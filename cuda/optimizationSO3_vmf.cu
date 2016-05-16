/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <nvidia/helper_cuda.h>

#define PI  3.141592653589793f
#define BLOCK_SIZE 256
#define N_PER_T 16                                                              

// step size of the normals 
// for PointXYZI
#define X_STEP 3 // 8
#define X_OFFSET 0

/* 
 */
template <int K>
__global__ void MMFvMFCostFctAssignment(float *cost, uint32_t* W, 
    float *x, uint32_t *z, float *mu, float* pi, int N)
{
  const int DIM = 3;
  //__shared__ float xi[BLOCK_SIZE*3];
  __shared__ float mui[DIM*K*6];
  __shared__ float pik[K*6];
  __shared__ float rho[BLOCK_SIZE];
  __shared__ uint32_t Wi[BLOCK_SIZE];
  
  const int tid = threadIdx.x ;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // caching 
  if(tid < K*DIM*6) mui[tid] = mu[tid];
  if(tid < K*6) pik[tid] = pi[tid];
  rho[tid] = 0.0f;
  Wi[tid] = 0;

  __syncthreads(); // make sure that ys have been cached
  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)                      
  {
    float xi[3];
    xi[0] = x[id*X_STEP+X_OFFSET+0];
    xi[1] = x[id*X_STEP+X_OFFSET+1];
    xi[2] = x[id*X_STEP+X_OFFSET+2];

    float err_max = -1e7f;
    uint32_t k_max = 6*K+1;
    if((xi[0]!=xi[0] || xi[1]!=xi[1] || xi[2]!=xi[2]) 
        || xi[0]*xi[0]+xi[1]*xi[1]+xi[2]*xi[2] < 0.9f )
    {
      // if nan
      k_max = 6*K+1;
      err_max = -1e7f; 
    }else{
#pragma unroll
      for (uint32_t k=0; k<6*K; ++k) {
        float err = pik[k] + xi[0]*mui[k] + xi[1]*mui[k+K*6] + xi[2]*mui[k+2*K*6];
        if(err_max < err) {
          err_max = err;
          k_max = k;
        }
      }
      rho[tid] += err_max;
      Wi[tid] += 1.;
    }
    z[id] = k_max;
  }
  //reduction.....
  // TODO: make it faster!
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLOCK_SIZE)/2; s>1; s>>=1) {
    if(tid < s) {
      rho[tid] += rho[tid + s];
      Wi[tid] += Wi[tid + s];
    }
    __syncthreads();
  }

  if(tid==0) {
    atomicAdd(&cost[0],rho[0]+rho[1]);
  }
  if(tid==1) {
    atomicAdd(W,Wi[0]+Wi[1]);
  }
}

template <int K>
__global__ void MMFvMFCostFctAssignment(float *cost, uint32_t* W, 
    float *x, float *weights, uint32_t *z, float *mu, float* pi,  int N)
{
  const int DIM = 3;
  //__shared__ float xi[BLOCK_SIZE*3];
  __shared__ float mui[DIM*6*K];
  __shared__ float pik[6*K];
  __shared__ float rho[BLOCK_SIZE];
  __shared__ uint32_t Wi[BLOCK_SIZE];
  
  const int tid = threadIdx.x ;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // caching 
  if(tid < K*DIM*6) mui[tid] = mu[tid];
  if(tid < K*6) pik[tid] = pi[tid];
  rho[tid] = 0.0f;
  Wi[tid] = 0;

  __syncthreads(); // make sure that ys have been cached
  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)                      
  {
    float xi[3];
    xi[0] = x[id*X_STEP+X_OFFSET+0];
    xi[1] = x[id*X_STEP+X_OFFSET+1];
    xi[2] = x[id*X_STEP+X_OFFSET+2];
    float weight = weights[id];

    float err_max = -1e7f;
    uint32_t k_max = K*6+1;
    if((xi[0]!=xi[0] || xi[1]!=xi[1] || xi[2]!=xi[2]) 
        || xi[0]*xi[0]+xi[1]*xi[1]+xi[2]*xi[2] < 0.9f )
    {
      // if nan
      k_max = K*6+1;
      err_max = -1e7f; 
    }else{
#pragma unroll
      for (uint32_t k=0; k<K*6; ++k) {
        float err = pik[k] + xi[0]*mui[k] + xi[1]*mui[k+K*6] + xi[2]*mui[k+K*2*6];
        if(err_max < err) {
          err_max = err;
          k_max = k;
        }
      }
      rho[tid] += weight*err_max;
      Wi[tid] += weight;
    }
    z[id] = k_max;
  }
  //reduction.....
  // TODO: make it faster!
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLOCK_SIZE)/2; s>1; s>>=1) {
    if(tid < s) {
      rho[tid] += rho[tid + s];
      Wi[tid] += Wi[tid + s];
    }
    __syncthreads();
  }

  if(tid==0) {
    atomicAdd(&cost[0],rho[0]+rho[1]);
  }
  if(tid==1) {
    atomicAdd(W,Wi[0]+Wi[1]);
  }
}


extern void MMFvMFCostFctAssignmentGPU(float *h_cost, float *d_cost,
  uint32_t *h_W, uint32_t *d_W, float *d_x, float* d_weights,
  uint32_t *d_z, float *d_mu, float* d_pi, int N, int K)
 {
   if (K>=7) {
    printf("currently only 7 MFvMFs are supported");
   }
  assert(K<8);

  for(uint32_t k=0; k<6*K; ++k) h_cost[k] =0.0f;
  *h_W =0;
  checkCudaErrors(cudaMemcpy(d_cost, h_cost, 6*K*sizeof(float), 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_W, h_W, sizeof(uint32_t), 
        cudaMemcpyHostToDevice));

  dim3 threads(BLOCK_SIZE,1,1);
  dim3 blocks(N/(BLOCK_SIZE*N_PER_T)+(N%(BLOCK_SIZE*N_PER_T)>0?1:0),1,1);
  if (K==1) {
    if (d_weights == NULL) {
      MMFvMFCostFctAssignment<1><<<blocks,threads>>>(d_cost,d_W,d_x,
          d_z,d_mu,d_pi,N);
    }else{
      MMFvMFCostFctAssignment<1><<<blocks,threads>>>(d_cost,d_W,d_x,
          d_weights,d_z,d_mu,d_pi,N);
    }
  } else if (K==2) {
    if (d_weights == NULL) {
      MMFvMFCostFctAssignment<2><<<blocks,threads>>>(d_cost,d_W,d_x,
          d_z,d_mu,d_pi,N);
    }else{
      MMFvMFCostFctAssignment<2><<<blocks,threads>>>(d_cost,d_W,d_x,
          d_weights,d_z,d_mu,d_pi,N);
    }
  } else if (K==3) {
    if (d_weights == NULL) {
      MMFvMFCostFctAssignment<3><<<blocks,threads>>>(d_cost,d_W,d_x,
          d_z,d_mu,d_pi,N);
    }else{
      MMFvMFCostFctAssignment<3><<<blocks,threads>>>(d_cost,d_W,d_x,
          d_weights,d_z,d_mu,d_pi,N);
    }
  } else if (K==4) {
    if (d_weights == NULL) {
      MMFvMFCostFctAssignment<4><<<blocks,threads>>>(d_cost,d_W,d_x,
          d_z,d_mu,d_pi,N);
    }else{
      MMFvMFCostFctAssignment<4><<<blocks,threads>>>(d_cost,d_W,d_x,
          d_weights,d_z,d_mu,d_pi,N);
    }
  } else if (K==5) {
    if (d_weights == NULL) {
      MMFvMFCostFctAssignment<5><<<blocks,threads>>>(d_cost,d_W,d_x,
          d_z,d_mu,d_pi,N);
    }else{
      MMFvMFCostFctAssignment<5><<<blocks,threads>>>(d_cost,d_W,d_x,
          d_weights,d_z,d_mu,d_pi,N);
    }
  } else if (K==6) {
    if (d_weights == NULL) {
      MMFvMFCostFctAssignment<6><<<blocks,threads>>>(d_cost,d_W,d_x,
          d_z,d_mu,d_pi,N);
    }else{
      MMFvMFCostFctAssignment<6><<<blocks,threads>>>(d_cost,d_W,d_x,
          d_weights,d_z,d_mu,d_pi,N);
    }
  } else if (K==7) {
    if (d_weights == NULL) {
      MMFvMFCostFctAssignment<7><<<blocks,threads>>>(d_cost,d_W,d_x,
          d_z,d_mu,d_pi,N);
    }else{
      MMFvMFCostFctAssignment<7><<<blocks,threads>>>(d_cost,d_W,d_x,
          d_weights,d_z,d_mu,d_pi,N);
    }
  }
  checkCudaErrors(cudaDeviceSynchronize());
  
  checkCudaErrors(cudaMemcpy(h_cost, d_cost, 6*K*sizeof(float), 
        cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_W, d_W, sizeof(uint32_t), 
        cudaMemcpyDeviceToHost));
}

extern void vMFCostFctAssignmentGPU(float *h_cost, float *d_cost,
  uint32_t *h_W, uint32_t *d_W, float *d_x, float* d_weights,
  uint32_t *d_z, float *d_mu, float* d_pi, int N) {
  return MMFvMFCostFctAssignmentGPU(h_cost, d_cost, h_W, d_W, d_x,
      d_weights, d_z, d_mu, d_pi, N, 1) ;
}


