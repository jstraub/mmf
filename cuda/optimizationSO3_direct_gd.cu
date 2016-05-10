/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */

#include <stdint.h>
#include <stdio.h>
#include <nvidia/helper_cuda.h>

#define PI  3.141592653589793f
#define BLOCK_SIZE 256
#define N_PER_T 16                                                              

// step size of the normals 
// for PointXYZI
#define X_STEP 3 // 8
#define X_OFFSET 0

/* 
 * Given assignments z of normals x to MF axes compute the costfunction value
 */
__global__ void directSquaredAngleCostFct(float *cost, float *x,
    uint32_t *z, float *mu, int N) 
{
  const int DIM = 3;
  //__shared__ float xi[BLOCK_SIZE*3];
  __shared__ float mui[DIM*6];
  __shared__ float rho[BLOCK_SIZE];
  
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // caching 
  if(tid < DIM*6) mui[tid] = mu[tid];
  rho[tid] = 0.0f;

  __syncthreads(); // make sure that ys have been cached
  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)                      
  {
    //  xi[tid*3] = x[tid];
    //  xi[tid*3+1] = x[tid+Nx];
    //  xi[tid*3+2] = x[tid+Nx*2];

    uint32_t k = z[id];
    if (k<6)
    {
      // k==6 means that xi is nan
      float xiTy = x[id*X_STEP+X_OFFSET]*mui[k] + x[id*X_STEP+X_OFFSET+1]*mui[k+6] 
        + x[id*X_STEP+X_OFFSET+2]*mui[k+12];
      float err = acosf(max(-1.0f,min(1.0f,xiTy)));
      //float errSq = err*err;
      rho[tid] += err*err;
    }
  }
  //reduction.....
  // TODO: make it faster!
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLOCK_SIZE)/2; s>1; s>>=1) {
    if(tid < s)
      rho[tid] += rho[tid + s];
    __syncthreads();
  }
  if(tid==0  && rho[0]!=0 ) {
    atomicAdd(&cost[0],rho[0]+rho[1]);
  }
}
__global__ void directSquaredAngleCostFct(float *cost, float *x, float
    *weights, uint32_t *z, float *mu, int N) 
{
  const int DIM = 3;
  //__shared__ float xi[BLOCK_SIZE*3];
  __shared__ float mui[DIM*6];
  __shared__ float rho[BLOCK_SIZE];
  
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // caching 
  if(tid < DIM*6) mui[tid] = mu[tid];
  rho[tid] = 0.0f;

  __syncthreads(); // make sure that ys have been cached
  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)                      
  {
    //  xi[tid*3] = x[tid];
    //  xi[tid*3+1] = x[tid+Nx];
    //  xi[tid*3+2] = x[tid+Nx*2];

    uint32_t k = z[id];
    float weight = weights[id];
    if (k<6)
    {
      // k==6 means that xi is nan
      float xiTy = x[id*X_STEP+X_OFFSET]*mui[k] + x[id*X_STEP+X_OFFSET+1]*mui[k+6] 
        + x[id*X_STEP+X_OFFSET+2]*mui[k+12];
      float err = acosf(max(-1.0f,min(1.0f,xiTy)));
      //float errSq = err*err;
      rho[tid] += weight*err*err;
    }
  }
  //reduction.....
  // TODO: make it faster!
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLOCK_SIZE)/2; s>1; s>>=1) {
    if(tid < s)
      rho[tid] += rho[tid + s];
    __syncthreads();
  }
  if(tid==0  && rho[0]!=0 ) {
    atomicAdd(&cost[0],rho[0]+rho[1]);
  }
}

extern void directSquaredAngleCostFctGPU(float *h_cost, float *d_cost,
   float *d_x, float *d_weights, uint32_t *d_z, float *d_mu,
   int N)
{
  for(uint32_t k=0; k<6; ++k) h_cost[k] =0.0f;
  checkCudaErrors(cudaMemcpy(d_cost, h_cost, 6* sizeof(float), 
        cudaMemcpyHostToDevice));

  dim3 threads(BLOCK_SIZE,1,1);
  dim3 blocks(N/(BLOCK_SIZE*N_PER_T)+(N%(BLOCK_SIZE*N_PER_T)>0?1:0),1,1);
  if (d_weights == NULL)
  {
    directSquaredAngleCostFct<<<blocks,threads>>>(d_cost,d_x,d_z,d_mu,N);
  }else{
    directSquaredAngleCostFct<<<blocks,threads>>>(d_cost,d_x,d_weights,d_z,d_mu,N);
  }
  checkCudaErrors(cudaDeviceSynchronize());
  
  checkCudaErrors(cudaMemcpy(h_cost, d_cost, 6*sizeof(float),
        cudaMemcpyDeviceToHost));
}

/*
 * compute the Jacobian of direct squared cost function 
 */
__global__ void directSquaredAngleCostFctJacobian(float *J, float *x, 
    uint32_t *z, float *mu, int N)
{
  const int DIM = 3;
  __shared__ float mui[DIM*6];
  // one J per column; BLOCK_SIZE columns;
  __shared__ float J_shared[BLOCK_SIZE*3];
  
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // caching 
  if(tid < DIM*6) mui[tid] = mu[tid];
#pragma unroll
  for(int s=0; s<3; ++s) {
    J_shared[tid+BLOCK_SIZE*s] = 0.0f;
  }

  __syncthreads(); // make sure that ys have been cached
  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)                      
  {
    float xi[3];
    xi[0] = x[id*X_STEP+X_OFFSET+0];
    xi[1] = x[id*X_STEP+X_OFFSET+1];
    xi[2] = x[id*X_STEP+X_OFFSET+2];
    uint32_t k = z[id]; // which MF axis does it belong to
    if (k<6)// && k!=4 && k!=5)
    {
//      int j = k/2; // which of the rotation columns does this belong to
//      float sign = (-float(k%2) +0.5f)*2.0f; // sign of the axis
      float dot = max(-1.0f,min(1.0f,xi[0]*mui[k] + xi[1]*mui[k+6] 
        + xi[2]*mui[k+12]));
      float eps = acosf(dot);
      float factor = 0.;
      if(-0.99f< dot && dot < 0.99f)
        factor = -(2.f*eps)/(sqrt(1.f-dot*dot));
      else if(dot >= 0.99f) { 
        // taylor series around 0.99 according to Mathematica
        factor = -2.f*(1.0033467240646519f - 0.33601724502488395f
            *(-0.99f + dot) + 0.13506297338381046f* (-0.99f + dot)*(-0.99f + dot));
      }else if (dot <= -0.99f) {
        factor = -2.f*(21.266813135156017f - 1108.2484926534892f*(0.99f
              + dot) + 83235.29739487475f*(0.99f + dot)*(0.99f + dot));
      }
      J_shared[tid+0*BLOCK_SIZE]+=factor*(xi[2]*mui[k+6]-xi[1]*mui[k+12]);   
      J_shared[tid+1*BLOCK_SIZE]+=factor*(xi[0]*mui[k+12]-xi[2]*mui[k]);   
      J_shared[tid+2*BLOCK_SIZE]+=factor*(xi[1]*mui[k]-xi[0]*mui[k+6]);   
    }
  }
  //reduction.....
    __syncthreads(); //sync the threads
#pragma unroll
    for(int s=(BLOCK_SIZE)/2; s>1; s>>=1) {
      if(tid < s)
#pragma unroll
        for( int k=0; k<3; ++k) {
          int tidk = k*BLOCK_SIZE+tid;
          J_shared[tidk] += J_shared[tidk + s];
        }
      __syncthreads();
    }

#pragma unroll
    for( int k=0; k<3; ++k) {
      if(tid==k  && J_shared[k*BLOCK_SIZE]!=0 ) {
        atomicAdd(&J[k],J_shared[k*BLOCK_SIZE] + J_shared[k*BLOCK_SIZE+1]);
      }
    }

//  //reduction.....
//#pragma unroll
//  for( int k=0; k<3*3; ++k) {
//    int tidk = k*BLOCK_SIZE+tid;
//    __syncthreads(); //sync the threads
//#pragma unroll
//    for(int s=(BLOCK_SIZE)/2; s>0; s>>=1) {
//      if(tid < s)
//        J_shared[tidk] += J_shared[tidk + s];
//      __syncthreads();
//    }
//
//    if(tid==0  && J_shared[k*BLOCK_SIZE]!=0 ) {
//      atomicAdd(&J[k],J_shared[k*BLOCK_SIZE]);
//    }
//  }
}
__global__ void directSquaredAngleCostFctJacobian(float *J, float *x, 
    float *weights, uint32_t *z, float *mu, int N)
{
  const int DIM = 3;
  __shared__ float mui[DIM*6];
  // one J per column; BLOCK_SIZE columns;
  __shared__ float J_shared[BLOCK_SIZE*3];
  
  const int tid = threadIdx.x;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // caching 
  if(tid < DIM*6) mui[tid] = mu[tid];
#pragma unroll
  for(int s=0; s<3; ++s) {
    J_shared[tid+BLOCK_SIZE*s] = 0.0f;
  }

  __syncthreads(); // make sure that ys have been cached
  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)                      
  {
    float xi[3];
    xi[0] = x[id*X_STEP+X_OFFSET+0];
    xi[1] = x[id*X_STEP+X_OFFSET+1];
    xi[2] = x[id*X_STEP+X_OFFSET+2];
    uint32_t k = z[id]; // which MF axis does it belong to
    float w = weights[id];
    if (k<6)// && k!=4 && k!=5)
    {
//      int j = k/2; // which of the rotation columns does this belong to
//      float sign = (-float(k%2) +0.5f)*2.0f; // sign of the axis
      float dot = max(-1.0f,min(1.0f,xi[0]*mui[k] + xi[1]*mui[k+6] 
        + xi[2]*mui[k+12]));
      float eps = acosf(dot);
      float factor = 0.;
      if(-0.99f< dot && dot < 0.99f)
        factor = -(2.f*eps)/(sqrt(1.f-dot*dot));
      else if(dot >= 0.99f) { 
        // taylor series around 0.99 according to Mathematica
        factor = -2.f*(1.0033467240646519f - 0.33601724502488395f
            *(-0.99f + dot) + 0.13506297338381046f* (-0.99f + dot)*(-0.99f + dot));
      }else if (dot <= -0.99f) {
        factor = -2.f*(21.266813135156017f - 1108.2484926534892f*(0.99f
              + dot) + 83235.29739487475f*(0.99f + dot)*(0.99f + dot));
      }
      J_shared[tid+0*BLOCK_SIZE]+=w*factor*(xi[2]*mui[k+6]-xi[1]*mui[k+12]);   
      J_shared[tid+1*BLOCK_SIZE]+=w*factor*(xi[0]*mui[k+12]-xi[2]*mui[k]);   
      J_shared[tid+2*BLOCK_SIZE]+=w*factor*(xi[1]*mui[k]-xi[0]*mui[k+6]);   
    }
  }
  //reduction.....
    __syncthreads(); //sync the threads
#pragma unroll
    for(int s=(BLOCK_SIZE)/2; s>1; s>>=1) {
      if(tid < s)
#pragma unroll
        for( int k=0; k<3; ++k) {
          int tidk = k*BLOCK_SIZE+tid;
          J_shared[tidk] += J_shared[tidk + s];
        }
      __syncthreads();
    }

#pragma unroll
    for( int k=0; k<3; ++k) {
      if(tid==k  && J_shared[k*BLOCK_SIZE]!=0 ) {
        atomicAdd(&J[k],J_shared[k*BLOCK_SIZE] + J_shared[k*BLOCK_SIZE+1]);
      }
    }

//  //reduction.....
//#pragma unroll
//  for( int k=0; k<3*3; ++k) {
//    int tidk = k*BLOCK_SIZE+tid;
//    __syncthreads(); //sync the threads
//#pragma unroll
//    for(int s=(BLOCK_SIZE)/2; s>0; s>>=1) {
//      if(tid < s)
//        J_shared[tidk] += J_shared[tidk + s];
//      __syncthreads();
//    }
//
//    if(tid==0  && J_shared[k*BLOCK_SIZE]!=0 ) {
//      atomicAdd(&J[k],J_shared[k*BLOCK_SIZE]);
//    }
//  }
}

extern void directSquaredAngleCostFctJacobianGPU(float *h_J, float *d_J,
   float *d_x, float *d_weights, uint32_t *d_z, float *d_mu,
   int N)
{
  for(uint32_t k=0; k<3; ++k) h_J[k] = 0.0f;
  checkCudaErrors(cudaMemcpy(d_J, h_J, 3*sizeof(float), 
        cudaMemcpyHostToDevice));

  dim3 threads(BLOCK_SIZE,1,1);
  dim3 blocks(N/(BLOCK_SIZE*N_PER_T)+(N%(BLOCK_SIZE*N_PER_T)>0?1:0),1,1);
  if (d_weights == NULL)
  {
    directSquaredAngleCostFctJacobian<<<blocks,threads>>>(d_J,d_x,d_z,d_mu, N);
  }else{
    directSquaredAngleCostFctJacobian<<<blocks,threads>>>(d_J,d_x,d_weights,
        d_z,d_mu,N);
  }
  checkCudaErrors(cudaDeviceSynchronize());
  
  checkCudaErrors(cudaMemcpy(h_J, d_J, 3*sizeof(float),
    cudaMemcpyDeviceToHost));
}


/* 
 * compute normal assignments as well as the costfunction value under that
 * assignment. Normal assignments are computed according based on nearest 
 * distance in the arclength sense.
 */
__global__ void directSquaredAngleCostFctAssignment(float *cost, uint32_t* W, 
    float *x, uint32_t *z, float *mu, int N)
{
  const int DIM = 3;
  //__shared__ float xi[BLOCK_SIZE*3];
  __shared__ float mui[DIM*6];
  __shared__ float rho[BLOCK_SIZE];
  __shared__ uint32_t Wi[BLOCK_SIZE];
  
  const int tid = threadIdx.x ;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // caching 
  if(tid < DIM*6) mui[tid] = mu[tid];
  rho[tid] = 0.0f;
  Wi[tid] = 0;

  __syncthreads(); // make sure that ys have been cached
  for(int id=idx*N_PER_T; id<min(N,(idx+1)*N_PER_T); ++id)                      
  {
    float xi[3];
    xi[0] = x[id*X_STEP+X_OFFSET+0];
    xi[1] = x[id*X_STEP+X_OFFSET+1];
    xi[2] = x[id*X_STEP+X_OFFSET+2];

    float err_min = 9999999.0f;
    uint32_t k_min = 6;
    if((xi[0]!=xi[0] || xi[1]!=xi[1] || xi[2]!=xi[2]) 
        || xi[0]*xi[0]+xi[1]*xi[1]+xi[2]*xi[2] < 0.9f )
    {
      // if nan
      k_min = 6;
      err_min = .1f; 
      //if(X_STEP == 8) x[id*X_STEP+4] = 6.0f;
    }else{
#pragma unroll
      for (uint32_t k=0; k<6; ++k)
      {
        float xiTy = xi[0]*mui[k] + xi[1]*mui[k+6] + xi[2]*mui[k+12];
        float err = acosf(max(-1.0f,min(1.0f,xiTy)));
        if(err_min > err)
        {
          err_min = err;
          k_min = k;
        }
      }
      rho[tid] += err_min*err_min;
      Wi[tid] += 1.;
    }
    z[id] = k_min;
//    errs[id] = err_min;
//    if(X_STEP == 8) 
//    {
//      x[id*X_STEP+X_OFFSET+4] = c_rgbForMFaxes[k_min];//float(k_min);
//      x[id*X_STEP+X_OFFSET+5] = float(k_min);//xi[0]; //float(k_min);
//      x[id*X_STEP+X_OFFSET+6] = err_min; //rgb;//xi[1]; //err_min;
////      x[id*X_STEP+X_OFFSET+7] = 0.0f;//err_min; //err_min;
//    }
  }
  //reduction.....
  // TODO: make it faster!
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLOCK_SIZE)/2; s>1; s>>=1) {
    if(tid < s)
    {
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
__global__ void directSquaredAngleCostFctAssignment(float *cost, uint32_t* W, 
    float *x, float *weights, uint32_t *z, float *mu, int N)
{
  const int DIM = 3;
  //__shared__ float xi[BLOCK_SIZE*3];
  __shared__ float mui[DIM*6];
  __shared__ float rho[BLOCK_SIZE];
  __shared__ uint32_t Wi[BLOCK_SIZE];
  
  const int tid = threadIdx.x ;
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // caching 
  if(tid < DIM*6) mui[tid] = mu[tid];
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

    float err_min = 9999999.0f;
    uint32_t k_min = 6;
    if((xi[0]!=xi[0] || xi[1]!=xi[1] || xi[2]!=xi[2]) 
        || xi[0]*xi[0]+xi[1]*xi[1]+xi[2]*xi[2] < 0.9f )
    {
      // if nan
      k_min = 6;
      err_min = .1f; 
      //if(X_STEP == 8) x[id*X_STEP+4] = 6.0f;
    }else{
#pragma unroll
      for (uint32_t k=0; k<6; ++k)
      {
        float xiTy = xi[0]*mui[k] + xi[1]*mui[k+6] + xi[2]*mui[k+12];
        float err = acosf(max(-1.0f,min(1.0f,xiTy)));
        if(err_min > err)
        {
          err_min = err;
          k_min = k;
        }
      }
      rho[tid] += weight*err_min*err_min;
      Wi[tid] += weight;
    }
    z[id] = k_min;
//    errs[id] = err_min;
//    if(X_STEP == 8) 
//    {
//      x[id*X_STEP+X_OFFSET+4] = c_rgbForMFaxes[k_min];//float(k_min);
//      x[id*X_STEP+X_OFFSET+5] = float(k_min);//xi[0]; //float(k_min);
//      x[id*X_STEP+X_OFFSET+6] = err_min; //rgb;//xi[1]; //err_min;
////      x[id*X_STEP+X_OFFSET+7] = 0.0f;//err_min; //err_min;
//    }
  }
  //reduction.....
  // TODO: make it faster!
  __syncthreads(); //sync the threads
#pragma unroll
  for(int s=(BLOCK_SIZE)/2; s>1; s>>=1) {
    if(tid < s)
    {
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

extern void directSquaredAngleCostFctAssignmentGPU(float *h_cost, float *d_cost,
  uint32_t *h_W, uint32_t *d_W, float *d_x, float* d_weights,
  uint32_t *d_z, float *d_mu, int N)
{

  for(uint32_t k=0; k<6; ++k) h_cost[k] =0.0f;
  *h_W =0;
  checkCudaErrors(cudaMemcpy(d_cost, h_cost, 6* sizeof(float), 
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_W, h_W, sizeof(uint32_t), 
        cudaMemcpyHostToDevice));

  dim3 threads(BLOCK_SIZE,1,1);
  dim3 blocks(N/(BLOCK_SIZE*N_PER_T)+(N%(BLOCK_SIZE*N_PER_T)>0?1:0),1,1);
  if (d_weights == NULL)
  {
    directSquaredAngleCostFctAssignment<<<blocks,threads>>>(d_cost,d_W,d_x,
        d_z,d_mu,N);
  }else{
    directSquaredAngleCostFctAssignment<<<blocks,threads>>>(d_cost,d_W,d_x,
        d_weights,d_z,d_mu,N);
  }
  checkCudaErrors(cudaDeviceSynchronize());
  
  checkCudaErrors(cudaMemcpy(h_cost, d_cost, 6*sizeof(float), 
        cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_W, d_W, sizeof(uint32_t), 
        cudaMemcpyDeviceToHost));
}

