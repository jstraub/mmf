#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "SimpleProfiler.h"
#include <math.h>
#include <limits.h>
#include "gpu_random.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
  //    if (abort) exit(code);
   }
}

//int n = 200;
using namespace std;

//__device__ uint rand_wanghash(uint& seed)
//{
//    seed = (seed ^ 61) ^ (seed >> 16);
//    seed *= 9;
//    seed = seed ^ (seed >> 4);
//    seed *= 0x27d4eb2d;
//    seed = seed ^ (seed >> 15);
//return seed;
//}
//
//__device__ float randn_wanghash(uint& seed, float buffer[2], bool& pos)
//{
//    if (pos){
//	pos=false;
//    float u1=((float)rand_wanghash(seed)/((float)UINT_MAX));
//    float u2=((float)rand_wanghash(seed)/((float)UINT_MAX));
//float lu1=__logf(u1);
//float R=sqrtf(-2*lu1);
//float theta=2*M_PI*u2;
//float tmp1;
//float tmp2;
//__sincosf(theta,&tmp1,&tmp2);
//	buffer[0]=tmp1*R;
//	buffer[1]=tmp2*R;
//	return buffer[0];
//    }
//    pos=true;
//    return buffer[1];
//}

__device__ float generate( curandState* globalState, int ind )
{
    //int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

inline __device__ float generateNormal( curandState& localState, int ind )
{

    //curandState localState = globalState[ind];
//    float RANDOM = curand_normal( &localState );
    //globalState[ind] = localState;
    return curand_normal( &localState );
}


__global__ void setup_kernel ( curandState* state, unsigned long seed, dim3 imageSize )
{
    int id_x = blockIdx.x*blockDim.x+threadIdx.x;
    int id_y = blockIdx.y*blockDim.y+threadIdx.y;
    int id=id_x+imageSize.x*id_y;
    curand_init ( seed, id, 0, &state[id] );
}

__global__ void kernel(float* U, curandState* globalState, int n_iter, float sig_proposal, dim3 imageSize, unsigned long seed2)
{
    int id_x = blockIdx.x*blockDim.x+threadIdx.x;
    int id_y = blockIdx.y*blockDim.y+threadIdx.y;
    int ind=id_x+imageSize.x*id_y;
    // generate random numbers
float sig_proposal_init=sig_proposal;
float randn_buff[2];
uint seed=ind+seed2;
bool randn_phase=true;
//curandState state=globalState[ind];
U[ind]=randn_wanghash(seed,randn_buff,randn_phase)*sig_proposal_init;
//U[ind]=generateNormal(state, ind) * sig_proposal_init;
//U[ind]=ind;
float u=U[ind];
    for(int i=0;i<n_iter;i++)
    {
        float du=randn_wanghash(seed,randn_buff,randn_phase)*sig_proposal;
        //float du = generateNormal(state, ind) * sig_proposal;
        float un=u+du;
        // accept or reject..
        u=un;
    }
U[ind]=u;
//globalState[ind]=state;
}

int main()
{
    SimpleProfiler *prof=SimpleProfiler::getInstance();
    int width=640;
    int height=480;
    const dim3 imageSize(width,height,1);
    const dim3 blockSize(32,16,1);

    size_t gridCols = (width + blockSize.x - 1) / blockSize.x;
    size_t gridRows = (height + blockSize.y - 1) / blockSize.y;
    const dim3 gridSize(gridCols,gridRows,1);
    curandState* devStates;
    int N=width*height;
    gpuErrchk(cudaMalloc ( &devStates, N*sizeof( curandState ) ));

    // setup seeds
    //setup_kernel <<< gridSize, blockSize >>> ( devStates,unsigned(time(NULL)),imageSize );
    gpuErrchk( cudaPeekAtLastError() );
    float U[N];
    float* dev_u;
    prof->StartWatch("overall computation");
    prof->StartWatch("alloc");
    cudaMalloc((void**) &dev_u, sizeof(float)*N);
    prof->StopWatch("alloc");
    gpuErrchk( cudaPeekAtLastError() );
    int n_iter=9000;
    float sig_proposal=1.f;
    prof->StartWatch("sampling");
    kernel<<<gridSize,blockSize>>> (dev_u, devStates, n_iter,sig_proposal,imageSize,unsigned(time(NULL)));
    gpuErrchk( cudaPeekAtLastError() );
    cudaThreadSynchronize();
    prof->StopWatch("sampling");

    cudaMemcpy(U, dev_u, sizeof(float)*N, cudaMemcpyDeviceToHost);
    gpuErrchk( cudaPeekAtLastError() );
    cudaThreadSynchronize();
    prof->StopWatch("overall computation");
    for(int i=0;i<10;i++)
    {
        cout<<U[i]<<endl;
    }
    cudaFree(dev_u);
    prof->PrintWatches();
    return 0;
}
