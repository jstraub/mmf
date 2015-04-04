#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "SimpleProfiler.h"
#include <math.h>
#include <limits.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
  //    if (abort) exit(code);
   }
}

int n = 200;
using namespace std;

__device__ uint rand_wanghash(uint& seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
return seed;
}

__device__ float randn_wanghash(uint& seed, float buffer[2], bool& pos)
{
    if (pos){
	pos=false;
    float u1=((float)rand_wanghash(seed)/((float)UINT_MAX));
    float u2=((float)rand_wanghash(seed)/((float)UINT_MAX));
float lu1=__logf(u1);
float R=sqrtf(-2*lu1);
float theta=2*M_PI*u2;
float tmp1;
float tmp2;
__sincosf(theta,&tmp1,&tmp2);
	buffer[0]=tmp1*R;
	buffer[1]=tmp2*R;
	return buffer[0];
    }
    pos=true;
    return buffer[1];
}
