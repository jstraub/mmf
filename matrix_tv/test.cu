#include "adaptive_camera.h"
#include "SimpleImage.h"
#include <cuda.h>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const uint MAX_IMAGE_CHANNELS=7;

__global__ void AWGN_resample(float *dest,float *src, dim3 imgSize){
    uint id_x = blockIdx.x*blockDim.x+threadIdx.x;
    uint id_y = blockIdx.y*blockDim.y+threadIdx.y;
    dest[0]=0.f;
  //  src[0]=0.f;
    if (id_x>=imgSize.x	|| id_y>=imgSize.y){
	return;
    }
    uint id=id_x+id_y*imgSize.x;
	float seed=id_x*id_y;
//	RandStruct rand_str(seed);
//	float n=randn_wanghash(rand_str);

//    float r0=src->getPixel2D(id_x,id_y);
//    dest->getPixel2D(id_x,id_y)=r0;
    //dest[id]=1.f;
    //src[id]=1.f;
}

void test_AWGN_resample(float *dest,float *src, uint width,uint height){
    dim3 imgSize(width,height,1);
    SimpleImage *dsrc=new SimpleImage(width,height,1,(float*)0);
    dsrc->allocate_2D_image(width,height,src);
    SimpleImage *ddest=new SimpleImage(width,height,1,(float*)0);
    ddest->allocate_2D_image(width,height,dest);

    AWGN_resample<<<1,1>>>(dest,src,imgSize);
    gpuErrchk( cudaPeekAtLastError() );
    ddest->copy_to_host(dest);
    dsrc->copy_to_host(src);
}


//
// Usage:
int main(){
uint width=16;
uint height=32;
float *pEstimate=(float*)malloc(sizeof(float)*width*height);
float *pStdEstimate=(float*)malloc(sizeof(float)*width*height);
SimpleImage img_estimate(width,height,1,pEstimate);
SimpleImage proposal_std(width,height,1,pStdEstimate);
test_AWGN_resample(img_estimate.data,proposal_std.data, width,height);

//res_MI.data[0]=0.f;
}

