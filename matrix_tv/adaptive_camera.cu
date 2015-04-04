#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "SimpleProfiler.h"
#include <math.h>
#include <iostream>
#include <limits.h>
#include "gpu_random.h"
#include "fixed_matrix.h"
#include "fixed_vector.h"
#include "SimpleImage.h"
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
template<class T, unsigned int N> __device__ void draw_image_given_r(RandStruct &str,T r, FixedVector<T,N> & img, T sigma)
{
    img[0]=randn_wanghash(str)*sigma+r;
}
__device__  float draw_surface_value(RandStruct &str, float r0)
{
    return randn_wanghash(str)+r0;
}
__device__ __host__ float log_prob_I_given_r(float r,FixedVector<float,MAX_IMAGE_CHANNELS> &img_given_r, float sigma)
{
    return 0;
}
__device__ __host__ float log_prob_I(RandStruct rand_str, float r,FixedVector<float,MAX_IMAGE_CHANNELS> &img_given_r, float sigma)
{
    return 0;
}

__global__ void compute_MI_proposal(SimpleImage* original,SimpleImage* sigmas,SimpleImage* o_MI)
{
    int id_x = blockIdx.x*blockDim.x+threadIdx.x;
    int id_y = blockIdx.y*blockDim.y+threadIdx.y;
    float x=id_x;
    float y=id_y;
    uint EX_ITER=10;
    uint seed=id_x+original->width*id_y;
    if (id_x>=original->width||id_y>=original->height)
        return;
    float r0=original->getPixel2D(id_x,id_y);
    // sample from the surface
    FixedVector<float,MAX_IMAGE_CHANNELS> img_given_r;
    float MI=0;
    RandStruct rand_str(seed);
    /*
    for (int ex_iter=0;ex_iter<EX_ITER;++ex_iter){
    // probably through metropolis-hastings
        float r=draw_surface_value(rand_str,r0);
    float sigma=sigmas->getPixel2D(x,y);
        draw_image_given_r<float,MAX_IMAGE_CHANNELS>(rand_str,r,img_given_r,sigma);
        float logp_I_given_r=log_prob_I_given_r(r,img_given_r,sigma);
        float logp_I=log_prob_I(rand_str,r,img_given_r,sigma);
    	MI=MI+logp_I_given_r-logp_I;
    	}
    MI=MI/EX_ITER;
    o_MI->getPixel2D(id_x,id_y)=0.f;*/
}

__global__ void AWGN_resample(SimpleImage dest, SimpleImage src, SimpleImage sigs){
    uint id_x = blockIdx.x*blockDim.x+threadIdx.x;
    uint id_y = blockIdx.y*blockDim.y+threadIdx.y;
  //  dest[0]=0.f;
//    src[0]=0.f;
  //  src[0]=0.f;
    if (id_x>=dest.width	|| id_y>=dest.height){
	return;
    }
    uint id=id_x+id_y*dest.pitch/sizeof(SimpleImage::T);
	float seed=id_x*id_y;
	RandStruct rand_str(seed);
	//float n=randn_wanghash(rand_str);
    float n=id;
//    float r0=src->getPixel2D(id_x,id_y);
//    dest->getPixel2D(id_x,id_y)=r0;
    dest.data[id]=src.data[id]+n*sigs.data[id];
    //src[id]=1.f;
}

void test_AWGN_resample(float *dest,float *src, uint width,uint height){
    dim3 imgSize(width,height,1);
    SimpleImage *dsrc=new SimpleImage(width,height,1,(float*)0);
    dsrc->allocate_2D_image(width,height,src);
    SimpleImage *ddest=new SimpleImage(width,height,1,(float*)0);
    ddest->allocate_2D_image(width,height,dest);

    AWGN_resample<<<1,1>>>(*ddest,*dsrc,*dsrc);
    gpuErrchk( cudaPeekAtLastError() );
}


class GPUImageOperations
{
protected:
dim3 gridSize;
dim3 blockSize;
dim3 imageSize;
public:
GPUImageOperations(const dim3 imageSize_,const dim3 blockSize_):blockSize(blockSize_),imageSize(imageSize_){
size_t width=imageSize.x;
size_t height=imageSize.y;
    size_t gridCols = (width + blockSize.x - 1) / blockSize.x;
    size_t gridRows = (height + blockSize.y - 1) / blockSize.y;
    gridSize=dim3(gridCols,gridRows,1);
cerr<<"gridSize: "<<gridSize.x<<","<<gridSize.y<<".\n";
cerr<<"blockSize: "<<blockSize.x<<","<<blockSize.y<<".\n";
}

    void operator() (SimpleImage &original_distribution, SimpleImage &dest_distribution,SimpleImage &observations)
    {

    }
};


class SampleLatentImage:GPUImageOperations
{
public:
SampleLatentImage(const dim3 imageSize_,const dim3 blockSize_):GPUImageOperations(imageSize_,blockSize_){
}
    void operator() (SimpleImage &dest_distribution,SimpleImage &original_distribution,SimpleImage &sigmas)
    {
        cerr<<"imageSize: "<<imageSize.x<<","<<imageSize.y<<"\n";

	AWGN_resample<<<gridSize,blockSize>>>(dest_distribution,original_distribution,sigmas);
    }
};

class UpdateLatentImage:GPUImageOperations
{
public:
UpdateLatentImage(const dim3 imageSize_,const dim3 blockSize_):GPUImageOperations(imageSize_,blockSize_){
}
    void operator() (SimpleImage &original_distribution, SimpleImage &dest_distribution,SimpleImage &observations)
    {
    }
};

class RangeSampleObservation:GPUImageOperations
{
public:
RangeSampleObservation(const dim3 imageSize_,const dim3 blockSize_):GPUImageOperations(imageSize_,blockSize_){
}

    void operator() (SimpleImage &original_distribution, SimpleImage &dest_distribution,SimpleImage &observations)
    {
    }
};

void sample_mutual_information_image(SimpleImage &latent_img, SimpleImage &proposed_latent_img, SimpleImage &sig_img, SimpleImage &observations_img,
                                     SampleLatentImage sample_latent, UpdateLatentImage update_latent, RangeSampleObservation range_sample_observation, SimpleImage &MI, uint external_iter, uint burnin, const dim3 gridSize, const dim3 blockSize)
{
    for(int i_e=0; i_e<external_iter; ++i_e)
    {
        for(int i_i=0; i_i<burnin; ++i_i)
        {
            sample_latent(proposed_latent_img,latent_img,sig_img);
            update_latent(latent_img,proposed_latent_img,observations_img);
        }

//range_sample_observation(latent_img,observations_img);
    }

    sample_latent(MI,latent_img,sig_img);
    //if( cudaPeekAtLastError() ~=);
}


void test_AWGN_resample2(float *dest,float *src, uint width,uint height){
    dim3 imgSize(width,height,1);
//    SimpleImage *dsrc=new SimpleImage(width,height,1,(float*)0);
//    dsrc->allocate_2D_image(width,height,src);
    SimpleImage *ddest=new SimpleImage(width,height,1,(float*)0);
    SimpleImage *dsrc=new SimpleImage(width,height,1,(float*)0);
    ddest->allocate_2D_image(width,height,dest);
    dsrc->allocate_2D_image(width,height,src);
    cerr<<"executing, with dest.data = "<<dest<<"\n";
    AWGN_resample<<<1,1>>>(*ddest,*dsrc,*dsrc);
    gpuErrchk( cudaPeekAtLastError() );
    ddest->copy_to_host(dest);
    //dsrc->copy_to_host(src);
}


void active_camera_estimation(SimpleImage& img_estimate,SimpleImage& observation_std,SimpleImage& proposal_std,SimpleImage& res_MI, double min_image_value, double max_image_value, double image_step)
{
    int width=img_estimate.width;
    int height=img_estimate.height;
    const dim3 imageSize(width,height,1);
    const dim3 blockSize(32,16,1);
// load to GPU
//float * tmp=(float*)malloc(sizeof(float)*width*height);
//////    SimpleImage *latent_img=new SimpleImage(width,height,1,(float*)0);
//////    latent_img->allocate_2D_image(img_estimate);
//////    cerr<<"latent_img: at "<<latent_img->data<<"\n";
//////    gpuErrchk( cudaPeekAtLastError() );
//////    SimpleImage *proposed_latent_img=new SimpleImage(width,height,1,(float*)0);
//////    proposed_latent_img->allocate_2D_image(img_estimate);
//////    gpuErrchk( cudaPeekAtLastError() );
//////    SimpleImage *sigs=new SimpleImage(width,height,1,(float*)0);
//////    sigs->allocate_2D_image(img_estimate);
//////    gpuErrchk( cudaPeekAtLastError() );
//////    SimpleImage *proposal_sigs=new SimpleImage(width,height,1,(float*)0);
//////    proposal_sigs->allocate_2D_image(proposal_std);
//////    gpuErrchk( cudaPeekAtLastError() );
    SimpleImage *o_MI=new SimpleImage(width,height,1,(float*)0);
    o_MI->allocate_2D_image(width,height,res_MI.data);
    gpuErrchk( cudaPeekAtLastError() );
//////    SampleLatentImage sample_latent(imageSize,blockSize);
//////    UpdateLatentImage update_latent(imageSize,blockSize);
//////    RangeSampleObservation range_sample_observation(imageSize,blockSize);
    size_t gridCols = (width + blockSize.x - 1) / blockSize.x;
    size_t gridRows = (height + blockSize.y - 1) / blockSize.y;
    const dim3 gridSize(gridCols,gridRows,1);
    //TODO use image and pitch
//////    cerr<<"blockSize: "<<blockSize.x<<","<<blockSize.y<<"\n";
//////    cerr<<"gridSize: "<<gridSize.x<<","<<gridSize.y<<"\n";
//////    cerr<<"imageSize: "<<imageSize.x<<","<<imageSize.y<<"\n";
//////    uint burnin=10;
//////    uint external_iter=10;
//////    gpuErrchk( cudaPeekAtLastError() );
//////    cerr<<"resampling at "<<sigs->data<<", size: "<<imageSize.x<<","<<imageSize.y<<"\n";
    cerr<<"writing to: "<<o_MI->data<<"\n";
    AWGN_resample<<<gridSize,blockSize>>>(*o_MI,*o_MI,proposal_std);
//////    //cudaMemset ((void *)o_MI->data, 0, sizeof(float) );
//////   //AWGN_resample<<<gridSize,blockSize>>>(sigs->data,o_MI->data,imageSize);
//////    gpuErrchk( cudaPeekAtLastError() );
//////    // setup seeds
//////    //sample_mutual_information_image ( *latent_img,*proposed_latent_img, *sigs,*o_MI ,sample_latent, update_latent, range_sample_observation,*o_MI, external_iter,burnin,gridSize, blockSize);
//////    //update_image_measurements
//////    //
//////    cerr<<"before: "<<res_MI.data[0]<<"\n";
//////    //cudaMemset ((void*)o_MI->data, 0,sizeof(float)*10 );
cerr<<"Copying..";
        o_MI->copy_to_host(res_MI);
        o_MI->dealloc();
//////    cerr<<"after: "<<res_MI.data[0]<<"\n";
//////
//////    //res_MI.data[0]=0.f;
//////    delete proposal_sigs;
//////    delete sigs;
    delete o_MI;
//////    delete proposed_latent_img;
//////    delete latent_img;
}

