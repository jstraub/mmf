#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include "SimpleImage.h"

using std::cerr;
void SimpleImage::allocate_2D_image(size_t _width,size_t _height,SimpleImage::T *h_img, size_t _depth)
{
    height=_height;
    width=_width;
    size_t bwidth=width*sizeof(SimpleImage::T);
    depth=_depth;
    channels=1;
    gpu_data=true;
    cerr<<"height= "<<height<<", width= "<<width<<"\n";
    cudaError_t err;

    cudaExtent extent = make_cudaExtent(bwidth, height,depth);
    cudaPitchedPtr dest_ptr;
    err=cudaMalloc3D(&dest_ptr,extent);
    if (!h_img)
    {
        this->data=(T*)dest_ptr.ptr;
        err=cudaMemset3D (dest_ptr,0,extent);
    }
    else
    {
        cudaPitchedPtr src_ptr=make_cudaPitchedPtr((void *)h_img,bwidth,width,height);
        size_t pitch=dest_ptr.pitch;
        if (err!=cudaSuccess)
        {
            cerr<<"Error on allocating pitched memory: "<<cudaGetErrorString(err)<<"\n";
        }
        cudaMemcpy3DParms myParms = {0};
        myParms.extent=extent;
        myParms.srcPtr=src_ptr;
        myParms.dstPtr=dest_ptr;
        cerr<<"cudaMalloc3D(2): allocated at "<<dest_ptr.ptr<<"\n";
        this->data=(float *)dest_ptr.ptr;
        this->pitch=dest_ptr.pitch;
        myParms.kind=cudaMemcpyHostToDevice;

        cerr<<"dpitch = "<<pitch<<", spitch="<<bwidth<<", width="<<bwidth<<", pitch="<<pitch<<"\n";
        cerr<<"from: "<<h_img<<", to: "<<this->data<<"\n";
        cerr<<"cudaMemcpyHostToDevice\n";
        cerr<<"h_img.pitch = "<<pitch<<"\n";
        err=cudaMemcpy3D(&myParms);
        cerr<<"..\n";
    }
    if (err!=cudaSuccess)
    {
        cerr<<"Error on copying to pitched memory: "<<cudaGetErrorString(err)<<"\n";
    }

    cerr<<"SimpleImage.allocate_2D_image: data="<<this->data<<"\n";
}




void SimpleImage::allocate_2D_image(const SimpleImage &h_img)
{
    height=h_img.height;
    width=h_img.width;
    depth=h_img.depth;
    channels=1;
    gpu_data=true;
    cerr<<"height= "<<height<<", width= "<<width<<"\n";
    //cudaError_t err=cudaMallocPitch(&this->data,&pitch,width*sizeof(SimpleImage::T),height);
    cudaPitchedPtr src_ptr=make_cudaPitchedPtr((void *)h_img.data,h_img.pitch,h_img.width,h_img.height);

    size_t bwidth=width*sizeof(SimpleImage::T);
    cudaExtent extent = make_cudaExtent(bwidth, height,depth);
    cudaPitchedPtr dest_ptr;
    cudaError_t err=cudaMalloc3D(&dest_ptr,extent);

    if (err!=cudaSuccess)
    {
        cerr<<"Error on allocating pitched memory: "<<cudaGetErrorString(err)<<"\n";
    }
    cudaMemcpy3DParms myParms = {0};

    myParms.srcPtr=src_ptr;
    myParms.dstPtr=dest_ptr;
    myParms.extent=extent;
    cerr<<"cudaMalloc3D(2): allocated at "<<dest_ptr.ptr<<"\n";
    this->data=(float *)dest_ptr.ptr;
    this->pitch=dest_ptr.pitch;

    cerr<<"dpitch = "<<pitch<<", spitch="<<bwidth<<", width="<<bwidth<<", pitch="<<pitch<<"\n";
    cerr<<"from: "<<h_img.data<<", to: "<<data<<"\n";
    if (h_img.gpu_data)
    {
        cerr<<"cudaMemcpyDeviceToDevice\n";
        myParms.kind=cudaMemcpyDeviceToDevice;
        err=cudaMemcpy3D(&myParms);
    }
    else
    {
        cerr<<"cudaMemcpyHostToDevice\n";
        cerr<<"h_img.pitch = "<<h_img.pitch<<"\n";
        myParms.kind=cudaMemcpyHostToDevice;
        err=cudaMemcpy3D(&myParms);
        //   err=cudaMemcpy2D(this->data, pitch, h_img.data, h_img.pitch, bwidth, height, cudaMemcpyHostToDevice);
    }

    //err=cudaMemcpy2D(data, pitch, h_img.data, bwidth, bwidth, height, cudaMemcpyHostToDevice);
//err=cudaMemcpy2D(data, pitch, h_img, sizeof(SimpleImage::T), 1, 1, cudaMemcpyHostToDevice);

    if (err!=cudaSuccess)
    {
        cerr<<"Error on copying to pitched memory: "<<cudaGetErrorString(err)<<"\n";
    }

    cerr<<"SimpleImage.allocate_2D_image: data="<<this->data<<"\n";
//	height=_height;width=_width;depth=1;channels=1;gpu_data=true;
//	cerr<<"(2)height= "<<height<<", width= "<<width<<"\n";
//	cudaError_t err=cudaMallocPitch(&this->data,&pitch,width*sizeof(SimpleImage::T),height);
//if (err!=cudaSuccess){
//cerr<<"Error on allocating pitched memory(2): "<<cudaGetErrorString(err)<<"\n";
//}
////size_t bwidth=width*sizeof(SimpleImage::T);
//size_t bwidth=width*sizeof(SimpleImage::T);
//cerr<<"dpitch = "<<pitch<<", spitch="<<bwidth<<", width="<<bwidth<<"\n";
//cerr<<"from: "<<h_img.data<<", to: "<<data<<"\n";
//
//err=cudaMemcpy2D(data, pitch, h_img.data, bwidth, bwidth, height, cudaMemcpyHostToDevice);
////cerr<<"(2)width="<<width<<"dpitch = "<<dpitch<<", width = "<<width<<", bwidth="<<bwidth<<"\n";
////err=cudaMemcpy2D(data, dpitch, h_img.data, bwidth, bwidth, height, cudaMemcpyHostToDevice);
//if (err!=cudaSuccess){
//cerr<<"Error on copying to pitched memory(2): "<<cudaGetErrorString(err)<<"\n";
//}
//
}
void SimpleImage::copy_to_host(SimpleImage::T *h_img)
{
    cudaError_t err;
    if (gpu_data)
    {
        cerr<<"Copying GPU data to host..\n";
        cerr<<"height: "<<height<<", width: "<<width<<", depth: "<<depth<<", channels: "<<channels<<"\n";
        cerr<<"from: "<<data<<", to: "<<h_img<<"\n";
//err=cudaMemcpy( (void*)h_img,(void*)data, pitch*height*depth*channels,cudaMemcpyDeviceToHost);
        err=cudaMemcpy( (void*)h_img,(void*)data, pitch*height*depth*channels,cudaMemcpyDeviceToHost);
//err=cudaMemcpy2D( (void*)h_img,(void*)pitch, data*height*depth*channels,cudaMemcpyDeviceToHost);
    }
    else
    {
        cerr<<"Copying CPU data to host..\n";
        err=cudaMemcpy( (void*)h_img,(void*)data, pitch*height*depth*channels,cudaMemcpyHostToHost);

    }
    if (err!=cudaSuccess)
    {
        cerr<<"Error on copying memory to host: "<<cudaGetErrorString(err)<<"\n";
    }
}

void SimpleImage::copy_to_host(SimpleImage& h_img)
{
    cudaError_t err;
    if (gpu_data && !h_img.gpu_data)
    {
        cerr<<"Copying GPU data to host (2)..\n";
        size_t bpitch=pitch;//*sizeof(SimpleImage::T);
        size_t dpitch=h_img.pitch;
        cerr<<"GPU pitch: "<<bpitch<<", host pitch: "<<dpitch<<"\n";
        cerr<<"height: "<<height<<", width: "<<width<<", depth: "<<depth<<", channels: "<<channels<<"\n";
        cerr<<"from: "<<data<<", to: "<<h_img.data<<"\n";
        //err=cudaMemcpy2D( (void *)h_img.data, dpitch, (void*)this->data,bpitch, width*sizeof(T),height,cudaMemcpyDeviceToHost);
        size_t bwidth=width*sizeof(T);
        cudaPitchedPtr src_ptr=make_cudaPitchedPtr((void *)data,pitch,width,height);
        cudaPitchedPtr dest_ptr=make_cudaPitchedPtr((void *)h_img.data,h_img.pitch,h_img.width,h_img.height);

        cudaExtent extent = make_cudaExtent(bwidth, height,depth);
        cudaMemcpy3DParms myParms = {0};

        myParms.srcPtr=src_ptr;
        myParms.dstPtr=dest_ptr;
        myParms.extent=extent;
        myParms.kind=cudaMemcpyDeviceToHost;
        err=cudaMemcpy3D(&myParms);

        if (err!=cudaSuccess)
        {
            cerr<<"Error on copying memory to host: "<<cudaGetErrorString(err)<<"\n";
        }
//err=cudaMemcpy( (void*)h_img.data,(void*)data, 1,cudaMemcpyDeviceToHost);
    }
    else
    {
//cerr<<"Copying CPU data to host..\n";
        cerr<<"TBD\n";
//err=cudaMemcpy2D( h_img.data, h_img.pitch*sizeof(SimpleImage::T), data,bpitch, width,height,cudaMemcpyDeviceToHost);
    }
}

SimpleImage::SimpleImage(unsigned int _width,unsigned int _height,unsigned int _depth,SimpleImage::T*_data, bool _gpu_data):width(_width),height(_height),depth(_depth),gpu_data(_gpu_data)
{
    pitch=_width*sizeof(T); // todo - replace in a different constructor..
    data=_data;
    //channels=0;
    //ctor
}

SimpleImage::~SimpleImage()
{
}
void SimpleImage::dealloc()
{
    if (data!=0)
    {

        if (gpu_data)
        {
            cerr<<"Freeing gpu data at "<<data<<"\n";
            cudaFree(data);
        }
        else
        {
            cerr<<"Freeing cpu data at"<<data<<"\n";
            delete[] data;
        }
        data=0;
    }
    //dtor
}



