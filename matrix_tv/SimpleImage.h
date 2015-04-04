#ifndef SIMPLEIMAGE_H
#define SIMPLEIMAGE_H
#include <stdlib.h>
#include <iostream>
//#include <cuda.h>

using std::cerr;
 class SimpleImage
{
    public:
    typedef float T;
    T* data;
    bool gpu_data;
    size_t pitch; // width, in bytes
    size_t width;
    size_t height;
    size_t depth;
    size_t channels;
    SimpleImage(unsigned int _width,unsigned int _height,unsigned int _depth,T*_data, bool _gpu_data=true);
    virtual ~SimpleImage();
    virtual void dealloc();
#ifdef __CUDACC__
    inline __device__ __host__ T& getPixel2D(uint x,uint y){ return *((T*)((char*)data + y * pitch) + x*sizeof(T)); }
    inline __device__ __host__ T& getPixel3D(uint x,uint y,uint z){ return data[(y+height*z) * pitch/sizeof(T) + x]; }
#else
    inline T& getPixel2D(uint x,uint y){ return *((T*)((char*)data + y * pitch) + x*sizeof(T)); }
    inline T& getPixel3D(uint x,uint y,uint z){ return *((T*)((char*)data + (y+height*z) * pitch) + x); }
#endif
// h_img is the host image data
    void allocate_2D_image(size_t _width,size_t _height,T *h_img, size_t _depth=1);
    void allocate_2D_image(const SimpleImage &h_img);
    void copy_to_host(T *h_img);
    void copy_to_host(SimpleImage& h_img);
    protected:
    private:
};


#endif // SIMPLEIMAGE_H
