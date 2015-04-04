
#include <stdint.h>
#include <nvidia/helper_cuda.h> 
#include <cuda_pc_helpers.h>


/*
 * compute the xyz images using the inverse focal length invF
 */
//__global__ void depth2xyz(unsigned short* d, float* x, float* y, float* z, 
//    float invF, int w, int h)
//{
//  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
//  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
//  const int id = idx+w*idy;
//
//  if(idx<w && idy<h)
//  {
//    float dd = float(d[id])*0.001f; // convert to mm
//    // have a buffer of nan pixels around the border to prohibit 
//    // the filters to do bad stuff at the corners
//    if ((dd>0.0f)&&( 3<idx && idx<w-3 && 3<idy && idy<h-3)){
//      // TODO directly do the transformation
////      x[id] = dd*float(idx-w/2)*invF;
////      z[id] = -dd*float(idy-h/2)*invF;
////      y[id] = dd;
//      x[id] = -dd*float(idx-w/2)*invF;
//      y[id] = -dd*float(idy-h/2)*invF;
//      z[id] = dd;
//    }else{
//      x[id] = 0.0f/0.0f;
//      y[id] = 0.0f/0.0f;
//      z[id] = 0.0f/0.0f;
//    }
//  }
//}

__global__ void depth2xyz(unsigned short* d, float* x, float* y, 
    float* z, float invF, int w, int h, float *xyz)
{
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if(idx<w && idy<h)
  {
    float dd = float(d[id])*0.001f; // convert to mm
    // have a buffer of nan pixels around the border to prohibit 
    // the filters to do bad stuff at the corners
    if ((0.0f<dd)&&(dd<4.0f)&&( BOARDER_SIZE<idx && idx<w-BOARDER_SIZE 
          && BOARDER_SIZE<idy && idy<h-BOARDER_SIZE)){
      // in combination with the normal computation this gives the right normals
      x[id] = dd*(float(idx)-(w-1.)*0.5)*invF;
      y[id] = dd*(float(idy)-(h-1.)*0.5)*invF;
      z[id] = dd;
    }else{
      x[id] = 0.0f/0.0f;
      y[id] = 0.0f/0.0f;
      z[id] = 0.0f/0.0f;
    }
    if (xyz != NULL){
      xyz[id*4] = x[id];
      xyz[id*4+1] = y[id];
      xyz[id*4+2] = z[id];
    }
  }
}

extern "C" void depth2xyzGPU(unsigned short* d, float* x, float* y, float* z,
    float invF, int w, int h, float *xyz=NULL)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

//  if (xyz==NULL)
//  {
//    depth2xyz<<<blocks, threads>>>(d,x,y,z,invF,w,h);
//    getLastCudaError("depth2xyzGPU() execution failed\n");
//  }else{
    depth2xyz<<<blocks, threads>>>(d,x,y,z,invF,w,h,xyz);
    getLastCudaError("depth2xyzGPU() execution failed\n");
//  }
}


__global__ void depth2float(unsigned short* d, float* d_float,
     int w, int h)
{
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if(idx<w && idy<h)
  {
    float dd = float(d[id])*0.001f; // convert to mm
    if ((dd>0.0f)){
      d_float[id] = dd;
    }else{
      d_float[id] = 0.0f/0.0f;
    }
  }
}
extern "C" void depth2floatGPU(unsigned short* d, float* d_float, 
    int w, int h)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

  depth2float<<<blocks, threads>>>(d,d_float,w,h);
  getLastCudaError("depth2floatGPU() execution failed\n");
}

//#define SQRT2 1.4142135623730951
__global__ void depthFilter(float* d,
     int w, int h)
{
  const float thresh = 0.2; // 5cm
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;
  const int tid = threadIdx.x + blockDim.x * threadIdx.y;

  __shared__ float duSq[8];
  if(tid==0) duSq[0] = 2.;
  if(tid==1) duSq[1] = 1.;
  if(tid==2) duSq[2] = 2.;
  if(tid==3) duSq[3] = 1.;
  if(tid==4) duSq[4] = 2.;
  if(tid==5) duSq[5] = 1.;
  if(tid==6) duSq[6] = 2.;
  if(tid==7) duSq[7] = 1.;

  __syncthreads(); // make sure that ys have been cached

  // filtering according to noise model from file:///home/jstraub/Downloads/Nguyen2012-ModelingKinectSensorNoise.pdf
  if(1<idx && idx<w-1 && 1<idy && idy<h-1)
  {
    float dd = d[id];
    if ((dd>0.0f))
    {
      float invSigSqL = 1.0f/0.5822699462742343; //for theta=30deg //0.8f + 0.035f*theta/(PI*0.5f -theta);
      float invSigSqZ = 1.0f/(0.0012f + 0.0019f*(dd-0.4f)*(dd-0.4f));
      invSigSqZ = invSigSqZ*invSigSqZ;
      float ds[8];
      ds[0] = d[idx-1+w*(idy-1)];
      ds[1] = d[idx  +w*(idy-1)];
      ds[2] = d[idx+1+w*(idy-1)];
      ds[3] = d[idx+1+w*idy];
      ds[4] = d[idx+1+w*(idy+1)];
      ds[5] = d[idx  +w*(idy+1)];
      ds[6] = d[idx-1+w*(idy+1)];
      ds[7] = d[idx-1+w*idy];
      float wSum = 0.0f;
      float dwSum = 0.0f;
#pragma unroll
      for(int32_t i=0; i<8; ++i)
      {
        float dz = fabs(ds[i]-dd);
        float wi = dz < thresh ? expf(-0.5f*(duSq[i]*invSigSqL + dz*dz*invSigSqZ)) : 0.0f;
        wSum += wi;
        dwSum += wi*ds[i];
      }
      d[id] = dwSum/wSum;
    }
  }
}
extern "C" void depthFilterGPU(float* d, int w, int h)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

  depthFilter<<<blocks, threads>>>(d,w,h);
  getLastCudaError("depthFilterGPU() execution failed\n");
}


__global__ void depth2xyzFloat(float* d, float* x, float* y, 
    float* z, float invF, int w, int h, float *xyz)
{
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if(idx<w && idy<h)
  {
    float dd = d[id]; // convert to mm
    // have a buffer of nan pixels around the border to prohibit 
    // the filters to do bad stuff at the corners
    if ((dd>0.0f)&&( BOARDER_SIZE<idx && idx<w-BOARDER_SIZE 
          && BOARDER_SIZE<idy && idy<h-BOARDER_SIZE)){
      // in combination with the normal computation this gives the right normals
      x[id] = dd*float(idx-w/2)*invF;
      y[id] = dd*float(idy-h/2)*invF;
      z[id] = dd;
    }else{
      x[id] = 0.0f/0.0f;
      y[id] = 0.0f/0.0f;
      z[id] = 0.0f/0.0f;
    }
    if (xyz != NULL){
      xyz[id*4] = x[id];
      xyz[id*4+1] = y[id];
      xyz[id*4+2] = z[id];
    }
  }
}

extern "C" void depth2xyzFloatGPU(float* d, float* x, float* y, float* z,
    float invF, int w, int h, float *xyz=NULL)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

  depth2xyzFloat<<<blocks, threads>>>(d,x,y,z,invF,w,h,xyz);
  getLastCudaError("depth2xyzGPU() execution failed\n");
}

inline __device__ float signf(float a)
{
  if (a<0.0f)
    return -1.0f;
  else
    return 1.0f;
//  else
//    return 0.0f;
}

inline __device__ float absf(float a)
{
  return a<0.0f?-a:a;
}

/*
 * derivatives2normals takes pointers to all derivatives and fills in d_n
 * d_n is a w*h*3 array for all three normal components (x,y,z)
 */
__global__ void derivatives2normals(float* d_x, float* d_y, float* d_z, 
    float* d_xu, float* d_yu, float* d_zu, 
    float* d_xv, float* d_yv, float* d_zv, 
    float* d_n, int w, int h)
{
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if(idx<w && idy<h)
  {
    // in combination with the depth to xyz computation this gives the right normals
    float xu=d_xu[id];
    float yu=d_yu[id];
    float zu=d_zu[id];
    float xv=d_xv[id];
    float yv=d_yv[id];
    float zv=d_zv[id];
    float invLenu = 1.0f/sqrtf(xu*xu + yu*yu + zu*zu);
    xu *= invLenu;
    yu *= invLenu;
    zu *= invLenu;
    float invLenv = 1.0f/sqrtf(xv*xv + yv*yv + zv*zv);
    xv *= invLenv;
    yv *= invLenv;
    zv *= invLenv;

    float nx = yu*zv - yv*zu;
    float ny = xv*zu - xu*zv;
    float nz = xu*yv - xv*yu;
    float lenn = sqrtf(nx*nx + ny*ny + nz*nz);
    float sgn = signf(d_x[id]*nx + d_y[id]*ny + d_z[id]*nz)/lenn;
    // normals are pointing away from where the kinect sensor is
    // ie. if pointed at the ceiling the normals will be (0,0,1)
    // the coordinate system is aligned with the image coordinates:
    // z points outward to the front 
    // x to the right (when standing upright on the foot and looking from behind)
    // y down (when standing upright on the foot and looking from behind)


//    if (absf(ny)<0.01f || absf(nx)<0.01f)
//{
//  nx=0.0f/0.0f;
//  ny=0.0f/0.0f;
//  nz=0.0f/0.0f;
//} 

    // the 4th component is always 1.0f - due to PCL conventions!
    d_n[id*X_STEP+X_OFFSET] = nx*sgn;
    d_n[id*X_STEP+X_OFFSET+1] = ny*sgn;
    d_n[id*X_STEP+X_OFFSET+2] = nz*sgn;
    d_n[id*X_STEP+X_OFFSET+3] = 1.0f;
    // f!=f only true for nans
    //d_nGood[id] = ((nx!=nx) | (ny!=ny) | (nz!=nz))?0:1;
  }
}

extern "C" void derivatives2normalsGPU(float* d_x, float* d_y, float* d_z, 
    float* d_xu, float* d_yu, float* d_zu, 
    float* d_xv, float* d_yv, float* d_zv, 
    float* d_n, int w, int h)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

  derivatives2normals<<<blocks, threads>>>(d_x,d_y,d_z,
      d_xu,d_yu,d_zu,
      d_xv,d_yv,d_zv,
      d_n,w,h);

  getLastCudaError("derivatives2normalsGPU() execution failed\n");
}

/*
 * derivatives2normals takes pointers to all derivatives and fills in d_n
 * d_n is a w*h*3 array for all three normal components (x,y,z)
 */
__global__ void derivatives2normalsCleaner(float* d_x, float* d_y, float* d_z, 
    float* d_xu, float* d_yu, float* d_zu, 
    float* d_xv, float* d_yv, float* d_zv, 
    float* d_n, int w, int h)
{
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if(idx<w && idy<h)
  {
    // in combination with the depth to xyz computation this gives the right normals
    float xu=d_xu[id];
    float yu=d_yu[id];
    float zu=d_zu[id];
    float xv=d_xv[id];
    float yv=d_yv[id];
    float zv=d_zv[id];
    float invLenu = 1.0f/sqrtf(xu*xu + yu*yu + zu*zu);
    xu *= invLenu;
    yu *= invLenu;
    zu *= invLenu;
    float invLenv = 1.0f/sqrtf(xv*xv + yv*yv + zv*zv);
    xv *= invLenv;
    yv *= invLenv;
    zv *= invLenv;

    float nx = 0.;
    float ny = 0.;
    float nz = 0.;
    float sgn = 1.;
    if (invLenu < 1./0.04 || invLenv < 1./0.04 )
    {
      nx=0.0f/0.0f;
      ny=0.0f/0.0f;
      nz=0.0f/0.0f;
    } else {
      nx = yu*zv - yv*zu;
      ny = xv*zu - xu*zv;
      nz = xu*yv - xv*yu;
      float lenn = sqrtf(nx*nx + ny*ny + nz*nz);
      sgn = signf(d_x[id]*nx + d_y[id]*ny + d_z[id]*nz)/lenn;
      // normals are pointing away from where the kinect sensor is
      // ie. if pointed at the ceiling the normals will be (0,0,1)
      // the coordinate system is aligned with the image coordinates:
      // z points outward to the front 
      // x to the right (when standing upright on the foot and looking from behind)
      // y down (when standing upright on the foot and looking from behind)
    }


    // the 4th component is always 1.0f - due to PCL conventions!
    d_n[id*X_STEP+X_OFFSET] = nx*sgn;
    d_n[id*X_STEP+X_OFFSET+1] = ny*sgn;
    d_n[id*X_STEP+X_OFFSET+2] = nz*sgn;
    d_n[id*X_STEP+X_OFFSET+3] = 1.0f;
    // f!=f only true for nans
    //d_nGood[id] = ((nx!=nx) | (ny!=ny) | (nz!=nz))?0:1;
  }
}

extern "C" void derivatives2normalsCleanerGPU(float* d_x, float* d_y, float* d_z, 
    float* d_xu, float* d_yu, float* d_zu, 
    float* d_xv, float* d_yv, float* d_zv, 
    float* d_n, int w, int h)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

  derivatives2normalsCleaner<<<blocks, threads>>>(d_x,d_y,d_z,
      d_xu,d_yu,d_zu,
      d_xv,d_yv,d_zv,
      d_n,w,h);

  getLastCudaError("derivatives2normalsGPU() execution failed\n");
}

__device__ inline float square(float a )
{ return a*a;}


__global__ void weightsFromCov(float* z, float* weights,
    float theta, float invF, int w, int h)
{
// according to ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6375037
// weights are the inverse of the determinant of the covariance of the noise ellipse
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if(idx<w && idy<h)
  {
    float z_i = z[id];
    //float ang = theta/180.0f*M_PI;
    float sigma_z = 0.0012f + 0.019f * square(z_i-0.4f);
    //float sigma_l = (0.8f + 0.035f*ang/(M_PI*0.5f-ang))*z_i*invF;
    weights[id] = 1.0f/sigma_z;
    //weights[id] = 1.0f/(square(sigma_z)+2.0f*square(sigma_l));
  }
}

extern "C" void weightsFromCovGPU(float* z, float* weights,
    float theta, float invF, int w, int h)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

    weightsFromCov<<<blocks, threads>>>(z,weights,theta,invF,w,h);
    getLastCudaError("depth2xyzGPU() execution failed\n");
}


__global__ void weightsFromArea(float* z, float* weights,
    int w, int h)
{
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;
  const int id = idx+w*idy;

  if(idx<w && idy<h)
  {
    // weight proportial to area that the pixel i observes at distance z_i
    // the area = z_i^2/f^2 but f is constant so we dont need divide by it.
    weights[id] = square(z[id]); 
  }
}

extern "C" void weightsFromAreaGPU(float* z, float* weights,
    int w, int h)
{
  dim3 threads(16,16,1);
  dim3 blocks(w/16 + (w%16>0?1:0),h/16 + (h%16>0?1:0),1);

  weightsFromArea<<<blocks, threads>>>(z,weights,w,h);
  getLastCudaError("weightsFromAreaGPU() execution failed\n");
}
