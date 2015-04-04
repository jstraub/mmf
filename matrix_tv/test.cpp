#include "adaptive_camera.h"
#include "SimpleImage.h"
//#include <cuda.h>
#include <iostream>
// Usage:
int main(){
uint width=16;
uint height=32;
float *pEstimate=(float*)malloc(sizeof(float)*width*height);
float *pStdEstimate=(float*)malloc(sizeof(float)*width*height);
SimpleImage img_estimate(width,height,1,pEstimate);
//SimpleImage proposal_std(width,height,1,pStdEstimate);
test_AWGN_resample2(img_estimate.data,img_estimate.data, width,height);

//res_MI.data[0]=0.f;
}

