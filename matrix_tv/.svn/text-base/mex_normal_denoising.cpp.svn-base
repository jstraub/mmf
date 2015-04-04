#include <mex.h>
#include <iostream>
#include "normal_smoothing.h"
#include "SimpleImage.h"
#include <string.h>
using namespace std;
#define CHECK_DATA(arr,type,str) if (mxGetClassID(arr)!=type) mexErrMsgTxt(str);
#define CHECK_DATA_DIM(arr,type,width,height,str) if (mxGetClassID(arr)!=type || mxGetN(arr)!=width || mxGetM(arr)!=height) mexErrMsgTxt(str);
//
// Usage:
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    if (nrhs<2) {
        mexErrMsgTxt("Need at least 2 arguments");
    }
CHECK_DATA(prhs[0], mxSINGLE_CLASS,"prhs[0] should be a float");
float *pImage=(float*)mxGetPr(prhs[0]);
uint width=mxGetDimensions(prhs[0])[1];
uint height=mxGetDimensions(prhs[0])[0];
uint depth=mxGetNumberOfDimensions(prhs[0])>2?mxGetDimensions(prhs[0])[2]:1;
if (depth!=3){
    mexErrMsgTxt("depth should be 3");
}
cerr<<"size = "<<depth<<","<<width<<","<<height<<"\n";
float *pImage2=new float[depth*width*height];
CHECK_DATA_DIM(prhs[1], mxSINGLE_CLASS,1,1,"prhs[1] should be a float");
float *pParams=(float*)mxGetPr(prhs[1]);
memcpy((void*)pImage2,(void*)pImage,width*height*depth*sizeof(float));
cerr<<width*height*depth*sizeof(float)<<"\n";
float fidelity=pParams[0];
uint iter=10;
SimpleImage res_MI(height,width,depth,pImage2,false);
normal_smoothing(res_MI,fidelity,iter);
memcpy(pImage,pImage2,depth*width*height*sizeof(float));
delete[] pImage2;

}

