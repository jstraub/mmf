#include <mex.h>
#include "adaptive_camera.h"
#include "SimpleImage.h"
#include <string.h>
#define CHECK_DATA(arr,type,str) if (mxGetClassID(arr)!=type) mexErrMsgTxt(str);
#define CHECK_DATA_DIM(arr,type,width,height,str) if (mxGetClassID(arr)!=type || mxGetN(arr)!=width || mxGetM(arr)!=height) mexErrMsgTxt(str);


//
// Usage:
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    if (nrhs<4) {
        mexErrMsgTxt("Need at least 4 arguments");
    }
CHECK_DATA(prhs[0], mxSINGLE_CLASS,"prhs[0] should be a float");
float *pEstimate=(float*)mxGetPr(prhs[0]);
uint width=mxGetN(prhs[0]);
uint height=mxGetM(prhs[0]);
CHECK_DATA_DIM(prhs[1], mxSINGLE_CLASS,width,height,"prhs[1] should be a float");
CHECK_DATA_DIM(prhs[2], mxSINGLE_CLASS,width,height,"prhs[2] should be a float");
float *pStdEstimate=(float*)mxGetPr(prhs[1]);
float *pObservationStd=(float*)mxGetPr(prhs[2]);
float *pResMI=(float *)mxGetPr(prhs[3]);
float *pResMI2=new float[width*height];
memcpy(pResMI2,pResMI,width*height*sizeof(float));
double min_img=-1;
double max_img=1;
double step_img=0.001;
SimpleImage img_estimate(width,height,1,0);
img_estimate.allocate_2D_image(width,height,pEstimate);
SimpleImage proposal_std(width,height,1,0);
proposal_std.allocate_2D_image(width,height,pStdEstimate);
SimpleImage observation_std(width,height,1,0);
//observation_std.allocate_2D_image(width,height,pObservationStd,false);
SimpleImage res_MI(width,height,1,pResMI2,false);
//pResMI2[0]=0;
active_camera_estimation(img_estimate, observation_std, proposal_std,res_MI,min_img,max_img,step_img);
memcpy(pResMI,pResMI2,width*height*sizeof(float));
//cerr<<"res_MI: "<<res_MI.data[0]<<"\n";
//free(pResMI2);
//pResMI[0]=0.f;
//res_MI.data[0]=0.f;
}

