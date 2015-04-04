#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "SimpleProfiler.h"
#include <math.h>
#include <limits.h>
#include "gpu_random.h"
#include "fixed_matrix.h"
#include "fixed_vector.h"

class Camera{
   FixedMatrix<float,3,3> K;
   FixedMatrix<float,3,3> R;
   FixedMatrix<float,3,1> t;
   FixedMatrix<float,3,4> P;
   void project(FixedMatrix<float,3,1> X3,FixedMatrix<float,2,1>&X2){
}
};

__global__ void compute_MI_proposal(Camera &cam,Camera &proj){
    int id_x = blockIdx.x*blockDim.x+threadIdx.x;
    int id_y = blockIdx.y*blockDim.y+threadIdx.y;
    float x=id_x;
    float y=id_y;
    FixedVector<float,2> xcam;
    FixedVector<float,3> x3;
    FixedVector<float,2> xproj;
    cam.backproject(xcam,x3);
    proj.project(x3,xproj);
    
}


int main(){
}
