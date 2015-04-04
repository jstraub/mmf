#pragma once
#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include "SimpleProfiler.h"
#include <math.h>
#include <limits.h>

struct RandStruct{
uint seed; 
float buffer1;
float buffer2;  
bool pos;
__device__ RandStruct(uint _seed):seed(_seed),pos(true){
buffer1=0.f;
buffer2=0.f;
}
};

__device__ uint rand_wanghash(RandStruct& str)
{
    str.seed = (str.seed ^ 61) ^ (str.seed >> 16);
    str.seed *= 9;
    str.seed = str.seed ^ (str.seed >> 4);
    str.seed *= 0x27d4eb2d;
    str.seed = str.seed ^ (str.seed >> 15);
return str.seed;
}

__device__ float randn_wanghash(RandStruct& str)
{
    if (str.pos){
	str.pos=false;
    float u1=((float)rand_wanghash(str)/((float)UINT_MAX));
    float u2=((float)rand_wanghash(str)/((float)UINT_MAX));
float lu1=__logf(u1);
float R=sqrtf(-2*lu1);
float theta=2*M_PI*u2;
float tmp1;
float tmp2;
__sincosf(theta,&tmp1,&tmp2);
	str.buffer1=tmp1*R;
	str.buffer2=tmp2*R;
	return str.buffer1;
    }else {
    str.pos=true;
    }
    return str.buffer2;
}


