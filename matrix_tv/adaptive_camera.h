#include <iostream>
#include <stdio.h>
#include "SimpleProfiler.h"
#include <math.h>
#include <limits.h>
#include "SimpleImage.h"
void active_camera_estimation(SimpleImage& img_estimate,SimpleImage& observation_std,SimpleImage& proposal_std,SimpleImage& res_MI, double min_image_value, double max_image_value, double image_step);
void test_AWGN_resample(float *dest,float *src, uint width, uint height);
void test_AWGN_resample2(float *dest,float *src, uint width,uint height);
