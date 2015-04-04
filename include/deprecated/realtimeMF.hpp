/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#ifndef REALTIME_MF_HPP_INCLUDED
#define REALTIME_MF_HPP_INCLUDED
#include <root_includes.hpp>
#include <defines.h>
#include <signal.h>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/thread.hpp>

#include <Eigen/Dense>

// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <nvidia/helper_functions.h>
#include <nvidia/helper_cuda.h>

#include <cuda_pc_helpers.h>
#include <convolutionSeparable_common.h>
#include <convolutionSeparable_common_small.h>
#include <timer.hpp>
#include <optimizationSO3.hpp>
#include <optimizationSO3_approx.hpp>
#include <cv_helpers.hpp>
#include <pcl_helpers.hpp>

using namespace Eigen;

class RealtimeMF
{
  public:
    RealtimeMF(std::string mode) : 
      dtNormals_(0.0), dtPrep_(0.0), dtCG_(0.0), dtTotal_(0.0),
      residual_(0.0), D_KL_(0.0), nCGIter_(10),
      update(false), updateRGB_(false), 
      mode_(mode), cuda_ready(false)
  {
    float theta = 0.1;
    kRw_<< 1,0,0,
      0,cos(theta),-sin(theta),
      0,sin(theta),cos(theta);

    kRw_ << 0.99942, -0.0300857, -0.0159247,
         0.0211853,   0.915914 , -0.400816,
         0.0266444,   0.400246 ,   0.91602;
  }

    ~RealtimeMF();

    /* process a depth image of size w*h and return rotation estimate
     * mfRc
     */
    Matrix3f depth_cb(const uint16_t *data, int w,int h);

    void getAxisAssignments();

    void visualizePc();

    void run();

    double dtNormals_;
    double dtPrep_;
    double dtCG_;
    double dtTotal_;
    double residual_, D_KL_;
    uint32_t nCGIter_;

  protected:
    // sets up the memory on the GPU device
    void prepareCUDA(uint32_t w,uint32_t h);
    // computes derivatives of d_x, d_y, d_z on GPU
    void computeDerivatives(uint32_t w,uint32_t h);
    // smoothes the derivatives (iterations) times
    void smoothDerivatives(uint32_t iterations, uint32_t w,uint32_t h);
    // convert the d_depth into smoothed xyz 
    void depth2smoothXYZ(float invF, uint32_t w,uint32_t h);

    bool update, updateRGB_;
    boost::mutex updateModelMutex;


    std::string mode_;

    bool cuda_ready;
    float* d_x, *d_y, *d_z;
    float* d_xu, *d_yu, *d_zu;
    float* d_xv, *d_yv, *d_zv;
    float *d_n, *d_xyz;
    float *a,*b,*c; // for intermediate computations
    float *d_weights;
    uint16_t* d_depth;
    float h_sobel_dif[KERNEL_LENGTH_S];
    float h_sobel_sum[KERNEL_LENGTH_S];
    float h_kernel_avg[KERNEL_LENGTH];
    float *h_n,*h_dbg,*h_xyz;
    

    Matrix3f kRw_;

    OptSO3 *optSO3_;
    pcl::PointCloud<pcl::PointXYZRGB> n_;
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr n_cp_;

    pcl::PointCloud<pcl::PointXYZ> pc_;
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr pc_cp_;

    cv::Mat rgb_;

    Vector3d vicon_t_; 
    Quaterniond vicon_q_; 
    Matrix3d kRv;
    Vector3d dt0;

    virtual void run_impl() = 0;
    virtual void run_cleanup_impl() = 0;
};

#endif
