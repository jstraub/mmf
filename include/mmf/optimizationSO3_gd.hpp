/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <stdint.h>
#include <string>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <unsupported/Eigen/MatrixFunctions>

// CUDA runtime
#include <cuda_runtime.h>
// Utilities and system includes
//#include <mmf/defines.h>
#include <nvidia/helper_cuda.h>

#include <jsCore/clDataGpu.hpp>
#include <jsCore/timer.hpp>

#include <mmf/optimizationSO3.hpp>
#include <manifold/SO3.h>

using namespace Eigen;
using namespace std;

extern void directSquaredAngleCostFctGPU(float *h_cost, float *d_cost,
    float *d_x, float* d_weights, uint32_t *d_z, float *d_mu,
    int N);

extern void directSquaredAngleCostFctAssignmentGPU(float *h_cost, float *d_cost,
  uint32_t *h_W, uint32_t *d_W, float *d_x, float* d_weights, 
  uint32_t *d_z, float *d_mu, int N);

extern void directSquaredAngleCostFctJacobianGPU(float *h_J, float *d_J,
    float *d_x, float *d_weights, uint32_t *d_z, float *d_mu, int N);

extern void meanInTpS2GPU(float *h_p, float *d_p, float *h_mu_karch,
    float *d_mu_karch, float *d_q, uint32_t *d_z, float* d_weights, int N);

extern void sufficientStatisticsOnTpS2GPU(float *h_p, float *d_p, float
    *h_Rnorths, float *d_Rnorths, float *d_q, uint32_t *d_z ,int N, float
    *h_SSs, float *d_SSs);

extern void loadRGBvaluesForMFaxes();

namespace mmf{

/// Implements Conjugate Gradient Optimization on SO3 using GPU
/// optimizations
class OptSO3GD : public OptSO3
{
public:
  // t_max and dt define the number linesearch steps, and how fine
  // grained to search 
  OptSO3GD(float sigma, float t_max = 1.0f, float dt = 0.1f, 
      float *d_weights =NULL)
    : OptSO3(sigma,t_max,dt,d_weights), thr_(1.e-7), c_(0.1), t_(0.5)
  {};

  virtual ~OptSO3GD() {};

protected:
  SO3f theta_;
  float thr_; // threshold for gradient descent
  float c_;
  float t_;

  void ComputeJacobian(const SO3f& theta, uint32_t N, Eigen::Vector3f* J, float* f);
  void LineSearch(uint32_t N, Eigen::Vector3f* J, float* f);

  virtual float conjugateGradientPreparation_impl(Matrix3f& R, uint32_t& N);
  virtual float conjugateGradientCUDA_impl(Matrix3f& R, float res0, uint32_t N, uint32_t maxIter=100);
  virtual void conjugateGradientPostparation_impl(Matrix3f& R){;};
  /* recompute assignment based on rotation R and return residual as well */
  virtual float computeAssignment(Matrix3f& R, uint32_t& N);
  /* mainly init GPU arrays */
  virtual void init();
};
}
