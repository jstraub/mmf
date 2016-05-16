/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <stdint.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <unsupported/Eigen/MatrixFunctions>

// CUDA runtime
#include <cuda_runtime.h>
// Utilities and system includes
//#include <helper_functions.h>
#include <nvidia/helper_cuda.h>

//#include <mmf/defines.h>
#include <mmf/sphereSimple.hpp>
#include <mmf/optimizationSO3.hpp>
#include <jsCore/vmf.h>

using namespace Eigen;

extern void MMFvMFCostFctAssignmentGPU(float *h_cost, float *d_cost,
  uint32_t *h_W, uint32_t *d_W, float *d_x, float* d_weights, 
  uint32_t *d_z, float *d_mu, float*d_pi, int N, int K);

namespace mmf{

// closed form solution for the vMF cost function.
class OptSO3MMFvMF : public OptSO3
{
  public:
  OptSO3MMFvMF(uint32_t K, float *d_weights =NULL):
    OptSO3(1.,1.,0.1,d_weights), 
    Rs_(K, Eigen::Matrix3f::Identity()),
    pi_(K*6), taus_(Eigen::VectorXf::Ones(K*6))
  { 
    // overwrite cld
    cld_ = jsc::ClDataGpu<float>(3,6*K);
    if(d_cost) checkCudaErrors(cudaFree(d_cost));
    if(d_mu_)  checkCudaErrors(cudaFree(d_mu_));
    if(d_N_) checkCudaErrors(cudaFree(d_N_));
    init();
  };

  virtual ~OptSO3MMFvMF() { };
  uint32_t K() {return Rs_.size();};
  virtual std::vector<Eigen::Matrix3f> GetRs() { return Rs_; };

protected:
  std::vector<Eigen::Matrix3f> Rs_;
  jsc::GpuMatrix<float> pi_;
  Eigen::VectorXf taus_;

  virtual float computeAssignment(uint32_t& N);

  virtual float conjugateGradientCUDA_impl(Matrix3f& R, float res0,
    uint32_t N, uint32_t maxIter);
  virtual void conjugateGradientPostparation_impl(Matrix3f& R);
  virtual float conjugateGradientPreparation_impl(Matrix3f& R, uint32_t& N);
  /* evaluate cost function for a given assignment of npormals to axes */
  virtual float evalCostFunction(Matrix3f& R);
  /* compute Jacobian */
  virtual void computeJacobian(Matrix3f&J, Matrix3f& R, float N);
  virtual void init();

  /* copy rotation to device */
  void Rot2Device();
};

}
