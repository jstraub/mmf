/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <stdint.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <unsupported/Eigen/MatrixFunctions>

// CUDA runtime
#include <cuda_runtime.h>
// Utilities and system includes
//#include <helper_functions.h>
#include <mmf/helper_cuda.h>

#include <mmf/defines.h>
#include <mmf/sphereSimple.hpp>
#include <mmf/optimizationSO3.hpp>

using namespace Eigen;

namespace mmf{

class OptSO3Approx : public OptSO3
{
  public:
  OptSO3Approx(float sigma, float t_max = 5.0f, float dt = 0.05f, 
      float *d_weights =NULL):
    OptSO3(sigma,t_max, dt, d_weights), Ss_(6,Matrix2f::Identity())
  {
    checkCudaErrors(cudaMalloc((void **)&d_mu_karch_, 6*4*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p_, 6*3*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_SSs_, 6*7*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Rnorths_, 6*6*sizeof(float)));
    Matrix2f S = Matrix2f::Identity()*(15.0*M_PI/180.0)*(15.0*M_PI/180.0);
    invSigma_ = S.inverse();
//    t_max_ = 5.0f;
//    dt_ = 0.05f; // 0.1
  };

  virtual ~OptSO3Approx()
  {
    checkCudaErrors(cudaFree(d_mu_karch_));
    checkCudaErrors(cudaFree(d_p_));
    checkCudaErrors(cudaFree(d_SSs_));
    checkCudaErrors(cudaFree(d_Rnorths_));
  };
  Matrix<float,3,6> karcherMeans(const Matrix<float,3,6>& p0, float
      thresh, uint32_t maxIter);
  void computeSuffcientStatistics();
  Matrix<float,3,6> qKarch_; // karcher means for all axes


protected:
  SphereSimple S2_;
  float *d_mu_karch_, *d_p_;
  float *d_Rnorths_, *d_SSs_;
  Matrix2f invSigma_; // inverse of the sigma in the tangent space
  Matrix<float,1,6> Ns_; // number of normals for each axis
  Matrix<float,2,6> xSums_; // sum over all vectors for each axis
  vector<Matrix2f> Ss_; // sum over outer products of data in tangent spaces


  virtual void conjugateGradientPostparation_impl(Matrix3f& R);
  virtual float conjugateGradientPreparation_impl(Matrix3f& R, uint32_t& N);
  /* evaluate cost function for a given assignment of npormals to axes */
  virtual float evalCostFunction(Matrix3f& R);
  /* compute Jacobian */
  virtual void computeJacobian(Matrix3f&J, Matrix3f& R, float N);
  /* recompute assignment based on rotation R and return residual as well */
  //float computeAssignment(Matrix3f& R, int& N);
  /* compute all karcher means (for 6 axis) */
  Matrix<float,3,6> meanInTpS2_GPU(Matrix<float,3,6>& p); 

  virtual void init() {};
};

}
