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
//#include <mmf/timer.hpp>

using namespace Eigen;

namespace mmf{

// closed form solution for the vMF cost function.
class OptSO3vMFCF : public OptSO3
{
  public:
  OptSO3vMFCF(float *d_weights =NULL):
    OptSO3(sigma,t_max, dt, d_weights)
  {
//    t_max_ = 5.0f;
//    dt_ = 0.05f; // 0.1
  };

  virtual ~OptSO3vMFCF() { };

  virtual double conjugateGradientCUDA(Matrix3f& R, uint32_t maxIter=0);

protected:

  virtual void conjugateGradientPostparation_impl(Matrix3f& R);
  virtual float conjugateGradientPreparation_impl(Matrix3f& R, uint32_t& N);
  /* evaluate cost function for a given assignment of npormals to axes */
  virtual float evalCostFunction(Matrix3f& R);
  /* compute Jacobian */
  virtual void computeJacobian(Matrix3f&J, Matrix3f& R, float N);
  virtual void init() {};
};

}
