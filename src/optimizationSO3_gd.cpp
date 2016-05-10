/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <mmf/optimizationSO3_gd.hpp>

void mmf::OptSO3GD::LineSearch(uint32_t N, Eigen::Vector3f* J, float* f) {
  float delta = 1.;
  SO3f thetaNew = theta_;
  ComputeJacobian(thetaNew, N, J, f);
  float fNew = *f;
  Eigen::Vector3f d = -(*J)/J->norm();
//  std::cout << "\tJ=" << J->transpose() << std::endl
//    << "\td=" << d.transpose() << std::endl;
  float m = J->dot(d);
  while (*f-fNew < -c_*m*delta && delta > 1e-16) {
    delta *= t_;
    thetaNew = theta_+delta*d;
    //std::cout << thetaNew << std::endl;
    ComputeJacobian(thetaNew, N, NULL, &fNew);
//    std::cout << *f-fNew << " <? " << -c_*m*delta 
//      << "\tfNew=" << fNew << "\tdelta=" << delta << std::endl;
  }
  *J = delta*d;
  *f = fNew;
}

void mmf::OptSO3GD::ComputeJacobian(const SO3f& theta, uint32_t N,
    Eigen::Vector3f* J, float* f) {
  Eigen::Matrix3f R = theta.matrix();

  Rot2Device(R);
  if (f) {
    float residuals[6]; // for all 6 different axes
    directSquaredAngleCostFctGPU(residuals, d_cost, cld_.d_x(),
        d_weights_, cld_.d_z(), d_mu_, cld_.N());
    *f = 0.0f;
    for (uint32_t i=0; i<6; ++i)  *f +=  residuals[i];
    *f /= -N;
  }
  if (J) {
    *J = Vector3f::Zero();
    directSquaredAngleCostFctJacobianGPU(J->data(), d_J,
        cld_.d_x(), d_weights_, cld_.d_z(), d_mu_, cld_.N());
    *J /= -N;
  }
}

void mmf::OptSO3GD::init()
{
  checkCudaErrors(cudaMalloc((void **)&d_cost, 6*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_J, 3*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_mu_, 6*3*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_N_, sizeof(int)));
  loadRGBvaluesForMFaxes();
};


float mmf::OptSO3GD::conjugateGradientPreparation_impl(Matrix3f& R, uint32_t& N)
{
  N = 0;
  return computeAssignment(R,N)/float(N);
};

float mmf::OptSO3GD::conjugateGradientCUDA_impl(Matrix3f& R, float res0,
    uint32_t N, uint32_t maxIter)
{
  theta_ = SO3f(R);
  SO3f thetaPrev(R);
  Eigen::Vector3f J = Eigen::Vector3f::Zero(); 
  float fPrev = 1e12;
  float f = res0;
//  T delta = 1e-2;
  uint32_t it=0;
  while((fPrev-f)/fabs(f) > thr_ && it < maxIter) {
    fPrev = f;
    LineSearch(N, &J, &f);
//    ComputeJacobian(theta_, &J, &f);
    thetaPrev = theta_;
    theta_ += J;
    std::cout << "@" << it << " f=" << f << " df/f=" <<
      (fPrev-f)/fabs(f) << std::endl;
    ++it;
  }
  if (f > fPrev) {
    theta_ = thetaPrev;
  }
  R = theta_.matrix();
}

float mmf::OptSO3GD::computeAssignment(Matrix3f& R, uint32_t& N)
{
  Rot2Device(R);
  float residuals[6]; // for all 6 different axes
  directSquaredAngleCostFctAssignmentGPU(residuals, d_cost, &N, d_N_, cld_.d_x(),
      d_weights_, cld_.d_z(), d_mu_, cld_.N());
  float residual = 0.0f;
  for (uint32_t i=0; i<6; ++i) residual +=  residuals[i];
  return residual;
};

