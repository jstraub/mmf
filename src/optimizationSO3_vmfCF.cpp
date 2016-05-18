/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <mmf/optimizationSO3_vmfCF.hpp>

float mmf::OptSO3vMFCF::computeAssignment(Matrix3f& R, uint32_t& N)
{
  Rot2Device(R);
  float residuals[6]; // for all 6 different axes
  vMFCostFctAssignmentGPU(residuals, d_cost, &N, d_N_, cld_.d_x(),
      d_weights_, cld_.d_z(), d_mu_, pi_.data(), cld_.N());
  float residual = 0.0f;
  for (uint32_t i=0; i<6; ++i) residual +=  residuals[i];
  return residual;
};

float mmf::OptSO3vMFCF::conjugateGradientPreparation_impl(Matrix3f& R, uint32_t& N)
{
//  Timer t0;
  N =0;
  float res0 = this->computeAssignment(R,N)/float(N);
  this->cld_.computeSS();
#ifndef NDEBUG
  cout<<"xSums: "<<endl<<this->cld_.xSums()<<endl;
#endif
//  t0.toctic("----- sufficient statistics");
  return res0; // cost fct value
}

void mmf::OptSO3vMFCF::conjugateGradientPostparation_impl(Matrix3f& R)
{ };

void mmf::OptSO3vMFCF::computeJacobian(Matrix3f&J, Matrix3f& R, float N) 
{ };

float mmf::OptSO3vMFCF::conjugateGradientCUDA_impl(Matrix3f& R, float res0,
    uint32_t n, uint32_t maxIter) {
  Eigen::Matrix3f N = Eigen::Matrix3f::Zero();
  // tauR_*R^T is the contribution of the motion prior between two
  // frames to regularize solution in case data exists only on certain
  // axes
  if (this->t_ >= 1) N += tauR_*R.transpose();
  for (uint32_t j=0; j<6; ++j) { 
    Eigen::Vector3f m = Eigen::Vector3f::Zero();
    m(j/2) = j%2==0?1.:-1.;
    N += m*this->cld_.xSums().col(j).transpose();
  }
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(N,Eigen::ComputeThinU|Eigen::ComputeThinV);
  if (svd.matrixV().determinant()*svd.matrixU().determinant() > 0)
    R = svd.matrixV()*svd.matrixU().transpose();
  else
    R = svd.matrixV()*Eigen::Vector3f(1.,1.,-1.).asDiagonal()*svd.matrixU().transpose();

//  if (R.determinant() < 0.) {
//    //R *= -1.;
////    R = svd.matrixV()*Eigen::Vector3f(1.,1.,-1.).asDiagonal()*svd.matrixU().transpose();
//    std::cout << "determinant of R < 0" << std::endl;
//  }
//  std::cout << R.determinant() << std::endl;
//  std::cout << "N" << std::endl << N << std::endl;
//  std::cout << "R" << std::endl << R << std::endl;
//  std::cout << this->cld_.xSums() << std::endl;
  return (N*R).trace();
}

/* evaluate cost function for a given assignment of npormals to axes */
float mmf::OptSO3vMFCF::evalCostFunction(Matrix3f& R)
{
  float c = 0.0f;
  for (uint32_t j=0; j<6; ++j) { 
    if(j%2 ==0){
      c -= this->cld_.xSums().col(j).transpose()*R.col(j/2);
    }else{
      c += this->cld_.xSums().col(j).transpose()*R.col(j/2);
    }
  }
  return c;
}
