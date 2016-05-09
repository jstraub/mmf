/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <mmf/optimizationSO3_vmfCF.hpp>

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
  for (uint32_t j=0; j<6; ++j) { 
    Eigen::Vector3f m = Eigen::Vector3f::Zero();
    m(j/2) = j%2==0?-1.:1.;
    N += m*this->cld_.xSums().col(j).transpose();
  }
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(N,Eigen::ComputeThinU|Eigen::ComputeThinV);
  R = svd.matrixV()*svd.matrixU().transpose();
  if (R.determinant() < 0.) R *= -1.;
//  std::cout << "N" << std::endl << N << std::endl;
//  std::cout << "R" << std::endl << R << std::endl;
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
