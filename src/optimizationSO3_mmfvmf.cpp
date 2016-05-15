/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <mmf/optimizationSO3_mmfvmf.hpp>

void mmf::OptSO3MMFvMF::init()
{
  checkCudaErrors(cudaMalloc((void **)&d_cost, K()*6*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_J, 3*3*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_mu_, K()*6*3*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_N_, sizeof(int)));
  loadRGBvaluesForMFaxes();
};


float mmf::OptSO3MMFvMF::computeAssignment(uint32_t& N)
{
  Rot2Device();
  Eigen::VectorXf residuals = Eigen::VectorXf::Zero(K()*6);
  MMFvMFCostFctAssignmentGPU((float*)residuals.data(), d_cost, &N, d_N_, cld_.d_x(),
      d_weights_, cld_.d_z(), d_mu_, cld_.N(), K());
  return residuals.sum();
};

float mmf::OptSO3MMFvMF::conjugateGradientPreparation_impl(Matrix3f& R, uint32_t& N)
{
  // TODO: ignoring R - cleanup
//  Timer t0;
  N = 0;
  float res0 = this->computeAssignment(N)/float(N);
  this->cld_.computeSS();
#ifndef NDEBUG
  cout<<"xSums: "<<endl<<this->cld_.xSums()<<endl;
#endif
  cout<<"xSums: "<<endl<<this->cld_.xSums()<<endl;
  cout<<"counts: "<<endl<<this->cld_.counts().transpose()<<endl;
//  t0.toctic("----- sufficient statistics");
  return res0; // cost fct value
}

void mmf::OptSO3MMFvMF::conjugateGradientPostparation_impl(Matrix3f& R)
{ };

void mmf::OptSO3MMFvMF::computeJacobian(Matrix3f&J, Matrix3f& R, float N) 
{ };

float mmf::OptSO3MMFvMF::conjugateGradientCUDA_impl(Matrix3f& R, float res0,
    uint32_t n, uint32_t maxIter) {
  float f = 0;
  for (uint32_t k=0; k<K(); ++k) {
    Eigen::Matrix3f N = Eigen::Matrix3f::Zero();
    for (uint32_t j=k*6; j<6*(k+1); ++j) { 
      Eigen::Vector3f m = Eigen::Vector3f::Zero();
      m(j/2) = j%2==0?-1.:1.;
      N += m*this->cld_.xSums().col(j).transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(N,Eigen::ComputeThinU|Eigen::ComputeThinV);
    Rs_[k] = svd.matrixV()*svd.matrixU().transpose();
    if (Rs_[k].determinant() < 0.) Rs_[k] *= -1.;
    f += (N*R).trace();
    std::cout << Rs_[k] << std::endl;
  }
//  std::cout << "N" << std::endl << N << std::endl;
//  std::cout << "R" << std::endl << R << std::endl;
  return f;
}

/* evaluate cost function for a given assignment of npormals to axes */
float mmf::OptSO3MMFvMF::evalCostFunction(Matrix3f& R)
{
  float c = 0.0f;
  for (uint32_t j=0; j<6*K(); ++j) { 
    if(j%2 ==0){
      c -= this->cld_.xSums().col(j).transpose()*Rs_[j/6].col(j/2);
    }else{
      c += this->cld_.xSums().col(j).transpose()*Rs_[j/6].col(j/2);
    }
  }
  return c;
}

void mmf::OptSO3MMFvMF::Rot2Device()
{
  Eigen::MatrixXf mu(3,K()*6);
  for(uint32_t k=0; k<6*K(); ++k){
    int j = k/2; // which of the rotation columns does this belong to
    float sign = (- float(k%2) +0.5f)*2.0f; // sign of the axis
    mu(k) = sign*Rs_[k/6](0,j);
    mu(k+6*K()) = sign*Rs_[k/6](1,j);
    mu(k+12*K()) = sign*Rs_[k/6](2,j);
  }
  checkCudaErrors(cudaMemcpy(d_mu_, (float*)mu.data(), 3*6*K()*sizeof(float),
        cudaMemcpyHostToDevice));
}
