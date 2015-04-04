/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <mmf/optimizationSO3_vmf.hpp>

float mmf::OptSO3vMF::conjugateGradientPreparation_impl(Matrix3f& R, uint32_t& N)
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

void mmf::OptSO3vMF::conjugateGradientPostparation_impl(Matrix3f& R)
{
//  if(this->t_ > 0)
//  {
//    // init karcher means with rotated version of previous karcher means
//    qKarch_ =  (this->Rprev_*R.transpose())*qKarch_; 
//    // rotations are cameraRworld
//  }
};

/* evaluate cost function for a given assignment of npormals to axes */
float mmf::OptSO3vMF::evalCostFunction(Matrix3f& R)
{
  float c = 0.0f;
  for (uint32_t j=0; j<6; ++j)
  { 
    if(j%2 ==0){
      c -= this->cld_.xSums().col(j).transpose()*R.col(j/2);
    }else{
      c += this->cld_.xSums().col(j).transpose()*R.col(j/2);
    }
  }
  return c;
}
/* compute Jacobian */
void mmf::OptSO3vMF::computeJacobian(Matrix3f&J, Matrix3f& R, float N)
{
  J = Matrix3f::Zero();
#ifndef NDEBUG
  cout<<"xSums_"<<endl<<this->cld_.xSums()<<endl;
#endif
  for (uint32_t j=0; j<6; ++j)
  {
    if(j%2 ==0){
      J.col(j/2) -= this->cld_.xSums().col(j);  // TODO shouldnt it be vice versa?
    }else{
      J.col(j/2) += this->cld_.xSums().col(j);  
    }
  }
//  J /= N;
}
