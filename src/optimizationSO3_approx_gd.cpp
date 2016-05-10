/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <mmf/optimizationSO3_approx_gd.hpp>

float mmf::OptSO3ApproxGD::conjugateGradientPreparation_impl(Matrix3f& R, uint32_t& N)
{
//  Timer t0;
  N =0;
  float res0 = this->computeAssignment(R,N)/float(N);
  if(this->t_ == 0){
   // init karcher means with columns of the rotation matrix (takes longer!)
  qKarch_ << R.col(0),-R.col(0),R.col(1),-R.col(1),R.col(2),-R.col(2);
//    // init karcher means with rotated version of previous karcher means
//    qKarch_ =  (this->Rprev_*R.transpose())*qKarch_; 
  }
  qKarch_ = karcherMeans(qKarch_, 5.e-5, 10);
//  t0.toctic("----- karcher mean");
  // compute a rotation matrix from the karcher means (not necessary)
  Matrix3f Rkarch;
  Rkarch.col(0) =  qKarch_.col(0);
  Rkarch.col(1) =  qKarch_.col(2) - qKarch_.col(2).dot(qKarch_.col(0))
    *qKarch_.col(0);
  Rkarch.col(1) /= Rkarch.col(1).norm();
  Rkarch.col(2) = Rkarch.col(0).cross(Rkarch.col(1));
#ifndef NDEBUG
  cout<<"R: "<<endl<<R<<endl;
  cout<<"Rkarch: "<<endl<<Rkarch<<endl<<"det(Rkarch)="<<Rkarch.determinant()<<endl;
#endif
//  t0.tic();
  computeSuffcientStatistics();
//  t0.toctic("----- sufficient statistics");
  return res0; // cost fct value
}

void mmf::OptSO3ApproxGD::conjugateGradientPostparation_impl(Matrix3f& R)
{ };

/* evaluate cost function for a given assignment of npormals to axes */
float mmf::OptSO3ApproxGD::evalCostFunction(Matrix3f& R)
{
  float c = 0.0f;
  for (uint32_t j=0; j<6; ++j)
  { 
    const float dot = max(-1.0f,min(1.0f,(qKarch_.col(j).transpose() * R.col(j/2))(0)));
    if(j%2 ==0){
      c += Ns_(j) * acos(dot)* acos(dot);
    }else{
      c += Ns_(j) * acos(-dot)* acos(-dot);
    }
  }
  return c;
}

float mmf::OptSO3ApproxGD::conjugateGradientCUDA_impl(Matrix3f& R, float res0,
    uint32_t n, uint32_t maxIter) {
  theta_ = SO3f(R);
  SO3f thetaPrev(R);
  Eigen::Vector3f J = Eigen::Vector3f::Zero(); 
  float fPrev = 1e12;
  float f = res0;
//  T delta = 1e-2;
  uint32_t it=0;
  while((fPrev-f)/fabs(f) > thr_ && it < maxIter) {
    fPrev = f;
    LineSearch(&J, &f);
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

/* compute Jacobian */
void mmf::OptSO3ApproxGD::ComputeJacobian(const SO3f& theta,
    Eigen::Vector3f* J, float* f) {
  Eigen::Matrix3f R = theta.matrix();
#ifndef NDEBUG
  cout<<"qKarch"<<endl<<qKarch_<<endl;
  cout<<"xSums_"<<endl<<xSums_<<endl;
  cout<<"Ns_"<<endl<<Ns_<<endl;
#endif
  if(J) *J = Vector3f::Zero();
  if(f) *f = 0.;
  for (uint32_t j=0; j<6; ++j) { 
    Eigen::Vector3f m = Eigen::Vector3f::Zero();
    m(j/2) = j%2==0?-1.:1.;
    float dot = max(-1.0f,min(1.0f,qKarch_.col(j).dot(R*m)));
    float eps = acos(dot);
    if (f) *f += Ns_(j)*eps*eps;
    if (J) {
      float factor = 0.;
      if(-0.99< dot && dot < 0.99)
        factor = -(2.*Ns_(j)*eps)/(sqrt(1.f-dot*dot));
      else if(dot >= 0.99) { 
        // taylor series around 0.99 according to Mathematica
        factor = -2.*Ns_(j)*(1.0033467240646519 - 0.33601724502488395
            *(-0.99 + dot) + 0.13506297338381046* (-0.99 + dot)*(-0.99 + dot));
      }else if (dot <= -0.99) {
        factor = -2.*Ns_(j)*(21.266813135156017 - 1108.2484926534892*(0.99
              + dot) + 83235.29739487475*(0.99 + dot)*(0.99 + dot));
      }
//      std::cout << "factor " << factor << std::endl;
      for (uint32_t k=0; k<3; ++k)
        (*J)(k) += factor*qKarch_.col(j).dot(SO3f::G(k)*R*m);
    }
  }
  if (f) *f /= -Ns_.sum();
  if (J) *J /= -Ns_.sum();
}

void mmf::OptSO3ApproxGD::LineSearch(Eigen::Vector3f* J, float* f) {
  float delta = 1.;
  SO3f thetaNew = theta_;
  ComputeJacobian(thetaNew, J, f);
  float fNew = *f;
  Eigen::Vector3f d = -(*J)/J->norm();
//  std::cout << "\tJ=" << J->transpose() << std::endl
//    << "\td=" << d.transpose() << std::endl;
  float m = J->dot(d);
  while (*f-fNew < -c_*m*delta && delta > 1e-16) {
    delta *= t_;
    thetaNew = theta_+delta*d;
    //std::cout << thetaNew << std::endl;
    ComputeJacobian(thetaNew, NULL, &fNew);
//    std::cout << *f-fNew << " <? " << -c_*m*delta 
//      << "\tfNew=" << fNew << "\tdelta=" << delta << std::endl;
  }
  *J = delta*d;
  *f = fNew;
}

Matrix<float,3,6> mmf::OptSO3ApproxGD::meanInTpS2_GPU(Matrix<float,3,6>& p)
{
  Matrix<float,4,6> mu_karch = Matrix<float,4,6>::Zero();
  float *h_mu_karch = mu_karch.data();
  float *h_p = p.data();
  meanInTpS2GPU(h_p, d_p_, h_mu_karch, d_mu_karch_, this->cld_.d_x(),
      this->cld_.d_z(), d_weights_, this->cld_.N());
  Matrix<float,3,6> mu = mu_karch.topRows(3);
  for(uint32_t i=0; i<6; ++i)
    if(mu_karch(3,i) >0)
      mu.col(i) /= mu_karch(3,i);
  return mu;
}

Matrix<float,3,6> mmf::OptSO3ApproxGD::karcherMeans(
    const Matrix<float,3,6>& p0, float thresh, uint32_t maxIter)
{
  Matrix<float,3,6> p = p0;
  Matrix<float,6,1> residuals;
  for(uint32_t i=0; i< maxIter; ++i)
  {
//    Timer t0;
    Matrix<float,3,6> mu_karch = meanInTpS2_GPU(p);
//    t0.toctic("meanInTpS2_GPU");
#ifndef NDEBUG
    cout<<"mu_karch"<<endl<<mu_karch<<endl;
#endif
    residuals.fill(0.0f);
    for (uint32_t j=0; j<6; ++j)
    {
      p.col(j) = S2_.Exp_p(p.col(j), mu_karch.col(j));
      residuals(j) = mu_karch.col(j).norm();
    }
#ifndef NDEBUG
//    cout<<"p"<<endl<<p<<endl;
    cout<<"karcherMeans "<<i<<" residuals="<<residuals.transpose()<<endl;
#endif
    if( (residuals.array() < thresh).all() )
    {
#ifndef NDEBUG
      cout<<"converged after "<<i<<" residuals="
        <<residuals.transpose()<<endl;
#endif
      break;
    }
  }
  return p;
}

void mmf::OptSO3ApproxGD::computeSuffcientStatistics()
{
  // compute rotations to north pole
  Matrix<float,2*6,3,RowMajor> Rnorths(2*6,3);
  for (uint32_t j=0; j<6; ++j)
  {
    Rnorths.middleRows<2>(j*2) = S2_.north_R_TpS2(qKarch_.col(j)).topRows<2>();
    //cout<<qKarch_.col(j).transpose()<<endl;
    //cout<<Rnorths.middleRows<2>(j*2)<<endl;
    //cout<<"----"<<endl;
  }
  //cout<<Rnorths<<endl;

  Matrix<float,7,6,ColMajor> SSs;
  sufficientStatisticsOnTpS2GPU(qKarch_.data(), d_mu_karch_, 
    Rnorths.data(), d_Rnorths_, this->cld_.d_x(), this->cld_.d_z() ,
    this->cld_.N(), SSs.data(), d_SSs_);
  
  //cout<<SSs<<endl; 
  for (uint32_t j=0; j<6; ++j)
  {
    xSums_.col(j) = SSs.block<2,1>(0,j);
    Ss_[j](0,0) =  SSs(2,j);
    Ss_[j](0,1) =  SSs(3,j);
    Ss_[j](1,0) =  SSs(4,j);
    Ss_[j](1,1) =  SSs(5,j);
    Ns_(j) = SSs(6,j);
    //cout<<"@j="<<j<<"\t"<< Ss_[j]<<endl;
  }
  //cout<<xSums_<<endl;
  //cout<<Ns_<<endl;
  
}
