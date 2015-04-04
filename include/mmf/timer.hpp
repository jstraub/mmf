/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#pragma once

#include <iostream>
#include <sys/time.h>
#include <string>

using namespace std;

namespace mmf{

class Timer
{
public:
  Timer()
  {
    tic();
    tInit_ = t0_;
  };

  void tic(void)
  {
    gettimeofday(&t0_, NULL);
  };

  float toc(void)
  {
    gettimeofday(&t1_, NULL);
    dt_ = getDtMs(t0_,t1_);
    return dt_;
  };

  float toctic(string description="")
  {
    toc();
#ifndef NDEBUG
    if (description.size()>0)
      cerr<<description<<" :"<<dt_<<"ms"<<endl;
#endif
    tic();
    return dt_;
  };

  float lastDt(void) const
  {
    return dt_;
  };

  float dtFromInit(void) 
  {
    timeval tNow; 
    gettimeofday(&tNow, NULL);
    return getDtMs(tInit_,tNow);
  };

  Timer& operator=(const Timer& t)
  {
    if(this != &t){
      dt_=t.lastDt();
    }
    return *this;
  };

private:
  float dt_;
  timeval tInit_, t0_, t1_;

  float getDtMs(const timeval& t0, const timeval& t1) 
  {
    float dt = (t1.tv_sec - t0.tv_sec) * 1000.0f; // sec to ms
    dt += (t1.tv_usec - t0.tv_usec) / 1000.0f; // us to ms
    return dt;
  };
};
}

inline ostream& operator<<(ostream &out, const mmf::Timer& t)
{
  out << t.lastDt() << "ms";
  return out;
};

