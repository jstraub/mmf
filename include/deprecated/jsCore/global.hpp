/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <stdint.h>

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>

//using namespace boost;
#ifdef BOOST_OLD
#  define shared_ptr boost::shared_ptr
#else
  using boost::shared_ptr;
#endif

typedef Eigen::Matrix<uint32_t,Eigen::Dynamic,1> VectorXu;
typedef Eigen::Matrix<uint32_t,Eigen::Dynamic,Eigen::Dynamic> MatrixXu;

// shared pointer typedefs
typedef shared_ptr<Eigen::MatrixXd> spMatrixXd;
typedef shared_ptr<Eigen::MatrixXf> spMatrixXf;

typedef shared_ptr<Eigen::VectorXd> spVectorXd;
typedef shared_ptr<Eigen::VectorXf> spVectorXf;
typedef shared_ptr<Eigen::VectorXi> spVectorXi;
typedef shared_ptr<VectorXu> spVectorXu;

#ifndef NDEBUG
#   define ASSERT(condition, message) \
  do { \
    if (! (condition)) { \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
      << " line " << __LINE__ << ": " << message << " " << std::endl; \
      assert(false); \
      std::exit(EXIT_FAILURE); \
    } \
  } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif
