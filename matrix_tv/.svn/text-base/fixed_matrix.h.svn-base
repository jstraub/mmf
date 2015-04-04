#ifndef FIXED_MATRIX_H
#define FIXED_MATRIX_H
#include <iostream>
#include "fixed_vector.h"
/**************************************************************************************************\
 *
 * FixedMatrix<T,m,n> - describes a matrix of fixed dimensions, with statically allocated memory
 *
 * Supports various BLAS operations, such as matrix-vector multiplication, matrix-matrix multiplication, addition, and so forth.
 *
 */
template<class T, int m, int n> struct FixedMatrix{
    protected:
        T _data[n*m];
        // Get the memory index for accessing row i, column j. Internally, a matrix is represented in row-major order.
        inline __device__ __host__ int getIdx(int i, int j) const {
            return j+i*n;
        }
        public:
            // Access element (i,j)
            __device__ __host__ T & operator()(int i, int j){
                return _data[getIdx(i,j)];
            }
            __device__ __host__ const T & operator()(int i, int j) const{
                return _data[getIdx(i,j)];
            }
            // Copy constructor
            __device__ __host__ FixedMatrix(const FixedMatrix<T, m,n>& rhs){
                for(int i=0;i<m;++i){
                    for(int j=0;j<n;++j){
                        this->operator()(i,j)=rhs(i,j);
                    }
                }
            }
            //
            __device__ __host__ FixedMatrix(const T&val=0){
                for(int i=0;i<m;++i){
                    for(int j=0;j<n;++j){
                        this->operator()(i,j)=val;
                    }
                }
            }
            __device__ __host__ FixedMatrix<T,m,n>  operator+(const FixedMatrix<T,m,n>& param2){
                FixedMatrix<T,m,n> res(*this);
                for (int i(0);i<(n*m);++i){
                    res._data[i]+=param2._data[i];
                }
                return res;
            }
            __device__ __host__ FixedMatrix<T,m,n>  operator-(const FixedMatrix<T,m,n>& param2){
                FixedMatrix<T,m,n> res(*this);
                for (int i(0);i<(n*m);++i){
                    res._data[i]-=param2._data[i];
                }
                return res;
            }
            __device__ __host__ FixedMatrix<T,n,m>  transpose(){
                FixedMatrix<T,n,m> res;
                for (int i(0);i<m;++i){
                    for (int j(0);j<(n);++j){
                        res(j,i)=this->operator()(i,j);
                    }
                }
                return res;
            }
            __device__ __host__ inline int rows(){
                return m;
            }
            __device__ __host__ inline int columns(){
                return n;
            }
            __device__ __host__ FixedVector<T,m> getCol(int i){
                FixedVector<T,m> res(0);
                for (int j=0;j<m;++j){
                    res[j]=this->operator()(j,i);
                }
                return res;
            }
            __device__ __host__ void setCol(int i,const FixedVector<T,m>&v){
                for (int j=0;j<m;++j){
                    this->operator()(j,i)=v[j];
                }
            }
            __device__ __host__ T frobeniusNorm(){
                T res;
                for (int i(0);i<m;++i){
                    for (int j(0);j<(n);++j){
                        T x=this->operator()(i,j);
                        res=res+x*x;
                    }
                }
                return sqrtf(res);
            }
            //

};

template<class T, int m, int n> __device__ __host__ FixedVector<T,m> operator*(const FixedMatrix<T,m,n>& param1, const FixedVector<T,n>& param2){
    FixedVector<T,m> res;
    for (int i(0);i<m;++i){
        res[i]=0;
        for (int j(0);j<n;++j){
            res[i]+=param1(i,j)*param2[j];
        }
    }
    return res;
}

template<class T, class S, int m, int n> __device__ __host__ FixedMatrix<T,m,n>  operator*(const S& alpha, const FixedMatrix<T,m,n>& param2){
    FixedMatrix<T,m,n> res(param2);
    for (int i(0);i<m;++i){
        for (int j(0);j<n;++j){
            res(i,j)=alpha*param2(i,j);
        }
    }
    return res;
}
template<class T, int m, int n, int l> __device__ __host__ FixedMatrix<T,m,l>  operator*(const FixedMatrix<T,m,n>& param1,const FixedMatrix<T,n,l>& param2){
    FixedMatrix<T,m,l> res;
    for (int i(0);i<m;++i){
        for (int k(0);k<l;++k){
            res(i,k)=0;
            for (int j(0);j<n;++j){
                res(i,k)+=param1(i,j)*param2(j,k);
            }
        }
    }
    return res;
}

template<class T, int m, int n> FixedVector<T,m> __device__ __host__ computeResidualVector(FixedVector<T,m> b, FixedVector<T,n> x,FixedMatrix<T,m,n> A){
    FixedVector<T,m> res;
    for (int i(0);i<m;++i){
        res[i]=b[i];
        for (int j(0);j<n;++j){
            res[i]-=A(i,j)*x.data[j];
        }
    }
    return res;
}


template<class T, int m, int n> std::ostream& operator<<(std::ostream& out, const FixedMatrix<T,m,n>& M){
    for (int i(0);i<m;++i){
        for (int j(0);j<n;++j){
            out<<M(i,j);
            if (j<(m+1))
                out<<" ";
        }
        out<<"\n";
    }
    return out;
}
#endif

