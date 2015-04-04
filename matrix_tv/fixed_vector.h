/**************************************************************************************************\
 *
 * FixedVector<T,n> - encapsulates a vector of type T, of length n, with a fixed size (statically allocated).
 *
 * Supports various BLAS operations, such as matrix-vector multiplication, vector inner product, addition, substraction, and so forth.
 *
 * \**************************************************************************************************/

#ifndef FIXED_VECTOR_H
#define FIXED_VECTOR_H
#include <iostream>

template<class T, int n> struct FixedVector{
    private:
        T _data[n];
        public:
            // A copy constructor
            __device__ __host__ FixedVector(const FixedVector<T, n>& rhs){
                for(int i=0;i<n;++i){
                    this->operator[](i)=rhs[i];
                }
            }
            // A default constructor (vector elements are initialized by default to 0)
            __device__ __host__ FixedVector(const T& val=0){
                for(int i=0;i<n;++i){
                    this->operator[](i)=val;
                }
            }
            
            __device__ __host__ T& operator[](int i){
                return _data[i];
            }
            __device__ __host__ T operator[](int i) const{
                return _data[i];
            }
            __device__ __host__ FixedVector<T,n>  operator+(const FixedVector<T,n>& param2){
                FixedVector<T,n> res(*this);
                for (int i(0);i<n;++i){
                    res._data[i]+=param2._data[i];
                }
                return res;
            }
            __device__ __host__ FixedVector<T,n>  operator/(T a){
                FixedVector<T,n> res(*this);
                for (int i(0);i<n;++i){
                    res._data[i]/=a;
                }
                return res;
            }
            __device__ __host__ T L2Norm(){
//                 T res(0);
//                 for (int i(0);i<n;++i){
//                     T v=_data[i];
//                     res+=v*v;
//                 }
                return sqrt(this->innerProduct(*this));
            }
            __device__ __host__ T L1Norm(){
                T res(0);
                for (int i(0);i<n;++i){
                    T v=_data[i];
                    res+=abs(v);
                }
                return res;
            }
            
            __device__ __host__ T innerProduct(const FixedVector<T,n>& b){
                T res(0);
                for (int i(0);i<n;++i){
                    res+=this->_data[i]*b._data[i];
                }
                return res;
            }
            __device__ __host__ FixedVector<T,n> addScalarVector(T alpha,const FixedVector<T,n>& b){
                FixedVector<T,n> res;
                for (int i(0);i<n;++i){
                    res._data[i]=this->_data[i]+alpha*b._data[i];
                }
                return res;
            }
            
            __device__ __host__ inline int length(){
                return n;
            }

};

template<class T, int m, int n> struct FixedMatrix;

template<class T, int n> __device__ __host__ FixedVector<T,n>  operator-(const FixedVector<T,n>& param1, const FixedVector<T,n>& param2){
    FixedVector<T,n> res(param1);
    for (int i(0);i<n;++i){
        res[i]=param1[i]-param2[i];
    }
    return res;
}

template<class T, class S, int n> __device__ __host__ FixedVector<T,n>  operator*(const S& alpha, const FixedVector<T,n>& param2){
    FixedVector<T,n> res(param2);
    for (int i(0);i<n;++i){
        res[i]=alpha*param2[i];
    }
    return res;
}

/*template<class T, int n> FixedVector<T,n> __device__ __host__ addFixedVector(FixedVector<T,n> a,FixedVector<T,n> b){
            FixedVector<T,n> res;
            for (int i(0);i<n;++i){
                res.data[i]=a.data[i]+b.data[i];
            }
            return res;
}
template<class T, int n> FixedVector<T,n> __device__ __host__ subFixedVector(FixedVector<T,n> a,FixedVector<T,n> b){
            FixedVector<T,n> res;
            for (int i(0);i<n;++i){
                res.data[i]=a.data[i]-b.data[i];
            }
            return res;
}
 */
template<class T, int n> FixedVector<T,n> __device__ __host__ addVectorAndScalarVector(FixedVector<T,n> a,T alpha,FixedVector<T,n> b){
    FixedVector<T,n> res;
    for (int i(0);i<n;++i){
        res._data[i]=a._data[i]+alpha*b._data[i];
    }
    return res;
}



template<class T, int n> std::ostream& operator<<(std::ostream& out, const FixedVector<T,n>& v){
    out<<"(";
    for (int i(0);i<n;++i){
        out<<v[i];
        if (i<(n+1))
            out<<" ";
    }
    out<<")";
    return out;
}


#endif

