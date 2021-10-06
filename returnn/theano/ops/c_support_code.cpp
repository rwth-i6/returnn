#include <limits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <iostream>
#include <stdint.h>

using namespace std;

template<typename T>
class Log
{
public:
    Log(T v = 0):
        logVal_(v < expMin ? logZero : log(v))
    {

    }

    Log(const Log& other)
    {
    	logVal_ = other.logVal_;
    }

    static Log<T> fromLogVal(T logVal)
    {
    	logVal = std::min(logVal, logInf);
    	logVal = std::max(logVal, logZero);
    	Log<T> res;
		res.logVal_ = logVal;
		return res;
    }

    Log<T> pow(T v)
    {
    	if(logVal_ == logZero)
    	{
    		return *this;
    	}
    	Log<T> res(*this);
    	res.logVal_ *= v;
    	res.logVal_ = std::min(res.logVal_, logInf);
    	res.logVal_ = std::max(res.logVal_, logZero);
    	return res;
    }

    Log<T>& operator*=(const Log<T>& rhs)
    {
        if(logVal_ == logZero || rhs.logVal_ == logZero)
        {
            logVal_ = logZero;
        }
        else
        {
            logVal_ += rhs.logVal_;
        }
        return *this;
    }

    T logVal() const
    {
        return logVal_;
    }

    Log<T>& operator/=(const Log<T>& rhs)
    {
        if(logVal_ == logZero)
        {
            //nothing to do
        }
        else if(rhs.logVal_ == logZero)
        {
            logVal_ = logInf;
        }
        else
        {
            logVal_ -= rhs.logVal_;
        }
        return *this;
    }

    Log<T>& operator+=(const Log<T>& rhs)
    {
        if(logVal_ == logZero)
        {
            *this = rhs;
        }
        else if(rhs.logVal_ != logZero)
        {
            T x = std::max(logVal_, rhs.logVal_);
            T y = std::min(logVal_, rhs.logVal_);
            logVal_ = x + ::log(T(1.0) + safeExp(y-x));
        }
        return *this;
    }

    bool operator==(const Log<T>& rhs)
    {
        return logVal_ == rhs.logVal_;
    }

    bool operator<(const Log<T>& rhs)
    {
    	return logVal_ < rhs.logVal_;
    }

    bool operator>(const Log<T>& rhs)
    {
    	return logVal_ > rhs.logVal_;
    }

    T expVal() const
    {
        return safeExp(logVal_);
    }

    static const T expMax;
    static const T expMin;
    static const T expLimit;
    static const T logZero;
    static const T logInf;
private:
    static T safeExp(T x)
    {
        if(x == logZero)
        {
            return 0;
        }
        if(x >= expLimit)
        {
            return expMax;
        }
        return ::exp(x);
    }

    T logVal_;
};

template<class T> const T Log<T>::expMax = std::numeric_limits<T>::max();
template<class T> const T Log<T>::expMin = std::numeric_limits<T>::min();
template<class T> const T Log<T>::expLimit = log(expMax);
template<class T> const T Log<T>::logInf = T(1e33);
template<class T> const T Log<T>::logZero = -Log<T>::logInf;

template<class T> Log<T> operator+(Log<T> lhs, Log<T> rhs)
{
    Log<T> res(lhs);
    res += rhs;
    return res;
}

template<class T> Log<T> operator*(Log<T> lhs, Log<T> rhs)
{
    Log<T> res(lhs);
    res *= rhs;
    return res;
}

template<class T> Log<T> operator/(Log<T> lhs, Log<T> rhs)
{
    Log<T> res(lhs);
    res /= rhs;
    return res;
}

typedef Log<float> myLog;

template<class T>
class PyArrayWrapper
{
public:
    PyArrayWrapper(PyArrayObject* arr):
        a_(arr)
    {

    }

    int dim(int idx) const
    {
        if(idx >= PyArray_NDIM(a_))
        {
            printf("index out of range1: %i / %i \n", idx, (int) PyArray_NDIM(a_));
        }
        return PyArray_DIM(a_,idx);
    }

    T& operator()()
    {
        if(PyArray_NDIM(a_) != 0)
        {
            printf("zero-dimensional index operator used on higher-dimensional array\n");
        }
        return *reinterpret_cast<T*>(PyArray_DATA(a_));
    }

    T& operator()(int idx)
    {
        if(PyArray_NDIM(a_) != 1)
        {
            printf("single-dimensional index operator used on multi-dimensional array\n");
        }
        idx *= PyArray_STRIDE(a_,0);
        if(idx >= PyArray_STRIDE(a_,0) * PyArray_DIM(a_,0))
        {
            printf("index out of range2: %i / %i \n", idx, (int) PyArray_DIM(a_,0));
        }
        return *reinterpret_cast<T*>(PyArray_DATA(a_) + idx);
    }

    T& operator()(int idx1, int idx2)
    {
        if(PyArray_NDIM(a_) != 2)
        {
            printf("2-dimensional index operator used on non 2-dimensional array");
        }
        int idx = idx1 * PyArray_STRIDE(a_,0) + idx2 * PyArray_STRIDE(a_,1);
        if(idx >= PyArray_STRIDE(a_,0) * PyArray_DIM(a_,0))
        {
            printf("index out of range3: %i / %i , %i / %i \n", idx1, (int) PyArray_DIM(a_,0), idx2, (int) PyArray_DIM(a_, 1));
        }
        return *reinterpret_cast<T*>(PyArray_DATA(a_) + idx);
    }

    T& operator()(int idx1, int idx2, int idx3)
    {
        if(PyArray_NDIM(a_) != 3)
        {
            printf("3-dimensional index operator used on non-3-dimensional array");
        }
        int idx = idx1 * PyArray_STRIDE(a_,0) + idx2 * PyArray_STRIDE(a_,1) + idx3 * PyArray_STRIDE(a_,2);
        if(idx >= PyArray_STRIDE(a_,0) * PyArray_DIM(a_,0))
        {
            printf("index out of range4: %i / %i, %i / %i, %i / %i \n", idx1, (int) PyArray_DIM(a_,0),
                   idx2, (int) PyArray_DIM(a_, 1), idx3, (int) PyArray_DIM(a_, 2));
        }
        return *reinterpret_cast<T*>(PyArray_DATA(a_) + idx);
    }

    const T& operator()() const
    {
        return const_cast<PyArrayWrapper<T>&>(*this)();
    }

    const T& operator()(int idx) const
    {
        return const_cast<PyArrayWrapper<T>&>(*this)(idx);
    }

    const T& operator()(int idx1, int idx2) const
    {
        return const_cast<PyArrayWrapper<T>&>(*this)(idx1, idx2);
    }

    const T& operator()(int idx1, int idx2, int idx3) const
    {
        return const_cast<PyArrayWrapper<T>&>(*this)(idx1, idx2, idx3);
    }

    void debugPrint(const char * name) const
    {
        int numDims = PyArray_NDIM(a_);
        if(numDims == 1)
        {
            printf("size %s: %i \n", name, (int) PyArray_DIM(a_,0));
            printf("stride %s: %i \n", name, (int) PyArray_STRIDE(a_,0));
        }
        else if(numDims == 2)
        {
            printf("size %s: %i, %i \n", name, (int) PyArray_DIM(a_,0), (int) PyArray_DIM(a_,1));
            printf("stride %s: %i, %i \n", name, (int) PyArray_STRIDE(a_,0), (int) PyArray_STRIDE(a_,1));
        }
        else if(numDims == 3)
        {
            printf("size %s: %i, %i, %i \n", name, (int) PyArray_DIM(a_,0), (int) PyArray_DIM(a_,1), (int) PyArray_DIM(a_,2));
            printf("stride %s: %i, %i, %i \n", name, (int) PyArray_STRIDE(a_,0), (int) PyArray_STRIDE(a_,1), (int) PyArray_STRIDE(a_,2));
        }
    }

private:
    PyArrayObject* a_;
};

template<class T>
class PySubArrayWrapper
{
public:
	//fixes index of dimension dim to idxValue and treats the array as if it had 1 dimension less than before
	PySubArrayWrapper(PyArrayWrapper<T>& arr, int dim, int idxValue):
	arr_(arr),
	dim_(dim),
	idxValue_(idxValue)
	{

	}

	int dim(int idx) const
	{
		if(idx < dim_)
		{
			return arr_.dim(idx);
		}
		else
		{
			return arr_.dim(idx + 1);
		}
	}

	T& operator()(int idx)
	{
		if(dim_ == 0)
		{
			return arr_(idxValue_, idx);
		}
		else if(dim_ == 1)
		{
			return arr_(idx, idxValue_);
		}
		else
		{
			std::cerr << "indexing error1" << std::endl;
			throw std::out_of_range("indexing error");
		}
	}

    T& operator()(int idx1, int idx2)
    {
    	if(dim_ == 0)
    	{
    		return arr_(idxValue_, idx1, idx2);
    	}
    	else if(dim_ == 1)
    	{
    		return arr_(idx1, idxValue_, idx2);
    	}
    	else if(dim_ == 2)
    	{
    		return arr_(idx1, idx2, idxValue_);
    	}
    	else
    	{
    		std::cerr << "indexing error2" << std::endl;
			throw std::out_of_range("indexing error");
    	}
    }

    const T& operator()(int idx) const
    {
    	return const_cast<PySubArrayWrapper<T>&>(*this)(idx);
    }

    const T& operator()(int idx1, int idx2) const
    {
    	return const_cast<PySubArrayWrapper<T>&>(*this)(idx1, idx2);
    }

    void debugPrint(const char* name) const
    {
        //TODO
        arr_.debugPrint(name);
    }
private:
	PyArrayWrapper<T>& arr_;
	int dim_;
	int idxValue_;
};

typedef PyArrayWrapper<float> ArrayF;
typedef PyArrayWrapper<int> ArrayI;
typedef const PyArrayWrapper<float> CArrayF;
typedef const PyArrayWrapper<int> CArrayI;

typedef PySubArrayWrapper<float> SArrayF;
typedef PySubArrayWrapper<int> SArrayI;
typedef const PySubArrayWrapper<float> CSArrayF;
typedef const PySubArrayWrapper<int> CSArrayI;

template<class T>
class TwoDArray
{
public:
	TwoDArray():
	size1_(0),
	size2_(0),
	data_(0)
	{

	}

	TwoDArray(size_t size1, size_t size2):
	size1_(size1),
	size2_(size2),
	data_(new T[size1*size2]())
	{

	}

	~TwoDArray()
	{
		delete[] data_;
	}

	void resize(size_t size1, size_t size2)
	{
		size1_ = size1;
		size2_ = size2;
		delete[] data_;
		data_ = new T[size1*size2]();
	}

	void swap(TwoDArray<T>& other)
	{
		std::swap(size1_, other.size1_);
		std::swap(size2_, other.size2_);
		std::swap(data_, other.data_);
	}

	size_t size(size_t idx) const
	{
		if(idx == 0)
		{
			return size1_;
		}
		else if(idx == 1)
		{
			return size2_;
		}
		else
		{
			std::cerr << "indexing error3" << std::endl;
			throw std::out_of_range("indexing error");
		}
	}

	T& operator()(size_t idx1, size_t idx2)
	{
		//optional range check
		if(idx1 >= size1_ || idx2 >= size2_)
		{
			std::cerr << "indexing error4: idx1: " << idx1 << " idx2: " << idx2 << " size1_:" << size1_ << " size2_: " << size2_ << std::endl;
			throw std::out_of_range("indexing error");
		}

		return data_[size2_ * idx1 + idx2];
	}

	const T& operator()(size_t idx1, size_t idx2) const
	{
		return const_cast<TwoDArray<T>&>(*this)(idx1, idx2);
	}

	T& at(size_t idx1, size_t idx2)
	{
		return (*this)(idx1, idx2);
	}

	const T& at(size_t idx1, size_t idx2) const
	{
		return (*this)(idx1,idx2);
	}
private:
	size_t size1_;
	size_t size2_;
	T * data_;
};

template <typename T>
T& data(PyArrayObject* arr)
{
  return *reinterpret_cast<T*>(PyArray_DATA(arr));
}

unsigned int& datau(PyArrayObject* arr)
{
  return data<unsigned int>(arr);
}

uint64_t& datau64(PyArrayObject* arr)
{
  return data<uint64_t>(arr);
}

int32_t& datai(PyArrayObject* arr)
{
  return data<int32_t>(arr);
}

int64_t& datai64(PyArrayObject* arr)
{
  return data<int64_t>(arr);
}

void*& datavoid(PyArrayObject* arr)
{
  return data<void*>(arr);
}

float& dataf(PyArrayObject* arr)
{
  return data<float>(arr);
}

//https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#C.2B.2B
template<class T>
int levenshteinDist(const T &s1, const T & s2)
{
    const size_t len1 = s1.size(), len2 = s2.size();
    std::vector<unsigned int> col(len2+1), prevCol(len2+1);

    for (unsigned int i = 0; i < prevCol.size(); i++)
    {
        prevCol[i] = i;
    }
    for (unsigned int i = 0; i < len1; i++)
    {
        col[0] = i+1;
        for (unsigned int j = 0; j < len2; j++)
        {
            col[j+1] = std::min(std::min( 1 + col[j], 1 + prevCol[1 + j]), prevCol[j] + (s1[i]==s2[j] ? 0 : 1));
        }
        col.swap(prevCol);
    }
    return (int) prevCol[len2];
}

void verify(bool pred, const char * msg = "")
{
  if(!pred)
  {
    cerr << "assertion failed: " << msg << endl;
    cout << "assertion failed: " << msg << endl;
    exit(1);
  }
}
