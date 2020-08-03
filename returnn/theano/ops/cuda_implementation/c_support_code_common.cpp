#include <thrust/device_vector.h>
#include <vector>
#include <string>
#include <cublas_v2.h>
using namespace std;

//also defining CUDA_LAUNCH_BLOCKING 1 as environment variable can help finding errors
//for productive use, this should definitely not be enabled!
//#define KERNEL_DEBUG_MODE

//TODO tune launch configuration
#define DIM_GRID 128
#define DIM_BLOCK 512

PyObject * MyCudaNdarray_NewDims(int nd, const int * dims);

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
  switch (error)
  {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

static void HandleError(cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess)
  {
    printf("%s in %s at line %d\n", cudaGetErrorString(err),
      file, line);
    exit(EXIT_FAILURE);
  }
}

static void HandleError(cublasStatus_t status, const char *file, int line)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    printf("%s in %s at line %d\n", _cudaGetErrorEnum(status),
      file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#ifdef KERNEL_DEBUG_MODE
  #define CHECK_KERNEL_ERROR() (HandleError( cudaDeviceSynchronize(), __FILE__, __LINE__ ))
#else
  #define CHECK_KERNEL_ERROR()
#endif

__global__ void check_gpu_ptr_kernel(const float ** ptr, int size, float * out)
{
  float val = 0;
  for(int i = 0; i < size; ++i)
  {
    if(ptr[i] != 0)
    {
      val += ptr[i][0];
    }
  }
  if(out != 0)
  {
    *out = val;
  }
}

//only useful together with cuda-memcheck
void check_gpu_ptr(const float ** ptr, int size)
{
  check_gpu_ptr_kernel<<<1,1>>>(ptr, size, 0);
  CHECK_KERNEL_ERROR();
}

CudaNdarray * CudaNdarray_zeros_like(CudaNdarray* a)
{
  const int * dim = CudaNdarray_HOST_DIMS(a);
  CudaNdarray * res = (CudaNdarray*) MyCudaNdarray_NewDims(a->nd, dim);
  int n = CudaNdarray_SIZE(a);
  HANDLE_ERROR(cudaMemset(CudaNdarray_DEV_DATA(res), 0, sizeof(float) * n));
  return res;
}

CudaNdarray * CudaNdarray_uninitialized_like(CudaNdarray* a)
{
  const int * dim = CudaNdarray_HOST_DIMS(a);
  CudaNdarray * res = (CudaNdarray*) MyCudaNdarray_NewDims(a->nd, dim);
  return res;
}

void CudaNdarray_print(CudaNdarray * a)
{
  //atm print the flattened array
  int n = CudaNdarray_SIZE(a);
  const float * a_data = CudaNdarray_DEV_DATA(a);
  vector<float> v(n);
  HANDLE_ERROR(cudaMemcpy(&v[0], a_data, n * sizeof(float), cudaMemcpyDeviceToHost));
  for(int i = 0; i < n; ++i)
  {
    cout << v[i] << " ";
  }
  cout << endl;
}

void CudaNdarray_print_part(CudaNdarray * a)
{
  //atm print the flattened array
  int n = CudaNdarray_SIZE(a);
  const float * a_data = CudaNdarray_DEV_DATA(a);
  vector<float> v(n);
  HANDLE_ERROR(cudaMemcpy(&v[0], a_data, n * sizeof(float), cudaMemcpyDeviceToHost));
  for(int i = 0; i < n && i < 100; ++i)
  {
    cout << v[i] << " ";
  }
  cout << endl;
}

__global__ void dummy_kernel()
{

}

//launch dummy kernel in default stream to synchronize all streams
void synchronize_streams()
{
  dummy_kernel<<<1,1>>>();
}

__global__ void initKernel(float * data, float val, int size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(; idx < size; idx += stride)
    {
      data[idx] = val;
    }
}

void CudaNdarray_fill(CudaNdarray * a, float val)
{
  const int * dim = CudaNdarray_HOST_DIMS(a);
  int n = CudaNdarray_SIZE(a);
  float * a_data = CudaNdarray_DEV_DATA(a);
  initKernel<<<DIM_GRID, DIM_BLOCK>>>(a_data, val, n);
  CHECK_KERNEL_ERROR();
}

PyObject * MyCudaNdarray_NewDims(int nd, const int * dims)
{
  PyObject * res = CudaNdarray_NewDims(nd, dims);
  if(!res)
  {
    PyErr_Format(PyExc_RuntimeError, "Out of GPU memory");
  }
  return res;
}
