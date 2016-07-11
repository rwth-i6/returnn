import theano
import theano.tensor as T
import numpy
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable, gpu_contiguous
from .Util import get_c_support_code_common


def pooling_regions_1D(sizes, factor):
  #here sizes is a 1D array
  out_sizes = numpy.zeros_like(sizes)
  max_size = sizes.max()
  regions = numpy.full((sizes.size, max_size), float("-inf"), dtype="float32")
  for idx, s in enumerate(sizes):
    out_size = numpy.ceil(s / factor)  #TODO we could also do floor or round
    #print s, factor, out_size
    if out_size == s:
      out_size -= 1
    assert out_size > 0
    n_twos = s - out_size - 1
    #n_ones = out_size - n_twos
    p = numpy.ones((out_size,), dtype="float32")
    p[:n_twos] = 2
    numpy.random.shuffle(p)  #TODO later we should use some theano random stream
    regions[idx, :out_size] = numpy.cumsum(p)
    out_sizes[idx] = out_size
  return regions, out_sizes


def pooling_regions(sizes, factor):
  regions_y, outsizes_y = pooling_regions_1D(sizes[:, 0], factor)
  regions_x, outsizes_x = pooling_regions_1D(sizes[:, 1], factor)
  out_sizes = numpy.concatenate([outsizes_y[:, numpy.newaxis], outsizes_x[:, numpy.newaxis]], axis=1)
  return regions_y, regions_x, out_sizes


class FractionalMaxPoolingHelperOp(theano.Op):
  __props__ = ()

  def make_node(self, sizes, factor):
    sizes = T.as_tensor_variable(sizes)
    assert sizes.dtype == "float32"
    assert sizes.ndim == 2
    factor = T.as_tensor_variable(factor)
    assert factor.dtype == "float32"
    assert factor.ndim == 0
    return theano.Apply(self, [sizes, factor], [sizes.type(), sizes.type(), sizes.type(), T.fvector()])

  def perform(self, node, inputs, out):
    sizes, factor = inputs
    regions_y, regions_x, out_sizes = pooling_regions(sizes, factor)
    out_height = out_sizes[:, 0].max()
    out_width = out_sizes[:, 1].max()
    max_out_size = numpy.array([out_height, out_width], dtype="float32")
    out[0][0] = regions_y
    out[1][0] = regions_x
    out[2][0] = out_sizes
    out[3][0] = max_out_size


class FractionalMaxPoolingOpGrad(theano.sandbox.cuda.GpuOp):
  __props__ = ()

  def make_node(self, X, DY, regions_y, regions_x):
    X = gpu_contiguous(as_cuda_ndarray_variable(X))
    assert X.dtype == "float32"
    assert X.ndim == 4
    DY = gpu_contiguous(as_cuda_ndarray_variable(DY))
    assert DY.dtype == "float32"
    assert DY.ndim == 4
    regions_y = gpu_contiguous(as_cuda_ndarray_variable(regions_y))
    assert regions_y.dtype == "float32"
    assert regions_y.ndim == 2
    regions_x = gpu_contiguous(as_cuda_ndarray_variable(regions_x))
    assert regions_x.dtype == "float32"
    assert regions_x.ndim == 2, regions_x.ndim
    return theano.Apply(self, [X, DY, regions_y, regions_x], [X.type()])

  def c_support_code(self):
    return get_c_support_code_common() + """
      #include <math_constants.h>

      __global__ void fmp_kernel_bwd(const float * X, const float * DY, float * DX, const float * regions_y_data,
                                     const float * regions_x_data, int n_regions_y, int n_regions_x,
                                     int h_out, int w_out, int h_inp, int w_inp, int n, int c)
      {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int n_stride = c;
        int w_stride = n * n_stride;
        int h_stride_region = n_regions_x * w_stride;
        int h_stride_inp = w_inp * w_stride;
        int h_stride_out = w_out * w_stride;
        int len = n_regions_y * n_regions_x * n * c;

        while (idx < len)
        {
          int y_idx_region = idx / h_stride_region;
          int x_idx_region = (idx % h_stride_region) / w_stride;
          int n_idx = (idx % w_stride) / n_stride;
          int c_idx = idx % n_stride;

          int y_start = 0;
          int x_start = 0;
          if(y_idx_region != 0)
          {
            y_start = static_cast<int>(regions_y_data[n_regions_y * n_idx + y_idx_region - 1]);
          }
          if(x_idx_region != 0)
          {
            x_start = static_cast<int>(regions_x_data[n_regions_x * n_idx + x_idx_region - 1]);
          }
          int y_end = static_cast<int>(regions_y_data[n_regions_y * n_idx + y_idx_region]);
          int x_end = static_cast<int>(regions_x_data[n_regions_x * n_idx + x_idx_region]);
          if(y_end < 0 || x_end < 0)
          {
            idx += gridDim.x * blockDim.x;
            continue;
          }

          float val = -CUDART_INF_F;
          int argmax_y = 0;
          int argmax_x = 0;
          for(int y = y_start; y <= y_end; ++y)
          {
            for(int x = x_start; x <= x_end; ++x)
            {
              float newVal = X[y * h_stride_inp + x * w_stride + n_idx * n_stride + c_idx];
              if(newVal > val)
              {
                val = newVal;
                argmax_y = y;
                argmax_x = x;
              }
            }
          }

          atomicAdd(&DX[argmax_y * h_stride_inp + argmax_x * w_stride + n_idx * n_stride + c_idx],
            DY[y_idx_region * h_stride_out + x_idx_region * w_stride + n_idx * n_stride + c_idx]);

          idx += gridDim.x * blockDim.x;
        }
      }
    """

  def c_code(self, node, name, input_names, output_names, sub):
    X, DY, regions_y, regions_x = input_names
    DX, = output_names
    fail = sub['fail']
    return """
      const int * X_dim = CudaNdarray_HOST_DIMS(%(X)s);
      const int * DY_dim = CudaNdarray_HOST_DIMS(%(DY)s);
      const int * regions_y_dim = CudaNdarray_HOST_DIMS(%(regions_y)s);
      const int * regions_x_dim = CudaNdarray_HOST_DIMS(%(regions_x)s);

      int h_inp = X_dim[0];
      int w_inp = X_dim[1];
      int n = X_dim[2];
      int c = X_dim[3];
      int h_out = DY_dim[0];
      int w_out = DY_dim[1];

      assert(regions_y_dim[0] == n);
      assert(regions_x_dim[0] == n);
      int n_regions_y = regions_y_dim[1];
      int n_regions_x = regions_x_dim[1];

      %(DX)s = CudaNdarray_zeros_like(%(X)s);

      float * DX_data = CudaNdarray_DEV_DATA(%(DX)s);
      float * DY_data = CudaNdarray_DEV_DATA(%(DY)s);
      const float * X_data = CudaNdarray_DEV_DATA(%(X)s);
      const float * regions_y_data = CudaNdarray_DEV_DATA(%(regions_y)s);
      const float * regions_x_data = CudaNdarray_DEV_DATA(%(regions_x)s);

      fmp_kernel_bwd<<<DIM_GRID, DIM_BLOCK>>>(X_data, DY_data, DX_data, regions_y_data, regions_x_data, n_regions_y,
                                              n_regions_x, h_out, w_out, h_inp, w_inp, n, c);

      CHECK_KERNEL_ERROR();
    """ % locals()

  def c_code_cache_version(self):
    return 1, 2


class FractionalMaxPoolingOp(theano.sandbox.cuda.GpuOp):
  __props__ = ()

  def make_node(self, X, regions_y, regions_x, out_size):
    X = gpu_contiguous(as_cuda_ndarray_variable(X))
    assert X.dtype == "float32"
    assert X.ndim == 4
    regions_y = gpu_contiguous(as_cuda_ndarray_variable(regions_y))
    assert regions_y.dtype == "float32"
    assert regions_y.ndim == 2
    regions_x = gpu_contiguous(as_cuda_ndarray_variable(regions_x))
    assert regions_x.dtype == "float32"
    assert regions_x.ndim == 2, regions_x.ndim
    out_size = T.as_tensor_variable(out_size)
    assert out_size.dtype == "float32"
    assert out_size.ndim == 1
    return theano.Apply(self, [X, regions_y, regions_x, out_size], [X.type()])

  def c_support_code(self):
    return get_c_support_code_common() + """
      #include <math_constants.h>

      __global__ void fmp_kernel(const float * X, float * Y, const float * regions_y_data,
                                 const float * regions_x_data, int n_regions_y, int n_regions_x, 
                                 int h_out, int w_out, int h_inp, int w_inp, int n, int c)
      {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int n_stride = c;
        int w_stride = n * n_stride;
        int h_stride_region = n_regions_x * w_stride;
        int h_stride_inp = w_inp * w_stride;
        int h_stride_out = w_out * w_stride;
        int len = n_regions_y * n_regions_x * n * c;

        while (idx < len)
        {
          int y_idx_region = idx / h_stride_region;
          int x_idx_region = (idx % h_stride_region) / w_stride;
          int n_idx = (idx % w_stride) / n_stride;
          int c_idx = idx % n_stride;

          int y_start = 0;
          int x_start = 0;
          if(y_idx_region != 0)
          {
            y_start = static_cast<int>(regions_y_data[n_regions_y * n_idx + y_idx_region - 1]);
          }
          if(x_idx_region != 0)
          {
            x_start = static_cast<int>(regions_x_data[n_regions_x * n_idx + x_idx_region - 1]);
          }
          int y_end = static_cast<int>(regions_y_data[n_regions_y * n_idx + y_idx_region]);
          int x_end = static_cast<int>(regions_x_data[n_regions_x * n_idx + x_idx_region]);
          if(y_end < 0 || x_end < 0)
          {
            idx += gridDim.x * blockDim.x;
            continue;
          }

          float val = -CUDART_INF_F;
          for(int y = y_start; y <= y_end; ++y)
          {
            for(int x = x_start; x <= x_end; ++x)
            {
              float newVal = X[y * h_stride_inp + x * w_stride + n_idx * n_stride + c_idx];
              val = max(val, newVal);
            }
          }

          Y[y_idx_region * h_stride_out + x_idx_region * w_stride + n_idx * n_stride + c_idx] = val;

          idx += gridDim.x * blockDim.x;
        }
      }
    """

  def c_code(self, node, name, input_names, output_names, sub):
    X, regions_y, regions_x, out_size = input_names
    Y, = output_names
    fail = sub['fail']
    return """
      const int * X_dim = CudaNdarray_HOST_DIMS(%(X)s);
      const int * regions_y_dim = CudaNdarray_HOST_DIMS(%(regions_y)s);
      const int * regions_x_dim = CudaNdarray_HOST_DIMS(%(regions_x)s);
      
      int h_inp = X_dim[0];
      int w_inp = X_dim[1];
      int n = X_dim[2];
      int c = X_dim[3];
      int h_out = static_cast<int>(reinterpret_cast<float*>(PyArray_DATA(%(out_size)s))[0]);
      int w_out = static_cast<int>(reinterpret_cast<float*>(PyArray_DATA(%(out_size)s))[1]);

      assert(regions_y_dim[0] == n);
      assert(regions_x_dim[0] == n);
      int n_regions_y = regions_y_dim[1];
      int n_regions_x = regions_x_dim[1];
      
      int Y_dim[] = {h_out, w_out, n, c};
      %(Y)s = (CudaNdarray*) MyCudaNdarray_NewDims(4, Y_dim);
      assert(%(Y)s);
      CudaNdarray_fill(%(Y)s, -1e20);
      
      float * Y_data = CudaNdarray_DEV_DATA(%(Y)s);
      const float * X_data = CudaNdarray_DEV_DATA(%(X)s);
      const float * regions_y_data = CudaNdarray_DEV_DATA(%(regions_y)s);
      const float * regions_x_data = CudaNdarray_DEV_DATA(%(regions_x)s);
      
      fmp_kernel<<<DIM_GRID, DIM_BLOCK>>>(X_data, Y_data, regions_y_data, regions_x_data, n_regions_y, n_regions_x,
                                          h_out, w_out, h_inp, w_inp, n, c);
      
      CHECK_KERNEL_ERROR();
    """ % locals()

  def grad(self, inputs, output_grads):
    X, regions_y, regions_x, out_size = inputs
    DY, = output_grads
    DX = FractionalMaxPoolingOpGrad()(X, DY, regions_y, regions_x)
    Dregions_y = theano.gradient.grad_undefined(self, 1, regions_y, 'cannot diff w.r.t. regions_y')
    Dregions_x = theano.gradient.grad_undefined(self, 2, regions_x, 'cannot diff w.r.t. regions_x')
    Dout_size = theano.gradient.grad_undefined(self, 3, out_size, 'cannot diff w.r.t. out_size')
    return [DX, Dregions_y, Dregions_x, Dout_size]

  def c_code_cache_version(self):
    return 1, 2


#use this function to do fmp!
def fmp(X, sizes, factor):
  regions_y, regions_x, out_sizes, max_out_size = FractionalMaxPoolingHelperOp()(sizes, factor)
  Y = FractionalMaxPoolingOp()(X, regions_y, regions_x, max_out_size)
  return Y, out_sizes
