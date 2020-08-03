import theano.sandbox.cuda
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable, gpu_contiguous, GpuContiguous, GpuDimShuffle)
from .Util import get_c_support_code_common
from theano.gof.opt import OpSub
from theano.compile import optdb
from theano import gof
from theano.gof import toolbox


class PoolHWBCOpGrad(theano.sandbox.cuda.GpuOp):
  __props__ = ("pool_shape", "inplace", "BCHW_grad_output")

  def __init__(self, pool_shape, inplace, BCHW_grad_output):
    pool_shape = tuple(pool_shape)
    super(PoolHWBCOpGrad, self).__init__()
    assert len(pool_shape) == 2, len(pool_shape)
    assert pool_shape[0] > 0, pool_shape[0]
    assert pool_shape[1] > 0, pool_shape[1]
    if BCHW_grad_output:
      assert inplace
    self.pool_shape = pool_shape
    self.inplace = inplace
    self.BCHW_grad_output = BCHW_grad_output

    if inplace:
      self.destroy_map = {0: [0]}
    #register optimization for this pool_shape
    else:
      if not hasattr(optdb, 'PoolHWBCOpGradInplaceOpt_registered'):
        optdb.PoolHWBCOpGradInplaceOpt_registered = []
      if pool_shape not in optdb.PoolHWBCOpGradInplaceOpt_registered:
        PoolHWBCOpGradInplaceOpt = OpSub(self, PoolHWBCOpGrad(self.pool_shape, inplace=True, BCHW_grad_output=False))
        optdb.PoolHWBCOpGradInplaceOpt_registered.append(pool_shape)
        optdb.register('PoolHWBCOpGradInplaceOpt' + str(pool_shape),
                       theano.gof.TopoOptimizer(PoolHWBCOpGradInplaceOpt, failure_callback=gof.TopoOptimizer.warn_inplace),
                       50.0, 'fast_run', 'inplace', 'gpuarray')

  def make_node(self, X, DY):
    X = gpu_contiguous(as_cuda_ndarray_variable(X))
    DY = gpu_contiguous(as_cuda_ndarray_variable(DY))
    assert X.dtype == "float32"
    assert DY.dtype == "float32"
    assert X.ndim == 4
    assert DY.ndim == 4
    return theano.Apply(self, [X, DY], [X.type()])

  def c_support_code(self):
    return get_c_support_code_common() + """
    #include <math_constants.h>

    //h, w, n, c are of input
    __global__ void pool_kernel_bwd(const float * X, const float * DY, float* DX,
                                    int h, int w, int n, int c, int poolHeight, int poolWidth)
    {
      int idx = threadIdx.x + blockDim.x * blockIdx.x;

      //all indices and strides (except for h_stride_big_image) are w.r.t. output (Y)
      int n_stride = c;
      int w_stride = n * n_stride;
      int h_stride = (w / poolWidth) * w_stride;
      int h_stride_big_image = w * w_stride;
      int len = (h / poolHeight) * (w / poolWidth) * n * c;
      while (idx < len)
      {
        int h_idx = idx / h_stride;
        int w_idx = (idx % h_stride) / w_stride;
        int n_idx = (idx % w_stride) / n_stride;
        int c_idx = idx % n_stride;

        float val = -CUDART_INF_F;
        int max_i = 0;
        int max_j = 0;
        for(int i = 0; i < poolHeight; ++i)
        {
          for(int j = 0; j < poolWidth; ++j)
          {
            float newVal = X[(h_idx * poolHeight + i) * h_stride_big_image + (w_idx * poolWidth + j) * w_stride
                             + n_idx * n_stride + c_idx];
            if(newVal > val)
            {
              val = newVal;
              max_i = i;
              max_j = j;
            }
          }
        }

        DX[(h_idx * poolHeight + max_i) * h_stride_big_image + (w_idx * poolWidth + max_j) * w_stride
                             + n_idx * n_stride + c_idx] = DY[h_idx * h_stride + w_idx * w_stride
                                                              + n_idx * n_stride + c_idx];

        idx += gridDim.x * blockDim.x;
      }
    }

    //DX is inplace in X
    //h, w, n, c are of input
    __global__ void pool_kernel_bwd_inplace(float * X, const float * DY,
                                    int h, int w, int n, int c, int poolHeight, int poolWidth)
    {
      int idx = threadIdx.x + blockDim.x * blockIdx.x;

      //all indices and strides (except for h_stride_big_image) are w.r.t. output (Y)
      int n_stride = c;
      int w_stride = n * n_stride;
      int h_stride = (w / poolWidth) * w_stride;
      int h_stride_big_image = w * w_stride;
      int len = (h / poolHeight) * (w / poolWidth) * n * c;
      while (idx < len)
      {
        int h_idx = idx / h_stride;
        int w_idx = (idx % h_stride) / w_stride;
        int n_idx = (idx % w_stride) / n_stride;
        int c_idx = idx % n_stride;

        float val = -CUDART_INF_F;
        int max_i = 0;
        int max_j = 0;
        for(int i = 0; i < poolHeight; ++i)
        {
          for(int j = 0; j < poolWidth; ++j)
          {
            float newVal = X[(h_idx * poolHeight + i) * h_stride_big_image + (w_idx * poolWidth + j) * w_stride
                             + n_idx * n_stride + c_idx];
            if(newVal > val)
            {
              val = newVal;
              max_i = i;
              max_j = j;
            }
          }
        }

        for(int i = 0; i < poolHeight; ++i)
        {
          for(int j = 0; j < poolWidth; ++j)
          {
            X[(h_idx * poolHeight + i) * h_stride_big_image + (w_idx * poolWidth + j) * w_stride
                             + n_idx * n_stride + c_idx] = 0.f;
          }
        }

        X[(h_idx * poolHeight + max_i) * h_stride_big_image + (w_idx * poolWidth + max_j) * w_stride
                             + n_idx * n_stride + c_idx] = DY[h_idx * h_stride + w_idx * w_stride
                                                              + n_idx * n_stride + c_idx];

        idx += gridDim.x * blockDim.x;
      }

      //set borders to 0
      //strides for full image here
      n_stride = c;
      w_stride = n * n_stride;
      h_stride = w * w_stride;

      idx = threadIdx.x + blockDim.x * blockIdx.x;
      len = (h % poolHeight) * h_stride;
      int h_start = h - (h % poolHeight);
      while(idx < len)
      {
        int h_idx = idx / h_stride + h_start;
        int w_idx = (idx % h_stride) / w_stride;
        int n_idx = (idx % w_stride) / n_stride;
        int c_idx = idx % n_stride;
        X[h_idx * h_stride + w_idx * w_stride + n_idx * n_stride + c_idx] = 0.f;

        idx += gridDim.x * blockDim.x;
      }

      idx = threadIdx.x + blockDim.x * blockIdx.x;
      int w_border = (w % poolWidth);
      len = h * w_border * n * c;
      int w_start = w - w_border;
      int n_stride_border = c;
      int w_stride_border = n * n_stride_border;
      int h_stride_border = w_border * w_stride_border;
      while(idx < len)
      {
        int h_idx = idx / h_stride_border;
        int w_idx = (idx % h_stride_border) / w_stride_border + w_start;
        int n_idx = (idx % w_stride_border) / n_stride_border;
        int c_idx = idx % n_stride_border;

        X[h_idx * h_stride + w_idx * w_stride + n_idx * n_stride + c_idx] = 0.f;

        idx += gridDim.x * blockDim.x;
      }
    }

    //h, w, n, c are of input
    __global__ void pool_kernel_bwd_create_argmax(const float * X, const float * DY, unsigned short * argmax,
                                    int h, int w, int n, int c, int poolHeight, int poolWidth)
    {
      int idx = threadIdx.x + blockDim.x * blockIdx.x;

      //all indices and strides (except for h_stride_big_image) are w.r.t. output (Y)
      int n_stride = c;
      int w_stride = n * n_stride;
      int h_stride = (w / poolWidth) * w_stride;
      int h_stride_big_image = w * w_stride;
      int len = (h / poolHeight) * (w / poolWidth) * n * c;
      while (idx < len)
      {
        int h_idx = idx / h_stride;
        int w_idx = (idx % h_stride) / w_stride;
        int n_idx = (idx % w_stride) / n_stride;
        int c_idx = idx % n_stride;

        float val = -CUDART_INF_F;
        int max_i = 0;
        int max_j = 0;
        for(int i = 0; i < poolHeight; ++i)
        {
          for(int j = 0; j < poolWidth; ++j)
          {
            float newVal = X[(h_idx * poolHeight + i) * h_stride_big_image + (w_idx * poolWidth + j) * w_stride
                             + n_idx * n_stride + c_idx];
            if(newVal > val)
            {
              val = newVal;
              max_i = i;
              max_j = j;
            }
          }
        }
        argmax[h_idx * h_stride + w_idx * w_stride + n_idx * n_stride + c_idx] = max_i + 256 * max_j;
        idx += gridDim.x * blockDim.x;
      }
    }

    __global__ void pool_kernel_bwd_BCHW_grad_output(float * DX, const float * DY, const unsigned short * argmax,
                                    int h, int w, int n, int c, int poolHeight, int poolWidth)
    {
      //slightly inconsistent: b for batch is also called n (b == n)
      int idx = threadIdx.x + blockDim.x * blockIdx.x;
      int len = (h / poolHeight) * (w / poolWidth) * n * c;

      int n_stride = c;
      int w_stride = n * n_stride;
      int h_stride = (w / poolWidth) * w_stride;

      int w_stride_bchw = 1;
      int h_stride_bchw = w;
      int c_stride_bchw = h * w;
      int n_stride_bchw = c * h * w;
      while(idx < len)
      {
        int h_idx = idx / h_stride;
        int w_idx = (idx % h_stride) / w_stride;
        int n_idx = (idx % w_stride) / n_stride;
        int c_idx = idx % n_stride;
        int DY_idx = h_idx * h_stride + w_idx * w_stride + n_idx * n_stride + c_idx;

        //argmax[DY_idx] == max_i + 256 * max_j;
        int max_i = argmax[DY_idx] % 256;
        int max_j = argmax[DY_idx] / 256;
        //BCHW layout
        DX[(h_idx * poolHeight + max_i) * h_stride_bchw + (w_idx * poolWidth + max_j) * w_stride_bchw
                             + n_idx * n_stride_bchw + c_idx * c_stride_bchw] = DY[DY_idx];

        idx += gridDim.x * blockDim.x;
      }
    }
    """

  def c_code(self, node, name, input_names, output_names, sub):
    X, DY = input_names
    DX, = output_names
    poolHeight, poolWidth = self.pool_shape
    inplace = "true" if self.inplace else "false"
    BCHW_grad_output = "true" if self.BCHW_grad_output else "false"
    fail = sub['fail']
    return """

    const int * X_dim = CudaNdarray_HOST_DIMS(%(X)s);
    int h = X_dim[0];
    int w = X_dim[1];
    int n = X_dim[2];
    int c = X_dim[3];
    if(%(inplace)s)
    {
      %(DX)s = %(X)s;
      Py_XINCREF(%(X)s);
    }
    else
    {
      %(DX)s = CudaNdarray_zeros_like(%(X)s);
      cout << "warning, PoolHWBCOpGrad inplace optimization failed, not working inplace" << endl;
    }
    const float * X_data = CudaNdarray_DEV_DATA(%(X)s);
    const float * DY_data = CudaNdarray_DEV_DATA(%(DY)s);
    float * DX_data = CudaNdarray_DEV_DATA(%(DX)s);
    if(%(inplace)s)
    {
      if(%(BCHW_grad_output)s)
      {
        //CudaNdarray is always float, but we use 2-byte unsigned shorts, so divide by 2
        int argmax_dims[] = {(CudaNdarray_SIZE(%(DY)s) + 1) / 2};
        CudaNdarray * argmax = (CudaNdarray*) MyCudaNdarray_NewDims(1, argmax_dims);
	assert(argmax);
        unsigned short * argmax_data = reinterpret_cast<unsigned short*>(CudaNdarray_DEV_DATA(argmax));
        pool_kernel_bwd_create_argmax<<<DIM_GRID, DIM_BLOCK>>>(DX_data, DY_data, argmax_data,
                                                                          h, w, n, c, %(poolHeight)s, %(poolWidth)s);
        CHECK_KERNEL_ERROR();
        HANDLE_ERROR(cudaMemset(DX_data, 0, sizeof(float) * CudaNdarray_SIZE(%(DX)s)));
        pool_kernel_bwd_BCHW_grad_output<<<DIM_GRID, DIM_BLOCK>>>(DX_data, DY_data, argmax_data,
                                                                          h, w, n, c, %(poolHeight)s, %(poolWidth)s);
        CHECK_KERNEL_ERROR();
        Py_XDECREF(argmax);
        CudaNdarray_set_dim(%(DX)s, 0, n);
        CudaNdarray_set_dim(%(DX)s, 1, c);
        CudaNdarray_set_dim(%(DX)s, 2, h);
        CudaNdarray_set_dim(%(DX)s, 3, w);
        CudaNdarray_set_stride(%(DX)s, 0, h * w * c);
        CudaNdarray_set_stride(%(DX)s, 1, h * w);
        CudaNdarray_set_stride(%(DX)s, 2, w);
        CudaNdarray_set_stride(%(DX)s, 3, 1);
      }
      else
      {
        //cout << "warning, RemoveConvGradDimshuffle optimization failed" << endl;
        //TODO: this optimization seems not to be applied anymore...
        pool_kernel_bwd_inplace<<<DIM_GRID, DIM_BLOCK>>>(DX_data, DY_data, h, w, n, c, %(poolHeight)s, %(poolWidth)s);
        CHECK_KERNEL_ERROR();
      }
    }
    else
    {
      pool_kernel_bwd<<<DIM_GRID, DIM_BLOCK>>>(X_data, DY_data, DX_data, h, w, n, c, %(poolHeight)s, %(poolWidth)s);
      CHECK_KERNEL_ERROR();
    }
    """ % locals()

  def infer_shape(self, node, input_shapes):
    return input_shapes[:1]

  def c_code_cache_version(self):
    return 2, 3


class RemoveConvGradDimshuffle(gof.Optimizer):
  def add_requirements(self, fgraph):
    fgraph.attach_feature(toolbox.ReplaceValidate())

  def apply(self, fgraph):
    for node in fgraph.toposort():
      #print node
      if type(node.op) == GpuDimShuffle and node.op.new_order == (2, 3, 0, 1):
        X = node.inputs[0]
        if hasattr(X.owner, "op") and type(X.owner.op) == PoolHWBCOpGrad and X.owner.op.inplace:
          fgraph.replace_validate(node.outputs[0], node.inputs[0])
          replace_op = PoolHWBCOpGrad(X.owner.op.pool_shape, inplace=True, BCHW_grad_output=True)
          fgraph.replace_validate(X.owner.outputs[0], replace_op(*X.owner.inputs))

RemoveConvGradDimshuffleOptimizer = RemoveConvGradDimshuffle()
if not hasattr(optdb, 'RemoveConvGradDimshuffleOptimizer_registered'):
  optdb.register('RemoveConvGradDimshuffle', RemoveConvGradDimshuffleOptimizer, 50.5, 'fast_run', 'inplace', 'gpuarray')
  optdb.RemoveConvGradDimshuffleOptimizer_registered = True

#---------------------------


#for the moment we implement only ignore_border = True and no padding
class PoolHWBCOp(theano.sandbox.cuda.GpuOp):
  __props__ = ("pool_shape",)

  def __init__(self, pool_shape):
    pool_shape = tuple(pool_shape)
    super(PoolHWBCOp, self).__init__()
    assert len(pool_shape) == 2, len(pool_shape)
    assert pool_shape[0] > 0, pool_shape[0]
    assert pool_shape[1] > 0, pool_shape[1]
    self.pool_shape = pool_shape

  def make_node(self, X):
    X = gpu_contiguous(as_cuda_ndarray_variable(X))
    assert X.dtype == "float32"
    assert X.ndim == 4
    return theano.Apply(self, [X], [X.type()])

  def c_support_code(self):
    return get_c_support_code_common() + """
    #include <math_constants.h>

    __global__ void pool_kernel(const float * X, float * Y, int h, int w, int n, int c, int poolHeight, int poolWidth)
    {
      int idx = threadIdx.x + blockDim.x * blockIdx.x;

      //all indices and strides (except for h_stride_big_image) are w.r.t. output (Y)
      int n_stride = c;
      int w_stride = n * n_stride;
      int h_stride = (w / poolWidth) * w_stride;
      int h_stride_big_image = w * w_stride;
      int len = (h / poolHeight) * (w / poolWidth) * n * c;
      while (idx < len)
      {
        int h_idx = idx / h_stride;
        int w_idx = (idx % h_stride) / w_stride;
        int n_idx = (idx % w_stride) / n_stride;
        int c_idx = idx % n_stride;

        float val = -CUDART_INF_F;
        for(int i = 0; i < poolHeight; ++i)
        {
          for(int j = 0; j < poolWidth; ++j)
          {
            float newVal = X[(h_idx * poolHeight + i) * h_stride_big_image + (w_idx * poolWidth + j) * w_stride
                             + n_idx * n_stride + c_idx];
            val = max(val, newVal);
          }
        }

        Y[h_idx * h_stride + w_idx * w_stride + n_idx * n_stride + c_idx] = val;

        idx += gridDim.x * blockDim.x;
      }
    }
    """

  def c_code(self, node, name, input_names, output_names, sub):
    X, = input_names
    Y, = output_names
    poolHeight, poolWidth = self.pool_shape
    fail = sub['fail']
    return """
    const int * X_dim = CudaNdarray_HOST_DIMS(%(X)s);
    int h = X_dim[0];
    int w = X_dim[1];
    int n = X_dim[2];
    int c = X_dim[3];
    int Y_dim[] = {h / %(poolHeight)s, w / %(poolWidth)s, n, c};
    %(Y)s = (CudaNdarray*) MyCudaNdarray_NewDims(4, Y_dim);
    assert(%(Y)s);
    const float * X_data = CudaNdarray_DEV_DATA(%(X)s);
    float * Y_data = CudaNdarray_DEV_DATA(%(Y)s);
    pool_kernel<<<DIM_GRID, DIM_BLOCK>>>(X_data, Y_data, h, w, n, c, %(poolHeight)s, %(poolWidth)s);
    CHECK_KERNEL_ERROR();
    """ % locals()

  def grad(self, inputs, output_grads):
    X, = inputs
    DY, = output_grads
    grad_instance = PoolHWBCOpGrad(self.pool_shape, inplace=False, BCHW_grad_output=False)
    DX = grad_instance(X, DY)
    return [DX]

  def infer_shape(self, node, input_shapes):
    Xs, = input_shapes
    h, w, n, c = Xs
    h //= self.pool_shape[0]
    w //= self.pool_shape[1]
    return [(h, w, n, c)]

  def c_code_cache_version(self):
    return 2, 3
