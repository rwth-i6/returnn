import theano
from theano import gof
from theano.compile import optdb
from theano.gof import OpSub
from theano.sandbox.cuda.basic_ops import gpu_contiguous, as_cuda_ndarray_variable
from .Util import get_c_support_code_common


class CropToBatchImageSizeOp(theano.sandbox.cuda.GpuOp):
  __props__ = ("fill_val", "inplace")

  def __init__(self, fill_val, inplace):
    super(CropToBatchImageSizeOp, self).__init__()
    self.fill_val = fill_val
    self.inplace = inplace
    if inplace:
      self.destroy_map = {0: [0]}

  def make_node(self, X, sizes):
    X = gpu_contiguous(as_cuda_ndarray_variable(X))
    sizes = gpu_contiguous(as_cuda_ndarray_variable(sizes))
    assert X.dtype == "float32"
    assert X.ndim == 4
    assert sizes.dtype == "float32"
    assert sizes.ndim == 2
    return theano.Apply(self, [X, sizes], [X.type()])

  def c_support_code(self):
    return get_c_support_code_common() + """
      __global__ void crop_kernel(float * Y, const float * X, const float * sizes, int height, int width,
                                  int n_batch, int n_feat, float fill_val)
      {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int batch_stride = n_feat;
        int x_stride = batch_stride * n_batch;
        int y_stride = x_stride * width;
        int len = y_stride * height;
        while (idx < len)
        {
          int y = idx / y_stride;
          int x = (idx % y_stride) / x_stride;
          int n = (idx % x_stride) / batch_stride;
          //sizes of this particular image of the minibatch
          int img_height = sizes[2 * n];
          int img_width = sizes[2 * n + 1];
          if(y < img_height && x < img_width)
          {
            Y[idx] = X[idx];
          }
          else
          {
            //set to real low value, so it will not change the max pooling
            Y[idx] = fill_val;
          }
          idx += gridDim.x * blockDim.x;
        }
      }
    """

  def c_code(self, node, name, input_names, output_names, sub):
    X, sizes = input_names
    Y, = output_names
    fill_val = str(self.fill_val)
    inplace = "true" if self.inplace else "false"
    fail = sub['fail']
    return """
    if(%(inplace)s)
    {
      %(Y)s = %(X)s;
      Py_XINCREF(%(X)s);
    }
    else
    {
      cout << "warning, CropToBatchImageSizeOp inplace optimization failed, not working inplace" << endl;
      %(Y)s = CudaNdarray_uninitialized_like(%(X)s);
    }
    float * Y_data = CudaNdarray_DEV_DATA(%(Y)s);
    const float * X_data = CudaNdarray_DEV_DATA(%(X)s);
    const float * sizes_data = CudaNdarray_DEV_DATA(%(sizes)s);
    int height = CudaNdarray_HOST_DIMS(%(X)s)[0];
    int width = CudaNdarray_HOST_DIMS(%(X)s)[1];
    int n_batch = CudaNdarray_HOST_DIMS(%(X)s)[2];
    int n_feat = CudaNdarray_HOST_DIMS(%(X)s)[3];
    crop_kernel<<<DIM_GRID, DIM_BLOCK>>>(Y_data, X_data, sizes_data, height, width, n_batch, n_feat, %(fill_val)s);
    CHECK_KERNEL_ERROR();
    """ % locals()

  def grad(self, inputs, output_grads):
    X, sizes = inputs
    DY, = output_grads
    DX = CropToBatchImageSizeZeroInstance(DY, sizes)
    Dsizes = theano.gradient.grad_undefined(self, len(inputs) - 1, inputs[-1], 'cannot diff w.r.t. sizes')
    return [DX, Dsizes]

  # noinspection PyMethodMayBeStatic
  def infer_shape(self, node, input_shapes):
    return input_shapes[0],

  #def c_code_cache_version(self):
  #  return 1, 0


CropToBatchImageSizeInstance = CropToBatchImageSizeOp(-1e20, False)
CropToBatchImageSizeInplaceInstance = CropToBatchImageSizeOp(-1e20, True)
CropToBatchImageSizeZeroInstance = CropToBatchImageSizeOp(0.0, False)
CropToBatchImageSizeZeroInplaceInstance = CropToBatchImageSizeOp(0.0, True)


CropToBatchImageSizeGradInplaceOpt1 = OpSub(CropToBatchImageSizeInstance, CropToBatchImageSizeInplaceInstance)
#hack to avoid being called twice
if not hasattr(optdb, 'CropToBatchImageSizeGradInplaceOpt1_registered'):
  optdb.register('CropToBatchImageSizeGradInplaceOpt1',
                 theano.gof.TopoOptimizer(CropToBatchImageSizeGradInplaceOpt1,
                                          failure_callback=gof.TopoOptimizer.warn_inplace),
                 50.0, 'fast_run', 'inplace', 'gpuarray')
  optdb.CropToBatchImageSizeGradInplaceOpt1_registered = True

CropToBatchImageSizeGradInplaceOpt2 = OpSub(CropToBatchImageSizeZeroInstance, CropToBatchImageSizeZeroInplaceInstance)
#hack to avoid being called twice
if not hasattr(optdb, 'CropToBatchImageSizeGradInplaceOpt2_registered'):
  optdb.register('CropToBatchImageSizeGradInplaceOpt2',
                 theano.gof.TopoOptimizer(CropToBatchImageSizeGradInplaceOpt2,
                                          failure_callback=gof.TopoOptimizer.warn_inplace),
                 50.0, 'fast_run', 'inplace', 'gpuarray')
  optdb.CropToBatchImageSizeGradInplaceOpt2_registered = True
