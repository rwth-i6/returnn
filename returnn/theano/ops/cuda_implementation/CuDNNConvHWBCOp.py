import os
import theano.sandbox.cuda
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable, gpu_contiguous)
from theano import gof
from theano.gof.opt import OpSub
from theano.compile import optdb
from .Util import get_c_support_code_common, get_c_support_code_cudnn


class CuDNNConvHWBCOpGrad(theano.sandbox.cuda.GpuOp):
  __props__ = ("border_mode", "inplace")

  def __init__(self, border_mode, inplace):
    super(CuDNNConvHWBCOpGrad, self).__init__()
    assert border_mode in ("valid", "full"), border_mode
    self.border_mode = border_mode
    self.inplace = inplace
    if inplace:
      #DX inplace in X
      self.destroy_map = {0: [0]}

  def make_node(self, X, W, b, DY):
    X = gpu_contiguous(as_cuda_ndarray_variable(X))
    W = gpu_contiguous(as_cuda_ndarray_variable(W))
    b = gpu_contiguous(as_cuda_ndarray_variable(b))
    DY = gpu_contiguous(as_cuda_ndarray_variable(DY))
    assert X.dtype == "float32"
    assert W.dtype == "float32"
    assert b.dtype == "float32"
    assert DY.dtype == "float32"
    assert X.ndim == 4
    assert W.ndim == 4
    assert b.ndim == 1
    assert DY.ndim == 4
    return theano.Apply(self, [X, W, b, DY], [X.type(), W.type(), b.type()])

  def c_support_code(self):
    support_code = get_c_support_code_common() + get_c_support_code_cudnn()
    return support_code + """
    static cudnnHandle_t cudnnHandle = 0;
    static cudnnTensorDescriptor_t srcTensorDesc = 0;
    static cudnnTensorDescriptor_t diffTensorDesc = 0;
    static cudnnTensorDescriptor_t srcGradDesc = 0;
    static cudnnTensorDescriptor_t biasGradTensorDesc = 0;
    static cudnnFilterDescriptor_t filterDesc = 0;
    static cudnnConvolutionDescriptor_t convDesc = 0;
    """

  def c_code(self, node, name, input_names, output_names, sub):
    X, W, b, DY = input_names
    DX, DW, Db = output_names
    cudnn_version = theano.sandbox.cuda.dnn.version()[0]/1000
    if cudnn_version < 5:
      cudnn_version_macro = "#define CUDNN_SMALLER_5"
    elif cudnn_version > 6:
      cudnn_version_macro = "#define CUDNN_GREATER_6"
    else:
      cudnn_version_macro = ""
    full_conv = "true" if self.border_mode == "full" else "false"
    inplace = "true" if self.inplace else "false"
    fail = sub['fail']
    return """
    //cout << "inplace: " << %(inplace)s << endl;

    %(cudnn_version_macro)s

    if(cudnnHandle == 0)
    {
      checkCUDNN(cudnnCreate(&cudnnHandle));
      checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
      checkCUDNN(cudnnCreateTensorDescriptor(&diffTensorDesc));
      checkCUDNN(cudnnCreateTensorDescriptor(&biasGradTensorDesc));
      checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
      checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    }
    assert(cudnnHandle);
    assert(srcTensorDesc);
    assert(diffTensorDesc);
    assert(biasGradTensorDesc);
    assert(filterDesc);
    assert(convDesc);

    const int * X_dim = CudaNdarray_HOST_DIMS(%(X)s);
    const int * W_dim = CudaNdarray_HOST_DIMS(%(W)s);
    const int * DY_dim = CudaNdarray_HOST_DIMS(%(DY)s);
    const int * b_dim = CudaNdarray_HOST_DIMS(%(b)s);

    int n = X_dim[2];
    int c_src = X_dim[3];
    int h_src = X_dim[0];
    int w_src = X_dim[1];
    int c_filter = W_dim[0];
    int h_filter = W_dim[2];
    int w_filter = W_dim[3];
    int c_diff = DY_dim[1];
    int h_diff = DY_dim[2];
    int w_diff = DY_dim[3];

    int n_src_stride = c_src;
    int c_src_stride = 1;
    int h_src_stride = c_src * n * w_src;
    int w_src_stride = c_src * n;

    if(%(inplace)s)
    {
      %(DX)s = %(X)s;
      Py_XINCREF(%(X)s);
    }
    else
    {
      %(DX)s = (CudaNdarray*) MyCudaNdarray_NewDims(4, X_dim);
      assert(%(DX)s);
    }
    %(DW)s = (CudaNdarray*) MyCudaNdarray_NewDims(4, W_dim);
    assert(%(DW)s);
    %(Db)s = (CudaNdarray*) MyCudaNdarray_NewDims(1, b_dim);
    assert(%(Db)s);

    const float * srcData = CudaNdarray_DEV_DATA(%(X)s);
    const float * filterData = CudaNdarray_DEV_DATA(%(W)s);
    const float * diffData = CudaNdarray_DEV_DATA(%(DY)s);
    float * filterGradData = CudaNdarray_DEV_DATA(%(DW)s);
    float * inputGradData = CudaNdarray_DEV_DATA(%(DX)s);
    float * biasGradData = CudaNdarray_DEV_DATA(%(Db)s);

    checkCUDNN(cudnnSetTensor4dDescriptorEx(srcTensorDesc, CUDNN_DATA_FLOAT,
                                            n, c_src, h_src, w_src,
                                            n_src_stride, c_src_stride, h_src_stride, w_src_stride));
    checkCUDNN(cudnnSetTensor4dDescriptor(diffTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          n, c_diff, h_diff, w_diff));
    checkCUDNN(cudnnSetTensor4dDescriptor(biasGradTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c_filter, 1, 1));

    //here we add padding to support "full" convolution mode
    int y_pad = 0;
    int x_pad = 0;
    if(%(full_conv)s)
    {
      y_pad = h_filter - 1;
      x_pad = w_filter - 1;
    }
    if(x_pad < w_filter - w_src)
    {
      x_pad = w_filter - w_src;
    }
    if(y_pad < h_filter - h_src)
    {
      y_pad = h_filter - h_src;
    }
    #ifdef CUDNN_GREATER_6
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, y_pad, x_pad, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
    #else
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, y_pad, x_pad, 1, 1, 1, 1, CUDNN_CONVOLUTION));
    #endif
    #ifdef CUDNN_SMALLER_5
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, c_filter, c_src, h_filter, w_filter));
    #else
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, c_filter, c_src, h_filter, w_filter));
    #endif

    //do the actual computations
    float alpha = 1.0;
    float beta = 0.0;
    checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, diffTensorDesc, diffData, &beta,
                                            biasGradTensorDesc, biasGradData));
    checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, srcTensorDesc, srcData, diffTensorDesc, diffData,
                                                 convDesc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, 0, 0, &beta, filterDesc,
                                                 filterGradData));
    checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle, &alpha, filterDesc, filterData, diffTensorDesc, diffData,
                                               convDesc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, 0, 0, &beta, srcTensorDesc,
                                               inputGradData));
    """ % locals()

  def c_libraries(self):
    return ['cudnn']

  #def infer_shape(self, node, input_shapes):
  #  pass

  def c_code_cache_version(self):
    return 3, 3

CuDNNConvHWBCOpGradValidNoInplaceInstance = CuDNNConvHWBCOpGrad("valid", inplace=False)
CuDNNConvHWBCOpGradValidInplaceInstance = CuDNNConvHWBCOpGrad("valid", inplace=True)
CuDNNConvHWBCOpGradFullNoInplaceInstance = CuDNNConvHWBCOpGrad("full", inplace=False)
CuDNNConvHWBCOpGradFullInplaceInstance = CuDNNConvHWBCOpGrad("full", inplace=True)

CuDNNConvHWBCOpGradValidInplaceOpt = OpSub(CuDNNConvHWBCOpGradValidNoInplaceInstance, CuDNNConvHWBCOpGradValidInplaceInstance)
#hack to avoid being called twice
if not hasattr(optdb, 'CuDNNConvHWBCOpGradValidInplaceOpt_registered'):
  optdb.register('CuDNNConvHWBCOpGradValidInplaceOpt',
                 theano.gof.TopoOptimizer(CuDNNConvHWBCOpGradValidInplaceOpt, failure_callback=gof.TopoOptimizer.warn_inplace),
                 50.0, 'fast_run', 'inplace', 'gpuarray')
  optdb.CuDNNConvHWBCOpGradValidInplaceOpt_registered = True

#TODO: maybe this optimization causes problems
#CuDNNConvHWBCOpGradFullInplaceOpt = OpSub(CuDNNConvHWBCOpGradFullNoInplaceInstance, CuDNNConvHWBCOpGradFullInplaceInstance)
##hack to avoid being called twice
#if not hasattr(optdb, 'CuDNNConvHWBCOpGradFullInplaceOpt_registered'):
#  optdb.register('CuDNNConvHWBCOpGradFullInplaceOpt',
#                 theano.gof.TopoOptimizer(CuDNNConvHWBCOpGradFullInplaceOpt, failure_callback=gof.TopoOptimizer.warn_inplace),
#                 50.0, 'fast_run', 'inplace', 'gpuarray')
#  optdb.CuDNNConvHWBCOpGradFullInplaceOpt_registered = True

#------------------------------------------------------


class CuDNNConvHWBCOp(theano.sandbox.cuda.GpuOp):
  __props__ = ("border_mode",)

  def __init__(self, border_mode):
    super(CuDNNConvHWBCOp, self).__init__()
    assert border_mode in ("valid", "full"), border_mode
    self.border_mode = border_mode

  def make_node(self, X, W, b):
    X = gpu_contiguous(as_cuda_ndarray_variable(X))
    W = gpu_contiguous(as_cuda_ndarray_variable(W))
    b = gpu_contiguous(as_cuda_ndarray_variable(b))
    assert X.dtype == "float32"
    assert W.dtype == "float32"
    assert b.dtype == "float32"
    assert X.ndim == 4
    assert W.ndim == 4
    assert b.ndim == 1
    return theano.Apply(self, [X, W, b], [X.type()])

  def c_support_code(self):
    support_code = get_c_support_code_common() + get_c_support_code_cudnn()
    return support_code + """
    static cudnnHandle_t cudnnHandle = 0;
    static cudnnTensorDescriptor_t srcTensorDesc = 0;
    static cudnnTensorDescriptor_t dstTensorDesc = 0;
    static cudnnTensorDescriptor_t biasTensorDesc = 0;
    static cudnnFilterDescriptor_t filterDesc = 0;
    static cudnnConvolutionDescriptor_t convDesc = 0;
    """

  def c_code(self, node, name, input_names, output_names, sub):
    X, W, b = input_names
    Y, = output_names
    cudnn_version = theano.sandbox.cuda.dnn.version()[0]/1000
    if cudnn_version < 5:
      cudnn_version_macro = "#define CUDNN_SMALLER_5"
    elif cudnn_version > 6:
      cudnn_version_macro = "#define CUDNN_GREATER_6"
    else:
      cudnn_version_macro = ""
    full_conv = "true" if self.border_mode == "full" else "false"
    fail = sub['fail']
    return """
    %(cudnn_version_macro)s

    if(cudnnHandle == 0)
    {
      checkCUDNN(cudnnCreate(&cudnnHandle));
      checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
      checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
      checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
      checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
      checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    }
    assert(cudnnHandle);
    assert(srcTensorDesc);
    assert(dstTensorDesc);
    assert(biasTensorDesc);
    assert(filterDesc);
    assert(convDesc);

    const float * srcData = CudaNdarray_DEV_DATA(%(X)s);
    const float * filterData = CudaNdarray_DEV_DATA(%(W)s);
    const float * biasData = CudaNdarray_DEV_DATA(%(b)s);
    const int * X_dim = CudaNdarray_HOST_DIMS(%(X)s);
    const int * W_dim = CudaNdarray_HOST_DIMS(%(W)s);
    int n = X_dim[2];
    int c_src = X_dim[3];
    int h_src = X_dim[0];
    int w_src = X_dim[1];
    int c_filter = W_dim[0];
    int h_filter = W_dim[2];
    int w_filter = W_dim[3];

    int n_src_stride = c_src;
    int c_src_stride = 1;
    int h_src_stride = c_src * n * w_src;
    int w_src_stride = c_src * n;

    checkCUDNN(cudnnSetTensor4dDescriptorEx(srcTensorDesc, CUDNN_DATA_FLOAT,
                                            n, c_src, h_src, w_src,
                                            n_src_stride, c_src_stride, h_src_stride, w_src_stride));

    #ifdef CUDNN_SMALLER_5
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, c_filter, c_src, h_filter, w_filter));
    #else
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, c_filter, c_src, h_filter, w_filter));
    #endif

    //here we add padding to support "full" convolution mode
    int y_pad = 0;
    int x_pad = 0;
    if(%(full_conv)s)
    {
      y_pad = h_filter - 1;
      x_pad = w_filter - 1;
    }
    if(x_pad < w_filter - w_src)
    {
      x_pad = w_filter - w_src;
    }
    if(y_pad < h_filter - h_src)
    {
      y_pad = h_filter - h_src;
    }
    #ifdef CUDNN_GREATER_6
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, y_pad, x_pad, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
    #else
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, y_pad, x_pad, 1, 1, 1, 1, CUDNN_CONVOLUTION));
    #endif

    int n_out = 0, c_out = 0, h_out = 0, w_out = 0;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, srcTensorDesc, filterDesc, &n_out, &c_out, &h_out, &w_out));

    int n_out_stride = c_out;
    int c_out_stride = 1;
    int w_out_stride = n_out_stride * n_out;
    int h_out_stride = w_out_stride * w_out;

    checkCUDNN(cudnnSetTensor4dDescriptorEx(dstTensorDesc, CUDNN_DATA_FLOAT,
                                          n_out, c_out, h_out, w_out, n_out_stride, c_out_stride, h_out_stride, w_out_stride));
    checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c_out, 1, 1));

    int Y_dims[] = {h_out, w_out, n_out, c_out};
    %(Y)s = (CudaNdarray*) MyCudaNdarray_NewDims(4, Y_dims);
    assert(%(Y)s);
    float * dstData = CudaNdarray_DEV_DATA(%(Y)s);

    //do the actual computations
    float alpha = 1.0f;
    float beta = 0.0f;
    checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, srcTensorDesc, srcData, filterDesc, filterData, convDesc,
                                       CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 0, 0, &beta, dstTensorDesc, dstData));

    //add bias
    beta = 1.0f;
    checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, biasTensorDesc, biasData, &beta,
                               dstTensorDesc, dstData));
    """ % locals()

  def c_libraries(self):
    return ['cudnn']

  def grad(self, inputs, output_grads):
    X, W, b = inputs
    DY, = output_grads
    DY_shuffled = DY.dimshuffle(2, 3, 0, 1)

    grad_instance = CuDNNConvHWBCOpGradValidNoInplaceInstance if self.border_mode == "valid" \
        else CuDNNConvHWBCOpGradFullNoInplaceInstance
    DX, DW, Db = grad_instance(X, W, b, DY_shuffled)

    return [DX, DW, Db]

  #def infer_shape(self, node, input_shapes):
  #  pass

  def c_code_cache_version(self):
    return 3, 3

CuDNNConvHWBCOpValidInstance = CuDNNConvHWBCOp("valid")
CuDNNConvHWBCOpFullInstance = CuDNNConvHWBCOp("full")
