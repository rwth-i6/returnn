import numpy
import theano
import theano.gradient
import theano.tensor as T
import theano.printing
import theano.gof
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           gpu_contiguous)
from theano.gof.opt import OpSub
from theano.compile import optdb
import os

class LSTMOpCellGrad(theano.sandbox.cuda.GpuOp):
  def __init__(self, inplace):
    self.inplace = inplace
    if inplace:
      #all outputs operate inplace on inputs 4 and 6 (which are DZ and H)
      #but when the input is marked multiple times, we get an error
      #so we only mark that output 0 destroys inputs 4 and 6
      #anyway theano knows that inputs 4 and 6 will be destroyed, so it should be OK
      #TODO
      #self.destroy_map = {0: [4], 1: [6]}
      #self.destroy_map = {0: [4]}
      self.destroy_map = {}

  def __eq__(self, other):
    return type(self) == type(other) and self.inplace == other.inplace

  def __str__(self):
    if self.inplace:
      return '%s{inplace}' % self.__class__.__name__
    else:
      return '%s{no_inplace}' % self.__class__.__name__

  def __hash__(self):
    return hash(type(self)) ^ hash(self.inplace)

  def make_node(self, V_h, c, idx, Dd, DY, Y, H):
    V_h = gpu_contiguous(as_cuda_ndarray_variable(V_h))
    c = gpu_contiguous(as_cuda_ndarray_variable(c))
    DY = gpu_contiguous(as_cuda_ndarray_variable(DY))
    idx = gpu_contiguous(as_cuda_ndarray_variable(idx))
    Dd = gpu_contiguous(as_cuda_ndarray_variable(Dd))
    assert V_h.dtype == "float32"
    assert DY.dtype == 'float32'
    assert Y.dtype == 'float32'
    assert H.dtype == 'float32'
    assert c.dtype == 'float32'
    assert V_h.ndim == 2
    assert DY.ndim == 2
    assert Y.ndim == 2
    assert H.ndim == 2
    assert c.ndim == 2
    assert idx.ndim == 1

    return theano.Apply(self, [V_h, c, idx, Dd, DY, Y, H], [H.type(), V_h.type(), c.type()])

  #def infer_shape(self, node, input_shapes):
  #  V_hs, cs, idxs, Dds, DYs, Ys, Hs = input_shapes
  #  return [Hs, V_hs, cs]

  def c_support_code(self):
    crnn_path = os.path.dirname(__file__)
    with open(crnn_path + "/c_support_code_mdlstm.cpp") as f:
      return f.read()

  def c_code(self, node, name, input_names, output_names, sub):
    V_h, c, i, Dd, DY, Y, H = input_names
    DZ, DV_h, Dc = output_names
    fail = sub['fail']
    inplace = "true" if self.inplace else "false"
    return """

    // std::cout << "LSTMOpCellGrad called" << std::endl;
    if(!%(inplace)s)
    {
      //std::cout << "warning, inplace optimization failed, not working inplace" << std::endl;
    }

    if(%(DZ)s || %(DV_h)s || %(Dc)s)
    {
      //printf("output storage already exists\\n");
      //TODO check if we can reuse it
      Py_XDECREF(%(DZ)s);
      Py_XDECREF(%(DV_h)s);
      Py_XDECREF(%(Dc)s);
    }

    CudaNdarray * epsilon = 0;
    CudaNdarray * delta = 0;
    if(%(inplace)s)
    {
      epsilon = %(DY)s;
      delta = %(H)s;
      Py_XINCREF(delta);
    }
    else
    {
      epsilon = (CudaNdarray *) CudaNdarray_Copy(%(DY)s);
      delta = (CudaNdarray *) CudaNdarray_Copy(%(H)s);
    }

    const int * H_dim = CudaNdarray_HOST_DIMS(%(H)s);

    int y = 0;
    int x = 0;
    do_lstm_bwd(delta, epsilon, %(Y)s, %(Dd)s, %(c)s, y, x, true, %(i)s);

    %(DV_h)s = CudaNdarray_uninitialized_like(%(V_h)s);
    affine_global(%(Y)s, delta, %(DV_h)s, true, false, 1, 0.0f);

    %(DZ)s = delta;

    %(Dc)s = CudaNdarray_uninitialized_like(%(c)s);
    const int * Y_dim = CudaNdarray_HOST_DIMS(%(Y)s);
    cudaMemcpy(CudaNdarray_DEV_DATA(%(Dc)s), CudaNdarray_DEV_DATA(epsilon),
      Y_dim[0]*sizeof(float), cudaMemcpyDeviceToDevice);

    if(!%(inplace)s)
    {
      Py_XDECREF(epsilon);
    }

    """ % locals()

  #!!! change this when changing the code!
  #def c_code_cache_version(self):
  #  return 1, 2

LSTMOpCellGradNoInplaceInstance = LSTMOpCellGrad(inplace=False)
LSTMOpCellGradInplaceInstance = LSTMOpCellGrad(inplace=True)

LSTMOpCellGradInplaceOpt = OpSub(LSTMOpCellGradNoInplaceInstance, LSTMOpCellGradInplaceInstance)

#hack to avoid being called twice
if not hasattr(optdb, 'LSTMOpCellGradInplaceOpt_registered'):
  optdb.register('LSTMOpCellGradInplaceOpt', theano.gof.TopoOptimizer(LSTMOpCellGradInplaceOpt),
                 50.0, 'fast_run', 'inplace', 'gpuarray')
  optdb.LSTMOpCellGradInplaceOpt_registered = True


#------------------------

class LSTMOpCell(theano.sandbox.cuda.GpuOp):
  def __init__(self, inplace):
    self.inplace = inplace
    if inplace:
      #all outputs operate inplace on input 0 (which is Z)
      #but when the input is marked multiple times, we get an error
      #so we only mark that output 0 destroys input 0
      #anyway theano knows that input 0 will be destroyed, so it should be OK
      #TODO
      #self.destroy_map = {0: [0]}
      self.destroy_map = {}

  def __eq__(self, other):
    return type(self) == type(other) and self.inplace == other.inplace

  def __str__(self):
    if self.inplace:
      return '%s{inplace}' % self.__class__.__name__
    else:
      return '%s{no_inplace}' % self.__class__.__name__

  def __hash__(self):
    return hash(type(self)) ^ hash(self.inplace)

  def make_node(self, Z, V_h, c, i):
    Z = gpu_contiguous(as_cuda_ndarray_variable(Z))
    V_h = gpu_contiguous(as_cuda_ndarray_variable(V_h))
    c = gpu_contiguous(as_cuda_ndarray_variable(c))
    i = gpu_contiguous(as_cuda_ndarray_variable(i))
    assert Z.dtype == "float32"
    assert V_h.dtype == "float32"
    assert c.dtype == 'float32'
    assert c.ndim == 2
    assert Z.ndim == 2
    assert i.ndim == 1
    assert V_h.ndim == 2

    #results: output Y, (gates and cell state) H
    return theano.Apply(self, [Z, V_h, c, i], [Z.type(), Z.type(), c.type()])

  def c_support_code(self):
    crnn_path = os.path.dirname(__file__)
    with open(crnn_path + "/c_support_code_mdlstm.cpp") as f:
      return f.read()

  def c_code(self, node, name, input_names, output_names, sub):
    X, V_h, c, i = input_names
    Z, H, d = output_names
    fail = sub['fail']
    return """
    if(%(Z)s || %(H)s || %(d)s)
    {
      //printf("Z or H already exist\\n");
      //TODO check if we can reuse it
      Py_XDECREF(%(Z)s);
      Py_XDECREF(%(H)s);
      Py_XDECREF(%(d)s);
    }

    const int * X_dim = CudaNdarray_HOST_DIMS(%(X)s);
    //we can't use the modulo operator easily as it should not be replaced
    const int dims_Z[] = {X_dim[0], X_dim[1] / 4};
    const int dims_H[] = {X_dim[0], X_dim[1]};
    const int dims_d[] = {X_dim[0], X_dim[1] / 4};
    int size_d = X_dim[0] * X_dim[1] / 4;

    %(Z)s = (CudaNdarray*) CudaNdarray_NewDims(2,dims_Z);
    %(d)s = (CudaNdarray*) CudaNdarray_NewDims(2, dims_d);
    %(H)s = (CudaNdarray*) CudaNdarray_NewDims(2,dims_H); //CudaNdarray_uninitialized_like(%(X)s);
    cudaMemcpy(CudaNdarray_DEV_DATA(%(H)s), CudaNdarray_DEV_DATA(%(X)s),
      dims_H[0]*dims_H[1]*sizeof(float), cudaMemcpyDeviceToDevice);

    int y = 0;
    int x = 0;
    if(x > 0)
    {
      //H += Z[x-1]*V_h
      affine_y_x(y, x-1, %(Z)s, y, x, %(V_h)s, y, x, %(H)s);
    }
    float * d_ptr =  CudaNdarray_DEV_DATA(%(d)s);
    do_lstm(%(H)s, %(Z)s, %(c)s, d_ptr, y, x, %(i)s);
    """ % locals()

  def grad(self, inputs, output_grads):
    Z, V_h, c, i = inputs
    DY, DH, Dd = output_grads

    Z_raw = Z.owner.inputs[0].owner.inputs[0]
    #TODO!!!
    V_h_raw = V_h.owner.inputs[0]
    c_raw = c.owner.inputs[0].owner.inputs[0]
    i_raw = i.owner.inputs[0].owner.inputs[0]
    #we have to make sure that this in only computed once!
    #for this we have to extract the raw variables before conversion to continuous gpu array
    #so that theano can merge the nodes
    Z, H, d = LSTMOpCellInstance(Z_raw, V_h_raw, c_raw, i_raw)
    if isinstance(DY.type, theano.gradient.DisconnectedType):
      DY = T.zeros_like(Z)
    if isinstance(Dd.type, theano.gradient.DisconnectedType):
      Dd = T.zeros_like(c)
    DZ, DV_h, Dc = LSTMOpCellGradNoInplaceInstance(V_h, c, i, Dd, DY, Z, H)
    Di = theano.gradient.grad_undefined(self, 3, inputs[3], 'cannot diff w.r.t. index')

    return [DZ, DV_h, Dc, Di]

  def infer_shape(self, node, input_shapes):
    Xs, V_hs, cs, idxs = input_shapes
    Z_shape = (Xs[0], Xs[1] / 4)
    H_shape = (Xs[0], Xs[1])
    D_shape = (Xs[0], Xs[1] / 4)
    return [Z_shape, H_shape, D_shape]

  #!!! change this when changing the code!
  #def c_code_cache_version(self):
  #  return 1, 2

LSTMOpCellInstance = LSTMOpCell(inplace=False)
LSTMOpCellInplaceInstance = LSTMOpCell(inplace=True)

LSTMOpCellInplaceOpt = OpSub(LSTMOpCellInstance, LSTMOpCellInplaceInstance)

#hack to avoid begin aclled twice
if not hasattr(optdb, 'LSTMOpCellInplaceOpt_registered'):
  optdb.register('LSTMOpCellInplaceOpt', theano.gof.TopoOptimizer(LSTMOpCellInplaceOpt),
                 50.0, 'fast_run', 'inplace', 'gpuarray')
  optdb.LSTMOpCellInplaceOpt_registered = True
