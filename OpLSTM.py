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

class LSTMOpGrad(theano.sandbox.cuda.GpuOp):
  def __init__(self, inplace):
    self.inplace = inplace
    if inplace:
      #all outputs operate inplace on inputs 4 and 6 (which are DZ and H)
      #but when the input is marked multiple times, we get an error
      #so we only mark that output 0 destroys inputs 4 and 6
      #anyway theano knows that inputs 4 and 6 will be destroyed, so it should be OK
      #TODO
      self.destroy_map = {0: [4], 1: [6]}

  def __eq__(self, other):
    return type(self) == type(other) and self.inplace == other.inplace

  def __str__(self):
    if self.inplace:
      return '%s{inplace}' % self.__class__.__name__
    else:
      return '%s{no_inplace}' % self.__class__.__name__

  def __hash__(self):
    return hash(type(self)) ^ hash(self.inplace)

  def make_node(self, X, V_h, c, idx, DZ, Z, H):
    X = gpu_contiguous(as_cuda_ndarray_variable(X))
    V_h = gpu_contiguous(as_cuda_ndarray_variable(V_h))
    c = gpu_contiguous(as_cuda_ndarray_variable(c))
    DZ = gpu_contiguous(as_cuda_ndarray_variable(DZ))
    idx = gpu_contiguous(as_cuda_ndarray_variable(idx))
    assert X.dtype == "float32"
    assert V_h.dtype == "float32"
    assert DZ.dtype == 'float32'
    assert Z.dtype == 'float32'
    assert H.dtype == 'float32'
    assert c.dtype == 'float32'
    assert X.ndim == 3
    assert V_h.ndim == 2
    assert DZ.ndim == 3
    assert Z.ndim == 3
    assert H.ndim == 3
    assert c.ndim == 2
    assert idx.ndim == 2

    return theano.Apply(self, [X, V_h, c, idx, DZ, Z, H], [X.type(), V_h.type(), c.type()])

  def infer_shape(self, node, input_shapes):
    Xs, V_hs, cs, idxs, DZs, Zs, Hs = input_shapes
    return [Xs, V_hs, cs]

  def c_support_code(self):
    crnn_path = os.path.dirname(__file__)
    with open(crnn_path + "/c_support_code_mdlstm.cpp") as f:
      return f.read()

  def c_code(self, node, name, input_names, output_names, sub):
    X, V_h, c, i, DZ, Z, H = input_names
    DX, DV_h, Dc = output_names
    fail = sub['fail']
    inplace = "true" if self.inplace else "false"
    return """

    // std::cout << "LSTMOpGrad called" << std::endl;
    if(!%(inplace)s)
    {
      std::cout << "warning, inplace optimization failed, not working inplace" << std::endl;
    }

    if(%(DX)s || %(DV_h)s || %(Dc)s)
    {
      printf("output storage already exists\\n");
      //TODO check if we can reuse it
      Py_XDECREF(%(DX)s);
      Py_XDECREF(%(DV_h)s);
      Py_XDECREF(%(Dc)s);
    }

    CudaNdarray * epsilon = 0;
    CudaNdarray * delta = 0;
    if(%(inplace)s)
    {
      epsilon = %(DZ)s;
      delta = %(H)s;
    }
    else
    {
      epsilon = (CudaNdarray *) CudaNdarray_Copy(%(DZ)s);
      delta = (CudaNdarray *) CudaNdarray_Copy(%(H)s);
    }

    const int * X_dim = CudaNdarray_HOST_DIMS(%(X)s);
    //float *index = new float[X_dim[0] * X_dim[1]];
    //cudaMemcpy(index, CudaNdarray_DEV_DATA(%(i)s),
    //  X_dim[0]*X_dim[1]*sizeof(float), cudaMemcpyDeviceToHost);

    int y = 0;
    for(int x = X_dim[0]-1; x >= 0; --x)
    {
      //add recurrent
      bool rightBorder = (x == X_dim[0]-1);
      if(!rightBorder)
      {
        affine_y_x(y, x+1, delta, y, x, %(V_h)s, y, x, epsilon, false, true);
      }

      do_lstm_bwd(delta, epsilon, %(Z)s, y, x, rightBorder);
    }

    %(DX)s = CudaNdarray_uninitialized_like(%(X)s);
    %(DV_h)s = CudaNdarray_uninitialized_like(%(V_h)s);
    //DV_h = Z[0..end-1]^T * delta[1..end]
    affine_global(%(Z)s, delta, %(DV_h)s, true, false, 1, 0.0f);
    //DX = delta * W^T
    //%(DX)s = (CudaNdarray *) CudaNdarray_Copy(delta);

    //const int * X_dim = CudaNdarray_HOST_DIMS(%(X)s);
    cudaMemcpy(CudaNdarray_DEV_DATA(%(DX)s), CudaNdarray_DEV_DATA(delta),
      X_dim[0]*X_dim[1]*X_dim[2]*sizeof(float), cudaMemcpyDeviceToDevice);

    %(Dc)s = CudaNdarray_uninitialized_like(%(c)s);
    const int * Z_dim = CudaNdarray_HOST_DIMS(%(Z)s);
    cudaMemcpy(CudaNdarray_DEV_DATA(%(Dc)s), CudaNdarray_DEV_DATA(epsilon),
      Z_dim[1]*Z_dim[2]*sizeof(float), cudaMemcpyDeviceToDevice);

    if(!%(inplace)s)
    {
      Py_XDECREF(epsilon);
      Py_XDECREF(delta);
    }

    """ % locals()

  #!!! change this when changing the code!
  #def c_code_cache_version(self):
  #  return 1, 1

LSTMOpGradNoInplaceInstance = LSTMOpGrad(inplace=False)
LSTMOpGradInplaceInstance = LSTMOpGrad(inplace=True)

LSTMOpInlaceOpt = OpSub(LSTMOpGradNoInplaceInstance, LSTMOpGradInplaceInstance)

#TODO: why is this called twice??
#hack to avoid this
if not hasattr(optdb, 'LSTMOpInlaceOpt_registered'):
  optdb.register('LSTMOpInlaceOpt', theano.gof.TopoOptimizer(LSTMOpInlaceOpt),
                 50.0, 'fast_run', 'inplace', 'gpuarray')
  optdb.LSTMOpInlaceOpt_registered = True


#------------------------

class LSTMOp(theano.sandbox.cuda.GpuOp):
  __props__ = ()

  def make_node(self, Z, V_h, c, i):
    Z = gpu_contiguous(as_cuda_ndarray_variable(Z))
    V_h = gpu_contiguous(as_cuda_ndarray_variable(V_h))
    c = gpu_contiguous(as_cuda_ndarray_variable(c))
    i = gpu_contiguous(as_cuda_ndarray_variable(i))
    assert Z.dtype == "float32"
    assert V_h.dtype == "float32"
    assert c.dtype == 'float32'
    assert c.ndim == 2
    assert Z.ndim == 3
    assert i.ndim == 2
    assert V_h.ndim == 2

    #results: output Y, (gates and cell state) H
    return theano.Apply(self, [Z, V_h, c, i], [Z.type(), Z.type()])

  def c_support_code(self):
    crnn_path = os.path.dirname(__file__)
    with open(crnn_path + "/c_support_code_mdlstm.cpp") as f:
      return f.read()

  def c_code(self, node, name, input_names, output_names, sub):
    X, V_h, c, i = input_names
    Z, H = output_names
    fail = sub['fail']
    return """
    if(%(Z)s || %(H)s)
    {
      printf("Z or H already exist\\n");
      //TODO check if we can reuse it
      Py_XDECREF(%(Z)s);
      Py_XDECREF(%(H)s);
    }

    const int * X_dim = CudaNdarray_HOST_DIMS(%(X)s);
    //we can't use the modulo operator easily as it should not be replaced
    const int dims_Z[] = {X_dim[0], X_dim[1], X_dim[2] / 4};
    const int dims_H[] = {X_dim[0], X_dim[1], X_dim[2]};

    %(Z)s = (CudaNdarray*) CudaNdarray_NewDims(3,dims_Z);
    %(H)s = (CudaNdarray*) CudaNdarray_NewDims(3,dims_H); //CudaNdarray_uninitialized_like(%(X)s);
    cudaMemcpy(CudaNdarray_DEV_DATA(%(H)s), CudaNdarray_DEV_DATA(%(X)s),
      dims_H[0]*dims_H[1]*dims_H[2]*sizeof(float), cudaMemcpyDeviceToDevice);
    //%(H)s = (CudaNdarray *) CudaNdarray_Copy(%(X)s); //(CudaNdarray*) CudaNdarray_NewDims(3,dims_H);
    //%(H)s = %(X)s;

    //float *index = new float[X_dim[0] * X_dim[1]];
    //cudaMemcpy(index, CudaNdarray_DEV_DATA(%(i)s),
    //  X_dim[0]*X_dim[1]*sizeof(float), cudaMemcpyDeviceToHost);

    int y = 0;
    for(int x = 0; x < X_dim[0]; ++x)
    {
      if(x > 0)
      {
        //H += Z[x-1]*V_h
        affine_y_x(y, x-1, %(Z)s, y, x, %(V_h)s, y, x, %(H)s);
      }
      do_lstm(%(H)s, %(Z)s, %(c)s, y, x);
    }
    """ % locals()

  def grad(self, inputs, output_grads):
    X, V_h, c, i = inputs
    DZ, DH = output_grads

    X_raw = X.owner.inputs[0].owner.inputs[0]
    #TODO!!!
    V_h_raw = V_h.owner.inputs[0]
    c_raw = c.owner.inputs[0].owner.inputs[0]
    i_raw = i.owner.inputs[0].owner.inputs[0]
    #we have to make sure that this in only computed once!
    #for this we have to extract the raw variables before conversion to continuous gpu array
    #so that theano can merge the nodes
    Z, H = LSTMOpInstance(X_raw, V_h_raw, c_raw, i_raw)

    DX, DV_h, Dc = LSTMOpGradNoInplaceInstance(X, V_h, c, i, DZ, Z, H)
    Di = theano.gradient.grad_undefined(self, 3, inputs[3], 'cannot diff w.r.t. index')

    return [DX, DV_h, Dc, Di]

  def infer_shape(self, node, input_shapes):
    Xs, V_hs, cs, idxs = input_shapes
    Z_shape = (Xs[0], Xs[1], Xs[2] / 4)
    H_shape = (Xs[0], Xs[1], Xs[2])

    return [Z_shape, H_shape]

  #!!! change this when changing the code!
  #def c_code_cache_version(self):
  #  return 1, 1

LSTMOpInstance = LSTMOp()