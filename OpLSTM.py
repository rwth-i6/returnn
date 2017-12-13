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
      #all outputs operate inplace on inputs 4 and 6 (which are DY and H)
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

  def make_node(self, V_h, c, idx, Dd, DY, Y, H):
    V_h = gpu_contiguous(as_cuda_ndarray_variable(V_h))
    c = gpu_contiguous(as_cuda_ndarray_variable(c))
    DY = gpu_contiguous(as_cuda_ndarray_variable(DY))
    idx = gpu_contiguous(as_cuda_ndarray_variable(T.cast(idx,'float32')))
    Dd = gpu_contiguous(as_cuda_ndarray_variable(Dd))
    assert V_h.dtype == "float32"
    assert DY.dtype == 'float32'
    assert Y.dtype == 'float32'
    assert H.dtype == 'float32'
    assert c.dtype == 'float32'
    assert V_h.ndim == 2
    assert DY.ndim == 3
    assert Y.ndim == 3
    assert H.ndim == 3
    assert c.ndim == 2
    assert idx.ndim == 2

    return theano.Apply(self, [V_h, c, idx, Dd, DY, Y, H], [H.type(), V_h.type(), c.type()])

  def infer_shape(self, node, input_shapes):
    V_hs, cs, idxs, Dds, DYs, Ys, Hs = input_shapes
    return [Hs, V_hs, cs]

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

    // std::cout << "LSTMOpGrad called" << std::endl;
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
    for(int x = H_dim[0]-1; x >= 0; --x)
    {
      //add recurrent
      bool rightBorder = (x == H_dim[0]-1);
      if(!rightBorder)
      {
        affine_y_x(y, x+1, delta, y, x, %(V_h)s, y, x, epsilon, false, true);
      }

      do_lstm_bwd(delta, epsilon, %(Y)s, %(Dd)s, %(c)s, y, x, rightBorder, %(i)s);

    }

    %(DV_h)s = CudaNdarray_uninitialized_like(%(V_h)s);
    //DV_h = Y[0..end-1]^T * delta[1..end]
    affine_global(%(Y)s, delta, %(DV_h)s, true, false, 1, 0.0f);

    %(DZ)s = delta;

    %(Dc)s = CudaNdarray_uninitialized_like(%(c)s);
    const int * Y_dim = CudaNdarray_HOST_DIMS(%(Y)s);
    cudaMemcpy(CudaNdarray_DEV_DATA(%(Dc)s), CudaNdarray_DEV_DATA(epsilon),
      Y_dim[1]*Y_dim[2]*sizeof(float), cudaMemcpyDeviceToDevice);

    if(!%(inplace)s)
    {
      Py_XDECREF(epsilon);
    }

    """ % locals()

  #!!! change this when changing the code!
  def c_code_cache_version(self):
    return 1, 5

LSTMOpGradNoInplaceInstance = LSTMOpGrad(inplace=False)
LSTMOpGradInplaceInstance = LSTMOpGrad(inplace=True)

LSTMOpGradInplaceOpt = OpSub(LSTMOpGradNoInplaceInstance, LSTMOpGradInplaceInstance)

#hack to avoid being called twice
if not hasattr(optdb, 'LSTMOpGradInplaceOpt_registered'):
  optdb.register('LSTMOpGradInplaceOpt', theano.gof.TopoOptimizer(LSTMOpGradInplaceOpt),
                 50.0, 'fast_run', 'inplace', 'gpuarray')
  optdb.LSTMOpGradInplaceOpt_registered = True


#------------------------

class LSTMOp(theano.sandbox.cuda.GpuOp):
  def __init__(self, inplace):
    self.inplace = inplace
    if inplace:
      #all outputs operate inplace on input 0 (which is Z)
      #but when the input is marked multiple times, we get an error
      #so we only mark that output 0 destroys input 0
      #anyway theano knows that input 0 will be destroyed, so it should be OK
      #TODO
      self.destroy_map = {0: [0]}

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
    """
    :param Z: {input,output,forget} gate + cell state. 3d (time,batch,dim*4)
    :param V_h: recurrent matrix. 2d (dim,dim*4)
    :param c: initial cell state. 2d (batch,dim)
    :param i: index. 2d (time,batch) -> 0 or 1
    """
    Z = gpu_contiguous(as_cuda_ndarray_variable(Z))
    V_h = gpu_contiguous(as_cuda_ndarray_variable(V_h))
    c = gpu_contiguous(as_cuda_ndarray_variable(c))
    i = gpu_contiguous(as_cuda_ndarray_variable(T.cast(i,'float32')))
    assert Z.dtype == "float32"
    assert V_h.dtype == "float32"
    assert c.dtype == 'float32'
    assert c.ndim == 2
    assert Z.ndim == 3
    assert i.ndim == 2
    assert V_h.ndim == 2

    # results: output Y, (gates and cell state) H, (final cell state) d
    return theano.Apply(self, [Z, V_h, c, i], [Z.type(), Z.type(), c.type()])

  def c_support_code(self):
    crnn_path = os.path.dirname(__file__)
    with open(crnn_path + "/c_support_code_mdlstm.cpp") as f:
      return f.read()

  def c_code(self, node, name, input_names, output_names, sub):
    Z, V_h, c, i = input_names
    Y, H, d = output_names
    fail = sub['fail']
    inplace = "true" if self.inplace else "false"
    return """
    if(%(Y)s || %(H)s || %(d)s)
    {
      //printf("Y or H or d already exist\\n");
      //TODO check if we can reuse it
      Py_XDECREF(%(Y)s);
      Py_XDECREF(%(H)s);
      Py_XDECREF(%(d)s);
    }

    const int * Z_dim = CudaNdarray_HOST_DIMS(%(Z)s);
    const int dims_Y[] = {Z_dim[0], Z_dim[1], Z_dim[2] / 4};
    const int dims_H[] = {Z_dim[0], Z_dim[1], Z_dim[2]};
    const int dims_d[] = {Z_dim[1], Z_dim[2] / 4};
    int size_d = Z_dim[1] * Z_dim[2] / 4;

    %(Y)s = (CudaNdarray*) CudaNdarray_NewDims(3,dims_Y);
    %(d)s = (CudaNdarray*) CudaNdarray_NewDims(2, dims_d);
    if(%(inplace)s)
    {
      %(H)s = %(Z)s;
      Py_INCREF(%(Z)s);
    }
    else
    {
      %(H)s = (CudaNdarray*) CudaNdarray_NewDims(3,dims_H);
      cudaMemcpy(CudaNdarray_DEV_DATA(%(H)s), CudaNdarray_DEV_DATA(%(Z)s),
      dims_H[0]*dims_H[1]*dims_H[2]*sizeof(float), cudaMemcpyDeviceToDevice);
    }

    int y = 0;
    for(int x = 0; x < Z_dim[0]; ++x)
    {
      if(x > 0)
      {
        //H += Y[x-1]*V_h
        affine_y_x(y, x-1, %(Y)s, y, x, %(V_h)s, y, x, %(H)s);
      }
      float * d_ptr = (x == Z_dim[0] - 1) ? CudaNdarray_DEV_DATA(%(d)s) : 0;
      do_lstm(%(H)s, %(Y)s, %(c)s, d_ptr, y, x, %(i)s);
    }
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
    Y, H, d = LSTMOpInstance(Z_raw, V_h_raw, c_raw, i_raw)
    if isinstance(DY.type, theano.gradient.DisconnectedType):
      DY = T.zeros_like(Z)
    if isinstance(Dd.type, theano.gradient.DisconnectedType):
      Dd = T.zeros_like(c)
    DZ, DV_h, Dc = LSTMOpGradNoInplaceInstance(V_h, c, i, Dd, DY, Y, H)
    Di = theano.gradient.grad_undefined(self, 3, inputs[3], 'cannot diff w.r.t. index')

    return [DZ, DV_h, Dc, Di]

  def infer_shape(self, node, input_shapes):
    Zs, V_hs, cs, idxs = input_shapes
    Y_shape = (Zs[0], Zs[1], Zs[2] // 4)
    H_shape = (Zs[0], Zs[1], Zs[2])
    d_shape = (Zs[1], Zs[2] // 4)
    return [Y_shape, H_shape, d_shape]

  #!!! change this when changing the code!
  def c_code_cache_version(self):
    return 1, 6

LSTMOpInstance = LSTMOp(inplace=False)
LSTMOpInplaceInstance = LSTMOp(inplace=True)

LSTMOpInplaceOpt = OpSub(LSTMOpInstance, LSTMOpInplaceInstance)

#hack to avoid begin called twice
if not hasattr(optdb, 'LSTMOpInplaceOpt_registered'):
  optdb.register('LSTMOpInplaceOpt', theano.gof.TopoOptimizer(LSTMOpInplaceOpt),
                 50.0, 'fast_run', 'inplace', 'gpuarray')
  optdb.LSTMOpInplaceOpt_registered = True

class LSTMSOp(theano.sandbox.cuda.GpuOp):
  def __init__(self, inplace):
    self.inplace = inplace
    if inplace:
      #all outputs operate inplace on input 0 (which is Z)
      #but when the input is marked multiple times, we get an error
      #so we only mark that output 0 destroys input 0
      #anyway theano knows that input 0 will be destroyed, so it should be OK
      #TODO
      self.destroy_map = {0: [0]}

  def __eq__(self, other):
    return type(self) == type(other) and self.inplace == other.inplace

  def __str__(self):
    if self.inplace:
      return '%s{inplace}' % self.__class__.__name__
    else:
      return '%s{no_inplace}' % self.__class__.__name__

  def __hash__(self):
    return hash(type(self)) ^ hash(self.inplace)

  def make_node(self, Z, V_h, c, i, att):
    """
    :param Z: {input,output,forget} gate + cell state. 3d (time,batch,dim*4)
    :param V_h: recurrent matrix. 2d (dim,dim*4)
    :param c: initial cell state. 2d (batch,dim)
    :param i: index. 2d (time,batch) -> 0 or 1
    :param att: attention from inverted alignment layer
    """
    Z = gpu_contiguous(as_cuda_ndarray_variable(Z))
    V_h = gpu_contiguous(as_cuda_ndarray_variable(V_h))
    c = gpu_contiguous(as_cuda_ndarray_variable(c))
    i = gpu_contiguous(as_cuda_ndarray_variable(T.cast(i,'float32')))
    att = gpu_contiguous(as_cuda_ndarray_variable(T.cast(att,'float32')))
    assert Z.dtype == "float32"
    assert V_h.dtype == "float32"
    assert c.dtype == 'float32'
    assert c.ndim == 2
    assert Z.ndim == 3
    assert i.ndim == 2
    assert V_h.ndim == 2
    assert att.ndim == 2
    # results: output Y, (gates and cell state) H, (final cell state) d
    return theano.Apply(self, [Z, V_h, c, i, att], [Z.type(), Z.type(), c.type()])

  def c_support_code(self):
    crnn_path = os.path.dirname(__file__)
    with open(crnn_path + "/c_support_code_mdlstm.cpp") as f:
      return f.read()

  def c_code(self, node, name, input_names, output_names, sub):
    Z, V_h, c, i, att = input_names
    Y, H, d = output_names
    fail = sub['fail']
    inplace = "true" if self.inplace else "false"
    return """
    if(%(Y)s || %(H)s || %(d)s)
    {
      //printf("Y or H or d already exist\\n");
      //TODO check if we can reuse it
      Py_XDECREF(%(Y)s);
      Py_XDECREF(%(H)s);
      Py_XDECREF(%(d)s);
    }

    const int * Z_dim = CudaNdarray_HOST_DIMS(%(Z)s);
    const int dims_Y[] = {Z_dim[0], Z_dim[1], Z_dim[2] / 4};
    const int dims_H[] = {Z_dim[0], Z_dim[1], Z_dim[2]};
    const int dims_d[] = {Z_dim[1], Z_dim[2] / 4};
    int size_d = Z_dim[1] * Z_dim[2] / 4;

    %(Y)s = (CudaNdarray*) CudaNdarray_NewDims(3,dims_Y);
    %(d)s = (CudaNdarray*) CudaNdarray_NewDims(2, dims_d);
    if(%(inplace)s)
    {
      %(H)s = %(Z)s;
      Py_INCREF(%(Z)s);
    }
    else
    {
      %(H)s = (CudaNdarray*) CudaNdarray_NewDims(3,dims_H);
      cudaMemcpy(CudaNdarray_DEV_DATA(%(H)s), CudaNdarray_DEV_DATA(%(Z)s),
      dims_H[0]*dims_H[1]*dims_H[2]*sizeof(float), cudaMemcpyDeviceToDevice);
    }

    int y = 0;
    for(int x = 0; x < Z_dim[0]; ++x)
    {
      if(x > 0)
      {
        //H += Y[x-1]*V_h
        affine_y_x(y, x-1, %(Y)s, y, x, %(V_h)s, y, x, %(H)s);
      }
      float * d_ptr = (x == Z_dim[0] - 1) ? CudaNdarray_DEV_DATA(%(d)s) : 0;
      do_lstms(%(H)s, %(Y)s, %(c)s, d_ptr, y, x, %(i)s, %(att)s);
    }
    """ % locals()

  def grad(self, inputs, output_grads):
    Z, V_h, c, i, att = inputs
    DY, DH, Dd = output_grads

    Z_raw = Z.owner.inputs[0].owner.inputs[0]
    #TODO!!!
    V_h_raw = V_h.owner.inputs[0]
    c_raw = c.owner.inputs[0].owner.inputs[0]
    i_raw = i.owner.inputs[0].owner.inputs[0]
    att_raw = att.owner.inputs[0].owner.inputs[0]
    #we have to make sure that this in only computed once!
    #for this we have to extract the raw variables before conversion to continuous gpu array
    #so that theano can merge the nodes
    Y, H, d = LSTMSOpInstance(Z_raw, V_h_raw, c_raw, i_raw, att_raw)
    if isinstance(DY.type, theano.gradient.DisconnectedType):
      DY = T.zeros_like(Z)
    if isinstance(Dd.type, theano.gradient.DisconnectedType):
      Dd = T.zeros_like(c)
    DZ, DV_h, Dc = LSTMOpGradNoInplaceInstance(V_h, c, i, Dd, DY, Y, H)
    Di = theano.gradient.grad_undefined(self, 3, inputs[3], 'cannot diff w.r.t. index')
    Datt = theano.gradient.grad_undefined(self, 4, inputs[4], 'cannot diff w.r.t. index')
    return [DZ, DV_h, Dc, Di, Datt]

  def infer_shape(self, node, input_shapes):
    Zs, V_hs, cs, idxs, atts = input_shapes
    Y_shape = (Zs[0], Zs[1], Zs[2] / 4)
    H_shape = (Zs[0], Zs[1], Zs[2])
    d_shape = (Zs[1], Zs[2] / 4)
    return [Y_shape, H_shape, d_shape]

  #!!! change this when changing the code!
  def c_code_cache_version(self):
    return 1,7.1

LSTMSOpInstance = LSTMSOp(inplace=False)
LSTMSOpInplaceInstance = LSTMSOp(inplace=True)

LSTMSOpInplaceOpt = OpSub(LSTMSOpInstance, LSTMSOpInplaceInstance)

#hack to avoid begin called twice
if not hasattr(optdb, 'LSTMSOpInplaceOpt_registered'):
  optdb.register('LSTMSOpInplaceOpt', theano.gof.TopoOptimizer(LSTMSOpInplaceOpt),
                 50.0, 'fast_run', 'inplace', 'gpuarray')
  optdb.LSTMSOpInplaceOpt_registered = True
