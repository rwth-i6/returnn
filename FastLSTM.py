import numpy
import theano
import theano.gradient
import theano.tensor as T
import theano.printing
import theano.gof
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           gpu_contiguous)
from theano.sandbox.cuda import CudaNdarrayType
from theano.gof.opt import OpSub
from theano.compile import optdb
import os

#renamed this to LSTMOp2 because there is almost the same in OpLSTM.py (without the 2)

class LSTMOp2Grad(theano.sandbox.cuda.GpuOp):
  def __init__(self, inplace):
    self.inplace = inplace
    if inplace:
      #all outputs operate inplace on inputs 2 and 4 (which are DZ and H)
      #but when the input is marked multiple times, we get an error
      #so we only mark that output 0 destroys inputs 2 and 4
      #anyway theano knows that inputs 2 and 4 will be destroyed, so it should be OK
      #TODO
      self.destroy_map = {0: [2], 1: [4]}

  def __eq__(self, other):
    return type(self) == type(other) and self.inplace == other.inplace

  def __str__(self):
    if self.inplace:
      return '%s{inplace}' % self.__class__.__name__
    else:
      return '%s{no_inplace}' % self.__class__.__name__

  def __hash__(self):
    return hash(type(self)) ^ hash(self.inplace)

  def make_node(self, V_h, b, DZ, Z, H, c, i, Dd, *args):
    num_inputs = len(args) / 2
    XS = args[:num_inputs]
    WS = args[num_inputs:]
    XS = [gpu_contiguous(as_cuda_ndarray_variable(X)) for X in XS]
    WS = [gpu_contiguous(as_cuda_ndarray_variable(W)) for W in WS]
    V_h = gpu_contiguous(as_cuda_ndarray_variable(V_h))
    b = gpu_contiguous(as_cuda_ndarray_variable(b))
    DZ = gpu_contiguous(as_cuda_ndarray_variable(DZ))
    Dd = gpu_contiguous(as_cuda_ndarray_variable(Dd))
    for X in XS:
      assert X.dtype == "float32"
      assert X.ndim == 3
    for W in WS:
      assert W.dtype == "float32"
      assert W.ndim == 2
    assert V_h.dtype == "float32"
    assert b.dtype == 'float32'
    assert DZ.dtype == 'float32'
    assert Z.dtype == 'float32'
    assert H.dtype == 'float32'
    assert c.dtype == 'float32'
    assert Dd.dtype == 'float32'
    assert V_h.ndim == 2
    assert b.ndim == 1
    assert DZ.ndim == 3
    assert Z.ndim == 3
    assert H.ndim == 3
    assert c.ndim == 2
    assert i.ndim == 2
    assert Dd.ndim == 2

    return theano.Apply(self, [V_h, b, DZ, Z, H, c, i, Dd] + XS + WS,
                        [V_h.type(), c.type(), b.type()] + [X.type() for X in XS] + [W.type() for W in WS])

  def infer_shape(self, node, input_shapes):
    n_inputs = (len(input_shapes) - 7) / 2
    V_hs, bs, DZs, Zs, Hs, cs, i_s, Dd_s = input_shapes[:8]
    XSs = input_shapes[8:8+n_inputs]
    WSs = input_shapes[8+n_inputs:]
    return [V_hs, cs, bs] + XSs + WSs

  def c_support_code(self):
    crnn_path = os.path.dirname(__file__)
    with open(crnn_path + "/c_support_code_mdlstm.cpp") as f:
      return f.read()

  def c_code(self, node, name, input_names, output_names, sub):
    n_inputs = (len(input_names) - 8) / 2
    V_h, b, DZ, Z, H, c, i, Dd = input_names[:8]
    XS = input_names[8:8+n_inputs]
    WS = input_names[8+n_inputs:]

    DV_h, Dc, Db = output_names[:3]
    DXS = output_names[3:n_inputs+3]
    DWS = output_names[n_inputs+3:2*n_inputs+3]

    if n_inputs == 0:
      XS_str = "CudaNdarray ** XS = 0;"
      WS_str = "CudaNdarray ** WS = 0;"
      DXS_str = "CudaNdarray *** DXS = 0;"
      DWS_str = "CudaNdarray *** DWS = 0;"
    else:
      XS_str = "CudaNdarray * XS[] = {" + ",".join(XS) + "}"
      WS_str = "CudaNdarray * WS[] = {" + ",".join(WS) + "}"
      DXS_str = "CudaNdarray ** DXS[] = {" + ",".join(["&" + DX for DX in DXS]) + "}"
      DWS_str = "CudaNdarray ** DWS[] = {" + ",".join(["&" + DW for DW in DWS]) + "}"

    fail = sub['fail']
    inplace = "true" if self.inplace else "false"
    return """

    // std::cout << "LSTMOpGrad called" << std::endl;
    if(!%(inplace)s)
    {
      std::cout << "warning, inplace optimization failed, not working inplace" << std::endl;
    }

    int n_inputs = %(n_inputs)s;
    //these declare CudaNdarray **(*) s (see above)
    %(XS_str)s;
    %(WS_str)s;
    %(DXS_str)s;
    %(DWS_str)s;

    //TODO: DX and DW are not checked here anymore, as they are harder to handle as arrays
    if(%(DV_h)s || %(Dc)s || %(Db)s)
    {
      printf("output storage already exists\\n");
      //TODO check if we can reuse it
      Py_XDECREF(%(DV_h)s);
      Py_XDECREF(%(Db)s);
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

    const int * Z_dim = CudaNdarray_HOST_DIMS(%(Z)s);
    int y = 0;
    for(int x = Z_dim[0]-1; x >= 0; --x)
    {
      //add recurrent
      bool rightBorder = (x == Z_dim[0]-1);
      if(!rightBorder)
      {
        affine_y_x(y, x+1, delta, y, x, %(V_h)s, y, x, epsilon, false, true);
      }

      do_lstm_bwd(delta, epsilon, %(Z)s, %(Dd)s, %(c)s, y, x, rightBorder, %(i)s);
    }

    %(DV_h)s = CudaNdarray_uninitialized_like(%(V_h)s);
    //DV_h = Z[0..end-1]^T * delta[1..end]
    affine_global(%(Z)s, delta, %(DV_h)s, true, false, 1, 0.0f);

    for(int j = 0; j < n_inputs; ++j)
    {
      *DXS[j] = CudaNdarray_uninitialized_like(XS[j]);
      //DX = delta * W^T
      affine_global(delta, WS[j], *DXS[j], false, true, 0, 0.0f);
    }

    //for testing!!!
    /* *DXS[0] = delta;
    Py_XINCREF(delta);*/

    for(int j = 0; j < n_inputs; ++j)
    {
      *DWS[j] = CudaNdarray_uninitialized_like(WS[j]);
      //DW = X^T * delta
      affine_global(XS[j], delta, *DWS[j], true, false, 0, 0.0f);
    }

    //Db = (1 ... 1) * delta
    %(Db)s = sumOverAllButLastDimensions(delta);

    %(Dc)s = CudaNdarray_uninitialized_like(%(c)s);
    cudaMemcpy(CudaNdarray_DEV_DATA(%(Dc)s), CudaNdarray_DEV_DATA(epsilon),
      Z_dim[1]*Z_dim[2]*sizeof(float), cudaMemcpyDeviceToDevice);

    if(!%(inplace)s)
    {
      Py_XDECREF(epsilon);
      Py_XDECREF(delta);
    }

    """ % locals()

  #!!! change this when changing the code!
  def c_code_cache_version(self):
    return 1, 2

LSTMOpGradNoInplaceInstance = LSTMOp2Grad(inplace=False)
LSTMOpGradInplaceInstance = LSTMOp2Grad(inplace=True)

LSTMOp2InplaceOpt = OpSub(LSTMOpGradNoInplaceInstance, LSTMOpGradInplaceInstance)

#TODO: why is this called twice??
#hack to avoid this
if not hasattr(optdb, 'LSTMOp2InplaceOpt_registered'):
  optdb.register('LSTMOp2InplaceOpt', theano.gof.TopoOptimizer(LSTMOp2InplaceOpt),
                 50.0, 'fast_run', 'inplace', 'gpuarray')
  optdb.LSTMOp2InplaceOpt_registered = True


#------------------------

class LSTMOp2(theano.sandbox.cuda.GpuOp):
  __props__ = ()

  #V_h: recurrent (horizontal) weight matrix
  #c: initial state
  #b: bias
  #i index
  #args: XS and WS (inputs and weight matrices)
  def make_node(self, V_h, c, b, i, *args):
    num_inputs = len(args) / 2
    XS = args[:num_inputs]
    WS = args[num_inputs:2*num_inputs]
    XS = [gpu_contiguous(as_cuda_ndarray_variable(X)) for X in XS]
    WS = [gpu_contiguous(as_cuda_ndarray_variable(W)) for W in WS]
    V_h = gpu_contiguous(as_cuda_ndarray_variable(V_h))
    b = gpu_contiguous(as_cuda_ndarray_variable(b))
    c = gpu_contiguous(as_cuda_ndarray_variable(c))
    i = gpu_contiguous(as_cuda_ndarray_variable(T.cast(i,'float32')))
    for X in XS:
      assert X.dtype == "float32"
      assert X.ndim == 3
    for W in WS:
      assert W.dtype == "float32"
      assert W.ndim == 2
    assert V_h.dtype == "float32"
    assert b.dtype == 'float32'
    assert c.dtype == 'float32'
    assert c.ndim == 2
    assert V_h.ndim == 2
    assert b.ndim == 1
    assert i.ndim == 2

    #results: output Y, (gates and cell state) H
    if len(XS) > 0:
      X_type = XS[0].type
    else:
      broad = (False, False, False)
      X_type = CudaNdarrayType(dtype=V_h.dtype, broadcastable=broad)
    return theano.Apply(self, [V_h, c, b, i] + XS + WS, [X_type(), X_type(), c.type()])

  def c_support_code(self):
    crnn_path = os.path.dirname(__file__)
    with open(crnn_path + "/c_support_code_mdlstm.cpp") as f:
      return f.read()

  def c_code(self, node, name, input_names, output_names, sub):
    n_inputs = (len(input_names) - 4) / 2
    V_h, c, b, i = input_names[:4]
    XS = input_names[4:n_inputs+4]
    WS = input_names[n_inputs+4:2*n_inputs+4]

    if len(XS) == 0:
      XS_str = "CudaNdarray ** XS = 0;"
      WS_str = "CudaNdarray ** WS = 0;"
    else:
      XS_str = "CudaNdarray * XS[] = {" + ",".join(XS) + "}"
      WS_str = "CudaNdarray * WS[] = {" + ",".join(WS) + "}"

    Z, H, d = output_names #d: endstate
    fail = sub['fail']
    return """
    if(%(Z)s || %(H)s || %(d)s)
    {
      printf("Z or H or d already exist\\n");
      //TODO check if we can reuse it
      Py_XDECREF(%(Z)s);
      Py_XDECREF(%(H)s);
      Py_XDECREF(%(d)s);
    }

    //std::cout << "LSTMOp called" << std::endl;

    int n_inputs = %(n_inputs)s;
    //these declare
    //CudaNdarray * XS[] and CudaNdarray * WS[]
    %(XS_str)s;
    %(WS_str)s;

    const int * c_dim = CudaNdarray_HOST_DIMS(%(c)s);
    const int * i_dim = CudaNdarray_HOST_DIMS(%(i)s);
    int T = i_dim[0];
    int batch = c_dim[0];
    int n_cells = c_dim[1];
    const int dims_Z[] = {T, batch, n_cells};
    const int dims_H[] = {T, batch, n_cells * 4};
    const int dims_d[] = {batch, n_cells};
    int size_d = batch * n_cells;

    %(Z)s = (CudaNdarray*) CudaNdarray_NewDims(3,dims_Z);
    %(d)s = (CudaNdarray*) CudaNdarray_NewDims(2,dims_d);
    %(H)s = (CudaNdarray*) CudaNdarray_NewDims(3,dims_H);
    //init H with b
    fillmat(%(b)s, %(H)s);
    //H+=XW (for all Xs and Ws)
    for(int j = 0; j < n_inputs; ++j)
    {
      affine_global(XS[j], WS[j], %(H)s);
    }

    int y = 0;
    for(int x = 0; x < T; ++x)
    {
      if(x > 0)
      {
        //H += Z[x-1]*V_h
        affine_y_x(y, x-1, %(Z)s, y, x, %(V_h)s, y, x, %(H)s);
      }
      float * d_ptr = (x == T - 1) ? CudaNdarray_DEV_DATA(%(d)s) : 0;
      do_lstm(%(H)s, %(Z)s, %(c)s, d_ptr, y, x, %(i)s);
    }
    """ % locals()

  def grad(self, inputs, output_grads):
    n_inputs = (len(inputs) - 4) / 2
    V_h, c, b, i = inputs[:4]
    XS = inputs[4:n_inputs+4]
    WS = inputs[n_inputs+4:2*n_inputs+4]

    DZ, DH, Dd = output_grads

    XS_raw = [X.owner.inputs[0].owner.inputs[0] for X in XS]
    #TODO
    WS_raw = [W.owner.inputs[0] for W in WS]
    V_h_raw = V_h.owner.inputs[0]
    #TODO: check
    c_raw = c.owner.inputs[0].owner.inputs[0]
    b_raw = b.owner.inputs[0]
    #we have to make sure that this in only computed once!
    #for this we have to extract the raw variables before conversion to continuous gpu array
    #so that theano can merge the nodes
    Z, H, d = LSTMOp2Instance(*([V_h_raw, c_raw, b_raw, i] + XS_raw + WS_raw))

    DZ_valid = not isinstance(DZ.type, theano.gradient.DisconnectedType)
    Dd_valid = not isinstance(Dd.type, theano.gradient.DisconnectedType)
    assert DZ_valid or Dd_valid, "both outputs disconnected"
    if not DZ_valid:
      DZ = T.zeros_like(Z)
    if not Dd_valid:
      Dd = T.zeros_like(c)

    grads = LSTMOpGradNoInplaceInstance(*([V_h, b, DZ, Z, H, c, i, Dd] + XS + WS))
    DV_h, Dc, Db = grads[:3]
    DXS = grads[3:n_inputs+3]
    DWS = grads[n_inputs+3:2*n_inputs+3]

    Di = theano.gradient.grad_undefined(self, 6, i, 'cannot diff w.r.t. index')
    return [DV_h, Dc, Db, Di] + DXS + DWS

  def infer_shape(self, node, input_shapes):
    V_hs, cs, bs, i_s = input_shapes[:4]
    time = i_s[0]
    batch = cs[0]
    n_cells = cs[1]
    Z_shape = (time, batch, n_cells)
    H_shape = (time, batch, n_cells * 4)
    d_shape = (batch, n_cells)
    return [Z_shape, H_shape, d_shape]

  #!!! change this when changing the code!
  def c_code_cache_version(self):
    return 1, 2

LSTMOp2Instance = LSTMOp2()
