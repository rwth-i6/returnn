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

class BLSTMOpGrad(theano.sandbox.cuda.GpuOp):
  def __init__(self, inplace):
    self.inplace = inplace
    if inplace:
      #all outputs operate inplace on inputs 4 and 6 (which are DY and H)
      #but when the input is marked multiple times, we get an error
      #so we only mark that output 0 destroys inputs 4 and 6
      #anyway theano knows that inputs 4 and 6 will be destroyed, so it should be OK
      #TODO
      self.destroy_map = {0: [7], 1: [8], 2: [11], 3: [12]}

  def __eq__(self, other):
    return type(self) == type(other) and self.inplace == other.inplace

  def __str__(self):
    if self.inplace:
      return '%s{inplace}' % self.__class__.__name__
    else:
      return '%s{no_inplace}' % self.__class__.__name__

  def __hash__(self):
    return hash(type(self)) ^ hash(self.inplace)

  def make_node(self, V_f, V_b, c_f, c_b, idx_f, idx_b, Dd_f, Dd_b, DY_f, DY_b, Y_f, Y_b, H_f, H_b):
    V_f = gpu_contiguous(as_cuda_ndarray_variable(V_f))
    V_b = gpu_contiguous(as_cuda_ndarray_variable(V_b))
    c_f = gpu_contiguous(as_cuda_ndarray_variable(c_f))
    c_b = gpu_contiguous(as_cuda_ndarray_variable(c_b))
    DY_f = gpu_contiguous(as_cuda_ndarray_variable(DY_f))
    DY_b = gpu_contiguous(as_cuda_ndarray_variable(DY_b))
    idx_f = gpu_contiguous(as_cuda_ndarray_variable(T.cast(idx_f,'float32')))
    idx_b = gpu_contiguous(as_cuda_ndarray_variable(T.cast(idx_b, 'float32')))
    Dd_f = gpu_contiguous(as_cuda_ndarray_variable(Dd_f))
    Dd_b = gpu_contiguous(as_cuda_ndarray_variable(Dd_b))
    assert V_f.dtype == "float32"
    assert V_b.dtype == "float32"
    assert DY_f.dtype == 'float32'
    assert DY_b.dtype == 'float32'
    assert Y_f.dtype == 'float32'
    assert Y_b.dtype == 'float32'
    assert H_f.dtype == 'float32'
    assert H_b.dtype == 'float32'
    assert c_f.dtype == 'float32'
    assert c_b.dtype == 'float32'
    assert V_f.ndim == 2
    assert V_b.ndim == 2
    assert DY_f.ndim == 3
    assert DY_b.ndim == 3
    assert Y_f.ndim == 3
    assert Y_b.ndim == 3
    assert H_f.ndim == 3
    assert H_b.ndim == 3
    assert c_f.ndim == 2
    assert c_b.ndim == 2
    assert idx_f.ndim == 2
    assert idx_b.ndim == 2

    return theano.Apply(self, [V_f, V_b, c_f, c_b, idx_f, idx_b, Dd_f, Dd_b, DY_f, DY_b, Y_f, Y_b, H_f, H_b],
                        [H_f.type(), H_b.type(), V_f.type(), V_b.type(), c_f.type(), c_b.type()])

  def infer_shape(self, node, input_shapes):
    V_fs, V_bs, c_fs, c_bs, idx_fs, idx_bs, Dd_fs, Dd_bs, DYs_fs, DYs_bs, Y_fs, Y_bs, H_fs, H_bs = input_shapes
    return [H_fs, H_bs, V_fs, V_bs, c_fs, c_bs]

  def c_support_code(self):
    crnn_path = os.path.dirname(__file__)
    with open(crnn_path + "/c_support_code_mdlstm.cpp") as f:
      return f.read()

  def c_code(self, node, name, input_names, output_names, sub):
    V_f, V_b, c_f, c_b, i_f, i_b, Dd_f, Dd_b, DY_f, DY_b, Y_f, Y_b, H_f, H_b = input_names
    DZ_f, DZ_b, DV_f, DV_b, Dc_f, Dc_b = output_names
    fail = sub['fail']
    inplace = "true" if self.inplace else "false"
    return """

    if(!%(inplace)s)
    {
      //std::cout << "warning, inplace optimization failed, not working inplace" << std::endl;
    }

    if(%(DZ_f)s || %(DV_f)s || %(Dc_b)s || %(DZ_b)s || %(DV_b)s || %(Dc_b)s)
    {
      //printf("output storage already exists\\n");
      //TODO check if we can reuse it
      Py_XDECREF(%(DZ_f)s);
      Py_XDECREF(%(DV_f)s);
      Py_XDECREF(%(Dc_f)s);
      Py_XDECREF(%(DZ_b)s);
      Py_XDECREF(%(DV_b)s);
      Py_XDECREF(%(Dc_b)s);
    }

    CudaNdarray * epsilon_f = 0;
    CudaNdarray * epsilon_b = 0;
    CudaNdarray * delta_f = 0;
    CudaNdarray * delta_b = 0;
    if(%(inplace)s)
    {
      epsilon_f = %(DY_f)s;
      epsilon_b = %(DY_b)s;
      delta_f = %(H_f)s;
      delta_b = %(H_b)s;
      Py_XINCREF(delta_f);
      Py_XINCREF(delta_b);
    }
    else
    {
      epsilon_f = (CudaNdarray *) CudaNdarray_Copy(%(DY_f)s);
      delta_f = (CudaNdarray *) CudaNdarray_Copy(%(H_f)s);
      epsilon_b = (CudaNdarray *) CudaNdarray_Copy(%(DY_b)s);
      delta_b = (CudaNdarray *) CudaNdarray_Copy(%(H_b)s);
    }

    const int * H_dim = CudaNdarray_HOST_DIMS(%(H_f)s);

    int y = 0;
    for(int x = H_dim[0]-1; x >= 0; --x)
    {
      //add recurrent
      bool rightBorder = (x == H_dim[0]-1);
      if(!rightBorder)
      {
        affine_y_x(y, x+1, delta_f, y, x, %(V_f)s, y, x, epsilon_f, false, true);
        affine_y_x(y, x+1, delta_b, y, x, %(V_b)s, y, x, epsilon_b, false, true);
      }
      /*
      do_lstm_bwd(delta_f, epsilon_f, %(Y_f)s, %(Dd_f)s, %(c_f)s, y, x, rightBorder, %(i_f)s);
      do_lstm_bwd(delta_b, epsilon_b, %(Y_b)s, %(Dd_b)s, %(c_b)s, y, x, rightBorder, %(i_b)s);
      */
      do_blstm_bwd(delta_f, delta_b, epsilon_f, epsilon_b, %(Y_f)s, %(Y_b)s, %(Dd_f)s, %(Dd_b)s, %(c_f)s, %(c_b)s,
                    y, x, rightBorder, %(i_f)s, %(i_b)s);
    }

    %(DV_f)s = CudaNdarray_uninitialized_like(%(V_f)s);
    //DV_h = Y[0..end-1]^T * delta[1..end]
    affine_global(%(Y_f)s, delta_f, %(DV_f)s, true, false, 1, 0.0f);

    %(DZ_f)s = delta_f;

    %(Dc_f)s = CudaNdarray_uninitialized_like(%(c_f)s);
    const int * Y_dim = CudaNdarray_HOST_DIMS(%(Y_f)s);
    cudaMemcpy(CudaNdarray_DEV_DATA(%(Dc_f)s), CudaNdarray_DEV_DATA(epsilon_f),
      Y_dim[1]*Y_dim[2]*sizeof(float), cudaMemcpyDeviceToDevice);

    %(DV_b)s = CudaNdarray_uninitialized_like(%(V_b)s);
    affine_global(%(Y_b)s, delta_b, %(DV_b)s, true, false, 1, 0.0f);

    %(DZ_b)s = delta_b;

    %(Dc_b)s = CudaNdarray_uninitialized_like(%(c_b)s);
    cudaMemcpy(CudaNdarray_DEV_DATA(%(Dc_b)s), CudaNdarray_DEV_DATA(epsilon_b),
      Y_dim[1]*Y_dim[2]*sizeof(float), cudaMemcpyDeviceToDevice);

    if(!%(inplace)s)
    {
      Py_XDECREF(epsilon_f);
      Py_XDECREF(epsilon_b);
    }

    """ % locals()

  #!!! change this when changing the code!
  def c_code_cache_version(self):
    return 1, 7

BLSTMOpGradNoInplaceInstance = BLSTMOpGrad(inplace=False)
BLSTMOpGradInplaceInstance = BLSTMOpGrad(inplace=True)

BLSTMOpGradInplaceOpt = OpSub(BLSTMOpGradNoInplaceInstance, BLSTMOpGradInplaceInstance)

#hack to avoid being called twice
if not hasattr(optdb, 'BLSTMOpGradInplaceOpt_registered'):
  optdb.register('BLSTMOpGradInplaceOpt', theano.gof.TopoOptimizer(BLSTMOpGradInplaceOpt),
                 50.0, 'fast_run', 'inplace', 'gpuarray')
  optdb.BLSTMOpGradInplaceOpt_registered = True


#------------------------

class BLSTMOp(theano.sandbox.cuda.GpuOp):
  def __init__(self, inplace):
    self.inplace = inplace
    if inplace:
      #all outputs operate inplace on input 0 (which is Z)
      #but when the input is marked multiple times, we get an error
      #so we only mark that output 0 destroys input 0
      #anyway theano knows that input 0 will be destroyed, so it should be OK
      #TODO
      #self.destroy_map = {0: [0], 1: [1]} # use this if z_fw and z_bw differ
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

  def make_node(self, Z_f, Z_b, V_f, V_b, c_f, c_b, i_f, i_b):
    """
    :param Z_f: {input,output,forget} gate + cell state forward. 3d (time,batch,dim*4)
    :param Z_b: {input,output,forget} gate + cell state backward. 3d (time,batch,dim*4)
    :param V_f: forward recurrent matrix. 2d (dim,dim*4)
    :param V_b: backward recurrent matrix. 2d (dim,dim*4)
    :param c_f: initial forward cell state. 2d (batch,dim)
    :param c_b: initial backward cell state. 2d (batch,dim)
    :param i: index. 2d (time,batch) -> 0 or 1
    """
    Z_f = gpu_contiguous(as_cuda_ndarray_variable(Z_f))
    Z_b = gpu_contiguous(as_cuda_ndarray_variable(Z_b))
    V_f = gpu_contiguous(as_cuda_ndarray_variable(V_f))
    V_b = gpu_contiguous(as_cuda_ndarray_variable(V_b))
    c_f = gpu_contiguous(as_cuda_ndarray_variable(c_f))
    c_b = gpu_contiguous(as_cuda_ndarray_variable(c_b))
    i_f = gpu_contiguous(as_cuda_ndarray_variable(T.cast(i_f,'float32')))
    i_b = gpu_contiguous(as_cuda_ndarray_variable(T.cast(i_b, 'float32')))
    assert Z_f.dtype == "float32"
    assert Z_f.dtype == "float32"
    assert V_f.dtype == "float32"
    assert V_b.dtype == "float32"
    assert c_f.dtype == 'float32'
    assert c_f.ndim == 2
    assert c_b.dtype == 'float32'
    assert c_b.ndim == 2
    assert Z_f.ndim == 3
    assert Z_b.ndim == 3
    assert i_f.ndim == 2
    assert i_b.ndim == 2
    assert V_f.ndim == 2
    assert V_b.ndim == 2

    # results: output Y, (gates and cell state) H, (final cell state) d
    return theano.Apply(self, [Z_f, Z_b, V_f, V_b, c_f, c_b, i_f, i_b],
                        [Z_f.type(), Z_f.type(), Z_f.type(), Z_f.type(), c_f.type(), c_b.type()])

  def c_support_code(self):
    crnn_path = os.path.dirname(__file__)
    with open(crnn_path + "/c_support_code_mdlstm.cpp") as f:
      return f.read()

  def c_code(self, node, name, input_names, output_names, sub):
    Z_f, Z_b, V_f, V_b, c_f, c_b, i_f, i_b = input_names
    Y_f, Y_b, H_f, H_b, d_f, d_b = output_names
    fail = sub['fail']
    inplace = "true" if self.inplace else "false"
    return """
    if(%(Y_f)s || %(H_f)s || %(d_f)s || %(Y_b)s || %(H_b)s || %(d_b)s)
    {
      //printf("Y or H or d already exist\\n");
      //TODO check if we can reuse it
      Py_XDECREF(%(Y_f)s);
      Py_XDECREF(%(Y_b)s);
      Py_XDECREF(%(H_f)s);
      Py_XDECREF(%(H_b)s);
      Py_XDECREF(%(d_f)s);
      Py_XDECREF(%(d_b)s);
    }

    const int * Z_dim = CudaNdarray_HOST_DIMS(%(Z_f)s);
    const int dims_Y[] = {Z_dim[0], Z_dim[1], Z_dim[2] / 4};
    const int dims_H[] = {Z_dim[0], Z_dim[1], Z_dim[2]};
    const int dims_d[] = {Z_dim[1], Z_dim[2] / 4};
    int size_d = Z_dim[1] * Z_dim[2] / 4;

    %(Y_f)s = (CudaNdarray*) CudaNdarray_NewDims(3,dims_Y);
    %(Y_b)s = (CudaNdarray*) CudaNdarray_NewDims(3,dims_Y);
    %(d_f)s = (CudaNdarray*) CudaNdarray_NewDims(2, dims_d);
    %(d_b)s = (CudaNdarray*) CudaNdarray_NewDims(2, dims_d);
    if(%(inplace)s)
    {
      %(H_f)s = %(Z_f)s;
      Py_INCREF(%(Z_f)s);
      %(H_b)s = %(Z_b)s;
      Py_INCREF(%(Z_b)s);
    }
    else
    {
      printf("no inplace\\n");
      %(H_f)s = (CudaNdarray*) CudaNdarray_NewDims(3,dims_H);
      cudaMemcpy(CudaNdarray_DEV_DATA(%(H_f)s), CudaNdarray_DEV_DATA(%(Z_f)s),
      dims_H[0]*dims_H[1]*dims_H[2]*sizeof(float), cudaMemcpyDeviceToDevice);

      %(H_b)s = (CudaNdarray*) CudaNdarray_NewDims(3,dims_H);
      cudaMemcpy(CudaNdarray_DEV_DATA(%(H_b)s), CudaNdarray_DEV_DATA(%(Z_b)s),
      dims_H[0]*dims_H[1]*dims_H[2]*sizeof(float), cudaMemcpyDeviceToDevice);
    }

    int y = 0;
    for(int x = 0; x < Z_dim[0]; ++x)
    {
      if(x > 0)
      {
        //H += Y[x-1]*V_h
        affine_y_x(y, x-1, %(Y_f)s, y, x, %(V_f)s, y, x, %(H_f)s);
        affine_y_x(y, x-1, %(Y_b)s, y, x, %(V_b)s, y, x, %(H_b)s);
      }
      float * d_ptr_f = (x == Z_dim[0] - 1) ? CudaNdarray_DEV_DATA(%(d_f)s) : 0;
      float * d_ptr_b = (x == Z_dim[0] - 1) ? CudaNdarray_DEV_DATA(%(d_b)s) : 0;
      /*float * d_ptr_f = (x == Z_dim[0] - 1) ? CudaNdarray_DEV_DATA(%(d_f)s) : 0;
      float * d_ptr_b = (x == Z_dim[0] - 1) ? CudaNdarray_DEV_DATA(%(d_b)s) : 0;
      do_lstm(%(H_f)s, %(Y_f)s, %(c_f)s, d_ptr, y, x, %(i_f)s);
      d_ptr = (x == Z_dim[0] - 1) ? CudaNdarray_DEV_DATA(%(d_b)s) : 0;
      do_lstm(%(H_b)s, %(Y_b)s, %(c_b)s, d_ptr, y, x, %(i_b)s);*/

      do_blstm(%(H_f)s, %(H_b)s, %(Y_f)s, %(Y_b)s, %(c_f)s, %(c_b)s, d_ptr_f, d_ptr_b, y, x, %(i_f)s, %(i_b)s);
    }

    """ % locals()

  def grad(self, inputs, output_grads):
    Z_f, Z_b, V_f, V_b, c_f, c_b, i_f, i_b = inputs
    DY_f, DY_b, DH_f, DH_b, Dd_f, Dd_b = output_grads

    Z_f_raw = Z_f.owner.inputs[0].owner.inputs[0]
    Z_b_raw = Z_b.owner.inputs[0].owner.inputs[0]
    #TODO!!!
    V_f_raw = V_f.owner.inputs[0]
    V_b_raw = V_b.owner.inputs[0]
    c_f_raw = c_f.owner.inputs[0].owner.inputs[0]
    c_b_raw = c_b.owner.inputs[0].owner.inputs[0]
    i_f_raw = i_f.owner.inputs[0].owner.inputs[0]
    i_b_raw = i_b.owner.inputs[0].owner.inputs[0]
    #we have to make sure that this in only computed once!
    #for this we have to extract the raw variables before conversion to continuous gpu array
    #so that theano can merge the nodes
    Y_f, Y_b, H_f, H_b, d_f, d_b = BLSTMOpInstance(Z_f_raw, Z_b_raw, V_f_raw, V_b_raw, c_f_raw, c_b_raw, i_f_raw, i_b_raw)
    if isinstance(DY_f.type, theano.gradient.DisconnectedType):
      DY_f = T.zeros_like(Z_f)
    if isinstance(DY_b.type, theano.gradient.DisconnectedType):
      DY_b = T.zeros_like(Z_b)
    if isinstance(Dd_f.type, theano.gradient.DisconnectedType):
      Dd_f = T.zeros_like(c_f)
    if isinstance(Dd_b.type, theano.gradient.DisconnectedType):
      Dd_b = T.zeros_like(c_b)
    DZ_f, DZ_b, DV_f, DV_b, Dc_f, Dc_b = BLSTMOpGradNoInplaceInstance(V_f, V_b, c_f, c_b, i_f, i_b, Dd_f, Dd_b, DY_f, DY_b, Y_f, Y_b, H_f, H_b)
    Di_f = theano.gradient.grad_undefined(self, 5, inputs[5], 'cannot diff w.r.t. index')
    Di_b = theano.gradient.grad_undefined(self, 6, inputs[6], 'cannot diff w.r.t. index')

    return [DZ_f, DZ_b, DV_f, DV_b, Dc_f, Dc_b, Di_f, Di_b]

  def infer_shape(self, node, input_shapes):
    Z_fs, Z_bs, V_fs, V_bs, c_fs, c_bs, idx_fs, idx_bs = input_shapes
    Y_shape = (Z_fs[0], Z_fs[1], Z_fs[2] / 4)
    H_shape = (Z_fs[0], Z_fs[1], Z_fs[2])
    d_shape = (Z_fs[1], Z_fs[2] / 4)
    return [Y_shape, Y_shape, H_shape, H_shape, d_shape, d_shape]

  #!!! change this when changing the code!
  def c_code_cache_version(self):
    return 1, 7

BLSTMOpInstance = BLSTMOp(inplace=False)
BLSTMOpInplaceInstance = BLSTMOp(inplace=True)

BLSTMOpInplaceOpt = OpSub(BLSTMOpInstance, BLSTMOpInplaceInstance)

#hack to avoid begin called twice
if not hasattr(optdb, 'BLSTMOpInplaceOpt_registered'):
  optdb.register('BLSTMOpInplaceOpt', theano.gof.TopoOptimizer(BLSTMOpInplaceOpt),
                 50.0, 'fast_run', 'inplace', 'gpuarray')
  optdb.BLSTMOpInplaceOpt_registered = True
