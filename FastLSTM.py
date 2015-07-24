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

  def make_node(self, X, W, V_h, b, DZ, Z, H, c):
    X = gpu_contiguous(as_cuda_ndarray_variable(X))
    W = gpu_contiguous(as_cuda_ndarray_variable(W))
    V_h = gpu_contiguous(as_cuda_ndarray_variable(V_h))
    b = gpu_contiguous(as_cuda_ndarray_variable(b))
    DZ = gpu_contiguous(as_cuda_ndarray_variable(DZ))
    assert X.dtype == "float32"
    assert W.dtype == "float32"
    assert V_h.dtype == "float32"
    assert b.dtype == 'float32'
    assert DZ.dtype == 'float32'
    assert Z.dtype == 'float32'
    assert H.dtype == 'float32'
    assert c.dtype == 'float32'
    assert X.ndim == 3
    assert W.ndim == 2
    assert V_h.ndim == 2
    assert b.ndim == 1
    assert DZ.ndim == 3
    assert Z.ndim == 3
    assert H.ndim == 3
    assert c.ndim == 2

    return theano.Apply(self, [X, W, V_h, b, DZ, Z, H, c], [X.type(), W.type(), V_h.type(), c.type(), b.type()])

  def infer_shape(self, node, input_shapes):
    Xs, Ws, V_hs, bs, DZs, Zs, Hs, cs = input_shapes
    return [Xs, Ws, V_hs, cs, bs]

  def c_support_code(self):
    crnn_path = os.path.dirname(__file__)
    with open(crnn_path + "/c_support_code_mdlstm.cpp") as f:
      return f.read()

  def c_code(self, node, name, input_names, output_names, sub):
    X, W, V_h, b, DZ, Z, H, c = input_names
    DX, DW, DV_h, Dc, Db = output_names
    fail = sub['fail']
    inplace = "true" if self.inplace else "false"
    return """

    // std::cout << "LSTMOpGrad called" << std::endl;
    if(!%(inplace)s)
    {
      std::cout << "warning, inplace optimization failed, not working inplace" << std::endl;
    }

    if(%(DX)s || %(DW)s || %(DV_h)s || %(Dc)s || %(Db)s)
    {
      printf("output storage already exists\\n");
      //TODO check if we can reuse it
      Py_XDECREF(%(DX)s);
      Py_XDECREF(%(DW)s);
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

    const int * X_dim = CudaNdarray_HOST_DIMS(%(X)s);
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
    %(DW)s = CudaNdarray_uninitialized_like(%(W)s);
    %(DV_h)s = CudaNdarray_uninitialized_like(%(V_h)s);
    //DV_h = Z[0..end-1]^T * delta[1..end]
    affine_global(%(Z)s, delta, %(DV_h)s, true, false, 1, 0.0f);
    //DX = delta * W^T
    affine_global(delta, %(W)s, %(DX)s, false, true, 0, 0.0f);
    //DW = X^T * delta
    affine_global(%(X)s, delta, %(DW)s, true, false, 0, 0.0f);
    //Db = (1 ... 1) * delta
    %(Db)s = sumOverAllButLastDimensions(delta);

    //TODO: calculate gradient w.r.t. c!
    %(Dc)s = CudaNdarray_zeros_like(%(c)s);

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

  def make_node(self, X, W, V_h, c, b):
    X = gpu_contiguous(as_cuda_ndarray_variable(X))
    W = gpu_contiguous(as_cuda_ndarray_variable(W))
    V_h = gpu_contiguous(as_cuda_ndarray_variable(V_h))
    b = gpu_contiguous(as_cuda_ndarray_variable(b))
    c = gpu_contiguous(as_cuda_ndarray_variable(c))
    assert X.dtype == "float32"
    assert W.dtype == "float32"
    assert V_h.dtype == "float32"
    assert b.dtype == 'float32'
    assert c.dtype == 'float32'
    assert c.ndim == 2
    assert X.ndim == 3
    assert W.ndim == 2
    assert V_h.ndim == 2
    assert b.ndim == 1

    #results: output Y, (gates and cell state) H
    return theano.Apply(self, [X, W, V_h, c, b], [X.type(), X.type()])

  def c_support_code(self):
    crnn_path = os.path.dirname(__file__)
    with open(crnn_path + "/c_support_code_mdlstm.cpp") as f:
      return f.read()

  def c_code(self, node, name, input_names, output_names, sub):
    X, W, V_h, c, b = input_names
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

    //std::cout << "LSTMOp called" << std::endl;

    const int * X_dim = CudaNdarray_HOST_DIMS(%(X)s);
    const int * W_dim = CudaNdarray_HOST_DIMS(%(W)s);
    //we can't use the modulo operator easily as it should not be replaced
    assert((W_dim[1] / 4) * 4 == W_dim[1] && "W has wrong shape");
    const int dims_Z[] = {X_dim[0], X_dim[1], W_dim[1] / 4};
    const int dims_H[] = {X_dim[0], X_dim[1], W_dim[1]};

    %(Z)s = (CudaNdarray*) CudaNdarray_NewDims(3,dims_Z);
    %(H)s = (CudaNdarray*) CudaNdarray_NewDims(3,dims_H);
    //init H with b
    fillmat(%(b)s, %(H)s);
    //H+=XW
    affine_global(%(X)s, %(W)s, %(H)s);

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
    X, W, V_h, c, b = inputs
    DZ, DH = output_grads

    X_raw = X.owner.inputs[0].owner.inputs[0]
    #TODO!!!
    W_raw = W.owner.inputs[0]
    V_h_raw = V_h.owner.inputs[0]
    c_raw = c.owner.inputs[0].owner.inputs[0]
    b_raw = b.owner.inputs[0]
    #we have to make sure that this in only computed once!
    #for this we have to extract the raw variables before conversion to continuous gpu array
    #so that theano can merge the nodes
    Z, H = LSTMOpInstance(X_raw, W_raw, V_h_raw, c_raw, b_raw)

    DX, DW, DV_h, Dc, Db = LSTMOpGradNoInplaceInstance(X, W, V_h, b, DZ, Z, H, c)

    return [DX, DW, DV_h, Dc, Db]

  def infer_shape(self, node, input_shapes):
    Xs, Ws, V_hs, cs, bs = input_shapes
    Z_shape = (Xs[0], Xs[1], Ws[1] / 4)
    H_shape = (Xs[0], Xs[1], Ws[1])
    return [Z_shape, H_shape]

  #!!! change this when changing the code!
  #def c_code_cache_version(self):
  #  return 1, 1

LSTMOpInstance = LSTMOp()

if __name__ == '__main__':
  #this is a test for the implementation

  X = T.ftensor3('X')
  W = T.fmatrix('W')
  V_h = T.fmatrix('V_h')
  b = T.fvector('b')
  Z, H = LSTMOpInstance(X, W, V_h, b)
  DX = T.grad(Z.sum(), X)
  DW = T.grad(Z.sum(), W)
  DV_h = T.grad(Z.sum(), V_h)
  Db = T.grad(Z.sum(), b)
  f = theano.function(inputs=[X, W, V_h, b], outputs=[Z, DX, DW, DV_h, Db])
  #g = theano.function(inputs=[X, W, V_h, b], outputs=[Z,H])

  X_val_mat0 = 0.1 * numpy.array([[1,2,3], [4,5,6]], dtype='float32')
  X_val_mat1 = 0.1 * numpy.array([[5,1,8], [7,0,1]], dtype='float32')
  X_val_mat2 = 0.1 * numpy.array([[2,1,1], [-7,0,-1]], dtype='float32')
  X_val = numpy.zeros((3,2,3), dtype='float32')
  X_val[0, :, :] = X_val_mat0
  X_val[1, :, :] = X_val_mat1
  X_val[2, :, :] = X_val_mat2
  #should be divisable by 4 for lstm, attention: note the .T
  W_val = 0.1 * numpy.array([[3,1,2], [4,8,0], [7,7,1], [4,2,-5],
                             [6,-1,-2], [-4,8,0], [-7,2,1], [4,-2,-5],
                             [6,5,-2], [-4,8,-6], [-7,3,-1], [4,2,-5]], dtype='float32').T
  #(for lstm) size 1/4th
  V_h_val = 0.1 * numpy.array([[1,3,5], [2,-1,-1], [4, 8,-5], [0,-2,3],
                               [7,7,7], [1,2,3], [5,2,1], [-4,8,-4],
                               [-3,7,-7], [2,-2,-3], [-5,2,1], [-4,-5,-4]],
                              dtype='float32').T
  b_val = 0.1 * numpy.array([1,2,3,4,5,6,7,8,9,10,11,12], dtype='float32')

  #print "calling g"
  #Z_val, H_val = g(X_val, W_val, V_h_val, b_val)
  #print numpy.asarray(Z_val), '\n', numpy.asarray(H_val)
  #print "done calling g"

  print "calling f"
  Z_val, DX_val, DW_val, DV_h_val, Db_val = f(X_val, W_val, V_h_val, b_val)
  print numpy.asarray(Z_val), '\n', numpy.asarray(DX_val), '\n', \
    numpy.asarray(DW_val), '\n', numpy.asarray(DV_h_val), '\n', numpy.asarray(Db_val)
  print "done calling f"

  print "verifying grad..."

  #def testOp_only_b(b):
  #  return TestOp()(X_val, W_val, V_h_val, b)[0]
  #theano.tests.unittest_tools.verify_grad(testOp_only_b, [b_val])

  def LSTMOp_Z(X, W, V_h, b):
    return LSTMOpInstance(X, W, V_h, b)[0]

  theano.tests.unittest_tools.verify_grad(LSTMOp_Z, [X_val, W_val, V_h_val, b_val])

  print "success"