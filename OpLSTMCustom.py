import os
from FunctionLoader import make_funloader_code
import theano
import theano.gradient
import theano.tensor as T
import theano.printing
import theano.gof
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           gpu_contiguous)


class LSTMCustomOpGrad(theano.sandbox.cuda.GpuOp):
  __props__ = ()

  def __init__(self):
    super(LSTMCustomOpGrad, self).__init__(self)
    self.inplace = False

  #TODO: later also accept X
  def make_node(self, Y, H, c, i, Dd, DY, W_re):
    c = gpu_contiguous(as_cuda_ndarray_variable(c))
    i = gpu_contiguous(as_cuda_ndarray_variable(T.cast(i,'float32')))
    Dd = gpu_contiguous(as_cuda_ndarray_variable(Dd))
    DY = gpu_contiguous(as_cuda_ndarray_variable(DY))
    W_re = gpu_contiguous(as_cuda_ndarray_variable(W_re))
    assert DY.dtype == 'float32'
    assert Y.dtype == 'float32'
    assert H.dtype == 'float32'
    assert c.dtype == 'float32'
    assert W_re.dtype == "float32"
    assert DY.ndim == 3
    assert Y.ndim == 3
    assert H.ndim == 3
    assert c.ndim == 2
    assert i.ndim == 2
    assert W_re.ndim == 2

    return theano.Apply(self, [Y, H, c, i, Dd, DY, W_re], [H.type(), c.type(), W_re.type()])

  def c_support_code(self):
    #do not remove this import as it is used in the c code
    import CustomLSTMFunctions
    crnn_path = os.path.dirname(__file__)
    #TODO!!!
    funloader = make_funloader_code("bwd_fun", 2)
    with open(crnn_path + "/c_support_code_mdlstm.cpp") as f:
      return funloader + f.read()

  def c_code(self, node, name, input_names, output_names, sub):
    Y, H, c, i, Dd, DY, W_re = input_names
    DZ, Dc, DW_re = output_names
    fail = sub['fail']
    inplace = "true" if self.inplace else "false"
    return """
    std::cout << "LSTMCustomOpGrad called" << std::endl;
    if(%(DZ)s || %(Dc)s || %(DW_re)s)
    {
      printf("output storage already exists\\n");
      //TODO check if we can reuse it
      Py_XDECREF(%(DZ)s);
      Py_XDECREF(%(Dc)s);
      Py_XDECREF(%(DW_re)s);
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
      bool leftBorder = (x == 0);

      //TODO: check if we need to handle boundary case specially
      if(!rightBorder)
      {
        //affine_y_x(y, x+1, delta, y, x, %(W_re)s, y, x, epsilon, false, true);

        //call custom function here
        CudaNdarray * y_p = 0;
        //TODO: handle leftBorder
        if(!leftBorder)
        {
          PyObject * y_p_obj = PyObject_CallMethod((PyObject*) %(Y)s, "__getitem__", "(i)", x-1);
          assert(y_p_obj);
          y_p = (CudaNdarray*) y_p_obj;
          PyObject * delta_x_obj = PyObject_CallMethod((PyObject*) delta, "__getitem__", "(i)", x+1);
          assert(delta_x_obj);
          CudaNdarray * delta_x = (CudaNdarray*) delta_x_obj;

          std::vector<PyObject*> res_vec = bwd_fun(y_p, %(W_re)s, delta_x);
          assert(res_vec.size() == 2);
          CudaNdarray * Dy_p = (CudaNdarray*) res_vec[0];
          CudaNdarray * DW_re_local = (CudaNdarray*) res_vec[1];

          //TODO
          //copy to epsilon
          float * epsilon_x_data = data_ptr(epsilon, y, x);
          do_add(epsilon_x_data, CudaNdarray_DEV_DATA(Dy_p), CudaNdarray_SIZE(Dy_p));

          Py_XDECREF(Dy_p);
          Py_XDECREF(DW_re_local);
        }
      }

      do_lstm_bwd(delta, epsilon, %(Y)s, %(Dd)s, %(c)s, y, x, rightBorder, %(i)s);
    }

    %(DW_re)s = CudaNdarray_uninitialized_like(%(W_re)s);
    //DW_re = Y[0..end-1]^T * delta[1..end]
    affine_global(%(Y)s, delta, %(DW_re)s, true, false, 1, 0.0f);

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

LSTMCustomOpGradInstance = LSTMCustomOpGrad()

#------------------------

class LSTMCustomOp(theano.sandbox.cuda.GpuOp):
  __props__ = ()

  def __init__(self):
    super(LSTMCustomOp, self).__init__(self)
    self.inplace = False

  #TODO: make recurrent weights customizable, atm fixed to single matrix (W_re)
  #X is base
  def make_node(self, Z, X, c, i, W_re):
    from Device import have_gpu
    assert have_gpu()

    #X: context (on which attention is applied)
    Z = gpu_contiguous(as_cuda_ndarray_variable(Z))
    X = gpu_contiguous(as_cuda_ndarray_variable(X))
    c = gpu_contiguous(as_cuda_ndarray_variable(c))
    i = gpu_contiguous(as_cuda_ndarray_variable(T.cast(i,'float32')))
    W_re = gpu_contiguous(as_cuda_ndarray_variable(W_re))
    assert Z.dtype == "float32"
    assert X.dtype == "float32"
    assert c.dtype == "float32"
    assert W_re.dtype == "float32"
    assert Z.ndim == 3
    assert X.ndim == 3
    assert c.ndim == 2
    assert i.ndim == 2
    assert W_re.ndim == 2

    #results: output Y, (gates and cell state) H
    return theano.Apply(self, [Z, X, c, i, W_re], [Z.type(), Z.type(), c.type()])

  def c_support_code(self):
    #do not remove this import as it is used in the c code
    import CustomLSTMFunctions
    funloader = make_funloader_code("fwd_fun", 1)
    crnn_path = os.path.dirname(__file__)
    with open(crnn_path + "/c_support_code_mdlstm.cpp") as f:
      return funloader + f.read()

  def c_code(self, node, name, input_names, output_names, sub):
    Z, X, c, i, W_re = input_names
    Y, H, d = output_names
    fail = sub['fail']
    return """
    std::cout << "LSTMCustomOp called" << std::endl;
    if(%(Y)s || %(H)s || %(d)s)
    {
      printf("Y or H or d already exist\\n");
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
    %(H)s = (CudaNdarray*) CudaNdarray_NewDims(3,dims_H); //CudaNdarray_uninitialized_like(%(Z)s);
    cudaMemcpy(CudaNdarray_DEV_DATA(%(H)s), CudaNdarray_DEV_DATA(%(Z)s),
      dims_H[0]*dims_H[1]*dims_H[2]*sizeof(float), cudaMemcpyDeviceToDevice);

    int y = 0;
    for(int x = 0; x < Z_dim[0]; ++x)
    {
      bool leftBorder = (x == 0);
      //TODO: later we also need to handle the first state, but atm we can let it be handled outside
      if(!leftBorder)
      {
        //affine_y_x(y, x-1, %(Y)s, y, x, %(W_re)s, y, x, %(H)s);

        //call custom function here
        PyObject * y_p_obj = PyObject_CallMethod((PyObject*) %(Y)s, "__getitem__", "(i)", x-1);
        assert(y_p_obj);
        CudaNdarray * y_p = (CudaNdarray*) y_p_obj;

        std::vector<PyObject*> res_vec = fwd_fun(y_p, %(W_re)s);
        assert(res_vec.size() == 1);
        CudaNdarray * res = (CudaNdarray*) res_vec[0];
        Py_XDECREF(y_p);

        //copy to H
        float * H_y_x_data = data_ptr(%(H)s, y, x);
        do_add(H_y_x_data, CudaNdarray_DEV_DATA(res), CudaNdarray_SIZE(res));

        Py_XDECREF(res);
      }
      float * d_ptr = (x == Z_dim[0] - 1) ? CudaNdarray_DEV_DATA(%(d)s) : 0;
      do_lstm(%(H)s, %(Y)s, %(c)s, d_ptr, y, x, %(i)s);
    }
    """ % locals()

  def grad(self, inputs, output_grads):
    Z, X, c, i, W_re = inputs
    DY, DH, Dd = output_grads

    Z_raw = Z.owner.inputs[0].owner.inputs[0]
    X_raw = X.owner.inputs[0].owner.inputs[0]
    c_raw = c.owner.inputs[0].owner.inputs[0]
    i_raw = i.owner.inputs[0].owner.inputs[0]
    W_re_raw = W_re.owner.inputs[0]
    #we have to make sure that this in only computed once!
    #for this we have to extract the raw variables before conversion to continuous gpu array
    #so that theano can merge the nodes
    Y, H, d = LSTMCustomOpInstance(Z_raw, X_raw, c_raw, i_raw, W_re_raw)
    if isinstance(DY.type, theano.gradient.DisconnectedType):
      DY = T.zeros_like(Z)
    if isinstance(Dd.type, theano.gradient.DisconnectedType):
      Dd = T.zeros_like(c)
    #TODO: later also pass X
    DZ, Dc, DW_re = LSTMCustomOpGradInstance(Y, H, c, i, Dd, DY, W_re)
    Di = theano.gradient.grad_undefined(self, 3, inputs[3], 'cannot diff w.r.t. index')
    #TODO
    DX = theano.gradient.grad_undefined(self, 2, inputs[2], 'cannot diff w.r.t. X yet')

    return [DZ, DX, Dc, Di, DW_re]

LSTMCustomOpInstance = LSTMCustomOp()
