import os
from FunctionLoader import make_funloader_code
import theano
import theano.gradient
import theano.tensor as T
import theano.printing
import theano.gof
from theano.gof.opt import OpSub
from theano.compile import optdb
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           gpu_contiguous)


class LSTMCustomOpGrad(theano.sandbox.cuda.GpuOp):
  __props__ = ("inplace", "fun_name", "recurrent_transform")

  def __init__(self, fun_name, inplace, recurrent_transform):
    """
    :type recurrent_transform: RecurrentTransform.RecurrentTransformBase
    """
    super(LSTMCustomOpGrad, self).__init__()
    self.inplace = inplace
    self.fun_name = fun_name
    self.recurrent_transform = recurrent_transform
    if inplace:
     # http://deeplearning.net/software/theano/extending/inplace.html
     # https://github.com/Theano/Theano/issues/3506
     # It's strange that we must mark which output operates on which input -
     # I would expect that it must only know which inputs are destroyed.
     # Anyway:
     # All outputs operate inplace on inputs 1 and 6 (which are H and DY)
     # but when the input is marked multiple times, we get an error.
     # This is also strange, and probably a bug in Theano.
     # So we could mark that output 0 destroys inputs 1 and 6.
     # That also doesn't work, it will not apply the inplace-optimization anymore.
     # So, we do it in some other way. From how I understand the Theano code,
     # the output index is ignored, so we can use any.
     # Anyway Theano knows what inputs will be destroyed, so it should be OK.
     destroy_input_list = [1, 6]
     self.destroy_map = {i: [i] for i in destroy_input_list}  # hack, see above

  def _get_num_custom_vars(self):
    return len(self.recurrent_transform.custom_vars)

  def _get_num_state_vars(self):
    return len(self.recurrent_transform.state_vars)

  def make_node(self, Y, H, c, y0, i, freq, Dd, DY, W_re, *args):
    c = gpu_contiguous(as_cuda_ndarray_variable(c))
    y0 = gpu_contiguous(as_cuda_ndarray_variable(y0))
    i = gpu_contiguous(as_cuda_ndarray_variable(T.cast(i,'float32')))
    Dd = gpu_contiguous(as_cuda_ndarray_variable(Dd))
    DY = gpu_contiguous(as_cuda_ndarray_variable(DY))
    W_re = gpu_contiguous(as_cuda_ndarray_variable(W_re))
    args = [gpu_contiguous(as_cuda_ndarray_variable(x)) for x in args]
    freq = gpu_contiguous(as_cuda_ndarray_variable(freq))

    # args = custom_inputs + state_vars_seqs
    assert len(args) == self._get_num_custom_vars() + self._get_num_state_vars()
    assert freq.dtype == 'float32'
    assert DY.dtype == 'float32'
    assert Y.dtype == 'float32'
    assert H.dtype == 'float32'
    assert c.dtype == 'float32'
    assert y0.dtype == "float32"
    assert W_re.dtype == "float32"
    for x in args:
      assert x.dtype == "float32"
    assert DY.ndim == 3
    assert Y.ndim == 3
    assert H.ndim == 3
    assert c.ndim == 2
    assert y0.ndim == 2
    assert i.ndim == 2
    assert W_re.ndim == 2

    custom_input_grads = [var.type() for var in args[:self._get_num_custom_vars()]]
    CudaNdarrayType = theano.sandbox.cuda.CudaNdarrayType
    # One ndim less because initial state var grads vs whole seq state vars.
    initial_state_var_grads = [CudaNdarrayType(dtype="float32", broadcastable=(False,) * (var.ndim - 1))()
                               for var in args[self._get_num_custom_vars():]]
    return theano.Apply(self, [Y, H, c, y0, i, freq, Dd, DY, W_re] + args,
                        # DZ, Dc, Dy0, DW_re, custom input grads, initial state var grads
                        [H.type(), c.type(), y0.type(), W_re.type()] + custom_input_grads + initial_state_var_grads)

  def c_support_code(self):
    crnn_path = os.path.dirname(__file__)
    fun_prefix = "%s_%i" % (self.fun_name, id(self.recurrent_transform))
    funloader = make_funloader_code(self.recurrent_transform, fun_prefix + "_fun_bwd", fun_prefix + "_fun_reset")
    with open(crnn_path + "/c_support_code_mdlstm.cpp") as f:
      return funloader + f.read()

  def c_code(self, node, name, input_names, output_names, sub):
    (Y, H, c, y0, i, freq, Dd, DY, W_re), remaining_inputs = input_names[:9], input_names[9:]
    freq = 1 # TODO
    assert len(remaining_inputs) == self._get_num_custom_vars() + self._get_num_state_vars()
    custom_inputs = remaining_inputs[:self._get_num_custom_vars()]
    seq_state_var_names = remaining_inputs[self._get_num_custom_vars():]
    custom_inputs_str = ",".join(custom_inputs)
    seq_state_var_names_str = ", ".join(seq_state_var_names)
    (DZ, Dc, Dy0, DW_re), remaining_outputs = output_names[:4], output_names[4:]
    assert len(remaining_outputs) == self._get_num_custom_vars() + self._get_num_state_vars()
    custom_output_names = remaining_outputs[:self._get_num_custom_vars()]
    initial_state_var_grad_names = remaining_outputs[self._get_num_custom_vars():]
    custom_outputs_str = ", ".join(["&" + grad for grad in custom_output_names])
    initial_state_var_grad_names_str = ", ".join(["&" + grad for grad in initial_state_var_grad_names])
    bwd_fun = "%s_%i_fun_bwd" % (self.fun_name, id(self.recurrent_transform))
    fail = sub['fail']
    inplace = "true" if self.inplace else "false"
    return """
    //std::cout << "LSTMCustomOpGrad called" << std::endl;
    if(!%(inplace)s)
    {
      //std::cout << "warning, inplace optimization failed, not working inplace" << std::endl;
    }

    if(%(DZ)s || %(Dc)s || %(DW_re)s || %(Dy0)s)
    {
      printf("output storage already exists\\n");
      //TODO check if we can reuse it
      Py_XDECREF(%(DZ)s);
      Py_XDECREF(%(Dc)s);
      Py_XDECREF(%(DW_re)s);
      Py_XDECREF(%(Dy0)s);
    }

    #define ARRAY_LEN(x) (sizeof(x) / sizeof(x[0]))
    CudaNdarray* custom_inputs[] = {%(custom_inputs_str)s}; // input
    %(bwd_fun)s.reset_shared(custom_inputs, ARRAY_LEN(custom_inputs)); // init the custom grads with zero

    CudaNdarray* seq_state_vars[] = {%(seq_state_var_names_str)s}; // input
    CudaNdarray** state_var_grads[] = {%(initial_state_var_grad_names_str)s}; // output
    for(int i = 0; i < ARRAY_LEN(state_var_grads); ++i) {
      Py_XDECREF(*state_var_grads[i]); // in case of earlier output storage
      // dims like seq_state_vars[i] without time, which is the first dim
      int ndim = CudaNdarray_NDIM(seq_state_vars[i]) - 1;
      const int* dims = CudaNdarray_HOST_DIMS(seq_state_vars[i]) + 1;
      *state_var_grads[i] = (CudaNdarray*) CudaNdarray_ZEROS(ndim, (int*) dims);
      assert(*state_var_grads[i]);
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
    const int * Y_dim = CudaNdarray_HOST_DIMS(%(Y)s);

    int y = 0;
    for(int x = H_dim[0]-1; x >= 0; --x)
    {
      //add recurrent
      bool rightBorder = (x == H_dim[0]-1);
      bool leftBorder = (x == 0);

      if(!rightBorder)
      {
        affine_y_x(y, x+1, delta, y, x, %(W_re)s, y, x, epsilon, false, true);
      }

      //call custom function here
      //const float *freqs = data_ptr(%(freq)s)
      //if(!rightBorder && x %% (int)(freqs[0]) == 0)
      if(!rightBorder && x %% %(freq)d == 0)
      {
        CudaNdarray * y_p = 0;
        //x-1?
        PyObject * y_p_obj = PyObject_CallMethod((PyObject*) %(Y)s, "__getitem__", "(i)", x);
        assert(y_p_obj);
        y_p = (CudaNdarray*) y_p_obj;

        PyObject * delta_x_obj = PyObject_CallMethod((PyObject*) delta, "__getitem__", "(i)", x+1);
        assert(delta_x_obj);
        CudaNdarray * delta_x = (CudaNdarray*) delta_x_obj;

        CudaNdarray* state_vars_prev[ARRAY_LEN(seq_state_vars)];
        for(int i = 0; i < ARRAY_LEN(seq_state_vars); ++i) {
          state_vars_prev[i] = (CudaNdarray*) PyObject_CallMethod((PyObject*) seq_state_vars[i], "__getitem__", "(i)", x+1);
          assert(state_vars_prev[i]);
        }

        // bwd_fun args: y_p, custom inputs, state vars prev, Dz_re, state var new grads
        CudaNdarray* bwd_fun_inputs[2 + ARRAY_LEN(custom_inputs) + 2 * ARRAY_LEN(seq_state_vars)];
        {
          int idx = 0;
          bwd_fun_inputs[idx++] = y_p;
          for(int i = 0; i < ARRAY_LEN(custom_inputs); ++i)
            bwd_fun_inputs[idx++] = custom_inputs[i];
          for(int i = 0; i < ARRAY_LEN(state_vars_prev); ++i)
            bwd_fun_inputs[idx++] = state_vars_prev[i];
          bwd_fun_inputs[idx++] = delta_x;
          for(int i = 0; i < ARRAY_LEN(state_var_grads); ++i)
            bwd_fun_inputs[idx++] = *state_var_grads[i];
          assert(idx == ARRAY_LEN(bwd_fun_inputs));
        }
        std::vector<CudaNdarray*> res_vec = %(bwd_fun)s.call(bwd_fun_inputs, ARRAY_LEN(bwd_fun_inputs));
        // result shared vars: Dy_p, custom input grads, state var grads
        assert(res_vec.size() == 1 + ARRAY_LEN(custom_inputs) + ARRAY_LEN(seq_state_vars));
        Py_XDECREF(delta_x);
        CudaNdarray * Dy_p = (CudaNdarray*) res_vec[0];

        //copy to epsilon
        float * epsilon_x_data = data_ptr(epsilon, y, x);
        do_add(epsilon_x_data, CudaNdarray_DEV_DATA(Dy_p), CudaNdarray_SIZE(Dy_p));

        // custom input grads will automatically be accumulated. see CustomLSTMFunctions.
        // copy state var grads
        {
          int idx = 1 + ARRAY_LEN(custom_inputs);
          for(int i = 0; i < ARRAY_LEN(seq_state_vars); ++i) {
            CudaNdarray* dst = *state_var_grads[i];
            CudaNdarray* src = res_vec[idx++];
            assert(CudaNdarray_SIZE(dst) == CudaNdarray_SIZE(src));
            cudaMemcpy(
              CudaNdarray_DEV_DATA(dst), CudaNdarray_DEV_DATA(src),
              CudaNdarray_SIZE(src) * sizeof(real), cudaMemcpyDeviceToDevice);
          }
          assert(res_vec.size() == idx);
        }

        for(int i = 0; i < res_vec.size(); ++i)
          Py_XDECREF(res_vec[i]);
        for(int i = 0; i < ARRAY_LEN(state_vars_prev); ++i)
          Py_XDECREF(state_vars_prev[i]);
        Py_XDECREF(y_p);
      }

      do_lstm_bwd(delta, epsilon, %(Y)s, %(Dd)s, %(c)s, y, x, rightBorder, %(i)s);
    }

    %(DW_re)s = CudaNdarray_uninitialized_like(%(W_re)s);
    //DW_re = Y[0..end-1]^T * delta[1..end]
    affine_global(%(Y)s, delta, %(DW_re)s, true, false, 1, 0.0f);
    //DW_re += y0^T * delta[0]
    affine_y_x(0, 0, %(y0)s, 0, 0, delta, 0, 0, %(DW_re)s, true, false);

    %(DZ)s = delta;

    %(Dc)s = CudaNdarray_uninitialized_like(%(c)s);
    HANDLE_ERROR(cudaMemcpy(CudaNdarray_DEV_DATA(%(Dc)s), CudaNdarray_DEV_DATA(epsilon),
      Y_dim[1]*Y_dim[2]*sizeof(float), cudaMemcpyDeviceToDevice));

    %(Dy0)s = CudaNdarray_zeros_like(%(y0)s);
    //calculation like epsilon
    affine_y_x(0, 0, delta, 0, 0, %(W_re)s, 0, 0, %(Dy0)s, false, true);

    //add custom function
    //TODO: move to function
    PyObject * delta_x_obj = PyObject_CallMethod((PyObject*) delta, "__getitem__", "(i)", 0);
    assert(delta_x_obj);
    CudaNdarray * delta_x = (CudaNdarray*) delta_x_obj;
    CudaNdarray* state_vars_prev[ARRAY_LEN(seq_state_vars)];
    for(int i = 0; i < ARRAY_LEN(seq_state_vars); ++i) {
      // left border
      state_vars_prev[i] = (CudaNdarray*) PyObject_CallMethod((PyObject*) seq_state_vars[i], "__getitem__", "(i)", 0);
      assert(state_vars_prev[i]);
    }
    // bwd_fun args: y_p, custom inputs, state vars prev, Dz_re, state var new grads
    CudaNdarray* bwd_fun_inputs[2 + ARRAY_LEN(custom_inputs) + 2 * ARRAY_LEN(seq_state_vars)];
    {
      int idx = 0;
      bwd_fun_inputs[idx++] = %(y0)s;
      for(int i = 0; i < ARRAY_LEN(custom_inputs); ++i)
        bwd_fun_inputs[idx++] = custom_inputs[i];
      for(int i = 0; i < ARRAY_LEN(state_vars_prev); ++i)
        bwd_fun_inputs[idx++] = state_vars_prev[i];
      bwd_fun_inputs[idx++] = delta_x;
      for(int i = 0; i < ARRAY_LEN(state_var_grads); ++i)
        bwd_fun_inputs[idx++] = *state_var_grads[i];
      assert(idx == ARRAY_LEN(bwd_fun_inputs));
    }
    std::vector<CudaNdarray*> res_vec = %(bwd_fun)s.call(bwd_fun_inputs, ARRAY_LEN(bwd_fun_inputs));
    // result shared vars: Dy_p, custom input grads, state var grads
    assert(res_vec.size() == 1 + ARRAY_LEN(custom_inputs) + ARRAY_LEN(seq_state_vars));
    Py_XDECREF(delta_x);
    {
      int idx = 0;
      CudaNdarray * Dy_p = res_vec[idx++];
      //copy to Dy0
      do_add(CudaNdarray_DEV_DATA(%(Dy0)s), CudaNdarray_DEV_DATA(Dy_p), CudaNdarray_SIZE(Dy_p));

      //custom grads
      CudaNdarray** custom_grads[] = {%(custom_outputs_str)s}; // output
      for(int i = 0; i < ARRAY_LEN(custom_grads); ++i) {
        *custom_grads[i] = (CudaNdarray*) CudaNdarray_Copy(res_vec[idx++]);
        assert(*custom_grads[i]);
      }

      // copy state var grads
      for(int i = 0; i < ARRAY_LEN(seq_state_vars); ++i) {
        CudaNdarray* dst = *state_var_grads[i];
        CudaNdarray* src = res_vec[idx++];
        assert(CudaNdarray_SIZE(dst) == CudaNdarray_SIZE(src));
        cudaMemcpy(
          CudaNdarray_DEV_DATA(dst), CudaNdarray_DEV_DATA(src),
          CudaNdarray_SIZE(src) * sizeof(real), cudaMemcpyDeviceToDevice);
      }
      assert(res_vec.size() == idx);
    }

    for(int i = 0; i < res_vec.size(); ++i)
      Py_XDECREF(res_vec[i]);
    for(int i = 0; i < ARRAY_LEN(state_vars_prev); ++i)
      Py_XDECREF(state_vars_prev[i]);
    if(!%(inplace)s)
      Py_XDECREF(epsilon);

    #undef ARRAY_LEN
    """ % locals()

#------------------------

class LSTMCustomOp(theano.sandbox.cuda.GpuOp):
  __props__ = ("inplace", "fun_name", "recurrent_transform")

  def __init__(self, fun_name, inplace, recurrent_transform):
    """
    :type recurrent_transform: RecurrentTransform.RecurrentTransformBase
    """
    super(LSTMCustomOp, self).__init__()
    self.inplace = inplace
    self.fun_name = fun_name
    self.recurrent_transform = recurrent_transform
    if inplace:
      #all outputs operate inplace on input 0 (which is Z)
      #but when the input is marked multiple times, we get an error
      #so we only mark that output 0 destroys input 0
      #anyway theano knows that input 0 will be destroyed, so it should be OK
      self.destroy_map = {0: [0]}

  def _get_num_custom_vars(self):
    return len(self.recurrent_transform.custom_vars)

  def _get_num_state_vars(self):
    return len(self.recurrent_transform.state_vars)

  def _seq_var_for_initial_state_var(self, v):
    type_class = v.type.__class__
    # One ndim more for time.
    seq_var_type = type_class(dtype="float32", broadcastable=(False,) * (v.ndim + 1))
    return seq_var_type()

  def make_node(self, Z, c, y0, i, freq, W_re, *args):
    """
    :param Z: {input,output,forget} gate + cell state. 3d (time,batch,dim*4)
    :param c: initial cell state. 2d (batch,dim)
    :param y0: output of t = -1 (for recursion at t = 0). 2d (batch,dim)
    :param i: index. 2d (time,batch) -> 0 or 1
    :param W_re: recurrent matrix. 2d (dim,dim*4)
    :param freq: call frequency to custom function. int
    :param args: custom_inputs + initial_state_vars: other inputs for the custom function
    """
    from Device import have_gpu
    assert have_gpu()

    assert len(args) == self._get_num_custom_vars() + self._get_num_state_vars(), self.recurrent_transform
    custom_inputs = args[:self._get_num_custom_vars()]
    initial_state_vars = args[self._get_num_custom_vars():]

    custom_inputs = [gpu_contiguous(as_cuda_ndarray_variable(x)) for x in custom_inputs]
    initial_state_vars = [gpu_contiguous(as_cuda_ndarray_variable(x)) for x in initial_state_vars]
    Z = gpu_contiguous(as_cuda_ndarray_variable(Z))
    c = gpu_contiguous(as_cuda_ndarray_variable(c))
    y0 = gpu_contiguous(as_cuda_ndarray_variable(y0))
    i = gpu_contiguous(as_cuda_ndarray_variable(T.cast(i,'float32')))
    W_re = gpu_contiguous(as_cuda_ndarray_variable(W_re))
    self.freq = gpu_contiguous(as_cuda_ndarray_variable(freq))
    assert Z.dtype == "float32"
    assert c.dtype == "float32"
    assert y0.dtype == "float32"
    assert W_re.dtype == "float32"
    for x in custom_inputs:
      assert x.dtype == "float32"
    for x in initial_state_vars:
      assert x.dtype == "float32"
    assert Z.ndim == 3
    assert c.ndim == 2
    assert y0.ndim == 2
    assert i.ndim == 2
    assert W_re.ndim == 2

    seq_state_vars = [self._seq_var_for_initial_state_var(x) for x in initial_state_vars]
    return theano.Apply(self,
                        [Z, c, y0, i, freq, W_re] + custom_inputs + initial_state_vars,
                        # results: (output) Y, (gates and cell state) H, (final cell state) d, state vars sequences
                        [Z.type(), Z.type(), c.type()] + seq_state_vars)

  def c_support_code(self):
    fun_prefix = "%s_%i" % (self.fun_name, id(self.recurrent_transform))
    funloader = make_funloader_code(self.recurrent_transform, fun_prefix + "_fun_fwd")
    crnn_path = os.path.dirname(__file__)
    with open(crnn_path + "/c_support_code_mdlstm.cpp") as f:
      return funloader + f.read()

  def c_code(self, node, name, input_names, output_names, sub):
    # Y: all the outputs. 3d (time,batch,dim)
    # Z/H: {input,output,forget} gate + cell state. 3d (time,batch,dim*4)
    # d: last state (= Y[T-1]). 2d (batch,dim)
    Z, c, y0, i, freq, W_re = input_names[:6]
    freq = 1 # TODO
    custom_inputs = input_names[6:]
    assert len(custom_inputs) == self._get_num_custom_vars() + self._get_num_state_vars()
    custom_inputs, initial_state_vars = custom_inputs[:self._get_num_custom_vars()], custom_inputs[self._get_num_custom_vars():]
    custom_inputs_str = ", ".join(custom_inputs)
    initial_state_vars_str = ", ".join(initial_state_vars)
    Y, H, d = output_names[:3]
    state_vars_seqs = output_names[3:]
    assert len(state_vars_seqs) == self._get_num_state_vars()
    state_vars_seqs_str_comma = "".join([", %s[x]" % x for x in state_vars_seqs])
    state_vars_seqs_ptr_str = ", ".join(["&" + x for x in state_vars_seqs])
    fwd_fun = "%s_%i_fun_fwd" % (self.fun_name, id(self.recurrent_transform))
    inplace = "true" if self.inplace else "false"
    fail = sub['fail']
    # see https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/cuda_ndarray.cuh for some doc
    return """
    //std::cout << "LSTMCustomOp called" << std::endl;
    if(%(Y)s || %(H)s || %(d)s)
    {
      printf("Y or H or d already exist\\n");
      //TODO check if we can reuse it
      Py_XDECREF(%(Y)s);
      Py_XDECREF(%(H)s);
      Py_XDECREF(%(d)s);
    }

    // outputs
    const int * Z_dim = CudaNdarray_HOST_DIMS(%(Z)s);
    const int dims_Y[] = {Z_dim[0], Z_dim[1], Z_dim[2] / 4};
    const int dims_H[] = {Z_dim[0], Z_dim[1], Z_dim[2]};
    const int dims_d[] = {Z_dim[1], Z_dim[2] / 4};
    int size_d = Z_dim[1] * Z_dim[2] / 4;

    %(Y)s = (CudaNdarray*) CudaNdarray_NewDims(3, dims_Y);
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

    CudaNdarray* custom_inputs[] = {%(custom_inputs_str)s};

    // custom state vars seqs outputs
    #define ARRAY_LEN(x) (sizeof(x) / sizeof(x[0]))
    CudaNdarray* initial_state_vars[] = {%(initial_state_vars_str)s};
    CudaNdarray** state_vars_seqs_ptr[] = {%(state_vars_seqs_ptr_str)s};
    assert(ARRAY_LEN(initial_state_vars) == ARRAY_LEN(state_vars_seqs_ptr));
    for(int i = 0; i < ARRAY_LEN(initial_state_vars); ++i) {
      const int initial_ndim = CudaNdarray_NDIM(initial_state_vars[i]);
      int ndim = initial_ndim + 1; // add time-dim
      const int* initial_dims = CudaNdarray_HOST_DIMS(initial_state_vars[i]);
      int dims[] = {Z_dim[0], 0, 0, 0};
      assert(ARRAY_LEN(dims) >= ndim);
      for(int d = 0; d < initial_ndim; ++d)
        dims[d + 1] = initial_dims[d];
      *state_vars_seqs_ptr[i] = (CudaNdarray*) CudaNdarray_NewDims(ndim, dims);
      // copy initial over
      cudaMemcpy(
        CudaNdarray_DEV_DATA(*state_vars_seqs_ptr[i]),
        CudaNdarray_DEV_DATA(initial_state_vars[i]),
        CudaNdarray_SIZE(initial_state_vars[i]) * sizeof(real), cudaMemcpyDeviceToDevice);
    }

    int y = 0;
    for(int x = 0; x < Z_dim[0]; ++x)
    {
      bool leftBorder = (x == 0);
      bool rightBorder = (x == Z_dim[0] - 1);
      if(leftBorder)
      {
        affine_y_x(y, x-1, %(y0)s, y, x, %(W_re)s, y, x, %(H)s);
      }
      else
      {
        affine_y_x(y, x-1, %(Y)s, y, x, %(W_re)s, y, x, %(H)s);
      }

      // call custom function here
      //const float *freqs = data_ptr(%(freq)s);
      //if(x %% (int)(freqs[0]) == 0)
      if(x %% %(freq)d == 0)
      {
        CudaNdarray* state_vars[ARRAY_LEN(state_vars_seqs_ptr)];
        for(int i = 0; i < ARRAY_LEN(state_vars_seqs_ptr); ++i) {
          state_vars[i] = (CudaNdarray*) PyObject_CallMethod((PyObject*) *state_vars_seqs_ptr[i], "__getitem__", "(i)", x);
          assert(state_vars[i]);
        }

        CudaNdarray * y_p = 0;
        if(leftBorder)
        {
          y_p = %(y0)s;
        }
        else
        {
          PyObject * y_p_obj = PyObject_CallMethod((PyObject*) %(Y)s, "__getitem__", "(i)", x-1);
          assert(y_p_obj);
          y_p = (CudaNdarray*) y_p_obj;
        }
        //std::cerr << "t=" << x << std::endl;
        // fwd fun args: y_p, custom inputs, state vars
        CudaNdarray* fun_args[1 + ARRAY_LEN(custom_inputs) + ARRAY_LEN(state_vars)];
        {
          int idx = 0;
          fun_args[idx++] = y_p;
          for(int i = 0; i < ARRAY_LEN(custom_inputs); ++i)
            fun_args[idx++] = custom_inputs[i];
          for(int i = 0; i < ARRAY_LEN(state_vars); ++i)
            fun_args[idx++] = state_vars[i];
          assert(idx == ARRAY_LEN(fun_args));
        }

        std::vector<CudaNdarray*> res_vec = %(fwd_fun)s.call(fun_args, ARRAY_LEN(fun_args));
        assert(res_vec.size() == 1 + ARRAY_LEN(initial_state_vars));

        // add to H
        {
          CudaNdarray * res = res_vec[0];
          float * H_y_x_data = data_ptr(%(H)s, y, x);
          do_add(H_y_x_data, CudaNdarray_DEV_DATA(res), CudaNdarray_SIZE(res));
        }

        if(!rightBorder) {
          // set new state vars
          for(int i = 0; i < ARRAY_LEN(initial_state_vars); ++i) {
            CudaNdarray* src = res_vec[i + 1];
            float* src_ptr = CudaNdarray_DEV_DATA(src);
            CudaNdarray* dst = *state_vars_seqs_ptr[i];
            float* dst_ptr = CudaNdarray_DEV_DATA(dst) + CudaNdarray_HOST_STRIDES(dst)[0] * (x + 1);
            assert(CudaNdarray_HOST_STRIDES(dst)[0] == CudaNdarray_SIZE(src));
            cudaMemcpy(dst_ptr, src_ptr, CudaNdarray_SIZE(src) * sizeof(real), cudaMemcpyDeviceToDevice);
          }
        }

        for(int i = 0; i < res_vec.size(); ++i)
          Py_XDECREF(res_vec[i]);
        if(!leftBorder)
          Py_XDECREF(y_p);
        for(int i = 0; i < ARRAY_LEN(state_vars); ++i)
          Py_XDECREF(state_vars[i]);
      }

      float * d_ptr = rightBorder ? CudaNdarray_DEV_DATA(%(d)s) : 0;
      do_lstm(%(H)s, %(Y)s, %(c)s, d_ptr, y, x, %(i)s);
    }
    #undef ARRAY_LEN
    """ % locals()

  def grad(self, inputs, output_grads):
    (Z, c, y0, index, freq, W_re), input_rest = inputs[:6], inputs[6:]
    assert len(input_rest) == self._get_num_custom_vars() + self._get_num_state_vars()
    custom_inputs = input_rest[:self._get_num_custom_vars()]
    initial_state_vars = input_rest[self._get_num_custom_vars():]
    (DY, DH, Dd), seq_state_var_grads = output_grads[:3], output_grads[3:]
    assert len(seq_state_var_grads) == self._get_num_state_vars()

    Z_raw = Z.owner.inputs[0].owner.inputs[0]
    c_raw = c.owner.inputs[0].owner.inputs[0]
    y0_raw = y0.owner.inputs[0].owner.inputs[0]
    i_raw = index.owner.inputs[0].owner.inputs[0]
    W_re_raw = W_re.owner.inputs[0]
    custom_inputs_raw = [x.owner.inputs[0] for x in custom_inputs]
    #we have to make sure that this in only computed once!
    #for this we have to extract the raw variables before conversion to continuous gpu array
    #so that theano can merge the nodes
    all_out = self(*([Z_raw, c_raw, y0_raw, i_raw, freq, W_re_raw] + custom_inputs + initial_state_vars))
    (Y, H, d), seq_state_vars = all_out[:3], all_out[3:]

    assert isinstance(DH.type, theano.gradient.DisconnectedType)  # DH is ignored.
    if isinstance(DY.type, theano.gradient.DisconnectedType):
      DY = T.zeros_like(Z)
    if isinstance(Dd.type, theano.gradient.DisconnectedType):
      Dd = T.zeros_like(c)
    for i in range(len(seq_state_var_grads)):
      if isinstance(seq_state_var_grads[i].type, theano.gradient.DisconnectedType):
        # First dim for time. One more for -1 element.
        shape = [Z.shape[0] + 1] + [initial_state_vars[i].shape[d] for d in range(initial_state_vars[i].ndim)]
        seq_state_var_grads[i] = T.zeros(shape, dtype="float32")

    grad_op = grad_ops[(self.fun_name, id(self.recurrent_transform))]
    all_grads = grad_op(*([Y, H, c, y0, index, freq, Dd, DY, W_re] + custom_inputs + seq_state_var_grads))
    (DZ, Dc, Dy0, DW_re), remaining_grads = all_grads[:4], all_grads[4:]
    # remaining grads = custom_inputs grads + initial state var grads
    assert len(remaining_grads) == self._get_num_custom_vars() + self._get_num_state_vars()
    custom_input_grads = remaining_grads[:self._get_num_custom_vars()]
    initial_state_var_grads = remaining_grads[self._get_num_custom_vars():]
    Di = theano.gradient.grad_undefined(self, 3, inputs[3], 'cannot diff w.r.t. index')
    Dfreq = theano.gradient.grad_undefined(self, 4, inputs[4], 'cannot diff w.r.t. frequency')

    return [DZ, Dc, Dy0, Di, Dfreq, DW_re] + custom_input_grads + initial_state_var_grads


function_ops = {}; ":type: dict[(str,int),LSTMCustomOp]"
grad_ops = {}; ":type: dict[(str,int),LSTMCustomOpGrad]"

def register_func(recurrent_transform):
  """
  :type recurrent_transform: RecurrentTransform.RecurrentTransformBase
  """
  fn = recurrent_transform.name
  key = (fn, id(recurrent_transform))
  if key in function_ops:
    return function_ops[key]

  # register op
  no_inpl = LSTMCustomOp(fun_name=fn, inplace=False, recurrent_transform=recurrent_transform)
  inpl = LSTMCustomOp(fun_name=fn, inplace=True, recurrent_transform=recurrent_transform)
  function_ops[key] = no_inpl

  # hack to avoid being called twice
  attr = 'LSTMCustomMOpInplaceOpt_%s_%i' % (fn, id(recurrent_transform))
  if not hasattr(optdb, attr):
    opt = OpSub(no_inpl, inpl)
    optdb.register(attr, theano.gof.TopoOptimizer(opt),
                   50.0, 'fast_run', 'inplace', 'gpuarray')
    setattr(optdb, attr, True)

  # the same for grad
  no_inpl = LSTMCustomOpGrad(fun_name=fn, inplace=False, recurrent_transform=recurrent_transform)
  inpl = LSTMCustomOpGrad(fun_name=fn, inplace=True, recurrent_transform=recurrent_transform)
  grad_ops[key] = no_inpl

  # hack to avoid being called twice
  attr = 'LSTMCustomMOpGradInplaceOpt_%s_%i' % (fn, id(recurrent_transform))
  if not hasattr(optdb, attr):
    opt = OpSub(no_inpl, inpl)
    optdb.register(attr, theano.gof.TopoOptimizer(opt),
                   50.0, 'fast_run', 'inplace', 'gpuarray')
    setattr(optdb, attr, True)

  return function_ops[key]
