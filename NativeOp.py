
"""
Generic interface which automatically creates:
* CPU and GPU op
* inplace and not inplace
* grad variants
"""

import os
import numpy
import theano
import theano.sandbox.cuda
import theano.tensor as T
from theano.gof.opt import OpSub
from theano.compile import optdb
from theano import gof
from Util import make_hashable, make_dll_name, escape_c_str
from TheanoUtil import try_register_gpu_opt


class NativeOp(theano.Op):
  __props__ = ("in_info", "out_info", "c_fw_code", "c_bw_code", "code_version", "name")

  def __init__(self, in_info, out_info, c_fw_code, c_bw_code=None, code_version=None, name=None):
    """
    :param list[dict(str)] in_info: each dict describes one input var.
      attribs in the dict:
        int ndim: the ndim.
        tuple shape: tuple and can contain None for specific dimensions.
      optional attribs:
        str dtype: "float32" by default.
        bool need_contiguous: false by default.
        int want_inplace: -1 by default. try to optimize to destroy input, on output-index.
        bool is_inplace: false by default. whether the optimization was applied.
        str gradient: can be "disconnected". see grad().
        bool bw_input: True by default. add this param to the bw input.
      other attribs are just ignored.
    :param list[dict(str)] out_info: like in_info.
      slightly different behavior for:
        shape: we also allow refs to the in_info in the form (in-idx,dim). see infer_shape().
        need_contiguous/want_inplace: used for bw, in case for bw_input == True.
    :param str c_fw_code: C code for forward pass
    :param str|None c_bw_code: C code for backward pass (for gradient)
    :param tuple[int] code_version: will be returned by c_code_cache_version.
    :param str name: name
    """
    super(NativeOp, self).__init__()
    self.in_info = make_hashable(in_info)
    self.out_info = make_hashable(out_info)
    self.c_fw_code = c_fw_code
    self.c_bw_code = c_bw_code
    self.code_version = code_version or ()
    self.name = name or "<anonNativeOp>"
    self.destroy_map = {}
    for in_idx, info in enumerate(self.in_info):
      if info.get("want_inplace", -1) >= 0 and info.get("is_inplace", False):
        out_idx = info["want_inplace"]
        # http://deeplearning.net/software/theano/extending/inplace.html
        # https://github.com/Theano/Theano/issues/3506
        # It's strange that we must mark which output operates on which input -
        # I would expect that it must only know which inputs are destroyed.
        assert out_idx not in self.destroy_map, "Theano cannot handle that yet"
        self.destroy_map[out_idx] = [in_idx]

  def __str__(self):
    return "%s{%s,%s}" % (
      self.__class__.__name__,
      self.name,
      "inplace" if self.destroy_map else "no_inplace")

  def as_tensor_var(self, v):
    return theano.tensor.as_tensor_variable(v)

  def contiguous(self, v):
    from theano.tensor.extra_ops import cpu_contiguous
    assert isinstance(v, theano.Variable)
    if getattr(v, 'owner', None):
      assert isinstance(v.owner, theano.Apply)
      if v.owner == cpu_contiguous:
        return v
    return cpu_contiguous(v)

  def _convert_input_var(self, v, info):
    v = T.cast(v, info.get("dtype", "float32"))
    v = self.as_tensor_var(v)
    if info.get("need_contiguous", False):
      v = self.contiguous(v)
    return v

  def infer_shape(self, node, input_shapes):
    out_shapes = []
    for info in self.out_info:
      out_shape = list(info["shape"])
      for idx, s in enumerate(out_shape):
        if isinstance(s, tuple):  # we interpret this as a reference to input shapes
          assert len(s) == 2
          out_shape[idx] = input_shapes[s[0]][s[1]]
      assert not any([s is None for s in out_shape]), "out_shape %r, out_info %r" % (out_shape, self.out_info)
      out_shapes += [tuple(out_shape)]
    return out_shapes

  def grad(self, inputs, output_grads):
    if not self.c_bw_code:
      # Unknown how to calculate gradient.
      return [T.DisconnectedType()() for inp in inputs]

    assert len(self.in_info) == len(inputs)
    assert len(self.out_info) == len(output_grads)
    out_info = [info.copy() for info in self.in_info]
    for idx, info in enumerate(out_info):
      # Refer to input shapes. See infer_shape().
      info["shape"] = [(idx, i) for i in range(info["ndim"])]
    out_info = [info for info in out_info if info.get("gradient", "") != "disconnected"]

    grad_op = self.__class__(
      name="grad-of-%s" % self.name,
      in_info=self.in_info + self.out_info,  # inputs + output_grads
      out_info=out_info,
      c_fw_code=self.c_bw_code
    )
    input_grads = grad_op(*(inputs + output_grads))
    assert len(out_info) == len(input_grads)

    results = []
    for info in self.in_info:
      if info.get("gradient", "") == "disconnected":
        results += [T.DisconnectedType()()]
      else:
        results += input_grads[:1]
        input_grads = input_grads[1:]
    assert len(input_grads) == 0
    assert len(results) == len(self.in_info)
    return results

  def connection_pattern(self, node):
    assert len(node.inputs) == len(self.in_info)
    pattern = [[info.get("gradient", "") != "disconnected"]
               for info in self.in_info]
    return pattern

  def make_node(self, *args):
    assert len(args) == len(self.in_info)
    args = [self._convert_input_var(arg, info) for arg, info in zip(args, self.in_info)]
    outputs = [T.TensorType(info.get("dtype", "float32"), (False,) * info["ndim"])()
               for info in self.out_info]
    return theano.Apply(self, args, outputs)

  def perform(self, node, inputs, output_storage):
    raise NotImplementedError("NativeOp: no pure Python implementation, only C implementation")

  def c_code_cache_version(self):
    return self.code_version

  def c_support_code(self):
    src = open(os.path.dirname(__file__) + "/NativeOp.cpp").read()
    return "#define CUDA 0\n\n" + src

  def c_code(self, node, name, inputs, outputs, sub):
    assert len(inputs) == len(self.in_info)
    assert len(outputs) == len(self.out_info)
    return """
      int n_inputs = %(n_inputs)i, n_outputs = %(n_outputs)i;
      Ndarray* inputs[] = {%(input_var_names_str)s};
      Ndarray** outputs[] = {%(output_var_names_str)s};
      int in_ndims[] = {%(input_ndims_str)s};
      int out_ndims[] = {%(output_ndims_str)s};
      int output_shapes_flat[] = {%(output_shapes_flat_str)s};
      int in_want_inplace[] = {%(input_want_inplace_str)s};
      bool in_is_inplace[] = {%(input_is_inplace_str)s};

      // Check if we can reuse any preallocated output.
      // Reset those which we cannot reuse.
      {
        int out_shape_idx = 0;
        for(int i = 0; i < n_outputs; ++i) {
          assert(out_shape_idx + out_ndims[i] <= ARRAY_LEN(output_shapes_flat));
          if(*outputs[i]) {
            bool can_reuse = true;
            for(int j = 0; j < out_ndims[i]; ++j)
              if(output_shapes_flat[out_shape_idx + j] != Ndarray_DIMS(*outputs[i])[j]) {
                can_reuse = false;
                break;
              }
            if(!can_reuse)
              Py_DECREF(*outputs[i]);
          }
          out_shape_idx += out_ndims[i];
        }
        assert(out_shape_idx == ARRAY_LEN(output_shapes_flat));
      }

      // Maybe reuse or otherwise copy input into output vars.
      for(int i = 0; i < n_inputs; ++i)
        if(in_want_inplace[i] >= 0) {
          assert(in_want_inplace[i] < n_outputs);
          Py_XDECREF(*outputs[in_want_inplace[i]]);
          if(in_is_inplace[i]) {
            *outputs[in_want_inplace[i]] = inputs[i];
            Py_INCREF(inputs[i]);
          } else {
            *outputs[in_want_inplace[i]] = (Ndarray*) Ndarray_Copy(inputs[i]);
            if(!*outputs[in_want_inplace[i]]) %(fail)s;
            inputs[i] = *outputs[in_want_inplace[i]];  // reset with copy
          }
        }

      // Init the remaining output vars. Note that they are initialized randomly!
      {
        int out_shape_idx = 0;
        for(int i = 0; i < n_outputs; ++i) {
          assert(out_shape_idx + out_ndims[i] <= ARRAY_LEN(output_shapes_flat));
          if(*outputs[i]) {
            for(int j = 0; j < out_ndims[i]; ++j)
              assert(output_shapes_flat[out_shape_idx + j] == Ndarray_DIMS(*outputs[i])[j]);
          }
          else {
            *outputs[i] = (Ndarray*) Ndarray_NewDims(out_ndims[i], &output_shapes_flat[out_shape_idx]);
            if(!*outputs[i]) %(fail)s;
          }
          out_shape_idx += out_ndims[i];
        }
        assert(out_shape_idx == ARRAY_LEN(output_shapes_flat));
      }

      // And the user C code starts here.
      // --------------------------------
      %(c_code)s;
    """ % {
      'name': name, 'fail': sub['fail'],
      'op_name': escape_c_str(self.name),
      'c_code': self.c_fw_code % {'fail': sub['fail']},
      'n_inputs': len(inputs), 'n_outputs': len(outputs),
      'input_var_names_str': ", ".join(["%s" % inp for inp in inputs]),
      'output_var_names_str': ", ".join(["&%s" % out for out in outputs]),
      'input_ndims_str': ', '.join(["%i" % info["ndim"] for info in self.in_info]),
      'output_ndims_str': ', '.join(["%i" % info["ndim"] for info in self.out_info]),
      'output_shapes_flat_str':
        ', '.join([(("%i" % s) if isinstance(s, (int, long))
                    else "Ndarray_DIMS(inputs[%i])[%i]" % s)
                   for info in self.out_info for s in info["shape"]]),
      "input_want_inplace_str": ", ".join([str(int(info.get("want_inplace", -1)))
                                           for info in self.in_info]),
      "input_is_inplace_str": ", ".join([str(int(info.get("is_inplace", False)))
                                         for info in self.in_info])
    }


class GpuNativeOp(NativeOp, theano.sandbox.cuda.GpuOp):

  def as_tensor_var(self, v):
    from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable
    return as_cuda_ndarray_variable(v)

  def contiguous(self, v):
    from theano.sandbox.cuda.basic_ops import gpu_contiguous
    assert isinstance(v, theano.sandbox.cuda.CudaNdarrayVariable)
    if getattr(v, 'owner', None):
      assert isinstance(v.owner, theano.Apply)
      if v.owner == gpu_contiguous:
        return v
    return gpu_contiguous(v)

  def c_support_code(self):
    src = open(os.path.dirname(__file__) + "/NativeOp.cpp").read()
    return "#define CUDA 1\n\n" + src


@gof.local_optimizer([NativeOp], inplace=True)
def inplace_NativeOp(node):
  if isinstance(node.op, NativeOp) and not node.op.inplace and not node.op.zero_with_shape:
    kwargs = {k: getattr(node.op, k) for k in node.op.__props__}
    # TODO: We could try to make each input inplace individually.
    # What we do now is just to try to make all inplace.
    kwargs["in_info"] = [dict(info) for info in node.op.in_info]
    any_inplace = False
    for info in kwargs["in_info"]:
      if info.get("want_inplace", -1) >= 0:
        any_inplace = True
        info["is_inplace"] = True
    if not any_inplace:
      return False
    new_op = node.op.__class__(**kwargs)
    new_v = new_op(*node.inputs)
    return [new_v]
  return False

optdb.register('inplace_NativeOp',
               gof.TopoOptimizer(inplace_NativeOp
                                 , failure_callback=gof.TopoOptimizer.warn_inplace
                                 ),
               60, 'fast_run', 'inplace')


@try_register_gpu_opt(NativeOp)
def local_gpu_NativeOp(node):
  if isinstance(node.op, NativeOp):
    # see also: https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/opt.py
    from theano.sandbox.cuda import host_from_gpu, gpu_from_host, as_cuda_ndarray_variable
    args = node.inputs
    if any([(x.owner and x.owner.op == host_from_gpu) for x in args]):
      gpu_op = GpuNativeOp(**{key: getattr(node.op, key) for key in node.op.__props__})
      args = [x.owner.inputs[0] if (x.owner and x.owner.op == host_from_gpu) else x
              for x in args]
      outputs = gpu_op(*args)
      if not isinstance(outputs, tuple):
        outputs = [outputs]
      return [host_from_gpu(out) for out in outputs]


class NativeOpGenBase:
  """
  Base interface for op generation.
  See NativeOp.__init__() for attribs.
  """
  in_info = None
  out_info = None
  c_fw_code = None
  c_bw_code = None
  code_version = None

  def make_op(self):
    name = self.__class__.__name__
    assert self.in_info is not None
    assert self.out_info is not None
    assert self.c_fw_code is not None
    return NativeOp(in_info=self.in_info, out_info=self.out_info,
                    c_fw_code=self.c_fw_code, c_bw_code=self.c_bw_code,
                    name=name)


class LstmGenericBase(NativeOpGenBase):
  """
  inputs:
    :param Z: {input,output,forget} gate + cell state. 3d (time,batch,dim*4)
    :param V_h: recurrent matrix. 2d (dim,dim*4)
    :param c: initial cell state. 2d (batch,dim)
    :param i: index. 2d (time,batch) -> 0 or 1
  outputs:
    :param Y: output. 3d (time,batch,dim)
    :param H: gates and cell state. 3d (time,batch,dim*4)
    :param d: final cell state. 2d (batch,dim)
  """
  in_info = (
    {"name": "Z", "ndim": 3, "shape": (None, None, None),
     "need_contiguous": True, "want_inplace": 1, "bw_input": False},
    {"name": "V_h", "ndim": 2, "shape": (None, None)},
    {"name": "c", "ndim": 2, "shape": (None, None)},
    {"name": "i", "ndim": 2, "shape": (None, None), "gradient": "disconnected"}
  )
  out_info = (
    {"name": "Y", "ndim": 3, "shape": ((0, 0), (0, 1), (1, 0))},
    {"name": "H", "ndim": 3, "shape": ((0, 0), (0, 1), (0, 2))},
    {"name": "d", "ndim": 2, "shape": ((2, 0), (2, 1))}
  )
  c_fw_code = """
    // Z, V_h, c, i = input_names
    // Y, H, d = output_names
    assert(n_inputs == 4);
    assert(n_outputs == 3);
    Ndarray* V_h = inputs[1];
    Ndarray* c = inputs[2];
    Ndarray* i = inputs[3];
    Ndarray* Y = *outputs[0];
    Ndarray* H = *outputs[1];
    Ndarray* d = *outputs[2];

    int T = Ndarray_DIMS(inputs[0])[0];
    assert(T > 0);
    for(int x = 0; x < T; ++x) {
      if(x > 0) {
        //H += Y[x-1]*V_h
        affine_y_x(/*y*/0, x-1, Y, /*y*/0, x, V_h, /*y*/0, x, H);
      }
      float* d_ptr = (x == T - 1) ? Ndarray_DEV_DATA(d) : 0;
      do_lstm(H, Y, c, d_ptr, /*y*/0, x, i);
    }
  """
  c_bw_code = """
    // V_h, c, i,   Y, H, d,   DY, DH, Dd = input_names
    // DZ, DV_h, Dc = output_names
    assert(n_inputs == 9);
    assert(n_outputs == 4);
    Ndarray* V_h = inputs[0];
    Ndarray* c = inputs[1];
    Ndarray* i = inputs[2];
    Ndarray* Y = inputs[3];
    Ndarray* H = inputs[4];
    Ndarray* Dd = inputs[8];
    Ndarray* DZ = *outputs[0];
    Ndarray* DV_h = *outputs[1];
    Ndarray* Dc = *outputs[2];

    int T = Ndarray_DIMS(Y)[0];
    assert(T > 0);
    for(int x = T - 1; x >= 0; --x) {
      // add recurrent
      bool rightBorder = (x == T - 1);
      if(!rightBorder)
        affine_y_x(/*y*/0, x+1, DZ, /*y*/0, x, V_h, /*y*/0, x, DY, false, true);
      do_lstm_bwd(DZ, DY, Y, Dd, c, /*y*/0, x, rightBorder, i);
    }

    //DV_h = Y[0..end-1]^T * delta[1..end]
    affine_global(Y, DZ, DV_h, true, false, 1, 0.0f);

    const int* Dc_dim = Ndarray_HOST_DIMS(Dc);
    Ndarray_memcpy(
      Ndarray_DEV_DATA(Dc), Ndarray_DEV_DATA(DY),
      Dc_dim[0] * Dc_dim[1] * sizeof(float));
  """
  code_version = (1, 1)
