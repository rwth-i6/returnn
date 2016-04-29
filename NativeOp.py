
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
from TheanoUtil import try_register_gpu_opt, make_var_tuple


class NativeOp(theano.Op):
  """
  We wrap some C code which can define a forward pass
  and optionally a backward pass (for gradient calculation).
  The C code should be Numpy and CUDA compatible. See NativeOp.cpp.
  We also support inplace operations, i.e. we can operate inplace on some inputs.
  You can define in a flexible way all the inputs and the outputs.
  See __init__() for the details.

  All output variables are created automatically with the right shape
   but their content is not initialized,
   except when its used by some input variable as the inplace output
   - in that case, it is either the input variable or it has a copy of its data.
  """

  __props__ = ("in_info", "out_info",
               "c_fw_code", "c_bw_code", "c_extra_support_code", "code_version",
               "grad_input_map", "name")

  def __init__(self, in_info, out_info,
               c_fw_code, c_bw_code=None, c_extra_support_code=None, code_version=None,
               grad_input_map=None, name=None):
    """
    :param list[dict(str)] in_info: each dict describes one input var.
      attribs in the dict:
        int ndim: the ndim.
        tuple shape: tuple and can contain None for specific dimensions.
      optional attribs:
        str dtype: "float32" by default.
        bool need_contiguous: false by default.
        int want_inplace: -1 by default. try to optimize to destroy input, on output-index.
          "dummy_out" is a special value which will add another output.
        bool is_inplace: false by default. whether the optimization was applied.
        str gradient: can be "disconnected". see grad().
        bool bw_input: True by default. add this param to the bw input.
      other attribs are just ignored.
    :param list[dict(str)] out_info: like in_info.
      slightly different behavior for:
        shape: we also allow refs to the in_info in the form (in-idx,dim). see infer_shape().
        need_contiguous/want_inplace: used for bw, in case for bw_input == True.
    :param str c_fw_code: C code for forward pass
    :param str c_extra_support_code: C support code (for c_support_code)
    :param str|None c_bw_code: C code for backward pass (for gradient)
    :param tuple[int] code_version: will be returned by c_code_cache_version.
    :param tuple|callable grad_input_map: selection of grad inputs.
      by default, we get all inputs + all outputs + all grad outputs.
    :param str name: name
    """
    super(NativeOp, self).__init__()
    in_info, out_info, num_dummy_outs = self._resolve_want_inplace_dummy(in_info, out_info)
    self.in_info = make_hashable(in_info)
    self.out_info = make_hashable(out_info)
    self.num_dummy_outs = num_dummy_outs
    self.c_fw_code = c_fw_code
    self.c_bw_code = c_bw_code
    self.c_extra_support_code = self._reduce_c_extra_support_code(c_extra_support_code)
    self.code_version = code_version or ()
    self.name = name or "<anonNativeOp>"
    self.grad_input_map = self._convert_grad_input_map(grad_input_map, len(in_info) + len(out_info) * 2)
    self.destroy_map = self._construct_destroy_map(in_info)

  @classmethod
  def _resolve_want_inplace_dummy(cls, in_info, out_info):
    in_info = [dict(info) for info in in_info]  # deep copy, don't modify original
    out_info = list(out_info)  # copying list is enough here
    num_dummy_outs = 0
    for in_idx, info in enumerate(in_info):
      if info.get("want_inplace", None) == "dummy_out":
        num_dummy_outs += 1
        dummy_out_idx = len(out_info)
        dummy_out = {"ndim": info["ndim"],
                     "shape": [(in_idx, i) for i in range(info["ndim"])],
                     "dtype": info.get("dtype", "float32")}
        out_info += [dummy_out]
        info["want_inplace"] = dummy_out_idx
    return in_info, out_info, num_dummy_outs

  @classmethod
  def _reduce_c_extra_support_code(cls, c):
    if c is None:
      return ""
    if isinstance(c, dict):
      c = [v for (k, v) in sorted(c.items())]
    if isinstance(c, (list, tuple)):
      c = "\n".join([v + "\n\n" for v in c])
    assert isinstance(c, (str, unicode))
    return c

  @classmethod
  def _construct_destroy_map(cls, in_info):
    destroy_map = {}
    for in_idx, info in enumerate(in_info):
      want_inplace = info.get("want_inplace", -1)
      assert isinstance(want_inplace, (int, long))
      if want_inplace >= 0 and info.get("is_inplace", False):
        out_idx = want_inplace
        # http://deeplearning.net/software/theano/extending/inplace.html
        # https://github.com/Theano/Theano/issues/3506
        # It's strange that we must mark which output operates on which input -
        # I would expect that it must only know which inputs are destroyed.
        assert out_idx not in destroy_map, "Theano cannot handle that yet"
        destroy_map[out_idx] = [in_idx]
    return destroy_map

  @classmethod
  def _convert_grad_input_map(cls, gi_map, num_params):
    if gi_map is None:
      gi_map = tuple(range(num_params))
    if callable(gi_map):
      gi_map = gi_map(*range(num_params))
    if isinstance(gi_map, list):
      gi_map = tuple(gi_map)
    assert isinstance(gi_map, tuple)
    return gi_map

  def _filter_grad_inputs(self, inputs):
    assert len(inputs) == len(self.in_info) + len(self.out_info) * 2
    return [inputs[i] for i in self.grad_input_map]

  def __str__(self):
    return "%s{%s,%s}" % (
      self.__class__.__name__,
      self.name,
      "inplace" if self.destroy_map else "no_inplace")

  @classmethod
  def as_tensor_var(cls, v):
    return theano.tensor.as_tensor_variable(v)

  @classmethod
  def tensor_type(cls, dtype, ndim):
    return T.TensorType(dtype=dtype, broadcastable=(False,) * ndim)

  @classmethod
  def contiguous(cls, v):
    from TheanoUtil import Contiguous
    assert isinstance(v, theano.Variable)
    if getattr(v, 'owner', None):
      assert isinstance(v.owner, theano.Apply)
      if isinstance(v.owner.op, Contiguous.__base__):
        return v
    return Contiguous()(v)

  def _convert_input_var(self, v, info):
    v = T.cast(v, info.get("dtype", "float32"))
    v = self.as_tensor_var(v)
    if v.ndim != info["ndim"]:
      raise TypeError("input var ndim %i does not match with info %r" % (v.ndim, info))
    if info.get("need_contiguous", False):
      v = self.contiguous(v)
    return v

  def infer_shape(self, node, input_shapes):
    assert len(input_shapes) == len(self.in_info)
    out_shapes = []
    for info in self.out_info:
      out_shape = list(info["shape"])
      for idx, s in enumerate(out_shape):
        if isinstance(s, tuple):  # we interpret this as a reference to input shapes
          assert len(s) == 2, "dim %r invalid in info %r" % (s, info)
          assert 0 <= s[0] < len(input_shapes), "dim %r invalid in info %r" % (s, info)
          assert 0 <= s[1] < self.in_info[s[0]]["ndim"], "dim idx %r invalid in input %i %r, info %r" % (s[1], s[0], self.in_info[s[0]], info)
          out_shape[idx] = input_shapes[s[0]][s[1]]
      assert not any([s is None for s in out_shape]), "out_shape %r, out_info %r" % (out_shape, self.out_info)
      out_shapes += [tuple(out_shape)]
    return out_shapes

  @classmethod
  def _bw_in_var_info(cls, info):
    if "bw_in_var" in info:
      info = dict(info)
      info.update(info.pop("bw_in_var"))
    return info

  @classmethod
  def _bw_grad_var_info(cls, info):
    info = dict(info)
    if "bw_grad_var" in info:
      info.update(info.pop("bw_grad_var"))
    if "name" in info:
      info["name"] = "D_" + info["name"]
    return info

  def grad(self, inputs, output_grads):
    if not self.c_bw_code:
      # Unknown how to calculate gradient.
      return [T.DisconnectedType()() for inp in inputs]

    assert len(self.in_info) == len(inputs)
    assert len(self.out_info) == len(output_grads)

    # Some of output_grads might be of disconnected type.
    out_shapes = self.infer_shape(None, [v.shape for v in inputs])
    assert len(out_shapes) == len(output_grads)
    for i, out_grad in enumerate(output_grads):
      if isinstance(out_grad.type, T.DisconnectedType):
        output_grads[i] = T.zeros(out_shapes[i], dtype="float32")

    # Inputs: inputs + outputs + output_grads, where outputs = op(inputs),
    # i.e. we might reuse some of the calculation.
    grad_inputs = inputs + list(make_var_tuple(self(*inputs))) + output_grads
    grad_inputs = self._filter_grad_inputs(grad_inputs)
    in_info = list(self.in_info)
    in_info += [self._bw_in_var_info(info) for info in self.out_info]
    in_info += [self._bw_grad_var_info(info) for info in self.out_info]
    in_info = self._filter_grad_inputs(in_info)
    assert len(in_info) == len(grad_inputs)
    in_idx_rev = {v: k for (k, v) in enumerate(self.grad_input_map)}
    # Outputs: All like original inputs. Filter our the disconnected.
    out_info = [info.copy() for info in self.in_info]
    for idx, info in enumerate(out_info):
      info.pop("shape")
      if "bw_out_var" in info:
        info.update(info["bw_out_var"])
      if "shape" not in info:
        # Refer to input shapes. See infer_shape().
        info["shape"] = [(in_idx_rev[idx], i) for i in range(info["ndim"])]
    out_info = [info for info in out_info if info.get("gradient", "") != "disconnected"]

    grad_op = self.__class__(
      name="grad-of-%s" % self.name,
      in_info=in_info,
      out_info=out_info,
      c_fw_code=self.c_bw_code,
      c_extra_support_code=self.c_extra_support_code,
      code_version=self.code_version
    )
    input_grads = make_var_tuple(grad_op(*grad_inputs))
    if grad_op.num_dummy_outs > 0:
      input_grads = input_grads[:-grad_op.num_dummy_outs]  # remove any dummy outputs
    assert len(out_info) == len(input_grads)

    def print_fn(op, x):
      import numpy
      first = x[(0,) * x.ndim]
      stats = (first, x.shape, numpy.min(x), numpy.max(x), numpy.mean(x), numpy.std(x),
               numpy.isinf(x).any(), numpy.isnan(x).any())
      print(op.message, "first/shape/min/max/mean/std/any-inf/any-nan:", stats)
    #input_grads = [theano.printing.Print("in grad %i" % i, global_fn=print_fn)(v)
    #               for (i, v) in enumerate(input_grads)]

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
    pattern = [[info.get("gradient", "") != "disconnected"] * len(self.out_info)
               for info in self.in_info]
    return pattern

  def make_node(self, *args):
    assert len(args) == len(self.in_info)
    args = [self._convert_input_var(arg, info) for arg, info in zip(args, self.in_info)]
    outputs = [self.tensor_type(dtype=info.get("dtype", "float32"), ndim=info["ndim"])()
               for info in self.out_info]
    return theano.Apply(self, args, outputs)

  def perform(self, node, inputs, output_storage):
    raise NotImplementedError("NativeOp: no pure Python implementation, only C implementation")

  def c_code_cache_version(self):
    return self.code_version

  def c_support_code(self):
    base_src = open(os.path.dirname(__file__) + "/NativeOp.cpp").read()
    return "\n\n".join([
      T.blas.blas_header_text(),
      "#define CUDA 0",
      base_src,
      self.c_extra_support_code])

  def c_libraries(self):
    return T.blas.ldflags()

  def c_compile_args(self):
    return T.blas.ldflags(libs=False, flags=True)

  def c_lib_dirs(self):
    return T.blas.ldflags(libs=False, libs_dir=True)

  def c_header_dirs(self):
    return T.blas.ldflags(libs=False, include_dir=True)

  def c_code(self, node, name, inputs, outputs, sub):
    assert len(inputs) == len(self.in_info)
    assert len(outputs) == len(self.out_info)
    return """
    {
      int n_inputs = %(n_inputs)i, n_outputs = %(n_outputs)i;
      Ndarray* inputs[] = {%(input_var_names_str)s};
      Ndarray** outputs[] = {%(output_var_names_str)s};
      int in_ndims[] = {%(input_ndims_str)s};
      int out_ndims[] = {%(output_ndims_str)s};
      Ndarray_DIM_Type output_shapes_flat[] = {%(output_shapes_flat_str)s};
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
    }
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

  @classmethod
  def as_tensor_var(cls, v):
    from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable
    return as_cuda_ndarray_variable(v)

  @classmethod
  def tensor_type(cls, dtype, ndim):
    from theano.sandbox.cuda import CudaNdarrayType
    if dtype != "float32":
      print("%s: WARNING: cannot handle type %r, will use float32 instead" % ("GpuNativeOp", dtype))
      dtype = "float32"
    return CudaNdarrayType(dtype=dtype, broadcastable=(False,) * ndim)

  @classmethod
  def contiguous(cls, v):
    from theano.sandbox.cuda.basic_ops import gpu_contiguous
    assert isinstance(v, theano.sandbox.cuda.CudaNdarrayVariable)
    if getattr(v, 'owner', None):
      assert isinstance(v.owner, theano.Apply)
      if v.owner.op == gpu_contiguous:
        return v
    return gpu_contiguous(v)

  def c_support_code(self):
    src = open(os.path.dirname(__file__) + "/NativeOp.cpp").read()
    return "#define CUDA 1\n\n" + src


@gof.local_optimizer([NativeOp], inplace=True)
def inplace_NativeOp(node):
  if isinstance(node.op, NativeOp) and not node.op.destroy_map:
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
    from TheanoUtil import make_var_tuple
    new_v = make_var_tuple(new_op(*node.inputs))
    return new_v
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
      from TheanoUtil import make_var_tuple
      outputs = make_var_tuple(gpu_op(*args))
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
  c_extra_support_code = None
  code_version = None
  grad_input_map = None

  @classmethod
  def make_op(cls):
    name = cls.__name__
    assert cls.in_info is not None
    assert cls.out_info is not None
    assert cls.c_fw_code is not None
    return NativeOp(in_info=cls.in_info, out_info=cls.out_info,
                    c_fw_code=cls.c_fw_code, c_bw_code=cls.c_bw_code,
                    c_extra_support_code=cls.c_extra_support_code,
                    grad_input_map=cls.grad_input_map,
                    name=name)

  @classmethod
  def map_layer_inputs_to_op(cls, *inputs):
    return inputs

  @classmethod
  def map_layer_output_from_op(cls, *outputs):
    return outputs[0]


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
    {"name": "Z", "ndim": 3, "shape": (None, None, None), "need_contiguous": True,
     "want_inplace": 1,
     "bw_out_var": {"shape": ((2, 0), (2, 1), (0, 1))}},  # see grad_input_map() for indices
    {"name": "V_h", "ndim": 2, "shape": (None, None), "need_contiguous": True},
    {"name": "c", "ndim": 2, "shape": (None, None), "need_contiguous": True},
    {"name": "i", "ndim": 2, "shape": (None, None), "need_contiguous": True,
     "gradient": "disconnected"}
  )
  out_info = (
    {"name": "Y", "ndim": 3, "shape": ((0, 0), (0, 1), (1, 0)), "need_contiguous": True,
     "bw_grad_var": {"want_inplace": "dummy_out"}},
    {"name": "H", "ndim": 3, "shape": ((0, 0), (0, 1), (0, 2)), "need_contiguous": True,
     "bw_in_var": {"want_inplace": 0}},
    {"name": "d", "ndim": 2, "shape": ((2, 0), (2, 1)), "need_contiguous": True}
  )
  @classmethod
  def grad_input_map(cls, Z, V_h, c, i,  Y, H, d,  DY, DH, Dd):
    return (V_h, c, i,  Y, H,  DY, Dd)

  @classmethod
  def map_layer_inputs_to_op(cls, Z, V_h, i):
    assert Z.ndim == 3
    assert V_h.ndim == 2
    assert i.ndim == 2
    n_batch = Z.shape[1]
    n_out = V_h.shape[0]
    c = T.zeros((n_batch, n_out), dtype="float32")
    return Z, V_h, c, i

  c_extra_support_code = {
    "lstm_kernel": """
      DEF_KERNEL
      void lstm_kernel(float* data, const float* old_state, bool old_state_strided,
                       float* output, float* state_out, int n_cells, int n_batch, const float* i) {
        //layout:
        //data[0*n_cells..1*n_cells-1] : input gate
        //data[1*n_cells..2*n_cells-1] : forget gate
        //data[2*n_cells..3*n_cells-1] : output gate
        //data[3*n_cells..4*n_cells-1] : cell state
        //output[0*n_cells..1*n_cells-1]: cell output
        //repeated for every mini-batch

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        while (idx < n_cells * n_batch) {
          int batch_idx = idx / n_cells;
          int start = batch_idx * 4 * n_cells + idx % n_cells;
          float i_batch = i[batch_idx];

          //input, forget and output gates
          float inpGate = 1.f / (1.f + expf(-data[start + n_cells]));
          float fgtGate = 1.f / (1.f + expf(-data[start + 2 * n_cells]));
          float outGate = 1.f / (1.f + expf(-data[start + 3 * n_cells]));
          float state = inpGate * tanhf(data[start]);
          float old_state_batch = old_state_strided ? old_state[start] : old_state[idx];

          state += fgtGate * old_state_batch;
          state = state * i_batch + old_state_batch * (1.f - i_batch);

          //cell output
          output[idx] = outGate * tanhf(state) * i_batch;

          data[start] = state;
          data[start + n_cells] = inpGate;
          data[start + 2 * n_cells] = fgtGate;
          data[start + 3 * n_cells] = outGate;
          if(state_out)
            state_out[idx] = state;

          idx += gridDim.x * blockDim.x;
        }
      }
    """,
    "lstm_bwd_kernel": """
      DEF_KERNEL
      void lstm_bwd_kernel(
            float* delta, float* epsilon, const float* next_epsilon, const float* old_state,
            bool old_state_strided, const float* Y, int n_cells, int n_batch, const float* i) {
        //layout:
        //delta[0*n_cells..1*n_cells-1] : input gate
        //delta[1*n_cells..2*n_cells-1] : forget gate
        //delta[2*n_cells..3*n_cells-1] : output gate
        //delta[3*n_cells..4*n_cells-1] : cell state
        //epsilon[0*n_cells..1*n_cells-1]: cell output derivative (later overwritten, see below)
        //next_epsilon[0*n_cells..1*n_cells-1]: cell state derivative * forget_gate (of next timestep)
        //repeated for every mini-batch

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        while (idx < n_cells * n_batch) {
          int batch_idx = idx / n_cells;
          int batch_offset = batch_idx * 4 * n_cells;
          int cell_offset = idx % n_cells;
          int start = batch_offset + cell_offset;
          float i_batch = i[batch_idx];

          float inpGate = delta[start + n_cells];
          float fgtGate = delta[start + 2 * n_cells];
          float outGate = delta[start + 3 * n_cells];
          float oldState = old_state_strided ? old_state[start] : old_state[idx];
          float state = delta[start];
          float eps = epsilon[idx];

          //avoid division by 0 (TODO: check if this is needed)
          float gc = 0.f; //g(c(t))
          float gzc = 0.f; //g(z_c(t))
          if (outGate != 0)
              gc = Y[idx] / outGate;
          if (inpGate != 0)
              gzc = (state - fgtGate * oldState) / inpGate;

          //delta_output
          delta[start + 3 * n_cells] = outGate * (1.f - outGate) * gc * eps * i_batch;

          //epsilon_c
          float epsilon_c = (1.f - (gc * gc)) * outGate * eps;
          epsilon_c += next_epsilon[idx];
          epsilon[idx] = epsilon_c * fgtGate * i_batch + next_epsilon[idx] * (1.f - i_batch);

          //delta_cell
          delta[start] = inpGate * (1.f - (gzc * gzc)) * epsilon_c * i_batch;

          //delta_forget
          delta[start + 2 * n_cells] = fgtGate * (1.f - fgtGate) * oldState * epsilon_c * i_batch;

          //delta_input
          delta[start + n_cells] = inpGate * (1.f - inpGate) * gzc * epsilon_c * i_batch;

          idx += gridDim.x * blockDim.x;
        }
      }
      """
  }

  c_fw_code = """
    // Z*, V_h, c, i = input_names (*: inplace)
    // Y, H, d = output_names
    assert(n_inputs == 4);
    assert(n_outputs == 3);
    Ndarray* V_h = inputs[1];
    Ndarray* c = inputs[2];
    Ndarray* i = inputs[3];
    Ndarray* Y = *outputs[0];
    Ndarray* H = *outputs[1]; // inplace on Z
    Ndarray* d = *outputs[2];

    long T = Ndarray_DIMS(i)[0];
    int n_batch = Ndarray_DIMS(i)[1];
    assert(Ndarray_DIMS(H)[2] %% 4 == 0); // 3 gates + cell
    int n_cells = Ndarray_DIMS(H)[2] / 4;

    assert(T > 0);
    for(int x = 0; x < T; ++x) {
      if(x > 0) {
        //H += Y[x-1]*V_h
        affine_y_x(x-1, Y,  x, V_h,  x, H);
      }

      start_dev_kernel(lstm_kernel, (
        data_ptr(H, x),
        x > 0 ? data_ptr(H, x - 1) : Ndarray_DEV_DATA(c),
        x > 0,
        data_ptr(Y, x),
        (x == T - 1) ? Ndarray_DEV_DATA(d) : 0,
        n_cells,
        n_batch,
        Ndarray_DEV_DATA(i) + x * n_batch
      ));
    }
  """

  c_bw_code = """
    // V_h, c, i,   Y, H*,   DY*, Dd = input_names (*: inplace)
    // DZ, DV_h, Dc, tmpDc = output_names
    assert(n_inputs == 7);
    assert(n_outputs == 4);
    Ndarray* V_h = inputs[0];
    Ndarray* c = inputs[1];
    Ndarray* i = inputs[2];
    Ndarray* Y = inputs[3];
    Ndarray* Dd = inputs[6];
    Ndarray* DZ = *outputs[0]; // inplace on H
    Ndarray* DV_h = *outputs[1];
    Ndarray* Dc = *outputs[2];
    Ndarray* tmpDc = *outputs[3]; // (old DY), inplace buffer

    long T = Ndarray_DIMS(i)[0];
    int n_batch = Ndarray_DIMS(i)[1];
    assert(Ndarray_DIMS(DZ)[2] %% 4 == 0); // 3 gates + cell
    int n_cells = Ndarray_DIMS(DZ)[2] / 4;

    assert(T > 0);
    for(int x = T - 1; x >= 0; --x) {
      // add recurrent
      bool rightBorder = (x == T - 1);
      if(!rightBorder)
        affine_y_x(x+1, DZ,  x, V_h,  x, tmpDc,  false, true);

      start_dev_kernel(lstm_bwd_kernel, (
        data_ptr(DZ, x),
        data_ptr(tmpDc, x),
        rightBorder ? Ndarray_DEV_DATA(Dd) : data_ptr(tmpDc, x + 1),
        x > 0 ? data_ptr(DZ, x - 1) : Ndarray_DEV_DATA(c),
        x > 0,
        data_ptr(Y, x),
        n_cells,
        n_batch,
        Ndarray_DEV_DATA(i) + x * n_batch
      ));
    }

    //DV_h = Y[0..end-1]^T * DZ[1..end]
    affine_global(Y, DZ, DV_h, true, false, 1, 0.0f);

    const Ndarray_DIM_Type* Dc_dim = Ndarray_HOST_DIMS(Dc);
    Ndarray_memcpy(
      Ndarray_DEV_DATA(Dc), Ndarray_DEV_DATA(tmpDc),
      Dc_dim[0] * Dc_dim[1] * sizeof(float));
  """

  code_version = ()
