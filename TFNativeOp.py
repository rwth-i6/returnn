
from __future__ import print_function

import os
import tensorflow as tf
from threading import RLock

import NativeOp
import TFUtil
from Util import camel_case_to_snake_case


class OpDescription(NativeOp.NativeOpBaseMixin):
  @classmethod
  def from_gen_base(cls, gen_base):
    """
    :param NativeOp.NativeOpGenBase|Type[NativeOp.NativeOpGenBase] gen_base:
    :rtype: OpDescription
    """
    name = gen_base.__name__
    assert gen_base.in_info is not None
    assert gen_base.out_info is not None
    assert gen_base.c_fw_code is not None
    assert gen_base.custom_grad is None  # not supported for TF currently
    return OpDescription(
      in_info=gen_base.in_info, out_info=gen_base.out_info,
      c_fw_code=gen_base.c_fw_code, c_bw_code=gen_base.c_bw_code,
      c_extra_support_code=gen_base.c_extra_support_code,
      cpu_support=gen_base.cpu_support,
      grad_input_map=gen_base.grad_input_map,
      name=name)

  @property
  def is_grad_defined(self):
    return bool(self.c_bw_code)

  def grad(self):
    """
    :rtype: OpDescription|None
    """
    if not self.is_grad_defined:
      return None
    kwargs = self.kwargs_for_grad_op()
    return OpDescription(**kwargs)


class OpMaker(object):
  """
  https://www.tensorflow.org/versions/master/how_tos/adding_an_op/
  """
  with_cuda = None  # type: None|bool
  global_lock = RLock()
  mod_cache = {}  # cache_key -> mod
  op_cache = {}  # cache_key -> op

  def __init__(self, description, compiler_opts=None):
    """
    :param OpDescription description:
    :param dict[str]|None compiler_opts: passed on to OpCodeCompiler as kwargs
    """
    self._cls_init()
    self.description = description
    self.name = description.name
    self.compiler_opts = compiler_opts or {}

  @classmethod
  def _cls_init(cls):
    if cls.with_cuda is None:
      cls.with_cuda = TFUtil.CudaEnv.get_instance().is_available()
      if cls.with_cuda:
        cls._load_cuda_blas_gemm()

  @classmethod
  def _load_cuda_blas_gemm(cls):
    """
    https://github.com/tensorflow/tensorflow/issues/6602
    As a workaround for TF issue 6602, we link to some functions which are implemented in contrib.rnn.kernels.blas_gemm.
    See NativeOp.cpp.
    To make the symbols available in the namespace, load the library now.
    """
    if TFUtil.CudaEnv.verbose_find_cuda:
      print("Load tf.contrib lstm_ops...")
    from tensorflow.contrib.rnn.python.ops import lstm_ops
    lstm_ops_so = "%s/_lstm_ops.so" % os.path.dirname(lstm_ops.__file__)
    assert os.path.exists(lstm_ops_so)
    if TFUtil.CudaEnv.verbose_find_cuda:
      print("Load tf.contrib lstm_ops lib:", lstm_ops_so)
    # Maybe a bit hacky: Just load all symbols into the global namespace.
    from ctypes import RTLD_GLOBAL, CDLL
    CDLL(lstm_ops_so, mode=RTLD_GLOBAL)
    if TFUtil.CudaEnv.verbose_find_cuda:
      print("tf.contrib lstm_ops lib loaded.")

  @property
  def op_name(self):
    return self.name

  @property
  def cache_key(self):
    return self.name

  @property
  def support_native_op_cpp_filename(self):
    my_dir = os.path.abspath(os.path.dirname(__file__) or os.getcwd())
    my_dir = os.path.realpath(my_dir)  # Make canonical path-name.
    support_native_op_cpp_filename = "%s/NativeOp.cpp" % my_dir
    assert os.path.exists(support_native_op_cpp_filename)
    return support_native_op_cpp_filename

  def _make_code(self):
    # In the user code, we assume that we have the following variables:
    # int n_inputs; int n_outputs;
    # Ndarray* inputs[n_inputs]; Ndarray** outputs[n_outputs];
    # Reference:
    # https://www.tensorflow.org/extend/adding_an_op
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/how_tos/adding_an_op/
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_kernel.h
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def_builder.h
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/pad_op.cc
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/debug_ops.h  CopyOp...
    # http://stackoverflow.com/questions/37565367/designing-an-accumulating-tensorflow-gpu-operator
    # We also include NativeOp.cpp.
    in_info, out_info, _ = NativeOp.NativeOp._resolve_want_inplace_dummy(
      in_info=self.description.in_info, out_info=self.description.out_info)
    out_is_ref = dict()  # output vars which are inplace, out_name -> in_idx
    # want_inplace: output-index which this input should operate on
    # Unlike the Theano variant, we always do it inplace,
    # so the user has to make a copy if this is not the intention.
    for in_idx, v in enumerate(in_info):
      out_idx = v.get("want_inplace", -1)
      if out_idx >= 0:
        out_name = out_info[out_idx]["name"]
        assert out_name not in out_is_ref
        out_is_ref[out_name] = in_idx
    def map_name(v, is_out=False):
      name = v["name"].lower()
      if is_out:
        # Maybe it clashes with some input name. TF doesn't allow the same name.
        if any([v["name"].lower() == name for v in in_info]):
          name = "out_%s" % name
      return name
    def map_type(v, is_out=False):
      t = v.get("dtype", "float32")
      return t
    code_register_op_io = ""
    for v in in_info:
      code_register_op_io += ".Input(\"%s: %s\")\n" % (map_name(v), map_type(v))
    for v in out_info:
      code_register_op_io += ".Output(\"%s: %s\")\n" % (map_name(v, is_out=True), map_type(v, is_out=True))
    code_set_out_shape = ""
    def make_dim_str(c):
      if isinstance(c, tuple):
        in_idx, in_dim = c
        return "c->Dim(c->input(%i), %i)" % (in_idx, in_dim)
      elif isinstance(c, int):
        return str(c)
      else:
        raise Exception("type: %s" % type(c))
    for i, v in enumerate(in_info):
      code_set_out_shape += """
      if(c->Rank(c->input(%(idx)i)) != tensorflow::shape_inference::InferenceContext::kUnknownRank && c->Rank(c->input(%(idx)i)) != %(rank)i)
        return errors::InvalidArgument(
          "wrong rank for input (%(idx)i) '%(name)s'. required %(rank)i but got ", c->Rank(c->input(%(idx)i)));
      """ % {"idx": i, "rank": v["ndim"], "name": v["name"]}
    for i, v in enumerate(out_info):
      code_set_out_shape += "c->set_output(%i, c->MakeShape({%s}));\n" % (
        i, ", ".join([make_dim_str(c) for c in v["shape"]]))
    code_register_op_io += """
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      if(c->num_inputs() != %(num_inputs)i)
        return errors::InvalidArgument("wrong number of inputs. required %(num_inputs)i but got ", c->num_inputs());
      if(c->num_outputs() != %(num_outputs)i)
        return errors::InvalidArgument("wrong number of outputs. required %(num_outputs)i but got ", c->num_outputs());
      %(code_set_out_shape)s
      return Status::OK();
    })
    """ % {
      "num_inputs": len(in_info),
      "num_outputs": len(out_info),
      "code_set_out_shape": code_set_out_shape
    }
    code_forward_io = ""
    for in_idx, v in enumerate(in_info):
      out_idx = v.get("want_inplace", -1)
      if out_idx >= 0:
        code_forward_io += "context->forward_ref_input_to_ref_output(%i, %i);\n" % (in_idx, out_idx)
    code_set_io = ""
    for in_idx, v in enumerate(in_info):
      ndim = len(v["shape"])
      code_set_io += """
      OP_REQUIRES(
        context, context->input(%i).dims() == %i,
        errors::InvalidArgument("shape ndim is not %i, got shape ",
                                context->input(%i).shape().DebugString()));
      """ % (in_idx, ndim, ndim, in_idx)
      for axis, d in enumerate(v["shape"]):
        if isinstance(d, int):
          code_set_io += """
          OP_REQUIRES(
            context, context->input(%i).dim_size(%i) == %i,
            errors::InvalidArgument("shape[%i] != %i, got shape ",
                                    context->input(%i).shape().DebugString()));
          """ % (in_idx, axis, d, axis, d, in_idx)
    code_set_io += """
    Ndarray* inputs[n_inputs];
    Ndarray** outputs[n_outputs];
    """
    for in_idx, v in enumerate(in_info):
      out_idx = v.get("want_inplace", -1)
      if out_idx >= 0:  # is ref
        # mutable_input if it is a ref-type, i.e. a Variable.
        #code_set_io += "Ndarray mutable_input_%i = context->mutable_input(%i, false);\n" % (in_idx, in_idx)
        #code_set_io += "inputs[%i] = &mutable_input_%i;\n" % (in_idx, in_idx)
        # Maybe we could use a TemporaryVariable or so but not sure if the gradient will flow through tf.assign().
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/state_ops.cc
        # but a normal tensor is never mutable, thus create a copy of the input now.
        code_set_io += "Ndarray* output_%i = NULL;\n" % (out_idx,)
        cshape = "TensorShape({%s})" % ", ".join(["context->input(%i).dim_size(%i)" % (in_idx, in_dim)
                                                  for in_dim in range(len(v["shape"]))])
        code_set_io += "OP_REQUIRES_OK(context, context->allocate_output(%i, %s, &output_%i));\n" % (out_idx, cshape, out_idx)
        code_set_io += "inputs[%i] = output_%i;\n" % (in_idx, out_idx)
        # We always make a copy for now.
        # I'm not sure if inplace is an option for TF because we don't know if any other operation in the graph
        # wants to access it. Maybe we can check the reference count or so?
        # Some references for inplace operations:
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/inplace_ops.cc
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/strided_slice_op.cc
        code_set_io += "make_copy(context, inputs[%i], &context->input(%i));\n" % (in_idx, in_idx)
      else:  # no ref
        # TODO: if not on GPU but GPU requested, move to GPU first, maybe via allocate_temp?
        code_set_io += "inputs[%i] = const_cast<Ndarray*>(&context->input(%i));\n" % (in_idx, in_idx)
    for out_idx, v in enumerate(out_info):
      out_name = out_info[out_idx]["name"]
      if out_name in out_is_ref:  # is ref on input
        in_idx = out_is_ref[out_name]
        code_set_io += "outputs[%i] = &inputs[%i];\n" % (out_idx, in_idx)
      else:  # no ref
        code_set_io += "Ndarray* output_%i = NULL;\n" % (out_idx,)
        code_set_io += "outputs[%i] = &output_%i;\n" % (out_idx, out_idx)
        cshape = "TensorShape({%s})" % ", ".join(["inputs[%i]->dim_size(%i)" % (in_idx, in_dim)
                                                  for (in_idx, in_dim) in v["shape"]])
        code_set_io += "OP_REQUIRES_OK(context, context->allocate_output(%i, %s, &output_%i));\n" % (out_idx, cshape, out_idx)
        code_set_io += "Ndarray_set_zero(*outputs[%i]);\n" % out_idx

    code_user = self.description.c_fw_code % {"fail": "assert(false);"}
    code_compute = "\n".join([
      code_forward_io,
      code_set_io,
      code_user])
    register_gpu_kernel_opts = ".Device(DEVICE_GPU)\n"
    for v in in_info:
      if v.get("host_memory", False):
        register_gpu_kernel_opts += """.HostMemory("%s")\n""" % map_name(v)
    format_args = {
      "op_name": self.op_name,
      "code_register_op_io": code_register_op_io,
      "code_forward_io": code_forward_io,
      "code_set_io": code_set_io,
      "code_compute": code_compute,
      "user_code_kernels": self.description._reduce_c_extra_support_code(self.description.c_extra_support_code),
      "native_op_cpp_filename": self.support_native_op_cpp_filename,
      "register_gpu_kernel_opts": register_gpu_kernel_opts,
      "n_inputs": len(in_info),
      "n_outputs": len(out_info)
    }
    code_header = ""
    if self.with_cuda:
      code_header += """
      // For Eigen::GpuDevice.
      #define EIGEN_USE_GPU 1
      """
    code_header += """
    // For Eigen::ThreadPoolDevice.
    #define EIGEN_USE_THREADS 1

    #include "tensorflow/core/framework/op.h"
    #include "tensorflow/core/framework/shape_inference.h"
    #include "tensorflow/core/framework/op_kernel.h"
    #include "tensorflow/core/common_runtime/device.h"
    """
    if self.with_cuda:
      # http://docs.nvidia.com/cuda/cublas
      code_header += """
      #include <cuda.h>
      #include <cuda_runtime.h>
      #include <cublas_v2.h>
      #include <math_constants.h>

      // https://github.com/tensorflow/tensorflow/issues/6602 ?
      //#include "tensorflow/core/platform/stream_executor.h"
      //#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
      //#include "tensorflow/core/common_runtime/gpu_device_context.h"
      """
    # sgemm
    code_header += """
    typedef float real;
    typedef int integer;
    extern "C" {
    extern int sgemm_(char *transa, char *transb,
      integer *m, integer *n, integer *k,
      const real *alpha,
      const real *a, integer *lda,
      const real *b, integer *ldb,
      const real *beta,
      real *c, integer *ldc);
    }
    """
    code_header += """
    using namespace tensorflow;

    #define _ns  // so _ns::something will use the root namespace
    #define TENSORFLOW 1
    #define CUDA 0
    #include "%(native_op_cpp_filename)s"

    static const int n_inputs = %(n_inputs)i, n_outputs = %(n_outputs)i;

    REGISTER_OP("%(op_name)s")
    %(code_register_op_io)s;
    """ % format_args
    if self.description.cpu_support:
      code_cpu_op = """
      %(user_code_kernels)s
  
      class %(op_name)sOp : public OpKernel {
      public:
        explicit %(op_name)sOp(OpKernelConstruction* context) : OpKernel(context) {}
        void Compute(OpKernelContext* context) override {
          %(code_compute)s
        }
      };
  
      REGISTER_KERNEL_BUILDER(Name("%(op_name)s").Device(DEVICE_CPU), %(op_name)sOp);
      """ % format_args
    else:
      code_cpu_op = ""
    if self.with_cuda:
      code_gpu_op = """
      namespace _gpu {
        #ifdef _ns
          #undef _ns
        #endif
        namespace _ns = ::_gpu;
        #undef CUDA
        #define CUDA 1
        #undef Ndarray_memcpy
        #undef Ndarray_memset
        #undef Ndarray_sgemm
        #undef DEF_KERNEL
        #undef start_dev_kernel
        #undef assert_cmp
        #undef threadIdx
        #undef blockIdx
        #undef blockDim
        #undef gridDim
        #include "%(native_op_cpp_filename)s"

        %(user_code_kernels)s

        class %(op_name)sGpuOp : public OpKernel {
        public:
          explicit %(op_name)sGpuOp(OpKernelConstruction* context) : OpKernel(context) {}
          void Compute(OpKernelContext* context) override {
            %(code_compute)s
          }
        };

        REGISTER_KERNEL_BUILDER(
          Name("%(op_name)s")
          %(register_gpu_kernel_opts)s,
          %(op_name)sGpuOp);
      }
      """ % format_args
    else:
      code_gpu_op = ""
    return code_header + code_cpu_op + code_gpu_op

  def _make_mod(self):
    if self.cache_key in self.mod_cache:
      return self.mod_cache[self.cache_key]
    from Util import find_lib
    # Note about BLAS linkage:
    # TensorFlow (or its Eigen lib) likely has linked against some BLAS lib itself.
    # For our CPU code, we directly call some BLAS functions such as `sgemm_`.
    # On platforms where there is a flat namespace (e.g. Mac),
    # it probably is not needed to explicitly link it again for this module.
    # In other cases, it's probably needed, but it's not so clear which lib has the
    # right symbols (e.g. the `sgemm_` symbol).
    # The current solution is just to link against blas/f77blas
    # (both can potentially have the symbol) if it finds the lib.
    ld_flags = []
    if find_lib("blas"):
      ld_flags += ["-lblas"]
    if find_lib("f77blas"):
      ld_flags += ["-lf77blas"]
    # Another option to find some BLAS lib.
    import numpy
    numpy_dir = os.path.dirname(numpy.__file__)
    if os.path.exists("%s/.libs" % numpy_dir):
      ld_flags += ["-L%s/.libs" % numpy_dir]
      from glob import glob
      for f in glob("%s/.libs/*.so" % numpy_dir):
        f = os.path.basename(f)
        if f.startswith("lib"):
          f = f[3:]
        if f.endswith(".so"):
          f = f[:-3]
        ld_flags += ["-l%s" % f]
    comp = TFUtil.OpCodeCompiler(
      base_name=self.name, code_version=self.description.code_version,
      code=self._make_code(),
      include_deps=[self.support_native_op_cpp_filename],
      ld_flags=ld_flags,
      use_cuda_if_available=self.with_cuda,
      **dict(self.compiler_opts))
    mod = comp.load_tf_module()
    self.mod_cache[self.cache_key] = mod
    return mod

  def make_op(self):
    with self.global_lock:
      if self.cache_key in self.op_cache:
        return self.op_cache[self.cache_key]
      mod = self._make_mod()
      op = getattr(mod, camel_case_to_snake_case(self.op_name))
      self.op_cache[self.cache_key] = op

      if self.description.is_grad_defined:
        grad_description = self.description.grad()
        grad_op_maker = OpMaker(description=grad_description, compiler_opts=self.compiler_opts)
        grad_op = grad_op_maker.make_op()

        from tensorflow.python.framework import ops
        def grad_wrapper(fwd_op, *bwd_grads):
          """
          :param tf.Operation fwd_op: for fwd_op.inputs and fwd_op.outputs
          :param list[tf.Tensor] bwd_grads:
          :return: list of tensors of gradients for each input
          :rtype: list[tf.Tensor]
          """
          assert len(bwd_grads) == len(fwd_op.outputs)

          grad_inputs = list(fwd_op.inputs) + list(fwd_op.outputs) + list(bwd_grads)
          grad_inputs = self.description._filter_grad_inputs(grad_inputs)
          grad_outputs = TFUtil.make_var_tuple(grad_op(*grad_inputs))
          if grad_description.num_dummy_outs > 0:
            grad_outputs = grad_outputs[:-grad_description.num_dummy_outs]
          grad_outputs = self.description.make_results_of_gradient(grad_outputs)
          return grad_outputs

        grad_wrapper.__name__ = grad_description.name
        grad_wrapper.grad_op = grad_op
        ops.RegisterGradient(self.name)(grad_wrapper)
        op.grad_wrapper = grad_wrapper
        op.grad_op = grad_op

    return op


def load_dump_file(filename):
  """
  See dump_to_file() in NativeOp.cpp.

  :param str filename:
  :rtype: numpy.ndarray
  """
  import numpy
  from struct import unpack

  with open(filename, "rb") as f:
    def _read_uint64():
      return int(unpack("Q", f.read(8))[0])

    def _read_bytes():
      size = _read_uint64()
      return f.read(size)

    def _read_str():
      return _read_bytes().decode("utf8")

    header = _read_str()
    assert header == "NativeOp_dump"
    dtype_name = _read_str()
    if dtype_name == "float":
      dtype_name = "float32"
    dtype = numpy.dtype(dtype_name)
    dtype_size = _read_uint64()
    assert dtype.itemsize == dtype_size, "dtype %r %r: %r != %r" % (dtype_name, dtype, dtype.itemsize, dtype_size)
    ndim = _read_uint64()
    dims = [_read_uint64() for i in range(ndim)]
    data = _read_bytes()
    assert len(data) == numpy.prod(dims) * dtype.itemsize
    v_flat = numpy.fromstring(data, dtype=dtype)
    v = v_flat.reshape(dims)
    return v


def make_op(cls, **kwargs):
  """
  :param Type[NativeOp.NativeOpGenBase] cls:
  :param kwargs: passed to OpMaker
  :return: op
  :rtype: (tf.Tensor) -> tuple[tf.Tensor]
  """
  maker = OpMaker(OpDescription.from_gen_base(cls), **kwargs)
  return maker.make_op()


def make_lstm_op(**kwargs):
  """
  See :class:`NativeLstmCell` for usage.

  :return: op
  :rtype: (tf.Tensor) -> tuple[tf.Tensor]
  """
  return make_op(NativeOp.LstmGenericBase, **kwargs)


class RecSeqCellOp(object):
  does_input_projection = False
  does_direction_handling = False

  def __init__(self, n_hidden, n_input_dim=None, input_is_sparse=False, step=None):
    """
    :param int n_hidden:
    :param int n_input_dim:
    :param bool input_is_sparse:
    :param int step: what direction and step to use
    """
    if n_input_dim is None:
      n_input_dim = n_hidden
    self.n_hidden = n_hidden  # hidden-dim and output-dim
    self.n_input_dim = n_input_dim  # input dim for the inputs in __call__
    self.input_is_sparse = input_is_sparse
    self.step = step if self.does_direction_handling else None

  @property
  def state_size(self):
    return self.n_hidden

  def __call__(self, inputs, index, initial_state=None, recurrent_weights_initializer=None):
    """
    :param tf.Tensor inputs: shape (time,batch,n_input_dim)
    :param tf.Tensor index: shape (time,batch)
    :param tf.Tensor|None initial_state: optional initial state of shape (batch,n_hidden)
    :param ()->tf.Tensor recurrent_weights_initializer:
    :returns: output fused tensor shape (time,batch,n_hidden), last hidden state (batch,n_hidden)
    :rtype: (tf.Tensor, tf.Tensor)
    """
    raise NotImplementedError


class NativeLstmCell(RecSeqCellOp):
  def __init__(self, **kwargs):
    super(NativeLstmCell, self).__init__(**kwargs)
    self.n_input_dim = self.n_hidden * 4
    self.op = make_lstm_op()

  @classmethod
  def map_layer_inputs_to_op(cls, Z, V_h, i, initial_state=None):
    """
    Just like NativeOp.LstmGenericBase.map_layer_inputs_to_op().
    :param tf.Tensor Z: inputs: shape (time,batch,n_hidden*4)
    :param tf.Tensor V_h: W_re: shape (n_hidden,n_hidden*4)
    :param tf.Tensor i: index: shape (time,batch)
    :param tf.Tensor|None initial_state: shape (batch,n_hidden)
    :rtype: (tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor)
    """
    assert Z.get_shape().ndims == 3
    assert V_h.get_shape().ndims == 2
    assert i.get_shape().ndims == 2
    if i.dtype != tf.float32:
      if not hasattr(i, "cast_float32"):
        from TFUtil import reuse_name_scope_of_tensor
        with reuse_name_scope_of_tensor(i):
          i_cast_float32 = tf.cast(i, dtype=tf.float32, name="index_cast_float32")
        i.cast_float32 = i_cast_float32
      i = i.cast_float32
    n_batch = tf.shape(Z)[1]
    n_out = tf.shape(V_h)[0]
    if initial_state is not None:
      from tensorflow.python.ops.nn import rnn_cell
      if isinstance(initial_state, rnn_cell.LSTMStateTuple):
        initial_state = initial_state.c
      c = initial_state
    else:
      c = tf.zeros((n_batch, n_out), dtype=tf.float32)
    return Z, V_h, c, i

  def __call__(self, inputs, index, initial_state=None, recurrent_weights_initializer=None):
    """
    :param tf.Tensor inputs: shape (time,batch,n_hidden*4)
    :param tf.Tensor index: shape (time,batch)
    :param tf.Tensor|None initial_state: shape (batch,n_hidden)
    :param ()->tf.Tensor recurrent_weights_initializer:
    :returns: shape (time,batch,n_hidden), shape (batch,n_hidden)
    :rtype: (tf.Tensor, tf.Tensor)
    """
    W_re = tf.get_variable(
      name="W_re", shape=(self.n_hidden, self.n_hidden * 4), initializer=recurrent_weights_initializer)
    out, _, final_state = self.op(
      *self.map_layer_inputs_to_op(Z=inputs, V_h=W_re, i=index, initial_state=initial_state))
    return out, final_state


class NativeLstmLowMemCell(RecSeqCellOp):
  does_input_projection = True
  does_direction_handling = True

  def __init__(self, **kwargs):
    super(NativeLstmLowMemCell, self).__init__(**kwargs)
    self.op = make_op(NativeOp.LstmLowMem)
    assert not self.input_is_sparse, "not supported"

  def map_layer_inputs_to_op(self, X, W, b, i, initial_state=None):
    """
    Just like NativeOp.LstmGenericBase.map_layer_inputs_to_op().
    :param tf.Tensor X: inputs: shape (time,batch,n_input_dim)
    :param tf.Tensor W: shape (n_input_dim+n_hidden,n_hidden*4)
    :param tf.Tensor b: shape (n_hidden*4,)
    :param tf.Tensor i: index: shape (time,batch)
    :param tf.Tensor|None initial_state: shape (batch,n_hidden)
    :rtype: tuple[tf.Tensor]
    """
    X.set_shape(tf.TensorShape([None, None, self.n_input_dim]))
    W.set_shape(tf.TensorShape([self.n_input_dim + self.n_hidden, self.n_hidden * 4]))
    i.set_shape(tf.TensorShape([None, None]))
    if i.dtype != tf.float32:
      if not hasattr(i, "cast_float32"):
        from TFUtil import reuse_name_scope_of_tensor
        with reuse_name_scope_of_tensor(i):
          i_cast_float32 = tf.cast(i, dtype=tf.float32, name="index_cast_float32")
        i.cast_float32 = i_cast_float32
      i = i.cast_float32
    n_batch = tf.shape(X)[1]
    if initial_state is not None:
      c0 = initial_state
    else:
      c0 = tf.zeros((n_batch, self.n_hidden), dtype=tf.float32, name="initial_c")
    # We could make `h` a variable exactly if `c` is a trainable variable.
    y0 = tf.zeros((n_batch, self.n_hidden), dtype=tf.float32, name="initial_h")
    start = tf.constant(0, name="start")
    step = tf.constant(self.step or 1, name="step")
    return X, W, b, y0, c0, i, start, step

  def __call__(self, inputs, index, initial_state=None, recurrent_weights_initializer=None):
    """
    :param tf.Tensor inputs: shape (time,batch,n_input_dim)
    :param tf.Tensor index: shape (time,batch)
    :param tf.Tensor|None initial_state: shape (batch,n_hidden)
    :param ()->tf.Tensor recurrent_weights_initializer:
    :returns: shape (time,batch,n_hidden), shape (batch,n_hidden)
    :rtype: (tf.Tensor, tf.Tensor)
    """
    W = tf.get_variable(
      name="W", shape=(self.n_input_dim + self.n_hidden, self.n_hidden * 4), initializer=recurrent_weights_initializer)
    b = tf.get_variable(name="b", shape=(self.n_hidden * 4,), initializer=tf.zeros_initializer())
    out, _, final_state = self.op(
      *self.map_layer_inputs_to_op(X=inputs, W=W, b=b, i=index, initial_state=initial_state))
    return out, final_state


class NativeLstm2(RecSeqCellOp):
  does_input_projection = False
  does_direction_handling = True

  def __init__(self, **kwargs):
    super(NativeLstm2, self).__init__(**kwargs)
    self.n_input_dim = self.n_hidden * 4
    self.op = make_op(NativeOp.NativeLstm2)

  @property
  def state_size(self):
    from tensorflow.python.ops.nn import rnn_cell
    return rnn_cell.LSTMStateTuple(c=self.n_hidden, h=self.n_hidden)

  def map_layer_inputs_to_op(self, X, W, i, initial_state=None):
    """
    Just like NativeOp.LstmGenericBase.map_layer_inputs_to_op().
    :param tf.Tensor X: inputs: shape (time,batch,n_input_dim)
    :param tf.Tensor W: shape (n_input_dim+n_hidden,n_hidden*4)
    :param tf.Tensor i: index: shape (time,batch)
    :param tf.Tensor|None initial_state: shape (batch,n_hidden)
    :rtype: tuple[tf.Tensor]
    """
    from tensorflow.python.ops.nn import rnn_cell
    X.set_shape(tf.TensorShape([None, None, self.n_hidden * 4]))
    W.set_shape(tf.TensorShape([self.n_hidden, self.n_hidden * 4]))
    i.set_shape(tf.TensorShape([None, None]))
    if i.dtype != tf.float32:
      if not hasattr(i, "cast_float32"):
        from TFUtil import reuse_name_scope_of_tensor
        with reuse_name_scope_of_tensor(i):
          i_cast_float32 = tf.cast(i, dtype=tf.float32, name="index_cast_float32")
        i.cast_float32 = i_cast_float32
      i = i.cast_float32
    n_batch = tf.shape(X)[1]
    if initial_state is None:
      c0 = tf.zeros((n_batch, self.n_hidden), dtype=tf.float32, name="initial_c")
      y0 = tf.zeros((n_batch, self.n_hidden), dtype=tf.float32, name="initial_h")
    elif isinstance(initial_state, rnn_cell.LSTMStateTuple):
      c0 = initial_state.c
      y0 = initial_state.h
    else:
      c0 = initial_state
      y0 = tf.zeros((n_batch, self.n_hidden), dtype=tf.float32, name="initial_h")
    start = tf.constant(0, name="start")
    step = tf.constant(self.step or 1, name="step")
    return X, W, y0, c0, i, start, step

  def __call__(self, inputs, index, initial_state=None, recurrent_weights_initializer=None):
    """
    :param tf.Tensor inputs: shape (time,batch,n_hidden)
    :param tf.Tensor index: shape (time,batch)
    :param tf.Tensor|None initial_state: shape (batch,n_hidden)
    :param ()->tf.Tensor recurrent_weights_initializer:
    :returns: shape (time,batch,n_hidden), shape (batch,n_hidden)
    :rtype: (tf.Tensor, tf.Tensor)
    """
    W = tf.get_variable(
      name="W_re", shape=(self.n_hidden, self.n_hidden * 4), initializer=recurrent_weights_initializer)
    out, _, _, final_cell_state = self.op(
      *self.map_layer_inputs_to_op(X=inputs, W=W, i=index, initial_state=initial_state))
    from tensorflow.python.ops.nn import rnn_cell
    return out, rnn_cell.LSTMStateTuple(h=out[-1], c=final_cell_state)


def make_fast_baum_welch_op(**kwargs):
  """
  :return: op
  :rtype: (tf.Tensor) -> tuple[tf.Tensor]
  """
  maker = OpMaker(OpDescription.from_gen_base(NativeOp.FastBaumWelchOp), **kwargs)
  return maker.make_op()


def fast_baum_welch(am_scores, edges, weights, start_end_states, float_idx, state_buffer=None):
  """
  :param tf.Tensor am_scores: (time, batch, dim), in -log space
  :param tf.Tensor edges: (4,num_edges), edges of the graph (from,to,emission_idx,sequence_idx)
  :param tf.Tensor weights: (num_edges,), weights of the edges
  :param tf.Tensor start_end_states: (2, batch), (start,end) state idx in automaton. there is only one single automaton.
  :param tf.Tensor float_idx: (time, batch) -> 0 or 1 (index mask, via seq lens)
  :param tf.Tensor state_buffer: (2, num_states)
  :return: (fwdbwd, obs_scores), fwdbwd is (time, batch, dim), obs_scores is (time, batch), in -log space
  :rtype: (tf.Tensor, tf.Tensor)
  """
  # edges, weights, start_end_states, state_buffer = SprintAlignmentAutomataOp(self.sprint_opts)(self.network.tags)
  op = make_fast_baum_welch_op()
  float_idx = tf.cast(float_idx, tf.float32)
  if state_buffer is None:
    last_state_idx = tf.reduce_max(start_end_states[1])  # see get_automata_for_batch
    state_buffer = tf.zeros((2, last_state_idx + 1))
  fwdbwd, obs_scores = op(am_scores, edges, weights, start_end_states, float_idx, state_buffer)
  return fwdbwd, obs_scores


def fast_baum_welch_by_sprint_automata(am_scores, float_idx, tags, sprint_opts, tdp_scale=1.0):
  """
  :param tf.Tensor am_scores: (time, batch, dim), in -log space
  :param tf.Tensor float_idx: (time, batch) -> 0 or 1 (index mask, via seq lens)
  :param tf.Tensor tags: (batch,) -> seq name (str)
  :param float tdp_scale: weights are multiplied by this
  :param dict[str] sprint_opts:
  :return: (fwdbwd, obs_scores), fwdbwd is (time, batch, dim), obs_scores is (time, batch), in -log space
  :rtype: (tf.Tensor, tf.Tensor)
  """
  from TFSprint import get_sprint_automata_for_batch_op
  edges, weights, start_end_states = get_sprint_automata_for_batch_op(sprint_opts=sprint_opts, tags=tags)
  if tdp_scale != 1:
    if tdp_scale == 0:
      weights = tf.zeros_like(weights)
    else:
      weights *= tdp_scale
  return fast_baum_welch(
    am_scores=am_scores, float_idx=float_idx,
    edges=edges, weights=weights, start_end_states=start_end_states)


def _debug_dumped_fast_baum_welch(prefix, postfix=".dump"):
  """
  If you uncomment the debug_print statements in FastBaumWelchOp, as well as dump_to_file inside debug_print,
  you will get some dump files in the current directory. These can be loaded here and evald again.

  :param str prefix: filename prefix, e.g. "ff_out_bw__FastBaumWelchOp_"
  :param str postfix: filename postfix
  :return: output from fast_baum_welch(), evald
  :rtype: (numpy.ndarray. numpy.ndarray)
  """
  with tf.Graph().as_default() as graph:
    with tf.Session(graph=graph) as session:
      arg_names = {
        "am_scores": None, "edges": None, "weights": None, "start_end_states": None, "float_idx": "index",
        "state_buffer": None}
      args = {}
      for name, file_postfix in list(arg_names.items()):
        if file_postfix is None:
          file_postfix = name
        filename = prefix + file_postfix + postfix
        print("load", filename)
        args[name] = tf.constant(load_dump_file(filename))
      print("run...")
      out_list = fast_baum_welch(**args)
      return session.run(out_list)


def have_blocksparse_requirements():
  import TFUtil
  if not TFUtil.is_gpu_available():
    return False
  min_compute_capability = TFUtil.get_available_gpu_min_compute_capability()
  if min_compute_capability < 3.5:
    return False
  return True


def init_blocksparse():
  import TFUtil
  assert TFUtil.is_gpu_available(), "we currently need a GPU"
  min_compute_capability = TFUtil.get_available_gpu_min_compute_capability()
  assert min_compute_capability and min_compute_capability >= 3.5, "we need at least compute capability 3.5"
  path = os.path.dirname(__file__) + "/extern/blocksparse"
  assert os.path.exists(path), "maybe submodule not checked out?"
  import sys
  if path not in sys.path:
    # At the beginning, to make sure we find it firs.t
    sys.path.insert(0, path)
  # test it
  from blocksparse import op_module
  op_module.get_module()


def demo():
  print("TFNativeOp demo")
  TFUtil.CudaEnv.verbose_find_cuda = True
  print("CUDA path: %s" % TFUtil.CudaEnv.get_instance().cuda_path)
  op = make_op(NativeOp.LstmLowMem, compiler_opts={"static_version_name": "demo"})
  print(op)


if __name__ == '__main__':
  import better_exchook
  better_exchook.install()
  demo()
