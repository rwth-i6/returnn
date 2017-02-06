
from __future__ import print_function

import tensorflow as tf
import NativeOp
import TFUtil
import os
import re


def _camel_case_to_snake_case(name):
  s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class OpDescription(NativeOp.NativeOpBaseMixin):
  @classmethod
  def from_gen_base(x, gen_base):
    """
    :param NativeOp.NativeOpGenBase gen_base:
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
  with_cuda = None
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
      cls.with_cuda = bool(TFUtil.CudaEnv.get_instance())
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
    import tensorflow.contrib.rnn.python.ops.lstm_ops as lstm_ops
    lstm_ops_so = "%s/_lstm_ops.so" % os.path.dirname(lstm_ops.__file__)
    assert os.path.exists(lstm_ops_so)
    # Maybe a bit hacky: Just load all symbols into the global namespace.
    from ctypes import RTLD_GLOBAL, CDLL
    CDLL(lstm_ops_so, mode=RTLD_GLOBAL)

  @property
  def op_name(self):
    return self.name

  @property
  def cache_key(self):
    return self.name

  @property
  def support_native_op_cpp_filename(self):
    my_dir = os.path.abspath(os.path.dirname(__file__) or os.getcwd())
    support_native_op_cpp_filename = "%s/NativeOp.cpp" % my_dir
    assert os.path.exists(support_native_op_cpp_filename)
    return support_native_op_cpp_filename

  def _make_code(self):
    # In the user code, we assume that we have the following variables:
    # int n_inputs; int n_outputs;
    # Ndarray* inputs[n_inputs]; Ndarray** outputs[n_outputs];
    # Reference:
    # https://www.tensorflow.org/versions/master/how_tos/adding_an_op/
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
    code_user = self.description.c_fw_code % {"fail": "assert(false);"}
    code_compute = "\n".join([
      code_forward_io,
      code_set_io,
      code_user])
    format_args = {
      "op_name": self.op_name,
      "code_register_op_io": code_register_op_io,
      "code_forward_io": code_forward_io,
      "code_set_io": code_set_io,
      "code_compute": code_compute,
      "user_code_kernels": self.description._reduce_c_extra_support_code(self.description.c_extra_support_code),
      "native_op_cpp_filename": self.support_native_op_cpp_filename,
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
    """
    if self.with_cuda:
      # http://docs.nvidia.com/cuda/cublas
      code_header += """
      #include <cuda.h>
      #include <cuda_runtime.h>
      #include <cublas_v2.h>

      // https://github.com/tensorflow/tensorflow/issues/6602 ?
      //#include "tensorflow/core/platform/stream_executor.h"
      //#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
      """
    code_header += """
    using namespace tensorflow;
    """
    # sgemm
    code_header += """
    typedef float real;
    typedef int integer;
    extern "C"
    extern int sgemm_(char *transa, char *transb,
      integer *m, integer *n, integer *k,
      const real *alpha,
      const real *a, integer *lda,
      const real *b, integer *ldb,
      const real *beta,
      real *c, integer *ldc);
    """
    code_register = """
    REGISTER_OP("%(op_name)s")
    %(code_register_op_io)s;
    """ % format_args
    code_op = """
    #define TENSORFLOW 1
    #define CUDA 0
    #include "%(native_op_cpp_filename)s"

    %(user_code_kernels)s

    static const int n_inputs = %(n_inputs)i, n_outputs = %(n_outputs)i;

    class %(op_name)sOp : public OpKernel {
    public:
      explicit %(op_name)sOp(OpKernelConstruction* context) : OpKernel(context) {}
      void Compute(OpKernelContext* context) override {
        %(code_compute)s
      }
    };

    REGISTER_KERNEL_BUILDER(Name("%(op_name)s").Device(DEVICE_CPU), %(op_name)sOp);
    """ % format_args
    if self.with_cuda:
      code_gpu_op = """
      namespace _gpu {
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

        REGISTER_KERNEL_BUILDER(Name("%(op_name)s").Device(DEVICE_GPU), %(op_name)sGpuOp);
      }
      """ % format_args
    else:
      code_gpu_op = ""
    return code_header + code_register + code_op + code_gpu_op

  def _make_mod(self):
    if self.cache_key in self.mod_cache:
      return self.mod_cache[self.cache_key]
    comp = TFUtil.OpCodeCompiler(
      base_name=self.name, code_version=self.description.code_version,
      code=self._make_code(),
      include_deps=[self.support_native_op_cpp_filename],
      ld_flags=["-lblas"],
      **dict(self.compiler_opts))
    mod = comp.load_module()
    self.mod_cache[self.cache_key] = mod
    return mod

  def make_op(self):
    if self.cache_key in self.op_cache:
      return self.op_cache[self.cache_key]
    mod = self._make_mod()
    op = getattr(mod, _camel_case_to_snake_case(self.op_name))
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
      ops.RegisterGradient(self.name)(grad_wrapper)

    return op


def make_lstm_op(**kwargs):
  """
  Demo.
  :return: op
  :rtype: (tf.Tensor) -> tuple[tf.Tensor]
  """
  maker = OpMaker(OpDescription.from_gen_base(NativeOp.LstmGenericBase), **kwargs)
  return maker.make_op()


class RecSeqCellOp(object):
  def __init__(self, n_hidden):
    self.n_hidden = n_hidden
    self.n_input_dim = n_hidden

  def __call__(self, inputs, index):
    """
    :param tf.Tensor inputs: shape (time,batch,n_input_dim)
    :param tf.Tensor index: shape (time,batch)
    :returns: shape (time,batch,n_hidden)
    :rtype: tf.Tensor
    """
    raise NotImplementedError


class NativeLstmCell(RecSeqCellOp):
  def __init__(self, n_hidden):
    super(NativeLstmCell, self).__init__(n_hidden=n_hidden)
    self.n_input_dim = n_hidden * 4
    self.op = make_lstm_op()

  @classmethod
  def map_layer_inputs_to_op(cls, Z, V_h, i):
    """
    Just like NativeOp.LstmGenericBase.map_layer_inputs_to_op().
    :param tf.Tensor Z: inputs: shape (time,batch,n_hidden*4)
    :param tf.Tensor V_h: W_re: shape (n_hidden,n_hidden*4)
    :param tf.Tensor i: index: shape (time,batch)
    :rtype: (tf.Tensor,tf.Tensor,tf.Tensor,tf.Tensor)
    """
    assert Z.get_shape().ndims == 3
    assert V_h.get_shape().ndims == 2
    assert i.get_shape().ndims == 2
    if i.dtype != tf.float32:
      i = tf.cast(i, dtype=tf.float32)
    n_batch = tf.shape(Z)[1]
    n_out = tf.shape(V_h)[0]
    c = tf.zeros((n_batch, n_out), dtype=tf.float32)
    return Z, V_h, c, i

  def __call__(self, inputs, index):
    """
    :param tf.Tensor inputs: shape (time,batch,n_hidden*4)
    :param tf.Tensor index: shape (time,batch)
    :returns: shape (time,batch,n_hidden)
    :rtype: tf.Tensor
    """
    W_re = tf.get_variable(name="W_re", shape=(self.n_hidden, self.n_hidden * 4))
    lstm_op = make_lstm_op()
    op_out = lstm_op(*self.map_layer_inputs_to_op(inputs, W_re, index))
    from TFUtil import make_var_tuple
    out = NativeOp.LstmGenericBase.map_layer_output_from_op(*make_var_tuple(op_out))
    return out


def demo():
  print("TFNativeOp demo")
  TFUtil.CudaEnv.verbose_find_cuda = True
  print("CUDA path: %s" % TFUtil.CudaEnv.get_instance().cuda_path)
  op = make_lstm_op(compiler_opts={"static_version_name": "demo"})
  print(op)


if __name__ == '__main__':
  import better_exchook
  better_exchook.install()
  demo()
