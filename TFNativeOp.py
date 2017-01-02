
from __future__ import print_function

import tensorflow as tf
import NativeOp
import TFUtil
import os
import re


def _camel_case_to_snake_case(name):
  s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class OpMaker(object):
  """
  https://www.tensorflow.org/versions/master/how_tos/adding_an_op/
  """

  def __init__(self, gen_base, name=None, compiler_opts=None):
    """
    :param NativeOp.NativeOpGenBase gen_base:
    :param str|None name: e.g. "LstmGenericBase", or automatically via gen_base.__name__
    :param dict[str]|None compiler_opts: passed on to OpCodeCompiler as kwargs
    """
    if not name:
      name = gen_base.__name__
    self.name = name
    self.gen_base = gen_base
    self.compiler_opts = compiler_opts or {}
    self.with_cuda = bool(TFUtil.CudaEnv.get_instance())

  @property
  def op_name(self):
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
    # http://stackoverflow.com/questions/37565367/designing-an-accumulating-tensorflow-gpu-operator
    # We also include NativeOp.cpp.
    in_info, out_info, _ = NativeOp.NativeOp._resolve_want_inplace_dummy(
      in_info=self.gen_base.in_info, out_info=self.gen_base.out_info)
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
    def map_name(v):
      return v["name"].lower()
    def map_type(v, is_out=False):
      t = v.get("dtype", "float32")
      if is_out:
        if v["name"] in out_is_ref:
          t = "Ref(%s)" % t
      else:
        if v.get("want_inplace", -1) >= 0:
          t = "Ref(%s)" % t
      return t
    code_register_op_io = ""
    for v in in_info:
      code_register_op_io += ".Input(\"%s: %s\")\n" % (map_name(v), map_type(v))
    for v in out_info:
      code_register_op_io += ".Output(\"%s: %s\")\n" % (map_name(v), map_type(v, is_out=True))
    code_forward_io = ""
    for in_idx, v in enumerate(in_info):
      out_idx = v.get("want_inplace", -1)
      if out_idx >= 0:
        code_forward_io += "context->forward_ref_input_to_ref_output(%i, %i);\n" % (in_idx, out_idx)
    code_set_io = ""
    for in_idx, v in enumerate(in_info):
      if v.get("want_inplace", -1) >= 0:  # is ref
        code_set_io += "Ndarray mutable_input_%i = context->mutable_input(%i, false);\n" % (in_idx, in_idx)
        code_set_io += "inputs[%i] = &mutable_input_%i;\n" % (in_idx, in_idx)
      else:  # no ref
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
        code_set_io += "OP_REQUIRES_OK(context, context->allocate_output(%i, %s, outputs[%i]));\n" % (out_idx, cshape, out_idx)
    format_args = {
      "op_name": self.op_name,
      "code_register_op_io": code_register_op_io,
      "code_forward_io": code_forward_io,
      "code_set_io": code_set_io,
      "user_code": self.gen_base.c_fw_code % {"fail": "assert(false);"},
      "user_code_kernels": NativeOp.NativeOp._reduce_c_extra_support_code(self.gen_base.c_extra_support_code),
      "native_op_cpp_filename": self.support_native_op_cpp_filename,
      "n_inputs": len(in_info),
      "n_outputs": len(out_info)
    }
    code_header = """
    #include "tensorflow/core/framework/op.h"
    #include "tensorflow/core/framework/shape_inference.h"
    #include "tensorflow/core/framework/op_kernel.h"

    using namespace tensorflow;
    """
    if self.with_cuda:
      # http://docs.nvidia.com/cuda/cublas
      code_header += """
      #include <cuda_runtime.h>
      #include "cublas_v2.h"
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
    #include "%(native_op_cpp_filename)s"

    %(user_code_kernels)s

    static const int n_inputs = %(n_inputs)i, n_outputs = %(n_outputs)i;

    class %(op_name)sOp : public OpKernel {
    public:
      explicit %(op_name)sOp(OpKernelConstruction* context) : OpKernel(context) {}
      void Compute(OpKernelContext* context) override {
        %(code_forward_io)s
        Ndarray* inputs[n_inputs];
        Ndarray** outputs[n_outputs];
        %(code_set_io)s
        %(user_code)s
      }
    };

    REGISTER_KERNEL_BUILDER(Name("%(op_name)s").Device(DEVICE_CPU), %(op_name)sOp);
    """ % format_args
    if self.with_cuda:
      code_gpu_op = """
      class %(op_name)sGpuOp : public OpKernel {
      public:
        explicit %(op_name)sGpuOp(OpKernelConstruction* context) : OpKernel(context) {}
        void Compute(OpKernelContext* context) override {
          // ...
        }
      };

      REGISTER_KERNEL_BUILDER(Name("%(op_name)s").Device(DEVICE_GPU), %(op_name)sGpuOp);
      """ % format_args
    else:
      code_gpu_op = ""
    return code_header + code_register + code_op + code_gpu_op

  def _make_mod(self):
    comp = TFUtil.OpCodeCompiler(
      base_name=self.name, code_version=self.gen_base.code_version,
      code=self._make_code(),
      include_deps=[self.support_native_op_cpp_filename],
      ld_flags=["-lblas"],
      **dict(self.compiler_opts))
    mod = comp.load_module()
    return mod

  def make_op(self):
    mod = self._make_mod()
    return getattr(mod, _camel_case_to_snake_case(self.op_name))


def make_lstm_op(**kwargs):
  """
  Demo.
  :return: op
  :rtype: (tf.Tensor) -> tf.Tensor
  """
  maker = OpMaker(NativeOp.LstmGenericBase, **kwargs)
  return maker.make_op()


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
