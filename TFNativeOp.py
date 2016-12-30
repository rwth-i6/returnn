
from __future__ import print_function

import tensorflow as tf
import NativeOp
import TFUtil
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

  def _make_code(self):
    code_register_op_io = ""
    for v in self.gen_base.in_info:
      code_register_op_io += ".Input(\"%s: %s\")\n" % (v["name"].lower(), v.get("dtype", "float32"))
    for v in self.gen_base.out_info:
      code_register_op_io += ".Output(\"%s: %s\")\n" % (v["name"].lower(), v.get("dtype", "float32"))
    format_args = {
      "op_name": self.op_name,
      "code_register_op_io": code_register_op_io
    }
    code_header = """
    #include "tensorflow/core/framework/op.h"
    #include "tensorflow/core/framework/shape_inference.h"
    #include "tensorflow/core/framework/op_kernel.h"

    using namespace tensorflow;
    """
    code_register = """
    REGISTER_OP("%(op_name)s")
    %(code_register_op_io)s;
    """ % format_args
    # In the user code, we assume that we have the following variables:
    # int n_inputs; int n_outputs;
    # Ndarray* inputs[n_inputs]; Ndarray** outputs[n_outputs];
    code_op = """
    class %(op_name)sOp : public OpKernel {
    public:
      explicit %(op_name)sOp(OpKernelConstruction* context) : OpKernel(context) {}
      void Compute(OpKernelContext* context) override {
        // ...
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
