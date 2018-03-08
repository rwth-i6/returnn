"""
Uses KenLM (extern/kenlm) to read n-gram LMs (ARPA format),
and provides a TF op to use them.

"""

import sys
import os
import tensorflow as tf

# https://www.tensorflow.org/extend/adding_an_op
_src_code = """
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("KenLmScoreStrings")
    .Input("strings: string")
    .Output("scores: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class KenLmScoreStringsOp : public OpKernel {
 public:
  explicit KenLmScoreStringsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    //auto input = input_tensor.flat<string>();

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<float>();

    //...
  }
};

REGISTER_KERNEL_BUILDER(Name("KenLmScoreStrings").Device(DEVICE_CPU), KenLmScoreStringsOp);
"""

_tf_mod = None


def get_tf_mod():
  global _tf_mod
  if _tf_mod:
    return _tf_mod
  import platform
  from glob import glob
  from TFUtil import OpCodeCompiler

  # References:
  # https://github.com/kpu/kenlm/blob/master/setup.py
  # https://github.com/kpu/kenlm/blob/master/compile_query_only.sh

  # Collect files.
  files = glob('util/*.cc') + glob('lm/*.cc') + glob('util/double-conversion/*.cc')
  files = [fn for fn in files if not (fn.endswith('main.cc') or fn.endswith('test.cc'))]
  libs = []
  if platform.system() != 'Darwin':
    libs.append('rt')

  # Put code all together in one big blob.
  src_code = ""
  for fn in files:
    src_code += "\n// ------------ %s : BEGIN { ------------\n"
    src_code += open(fn).read()
    src_code += "\n// ------------ %s : END } --------------\n\n"
  src_code += "\n\n// ------------ our code now: ------------\n\n"
  src_code += _src_code

  compiler = OpCodeCompiler(
    base_name="KenLM", code_version=1, code=src_code,
    c_macro_defines={"NDEBUG": 1, "KENLM_MAX_ORDER": 6},
    ld_flags=["-l%s" % lib for lib in libs],
    is_cpp=True, use_cuda_if_available=False)
  tf_mod = compiler.load_tf_module()
  assert hasattr(tf_mod, "ken_lm_score_strings"), "content of mod: %r" % (dir(tf_mod),)
  _tf_mod = tf_mod
  return tf_mod


def ken_lm_score_strings(*args, **kwargs):
  return get_tf_mod().ken_lm_score_strings(*args, **kwargs)


if __name__ == "__main__":
  import better_exchook
  better_exchook.install()
  # Try to compile now.
  get_tf_mod()
  # Some demo.
  input_strings_tf = tf.placeholder(tf.string, [None])
  output_scores_tf = ken_lm_score_strings(input_strings_tf)
  with tf.Session() as session:
    output_scores = session.run(output_scores_tf, feed_dict={input_strings_tf: ["hello world"]})
    print("output scores:", output_scores)

