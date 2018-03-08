"""
Uses KenLM (extern/kenlm) to read n-gram LMs (ARPA format),
and provides a TF op to use them.

"""

import sys
import os
import tensorflow as tf

returnn_dir = os.path.dirname(os.path.abspath(__file__))
kenlm_dir = returnn_dir + "/extern/kenlm"

# https://www.tensorflow.org/extend/adding_an_op
# Also see TFUitl.TFArrayContainer for TF resources.
_src_code = """
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

using namespace tensorflow;


REGISTER_OP("KenLmLoadModel")
.Attr("filename: string")
.Attr("container: string = ''")
.Attr("shared_name: string = ''")
.Output("handle: resource")
.SetIsStateful()
.SetShapeFn(shape_inference::ScalarShape)
.Doc("KenLmLoadModel: loads KenLM model, creates TF resource, persistent across runs in the session");


REGISTER_OP("KenLmScoreStrings")
.Input("handle: resource")
.Input("strings: string")
.Output("scores: float32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(1));
  return Status::OK();
})
.Doc("KenLmScoreStrings: scores texts");


// https://github.com/kpu/kenlm/blob/master/lm/model.hh
struct KenLmModel : public ResourceBase {
  explicit KenLmModel(const string& filename)
      : filename_(filename), model_(filename.c_str()) {}

  float score(const string& text) {
    float total = 0;
    mutex_lock l(mu_);
    lm::ngram::State state;
    model_.BeginSentenceWrite(&state);
    lm::ngram::State out_state;
    for(const string& word : tensorflow::str_util::Split(text, ' ')) {
      if(word.empty()) continue;
      auto word_idx = model_.BaseVocabulary().Index(word);
      total += model_.BaseScore(&state, word_idx, &out_state);
      state = out_state;
    }
    return total;
  }
  
  string DebugString() override {
    return strings::StrCat("KenLmModel[", filename_, "]");
  }

  const string filename_;
  mutex mu_;
  lm::ngram::RestProbingModel model_ GUARDED_BY(mu_);
};


// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_op_kernel.h
// TFUtil.TFArrayContainer
class KenLmLoadModelOp : public ResourceOpKernel<KenLmModel> {
 public:
  explicit KenLmLoadModelOp(OpKernelConstruction* context)
      : ResourceOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filename", &filename_));
  }

 private:
  virtual bool IsCancellable() const { return false; }
  virtual void Cancel() {}

  Status CreateResource(KenLmModel** ret) override EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    *ret = new KenLmModel(filename_);
    if(*ret == nullptr)
      return errors::ResourceExhausted("Failed to allocate");
    return Status::OK();
  }

  Status VerifyResource(KenLmModel* lm) override {
    if(lm->filename_ != filename_)
      return errors::InvalidArgument("Filename mismatch: expected ", filename_,
                                     " but got ", lm->filename_, ".");
    return Status::OK();
  }

  string filename_;
};

REGISTER_KERNEL_BUILDER(Name("KenLmLoadModel").Device(DEVICE_CPU), KenLmLoadModelOp);


class KenLmScoreStringsOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* context) override {
    KenLmModel* lm;
    {
      const Tensor* handle;
      OP_REQUIRES_OK(context, context->input("handle", &handle));        
      OP_REQUIRES_OK(context, GetResourceFromContext(context, "handle", &lm));
    }
    core::ScopedUnref unref(lm);

    const Tensor& input_tensor = context->input(1);
    auto input_flat = input_tensor.flat<string>();

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<float>();

    for(int i = 0; i < input_flat.size(); ++i) {
      output_flat(i) = lm->score(input_flat(i));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("KenLmScoreStrings").Device(DEVICE_CPU), KenLmScoreStringsOp);
"""

_kenlm_src_code_workarounds = """
// ------- start with some workarounds { ------
// The KenLM code (util/integer_to_string.cc) includes this file in the wrong namespace.
// Thus include it here now.
#include <emmintrin.h>
// ------- end with workarounds } -------------
"""


_tf_mod = None


def get_tf_mod(verbose=False):
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
  files = glob('%s/util/*.cc' % kenlm_dir)
  files += glob('%s/lm/*.cc' % kenlm_dir)
  files += glob('%s/util/double-conversion/*.cc' % kenlm_dir)
  files = [fn for fn in files if not (fn.endswith('main.cc') or fn.endswith('test.cc'))]
  libs = []
  if platform.system() != 'Darwin':
    libs.append('rt')

  # Put code all together in one big blob.
  src_code = ""
  src_code += _kenlm_src_code_workarounds
  for fn in files:
    f_code = open(fn).read()
    # We need to do some replacements to not clash symbol names.
    fn_short = os.path.basename(fn).replace(".", "_")
    for word in ["kConverter"]:
      f_code = f_code.replace(word, "%s_%s" % (fn_short, word))
    src_code += "\n// ------------ %s : BEGIN { ------------\n" % os.path.basename(fn)
    src_code += f_code
    src_code += "\n// ------------ %s : END } --------------\n\n" % os.path.basename(fn)
  src_code += "\n\n// ------------ our code now: ------------\n\n"
  src_code += _src_code

  compiler = OpCodeCompiler(
    base_name="KenLM", code_version=1, code=src_code,
    include_paths=(kenlm_dir, kenlm_dir + "/util/double-conversion"),
    c_macro_defines={"NDEBUG": 1, "KENLM_MAX_ORDER": 6},
    ld_flags=["-l%s" % lib for lib in libs],
    is_cpp=True, use_cuda_if_available=False,
    verbose=verbose)
  tf_mod = compiler.load_tf_module()
  assert hasattr(tf_mod, "ken_lm_score_strings"), "content of mod: %r" % (dir(tf_mod),)
  _tf_mod = tf_mod
  return tf_mod


def ken_lm_load(filename):
  return get_tf_mod().ken_lm_load_model(filename=filename)


def ken_lm_score_strings(handle, strings):
  return get_tf_mod().ken_lm_score_strings(handle=handle, strings=strings)


if __name__ == "__main__":
  import better_exchook
  better_exchook.install()
  # Try to compile now.
  get_tf_mod(verbose=True)
  # Some demo.
  input_strings = sys.argv[1:] or ["hello world </s>"]
  test_lm_file = kenlm_dir + "/lm/test.arpa"
  assert os.path.exists(test_lm_file)
  lm_tf = ken_lm_load(filename=test_lm_file)
  input_strings_tf = tf.placeholder(tf.string, [None])
  output_scores_tf = ken_lm_score_strings(handle=lm_tf, strings=input_strings_tf)
  with tf.Session() as session:
    output_scores = session.run(output_scores_tf, feed_dict={input_strings_tf: input_strings})
    print("input strings:", input_strings, "(sys.argv[1:])")
    print("output scores:", output_scores)

