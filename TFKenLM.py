"""
Uses KenLM (http://kheafield.com/code/kenlm/) (extern/kenlm) to read n-gram LMs (ARPA format),
and provides a TF op to use them.

"""

import sys
import os
import tensorflow as tf

returnn_dir = os.path.dirname(os.path.abspath(__file__))
kenlm_dir = returnn_dir + "/extern/kenlm"

# https://www.tensorflow.org/extend/adding_an_op
# Also see TFUitl.TFArrayContainer for TF resources.
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.h
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.h
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_types.h
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/strings/str_util.h
_src_code = """
#include <exception>
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


REGISTER_OP("KenLmAbsScoreStrings")
.Input("handle: resource")
.Input("strings: string")
.Output("scores: float32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(1));
  return Status::OK();
})
.Doc("KenLmScoreStrings: scores texts. returns in +log space (natural log, not base 10)");


REGISTER_OP("KenLmAbsScoreBpeStrings")
.Input("handle: resource")
.Input("bpe_merge_symbol: string")
.Input("strings: string")
.Output("scores: float32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(2));
  return Status::OK();
})
.Doc("KenLmAbsScoreBpeStrings: optionally BPE-merges, remove surrounding whitespaces and scores texts."
  " returns in +log space (natural log, not base 10)."
  " relative score, relative to previous text."
  );


REGISTER_OP("KenLmAbsScoreBpeStringsDense")
.Input("handle: resource")
.Input("bpe_merge_symbol: string")
.Input("strings: string")
.Input("labels: string")
.Output("scores: float32")
.Output("dense_scores: float32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(2));
  ::tensorflow::shape_inference::ShapeHandle out_shape;
  TF_RETURN_IF_ERROR(c->Concatenate(c->input(2), c->input(3), &out_shape));
  c->set_output(1, out_shape);
  return Status::OK();
})
.Doc("KenLmAbsScoreBpeStrings: optionally BPE-merges, remove surrounding whitespaces and scores texts."
  " returns in +log space (natural log, not base 10)."
  " relative score, relative to previous text."
  " dense output, for all possible succeeding labels.");


// https://github.com/kpu/kenlm/blob/master/lm/model.hh
// https://github.com/kpu/kenlm/blob/master/lm/virtual_interface.hh
// https://github.com/kpu/kenlm/blob/master/python/kenlm.pyx
struct KenLmModel : public ResourceBase {
  explicit KenLmModel(const string& filename)
      : filename_(filename), model_(filename.c_str()) {}

  float abs_score(const string& text) {
    float total = 0;
    mutex_lock l(mu_);
    lm::ngram::State state, out_state;
    model_.BeginSentenceWrite(&state);
    for(const string& word : tensorflow::str_util::Split(text, ' ')) {
      if(word.empty()) continue;
      auto word_idx = model_.BaseVocabulary().Index(word);
      total += model_.FullScore(state, word_idx, out_state).prob;
      state = out_state;
    }
    // KenLM returns score in +log10 space.
    // We want to return in (natural) +log space.
    // 10 ** x = e ** (x * log(10))
    return total * logf(10.);
  }

  // See comments below.
  // We expect that the text either ends with a space or not, i.e. "... word " or "... subword".
  float abs_score_dense(
        const string& text, const string& last_word_join,
        const TTypes<string>::ConstFlat labels, TTypes<float>::UnalignedFlat out_dense_scores) {
    assert(labels.size() == out_dense_scores.size());
    mutex_lock l(mu_);
    lm::ngram::State state, out_state;
    model_.BeginSentenceWrite(&state);
    // We expect that the text either ends with a space or not, i.e. "... word " or "... subword".
    // We split the text into words. In the first case, we would have an empty word at the end, otherwise not.
    std::vector<string> words = tensorflow::str_util::Split(text, ' ');
    float total_score = 0;
    string last_word = "";
    if(!words.empty()) {
      last_word = words[words.size() - 1];
      // Only up to the last word, which is either empty or a subword, which we join below.
      for(int i = 0; i < words.size() - 1; ++i) {
        const string& word = words[i];
        if(word.empty()) continue;
        auto word_idx = model_.BaseVocabulary().Index(word);
        total_score += model_.FullScore(state, word_idx, out_state).prob;
        state = out_state;
      }
    }
    for(int i = 0; i < labels.size(); ++i) {
      string word = last_word + labels(i);
      auto word_idx = model_.BaseVocabulary().Index(word);
      float score = model_.FullScore(state, word_idx, out_state).prob;
      out_dense_scores(i) = (total_score + score) * logf(10.);
    }
    // Return the score from the prev step.
    if(!last_word.empty()) {
      auto word_idx = model_.BaseVocabulary().Index(last_word + last_word_join);
      total_score += model_.FullScore(state, word_idx, out_state).prob;
    }
    return total_score * logf(10.);
  }

  string DebugString() override {
    return strings::StrCat("KenLmModel[", filename_, "]");
  }

  const string filename_;
  mutex mu_;
  lm::ngram::ProbingModel model_ GUARDED_BY(mu_);
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
    try {
      *ret = new KenLmModel(filename_);
    } catch (std::exception& exc) {
      return errors::Internal("Could not load KenLmModel ", filename_, ", exception: ", exc.what());
    }
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


class KenLmAbsScoreStringsOp : public OpKernel {
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
      output_flat(i) = lm->abs_score(input_flat(i));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("KenLmAbsScoreStrings").Device(DEVICE_CPU), KenLmAbsScoreStringsOp);


class KenLmAbsScoreBpeStringsOp : public OpKernel {
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

    OP_REQUIRES(context, context->input(1).NumElements() == 1,
      errors::InvalidArgument(
        "bpe_merge_symbol must be a single element but got shape ",
        context->input(1).shape().DebugString()));
    const string& bpe_merge_symbol = context->input(1).flat<string>()(0);

    const Tensor& input_tensor = context->input(2);
    auto input_flat = input_tensor.flat<string>();

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<float>();

    for(int i = 0; i < input_flat.size(); ++i) {
      std::string text = input_flat(i);
      if(!bpe_merge_symbol.empty())
        text = tensorflow::str_util::StringReplace(text, bpe_merge_symbol + " ", "", /* replace_all */ true);
      tensorflow::StringPiece sp(text);
      tensorflow::str_util::RemoveWhitespaceContext(&sp);
      text = std::string(sp.data(), sp.size());
      output_flat(i) = lm->abs_score(text);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("KenLmAbsScoreBpeStrings").Device(DEVICE_CPU), KenLmAbsScoreBpeStringsOp);


class KenLmAbsScoreBpeStringsDenseOp : public OpKernel {
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

    OP_REQUIRES(context, context->input(1).NumElements() == 1,
      errors::InvalidArgument(
        "bpe_merge_symbol must be a single element but got shape ",
        context->input(1).shape().DebugString()));
    const string& bpe_merge_symbol = context->input(1).flat<string>()(0);

    const Tensor& input_tensor = context->input(2);
    auto input_flat = input_tensor.flat<string>();

    const Tensor& labels_tensor = context->input(3);
    auto labels_flat = labels_tensor.flat<string>();

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<float>();

    Tensor* output_dense_tensor = NULL;
    TensorShape output_dense_shape(input_tensor.shape());
    output_dense_shape.AppendShape(labels_tensor.shape());
    OP_REQUIRES_OK(context, context->allocate_output(1, output_dense_shape, &output_dense_tensor));
    Tensor output_dense_flat_tensor;
    OP_REQUIRES(context,
      output_dense_flat_tensor.CopyFrom(
        *output_dense_tensor,
        TensorShape({input_tensor.NumElements(), labels_tensor.NumElements()})),
      errors::Internal("CopyFrom failed"));

    for(int i = 0; i < input_flat.size(); ++i) {
      string text = input_flat(i);
      if(!bpe_merge_symbol.empty())
        text = tensorflow::str_util::StringReplace(text, bpe_merge_symbol + " ", "", /* replace_all */ true);
      output_flat(i) = lm->abs_score_dense(
        text, bpe_merge_symbol, labels_flat, output_dense_flat_tensor.Slice(i, i + 1).unaligned_flat<float>());
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("KenLmAbsScoreBpeStringsDense").Device(DEVICE_CPU), KenLmAbsScoreBpeStringsDenseOp);

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
  """
  :param bool verbose:
  :return: module
  """
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
  assert files, "submodule in %r not checked out?" % kenlm_dir
  libs = ["z"]
  if platform.system() != 'Darwin':
    libs.append('rt')

  # Put code all together in one big blob.
  src_code = ""
  src_code += _kenlm_src_code_workarounds
  for fn in files:
    f_code = open(fn).read()
    f_code = ''.join([x for x in f_code if ord(x) < 128])  # enforce ASCII
    # We need to do some replacements to not clash symbol names.
    fn_short = os.path.basename(fn).replace(".", "_")
    for word in ["kConverter"]:
      f_code = f_code.replace(word, "%s_%s" % (fn_short, word))
    src_code += "\n// ------------ %s : BEGIN { ------------\n" % os.path.basename(fn)
    # https://gcc.gnu.org/onlinedocs/cpp/Line-Control.html#Line-Control
    src_code += "#line 1 \"%s\"\n" % os.path.basename(fn)
    src_code += f_code
    src_code += "\n// ------------ %s : END } --------------\n\n" % os.path.basename(fn)
  src_code += "\n\n// ------------ our code now: ------------\n\n"
  src_code += _src_code

  compiler = OpCodeCompiler(
    base_name="KenLM", code_version=1, code=src_code,
    include_paths=(kenlm_dir, kenlm_dir + "/util/double-conversion"),
    c_macro_defines={"NDEBUG": 1, "KENLM_MAX_ORDER": 6, "HAVE_ZLIB": 1},
    ld_flags=["-l%s" % lib for lib in libs],
    is_cpp=True, use_cuda_if_available=False,
    verbose=verbose)
  tf_mod = compiler.load_tf_module()
  assert hasattr(tf_mod, "ken_lm_abs_score_strings"), "content of mod: %r" % (dir(tf_mod),)
  _tf_mod = tf_mod
  return tf_mod


def ken_lm_load(filename):
  """
  :param str filename:
  :return: TF resource handle
  :rtype: tf.Tensor
  """
  return get_tf_mod().ken_lm_load_model(filename=filename)


def ken_lm_abs_score_strings(handle, strings):
  """
  :param tf.Tensor handle: TF resource handle returned by :func:`ken_lm_load`
  :param tf.Tensor strings: strings which are being scores. white-space delimited words.
  :return: same shape as `strings`, float32
  :rtype: tf.Tensor
  """
  return get_tf_mod().ken_lm_abs_score_strings(handle=handle, strings=strings)


def ken_lm_abs_score_bpe_strings(handle, bpe_merge_symbol, strings):
  """
  :param tf.Tensor handle: TF resource handle returned by :func:`ken_lm_load`
  :param str bpe_merge_symbol: e.g. "@@"
  :param tf.Tensor strings: strings which are being scores. white-space delimited words.
  :return: same shape as `strings`, float32
  :rtype: tf.Tensor
  """
  return get_tf_mod().ken_lm_abs_score_bpe_strings(
    handle=handle, bpe_merge_symbol=bpe_merge_symbol, strings=strings)


def ken_lm_abs_score_bpe_strings_dense(handle, bpe_merge_symbol, strings, labels):
  """
  :param tf.Tensor handle: TF resource handle returned by :func:`ken_lm_load`
  :param str bpe_merge_symbol: e.g. "@@"
  :param tf.Tensor strings: strings which are being scores. white-space delimited words.
  :param tf.Tensor|tf.Variable labels:
  :return: same shape as `strings`, float32
  :rtype: tf.Tensor
  """
  return get_tf_mod().ken_lm_abs_score_bpe_strings_dense(
    handle=handle, bpe_merge_symbol=bpe_merge_symbol, strings=strings, labels=labels)


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
  output_scores_tf = ken_lm_abs_score_strings(handle=lm_tf, strings=input_strings_tf)
  with tf.Session() as session:
    output_scores = session.run(output_scores_tf, feed_dict={input_strings_tf: input_strings})
    print("input strings:", input_strings, "(sys.argv[1:])")
    print("output scores:", output_scores)
