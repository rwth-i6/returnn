"""
This provides ops as wrappers around OpenFst.
"""

import os
import platform
import tensorflow as tf

returnn_dir = os.path.dirname(os.path.abspath(__file__))
openfst_dir = returnn_dir + "/extern/openfst"


def get_fst(filename):
    """
    :param str filename: to OpenFst file
    :return: TF resource handle representing the FST
    :rtype: tf.Tensor
    """
    return get_tf_mod().open_fst_load(filename=filename)


def fst_transition(fst_handle, states, inputs):
    """
    :param tf.Tensor fst_handle: via :func:`get_fst`
    :param tf.Tensor states: [batch], int32
    :param tf.Tensor inputs: [batch], int32
    :return: (next_states, output_labels, weights). next_states can be -1 if invalid. all are shape [batch].
    :rtype: (tf.Tensor,tf.Tensor,tf.Tensor)
    """
    return get_tf_mod().open_fst_transition(handle=fst_handle, states=states, inputs=inputs)


# https://www.tensorflow.org/guide/extend/op
# Also see TFUtil.TFArrayContainer for TF resources.
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.h
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.h
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_types.h
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/strings/str_util.h
# https://github.com/kaldi-asr/kaldi/blob/master/src/tfrnnlm/tensorflow-rnnlm.h
_src_code = """
#include <exception>
#include <limits>
#include <fst/fstlib.h>

// Defined by OpenFst and also by TensorFlow.
// Also: https://github.com/kaldi-asr/kaldi/blob/master/src/tfrnnlm/tensorflow-rnnlm.h
#undef LOG
#undef VLOG
#undef CHECK
#undef CHECK_EQ
#undef CHECK_LT
#undef CHECK_GT
#undef CHECK_LE
#undef CHECK_GE
#undef CHECK_NE
#undef DCHECK
#undef DCHECK_EQ
#undef DCHECK_LT
#undef DCHECK_GT
#undef DCHECK_LE
#undef DCHECK_GE
#undef DCHECK_NE

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/version.h"

using namespace tensorflow;


REGISTER_OP("OpenFstLoad")
.Attr("filename: string")
.Attr("container: string = ''")
.Attr("shared_name: string = ''")
.Output("handle: resource")
.SetIsStateful()
.SetShapeFn(shape_inference::ScalarShape)
.Doc("OpenFstLoad: loads FST, creates TF resource, persistent across runs in the session");


REGISTER_OP("OpenFstTransition")
.Input("handle: resource")
.Input("states: int32")
.Input("inputs: int32")
.Output("next_states: int32")
.Output("output_labels: int32")
.Output("weights: float32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(1));
  c->set_output(1, c->input(1));
  c->set_output(2, c->input(1));
  return Status();
})
.Doc("OpenFstTransition: performs a transition");


struct OpenFstInstance : public ResourceBase {
  typedef fst::StdArc Arc;  // FSTs are usually saved using this type, and we have to use the same
  typedef fst::VectorFst<Arc> Fst;

  explicit OpenFstInstance(const string& filename)
      : filename_(filename), fst_(Fst::Read(filename)) {
    if(!fst_)
      throw std::runtime_error("failed to load FST; see stdout for errors");
  }

  virtual ~OpenFstInstance() {
    delete fst_;
  }

  string DebugString()
#if (TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 14) || (TF_MAJOR_VERSION > 1)
const
#endif
  override {
    return strings::StrCat("OpenFstInstance[", filename_, "]");
  }

  // This assumes a deterministic FST, i.e. it either has a single matching arc, or none.
  bool transition(int curState, int inputLabel, int* nextState, int* outputLabel, float* weight) {
    fst::Matcher<Fst> matcher(fst_, fst::MATCH_INPUT);
    if(curState >= 0)
      matcher.SetState(curState);
    if(curState >= 0 && matcher.Find(inputLabel)) {
      const Arc& arc = matcher.Value();
      *nextState = arc.nextstate;
      *outputLabel = arc.olabel;
      *weight = arc.weight.Value();
      return true;
    }
    else {
      *nextState = -1;
      *outputLabel = -1;
      *weight = -std::numeric_limits<float>::infinity();
      return false;
    }
  }

  const string filename_;
  mutex mu_;
  Fst* fst_;
};


// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_op_kernel.h
// TFUtil.TFArrayContainer
class OpenFstLoadOp : public ResourceOpKernel<OpenFstInstance> {
 public:
  explicit OpenFstLoadOp(OpKernelConstruction* context)
      : ResourceOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filename", &filename_));
  }

 private:
  virtual bool IsCancellable() const { return false; }
  virtual void Cancel() {}

  Status CreateResource(OpenFstInstance** ret) override {
    try {
      *ret = new OpenFstInstance(filename_);
    } catch (std::exception& exc) {
      return errors::Internal("Could not load OpenFst ", filename_, ", exception: ", exc.what());
    }
    if(*ret == nullptr)
      return errors::ResourceExhausted("Failed to allocate");
    return Status();
  }

  Status VerifyResource(OpenFstInstance* fst) override {
    if(fst->filename_ != filename_)
      return errors::InvalidArgument("Filename mismatch: expected ", filename_,
                                     " but got ", fst->filename_, ".");
    return Status();
  }

  string filename_;
};

REGISTER_KERNEL_BUILDER(Name("OpenFstLoad").Device(DEVICE_CPU), OpenFstLoadOp);


class OpenFstTransitionOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* context) override {
    OpenFstInstance* fst;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "handle", &fst));
    core::ScopedUnref unref(fst);

    const Tensor& states_tensor = context->input(1);
    auto states_flat = states_tensor.flat<int32>();

    const Tensor& inputs_tensor = context->input(2);
    auto inputs_flat = inputs_tensor.flat<int32>();

    OP_REQUIRES(
      context,
      TensorShapeUtils::IsVector(states_tensor.shape()) &&
      TensorShapeUtils::IsVector(inputs_tensor.shape()) &&
      states_flat.size() == inputs_flat.size(),
      errors::InvalidArgument(
        "Shape mismatch. states ", states_tensor.shape().DebugString(),
        " vs inputs ", inputs_tensor.shape().DebugString()));

    Tensor* output_next_states_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, states_tensor.shape(), &output_next_states_tensor));
    auto output_next_states_flat = output_next_states_tensor->flat<int32>();
    Tensor* output_output_labels_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, states_tensor.shape(), &output_output_labels_tensor));
    auto output_output_labels_flat = output_output_labels_tensor->flat<int32>();
    Tensor* output_weights_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, states_tensor.shape(), &output_weights_tensor));
    auto output_weights_flat = output_weights_tensor->flat<float>();

    for(int i = 0; i < inputs_flat.size(); ++i) {
      fst->transition(
        states_flat(i),
        inputs_flat(i),
        &output_next_states_flat(i),
        &output_output_labels_flat(i),
        &output_weights_flat(i));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("OpenFstTransition").Device(DEVICE_CPU), OpenFstTransitionOp);

"""


def openfst_checked_out():
    """
    :return: whether the Git submodule is checked out
    :rtype: bool
    """
    return os.path.exists("%s/src/include/fst/fst.h" % openfst_dir)


_tf_mod = None


def get_tf_mod(verbose=False):
    """
    :param bool verbose:
    :return: module
    """
    global _tf_mod
    if _tf_mod:
        return _tf_mod
    from glob import glob
    from returnn.tf.util.basic import OpCodeCompiler

    # Collect files.
    assert openfst_checked_out(), "submodule in %r not checked out?" % openfst_dir
    files = glob("%s/src/lib/*.cc" % openfst_dir)
    assert files, "submodule in %r not checked out?" % openfst_dir
    files = sorted(files)  # make somewhat deterministic
    libs = []
    if platform.system() != "Darwin":
        libs.append("rt")

    # Put code all together in one big blob.
    src_code = ""
    for fn in files:
        f_code = open(fn).read()
        f_code = "".join([x for x in f_code if ord(x) < 128])  # enforce ASCII
        src_code += "\n// ------------ %s : BEGIN { ------------\n" % os.path.basename(fn)
        # https://gcc.gnu.org/onlinedocs/cpp/Line-Control.html#Line-Control
        src_code += '#line 1 "%s"\n' % os.path.basename(fn)
        src_code += f_code
        src_code += "\n// ------------ %s : END } --------------\n\n" % os.path.basename(fn)
    src_code += "\n\n// ------------ our code now: ------------\n\n"
    src_code += '#line 1 "returnn/tf/util/open_fst.py:_src_code"\n'
    src_code += _src_code

    compiler = OpCodeCompiler(
        base_name="OpenFst",
        code_version=1,
        code=src_code,
        include_paths=("%s/src/include" % openfst_dir,),
        c_macro_defines={
            "NDEBUG": 1,  # https://github.com/tensorflow/tensorflow/issues/17316
        },
        ld_flags=["-l%s" % lib for lib in libs],
        is_cpp=True,
        use_cuda_if_available=False,
        verbose=verbose,
    )
    tf_mod = compiler.load_tf_module()
    assert hasattr(tf_mod, "open_fst_transition"), "content of mod: %r" % (dir(tf_mod),)
    _tf_mod = tf_mod
    return tf_mod


def _demo():
    def _make_int_list(s):
        """
        :param str s:
        :rtype: list[int]
        """
        return [int(s_) for s_ in s.split(",")]

    from returnn.util import better_exchook

    better_exchook.install()
    from argparse import ArgumentParser

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--states", type=_make_int_list, default=[0])
    arg_parser.add_argument("--inputs", type=_make_int_list, default=[0])
    # Other example FST files can be found online, e.g.: https://github.com/placebokkk/gofst/tree/master/ex01
    # Or: https://github.com/zh794390558/deeplearning/tree/master/kaldi/fst/data/test
    arg_parser.add_argument("--fst", default=returnn_dir + "/tests/lexicon_opt.fst")
    args = arg_parser.parse_args()
    # Try to compile now.
    get_tf_mod(verbose=True)
    # Some demo.
    assert os.path.exists(args.fst)
    fst_tf = get_fst(filename=args.fst)
    states_tf = tf.compat.v1.placeholder(tf.int32, [None])
    inputs_tf = tf.compat.v1.placeholder(tf.int32, [None])
    output_tf = fst_transition(fst_handle=fst_tf, states=states_tf, inputs=inputs_tf)
    with tf.compat.v1.Session() as session:
        out_next_states, out_labels, out_scores = session.run(
            output_tf, feed_dict={states_tf: args.states, inputs_tf: args.inputs}
        )
        print("states:", args.states)
        print("inputs:", args.inputs)
        print("output next states:", out_next_states)
        print("output labels:", out_labels)
        print("output scores:", out_scores)


if __name__ == "__main__":
    _demo()
