"""
Provides a RETURNN wrapper around `warp-transducer`:
  https://github.com/1ytic/warp-rna

Importing this module immediately compiles the library and TF module.
"""

import os
import typing
import tensorflow as tf
from tensorflow.python.framework import ops
from returnn.tf.util.basic import OpCodeCompiler


warprna_dir = os.path.dirname(os.path.abspath(__file__))
submodule_dir = os.path.join(warprna_dir, "warp-rna")
_tf_mod = None  # type: typing.Optional[typing.Any]


def is_checked_out():
  """
  Checks if the git submodule is checkout out.

  :rtype: bool
  """
  return os.path.isfile("%s/core.cu" % submodule_dir)


def init_warprna(verbose=False):
  """
  Initializes and compiles the library. Caches the TF module.

  :param bool verbose:
  """
  global _tf_mod
  if _tf_mod:
    return
  assert is_checked_out(), "submodule not checked out? Run `git submodule update --init --recursive`"

  src_files = [
    '%s/tensorflow_binding/src/warp_rna_op.cc' % submodule_dir,
    '%s/core.cu' % submodule_dir,
  ]
  src_code = ""
  for fn in src_files:
    f_code = open(fn).read()
    src_code += "\n// ------------ %s : BEGIN { ------------\n" % os.path.basename(fn)
    # https://gcc.gnu.org/onlinedocs/cpp/Line-Control.html#Line-Control
    src_code += "#line 1 \"%s\"\n" % os.path.basename(fn)
    src_code += f_code
    src_code += "\n// ------------ %s : END } --------------\n\n" % os.path.basename(fn)

  compiler = OpCodeCompiler(
    base_name="warprna_kernels", code_version=1, code=src_code,
    include_paths=(submodule_dir, os.path.dirname(submodule_dir)),
    c_macro_defines={"WARPRNA_ENABLE_GPU": 1},
    is_cpp=True, use_cuda_if_available=True,
    verbose=verbose)
  tf_mod = compiler.load_tf_module()
  assert hasattr(tf_mod, "WarpRNA"), "content of mod: %r" % (dir(tf_mod),)
  _tf_mod = tf_mod
  return tf_mod


def rna_loss(log_probs, labels, input_lengths, label_lengths, blank_label=0):
  """Computes the RNA loss between a sequence of activations and a
  ground truth labeling.
  Args:
      log_probs: A 4-D Tensor of floats.  The dimensions
                   should be (B, T, U, V), where B is the minibatch index,
                   T is the time index, U is the prediction network sequence
                   length, and V indexes over activations for each
                   symbol in the alphabet.
      labels: A 2-D Tensor of ints, a padded label sequences to make sure
                   labels for the minibatch are same length.
      input_lengths: A 1-D Tensor of ints, the number of time steps
                     for each sequence in the minibatch.
      label_lengths: A 1-D Tensor of ints, the length of each label
                     for each example in the minibatch.
      blank_label: int, the label value/index that the RNA
                   calculation should use as the blank label
  Returns:
      1-D float Tensor, the cost of each example in the minibatch
      (as negative log probabilities).
  * This class performs the softmax operation internally.
  * The label reserved for the blank symbol should be label 0.
  """
  init_warprna()
  loss, _ = _tf_mod.warp_rna(log_probs, labels, input_lengths,
                             label_lengths, tf.reduce_min(label_lengths), blank_label)
  return loss


@ops.RegisterGradient("WarpRNA")
def _warprna_loss_grad(op, grad_loss, _):
  grad = op.outputs[1]
  # NOTE since here we are batch first, cannot use _BroadcastMul
  grad_loss = tf.reshape(grad_loss, (-1, 1, 1, 1))
  return [grad_loss * grad, None, None, None, None]
