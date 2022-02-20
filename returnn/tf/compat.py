
"""
This module provides some helpers to support earlier versions of TF 1 and also TF 2.
In all cases, we use graph-mode execution, i.e. we will disable eager-mode.

For an initial draft about TF 2 support, see here:
https://github.com/rwth-i6/returnn/issues/283

Also see this for a similar (but different) approach:
https://github.com/tensorflow/lingvo/blob/master/lingvo/compat.py
"""

import tensorflow as tf
import os as _os

if not getattr(tf, "compat", None) or not getattr(tf.compat, "v2", None):
  v1 = tf
  v2 = None
else:
  # PyCharm type-inference will take the latest reference,
  # so this `else` branch should lead us to a valid reference for "modern" TF versions (TF >=1.14, or TF 2).
  v1 = tf.compat.v1
  v2 = tf.compat.v2

if v2 and tf.__version__.startswith("2."):
  tf.compat.v1.disable_eager_execution()
  tf.compat.v1.disable_v2_tensorshape()
  # There are still issues with control flow v2,
  # so we do not enable it yet by default.
  if _os.environ.get("TF_ENABLE_COND_V2", "") == "1":
    # https://github.com/tensorflow/tensorflow/issues/54458
    tf.compat.v1.experimental.output_all_intermediates(True)
  else:
    tf.compat.v1.disable_control_flow_v2()
  # There is also disable_v2_behavior() which disables all the TF2 behavior,
  # including control flow, eager execution, etc.,
  # but we want to make use of TF2 features as much as possible.

try:
  import tensorflow.contrib
  have_contrib = True
except ImportError:
  have_contrib = False
