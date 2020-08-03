
"""
This module provides some helpers to support earlier versions of TF 1 and also TF 2.
In all cases, we use graph-mode execution, i.e. we will disable eager-mode.

For an initial draft about TF 2 support, see here:
https://github.com/rwth-i6/returnn/issues/283

Also see this for a similar (but different) approach:
https://github.com/tensorflow/lingvo/blob/master/lingvo/compat.py
"""

import tensorflow as tf

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
  tf.compat.v1.disable_control_flow_v2()
  # tf.compat.v1.disable_v2_behavior()  -- not sure on this

try:
  import tensorflow.contrib
  have_contrib = True
except ImportError:
  have_contrib = False
