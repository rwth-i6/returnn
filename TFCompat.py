
"""
This module provides some helpers to support earlier versions of TF 1 and also TF 2.
In all cases, we use graph-mode execution, i.e. we will disable eager-mode.

For an initial draft about TF 2 support, see here:
https://github.com/rwth-i6/returnn/issues/283

Also see this for a similar (but different) approach:
https://github.com/tensorflow/lingvo/blob/master/lingvo/compat.py
"""

from Util import attr_chain
import tensorflow as tf

if attr_chain(tf, ["compat", "v1"]):
  tf.compat.v1.disable_eager_execution()
  tf.compat.v1.disable_v2_tensorshape()
  # tf.compat.v1.disable_v2_behavior()  -- not sure on this

if not attr_chain(tf, ["compat", "v1"]):
  v1 = tf
  v2 = None
else:
  # PyCharm type-inference will take the latest reference,
  # so this `else` branch should lead us to a valid reference for "modern" TF versions (TF >=1.14, or TF 2).
  v1 = tf.compat.v1
  v2 = tf.compat.v2
