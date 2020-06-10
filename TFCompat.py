
"""
This module provides some helpers to support earlier versions of TF 1 and also TF 2.
In all cases, we use graph-mode execution, i.e. we will disable eager-mode.
"""

from Util import attr_chain
import tensorflow as tf

if attr_chain(tf, ["compat", "v1"]):
  tf.compat.v1.disable_eager_execution()
  # tf.compat.v1.disable_v2_behavior()  -- not sure on this

try:
  # noinspection PyUnresolvedReferences
  Session = tf.Session
except AttributeError:
  Session = tf.compat.v1.Session

try:
  # noinspection PyUnresolvedReferences
  InteractiveSession = tf.InteractiveSession
except AttributeError:
  InteractiveSession = tf.compat.v1.InteractiveSession

try:
  from tensorflow.python.training.optimizer import Optimizer
except ImportError:
  Optimizer = tf.compat.v1.train.Optimizer

try:
  # noinspection PyUnresolvedReferences
  VariableScope = tf.VariableScope
except AttributeError:
  VariableScope = tf.compat.v1.VariableScope

try:
  # noinspection PyUnresolvedReferences
  Iterator = tf.data.Iterator
except AttributeError:
  Iterator = tf.compat.v1.data.Iterator
