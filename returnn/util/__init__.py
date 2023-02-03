"""
Basic utilities for RETURNN, such as some Numpy utilities, or system utilities, or other helpers.
Independent from other parts of RETURNN.
Also independent from TensorFlow or Theano (see :mod:`returnn.tf.util` or :mod:`returnn.theano.util`).
"""

# Some basic imports.
from .basic import BackendEngine, NumbersDict, BehaviorVersion
