
"""
Like SprintErrorSignals.py but for TensorFlow.
"""

from SprintErrorSignals import SprintInstancePool
from Log import log
import tensorflow as tf


def py_get_sprint_automata_for_batch(sprint_opts, tags):
  """
  :param dict[str] sprint_opts:
  :param list[str] tags:
  :return: (edges, weights, start_end_states)
  :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
  """
  # Also see :class:`SprintAlignmentAutomataOp`.
  sprint_instance_pool = SprintInstancePool.get_global_instance(sprint_opts=sprint_opts)
  edges, weights, start_end_states = sprint_instance_pool.get_automata_for_batch(tags)
  return edges, weights, start_end_states


def get_sprint_automata_for_batch_op(sprint_opts, tags):
  """
  :param dict[str] sprint_opts:
  :param tf.Tensor tags: shape (batch,), of dtype string
  :return: (edges, weights, start_end_states). all together in one automaton.
    edges are of shape (4, num_edges), each (from, to, emission-idx, seq-idx), of dtype int32.
    weights are of shape (num_edges,), of dtype float32.
    start_end_states are of shape (2, batch), each (start,stop) state idx, batch = len(tags), of dtype int32.
  :rtype: (tf.Tensor, tf.Tensor, tf.Tensor)
  """

  def py_wrap_get_sprint_automata_for_batch(py_tags):
    return py_get_sprint_automata_for_batch(sprint_opts=sprint_opts, tags=py_tags)

  tags.set_shape((None,))  # (batch,)
  edges, weights, start_end_states = tf.py_func(
    py_wrap_get_sprint_automata_for_batch,
    [tags], [tf.int32, tf.float32, tf.int32],
    name="get_sprint_automata_for_batch")
  assert isinstance(edges, tf.Tensor)
  assert isinstance(weights, tf.Tensor)
  assert isinstance(start_end_states, tf.Tensor)
  edges.set_shape((4, None))  # (4, num_edges)
  weights.set_shape((None,))  # (num_edges,)
  start_end_states.set_shape((2, tags.get_shape().dims[1]))  # (2, batch)
  return edges, weights, start_end_states


def py_get_sprint_loss_and_error_signal(sprint_opts, log_posteriors, seq_lengths, seq_tags):
  """
  :param dict[str] sprint_opts:
  :param numpy.ndarray log_posteriors: 3d (time,batch,label)
  :param numpy.ndarray seq_lengths: 1d (batch)
  :param list[str] seq_tags: seq names
  :return: (loss, error_signal), error_signal has the same shape as posteriors. loss is a 1d-array (batch).
  :rtype: (numpy.ndarray, numpy.ndarray)
  """
  # Also see :class:`SprintErrorSigOp`.
  sprint_instance_pool = SprintInstancePool.get_global_instance(sprint_opts=sprint_opts)
  loss, error_signal = sprint_instance_pool.get_batch_loss_and_error_signal(
    log_posteriors=log_posteriors, seq_lengths=seq_lengths, tags=seq_tags)
  return loss, error_signal


def get_sprint_loss_and_error_signal(sprint_opts, log_posteriors, seq_lengths, seq_tags):
  """
  :param dict[str] sprint_opts:
  :param tf.Tensor log_posteriors: 3d (time,batch,label)
  :param tf.Tensor seq_lengths: 1d (batch,)
  :param tf.Tensor seq_tags: 1d (batch,), seq names
  :return: (loss, error_signal), error_signal has the same shape as posteriors. loss is a 1d-array (batch).
  :rtype: (tf.Tensor, tf.Tensor)
  """

  def py_wrap_get_sprint_loss_and_error_signal(py_log_posteriors, py_seq_lengths, py_seq_tags):
    return py_get_sprint_loss_and_error_signal(
      sprint_opts=sprint_opts, log_posteriors=py_log_posteriors, seq_lengths=py_seq_lengths, seq_tags=py_seq_tags)

  log_posteriors.set_shape((None, None, None))  # (time,batch,label)
  seq_lengths.set_shape((None,))  # (batch,)
  seq_tags.set_shape((None,))  # (batch,)
  loss, error_signal = tf.py_func(
    py_wrap_get_sprint_loss_and_error_signal,
    [log_posteriors, seq_lengths, seq_tags], [tf.float32, tf.float32],
    name="get_sprint_loss_and_error_signal")
  assert isinstance(loss, tf.Tensor)
  assert isinstance(error_signal, tf.Tensor)
  loss.set_shape((None,))  # (batch,)
  error_signal.set_shape(log_posteriors.get_shape())  # (time,batch,label)
  return loss, error_signal
