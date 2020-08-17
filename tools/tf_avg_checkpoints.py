#!/usr/bin/env python3
# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to average values of variables in a list of checkpoint files.

This script is generic for any TF checkpoint. It is not specific to RETURNN.

Original code:
https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/avg_checkpoints.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy
import logging
import tensorflow as tf

import _setup_returnn_env  # noqa
import returnn.tf.compat as tf_compat
from returnn.util import better_exchook

better_exchook.install()

flags = tf_compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
  "checkpoints", "",
  "Comma-separated list of checkpoints to average.")
flags.DEFINE_integer(
  "num_last_checkpoints", 0,
  "Averages the last N saved checkpoints. If the checkpoints flag is set, this is ignored.")
flags.DEFINE_string(
  "prefix", "",
  "Prefix (e.g., directory) to append to each checkpoint.")
flags.DEFINE_string(
  "output_path", "/tmp/averaged.ckpt",
  "Path to output the averaged checkpoint to.")


def checkpoint_exists(path):
  """
  :param str path:
  :rtype: bool
  """
  return (
    tf_compat.v1.gfile.Exists(path) or tf_compat.v1.gfile.Exists(path + ".meta") or
    tf_compat.v1.gfile.Exists(path + ".index"))


def main(_):
  """
  Main entry.
  """
  _logger = logging.getLogger("tensorflow")
  _logger.setLevel("INFO")
  tf_compat.v1.logging.info("%s startup. TF version: %s" % (__file__, tf.__version__))

  if FLAGS.checkpoints:
    # Get the checkpoints list from flags and run some basic checks.
    checkpoints = [c.strip() for c in FLAGS.checkpoints.split(",")]
    checkpoints = [c for c in checkpoints if c]
    if not checkpoints:
      raise ValueError("No checkpoints provided for averaging.")
    if FLAGS.prefix:
      checkpoints = [FLAGS.prefix + c for c in checkpoints]
  else:
    assert FLAGS.num_last_checkpoints >= 1, "Must average at least one model"
    assert FLAGS.prefix, "Prefix must be provided when averaging last N checkpoints"
    checkpoint_state = tf.train.get_checkpoint_state(os.path.dirname(FLAGS.prefix))
    # Checkpoints are ordered from oldest to newest.
    checkpoints = checkpoint_state.all_model_checkpoint_paths[-FLAGS.num_last_checkpoints:]

  checkpoints = [c for c in checkpoints if checkpoint_exists(c)]
  if not checkpoints:
    if FLAGS.checkpoints:
      raise ValueError("None of the provided checkpoints exist. %s" % FLAGS.checkpoints)
    else:
      raise ValueError("Could not find checkpoints at %s" % os.path.dirname(FLAGS.prefix))

  # Read variables from all checkpoints and average them.
  tf_compat.v1.logging.info("Reading variables and averaging checkpoints:")
  for c in checkpoints:
    tf_compat.v1.logging.info("%s ", c)
  var_list = tf.train.list_variables(checkpoints[0])
  var_values, var_dtypes = {}, {}
  for (name, shape) in var_list:
    var_values[name] = numpy.zeros(shape)
  for checkpoint in checkpoints:
    reader = tf.train.load_checkpoint(checkpoint)
    for name in var_values:
      tensor = reader.get_tensor(name)
      if not isinstance(tensor, numpy.ndarray):  # e.g. int (scalar)
        tensor = numpy.array(tensor)
      assert isinstance(tensor, numpy.ndarray)
      var_dtypes[name] = tensor.dtype
      if isinstance(tensor.dtype, numpy.integer):
        var_values[name] = tensor  # just take last
      else:
        var_values[name] += tensor
    tf_compat.v1.logging.info("Read from checkpoint %s", checkpoint)
  for name in var_values:  # Average.
    if not isinstance(var_dtypes[name], numpy.integer):
      var_values[name] /= len(checkpoints)

  with tf_compat.v1.variable_scope(tf_compat.v1.get_variable_scope(), reuse=tf_compat.v1.AUTO_REUSE):
    tf_vars = [
      tf_compat.v1.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
      for v in var_values]
  placeholders = [tf_compat.v1.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
  assign_ops = [tf_compat.v1.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
  saver = tf_compat.v1.train.Saver(tf_compat.v1.all_variables())

  # Build a model consisting only of variables, set them to the average values.
  with tf_compat.v1.Session() as sess:
    sess.run(tf_compat.v1.global_variables_initializer())
    for p, assign_op, (name, value) in zip(placeholders, assign_ops, var_values.items()):
      sess.run(assign_op, {p: value})
    # Use the built saver to save the averaged checkpoint.
    saver.save(sess, FLAGS.output_path)

  tf_compat.v1.logging.info("Averaged checkpoints saved in %s", FLAGS.output_path)


if __name__ == "__main__":
  tf_compat.v1.app.run()
