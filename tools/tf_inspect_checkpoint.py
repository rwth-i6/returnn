#!/usr/bin/env python3

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
This is mostly a copy of:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/inspect_checkpoint.py

A simple script for inspect checkpoint files.

We extended it a bit.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _setup_returnn_env  # noqa
import sys
import argparse
import numpy

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import returnn.tf.compat as tf_compat

FLAGS = None


def print_tensor(v):
  """
  :param numpy.ndarray v:
  """
  print(v)
  # See :func:`variable_scalar_summaries_dict`.
  mean = numpy.mean(v)
  print("mean:", mean)
  print("stddev:", numpy.sqrt(numpy.mean(numpy.square(v - mean))))
  print("rms:", numpy.sqrt(numpy.mean(numpy.square(v))))
  print("min:", numpy.min(v))
  print("max:", numpy.max(v))


def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors):
  """Prints tensors in a checkpoint file.

  If no `tensor_name` is provided, prints the tensor names and shapes
  in the checkpoint file.

  If `tensor_name` is provided, prints the content of the tensor.

  Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
    all_tensors: Boolean indicating whether to print all tensors.
  """
  try:
    reader = tf_compat.v1.train.NewCheckpointReader(file_name)
    if all_tensors:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        print("tensor_name: ", key)
        print(reader.get_tensor(key))
    elif not tensor_name:
      print(reader.debug_string().decode("utf-8"))
    else:
      print("tensor_name: ", tensor_name)
      v = reader.get_tensor(tensor_name)
      print_tensor(v)
      if tensor_name.endswith("/Adam") and reader.has_tensor(tensor_name + "_1"):
        # https://github.com/tensorflow/tensorflow/blob/03beb65cecbc1e49ea477bca7f54543134b31d53/tensorflow/core/kernels/training_ops_gpu.cu.cc
        print("Guessing Adam m/v")
        v2 = reader.get_tensor(tensor_name + "_1")
        eps = 1e-8
        print("Adam update (m / (eps + sqrt(v))) with eps=%r:" % eps)
        print_tensor(v / (eps + numpy.sqrt(v2)))
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed with SNAPPY.")
    if "Data loss" in str(e) and (any([e in file_name for e in [".index", ".meta", ".data"]])):
      proposed_file = ".".join(file_name.split(".")[0:-1])
      v2_file_error_template = """
It's likely that this is a V2 checkpoint and you need to provide the filename
*prefix*.  Try removing the '.' and extension.  Try:
inspect checkpoint --file_name = {}"""
      print(v2_file_error_template.format(proposed_file))


def parse_numpy_printoption(kv_str):
  """Sets a single numpy printoption from a string of the form 'x=y'.

  See documentation on numpy.set_printoptions() for details about what values
  x and y can take. x can be any option listed there other than 'formatter'.

  Args:
    kv_str: A string of the form 'x=y', such as 'threshold=100000'

  Raises:
    argparse.ArgumentTypeError: If the string couldn't be used to set any
        nump printoption.
  """
  k_v_str = kv_str.split("=", 1)
  if len(k_v_str) != 2 or not k_v_str[0]:
    raise argparse.ArgumentTypeError("'%s' is not in the form k=v." % kv_str)
  k, v_str = k_v_str
  printoptions = numpy.get_printoptions()
  if k not in printoptions:
    raise argparse.ArgumentTypeError("'%s' is not a valid printoption." % k)
  v_type = type(printoptions[k])
  if printoptions[k] is None:
    raise argparse.ArgumentTypeError(
      "Setting '%s' from the command line is not supported." % k)
  try:
    v = v_type(v_str) if v_type is not bool else flags.BooleanParser().parse(v_str)
  except ValueError as e:
    raise argparse.ArgumentTypeError(str(e))
  numpy.set_printoptions(**{k: v})


# noinspection PyUnusedLocal
def main(unused_argv):
  """
  Main entry:
  """
  if not FLAGS.file_name:
    print(
      "Usage: inspect_checkpoint --file_name=checkpoint_file_name "
      "[--tensor_name=tensor_to_print]")
    sys.exit(1)
  else:
    print_tensors_in_checkpoint_file(FLAGS.file_name, FLAGS.tensor_name, FLAGS.all_tensors)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
    "--file_name", type=str, default="", help="Checkpoint filename. "
    "Note, if using Checkpoint V2 format, file_name is the "
    "shared prefix between all files in the checkpoint.")
  parser.add_argument(
    "--tensor_name",
    type=str, default="",
    help="Name of the tensor to inspect")
  parser.add_argument(
    "--all_tensors",
    nargs="?",
    const=True,
    type=bool,
    default=False,
    help="If True, print the values of all the tensors.")
  parser.add_argument(
    "--printoptions",
    nargs="*",
    type=parse_numpy_printoption,
    help="Argument for numpy.set_printoptions(), in the form 'k=v'.")
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
