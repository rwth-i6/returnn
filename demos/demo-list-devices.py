#!/usr/bin/env python3

"""
List all CPU/GPU devices which are accessible to TensorFlow.
"""

import os
from pprint import pprint
from argparse import ArgumentParser
from tensorflow.python.client import device_lib

import _setup_returnn_env  # noqa
import returnn.tf.compat as tf_compat
from returnn.tf.util.basic import get_device_attr, setup_tf_thread_pools
from returnn.tf.util.basic import print_available_devices, get_tf_list_local_devices


def dump_devs(tf_session_opts, use_device_lib=False, filter_gpu=True):
  """
  :param dict[str] tf_session_opts:
  :param bool use_device_lib:
  :param bool filter_gpu:
  """
  s = os.environ.get("CUDA_VISIBLE_DEVICES", None)
  cuda_num_visible = None
  if s is not None:
    cuda_num_visible = len(s.split(","))
  tf_num_visible = None
  if tf_session_opts.get("gpu_options", {}).get("visible_device_list"):
    tf_num_visible = len(tf_session_opts.get("gpu_options", {}).get("visible_device_list").split(","))
  if use_device_lib:
    devs = list(device_lib.list_local_devices())
  else:
    devs = get_tf_list_local_devices()
  if filter_gpu:
    devs = [dev for dev in devs if dev.device_type == 'GPU']
  # TF, or more likely the CUDA driver, will cache the list_local_devices internally after the first call.
  # Thus even when we update CUDA_VISIBLE_DEVICES, it will not notice that.
  # get_tf_list_local_devices will also cache.
  print("num devs %i, CUDA num visible %r, TF num visible %r" % (len(devs), cuda_num_visible, tf_num_visible))
  print("devs:")
  pprint(devs)
  with tf_compat.v1.Session(config=tf_compat.v1.ConfigProto(**tf_session_opts)) as session:
    for dev in devs:
      print("dev name:", dev.name)
      print("dev attribs:", session.run(get_device_attr(dev.name)))


def main():
  """
  Main entry.
  """
  arg_parser = ArgumentParser()
  arg_parser.add_argument("--try_subsets", action="store_true")
  arg_parser.add_argument("--visible_device_list")
  arg_parser.add_argument("--use_device_lib", action="store_true")
  args = arg_parser.parse_args()

  orig_cuda_visible_devs_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
  print("original CUDA_VISIBLE_DEVICES:", orig_cuda_visible_devs_str)

  tf_session_opts = {}
  if args.visible_device_list:
    tf_session_opts.setdefault("gpu_options", {})["visible_device_list"] = args.visible_device_list
    print("Using TF gpu_options.visible_device_list %r" % args.visible_device_list)

  setup_tf_thread_pools(tf_session_opts=tf_session_opts)
  print_available_devices(tf_session_opts=tf_session_opts)
  dump_devs(tf_session_opts=tf_session_opts, use_device_lib=args.use_device_lib, filter_gpu=False)

  if args.try_subsets:
    print("Trying subsets of CUDA_VISIBLE_DEVICES to see whether list_local_devices is cached.")
    cuda_visible_devs_str = orig_cuda_visible_devs_str
    while cuda_visible_devs_str:
      cuda_visible_devs_str = ",".join(cuda_visible_devs_str.split(",")[:-1])
      print("set CUDA_VISIBLE_DEVICES:", cuda_visible_devs_str)
      os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devs_str
      dump_devs(tf_session_opts=tf_session_opts, use_device_lib=args.use_device_lib)
    os.environ["CUDA_VISIBLE_DEVICES"] = orig_cuda_visible_devs_str
    print("Recovered original CUDA_VISIBLE_DEVICES")
    dump_devs(tf_session_opts=tf_session_opts, use_device_lib=args.use_device_lib)

  print("Quit.")


if __name__ == "__main__":
  from returnn.util import better_exchook
  better_exchook.install()
  main()
