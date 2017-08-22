#!/usr/bin/env python3

"""
This will dump some content from an event log, i.e. summaries/events written via `tf.summary.FileWriter`,
which will be done by our TFEngine.

Also see:
https://stackoverflow.com/questions/37304461/tensorflow-importing-data-from-a-tensorboard-tfevent-file

    tensorboard --inspect --event_file events.out.tfevents...

Usage:

    tf_inspect_summary_log.py events.out.tfevents...

"""

from __future__ import print_function

import tensorflow as tf
import os
import sys

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.append(returnn_dir)

from argparse import ArgumentParser
import numpy


def main():
  argparser = ArgumentParser()
  argparser.add_argument("file", help="e.g. events.out.tfevents...")
  argparser.add_argument("--tag", default="objective/loss", help="default is 'objective/loss'")
  args = argparser.parse_args()

  print("file: %s" % args.file)
  print("tag: %s" % args.tag)
  for e in tf.train.summary_iterator(args.file):
    for v in e.summary.value:
      if v.tag == args.tag:
        print("step %i: %r" % (e.step, v.simple_value))
  print("done")


if __name__ == "__main__":
  main()

