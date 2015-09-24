#!/usr/bin/env python

# This script will emulate a Sprint executable, so that we can use it for SprintDataset.
# This is useful for tests.
# To generate data, we can use the GeneratingDataset code.

import sys
import os
from importlib import import_module

# Add parent dir to Python path so that we can use GeneratingDataset and other CRNN code.
my_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.normpath(my_dir + "/..")
if parent_dir not in sys.path:
  sys.path += [parent_dir]

import GeneratingDataset
from Dataset import Dataset
from Util import ObjAsDict


class ArgParser:

  def __init__(self):
    self.args = {}

  def add(self, key, value):
    self.args[key] = value

  def get(self, key, default=None):
    return self.args.get(key, default)

  def parse(self, argv):
    """
    :type argv: list[str]
    """
    i = 0
    while i < len(argv):
      arg = argv[i]
      if arg.startswith("--"):
        key, value = arg[2:].split("=", 1)
        if key.startswith("*."):  # simple hack. example "--*.input-dim=5"
          key = key[2:]
        self.add(key, value)
      i += 1


def main(argv):
  print "DummySprintExec init", argv
  args = ArgParser()
  args.parse(argv[1:])

  if args.get("pymod-name"):
    SprintAPI = import_module(args.get("pymod-name"))
  else:
    import SprintExternInterface as SprintAPI

  inputDim = int(args.get("feature-dimension"))
  assert inputDim > 0
  outputDim = int(args.get("trainer-output-dimension"))
  assert outputDim > 0
  sprintConfig = args.get("pymod-config", "")
  targetMode = args.get("target-mode", "target-generic")
  SprintAPI.init(inputDim=inputDim, outputDim=outputDim,
                 config=sprintConfig, targetMode=targetMode)

  if args.get("crnn-dataset"):
    dataset = eval(args.get("crnn-dataset"), {}, ObjAsDict(GeneratingDataset))
    assert isinstance(dataset, Dataset)
    assert dataset.num_inputs == inputDim
    assert dataset.num_outputs == {"classes": [outputDim, 1], "data": [inputDim, 2]}
    dataset.init_seq_order(epoch=1)

    seq_idx = 0
    while dataset.is_less_than_num_seqs(seq_idx):
      dataset.load_seqs(seq_idx, seq_idx + 1)
      features = dataset.get_data(seq_idx, "data")
      features = features.T  # Sprint-like
      kwargs = {"features": features}
      if targetMode == "target-generic":
        if "orth" in dataset.get_target_list():
          kwargs["orthography"] = dataset.get_targets("orth", seq_idx)
        if "classes" in dataset.get_target_list():
          kwargs["alignment"] = dataset.get_targets("classes", seq_idx)
        SprintAPI.feedInputAndTarget(**kwargs)
      else:
        raise NotImplementedError("targetMode = %s" % targetMode)

  print "DummySprintExec exit"
  SprintAPI.exit()


if __name__ == "__main__":
  import better_exchook
  better_exchook.install()
  main(sys.argv)
