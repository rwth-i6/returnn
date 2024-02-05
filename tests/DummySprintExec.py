#!/usr/bin/env python

"""
This script will emulate a Sprint executable, so that we can use it for SprintDatasetBase.
This is useful for tests.
To generate data, we can use the GeneratingDataset code.
"""

from __future__ import annotations

import sys
from importlib import import_module

import _setup_test_env  # noqa
import returnn.datasets.generating as generating_dataset
from returnn.datasets import Dataset
from returnn.util.basic import ObjAsDict


class ArgParser:
    """
    Emulate the Sprint argument parser.
    """

    def __init__(self):
        self.args = {}

    def add(self, key, value):
        """
        :param str key:
        :param str value:
        """
        self.args[key] = value

    def get(self, key, default=None):
        """
        :param str key:
        :param str|T|None default:
        :rtype: str|T|None
        """
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
    """
    Main entry.
    """
    print("DummySprintExec init", argv)
    args = ArgParser()
    args.parse(argv[1:])

    if args.get("pymod-name"):
        sprint_api = import_module(args.get("pymod-name"))
    else:
        import returnn.sprint.extern_interface as sprint_api

    input_dim = int(args.get("feature-dimension"))
    assert input_dim > 0
    output_dim = int(args.get("trainer-output-dimension"))
    assert output_dim > 0
    sprint_config = args.get("pymod-config", "")
    target_mode = args.get("target-mode", "target-generic")
    sprint_api.init(inputDim=input_dim, outputDim=output_dim, config=sprint_config, targetMode=target_mode)

    if args.get("crnn-dataset"):
        dataset = eval(args.get("crnn-dataset"), {}, ObjAsDict(generating_dataset))
        assert isinstance(dataset, Dataset)
        assert dataset.num_inputs == input_dim
        assert dataset.num_outputs == {"classes": (output_dim, 1), "data": (input_dim, 2)}
        dataset.init_seq_order(epoch=1)

        seq_idx = 0
        while dataset.is_less_than_num_seqs(seq_idx):
            dataset.load_seqs(seq_idx, seq_idx + 1)
            features = dataset.get_data(seq_idx, "data")
            features = features.T  # Sprint-like
            kwargs = {"features": features}
            if target_mode == "target-generic":
                if "orth" in dataset.get_target_list():
                    kwargs["orthography"] = dataset.get_data(seq_idx, "orth")
                if "classes" in dataset.get_target_list():
                    kwargs["alignment"] = dataset.get_data(seq_idx, "classes")
                print("DummySprintExec seq_idx %i feedInputAndTarget(**%r)" % (seq_idx, kwargs))
                sprint_api.feedInputAndTarget(**kwargs)
            else:
                raise NotImplementedError("targetMode = %s" % target_mode)
            seq_idx += 1

    print("DummySprintExec exit")
    sprint_api.exit()


if __name__ == "__main__":
    from returnn.util import better_exchook

    better_exchook.install()
    main(sys.argv)
