"""
tests for MultiProcDataset
"""

from __future__ import annotations
import _setup_test_env  # noqa
import sys
import unittest
from returnn.util import better_exchook
from returnn.datasets.basic import init_dataset
from returnn.datasets.multi_proc import MultiProcDataset
from returnn.datasets.map import MapDatasetBase
from test_HDFDataset import generate_hdf_from_other
from test_Dataset import dummy_iter_dataset, compare_dataset_seqs


def test_MultiProcDataset_n1_b1_default():
    hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 23})
    hdf_dataset_dict = {"class": "HDFDataset", "files": [hdf_fn]}
    hdf_dataset = init_dataset(hdf_dataset_dict)
    hdf_dataset_seqs = dummy_iter_dataset(hdf_dataset)

    mp_dataset = MultiProcDataset(dataset=hdf_dataset_dict, num_workers=1, buffer_size=1)
    mp_dataset.initialize()
    mp_dataset_seqs = dummy_iter_dataset(mp_dataset)

    compare_dataset_seqs(hdf_dataset_seqs, mp_dataset_seqs)


def test_MultiProcDataset_n3_b5_shuffle():
    hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 23})
    hdf_dataset_dict = {"class": "HDFDataset", "files": [hdf_fn], "seq_ordering": "random"}
    hdf_dataset = init_dataset(hdf_dataset_dict)
    hdf_dataset_seqs = dummy_iter_dataset(hdf_dataset)

    mp_dataset = MultiProcDataset(dataset=hdf_dataset_dict, num_workers=3, buffer_size=5)
    mp_dataset.initialize()
    mp_dataset_seqs = dummy_iter_dataset(mp_dataset)

    compare_dataset_seqs(hdf_dataset_seqs, mp_dataset_seqs)


def test_MultiProcDataset_meta():
    hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 23})
    meta_dataset_dict = {
        "class": "MetaDataset",
        "data_map": {"classes": ("hdf", "classes"), "data": ("hdf", "data")},
        "datasets": {"hdf": {"class": "HDFDataset", "files": [hdf_fn]}},
    }
    meta_dataset = init_dataset(meta_dataset_dict)
    meta_dataset_seqs = dummy_iter_dataset(meta_dataset)

    mp_dataset = MultiProcDataset(dataset=meta_dataset_dict, num_workers=1, buffer_size=1)
    mp_dataset.initialize()
    mp_dataset_seqs = dummy_iter_dataset(mp_dataset)

    compare_dataset_seqs(meta_dataset_seqs, mp_dataset_seqs)


if __name__ == "__main__":
    better_exchook.install()
    if len(sys.argv) <= 1:
        for k, v in sorted(globals().items()):
            if k.startswith("test_"):
                print("-" * 40)
                print("Executing: %s" % k)
                try:
                    v()
                except unittest.SkipTest as exc:
                    print("SkipTest:", exc)
                print("-" * 40)
        print("Finished all tests.")
    else:
        assert len(sys.argv) >= 2
        for arg in sys.argv[1:]:
            print("Executing: %s" % arg)
            if arg in globals():
                globals()[arg]()  # assume function and execute
            else:
                eval(arg)  # assume Python code and execute
