"""
tests for MultiProcDataset
"""

from __future__ import annotations
import _setup_test_env  # noqa
from typing import List
import sys
import unittest
from returnn.util import better_exchook
from returnn.datasets.basic import Dataset, DatasetSeq, init_dataset
from returnn.datasets.multi_proc import MultiProcDataset
from test_HDFDataset import generate_hdf_from_other


def dummy_iter_dataset(dataset: Dataset) -> List[DatasetSeq]:
    """
    :param Dataset dataset:
    :return: seqs
    """
    dataset.init_seq_order(epoch=1)
    data_keys = dataset.get_data_keys()
    seq_idx = 0
    seqs = []
    while dataset.is_less_than_num_seqs(seq_idx):
        dataset.load_seqs(seq_idx, seq_idx + 1)
        data = {}
        for key in data_keys:
            data[key] = dataset.get_data(seq_idx=seq_idx, key=key)
        tag = dataset.get_tag(seq_idx)
        seq = DatasetSeq(seq_idx=seq_idx, seq_tag=tag, features=data)
        seqs.append(seq)
        seq_idx += 1
    print("Iterated through %r, num seqs %i" % (dataset, seq_idx))
    return seqs


def compare_dataset_seqs(seqs1: List[DatasetSeq], seqs2: List[DatasetSeq]):
    """
    :param list[DatasetSeq] seqs1:
    :param list[DatasetSeq] seqs2:
    """
    assert len(seqs1) == len(seqs2)
    for i, (seq1, seq2) in enumerate(zip(seqs1, seqs2)):
        assert seq1.seq_idx == seq2.seq_idx == i
        assert seq1.seq_tag == seq2.seq_tag, f"seq1 tag {seq1.seq_tag!r} != seq2 tag {seq2.seq_tag!r} for seq idx {i}"
        assert set(seq1.features.keys()) == set(seq2.features.keys())
        for key in seq1.features.keys():
            assert seq1.features[key].shape == seq2.features[key].shape
            assert seq1.features[key].dtype == seq2.features[key].dtype
            assert (seq1.features[key] == seq2.features[key]).all()


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
