from __future__ import annotations

import _setup_test_env  # noqa
import sys
import unittest
import torch
import torch.utils.data.datapipes as dp
from torchdata.dataloader2 import DataLoader2

from returnn.datasets.basic import Dataset
from returnn.datasets.generating import Task12AXDataset
from returnn.torch.data import pipeline as data_pipeline
from returnn.torch.data import returnn_dataset_wrapper
from returnn.util import better_exchook


def get_loader_from_returnn_dataset(dataset: Dataset, mp_manager: torch.multiprocessing.Manager) -> DataLoader2:
    # Follow mostly similar logic as in the PT engine.

    epoch_mp_shared = mp_manager.Value("i", 0)
    epoch_mp_shared.value = 1
    reset_callback = returnn_dataset_wrapper.ReturnnDatasetResetMpSharedEpochCallback(
        dataset=dataset, epoch_mp_shared=epoch_mp_shared
    )

    wrapped_dataset = returnn_dataset_wrapper.ReturnnDatasetIterDataPipe(dataset, reset_callback=reset_callback)

    batch_size = 5
    max_seqs = 2
    batches_dataset = data_pipeline.BatchingIterDataPipe(wrapped_dataset, batch_size=batch_size, max_seqs=max_seqs)
    batches_dataset = dp.iter.Collator(batches_dataset, collate_fn=data_pipeline.collate_batch)

    # Test different ways to deepcopy/serialize the dataset.
    # This is what DataLoader2 also would do, although DataLoader2 also uses dill as a fallback,
    # if it is available.
    # Dill is not always available though,
    # so it is important that we make sure that it also works without dill.

    from copy import deepcopy

    deepcopy(batches_dataset)

    import pickle

    pickle.loads(pickle.dumps(batches_dataset))

    loader = DataLoader2(batches_dataset)
    return loader


def test_pipeline_serialization():
    dataset = Task12AXDataset(num_seqs=1000)

    mp_manager = torch.multiprocessing.Manager()
    loader = get_loader_from_returnn_dataset(dataset, mp_manager)

    c = 0
    n = 3
    for batch in loader:
        print(batch)
        c += 1
        if c >= n:
            break

    assert c == n


def test_HDFDataset():
    # https://github.com/rwth-i6/returnn/issues/1281
    from test_HDFDataset import generate_hdf_from_other, HDFDataset

    hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 23})
    hdf_dataset = HDFDataset(files=[hdf_fn], cache_byte_size=0)

    mp_manager = torch.multiprocessing.Manager()
    loader = get_loader_from_returnn_dataset(hdf_dataset, mp_manager)
    c = 0
    n = 3
    for batch in loader:
        print(batch)
        c += 1
        if c >= n:
            break

    assert c == n


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
