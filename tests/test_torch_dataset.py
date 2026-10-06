from __future__ import annotations

import os

import _setup_test_env  # noqa
from typing import Optional, Any, Dict
import sys
import unittest
from multiprocessing.managers import SyncManager
import torch
from torch.utils.data import DataLoader

from returnn.config import Config, get_global_config, global_config_ctx
from returnn.datasets.basic import init_dataset, Dataset, DatasetSeq
from returnn.datasets.generating import Task12AXDataset
from returnn.torch.data import pipeline as data_pipeline
from returnn.torch.data import returnn_dataset_wrapper
from returnn.util import better_exchook
from returnn.util.basic import BehaviorVersion
from returnn.util import multi_proc_manager_with_watchdog


def get_loader_from_returnn_dataset(
    dataset: Dataset, mp_manager: SyncManager, *, batch_size: int = 5, max_seqs: int = 2
) -> DataLoader:
    # Follow mostly similar logic as in the PT engine.

    epoch_mp_shared = mp_manager.Value("i", 0)
    epoch_mp_shared.value = 1
    reset_callback = returnn_dataset_wrapper.ReturnnDatasetResetMpSharedEpochCallback(
        dataset=dataset, epoch_mp_shared=epoch_mp_shared
    )

    wrapped_dataset = returnn_dataset_wrapper.ReturnnDatasetIterDataPipe(dataset, reset_callback=reset_callback)

    batches_dataset = data_pipeline.BatchingIterDataPipe(wrapped_dataset, batch_size=batch_size, max_seqs=max_seqs)

    # Test different ways to deepcopy/serialize the dataset.
    # This is what DataLoader2 also would do, although DataLoader2 also uses dill as a fallback,
    # if it is available.
    # Dill is not always available though,
    # so it is important that we make sure that it also works without dill.

    from copy import deepcopy

    deepcopy(batches_dataset)

    import pickle

    pickle.loads(pickle.dumps(batches_dataset))

    return data_pipeline.create_data_loader_from_batches(batches_dataset, {"num_workers": 1})


def test_pipeline_serialization():
    dataset = Task12AXDataset(num_seqs=1000)

    mp_manager = multi_proc_manager_with_watchdog.create_manager()
    loader = get_loader_from_returnn_dataset(dataset, mp_manager)

    c = 0
    n = 3
    for batch in loader:
        print(batch)
        c += 1
        if c >= n:
            break

    assert c == n


class _DummyDatasetWithChecks(Task12AXDataset):
    def __init__(
        self,
        *,
        parent_pid: int,
        num_seqs: int = 1000,
        check_in_global_config: Optional[Dict[str, Any]] = None,
        check_behavior_version: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(num_seqs=num_seqs, **kwargs)
        self.parent_pid = parent_pid
        self.check_in_global_config = check_in_global_config
        self.check_behavior_version = check_behavior_version

    def generate_seq(self, seq_idx: int) -> DatasetSeq:
        """generate seq"""
        seq = super().generate_seq(seq_idx)
        assert os.getpid() != self.parent_pid  # check we are in a subproc
        if self.check_in_global_config:
            config = get_global_config()
            for k, v in self.check_in_global_config.items():
                assert config.typed_dict[k] == v
        if self.check_behavior_version is not None:
            assert BehaviorVersion.get() == self.check_behavior_version, (
                f"behavior_version {BehaviorVersion.get()} in dataset worker proc,"
                f" expected {self.check_behavior_version} as in the parent proc"
            )
        return seq


def test_correct_global_config():
    config = Config({"test_value": 43})
    with global_config_ctx(config):
        dataset = _DummyDatasetWithChecks(parent_pid=os.getpid(), check_in_global_config={"test_value": 43})

        mp_manager = multi_proc_manager_with_watchdog.create_manager()
        loader = get_loader_from_returnn_dataset(dataset, mp_manager)

        c = 0
        n = 3
        for batch in loader:
            print(batch)
            c += 1
            if c >= n:
                break

        assert c == n


def test_correct_behavior_version():
    # The dataset lives in a spawned worker proc, and the behavior version must be visible there:
    # code such as DistributeFilesDataset sharding decides via BehaviorVersion.get().
    # https://github.com/rwth-i6/returnn/issues/1738
    behavior_version_orig_state = BehaviorVersion._get_state()
    try:
        BehaviorVersion._reset()
        BehaviorVersion.set(26)
        config = Config({"behavior_version": 26, "propagate_behavior_version_to_subprocs": True})
        with global_config_ctx(config):
            dataset = _DummyDatasetWithChecks(parent_pid=os.getpid(), check_behavior_version=26)

            mp_manager = multi_proc_manager_with_watchdog.create_manager()
            loader = get_loader_from_returnn_dataset(dataset, mp_manager)

            c = 0
            n = 3
            for batch in loader:
                print(batch)
                c += 1
                if c >= n:
                    break

            assert c == n
    finally:
        BehaviorVersion._reset(behavior_version_orig_state)


def test_DistributeFilesDataset_no_worker_proc():
    # The use case of use_worker_proc=False: the DFD itself runs in the DataLoader worker,
    # and the sub epoch dataset has its own parallel worker procs.
    from test_Dataset import (
        _dfd_make_hdf_files,
        _dfd_get_sub_epoch_dataset_multi_proc_random,
        _dfd_test_random_seed_offset,
    )

    files = _dfd_make_hdf_files(num_hdf_files=4)
    opts = {
        "class": "DistributeFilesDataset",
        "files": files,
        "get_sub_epoch_dataset": _dfd_get_sub_epoch_dataset_multi_proc_random,
        "partition_epoch": 2,
        "seq_ordering": "random",
        "random_seed_offset": _dfd_test_random_seed_offset,
    }

    ref_dataset = init_dataset(opts)
    ref_dataset.init_seq_order(epoch=1)
    ref = []
    seq_idx = 0
    while ref_dataset.is_less_than_num_seqs(seq_idx):
        ref_dataset.load_seqs(seq_idx, seq_idx + 1)
        ref.append(ref_dataset.get_data(seq_idx, "classes").tolist())
        seq_idx += 1
    ref_dataset.finish_epoch(free_resources=True)
    assert ref

    dataset = init_dataset({**opts, "use_worker_proc": False})
    mp_manager = multi_proc_manager_with_watchdog.create_manager()
    loader = get_loader_from_returnn_dataset(dataset, mp_manager, batch_size=100, max_seqs=3)
    res = []
    for batch in loader:
        for b in range(batch["classes"].shape[0]):
            res.append(batch["classes"][b, : batch["classes:seq_len"][b]].tolist())
    assert res == ref


def test_LmDataset_SentencePieces_add_eos():
    # The dataset with its vocab is pickled into the spawned worker proc.
    import tempfile
    from test_GeneratingDataset import generate_sentencepiece_model

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(f"{tmp_dir}/corpus.txt", "w") as f:
            f.write("HELLO WORLD\nGOOD MORNING\nHELLO\n")
        opts = {
            "class": "LmDataset",
            "corpus_file": f"{tmp_dir}/corpus.txt",
            "orth_vocab": {
                "class": "SentencePieces",
                "model_file": generate_sentencepiece_model(tmp_dir),
                "add_eos": True,
            },
        }

        ref_dataset = init_dataset(opts)
        ref_dataset.init_seq_order(epoch=1)
        ref = []
        seq_idx = 0
        while ref_dataset.is_less_than_num_seqs(seq_idx):
            ref_dataset.load_seqs(seq_idx, seq_idx + 1)
            ref.append(ref_dataset.get_data(seq_idx, "data").tolist())
            seq_idx += 1
        assert len(ref) == 3 and all(seq[-1] == ref_dataset.orth_vocab.eos_label_id for seq in ref)

        dataset = init_dataset(opts)
        mp_manager = multi_proc_manager_with_watchdog.create_manager()
        loader = get_loader_from_returnn_dataset(dataset, mp_manager, batch_size=100, max_seqs=3)
        res = []
        for batch in loader:
            for b in range(batch["data"].shape[0]):
                res.append(batch["data"][b, : batch["data:seq_len"][b]].tolist())
        assert res == ref


def test_func_in_global_config():
    # Very similar to test_MultiProcDataset_via_config.
    # https://github.com/rwth-i6/returnn/issues/1495
    from io import StringIO
    import textwrap

    config = Config()
    config.load_file(
        StringIO(
            textwrap.dedent(
                """\
                #!returnn.py

                import numpy
                from returnn.datasets.map import MapDatasetBase

                class MyCustomMapDatasetInConfig(MapDatasetBase):
                    def __init__(self):
                        super().__init__(data_types={"data": {"shape": (None, 3)}})

                    def __len__(self):
                        return 10

                    def __getitem__(self, item):
                        return {"data": numpy.zeros((5, 3))}
                """
            )
        )
    )

    with global_config_ctx(config):
        dataset = init_dataset(
            {"class": "MapDatasetWrapper", "map_dataset": config.typed_dict["MyCustomMapDatasetInConfig"]}
        )

        mp_manager = multi_proc_manager_with_watchdog.create_manager()
        loader = get_loader_from_returnn_dataset(dataset, mp_manager, batch_size=100, max_seqs=4)

        c = 0
        for batch in loader:
            print(batch)
            c += 1

        assert c == 3


def test_HDFDataset():
    # https://github.com/rwth-i6/returnn/issues/1281
    from test_HDFDataset import generate_hdf_from_other, HDFDataset

    hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 23})
    hdf_dataset = HDFDataset(files=[hdf_fn], cache_byte_size=0)

    mp_manager = multi_proc_manager_with_watchdog.create_manager()
    loader = get_loader_from_returnn_dataset(hdf_dataset, mp_manager)
    c = 0
    n = 3
    for batch in loader:
        print(batch)
        c += 1
        if c >= n:
            break

    assert c == n


def test_MultiProcDataset_HDFDataset():
    from test_HDFDataset import generate_hdf_from_other
    from test_MultiProcDataset import timeout
    from returnn.datasets.multi_proc import MultiProcDataset

    hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 23})
    with timeout(10):
        mp_dataset = MultiProcDataset(
            dataset={"class": "HDFDataset", "files": [hdf_fn], "cache_byte_size": 0},
            num_workers=1,
            buffer_size=1,
        )
        mp_dataset.initialize()

        mp_manager = multi_proc_manager_with_watchdog.create_manager()
        loader = get_loader_from_returnn_dataset(mp_dataset, mp_manager)
        c = 0
        n = 3
        for batch in loader:
            print(batch)
            c += 1
            if c >= n:
                break

        assert c == n


class _BigTagBatches(torch.utils.data.IterableDataset):
    """the tag of each batch goes through the pipe of the worker (unlike tensors), and is larger than its buffer"""

    def __iter__(self):
        for i in range(10):
            yield {"data": torch.full((2,), i), "tag": "x" * 100000}


def _check_shutdown_data_loader(*, pin_memory: bool):
    """
    Stops the persistent worker with an unread batch, also repeatedly,
    and also when the worker already left its loop on its own SIGINT (Ctrl+C goes to the whole process group):
    then it does not get the shutdown signal, but must not wait on its unread batch until the join timeout (5 s)
    """
    import signal
    import time
    from returnn.util.multi_proc_non_daemonic_spawn import NonDaemonicSpawnContext

    loader = DataLoader(
        _BigTagBatches(),
        batch_size=None,
        num_workers=1,
        persistent_workers=True,
        pin_memory=pin_memory,
        multiprocessing_context=NonDaemonicSpawnContext(),
    )
    data_pipeline.shutdown_data_loader(loader)  # nothing started yet
    for interrupted in [False, True]:
        next(iter(loader))
        data_iter = loader._iterator
        worker = data_iter._workers[0]
        if pin_memory:  # the Torch pin thread reads the pipe: wait until it has the next batch
            end_time = time.monotonic() + 60
            while data_iter._data_queue.qsize() == 0:
                assert time.monotonic() < end_time
                time.sleep(0.01)
        else:  # the worker sends the next batch
            assert data_iter._worker_result_queue._reader.poll(timeout=60)
        if interrupted:
            os.kill(worker.pid, signal.SIGINT)
            time.sleep(0.5)  # the worker leaves its loop
        start_time = time.monotonic()
        data_pipeline.shutdown_data_loader(loader)
        assert time.monotonic() - start_time < 3
        # no atexit handler left which would signal it at interpreter exit
        assert not worker.is_alive() and worker.exitcode == 0 and worker._at_exit_cleanup_handler is None
        assert loader._iterator is None
        data_pipeline.shutdown_data_loader(loader)


def test_shutdown_data_loader():
    _check_shutdown_data_loader(pin_memory=False)


def test_shutdown_data_loader_torch_pin_memory():
    """with the Torch pin thread, which reads the worker results itself"""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    _check_shutdown_data_loader(pin_memory=True)


def test_batching_packed_batch_cost_bounds_a_product_of_lengths():
    """
    A monotonic RNN-T lattice has frames times prefixes cells per sequence, a product no per-key length
    budget can bound, so a batch cost derived from the lengths joins the packed budget check.
    """
    import numpy

    def _seq(frames: int, labels: int):
        return {"text_codes": numpy.zeros((frames,), dtype="int32"), "labels": numpy.zeros((labels,), dtype="int32")}

    seqs = [_seq(30, 9), _seq(30, 9), _seq(30, 9), _seq(30, 9), _seq(6, 1), _seq(6, 1)]
    cells = lambda lengths: lengths["text_codes"] * (lengths["labels"] + 1)  # noqa: E731
    limit = 700  # two 300-cell sequences fit, a third would not, both 12-cell ones join the second batch

    batches = list(
        data_pipeline.BatchingIterDataPipe(
            seqs,
            batch_size=None,
            max_seqs=100,
            packed_batch_size={"lattice": limit},
            packed_batch_cost={"lattice": cells},
        )
    )
    for batch in batches:
        used = sum(len(s["text_codes"]) * (len(s["labels"]) + 1) for s in batch)
        assert used <= limit, (used, [len(s["text_codes"]) for s in batch])
    assert [len(b) for b in batches] == [2, 4], [len(b) for b in batches]

    without = list(data_pipeline.BatchingIterDataPipe(seqs, batch_size=None, max_seqs=100))
    assert len(without) == 1, "the length budgets alone must not bound the product"


def test_batching_packed_batch_cost_meets_only_its_own_limit():
    """
    A cost is bounded by the packed_batch_size entry of its name and nothing else. The padded batch_size
    check must not see it (the largest cost times the batch size against the frame limit would cut
    batches early), and a packed_batch_size without an entry for the cost is refused instead of
    leaving the cost unbounded.
    """
    import numpy
    import pytest

    def _seq(frames: int, labels: int):
        return {"text_codes": numpy.zeros((frames,), dtype="int32"), "labels": numpy.zeros((labels,), dtype="int32")}

    seqs = [_seq(30, 9), _seq(30, 9), _seq(30, 9), _seq(30, 9), _seq(6, 1), _seq(6, 1)]
    cells = lambda lengths: lengths["text_codes"] * (lengths["labels"] + 1)  # noqa: E731

    batches = list(
        data_pipeline.BatchingIterDataPipe(
            seqs,
            batch_size=1000,
            max_seqs=100,
            packed_batch_size={"lattice": 700},
            packed_batch_cost={"lattice": cells},
        )
    )
    assert [len(b) for b in batches] == [2, 4], [len(b) for b in batches]

    with pytest.raises(AssertionError, match="packed_batch_size"):
        list(
            data_pipeline.BatchingIterDataPipe(
                seqs, batch_size=None, max_seqs=100, packed_batch_size=700, packed_batch_cost={"lattice": cells}
            )
        )


def test_bucket_ordering_monotonic_data_keys():
    """
    A partially filled bucket is emitted at the end, after batches of later seqs.
    With monotonic_data_keys, complete_frac and seq_idx keep the input order,
    while the payload and seq_tag stay with their seq.
    """
    import pickle
    import numpy

    lens = [8, 1, 20, 1, 1, 1]  # the seq of len 20 is too long for all buckets and dropped
    seqs = [
        {
            "data": numpy.full((n,), i, dtype="int32"),
            "seq_tag": numpy.array(f"seq-{i}"),
            "seq_idx": numpy.array(i),
            "complete_frac": numpy.array((i + 1) / len(lens)),
        }
        for i, n in enumerate(lens)
    ]
    seqs_orig = [dict(seq) for seq in seqs]

    batches = list(data_pipeline.BucketOrderingIterDataPipe(seqs, buckets=[(2, 2), (8, 2)], length_key="data"))
    assert [[int(s["seq_idx"]) for s in b] for b in batches] == [[1, 3], [4, 5], [0]]  # default: original values
    assert [max(float(s["complete_frac"]) for s in b) for b in batches] == [4 / 6, 1.0, 1 / 6]

    pipe = data_pipeline.BucketOrderingIterDataPipe(
        seqs, buckets=[(2, 2), (8, 2)], length_key="data", monotonic_data_keys=("complete_frac", "seq_idx")
    )
    for pipe_ in [pipe, pipe, pickle.loads(pickle.dumps(pipe))]:  # repeated iteration, pickling
        batches = list(pipe_)
        assert [[str(s["seq_tag"]) for s in b] for b in batches] == [["seq-1", "seq-3"], ["seq-4", "seq-5"], ["seq-0"]]
        assert [[int(s["data"][0]) for s in b] for b in batches] == [[1, 3], [4, 5], [0]]
        assert [[int(s["seq_idx"]) for s in b] for b in batches] == [[0, 1], [3, 4], [5]]
        assert [max(float(s["complete_frac"]) for s in b) for b in batches] == [2 / 6, 5 / 6, 1.0]
    assert all(seq == seq_orig for seq, seq_orig in zip(seqs, seqs_orig)), "input dicts must not be modified"


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
