"""
tests for MultiProcDataset
"""

from __future__ import annotations
import _setup_test_env  # noqa
import os
import sys
import time
import shutil
import tempfile
import threading
import functools
import numpy
import signal
import contextlib
import unittest
from returnn.util import better_exchook
from returnn.datasets.basic import init_dataset
from returnn.datasets.multi_proc import MultiProcDataset
from returnn.datasets.map import MapDatasetBase
from test_HDFDataset import generate_hdf_from_other
from test_Dataset import dummy_iter_dataset, compare_dataset_seqs


def _sig_alarm_handler(signum, frame):
    raise Exception("Alarm (timeout) signal handler")


signal.signal(signal.SIGALRM, _sig_alarm_handler)


@contextlib.contextmanager
def timeout(seconds=10):
    """
    :param seconds: when the context is not closed within this time, an exception will be raised
    """
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def test_MultiProcDataset_n1_b1_default():
    hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 23})
    hdf_dataset_dict = {"class": "HDFDataset", "files": [hdf_fn]}
    hdf_dataset = init_dataset(hdf_dataset_dict)
    hdf_dataset_seqs = dummy_iter_dataset(hdf_dataset)

    with timeout():
        mp_dataset = MultiProcDataset(dataset=hdf_dataset_dict, num_workers=1, buffer_size=1)
        mp_dataset.initialize()
        mp_dataset_seqs = dummy_iter_dataset(mp_dataset)

        compare_dataset_seqs(hdf_dataset_seqs, mp_dataset_seqs)


def test_MultiProcDataset_n3_b5_shuffle():
    hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 23})
    hdf_dataset_dict = {"class": "HDFDataset", "files": [hdf_fn], "seq_ordering": "random"}
    hdf_dataset = init_dataset(hdf_dataset_dict)
    hdf_dataset_seqs = dummy_iter_dataset(hdf_dataset)

    with timeout():
        mp_dataset = MultiProcDataset(dataset=hdf_dataset_dict, num_workers=3, buffer_size=5)
        mp_dataset.initialize()
        mp_dataset_seqs = dummy_iter_dataset(mp_dataset)

        compare_dataset_seqs(hdf_dataset_seqs, mp_dataset_seqs)


def test_MultiProcDataset_n3_b5_shuffle_sharding():
    hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 23})
    hdf_dataset_dict = {"class": "HDFDataset", "files": [hdf_fn], "seq_ordering": "random"}
    hdf_dataset = init_dataset(hdf_dataset_dict)
    hdf_dataset_seqs = dummy_iter_dataset(hdf_dataset)

    with timeout():
        mp_dataset = MultiProcDataset(
            dataset=hdf_dataset_dict, num_workers=3, buffer_size=5, sharding_method="dedicated"
        )
        mp_dataset.initialize()
        mp_dataset_seqs = dummy_iter_dataset(mp_dataset)

        assert len(hdf_dataset_seqs) == len(mp_dataset_seqs)
        assert set(seq.seq_idx for seq in hdf_dataset_seqs) == set(seq.seq_idx for seq in mp_dataset_seqs)
        assert set(seq.seq_tag for seq in hdf_dataset_seqs) == set(seq.seq_tag for seq in mp_dataset_seqs)


def test_MultiProcDataset_sharded():
    # The MultiProcDataset itself is sharded (e.g. per distributed rank, or per DataLoader worker),
    # given at construction, or via set_sharding, with the workers running,
    # or on an unpickled copy (as in a DataLoader worker), where the workers start later.
    # With both sharding methods, the shards together cover every seq once.
    import pickle

    hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 23})
    hdf_dataset_dict = {"class": "HDFDataset", "files": [hdf_fn], "seq_ordering": "random"}
    all_tags = sorted(seq.seq_tag for seq in dummy_iter_dataset(init_dataset(hdf_dataset_dict)))

    with timeout(60):
        for sharding_method in ["seq_order", "dedicated"]:
            for set_sharding_when in [None, "after_init", "after_pickle"]:
                tags = []
                for shard_index in range(2):
                    shard_opts = {"_num_shards": 2, "_shard_index": shard_index}
                    mp_dataset = MultiProcDataset(
                        dataset=hdf_dataset_dict,
                        num_workers=2,
                        buffer_size=5,
                        sharding_method=sharding_method,
                        **(shard_opts if set_sharding_when is None else {}),
                    )
                    mp_dataset.initialize()
                    if set_sharding_when == "after_pickle":
                        mp_dataset = pickle.loads(pickle.dumps(mp_dataset))
                    if set_sharding_when is not None:
                        mp_dataset.set_sharding(num_shards=2, shard_index=shard_index)
                    assert mp_dataset.supports_sharding()
                    tags.append([seq.seq_tag for seq in dummy_iter_dataset(mp_dataset)])
                assert [len(t) for t in tags] == [12, 11], (sharding_method, set_sharding_when, tags)
                assert sorted(tags[0] + tags[1]) == all_tags, (sharding_method, set_sharding_when, tags)


def test_MultiProcDataset_sharded_random_seed_offset():
    # The shards (e.g. per distributed rank) are cut from the seq order, so with seq_ordering "random"
    # they are disjoint only when every rank uses the same random seed offset
    # (the default differs per rank, except with torch_distributed dataset_distribution "shard").
    hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 23})
    hdf_dataset_dict = {"class": "HDFDataset", "files": [hdf_fn], "seq_ordering": "random"}
    all_tags = sorted(seq.seq_tag for seq in dummy_iter_dataset(init_dataset(hdf_dataset_dict)))

    with timeout(60):
        for same_seed in [True, False]:
            tags = []
            for rank in range(2):
                mp_dataset = MultiProcDataset(
                    dataset=hdf_dataset_dict,
                    num_workers=2,
                    buffer_size=5,
                    random_seed_offset=0 if same_seed else rank * 16127,
                )
                mp_dataset.initialize()
                mp_dataset.set_sharding(num_shards=2, shard_index=rank)
                tags.append([seq.seq_tag for seq in dummy_iter_dataset(mp_dataset)])
            assert [len(t) for t in tags] == [12, 11], (same_seed, tags)
            if same_seed:
                assert sorted(tags[0] + tags[1]) == all_tags, tags
            else:
                assert set(tags[0]) & set(tags[1]), tags


def test_MultiProcDataset_meta():
    hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 23})
    meta_dataset_dict = {
        "class": "MetaDataset",
        "data_map": {"classes": ("hdf", "classes"), "data": ("hdf", "data")},
        "datasets": {"hdf": {"class": "HDFDataset", "files": [hdf_fn]}},
    }
    meta_dataset = init_dataset(meta_dataset_dict)
    meta_dataset_seqs = dummy_iter_dataset(meta_dataset)

    with timeout():
        mp_dataset = MultiProcDataset(dataset=meta_dataset_dict, num_workers=1, buffer_size=1)
        mp_dataset.initialize()
        mp_dataset_seqs = dummy_iter_dataset(mp_dataset)

        compare_dataset_seqs(meta_dataset_seqs, mp_dataset_seqs)


def test_MultiProcDataset_via_config():
    # https://github.com/rwth-i6/returnn/issues/1384
    from io import StringIO
    import textwrap
    from returnn.config import Config, global_config_ctx

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
                        return 2

                    def __getitem__(self, item):
                        return {"data": numpy.zeros((5, 3))}
                """
            )
        )
    )

    with timeout(), global_config_ctx(config):
        mp_dataset = MultiProcDataset(
            dataset={"class": "MapDatasetWrapper", "map_dataset": config.typed_dict["MyCustomMapDatasetInConfig"]},
            num_workers=1,
            buffer_size=1,
        )
        mp_dataset.initialize()
        items = dummy_iter_dataset(mp_dataset)
        assert len(items) == 2


class _MyCustomMapDatasetException(Exception):
    pass


class _MyCustomMapDatasetThrowingExceptionAtInit(MapDatasetBase):
    def __init__(self):
        super().__init__()
        raise _MyCustomMapDatasetException("test exception at init")


class _MyCustomMapDatasetThrowingExceptionAtItem(MapDatasetBase):
    def __init__(self):
        super().__init__(data_types={"data": {"shape": (None, 3)}})

    def __len__(self):
        return 2

    def __getitem__(self, item):
        if item == 0:
            return {"data": numpy.zeros((5, 3))}
        raise _MyCustomMapDatasetException("test exception at getitem")


def test_MultiProcDataset_exception_at_init():
    with timeout():
        mp_dataset = MultiProcDataset(
            dataset={"class": "MapDatasetWrapper", "map_dataset": _MyCustomMapDatasetThrowingExceptionAtInit},
            num_workers=1,
            buffer_size=1,
        )
        try:
            mp_dataset.initialize()
        except Exception as exc:
            # Accept any exception. We do not properly forward it. But this is ok.
            print("Got expected exception:", exc)
        else:
            raise Exception("Expected exception")


def test_MultiProcDataset_exception_at_item():
    with timeout():
        mp_dataset = MultiProcDataset(
            dataset={"class": "MapDatasetWrapper", "map_dataset": _MyCustomMapDatasetThrowingExceptionAtItem},
            num_workers=1,
            buffer_size=1,
        )
        mp_dataset.initialize()
        try:
            dummy_iter_dataset(mp_dataset)
        except Exception as exc:
            # Accept any exception. We do not properly forward it. But this is ok.
            print("Got expected exception:", exc)
        else:
            raise Exception("Expected exception")


class _MyCustomDummyMapDataset(MapDatasetBase):
    def __init__(self):
        super().__init__(data_types={"data": {"shape": (None, 3)}})

    def __len__(self):
        return 2

    def __getitem__(self, item):
        return {"data": numpy.zeros((item * 2 + 5, 3))}


def test_MultiProcDataset_pickle():
    import pickle

    with timeout():
        mp_dataset = MultiProcDataset(
            dataset={"class": "MapDatasetWrapper", "map_dataset": _MyCustomDummyMapDataset},
            num_workers=1,
            buffer_size=1,
        )
        mp_dataset.initialize()
        mp_dataset_seqs = dummy_iter_dataset(mp_dataset)
        mp_dataset_serialized = pickle.dumps(mp_dataset)

    with timeout():
        mp_dataset_ = pickle.loads(mp_dataset_serialized)
        mp_dataset_seqs_ = dummy_iter_dataset(mp_dataset_)

    compare_dataset_seqs(mp_dataset_seqs, mp_dataset_seqs_)


class _MyCustomBlockingMapDataset(MapDatasetBase):
    # Loading item block_at creates the file started_fn and waits until the file release_fn exists.
    def __init__(self, *, started_fn: str, release_fn: str, block_at: int):
        super().__init__(data_types={"data": {"shape": (None, 3)}})
        self.started_fn = started_fn
        self.release_fn = release_fn
        self.block_at = block_at

    def __len__(self):
        return 5

    def __getitem__(self, item):
        if item == self.block_at:
            open(self.started_fn, "w").close()
            for _ in range(600):
                if os.path.exists(self.release_fn):
                    break
                time.sleep(0.1)
            else:
                raise Exception("test: release not set")
        return {"data": numpy.full((item * 2 + 5, 3), item, dtype="float32")}


def _wait_for_file(fn: str):
    for _ in range(100):
        if os.path.exists(fn):
            return
        time.sleep(0.1)
    raise Exception(f"timeout waiting for {fn!r}")


def test_MultiProcDataset_serves_cached_while_loading():
    # The worker serves a cached seq while it loads a later seq.
    tmp_dir = tempfile.mkdtemp()
    started_fn, release_fn = f"{tmp_dir}/started", f"{tmp_dir}/release"
    map_dataset = functools.partial(
        _MyCustomBlockingMapDataset, started_fn=started_fn, release_fn=release_fn, block_at=2
    )
    try:
        with timeout(30):
            mp_dataset = MultiProcDataset(
                dataset={"class": "MapDatasetWrapper", "map_dataset": map_dataset}, num_workers=1, buffer_size=3
            )
            mp_dataset.initialize()
            mp_dataset.init_seq_order(epoch=1)
            mp_dataset.load_seqs(0, 1)
            _wait_for_file(started_fn)  # item 1 is cached, item 2 is loading
            thread = threading.Thread(target=mp_dataset.load_seqs, args=(1, 2))
            thread.start()
            thread.join(timeout=10)
            assert not thread.is_alive(), "seq 1 not served while the worker loads seq 2"
            open(release_fn, "w").close()
            for seq_idx in range(1, 5):
                mp_dataset.load_seqs(seq_idx, seq_idx + 1)
                numpy.testing.assert_array_equal(
                    mp_dataset.get_data(seq_idx, "data"), numpy.full((seq_idx * 2 + 5, 3), seq_idx)
                )
            assert not mp_dataset.is_less_than_num_seqs(5)
            mp_dataset.finish_epoch(free_resources=True)
    finally:
        open(release_fn, "w").close()
        shutil.rmtree(tmp_dir)


def test_MultiProcDataset_del_after_init_error():
    from test_Dataset import check_del_after_init_error

    check_del_after_init_error(
        lambda: MultiProcDataset(dataset={"class": "Task12AXDataset", "num_seqs": 3}, num_workers=0, buffer_size=1),
        AssertionError,
    )


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
