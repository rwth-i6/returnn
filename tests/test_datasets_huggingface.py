from __future__ import annotations

import _setup_test_env  # noqa
from typing import List, Tuple
import os
import tempfile
import atexit
import shutil
import pickle
import numpy

from returnn.datasets import init_dataset
from returnn.datasets.huggingface import HuggingFaceDataset
from test_Dataset import dummy_iter_dataset, check_predefined_seq_order


def _setup_hf_env():
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = _get_tmp_dir()


def _get_tmp_dir() -> str:
    fn = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, fn)
    return fn


_setup_hf_env()


def test_HuggingFaceDataset_audio():
    ds = HuggingFaceDataset(
        {"path": "datasets-examples/doc-audio-6", "split": "train"},
        cast_columns={"audio": {"_type": "Audio", "sample_rate": 16_000}},
        data_format={"audio": {"dtype": "float32", "shape": [None]}},
        seq_tag_column=None,
    )
    ds.initialize()
    res = dummy_iter_dataset(ds)
    print(res[0].features["audio"])


def test_HuggingFaceDataset_text1():
    ds = HuggingFaceDataset(
        {"path": "openai/gdpval", "split": "train"},
        seq_tag_column="task_id",
        data_format={
            "prompt": {"dtype": "string", "shape": ()},
            "sector": {"dtype": "string", "shape": ()},
            "occupation": {"dtype": "string", "shape": ()},
        },
    )
    ds.initialize()
    res = dummy_iter_dataset(ds)
    print(repr(res[0].seq_tag))
    assert type(res[0].seq_tag) is str


def test_HuggingFaceDataset_text2():
    ds = HuggingFaceDataset(
        {"path": "lavita/medical-qa-shared-task-v1-toy", "split": "train"},
        seq_tag_column="id",
        data_format={
            "id": {"dtype": "int64", "shape": ()},
            "startphrase": {"dtype": "string", "shape": ()},
            "label": {"dtype": "int64", "shape": ()},
        },
    )
    ds.initialize()
    assert dummy_iter_dataset(ds)


def test_HuggingFaceDataset_rename_tokens():
    ds = HuggingFaceDataset(
        {"path": "lavita/medical-qa-shared-task-v1-toy", "split": "train"},
        seq_tag_column="id",
        rename_columns={"startphrase": "text"},
        data_format={
            "id": {"dtype": "int64", "shape": ()},
            "text": {"dtype": "string", "shape": ()},
            "label": {"dtype": "int64", "shape": ()},
        },
    )
    ds.initialize()
    assert dummy_iter_dataset(ds)


def test_HuggingFaceDataset_text_tokenize():
    ds = HuggingFaceDataset(
        {"path": "lavita/medical-qa-shared-task-v1-toy", "split": "train"},
        seq_tag_column="id",
        data_format={
            "id": {"dtype": "int64", "shape": ()},
            "startphrase": {"dtype": "int32", "vocab": {"class": "Utf8ByteTargets"}},
            "label": {"dtype": "int64", "shape": ()},
        },
    )
    ds.initialize()
    res = dummy_iter_dataset(ds)
    txt = res[0].features["startphrase"]
    print("startphrase:", txt)
    assert isinstance(txt, numpy.ndarray) and txt.dtype == numpy.int32
    txt_ = ds.data_format["startphrase"].vocab.get_seq_labels(txt)
    print("startphrase labels:", txt_)


def test_HuggingFaceDataset_pickle():
    ds = HuggingFaceDataset(
        {"path": "lavita/medical-qa-shared-task-v1-toy", "split": "train"},
        seq_tag_column="id",
        data_format={
            "id": {"dtype": "int64", "shape": ()},
            "startphrase": {"dtype": "string", "shape": ()},
            "label": {"dtype": "int64", "shape": ()},
        },
    )
    ds.initialize()
    s = pickle.dumps(ds)
    ds = pickle.loads(s)
    assert isinstance(ds, HuggingFaceDataset)
    assert dummy_iter_dataset(ds)


def test_HuggingFaceDataset_load_from_disk():
    from datasets import load_dataset

    datadir_path = _get_tmp_dir() + "/hf-dataset-save-to-disk"
    hf_ds = load_dataset("lavita/medical-qa-shared-task-v1-toy", split="train")
    hf_ds.save_to_disk(datadir_path)

    ds = HuggingFaceDataset(
        datadir_path,
        seq_tag_column="id",
        data_format={
            "id": {"dtype": "int64", "shape": ()},
            "startphrase": {"dtype": "string", "shape": ()},
            "label": {"dtype": "int64", "shape": ()},
        },
    )
    ds.initialize()
    assert dummy_iter_dataset(ds)


def test_HuggingFaceDataset_single_arrows():
    import datasets

    datadir_path = _get_tmp_dir() + "/hf-dataset-save-to-disk"
    hf_ds = datasets.Dataset.from_list([{"data": i} for i in range(100_000)])
    hf_ds.save_to_disk(datadir_path, num_shards=100)

    content = os.listdir(datadir_path)
    print("Saved dir content:", content)
    assert "state.json" in content
    assert "dataset_info.json" in content
    assert all(f"data-{i:05}-of-00100.arrow" in content for i in range(100))

    ds = HuggingFaceDataset(
        [f"{datadir_path}/data-{i:05}-of-00100.arrow" for i in range(0, 100, 5)],
        seq_tag_column=None,
        data_format={"data": {"dtype": "int64", "shape": ()}},
    )
    ds.initialize()
    assert dummy_iter_dataset(ds)


def test_HuggingFaceDataset_file_cache_with_sharded():
    import datasets

    datadir_path = _get_tmp_dir() + "/hf-dataset-save-to-disk"
    hf_ds = datasets.Dataset.from_list([{"data": i} for i in range(100_000)])
    hf_ds.save_to_disk(datadir_path, num_shards=100)

    ds = HuggingFaceDataset(
        datadir_path,
        use_file_cache=True,
        seq_tag_column=None,
        data_format={"data": {"dtype": "int64", "shape": ()}},
    )
    ds.initialize()
    assert dummy_iter_dataset(ds)


def test_HuggingFaceDataset_in_multi_proc():
    ds_dict = {
        "class": "HuggingFaceDataset",
        "dataset_opts": {"path": "lavita/medical-qa-shared-task-v1-toy", "split": "train"},
        "seq_tag_column": "id",
        "data_format": {
            "id": {"dtype": "int64", "shape": ()},
            "startphrase": {"dtype": "string", "shape": ()},
            "label": {"dtype": "int64", "shape": ()},
        },
    }
    ds_dict = {
        "class": "MultiProcDataset",
        "num_workers": 2,
        "buffer_size": 5,
        "dataset": ds_dict,
    }
    ds = init_dataset(ds_dict)
    ds.initialize()
    assert dummy_iter_dataset(ds)


def _dummy_tags_dataset(tags: List[str], **kwargs) -> HuggingFaceDataset:
    import datasets

    ds = HuggingFaceDataset(
        lambda: datasets.Dataset.from_dict({"id": tags, "data": list(range(len(tags)))}),
        data_format={"data": {"dtype": "int64", "shape": ()}},
        **kwargs,
    )
    ds.initialize()
    return ds


def _get_tags_and_data(ds: HuggingFaceDataset) -> Tuple[List[str], List[int]]:
    ds.load_seqs(0, ds.num_seqs)
    return (
        [ds.get_tag(seq_idx) for seq_idx in range(ds.num_seqs)],
        [int(ds.get_data(seq_idx, "data")) for seq_idx in range(ds.num_seqs)],
    )


def test_HuggingFaceDataset_seq_list():
    tags = [f"item-{i}" for i in range(7)]
    ds = _dummy_tags_dataset(tags)
    seq_list = ["item-5", "item-1", "item-5", "item-3", "item-3"]
    ds.init_seq_order(epoch=1, seq_list=seq_list)
    assert list(ds.get_current_seq_order()) == [5, 1, 5, 3, 3]
    assert ds.num_seqs == 5
    assert _get_tags_and_data(ds) == (seq_list, [5, 1, 5, 3, 3])

    ds.init_seq_order(epoch=2, seq_list=tags[::-1])
    assert list(ds.get_current_seq_order()) == [6, 5, 4, 3, 2, 1, 0]
    assert _get_tags_and_data(ds) == (tags[::-1], [6, 5, 4, 3, 2, 1, 0])


def test_HuggingFaceDataset_seq_list_duplicate_tags():
    ds = _dummy_tags_dataset(["a", "b", "a"])
    ds.init_seq_order(epoch=1, seq_list=["a", "b", "a"])
    assert list(ds.get_current_seq_order()) == [0, 1, 0]
    assert _get_tags_and_data(ds) == (["a", "b", "a"], [0, 1, 0])


def test_HuggingFaceDataset_seq_list_empty():
    ds = _dummy_tags_dataset(["a", "b", "c"])
    ds.init_seq_order(epoch=1, seq_list=[])
    assert list(ds.get_current_seq_order()) == []
    assert ds.num_seqs == 0


def test_HuggingFaceDataset_seq_list_unknown_tag():
    ds = _dummy_tags_dataset(["a", "b", "c"])
    try:
        ds.init_seq_order(epoch=1, seq_list=["b", "x", "a"])
    except ValueError as exc:
        print("Got expected exception:", exc)
        assert "'x'" in str(exc)
    else:
        raise Exception("expected ValueError for unknown tag")


def test_HuggingFaceDataset_seq_list_no_seq_tag_column():
    ds = _dummy_tags_dataset(["a", "b", "c", "d"], seq_tag_column=None)
    assert ds.get_all_tags() == ["seq-0", "seq-1", "seq-2", "seq-3"]
    ds.init_seq_order(epoch=1, seq_list=["seq-2", "seq-0", "seq-3"])
    assert list(ds.get_current_seq_order()) == [2, 0, 3]
    assert _get_tags_and_data(ds) == (["seq-2", "seq-0", "seq-3"], [2, 0, 3])
    try:
        ds.init_seq_order(epoch=1, seq_list=["a"])
    except ValueError as exc:
        print("Got expected exception:", exc)
    else:
        raise Exception("expected ValueError for unknown tag")


def test_HuggingFaceDataset_seq_order_precedence():
    tags = ["a", "b", "c", "d"]
    ds = _dummy_tags_dataset(tags, seq_ordering="reverse")

    # Explicit seq_order has precedence over seq_list. seq_list is not even resolved then.
    ds.init_seq_order(epoch=1, seq_list=["d", "x"], seq_order=[2, 0, 2])
    assert list(ds.get_current_seq_order()) == [2, 0, 2]
    assert _get_tags_and_data(ds) == (["c", "a", "c"], [2, 0, 2])

    # Explicit seq_list has precedence over the seq_ordering of the epoch.
    ds.init_seq_order(epoch=1, seq_list=["b", "c"])
    assert list(ds.get_current_seq_order()) == [1, 2]

    # Ordinary epoch ordering.
    ds.init_seq_order(epoch=1)
    assert list(ds.get_current_seq_order()) == [3, 2, 1, 0]
    assert _get_tags_and_data(ds) == (tags[::-1], [3, 2, 1, 0])

    ds.init_seq_order(epoch=None)
    assert list(ds.get_current_seq_order()) == []
    assert ds.num_seqs == 0


def test_HuggingFaceDataset_seq_list_no_audio_decode():
    import datasets
    from unittest import mock

    tags = ["a", "b", "c"]
    ds = HuggingFaceDataset(
        lambda: datasets.Dataset.from_dict(
            {"id": tags, "audio": [{"bytes": b"not an audio file", "path": f"{tag}.wav"} for tag in tags]}
        ),
        cast_columns={"audio": {"_type": "Audio", "sampling_rate": 16_000}},
        data_format={"audio": {"dtype": "float32", "shape": [None]}},
    )
    ds.initialize()

    class _AudioDecoded(Exception):
        pass

    with mock.patch.object(datasets.Audio, "decode_example", side_effect=_AudioDecoded):
        ds.init_seq_order(epoch=1, seq_list=["c", "a", "c"])
        assert list(ds.get_current_seq_order()) == [2, 0, 2]

        # Check that the mock is effective, i.e. reading a row does decode the audio.
        try:
            ds.load_seqs(0, 1)
        except _AudioDecoded:
            pass
        else:
            raise Exception("expected that loading the seq decodes the audio")


def test_HuggingFaceDataset_seq_list_via_MetaDataset():
    import datasets

    num_seqs = 11
    tags = [f"item-{i}" for i in range(num_seqs)]
    perm = [(i * 7 + 3) % num_seqs for i in range(num_seqs)]  # corpus order of the second dataset
    ds = init_dataset(
        {
            "class": "MetaDataset",
            "datasets": {
                "ctrl": {
                    "class": "HuggingFaceDataset",
                    "dataset_opts": lambda: datasets.Dataset.from_dict({"id": tags, "data": list(range(num_seqs))}),
                    "data_format": {"data": {"dtype": "int64", "shape": ()}},
                    "seq_ordering": "reverse",
                },
                "other": {
                    "class": "HuggingFaceDataset",
                    "dataset_opts": lambda: datasets.Dataset.from_dict({"id": [tags[i] for i in perm], "data": perm}),
                    "data_format": {"data": {"dtype": "int64", "shape": ()}},
                },
            },
            "data_map": {"data": ("ctrl", "data"), "other": ("other", "data")},
            "seq_order_control_dataset": "ctrl",
        }
    )
    ds.initialize()
    ds.init_seq_order(epoch=1)
    assert ds.num_seqs == num_seqs
    ds.load_seqs(0, num_seqs)
    for seq_idx in range(num_seqs):
        assert ds.get_tag(seq_idx) == tags[num_seqs - 1 - seq_idx]
        assert int(ds.get_data(seq_idx, "data")) == int(ds.get_data(seq_idx, "other")) == num_seqs - 1 - seq_idx


def test_HuggingFaceDataset_predefined_seq_order():
    import datasets

    datadir_path = _get_tmp_dir() + "/hf-dataset-predefined-seq-order"
    datasets.Dataset.from_dict({"text": [f"text {i}" for i in range(5)]}).save_to_disk(datadir_path)
    ds = HuggingFaceDataset(datadir_path, seq_tag_column=None, data_format={"text": {"dtype": "string", "shape": ()}})
    ds.initialize()
    check_predefined_seq_order(ds)
