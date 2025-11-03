from __future__ import annotations

import _setup_test_env  # noqa
import os
import tempfile
import atexit
import shutil
import pickle
import numpy

from returnn.datasets import init_dataset
from returnn.datasets.huggingface import HuggingFaceDataset
from test_Dataset import dummy_iter_dataset


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
