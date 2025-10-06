from __future__ import annotations

import _setup_test_env  # noqa
import os
import tempfile
import atexit
import shutil
import pickle
from returnn.datasets.huggingface import HuggingfaceDataset
from test_Dataset import dummy_iter_dataset


def _setup_hf_env():
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = _get_tmp_dir()


def _get_tmp_dir() -> str:
    fn = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, fn)
    return fn


_setup_hf_env()


def test_HuggingfaceDataset_audio():
    ds = HuggingfaceDataset(
        {"path": "datasets-examples/doc-audio-6", "split": "train"},
        cast_columns={"audio": {"_type": "Audio", "sample_rate": 16_000}},
        data_format={"audio": {"dtype": "float32", "shape": [None]}},
        seq_tag_column=None,
    )
    ds.initialize()
    res = dummy_iter_dataset(ds)
    print(res[0].features["audio"])


def test_HuggingfaceDataset_text1():
    ds = HuggingfaceDataset(
        {"path": "openai/gdpval", "split": "train"},
        seq_tag_column="task_id",
        data_format={
            "prompt": {"dtype": "string", "shape": ()},
            "sector": {"dtype": "string", "shape": ()},
            "occupation": {"dtype": "string", "shape": ()},
        },
    )
    ds.initialize()
    assert dummy_iter_dataset(ds)


def test_HuggingfaceDataset_text2():
    ds = HuggingfaceDataset(
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


def test_HuggingfaceDataset_pickle():
    ds = HuggingfaceDataset(
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
    assert isinstance(ds, HuggingfaceDataset)
    assert dummy_iter_dataset(ds)


def test_HuggingfaceDataset_load_from_disk():
    from datasets import load_dataset

    datadir_path = _get_tmp_dir() + "/hf-dataset-save-to-disk"
    hf_ds = load_dataset("lavita/medical-qa-shared-task-v1-toy", split="train")
    hf_ds.save_to_disk(datadir_path)

    ds = HuggingfaceDataset(
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


def test_HuggingfaceDataset_single_arrow():
    # TODO...
    ds = HuggingfaceDataset(
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
