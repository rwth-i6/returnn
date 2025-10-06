from __future__ import annotations

import _setup_test_env  # noqa
import os
import tempfile
import atexit
import shutil
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
    ds = HuggingfaceDataset({"path": "datasets-examples/doc-audio-6", "split": "train"}, seq_tag_column=None)
    ds.initialize()
    assert dummy_iter_dataset(ds)
