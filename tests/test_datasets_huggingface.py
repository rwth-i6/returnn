from __future__ import annotations

import _setup_test_env  # noqa
from returnn.datasets.huggingface import HuggingfaceDataset
from test_Dataset import dummy_iter_dataset


def test_HuggingfaceDataset():
    ds = HuggingfaceDataset({"path": "datasets-examples/doc-audio-6", "split": "train"}, seq_tag_key=None)
    ds.initialize()
    dummy_iter_dataset(ds)
