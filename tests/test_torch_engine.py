"""
Tests for PyTorch engine.
"""

from __future__ import annotations
import _setup_test_env  # noqa
from typing import Optional, Any, Dict, List, Tuple
import contextlib
import copy
import json
import os
import sys
import unittest
import unittest.mock
import tempfile
import numpy
import torch

from returnn.util import better_exchook
from returnn.config import Config, global_config_ctx
from returnn.tensor import TensorDict, Tensor, Dim
from returnn.torch.engine import Engine
from returnn.torch.updater import Updater
import returnn.frontend as rf
from returnn.forward_iface import ForwardCallbackIface
from returnn.datasets import init_dataset
from returnn.datasets.generating import StaticDataset


# must be in the global scope due to pickling
class TrainTestModel(torch.nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.lin = torch.nn.Linear(9, 2)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B,T,D]
        :return: [B,T,D']
        """
        x = self.lin(x)
        return torch.nn.functional.log_softmax(x, dim=-1)

    @classmethod
    def train_step(cls, *, model: TrainTestModel, extern_data: TensorDict, **_kwargs):
        """train step"""
        data: Tensor = extern_data["data"]
        logits = model(data.raw_tensor)
        logits_packed = torch.nn.utils.rnn.pack_padded_sequence(
            logits, data.dims[1].dyn_size_ext.raw_tensor, batch_first=True, enforce_sorted=False
        )
        targets = extern_data["classes"]
        targets_packed = torch.nn.utils.rnn.pack_padded_sequence(
            targets.raw_tensor, data.dims[1].dyn_size_ext.raw_tensor, batch_first=True, enforce_sorted=False
        )
        loss = torch.nn.CrossEntropyLoss(reduction="none")(logits_packed.data, targets_packed.data.long())
        rf.get_run_ctx().mark_as_loss(name="ce", loss=loss)
        frame_error = torch.argmax(logits_packed.data, dim=-1).not_equal(targets_packed.data)
        rf.get_run_ctx().mark_as_loss(name="fer", loss=frame_error, as_error=True)


def test_torch_engine_train():
    config = Config(
        dict(
            task="train",
            device="cpu",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainTestModel,
            train_step=TrainTestModel.train_step,
            batch_size=500,
            optimizer={"class": "adam"},
        )
    )
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "train"})
    dataset.init_seq_order(epoch=1)

    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()


_packed_time_dim = Dim(None, name="time")
_packed_in_dim = Dim(9, name="in")
_packed_classes_dim = Dim(2, name="classes")
_packed_train_losses = []
_packed_train_is_packed = []  # per step: whether extern_data["data"] arrived packed


# must be in the global scope due to pickling
class RFPackedTrainModel(rf.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.lin = rf.Linear(_packed_in_dim, _packed_classes_dim)

    @classmethod
    def train_step(cls, *, model: RFPackedTrainModel, extern_data: TensorDict, **_kwargs):
        """train step, same code for padded and packed extern_data"""
        logits = model.lin(extern_data["data"])
        loss = rf.cross_entropy(
            estimated=logits, target=extern_data["classes"], axis=_packed_classes_dim, estimated_type="logits"
        )
        rf.get_run_ctx().mark_as_loss(name="ce", loss=loss)
        _packed_train_losses.append(float(rf.reduce_sum(loss, axis=loss.dims, use_mask=True).raw_tensor))
        from returnn.frontend import _packed_backend

        _packed_train_is_packed.append(_packed_backend.is_packed(extern_data["data"]))


def _run_packed_train(packed_tensors):
    from returnn.tensor import batch_dim
    from returnn.frontend import _packed_backend

    _packed_train_losses.clear()
    _packed_train_is_packed.clear()
    warned_before = set(_packed_backend._warned_fallback_ops)
    _packed_backend._warned_fallback_ops.clear()
    config = Config(
        dict(
            task="train",
            device="cpu",
            random_seed=42,
            extern_data={
                "data": {"dims": [batch_dim, _packed_time_dim, _packed_in_dim], "dtype": "float32"},
                "classes": {
                    "dims": [batch_dim, _packed_time_dim],
                    "sparse_dim": _packed_classes_dim,
                    "dtype": "int32",
                },
            },
            get_model=RFPackedTrainModel,
            train_step=RFPackedTrainModel.train_step,
            batch_size=500,
            optimizer={"class": "adam"},
            torch_dataloader_opts={"num_workers": 0},
            packed_tensors=packed_tensors,
        )
    )
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 20, "name": "train", "fixed_random_seed": 1})
    dataset.init_seq_order(epoch=1)
    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()
    warnings = set(_packed_backend._warned_fallback_ops)
    _packed_backend._warned_fallback_ops.clear()
    _packed_backend._warned_fallback_ops.update(warned_before)
    return list(_packed_train_losses), list(_packed_train_is_packed), warnings


def test_torch_engine_train_packed():
    # full packed data pipeline: collate_batch(packing) -> raw_dict_to_extern_data (packed) ->
    # model (RF, unchanged) -> loss -> backprop. Per-step losses must match the padded run,
    # both dense (gap 0) and with a gap in the packing (the gap frames must not affect the loss).
    losses_padded, is_packed_padded, _ = _run_packed_train(False)
    assert not any(is_packed_padded)  # padded run: nothing is packed
    for packed_tensors in [True, {"gap": 8, "align": 2}, {"per_key": {"data": {"gap": 8, "align": 2}}}]:
        losses_packed, is_packed, warnings = _run_packed_train(packed_tensors)
        assert is_packed and all(is_packed), (packed_tensors, is_packed)  # data actually packed each step
        assert not warnings, (packed_tensors, warnings)  # packed ops took the fast path, no unpack fallback
        assert losses_packed and all(numpy.isfinite(losses_packed)), (packed_tensors, losses_packed)
        assert len(losses_padded) == len(losses_packed)
        numpy.testing.assert_allclose(losses_packed, losses_padded, rtol=1e-4, atol=1e-4, err_msg=repr(packed_tensors))


def test_raw_dict_split_batch_packed():
    # packed OOM auto-split: split by sequences, each key sliced at its own frame boundaries.
    from returnn.torch.data.extern_data import raw_dict_can_split_batch, raw_dict_split_batch

    data_lens, cls_lens = [4, 2, 3], [2, 1, 2]
    raw = {
        "data": torch.arange(sum(data_lens) * 5, dtype=torch.float32).reshape(sum(data_lens), 5),
        "data:seq_len": torch.tensor(data_lens, dtype=torch.int32),
        "data:packed": {"gap": 0, "align": 1},
        "classes": torch.arange(sum(cls_lens), dtype=torch.int32),
        "classes:seq_len": torch.tensor(cls_lens, dtype=torch.int32),
        "classes:packed": {"gap": 0, "align": 1},
        "seq_tag": numpy.array(["s0", "s1", "s2"]),
        "num_seqs": 3,
    }
    assert raw_dict_can_split_batch(raw, num_splits=2)
    assert not raw_dict_can_split_batch(raw, num_splits=4)  # only 3 seqs
    parts = raw_dict_split_batch(raw, splits=2)  # seqs [0,1] and [2]
    assert len(parts) == 2
    d_cs, c_cs = [0, 4, 6, 9], [0, 2, 3, 5]
    for part, (lo, hi) in [(parts[0], (0, 2)), (parts[1], (2, 3))]:
        torch.testing.assert_close(part["data"], raw["data"][d_cs[lo] : d_cs[hi]])
        torch.testing.assert_close(part["data:seq_len"], raw["data:seq_len"][lo:hi])
        assert part["data:packed"] == {"gap": 0, "align": 1}
        torch.testing.assert_close(part["classes"], raw["classes"][c_cs[lo] : c_cs[hi]])
        torch.testing.assert_close(part["classes:seq_len"], raw["classes:seq_len"][lo:hi])
        assert list(part["seq_tag"]) == list(raw["seq_tag"][lo:hi])
        assert part["num_seqs"] == 3
    torch.testing.assert_close(torch.cat([parts[0]["data"], parts[1]["data"]], dim=0), raw["data"])


def test_get_batch_size_info_raw():
    from returnn.torch.engine import _get_batch_size_info_raw

    raw = {
        "data": torch.zeros(3, 4, 5),
        "data:seq_len": torch.tensor([4, 2, 3], dtype=torch.int32),
        "classes": numpy.zeros((3, 2), dtype="int32"),
        "classes:seq_len": numpy.array([2, 1, 2], dtype="int32"),
    }
    info = _get_batch_size_info_raw(raw)
    assert info == {"num_seqs": 3, "max_size:data": 4, "sum_size:data": 9, "max_size:classes": 2, "sum_size:classes": 5}
    assert all(type(v) is int for v in info.values())

    empty = {"data": torch.zeros(0, 0, 5), "data:seq_len": torch.zeros(0, dtype=torch.int32)}
    assert _get_batch_size_info_raw(empty) == {"num_seqs": 0, "max_size:data": 0, "sum_size:data": 0}

    # int32 total beyond 2**31 must not overflow
    large = {"data": torch.zeros(3, 1), "data:seq_len": torch.full((3,), 2**30, dtype=torch.int32)}
    assert _get_batch_size_info_raw(large)["sum_size:data"] == 3 * 2**30


def test_torch_engine_forward_simple():
    def _get_model(**_kwargs):
        return torch.nn.Module()

    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        rf.get_run_ctx().mark_as_default_output(extern_data["data"])

    config = Config(
        dict(
            task="forward",
            extern_data={"data": {"dim": 9}},
            batch_size=500,
            get_model=_get_model,
            forward_step=_forward_step,
        )
    )
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "dev", "fixed_random_seed": 1})
    dataset.init_seq_order(epoch=1)
    callback = ForwardCallbackIface()

    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_network_from_config()
        engine.forward_with_callback(callback=callback, dataset=dataset)


def test_torch_engine_forward():
    def _get_model(**_kwargs):
        return torch.nn.Module()

    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        rf.get_run_ctx().mark_as_default_output(extern_data["data"])

    class _ForwardCallback(ForwardCallbackIface):
        def __init__(self):
            self.num_seqs = 0
            self.init_called = False
            self.finish_called = False

        def init(self, *, model):
            assert isinstance(model, torch.nn.Module)
            assert self.num_seqs == 0
            self.init_called = True

        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            assert isinstance(seq_tag, str) and seq_tag.startswith("seq-")
            assert isinstance(outputs, TensorDict)
            out = outputs["output"]
            assert isinstance(out, Tensor)
            assert out.batch_ndim == 2 and out.batch_shape[-1] == 9
            self.num_seqs += 1

        def finish(self):
            self.finish_called = True

    config = Config(
        dict(
            task="forward",
            extern_data={"data": {"dim": 9}},
            batch_size=500,
            get_model=_get_model,
            forward_step=_forward_step,
        )
    )
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "dev", "fixed_random_seed": 1})
    dataset.init_seq_order(epoch=1)
    callback = _ForwardCallback()

    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_network_from_config()
        engine.forward_with_callback(callback=callback, dataset=dataset)
        assert callback.num_seqs == 100
        assert callback.init_called and callback.finish_called


def test_torch_engine_forward_pure_torch_no_model_out():
    # https://github.com/rwth-i6/returnn/issues/1385
    # Automatically assume that we have batch-dim first in mark_as_output with raw tensor.
    def _get_model(**_kwargs):
        return torch.nn.Module()

    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        rf.get_run_ctx().mark_as_default_output(extern_data["data"].raw_tensor)

    config = Config(
        dict(
            task="forward",
            extern_data={"data": {"dim": 9}},
            batch_size=500,
            get_model=_get_model,
            forward_step=_forward_step,
        )
    )
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "dev", "fixed_random_seed": 1})
    dataset.init_seq_order(epoch=1)
    callback = ForwardCallbackIface()

    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_network_from_config()
        engine.forward_with_callback(callback=callback, dataset=dataset)


def test_torch_forward_raw_strings():
    # In OggZipDataset, but maybe also other datasets,
    # in combination with forward task, we get all kind of different string formats:
    #   - seq_tag (numpy U8 str)
    #   - raw (numpy object -> str)
    #   - orth (uint8 bytes)
    # Test all of them.

    from test_Dataset import create_ogg_zip_txt_only_dataset

    def _get_model(**_kwargs):
        return torch.nn.Module()

    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        for key, value in extern_data.data.items():
            rf.get_run_ctx().mark_as_output(value, key)

    config = Config(
        dict(
            task="forward",
            extern_data={
                "classes": {"shape": (None,), "dim": 29, "sparse": True},
                "orth": {"shape": (None,), "dim": 256, "sparse": True},
                "raw": {"shape": (), "dtype": "string"},
            },
            batch_size=500,
            get_model=_get_model,
            forward_step=_forward_step,
        )
    )

    _demo_txt = "hello world"
    _demo_seq_tag = "seq-000000"

    class _ForwardCallback(ForwardCallbackIface):
        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            assert isinstance(seq_tag, str) and seq_tag == _demo_seq_tag
            raw = outputs["raw"].raw_tensor
            orth = outputs["orth"].raw_tensor
            classes = outputs["classes"].raw_tensor
            assert isinstance(raw, numpy.ndarray) and raw.dtype.name.startswith("str") and raw.shape == ()
            raw_ = raw.item()
            assert isinstance(raw_, str) and raw_ == _demo_txt
            assert isinstance(orth, numpy.ndarray) and orth.dtype == numpy.uint8 and orth.ndim == 1
            orth_ = orth.tobytes()
            assert orth_.decode("utf8") == _demo_txt
            assert isinstance(classes, numpy.ndarray) and classes.dtype == numpy.int32 and classes.ndim == 1
            classes_ = "".join([dataset.targets.id_to_label(c) for c in classes])
            assert classes_ == _demo_txt + "."

    with global_config_ctx(config), create_ogg_zip_txt_only_dataset(text=_demo_txt, seq_tag=_demo_seq_tag) as dataset:
        dataset.init_seq_order(epoch=1)
        engine = Engine(config=config)
        engine.init_network_from_config()
        engine.forward_with_callback(callback=_ForwardCallback(), dataset=dataset)


def test_forward_beam_seq_lens():
    from returnn.tensor import Dim, batch_dim

    def _get_model(**_kwargs):
        return torch.nn.Module()

    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        data = extern_data["data"]  # [batch, time, dim]
        assert data.dims[0] == batch_dim
        time_dim = data.dims[1]
        feat_dim = data.dims[2]
        beam_dim = Dim(dimension=5, name="beam")
        with rf.set_default_device_ctx(time_dim.dyn_size_ext.device):
            ext_seq_lens = rf.relu(
                rf.combine_bc(
                    time_dim.dyn_size_ext, "-", rf.range_over_dim(beam_dim, dtype=time_dim.dyn_size_ext.dtype)
                )
            )
        assert set(ext_seq_lens.dims) == {batch_dim, beam_dim}
        ext_time_dim = Dim(ext_seq_lens, name="time_with_beam")
        ext_data = rf.expand_dim(data, beam_dim)
        ext_data, _ = rf.replace_dim(ext_data, in_dim=time_dim, out_dim=ext_time_dim)
        assert set(ext_data.dims) == {batch_dim, beam_dim, ext_time_dim, feat_dim}
        rf.get_run_ctx().mark_as_output(ext_data, "ext_data", dims=(batch_dim, beam_dim, ext_time_dim, feat_dim))

    max_sizes = set()

    class _ForwardCallback(ForwardCallbackIface):
        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            out: Tensor = outputs["ext_data"]
            beam_dim, ext_time_dim, feat_dim = out.dims
            assert isinstance(ext_time_dim.dyn_size_ext.raw_tensor, numpy.ndarray)
            assert ext_time_dim.dyn_size_ext.dims == (beam_dim,)
            max_size = max(ext_time_dim.dyn_size_ext.raw_tensor)
            assert set(ext_time_dim.dyn_size_ext.raw_tensor) == set(
                range(max(max_size - beam_dim.dimension + 1, 0), max_size + 1)
            )
            max_sizes.add(max_size)

    config = Config(
        dict(
            task="forward",
            batch_size=500,
            extern_data={"data": {"dim": 9}},
            get_model=_get_model,
            forward_step=_forward_step,
        )
    )
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "dev", "fixed_random_seed": 1})
    callback = _ForwardCallback()

    with global_config_ctx(config):
        dataset.init_seq_order(epoch=1)
        engine = Engine(config=config)
        engine.init_network_from_config()
        engine.forward_with_callback(callback=callback, dataset=dataset)
        assert len(max_sizes) > 1


def test_torch_engine_forward_dataset_epoch():
    import tempfile
    import shutil
    import atexit
    import os
    import returnn

    model_dir_name = tempfile.mkdtemp()
    assert model_dir_name and os.path.isdir(model_dir_name) and not os.listdir(model_dir_name)
    atexit.register(lambda: shutil.rmtree(model_dir_name))

    in_dim, out_dim = 9, 13

    def _get_model(**_kwargs):
        return torch.nn.Linear(in_dim, out_dim)

    epoch = 17
    filename = Engine.epoch_model_filename(f"{model_dir_name}/model", epoch=epoch) + ".pt"

    # That's how RETURNN now saves the model (2024-10-25).
    # Maybe leave it like this for the test, even when RETURNN itself changes it,
    # so that we also test that we still support this format.
    torch.save(
        {
            "model": _get_model().state_dict(),  # some random model
            "epoch": epoch,
            "step": 123,
            "effective_learning_rate": 0.13,
            "returnn_version": returnn.__long_version__,
        },
        filename,
    )

    recent_seen_seq_idx: Optional[int] = None

    class _ForwardCallback(ForwardCallbackIface):
        def process_seq(self, *, seq_tag: str, outputs: TensorDict):
            print("*** forward callback process seq", seq_tag)
            d = eval(seq_tag)  # we prepared the dataset this way that we get some dict repr here...
            assert isinstance(d, dict)
            assert d["epoch"] == epoch
            nonlocal recent_seen_seq_idx
            seq_idx = d["seq_idx"]
            if seq_idx == 0:
                assert recent_seen_seq_idx is None
            else:
                assert recent_seen_seq_idx is not None
                assert seq_idx == recent_seen_seq_idx + 1
            recent_seen_seq_idx = seq_idx

    forward_callback = _ForwardCallback()

    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        print("*** forward step", extern_data)
        data = extern_data["data"]
        # Doesn't matter what we set as output here, not used...
        # (Without output, maybe RETURNN complains, so put sth.)
        # We just use the seq_tag in the forward callback, which is anyway available.
        data.mark_as_default_output(shape=data.dims)

    config = Config(
        dict(
            task="forward",
            batch_size=50,
            extern_data={"data": {"dim": in_dim}},
            get_model=_get_model,
            load=filename,
            forward_step=_forward_step,
            torch_dataloader_opts=dict(num_workers=0),  # simplifies the test
        )
    )

    from returnn.datasets.cached2 import CachedDataset2
    from returnn.datasets.basic import DatasetSeq

    num_seqs = 10

    class _MyDataset(CachedDataset2):
        def __init__(self):
            super().__init__()
            self.num_inputs = in_dim
            self.num_outputs = {"classes": out_dim}

        # noinspection PyShadowingNames
        def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
            """init seq order"""
            super().init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
            self._num_seqs = num_seqs

        def _collect_single_seq(self, seq_idx: int) -> Optional[DatasetSeq]:
            if seq_idx >= self._num_seqs:
                return None
            return DatasetSeq(
                seq_idx=seq_idx,
                seq_tag=repr({"epoch": self.epoch, "seq_idx": seq_idx}),
                features=numpy.zeros((10, in_dim)),
                targets={"classes": numpy.zeros((10,), dtype=numpy.int32)},
            )

    dataset = _MyDataset()
    dataset.initialize()

    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_network_from_config()
        # We expect that the engine epoch is set to the epoch of the checkpoint.
        assert engine.epoch == epoch

        for epoch in [3, 7, 11]:
            engine.set_epoch(epoch)
            assert engine.epoch == epoch
            dataset.init_seq_order(epoch=epoch)
            assert dataset.num_seqs == num_seqs
            recent_seen_seq_idx = None
            engine.forward_with_callback(callback=forward_callback, dataset=dataset)
            assert recent_seen_seq_idx == num_seqs - 1


def test_torch_engine_forward_load_epoch():
    import tempfile
    import shutil
    import atexit
    import os
    import returnn

    model_dir_name = tempfile.mkdtemp()
    assert model_dir_name and os.path.isdir(model_dir_name) and not os.listdir(model_dir_name)
    atexit.register(lambda: shutil.rmtree(model_dir_name))

    in_dim, out_dim = 9, 13

    def _get_model(**_kwargs):
        return torch.nn.Linear(in_dim, out_dim)

    epoch = 17
    load_epoch = 11  # some other epoch
    filename = Engine.epoch_model_filename(f"{model_dir_name}/model", epoch=epoch) + ".pt"

    # That's how RETURNN now saves the model (2024-10-25).
    # Maybe leave it like this for the test, even when RETURNN itself changes it,
    # so that we also test that we still support this format.
    torch.save(
        {
            "model": _get_model().state_dict(),  # some random model
            "epoch": epoch,
            "step": 123,
            "effective_learning_rate": 0.13,
            "returnn_version": returnn.__long_version__,
        },
        filename,
    )

    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        print("*** forward step", extern_data)
        data = extern_data["data"]
        data.mark_as_default_output(shape=data.dims)  # dummy...

    config = Config(
        dict(
            task="forward",
            batch_size=50,
            extern_data={"data": {"dim": in_dim}},
            get_model=_get_model,
            load=filename,
            load_epoch=load_epoch,
            forward_step=_forward_step,
            torch_dataloader_opts=dict(num_workers=0),  # simplifies the test
        )
    )

    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_network_from_config()
        # We expect that even though we loaded the checkpoint, we now have the load_epoch.
        assert engine.epoch == load_epoch


def test_min_seq_len():
    from returnn.datasets.generating import DummyDataset

    config = Config({"min_seq_length": 2, "batch_size": 3})
    dataset = DummyDataset(input_dim=1, output_dim=4, num_seqs=1, seq_len=1)
    dataset.initialize()
    dataset.init_seq_order(epoch=1)
    engine = Engine(config=config)
    engine.set_epoch(1)
    data_loader = engine._create_data_loader(dataset)
    for _ in data_loader:
        assert False, "Should not contain sequences"

    config = Config(dict(batch_size=3))
    dataset = DummyDataset(input_dim=1, output_dim=4, num_seqs=1, seq_len=3)
    dataset.initialize()
    dataset.init_seq_order(epoch=1)
    engine = Engine(config=config)
    engine.set_epoch(1)
    data_loader = engine._create_data_loader(dataset)
    for _ in data_loader:
        return
    assert False, "Should have contained sequences"


def test_max_seq_len():
    from returnn.datasets.generating import DummyDataset

    config = Config({"max_seq_length": 4, "batch_size": 3})
    dataset = DummyDataset(input_dim=1, output_dim=4, num_seqs=1, seq_len=5)
    dataset.initialize()
    dataset.init_seq_order(epoch=1)
    engine = Engine(config=config)
    engine.set_epoch(1)
    data_loader = engine._create_data_loader(dataset)
    for _ in data_loader:
        assert False, "Should not contain sequences"

    config = Config(dict(batch_size=3))
    dataset = DummyDataset(input_dim=1, output_dim=4, num_seqs=1, seq_len=3)
    dataset.initialize()
    dataset.init_seq_order(epoch=1)
    engine = Engine(config=config)
    engine.set_epoch(1)
    data_loader = engine._create_data_loader(dataset)
    for _ in data_loader:
        return
    assert False, "Should have contained sequences"


def _run_maybe_stop_for_resubmission(*, epoch: int, final_epoch: int, time_left: int):
    """
    :return: (number of SLURM time-left queries, signals sent via os.kill)
    """
    import os
    from unittest import mock

    config = Config({"stop_for_resubmission_when_low_time_left": True})
    engine = Engine(config=config)
    engine._final_epoch = final_epoch  # normally set by init_train_from_config
    engine.set_epoch(epoch)
    with mock.patch("returnn.util.basic.slurm_time_left_sec", return_value=time_left) as time_left_mock:
        with mock.patch.object(os, "kill") as kill_mock:
            engine._maybe_stop_for_resubmission(last_epoch_wall_sec=100.0)
    return time_left_mock.call_count, [call.args[1] for call in kill_mock.call_args_list]


def test_stop_for_resubmission_final_epoch():
    # low wall-time left, but the training is complete: no scheduler query, no signal
    assert _run_maybe_stop_for_resubmission(epoch=38, final_epoch=38, time_left=10) == (0, [])


def test_stop_for_resubmission_beyond_final_epoch():
    assert _run_maybe_stop_for_resubmission(epoch=39, final_epoch=38, time_left=10) == (0, [])


def test_stop_for_resubmission_low_time_left():
    import signal

    assert _run_maybe_stop_for_resubmission(epoch=37, final_epoch=38, time_left=10) == (1, [signal.SIGINT])


def test_stop_for_resubmission_enough_time_left():
    assert _run_maybe_stop_for_resubmission(epoch=37, final_epoch=38, time_left=1000) == (1, [])


def test_data_loader_oggzip():
    from test_Dataset import create_ogg_zip_txt_only_dataset_mult_seqs

    ds_num_seqs = 23
    ds_max_seq_len = 11
    max_seqs = 3
    config = Config({"max_seqs": max_seqs, "batch_size": max_seqs * ds_max_seq_len})
    with create_ogg_zip_txt_only_dataset_mult_seqs(num_seqs=ds_num_seqs, max_seq_len=ds_max_seq_len) as dataset:
        dataset.init_seq_order(epoch=1)
        engine = Engine(config=config)
        engine.set_epoch(1)
        data_loader = engine._create_data_loader(dataset)
        num_batches = 0
        num_seqs = 0
        last_batch_num_seqs = None
        for batch in data_loader:
            assert isinstance(batch, dict)
            data: torch.Tensor = batch["classes"]
            assert isinstance(data, torch.Tensor)
            num_batches += 1
            num_seqs += data.shape[0]
            if last_batch_num_seqs is not None:
                assert last_batch_num_seqs == max_seqs
            last_batch_num_seqs = data.shape[0]
        assert 1 <= last_batch_num_seqs <= max_seqs
        assert num_batches == -(-num_seqs // max_seqs) and num_seqs == ds_num_seqs

    ds_num_seqs = 5
    ds_max_seq_len = 5
    max_seqs = 2
    config = Config({"max_seqs": max_seqs, "batch_size": max_seqs * ds_max_seq_len})
    batches = []
    with create_ogg_zip_txt_only_dataset_mult_seqs(num_seqs=ds_num_seqs, max_seq_len=ds_max_seq_len) as dataset:
        dataset.init_seq_order(epoch=1)
        engine = Engine(config=config)
        engine.set_epoch(1)
        data_loader = engine._create_data_loader(dataset)
        for batch in data_loader:
            assert isinstance(batch, dict)
            data: torch.Tensor = batch["classes"]
            batches.append(data.numpy().tolist())
    print(batches)
    # The following depends on the random data generation in create_ogg_zip_txt_only_dataset_mult_seqs,
    # but we fixed the seed and the random number generator, so this should stay the same, unless we change the code.
    assert batches == [[[12, 8, 9, 11], [16, 0, 0, 0]], [[6, 25, 18, 20, 5], [28, 10, 28, 14, 0]], [[17, 23]]]


def test_load_optimizer_old_format():
    config = Config(dict(optimizer={"class": "adamw", "weight_decay": 1e-3}))
    model = torch.nn.Linear(7, 5)
    updater = Updater(config=config, network=model, device=torch.device("cpu"))
    updater.create_optimizer()

    with tempfile.TemporaryDirectory(prefix="returnn_test_load_optimizer_old_format") as tmp_dir:
        torch.save(updater.optimizer.state_dict(), tmp_dir + "/model.opt.old_format.pt")
        updater.load_optimizer(tmp_dir + "/model.opt.old_format.pt")

        updater.save_optimizer(tmp_dir + "/model.opt.new_format.pt")
        updater.load_optimizer(tmp_dir + "/model.opt.new_format.pt")


def test_updater_weight_decay_blacklist():
    from returnn.util.basic import DictRefKeys

    # Don't specify weight_decay_modules_blacklist, so it should use the default,
    # which should exclude Embedding and LayerNorm, and all biases.
    # So this also tests that the default behavior does not change unexpectedly.
    config = Config(dict(optimizer={"class": "adamw", "weight_decay": 1e-3}))
    model = torch.nn.Sequential(
        torch.nn.Embedding(10, 5),
        torch.nn.LayerNorm(5),
        torch.nn.Linear(5, 5),
        torch.nn.ReLU(),
    )
    updater = Updater(config=config, network=model, device=torch.device("cpu"))
    updater.create_optimizer()
    updater.set_current_train_step(global_train_step=0, epoch=1)

    opt = updater.get_optimizer()
    assert isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups) == 2
    groups_by_wd = {pg.get("weight_decay", 0.0): pg for pg in opt.param_groups}
    assert set(groups_by_wd.keys()) == {0.0, 1e-3}
    param_to_name = DictRefKeys((param, name) for name, param in model.named_parameters())
    params_by_wd = {wd: set(map(param_to_name.__getitem__, group["params"])) for wd, group in groups_by_wd.items()}
    print("params by wd:", params_by_wd)
    assert params_by_wd[0.0] == {"0.weight", "1.weight", "1.bias", "2.bias"}
    assert params_by_wd[1e-3] == {"2.weight"}


def test_updater_step_does_not_walk_the_modules():
    """the grad clip and norm take the parameters of the optimizer creation, walking the modules costs every step"""
    from unittest import mock

    config = Config(dict(optimizer={"class": "adamw"}, gradient_clip_global_norm=1.0, log_grad_norm=True))
    model = torch.nn.Sequential(torch.nn.Linear(5, 5), torch.nn.Linear(5, 2))
    updater = Updater(config=config, network=model, device=torch.device("cpu"))
    updater.create_optimizer()
    model(torch.randn(3, 5)).sum().backward()
    with mock.patch.object(torch.nn.Module, "named_modules", side_effect=AssertionError("walked the modules")):
        updater.step()
    assert updater.last_grad_norm is not None


def test_updater_lr_multipliers():
    from collections import defaultdict
    from fnmatch import fnmatchcase
    from typing import List, Set
    from returnn.util.basic import DictRefKeys, FrozenDict
    from returnn.torch.updater import wrap_user_blacklist_wd_modules
    from returnn.torch.frontend.bridge import wrapped_pt_module_to_rf_module

    # noinspection PyShadowingNames
    def _param_groups_custom(*, model: torch.nn.Module, optimizer_opts: Dict[str, Any], **_kwargs):
        default_weight_decay = optimizer_opts.get("weight_decay", 0.0)

        blacklist_wd_modules = wrap_user_blacklist_wd_modules(
            optimizer_opts.pop("weight_decay_modules_blacklist", None)
        )
        lr_multipliers_by_patterns = optimizer_opts.pop("learning_rate_multipliers_by_patterns")

        # Tracker of visited parameters to only add each parameter once, in case two modules share common parameters.
        # We need the wrapper class RefIdEq because Parameters are compared by value and not by reference.
        params_by_opts: defaultdict[FrozenDict, List[torch.nn.Parameter]] = defaultdict(list)
        visited_params = DictRefKeys()
        for module_name, module in model.named_modules():
            module_name: str
            module: torch.nn.Module
            rf_module = wrapped_pt_module_to_rf_module(module)
            for param_name, param in module.named_parameters(recurse=False):
                param_name: str
                param: torch.nn.Parameter
                if param in visited_params:
                    continue
                visited_params[param] = True
                full_param_name = "%s.%s" % (module_name, param_name) if module_name else param_name

                opts = {}
                if (
                    param_name.endswith("bias")
                    or isinstance(module, blacklist_wd_modules)
                    or isinstance(rf_module, blacklist_wd_modules)
                ):
                    opts["weight_decay"] = 0.0
                else:
                    opts["weight_decay"] = default_weight_decay
                for pattern, lr_multiplier in lr_multipliers_by_patterns.items():
                    if fnmatchcase(full_param_name, pattern):
                        if lr_multiplier != 1.0:
                            opts["learning_rate_multiplier"] = lr_multiplier
                        break
                params_by_opts[FrozenDict(opts)].append(param)

        return [{"params": params, **opts} for opts, params in params_by_opts.items()]

    config = Config(
        dict(
            optimizer={
                "class": "adamw",
                "weight_decay": 1e-3,
                "param_groups_custom": _param_groups_custom,
                "learning_rate_multipliers_by_patterns": {"0.*": 1.0, "1.*": 0.5, "2.*": 0.1},
            }
        )
    )
    model = torch.nn.Sequential(
        torch.nn.Embedding(10, 5),
        torch.nn.LayerNorm(5),
        torch.nn.Linear(5, 5),
        torch.nn.ReLU(),
    )
    updater = Updater(config=config, network=model, device=torch.device("cpu"))
    updater.create_optimizer()
    updater.set_current_train_step(global_train_step=0, epoch=1)

    param_to_name = DictRefKeys((param, name) for name, param in model.named_parameters())
    opt = updater.get_optimizer()
    param_names_by_opts: Dict[FrozenDict, Set[str]] = {}
    for group in opt.param_groups:
        group_opts = FrozenDict({k: group[k] for k in ["weight_decay", "lr"]})
        assert group_opts not in param_names_by_opts  # unique
        param_names_by_opts[group_opts] = {param_to_name[p] for p in group["params"]}
    assert len(param_names_by_opts) == 4, "Expected 4 param groups"
    for opts, ref_param_names in [
        ({"weight_decay": 0.0, "lr": 1.0}, {"0.weight"}),
        ({"weight_decay": 0.0, "lr": 0.5}, {"1.weight", "1.bias"}),
        ({"weight_decay": 0.001, "lr": 0.1}, {"2.weight"}),
        ({"weight_decay": 0.0, "lr": 0.1}, {"2.bias"}),
    ]:
        opts = FrozenDict(opts)
        assert opts in param_names_by_opts, f"Expected param group with opts {opts} not found"
        param_names = param_names_by_opts[opts]
        assert param_names == ref_param_names, (
            f"For opts {opts}, expected param names {ref_param_names} but got {param_names}"
        )


def test_optimizer_convert_aux_param():
    # See rf_module_to_pt_module aux_params_as_buffers option.
    # This causes a change in the optimizer state dict.
    # But we should be able to convert it back, in both directions.

    from returnn.torch.frontend.bridge import rf_module_to_pt_module

    config = Config(dict(optimizer={"class": "adamw", "weight_decay": 1e-3}))
    rf.select_backend_torch()

    class _Model(rf.Module):
        def __init__(self):
            super().__init__()
            self.batch_norm = rf.BatchNorm(in_dim=rf.Dim(3))
            self.linear = rf.Linear(in_dim=rf.Dim(2), out_dim=rf.Dim(3))

    rf_model = _Model()
    pt_model_buf = rf_module_to_pt_module(rf_model, aux_params_as_buffers=True)
    pt_model_param = rf_module_to_pt_module(rf_model, aux_params_as_buffers=False)
    pt_model_buf_param_names = set(name for name, _ in pt_model_buf.named_parameters())
    pt_model_param_param_names = set(name for name, _ in pt_model_param.named_parameters())
    print("buf params:", pt_model_buf_param_names)
    print("all params:", pt_model_param_param_names)
    assert len(pt_model_buf_param_names) < len(pt_model_param_param_names)
    assert pt_model_buf_param_names.issubset(pt_model_param_param_names)
    updater_buf = Updater(config=config, network=pt_model_buf, device=torch.device("cpu"))
    updater_buf.create_optimizer()
    updater_param = Updater(config=config, network=pt_model_param, device=torch.device("cpu"))
    updater_param.create_optimizer()

    with tempfile.TemporaryDirectory(prefix="returnn_test_optimizer_convert_aux_param") as tmp_dir:
        updater_buf.save_optimizer(tmp_dir + "/model_buf.opt.pt")
        updater_param.save_optimizer(tmp_dir + "/model_param.opt.pt")
        updater_buf.load_optimizer(tmp_dir + "/model_buf.opt.pt")
        updater_param.load_optimizer(tmp_dir + "/model_param.opt.pt")
        # Ok, now test whether we can convert them.
        updater_buf.load_optimizer(tmp_dir + "/model_param.opt.pt")
        updater_param.load_optimizer(tmp_dir + "/model_buf.opt.pt")


class _DemoException(Exception):
    pass


class _TestTorchSubModelRaisingException(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin = torch.nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B,T,D]
        :return: [B,T,D']
        """
        x = self.lin(x)
        if int("1") == 1:
            raise _DemoException("uh")
        return x


# must be in the global scope due to pickling
class TrainExceptionModel(torch.nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.sub = _TestTorchSubModelRaisingException(9, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B,T,D]
        :return: [B,T,D']
        """
        x = self.sub(x)
        return torch.nn.functional.log_softmax(x, dim=-1)

    @classmethod
    def train_step(cls, *, model: TrainExceptionModel, extern_data: TensorDict, **_kwargs):
        """train step"""
        data: Tensor = extern_data["data"]
        logits = model(data.raw_tensor)
        logits_packed = torch.nn.utils.rnn.pack_padded_sequence(
            logits, data.dims[1].dyn_size_ext.raw_tensor, batch_first=True, enforce_sorted=False
        )
        targets = extern_data["classes"]
        targets_packed = torch.nn.utils.rnn.pack_padded_sequence(
            targets.raw_tensor, data.dims[1].dyn_size_ext.raw_tensor, batch_first=True, enforce_sorted=False
        )
        loss = torch.nn.CrossEntropyLoss(reduction="none")(logits_packed.data, targets_packed.data.long())
        rf.get_run_ctx().mark_as_loss(name="ce", loss=loss)
        frame_error = torch.argmax(logits_packed.data, dim=-1).not_equal(targets_packed.data)
        rf.get_run_ctx().mark_as_loss(name="fer", loss=frame_error, as_error=True)


def test_torch_engine_train_exception():
    config = Config(
        dict(
            task="train",
            device="cpu",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainExceptionModel,
            train_step=TrainExceptionModel.train_step,
            batch_size=500,
            optimizer={"class": "adam"},
        )
    )
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "train"})
    dataset.init_seq_order(epoch=1)

    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        try:
            engine.train()
        except _DemoException as exc:
            print("got demo exception:", exc)
            exc_lines = str(exc).splitlines()
            assert "Module call stack:" in exc_lines and "(_TestTorchSubModelRaisingException.forward) sub" in exc_lines
        else:
            raise Exception("did not get expected exception")


# Bounded profile window: stops and exports after 4 steps.
_torch_profile_window = {
    "schedule": dict(wait=1, warmup=1, active=2, repeat=1),
    "profile_memory": False,  # the memory timeline HTML export needs matplotlib
    # torch 2.5 Python tracer: "Python replay stack is empty" internal assert when the window ends
    "with_stack": False,
}
_torch_profile_max_step = 4


@contextlib.contextmanager
def _cwd(path: str):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _torch_profile_train_config(tmp_dir: str, torch_profile: Dict[str, Any], **kwargs) -> Config:
    opts = dict(
        task="train",
        device="cpu",
        extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
        get_model=TrainTestModel,
        train_step=TrainTestModel.train_step,
        batch_size=500,
        max_seqs=4,  # 25 steps per epoch, i.e. many steps after the profile window
        optimizer={"class": "adam"},
        num_epochs=2,
        model=f"{tmp_dir}/model",
        learning_rate_file=f"{tmp_dir}/learning_rates",
        torch_profile=torch_profile,
    )
    opts.update(kwargs)
    return Config(opts)


def _torch_profile_train(config: Config) -> Engine:
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "train", "fixed_random_seed": 1})
    dataset.init_seq_order(epoch=1)
    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()
    return engine


def _check_torch_profile_trace(filename: str):
    with open(filename) as f:
        trace = json.load(f)
    assert trace["traceEvents"], f"{filename}: no trace events"


def _check_torch_profile_checkpoints(tmp_dir: str, *, num_epochs: int):
    for epoch in range(1, num_epochs + 1):
        model = torch.load(f"{tmp_dir}/model.{epoch:03d}.pt", map_location="cpu")
        assert model["epoch"] == epoch and model["model"], f"epoch {epoch}: invalid model checkpoint"
        opt = torch.load(f"{tmp_dir}/model.{epoch:03d}.opt.pt", map_location="cpu")
        assert opt["optimizer"]["state"], f"epoch {epoch}: invalid optimizer checkpoint"


def test_torch_engine_profile_exit_after_profile():
    # default: exit after the profile, and nothing is written
    from returnn.util.basic import should_write_to_disk

    with tempfile.TemporaryDirectory(prefix="returnn_test_torch_profile_exit") as tmp_dir, _cwd(tmp_dir):
        config = _torch_profile_train_config(tmp_dir, _torch_profile_window)
        assert not should_write_to_disk(config)
        try:
            _torch_profile_train(config)
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise Exception("did not exit after profiling")
        _check_torch_profile_trace(f"{tmp_dir}/torch_profile.json")
        assert sorted(os.listdir(tmp_dir)) == ["torch_profile.json"]


def test_torch_engine_profile_continue_training():
    # bounded profile in epoch 1, then further updates, epoch 2 without profiling (once),
    # and model, optimizer and scores are saved as in ordinary training
    import returnn.torch.engine as engine_module
    from returnn.util.basic import should_write_to_disk
    from returnn.learning_rate_control import load_learning_rate_control_from_config

    orig_opt_torch_profiler_from_opts = engine_module._opt_torch_profiler_from_opts
    profilers = []  # per epoch

    def _opt_torch_profiler_from_opts(**kwargs):
        prof = orig_opt_torch_profiler_from_opts(**kwargs)
        profilers.append(prof)
        return prof

    orig_export_chrome_trace = torch.profiler.profile.export_chrome_trace
    with contextlib.ExitStack() as stack:
        tmp_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix="returnn_test_torch_profile_continue"))
        stack.enter_context(_cwd(tmp_dir))
        stack.enter_context(
            unittest.mock.patch.object(engine_module, "_opt_torch_profiler_from_opts", _opt_torch_profiler_from_opts)
        )
        export_chrome_trace = stack.enter_context(
            unittest.mock.patch.object(
                torch.profiler.profile, "export_chrome_trace", autospec=True, side_effect=orig_export_chrome_trace
            )
        )
        config = _torch_profile_train_config(
            tmp_dir, {**_torch_profile_window, "exit_after_profile": False, "once": True}
        )
        engine = _torch_profile_train(config)

        assert len(profilers) == 2 and profilers[0] is not None and profilers[1] is None, profilers
        assert profilers[0].exported and not profilers[0].entered
        assert export_chrome_trace.call_count == 1, export_chrome_trace.call_args_list
        _check_torch_profile_trace(f"{tmp_dir}/torch_profile.json")
        epoch1_num_steps = engine.learning_rate_control.epoch_data[1].meta["epoch_num_train_steps"]
        assert epoch1_num_steps > _torch_profile_max_step + 1, "no updates after the profile window"
        assert engine.epoch == 2 and engine.learning_rate_control.epoch_data[2].meta["epoch_num_train_steps"] > 0

        _check_torch_profile_checkpoints(tmp_dir, num_epochs=2)
        scores = load_learning_rate_control_from_config(config).epoch_data
        for epoch in [1, 2]:
            assert "train_loss_ce" in scores[epoch].error, f"epoch {epoch}: scores not saved"
        assert should_write_to_disk(config)


class _FakeTorchDistributedContext:
    def __init__(self, *, rank: int, size: int, local_rank: int, local_size: int):
        self._rank, self._size, self._local_rank, self._local_size = rank, size, local_rank, local_size

    def rank(self) -> int:
        """global rank"""
        return self._rank

    def size(self) -> int:
        """global size"""
        return self._size

    def local_rank(self) -> int:
        """local rank"""
        return self._local_rank

    def local_size(self) -> int:
        """local size"""
        return self._local_size


def test_torch_profile_rank_selection():
    from returnn.torch.engine import _opt_torch_profiler_from_opts

    def _selected(*, num_nodes: int, local_size: int, **opts) -> Dict[int, Tuple[str, str]]:
        size = num_nodes * local_size
        res = {}
        for rank in range(size):
            ctx = _FakeTorchDistributedContext(
                rank=rank, size=size, local_rank=rank % local_size, local_size=local_size
            )
            prof = _opt_torch_profiler_from_opts({**_torch_profile_window, **opts}, ctx)
            if prof:
                res[rank] = (prof.trace_filename, prof.memory_filename)
        filenames = [fn for fns in res.values() for fn in fns]
        assert len(set(filenames)) == len(filenames), f"filename collision: {res}"
        return res

    def _names(rank: Optional[int]) -> Tuple[str, str]:
        suffix = f".rank{rank}" if rank is not None else ""
        return f"torch_profile{suffix}.json", f"torch_memory_profile{suffix}.html"

    # not distributed
    prof = _opt_torch_profiler_from_opts(_torch_profile_window, None)
    assert (prof.trace_filename, prof.memory_filename) == _names(None)
    # default: local rank 0 of each node, unchanged filenames on a single node
    assert _selected(num_nodes=1, local_size=4) == {0: _names(None)}
    assert _selected(num_nodes=2, local_size=4) == {0: _names(0), 4: _names(4)}
    # explicit global ranks
    assert _selected(num_nodes=2, local_size=4, ranks=[0]) == {0: _names(0)}
    assert _selected(num_nodes=2, local_size=4, ranks=[1, 5]) == {1: _names(1), 5: _names(5)}
    assert _selected(num_nodes=2, local_size=2, ranks="all") == {r: _names(r) for r in range(4)}
    # once
    assert _opt_torch_profiler_from_opts({**_torch_profile_window, "once": True}, None, exported_before=True) is None
    assert _opt_torch_profiler_from_opts(_torch_profile_window, None, exported_before=True) is not None
    try:
        _opt_torch_profiler_from_opts({**_torch_profile_window, "ranks": 0}, None)
    except TypeError as exc:
        print("got expected exception:", exc)
    else:
        raise Exception("did not get expected TypeError for invalid ranks")


def _torch_profile_distributed_worker(rank: int, world_size: int, port: int, tmp_dir: str, torch_profile: Dict):
    from returnn.util.basic import should_write_to_disk

    # each rank as its own single-GPU node, i.e. local rank 0 everywhere.
    # reduce_type param with sync_on_cpu only needs Gloo, so the ranks can share one GPU.
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK="0",
        LOCAL_WORLD_SIZE="1",
    )
    os.chdir(tmp_dir)
    config = _torch_profile_train_config(
        tmp_dir,
        torch_profile,
        backend="torch",
        device="cuda",
        torch_distributed={"reduce_type": "param", "param_sync_step": 1, "sync_on_cpu": True},
    )
    engine = _torch_profile_train(config)
    assert engine.epoch == 2
    with global_config_ctx(config):
        assert should_write_to_disk(config) == (rank == 0)  # only rank 0 writes checkpoints
    torch.distributed.destroy_process_group()


def _run_torch_profile_distributed(tmp_dir: str, torch_profile: Dict[str, Any], *, world_size: int = 2):
    import socket
    import torch.multiprocessing

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    torch.multiprocessing.spawn(
        _torch_profile_distributed_worker, args=(world_size, port, tmp_dir, torch_profile), nprocs=world_size
    )


def test_torch_engine_profile_continue_distributed():
    # multi-node-like setup: two ranks with local rank 0, which previously wrote the same trace file
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")  # torch_distributed in the engine requires CUDA
    for ranks, expected_ranks in [(None, [0, 1]), ([1], [1]), ("all", [0, 1])]:
        torch_profile = {**_torch_profile_window, "exit_after_profile": False, "once": True}
        if ranks is not None:
            torch_profile["ranks"] = ranks
        with tempfile.TemporaryDirectory(prefix="returnn_test_torch_profile_distributed") as tmp_dir:
            _run_torch_profile_distributed(tmp_dir, torch_profile)
            traces = sorted(fn for fn in os.listdir(tmp_dir) if fn.startswith("torch_profile"))
            assert traces == [f"torch_profile.rank{r}.json" for r in expected_ranks], (ranks, traces)
            for fn in traces:
                _check_torch_profile_trace(f"{tmp_dir}/{fn}")
            _check_torch_profile_checkpoints(tmp_dir, num_epochs=2)


def _torch_distributed_cpu_worker(rank: int, world_size: int, port: int, tmp_dir: str):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world_size),
    )
    config = Config(
        dict(
            task="train",
            device="cpu",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainTestModel,
            train_step=TrainTestModel.train_step,
            batch_size=500,
            optimizer={"class": "adam"},
            learning_rate=0.01,
            num_epochs=1,
            torch_dataloader_opts={"num_workers": 0},
            torch_distributed={"backend": "gloo"},
        )
    )
    with global_config_ctx(config):
        dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 20})
        dataset.init_seq_order(epoch=1)
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()
        param_sum = float(sum(param.detach().double().sum() for param in engine._pt_model.parameters()))
    with open(f"{tmp_dir}/rank{rank}.txt", "w") as f:
        f.write(repr(param_sum))
    torch.distributed.destroy_process_group()


def test_torch_engine_distributed_cpu():
    # DistributedDataParallel on the CPU with gloo. The averaged grads keep the params of both ranks identical.
    import socket
    import torch.multiprocessing

    world_size = 2
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="returnn_test_torch_distributed_cpu") as tmp_dir:
        torch.multiprocessing.spawn(_torch_distributed_cpu_worker, args=(world_size, port, tmp_dir), nprocs=world_size)
        param_sums = []
        for rank in range(world_size):
            with open(f"{tmp_dir}/rank{rank}.txt") as f:
                param_sums.append(float(f.read()))
    assert param_sums[0] == param_sums[1], param_sums


def test_dynamic_learning_rate():
    num_epochs = 3
    last_global_train_step: Optional[float] = None
    last_epoch_continuous: Optional[float] = None
    epoch_continuous_diffs = []

    def _dynamic_learning_rate(
        *, global_train_step: int, epoch: int, epoch_continuous: float, learning_rate: float, **_kwargs
    ) -> float:
        nonlocal last_global_train_step, last_epoch_continuous
        assert isinstance(global_train_step, int)
        assert isinstance(epoch, int)
        assert isinstance(epoch_continuous, (int, float))
        assert isinstance(learning_rate, (int, float))
        print(f"global_train_step: {global_train_step}, epoch: {epoch}, epoch_continuous: {epoch_continuous}")
        if last_global_train_step is None:
            assert global_train_step == 0 and epoch == 1
        else:
            # The call to this function could be repeated.
            assert global_train_step in (last_global_train_step, last_global_train_step + 1)
        if last_epoch_continuous is None:
            assert epoch_continuous == 0
        elif global_train_step == last_global_train_step:  # repeated call
            assert epoch_continuous == last_epoch_continuous
        else:
            assert epoch_continuous > last_epoch_continuous
            assert epoch >= epoch_continuous >= epoch - 1
            epoch_continuous_diffs.append(epoch_continuous - last_epoch_continuous)
        last_global_train_step = global_train_step
        last_epoch_continuous = epoch_continuous
        return learning_rate * epoch_continuous / num_epochs

    config = Config(
        dict(
            task="train",
            device="cpu",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainTestModel,
            train_step=TrainTestModel.train_step,
            batch_size=500,
            optimizer={"class": "adam"},
            dynamic_learning_rate=_dynamic_learning_rate,
            num_epochs=num_epochs,
        )
    )
    num_seqs_per_epoch = 100
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": num_seqs_per_epoch, "name": "train"})
    dataset.init_seq_order(epoch=1)

    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()

    assert last_epoch_continuous == num_epochs
    assert epoch_continuous_diffs
    print("epoch continuous diffs:", epoch_continuous_diffs)
    # Just some sanity check. The exact number here depends on num_seqs_per_epoch, batch_size, etc.
    eps = 0.001
    assert numpy.min(epoch_continuous_diffs) >= (0.01 - eps)
    assert numpy.max(epoch_continuous_diffs) <= 0.1
    # It's one more (non-repeated) call than num steps (first + very last),
    # and the diffs is one less, so the length should match final global train step.
    assert len(epoch_continuous_diffs) == engine.global_train_step


def test_torch_engine_train_lion_optimizer():
    config = Config(
        dict(
            task="train",
            device="cpu",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainTestModel,
            train_step=TrainTestModel.train_step,
            batch_size=500,
            optimizer={"class": "returnn.torch.optim.lion.Lion"},
            num_epochs=1,
        )
    )
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 10, "name": "train"})
    dataset.init_seq_order(epoch=1)

    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()


def test_torch_engine_bf16():
    config = Config(
        dict(
            task="train",
            device="cpu",
            default_float_dtype="bfloat16",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainTestModel,
            train_step=TrainTestModel.train_step,
            batch_size=500,
            optimizer={"class": "adam"},
            num_epochs=1,
        )
    )
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 10, "name": "train"})
    dataset.init_seq_order(epoch=1)

    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()
        params = list(engine.get_pt_model().parameters())
        assert params
        for p in params:
            assert p.dtype == torch.bfloat16


def test_torch_engine_train_shuffle_batches():
    config = Config(
        dict(
            task="train",
            device="cpu",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainTestModel,
            train_step=TrainTestModel.train_step,
            batch_size=100,
            optimizer={"class": "adam"},
            num_epochs=3,
            online_shuffle_batches=10,
        )
    )
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "train"})
    dataset.init_seq_order(epoch=1)

    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()


def test_torch_engine_sub_proc_cleanup():
    from multiprocessing import Process, Pipe
    import psutil
    import time

    parent_conn, child_conn = Pipe()

    # start in subproc so that we can modify the env
    p = Process(target=_torch_engine_sub_proc_cleanup_test_main, args=(child_conn,))
    p.start()

    # wait until the engine is initialized, so that we have the sub procs started
    msg = parent_conn.recv()
    assert msg == "initialized_engine"
    msg = parent_conn.recv()
    assert msg == "first_global_step"

    # Collect sub procs
    train_proc = psutil.Process(p.pid)
    child_procs = train_proc.children(recursive=True)
    print("train proc:", train_proc)
    for child_proc in child_procs:
        print("child proc:", child_proc, child_proc.cmdline())
    assert child_procs  # e.g. the multiproccessing manager + resource tracker?

    p.kill()
    p.join()

    counter = 0
    while True:
        # Check that all sub procs are also killed.
        any_alive = False
        for child_proc in child_procs:
            try:
                # is_running() is True for a ZOMBIE as well,
                # i.e. a proc which already exited and only waits to be reaped by its parent.
                # That counts as killed. (The parent is gone here, so init reaps them,
                # but on a loaded machine that can take a moment.)
                if not child_proc.is_running() or child_proc.status() == psutil.STATUS_ZOMBIE:
                    continue
                cmdline = child_proc.cmdline()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue  # exited in between
            print(f"Child proc still running: {child_proc} {cmdline}")
            any_alive = True
        counter += 1
        if any_alive:
            if counter > 100:
                raise Exception("Sub procs still alive")
            time.sleep(0.5)
            # repeat
        else:
            print("All sub procs are killed")
            break


def _torch_engine_sub_proc_cleanup_test_main(conn):
    try:
        import time

        config = Config(
            dict(
                task="train",
                device="cpu",
                extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
                get_model=TrainTestModel,
                train_step=TrainTestModel.train_step,
                batch_size=100,
                optimizer={"class": "adam"},
                num_epochs=100,
            )
        )
        dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 1000, "name": "train"})
        dataset.init_seq_order(epoch=1)

        with global_config_ctx(config):
            engine = Engine(config=config)
            engine.init_train_from_config(train_data=dataset)
            conn.send("initialized_engine")
            # to avoid pickling issues due to referencing __main__ here...
            config.typed_dict.pop("get_model")
            config.typed_dict.pop("train_step")
            epoch = 1
            while True:
                data_iter = iter(engine._train_dataloader)
                step = 0
                for _ in data_iter:
                    if step == 0 and epoch == 1:
                        conn.send("first_global_step")
                    step += 1
                print(f"Finished epoch {epoch} after {step} steps")
                assert step > 0
                time.sleep(0.1)

    except Exception as exc:
        conn.send(("exception", str(exc)))
        raise

    finally:
        conn.close()


# (epoch, seq len) of every dev seq evaluated in this process
_eval_on_all_ranks_seq_lens = []
_EVAL_ON_ALL_RANKS_DIST_OPTS = {"backend": "gloo", "reduce_type": "grad_explicit", "timeout_sec": 60}
_EVAL_ON_ALL_RANKS_DEV_SEQ_LENS = list(range(3, 26))


def _eval_on_all_ranks_train_step(*, model: TrainTestModel, extern_data: TensorDict, **kwargs):
    run_ctx = rf.get_run_ctx()
    if not run_ctx.train_flag:
        seq_lens = extern_data["data"].dims[1].dyn_size_ext.raw_tensor.tolist()
        _eval_on_all_ranks_seq_lens.extend((run_ctx.epoch, seq_len) for seq_len in seq_lens)
    TrainTestModel.train_step(model=model, extern_data=extern_data, **kwargs)


class _StaticDatasetWithoutPredefinedSeqOrder(StaticDataset):
    def supports_predefined_seq_order(self) -> bool:
        """:return: False, so that it cannot be split over the ranks"""
        return False


def _eval_on_all_ranks_run(
    *,
    torch_distributed: Optional[dict] = None,
    num_epochs: int = 2,
    learning_rate: float = 0.0,
    train_seq_ordering: str = "default",
    num_dev_seqs: int = len(_EVAL_ON_ALL_RANKS_DEV_SEQ_LENS),
    dev_seq_ordering: str = "default",
    dev_meta: bool = False,
    dev_predefined_seq_order: bool = True,
    calculate_exp_loss: bool = False,
    eval_before_train: bool = False,
    eval_again: bool = False,
) -> Dict[str, Any]:
    rnd = numpy.random.RandomState(42)
    # Every seq has its own length, so the lengths a rank saw tell which dev seqs it evaluated.
    seqs = [
        {"data": rnd.normal(size=(n, 9)).astype("float32"), "classes": rnd.randint(0, 2, size=(n,)).astype("int32")}
        for n in _EVAL_ON_ALL_RANKS_DEV_SEQ_LENS
    ]
    config = Config(
        dict(
            task="train",
            device="cpu",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainTestModel,
            train_step=_eval_on_all_ranks_train_step,
            batch_size=60,
            batch_size_train=20,
            optimizer={"class": "adam"},
            # 0 (the default here) keeps the initial model, so every run evaluates the same model
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            calculate_exp_loss=calculate_exp_loss,
            torch_dataloader_opts={"num_workers": 0},
            **({"torch_distributed": torch_distributed} if torch_distributed is not None else {}),
        )
    )
    dev_opts = {
        "class": "StaticDataset" if dev_predefined_seq_order else _StaticDatasetWithoutPredefinedSeqOrder,
        "data": seqs[:num_dev_seqs],
        "input_dim": 9,
        "output_dim": 2,
        "seq_ordering": dev_seq_ordering,
    }
    if dev_meta:
        dev_opts = {
            "class": "MetaDataset",
            "datasets": {"static": dev_opts},
            "data_map": {"data": ("static", "data"), "classes": ("static", "classes")},
            "seq_order_control_dataset": "static",
        }
    # The datasets are created under the config, as in a real run,
    # where each rank's datasets get their own random seed offset.
    with global_config_ctx(config):
        train_dataset = init_dataset(
            {
                "class": "StaticDataset",
                "data": seqs[:12],
                "input_dim": 9,
                "output_dim": 2,
                "seq_ordering": train_seq_ordering,
            }
        )
        train_dataset.init_seq_order(epoch=1)
        dev_dataset = init_dataset(dev_opts)
        dev_dataset.init_seq_order(epoch=1)
        engine = Engine(config=config)
        # init_seq_order calls on the dev dataset while the engine sets up
        dev_init_seq_order, dev_inits_at_startup = dev_dataset.init_seq_order, []

        def _dev_init_seq_order(*args, **kwargs):
            dev_inits_at_startup.append(kwargs)
            return dev_init_seq_order(*args, **kwargs)

        dev_dataset.init_seq_order = _dev_init_seq_order
        engine.init_train_from_config(train_data=train_dataset, dev_data=dev_dataset)
        del dev_dataset.init_seq_order
        # number of values this rank keeps aside for each split eval, see Engine._take_rank0_model_state
        own_model_state_sizes = []
        take_rank0_model_state = engine._take_rank0_model_state

        def _take_rank0_model_state():
            own = take_rank0_model_state()
            own_model_state_sizes.append(sum(value.numel() for value in own or []))
            return own

        engine._take_rank0_model_state = _take_rank0_model_state
        del _eval_on_all_ranks_seq_lens[:]
        if eval_before_train:
            # like the eval of a loaded checkpoint before the training goes on: this process ran no step yet
            engine.set_epoch(1)
            engine.eval_model()
        else:
            engine.train()
        evals = [list(_eval_on_all_ranks_seq_lens)]
        if eval_again:
            del _eval_on_all_ranks_seq_lens[:]
            engine.eval_model()
            evals.append(list(_eval_on_all_ranks_seq_lens))
        return {
            "evals": evals,
            "errors": {epoch: dict(data.error) for epoch, data in engine.learning_rate_control.epoch_data.items()},
            "param_sum": float(sum(param.detach().double().sum() for param in engine._pt_model.parameters())),
            "own_model_state_sizes": own_model_state_sizes,
            "dev_inits_at_startup": len(dev_inits_at_startup),
        }


def _eval_on_all_ranks_rank_main(rank: int, world_size: int, port: int, run_kwargs: Dict[str, Any], conn):
    import os

    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "LOCAL_WORLD_SIZE": str(world_size),
        }
    )
    try:
        conn.send(_eval_on_all_ranks_run(**run_kwargs))
    except Exception as exc:
        conn.send({"exception": repr(exc)})
        raise
    finally:
        conn.close()


def _eval_on_ranks(*, world_size: int = 2, **run_kwargs) -> List[Dict[str, Any]]:
    import multiprocessing
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = multiprocessing.get_context("spawn")
    conns, procs = [], []
    for rank in range(world_size):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(target=_eval_on_all_ranks_rank_main, args=(rank, world_size, port, run_kwargs, child_conn))
        proc.start()
        conns.append(parent_conn)
        procs.append(proc)
    results = []
    try:
        for rank, conn in enumerate(conns):
            assert conn.poll(300), f"rank {rank} did not report back"
            results.append(conn.recv())
    finally:
        for proc in procs:
            proc.join(timeout=60)
            if proc.is_alive():
                proc.kill()
    for rank, result in enumerate(results):
        assert "exception" not in result, f"rank {rank} failed: {result['exception']}"
    return results


def _eval_on_all_ranks_seq_lens_per_rank(
    results: List[Dict[str, Any]], *, epoch: int, eval_idx: int = 0
) -> List[List[int]]:
    return [sorted(seq_len for e, seq_len in result["evals"][eval_idx] if e == epoch) for result in results]


def _assert_eval_split(rank_seq_lens: List[List[int]], all_seq_lens: List[int]):
    assert all(rank_seq_lens), f"every rank must evaluate a share of the dev seqs, got {rank_seq_lens}"
    assert sorted(sum(rank_seq_lens, [])) == all_seq_lens, f"the shares must cover every dev seq once: {rank_seq_lens}"


def _assert_eval_scores(results: List[Dict[str, Any]], reference: Dict[str, Any], *, epochs, keys):
    for rank, result in enumerate(results):
        for epoch in epochs:
            for key in keys:
                numpy.testing.assert_allclose(
                    result["errors"][epoch][key],
                    reference["errors"][epoch][key],
                    rtol=1e-6,
                    err_msg=f"rank {rank} epoch {epoch} {key}",
                )


def test_torch_engine_eval_on_all_ranks():
    # With eval_on_all_ranks, the dev seqs are split over the ranks in every epoch,
    # also with a random seq order, which is seeded per rank (rank 0's order is split then),
    # also with DistributedDataParallel on CPU,
    # and every rank gets the score of one process over the whole dev set.
    # Without it (the default), or for a dataset without a predefined seq order, rank 0 evaluates them alone.
    reference = _eval_on_all_ranks_run()
    for opts, run_kwargs, split in [
        ({"eval_on_all_ranks": True}, {}, True),
        ({}, {}, False),
        ({"eval_on_all_ranks": True}, {"dev_seq_ordering": "random"}, True),
        ({"eval_on_all_ranks": True, "reduce_type": "grad"}, {}, True),
        ({"eval_on_all_ranks": True}, {"dev_predefined_seq_order": False}, False),
    ]:
        results = _eval_on_ranks(torch_distributed={**_EVAL_ON_ALL_RANKS_DIST_OPTS, **opts}, **run_kwargs)
        # The engine asks the dataset whether it can be split, without initializing it.
        assert [result["dev_inits_at_startup"] for result in results] == [0, 0]
        for epoch in (1, 2):
            rank_seq_lens = _eval_on_all_ranks_seq_lens_per_rank(results, epoch=epoch)
            print(f"{opts} {run_kwargs} epoch {epoch}: dev seq lens per rank:", rank_seq_lens)
            if split:
                _assert_eval_split(rank_seq_lens, _EVAL_ON_ALL_RANKS_DEV_SEQ_LENS)
            else:
                assert rank_seq_lens == [_EVAL_ON_ALL_RANKS_DEV_SEQ_LENS, []]
        _assert_eval_scores(results, reference, epochs=(1, 2), keys=("dev_loss_ce", "dev_loss_fer"))
        # Averaged grads keep the params identical, so rank 1 keeps none aside (and this model has no buffers).
        assert results[1]["own_model_state_sizes"] == ([0, 0] if split else [])


def test_torch_engine_eval_on_all_ranks_rank0_model():
    # With parameter averaging, the ranks train apart between the syncs.
    # A split eval must still score rank 0's model, like an eval on rank 0 alone,
    # and leave every rank's own training state as it was.
    opts = {**_EVAL_ON_ALL_RANKS_DIST_OPTS, "reduce_type": "param", "param_sync_step": 1000}
    run_kwargs = dict(learning_rate=0.01, train_seq_ordering="random")
    on_rank0 = _eval_on_ranks(torch_distributed={**opts, "eval_on_all_ranks": False}, **run_kwargs)
    split = _eval_on_ranks(torch_distributed={**opts, "eval_on_all_ranks": True}, **run_kwargs)
    assert split[0]["param_sum"] != split[1]["param_sum"], "the ranks must train apart for this test"
    _assert_eval_split(_eval_on_all_ranks_seq_lens_per_rank(split, epoch=1), _EVAL_ON_ALL_RANKS_DEV_SEQ_LENS)
    _assert_eval_scores(split, on_rank0[0], epochs=(1, 2), keys=("dev_loss_ce", "dev_loss_fer"))
    assert [result["param_sum"] for result in split] == [result["param_sum"] for result in on_rank0]


def test_torch_engine_eval_on_all_ranks_empty_share():
    # One dev seq on two ranks leaves rank 1 without any.
    # When the eval comes before any step of this process (e.g. the eval of a loaded checkpoint),
    # rank 1 must still take part and extend the scores (calculate_exp_loss) like rank 0.
    run_kwargs = dict(num_dev_seqs=1, calculate_exp_loss=True, eval_before_train=True)
    reference = _eval_on_all_ranks_run(**run_kwargs)
    results = _eval_on_ranks(
        torch_distributed={**_EVAL_ON_ALL_RANKS_DIST_OPTS, "eval_on_all_ranks": True}, **run_kwargs
    )
    assert _eval_on_all_ranks_seq_lens_per_rank(results, epoch=1) == [[3], []]
    _assert_eval_scores(results, reference, epochs=(1,), keys=("dev_loss_ce", "dev_loss_ce:exp", "dev_loss_fer"))


def test_torch_engine_eval_on_all_ranks_repeated_eval():
    # A second eval in the same epoch must split the same full seq order again.
    # MetaDataset keeps its current seq order when the epoch does not change
    # and it did not load beyond its first seq, as with one dev seq per rank here.
    results = _eval_on_ranks(
        torch_distributed={**_EVAL_ON_ALL_RANKS_DIST_OPTS, "eval_on_all_ranks": True},
        dev_meta=True,
        num_dev_seqs=2,
        eval_again=True,
    )
    for eval_idx in (0, 1):
        rank_seq_lens = _eval_on_all_ranks_seq_lens_per_rank(results, epoch=2, eval_idx=eval_idx)
        print(f"eval {eval_idx}: dev seq lens per rank:", rank_seq_lens)
        _assert_eval_split(rank_seq_lens, _EVAL_ON_ALL_RANKS_DEV_SEQ_LENS[:2])


def test_graph_capture_bounds_from_config():
    # the bounds a config already implies must not have to be repeated by hand:
    # max_seqs is the batch bound, max_seq_length the capacity, batch_size plus the per-seq
    # gap and align slack the packed total bound
    from returnn.tensor import batch_dim
    from returnn.torch.util.graph_capture import bounds_from_config

    time_dim = Dim(None, name="time-bounds")
    text_dim = Dim(None, name="text-bounds")
    labels_dim = Dim(None, name="labels-bounds")
    classes_dim = Dim(7, name="classes")
    template = TensorDict()
    template.update(
        {
            "data": {"dims": [batch_dim, time_dim], "dtype": "float32"},
            "text_codes": {"dims": [batch_dim, text_dim], "dtype": "int32", "sparse_dim": classes_dim},
            "labels": {"dims": [batch_dim, labels_dim], "dtype": "int32", "sparse_dim": classes_dim},
        },
        auto_convert=True,
    )
    config = Config(
        dict(
            max_seqs=48,
            max_seq_length={"data": 680000, "labels": 720},
            batch_size=4_000_000,
            packed_tensors={
                "gap": 8640,
                "align": 960,
                "per_key": {"text_codes": {"packed": False}, "labels": {"packed": False}},
            },
        )
    )
    with global_config_ctx(config):
        opts = bounds_from_config(
            {"warmup_steps": 2, "dim_capacity": {"text_codes": 720}}, config=config, extern_data_template=template
        )
    assert opts["batch_size_bound"] == 48
    assert opts["dim_capacity"] == {"data": 680000, "text_codes": 720, "labels": 720}
    assert opts["packed_total_bound"] == {"data": 4_000_000 + 48 * (8640 + 959)}
    assert opts["warmup_steps"] == 2
    with global_config_ctx(config):
        explicit = bounds_from_config(
            {
                "batch_size_bound": 32,
                "dim_capacity": {"data": 1, "text_codes": 1, "labels": 1},
                "packed_total_bound": {"data": 5},
            },
            config=config,
            extern_data_template=template,
        )
    assert explicit["batch_size_bound"] == 32
    assert explicit["dim_capacity"] == {"data": 1, "text_codes": 1, "labels": 1}
    assert explicit["packed_total_bound"] == {"data": 5}
    config = Config(dict(max_seqs=4, batch_size=1000, packed_batch_size={"data": 100}, packed_tensors=True))
    with global_config_ctx(config):
        opts = bounds_from_config({}, config=config, extern_data_template=template)
    assert opts["packed_total_bound"] == {"data": 100, "text_codes": 1000, "labels": 1000}


def test_graph_capture_boundary_cuts_to_the_longest_sequence():
    """
    The captured first segment of a segmented step returns buffers padded to the capacity,
    and the eager segment must see them as an eager step would: cut along every dynamic dim
    to the batch's longest sequence, a dim returned next to a tensor being that tensor's dim,
    sparse and feature dims and plain values kept.
    """
    from returnn.torch.util.graph_capture import _flatten_boundary, _unflatten_boundary

    batch = Dim(3, name="batch-boundary")
    lens = torch.tensor([4, 2, 0], dtype=torch.int32)
    time = Dim(Tensor("time:size", dims=[batch], dtype="int32", raw_tensor=lens), name="time-boundary")
    feat = Dim(2, name="feat-boundary")
    vocab = Dim(5, name="vocab-boundary")
    x = Tensor("x", dims=[batch, time, feat], dtype="float32", feature_dim=feat, raw_tensor=torch.randn(3, 7, 2))
    labels = Tensor(
        "labels",
        dims=[batch, time],
        dtype="int32",
        sparse_dim=vocab,
        raw_tensor=torch.randint(0, 5, (3, 7), dtype=torch.int32),
    )
    spec, tensors = _flatten_boundary({"x": x, "time": time, "labels": labels, "scale": 0.5}, batch_dim=batch)
    values = _unflatten_boundary(spec, tensors, batch_dim=batch)
    assert values["x"].dims[1] is values["time"] and values["labels"].dims[1] is values["time"]
    assert values["x"].feature_dim == feat and values["labels"].sparse_dim == vocab and values["scale"] == 0.5
    assert values["time"].dyn_size_ext.raw_tensor.tolist() == [4, 2, 0]
    assert torch.equal(values["x"].raw_tensor, x.raw_tensor[:, :4])
    assert torch.equal(values["labels"].raw_tensor, labels.raw_tensor[:, :4])


def test_graph_capture_optimizer_state_materialization():
    """the lazy optimizer state is only created for the captured optimizer, and only where zero-init is right"""
    from returnn.torch.util.graph_capture import GraphCapturedTrainStep, _optimizer_state_zero_init

    def _stub(opt, *, captured: bool):
        obj = GraphCapturedTrainStep.__new__(GraphCapturedTrainStep)
        obj._get_optimizer = lambda: opt
        obj._grad_params = list(opt.param_groups[0]["params"])
        obj._post_step = (lambda: None) if captured else None
        return obj

    p = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.Rprop([p], lr=0.1)
    assert not _optimizer_state_zero_init(opt)
    assert _optimizer_state_zero_init(torch.optim.AdamW([p], lr=0.1))
    try:
        _stub(opt, captured=True)._materialize_optimizer_state()
    except NotImplementedError:
        pass
    else:
        raise AssertionError("a captured Rprop must be rejected")
    _stub(opt, captured=False)._materialize_optimizer_state()
    assert not opt.state
    for _ in range(3):
        p.grad = torch.tensor(1.0)
        opt.step()
    torch.testing.assert_close(p.detach(), torch.tensor(0.636))
    q = torch.nn.Parameter(torch.tensor(1.0))
    adamw = torch.optim.AdamW([q], lr=0.1)
    _stub(adamw, captured=True)._materialize_optimizer_state()
    assert adamw.state and all(float(v) == 0 for v in adamw.state[q].values() if v.dim() == 0)


def _build_cuda_graph_train_config_and_dataset(
    *,
    compile_: bool,
    torch_model: bool = False,
    warmup_steps: Optional[int] = 2,
    cuda_graph: bool = True,
    optimizer_step: bool = False,
):
    """
    small model + Task12AXDataset config with torch_cuda_graph, see the tests below.
    With optimizer_step: the optimizer step not in the model graph but separately (torch_optimizer_step).

    :param compile_: the step goes through Inductor before the capture
    :param torch_model: the model is a torch module around an RF part, with parameters torch owns itself
        (one of them tied, read from two modules, and a conv weight whose grad cuDNN computes channels-last),
        instead of an RF module
    :param warmup_steps: eager steps before the capture, None for the default
    :param cuda_graph: whether the model step is captured at all
    :param optimizer_step: whether the optimizer step is compiled and captured separately
    """
    from returnn.datasets import init_dataset
    from returnn.tensor import Dim, batch_dim
    from returnn.torch.frontend.bridge import rf_module_to_pt_module

    # fresh dims per test: capacities get set on them
    time_dim = Dim(None, name=f"time-cudagraph-{compile_}-{torch_model}-{warmup_steps}-{cuda_graph}-{optimizer_step}")
    feat_dim = Dim(9, name="feat")
    classes_dim = Dim(2, name="classes")
    hidden = Dim(64, name="hidden")

    class _Model(rf.Module):
        def __init__(self):
            super().__init__()
            self.out_dim = classes_dim
            self.layer = rf.Linear(feat_dim, hidden)
            self.out = rf.Linear(hidden, classes_dim)
            self.steps_seen = rf.Parameter([], dtype="int64", auxiliary=True)
            self.steps_seen.initial = 0

    class _TorchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.out_dim = classes_dim
            self.layer = rf_module_to_pt_module(rf.Linear(feat_dim, hidden))
            self.mid = torch.nn.Linear(hidden.dimension, hidden.dimension, bias=False)
            self.mid_tied = torch.nn.Linear(hidden.dimension, hidden.dimension, bias=False)
            self.mid_tied.weight = self.mid.weight
            self.out = torch.nn.Linear(hidden.dimension, classes_dim.dimension)
            self.conv = torch.nn.Conv2d(3, 4, 3, padding=1, bias=False)
            self.initial = {name: p.detach().clone() for name, p in self.named_parameters()}

        def logits(self, data: Tensor) -> Tensor:
            """the RF part through its RF module, the torch parameters through the attributes of their modules"""
            x = rf.relu(self.layer.rf_module(data))
            # (batch, 3, time, 3) with channels-last strides: cuDNN then computes the weight grad channels-last
            conv_in = data.raw_tensor.unflatten(-1, (3, 3)).permute(0, 3, 1, 2)
            conv_out = torch.nn.functional.conv2d(conv_in, self.conv.weight, padding=1).mean(dim=(1, 3))
            x = x + Tensor("conv", dims=data.dims[:2], dtype="float32", raw_tensor=conv_out)
            for linear in (self.mid, self.mid_tied):
                weight = Tensor("weight", dims=[hidden.copy(match_priority=1), hidden], dtype="float32")
                weight.raw_tensor = linear.weight
                x = rf.relu(rf.matmul(x, weight, reduce=hidden))
                x, _ = rf.replace_dim(x, in_dim=weight.dims[0], out_dim=hidden)
            weight = Tensor("weight", dims=[classes_dim, hidden], dtype="float32", raw_tensor=self.out.weight)
            bias = Tensor("bias", dims=[classes_dim], dtype="float32", raw_tensor=self.out.bias)
            return rf.matmul(x, weight, reduce=hidden) + bias

    def _get_model(*, epoch, step, **_kwargs):
        return _TorchModel() if torch_model else _Model()

    def _train_step(*, model, extern_data: TensorDict, **_kwargs):
        data = extern_data["data"]
        classes = extern_data["classes"]
        if not torch_model and rf.get_run_ctx().train_flag:  # the eval between the epochs is no train step
            model.steps_seen.assign_add(1)
        if torch_model:
            logits = model.logits(data)
        else:
            x = rf.relu(model.layer(data))
            logits = model.out(x)
        loss = rf.cross_entropy(target=classes, estimated=logits, estimated_type="logits", axis=model.out_dim)
        loss.mark_as_loss("ce")
        # an error measure is not part of the total loss, its reduction must be recorded in the step as well
        frame_err = rf.cast(rf.reduce_argmax(logits, axis=model.out_dim) != classes, "float32")
        frame_err.mark_as_loss("fer", as_error=True)
        # a per-sequence measure of exactly one, its reported mean must not depend on the batch bound
        seq_one = rf.reduce_sum(loss, axis=time_dim) * 0.0 + 1.0
        seq_one.mark_as_loss("seq_one", as_error=True)

    def _dyn_lr(*, global_train_step: int, learning_rate: float, **_kwargs) -> float:
        # per-step LR schedule: under capture_optimizer this exercises the device-tensor LR input
        return learning_rate * (1.0 + 0.1 * global_train_step)

    config = Config(
        dict(
            task="train",
            device="gpu",
            extern_data={
                "data": {"dims": [batch_dim, time_dim, feat_dim], "dtype": "float32"},
                # targets are declared like this in real configs, the eager warmup step must keep them
                "classes": {
                    "dims": [batch_dim, time_dim],
                    "dtype": "int32",
                    "sparse_dim": classes_dim,
                    "available_for_inference": False,
                },
            },
            get_model=_get_model,
            train_step=_train_step,
            batch_size=400,
            max_seqs=10,
            num_epochs=2,
            learning_rate=1e-3,
            dynamic_learning_rate=_dyn_lr,
            # covers the pre-clip grad-norm recording in updater.step,
            # which under capture_optimizer runs in-graph (a static tensor updated per replay)
            log_grad_norm=True,
            gradient_clip_global_norm=5.0,
            optimizer={"class": "adamw", "capturable": True},
            torch_dataloader_opts={"num_workers": 0},
        )
    )
    if cuda_graph:
        config.typed_dict["torch_cuda_graph"] = dict(
            batch_size_bound=10,
            dim_capacity={"data": 100, "classes": 100},
            capture_optimizer=not optimizer_step,
            **({"warmup_steps": warmup_steps} if warmup_steps is not None else {}),
            **({"compile": True} if compile_ else {}),
        )
    if optimizer_step:
        config.typed_dict["torch_optimizer_step"] = {}
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "train", "fixed_random_seed": 1})
    dataset.init_seq_order(epoch=1)
    # the eval at the epoch end rebinds the dyn sizes the train losses are normalized by
    dev_dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 5, "name": "dev", "fixed_random_seed": 3})
    dev_dataset.init_seq_order(epoch=1)
    return config, dataset, dev_dataset


def _run_cuda_graph_train(
    *,
    compile_: bool,
    torch_model: bool = False,
    warmup_steps: Optional[int] = 2,
    cuda_graph: bool = True,
    optimizer_step: bool = False,
) -> Engine:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    import re
    from unittest import mock
    from returnn.log import log as returnn_log
    import returnn.torch.engine as torch_engine_module

    config, dataset, dev_dataset = _build_cuda_graph_train_config_and_dataset(
        compile_=compile_,
        torch_model=torch_model,
        warmup_steps=warmup_steps,
        cuda_graph=cuda_graph,
        optimizer_step=optimizer_step,
    )
    log_file = tempfile.NamedTemporaryFile(mode="wt", suffix=f"-cudagraph-{compile_}.log", delete=False)
    log_file.close()
    returnn_log.initialize(logs=[log_file.name], verbosity=[5])
    try:
        with global_config_ctx(config):
            engine = Engine(config=config)
            engine.init_train_from_config(train_data=dataset, dev_data=dev_dataset)
            if cuda_graph:
                order = []
                run_train_step = engine._graph_capture.run_train_step
                print_process = torch_engine_module._print_process

                def _launch(*args, **kwargs):
                    order.append((engine.epoch, "l"))
                    return run_train_step(*args, **kwargs)

                def _report(report_prefix, **kwargs):
                    if report_prefix.endswith(" train"):
                        order.append((engine.epoch, "r"))
                    return print_process(report_prefix, **kwargs)

                engine._graph_capture.run_train_step = _launch
                with mock.patch.object(torch_engine_module, "_print_process", _report):
                    engine.train()
                # the host reports every step only after it launched the next one, so the GPU never waits for it
                for epoch in sorted({e for e, _ in order}):
                    kinds = "".join(k for e, k in order if e == epoch)
                    n = kinds.count("l")
                    assert n > 2 and kinds == "l" + "lr" * (n - 1) + "r", (epoch, kinds)
                assert engine._graph_capture is not None
                assert engine._graph_capture._graph is not None, "graph never captured"
                # the loss denominators must follow the static length buffers, not the dyn sizes
                # the eval left behind (it rebinds the very size tensors they are reduced from)
                n_frames = int(engine._graph_capture._lens_bufs["data"].sum())
                assert n_frames > 0
                for name, loss in engine._graph_capture._ctx.losses.items():
                    assert int(loss.get_inv_norm_factor().raw_tensor) == n_frames, name
                assert engine._graph_capture.captures_optimizer == (not optimizer_step)
                # the cached reduction of the error measure must be a graph output, refreshed by every replay,
                # not an eager reduction from the first readout, which every later replay would repeat
                fer = engine._graph_capture._ctx.losses["fer"]
                fresh = rf.reduce_sum(fer.loss, axis=fer.loss.dims) if fer.loss.dims else fer.loss
                torch.testing.assert_close(fer.get_summed_loss().raw_tensor, fresh.raw_tensor)
            else:
                engine.train()
            if optimizer_step:
                assert engine._updater._optimizer_step._graph is not None, "optimizer step never captured"
            for param_group in engine._updater.optimizer.param_groups:
                lr = param_group["lr"]  # device-tensor LR input of the captured optimizer
                assert isinstance(lr, torch.Tensor) and lr.is_cuda
                assert lr.item() > 1e-3  # the per-step schedule advanced it
            for name, p in engine._pt_model.named_parameters():
                assert torch.isfinite(p).all(), f"non-finite param {name}"
                # the foreach and fused optimizer kernels need every grad in the layout of its parameter
                assert p.grad is None or p.grad.stride() == p.stride(), (name, p.grad.stride(), p.stride())
                if torch_model:
                    assert not torch.equal(p.detach().cpu(), engine._pt_model.initial[name]), f"{name} never trained"
            if not torch_model:
                # every train step counts once: the runs which only warm the kernels
                # or trace and compile the step must leave the training state alone
                seen = int(engine._orig_model.steps_seen.raw_tensor)
                assert seen == engine.global_train_step, (seen, engine.global_train_step)
            if cuda_graph:
                # the run end must destroy the graph, else a NCCL comm it captured never shuts down
                engine.finalize()
                assert engine._graph_capture is None
    finally:
        returnn_log.initialize()
    with open(log_file.name, "rt", encoding="utf-8") as f:
        txt = f.read()
    os.remove(log_file.name)
    # the error measure is not part of the total loss, so its reduction must still be recorded
    # in the graph: a value reduced once at the first readout would repeat on every replay
    fer = [float(v) for _, v in re.findall(r"train, step (\d+), ce [0-9.]+, fer ([0-9.]+)", txt)]
    assert len(fer) >= 10, f"only {len(fer)} steps with the error measure parsed from the log"
    warmup = (config.typed_value("torch_cuda_graph") or {}).get("warmup_steps", 0)
    assert len(set(fer[warmup + 1 :])) > 1, f"the error measure is frozen under replay: {fer}"
    return engine


def _cuda_graph_packed_decoder_setup(mode: str):
    """config+dataset for the packed-decoder capture parity test, see below"""
    from returnn.datasets import init_dataset
    from returnn.tensor import Dim, batch_dim
    import numpy

    time_dim = Dim(None, name=f"time-pdec-{mode}")
    tgt_time_dim = Dim(None, name=f"tgt-time-pdec-{mode}")
    feat_dim = Dim(8, name="feat")
    vocab_dim = Dim(11, name="vocab")
    wb_vocab_dim = Dim(12, name="vocab_wb")

    class _Model(rf.Module):
        def __init__(self):
            super().__init__()
            from returnn.frontend.encoder.conformer import (
                ConformerEncoder,
                ConformerEncoderLayer,
                ConformerConvSubsample,
                ConformerPositionwiseFeedForward,
            )
            from returnn.frontend.decoder.transformer import TransformerDecoder, FeedForwardGated

            enc_dim = Dim(32, name="enc")
            self.encoder = ConformerEncoder(
                feat_dim,
                enc_dim,
                ff_dim=Dim(24, name="enc-ff"),
                # the strided subsample frontend, like the real training model:
                # covers the strided-conv packed relayout (incl. auto-realign) under capture
                input_layer=ConformerConvSubsample(
                    feat_dim,
                    out_dims=[Dim(4, name="conv1"), Dim(4, name="conv2")],
                    filter_sizes=[(3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1)],
                ),
                input_dropout=0.0,
                dropout=0.0,
                encoder_layer=rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square)),
                    num_heads=2,
                    conv_kernel_size=8,
                    dropout=0.0,
                    att_dropout=0.0,
                    conv_norm_opts={"use_mask": True},
                ),
                num_layers=2,
            )
            self.decoder = TransformerDecoder(
                enc_dim,
                vocab_dim,
                Dim(32, name="dec"),
                num_layers=2,
                num_heads=2,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                dropout=0.0,
                att_dropout=0.0,
            )
            self.aux_logits = rf.Linear(enc_dim, wb_vocab_dim)

    def _get_model(*, epoch, step, **_kwargs):
        return _Model()

    def _train_step(*, model: _Model, extern_data: TensorDict, **_kwargs):
        data = extern_data["data"]
        targets = extern_data["classes"]
        enc, enc_sp = model.encoder(data, in_spatial_dim=time_dim)
        log_probs = rf.log_softmax(model.aux_logits(enc), axis=model.aux_logits.out_dim)
        ctc = rf.ctc_loss(
            logits=log_probs,
            logits_normalized=True,
            targets=targets,
            input_spatial_dim=enc_sp,
            targets_spatial_dim=tgt_time_dim,
            blank_index=11,
        )
        # constant inv norm: under bound shapes the default norm counts the FILLER seqs
        # (static batch dim = the bound), eager counts real seqs -- the reported values
        # then differ by that ratio while the sums are identical.
        # Raw sums are what this parity test must compare.
        one = rf.constant(1.0, dims=[])
        ctc.mark_as_loss("ctc", custom_inv_norm_factor=one)
        enc_state = model.decoder.transform_encoder(enc, axis=enc_sp)
        logits, _ = model.decoder(
            targets,
            spatial_dim=tgt_time_dim,
            state=model.decoder.default_initial_state(batch_dims=[batch_dim]),
            encoder=enc_state,
        )
        ce = rf.cross_entropy(estimated=logits, target=targets, axis=vocab_dim, estimated_type="logits")
        ce.mark_as_loss("ce", custom_inv_norm_factor=one)

    # varying lens AND varying seq counts per batch (frame-budget batching):
    # under capture this varies the filler-seq count and every packed extent per replay
    rnd = numpy.random.RandomState(7)
    seqs = []
    for _i in range(36):
        t = int(rnd.randint(15, 99))
        s = 2 + t // 12
        seqs.append(
            {
                "data": rnd.randn(t, 8).astype("float32"),
                "classes": rnd.randint(0, 11, (s,)).astype("int32"),
            }
        )
    cfg = dict(
        task="train",
        device="gpu",
        random_seed=42,
        extern_data={
            "data": {"dims": [batch_dim, time_dim, feat_dim], "dtype": "float32"},
            "classes": {"dims": [batch_dim, tgt_time_dim], "dtype": "int32", "sparse_dim": vocab_dim},
        },
        get_model=_get_model,
        train_step=_train_step,
        batch_size=600,
        max_seqs=10,
        num_epochs=1,
        learning_rate=0.0,  # frozen params: each step's losses depend only on its batch
        optimizer={"class": "adamw", "capturable": True},
        torch_dataloader_opts={"num_workers": 0},
        # as the real trainings: the packed decoder self-att fast path is flash varlen,
        # which exists for cuda bf16/fp16 only
        torch_amp="bfloat16",
        grad_scaler=None,  # bf16 needs no scaler; capture forbids one
    )
    # both modes use packed batching, so the per-step batches are IDENTICAL;
    # a padded mode would batch by padded frames and see different batches,
    # making per-step comparison meaningless.
    # packed_eager is equivalent to padded by the rf-level full-model tests,
    # so this test isolates exactly the capture layer.
    cfg["packed_tensors"] = {"per_key": {"data": {"gap": 8, "align": 1}, "classes": {"gap": 2, "align": 1}}}
    if mode == "packed_graphc":
        cfg["torch_cuda_graph"] = dict(
            batch_size_bound=10,
            dim_capacity={"data": 100, "classes": 16},
            packed_total_bound={"data": 600 + 10 * 9, "classes": 10 * 18},
            warmup_steps=2,
            capture_optimizer=True,
            compile=True,
        )
    config = Config(cfg)
    dataset = init_dataset({"class": "StaticDataset", "data": seqs, "input_dim": 8, "output_dim": 11})
    dataset.init_seq_order(epoch=1)
    return config, dataset


def _cuda_graph_packed_decoder_run(mode: str):
    """run one epoch in-process; per-step losses parsed from the RETURNN log file"""
    import os
    import re
    import tempfile
    from returnn.log import log as returnn_log

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config, dataset = _cuda_graph_packed_decoder_setup(mode)
    log_file = tempfile.NamedTemporaryFile(mode="wt", suffix=f"-{mode}.log", delete=False)
    log_file.close()
    # the engine logs the per-step losses via the RETURNN log module (bound at init):
    # direct it to a file for this run, so the losses are parseable in-process
    # (stdout stays attached too, see Log.initialize)
    returnn_log.initialize(logs=[log_file.name], verbosity=[5])
    try:
        with global_config_ctx(config):
            engine = Engine(config=config)
            engine.init_train_from_config(train_data=dataset)
            engine.train()
    finally:
        returnn_log.initialize()  # back to the default (stdout only)
    with open(log_file.name, "rt", encoding="utf-8") as f:
        txt = f.read()
    os.remove(log_file.name)
    steps = re.findall(r"train, step (\d+), ctc ([0-9.]+), ce ([0-9.]+)", txt)
    assert len(steps) >= 5, f"{mode}: only {len(steps)} steps parsed from the log"
    return {int(s): (float(a), float(b)) for s, a, b in steps}


def test_torch_engine_cuda_graph_packed_decoder_parity():
    """
    Full model INCL cross-attention decoder + aux CTC under whole-step capture,
    with per-step varying seq lens AND seq counts (so the filler count and every
    packed extent change per replay):
    the per-step losses must match the padded eager run on identical batches.

    The rf-level full-model tests cover eager and the static-traceable path;
    capture semantics (host code runs ONCE, persistent buffers, replays reuse
    baked host scalars) exist only here.
    A host-derived per-batch scalar frozen at capture time shows up ONLY in this test.

    Both modes run in-process, sequentially; the same config random_seed gives
    identical model init and batches, so the per-step losses are comparable.
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    losses = {mode: _cuda_graph_packed_decoder_run(mode) for mode in ["packed_eager", "packed_graphc"]}
    common = sorted(set(losses["packed_eager"]) & set(losses["packed_graphc"]))
    assert len(common) >= 5
    for s in common:
        (ctc_a, ce_a), (ctc_b, ce_b) = losses["packed_eager"][s], losses["packed_graphc"][s]
        assert abs(ctc_a - ctc_b) / max(abs(ctc_a), 1e-6) < 2e-2, f"step {s} ctc: {ctc_a} vs {ctc_b}"
        assert abs(ce_a - ce_b) / max(abs(ce_a), 1e-6) < 2e-2, f"step {s} ce: {ce_a} vs {ce_b}"


def _cuda_graph_segmented_setup(mode: str, *, compile_: bool):
    """config+dataset for the segmented capture parity test, see below"""
    from returnn.datasets import init_dataset
    from returnn.tensor import Dim, batch_dim
    import numpy

    time_dim = Dim(None, name=f"time-segmented-{mode}-{compile_}")
    feat_dim = Dim(8, name="feat")
    classes_dim = Dim(5, name="classes")
    hidden = Dim(16, name="hidden")

    class _Model(rf.Module):
        def __init__(self):
            super().__init__()
            self.layer = rf.Linear(feat_dim, hidden)
            self.out = rf.Linear(hidden, classes_dim)

    def _get_model(*, epoch, step, **_kwargs):
        return _Model()

    def _encode(*, model: _Model, extern_data: TensorDict, **_kwargs):
        """the first segment, static shapes: the hidden layer and a loss on it"""
        x = rf.relu(model.layer(extern_data["data"]))
        rf.reduce_mean(x * x, axis=hidden).mark_as_loss("aux", scale=0.1)
        return {"x": x, "classes": extern_data["classes"], "time": time_dim}

    def _classify(*, model: _Model, x, classes, time, **_kwargs):
        """the second segment, eager: the real frames selected at their exact count, then the output layer"""
        mask = rf.sequence_mask([batch_dim, time], device=x.device)
        flat_x, flat_dim = rf.masked_select(x, mask=mask, dims=[batch_dim, time])
        flat_classes, _ = rf.masked_select(classes, mask=mask, dims=[batch_dim, time], out_dim=flat_dim)
        logits = model.out(flat_x)
        ce = rf.cross_entropy(estimated=logits, target=flat_classes, axis=classes_dim, estimated_type="logits")
        ce.mark_as_loss("ce")
        fer = rf.cast(rf.reduce_argmax(logits, axis=classes_dim) != flat_classes, "float32")
        fer.mark_as_loss("fer", as_error=True)

    def _train_step(*, model: _Model, extern_data: TensorDict, **_kwargs):
        _classify(model=model, **_encode(model=model, extern_data=extern_data))

    _train_step.graph_segments = (_encode, _classify)

    rnd = numpy.random.RandomState(7)
    seqs = []
    for _i in range(80):
        t = int(rnd.randint(15, 99))
        seqs.append({"data": rnd.randn(t, 8).astype("float32"), "classes": rnd.randint(0, 5, (t,)).astype("int32")})
    cfg = dict(
        task="train",
        device="gpu",
        random_seed=42,
        extern_data={
            "data": {"dims": [batch_dim, time_dim, feat_dim], "dtype": "float32"},
            "classes": {"dims": [batch_dim, time_dim], "dtype": "int32", "sparse_dim": classes_dim},
        },
        get_model=_get_model,
        train_step=_train_step,
        batch_size=600,
        max_seqs=10,
        num_epochs=1,
        # the losses of every step depend on the gradients of all steps before it
        learning_rate=0.5,
        optimizer={"class": "sgd"},
        torch_dataloader_opts={"num_workers": 0},
    )
    if mode == "segmented":
        cfg["torch_cuda_graph"] = dict(
            batch_size_bound=10, dim_capacity={"data": 100, "classes": 100}, warmup_steps=2, segmented=True
        )
        if compile_:
            cfg["torch_cuda_graph"]["compile"] = True
    config = Config(cfg)
    dataset = init_dataset({"class": "StaticDataset", "data": seqs, "input_dim": 8, "output_dim": 5})
    dataset.init_seq_order(epoch=1)
    return config, dataset


def _cuda_graph_segmented_run(mode: str, *, compile_: bool):
    """one epoch in-process, the per-step losses parsed from the log"""
    import os
    import re
    import tempfile
    from returnn.log import log as returnn_log

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config, dataset = _cuda_graph_segmented_setup(mode, compile_=compile_)
    log_file = tempfile.NamedTemporaryFile(mode="wt", suffix=f"-{mode}.log", delete=False)
    log_file.close()
    returnn_log.initialize(logs=[log_file.name], verbosity=[5])
    try:
        with global_config_ctx(config):
            engine = Engine(config=config)
            engine.init_train_from_config(train_data=dataset)
            engine.train()
            if mode == "segmented":
                assert engine._graph_capture._graph is not None and engine._graph_capture._graph_bw is not None
            engine.finalize()
    finally:
        returnn_log.initialize()
    with open(log_file.name, "rt", encoding="utf-8") as f:
        txt = f.read()
    os.remove(log_file.name)
    number = r"([-+0-9.e]+)"
    steps = re.findall(rf"train, step (\d+), aux {number}, ce {number}, fer {number}", txt)
    assert len(steps) >= 8, (mode, compile_, re.findall(r"train, step .*", txt)[:2])
    return {int(s): (float(a), float(c), float(e)) for s, a, c, e in steps}


def test_torch_engine_cuda_graph_segmented_parity():
    """
    Opts "segmented": the first segment replays as a forward and a backward graph, the second runs eagerly
    in between at its own shapes (here a masked_select at the exact frame count), plain and compiled.
    The per-step losses must follow the eager run on identical batches. The model trains, so from the
    first captured step on they only do if every gradient does, the first segment's own and the one
    it gets through the boundary.
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    eager = _cuda_graph_segmented_run("eager", compile_=False)
    for compile_ in (False, True):
        segmented = _cuda_graph_segmented_run("segmented", compile_=compile_)
        assert set(segmented) == set(eager)
        for s in sorted(eager):
            for name, a, b in zip(("aux", "ce", "fer"), eager[s], segmented[s]):
                assert abs(a - b) <= 2e-3 * max(1.0, abs(a)), f"compile {compile_} step {s} {name}: {a} vs {b}"


def test_graph_capture_optimizer_state_snapshot_roundtrip():
    """a resumed optimizer state survives the kernel warmup runs through the snapshot, the lr object stays"""
    from returnn.torch.util.graph_capture import _snapshot_optimizer_state, _restore_optimizer_state

    torch.manual_seed(3)
    model = torch.nn.Linear(4, 3)
    opt = torch.optim.Adam(model.parameters(), lr=torch.tensor(0.1))
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    before = {k: v.clone() for k, v in opt.state[model.weight].items()}
    lr = opt.param_groups[0]["lr"]
    snapshot = _snapshot_optimizer_state(opt)
    opt.step()
    assert not torch.equal(opt.state[model.weight]["exp_avg"], before["exp_avg"])
    _restore_optimizer_state(opt, snapshot)
    for k, v in before.items():
        assert torch.equal(opt.state[model.weight][k], v), k
    assert opt.param_groups[0]["lr"] is lr


def test_torch_engine_cuda_graph_train():
    """whole-train-step CUDA-graph capture/replay (torch_cuda_graph), 2 epochs across an epoch boundary,
    in-graph optimizer + per-step LR schedule via the device-tensor LR input"""
    _run_cuda_graph_train(compile_=False)


def test_torch_engine_cuda_graph_compile_train_torch_module():
    """
    the compiled step differentiates every parameter of a torch module model as well:
    those an RF part reads through its :class:`rf.Parameter`, and those a torch module reads through its attribute,
    a tied one through all of its attributes, and binds every grad in the layout of its parameter
    """
    _run_cuda_graph_train(compile_=True, torch_model=True)


def test_torch_engine_cuda_graph_compile_train():
    """torch_cuda_graph "compile": the whole step Inductor-compiled (aot_function + compile_fx,
    no Dynamo), then captured; otherwise as :func:`test_torch_engine_cuda_graph_train`"""
    _run_cuda_graph_train(compile_=True)


def test_torch_engine_cuda_graph_compile_train_default_warmup():
    """as :func:`test_torch_engine_cuda_graph_compile_train` with the default warmup_steps (0):
    no eager step, the lazy optimizer state is created directly before the capture"""
    _run_cuda_graph_train(compile_=True, warmup_steps=None)


def test_torch_engine_optimizer_step_train():
    """
    torch_optimizer_step: eager model step, the optimizer step compiled + captured separately,
    2 epochs with the per-step LR schedule; then the optimizer checkpoint save + load
    """
    import os

    engine = _run_cuda_graph_train(compile_=False, cuda_graph=False, optimizer_step=True)
    updater = engine._updater
    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = os.path.join(tmp_dir, "opt.pt")
        updater.save_optimizer(filename)
        state = torch.load(filename)
        for group in state["optimizer"]["param_groups"]:
            assert isinstance(group["lr"], float)  # the checkpoint keeps the ordinary Python scalar
        updater.load_optimizer(filename)
    assert updater._optimizer_step._graph is None  # new state tensors: invalidated


def test_torch_engine_cuda_graph_compile_optimizer_step_train():
    """
    Compiled + captured model step (torch_cuda_graph, without capture_optimizer),
    the optimizer step compiled + captured separately (torch_optimizer_step).
    The model graph capture rebinds the grads once: the optimizer step recaptures on the new addresses.
    """
    _run_cuda_graph_train(compile_=True, optimizer_step=True)


class _MuonLike(torch.optim.Optimizer):
    """
    Written like the i6 Muon (i6_experiments exp2024_04_23_baselines optim_ext/muon.py), small:
    Muon (bf16 Newton-Schulz) on 2D params, Adam on the rest, with a Python-float lr
    (``alpha=`` / ``value=`` arguments) and a Python-int step counter.
    """

    def __init__(self, params, lr=2e-2, momentum=0.95, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        """step"""
        assert closure is None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if p.ndim == 2:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(p.grad)
                    x = p.grad.add(buf, alpha=group["momentum"]).bfloat16()
                    x = x / (x.norm() + 1e-7)
                    for _ in range(3):
                        a = x @ x.T
                        x = 3.4445 * x + (-4.7750 * a + 2.0315 * (a @ a)) @ x
                    p.mul_(1 - lr * wd)
                    p.add_(x.to(p.dtype), alpha=-lr)
                else:
                    if "exp_avg" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    t = state["step"]
                    state["exp_avg"].mul_(b1).add_(p.grad, alpha=1 - b1)
                    state["exp_avg_sq"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    denom = (state["exp_avg_sq"].sqrt() / ((1 - b2**t) ** 0.5)).add_(group["eps"])
                    p.addcdiv_(state["exp_avg"], denom, value=-lr / (1 - b1**t))


def _optimizer_step_device() -> str:
    """the non-capture optimizer step tests also run on CPU"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _adamw(params):
    # capturable: the device-tensor lr and step counters are CUDA only
    return torch.optim.AdamW(params, lr=1e-2, weight_decay=0.01, capturable=params[0].is_cuda)


def _run_optimizer_step(opt_factory, *, opts=None, num_steps=8, replace_grads_at=None, reload_at=None):
    """
    :return: params after num_steps updates with a per-step LR schedule and random grads,
        via the plain ``optimizer.step()`` if opts is None, otherwise via :class:`OptimizerStep`;
        the optimizer; and the OptimizerStep
    """
    from returnn.torch.util.optimizer_step import OptimizerStep

    device = _optimizer_step_device()
    gen = torch.Generator().manual_seed(0)
    shapes = [(32, 16), (16,), (64, 32), (32,), (8, 1, 3)]
    params = [torch.nn.Parameter(torch.randn(s, generator=gen).to(device)) for s in shapes]
    opt = opt_factory(params)
    opt_step = OptimizerStep(optimizer=opt, opts=opts) if opts is not None else None
    gen = torch.Generator(device=device).manual_seed(1)
    for p in params:
        p.grad = torch.zeros_like(p)
    for i in range(num_steps):
        for group in opt.param_groups:
            lr = 1e-2 * (i + 1) / num_steps
            if isinstance(group["lr"], torch.Tensor):
                group["lr"].fill_(lr)
            else:
                group["lr"] = lr
        if i == replace_grads_at:
            for p in params:
                p.grad = torch.zeros_like(p)
        for p in params:
            p.grad.copy_(torch.randn(p.shape, device=device, generator=gen))
        if i == reload_at:
            state_dict = opt.state_dict()
            if opt_step:
                state_dict = opt_step.state_dict_to_host_scalars(state_dict)
                for group in state_dict["param_groups"]:
                    assert isinstance(group["lr"], float)
            opt.load_state_dict(copy.deepcopy(state_dict))
            if opt_step:
                opt_step.invalidate()
        if opt_step:
            opt_step.step()
        else:
            opt.step()
    return params, opt, opt_step


def _check_optimizer_step_same_as_eager(opt_factory, opts, **kwargs):
    """params and optimizer state (incl. the checkpoint scalars) as the plain eager optimizer.step()"""
    ref_params, ref_opt, _ = _run_optimizer_step(opt_factory, **kwargs)
    params, opt, opt_step = _run_optimizer_step(opt_factory, opts=opts, **kwargs)
    # torch < 2.12 emulates the eager bf16 rounding only partially (Muon-like: up to 4e-5 abs diff on 2.7)
    tol = dict(rtol=1e-5, atol=1e-4 if torch.__version__ < (2, 12) else 1e-6)
    for p_ref, p in zip(ref_params, params):
        torch.testing.assert_close(p, p_ref, **tol)
    ref_state = ref_opt.state_dict()["state"]
    state = opt_step.state_dict_to_host_scalars(opt.state_dict())["state"]
    assert set(state) == set(ref_state)
    for i, s_ref in ref_state.items():
        assert set(state[i]) == set(s_ref)
        for k, v_ref in s_ref.items():
            v = state[i][k]
            if isinstance(v_ref, torch.Tensor):
                torch.testing.assert_close(v, v_ref, **tol)
            else:  # Python scalar, e.g. the step counter of _MuonLike
                assert type(v) is type(v_ref) and v == v_ref, f"state {i} {k}: {v!r} vs {v_ref!r}"
    return opt_step


def test_torch_optimizer_step_eager():
    opt_step = _check_optimizer_step_same_as_eager(_adamw, {"compile": False, "capture": False}, reload_at=5)
    assert opt_step._graph is None


def test_torch_optimizer_step_dynamic_state_keys():
    """
    Python-float lr and Python-int step counter as device tensors (incl. alpha= / value= args),
    back to Python scalars in the checkpoint
    """
    opts = {"dynamic_state_keys": ["step"]}
    if not torch.cuda.is_available():
        opts.update({"compile": False, "capture": False})
    opt_step = _check_optimizer_step_same_as_eager(_MuonLike, opts, reload_at=5)
    state_dict = opt_step.state_dict_to_host_scalars(opt_step._optimizer.state_dict())
    for state in state_dict["state"].values():
        if "step" in state:
            assert state["step"] == 8 and isinstance(state["step"], int)
    for group in state_dict["param_groups"]:
        assert isinstance(group["lr"], float)


def test_torch_optimizer_step_python_scalar_state_not_selected():
    try:
        _run_optimizer_step(_MuonLike, opts={"compile": False, "capture": False})
    except ValueError as exc:
        assert "dynamic_state_keys" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_torch_optimizer_step_capture():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    opt_step = _check_optimizer_step_same_as_eager(_adamw, {"compile": False})
    assert opt_step._graph is not None and opt_step._num_captures == 1


def test_torch_optimizer_step_compile_capture():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    opt_step = _check_optimizer_step_same_as_eager(_adamw, {})
    assert opt_step._graph is not None and (opt_step._num_traces, opt_step._num_captures) == (1, 1)


def test_torch_optimizer_step_compile_capture_muon_like():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    opt_step = _check_optimizer_step_same_as_eager(_MuonLike, {"dynamic_state_keys": ["step"]})
    assert (opt_step._num_traces, opt_step._num_captures) == (1, 1)


def test_torch_optimizer_step_replaced_grads():
    """new grad buffers: recapture on the new addresses, no retrace"""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    opt_step = _check_optimizer_step_same_as_eager(_adamw, {}, replace_grads_at=5)
    assert (opt_step._num_traces, opt_step._num_captures) == (1, 2)


def test_torch_optimizer_step_reload():
    """checkpoint round trip: new state tensors and lr, recapture, no retrace"""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    opt_step = _check_optimizer_step_same_as_eager(_MuonLike, {"dynamic_state_keys": ["step"]}, reload_at=5)
    assert (opt_step._num_traces, opt_step._num_captures) == (1, 2)


def test_torch_optimizer_step_updater_invalid_grad_skipped():
    """Updater: an update with a non-finite grad norm is skipped, also under the captured optimizer step"""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    config = Config(
        dict(
            optimizer={"class": "adamw", "capturable": True},
            torch_optimizer_step={},
            num_allowed_consec_invalid_gradient_steps=1,
        )
    )
    model = torch.nn.Linear(4, 3).cuda()
    updater = Updater(config=config, network=model, device="cuda", initial_learning_rate=1e-2)
    updater.create_optimizer()
    x = torch.randn(5, 4, device="cuda")
    for i in range(6):
        updater.zero_grad()
        model(x).square().sum().backward()
        if i == 4:  # after the capture (step 3)
            model.weight.grad[0, 0] = float("nan")
        before = [p.detach().clone() for p in model.parameters()]
        updater.step()
        changed = any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
        assert changed == (i != 4), f"step {i}: params changed {changed}"
    assert updater._optimizer_step._num_captures == 1


def _pin_memory_iter(src, **kwargs):
    from returnn.torch.data.pin_memory import PinMemoryIter

    return PinMemoryIter(iter(src), device=torch.device("cuda", torch.cuda.current_device()), **kwargs)


def test_pin_memory_iter_batches_unchanged():
    """same batches in the same order, all tensors pinned (also nested), other values as they are"""
    from collections import namedtuple

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    _Pair = namedtuple("_Pair", ["a", "b"])
    gen = torch.Generator().manual_seed(0)
    batches = [
        {
            "data": torch.randn(3, 5 + i, 2, generator=gen),
            "data:seq_len": torch.tensor([5 + i, 4, 2], dtype=torch.int32),
            "seq_tag": numpy.array([f"seq-{i}-{j}" for j in range(3)]),
            "num_seqs": 100,
            "list": [torch.full((2,), i)],
            "nested": {"pair": _Pair(torch.full((3,), i), "x")},
        }
        for i in range(10)
    ]
    out = list(_pin_memory_iter(batches))
    assert len(out) == len(batches)
    for ref, batch in zip(batches, out):
        assert list(batch) == list(ref)  # same keys, same order
        for k in ["data", "data:seq_len"]:
            assert batch[k].is_pinned() and batch[k].dtype == ref[k].dtype and torch.equal(batch[k], ref[k])
        assert batch["list"][0].is_pinned() and torch.equal(batch["list"][0], ref["list"][0])
        pair = batch["nested"]["pair"]
        assert type(pair) is _Pair and pair.b == "x"
        assert pair.a.is_pinned() and torch.equal(pair.a, ref["nested"]["pair"].a)
        assert batch["seq_tag"] is ref["seq_tag"] and batch["num_seqs"] == 100


def test_pin_memory_iter_buffer_lifetime():
    """
    Each pinned batch stays valid while the consumer holds it, also while further batches get pinned,
    and a batch dropped right after its async H2D copy is not reused before the copy is done.
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    n, shape = 30, (256, 1024)

    def _src():
        for i in range(n):
            yield {"data": torch.full(shape, float(i))}

    it = _pin_memory_iter(_src())
    first = next(it)
    first_ptr = first["data"].data_ptr()
    stream = torch.cuda.Stream()
    dev = []
    with torch.cuda.stream(stream):
        for batch in it:
            # delays the stream: the H2D copy below is still pending when the batch is dropped,
            # while the next batches get pinned
            torch.cuda._sleep(1_000_000)
            dev.append(batch["data"].to("cuda", non_blocking=True))
            del batch
    stream.synchronize()
    assert first["data"].data_ptr() == first_ptr and bool((first["data"] == 0.0).all())
    assert [float(d[0, 0]) for d in dev] == [float(i) for i in range(1, n)]
    for i, d in enumerate(dev):
        assert bool((d == float(i + 1)).all()), f"batch {i + 1} overwritten"


def test_pin_memory_iter_release_after_handoff():
    """
    After handing a batch over, the pin thread may hold the last reference to it
    (the consumer already copied from it and dropped it); that release must not run during a capture.
    Forced: the pin thread pauses right after publishing the batch, the consumer copies and drops it,
    starts a capture, and the pin thread resumes inside that capture.
    """
    import threading
    import time
    import weakref
    from returnn.torch.util.capture_lock import cuda_graph_capture

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    src_gate = threading.Event()

    def _src():
        src_gate.wait(timeout=60)
        yield {"data": torch.arange(1024, dtype=torch.float32)}

    it = _pin_memory_iter(_src())
    published, resume = threading.Event(), threading.Event()
    orig_put = it._queue.put
    num_puts = 0

    def _put(item, *args, **kwargs):
        nonlocal num_puts
        orig_put(item, *args, **kwargs)
        num_puts += 1
        if num_puts == 1:  # the batch: pause before the pin thread drops its reference
            published.set()
            resume.wait(timeout=60)

    it._queue.put = _put  # the pin thread uses the same queue object, and does not put before src_gate
    src_gate.set()
    assert published.wait(timeout=60)

    state = {"capturing": False}
    released = []  # per release: whether it ran during the capture
    batch = next(it)
    data = batch["data"]
    assert data.is_pinned()
    weakref.finalize(data, lambda: released.append(state["capturing"]))
    dev = data.to("cuda", non_blocking=True)
    del batch, data
    assert not released  # the pin thread still holds a reference

    x = torch.zeros(8, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with cuda_graph_capture(graph):
        state["capturing"] = True
        resume.set()
        time.sleep(0.3)  # an unguarded release would run now
        x.add_(1)
        state["capturing"] = False
    for _ in range(200):
        if released:
            break
        time.sleep(0.05)
    assert released == [False], f"release during the capture: {released}"
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(dev.cpu(), torch.arange(1024, dtype=torch.float32))
    assert torch.equal(x.cpu(), torch.ones(8))
    assert list(it) == []


def test_pin_memory_iter_num_threads():
    """the pin thread uses one intra-op thread, the consumer thread keeps its setting"""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    num_threads_main = torch.get_num_threads()
    num_threads_pin = []

    def _src():
        num_threads_pin.append(torch.get_num_threads())  # runs in the pin thread
        yield {"data": torch.zeros(2)}

    assert len(list(_pin_memory_iter(_src()))) == 1
    assert num_threads_pin == [1]
    assert torch.get_num_threads() == num_threads_main


def test_pin_memory_iter_exception():
    """an exception of the source is re-raised in the consumer, after the batches before it"""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")

    class _Error(Exception):
        pass

    def _src():
        yield {"data": torch.zeros(2)}
        yield {"data": torch.ones(2)}
        raise _Error("source failed")

    it = _pin_memory_iter(_src())
    assert float(next(it)["data"][0]) == 0.0
    assert float(next(it)["data"][0]) == 1.0
    try:
        next(it)
    except _Error as exc:
        assert str(exc) == "source failed"
    else:
        raise AssertionError("expected _Error")
    assert not it._thread.is_alive()
    try:
        next(it)
    except StopIteration:
        pass
    else:
        raise AssertionError("expected StopIteration after the error")


def test_pin_memory_data_loader_queue_size():
    """default queue size: as many batches as the DataLoader has in flight, at least 2"""
    from returnn.torch.data.pin_memory import PinMemoryDataLoader

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    batches = [{"data": torch.full((2,), i)} for i in range(5)]

    def _loader(**opts):
        return torch.utils.data.DataLoader(batches, batch_size=None, **opts)

    assert PinMemoryDataLoader(_loader(num_workers=3, prefetch_factor=4), device="cuda").queue_size == 12
    assert PinMemoryDataLoader(_loader(num_workers=1), device="cuda").queue_size == 2  # default prefetch_factor 2
    assert PinMemoryDataLoader(_loader(num_workers=0), device="cuda").queue_size == 2
    assert PinMemoryDataLoader(_loader(num_workers=3), device="cuda", queue_size=1).queue_size == 1
    loader = PinMemoryDataLoader(_loader(num_workers=0), device="cuda", queue_size=3)
    it = iter(loader)
    assert it._queue.maxsize == 3
    assert [float(b["data"][0]) for b in it] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_pin_memory_iter_shutdown():
    """
    The thread stops: at the end, on close() of an unfinished iterator (thread blocked on the full queue),
    when the consumer drops it, and for the previous iterator on the next iter() of the loader
    """
    import gc
    from itertools import count
    from returnn.torch.data.pin_memory import PinMemoryDataLoader

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")

    def _endless():
        for i in count():
            yield {"data": torch.full((4,), i)}

    it = _pin_memory_iter([{"data": torch.zeros(2)}])
    assert len(list(it)) == 1 and not it._thread.is_alive()

    it = _pin_memory_iter(_endless(), queue_size=1)
    assert float(next(it)["data"][0]) == 0.0
    it.close()
    assert not it._thread.is_alive()
    assert list(it) == []

    it = _pin_memory_iter(_endless())
    next(it)
    thread = it._thread
    del it
    gc.collect()
    thread.join(timeout=10)
    assert not thread.is_alive()

    loader = PinMemoryDataLoader([{"data": torch.full((2,), i)} for i in range(5)], device="cuda")
    it1 = iter(loader)
    assert float(next(it1)["data"][0]) == 0.0
    it2 = iter(loader)
    assert not it1._thread.is_alive()
    assert [float(b["data"][0]) for b in it2] == [0.0, 1.0, 2.0, 3.0, 4.0]


class _PinCaptureOverlapForcer:
    """
    Forces the background pinning (:class:`returnn.torch.data.pin_memory.PinMemoryIter`)
    to overlap each CUDA graph capture:
    the pin thread holds back each batch until a capture has begun or the consumer waits for a batch,
    and each capture, once begun, waits until the pin thread tries to pin, and then a bit longer
    (an unguarded pin would now run inside the capture).
    Records per capture whether a pin was attempted during it, and whether that pin finished during it.
    """

    def __init__(self):
        import threading
        from contextlib import ExitStack
        from returnn.torch.data import pin_memory

        self._pin_memory = pin_memory
        self._cond = threading.Condition()
        self._capturing = False
        self._capture_released_pin = True  # the current capture released a held-back pin already
        self._consumer_waiting = False
        self.captures = []  # per capture: dict(pin_attempted=..., pin_finished_during=...)
        self._exit_stack = ExitStack()

    def __enter__(self):
        import time
        from unittest import mock

        orig_pin_batch = self._pin_memory._pin_batch
        orig_next = self._pin_memory.PinMemoryIter.__next__
        orig_capture_begin = torch.cuda.CUDAGraph.capture_begin
        orig_capture_end = torch.cuda.CUDAGraph.capture_end
        forcer = self

        def _pin_batch(batch):
            with forcer._cond:
                forcer._cond.wait_for(lambda: not forcer._capture_released_pin or forcer._consumer_waiting, timeout=60)
                capture_idx = None
                if forcer._capturing and not forcer._capture_released_pin:
                    forcer._capture_released_pin = True
                    capture_idx = len(forcer.captures) - 1
                    forcer.captures[capture_idx]["pin_attempted"] = True
                    forcer._cond.notify_all()
            res = orig_pin_batch(batch)
            if capture_idx is not None:
                with forcer._cond:
                    forcer.captures[capture_idx]["pin_finished_during"] = (
                        forcer._capturing and len(forcer.captures) - 1 == capture_idx
                    )
            return res

        def _next(it):
            with forcer._cond:
                forcer._consumer_waiting = True
                forcer._cond.notify_all()
            try:
                return orig_next(it)
            finally:
                with forcer._cond:
                    forcer._consumer_waiting = False

        def _capture_begin(graph, *args, **kwargs):
            orig_capture_begin(graph, *args, **kwargs)
            with forcer._cond:
                forcer._capturing = True
                forcer._capture_released_pin = False
                forcer.captures.append(dict(pin_attempted=False, pin_finished_during=False))
                forcer._cond.notify_all()
                forcer._cond.wait_for(lambda: forcer.captures[-1]["pin_attempted"], timeout=10)
            time.sleep(0.2)

        def _capture_end(graph, *args, **kwargs):
            try:
                return orig_capture_end(graph, *args, **kwargs)
            finally:
                with forcer._cond:
                    forcer._capturing = False
                    forcer._capture_released_pin = True
                    forcer._cond.notify_all()

        patch = mock.patch.object
        self._exit_stack.enter_context(patch(self._pin_memory, "_pin_batch", _pin_batch))
        self._exit_stack.enter_context(patch(self._pin_memory.PinMemoryIter, "__next__", _next))
        self._exit_stack.enter_context(patch(torch.cuda.CUDAGraph, "capture_begin", _capture_begin))
        self._exit_stack.enter_context(patch(torch.cuda.CUDAGraph, "capture_end", _capture_end))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_stack.close()


def _run_pin_memory_capture_overlap(
    *, cuda_graph: bool, compile_: bool, optimizer_step: bool, invalidate_optimizer_step_at: Optional[int] = None
) -> int:
    """
    Train 2 epochs with DataLoader pin_memory (-> the RETURNN pinning thread), each capture forced to overlap
    with a background pin (see :class:`_PinCaptureOverlapForcer`), then check the checkpoints.

    :param invalidate_optimizer_step_at: before this optimizer update, drop the captured optimizer step
        (as loading the optimizer state does), so the next update recaptures
    :return: number of captures
    """
    from unittest import mock
    from contextlib import ExitStack
    import os
    import glob
    from returnn.torch.data.pin_memory import PinMemoryDataLoader

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    config, dataset, _dev_dataset = _build_cuda_graph_train_config_and_dataset(
        compile_=compile_, warmup_steps=None, cuda_graph=cuda_graph, optimizer_step=optimizer_step
    )
    config.typed_dict["torch_dataloader_opts"] = {"num_workers": 1, "pin_memory": True}
    with tempfile.TemporaryDirectory() as tmp_dir, global_config_ctx(config):
        config.typed_dict["model"] = f"{tmp_dir}/model"
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        loader = engine._train_dataloader
        assert isinstance(loader, PinMemoryDataLoader) and not loader.data_loader.pin_memory
        # all read by now; the global config gets pickled to the DataLoader worker, local functions cannot
        for key in ["get_model", "train_step", "dynamic_learning_rate"]:
            config.typed_dict.pop(key)
        with _PinCaptureOverlapForcer() as forcer, ExitStack() as stack:
            opt_step = engine._updater._optimizer_step
            if invalidate_optimizer_step_at is not None:
                orig_opt_step = opt_step.step
                num_opt_steps = 0

                def _opt_step():
                    nonlocal num_opt_steps
                    if num_opt_steps == invalidate_optimizer_step_at:
                        assert opt_step._graph is not None, "optimizer step not captured yet"
                        opt_step.invalidate()
                    num_opt_steps += 1
                    orig_opt_step()

                stack.enter_context(mock.patch.object(opt_step, "step", _opt_step))
            engine.train()
        print("captures:", forcer.captures)
        if invalidate_optimizer_step_at is not None:
            assert opt_step._num_captures == 2, "no recapture"
        assert forcer.captures, "nothing captured"
        for i, capture in enumerate(forcer.captures):
            assert capture["pin_attempted"], f"capture {i}: no pin attempt during the capture, overlap not forced"
            assert not capture["pin_finished_during"], f"capture {i}: pinned during the capture"
        if cuda_graph:
            assert engine._graph_capture._graph is not None, "graph never captured"
        if optimizer_step:
            assert engine._updater._optimizer_step._graph is not None, "optimizer step never captured"

        params = {k: v.detach().cpu() for k, v in engine._pt_model.state_dict().items()}
        model_files = sorted(glob.glob(f"{tmp_dir}/model.*.pt"))
        model_files = [fn for fn in model_files if not fn.endswith(".opt.pt")]
        assert [os.path.basename(fn) for fn in model_files] == ["model.001.pt", "model.002.pt"], model_files
        for fn in model_files:
            ckpt = torch.load(fn, map_location="cpu")
            assert set(ckpt["model"]) == set(params)
            for k, v in ckpt["model"].items():
                assert torch.isfinite(v).all(), f"{fn}: non-finite {k}"
        for k, v in torch.load(model_files[-1], map_location="cpu")["model"].items():
            assert torch.equal(v, params[k]), f"last checkpoint {k} differs from the trained param"
        opt_ckpt = torch.load(f"{tmp_dir}/model.002.opt.pt", map_location="cpu")
        assert opt_ckpt["optimizer"]["state"]
        if optimizer_step:  # the checkpoint keeps the ordinary Python scalar
            for group in opt_ckpt["optimizer"]["param_groups"]:
                assert isinstance(group["lr"], float)
    return len(forcer.captures)


def test_pin_memory_capture_overlap_model_step():
    """background pinning forced to overlap the model step capture (incl. the optimizer, capture_optimizer)"""
    n = _run_pin_memory_capture_overlap(cuda_graph=True, compile_=True, optimizer_step=False)
    assert n == 1


def test_pin_memory_capture_overlap_optimizer_step():
    """
    background pinning forced to overlap the separate optimizer step capture (torch_optimizer_step)
    and its recapture
    """
    n = _run_pin_memory_capture_overlap(
        cuda_graph=False, compile_=False, optimizer_step=True, invalidate_optimizer_step_at=6
    )
    assert n == 2


def test_pin_memory_capture_overlap_model_and_optimizer_step():
    """
    background pinning forced to overlap the model step capture, the separate optimizer step capture,
    and its recapture
    """
    n = _run_pin_memory_capture_overlap(
        cuda_graph=True, compile_=True, optimizer_step=True, invalidate_optimizer_step_at=6
    )
    assert n == 3


def _check_pin_memory_engine_batches_unchanged(*, optimizer_step: bool, num_workers: int = 1):
    """
    Engine with pin_memory: with num_workers > 0 the RETURNN pinning thread instead of the DataLoader one,
    with or without a CUDA graph capture configured,
    same batches (tensors, lengths, tags, order) as the unwrapped DataLoader;
    then one training epoch
    """
    from returnn.torch.data.pin_memory import PinMemoryDataLoader

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    config = Config(
        dict(
            task="train",
            device="gpu",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainTestModel,
            train_step=TrainTestModel.train_step,
            batch_size=100,
            max_seqs=10,
            num_epochs=1,
            optimizer={"class": "adamw", "capturable": True},
            torch_dataloader_opts={"num_workers": num_workers, "pin_memory": True},
        )
    )
    if optimizer_step:
        config.typed_dict["torch_optimizer_step"] = {}
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "train", "fixed_random_seed": 1})
    dataset.init_seq_order(epoch=1)
    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        loader = engine._train_dataloader
        if num_workers == 0:  # the DataLoader pins in the main thread, unchanged
            assert isinstance(loader, torch.utils.data.DataLoader) and loader.pin_memory
            return
        assert isinstance(loader, PinMemoryDataLoader) and not loader.data_loader.pin_memory
        pinned = list(loader)
        plain = list(loader.data_loader)
        engine.train()
    assert len(pinned) == len(plain) > 1
    for batch, ref in zip(pinned, plain):
        assert set(batch) == set(ref)
        assert "seq_tag" in batch and "data:seq_len" in batch
        for k, v_ref in ref.items():
            v = batch[k]
            if isinstance(v_ref, torch.Tensor):
                assert v.is_pinned() and v.dtype == v_ref.dtype and torch.equal(v, v_ref), k
            elif isinstance(v_ref, numpy.ndarray):
                assert v.dtype == v_ref.dtype and numpy.array_equal(v, v_ref), k
            else:
                assert v == v_ref, k


def test_pin_memory_engine_batches_unchanged():
    """no CUDA graph capture configured: also the RETURNN pinning thread"""
    _check_pin_memory_engine_batches_unchanged(optimizer_step=False)


def test_pin_memory_engine_batches_unchanged_optimizer_step():
    """with the separately captured optimizer step"""
    _check_pin_memory_engine_batches_unchanged(optimizer_step=True)


def test_pin_memory_engine_no_workers():
    """num_workers 0: the plain DataLoader, which pins in the main thread"""
    _check_pin_memory_engine_batches_unchanged(optimizer_step=False, num_workers=0)


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
