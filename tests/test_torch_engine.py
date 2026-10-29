"""
Tests for PyTorch engine.
"""

from __future__ import annotations
import _setup_test_env  # noqa
from typing import Optional, Any, Dict, Tuple, List
import contextlib
import copy
import functools
import io
import json
import os
import sys
import time
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


def test_get_train_start_epoch_with_load_epoch():
    from returnn.util.basic import BackendEngine

    with tempfile.TemporaryDirectory(prefix="returnn_test_start_epoch") as tmp_dir:
        model = f"{tmp_dir}/model"
        for epoch in [43, 44]:
            for postfix in [".pt", ".opt.pt"]:
                open(Engine.epoch_model_filename(model, epoch) + postfix, "w").close()

        def _select(**opts) -> Tuple[Optional[int], Optional[str], int]:
            config = Config(dict(backend="torch", task="train", model=model, num_epochs=45, **opts))
            BackendEngine.select_engine(config=config, _select_rf_backend=False)
            return (*Engine.get_epoch_model(config), Engine.get_train_start_epoch(config))

        # continuation after epoch 44, however it is specified
        assert _select() == (44, f"{model}.044", 45)
        assert _select(start_epoch=45) == (44, f"{model}.044", 45)
        assert _select(load_epoch=44) == (44, f"{model}.044", 45)
        assert _select(start_epoch=45, load_epoch=44) == (44, f"{model}.044", 45)
        # continuation after an earlier epoch
        assert _select(start_epoch=44) == (43, f"{model}.043", 44)
        assert _select(start_epoch=44, load_epoch=43) == (43, f"{model}.043", 44)
        # contradiction
        for start_epoch, load_epoch in [(45, 43), (44, 44), (2, 44)]:
            try:
                _select(start_epoch=start_epoch, load_epoch=load_epoch)
            except ValueError as exc:
                assert f"start_epoch {start_epoch} with load_epoch {load_epoch}" in str(exc)
            else:
                raise Exception(f"start_epoch {start_epoch} with load_epoch {load_epoch}: did not get ValueError")
        # new training
        assert _select(start_epoch=1) == (None, None, 1)
        # new training with model import
        assert _select(start_epoch=1, load_epoch=44) == (None, f"{model}.044", 1)
        assert _select(start_epoch=1, load=f"{model}.044") == (None, f"{model}.044", 1)


def test_torch_engine_train_start_epoch_with_load_epoch():
    # State right before the first update, with real model and optimizer checkpoints of epoch 2.
    class _StopBeforeFirstUpdate(Exception):
        pass

    def _state_before_first_update(tmp_dir: str, **opts) -> Dict[str, Any]:
        state = {}

        def _train_step(**_kwargs):
            optimizer = engine.get_pt_optimizer()
            state.update(
                epoch=engine.epoch,
                run_ctx_epoch=rf.get_run_ctx().epoch,
                global_train_step=engine.global_train_step,
                learning_rate=optimizer.param_groups[0]["lr"],
                dataset_epoch=dataset.epoch,
                model=copy.deepcopy(engine.get_pt_model().state_dict()),
                optimizer=copy.deepcopy(optimizer.state_dict()["state"]),
            )
            raise _StopBeforeFirstUpdate()

        config = _start_epoch_train_config(tmp_dir, num_epochs=3, train_step=_train_step, **opts)
        dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 20, "name": "train", "fixed_random_seed": 1})
        with global_config_ctx(config):
            engine = Engine(config=config)
            engine.init_train_from_config(train_data=dataset)
            try:
                engine.train()
            except _StopBeforeFirstUpdate:
                pass
            else:
                raise Exception("train step was not called")
        return state

    def _assert_tensors_equal(a: Dict[str, Any], b: Dict[str, Any]):
        assert a.keys() == b.keys()
        for key in a:
            assert torch.equal(torch.as_tensor(a[key]), torch.as_tensor(b[key])), key

    with tempfile.TemporaryDirectory(prefix="returnn_test_start_epoch") as tmp_dir:
        config = _start_epoch_train_config(tmp_dir, num_epochs=2)
        dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 20, "name": "train", "fixed_random_seed": 1})
        with global_config_ctx(config):
            engine = Engine(config=config)
            engine.init_train_from_config(train_data=dataset)
            engine.train()
        model_ckpt = torch.load(f"{tmp_dir}/model.002.pt", map_location="cpu")
        opt_ckpt = torch.load(f"{tmp_dir}/model.002.opt.pt", map_location="cpu")["optimizer"]["state"]
        assert model_ckpt["step"] == engine.global_train_step > 0 and opt_ckpt

        for opts in [dict(start_epoch=3, load_epoch=2), dict(start_epoch=3), dict(load_epoch=2), dict()]:
            state = _state_before_first_update(tmp_dir, **opts)
            assert state["epoch"] == state["run_ctx_epoch"] == state["dataset_epoch"] == 3, (opts, state)
            assert state["global_train_step"] == model_ckpt["step"], (opts, state)
            assert state["learning_rate"] == 0.03, (opts, state)
            _assert_tensors_equal(state["model"], model_ckpt["model"])
            assert state["optimizer"].keys() == opt_ckpt.keys()
            for param_idx, param_state in state["optimizer"].items():
                _assert_tensors_equal(param_state, opt_ckpt[param_idx])

        # New training with model import: only the parameters are taken from the checkpoint.
        state = _state_before_first_update(tmp_dir, start_epoch=1, load_epoch=2)
        assert state["epoch"] == state["run_ctx_epoch"] == state["dataset_epoch"] == 1, state
        assert state["learning_rate"] == 0.01, state
        _assert_tensors_equal(state["model"], model_ckpt["model"])
        assert not state["optimizer"], state


def _start_epoch_train_config(tmp_dir: str, **kwargs) -> Config:
    opts = dict(
        task="train",
        device="cpu",
        extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
        get_model=TrainTestModel,
        train_step=TrainTestModel.train_step,
        batch_size=500,
        max_seqs=4,
        optimizer={"class": "adam"},
        learning_rates=[0.01, 0.02, 0.03],
        model=f"{tmp_dir}/model",
        learning_rate_file=f"{tmp_dir}/learning_rates",
        torch_dataloader_opts=dict(num_workers=0),  # the engine then uses our dataset instance
    )
    opts.update(kwargs)
    return Config(opts)


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


def _env_without_wandb_defaults() -> Dict[str, str]:
    """the default id / name must not come from the env the test runs in (e.g. a Slurm job)"""
    return {k: v for k, v in os.environ.items() if k not in {"SLURM_JOB_NAME", "WANDB_RUN_ID"}}


def _train_with_wandb_stub(
    *,
    num_epochs: int,
    stop_for_resubmission: bool = False,
    model: Optional[str] = None,
    config_file: Optional[str] = None,
):
    """
    :return: (wandb_opts of the config, wandb.init kwargs, events, global train step at the end),
        events: ("log", step, data), ("finish", exit_code), ("kill", sig)
    """
    import types

    init_kwargs = []
    events: List[Tuple[Any, ...]] = []

    class _WandbRun:
        def log(self, data: Dict[str, Any], *, step: int):
            events.append(("log", step, data))

        def finish(self, *, exit_code: int):
            events.append(("finish", exit_code))

    def _wandb_init(**kwargs):
        init_kwargs.append(kwargs)
        return _WandbRun()

    wandb_opts = {"project": "test", "log_every_n_train_steps": 2}
    config = Config(
        dict(
            task="train",
            device="cpu",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainTestModel,
            train_step=TrainTestModel.train_step,
            batch_size=500,
            optimizer={"class": "adam"},
            num_epochs=num_epochs,
            wandb_opts=wandb_opts,
            stop_for_resubmission_when_low_time_left=stop_for_resubmission,
        )
    )
    if model:
        config.set("model", model)
    if config_file:
        config.files.append(config_file)  # as if loaded from this file
    train_dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 30, "name": "train"})
    train_dataset.init_seq_order(epoch=1)
    dev_dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 10, "name": "dev", "fixed_random_seed": 1})

    # stub module, as CI has no wandb. Not mock.patch.dict(sys.modules): it also drops all modules imported meanwhile
    sys.modules["wandb"] = types.SimpleNamespace(init=_wandb_init)
    try:
        with unittest.mock.patch.dict(os.environ, _env_without_wandb_defaults(), clear=True):
            with unittest.mock.patch("returnn.util.basic.slurm_time_left_sec", return_value=0):
                with unittest.mock.patch.object(os, "kill", side_effect=lambda pid, sig: events.append(("kill", sig))):
                    with global_config_ctx(config):
                        engine = Engine(config=config)
                        engine.init_train_from_config(train_data=train_dataset, dev_data=dev_dataset)
                        engine.train()
                        # like returnn.__main__: the stop for resubmission ends in a KeyboardInterrupt
                        engine.finalize(error_occurred=stop_for_resubmission)
    finally:
        del sys.modules["wandb"]
    return wandb_opts, init_kwargs, events, engine.global_train_step


def test_torch_engine_wandb():
    wandb_opts, init_kwargs, events, num_steps = _train_with_wandb_stub(num_epochs=1)
    assert wandb_opts == {"project": "test", "log_every_n_train_steps": 2}  # the config is not modified
    assert init_kwargs == [{"project": "test", "job_type": "train", "resume": "allow"}]  # no config file, so no id
    assert num_steps >= 4
    logs = [event[1:] for event in events if event[0] == "log"]
    step_keys = ["epoch", "train_step/ce", "train_step/fer", "train_step/learning_rate"]
    assert [(step, sorted(data)) for step, data in logs] == [(step, step_keys) for step in range(0, num_steps, 2)] + [
        (num_steps, ["epoch", "train/ce", "train/fer"]),
        (num_steps, ["dev/ce", "dev/fer", "epoch"]),
    ]
    assert all(data["epoch"] == 1 for _, data in logs)
    assert events[-1] == ("finish", 0)


def test_torch_engine_wandb_continue_from_checkpoint():
    import hashlib

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_file = f"{tmp_dir}/returnn.config"
        run_id = hashlib.sha256(repr(("train", [config_file])).encode("utf8")).hexdigest()
        kwargs = dict(model=f"{tmp_dir}/model", config_file=config_file)
        _, init_kwargs, _, step_after_ep1 = _train_with_wandb_stub(num_epochs=1, **kwargs)
        assert init_kwargs == [{"project": "test", "job_type": "train", "id": run_id, "resume": "allow"}]
        # e.g. a resubmission: continues from the epoch 1 checkpoint, in the same W&B run
        _, init_kwargs, events, _ = _train_with_wandb_stub(num_epochs=2, **kwargs)
        assert init_kwargs == [{"project": "test", "job_type": "train", "id": run_id, "resume": "allow"}]
        assert min(event[1] for event in events if event[0] == "log") == step_after_ep1


def test_torch_engine_wandb_default_id_per_task():
    import types

    init_kwargs = []
    sys.modules["wandb"] = types.SimpleNamespace(init=lambda **kwargs: init_kwargs.append(kwargs))
    try:
        with unittest.mock.patch.dict(os.environ, _env_without_wandb_defaults(), clear=True):
            for task in ["train", "eval"]:
                config = Config(dict(task=task, wandb_opts={"project": "test"}))
                config.files.append("/tmp/returnn.config")  # as if loaded from this file
                Engine(config=config)._wandb_init()
    finally:
        del sys.modules["wandb"]
    assert [kwargs["job_type"] for kwargs in init_kwargs] == ["train", "eval"]
    # the same config file with another task (e.g. ++task eval) does not continue the training run
    assert init_kwargs[0]["id"] != init_kwargs[1]["id"]


def test_torch_engine_wandb_stop_for_resubmission():
    import signal

    _, _, events, _ = _train_with_wandb_stub(num_epochs=2, stop_for_resubmission=True)
    # the planned stop after epoch 1 finishes the run as successful, before the SIGINT
    assert events[-2:] == [("finish", 0), ("kill", signal.SIGINT)]
    assert [event[0] for event in events].count("finish") == 1
    logs = [event[2] for event in events if event[0] == "log"]
    assert any("dev/ce" in data for data in logs)
    assert all(data["epoch"] == 1 for data in logs)


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


@unittest.skipIf(torch.__version__ < (2,), "torch.load weights_only cannot load floats before PyTorch 2.0")
def test_save_optimizer_opts_weights_only():
    include_check_partial = functools.partial(lambda **_kwargs: None)
    for optimizer, expected_saved_opts in [
        (torch.optim.AdamW, "adamw"),
        (
            {
                "class": torch.optim.AdamW,
                "betas": (0.9, 0.98),
                "weight_decay": 1e-3,
                "weight_decay_modules_blacklist": [torch.nn.LayerNorm, "rf.Embedding"],
                "weight_decay_custom_include_check": lambda **_kwargs: None,
            },
            {
                "class": "adamw",
                "betas": (0.9, 0.98),
                "weight_decay": 1e-3,
                "weight_decay_modules_blacklist": ["torch.nn.modules.normalization.LayerNorm", "rf.Embedding"],
                "weight_decay_custom_include_check": f"{__name__}.<lambda>",
            },
        ),
        (
            {"class": "adamw", "weight_decay": 1e-3, "weight_decay_custom_include_check": include_check_partial},
            {
                "class": "adamw",
                "weight_decay": 1e-3,
                "weight_decay_custom_include_check": f"object:{include_check_partial!r}",
            },
        ),
    ]:
        config = Config(dict(optimizer=optimizer))
        model = torch.nn.Linear(7, 5)
        updater = Updater(config=config, network=model, device=torch.device("cpu"))
        updater.create_optimizer()

        with tempfile.TemporaryDirectory(prefix="returnn_test_save_optimizer_opts_weights_only") as tmp_dir:
            updater.save_optimizer(tmp_dir + "/model.opt.pt")
            saved = torch.load(tmp_dir + "/model.opt.pt", weights_only=True)
            assert saved["optimizer_opts"] == expected_saved_opts
            updater.load_optimizer(tmp_dir + "/model.opt.pt")

    # numpy values given to the optimizer itself would break its own state dict, so check the conversion directly
    from returnn.torch.updater import _optimizer_opts_for_checkpoint

    buffer = io.BytesIO()
    torch.save(_optimizer_opts_for_checkpoint({"a": numpy.float32(0.5), "b": numpy.arange(4).reshape(2, 2)}), buffer)
    buffer.seek(0)
    assert torch.load(buffer, weights_only=True) == {"a": 0.5, "b": [[0, 1], [2, 3]]}


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


class _AnchoredSGD(torch.optim.Optimizer):
    """
    SGD from an anchor, the state starting as a copy of the param (like the z of a schedule-free optimizer):
    a lazily created optimizer state which is not all zeros, and an init_state() to create it in advance
    """

    def __init__(self, params, lr: float):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def init_state(self):
        """the state of the params with grad, as the first step creates it"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and not self.state[p]:
                    self.state[p]["anchor"] = p.detach().clone()
                    self.state[p]["grad_sum"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        """one update"""
        assert closure is None
        self.init_state()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["grad_sum"].add_(p.grad)
                p.copy_(state["anchor"] - group["lr"] * state["grad_sum"])


def test_init_optimizer_state():
    """
    init_optimizer_state creates the state as the first step would:
    the state created in advance, then some steps, equals the same steps without it
    (params and state), for the torch optimizers and an optimizer with init_state().
    """
    from returnn.torch.updater import init_optimizer_state

    cases = [
        ("SGD", dict(lr=0.1, momentum=0.9)),
        ("SGD", dict(lr=0.1, momentum=0.9, nesterov=True)),
        ("SGD", dict(lr=0.1)),
        ("Adam", dict(lr=0.1, amsgrad=True)),
        ("AdamW", dict(lr=0.1)),
        ("Adamax", dict(lr=0.1)),
        ("NAdam", dict(lr=0.1)),
        ("RAdam", dict(lr=0.1)),
        ("Adadelta", dict(lr=0.1)),
        ("RMSprop", dict(lr=0.1, momentum=0.9, centered=True)),
        ("ASGD", dict(lr=0.1)),
        ("Adagrad", dict(lr=0.1, initial_accumulator_value=0.1)),
        ("Rprop", dict(lr=0.1)),
        ("Adafactor", dict(lr=0.1)),
        ("Muon", dict(lr=0.1)),
        (_AnchoredSGD, dict(lr=0.1)),
    ]
    num_checked = 0
    for name, kwargs in cases:
        cls = getattr(torch.optim, name, None) if isinstance(name, str) else name
        if cls is None or not (cls is torch.optim.SGD or hasattr(cls, "init_state") or hasattr(cls, "_init_group")):
            print(f"{name}: not in this torch version, or no _init_group")
            continue
        print(f"{name} {kwargs}")
        num_checked += 1
        shapes = [(4, 3), (3, 2)] if name == "Muon" else [(4, 3), (3,)]  # Muon: matrix params only
        gen = torch.Generator().manual_seed(42)
        init_values = [torch.randn(shape, generator=gen) for shape in shapes]
        grads_per_step = [[torch.randn(shape, generator=gen) for shape in shapes] for _ in range(3)]
        results = []
        for in_advance in (False, True):
            params = [torch.nn.Parameter(v.clone()) for v in init_values]
            opt = cls(params, **kwargs)
            if in_advance:
                for p in params:
                    p.grad = torch.zeros_like(p)
                init_optimizer_state(opt)
                for p, v in zip(params, init_values):
                    assert torch.equal(p, v), f"{name}: params changed without a step"
                    if kwargs.get("momentum", 0) != 0 or cls is not torch.optim.SGD:  # plain SGD has no state
                        assert opt.state[p], f"{name}: no state created in advance"
            for grads in grads_per_step:
                for p, g in zip(params, grads):
                    p.grad = g.clone()
                opt.step()
            results.append((params, opt))
        (params_ref, opt_ref), (params_, opt_) = results
        for i, (p_ref, p) in enumerate(zip(params_ref, params_)):
            torch.testing.assert_close(p, p_ref, msg=lambda m: f"{name} param {i}: {m}")
            state_ref, state = opt_ref.state[p_ref], opt_.state[p]
            assert set(state) == set(state_ref), f"{name} param {i}: state keys {set(state)} != {set(state_ref)}"
            for k, v_ref in state_ref.items():
                if isinstance(v_ref, torch.Tensor):
                    torch.testing.assert_close(state[k], v_ref, msg=lambda m: f"{name} param {i} state {k}: {m}")
                else:
                    assert state[k] == v_ref, f"{name} param {i} state {k}: {state[k]} != {v_ref}"
    assert num_checked >= 2

    # no _init_group, no init_state: loud
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = torch.zeros(3)
    try:
        init_optimizer_state(torch.optim.LBFGS([p]))
    except NotImplementedError as exc:
        print("LBFGS:", exc)
    else:
        raise AssertionError("LBFGS: expected NotImplementedError")


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
    engine._torch_distributed_ctx.destroy_process_groups()


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
    engine._torch_distributed_ctx.destroy_process_groups()


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


def _torch_distributed_shard_worker(rank: int, world_size: int, port: int, tmp_dir: str, hdf_fn: str):
    from returnn.__main__ import load_data
    from returnn.datasets import Dataset

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world_size),
    )
    seq_tags = []

    def _train_step(*, model: TrainTestModel, extern_data: TensorDict, **kwargs):
        seq_tags.extend(extern_data["seq_tag"].raw_tensor.tolist())
        TrainTestModel.train_step(model=model, extern_data=extern_data, **kwargs)

    config = Config(
        dict(
            task="train",
            device="cpu",
            train={"class": "HDFDataset", "files": [hdf_fn], "seq_ordering": "random"},
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainTestModel,
            train_step=_train_step,
            batch_size=1000,
            max_seqs=1,
            optimizer={"class": "adam"},
            learning_rate=0.01,
            num_epochs=1,
            torch_dataloader_opts={"num_workers": 1},
            torch_distributed={"backend": "gloo", "dataset_distribution": "shard"},
        )
    )
    with global_config_ctx(config):
        # as returnn.__main__.init_data
        dataset, _ = load_data(config, 0, "train", **Dataset.get_default_kwargs_train(config=config))
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()
    with open(f"{tmp_dir}/rank{rank}.txt", "w") as f:
        f.write(repr(seq_tags))
    engine._torch_distributed_ctx.destroy_process_groups()


def test_torch_engine_distributed_cpu_dataset_distribution_shard():
    # dataset_distribution "shard": the ranks together train on every seq once per epoch, with disjoint shards,
    # also with the DataLoader worker (the dataset is sharded in the main proc, then pickled into the worker).
    import socket
    import torch.multiprocessing
    from test_HDFDataset import generate_hdf_from_other
    from test_Dataset import dummy_iter_dataset

    world_size = 2
    hdf_fn = generate_hdf_from_other({"class": "Task12AXDataset", "num_seqs": 22})
    all_tags = sorted(
        seq.seq_tag for seq in dummy_iter_dataset(init_dataset({"class": "HDFDataset", "files": [hdf_fn]}))
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="returnn_test_torch_distributed_shard") as tmp_dir:
        torch.multiprocessing.spawn(
            _torch_distributed_shard_worker, args=(world_size, port, tmp_dir, hdf_fn), nprocs=world_size
        )
        seq_tags = []
        for rank in range(world_size):
            with open(f"{tmp_dir}/rank{rank}.txt") as f:
                seq_tags.append(eval(f.read()))
    assert [len(tags) for tags in seq_tags] == [11, 11], seq_tags
    assert sorted(seq_tags[0] + seq_tags[1]) == all_tags, seq_tags


def _torch_distributed_lr_worker(rank: int, world_size: int, port: int, tmp_dir: str, opts: Dict[str, Any]):
    device = opts.get("device", "cpu")
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        # on CUDA: both ranks on one GPU, gloo also for the CUDA tensors
        LOCAL_RANK=str(rank) if device == "cpu" else "0",
        LOCAL_WORLD_SIZE=str(world_size) if device == "cpu" else "1",
    )
    applied_lrs = {}  # global train step -> LR of the update in that step

    def _dyn_lr(*, global_train_step: int, epoch_continuous: float, learning_rate: float, **_kwargs) -> float:
        lr = learning_rate * (1.0 + epoch_continuous)
        applied_lrs[global_train_step] = lr  # a repeated call for the same step overwrites
        return lr

    def _train(num_epochs: int) -> Engine:
        config = Config(
            dict(
                task="train",
                device=device,
                extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
                get_model=TrainTestModel,
                train_step=TrainTestModel.train_step,
                batch_size=300,
                optimizer={"class": "adamw", "capturable": device != "cpu"},
                learning_rate=0.01,
                dynamic_learning_rate=_dyn_lr,
                accum_grad_multiple_step=opts["accum_grad_multiple_step"],
                num_epochs=num_epochs,
                model=f"{tmp_dir}/model",
                learning_rate_file=f"{tmp_dir}/learning_rates",
                eval_datasets={"dev": {"class": "Task12AXDataset", "num_seqs": 10}},
                torch_dataloader_opts={"num_workers": 0},
                torch_distributed={
                    "backend": "gloo",
                    "reduce_type": opts["reduce_type"],
                    "sync_complete_frac": opts["sync_complete_frac"],
                },
                **({"torch_optimizer_step": {}} if device != "cpu" else {}),
            )
        )
        with global_config_ctx(config):
            # no fixed seed: each rank its own seq order and seq lens, thus its own complete_frac per step
            dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 50, "name": "train"})
            dataset.init_seq_order(epoch=1)
            engine = Engine(config=config)
            engine.init_train_from_config(train_data=dataset)
            engine.train()
        return engine

    _train(num_epochs=2)
    torch.distributed.barrier()  # rank 0 wrote the checkpoint
    engine = _train(num_epochs=3)  # continues from the epoch 2 checkpoint
    assert engine.epoch == 3
    params = [param.detach().cpu() for param in engine._pt_model.parameters()]
    torch.save({"lrs": applied_lrs, "params": params}, f"{tmp_dir}/rank{rank}.pt")
    engine._torch_distributed_ctx.destroy_process_groups()


def _run_torch_distributed_lr(**opts) -> List[Dict[str, Any]]:
    import socket
    import torch.multiprocessing

    world_size = 2
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="returnn_test_torch_distributed_lr") as tmp_dir:
        torch.multiprocessing.spawn(
            _torch_distributed_lr_worker, args=(world_size, port, tmp_dir, opts), nprocs=world_size
        )
        return [torch.load(f"{tmp_dir}/rank{rank}.pt") for rank in range(world_size)]


def _check_torch_distributed_lr(**opts):
    for sync_complete_frac in [False, True]:
        res = _run_torch_distributed_lr(sync_complete_frac=sync_complete_frac, **opts)
        lrs0, lrs1 = res[0]["lrs"], res[1]["lrs"]
        assert sorted(lrs0) == sorted(lrs1), (opts, lrs0, lrs1)
        params_equal = all(torch.equal(p0, p1) for p0, p1 in zip(res[0]["params"], res[1]["params"]))
        print(f"{opts}, sync_complete_frac {sync_complete_frac}: LRs equal {lrs0 == lrs1}, params equal {params_equal}")
        if sync_complete_frac:
            assert lrs0 == lrs1, (opts, lrs0, lrs1)
            assert params_equal, opts
        else:  # rank-local progress: the setting really triggers the divergence
            assert lrs0 != lrs1, opts
            assert not params_equal, opts


def test_torch_engine_distributed_cpu_sync_complete_frac_ddp():
    _check_torch_distributed_lr(reduce_type="grad", accum_grad_multiple_step=1)


def test_torch_engine_distributed_cpu_sync_complete_frac_grad_explicit_accum():
    _check_torch_distributed_lr(reduce_type="grad_explicit", accum_grad_multiple_step=2)


def test_torch_engine_distributed_sync_complete_frac_optimizer_step():
    """captured optimizer step (torch_optimizer_step), its LR is a device tensor updated in place"""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    _check_torch_distributed_lr(device="cuda", reduce_type="grad_explicit", accum_grad_multiple_step=1)


def test_torch_distributed_sync_levels_spec():
    from returnn.torch.distributed import _make_group_ranks, _parse_sync_levels

    assert _make_group_ranks("world", size=4, hostnames=None) == [[0, 1, 2, 3]]
    assert _make_group_ranks(2, size=4, hostnames=None) == [[0, 1], [2, 3]]
    assert _make_group_ranks([[3, 0], [1, 2]], size=4, hostnames=None) == [[0, 3], [1, 2]]
    assert _make_group_ranks("node", size=4, hostnames=["a", "b", "a", "b"]) == [[0, 2], [1, 3]]
    for bad in [3, [[0, 1], [2]], [[0, 1], [1, 2, 3]], "foo", 2.0]:
        try:
            _make_group_ranks(bad, size=4, hostnames=None)
        except (AssertionError, TypeError) as exc:
            print("got expected exception:", exc)
        else:
            raise Exception(f"did not get expected exception for group {bad!r}")

    levels = _parse_sync_levels(
        [
            {"group": 2, "type": "grad"},
            {"group": 4, "type": "param", "every": 10},
            {"group": "world", "type": "param", "every": 20},
        ],
        size=8,
        local_size=4,
        hostnames=None,
    )
    assert [(level.sync_type, level.every, level.group_ranks) for level in levels] == [
        ("grad", 1, [[0, 1], [2, 3], [4, 5], [6, 7]]),
        ("param", 10, [[0, 1, 2, 3], [4, 5, 6, 7]]),
        ("param", 20, [[0, 1, 2, 3, 4, 5, 6, 7]]),
    ]
    assert [level.fires(epoch_update_step_idx=step) for level in levels for step in [0, 9, 10, 19]] == [
        True, True, True, True,  # grad
        False, True, False, True,  # param every 10
        False, False, False, True,  # param every 20
    ]  # fmt: skip
    from returnn.torch.distributed import _control_sync_chain

    # the control syncs a stopped group joins: the outermost level firing at each next outer firing.
    # A stop at the pair with update 19 pending: quad and world both fire after 19, so only the world check follows.
    assert [level.group_spec for level in _control_sync_chain(levels, 0, 19)] == [2, "world"]
    assert [level.group_spec for level in _control_sync_chain(levels, 0, 3)] == [2, 4, "world"]  # quad 9, world 19
    assert [level.group_spec for level in _control_sync_chain(levels, 1, 9)] == [4, "world"]
    assert [level.group_spec for level in _control_sync_chain(levels, 2, 19)] == ["world"]

    hostnames = ["a", "a", "b", "b"]
    (level,) = _parse_sync_levels([{"group": "node", "type": "param"}], size=4, local_size=2, hostnames=hostnames)
    assert level.group_ranks == [[0, 1], [2, 3]]
    for bad, kwargs in [
        ([{"group": "world", "type": "param", "every": 10}, {"group": "world", "type": "grad"}], {}),  # grad not first
        ([{"group": "world", "type": "grad", "every": 2}], {}),
        ([{"group": "world", "type": "param", "every": 3}, {"group": "world", "type": "param", "every": 4}], {}),
        ([{"group": "world", "type": "param", "every": 0}], {}),
        ([{"group": "world", "type": "foo"}], {}),
        ([{"group": "world", "type": "param", "foo": 1}], {}),
        ([{"group": "world"}], {}),
        ({"group": "world", "type": "param"}, {}),
        (
            [{"group": [[0, 1], [2, 3]], "type": "param"}, {"group": [[0, 2], [1, 3]], "type": "param"}],
            {},
        ),  # not nested
        ([{"group": "node", "type": "param"}], {"local_size": 1, "hostnames": hostnames}),  # local size vs hosts
        ([{"group": "node", "type": "param"}], {"local_size": 2, "hostnames": None}),  # hostnames not gathered
    ]:
        try:
            _parse_sync_levels(bad, **{"size": 4, "local_size": 2, "hostnames": None, **kwargs})
        except AssertionError as exc:
            print("got expected exception:", exc)
        else:
            raise Exception(f"did not get expected exception for sync {bad!r}")


def _torch_distributed_sync_levels_worker(rank: int, world_size: int, port: int, tmp_dir: str, variant: str):
    from returnn.torch.distributed import DistributedContext

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world_size),
    )
    module = torch.nn.Linear(3, 2)
    with torch.no_grad():
        for i, param in enumerate(module.parameters()):
            param.fill_(float(rank + 10 * i))  # rank-specific, so every average is distinguishable
            param.grad = torch.full_like(param, float(100 * rank + i))
    optimizer = torch.optim.Adam(module.parameters(), lr=0.01)

    def _values() -> List[List[float]]:
        return [[float(param.detach().flatten()[0]), float(param.grad.flatten()[0])] for param in module.parameters()]

    results = {}
    if variant == "grad_pairs":
        # grads in pairs every step, params over all ranks every 2 steps
        ctx = DistributedContext(
            {"backend": "gloo", "sync": [{"group": 2, "type": "grad"}, {"group": "world", "type": "param", "every": 2}]}
        )
        assert ctx.has_grad_sync() and ctx.maybe_make_distributed_module(torch.nn.Linear(1, 1)) is None
        pair = [rank - rank % 2, rank - rank % 2 + 1]
        assert torch.distributed.get_process_group_ranks(ctx.grad_sync_group()) == pair
        # the grad level: control sync on its group every step, then the world level to join when out of data
        levels = ctx.control_sync_levels(epoch_update_step_idx=0, update_step=False, world=False, no_data=False)
        assert [level.group_spec for level in levels] == [2, "world"], levels
        # out of data mid accumulation while the world fires after the pending update: the pair check comes first,
        # as the partner does it in this step, the world check after
        levels = ctx.control_sync_levels(epoch_update_step_idx=1, update_step=False, world=False, no_data=True)
        assert [level.group_spec for level in levels] == [2, "world"], levels
        ctx.maybe_reduce_grads(module=module)
        results["step0_grads"] = _values()
        ctx.step_after_param_update(module=module, optimizer=optimizer, epoch_update_step_idx=0)
        results["step0"] = _values()
        ctx.step_after_param_update(module=module, optimizer=optimizer, epoch_update_step_idx=1)
        results["step1"] = _values()
    elif variant == "hier":
        # params in pairs every 2 steps, params over the node (= all ranks here) every 4 steps;
        # the pairs are given in a different rank order, which must not matter
        ctx = DistributedContext(
            {
                "backend": "gloo",
                "sync": [
                    {"group": [[2, 0], [1, 3]], "type": "param", "every": 2, "sync_on_cpu": True},
                    {"group": "node", "type": "param", "every": 4},
                ],
            }
        )
        assert not ctx.has_grad_sync()
        try:
            ctx.grad_sync_group()
        except ValueError:
            pass
        else:
            raise AssertionError("no grad sync, expected ValueError")
        # pairs every 2 updates, node (= world here) every 4: the control sync on the outermost firing level
        assert [
            [level.group_spec for level in ctx.control_sync_levels(
                epoch_update_step_idx=step, update_step=True, world=False, no_data=False
            )]
            for step in range(4)
        ] == [[], [[[2, 0], [1, 3]], "node"], [], ["node"]]  # fmt: skip
        # mid grad accumulation: nothing; out of data: the next control sync of the pair, then the node to join
        assert not ctx.control_sync_levels(epoch_update_step_idx=1, update_step=False, world=False, no_data=False)
        assert [
            level.group_spec
            for level in ctx.control_sync_levels(epoch_update_step_idx=0, update_step=False, world=False, no_data=True)
        ] == [[[2, 0], [1, 3]], "node"]
        # out of data in the step completing update 2, which fires nothing: the next is the node after update 3
        assert [
            level.group_spec
            for level in ctx.control_sync_levels(epoch_update_step_idx=2, update_step=True, world=False, no_data=True)
        ] == ["node"]
        assert [
            level.group_spec
            for level in ctx.control_sync_levels(epoch_update_step_idx=0, update_step=False, world=True, no_data=False)
        ] == ["world"]
        ctx.maybe_reduce_grads(module=module)  # no-op
        ctx.step_after_param_update(module=module, optimizer=optimizer, epoch_update_step_idx=0)
        results["step0"] = _values()
        ctx.step_after_param_update(module=module, optimizer=optimizer, epoch_update_step_idx=1)
        results["step1"] = _values()
        ctx.step_after_param_update(module=module, optimizer=optimizer, epoch_update_step_idx=3)
        results["step3"] = _values()
    elif variant == "hier_opt":
        # params and optimizer state in pairs every 2 steps, params only over all ranks every 4 steps:
        # in step 3 the pair still averages the optimizer state
        ctx = DistributedContext(
            {
                "backend": "gloo",
                "sync": [
                    {"group": 2, "type": "param", "every": 2, "sync_optimizer_state": True},
                    {"group": "world", "type": "param", "every": 4},
                ],
            }
        )

        def _set_opt_state(value: float):
            for param in module.parameters():
                optimizer.state[param] = {"exp_avg": torch.full_like(param, value)}

        def _opt_values() -> List[float]:
            return [float(optimizer.state[param]["exp_avg"].flatten()[0]) for param in module.parameters()]

        _set_opt_state(float(rank))
        ctx.step_after_param_update(module=module, optimizer=optimizer, epoch_update_step_idx=1)
        results["step1"] = _values()
        results["step1_opt"] = _opt_values()
        _set_opt_state(float(10 * rank))
        ctx.step_after_param_update(module=module, optimizer=optimizer, epoch_update_step_idx=3)
        results["step3"] = _values()
        results["step3_opt"] = _opt_values()
    else:
        raise ValueError(f"invalid variant {variant!r}")

    torch.save(results, f"{tmp_dir}/rank{rank}.pt")
    ctx.destroy_process_groups()


def _run_torch_distributed_sync_levels(variant: str) -> List[Dict[str, List[Any]]]:
    import socket
    import torch.multiprocessing

    world_size = 4
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="returnn_test_torch_distributed_sync_levels") as tmp_dir:
        torch.multiprocessing.spawn(
            _torch_distributed_sync_levels_worker, args=(world_size, port, tmp_dir, variant), nprocs=world_size
        )
        return [torch.load(f"{tmp_dir}/rank{rank}.pt") for rank in range(world_size)]


def _check_torch_distributed_sync_levels(variant: str, expected: Dict[str, List[Any]]):
    res = _run_torch_distributed_sync_levels(variant)
    for key, expected_values in expected.items():
        got = [res[rank][key] for rank in range(len(res))]
        assert got == expected_values, (key, got, expected_values)


def test_torch_distributed_sync_levels_cpu_grad_pairs():
    # per rank: [param, grad] of (weight, bias). Params start at rank + 10 * i, grads at 100 * rank + i.
    unsynced = [[[0, 50], [10, 51]], [[1, 50], [11, 51]], [[2, 250], [12, 251]], [[3, 250], [13, 251]]]
    _check_torch_distributed_sync_levels(
        "grad_pairs",
        {
            "step0_grads": unsynced,
            "step0": unsynced,
            "step1": [
                [[1.5, 50], [11.5, 51]],
                [[1.5, 50], [11.5, 51]],
                [[1.5, 250], [11.5, 251]],
                [[1.5, 250], [11.5, 251]],
            ],
        },
    )


def test_torch_distributed_sync_levels_cpu_hier():
    _check_torch_distributed_sync_levels(
        "hier",
        {
            "step0": [[[0, 0], [10, 1]], [[1, 100], [11, 101]], [[2, 200], [12, 201]], [[3, 300], [13, 301]]],
            "step1": [[[1, 0], [11, 1]], [[2, 100], [12, 101]], [[1, 200], [11, 201]], [[2, 300], [12, 301]]],
            "step3": [
                [[1.5, 0], [11.5, 1]],
                [[1.5, 100], [11.5, 101]],
                [[1.5, 200], [11.5, 201]],
                [[1.5, 300], [11.5, 301]],
            ],
        },
    )


def test_torch_distributed_sync_levels_cpu_hier_opt():
    _check_torch_distributed_sync_levels(
        "hier_opt",
        {
            "step1": [
                [[0.5, 0], [10.5, 1]],
                [[0.5, 100], [10.5, 101]],
                [[2.5, 200], [12.5, 201]],
                [[2.5, 300], [12.5, 301]],
            ],
            "step1_opt": [[0.5, 0.5], [0.5, 0.5], [2.5, 2.5], [2.5, 2.5]],
            "step3": [
                [[1.5, 0], [11.5, 1]],
                [[1.5, 100], [11.5, 101]],
                [[1.5, 200], [11.5, 201]],
                [[1.5, 300], [11.5, 301]],
            ],
            "step3_opt": [[5, 5], [5, 5], [25, 25], [25, 25]],  # pair means, not the per-rank 10 * rank
        },
    )


def _torch_spawn_with_timeout(fn, *, args: Tuple[Any, ...], nprocs: int, timeout: float):
    import torch.multiprocessing

    ctx = torch.multiprocessing.spawn(fn, args=args, nprocs=nprocs, join=False)
    deadline = time.monotonic() + timeout
    while not ctx.join(timeout=max(0.0, deadline - time.monotonic())):  # returns once any process finished
        if time.monotonic() > deadline:
            for proc in ctx.processes:
                proc.kill()
            raise Exception("timeout, the ranks hang")


def _torch_distributed_destroy_process_groups_worker(rank: int, world_size: int, port: int):
    from returnn.torch.distributed import DistributedContext

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world_size),
    )
    # One core per rank: the gloo worker thread only runs while the main thread waits, as on a loaded machine.
    os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[rank]})
    # explicit ranks: a group of its own (new_group), not the default group
    ctx = DistributedContext({"backend": "gloo", "sync": [{"group": [[0, 1]], "type": "param"}]})
    module = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(module.parameters())
    ctx.step_after_param_update(module=module, optimizer=optimizer, epoch_update_step_idx=0)
    ctx.destroy_process_groups()  # right after the collective


@unittest.skipIf(not hasattr(os, "sched_setaffinity"), "Linux only")
def test_torch_distributed_destroy_process_groups_after_sync_cpu():
    """
    Freeing a gloo process group right after a collective on it deadlocked with torch 2.0 to 2.7,
    see :func:`returnn.torch.distributed._destroy_process_groups`.
    """
    import socket

    world_size = 2
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    _torch_spawn_with_timeout(
        _torch_distributed_destroy_process_groups_worker, args=(world_size, port), nprocs=world_size, timeout=60
    )


def _torch_distributed_param_sync_accum_worker(rank: int, world_size: int, port: int):
    import returnn.torch.distributed as torch_distributed
    from returnn.torch.updater import Updater

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world_size),
    )
    # The param sync must come right after an optimizer update, never between accumulated grads,
    # and its period counts updates per epoch.
    state = {"last": None, "updates": 0, "updates_at_epoch_start": 0, "syncs": []}
    orig_run_step = Engine._run_step
    orig_train_epoch = Engine.train_epoch
    orig_updater_step = Updater.step
    orig_sync_state = torch_distributed._sync_state

    def _run_step(self, *args, train_flag: bool = False, **kwargs):
        if train_flag:
            state["last"] = "step"
        return orig_run_step(self, *args, train_flag=train_flag, **kwargs)

    def _train_epoch(self):
        state["updates_at_epoch_start"] = state["updates"]
        return orig_train_epoch(self)

    def _updater_step(self, **kwargs):
        state["last"] = "update"
        state["updates"] += 1
        return orig_updater_step(self, **kwargs)

    def _sync_state(**kwargs):
        if kwargs.get("counters"):  # the epoch-end sync (sync_at_epoch_end), not a param sync
            return orig_sync_state(**kwargs)
        assert state["last"] == "update", state
        updates_in_epoch = state["updates"] - state["updates_at_epoch_start"]
        assert updates_in_epoch > 0 and updates_in_epoch % 3 == 0, state
        state["syncs"].append(updates_in_epoch)
        state["last"] = "sync"
        return orig_sync_state(**kwargs)

    Engine._run_step = _run_step
    Engine.train_epoch = _train_epoch
    Updater.step = _updater_step
    torch_distributed._sync_state = _sync_state

    config = Config(
        dict(
            task="train",
            device="cpu",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainTestModel,
            train_step=TrainTestModel.train_step,
            batch_size=100,
            optimizer={"class": "adam"},
            learning_rate=0.01,
            accum_grad_multiple_step=2,
            num_epochs=2,
            torch_dataloader_opts={"num_workers": 0},
            torch_distributed={"backend": "gloo", "reduce_type": "param", "param_sync_step": 3},
        )
    )
    with global_config_ctx(config):
        dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 50})
        dataset.init_seq_order(epoch=1)
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()
    assert engine.epoch == 2
    assert len(state["syncs"]) >= 2, state  # at least one sync per epoch, so the epoch boundary reset is covered
    engine._torch_distributed_ctx.destroy_process_groups()


def test_torch_engine_distributed_param_sync_accum_cpu():
    import socket
    import torch.multiprocessing

    world_size = 2
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    torch.multiprocessing.spawn(_torch_distributed_param_sync_accum_worker, args=(world_size, port), nprocs=world_size)


def _torch_distributed_param_sync_accum_dyn_worker(rank: int, world_size: int, port: int):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world_size),
    )

    def _accum(*, global_train_step: int, **_kwargs) -> int:
        # Rank 0 runs out of data in step 4 (one seq per batch, 4 seqs),
        # while rank 1 finishes an accumulation there under the new target 2, which the old target 3 did not predict.
        return 2 if global_train_step == 4 else 3

    config = Config(
        dict(
            task="train",
            device="cpu",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainTestModel,
            train_step=TrainTestModel.train_step,
            batch_size=10**9,
            max_seqs=1,
            optimizer={"class": "adam"},
            learning_rate=0.01,
            accum_grad_multiple_step=_accum,
            num_epochs=1,
            torch_dataloader_opts={"num_workers": 0},
            torch_distributed={"backend": "gloo", "reduce_type": "param", "param_sync_step": 1},
        )
    )
    with global_config_ctx(config):
        dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 4 if rank == 0 else 10})
        dataset.init_seq_order(epoch=1)
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()
    assert engine.global_train_step == 4, engine.global_train_step  # both stop when rank 0 runs out
    engine._torch_distributed_ctx.destroy_process_groups()


def _torch_distributed_hier_control_sync_worker(
    rank: int, world_size: int, port: int, tmp_dir: str, accum_grad_multiple_step: int, param_sync_every: int
):
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
            batch_size=10**9,
            max_seqs=1,  # one seq per step
            optimizer={"class": "adam"},
            learning_rate=0.01,
            accum_grad_multiple_step=accum_grad_multiple_step,
            num_epochs=1,
            torch_dataloader_opts={"num_workers": 0},
            torch_distributed={
                "backend": "gloo",
                # grads within the pair every step, params across the pairs every param_sync_every updates
                "sync": [
                    {"group": 2, "type": "grad"},
                    {"group": "world", "type": "param", "every": param_sync_every},
                ],
                "sync_at_epoch_end": False,  # keeps the per-rank step counts, which the test reads
            },
        )
    )
    with global_config_ctx(config):
        # Rank 0 runs out in step 2, so its pair stops there and joins the next world control sync,
        # which the other pair reaches in step 3: without accumulation, the world level fires after update 3;
        # with accumulation 2, step 2 is mid accumulation and the world level fires after update 1, in step 3.
        dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 2 if rank == 0 else 10})
        dataset.init_seq_order(epoch=1)
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()
    with open(f"{tmp_dir}/rank{rank}.txt", "w") as f:
        f.write(str(engine.global_train_step))
    engine._torch_distributed_ctx.destroy_process_groups()


def _check_torch_distributed_hier_control_sync(*, accum_grad_multiple_step: int, param_sync_every: int):
    import socket

    world_size = 4
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="returnn_test_torch_distributed_hier_control_sync") as tmp_dir:
        _torch_spawn_with_timeout(
            _torch_distributed_hier_control_sync_worker,
            args=(world_size, port, tmp_dir, accum_grad_multiple_step, param_sync_every),
            nprocs=world_size,
            timeout=120,
        )
        steps = []
        for rank in range(world_size):
            with open(f"{tmp_dir}/rank{rank}.txt") as f:
                steps.append(int(f.read()))
    assert steps == [2, 2, 3, 3], steps


def test_torch_engine_distributed_hier_control_sync_cpu():
    """
    The control sync follows the levels: the pair of rank 0 stops when rank 0 runs out,
    the other pair only at its next world sync.
    """
    _check_torch_distributed_hier_control_sync(accum_grad_multiple_step=1, param_sync_every=4)


def test_torch_engine_distributed_hier_control_sync_accum_cpu():
    """
    Rank 0 runs out mid accumulation while the world level fires once the pending update completes:
    it must still do the pair check its partner does in this step, not the world check.
    """
    _check_torch_distributed_hier_control_sync(accum_grad_multiple_step=2, param_sync_every=1)


def test_torch_engine_distributed_param_sync_accum_dyn_cpu():
    """
    Param sync with a callable accum_grad_multiple_step and uneven data:
    the has-data check must not be skipped in the step where the callable makes an update due, else the ranks hang.
    """
    import socket
    import torch.multiprocessing

    world_size = 2
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = torch.multiprocessing.spawn(
        _torch_distributed_param_sync_accum_dyn_worker, args=(world_size, port), nprocs=world_size, join=False
    )
    deadline = time.monotonic() + 120
    while not ctx.join(timeout=max(0.0, deadline - time.monotonic())):  # returns once any process finished
        if time.monotonic() > deadline:
            for proc in ctx.processes:
                proc.kill()
            raise Exception("timeout, the ranks hang")


def _torch_distributed_param_sync_optimizer_state_worker(
    rank: int, world_size: int, port: int, tmp_dir: str, sync_optimizer_state: bool
):
    from returnn.torch.distributed import DistributedContext

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world_size),
    )
    module = torch.nn.Linear(3, 2)
    with torch.no_grad():
        for i, param in enumerate(module.parameters()):
            param.fill_(float(rank + 10 * i))  # rank-specific, so every average is distinguishable
            param.grad = torch.full_like(param, float(100 * rank + i))
    optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
    optimizer.step()  # exp_avg = 0.1 * grad, exp_avg_sq = 0.001 * grad^2
    if rank == 1:
        optimizer.step()  # an extra update on one rank, as after an epoch with more batches on it
    # nested state with a Python value, as a custom optimizer might keep it: taken from rank 0
    optimizer.state[next(module.parameters())]["extra"] = {"nested": [f"rank {rank}"]}
    ctx = DistributedContext(
        {
            "backend": "gloo",
            "reduce_type": "param",
            "param_sync_step": 1,
            "sync_optimizer_state": sync_optimizer_state,
        }
    )
    assert [
        level.group_spec
        for level in ctx.control_sync_levels(epoch_update_step_idx=0, update_step=True, world=False, no_data=False)
    ] == ["world"]
    ctx.step_after_param_update(module=module, optimizer=optimizer, epoch_update_step_idx=0)
    results = {
        "params": [float(param.detach().flatten()[0]) for param in module.parameters()],
        "state": [
            [float(optimizer.state[param][key].flatten()[0]) for key in ["exp_avg", "exp_avg_sq"]]
            for param in module.parameters()
        ],
        "steps": [float(optimizer.state[param]["step"]) for param in module.parameters()],
        "extra": optimizer.state[next(module.parameters())]["extra"],
    }
    torch.save(results, f"{tmp_dir}/rank{rank}.pt")
    ctx.destroy_process_groups()


def _check_torch_distributed_param_sync_optimizer_state(sync_optimizer_state: bool):
    import socket
    import torch.multiprocessing

    world_size = 4
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="returnn_test_torch_distributed_param_sync_optimizer_state") as tmp_dir:
        torch.multiprocessing.spawn(
            _torch_distributed_param_sync_optimizer_state_worker,
            args=(world_size, port, tmp_dir, sync_optimizer_state),
            nprocs=world_size,
        )
        res = [torch.load(f"{tmp_dir}/rank{rank}.pt") for rank in range(world_size)]
    # Adam with a constant grad g, after n steps: exp_avg = g * (1 - 0.9^n), exp_avg_sq = g^2 * (1 - 0.999^n).
    # Grads 100 * rank + i, n = 2 on rank 1, 1 elsewhere.
    own_state = [
        [[g * (1 - 0.9**n), g**2 * (1 - 0.999**n)] for g in [100.0 * rank + i for i in range(2)]]
        for rank, n in enumerate([1, 2, 1, 1])
    ]
    for rank, res_rank in enumerate(res):
        assert res_rank["params"] == res[0]["params"], (rank, res_rank["params"], res[0]["params"])  # params synced
        print(f"sync_optimizer_state {sync_optimizer_state}, rank {rank}: optimizer state {res_rank['state']}")
    if sync_optimizer_state:
        mean_state = numpy.mean(own_state, axis=0)
        for rank, res_rank in enumerate(res):
            assert res_rank["state"] == res[0]["state"], (rank, res_rank["state"])
            numpy.testing.assert_allclose(res_rank["state"], mean_state, rtol=1e-5)
            assert res_rank["steps"] == [2.0, 2.0], (rank, res_rank["steps"])  # the max, not the mean 1.25
            assert res_rank["extra"] == {"nested": ["rank 0"]}, (rank, res_rank["extra"])
    else:  # each rank keeps its own
        for rank, res_rank in enumerate(res):
            numpy.testing.assert_allclose(res_rank["state"], own_state[rank], rtol=1e-5)
            assert res_rank["steps"] == [float(n) for n in [[1, 2, 1, 1][rank]] * 2], (rank, res_rank["steps"])
            assert res_rank["extra"] == {"nested": [f"rank {rank}"]}, (rank, res_rank["extra"])


def test_torch_distributed_param_sync_optimizer_state_cpu():
    _check_torch_distributed_param_sync_optimizer_state(sync_optimizer_state=True)


def test_torch_distributed_param_sync_no_optimizer_state_cpu():
    _check_torch_distributed_param_sync_optimizer_state(sync_optimizer_state=False)


def _torch_distributed_sync_at_epoch_end_worker(
    rank: int, world_size: int, port: int, tmp_dir: str, sync_at_epoch_end: bool
):
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
            batch_size=10**9,
            max_seqs=1,  # one seq per step: 7 / 9 steps per epoch, the param sync every 3 steps leaves local steps
            optimizer={"class": "adam"},
            learning_rate=0.01,
            num_epochs=2,
            model=f"{tmp_dir}/model",
            torch_dataloader_opts={"num_workers": 0},
            torch_distributed={
                "backend": "gloo",
                "reduce_type": "param",
                "param_sync_step": 3,
                "sync_at_epoch_end": sync_at_epoch_end,
            },
        )
    )
    with global_config_ctx(config):
        # no fixed seed: each rank its own seq order, thus its own params and optimizer state between the syncs.
        # Rank 0 runs out after 7 steps, rank 1 then stops at its next sync, after 8 steps: different step counts.
        dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 7 if rank == 0 else 9})
        dataset.init_seq_order(epoch=1)
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        # buffers as a batch norm has them: running stats (float, the mean) and a count (int, the max)
        engine._pt_model.register_buffer("stat", torch.full((2,), float(rank)))
        engine._pt_model.register_buffer("count", torch.tensor(rank, dtype=torch.int64))
        # nested optimizer state as a custom optimizer might keep it: a tensor (mean) and a Python value (from rank 0),
        # added before epoch 2, when the lazy Adam state exists (Adam creates it only when the state is empty)
        optimizer = engine._updater.get_optimizer()
        orig_train_epoch = Engine.train_epoch

        def _train_epoch(self):
            if self.epoch == 2:
                optimizer.state[next(self._pt_model.parameters())]["extra"] = {
                    "nested": [torch.full((3,), float(rank)), {"note": f"rank {rank}"}]
                }
            return orig_train_epoch(self)

        Engine.train_epoch = _train_epoch
        engine.train()
        params = [param.detach().clone() for param in engine._pt_model.parameters()]
        state = [
            {key: value.clone() for key, value in optimizer.state[param].items() if isinstance(value, torch.Tensor)}
            for param in engine._pt_model.parameters()
        ]
        extra = optimizer.state[next(engine._pt_model.parameters())]["extra"]
        buffers = {name: buf.clone() for name, buf in engine._pt_model.named_buffers()}
    torch.save(
        {
            "params": params,
            "state": state,
            "extra": extra,
            "buffers": buffers,
            "global_train_step": engine.global_train_step,
        },
        f"{tmp_dir}/rank{rank}.pt",
    )
    engine._torch_distributed_ctx.destroy_process_groups()


def _check_torch_distributed_sync_at_epoch_end(sync_at_epoch_end: bool):
    import socket
    import torch.multiprocessing

    world_size = 2
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="returnn_test_torch_distributed_sync_at_epoch_end") as tmp_dir:
        torch.multiprocessing.spawn(
            _torch_distributed_sync_at_epoch_end_worker,
            args=(world_size, port, tmp_dir, sync_at_epoch_end),
            nprocs=world_size,
        )
        res = [torch.load(f"{tmp_dir}/rank{rank}.pt") for rank in range(world_size)]
        saved_opt = torch.load(f"{tmp_dir}/model.002.opt.pt")  # written by rank 0
    params_equal = all(torch.equal(p0, p1) for p0, p1 in zip(res[0]["params"], res[1]["params"]))
    state_keys = sorted(res[0]["state"][0].keys())
    assert "exp_avg" in state_keys and "exp_avg_sq" in state_keys, state_keys
    state_equal = all(
        torch.equal(s0[key], s1[key]) for s0, s1 in zip(res[0]["state"], res[1]["state"]) for key in state_keys
    )
    steps = [res[rank]["global_train_step"] for rank in range(world_size)]
    print(
        f"sync_at_epoch_end {sync_at_epoch_end}: params equal {params_equal}, optimizer state equal {state_equal},"
        f" global train steps {steps}"
    )
    assert params_equal == sync_at_epoch_end and state_equal == sync_at_epoch_end
    # the max over the ranks after each of the 2 epochs: 8 + 8; without the sync, 7 + 7 and 8 + 8
    assert steps == ([16, 16] if sync_at_epoch_end else [14, 16]), steps
    for rank in range(world_size):
        buffers = res[rank]["buffers"]
        assert torch.equal(buffers["stat"], torch.full((2,), 0.5 if sync_at_epoch_end else float(rank))), (
            rank,
            buffers,
        )
        assert int(buffers["count"]) == (1 if sync_at_epoch_end else rank), (rank, buffers)
        extra = res[rank]["extra"]["nested"]
        assert torch.equal(extra[0], torch.full((3,), 0.5 if sync_at_epoch_end else float(rank))), (rank, extra)
        assert extra[1] == {"note": "rank 0" if sync_at_epoch_end else f"rank {rank}"}, (rank, extra)
    assert all(
        float(s["step"]) == (16 if sync_at_epoch_end else steps[rank])
        for rank in range(world_size)
        for s in res[rank]["state"]
    )
    # The checkpoint is rank 0's state. With the sync it is thus also what rank 1 continues with.
    saved_state = saved_opt["state"] if "state" in saved_opt else saved_opt["optimizer"]["state"]
    for i, s0 in enumerate(res[0]["state"]):
        for key in state_keys:
            assert torch.equal(saved_state[i][key], s0[key]), (i, key)


def test_torch_engine_distributed_param_sync_at_epoch_end_cpu():
    _check_torch_distributed_sync_at_epoch_end(sync_at_epoch_end=True)


def test_torch_engine_distributed_param_no_sync_at_epoch_end_cpu():
    _check_torch_distributed_sync_at_epoch_end(sync_at_epoch_end=False)


def _torch_distributed_all_reduce_avg_flat_worker(rank: int, world_size: int, port: int, tmp_dir: str):
    from returnn.torch.distributed import _all_reduce_avg_flat, _destroy_process_groups, _FlatReducer

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size))
    torch.distributed.init_process_group(backend="gloo")
    # 5 float32 tensors of 3, 7, 2, 11 and 5 elements; 8 bytes per bucket: a tensor of 2 fills one,
    # every other tensor is larger than a bucket and gets its own
    tensors = [torch.full((n,), float(100 * rank + i)) for i, n in enumerate([3, 7, 2, 11, 5])]
    _all_reduce_avg_flat(tensors, bucket_bytes=8)
    results = [[float(tensor[0]), float(tensor[-1])] for tensor in tensors]
    tensors = [torch.full((n,), float(100 * rank + i)) for i, n in enumerate([3, 7, 2, 11, 5])]
    _all_reduce_avg_flat(tensors, sync_on_cpu=True, bucket_bytes=8)  # on CPU tensors: the same path, no copy
    assert [[float(tensor[0]), float(tensor[-1])] for tensor in tensors] == results
    # mixed dtypes: one bucket per dtype, no promotion
    tensors = [
        torch.full((n,), float(100 * rank + i), dtype=[torch.float32, torch.float64][i % 2])
        for i, n in enumerate([3, 7, 2, 11, 5])
    ]
    _all_reduce_avg_flat(tensors, bucket_bytes=8)
    assert [[float(tensor[0]), float(tensor[-1])] for tensor in tensors] == results
    assert [tensor.dtype for tensor in tensors] == [[torch.float32, torch.float64][i % 2] for i in range(5)]
    # a kept reducer (the grad sync): the same buffer for new tensors of the same shapes, bucket layout as above
    reducer = _FlatReducer(bucket_bytes=8)
    for _ in range(2):
        tensors = [torch.full((n,), float(100 * rank + i)) for i, n in enumerate([3, 7, 2, 11, 5])]
        reducer.reduce(tensors)
        assert [[float(tensor[0]), float(tensor[-1])] for tensor in tensors] == results
        buffers = {flat.untyped_storage().data_ptr() for _, _, flat, _ in reducer._buckets}
        assert len(reducer._buckets) == 5 and len(buffers) == 1
    # in bfloat16: rounded to it (the small integers rank + i and their sums are exact there), the dtype kept
    tensors = [torch.full((n,), float(rank + i) + 1e-3) for i, n in enumerate([3, 7, 2, 11, 5])]
    _FlatReducer(dtype=torch.bfloat16).reduce(tensors)
    assert [[float(tensor[0]), float(tensor[-1])] for tensor in tensors] == [[1.0 + i, 1.0 + i] for i in range(5)]
    assert all(tensor.dtype == torch.float32 for tensor in tensors)
    with open(f"{tmp_dir}/rank{rank}.txt", "w") as f:
        f.write(repr(results))
    _destroy_process_groups([None])


class _PartlyUsedTestModel(TrainTestModel):
    """with a param only some steps use, like an audio encoder in a batch without audio"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.extra = torch.nn.Linear(9, 9)


def _torch_distributed_epoch_end_worker(rank: int, world_size: int, port: int, tmp_dir: str, opts: Dict[str, Any]):
    from returnn.log import log

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world_size),
    )
    log.initialize(logs=[f"{tmp_dir}/log.rank{rank}.txt"], verbosity=[4])
    num_seqs = [9, 41, 23, 30][rank]  # max_seqs 3: 3, 14, 8, 10 batches
    num_seen_seqs = 0
    opts = dict(opts)
    partly_used = opts.pop("partly_used", False)  # model.extra unused in the first step of each epoch
    steps_in_epoch = {}

    def _train_step(*, model: TrainTestModel, extern_data: TensorDict, **kwargs):
        nonlocal num_seen_seqs
        num_seen_seqs += extern_data["data"].raw_tensor.shape[0]
        if partly_used:
            epoch = rf.get_run_ctx().epoch
            steps_in_epoch[epoch] = steps_in_epoch.get(epoch, -1) + 1
            if steps_in_epoch[epoch] > 0:
                data = extern_data["data"]
                data.raw_tensor = data.raw_tensor + 0.01 * model.extra(data.raw_tensor)
        TrainTestModel.train_step(model=model, extern_data=extern_data, **kwargs)

    config = Config(
        dict(
            task="train",
            device="cpu",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=_PartlyUsedTestModel if partly_used else TrainTestModel,
            train_step=_train_step,
            batch_size=10**9,
            max_seqs=3,
            optimizer={"class": "adamw", "weight_decay": 0.1},
            learning_rate=0.01,
            # depends on epoch_continuous: the same on all ranks only with sync_complete_frac
            dynamic_learning_rate=lambda *, learning_rate, epoch_continuous=None, **_kw: (
                learning_rate * (1.0 + (epoch_continuous or 0.0))
            ),
            num_epochs=2,
            accum_grad_multiple_step=opts.pop("accum", 1),
            torch_dataloader_opts={"num_workers": 0},
            torch_distributed={"backend": "gloo", "sync_complete_frac": True, "sync_at_epoch_end": True, **opts},
        )
    )
    with global_config_ctx(config):
        dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": num_seqs})
        dataset.init_seq_order(epoch=1)
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        if engine._torch_distributed_ctx.reduce_type() in ("grad", "grad_explicit"):
            assert engine._torch_distributed_ctx.grad_sync_group() is None  # all ranks
        engine.train()
        param_sum = float(sum(param.detach().double().sum() for param in engine._pt_model.parameters()))
        meta_keys = sorted(k for k in engine.learning_rate_control.epoch_data[1].meta if k.startswith("epoch_end_"))
    with open(f"{tmp_dir}/rank{rank}.txt", "w") as f:
        f.write(repr((num_seen_seqs, num_seqs, param_sum, meta_keys)))
    engine._torch_distributed_ctx.destroy_process_groups()


def _check_torch_distributed_epoch_end(
    opts: Dict[str, Any],
) -> Tuple[List[Tuple[int, int, float, List[str]]], List[str]]:
    import socket

    world_size = 4
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="returnn_test_torch_distributed_epoch_end") as tmp_dir:
        _torch_spawn_with_timeout(
            _torch_distributed_epoch_end_worker, args=(world_size, port, tmp_dir, opts), nprocs=world_size, timeout=120
        )
        results, logs = [], []
        for rank in range(world_size):
            with open(f"{tmp_dir}/rank{rank}.txt") as f:
                results.append(eval(f.read()))
            with open(f"{tmp_dir}/log.rank{rank}.txt") as f:
                logs.append(f.read())
    return results, logs


def test_torch_distributed_epoch_end_join_cpu():
    """
    epoch_end "join", uneven batches per rank: every rank consumes all its batches, the params end up the same,
    and the idle steps get logged.
    """
    import re

    for opts in [
        {"reduce_type": "grad"},
        {"reduce_type": "grad", "accum": 2},
        {"reduce_type": "grad_explicit", "accum": 3},
        {"reduce_type": "grad_explicit", "grad_reduce_dtype": "bfloat16"},
        {"reduce_type": "param", "param_sync_step": 4},
        {"sync": [{"group": 2, "type": "grad"}, {"group": "world", "type": "param", "every": 2}]},
    ]:
        results, logs = _check_torch_distributed_epoch_end({**opts, "epoch_end": "join"})
        for num_seen_seqs, num_seqs, _, _ in results:
            assert num_seen_seqs == 2 * num_seqs, (opts, results)  # nothing dropped
        assert len({param_sum for _, _, param_sum, _ in results}) == 1, (opts, results)
        assert "epoch_end_idle_time_frac_mean" in results[0][3], (opts, results[0])
        for rank, (_, num_seqs, _, _) in enumerate(results):
            idle = re.findall(rf"Epoch \d+: epoch_end join: rank {rank}/4: (\d+) of (\d+) steps idle", logs[rank])
            assert len(idle) == 2 and all(int(n) - int(i) == -(-num_seqs // 3) for i, n in idle), (opts, rank, idle)
        assert len(re.findall(r"epoch_end join: idle steps per rank min/mean/max", logs[0])) == 2, (opts, logs[0])


def test_torch_distributed_epoch_end_join_partly_used_params_cpu():
    """
    epoch_end "join" with DistributedDataParallel and find_unused_parameters, and a param that the first step of
    each epoch leaves unused: the grad_join steps (past the wrapper) at the end of epoch 1 must not make the first
    reduce of epoch 2 expect it ("Encountered gradient which is undefined").
    """
    opts = {"reduce_type": "grad", "options": {"find_unused_parameters": True}, "partly_used": True}
    results, _ = _check_torch_distributed_epoch_end({**opts, "epoch_end": "join"})
    for num_seen_seqs, num_seqs, _, _ in results:
        assert num_seen_seqs == 2 * num_seqs, results  # nothing dropped, epoch 2 ran
    assert len({param_sum for _, _, param_sum, _ in results}) == 1, results


def test_torch_distributed_epoch_end_stop_cpu():
    """
    epoch_end "stop" (default): the first rank out of data ends the epoch, without epoch end log.
    """
    results, logs = _check_torch_distributed_epoch_end({"reduce_type": "grad"})
    for rank, (num_seen_seqs, _, _, meta_keys) in enumerate(results):
        assert num_seen_seqs == 2 * 9  # 3 steps per epoch, as rank 0
        assert "epoch_end" not in logs[rank] and not meta_keys, (rank, meta_keys)


def test_torch_distributed_all_reduce_avg_flat_buckets():
    import socket
    import torch.multiprocessing

    world_size = 3
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="returnn_test_torch_distributed_all_reduce_avg_flat") as tmp_dir:
        torch.multiprocessing.spawn(
            _torch_distributed_all_reduce_avg_flat_worker, args=(world_size, port, tmp_dir), nprocs=world_size
        )
        for rank in range(world_size):
            with open(f"{tmp_dir}/rank{rank}.txt") as f:
                results = eval(f.read())
            # the mean over the ranks of 100 * rank + i
            assert results == [[100.0 + i, 100.0 + i] for i in range(5)], (rank, results)


class _AllGatherSGD(torch.optim.Optimizer):
    """SGD with a collective in its step: the grads all-gathered over the ranks (all equal), their mean applied"""

    def __init__(self, params, lr: float = 0.1):
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self, closure=None):
        """update"""
        group = self.param_groups[0]
        params = [param for param in group["params"] if param.grad is not None]
        size = torch.distributed.get_world_size()
        flat = torch.cat([param.grad.flatten() for param in params])
        # the functional collective: traceable for the compiled step
        gathered = torch.ops._c10d_functional.all_gather_into_tensor(
            flat, size, torch.distributed.group.WORLD.group_name
        )
        gathered = torch.ops._c10d_functional.wait_tensor(gathered).view(size, -1).mean(dim=0)
        offset = 0
        for param in params:
            param.add_(gathered[offset : offset + param.numel()].view_as(param), alpha=-group["lr"])
            offset += param.numel()


def _torch_distributed_cuda_graph_nccl_worker(
    rank: int, world_size: int, port: int, tmp_dir: str, opts: Dict[str, Any]
):
    from returnn.datasets import init_dataset

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world_size),
    )
    opts = dict(opts)
    optimizer = opts.pop("optimizer", None)
    config, _ = _build_cuda_graph_train_config_and_dataset(compile_=False, optimizer_step=True, optimizer=optimizer)
    config.typed_dict["torch_distributed"] = {
        "reduce_type": "grad_explicit",
        "sync_complete_frac": True,
        "epoch_end": "join",
        **opts,
    }
    # fewer seqs on rank 1: it joins the last steps of each epoch idle
    num_seqs = [100, 60][rank]
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": num_seqs, "name": "train", "fixed_random_seed": 1})
    dataset.init_seq_order(epoch=1)
    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()
        engine.finalize()
    torch.save([param.detach().cpu() for param in engine._pt_model.parameters()], f"{tmp_dir}/rank{rank}.pt")
    engine._torch_distributed_ctx.destroy_process_groups()


def test_torch_distributed_cuda_graph_grad_explicit_nccl():
    """
    2 GPUs, NCCL: captured model step, captured optimizer step, grad_explicit with the kept flat reduce buffer,
    epoch_end join; also with the grad allreduce in bfloat16,
    and with a collective in the captured optimizer step (its graph released before the process group is destroyed,
    else that blocks). The ranks end with the same params.
    """
    import socket

    if torch.cuda.device_count() < 2:
        raise unittest.SkipTest("needs 2 GPUs")
    results = {}
    for name, opts in {
        "float32": {},
        "bfloat16": {"grad_reduce_dtype": "bfloat16"},
        "optimizer_collective": {"optimizer": {"class": _AllGatherSGD}},
    }.items():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with tempfile.TemporaryDirectory(prefix="returnn_test_torch_distributed_cuda_graph_nccl") as tmp_dir:
            _torch_spawn_with_timeout(
                _torch_distributed_cuda_graph_nccl_worker, args=(2, port, tmp_dir, opts), nprocs=2, timeout=300
            )
            params = [torch.load(f"{tmp_dir}/rank{rank}.pt") for rank in range(2)]
        assert all(torch.equal(p0, p1) for p0, p1 in zip(*params)), name
        assert all(p.isfinite().all() for p in params[0]), name
        results[name] = params[0]
    for p, p_bf16 in zip(results["float32"], results["bfloat16"]):
        assert not torch.equal(p, p_bf16)  # really reduced in bfloat16
        assert (p - p_bf16).norm() < 0.05 * p.norm(), ((p - p_bf16).norm(), p.norm())


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


def _check_torch_engine_finalize_data_loaders(*, pin_memory: bool = False):
    """
    :func:`Engine.finalize` stops the DataLoader workers of the train and dev data with unread batches,
    and the pinning threads, also when repeated (the engine stays alive until the interpreter exits)
    """
    import threading

    config = Config(
        dict(
            task="train",
            device="gpu" if pin_memory else "cpu",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainTestModel,
            train_step=TrainTestModel.train_step,
            batch_size=100,
            max_seqs=2,
            optimizer={"class": "adam"},
            torch_dataloader_opts={"num_workers": 1, "pin_memory": pin_memory},
        )
    )
    datasets = {
        name: init_dataset({"class": "Task12AXDataset", "num_seqs": 10, "name": name}) for name in ["train", "dev"]
    }
    for dataset in datasets.values():
        dataset.init_seq_order(epoch=1)
    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=datasets["train"], dev_data=datasets["dev"])
        workers = []
        data_iters = []  # kept alive: the pinning thread of a dropped iterator stops on its own
        for loader in [engine._train_dataloader, engine._eval_dataloaders["dev"]]:
            data_iter = iter(loader)
            next(data_iter)
            data_iters.append(data_iter)
            workers += (loader.data_loader if pin_memory else loader)._iterator._workers
        if pin_memory:
            assert all(data_iter._thread.is_alive() for data_iter in data_iters)
        engine.finalize()
        engine.finalize()
    for worker in workers:
        # no atexit handler left which would signal it at interpreter exit
        assert not worker.is_alive() and worker.exitcode == 0 and worker._at_exit_cleanup_handler is None
    assert not [thread for thread in threading.enumerate() if thread.name == "RETURNN pin memory"]


def test_torch_engine_finalize_data_loaders():
    _check_torch_engine_finalize_data_loaders()


def test_torch_engine_finalize_data_loaders_pin_memory():
    """with :class:`returnn.torch.data.pin_memory.PinMemoryDataLoader`"""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    _check_torch_engine_finalize_data_loaders(pin_memory=True)


def _build_cuda_graph_train_config_and_dataset(
    *,
    compile_: bool,
    warmup_steps: Optional[int] = 2,
    cuda_graph: bool = True,
    optimizer_step: bool = False,
    optimizer: Optional[Dict[str, Any]] = None,
    count_steps: bool = False,
    graph_opts: Optional[Dict[str, Any]] = None,
    conv_in: bool = False,
):
    """
    small RF model + Task12AXDataset config with torch_cuda_graph, see the tests below.
    With optimizer_step: the optimizer step not in the model graph but separately (torch_optimizer_step).
    optimizer: the config entry, capturable AdamW by default.
    count_steps: the model counts its train step calls in an auxiliary parameter.
    graph_opts: further torch_cuda_graph entries.
    conv_in: two Conv2d over (time, feat) before the input layer (cuDNN gives their filter grads in another layout).
    """
    from returnn.datasets import init_dataset
    from returnn.tensor import Dim, batch_dim

    # fresh dims per test: capacities get set on them
    time_dim = Dim(None, name=f"time-cudagraph-{compile_}-{warmup_steps}-{cuda_graph}-{optimizer_step}")
    feat_dim = Dim(9, name="feat")
    classes_dim = Dim(2, name="classes")

    class _ConvIn(rf.Module):
        def __init__(self, out_dim: Dim):
            super().__init__()
            self.channel_dim = Dim(1, name="channel")
            self.conv1 = rf.Conv2d(self.channel_dim, Dim(4, name="conv1"), filter_size=(3, 3), padding="same")
            self.conv2 = rf.Conv2d(self.conv1.out_dim, Dim(4, name="conv2"), filter_size=(3, 3), padding="same")
            self.linear = rf.Linear(feat_dim, out_dim)

        def __call__(self, x: Tensor) -> Tensor:
            x, _ = self.conv1(rf.expand_dim(x, self.channel_dim), in_spatial_dims=[time_dim, feat_dim])
            x, _ = self.conv2(x, in_spatial_dims=[time_dim, feat_dim])
            return self.linear(rf.reduce_mean(x, axis=self.conv2.out_dim))

    class _Model(rf.Module):
        def __init__(self):
            super().__init__()
            self.out_dim = classes_dim
            hidden = Dim(64, name="hidden")
            self.layer = _ConvIn(hidden) if conv_in else rf.Linear(feat_dim, hidden)
            self.out = rf.Linear(hidden, classes_dim)
            if count_steps:
                self.steps_seen = rf.Parameter([], dtype="int64", auxiliary=True)
                self.steps_seen.initial = 0

    def _get_model(*, epoch, step, **_kwargs):
        return _Model()

    def _train_step(*, model: _Model, extern_data: TensorDict, **_kwargs):
        data = extern_data["data"]
        classes = extern_data["classes"]
        if count_steps:
            model.steps_seen.assign_add(1)
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
            optimizer=optimizer or {"class": "adamw", "capturable": True},
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
            **(graph_opts or {}),
        )
    if optimizer_step:
        config.typed_dict["torch_optimizer_step"] = {}
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "train", "fixed_random_seed": 1})
    dataset.init_seq_order(epoch=1)
    return config, dataset


def _run_cuda_graph_train(
    *,
    compile_: bool,
    warmup_steps: Optional[int] = 2,
    cuda_graph: bool = True,
    optimizer_step: bool = False,
    optimizer: Optional[Dict[str, Any]] = None,
    count_steps: bool = False,
    graph_opts: Optional[Dict[str, Any]] = None,
    conv_in: bool = False,
) -> Engine:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    config, dataset = _build_cuda_graph_train_config_and_dataset(
        compile_=compile_,
        warmup_steps=warmup_steps,
        cuda_graph=cuda_graph,
        optimizer_step=optimizer_step,
        optimizer=optimizer,
        count_steps=count_steps,
        graph_opts=graph_opts,
        conv_in=conv_in,
    )
    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()
        if count_steps:
            # the warm runs before the capture leave no trace on the model state
            seen = int(engine._orig_model.steps_seen.raw_tensor)
            assert seen == engine.global_train_step, (seen, engine.global_train_step)
        # the dynamic warmup steps run at the batch's own size, every batch here is smaller than the bound,
        # and the mean of the per-sequence one must still be reported as one
        seq_one = engine.learning_rate_control.epoch_data[1].error["train_loss_seq_one"]
        assert abs(seq_one - 1.0) < 1e-6, seq_one
        if cuda_graph:
            assert engine._graph_capture is not None
            assert engine._graph_capture._graph is not None, "graph never captured"
            assert engine._graph_capture.captures_optimizer == (not optimizer_step)
            # the cached reduction of the error measure must be a graph output, refreshed by every replay,
            # not an eager reduction from the first readout, which every later replay would repeat
            fer = engine._graph_capture._ctx.losses["fer"]
            fresh = rf.reduce_sum(fer.loss, axis=fer.loss.dims) if fer.loss.dims else fer.loss
            torch.testing.assert_close(fer.get_summed_loss().raw_tensor, fresh.raw_tensor)
        if optimizer_step:
            assert engine._updater._optimizer_step._graph is not None, "optimizer step never captured"
        for param_group in engine._updater.optimizer.param_groups:
            lr = param_group["lr"]  # device-tensor LR input of the captured optimizer
            if cuda_graph or optimizer_step:
                assert isinstance(lr, torch.Tensor) and lr.is_cuda
            assert float(lr) > 1e-3  # the per-step schedule advanced it
        for name, p in engine._pt_model.named_parameters():
            assert torch.isfinite(p).all(), f"non-finite param {name}"
        if cuda_graph:
            # the run end must destroy the graph, else a NCCL comm it captured never shuts down
            engine.finalize()
            assert engine._graph_capture is None
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
        ctc.mark_as_loss("ctc")
        enc_state = model.decoder.transform_encoder(enc, axis=enc_sp)
        logits, _ = model.decoder(
            targets,
            spatial_dim=tgt_time_dim,
            state=model.decoder.default_initial_state(batch_dims=[batch_dim]),
            encoder=enc_state,
        )
        ce = rf.cross_entropy(estimated=logits, target=targets, axis=vocab_dim, estimated_type="logits")
        ce.mark_as_loss("ce")

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
    losses, _ = _cuda_graph_run_step_losses(
        *_cuda_graph_packed_decoder_setup(mode), mode=mode, loss_names=["ctc", "ce"]
    )
    return losses


def _cuda_graph_run_step_losses(
    config: Config, dataset, *, mode: str, loss_names: List[str]
) -> Tuple[Dict[Tuple[int, int], Tuple[float, ...]], Engine]:
    """
    train in-process

    :return: per (epoch, step) the losses parsed from the RETURNN log file, and the engine
    """
    import os
    import re
    import tempfile
    from returnn.log import log as returnn_log

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
    steps = re.findall(r"ep (\d+) train, step (\d+), " + ", ".join(f"{name} ([0-9.]+)" for name in loss_names), txt)
    assert len(steps) >= 5, f"{mode}: only {len(steps)} steps parsed from the log"
    return {(int(ep), int(s)): tuple(float(v) for v in values) for ep, s, *values in steps}, engine


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


def test_torch_engine_cuda_graph_finalize_then_eager():
    """
    A graphc training which stops in the middle of an epoch (an exception), with the bound shapes still enabled,
    then an eager packed engine in the same process with the same dims, incl. the global batch dim:
    after finalize, the dims are as before the graphc engine.
    """

    class _StopTraining(Exception):
        pass

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    config, dataset = _build_cuda_graph_train_config_and_dataset(compile_=False)
    time_dim = config.typed_dict["extern_data"]["data"]["dims"][1]
    dyn_lr = config.typed_dict["dynamic_learning_rate"]

    def _dyn_lr_stop(*, global_train_step: int, **kwargs) -> float:
        if global_train_step == 5:  # after the capture
            raise _StopTraining()
        return dyn_lr(global_train_step=global_train_step, **kwargs)

    config.typed_dict["dynamic_learning_rate"] = _dyn_lr_stop
    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        try:
            engine.train()
        except _StopTraining:
            pass
        else:
            raise Exception("did not get expected exception")
        assert engine._graph_capture._graph is not None, "graph never captured"
        engine.finalize()
    config.typed_dict["dynamic_learning_rate"] = dyn_lr
    del config.typed_dict["torch_cuda_graph"]
    config.typed_dict["packed_tensors"] = True
    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()
        engine.finalize()
    assert time_dim.capacity is None


def test_torch_engine_cuda_graph_packed_total_bound():
    """
    The gapped packed tensor which the captured step sees has the extent of the input buffer:
    the declared packed_total_bound, else the one inferred from packed_batch_size,
    capped by every seq at full capacity.
    """
    from returnn.datasets import init_dataset
    from returnn.tensor import Dim, batch_dim
    import numpy

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    rnd = numpy.random.RandomState(3)
    # first a batch of every seq at full capacity, then random lens
    seqs = [{"data": rnd.randn(8, 5).astype("float32")} for _ in range(4)]
    seqs += [{"data": rnd.randn(int(rnd.randint(1, 9)), 5).astype("float32")} for _ in range(20)]
    # (gap, align, packed_batch_size, declared packed_total_bound) -> expected extent;
    # batch_size_bound 4, dim_capacity 8: every seq at full capacity is 4 * (8 + gap + align - 1) frames
    for gap, align, budget, declared, expected in [
        (1, 1, 12, None, 16),
        (2, 2, 12, None, 24),
        (0, 1, 12, None, 12),
        (1, 1, 12, 24, 24),
        (1, 1, 40, None, 36),
        (1, 1, 12, 100, 36),
        (2, 2, 40, None, 44),
    ]:
        time_dim = Dim(None, name=f"time-bound-{gap}-{align}-{budget}-{declared}")
        feat_dim = Dim(5, name="feat")
        seen = []

        def _get_model(**_kwargs):
            return rf.Linear(feat_dim, Dim(3, name="out"))

        def _train_step(*, model: rf.Linear, extern_data: TensorDict, **_kwargs):
            data = extern_data["data"]
            seen.append(data.raw_tensor.packed_dim.dimension)
            loss = rf.reduce_sum(model(data) ** 2, axis=model.out_dim)
            loss.mark_as_loss("l2")

        config = Config(
            dict(
                task="train",
                device="gpu",
                extern_data={"data": {"dims": [batch_dim, time_dim, feat_dim], "dtype": "float32"}},
                get_model=_get_model,
                train_step=_train_step,
                batch_size=None,
                packed_batch_size={"data": budget},
                max_seqs=4,
                num_epochs=1,
                learning_rate=1e-3,
                optimizer={"class": "adamw", "capturable": True},
                torch_dataloader_opts={"num_workers": 0},
                packed_tensors={"per_key": {"data": {"gap": gap, "align": align}}},
                torch_cuda_graph=dict(
                    batch_size_bound=4,
                    dim_capacity={"data": 8},
                    **({"packed_total_bound": {"data": declared}} if declared is not None else {}),
                    capture_optimizer=True,
                ),
            )
        )
        dataset = init_dataset({"class": "StaticDataset", "data": seqs, "input_dim": 5})
        dataset.init_seq_order(epoch=1)
        with global_config_ctx(config):
            engine = Engine(config=config)
            engine.init_train_from_config(train_data=dataset)
            engine.train()
            assert engine._graph_capture._graph is not None, "graph never captured"
            assert engine._graph_capture.data_bound_sizes() == {"data": expected}
            assert seen and set(seen) == {expected}, (gap, align, budget, declared, seen)
            engine.finalize()


def _cuda_graph_lm_eos_setup(storage: str, *, graphc: bool):
    """config+dataset for the padding seqs test, see below"""
    from returnn.datasets import init_dataset
    from returnn.tensor import Dim, batch_dim
    from returnn.frontend.decoder.transformer import TransformerDecoder
    import numpy

    time_dim = Dim(None, name=f"time-lm-{storage}-{graphc}")
    tgt_time_dim = Dim(None, name=f"tgt-time-lm-{storage}-{graphc}")
    feat_dim = Dim(8, name="feat")
    vocab_dim = Dim(11, name="vocab")

    def _get_model(**_kwargs):
        return TransformerDecoder(
            None, vocab_dim, Dim(32, name="dec"), num_layers=1, num_heads=2, dropout=0.0, att_dropout=0.0
        )

    def _train_step(*, model: TransformerDecoder, extern_data: TensorDict, **_kwargs):
        # as an LM or AED train step: BOS prepended to the input, EOS appended to the targets
        targets = extern_data["classes"]
        inputs, (targets_w_eos_dim,) = rf.pad(targets, axes=[tgt_time_dim], padding=[(1, 0)], value=0)
        targets_w_eos, _ = rf.pad(targets, axes=[tgt_time_dim], padding=[(0, 1)], value=0, out_dims=[targets_w_eos_dim])
        logits, _ = model(
            inputs, spatial_dim=targets_w_eos_dim, state=model.default_initial_state(batch_dims=[batch_dim])
        )
        ce = rf.cross_entropy(estimated=logits, target=targets_w_eos, axis=vocab_dim, estimated_type="logits")
        ce.mark_as_loss("ce")
        rf.reduce_sum(ce, axis=targets_w_eos_dim).mark_as_loss("seq_ce", as_error=True)

    rnd = numpy.random.RandomState(7)
    seqs = []
    for i in range(30):
        t = int(rnd.randint(15, 99))
        n = 0 if i % 7 == 3 else 2 + t // 12  # some real empty seqs: they keep their EOS target
        seqs.append({"data": rnd.randn(t, 8).astype("float32"), "classes": rnd.randint(1, 11, (n,)).astype("int32")})
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
        batch_size=400,
        max_seqs=10,
        # the dev eval between the epochs runs the dynamic-shape path, then the captured steps resume
        num_epochs=2,
        eval_datasets={"dev": {"class": "StaticDataset", "data": seqs, "input_dim": 8, "output_dim": 11}},
        learning_rate=0.0,  # frozen params: each step's losses depend only on its batch
        optimizer={"class": "adamw", "capturable": True},
        log_grad_norm=True,
        torch_dataloader_opts={"num_workers": 0},
        torch_amp="bfloat16",
        grad_scaler=None,
    )
    if storage == "packed":
        cfg["packed_tensors"] = {"per_key": {"data": {"gap": 8, "align": 1}, "classes": {"gap": 2, "align": 1}}}
    if graphc:
        cfg["torch_cuda_graph"] = dict(
            batch_size_bound=16,  # above max_seqs: every batch has padding seqs
            dim_capacity={"data": 100, "classes": 16},
            **({"packed_total_bound": {"data": 400 + 16 * 9, "classes": 16 * 18}} if storage == "packed" else {}),
            capture_optimizer=True,
            compile=True,
            mask_padding_seqs=True,
        )
    dataset = init_dataset({"class": "StaticDataset", "data": seqs, "input_dim": 8, "output_dim": 11})
    dataset.init_seq_order(epoch=1)
    return Config(cfg), dataset


def test_torch_engine_cuda_graph_mask_padding_seqs():
    """
    With mask_padding_seqs, the padding seqs which fill each batch up to batch_size_bound get no EOS target:
    the per-step losses of an LM step (default normalization, also per seq) and the gradient norm
    match the eager run, padded and packed, also after the dev eval, which matches as well
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    names = ["ce", "seq_ce", "grad_norm:p2"]
    for storage in ["padded", "packed"]:
        (losses_a, engine_a), (losses_b, engine_b) = [
            _cuda_graph_run_step_losses(
                *_cuda_graph_lm_eos_setup(storage, graphc=graphc), mode=storage, loss_names=names
            )
            for graphc in [False, True]
        ]
        common = sorted(set(losses_a) & set(losses_b))
        assert {ep for ep, _ in common} == {1, 2}, common
        for s in common:
            for name, a, b in zip(names, losses_a[s], losses_b[s]):
                assert abs(a - b) / max(abs(a), 1e-6) < 2e-2, f"{storage} step {s} {name}: {a} vs {b}"
        for key in ["dev_loss_ce", "dev_loss_seq_ce"]:
            a, b = (engine.learning_rate_control.epoch_data[1].error[key] for engine in (engine_a, engine_b))
            assert abs(a - b) / max(abs(a), 1e-6) < 2e-2, f"{storage} {key}: {a} vs {b}"


def test_torch_engine_cuda_graph_train():
    """whole-train-step CUDA-graph capture/replay (torch_cuda_graph), 2 epochs across an epoch boundary,
    in-graph optimizer + per-step LR schedule via the device-tensor LR input"""
    _run_cuda_graph_train(compile_=False)


def test_torch_engine_cuda_graph_warm_runs_keep_state():
    """
    The runs which only warm the kernels or trace the compiled step leave no trace on the model:
    its buffers and auxiliary parameters (a step counter here, running statistics in a real model)
    are back afterwards, so the model sees every batch exactly once.
    """
    for compile_ in (False, True):
        # the traced step with eager kernels, Inductor would only add compile time here
        graph_opts = {"debug_aot_eager": True} if compile_ else None
        _run_cuda_graph_train(compile_=compile_, count_steps=True, graph_opts=graph_opts)


def test_torch_engine_cuda_graph_compile_train():
    """torch_cuda_graph "compile": the whole step Inductor-compiled (aot_function + compile_fx,
    no Dynamo), then captured; otherwise as :func:`test_torch_engine_cuda_graph_train`"""
    _run_cuda_graph_train(compile_=True)


def test_torch_engine_cuda_graph_compile_train_default_warmup():
    """as :func:`test_torch_engine_cuda_graph_compile_train` with the default warmup_steps (0):
    no eager step, the lazy optimizer state is created directly before the capture"""
    _run_cuda_graph_train(compile_=True, warmup_steps=None)


def test_torch_engine_cuda_graph_compile_grad_layout_fused_adamw():
    """
    The compiled step gives each grad in the strides of its param, as eager autograd does,
    also for a conv filter (whose grad cuDNN gives in another layout):
    fused AdamW (here the in-graph optimizer step) requires that
    """
    engine = _run_cuda_graph_train(
        compile_=True, optimizer={"class": "adamw", "fused": True, "capturable": True}, conv_in=True
    )
    for name, p in engine._pt_model.named_parameters():
        assert p.grad.stride() == p.stride(), f"{name}: grad strides {p.grad.stride()} vs param {p.stride()}"


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


def test_torch_engine_cuda_graph_optimizer_state_in_advance():
    """
    With the in-graph optimizer step and no warmup step, the optimizer state is created in advance
    as the first step would create it (init_optimizer_state), so the params after training match the eager engine,
    also for optimizers whose fresh state is not all zeros (NAdam: mu_product 1; _AnchoredSGD: a copy of the param).
    """
    for optimizer in [{"class": "nadam", "capturable": True}, {"class": _AnchoredSGD}]:
        eager = _run_cuda_graph_train(compile_=False, cuda_graph=False, optimizer=optimizer)
        for compile_ in (False, True):
            captured = _run_cuda_graph_train(compile_=compile_, warmup_steps=None, optimizer=optimizer)
            for (name, p), (_, q) in zip(eager._pt_model.named_parameters(), captured._pt_model.named_parameters()):
                torch.testing.assert_close(
                    q, p, rtol=1e-4, atol=1e-5, msg=lambda m: f"{optimizer}, compile {compile_}, {name}: {m}"
                )


class _MuonLike(torch.optim.Optimizer):
    """
    Written like the i6 Muon (i6_experiments exp2024_04_23_baselines optim_ext/muon.py), small:
    Muon (bf16 Newton-Schulz) on 2D params, Adam on the rest, with a Python-float lr
    (``alpha=`` / ``value=`` arguments) and a Python-int step counter.
    ``ns_dtype=torch.float64`` gives the reference.
    """

    def __init__(
        self, params, lr=2e-2, momentum=0.95, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8, ns_dtype=torch.bfloat16
    ):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, betas=betas, eps=eps))
        self.ns_dtype = ns_dtype

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
                    x = p.grad.add(buf, alpha=group["momentum"]).to(self.ns_dtype)
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


def _run_optimizer_step(
    opt_factory, *, opts=None, num_steps=8, replace_grads_at=None, reload_at=None, dtype=torch.float32
):
    """
    :return: params after num_steps updates with a per-step LR schedule and random grads,
        via the plain ``optimizer.step()`` if opts is None, otherwise via :class:`OptimizerStep`;
        the optimizer; and the OptimizerStep
    """
    from returnn.torch.util.optimizer_step import OptimizerStep

    device = _optimizer_step_device()
    gen = torch.Generator().manual_seed(0)
    shapes = [(32, 16), (16,), (64, 32), (32,), (8, 1, 3)]
    params = [torch.nn.Parameter(torch.randn(s, generator=gen).to(device, dtype)) for s in shapes]
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


def _check_optimizer_step(opt_factory, opts, *, ref_opt_factory=None, **kwargs):
    """
    Params and optimizer state (incl. the checkpoint scalars) at least as accurate as the plain eager optimizer.step(),
    both measured against a float64 run (with ``ref_opt_factory``, default ``opt_factory``).
    Not bitwise eager: Inductor keeps low-precision intermediates in float32.
    """
    ref_params, ref_opt, _ = _run_optimizer_step(ref_opt_factory or opt_factory, dtype=torch.float64, **kwargs)
    eager_params, eager_opt, _ = _run_optimizer_step(opt_factory, **kwargs)
    params, opt, opt_step = _run_optimizer_step(opt_factory, opts=opts, **kwargs)

    def _check(name: str, v: torch.Tensor, v_eager: torch.Tensor, v_ref: torch.Tensor):
        err, err_eager = (v.double() - v_ref).norm().item(), (v_eager.double() - v_ref).norm().item()
        assert err <= 1.1 * err_eager + 1e-6 * v_ref.norm().item(), f"{name}: error {err} vs eager {err_eager}"

    for i, (p_ref, p_eager, p) in enumerate(zip(ref_params, eager_params, params)):
        _check(f"param {i}", p, p_eager, p_ref)
    ref_state = ref_opt.state_dict()["state"]
    eager_state = eager_opt.state_dict()["state"]
    state = opt_step.state_dict_to_host_scalars(opt.state_dict())["state"]
    assert set(state) == set(eager_state)
    for i, s_eager in eager_state.items():
        assert set(state[i]) == set(s_eager)
        for k, v_eager in s_eager.items():
            v = state[i][k]
            if isinstance(v_eager, torch.Tensor):
                _check(f"state {i} {k}", v, v_eager, ref_state[i][k])
            else:  # Python scalar, e.g. the step counter of _MuonLike
                assert type(v) is type(v_eager) and v == v_eager, f"state {i} {k}: {v!r} vs {v_eager!r}"
    return opt_step


def _muon_like_float64(params):
    return _MuonLike(params, ns_dtype=torch.float64)


def test_torch_optimizer_step_eager():
    opt_step = _check_optimizer_step(_adamw, {"compile": False, "capture": False}, reload_at=5)
    assert opt_step._graph is None


def test_torch_optimizer_step_dynamic_state_keys():
    """
    Python-float lr and Python-int step counter as device tensors (incl. alpha= / value= args),
    back to Python scalars in the checkpoint
    """
    opts = {"dynamic_state_keys": ["step"]}
    if not torch.cuda.is_available():
        opts.update({"compile": False, "capture": False})
    opt_step = _check_optimizer_step(_MuonLike, opts, ref_opt_factory=_muon_like_float64, reload_at=5)
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
    opt_step = _check_optimizer_step(_adamw, {"compile": False})
    assert opt_step._graph is not None and opt_step._num_captures == 1


def test_torch_optimizer_step_compile_capture():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    opt_step = _check_optimizer_step(_adamw, {})
    assert opt_step._graph is not None and (opt_step._num_traces, opt_step._num_captures) == (1, 1)


def test_torch_optimizer_step_compile_capture_muon_like():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    opt_step = _check_optimizer_step(_MuonLike, {"dynamic_state_keys": ["step"]}, ref_opt_factory=_muon_like_float64)
    assert (opt_step._num_traces, opt_step._num_captures) == (1, 1)


def test_torch_optimizer_step_compile_capture_inductor_config():
    """inductor_config applies while the step compiles, by default deterministic, not after"""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    import torch._inductor.config as inductor_config
    from torch._inductor.custom_graph_pass import CustomGraphPass

    class _Recorder(CustomGraphPass):
        def __init__(self):
            self.seen = []

        def __call__(self, graph: torch.fx.Graph):
            self.seen.append((inductor_config.deterministic, inductor_config.max_autotune_gemm))

        def uuid(self):
            return None  # not cacheable: compiles every time

    for user_config, expected in [
        ({}, (True, False)),
        ({"deterministic": False, "max_autotune_gemm": True}, (False, True)),
    ]:
        recorder = _Recorder()
        opts = {
            "dynamic_state_keys": ["step"],
            "inductor_config": {**user_config, "post_grad_custom_post_pass": recorder},
        }
        opt_step = _check_optimizer_step(_MuonLike, opts, ref_opt_factory=_muon_like_float64)
        assert (opt_step._num_traces, opt_step._num_captures) == (1, 1)
        assert recorder.seen == [expected]
    assert inductor_config.post_grad_custom_post_pass is None and not inductor_config.max_autotune_gemm


def test_torch_optimizer_step_compile_deterministic():
    """
    Compiles of the same step (fresh Inductor caches, as on several ranks) give bitwise the same update
    with the default inductor_config, i.e. the ranks of a distributed training keep the same params.
    The step normalizes stacked matrices (as Muon over experts): without "deterministic", the reduction kernel config
    is picked by timing, and about every second compile differed (torch 2.14, RTX 5090).
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    from torch._inductor.utils import fresh_cache
    from returnn.torch.util.optimizer_step import OptimizerStep

    class _NormalizedSgd(torch.optim.Optimizer):
        @torch.no_grad()
        def step(self, closure=None):
            """update: -lr * the grad of each matrix normalized, in bfloat16"""
            for group in self.param_groups:
                for p in group["params"]:
                    x = p.grad.bfloat16()
                    p.add_((x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)).float(), alpha=-group["lr"])

    def _run() -> torch.Tensor:
        gen = torch.Generator(device="cuda").manual_seed(0)
        param = torch.nn.Parameter(torch.randn((16, 1024, 1024), device="cuda", generator=gen))
        param.grad = torch.zeros_like(param)
        opt_step = OptimizerStep(optimizer=_NormalizedSgd([param], {"lr": 1e-2}), opts={})
        for _ in range(3):
            param.grad.copy_(torch.randn(param.shape, device="cuda", generator=gen))
            opt_step.step()
        assert (opt_step._num_traces, opt_step._num_captures) == (1, 1)
        return param.detach().clone()

    results = []
    for _ in range(3):
        with fresh_cache():
            results.append(_run())
    assert all(torch.equal(results[0], res) for res in results[1:])


def test_torch_optimizer_step_replaced_grads():
    """new grad buffers: recapture on the new addresses, no retrace"""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    opt_step = _check_optimizer_step(_adamw, {}, replace_grads_at=5)
    assert (opt_step._num_traces, opt_step._num_captures) == (1, 2)


class _TunedSgd(torch.optim.Optimizer):
    """SGD which "autotunes" (host sync, not allowed in a capture) for each new set of shapes with grads, as Triton"""

    def __init__(self, params):
        super().__init__(params, {"lr": 0.1})
        self.tuned = set()

    @torch.no_grad()
    def step(self, closure=None):
        """update"""
        group = self.param_groups[0]
        params = [p for p in group["params"] if p.grad is not None]
        key = tuple(tuple(p.shape) for p in params)
        if key not in self.tuned:
            torch.cuda.synchronize()
            self.tuned.add(key)
        for p in params:
            p.add_(p.grad, alpha=-group["lr"])


def test_torch_optimizer_step_recapture_after_warm_step():
    """
    another set of params with grads (other shapes) under capture without compile:
    one uncaptured step before the recapture, so the optimizer's lazy work (here a host sync) is outside of it
    """
    from returnn.torch.util.optimizer_step import OptimizerStep

    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    params = [torch.nn.Parameter(torch.zeros(s, device="cuda")) for s in [(4, 3), (5,)]]
    opt_step = OptimizerStep(optimizer=_TunedSgd(params), opts={"compile": False})
    for i in range(6):
        for j, p in enumerate(params):
            p.grad = None if (i >= 3 and j == 1) else torch.ones_like(p)
        opt_step.step()
    assert opt_step._num_captures == 2
    torch.testing.assert_close(params[0], torch.full_like(params[0], -0.6))
    torch.testing.assert_close(params[1], torch.full_like(params[1], -0.3))


def test_torch_optimizer_step_reload():
    """checkpoint round trip: new state tensors and lr, recapture, no retrace"""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    opt_step = _check_optimizer_step(
        _MuonLike, {"dynamic_state_keys": ["step"]}, ref_opt_factory=_muon_like_float64, reload_at=5
    )
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
    config, dataset = _build_cuda_graph_train_config_and_dataset(
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
