"""
Tests for PyTorch engine.
"""

from __future__ import annotations
import _setup_test_env  # noqa
from typing import Optional
import contextlib
import sys
import unittest
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


def test_save_optimizer_callable_config():
    # The optimizer config can be a callable (e.g. the optimizer class itself).
    # The saved checkpoint metadata must still be loadable under the torch >= 2.6 weights_only default.
    config = Config(dict(optimizer=torch.optim.AdamW))
    model = torch.nn.Linear(7, 5)
    updater = Updater(config=config, network=model, device=torch.device("cpu"))
    updater.create_optimizer()

    with tempfile.TemporaryDirectory(prefix="returnn_test_save_optimizer_callable_config") as tmp_dir:
        updater.save_optimizer(tmp_dir + "/model.opt.pt")
        updater.load_optimizer(tmp_dir + "/model.opt.pt")


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


def test_load_optimizer_changed_weight_decay_split():
    # A changed weight-decay split moves params between the two param groups.
    # load_optimizer must not fail on that (it warns and remaps the per-param state by name),
    # and the state must survive the move.
    model = torch.nn.Sequential(torch.nn.Linear(7, 5), torch.nn.LayerNorm(5))

    config1 = Config(dict(optimizer={"class": "adamw", "weight_decay": 1e-3}))
    updater1 = Updater(config=config1, network=model, device=torch.device("cpu"))
    updater1.create_optimizer()
    updater1.set_current_train_step(global_train_step=0, epoch=1)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    updater1.get_optimizer().step()

    ln_weight = model[1].weight
    state1 = updater1.get_optimizer().state[ln_weight]
    assert "exp_avg" in state1
    exp_avg1 = state1["exp_avg"].clone()

    def _include_check(*, module, **_kwargs):
        if isinstance(module, torch.nn.LayerNorm):
            return True
        return None

    config2 = Config(
        dict(optimizer={"class": "adamw", "weight_decay": 1e-3, "weight_decay_custom_include_check": _include_check})
    )
    updater2 = Updater(config=config2, network=model, device=torch.device("cpu"))
    updater2.create_optimizer()
    updater2.set_current_train_step(global_train_step=0, epoch=1)

    with tempfile.TemporaryDirectory(prefix="returnn_test_load_opt_changed_wd_split") as tmp_dir:
        updater1.save_optimizer(tmp_dir + "/model.opt.pt")
        updater2.load_optimizer(tmp_dir + "/model.opt.pt")

    opt2 = updater2.get_optimizer()
    groups_by_wd = {group["weight_decay"]: group for group in opt2.param_groups}
    assert any(p is ln_weight for p in groups_by_wd[1e-3]["params"])
    assert torch.equal(opt2.state[ln_weight]["exp_avg"], exp_avg1)


def test_multi_optimizer_load_cross_algorithm_error():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))

    def _filter_first(*, full_param_name, **_kwargs):
        return full_param_name.startswith("0.")

    def _filter_second(*, full_param_name, **_kwargs):
        return full_param_name.startswith("1.")

    def _make_updater(params_filter):
        config = Config(
            dict(
                optimizer={
                    "class": "multi",
                    "optimizers": [
                        {"class": "adamw", "params_filter": params_filter, "weight_decay": 1e-3},
                        {"class": "sgd", "momentum": 0.9},
                    ],
                }
            )
        )
        updater = Updater(config=config, network=model, device=torch.device("cpu"))
        updater.create_optimizer()
        updater.set_current_train_step(global_train_step=0, epoch=1)
        return updater

    updater1 = _make_updater(_filter_first)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    updater1.get_optimizer().step()
    updater2 = _make_updater(_filter_second)

    with tempfile.TemporaryDirectory(prefix="returnn_test_multi_load_cross_algo") as tmp_dir:
        updater1.save_optimizer(tmp_dir + "/model.opt.pt")
        try:
            updater2.load_optimizer(tmp_dir + "/model.opt.pt")
        except ValueError as exc:
            assert "moved" in str(exc) and "AdamW" in str(exc) and "SGD" in str(exc)
        else:
            raise AssertionError("expected ValueError for a cross-optimizer param move")


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


@contextlib.contextmanager
def set_behavior_version(version: int):
    """
    This is a context manager which sets the behavior version to the given value.
    """
    from returnn.util.basic import BehaviorVersion

    # noinspection PyProtectedMember
    old = BehaviorVersion._get_state()
    try:
        # noinspection PyProtectedMember
        BehaviorVersion._reset()
        BehaviorVersion.set(version)
        yield
    finally:
        # noinspection PyProtectedMember
        BehaviorVersion._reset(old)


def test_updater_weight_decay_blacklist_rf_modules():
    # Since behavior version 30, the default weight-decay blacklist also covers
    # rf.LayerNorm and rf.Embedding, matching torch.nn.LayerNorm / torch.nn.Embedding.
    from returnn.torch.frontend.bridge import rf_module_to_pt_module
    from returnn.util.basic import DictRefKeys

    rf.select_backend_torch()

    class _Model(rf.Module):
        def __init__(self):
            super().__init__()
            in_dim, embed_dim, out_dim = rf.Dim(11), rf.Dim(5), rf.Dim(7)
            self.embed = rf.Embedding(in_dim, embed_dim)
            self.layer_norm = rf.LayerNorm(embed_dim)
            self.linear = rf.Linear(embed_dim, out_dim)

    config = Config(dict(optimizer={"class": "adamw", "weight_decay": 1e-3}))

    def _params_by_wd():
        pt_model = rf_module_to_pt_module(_Model())
        updater = Updater(config=config, network=pt_model, device=torch.device("cpu"))
        updater.create_optimizer()
        opt = updater.get_optimizer()
        assert len(opt.param_groups) == 2
        param_to_name = DictRefKeys((param, name) for name, param in pt_model.named_parameters())
        return {pg["weight_decay"]: {param_to_name[p] for p in pg["params"]} for pg in opt.param_groups}

    with set_behavior_version(29):
        params_by_wd = _params_by_wd()
        assert params_by_wd[1e-3] == {"embed.weight", "layer_norm.scale", "linear.weight"}
        assert params_by_wd[0.0] == {"layer_norm.bias", "linear.bias"}

    with set_behavior_version(30):
        params_by_wd = _params_by_wd()
        assert params_by_wd[1e-3] == {"linear.weight"}
        assert params_by_wd[0.0] == {"embed.weight", "layer_norm.scale", "layer_norm.bias", "linear.bias"}


def test_updater_lr_multipliers():
    from collections import defaultdict
    from fnmatch import fnmatchcase
    from typing import Dict, List, Set, Any
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


def _build_cuda_graph_train_config_and_dataset(*, compile_: bool):
    """small RF model + Task12AXDataset config with torch_cuda_graph, see the tests below"""
    from returnn.datasets import init_dataset
    from returnn.tensor import Dim, batch_dim

    time_dim = Dim(None, name=f"time-cudagraph-{compile_}")  # fresh dims per test: capacities get set on them
    feat_dim = Dim(9, name="feat")
    classes_dim = Dim(2, name="classes")

    class _Model(rf.Module):
        def __init__(self):
            super().__init__()
            self.out_dim = classes_dim
            hidden = Dim(64, name="hidden")
            self.layer = rf.Linear(feat_dim, hidden)
            self.out = rf.Linear(hidden, classes_dim)

    def _get_model(*, epoch, step, **_kwargs):
        return _Model()

    def _train_step(*, model: _Model, extern_data: TensorDict, **_kwargs):
        data = extern_data["data"]
        classes = extern_data["classes"]
        x = rf.relu(model.layer(data))
        logits = model.out(x)
        loss = rf.cross_entropy(target=classes, estimated=logits, estimated_type="logits", axis=model.out_dim)
        loss.mark_as_loss("ce")

    def _dyn_lr(*, global_train_step: int, learning_rate: float, **_kwargs) -> float:
        # per-step LR schedule: under capture_optimizer this exercises the device-tensor LR input
        return learning_rate * (1.0 + 0.1 * global_train_step)

    config = Config(
        dict(
            task="train",
            device="gpu",
            extern_data={
                "data": {"dims": [batch_dim, time_dim, feat_dim], "dtype": "float32"},
                "classes": {"dims": [batch_dim, time_dim], "dtype": "int32", "sparse_dim": classes_dim},
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
            torch_cuda_graph=dict(
                batch_size_bound=10,
                dim_capacity={"data": 100, "classes": 100},
                warmup_steps=2,
                capture_optimizer=True,
                **({"compile": True} if compile_ else {}),
            ),
            torch_dataloader_opts={"num_workers": 0},
        )
    )
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "train", "fixed_random_seed": 1})
    dataset.init_seq_order(epoch=1)
    return config, dataset


def _run_cuda_graph_train(*, compile_: bool):
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    config, dataset = _build_cuda_graph_train_config_and_dataset(compile_=compile_)
    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()
        assert engine._graph_capture is not None
        assert engine._graph_capture._graph is not None, "graph never captured"
        assert engine._graph_capture.captures_optimizer
        for param_group in engine._updater.optimizer.param_groups:
            lr = param_group["lr"]  # device-tensor LR input of the captured optimizer
            assert isinstance(lr, torch.Tensor) and lr.is_cuda
            assert lr.item() > 1e-3  # the per-step schedule advanced it
        for name, p in engine._pt_model.named_parameters():
            assert torch.isfinite(p).all(), f"non-finite param {name}"


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


def test_torch_engine_cuda_graph_train():
    """whole-train-step CUDA-graph capture/replay (torch_cuda_graph), 2 epochs across an epoch boundary,
    in-graph optimizer + per-step LR schedule via the device-tensor LR input"""
    _run_cuda_graph_train(compile_=False)


def test_torch_engine_cuda_graph_compile_train():
    """torch_cuda_graph "compile": the whole step Inductor-compiled (aot_function + compile_fx,
    no Dynamo), then captured; otherwise as :func:`test_torch_engine_cuda_graph_train`"""
    _run_cuda_graph_train(compile_=True)


# must be in the global scope due to pickling
def _multi_test_hidden_matrix_filter(*, full_param_name: str, param: torch.nn.Parameter, module, **_kwargs) -> bool:
    return param.dim() >= 2 and not isinstance(module, torch.nn.Embedding)


# must be in the global scope due to pickling
class _RecordingScheduleFreeSGD(torch.optim.SGD):
    """SGD with recording schedule-free train()/eval() methods, for testing the engine hooks."""

    calls = []

    def train(self):
        """record train mode switch"""
        type(self).calls.append("train")

    def eval(self):
        """record eval mode switch"""
        type(self).calls.append("eval")


def _make_multi_test_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Embedding(10, 5),
        torch.nn.LayerNorm(5),
        torch.nn.Linear(5, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 5),
    )


def _multi_test_layer2_weight_filter(*, full_param_name: str, param: torch.nn.Parameter, **_kwargs) -> bool:
    return full_param_name.startswith("2.") and param.dim() >= 2


def test_multi_optimizer():
    from returnn.util.basic import DictRefKeys
    from returnn.torch.optim.multi import MultiOptimizer

    config = Config(
        dict(
            optimizer={
                "class": "multi",
                "optimizers": [
                    {
                        "class": "sgd",
                        "params_filter": _multi_test_layer2_weight_filter,
                        "learning_rate_multiplier": 2.0,
                        "momentum": 0.9,
                    },
                    {"class": "adamw", "weight_decay": 1e-3, "epsilon": 1e-8},
                ],
            }
        )
    )
    model = _make_multi_test_model()
    updater = Updater(config=config, network=model, device=torch.device("cpu"))
    updater.create_optimizer()
    updater.set_current_train_step(global_train_step=0, epoch=1)

    opt = updater.get_optimizer()
    assert isinstance(opt, MultiOptimizer)
    assert len(opt.sub_optimizers) == 2
    sgd_sub, adamw_sub = opt.sub_optimizers
    assert isinstance(sgd_sub, torch.optim.SGD) and isinstance(adamw_sub, torch.optim.AdamW)

    param_to_name = DictRefKeys((param, name) for name, param in model.named_parameters())
    assert len(sgd_sub.param_groups) == 1
    assert {param_to_name[p] for p in sgd_sub.param_groups[0]["params"]} == {"2.weight"}
    assert sgd_sub.param_groups[0]["momentum"] == 0.9
    # AdamW sub: default weight-decay split, embedding/LayerNorm/biases without decay.
    assert len(adamw_sub.param_groups) == 2
    adamw_groups_by_wd = {pg["weight_decay"]: pg for pg in adamw_sub.param_groups}
    assert set(adamw_groups_by_wd.keys()) == {0.0, 1e-3}
    assert {param_to_name[p] for p in adamw_groups_by_wd[1e-3]["params"]} == {"4.weight"}
    assert {param_to_name[p] for p in adamw_groups_by_wd[0.0]["params"]} == {
        "0.weight",
        "1.weight",
        "1.bias",
        "2.bias",
        "4.bias",
    }
    assert adamw_sub.param_groups[0]["eps"] == 1e-8

    # The concatenated param_groups view covers all params exactly once.
    assert len(opt.param_groups) == 3
    param_names, _ = updater._get_opt_param_names()
    assert sorted(param_names) == sorted(name for name, _ in model.named_parameters())

    # LR schedule propagates into the sub-optimizers, with the multiplier.
    updater.set_learning_rate(0.5)
    assert sgd_sub.param_groups[0]["lr"] == 0.5 * 2.0
    assert all(pg["lr"] == 0.5 for pg in adamw_sub.param_groups)


def test_multi_optimizer_save_load():
    def _make_updater():
        config = Config(
            dict(
                optimizer={
                    "class": "multi",
                    "optimizers": [
                        {
                            "class": "sgd",
                            "params_filter": _multi_test_layer2_weight_filter,
                            "learning_rate_multiplier": 2.0,
                            "momentum": 0.9,
                        },
                        {"class": "adamw", "weight_decay": 1e-3},
                    ],
                }
            )
        )
        model_ = _make_multi_test_model()
        updater_ = Updater(config=config, network=model_, device=torch.device("cpu"))
        updater_.create_optimizer()
        updater_.set_current_train_step(global_train_step=0, epoch=1)
        return updater_, model_

    updater, model = _make_updater()
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    updater.get_optimizer().step()

    with tempfile.TemporaryDirectory(prefix="returnn_test_multi_optimizer_save_load") as tmp_dir:
        updater.save_optimizer(tmp_dir + "/model.opt.pt")

        updater2, model2 = _make_updater()
        updater2.load_optimizer(tmp_dir + "/model.opt.pt")

        state_dict1 = updater.get_optimizer().state_dict()
        state_dict2 = updater2.get_optimizer().state_dict()
        assert set(state_dict1["state"].keys()) == set(state_dict2["state"].keys())
        for param_idx, param_state1 in state_dict1["state"].items():
            param_state2 = state_dict2["state"][param_idx]
            assert set(param_state1.keys()) == set(param_state2.keys())
            for key, value1 in param_state1.items():
                value2 = param_state2[key]
                if isinstance(value1, torch.Tensor):
                    assert torch.equal(value1, value2), f"state {param_idx} {key} differs"
                else:
                    assert value1 == value2, f"state {param_idx} {key} differs"
        assert len(state_dict1["param_groups"]) == len(state_dict2["param_groups"])
        for group1, group2 in zip(state_dict1["param_groups"], state_dict2["param_groups"]):
            assert group1["params"] == group2["params"]

        # After loading, the composite's param_groups must alias the sub-optimizers' rebuilt
        # group dicts, so that the LR schedule keeps reaching the sub-optimizers.
        opt2 = updater2.get_optimizer()
        flat_sub_groups = [group for sub in opt2.sub_optimizers for group in sub.param_groups]
        assert len(opt2.param_groups) == len(flat_sub_groups)
        assert all(a is b for a, b in zip(opt2.param_groups, flat_sub_groups))
        updater2.set_learning_rate(0.125)
        sgd_sub2, adamw_sub2 = opt2.sub_optimizers
        assert all(pg["lr"] == 0.125 * 2.0 for pg in sgd_sub2.param_groups)
        assert all(pg["lr"] == 0.125 for pg in adamw_sub2.param_groups)


def test_multi_optimizer_leftover_params_error():
    config = Config(
        dict(
            optimizer={
                "class": "multi",
                "optimizers": [
                    {"class": "sgd", "params_filter": _multi_test_layer2_weight_filter, "momentum": 0.9},
                ],
            }
        )
    )
    model = _make_multi_test_model()
    updater = Updater(config=config, network=model, device=torch.device("cpu"))
    try:
        updater.create_optimizer()
    except ValueError as exc:
        assert "params matched by no sub-optimizer" in str(exc)
    else:
        raise AssertionError("expected ValueError for params not covered by any params_filter")


def test_multi_optimizer_engine_train():
    config = Config(
        dict(
            task="train",
            device="cpu",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainTestModel,
            train_step=TrainTestModel.train_step,
            batch_size=500,
            torch_dataloader_opts={"num_workers": 0},
            optimizer={
                "class": "multi",
                "optimizers": [
                    {"class": "sgd", "params_filter": _multi_test_hidden_matrix_filter, "momentum": 0.9},
                    {"class": "adamw", "weight_decay": 1e-3},
                ],
            },
        )
    )
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "train"})
    dataset.init_seq_order(epoch=1)

    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()


def test_engine_schedule_free_optimizer_hooks():
    _RecordingScheduleFreeSGD.calls = []
    config = Config(
        dict(
            task="train",
            device="cpu",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainTestModel,
            train_step=TrainTestModel.train_step,
            batch_size=500,
            torch_dataloader_opts={"num_workers": 0},
            optimizer={"class": _RecordingScheduleFreeSGD},
        )
    )
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "train"})
    dataset.init_seq_order(epoch=1)

    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()

    calls = _RecordingScheduleFreeSGD.calls
    assert calls, "engine did not call the schedule-free optimizer train()/eval() hooks"
    assert calls[0] == "train" and calls[-1] == "eval", f"unexpected hook call sequence {calls}"


def test_multi_optimizer_schedule_free_forwarding():
    from returnn.torch.optim.multi import MultiOptimizer

    _RecordingScheduleFreeSGD.calls = []
    model = torch.nn.Linear(4, 3)
    sub1 = _RecordingScheduleFreeSGD([model.weight], lr=0.1)
    sub2 = torch.optim.AdamW([model.bias], lr=0.1)
    opt = MultiOptimizer(sub_optimizers=[sub1, sub2])
    opt.train()
    opt.eval()
    assert _RecordingScheduleFreeSGD.calls == ["train", "eval"]


def test_amuse_optimizer():
    from returnn.torch.optim.amuse import AMUSE

    config = Config(dict(optimizer={"class": "amuse", "update_type": "adamw", "warmup_steps": 5}))
    model = torch.nn.Linear(4, 3)
    updater = Updater(config=config, network=model, device=torch.device("cpu"))
    updater.create_optimizer()
    updater.set_learning_rate(1e-3)
    updater.set_current_train_step(global_train_step=0, epoch=1)

    opt = updater.get_optimizer()
    assert isinstance(opt, AMUSE)
    assert opt.update_type == "adamw"
    opt.train()
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    updater.step()
    opt.eval()
    assert all("z" in opt.state[p] for p in model.parameters())

    with tempfile.TemporaryDirectory(prefix="returnn_test_amuse_optimizer") as tmp_dir:
        updater.save_optimizer(tmp_dir + "/model.opt.pt")
        updater.load_optimizer(tmp_dir + "/model.opt.pt")


def test_amuse_engine_train():
    # Also tests the engine schedule-free hooks: AMUSE raises in step() if not in train mode.
    config = Config(
        dict(
            task="train",
            device="cpu",
            extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
            get_model=TrainTestModel,
            train_step=TrainTestModel.train_step,
            batch_size=500,
            torch_dataloader_opts={"num_workers": 0},
            optimizer={"class": "amuse", "update_type": "adamw", "warmup_steps": 5},
        )
    )
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "train"})
    dataset.init_seq_order(epoch=1)

    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=dataset)
        engine.train()
        # The engine must have switched to eval mode at the train epoch end,
        # so params (and thus any saved checkpoint) hold the averaged weights.
        assert engine._updater.get_optimizer().train_mode is False


def test_multi_optimizer_contract():
    import copy
    import io

    from returnn.torch.optim.multi import MultiOptimizer

    model = torch.nn.Linear(4, 3)
    sub1 = torch.optim.SGD([model.weight], lr=0.1, momentum=0.9)
    sub2 = torch.optim.AdamW([model.bias], lr=0.1)
    opt = MultiOptimizer(sub_optimizers=[sub1, sub2])

    # Membership tests and get() must not mutate the state (unlike the MutableMapping defaults).
    assert model.weight not in opt.state
    assert opt.state.get(model.weight) is None
    assert len(opt.state) == 0

    # State view: auto-creates empty entries like the base class defaultdict,
    # mutations reach the owning sub-optimizer.
    assert opt.state[model.weight] == {}
    opt.state[model.weight]["marker"] = 1
    assert sub1.state[model.weight]["marker"] == 1
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    opt.step()
    assert "momentum_buffer" in opt.state[model.weight]
    assert "exp_avg" in opt.state[model.bias]
    assert len(opt.state) == 2
    opt.state.clear()
    assert len(sub1.state) == 0 and len(sub2.state) == 0 and len(opt.state) == 0

    # Hook registration via the base class machinery.
    if hasattr(opt, "register_step_pre_hook"):
        hook_calls = []
        handle = opt.register_step_pre_hook(lambda _opt, _args, _kwargs: hook_calls.append("pre"))
        opt.step()
        handle.remove()
        assert hook_calls == ["pre"]
    if hasattr(opt, "register_state_dict_pre_hook"):
        sd_hook_calls = []
        handles = [
            opt.register_state_dict_pre_hook(lambda _opt: sd_hook_calls.append("sd_pre")),
            opt.register_state_dict_post_hook(lambda _opt, _sd: sd_hook_calls.append("sd_post")),
            opt.register_load_state_dict_pre_hook(lambda _opt, _sd: sd_hook_calls.append("load_pre")),
            opt.register_load_state_dict_post_hook(lambda _opt: sd_hook_calls.append("load_post")),
        ]
        opt.load_state_dict(opt.state_dict())
        for handle in handles:
            handle.remove()
        assert sd_hook_calls == ["sd_pre", "sd_post", "load_pre", "load_post"]

    # deepcopy and pickle produce consistent objects.
    opt2 = copy.deepcopy(opt)
    assert len(opt2.sub_optimizers) == 2 and len(opt2.param_groups) == len(opt.param_groups)
    assert opt2.param_groups[0] is opt2.sub_optimizers[0].param_groups[0]
    buf = io.BytesIO()
    torch.save(opt, buf)
    buf.seek(0)
    opt3 = torch.load(buf, weights_only=False)
    assert len(opt3.sub_optimizers) == 2
    assert opt3.param_groups[0] is opt3.sub_optimizers[0].param_groups[0]

    try:
        opt.add_param_group({"params": [torch.nn.Parameter(torch.zeros(2))]})
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError from add_param_group")


def test_multi_optimizer_duplicate_params_error():
    from returnn.torch.optim.multi import MultiOptimizer

    model = torch.nn.Linear(4, 3)
    sub1 = torch.optim.SGD([model.weight], lr=0.1)
    sub2 = torch.optim.SGD([model.weight, model.bias], lr=0.2)
    try:
        MultiOptimizer(sub_optimizers=[sub1, sub2])
    except ValueError as exc:
        assert "disjoint" in str(exc)
    else:
        raise AssertionError("expected ValueError for overlapping sub-optimizer params")


def test_multi_optimizer_config_errors():
    config = Config(
        dict(
            decouple_constraints=False,
            optimizer={
                "class": "multi",
                "optimizers": [
                    {"class": "sgd", "params_filter": _multi_test_layer2_weight_filter, "momentum": 0.9},
                    {"class": "adamw", "weight_decay": 1e-3},
                ],
            },
        )
    )
    updater = Updater(config=config, network=_make_multi_test_model(), device=torch.device("cpu"))
    try:
        updater.create_optimizer()
    except AssertionError as exc:
        assert "decouple_constraints" in str(exc)
    else:
        raise AssertionError("expected AssertionError for decouple_constraints=False under multi")

    config = Config(
        dict(
            optimizer={
                "class": "multi",
                "optimizers": [
                    {
                        "class": "adamw",
                        "weight_decay": 1e-3,
                        "param_groups_custom": lambda **_kwargs: [],
                    },
                ],
            }
        )
    )
    updater = Updater(config=config, network=_make_multi_test_model(), device=torch.device("cpu"))
    try:
        updater.create_optimizer()
    except ValueError as exc:
        assert "param_groups_custom" in str(exc) and "params_filter" in str(exc)
    else:
        raise AssertionError("expected ValueError for param_groups_custom in sub-optimizer opts")


# must be in the global scope due to pickling
class _SubclassMultiOptimizer:
    """placeholder, replaced below (needs the import)"""


def _init_subclass_multi_optimizer():
    global _SubclassMultiOptimizer
    from returnn.torch.optim.multi import MultiOptimizer

    class _SubclassMultiOptimizerImpl(MultiOptimizer):
        """MultiOptimizer subclass for testing that the updater instantiates the resolved class."""

    _SubclassMultiOptimizerImpl.__name__ = "_SubclassMultiOptimizer"
    _SubclassMultiOptimizer = _SubclassMultiOptimizerImpl
    return _SubclassMultiOptimizerImpl


def test_multi_optimizer_subclass():
    import copy

    subclass = _init_subclass_multi_optimizer()
    config = Config(
        dict(
            optimizer={
                "class": subclass,
                "optimizers": [
                    {"class": "sgd", "params_filter": _multi_test_layer2_weight_filter, "momentum": 0.9},
                    {"class": "adamw", "weight_decay": 1e-3},
                ],
            }
        )
    )
    updater = Updater(config=config, network=_make_multi_test_model(), device=torch.device("cpu"))
    updater.create_optimizer()
    opt = updater.get_optimizer()
    assert type(opt) is subclass
    assert type(copy.deepcopy(opt)) is subclass


def test_updater_weight_decay_custom_include_check_local_name():
    # For backward compatibility, the callback receives the module-local param name
    # as full_param_name (despite the name), both in the single-optimizer and the multi case.
    seen_names = []

    def _include_check(*, full_param_name, **_kwargs):
        seen_names.append(full_param_name)
        return None

    config = Config(
        dict(
            optimizer={
                "class": "adamw",
                "weight_decay": 1e-3,
                "weight_decay_custom_include_check": _include_check,
            }
        )
    )
    updater = Updater(config=config, network=_make_multi_test_model(), device=torch.device("cpu"))
    updater.create_optimizer()
    assert seen_names and all("." not in name for name in seen_names), seen_names

    seen_names = []
    config = Config(
        dict(
            optimizer={
                "class": "multi",
                "optimizers": [
                    {"class": "sgd", "params_filter": _multi_test_layer2_weight_filter, "momentum": 0.9},
                    {
                        "class": "adamw",
                        "weight_decay": 1e-3,
                        "weight_decay_custom_include_check": _include_check,
                    },
                ],
            }
        )
    )
    updater = Updater(config=config, network=_make_multi_test_model(), device=torch.device("cpu"))
    updater.create_optimizer()
    assert seen_names and all("." not in name for name in seen_names), seen_names


def test_multi_optimizer_non_param_state_error():
    from returnn.torch.optim.multi import MultiOptimizer

    model = torch.nn.Linear(4, 3)
    opt = MultiOptimizer(
        sub_optimizers=[
            torch.optim.SGD([model.weight], lr=0.1, momentum=0.9),
            torch.optim.AdamW([model.bias], lr=0.1),
        ]
    )
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    opt.step()
    state_dict = opt.state_dict()
    state_dict["state"][999] = {"foo": 1}
    try:
        opt.load_state_dict(state_dict)
    except NotImplementedError as exc:
        assert "999" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError for non-parameter state key on load")


def test_amuse_legacy_group_keys_error():
    from returnn.torch.optim.amuse import AMUSE

    model = torch.nn.Linear(4, 3)
    for legacy_group_opts in ({"use_muon": True}, {"update_type": "muon"}, {"aux_update_type": "sgd"}):
        try:
            AMUSE([{"params": list(model.parameters()), **legacy_group_opts}], warmup_steps=5)
        except ValueError as exc:
            assert "no longer supported" in str(exc) or "Per-group update types" in str(exc)
        else:
            raise AssertionError(f"expected ValueError for legacy group opts {legacy_group_opts}")


def test_amuse_zero_lr():
    from returnn.torch.optim.amuse import AMUSE

    # With lr 0 throughout, all per-step averaging weights are zero, so ckp1 stays at its 1.0 fallback.
    # Past warmup, the beta1 ramp must not divide by (1 - ckp1) == 0 then.
    model = torch.nn.Linear(4, 3)
    opt = AMUSE(list(model.parameters()), lr=0.0, warmup_steps=2)
    opt.train()
    for _ in range(4):
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        opt.step()
    opt.eval()


def test_muon_update_higher_rank():
    from returnn.torch.optim.amuse import muon_update

    # 3D params are orthogonalized batch-wise over the last two dims,
    # so the update scaling must be based on those dims as well.
    grad = torch.randn(8, 1, 5)
    momentum = torch.zeros_like(grad)
    batched = muon_update(grad.clone(), momentum.clone(), aux_update_type="adamw")
    per_slice = torch.stack(
        [muon_update(grad[i].clone(), momentum[i].clone(), aux_update_type="adamw") for i in range(len(grad))]
    )
    assert torch.allclose(batched, per_slice, atol=1e-2)

    # Channels-last conv grads are non-contiguous, the 4D flatten must handle that.
    grad4 = torch.randn(8, 4, 3, 3)
    ref = muon_update(grad4.clone(), torch.zeros_like(grad4), aux_update_type="adamw")
    out = muon_update(
        grad4.clone().to(memory_format=torch.channels_last), torch.zeros_like(grad4), aux_update_type="adamw"
    )
    assert torch.allclose(out, ref, atol=1e-2)


def test_amuse_zero_lr_at_warmup_boundary():
    from returnn.torch.optim.amuse import AMUSE

    # An externally scheduled lr of 0 exactly at the warmup boundary step records c_warmup 0.
    # The next positive-lr step must re-anchor the beta1 ramp there,
    # not crash and not ramp away from beta1_init.
    model = torch.nn.Linear(4, 3)
    opt = AMUSE(list(model.parameters()), lr=0.1, warmup_steps=3)
    opt.train()
    betas = []
    for lr in (0.1, 0.1, 0.0, 0.1, 0.1):
        for group in opt.param_groups:
            group["lr"] = lr
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        opt.step()
        betas.append(opt.param_groups[0]["beta1"])
    opt.eval()
    assert all(opt.beta1_init <= beta < 1.0 for beta in betas), betas
    assert betas[-1] > opt.beta1_init, betas


def test_amuse_constructor_validation():
    from returnn.torch.optim.amuse import AMUSE

    model = torch.nn.Linear(4, 3)
    params = list(model.parameters())
    for bad_kwargs in (
        {"warmup_steps": 0},
        {"warmup_steps": 0.5},
        {"warmup_steps": 5, "beta1": 0.0},
        {"warmup_steps": 5, "beta1": 1.0},
        {"warmup_steps": 5, "rho": 2.0},
        {"warmup_steps": 5, "rho": -1.0},
    ):
        try:
            AMUSE(params, **bad_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad_kwargs}")

    # rho 0 is the fixed-beta1 AMUSE variant from the paper, must be accepted.
    AMUSE(params, warmup_steps=5, rho=0.0)

    # Muon needs matrix params, so the 1D bias must be rejected at construction, not at step time.
    try:
        AMUSE(params, warmup_steps=5, update_type="muon")
    except ValueError as exc:
        assert "ndim" in str(exc)
    else:
        raise AssertionError("expected ValueError for a 1D param with update_type muon")
    AMUSE([p for p in params if p.ndim >= 2], warmup_steps=5, update_type="muon")


def test_multi_optimizer_amuse():
    from returnn.torch.optim.amuse import AMUSE
    from returnn.torch.optim.multi import MultiOptimizer

    config = Config(
        dict(
            optimizer={
                "class": "multi",
                "optimizers": [
                    {
                        "class": "amuse",
                        "update_type": "muon",
                        "params_filter": _multi_test_hidden_matrix_filter,
                        "momentum": 0.95,
                        "weight_decay": 0.05,
                        "warmup_steps": 5,
                    },
                    {
                        "class": "amuse",
                        "update_type": "adamw",
                        "learning_rate_multiplier": 0.015,
                        "weight_decay": 0.05,
                        "warmup_steps": 5,
                    },
                ],
            }
        )
    )
    model = _make_multi_test_model()
    updater = Updater(config=config, network=model, device=torch.device("cpu"))
    updater.create_optimizer()
    updater.set_learning_rate(0.02)
    updater.set_current_train_step(global_train_step=0, epoch=1)

    opt = updater.get_optimizer()
    assert isinstance(opt, MultiOptimizer)
    muon_sub, adamw_sub = opt.sub_optimizers
    assert isinstance(muon_sub, AMUSE) and muon_sub.update_type == "muon"
    assert isinstance(adamw_sub, AMUSE) and adamw_sub.update_type == "adamw"
    assert all(pg["lr"] == 0.02 for pg in muon_sub.param_groups)
    assert all(pg["lr"] == 0.02 * 0.015 for pg in adamw_sub.param_groups)

    updater.set_optimizer_training_mode(train=True)
    assert muon_sub.train_mode and adamw_sub.train_mode
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    updater.step()
    updater.set_optimizer_training_mode(train=False)
    assert not muon_sub.train_mode and not adamw_sub.train_mode


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
