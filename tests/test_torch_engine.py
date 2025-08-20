"""
Tests for PyTorch engine.
"""

from __future__ import annotations
import _setup_test_env  # noqa
from typing import Optional
import sys
import unittest
import tempfile
import numpy
import torch

from returnn.util import better_exchook
from returnn.config import Config, global_config_ctx
from returnn.tensor import TensorDict, Tensor
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
            orth_ = orth.tostring()
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
