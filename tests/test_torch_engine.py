"""
Tests for PyTorch engine.
"""

import _setup_test_env  # noqa
import torch

from returnn.config import Config, global_config_ctx
from returnn.tensor import TensorDict, Tensor
from returnn.torch.engine import Engine
import returnn.frontend as rf
from returnn.forward_iface import ForwardCallbackIface
from returnn.datasets import init_dataset


def test_torch_engine():
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
            out = outputs["output"]
            assert isinstance(out, Tensor)
            assert out.batch_ndim == 2 and out.batch_shape[-1] == 9
            self.num_seqs += 1

        def finish(self):
            self.finish_called = True

    config = Config(
        dict(task="forward", extern_data={"data": {"dim": 9}}, get_model=_get_model, forward_step=_forward_step)
    )
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 100, "name": "dev", "fixed_random_seed": 1})
    callback = _ForwardCallback()

    with global_config_ctx(config):
        engine = Engine(config=config)
        engine.init_network_from_config()
        engine.forward_with_callback(callback=callback, dataset=dataset)
        assert callback.num_seqs == 100
        assert callback.init_called and callback.finish_called


def test_min_seq_len():

    from returnn.datasets.generating import DummyDataset

    config = Config({"min_seq_length": 2})
    dataset = DummyDataset(input_dim=1, output_dim=4, num_seqs=1, seq_len=1)
    engine = Engine(config=config)
    data_loader = engine._create_data_loader(dataset)
    for _ in data_loader:
        assert False, "Should not contain sequences"

    config = Config()
    dataset = DummyDataset(input_dim=1, output_dim=4, num_seqs=1, seq_len=3)
    engine = Engine(config=config)
    data_loader = engine._create_data_loader(dataset)
    for _ in data_loader:
        return
    assert False, "Should have contained sequences"


def test_max_seq_len():

    from returnn.datasets.generating import DummyDataset

    config = Config({"max_seq_length": 4})
    dataset = DummyDataset(input_dim=1, output_dim=4, num_seqs=1, seq_len=5)
    engine = Engine(config=config)
    data_loader = engine._create_data_loader(dataset)
    for _ in data_loader:
        assert False, "Should not contain sequences"

    config = Config()
    dataset = DummyDataset(input_dim=1, output_dim=4, num_seqs=1, seq_len=3)
    engine = Engine(config=config)
    data_loader = engine._create_data_loader(dataset)
    for _ in data_loader:
        return
    assert False, "Should have contained sequences"
