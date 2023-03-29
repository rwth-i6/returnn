"""
RETURNN frontend (returnn.frontend) utils
"""

from __future__ import annotations
from typing import Callable
import contextlib
import numpy
import torch

from returnn.config import Config, global_config_ctx
import returnn.frontend as rf
from returnn.tensor import Tensor, TensorDict
import returnn.tf.compat as tf_compat
from returnn.tf.network import TFNetwork, ExternData
import returnn.tf.frontend_layers as rfl


@contextlib.contextmanager
def tf_scope():
    """tf scope"""
    with tf_compat.v1.Graph().as_default(), tf_compat.v1.Session().as_default() as session:
        yield session


def run_model(get_model: Callable[[], rf.Module], extern_data: Tensor):
    """run"""
    out_pt = run_model_torch(get_model, extern_data)
    out_tf = run_model_net_dict_tf(get_model, extern_data)
    out_pt, out_tf  # noqa  # TODO ...


def run_model_torch(get_model: Callable[[], rf.Module], extern_data: Tensor) -> Tensor:
    """run"""
    rnd = numpy.random.RandomState(42)
    rf.select_backend_torch()
    _fill_random(extern_data, rnd=rnd, raw_tensor_type=torch.Tensor)
    model = get_model()
    out = model(extern_data)
    assert isinstance(out, Tensor)
    return out


def run_model_net_dict_tf(get_model: Callable[[], rf.Module], extern_data: Tensor):
    """run"""
    rnd = numpy.random.RandomState(42)
    rf.select_backend_returnn_layers_tf()

    # noinspection PyUnusedLocal
    def _get_model(*, epoch: int, step: int) -> rf.Module:
        return get_model()

    # noinspection PyShadowingNames
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        out = model(extern_data)
        out.mark_as_default_output(shape=out.dims)

    extern_data_ = ExternData()
    extern_data_.register_data(extern_data)
    config = Config()

    from returnn.tf.frontend_layers.config_entry_points import get_net_dict

    with tf_scope() as session, global_config_ctx(config):
        _fill_random(extern_data, rnd=rnd, raw_tensor_type=rfl.Layer)

        net_dict = get_net_dict(
            epoch=1, step=0, get_model_func=_get_model, extern_data=extern_data_, step_func=_forward_step
        )

        net = TFNetwork(config=config, extern_data=extern_data_, train_flag=False)
        net.construct_from_dict(net_dict)
        out = net.get_default_output_layer().output

        session.run(tf_compat.v1.global_variables_initializer())
        out_v = session.run(out.placeholder)
        out = out.copy_template()
        out.raw_tensor = out_v
        return out


def _fill_random(x: Tensor, *, rnd: numpy.random.RandomState, raw_tensor_type: type) -> bool:
    """fill. return whether sth was filled"""
    filled = False
    while True:
        have_unfilled = False
        filled_this_round = False

        for dim in x.dims:
            if not dim.dyn_size_ext:
                continue
            if _fill_random(dim.dyn_size_ext, rnd=rnd, raw_tensor_type=raw_tensor_type):
                filled = True
                filled_this_round = True
            if dim.dyn_size_ext.raw_tensor is None:
                have_unfilled = True

        if have_unfilled:
            assert filled_this_round, f"should have filled something, {x}"

        if not have_unfilled:
            break

    if x.raw_tensor is not None:
        if not isinstance(x.raw_tensor, raw_tensor_type):
            x.raw_tensor = None

    if x.raw_tensor is None:
        shape = [dim.get_dim_value() for dim in x.dims]
        if x.dtype.startswith("int"):
            min_val = 0
            max_val = 10
            if x.sparse_dim and x.sparse_dim.dimension is not None:
                max_val = x.sparse_dim.dimension
            values = rnd.randint(min_val, max_val, shape).astype(x.dtype)
        elif x.dtype.startswith("float"):
            values = rnd.uniform(-1, 1, shape).astype(x.dtype)
        else:
            raise NotImplementedError(f"not implemented for {x} dtype {x.dtype}")
        x.raw_tensor = rf.convert_to_tensor(values, dims=x.dims, dtype=x.dtype, sparse_dim=x.sparse_dim).raw_tensor
        filled = True

    assert isinstance(x.raw_tensor, raw_tensor_type)

    return filled
