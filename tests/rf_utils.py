"""
RETURNN frontend (returnn.frontend) utils
"""

from __future__ import annotations
from typing import Callable
import contextlib
import numpy
import torch

from returnn.config import Config, global_config_ctx
from returnn.util.pprint import pprint
import returnn.frontend as rf
from returnn.tensor import Tensor, TensorDict
import returnn.tf.compat as tf_compat
import returnn.tf.frontend_layers as rfl
from returnn.tf.network import TFNetwork, ExternData


@contextlib.contextmanager
def tf_scope():
    """tf scope"""
    with tf_compat.v1.Graph().as_default(), tf_compat.v1.Session().as_default() as session:
        yield session


def run_model(get_model: Callable[[], rf.Module], extern_data: Tensor):
    """run"""
    out_pt = run_model_torch(get_model, extern_data)
    out_tf = run_model_net_dict_tf(get_model, extern_data)
    # out_tf = out_tf.copy_compatible_to(out_pt, add_dims=False)  # TODO does not work yet, dim tags are not complete
    print(out_pt.raw_tensor.shape, out_tf.raw_tensor.shape)
    # TODO ...


def run_model_torch(get_model: Callable[[], rf.Module], extern_data: Tensor) -> Tensor:
    """run"""
    rf.select_backend_torch()
    rf.set_random_seed(42)
    _fill_random(extern_data)
    model = get_model()
    out = model(extern_data)
    assert isinstance(out, Tensor) and isinstance(out.raw_tensor, torch.Tensor)
    out_np = out.copy_template()
    out_np.raw_tensor = out.raw_tensor.detach().cpu().numpy()
    return out_np


def run_model_net_dict_tf(get_model: Callable[[], rf.Module], extern_data: Tensor):
    """run"""
    rf.select_backend_returnn_layers_tf()
    rf.set_random_seed(42)
    extern_data_name = extern_data.name

    def _get_extern_data() -> TensorDict:
        _fill_random(extern_data)
        rfl.register_extern_data(extern_data)
        return TensorDict([extern_data])

    # noinspection PyUnusedLocal
    def _get_model(*, epoch: int, step: int) -> rf.Module:
        return get_model()

    # noinspection PyShadowingNames
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        rf.init_forward_step_run_ctx()
        out = model(extern_data.data[extern_data_name])
        out.mark_as_default_output(shape=out.dims)

    from returnn.tf.frontend_layers.config_entry_points import get_net_dict

    config = Config(
        {
            "debug_runtime_sanity_checks": True,
        }
    )

    with tf_scope() as session, global_config_ctx(config):
        net_dict = get_net_dict(
            epoch=1, step=0, get_model_func=_get_model, extern_data=_get_extern_data, step_func=_forward_step
        )
        print("*** TF net dict:")
        pprint(net_dict)

        tf_extern_data = ExternData()
        tf_extern_data.set_batch_info(rfl.Layer.top().root.global_batch)
        net = TFNetwork(config=config, extern_data=tf_extern_data, train_flag=False)
        net.construct_from_dict(net_dict)
        out = net.get_default_output_layer().output

        session.run(tf_compat.v1.global_variables_initializer())
        out_v = session.run(out.placeholder)
        assert isinstance(out_v, numpy.ndarray)
        out = out.copy_template()
        out.raw_tensor = out_v
        return out


def _fill_random(x: Tensor, *, min_val: int = 0) -> bool:
    """fill. return whether sth was filled"""
    raw_tensor_type = rf.get_raw_tensor_type()
    filled = False
    while True:
        have_unfilled = False
        filled_this_round = False

        for dim in x.dims:
            if not dim.dyn_size_ext:
                continue
            if _fill_random(dim.dyn_size_ext, min_val=2):
                filled = True
                filled_this_round = True
            if dim.dyn_size_ext.raw_tensor is None:
                have_unfilled = True
            elif not isinstance(dim.dyn_size_ext.raw_tensor, raw_tensor_type):
                have_unfilled = True

        if have_unfilled:
            assert filled_this_round, f"should have filled something, {x}"

        if not have_unfilled:
            break

    if x.raw_tensor is not None:
        if not isinstance(x.raw_tensor, raw_tensor_type):
            x.raw_tensor = None

    if x.raw_tensor is None:
        if x.dtype.startswith("int"):
            max_val = 10
            if x.sparse_dim and x.sparse_dim.dimension is not None:
                max_val = x.sparse_dim.dimension
            x.raw_tensor = rf.random(
                dims=x.dims,
                dtype=x.dtype,
                sparse_dim=x.sparse_dim,
                minval=min_val,
                maxval=max_val,
                distribution="uniform",
            ).raw_tensor
        elif x.dtype.startswith("float"):
            x.raw_tensor = rf.random(
                dims=x.dims, dtype=x.dtype, sparse_dim=x.sparse_dim, distribution="normal", mean=0.0, stddev=1.0
            ).raw_tensor
        else:
            raise NotImplementedError(f"not implemented for {x} dtype {x.dtype}")
        filled = True

    assert isinstance(x.raw_tensor, raw_tensor_type)

    return filled
