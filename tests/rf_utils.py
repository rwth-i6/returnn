"""
RETURNN frontend (returnn.frontend) utils
"""

from __future__ import annotations
import contextlib
import numpy

from returnn.config import Config, global_config_ctx
from returnn.util.pprint import pprint
import returnn.frontend as rf
from returnn.tensor import Tensor, TensorDict
import returnn.tf.compat as tf_compat
import returnn.torch.frontend as rft
import returnn.tf.frontend_layers as rfl
from returnn.tf.network import TFNetwork
from returnn.torch.data.tensor_utils import tensor_numpy_to_torch_, tensor_torch_to_numpy_


@contextlib.contextmanager
def tf_scope():
    """tf scope"""
    with tf_compat.v1.Graph().as_default(), tf_compat.v1.Session().as_default() as session:
        yield session


def run_model(extern_data: TensorDict, get_model: rf.GetModelFunc, forward_step: rf.StepFunc) -> TensorDict:
    """run"""
    for v in extern_data.data.values():
        _reset_tensor(v)
    rnd = numpy.random.RandomState(42)
    for v in extern_data.data.values():
        _fill_random(v, rnd=rnd)

    with rft.TorchBackend.random_journal_record() as random_journal:
        out_pt = run_model_torch(extern_data, get_model, forward_step)
        out_pt_raw = out_pt.as_raw_tensor_dict()  # get them now because dims might get overwritten

    with rfl.ReturnnLayersBackend.random_journal_replay(random_journal):
        out_tf = run_model_net_dict_tf(extern_data, get_model, forward_step)
        out_tf_raw = out_tf.as_raw_tensor_dict()

    print(out_pt, out_tf)
    assert set(out_pt.data.keys()) == set(out_tf.data.keys())
    for k, v_pt in out_pt.data.items():
        v_tf = out_tf[k]
        assert v_pt.dims == v_tf.dims
    assert set(out_pt_raw.keys()) == set(out_tf_raw.keys())
    for k, v_pt in out_pt_raw.items():
        v_tf = out_tf_raw[k]
        assert numpy.allclose(v_pt, v_tf, atol=1e-5, rtol=1e-5)
    return out_pt


def run_model_torch(extern_data: TensorDict, get_model: rf.GetModelFunc, forward_step: rf.StepFunc) -> TensorDict:
    """run"""
    extern_data_raw = extern_data.as_raw_tensor_dict()
    rf.select_backend_torch()
    rf.set_random_seed(42)

    # Inplace replace Numpy by Torch.
    # Inplace because dim tags cannot easily be copied (then they are not the same).
    # We recover extern_data in the end.
    for v in extern_data.data.values():
        tensor_numpy_to_torch_(v)

    model = get_model(epoch=1, step=0)
    rf.init_forward_step_run_ctx()
    forward_step(model=model, extern_data=extern_data)
    outputs = rf.get_run_ctx().outputs
    assert outputs.data
    for v in outputs.data.values():
        tensor_torch_to_numpy_(v)

    extern_data.assign_from_raw_tensor_dict_(extern_data_raw)
    return outputs


def run_model_net_dict_tf(extern_data: TensorDict, get_model: rf.GetModelFunc, forward_step: rf.StepFunc) -> TensorDict:
    """run"""
    extern_data_raw = extern_data.as_raw_tensor_dict()
    for v in extern_data.data.values():
        _reset_tensor(v)
    rf.select_backend_returnn_layers_tf()
    rf.set_random_seed(42)

    from returnn.tf.frontend_layers.config_entry_points import get_net_dict

    config = Config(
        {
            "debug_runtime_sanity_checks": True,
            "extern_data": extern_data,
            "get_model": get_model,
            "task": "forward",
            "forward_step": forward_step,
        }
    )

    with tf_scope() as session, global_config_ctx(config):
        net_dict = get_net_dict(epoch=1, step=0)
        print("*** TF net dict:")
        pprint(net_dict)
        outputs_layers = rf.get_run_ctx().outputs
        print("*** outputs:", outputs_layers)

        net = TFNetwork(config=config, train_flag=False)
        net.construct_from_dict(net_dict)
        session.run(tf_compat.v1.global_variables_initializer())

        outputs_tf = TensorDict()
        for k, v in outputs_layers.data.items():
            v: Tensor[rfl.Layer]
            assert isinstance(v.raw_tensor, rfl.Layer)
            layer_name = v.raw_tensor.get_abs_name()
            layer = net.get_layer(layer_name)
            outputs_tf.data[k] = layer.output.copy()

        fetches = outputs_tf.as_raw_tensor_dict()
        assert set(extern_data.data.keys()) == set(net.extern_data.data.keys())
        extern_data_tf_placeholders = net.extern_data.as_raw_tensor_dict()
        assert set(extern_data_tf_placeholders.keys()) == set(extern_data_raw.keys())
        feed_dict = {extern_data_tf_placeholders[k]: v for k, v in extern_data_raw.items()}

        outputs_numpy_raw = session.run(fetches, feed_dict=feed_dict)
        outputs_numpy = outputs_tf.copy_template()
        for v in outputs_numpy.data.values():
            _reset_tensor(v)
        outputs_numpy.assign_from_raw_tensor_dict_(outputs_numpy_raw)

    extern_data.assign_from_raw_tensor_dict_(extern_data_raw)
    return outputs_numpy


def _reset_tensor(x: Tensor):
    """reset"""
    x.raw_tensor = None
    for dim in x.dims:
        if dim.dyn_size_ext:
            _reset_tensor(dim.dyn_size_ext)


def _fill_random(x: Tensor, *, min_val: int = 0, rnd: numpy.random.RandomState) -> bool:
    """fill. return whether sth was filled"""
    filled = False
    while True:
        have_unfilled = False
        filled_this_round = False

        for dim in x.dims:
            if dim.is_batch_dim() and not dim.dyn_size_ext:
                dim.dyn_size_ext = Tensor("batch", [], dtype="int32")
            if not dim.dyn_size_ext:
                continue
            if _fill_random(dim.dyn_size_ext, min_val=2, rnd=rnd):
                filled = True
                filled_this_round = True
            if dim.dyn_size_ext.raw_tensor is None:
                have_unfilled = True
            elif not isinstance(dim.dyn_size_ext.raw_tensor, numpy.ndarray):
                have_unfilled = True

        if have_unfilled:
            assert filled_this_round, f"should have filled something, {x}"

        if not have_unfilled:
            break

    if x.raw_tensor is not None:
        if not isinstance(x.raw_tensor, numpy.ndarray):
            x.raw_tensor = None

    if x.raw_tensor is None:
        shape = [d.get_dim_value() for d in x.dims]
        if x.dtype.startswith("int"):
            max_val = 10
            if x.sparse_dim and x.sparse_dim.dimension is not None:
                max_val = x.sparse_dim.dimension
            x.raw_tensor = rnd.randint(min_val, max_val, size=shape, dtype=x.dtype)
        elif x.dtype.startswith("float"):
            x.raw_tensor = rnd.normal(0.0, 1.0, size=shape).astype(x.dtype)
        else:
            raise NotImplementedError(f"not implemented for {x} dtype {x.dtype}")
        filled = True

    assert isinstance(x.raw_tensor, numpy.ndarray)

    return filled
