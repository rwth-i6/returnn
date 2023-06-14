"""
RETURNN frontend (returnn.frontend) utils
"""

from __future__ import annotations
from typing import Optional, Union, Dict
import contextlib
import re
import numpy
import numpy.testing
import tensorflow as tf

from returnn.config import Config, global_config_ctx
from returnn.util.pprint import pprint
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict
from returnn.tensor.utils import tensor_dict_fill_random_numpy_
import returnn.tf.compat as tf_compat
import returnn.torch.frontend as rft
import returnn.tf.frontend_layers as rfl
from returnn.tf.network import TFNetwork
from returnn.torch.data.tensor_utils import tensor_dict_numpy_to_torch_, tensor_dict_torch_to_numpy_


@contextlib.contextmanager
def tf_scope():
    """tf scope"""
    with tf_compat.v1.Graph().as_default(), tf_compat.v1.Session().as_default() as session:
        yield session


def run_model(
    extern_data: TensorDict,
    get_model: rf.GetModelFunc,
    forward_step: rf.StepFunc,
    *,
    dyn_dim_max_sizes: Optional[Dict[Dim, int]] = None,
    dyn_dim_min_sizes: Optional[Dict[Dim, int]] = None,
    test_tensorflow: bool = True,
) -> TensorDict:
    """run"""
    print(f"* run_model with dyn_dim_max_sizes={dyn_dim_max_sizes!r}")
    extern_data.reset_content()
    tensor_dict_fill_random_numpy_(
        extern_data, dyn_dim_max_sizes=dyn_dim_max_sizes, dyn_dim_min_sizes=dyn_dim_min_sizes
    )

    print("** run with PyTorch backend")
    with rft.TorchBackend.random_journal_record() as random_journal:
        out_pt = run_model_torch(extern_data, get_model, forward_step)
        _pad_mask_zeros(out_pt)
        # get the values now because dims might get overwritten
        out_pt_raw = out_pt.as_raw_tensor_dict(include_const_sizes=True)

    if not test_tensorflow:
        return out_pt

    print("** run with TensorFlow-net-dict backend")
    with rfl.ReturnnLayersBackend.random_journal_replay(random_journal):
        out_tf = run_model_net_dict_tf(extern_data, get_model, forward_step)
        _pad_mask_zeros(out_tf)
        out_tf_raw = out_tf.as_raw_tensor_dict(include_const_sizes=True)

    print(out_pt, out_tf)
    assert set(out_pt.data.keys()) == set(out_tf.data.keys())
    for k, v_pt in out_pt.data.items():
        v_tf = out_tf[k]
        # We cannot really check the dims directly for equality,
        # because the model code often creates new dims, which are different in each call.
        # However, via mark_as_output, the order of dims is well-defined.
        # So we can check the values.
        assert len(v_pt.dims) == len(v_tf.dims)
        assert v_pt.feature_dim_axis == v_tf.feature_dim_axis
        for d_pt, d_tf in zip(v_pt.dims, v_tf.dims):
            _check_dim(d_pt, d_tf)
        if v_pt.dtype.startswith("int"):
            assert v_tf.dtype.startswith("int")  # allow maybe different bit depth
        else:
            assert v_pt.dtype == v_tf.dtype
        assert bool(v_pt.sparse_dim) == bool(v_tf.sparse_dim)
        if v_pt.sparse_dim:
            _check_dim(v_pt.sparse_dim, v_tf.sparse_dim)
    assert set(out_pt_raw.keys()) == set(out_tf_raw.keys())
    for k, v_pt in out_pt_raw.items():
        v_tf = out_tf_raw[k]
        numpy.testing.assert_allclose(v_pt, v_tf, atol=1e-5, rtol=1e-5, err_msg=f"output {k!r} differs")
    return out_pt


def run_model_torch(extern_data: TensorDict, get_model: rf.GetModelFunc, forward_step: rf.StepFunc) -> TensorDict:
    """run"""
    extern_data_raw = extern_data.as_raw_tensor_dict(expected_type=numpy.ndarray)
    rf.select_backend_torch()
    rf.set_random_seed(42)

    # Inplace replace Numpy by Torch.
    # Inplace because dim tags cannot easily be copied (then they are not the same).
    # We recover extern_data in the end.
    tensor_dict_numpy_to_torch_(extern_data)

    model = get_model(epoch=1, step=0)
    rf.init_forward_step_run_ctx()
    forward_step(model=model, extern_data=extern_data)
    outputs = rf.get_run_ctx().outputs
    assert outputs.data
    tensor_dict_torch_to_numpy_(outputs)

    extern_data.assign_from_raw_tensor_dict_(extern_data_raw)
    return outputs


def run_model_net_dict_tf(extern_data: TensorDict, get_model: rf.GetModelFunc, forward_step: rf.StepFunc) -> TensorDict:
    """run"""
    extern_data_raw = extern_data.as_raw_tensor_dict(expected_type=numpy.ndarray)
    extern_data.reset_content()
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
        net_dict, model = get_net_dict(epoch=1, step=0)
        print("*** TF net dict:")
        pprint(net_dict)
        outputs_layers = rf.get_run_ctx().outputs
        print("*** outputs:", outputs_layers)

        net = TFNetwork(config=config, train_flag=False)
        net.construct_from_dict(net_dict)

        rf_params = {name.replace(".", "/"): p for name, p in model.named_parameters()}
        tf_params = {re.sub("/param:0$", "", p.name): p for p in net.get_params_list()}
        rf_params_not_in_tf = set(rf_params.keys()) - set(tf_params.keys())
        tf_params_not_in_rf = set(tf_params.keys()) - set(rf_params.keys())
        if rf_params_not_in_tf or tf_params_not_in_rf:
            raise Exception(
                "params not equal:\n"
                f"RF params not in TF: {rf_params_not_in_tf}\n"
                f"TF params not in RF: {tf_params_not_in_rf}"
            )

        session.run(tf_compat.v1.global_variables_initializer())

        outputs_tf = TensorDict()
        for k, v in outputs_layers.data.items():
            v: Tensor[rfl.Layer]
            assert isinstance(v.raw_tensor, rfl.Layer)
            layer_name = v.raw_tensor.get_abs_name()
            layer = net.get_layer(layer_name)
            outputs_tf.data[k] = layer.output.copy()

        fetches = outputs_tf.as_raw_tensor_dict(expected_type=tf.Tensor)
        assert set(extern_data.data.keys()) == set(net.extern_data.data.keys())
        extern_data_tf_placeholders = net.extern_data.as_raw_tensor_dict(expected_type=tf.Tensor)
        assert set(extern_data_tf_placeholders.keys()) == set(extern_data_raw.keys())
        feed_dict = {extern_data_tf_placeholders[k]: v for k, v in extern_data_raw.items()}

        outputs_numpy_raw = session.run(fetches, feed_dict=feed_dict)

        # Scalars are not Numpy arrays, but our code below assumes that we only have Numpy arrays.
        # So we convert them here.
        def _make_numpy_array(x):
            if isinstance(x, numpy.ndarray):
                return x
            return numpy.array(x)

        outputs_numpy_raw = {k: _make_numpy_array(v) for k, v in outputs_numpy_raw.items()}

        outputs_numpy = outputs_tf.copy_template()
        outputs_numpy.reset_content()
        outputs_numpy.assign_from_raw_tensor_dict_(outputs_numpy_raw)

    extern_data.assign_from_raw_tensor_dict_(extern_data_raw)
    return outputs_numpy


def _dim_is_scalar_size(dim: Dim) -> bool:
    """dim is scalar size"""
    if dim.size is not None:
        return True
    if dim.dyn_size_ext:
        return dim.dyn_size_ext.dims == ()
    return False


def _dim_scalar_size(dim: Dim) -> int:
    """dim scalar size"""
    if dim.size is not None:
        return dim.size
    if dim.dyn_size_ext:
        assert dim.dyn_size_ext.dims == ()
        return dim.dyn_size_ext.raw_tensor
    raise Exception(f"dim {dim} has no known size")


def _pad_mask_zeros(x: Union[TensorDict, Tensor, Dim]):
    if isinstance(x, TensorDict):
        for v in x.data.values():
            _pad_mask_zeros(v)
        return

    if isinstance(x, Dim):
        if x.dyn_size_ext:
            _pad_mask_zeros(x.dyn_size_ext)
        return

    assert isinstance(x, Tensor)
    for i, d in enumerate(x.dims):
        _pad_mask_zeros(d)
        if d.need_masking():
            mask = x.get_sequence_mask_tensor(i)
            if not set(mask.dims).issubset(set(x.dims)):
                print(f"Warning: cannot apply mask {mask} for dim {d} on tensor {x}.")
                continue
            mask = mask.copy_compatible_to(x, check_sparse=False, check_dtype=False)
            x.raw_tensor = numpy.where(mask.raw_tensor, x.raw_tensor, 0)


def _check_dim(d_pt: Dim, d_tf: Dim):
    assert isinstance(d_pt, Dim) and isinstance(d_tf, Dim)
    assert _dim_is_scalar_size(d_pt) == _dim_is_scalar_size(d_tf)
    if _dim_is_scalar_size(d_pt):
        assert _dim_scalar_size(d_pt) == _dim_scalar_size(d_tf)
    else:
        assert d_pt.dyn_size_ext and d_tf.dyn_size_ext
        # There might be cases where the dims are maybe not equal
        # (same reasoning as above, or also different order),
        # although this would be quite exotic.
        # Let's just assume for now that this does not happen.
        assert d_pt.dyn_size_ext.dims == d_tf.dyn_size_ext.dims
        assert (d_pt.dyn_size_ext.raw_tensor == d_tf.dyn_size_ext.raw_tensor).all()
