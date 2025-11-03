"""
RETURNN frontend (returnn.frontend) utils
"""

from __future__ import annotations
from typing import Optional, Union, Any, Dict, Tuple
import contextlib
import os
import re
import numpy
import numpy.testing

from returnn.config import Config, global_config_ctx
from returnn.util.pprint import pprint
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from returnn.tensor.utils import tensor_dict_fill_random_numpy_
import returnn.torch.frontend as rft
from returnn.torch.data.tensor_utils import tensor_dict_numpy_to_torch_, tensor_dict_torch_to_numpy_
from returnn.torch.util.debug_inf_nan import debug_inf_nan as torch_debug_inf_nan

# noinspection PyProtectedMember
from returnn.frontend._random_journal import RandomJournal


disable_tf = "RETURNN_DISABLE_TF" in os.environ and int(os.environ["RETURNN_DISABLE_TF"]) == 1

if disable_tf:
    tf = None
else:
    try:
        import tensorflow as tf
    except ImportError:
        tf = globals().get("tf", None)  # type: ignore

if tf:
    import returnn.tf.compat as tf_compat
    import returnn.tf.frontend_layers as rfl
    from returnn.tf.network import TFNetwork
else:
    tf_compat = rfl = TFNetwork = None


@contextlib.contextmanager
def tf_scope():
    """tf scope"""
    with tf_compat.v1.Graph().as_default(), tf_compat.v1.Session().as_default() as session:
        yield session


class RunModelException(Exception):
    """run model exception"""


class NonFiniteValuesException(RunModelException):
    """non-finite values exception"""


class CompareResultsMismatchException(RunModelException):
    """compare results exception"""


class CompareResultsMismatchTfVsPtException(CompareResultsMismatchException):
    """compare results TF vs PT exception"""


class CompareResultsMismatchSingleVsMultiBatchException(CompareResultsMismatchException):
    """compare results single vs multi batch exception"""


def run_model(
    extern_data: TensorDict,
    get_model: rf.GetModelFunc,
    forward_step: rf.StepFunc,
    *,
    dyn_dim_max_sizes: Optional[Dict[Dim, int]] = None,
    dyn_dim_min_sizes: Optional[Dict[Dim, int]] = None,
    test_tensorflow: bool = True,
    allow_inf_nan_in_output: bool = False,
    test_single_batch_entry: bool = True,
) -> TensorDict:
    """run"""
    print(f"* run_model with dyn_dim_max_sizes={dyn_dim_max_sizes!r}")
    extern_data_dims = extern_data.all_dims()
    extern_data.reset_content()
    tensor_dict_fill_random_numpy_(
        extern_data, dyn_dim_max_sizes=dyn_dim_max_sizes, dyn_dim_min_sizes=dyn_dim_min_sizes
    )

    print("** run with PyTorch backend")
    with rft.TorchBackend.random_journal_record() as random_journal:
        out_pt = _run_model_torch(extern_data, get_model, forward_step)
        _pad_mask_zeros(out_pt)
        # get the values now because dims might get overwritten
        out_pt_raw = out_pt.as_raw_tensor_dict(include_const_sizes=True)

    if not allow_inf_nan_in_output:
        non_finite_outputs = {}
        for k, v in out_pt.data.items():
            if not numpy.isfinite(v.raw_tensor).all():
                non_finite_outputs[k] = v
                print(f"ERROR: output {k!r} has non-finite values:\n{v.raw_tensor}")
        if non_finite_outputs:
            torch_debug_inf_nan(
                lambda: (_run_model_torch(extern_data, get_model, forward_step), None)[-1],
                stop_reporting_after_first_inf_nan=False,
            )
            raise NonFiniteValuesException(f"Non-finite values in output: {non_finite_outputs}. See log above.")

    if test_single_batch_entry and batch_dim in extern_data_dims:
        dyn_dims = [
            d
            for d in extern_data_dims
            if d.dyn_size_ext is not None
            and d.dyn_size_ext.dims == (batch_dim,)
            and d.dyn_size_ext.raw_tensor.min() != d.dyn_size_ext.raw_tensor.max()
        ]
        if dyn_dims:  # e.g. the typical time dim with dyn size shape [batch_dim]
            batch_idx = dyn_dims[0].dyn_size_ext.raw_tensor.argmin().item()
            print(f"** run with PyTorch backend with single batch entry for some smaller sequence {batch_idx=}")
            for d in dyn_dims:
                print(f"  {d}: {d.dyn_size_ext.raw_tensor}")
            _run_model_torch_single_batch(extern_data, get_model, forward_step, batch_idx=batch_idx, ref_output=out_pt)

    if not test_tensorflow:
        return out_pt
    if disable_tf:
        print("** TensorFlow disabled (RETURNN_DISABLE_TF)")
        return out_pt

    assert tf, "TensorFlow not available"
    print("** run with TensorFlow-net-dict backend")
    with rfl.ReturnnLayersBackend.random_journal_replay(random_journal):
        out_tf = _run_model_net_dict_tf(extern_data, get_model, forward_step)
        _pad_mask_zeros(out_tf)
        out_tf_raw = out_tf.as_raw_tensor_dict(include_const_sizes=True)

    random_journal: RandomJournal
    assert random_journal.reached_end()

    print("Output PT/TF:", out_pt, out_tf)
    assert set(out_pt.data.keys()) == set(out_tf.data.keys()), (
        f"PT output {list(out_pt.data.keys())} vs TF output {list(out_tf.data.keys())}"
    )
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
            assert v_pt.dtype == v_tf.dtype, f"PT dtype {v_pt.dtype} vs TF dtype {v_tf.dtype}"
        assert bool(v_pt.sparse_dim) == bool(v_tf.sparse_dim)
        if v_pt.sparse_dim:
            _check_dim(v_pt.sparse_dim, v_tf.sparse_dim)
    assert set(out_pt_raw.keys()) == set(out_tf_raw.keys())
    for k, v_pt in out_pt_raw.items():
        v_tf = out_tf_raw[k]
        print(f"  comparing {k!r} {_array_repr(v_pt)} PT vs TF")
        if not numpy.allclose(v_pt, v_tf, atol=1e-5, rtol=1e-5):
            print(f"  PT:\n{v_pt}")
            print(f"  TF:\n{v_tf}")
            raise CompareResultsMismatchTfVsPtException(f"output {k!r} differs")
    return out_pt


def _run_model_torch(extern_data: TensorDict, get_model: rf.GetModelFunc, forward_step: rf.StepFunc) -> TensorDict:
    """run"""
    extern_data_raw = extern_data.as_raw_tensor_dict(expected_value_type=numpy.ndarray)
    rf.select_backend_torch()
    rf.set_random_seed(42)

    # Inplace replace Numpy by Torch.
    # Inplace because dim tags cannot easily be copied (then they are not the same).
    # We recover extern_data in the end.
    tensor_dict_numpy_to_torch_(extern_data)

    for k, v in extern_data.data.items():
        if v.raw_tensor.dtype.is_floating_point:
            v.raw_tensor.requires_grad = True

    model = get_model(epoch=1, step=0)
    rf.init_forward_step_run_ctx(epoch=1, step=0)
    forward_step(model=model, extern_data=extern_data)
    outputs = rf.get_run_ctx().outputs
    assert outputs.data

    if "loss" in outputs.data:
        loss = outputs.data["loss"]
        assert isinstance(loss, Tensor)
        assert loss.raw_tensor.dtype.is_floating_point
        loss = rf.reduce_sum(loss, axis=loss.dims)
        print("loss:", loss.raw_tensor.detach().numpy().item())
        loss.raw_tensor.backward()
        for k, v in list(extern_data.data.items()):
            if v.raw_tensor.dtype.is_floating_point:
                assert v.raw_tensor.grad is not None, f"no grad for {k}"
                v_grad = v.copy_template()
                v_grad.raw_tensor = v.raw_tensor.grad
                assert f"{k}_grad" not in outputs.data
                outputs.data[f"{k}_grad"] = v_grad

    tensor_dict_torch_to_numpy_(outputs)

    extern_data.assign_from_raw_tensor_dict_(extern_data_raw)
    return outputs


def run_model_torch_train(
    extern_data: TensorDict,
    get_model: rf.GetModelFunc,
    train_step: rf.StepFunc,
    *,
    dyn_dim_max_sizes: Optional[Dict[Dim, int]] = None,
    dyn_dim_min_sizes: Optional[Dict[Dim, int]] = None,
) -> Dict[str, float]:
    """run"""
    rf.select_backend_torch()
    rf.set_random_seed(42)

    extern_data.reset_content()
    tensor_dict_fill_random_numpy_(
        extern_data, dyn_dim_max_sizes=dyn_dim_max_sizes, dyn_dim_min_sizes=dyn_dim_min_sizes
    )
    tensor_dict_numpy_to_torch_(extern_data)

    # We want to be able to calculate gradients for testing,
    # so we need to set requires_grad=True.
    for v in extern_data.data.values():
        v: Tensor
        v.raw_tensor.requires_grad = True

    model = get_model(epoch=1, step=0)
    rf.init_train_step_run_ctx(train_flag=True, step=0, epoch=1)
    train_step(model=model, extern_data=extern_data)
    total_loss = rf.get_run_ctx().total_loss()
    assert isinstance(total_loss, Tensor) and not total_loss.dims and total_loss.raw_tensor.dtype.is_floating_point
    total_loss_v = total_loss.raw_tensor.detach().numpy().item()
    print("total loss (for backprop):", total_loss_v)
    res = {"total_loss": total_loss_v}

    total_loss.raw_tensor.backward()  # test backprop

    for k, loss in rf.get_run_ctx().losses.items():
        loss_v = loss.get_summed_loss().raw_tensor.detach().cpu().numpy().item()
        res[f"{k}:summed"] = loss_v
        print(f"loss (summed) {k!r}: {loss_v}")
        loss_v = loss.get_mean_loss().raw_tensor.detach().cpu().numpy().item()
        print(f"loss (mean) {k!r}: {loss_v}")
        res[f"{k}:mean"] = loss_v
        inv_norm_factor = loss.get_inv_norm_factor()
        if isinstance(inv_norm_factor, Tensor):
            inv_norm_factor = inv_norm_factor.raw_tensor.detach().sum().cpu().numpy().item()
        print(f"inv_norm_factor {k!r}: {inv_norm_factor}")
        res[f"{k}:inv_norm_factor"] = inv_norm_factor

    return res


def _run_model_torch_single_batch(
    extern_data: TensorDict,
    get_model: rf.GetModelFunc,
    forward_step: rf.StepFunc,
    *,
    batch_idx: int,
    ref_output: TensorDict,
):
    """
    Restrict batch_dim to batch_idx only.
    This is somewhat hacky:
    We want to keep the same dim tags.
    Thus, we overwrite all the data by the sliced data.
    We also want to keep the batch_dim for the forward_step as the user code probably expects it.
    Afterward, we want to recover the original data.
    """
    # Store original data to be able to recover it later.
    extern_data_raw = extern_data.as_raw_tensor_dict(expected_value_type=numpy.ndarray)

    # noinspection PyShadowingNames
    def _get_slices(x: Tensor) -> Tuple[slice, ...]:
        slices = []
        for dim in x.dims:
            if dim == batch_dim:
                slices.append(slice(batch_idx, batch_idx + 1))
            elif dim.dyn_size_ext is not None and batch_dim in dim.dyn_size_ext.dims:
                slices.append(slice(0, dim.dyn_size_ext.raw_tensor.max().item()))
            else:
                slices.append(slice(None))
        return tuple(slices)

    # inplace
    # noinspection PyShadowingNames
    def tensor_numpy_restrict_batch_dim_(x: Tensor[numpy.ndarray]):
        if batch_dim not in x.dims:
            return
        if x.raw_tensor is not None:
            assert isinstance(x.raw_tensor, numpy.ndarray)
            x.raw_tensor = x.raw_tensor[_get_slices(x)]

    batch_dim.reset_eager()
    batch_dim.dyn_size_ext = Tensor("batch_size", dims=(), dtype="int32", raw_tensor=numpy.array(1, dtype="int32"))
    for dim in extern_data.all_dims():
        dim.transform_tensors(tensor_numpy_restrict_batch_dim_)
    for v in extern_data.data.values():
        tensor_numpy_restrict_batch_dim_(v)

    output = _run_model_torch(extern_data, get_model, forward_step)
    for key, ref_output_ in ref_output.data.items():
        output_ = output.data[key]
        if batch_dim not in ref_output_.dims:
            continue
        batch_axis = ref_output_.dims.index(batch_dim)
        assert output_.dims[batch_axis] == batch_dim and len(output_.dims) == len(ref_output_.dims)
        # Slice the raw ref output to be able to match it to the raw single output.
        ref_output_raw = ref_output_.raw_tensor[_get_slices(output_)]
        single_output_raw = output_.raw_tensor
        if not numpy.allclose(ref_output_raw, single_output_raw, atol=1e-5, rtol=1e-5):
            print(f"  Batched:\n{ref_output_raw}")
            print(f"  Single:\n{single_output_raw}")
            raise CompareResultsMismatchSingleVsMultiBatchException(f"output {key!r} differs")

    # Recover original data.
    extern_data.reset_content()
    extern_data.assign_from_raw_tensor_dict_(extern_data_raw)


def _run_model_net_dict_tf(
    extern_data: TensorDict, get_model: rf.GetModelFunc, forward_step: rf.StepFunc
) -> TensorDict:
    """run"""
    extern_data_raw = extern_data.as_raw_tensor_dict(expected_value_type=numpy.ndarray)
    extern_data.reset_content()
    rf.select_backend_returnn_layers_tf()
    rf.set_random_seed(42)

    from returnn.tf.frontend_layers.config_entry_points import get_net_dict

    # noinspection PyProtectedMember
    from returnn.frontend import _backend

    config = Config(
        {
            "debug_runtime_sanity_checks": True,
            # "debug_print_layer_output": True,  # enable temporarily for debugging
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

        _backend.select_backend_tf()
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

        if "loss" in outputs_tf.data:
            data_ = {name: data for name, data in net.extern_data.data.items() if data.dtype.startswith("float")}
            loss = outputs_tf.data["loss"]
            assert isinstance(loss, Tensor)
            assert loss.dtype.startswith("float")
            loss = rf.reduce_sum(loss, axis=loss.dims)
            d_grads = tf.gradients(loss.raw_tensor, [d.raw_tensor for d in data_.values()])
            for (name, data), d_grad_tf in zip(data_.items(), d_grads):
                assert isinstance(data, Tensor)
                assert isinstance(d_grad_tf, tf.Tensor)
                d_grad = data.copy_template()
                d_grad.raw_tensor = d_grad_tf
                outputs_tf.data[f"{name}_grad"] = d_grad

        fetches = outputs_tf.as_raw_tensor_dict(expected_value_type=tf.Tensor)
        assert set(extern_data.data.keys()) == set(net.extern_data.data.keys())
        extern_data_tf_placeholders = net.extern_data.as_raw_tensor_dict(expected_value_type=tf.Tensor)
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
    if dim.dyn_size_ext is not None:
        return dim.dyn_size_ext.dims == ()
    return False


def _dim_scalar_size(dim: Dim) -> int:
    """dim scalar size"""
    if dim.size is not None:
        return dim.size
    if dim.dyn_size_ext is not None:
        assert dim.dyn_size_ext.dims == ()
        return dim.dyn_size_ext.raw_tensor
    raise Exception(f"dim {dim} has no known size")


def _pad_mask_zeros(x: Union[TensorDict, Tensor, Dim]):
    if isinstance(x, TensorDict):
        for v in x.data.values():
            _pad_mask_zeros(v)
        return

    if isinstance(x, Dim):
        if x.dyn_size_ext is not None:
            _pad_mask_zeros(x.dyn_size_ext)
        return

    assert isinstance(x, Tensor)
    x.raw_tensor = x.copy_masked(0).raw_tensor
    for d in x.dims:
        _pad_mask_zeros(d)


def _check_dim(d_pt: Dim, d_tf: Dim):
    assert isinstance(d_pt, Dim) and isinstance(d_tf, Dim)
    assert d_pt.size == d_tf.size
    assert _dim_is_scalar_size(d_pt) == _dim_is_scalar_size(d_tf)
    if not _dim_is_scalar_size(d_pt):
        assert d_pt.dyn_size_ext is not None and d_tf.dyn_size_ext is not None
        # There might be cases where the dims are maybe not equal
        # (same reasoning as above, or also different order),
        # although this would be quite exotic.
        # Let's just assume for now that this does not happen.
        assert d_pt.dyn_size_ext.dims == d_tf.dyn_size_ext.dims
        # Do not compare the raw tensor directly here.
        # This will be done later for all the raw output.
        # The reason is that the different backends (TF, PT, etc)
        # might share the same dim tag but then reset them.


def _walk_dims(start: Dim, *, func=print):
    visited = set()  # ids
    queue = [((), start)]
    while queue:
        path, dim = queue.pop(0)
        path: Tuple[Any, ...]
        dim: Dim
        if id(dim) in visited:
            continue
        visited.add(id(dim))
        func(path, dim)
        # noinspection PyProtectedMember
        dim_extra = dim._extra
        if dim_extra:
            if dim_extra.cache_dim_math:
                for k, v in dim_extra.cache_dim_math.items():
                    k: Any
                    queue.append((path + ("_extra.cache_dim_math", k), v))
            if dim_extra.same_as:
                queue.append((path + ("_extra.same_as",), dim_extra.same_as))
            if dim_extra.derived_from_op:
                for i, v in enumerate(dim_extra.derived_from_op.inputs):
                    queue.append((path + ("_extra.derived_from_op.inputs", i), v))
            for k, v in dim_extra.same_for_batch_ctx.items():
                k: Any
                queue.append((path + ("_extra.same_for_batch_ctx", k), v))


def _array_repr(x: Union[numpy.ndarray, numpy.number]) -> str:
    if not isinstance(x, numpy.ndarray):
        return f"<{type(x).__name__} {x!r}>"

    try:
        import lovely_numpy

        return f"<{lovely_numpy.lovely(x)}>"
    except ImportError:
        if x.size <= 10:
            return repr(x)
        return f"<array shape={x.shape} dtype={x.dtype} min={x.min()} max={x.max()}>"
