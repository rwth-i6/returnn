"""
High-level backend for RETURNN layers
"""

from __future__ import annotations
from typing import Union, Sequence, Optional, Any, Tuple, Dict
import contextlib
import numpy
import tensorflow as tf
import returnn.tf.compat as tf_compat

from returnn.util.basic import NotSpecified
from returnn.tensor import Tensor, Dim

# noinspection PyProtectedMember
from returnn.frontend._backend import Backend

from .layer import Layer
from .. import frontend_layers as rfl
from ... import frontend as rf
from . import dims as _dims
from returnn.frontend import RawTensorTypes


# Ignore this warning until we really expect that we implemented everything.
# noinspection PyAbstractClass
class ReturnnLayersBackend(Backend[Layer]):
    """
    RETURNN layers backend (using TF), where raw_tensor represents a RETURNN layer
    """

    RawTensorType = Layer
    is_tensorflow = True

    @staticmethod
    def executing_eagerly() -> bool:
        """executing eagerly"""
        return False

    @staticmethod
    def set_random_seed(seed: int):
        """
        :param seed:
        """
        tf_compat.v1.set_random_seed(seed)

    @staticmethod
    def get_random_state() -> Dict[str, bytes]:
        """
        :return: random state
        """
        # Not sure... we could cover all state vars used by RandomLayer...
        return {}

    @staticmethod
    def set_random_state(state: Dict[str, bytes]):
        """
        :param state: as returned by :func:`get_random_state`.
            This might not always be successful (e.g. different hardware, different backend version),
            so the calling code should always have called set_random_seed before to have the random generators
            in a reasonable fallback state.
        """
        assert not state  # not implemented... unexpected currently

    @staticmethod
    def get_dtype_name_raw(raw_tensor: Layer) -> str:
        """dtype"""
        return raw_tensor.tensor.dtype

    @staticmethod
    def as_dtype_raw(dtype_name: str) -> str:
        """dtype"""
        return dtype_name

    @staticmethod
    def get_ndim_raw(raw_tensor: Layer) -> int:
        """ndim"""
        return raw_tensor.tensor.batch_ndim

    @staticmethod
    def get_shape_raw(raw_tensor: Layer) -> Layer:
        """shape"""
        raise NotImplementedError

    @staticmethod
    def get_shape_tuple_raw(raw_tensor: Layer) -> Tuple[Union[int, Layer]]:
        """shape"""
        raise NotImplementedError

    @staticmethod
    def get_known_shape_raw(raw_tensor: Layer) -> Tuple[Optional[int]]:
        """known shape"""
        return raw_tensor.tensor.batch_shape

    @staticmethod
    def set_known_shape_raw(raw_tensor: Layer, shape: Tuple[Optional[int]]) -> None:
        """set known shape"""
        pass  # ignore

    @staticmethod
    def fill_raw(shape: Union[Sequence[Union[int, Layer]], Layer], value: Union[Any, Layer]) -> Layer:
        """fill raw"""
        raise Exception("fill_raw not supported in layers backend because dim tags would be unknown")

    @staticmethod
    def compare_raw(a: Layer, kind: str, b: Layer) -> Layer:
        """compare"""
        raise Exception("compare_raw should not get called")

    @staticmethod
    def combine_raw(a: Layer, kind: str, b: Layer) -> Layer:
        """combine"""
        raise Exception("combine_raw should not get called")

    @staticmethod
    def reshape_raw(raw_tensor: Layer, shape: Union[Sequence[Union[int, Layer]], Layer]) -> Layer:
        """reshape_raw"""
        raise Exception("reshape_raw not supported in layers backend because dim tags would be unknown")

    @staticmethod
    def transpose_raw(raw_tensor: Layer, perm: Sequence[int]) -> Layer:
        """transpose_raw is a no-op in this backend"""
        return raw_tensor

    @staticmethod
    def activation(tensor: Tensor, func: str) -> Tensor:
        """activation"""
        return rfl.make_layer({"class": "activation", "activation": func, "from": tensor}, name=func)

    @staticmethod
    def activation_raw(raw_tensor: Layer, func: str) -> Layer:
        """activation"""
        return rfl.make_layer(
            {"class": "activation", "activation": func, "from": raw_tensor.tensor}, name=func
        ).raw_tensor

    @staticmethod
    def softmax(tensor: Tensor, *, axis: Dim) -> Tensor:
        """softmax"""
        return rfl.make_layer({"class": "softmax_over_spatial", "axis": axis, "from": tensor}, name="softmax")

    @staticmethod
    def log_softmax(tensor: Tensor, *, axis: Dim) -> Tensor:
        """log softmax"""
        return rfl.make_layer(
            {"class": "softmax_over_spatial", "axis": axis, "from": tensor, "log_space": True}, name="log_softmax"
        )

    @staticmethod
    def softmax_cross_entropy_with_logits(*, logits: Tensor, targets: Tensor, axis: Dim):
        """
        Efficient cross entropy.

        :param logits: target estimates given as inputs to softmax (i.e. unnormalized)
        :param targets: probabilities, i.e. normalized, can also be sparse
        :param axis: class labels dim over which softmax is computed
        :return: cross entropy (same Dims as 'logits' but without 'axis')
        """
        if targets.sparse_dim:
            assert axis == targets.sparse_dim and axis in logits.dims
            return rfl.make_layer(
                {
                    "class": "sparse_softmax_cross_entropy_with_logits",
                    "logits": logits,
                    "targets": targets,
                    "axis": axis,
                },
                name="sparse_softmax_cross_entropy_with_logits",
            )
        else:  # dense targets
            assert axis in targets.dims and axis in logits.dims
            log_probs = rf.log_softmax(logits, axis=axis)
            return -rf.matmul(targets, log_probs, reduce=axis)

    @staticmethod
    def sequence_mask_raw(lengths: Layer, *, batch_major: bool = True) -> Layer:
        """sequence mask"""
        raise NotImplementedError  # TODO

    @staticmethod
    def create_parameter_raw(tensor: rf.Parameter) -> Layer:
        """create parameter"""
        return rfl.make_layer(
            {"class": "variable", "shape": tensor.dims, "dtype": tensor.dtype}, name=tensor.name, existing_tensor=tensor
        ).raw_tensor

    @staticmethod
    def set_parameter_initial_value(param: rf.Parameter[Layer], value: Union[None, Tensor, rf.RawTensorTypes]) -> None:
        """set parameter initial value"""
        if value is None:
            param.raw_tensor.layer_dict.pop("init", None)
            param.raw_tensor.layer_dict.pop("init_by_layer", None)
        elif isinstance(value, Tensor):
            param.raw_tensor.layer_dict.pop("init", None)
            if not value.raw_tensor.parent.can_access_children_from_root:
                accessible_parent = value.raw_tensor.parent
                while not accessible_parent.can_access_children_from_root:
                    accessible_parent = accessible_parent.parent
                value.raw_tensor.assign_parent(accessible_parent)
                # We could also maybe move out all the dependencies.
                # However, it's not clear whether this is always safe.
                for dep in value.raw_tensor.get_tensor_dependencies():
                    assert (
                        dep.parent.can_access_children_from_root
                    ), f"dep {dep} of moved value {value} is not accessible"
            param.raw_tensor.layer_dict["init_by_layer"] = value
        else:
            param.raw_tensor.layer_dict.pop("init_by_layer", None)
            param.raw_tensor.layer_dict["init"] = value
        if rfl.is_debug_eager_mode_enabled():
            shape = [d.get_dim_value() for d in param.dims]
            if isinstance(value, Tensor):
                assert value.placeholder is not None
                value_tf = value.placeholder
            else:
                value_tf = tf.broadcast_to(tf.convert_to_tensor(value), shape)
            if param.raw_tensor.debug_layer.output.placeholder is None:
                var = tf.Variable(value_tf, shape=[d.get_dim_value() for d in param.dims], dtype=param.dtype)
                param.raw_tensor.debug_layer.output.placeholder = var
            else:
                var = param.raw_tensor.debug_layer.output.placeholder
                assert isinstance(var, tf.Variable)
                var.assign(value_tf)

    @staticmethod
    def convert_to_tensor(
        value: Union[Tensor, Layer, RawTensorTypes],
        *,
        dims: Sequence[Dim] = (),
        dtype: Optional[str] = None,
        sparse_dim: Optional[Dim] = None,
    ) -> Tensor[Layer]:
        """convert to tensor"""
        if isinstance(value, Tensor):
            return value
        kwargs = {}
        dim_deps = _dims.get_dim_deps(dims)
        if dim_deps:
            kwargs["shape_deps"] = dim_deps
        return rfl.make_layer(
            {"class": "constant", "value": value, "shape": dims, "dtype": dtype, **kwargs}, name="constant"
        )

    @classmethod
    def compare(
        cls,
        a: Union[Tensor, RawTensorTypes],
        kind: str,
        b: Union[Tensor, RawTensorTypes],
        *,
        allow_broadcast_all_sources: Optional[bool] = None,
        dim_order: Optional[Sequence[Dim]] = None,
    ) -> Tensor:
        """compare"""
        kwargs = {}
        if allow_broadcast_all_sources is not None:
            kwargs["allow_broadcast_all_sources"] = allow_broadcast_all_sources
        a = cls.convert_to_tensor(a)
        b = cls.convert_to_tensor(b)
        return rfl.make_layer({"class": "compare", "from": [a, b], "kind": kind, **kwargs}, name=kind)

    @classmethod
    def combine(
        cls,
        a: Union[Tensor, RawTensorTypes],
        kind: str,
        b: Union[Tensor, RawTensorTypes],
        *,
        allow_broadcast_all_sources: Optional[bool] = None,
        dim_order: Optional[Sequence[Dim]] = None,
    ) -> Tensor:
        """combine"""
        kwargs = {}
        if allow_broadcast_all_sources is not None:
            kwargs["allow_broadcast_all_sources"] = allow_broadcast_all_sources
        a = cls.convert_to_tensor(a)
        b = cls.convert_to_tensor(b)
        return rfl.make_layer({"class": "combine", "from": [a, b], "kind": kind, **kwargs}, name=kind)

    @staticmethod
    def gather(
        source: Tensor,
        *,
        indices: Union[Tensor, int],
        axis: Dim,
        clip_to_valid: bool = False,
    ) -> Tensor:
        """gather"""
        args = {}
        if clip_to_valid:
            args["clip_to_valid"] = clip_to_valid
        return rfl.make_layer(
            {"class": "gather", "from": source, "position": indices, "axis": axis, **args}, name="gather"
        )

    @staticmethod
    def matmul(a: Tensor, b: Tensor, *, reduce: Union[Dim, Sequence[Dim]]) -> Tensor:
        """matmul"""
        return rfl.make_layer({"class": "dot", "from": [a, b], "reduce": reduce}, name="matmul")

    @staticmethod
    def range_over_dim(dim: Dim) -> Tensor:
        """range over dim"""
        return rfl.make_layer(
            {"class": "range_in_axis", "from": _dims.get_dim_deps(dim), "axis": dim}, name="range_over_dim"
        )

    @staticmethod
    def reduce(
        source: Tensor, *, mode: str, axis: Union[Dim, Sequence[Dim]], use_time_mask: bool = NotSpecified
    ) -> Tensor:
        """Reduce"""
        assert mode in Backend._AllowedReduceModes
        kwargs = {}
        if use_time_mask is not NotSpecified:
            kwargs["use_time_mask"] = use_time_mask
        return rfl.make_layer(
            {"class": "reduce", "from": source, "mode": mode, "axis": axis, **kwargs}, name=f"reduce_{mode}"
        )

    @staticmethod
    @contextlib.contextmanager
    def random_journal_replay(journal: Sequence[Dict[str, Any]]):
        """
        Replays the journal.
        At exit, the journal is cleared, and we check that we replayed everything.
        """
        try:
            ReturnnLayersBackend._random_journal_replay_enabled = True
            ReturnnLayersBackend._random_journal_replay_idx = 0
            ReturnnLayersBackend._random_journal = journal
            yield
        finally:
            assert ReturnnLayersBackend._random_journal_replay_idx == len(journal)
            ReturnnLayersBackend._random_journal_replay_enabled = False
            ReturnnLayersBackend._random_journal_replay_idx = 0
            ReturnnLayersBackend._random_journal = None

    _random_journal_replay_enabled = False
    _random_journal_replay_idx = 0
    _random_journal = None  # type: Optional[Sequence[Dict[str, Any]]]

    @staticmethod
    def random(
        *,
        dims: Sequence[Dim],
        dtype: str,
        sparse_dim: Optional[Dim] = None,
        distribution: str,
        mean: Optional[Union[int, float, Tensor]] = None,
        stddev: Optional[Union[int, float, Tensor]] = None,
        bound: Optional[Union[int, float, Tensor]] = None,
        minval: Optional[Union[int, float, Tensor]] = None,
        maxval: Optional[Union[int, float, Tensor]] = None,
        seed: Optional[Union[int, Sequence[int], numpy.ndarray]] = None,
        algorithm: Optional[str] = None,
        explicit_state: Optional[Tensor] = None,
        auto_update_state: Optional[bool] = None,
        static: Optional[bool] = None,
        out: Optional[Tensor] = None,
    ) -> Tensor:
        """random"""
        if ReturnnLayersBackend._random_journal_replay_enabled:
            idx = ReturnnLayersBackend._random_journal_replay_idx
            ReturnnLayersBackend._random_journal_replay_idx += 1
            elem = ReturnnLayersBackend._random_journal[idx]
            assert isinstance(elem, dict)
            assert elem["dims"] == dims
            assert elem["dtype"] == dtype
            assert elem["sparse_dim"] == sparse_dim
            assert elem["distribution"] == distribution
            assert rfl.Layer.inner_control_flow() is None  # not implemented yet
            return rfl.make_layer(
                {
                    "class": "eval",
                    "from": (),
                    "eval": _random_replay_eval,
                    "eval_locals": {"idx": idx},
                    "out_type": {"dims": dims, "dtype": dtype, "sparse_dim": sparse_dim},
                },
                name="random_replay",
            )
        kwargs = {
            "mean": mean,
            "stddev": stddev,
            "bound": bound,
            "minval": minval,
            "maxval": maxval,
            "seed": seed,
            "algorithm": algorithm,
            "explicit_state": explicit_state,
            "auto_update_state": auto_update_state,
            "static": static,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return rfl.make_layer(
            {
                "class": "random",
                "shape": dims,
                "shape_deps": _dims.get_dim_deps(dims),
                "dtype": dtype,
                "sparse_dim": sparse_dim,
                "distribution": distribution,
                "stop_grad": True,
                **kwargs,
            },
            name="random",
        )


def _random_replay_eval(idx, **_kwargs):
    # noinspection PyProtectedMember
    elem = ReturnnLayersBackend._random_journal[idx]
    assert isinstance(elem, dict)
    out = elem["out"]
    assert isinstance(out, Tensor)
    assert isinstance(out.raw_tensor, numpy.ndarray)
    return tf.constant(out.raw_tensor, dtype=out.dtype)
