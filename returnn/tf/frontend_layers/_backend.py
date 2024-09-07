"""
High-level backend for RETURNN layers
"""

from __future__ import annotations
from typing import TypeVar, Union, Sequence, Optional, Any, Callable, Tuple, Dict, List
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

# noinspection PyProtectedMember
from returnn.frontend import _random_journal


S = TypeVar("S")  # any nested structure, can be None


# Ignore this warning until we really expect that we implemented everything.
# noinspection PyAbstractClass
class ReturnnLayersBackend(Backend[Layer]):
    """
    RETURNN layers backend (using TF), where raw_tensor represents a RETURNN layer
    """

    name = "returnn_layers_tf"
    RawTensorType = Layer
    is_tensorflow = True
    is_backend_raw_tensor_dim_tag_independent = False

    @staticmethod
    def executing_eagerly() -> bool:
        """executing eagerly"""
        return False

    @staticmethod
    def get_tensor_dependencies(x: Tensor[Layer]) -> Sequence[Tensor]:
        """get tensor inputs"""
        deps: List[Tensor] = []
        visited = set()
        for dep in x.raw_tensor.get_tensor_dependencies():
            if not dep.tensor or dep.tensor in visited:
                continue
            visited.add(dep.tensor)
            deps.append(dep.tensor)
        return deps

    @staticmethod
    def get_tensor_consumers(x: Tensor[Layer]) -> Sequence[Tensor]:
        """get tensor consumers"""
        usages: List[Tensor] = []
        visited = set()
        for use in x.raw_tensor.usages:
            if not use.tensor or use.tensor in visited:
                continue
            visited.add(use.tensor)
            usages.append(use.tensor)
        return usages

    @staticmethod
    def cond(pred: Tensor, true_fn: Callable, false_fn: Callable):
        """cond"""
        with rfl.Cond(pred) as cond:
            cond.true = true_fn()
            cond.false = false_fn()
        return cond.result

    @staticmethod
    def while_loop(
        cond: Callable[[S], Union[bool, Tensor]],
        body: Callable[[S], S],
        initial: S,
    ) -> S:
        """while loop"""

        # Have to put some arbitrary max_seq_len limit, otherwise the RecLayer will complain.
        # Note that we could put the `loop.end(..., include_eos=False)` at the beginning of the loop.
        # However, then all layers run one more iteration than you might expect,
        # specifically the last iteration will be run with end being True (condition being False).
        # There are multiple cases when this can be problematic:
        #  - Layers with side-effects.
        #  - Gathering or similar where the indices can get out of bounds.
        # When we have `loop.end(..., include_eos=True)` at the end of the loop, this is fixed.
        # However, the case where the `cond(...)` is False already in the very beginning is wrong now.
        # To check for this, we have to add an additional `cond(...)` around the loop.
        def _loop():
            loop = rfl.Loop(max_seq_len=rf.constant(2**31 - 1, dims=()))
            loop.state.state = initial
            with loop:
                loop.state.state = body(loop.state.state)
                loop.end(rf.logical_not(cond(loop.state.state)), include_eos=True)
            return loop.state.state

        return ReturnnLayersBackend.cond(cond(initial), _loop, lambda: initial)

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
    def get_shape_tuple_raw(raw_tensor: Layer) -> Tuple[Union[int, Layer], ...]:
        """shape"""
        raise NotImplementedError

    @staticmethod
    def get_known_shape_raw(raw_tensor: Layer) -> Tuple[Optional[int], ...]:
        """known shape"""
        return raw_tensor.tensor.batch_shape

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
    def make_output_tensor(tensor: Tensor, dims: Sequence[Dim], *, name: str) -> Tensor:
        """only func where we have explicitly defined dim order in the output"""
        return rfl.make_layer({"class": "transpose", "from": tensor, "perm": dims}, name=name)

    @staticmethod
    def copy(tensor: Tensor) -> Tensor:
        """copy"""
        return rfl.make_layer({"class": "identity", "from": tensor}, name="copy")

    @staticmethod
    def cast(tensor: Tensor, dtype: str) -> Tensor:
        """cast"""
        return rfl.make_layer({"class": "cast", "from": tensor, "dtype": dtype}, name="cast")

    @staticmethod
    def set_requires_gradient(tensor: Tensor):
        """
        set requires gradient; not needed for TensorFlow, will always calculate whatever is needed
        """

    @staticmethod
    def gradient(y: Tensor, x: Tensor) -> Tensor:
        """gradient"""
        return rfl.make_layer({"class": "gradient", "y": y, "x": x}, name="gradient")

    @staticmethod
    def stop_gradient(tensor: Tensor) -> Tensor:
        """stop grad"""
        return rfl.make_layer({"class": "scaled_grad", "from": tensor, "scale": 0}, name="stop_gradient")

    @staticmethod
    def scaled_gradient(tensor: Tensor, scale: Union[float, Tensor]) -> Tensor:
        """scaled gradient"""
        return rfl.make_layer({"class": "scaled_grad", "from": tensor, "scale": scale}, name="scaled_gradient")

    @staticmethod
    def scaled_gradient_ext(
        x: Tensor,
        *,
        scale: Union[float, Tensor] = 1.0,
        shift: Optional[Union[float, Tensor]] = None,
        scale_shift_by_sum_over_axis: Optional[Dim] = None,
    ):
        """scaled gradient ext"""
        return rfl.make_layer(
            {
                "class": "scaled_grad",
                "from": x,
                "scale": scale,
                "shift": shift,
                "scale_shift_by_sum_over_axis": scale_shift_by_sum_over_axis,
            },
            name="scaled_gradient_ext",
        )

    @staticmethod
    def merge_dims(
        source: Tensor,
        *,
        dims: Sequence[Dim],
        out_dim: Optional[Dim] = None,
    ) -> Tuple[Tensor, Dim]:
        """
        Merges a list of axes into a single one. (Flatten the dims.)
        E.g. input is (batch, width, height, dim) and dims=(width,height), then we get (batch, width*height, dim).
        Or input is (batch, time, height, dim) and axes=(height,dim), then we get (batch, time, height*dim).

        :param source:
        :param dims:
        :param out_dim:
        :return: tensor, out_dim
        """
        if not isinstance(source, Tensor):
            raise TypeError(f"merge_dims: unexpected type for source {source!r}, need tensor")
        if out_dim is None:
            out_dim = dims[0]
            for d in dims[1:]:
                out_dim = out_dim * d
        layer = rfl.make_layer(
            {"class": "merge_dims", "from": source, "axes": dims, "out_dim": out_dim}, name="merge_dims"
        )
        return layer, out_dim

    @staticmethod
    def split_dims(
        source: Tensor,
        *,
        axis: Dim,
        dims: Sequence[Dim],
        pad_to_multiples: Optional[bool] = None,
        pad_value: Union[None, int, float] = None,
    ) -> Tensor:
        """split dims"""
        args = {}
        if pad_to_multiples is not None or pad_value is not None:
            args["pad_to_multiples"] = pad_to_multiples
            args["pad_value"] = pad_value
        args = {key: value for (key, value) in args.items() if value is not NotSpecified}
        return rfl.make_layer(
            {"class": "split_dims", "from": source, "axis": axis, "dims": dims, **args}, name="split_dims"
        )

    @staticmethod
    def reshape(source: Tensor, in_dims: Sequence[Dim], out_dims: Sequence[Dim]) -> Tensor:
        """reshape"""
        return rfl.make_layer(
            {
                "class": "reshape",
                "from": source,
                "in_dims": in_dims,
                "out_dims": out_dims,
                "extra_deps": rfl.get_dim_deps(out_dims),
            },
            name="reshape",
        )

    @staticmethod
    def split(source: Tensor, *, axis: Dim, out_dims: Sequence[Dim]) -> Tuple[Tensor, ...]:
        """split"""
        res = rfl.make_layer({"class": "split", "from": source, "axis": axis, "out_dims": out_dims}, name="split")

        src_axis_int = source.get_axis_from_description(axis)
        # noinspection PyProtectedMember
        return tuple(
            rfl._get_sub_layer(
                layer=res,
                name=str(i),
                data=source.copy_template_replace_dim_tag(
                    axis=src_axis_int, new_dim_tag=dim, name=f"{source.name}/split:{i}:{dim.description}"
                ),
            )
            for i, dim in enumerate(out_dims)
        )

    @staticmethod
    def expand_dim(source: Tensor, dim: Dim) -> Tensor:
        """expand dim"""
        # Some heuristic on the kind, which just determines where RETURNN puts the new axis.
        if source.have_feature_axis():
            axis = "spatial"
        elif dim.dimension is not None:
            axis = "feature"
        else:
            axis = "spatial"
        return rfl.make_layer({"class": "expand_dims", "from": source, "axis": axis, "dim": dim}, name="expand_dims")

    @staticmethod
    def squeeze(source: Tensor, axis: Dim) -> Tensor:
        """squeeze"""
        return rfl.make_layer({"class": "squeeze", "from": source, "axis": axis}, name="squeeze")

    @staticmethod
    def concat(
        *sources: Tuple[Tensor, Dim],
        allow_broadcast: bool = False,
        out_dim: Dim,
    ) -> Tensor:
        """concat"""
        opts = {}
        if allow_broadcast:
            opts["allow_broadcast"] = True
        out_dim = sum(d for _, d in sources)
        return rfl.make_layer(
            {"class": "concat", "from": sources, "out_dim": out_dim, **opts},
            name="concat",
        )

    @staticmethod
    def pad(
        source: Tensor,
        *,
        axes: Sequence[Dim],
        padding: Sequence[Tuple[Union[Dim, int], Union[Dim, int]]],
        out_dims: Sequence[Dim],
        handle_dynamic_dims: bool,
        mode: str = "constant",
        value: Union[rf.RawTensorTypes, Tensor] = None,
    ) -> Tensor:
        """pad"""
        assert isinstance(value, (int, float, type(None)))  # not implemented otherwise
        return rfl.make_layer(
            {
                "class": "pad",
                "from": source,
                "axes": axes,
                "padding": padding,
                "out_dims": out_dims,
                "handle_dynamic_dims": handle_dynamic_dims,
                "mode": mode,
                "value": value,
            },
            name="pad",
        )

    @staticmethod
    def cum_concat_step(source: Tensor, *, prev_accum: Tensor, axis: Dim, out_spatial_dim: Dim) -> Tensor:
        """cum_concat_step"""
        return rfl.make_layer(
            {
                "class": "cum_concat",
                "from": source,
                "state": {"state": prev_accum},
                "out_spatial_dim": out_spatial_dim,
                "axis": axis,
            },
            name="cum_concat",
        )

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
    def safe_log(tensor: Tensor, *, eps: float) -> Tensor:
        """safe log"""
        return rfl.make_layer(
            {"class": "activation", "activation": "safe_log", "eps": eps, "from": tensor}, name="safe_log"
        )

    @staticmethod
    def softmax(tensor: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
        """softmax"""
        args = {}
        if not use_mask:
            args["use_time_mask"] = False
        return rfl.make_layer({"class": "softmax_over_spatial", "axis": axis, "from": tensor, **args}, name="softmax")

    @staticmethod
    def log_softmax(tensor: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
        """log softmax"""
        args = {}
        if not use_mask:
            args["use_time_mask"] = False
        return rfl.make_layer(
            {"class": "softmax_over_spatial", "axis": axis, "from": tensor, "log_space": True, **args},
            name="log_softmax",
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
    def ctc_loss(
        *,
        logits: Tensor,
        logits_normalized: bool = False,
        targets: Tensor,
        input_spatial_dim: Dim,
        targets_spatial_dim: Dim,
        blank_index: int,
        max_approx: bool = False,
    ) -> Tensor:
        """CTC"""
        assert targets.sparse_dim and targets.sparse_dim.dimension <= logits.feature_dim.dimension
        logits = rfl.make_layer(
            {"class": "reinterpret_data", "from": logits, "set_axes": {"T": input_spatial_dim}}, name="logits"
        )
        targets = rfl.make_layer(
            {"class": "reinterpret_data", "from": targets, "set_axes": {"T": targets_spatial_dim}}, name="targets"
        )
        return rfl.make_layer(
            {
                "class": "ctc_loss",
                "logits": logits,
                "logits_normalized": logits_normalized,
                "targets": targets,
                "blank_index": blank_index,
                "max_approx": max_approx,
            },
            name="ctc_loss",
        )

    @staticmethod
    def create_parameter_raw(tensor: rf.Parameter, *, device: Optional[str] = None) -> Layer:
        """create parameter"""
        return rfl.make_layer(
            {
                "class": "variable",
                "shape": tensor.dims,
                "dtype": tensor.dtype,
                "param_name": "param",
                "param_device": device,
            },
            name=tensor.name,
            out=tensor,
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
    def set_parameter_trainable(param: rf.Parameter, trainable: bool) -> None:
        """set parameter trainable"""
        if trainable:
            # pop it, as it's the default
            param.raw_tensor.layer_dict.pop("trainable", None)
        else:
            param.raw_tensor.layer_dict["trainable"] = False

    @staticmethod
    def parameter_assign(param: rf.Parameter, value: Tensor, *, op: str = "assign") -> None:
        """param assign"""
        from .parameter_assign import parameter_assign

        parameter_assign(param=param, value=value, op=op)

    @staticmethod
    def convert_to_tensor(
        value: Union[Tensor, Layer, RawTensorTypes],
        *,
        dims: Sequence[Dim],
        dtype: str,
        sparse_dim: Optional[Dim] = None,
        device: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Tensor[Layer]:
        """convert to tensor"""
        if isinstance(value, Tensor):
            return value
        kwargs = {}
        if sparse_dim:
            kwargs["sparse_dim"] = sparse_dim
        dim_deps = _dims.get_dim_deps(dims)
        if dim_deps:
            kwargs["shape_deps"] = dim_deps
        return rfl.make_layer(
            {"class": "constant", "value": value, "shape": dims, "dtype": dtype, **kwargs},
            name=name or "convert_to_tensor",
        )

    @staticmethod
    def full(
        dims: Sequence[Dim],
        fill_value: Union[RawTensorTypes, Tensor],
        *,
        dtype: str,
        device: Optional[str] = None,
        sparse_dim: Optional[Dim] = None,
        feature_dim: Optional[Dim] = None,
    ) -> Tensor:
        """full"""
        if isinstance(fill_value, Tensor):
            return fill_value + rf.zeros_like(fill_value)
        kwargs = {}
        if sparse_dim:
            kwargs["sparse_dim"] = sparse_dim
        kwargs["feature_dim"] = feature_dim
        dim_deps = _dims.get_dim_deps(dims)
        if dim_deps:
            kwargs["shape_deps"] = dim_deps
        return rfl.make_layer(
            {"class": "constant", "value": fill_value, "shape": dims, "dtype": dtype, **kwargs}, name="full"
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
        a = rf.convert_to_tensor(a, _backend=cls)
        b = rf.convert_to_tensor(b, _backend=cls)
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
        a = rf.convert_to_tensor(a, _backend=cls)
        b = rf.convert_to_tensor(b, _backend=cls)
        return rfl.make_layer({"class": "combine", "from": [a, b], "kind": kind, **kwargs}, name=kind)

    @staticmethod
    def gather(source: Tensor, *, indices: Union[Tensor, int], axis: Dim, clip_to_valid: bool = False) -> Tensor:
        """gather"""
        args = {}
        if clip_to_valid:
            args["clip_to_valid"] = clip_to_valid
        return rfl.make_layer(
            {"class": "gather", "from": source, "position": indices, "axis": axis, **args}, name="gather"
        )

    @staticmethod
    def slice(
        source: Tensor,
        *,
        axis: Dim,
        start: Optional[Union[int, Tensor]] = None,
        end: Optional[Union[int, Tensor]] = None,
        step: Optional[Union[int, Tensor]] = None,
        size: Optional[Union[int, Tensor, Dim]] = None,
        out_dim: Dim,
    ) -> Tensor:
        """slice"""
        if size is not None:
            assert end is None  # not implemented
            assert step is None  # not implemented
            assert size is not None  # not implemented
            return rfl.make_layer(
                {
                    "class": "slice_nd",
                    "from": source,
                    "axis": axis,
                    "start": start,
                    "size": size,
                    "out_spatial_dim": out_dim,
                },
                name="slice_nd",
            )
        assert size is None  # not implemented
        assert isinstance(start, (int, type(None)))  # not implemented
        assert isinstance(end, (int, type(None)))  # not implemented
        assert isinstance(step, (int, type(None)))  # not implemented
        args = {}
        if start is not None:
            args["slice_start"] = start
        if end is not None:
            args["slice_end"] = end
        if step is not None:
            args["slice_step"] = step
        return rfl.make_layer(
            {"class": "slice", "from": source, "axis": axis, "out_dim": out_dim, **args}, name="slice"
        )

    @staticmethod
    def flip(source: Tensor, *, axis: Dim) -> Tensor:
        """flip"""
        return rfl.make_layer(
            {"class": "slice", "from": source, "axis": axis, "out_dim": axis, "slice_step": -1}, name="flip"
        )

    @staticmethod
    def where(
        cond: Tensor,
        true_: Union[Tensor, rf.RawTensorTypes],
        false_: Union[Tensor, rf.RawTensorTypes],
        *,
        allow_broadcast_all_sources: bool = False,
    ) -> Tensor:
        """where"""
        allow_broadcast_all_sources  # noqa # ignore allow_broadcast_all_sources for now..., not implemented
        return rfl.make_layer(
            {"class": "switch", "condition": cond, "true_from": true_, "false_from": false_}, name="where"
        )

    @staticmethod
    def clip_by_value(
        x: Tensor,
        clip_value_min: Union[Tensor, rf.RawTensorTypes],
        clip_value_max: Union[Tensor, rf.RawTensorTypes],
        *,
        allow_broadcast_all_sources: bool = False,
    ) -> Tensor:
        """clip by value"""
        clip_value_min = rf.convert_to_tensor(clip_value_min, _backend=ReturnnLayersBackend)
        clip_value_max = rf.convert_to_tensor(clip_value_max, _backend=ReturnnLayersBackend)
        return rfl.make_layer(
            {
                "class": "eval",
                "eval": "tf.clip_by_value(source(0), source(1), source(2))",
                "from": [x, clip_value_min, clip_value_max],
                "allow_broadcast_all_sources": allow_broadcast_all_sources,
            },
            name="clip_by_value",
        )

    @staticmethod
    def cumsum(source: Tensor, *, spatial_dim: Dim) -> Tensor:
        """cumsum"""
        return rfl.make_layer({"class": "cumsum", "from": source, "axis": spatial_dim}, name="cumsum")

    @staticmethod
    def matmul(a: Tensor, b: Tensor, *, reduce: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> Tensor:
        """matmul"""
        args = {}
        if not use_mask:
            args["use_mask"] = False
        return rfl.make_layer({"class": "dot", "from": [a, b], "reduce": reduce, **args}, name="matmul")

    @staticmethod
    def range_over_dim(dim: Dim, *, dtype: Optional[str] = None, device: Optional[str] = None) -> Tensor:
        """range over dim"""
        if not dtype and dim.dyn_size_ext:
            dtype = dim.dyn_size_ext.dtype
        if not dtype:
            dtype = rf.get_default_array_index_dtype()
        return rfl.make_layer(
            {
                "class": "range_in_axis",
                "from": _dims.get_dim_deps(dim),
                "axis": dim,
                "dtype": dtype,
                "sparse": dtype.startswith("int") or dtype.startswith("uint"),
            },
            name="range_over_dim",
        )

    @staticmethod
    def replace_dim(source: Tensor, *, in_dim: Dim, out_dim: Dim) -> Tensor:
        """
        :param source:
        :param in_dim:
        :param out_dim:
        :return: source with in_dim replaced by out_dim.
        """
        return rfl.make_layer(
            {"class": "reinterpret_data", "set_dim_tags": {in_dim: out_dim}, "from": source}, name="new_dim"
        )

    @staticmethod
    def reduce(source: Tensor, *, mode: str, axis: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> Tensor:
        """Reduce"""
        assert mode in Backend._AllowedReduceModes
        kwargs = {}
        if not use_mask:
            kwargs["use_time_mask"] = False
        return rfl.make_layer(
            {"class": "reduce", "from": source, "mode": mode, "axis": axis, **kwargs}, name=f"reduce_{mode}"
        )

    # noinspection PyShadowingBuiltins
    @staticmethod
    def top_k(
        source: Tensor,
        *,
        axis: Union[Dim, Sequence[Dim]],
        k: Union[int, Tensor],
        k_dim: Optional[Dim] = None,
        sorted: bool = True,
    ) -> Tuple[Tensor, Union[Tensor, Sequence[Tensor]], Dim]:
        """top_k"""
        if not k_dim:
            k_dim = Dim(k, name="top-k-dim")
        values = rfl.make_layer(
            {
                "class": "top_k",
                "from": source,
                "axis": axis,
                "k": k,
                "k_dim": k_dim,
                "sorted": sorted,
            },
            name="top_k",
        )
        if isinstance(axis, (tuple, list)):
            axes = axis
            single_axis = False
        else:
            assert isinstance(axis, Dim)
            axes = [axis]
            single_axis = True
        indices = []
        for i, a in enumerate(axes):
            assert isinstance(a, Dim)
            sub_name = "indices" if single_axis else f"indices{i}"
            indices_data = values.copy_template(name=f"{values.name}_{sub_name}_{a.description}")
            indices_data.dtype = "int32"
            indices_data.sparse_dim = a
            # noinspection PyProtectedMember
            indices.append(rfl._get_sub_layer(values, sub_name, data=indices_data))
        if single_axis:
            indices = indices[0]
        return values, indices, k_dim

    @staticmethod
    @contextlib.contextmanager
    def random_journal_replay(journal: _random_journal.RandomJournal):
        """
        Replays the journal.
        At exit, the journal is cleared, and we check that we replayed everything.
        """
        try:
            ReturnnLayersBackend._random_journal = journal
            yield
        finally:
            ReturnnLayersBackend._random_journal = None

    _random_journal = None  # type: Optional[_random_journal.RandomJournal]

    @staticmethod
    def random(
        *,
        dims: Sequence[Dim],
        dtype: str,
        device: Optional[str] = None,
        sparse_dim: Optional[Dim] = None,
        feature_dim: Optional[Dim] = None,
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
        if ReturnnLayersBackend._random_journal:
            recent = ReturnnLayersBackend._random_journal.get_recent_graph_reader_node_in_accessible_ctx()
            out = rfl.make_layer(
                {
                    "class": "eval",
                    "from": [recent] if recent else [],  # use as control dependency
                    "eval": _random_replay_eval,
                    "eval_locals": {"idx": ReturnnLayersBackend._random_journal.get_graph_reader_idx()},
                    "out_type": {"dims": dims, "dtype": dtype, "sparse_dim": sparse_dim, "feature_dim": feature_dim},
                },
                name="random_replay",
            )
            if out.control_flow_ctx and out.control_flow_ctx.is_loop():
                # Make sure it stays inside the loop.
                out.raw_tensor.layer_dict["from"].append(
                    rfl.PrevTensorRef.get_prev_ref(
                        cur_layer_name_ctx=out.raw_tensor,
                        initial=rf.zeros(dims, dtype=dtype, sparse_dim=sparse_dim, feature_dim=feature_dim),
                    )
                )
            ReturnnLayersBackend._random_journal.add_graph_reader_node(out)
            return out
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
                "feature_dim": feature_dim,
                "distribution": distribution,
                "stop_grad": True,
                **kwargs,
            },
            name="random",
        )

    @staticmethod
    def masked_select(
        tensor: Tensor, *, mask: Tensor, dims: Sequence[Dim], out_dim: Optional[Dim] = None
    ) -> Tuple[Tensor, Dim]:
        """
        :param tensor:
        :param mask:
        :param dims: the order of the dims defines the format. those dims should be exactly the dims of the mask.
        :param out_dim:
        :return: tensor where all dims in mask/dims are removed and replaced by a new dim.
            the new dim is also returned.
            if mask==True for all elements, the returned tensor would be simply the flattened input tensor.
        """
        assert mask.dtype == "bool"
        assert set(mask.dims) == set(dims)
        assert set(mask.dims).issubset(set(tensor.dims))
        if not out_dim:
            out_dim = Dim(None, name="mask")
        return (
            rfl.make_layer(
                {"class": "boolean_mask", "from": tensor, "mask": mask, "dims": dims, "out_dim": out_dim},
                name="boolean_mask",
            ),
            out_dim,
        )

    @staticmethod
    def batch_norm(
        source: Tensor,
        *,
        in_dim: Union[Dim, Sequence[Dim]],
        running_mean: Tensor,
        running_variance: Tensor,
        gamma: Optional[Tensor],
        beta: Optional[Tensor],
        epsilon: float,
        momentum: float,
        affine: bool,
        use_mask: bool,
    ) -> Tensor:
        """batch norm"""
        reuse_params = {
            "batch_norm/v2_mean": running_mean,
            "batch_norm/v2_variance": running_variance,
        }
        if affine:
            reuse_params["batch_norm/v2_gamma"] = gamma
            reuse_params["batch_norm/v2_beta"] = beta
        reuse_params = {"map": {k: {"layer_output": v} for k, v in reuse_params.items()}}
        return rfl.make_layer(
            {
                "class": "batch_norm",
                "from": source,
                "in_dim": in_dim,
                "use_std": affine,
                "use_shift": affine,
                "param_version": 2,
                "reuse_params": reuse_params,
                "momentum": momentum,
                "epsilon": epsilon,
                "masked_time": use_mask,
            },
            name="batch_norm",
        )

    # noinspection PyShadowingBuiltins
    @staticmethod
    def conv(
        source: Tensor,
        *,
        in_dim: Dim,
        out_dim: Dim,
        in_spatial_dims: Sequence[Dim],
        out_spatial_dims: Optional[Sequence[Dim]] = None,
        filter: Tensor,
        filter_size: Sequence[Dim],  # to have the order well-defined
        padding: str,
        strides: Optional[Union[int, Sequence[int]]] = None,
        dilation_rate: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[int] = None,
        bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Sequence[Dim]]:
        """conv"""
        if not out_spatial_dims:
            out_spatial_dims = rf.make_conv_out_spatial_dims(
                description_prefix="conv",
                in_spatial_dims=in_spatial_dims,
                filter_size=[d.dimension for d in filter_size],
                strides=strides or 1,
                dilation_rate=dilation_rate or 1,
                padding=padding,
            )
        layer_dict = {
            "class": "conv",
            "from": source,
            "in_dim": in_dim,
            "in_spatial_dims": in_spatial_dims,
            "out_dim": out_dim,
            "out_spatial_dims": out_spatial_dims,
            "filter_size": filter_size,
            "padding": padding,
        }
        if strides:
            layer_dict["strides"] = strides
        if dilation_rate:
            layer_dict["dilation_rate"] = dilation_rate
        if groups:
            layer_dict["groups"] = groups
        layer_dict.update({"filter": filter, "with_bias": bias is not None})
        if bias is not None:
            layer_dict["bias"] = bias
        out = rfl.make_layer(layer_dict, name="conv")
        return out, out_spatial_dims

    # noinspection PyShadowingBuiltins
    @staticmethod
    def transposed_conv(
        source: Tensor,
        *,
        in_dim: Dim,
        out_dim: Dim,
        in_spatial_dims: Sequence[Dim],
        out_spatial_dims: Optional[Sequence[Dim]] = None,
        filter: Tensor,
        filter_size: Sequence[Dim],
        padding: str,
        remove_padding: Union[Sequence[int], int] = 0,
        output_padding: Optional[Union[Sequence[Optional[int]], int]] = None,
        strides: Optional[Sequence[int]] = None,
        bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Sequence[Dim]]:
        """transposed conv"""
        if not out_spatial_dims:
            out_spatial_dims = [Dim(None, name=f"out-spatial-dim{i}") for i, s in enumerate(filter_size)]
            for i in range(len(filter_size)):
                s = filter_size[i].dimension if not strides else strides[i]
                if filter_size[i].dimension == s == 1 or (s == 1 and padding.lower() == "same"):
                    out_spatial_dims[i] = in_spatial_dims[i]
        layer_dict = {
            "class": "transposed_conv",
            "from": source,
            "in_dim": in_dim,
            "in_spatial_dims": in_spatial_dims,
            "out_dim": out_dim,
            "out_spatial_dims": out_spatial_dims,
            "filter_size": filter_size,
            "padding": padding,
        }
        if remove_padding:
            layer_dict["remove_padding"] = remove_padding
        if output_padding:
            layer_dict["output_padding"] = output_padding
        if strides:
            layer_dict["strides"] = strides
        layer_dict.update({"filter": filter, "with_bias": bias is not None})
        if bias is not None:
            layer_dict["bias"] = bias
        out = rfl.make_layer(layer_dict, name="transposed_conv")
        return out, out_spatial_dims

    @staticmethod
    def pool(
        source: Tensor,
        *,
        mode: str,
        pool_size: Sequence[int],
        padding: str = "valid",
        dilation_rate: Union[Sequence[int], int] = 1,
        strides: Sequence[int],
        in_spatial_dims: Sequence[Dim],
        out_spatial_dims: Optional[Sequence[Dim]] = None,
    ) -> Tuple[Tensor, Sequence[Dim]]:
        """pool"""
        if out_spatial_dims is None:
            out_spatial_dims = rf.make_conv_out_spatial_dims(
                description_prefix="pool",
                in_spatial_dims=in_spatial_dims,
                filter_size=pool_size,
                strides=strides,
                dilation_rate=dilation_rate,
                padding=padding,
            )
        other_dims = [d for d in source.dims if d not in in_spatial_dims and not d.is_batch_dim()]
        dummy_in_dim = None
        if source.feature_dim and source.feature_dim in other_dims:
            in_dim = source.feature_dim
        elif other_dims:
            in_dim = other_dims[-1]
        else:
            # PoolLayer currently needs some in_dim.
            dummy_in_dim = Dim(1, name="dummy_in")
            in_dim = dummy_in_dim
            source = rf.expand_dim(source, dim=dummy_in_dim)
        assert source.have_batch_axis(), "PoolLayer without batch dim not implemented"
        args = {
            "mode": mode,
            "pool_size": pool_size,
            "padding": padding,
            "dilation_rate": dilation_rate,
            "strides": strides,
            "in_spatial_dims": in_spatial_dims,
            "out_spatial_dims": out_spatial_dims,
            "in_dim": in_dim,  # it does not really matter, but we need sth currently
        }
        layer = rfl.make_layer({"class": "pool", "from": source, **args}, name="pool")
        if dummy_in_dim:
            layer = rf.squeeze(layer, axis=dummy_in_dim)
        if source.feature_dim != in_dim:
            # We want that the feature-dim stays consistent. PoolLayer currently just sets it to the in_dim.
            layer = rfl.make_layer(
                {"class": "reinterpret_data", "from": layer, "set_axes": {"F": source.feature_dim}},
                name="pool_reset_feature",
            )
        return layer, out_spatial_dims

    @staticmethod
    def stft(
        x: Tensor,
        *,
        in_spatial_dim: Dim,
        frame_step: int,
        frame_length: int,
        fft_length: int,
        window_use_frame_length: bool = True,
        align_window_left: bool = True,
        window_enforce_even: bool = True,
        out_spatial_dim: Dim,
        out_dim: Dim,
    ) -> Tensor:
        """stft"""
        if frame_length < fft_length:
            assert window_use_frame_length, "not implemented otherwise"
            assert align_window_left, "not implemented otherwise"
        if fft_length % 2 != 0:
            assert window_enforce_even, "not implemented otherwise"
        return rfl.make_layer(
            {
                "class": "stft",
                "from": x,
                "in_spatial_dims": [in_spatial_dim],
                "out_spatial_dims": [out_spatial_dim],
                "out_dim": out_dim,
                "frame_shift": frame_step,
                "frame_size": frame_length,
                "fft_size": fft_length,
            }
        )

    @staticmethod
    def lstm(
        source: Tensor,
        *,
        state_h: Tensor,
        state_c: Tensor,
        ff_weight: Tensor,
        rec_weight: Tensor,
        bias: Tensor,
        spatial_dim: Dim,
        in_dim: Dim,
        out_dim: Dim,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        :return: output, (h, c)
        """
        from ._utils import get_last_hidden_state

        # PyTorch (cuDNN) weights are in ifco order (?),
        # which we defined as the standard for the RF.
        # Also, they are (in_dim|out_dim, 4*out_dim).
        # RETURNN NativeLstm2 weight order: cell-in + input, forget and output gates (cifo).
        # And they are (4*out_dim, in_dim|out_dim).
        # So we need to reorder the params (ifco->cifo) and transpose them.
        # See also CustomCheckpointLoader and convert_cudnn_canonical_to_lstm_block.
        # TODO: ideally, we would create a new NativeLstm variant which just uses the same order.
        # Need new dim because after the split, we would get (out_dim,out_dim) which is ambiguous...
        out_dim_ = out_dim.copy(same_as_self=False, description="(out-dim)")
        rec_weight_ = rf.split(rec_weight, axis=4 * out_dim, out_dims=[out_dim_] * 4)
        ff_weight_ = rf.split(ff_weight, axis=4 * out_dim, out_dims=[out_dim_] * 4)
        bias_ = rf.split(bias, axis=4 * out_dim, out_dims=[out_dim_] * 4)
        rec_weight, _ = rf.concat(
            (rec_weight_[2], out_dim_),
            (rec_weight_[0], out_dim_),
            (rec_weight_[1], out_dim_),
            (rec_weight_[3], out_dim_),
        )
        ff_weight, _ = rf.concat(
            (ff_weight_[2], out_dim_), (ff_weight_[0], out_dim_), (ff_weight_[1], out_dim_), (ff_weight_[3], out_dim_)
        )
        bias, _ = rf.concat((bias_[2], out_dim_), (bias_[0], out_dim_), (bias_[1], out_dim_), (bias_[3], out_dim_))
        rec_weight, _ = rf.replace_dim(rec_weight, in_dim=4 * out_dim_, out_dim=4 * out_dim)
        ff_weight, _ = rf.replace_dim(ff_weight, in_dim=4 * out_dim_, out_dim=4 * out_dim)
        bias, _ = rf.replace_dim(bias, in_dim=4 * out_dim_, out_dim=4 * out_dim)

        output = rfl.make_layer(
            {
                "class": "rec",
                "from": source,
                "in_dim": in_dim,
                "axis": spatial_dim,
                "out_dim": out_dim,
                "unit": "lstm",
                "reuse_params": {
                    "map": {
                        "W_re": {"layer_output": rec_weight, "shape": (out_dim, 4 * out_dim)},
                        "W": {"layer_output": ff_weight, "shape": (in_dim, 4 * out_dim)},
                        "b": {"layer_output": bias, "shape": (4 * out_dim,)},
                    }
                },
                "initial_state": {"h": state_h, "c": state_c},
            },
            name="lstm",
        )
        h = get_last_hidden_state(output, out_dim=out_dim, key="h")
        c = get_last_hidden_state(output, out_dim=out_dim, key="c")
        return output, (h, c)


def _random_replay_eval(*, self, source, idx: int, **_kwargs):
    from returnn.tf.layers.basic import LayerBase

    assert isinstance(self, LayerBase)
    idx  # noqa  # unused - this can be helpful for debugging that the execution order is correct

    def _py_func() -> numpy.ndarray:
        # noinspection PyProtectedMember
        elem = ReturnnLayersBackend._random_journal.get_next(new_out_template=self.output)
        assert isinstance(elem.out, Tensor)
        assert isinstance(elem.out.raw_tensor, numpy.ndarray)
        return elem.out.raw_tensor

    def _func() -> tf.Tensor:
        (out,) = tf.numpy_function(_py_func, [], [self.output.dtype])
        assert isinstance(out, tf.Tensor)
        out.set_shape(self.output.batch_shape)
        return out

    with (
        tf.control_dependencies([source(i, auto_convert=False) for i in range(len(self.sources))])
        if self.sources
        else contextlib.nullcontext()
    ):
        return _func()
