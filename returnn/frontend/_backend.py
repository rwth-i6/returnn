"""
Backends for the frontend API
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, Union, TypeVar, Generic, Type, Callable, Sequence, Dict, Tuple
import contextlib
import numpy
import returnn.frontend as rf

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim
    from .types import RawTensorTypes

T = TypeVar("T")  # tf.Tensor, torch.Tensor or so
T2 = TypeVar("T2")


class Backend(Generic[T]):
    """
    Abstract base class for the backend, operating on tensor type T, i.e. :class:`Tensor[T]`.

    This class and instances do not have any state,
    and all functions are staticmethod (or classmethod).
    """

    # class attribs set by derived classes
    RawTensorType: Type[T]
    is_tensorflow: bool = False  # whether this framework uses TensorFlow

    def __init__(self):
        raise Exception("do not instantiate this class")

    # --- functions to override

    @staticmethod
    def executing_eagerly() -> bool:
        """
        :return: whether we are in eager execution mode
        """
        raise NotImplementedError

    @staticmethod
    def cond(pred: Tensor, true_fn: Callable, false_fn: Callable):
        """
        cond: conditional execution.

        Note that this does not need an implementation for eager-based frameworks
        (:func:`executing_eagerly` returns True),
        as the :func:`returnn.frontend.cond` function already covers that case.
        """
        # noinspection PyProtectedMember
        assert not pred._raw_backend.executing_eagerly(), "should not get here"
        raise NotImplementedError

    @staticmethod
    def set_random_seed(seed: int):
        """
        :param seed:
        """
        raise NotImplementedError

    @staticmethod
    def get_random_state() -> Dict[str, bytes]:
        """
        :return: random state
        """
        raise NotImplementedError

    @staticmethod
    def set_random_state(state: Dict[str, bytes]):
        """
        :param state: as returned by :func:`get_random_state`.
            This might not always be successful (e.g. different hardware, different backend version),
            so the calling code should always have called set_random_seed before to have the random generators
            in a reasonable fallback state.
        """
        raise NotImplementedError

    @staticmethod
    def get_dtype_name_raw(raw_tensor: T) -> str:
        """
        :return: dtype of raw tensor, as string
        """
        raise NotImplementedError

    @staticmethod
    def as_dtype_raw(dtype_name: str) -> Any:
        """
        :param dtype_name: e.g. "float32"
        :return: dtype object
        """
        raise NotImplementedError

    @staticmethod
    def get_ndim_raw(raw_tensor: T) -> int:
        """
        :return: ndim of raw tensor. assumes it is known
        """
        raise NotImplementedError

    @staticmethod
    def get_shape_raw(raw_tensor: T) -> Union[T, Tuple[Union[int, T]]]:
        """
        :return: shape of raw tensor
        """
        raise NotImplementedError

    @staticmethod
    def get_shape_tuple_raw(raw_tensor: T) -> Tuple[Union[int, T]]:
        """
        :return: shape of raw tensor. assumes that ndim is known.
            In eager frameworks, all dims are int.
        """
        raise NotImplementedError

    @staticmethod
    def get_known_shape_raw(raw_tensor: T) -> Tuple[Optional[int]]:
        """
        :return: shape of raw tensor, int for static known, None otherwise. assumes that ndim is known.
            This will not create any ops.
            In eager frameworks, all dims are known.
        """
        raise NotImplementedError

    @staticmethod
    def set_known_shape_raw(raw_tensor: T, shape: Tuple[Optional[int]]) -> None:
        """
        Sets the known shape of the raw tensor.
        This is only supported in graph-based frameworks,
        and just performs a check in eager frameworks.
        """
        # Default implementation for eager-based frameworks.
        backend = get_backend_by_raw_tensor_type(type(raw_tensor))
        existing_shape = backend.get_known_shape_raw(raw_tensor)
        assert all(dim is not None for dim in existing_shape)
        assert len(shape) == len(existing_shape)
        assert all(dim is None or dim == existing_shape[i] for i, dim in enumerate(shape))

    @staticmethod
    def get_new_dim_raw(raw_tensor: T, axis: int, *, name: str) -> Dim:
        """
        :param raw_tensor:
        :param axis:
        :param name:
        :return: dim tag of axis
        """
        raise NotImplementedError

    @staticmethod
    def get_device(x: Tensor) -> Optional[str]:
        """
        :param x:
        :return: device, or none if unknown or logic not supported
        """
        # default implementation: ignore device
        return None

    @staticmethod
    def copy_to_device(x: Tensor, device: Optional[str]) -> Tensor:
        """
        :param x: tensor
        :param device: e.g. "cpu" or "gpu"
        :return: tensor on device
        """
        # default implementation: ignore device
        return x

    @staticmethod
    def fill_raw(shape: Union[Sequence[Union[int, T]], T], value: Union[Any, T]) -> T:
        """
        :param shape: shape
        :param value: scalar value to fill
        :return: raw tensor filled with value everywhere
        """
        raise NotImplementedError

    @staticmethod
    def compare_raw(a: T, kind: str, b: T) -> T:
        """
        :param a:
        :param kind: "equal", "less", "less_equal", "greater", "greater_equal", "not_equal"
        :param b:
        :return: a `kind` b
        """
        raise NotImplementedError

    @staticmethod
    def combine_raw(a: T, kind: str, b: T) -> T:
        """
        :param a:
        :param kind: "add", "sub", "mul", "truediv", "floordiv", "mod", "pow",
            "maximum", "minimum", "logical_and", "logical_or", "squared_difference"
        :param b:
        :return: a `kind` b
        """
        raise NotImplementedError

    @staticmethod
    def reshape_raw(raw_tensor: T, shape: Union[Sequence[Union[int, T]], T]) -> T:
        """
        :param raw_tensor: raw tensor
        :param shape: new shape
        :return: reshaped raw tensor
        """
        raise NotImplementedError

    @classmethod
    def squeeze_raw(cls, raw_tensor: T, axes: Sequence[int]) -> T:
        """
        :param raw_tensor: raw tensor
        :param axes: axes to squeeze
        :return: squeezed raw tensor
        """
        # Default implementation using reshape_raw.
        known_shape = cls.get_known_shape_raw(raw_tensor)
        assert all([known_shape[axis] == 1 for axis in axes])
        new_shape = [dim for a, dim in enumerate(cls.get_shape_tuple_raw(raw_tensor)) if a not in axes]
        return cls.reshape_raw(raw_tensor, new_shape)

    @staticmethod
    def transpose_raw(raw_tensor: T, perm: Sequence[int]) -> T:
        """
        :param raw_tensor: raw tensor
        :param perm: permutation
        :return: transposed raw tensor
        """
        raise NotImplementedError

    @staticmethod
    def transpose(tensor: Tensor, perm: Sequence[Union[Dim, int]], *, allow_int: bool = False) -> Tensor:
        """
        :param tensor: tensor
        :param perm: permutation
        :param allow_int: allow int as axis in perm
        :return: transposed tensor
        """
        # Default implementation using transpose_raw.
        out, perm_ = tensor.copy_template_transpose(perm, allow_int=allow_int)
        backend = get_backend_by_tensor(tensor)
        out.raw_tensor = backend.transpose_raw(tensor.raw_tensor, perm_)
        return out

    @staticmethod
    def expand_dims_raw(raw_tensor: T, axis: int) -> T:
        """
        :param raw_tensor:
        :param axis:
        :return: raw tensor with new axis
        """
        raise NotImplementedError

    @staticmethod
    def expand_raw(raw_tensor: T, axis: int, dim: Union[int, T]) -> T:
        """
        :param raw_tensor:
        :param axis: shape[axis] must be 1
        :param dim: the new dim for shape[axis]
        :return: shape[axis] expands to dim.
            in PyTorch or other frameworks which support custom strides,
            this is an efficient view and not a copy.
        """
        raise NotImplementedError

    @staticmethod
    def copy(tensor: Tensor) -> Tensor:
        """copy"""
        raise NotImplementedError

    @staticmethod
    def cast_raw(raw_tensor: T, dtype: str) -> T:
        """
        :param raw_tensor:
        :param dtype: e.g. "float32"
        :return: raw tensor with dtype casted
        """
        raise NotImplementedError

    @staticmethod
    def cast(tensor: Tensor, dtype: str) -> Tensor:
        """
        :param tensor:
        :param dtype: e.g. "float32"
        :return: tensor with dtype casted
        """
        # Default implementation using cast_raw.
        res = tensor.copy_template()
        res.dtype = dtype
        # noinspection PyProtectedMember
        res.raw_tensor = tensor._raw_backend.cast_raw(tensor.raw_tensor, dtype)
        return res

    @staticmethod
    def stop_gradient(tensor: Tensor) -> Tensor:
        """
        :param tensor:
        :return: tensor with stopped gradient
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @staticmethod
    def split_dims(
        source: Tensor,
        *,
        axis: Dim,
        dims: Sequence[Dim],
        pad_to_multiples: Optional[bool] = None,
        pad_value: Union[None, int, float] = None,
    ) -> Tensor:
        """
        :param source:
        :param axis:
        :param dims:
        :param pad_to_multiples:
        :param pad_value:
        :return: source with axis replaced by dims
        """
        raise NotImplementedError

    @staticmethod
    def reshape(source: Tensor, in_dims: Sequence[Dim], out_dims: Sequence[Dim]) -> Tensor:
        """
        :param source: e.g. (..., old_dims, ...)
        :param in_dims: the old dims which should be reshaped into new_dims.
            This should only cover those dims which should be reshaped,
            not all the dims of the source.
        :param out_dims: the new dims which should be reshaped from old_dims.
            This is excluding any of the other dims in the source.
        :return: e.g. (..., new_dims, ...)
        """
        raise NotImplementedError

    @staticmethod
    def split(source: Tensor, *, axis: Dim, out_dims: Sequence[Dim]) -> Tuple[Tensor, ...]:
        """
        Split the input on the specified axis (by default feature).
        Basically a wrapper around tf.split.

        :param source: {..., axis}
        :param axis: some static axis
        :param out_dims: list of dims where sum(out_dims) == axis
        :return: tuple of tensors, same amount as out_dims,
            with the same shape as source, but with the specified axis replaced by the out_dims
        """
        raise NotImplementedError

    @staticmethod
    def expand_dim(source: Tensor, dim: Dim) -> Tensor:
        """
        :param source:
        :param dim:
        :return: source with dim added
        """
        raise NotImplementedError

    @staticmethod
    def concat(
        *sources: Tuple[Tensor, Dim],
        allow_broadcast: bool = False,
        out_dim: Dim,
    ) -> Tensor:
        """concat"""
        raise NotImplementedError

    @staticmethod
    def pad(
        source: Tensor,
        *,
        axes: Sequence[Dim],
        padding: Sequence[Tuple[Union[Dim, int], Union[Dim, int]]],
        out_dims: Sequence[Dim],
        mode: str = "constant",
        value: Optional[Union[rf.RawTensorTypes, Tensor]] = None,
    ) -> Tensor:
        """
        :param source:
        :param axes:
        :param padding:
        :param out_dims:
        :param mode:
        :param value:
        :return: padded tensor
        """
        raise NotImplementedError

    @staticmethod
    def cum_concat_step(source: Tensor, *, prev_accum: Tensor, axis: Dim, out_spatial_dim: Dim) -> Tensor:
        """
        Concatenates all previous frames over a time-axis.
        See RETURNN :class:`CumConcatLayer` for details.

        :param source: same dims as prev_accum except for the accum axis
        :param prev_accum: previous accumulated tensor, shape {..., axis}
        :param axis: the axis to accumulate over
        :param out_spatial_dim: the spatial dim of the output will be this dim. like axis+1.
        :return: accumulated. accumulated shape {..., out_spatial_dim},
            same shape as prev_accum with axis replaced by out_spatial_dim.
        """
        raise NotImplementedError

    # Restrict the possible activation function names,
    # to not get unexpected behavior,
    # or unwanted incompatibilities.
    _AllowedActivationFuncs = {
        "exp",
        "expm1",
        "log",
        "log1p",
        "sqrt",
        "rsqrt",
        "square",
        "abs",
        "tanh",
        "sigmoid",
        "log_sigmoid",
        "sin",
        "cos",
        "ceil",
        "floor",
        "round",
        "relu",
        "elu",
        "selu",
        "silu",
        "logical_not",
        "neg",
    }

    @staticmethod
    def activation(tensor: Tensor, func: str) -> Tensor:
        """
        :param tensor:
        :param func: "tanh", "sigmoid", "relu", ...
        :return: tensor with elementwise activation applied
        """
        out = tensor.copy_template()
        if func == "abs" and out.dtype.startswith("complex"):
            num_bits = int(out.dtype[len("complex") :])
            out.dtype = f"float{num_bits // 2}"
        # noinspection PyProtectedMember
        out.raw_tensor = tensor._raw_backend.activation_raw(tensor.raw_tensor, func)
        return out

    @staticmethod
    def activation_raw(raw_tensor: T, func: str) -> T:
        """
        :param raw_tensor:
        :param func: "tanh", "sigmoid", "relu", ...
        :return: raw tensor with elementwise activation applied
        """
        raise NotImplementedError

    @staticmethod
    def safe_log(tensor: Tensor, *, eps: float) -> Tensor:
        """
        :param tensor:
        :param eps:
        :return: log(tensor + eps) in the default case. but some backends might do more things,
            like if tensor = softmax(logits), then this would be log_softmax(logits) instead.
        """
        return rf.log(rf.maximum(tensor, eps))

    @staticmethod
    def softmax(tensor: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
        """
        :param tensor:
        :param axis:
        :param use_mask:
        :return: softmax over axis
        """
        raise NotImplementedError

    @staticmethod
    def log_softmax(tensor: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
        """
        :param tensor:
        :param axis:
        :param use_mask:
        :return: log_softmax over axis
        """
        raise NotImplementedError

    @staticmethod
    def softmax_cross_entropy_with_logits(*, logits: Tensor, targets: Tensor, axis: Dim):
        """
        Efficient cross entropy.

        :param logits: target estimates given as inputs to softmax (i.e. unnormalized)
        :param targets: probabilities, i.e. normalized, can also be sparse
        :param axis: class labels dim over which softmax is computed
        :return: cross entropy (same Dims as 'logits' but without 'axis')
        """
        raise NotImplementedError

    @staticmethod
    def have_sequence_mask_raw() -> bool:
        """
        :return: whether we have a sequence_mask_raw implementation
        """
        return False

    @staticmethod
    def sequence_mask_raw(lengths: T, *, batch_major: bool = True) -> T:
        """
        Like tf.sequence_mask().

        :param lengths: shape (batch,)
        :param batch_major:
        :return: tensor mask of shape (batch,maxlen) if batch_major else (maxlen,batch) of type bool
        """
        raise NotImplementedError

    @staticmethod
    @contextlib.contextmanager
    def name_scope_raw(name: str) -> Any:
        """
        Default implementation for eager-based frameworks:
        Do nothing, tensors do not have a name.

        :param name:
        :return: context manager
        """
        # Default implementation for eager-based frameworks
        yield  # nothing to do

    @staticmethod
    @contextlib.contextmanager
    def control_dependencies_raw(dependencies: Sequence[Any]) -> Any:
        """
        Default implementation for eager-based frameworks:
        Do nothing, we expect that the dependencies are already executed.

        :param dependencies: raw tensors or ops
        :return: context manager
        """
        # Default implementation for eager-based frameworks
        yield

    @staticmethod
    def identity_with_control_dependencies_raw(raw_tensor: T, dependencies: Sequence[Any]) -> T:
        """
        Default implementation for eager-based frameworks:
        Do nothing, we expect that the dependencies are already executed.

        :param raw_tensor: raw tensor
        :param dependencies: raw tensors or ops
        :return: raw tensor
        """
        # Default implementation for eager-based frameworks
        return raw_tensor

    @staticmethod
    def create_placeholder_raw(tensor: Tensor) -> T:
        """
        :return: tf.placeholder in TF

        This is really only for TensorFlow for the deprecated option auto_create_placeholders
        and should not be used in other backends,
        even in graph-based backends.
        Rather, the logic to create placeholders should be done elsewhere.
        """
        raise Exception("create_placeholder not supported by backend")

    @staticmethod
    def create_parameter_raw(tensor: rf.Parameter) -> T:
        """
        :return: parameter (by default trainable)
        """
        raise NotImplementedError

    @staticmethod
    def set_parameter_initial_value(param: rf.Parameter, value: Union[None, Tensor, rf.RawTensorTypes]) -> None:
        """
        :param param: parameter
        :param value: initial value
        """
        raise NotImplementedError

    @staticmethod
    def set_parameter_trainable(param: rf.Parameter, trainable: bool) -> None:
        """
        :param param: parameter
        :param trainable: whether the parameter should be trainable
        """
        raise NotImplementedError

    @staticmethod
    def parameter_assign(param: rf.Parameter, value: Tensor, op: str = "assign") -> None:
        """
        :param param: parameter
        :param value: new value
        :param op: "assign" or "add"
        """
        raise NotImplementedError

    @staticmethod
    def runtime_sanity_checks(tensor: Tensor) -> Any:
        """
        Checks whether the tensor.raw_tensor is consistent with the tensor metadata.

        In graph-based frameworks (TF graph), we return some operation here.
        In eager frameworks, we would not return anything but instead directly perform the checks.
        """
        # By default, we do not do any checks. This is optional for the backend.
        pass

    @staticmethod
    def is_valid_in_current_graph(tensor: Tensor) -> bool:
        """
        :return: whether the raw tensor is valid in the current graph.
            In eager-mode frameworks, this is always true -- there is no graph.
        """
        return True

    @staticmethod
    def format_graph_output(raw_tensor: T, *, max_depth: Optional[int] = None) -> str:
        """
        :return: the computation graph leading to this tensor formatted.
            In eager-mode frameworks, this is not supported and returns None.
        """
        return "<no-graph>"

    @staticmethod
    def convert_to_tensor(
        value: Union[Tensor, T, RawTensorTypes],
        *,
        dims: Sequence[Dim],
        dtype: str,
        sparse_dim: Optional[Dim] = None,
        name: Optional[str] = None,
    ) -> Tensor[T]:
        """
        :param value: tensor, or scalar raw tensor or some other scalar value
        :param dims:
        :param dtype:
        :param sparse_dim:
        :param name:
        :return: tensor
        """
        raise NotImplementedError

    @staticmethod
    def full(
        dims: Sequence[Dim], fill_value: RawTensorTypes, *, dtype: str, sparse_dim: Optional[Dim] = None
    ) -> Tensor:
        """
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.full.html

        :param dims:
        :param fill_value:
        :param dtype:
        :param sparse_dim:
        :return: tensor
        """
        raise NotImplementedError

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
        """compare, default implementation using compare_raw"""
        from . import _utils

        out, a, b = _utils.bin_op_out_template(
            cls,
            a,
            b,
            name="compare",
            res_dtype="bool",
            allow_broadcast_all_sources=allow_broadcast_all_sources,
            dim_order=dim_order,
        )
        out.raw_tensor = cls.compare_raw(a.raw_tensor, kind, b.raw_tensor)
        return out

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
        """combine, default implementation using combine_raw"""
        from . import _utils

        out, a, b = _utils.bin_op_out_template(
            cls,
            a,
            b,
            name="combine",
            res_dtype=None,
            allow_broadcast_all_sources=allow_broadcast_all_sources,
            dim_order=dim_order,
        )
        out.raw_tensor = cls.combine_raw(a.raw_tensor, kind, b.raw_tensor)
        return out

    @staticmethod
    def gather(
        source: Tensor,
        *,
        indices: Union[Tensor, int],
        axis: Dim,
        clip_to_valid: bool = False,
    ) -> Tensor:
        """
        Gathers slices on a specified axis from the source using indices.
        If the source is of the shape ``[B,D,F1]``, and indices of shape ``[B,F2]``,
        this will yield output of the shape ``[B,F2,F1]`` where

        ``output[b,f2,f1] = source[b,indices[b,f2],f1]``

        (if ``D`` is the axis to gather from).
        In general, all shared axes of the input and the positions will be considered as batch-axes.

        The ``indices`` argument can also be an ``int``.
        In this case, this simply gives ``source[indices]`` on the specified ``axis``.

        :param source:
        :param indices: indices used to select the slices of the source from.
            If another tensor, must be of type ``int32`` or ``int64``.
            Can also specify a constant ``int``.
        :param axis: The axis into which we gather the indices into
        :param clip_to_valid: if True, the indices will be clipped to the valid range of the input
            Also taking seq lengths into account.
        :return: gathered values
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @staticmethod
    def matmul(a: Tensor[T], b: Tensor[T], *, reduce: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> Tensor[T]:
        """
        This performs a batched matmul of two sources a and b
        (non-batched matmul and dot product are special cases).
        The underlying operation is a batched matmul (shared..., I, J) * (shared..., J, K) -> (shared..., I, K).
        The inputs a and b are transformed internally into the required shapes in the following way:
        The axis J is specified via the Dim given as 'reduce'. If multiple reduce Dims are given the corresponding axes
        are merged into one before the matmul via a reshape. All other matching Dims in a and b will be treated as
        batch dimensions ('shared...'). Dims unique to a and b define the axes I and K, respectively. (Multiple or no
        unique axes in a and b are supported too.)

        Depending on which Dims exist in a, b and reduce this dot operation can be used to compute scaling, scalar
        product, outer product, matrix-vector multiplication, matrix-matrix multiplication etc. (all possibly batched).

        :param a:
        :param b:
        :param reduce: Dims over which to perform the product, have to be present in both a and b
        :param use_mask: If the reduction is over dynamic axes, to get the correct sum reduction,
            we need to apply masking to one of the inputs. This is done automatically.
            By disabling this flag, this would be disabled.
        :return: result of dot product, Dim order: common axes as sorted in a, unique axes of a (in order),
            unique axes of b (in order)
        """
        raise NotImplementedError

    @staticmethod
    def range_over_dim(dim: Dim, *, dtype: Optional[str] = None) -> Tensor[T]:
        """
        :param dim:
        :param dtype:
        :return: tensor with shape [dim]
        """
        raise NotImplementedError

    @staticmethod
    def replace_dim(source: Tensor, *, in_dim: Dim, out_dim: Dim) -> Tensor:
        """
        :param source:
        :param in_dim:
        :param out_dim:
        :return: source with in_dim replaced by out_dim.
        """
        # This default implementation works fine as long as the backend
        # does not have special treatments of Tensor and dim tags itself (like TF net dict backend).
        if not out_dim.is_dim_known():
            out_dim.copy_from(in_dim)
        out = source.copy_template_replace_dim_tag(axis=source.get_axis_from_description(in_dim), new_dim_tag=out_dim)
        out.raw_tensor = source.raw_tensor
        return out

    _AllowedReduceModes = {"sum", "max", "min", "mean", "logsumexp", "any", "all", "argmin", "argmax"}

    @staticmethod
    def reduce(
        source: Tensor[T],
        *,
        mode: str,
        axis: Union[Dim, Sequence[Dim]],
        use_mask: bool = True,
    ) -> Tensor[T]:
        """
        Reduce the tensor along the given axis

        :param source:
        :param mode: "sum", "max", "min", "mean", "logsumexp", "any", "all", "argmin", "argmax"
        :param axis:
        :param use_mask: if True (default), use the time mask (part of dim tag) to ignore padding frames
        :return: tensor with axis removed
        """
        raise NotImplementedError

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
        """
        random. See `rf.random` for details.
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        """
        :param source:
        :param in_dim:
        :param running_mean:
        :param running_variance:
        :param gamma:
        :param beta:
        :param epsilon:
        :param momentum:
        :param affine:
        :param use_mask:
        :return:
        """
        raise NotImplementedError

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
        """convolution"""
        raise NotImplementedError

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
        """transposed convolution"""
        raise NotImplementedError

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
        """pooling"""
        raise NotImplementedError

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
        """stft. see :func:`stft` for details."""
        raise NotImplementedError

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
        Functional LSTM.

        :param source: Tensor of shape [*, in_dim].
        :param state_c:
        :param state_h:
        :param ff_weight: Parameters for the weights of the feed-forward part.
        :param rec_weight: Parameters for the weights of the recurrent part.
        :param bias: Parameters for the bias.
        :param spatial_dim: Dimension in which the LSTM operates.
        :param in_dim:
        :param out_dim:
        :return: output, (state_h, state_c)
        """
        raise NotImplementedError


# We use a global instance, and we modify __class__ inplace,
# such that any reference to this can be updated.
# This is exposed to the user as `returnn.frontend`.
# The __class__ assignment is done in `select_engine`.
# Use object.__new__ because we disallow creating instances of Frontend.
global_backend = object.__new__(Backend)

_dispatch_table = {}  # type: Dict[Type, Type[Backend]]


def select_backend_tf():
    """
    Selects the RETURNN layers backend (based on TF).
    """
    import tensorflow as tf

    backend = get_backend_by_raw_tensor_type(tf.Tensor)  # side-effect: register it
    global_backend.__class__ = backend


def select_backend_returnn_layers_tf():
    """
    Selects the RETURNN layers backend (based on TF).
    """
    from returnn.tf.frontend_layers import Layer

    backend = get_backend_by_raw_tensor_type(Layer)  # side-effect: register it
    global_backend.__class__ = backend


def select_backend_torch():
    """
    Selects the PyTorch (low-level) backend.
    """
    import torch

    backend = get_backend_by_raw_tensor_type(torch.Tensor)  # side-effect: register it
    global_backend.__class__ = backend


def get_backend_by_tensor(tensor: Tensor, *, fallback: Optional[T2] = None) -> Union[Type[Backend[T]], T2]:
    """
    :param tensor:
    :param fallback:
    """
    if fallback and tensor.raw_tensor is None:
        return fallback
    assert tensor.raw_tensor is not None
    return get_backend_by_raw_tensor_type(type(tensor.raw_tensor))


def get_backend_by_raw_tensor_type(tensor_type: Type[T]) -> Union[Type[Backend[T]]]:
    """
    :param tensor_type:
    """
    if tensor_type in _dispatch_table:  # fast path
        return _dispatch_table[tensor_type]

    if not isinstance(tensor_type, type):
        raise TypeError(f"Expected type, got {tensor_type!r} of type {type(tensor_type)}")
    tensor_type: Type[T]

    # We don't register all possible subclasses in the dispatch table.
    # Check through the MRO.
    for base_type in tensor_type.__mro__:
        if base_type in _dispatch_table:
            # Also register it for faster future lookups.
            register_backend_by_tensor_type(tensor_type, _dispatch_table[base_type])
            return _dispatch_table[base_type]

    # It would be registered if there was any select_engine or select_backend_* call.
    # However, some code might not have done that, so for the common cases,
    # we do it here.
    for base_type in tensor_type.__mro__:
        if base_type.__module__.split(".")[0] == "tensorflow":
            from returnn.tf.frontend_low_level import TFBackend

            backend_type = TFBackend
            tensor_types = _get_tensor_types_tf()
        elif base_type.__module__.split(".")[0] == "torch":
            from returnn.torch.frontend import TorchBackend

            backend_type = TorchBackend
            tensor_types = _get_tensor_types_torch()
        elif base_type.__module__.startswith("returnn.tf.frontend_layers."):
            from returnn.tf.frontend_layers import ReturnnLayersBackend, Layer

            backend_type = ReturnnLayersBackend
            tensor_types = (Layer,)
        elif issubclass(base_type, numpy.ndarray):
            from ._numpy_backend import NumpyBackend

            backend_type = NumpyBackend
            tensor_types = (numpy.ndarray,)
        else:
            continue

        assert any(
            issubclass(base_type, type_) for type_ in tensor_types
        ), f"tensor type {tensor_type} base_type {base_type} not in {tensor_types}, expected for backend {backend_type}"
        for base_type_ in tensor_types:
            register_backend_by_tensor_type(base_type_, backend_type)
        return backend_type

    raise TypeError(f"unknown tensor type {tensor_type} with mro {tensor_type.__mro__}")


def register_backend_by_tensor_type(tensor_type: Type[T], backend: Type[Backend[T]]):
    """
    :param tensor_type:
    :param backend:
    """
    _dispatch_table[tensor_type] = backend


def _get_tensor_types_tf():
    """
    :return: tuple of relevant tensor types in TF.
        Note that it is not so important to cover all, as we also check issubclass as a fallback.
    """
    import tensorflow as tf

    ls = [tf.Tensor, tf.Variable]
    return tuple(ls)


def _get_tensor_types_torch():
    """
    :return: tuple of relevant tensor types in PyTorch.
        Note that it is not so important to cover all, as we also check issubclass as a fallback.
    """
    import torch

    ls = [torch.Tensor, torch.nn.Parameter]
    return tuple(ls)
