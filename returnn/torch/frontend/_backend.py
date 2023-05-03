"""
Backend for exposing PyTorch-specific functionality.
"""

from __future__ import annotations
from typing import Optional, Union, Any, Sequence, Tuple, List, Dict
import contextlib
import torch
import numpy

from returnn.tensor import Tensor, Dim
from returnn.util.basic import prod, get_global_inf_value

# noinspection PyProtectedMember
from returnn.frontend._backend import Backend
from returnn.frontend import RawTensorTypes
import returnn.frontend as rf


_TT = Tensor[torch.Tensor]


# Ignore this warning until we really expect that we implemented everything.
# noinspection PyAbstractClass
class TorchBackend(Backend[torch.Tensor]):
    """
    PyTorch backend
    """

    RawTensorType = torch.Tensor

    @staticmethod
    def executing_eagerly() -> bool:
        """
        :return: whether we are executing eagerly
        """
        return True

    @staticmethod
    def set_random_seed(seed: int):
        """
        :param seed:
        """
        torch.random.manual_seed(seed)

    @staticmethod
    def get_random_state() -> Dict[str, bytes]:
        """
        :return: random state
        """
        res = {
            "cpu": torch.random.get_rng_state().detach().cpu().numpy().tobytes(),
        }
        cuda_states = [state.detach().cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all()]
        if len(cuda_states) == 1:
            res["cuda"] = cuda_states[0]
        elif len(cuda_states) > 1:
            for i, state in enumerate(cuda_states):
                res[f"cuda{i}"] = state
        return res

    @staticmethod
    def set_random_state(state: Dict[str, bytes]):
        """
        :param state: as returned by :func:`get_random_state`.
            This might not always be successful (e.g. different hardware, different backend version),
            so the calling code should always have called set_random_seed before to have the random generators
            in a reasonable fallback state.
        """
        if "cpu" in state:
            torch.random.set_rng_state(torch.from_numpy(numpy.frombuffer(state["cpu"], dtype="uint8")))
        if "cuda" in state:
            torch.cuda.set_rng_state_all(torch.from_numpy(numpy.frombuffer(state["cuda"], dtype="uint8")))
        for k, v in state.items():
            if k.startswith("cuda"):
                i = int(k[4:])
                torch.cuda.set_rng_state(torch.from_numpy(numpy.frombuffer(v, dtype="uint8")), i)

    @staticmethod
    def get_dtype_name_raw(raw_tensor: torch.Tensor) -> str:
        """
        :return: dtype of raw tensor, as string
        """
        return str(raw_tensor.dtype).replace("torch.", "")

    @staticmethod
    def as_dtype_raw(dtype_name: str) -> torch.dtype:
        """
        :param dtype_name: e.g. "float32"
        :return: dtype object
        """
        dtype = getattr(torch, dtype_name)
        assert isinstance(dtype, torch.dtype)
        return dtype

    @staticmethod
    def get_ndim_raw(raw_tensor: torch.Tensor) -> int:
        """
        :return: ndim of raw tensor
        """
        return raw_tensor.dim()

    @staticmethod
    def get_known_shape_raw(raw_tensor: torch.Tensor) -> Tuple[Optional[int]]:
        """
        :return: shape of raw tensor; here for PyTorch the full shape is always known
        """
        return tuple(raw_tensor.size())

    @staticmethod
    def get_new_dim_raw(raw_tensor: torch.Tensor, axis: int, *, name: str) -> Dim:
        """
        :param raw_tensor:
        :param axis:
        :param name:
        :return: new Dim object
        """
        return Dim(raw_tensor.size(axis), name=name)

    @staticmethod
    def get_device(x: Tensor[torch.Tensor]) -> Optional[str]:
        """device"""
        if x.raw_tensor is None:
            return None
        return x.raw_tensor.device.type

    @staticmethod
    def copy_to_device(x: Tensor, device: Optional[str]) -> Tensor:
        """
        :param x:
        :param device:
        """
        if not device:
            return x
        x = x.copy()
        x.raw_tensor = x.raw_tensor.to(device)
        return x

    @staticmethod
    def expand_dims_raw(raw_tensor: torch.Tensor, axis: int) -> torch.Tensor:
        """
        :param raw_tensor:
        :param axis: e.g. 1
        :return: raw tensor with new axis
        """
        return raw_tensor.unsqueeze(axis)

    @staticmethod
    def copy(tensor: Tensor[torch.Tensor]) -> Tensor[torch.Tensor]:
        """copy"""
        out = tensor.copy_template()
        out.raw_tensor = tensor.raw_tensor.clone()
        return out

    @staticmethod
    def cast_raw(raw_tensor: torch.Tensor, dtype: str) -> torch.Tensor:
        """cast"""
        return raw_tensor.to(dtype=TorchBackend.as_dtype_raw(dtype))

    @staticmethod
    def stop_gradient(tensor: Tensor) -> Tensor:
        """stop grad"""
        out = tensor.copy()
        out.raw_tensor = out.raw_tensor.detach()
        return out

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
        assert dims
        if len(dims) == 1:
            return source, dims[0]
        first_axis = min(source.dims.index(d) for d in dims)
        pre_dims = source.dims[:first_axis]
        post_dims = [d for d in source.dims if d not in dims and d not in pre_dims]
        if out_dim is None:
            out_dim = dims[0]
            for d in dims[1:]:
                out_dim = out_dim * d
        source = source.copy_transpose(tuple(pre_dims) + tuple(dims) + tuple(post_dims), allow_int=False)
        out = Tensor(
            "merge_dims",
            dims=pre_dims + (out_dim,) + tuple(post_dims),
            dtype=source.dtype,
            sparse_dim=source.sparse_dim,
        )
        out_shape = [d.get_dim_value() for d in out.dims]
        out.raw_tensor = torch.reshape(source.raw_tensor, out_shape)
        return out, out_dim

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
        assert not axis.need_masking()  # not implemented
        assert pad_to_multiples in (None, False)  # not implemented
        axis_ = source.get_axis_from_description(axis)
        out_dims = source.dims[:axis_] + tuple(dims) + source.dims[axis_ + 1 :]
        out_shape = [d.get_dim_value() for d in out_dims]
        out_raw = torch.reshape(source.raw_tensor, out_shape)
        return Tensor(
            "split_dims",
            dims=out_dims,
            dtype=source.dtype,
            sparse_dim=source.sparse_dim,
            raw_tensor=out_raw,
        )

    @staticmethod
    def reshape(source: Tensor, in_dims: Sequence[Dim], out_dims: Sequence[Dim]) -> Tensor:
        """reshape"""
        in_dims_axes = [source.get_axis_from_description(d, allow_int=False) for d in in_dims]
        assert sorted(set(in_dims_axes)) == sorted(in_dims_axes), f"reshape {source}: invalid in_dims {in_dims}"
        insert_axis = min(in_dims_axes)
        dims = list(source.dim_tags)
        permute = list(range(source.batch_ndim))
        for axis in sorted(set(in_dims_axes), reverse=True):
            dims.pop(axis)
            permute.pop(axis)
        permute = permute[:insert_axis] + in_dims_axes + permute[insert_axis:]
        source = source.copy_transpose(permute)
        dims = dims[:insert_axis] + list(out_dims) + dims[insert_axis:]
        out = Tensor("reshape", dims=dims, dtype=source.dtype, sparse_dim=source.sparse_dim)
        if source.feature_dim and source.feature_dim not in in_dims:
            out.feature_dim = source.feature_dim
        out.raw_tensor = torch.reshape(source.placeholder, [d.get_dim_value() for d in dims])
        return out

    @staticmethod
    def split(source: Tensor, *, axis: Dim, out_dims: Sequence[Dim]) -> Tuple[Tensor, ...]:
        """split"""
        src_axis_int = source.get_axis_from_description(axis)
        out_raw_list = torch.split(
            source.raw_tensor,
            split_size_or_sections=[d.get_dim_value() for d in out_dims],
            dim=src_axis_int,
        )
        out_tuple = tuple(
            source.copy_template_replace_dim_tag(axis=src_axis_int, new_dim_tag=dim, name=f"split{i}")
            for i, dim in enumerate(out_dims)
        )
        for i, out in enumerate(out_tuple):
            out.raw_tensor = out_raw_list[i]
        return out_tuple

    @staticmethod
    def expand_dim(source: Tensor, dim: Dim) -> Tensor:
        """expand dim"""
        assert dim not in source.dims
        # Some heuristic where to put the new dim.
        axis = len(source.dims)  # default: at the end
        if dim.is_static():
            if source.have_feature_axis():
                axis = source.feature_dim_axis
        if dim.is_dynamic():
            for i, d in reversed(list(enumerate(source.dims))):
                assert isinstance(d, Dim)
                if d.is_dynamic():
                    axis = i + 1
                    break
        new_dim_tags = list(source.dims)
        new_dim_tags.insert(axis, dim)
        out = source.copy_template_new_dim_tags(new_dim_tags)
        if source.feature_dim:
            out.feature_dim = source.feature_dim
        out_raw = torch.unsqueeze(source.raw_tensor, axis)
        if dim.is_dynamic() or dim.dimension != 1:
            out_raw = torch.tile(out_raw, [dim.get_dim_value() if d == dim else 1 for d in out.dims])
        out.raw_tensor = out_raw
        return out

    @staticmethod
    def concat(
        *sources: Tuple[Tensor, Dim],
        allow_broadcast: bool = False,
        out_dim: Dim,
    ) -> Tensor:
        """concat"""
        axis = sources[0][0].get_axis_from_description(sources[0][1])
        dims = list(sources[0][0].dims)
        dims.remove(sources[0][1])
        if allow_broadcast:
            for source, dim in sources[1:]:
                assert dim in source.dims
                for dim_ in source.dims:
                    if dim_ == dim:
                        continue
                    if dim_ not in dims:
                        dims.append(dim_)
        sources_ = []
        for source, dim in sources:
            # Maybe extend dims, and also transpose in the right dim order.
            templ = Tensor(
                source.name, dims=dims[:axis] + [dim] + dims[axis:], dtype=source.dtype, sparse_dim=source.sparse_dim
            )
            if not allow_broadcast:
                assert set(templ.dims) == set(source.dims)
            source_ = source.copy_compatible_to(templ, add_dims=allow_broadcast, unbroadcast=True)
            sources_.append(source_)
        out = Tensor(
            "concat",
            dims=dims[:axis] + [out_dim] + dims[axis:],
            dtype=sources[0][0].dtype,
            sparse_dim=sources[0][0].sparse_dim,
        )
        if sources[0][0].feature_dim and sources[0][0].feature_dim != sources[0][1]:
            out.feature_dim = sources[0][0].feature_dim
        out.raw_tensor = torch.cat([s.raw_tensor for s in sources_], dim=axis)
        return out

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
        """pad"""
        assert len(out_dims) == len(axes) == len(padding)
        out = source.copy_template_new_dim_tags(
            [out_dims[axes.index(dim)] if dim in axes else dim for dim in source.dim_tags]
        )
        remaining_dims = set(axes)
        raw_pad = []
        for dim in reversed(source.dims):
            if dim not in remaining_dims:
                continue
            remaining_dims.remove(dim)
            pad_ = padding[axes.index(dim)]
            raw_pad.extend(
                (
                    pad_[0].get_dim_value() if isinstance(pad_[0], Dim) else pad_[0],
                    pad_[1].get_dim_value() if isinstance(pad_[1], Dim) else pad_[1],
                )
            )
        if isinstance(value, Tensor):
            assert value.dims == (), f"value {value} must be a scalar"
            value = value.raw_tensor
        out.raw_tensor = torch.nn.functional.pad(source.raw_tensor, pad=raw_pad, mode=mode, value=value)
        return out

    @staticmethod
    def cum_concat_step(source: Tensor, *, prev_accum: Tensor, axis: Dim, out_spatial_dim: Dim) -> Tensor:
        """cum concat step"""
        out = prev_accum.copy_template_replace_dim_tag(
            axis=prev_accum.get_axis_from_description(axis),
            new_dim_tag=out_spatial_dim,
            name=f"{source.name}/cum_concat_step",
        )
        source_ = source.copy_compatible_to(prev_accum)
        out.raw_tensor = torch.cat(
            (prev_accum.raw_tensor, source_.raw_tensor), dim=prev_accum.get_axis_from_description(axis)
        )
        return out

    @staticmethod
    def activation_raw(raw_tensor: torch.Tensor, func: str) -> torch.Tensor:
        """
        :param raw_tensor:
        :param func: e.g. "tanh"
        :return: raw tensor after activation
        """
        assert func in Backend._AllowedActivationFuncs
        if hasattr(torch, func):
            f = getattr(torch, func)
        elif hasattr(torch.nn.functional, func):
            f = getattr(torch.nn.functional, func)
        else:
            raise ValueError(f"unknown activation function {func!r}")
        return f(raw_tensor)

    @staticmethod
    def softmax(tensor: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
        """
        :param tensor:
        :param axis:
        :param use_mask:
        :return: softmax over axis
        """
        out = tensor.copy_template("softmax")
        if use_mask and axis.need_masking():
            tensor = tensor.copy()
            mask = tensor.get_sequence_mask_broadcast(axis=axis)
            inf_value = get_global_inf_value()
            tensor.raw_tensor = torch.where(mask, tensor.raw_tensor, -inf_value)
        out.raw_tensor = torch.softmax(tensor.raw_tensor, dim=tensor.dims.index(axis))
        return out

    @staticmethod
    def log_softmax(tensor: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
        """
        :param tensor:
        :param axis:
        :param use_mask:
        :return: log_softmax over axis
        """
        out = tensor.copy_template("log_softmax")
        if use_mask and axis.need_masking():
            tensor = tensor.copy()
            mask = tensor.get_sequence_mask_broadcast(axis=axis)
            inf_value = get_global_inf_value()
            tensor.raw_tensor = torch.where(mask, tensor.raw_tensor, -inf_value)
        out.raw_tensor = torch.log_softmax(tensor.raw_tensor, dim=tensor.dims.index(axis))
        return out

    @staticmethod
    def softmax_cross_entropy_with_logits(*, logits: Tensor, targets: Tensor, axis: Dim):
        """
        Efficient cross entropy. For PyTorch this is actually the default cross entropy function.
        (torch.nn.functional.cross_entropy)

        :param logits: target estimates given as inputs to softmax (i.e. unnormalized)
        :param targets: probabilities, i.e. normalized, can also be sparse
        :param axis: class labels dim over which softmax is computed
        :return: cross entropy (same Dims as 'logits' but without 'axis')
        """
        assert axis in logits.dims, "Specified axis not present in logits."

        if axis == targets.sparse_dim:
            assert (
                logits.dims_set - {axis} == targets.dims_set
            ), "logits Dims and target Dims have to match (except for implicit sparse_dim)."

            logits_dim_order = list(targets.dims)
            if len(logits_dim_order) > 0:
                # PyTorch's cross_entropy expects class probabilities over second axis.
                logits_dim_order.insert(1, axis)
            else:
                logits_dim_order = [axis]

            if targets.dtype != "int64":
                targets = targets.copy()
                targets.dtype = "int64"
                targets.raw_tensor = targets.raw_tensor.long()

        else:
            assert (
                not targets.sparse_dim
            ), "We expect that cross entropy would always be calculated along the sparse dim, if there is one."
            assert logits.dims_set == targets.dims_set, "logits Dims and target Dims have to match."
            assert axis in targets.dims, "Specified axis not present in targets."

            if len(targets.dims) > 1:
                # PyTorch's cross_entropy expects class probabilities over second axis.
                targets = targets.copy_move_axis(targets.dims.index(axis), 1)

            logits_dim_order = targets.dims

        # We need same order of axes as in target.
        logits_axes_permutation = [logits_dim_order.index(dim) for dim in logits.dims]
        logits = logits.copy_transpose(logits_axes_permutation)

        raw_cross_entropy = torch.nn.functional.cross_entropy(
            input=logits.raw_tensor, target=targets.raw_tensor, reduction="none"
        )

        out_dims = list(logits.dims)
        out_dims.remove(axis)

        cross_entropy = Tensor(name="cross_entropy", dims=out_dims, raw_tensor=raw_cross_entropy, dtype=logits.dtype)

        return cross_entropy

    @staticmethod
    def create_parameter_raw(tensor: rf.Parameter) -> torch.nn.Parameter:
        """
        :return: parameter
        """
        assert all(d.is_static() for d in tensor.dims)
        data = torch.zeros([d.dimension for d in tensor.dims], dtype=TorchBackend.as_dtype_raw(tensor.dtype))
        if tensor.dtype.startswith("int"):
            requires_grad = False
        else:
            requires_grad = True
        return torch.nn.Parameter(data, requires_grad=requires_grad)

    @staticmethod
    def set_parameter_initial_value(param: rf.Parameter, value: Union[None, Tensor, rf.RawTensorTypes]) -> None:
        """
        :param param: parameter
        :param value: initial value
        """
        if value is None:
            value = 0
        raw_param = param.raw_tensor
        assert isinstance(raw_param, torch.nn.Parameter)
        with torch.no_grad():
            if isinstance(value, Tensor):
                value_ = value.copy_compatible_to(param)
                raw_param.copy_(value_.raw_tensor)
            elif isinstance(value, numpy.ndarray):
                raw_param.copy_(torch.from_numpy(value))
            else:
                raw_param.copy_(value)

    @staticmethod
    def set_parameter_trainable(param: rf.Parameter, trainable: bool) -> None:
        """set trainable"""
        raw_param = param.raw_tensor
        assert isinstance(raw_param, torch.nn.Parameter)
        raw_param.requires_grad = trainable

    @staticmethod
    def parameter_assign(param: rf.Parameter, value: Tensor, op: str = "assign") -> None:
        """param assign"""
        value_ = value.copy_compatible_to(param)
        raw_param = param.raw_tensor
        assert isinstance(raw_param, torch.nn.Parameter)
        with torch.no_grad():
            if op == "assign":
                raw_param.copy_(value_.raw_tensor)
            elif op == "add":
                raw_param.add_(value_.raw_tensor)
            else:
                raise ValueError(f"Parameter {param} assign: Unsupported op: {op}")

    @staticmethod
    def compare_raw(a: torch.Tensor, kind: str, b: torch.Tensor) -> torch.Tensor:
        """
        :param a:
        :param kind: "equal", "less", "less_equal", "greater", "greater_equal", "not_equal"
        :param b:
        :return: a `kind` b
        """
        assert a.dim() == b.dim()
        if kind == "equal":
            kind = "eq"  # eq is different to equal; eq returns a torch Tensor
        op = getattr(torch, kind)  # e.g. torch.equal
        return op(a, b)

    @staticmethod
    def combine_raw(a: torch.Tensor, kind: str, b: torch.Tensor) -> torch.Tensor:
        """
        :param a:
        :param kind: "add", "sub", "mul", "truediv", "floordiv", "mod", "pow",
            "maximum", "minimum", "logical_and", "logical_or", "squared_difference"
        :param b:
        :return: a `kind` b
        """
        assert a.dim() == b.dim()
        if kind == "squared_difference":
            return (a - b) ** 2
        kind = {
            "truediv": "true_divide",
            "floordiv": "floor_divide",
            "mod": "remainder",
        }.get(kind, kind)
        op = getattr(torch, kind)  # e.g. torch.add
        return op(a, b)

    @staticmethod
    def transpose_raw(raw_tensor: torch.Tensor, perm: Sequence[int]) -> torch.Tensor:
        """
        :param raw_tensor:
        :param perm: e.g. [0, 2, 1]
        :return: permuted (transposed) raw tensor; wraps torch.permute
        """
        return torch.permute(raw_tensor, tuple(perm))

    @staticmethod
    def convert_to_tensor(
        value: Union[Tensor, torch.Tensor, RawTensorTypes],
        *,
        dims: Sequence[Dim],
        dtype: str,
        sparse_dim: Optional[Dim] = None,
        name: Optional[str] = None,
    ) -> Tensor[torch.Tensor]:
        """
        :param value:
        :param dims:
        :param dtype:
        :param sparse_dim:
        :param name:
        :return: tensor
        """
        if isinstance(value, Tensor):
            return value
        if isinstance(value, torch.Tensor):
            name = name or "raw_tensor"
        else:
            name = name or "const"
            value = torch.tensor(value, dtype=TorchBackend.as_dtype_raw(dtype))
        assert isinstance(value, torch.Tensor)
        return Tensor(name, dims=dims, dtype=dtype, sparse_dim=sparse_dim, raw_tensor=value)

    @staticmethod
    def full(
        dims: Sequence[Dim], fill_value: RawTensorTypes, *, dtype: str, sparse_dim: Optional[Dim] = None
    ) -> Tensor:
        """
        :param dims:
        :param fill_value:
        :param dtype:
        :param sparse_dim:
        :return: tensor
        """
        shape = [dim.get_dim_value() for dim in dims]
        raw_tensor = torch.full(shape, fill_value, dtype=TorchBackend.as_dtype_raw(dtype))
        return Tensor("full", dims=dims, sparse_dim=sparse_dim, dtype=dtype, raw_tensor=raw_tensor)

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
        axis_int = source.get_axis_from_description(axis, allow_int=False)
        out = source.copy_template_replace_dim_tag(axis=axis_int, new_dim_tag=out_dim)
        if isinstance(start, Tensor):
            assert start.dims == ()
            start = start.raw_tensor
        if start is None:
            start = 0
        if isinstance(size, Dim):
            size = size.get_dim_value()
        if size is not None:
            assert end is None
            out.raw_tensor = torch.narrow(source.raw_tensor, dim=axis_int, start=start, length=size)
        else:
            if isinstance(end, Tensor):
                assert end.dims == ()
                end = end.raw_tensor
            if end is None:
                end = axis.get_dim_value()
            out.raw_tensor = torch.narrow(source.raw_tensor, dim=axis_int, start=start, length=end - start)
        return out

    @staticmethod
    def matmul(a: _TT, b: _TT, *, reduce: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> _TT:
        """
        batched matmul of a and b, see base class doc string
        """
        if isinstance(reduce, Dim):
            reduce = [reduce]

        if use_mask and any(dim.dyn_size_ext for dim in reduce):
            raise NotImplementedError("masking in matmul reduce not yet implemented")
        assert a.dtype == b.dtype, f"matmul: dtypes do not match: {a} vs {b}"

        a_dims = a.dims
        b_dims = b.dims

        assert all(
            dim in a_dims for dim in reduce
        ), f"'a' does not have the specified reduce dim(s) {reduce} (a dims: {a_dims})"
        assert all(
            dim in b_dims for dim in reduce
        ), f"'b' does not have the specified reduce dim(s) {reduce} (b dims: {b_dims})"

        if len(reduce) > 1:
            reduce = list(reduce)
            reduce.sort(key=lambda dim: a_dims.index(dim))

        # matmul might get square matrices as arguments, where a dim could occur multiple times.
        # This complicates the logic here, and we properly have to handle match_priority.
        a_reduce_axes = [a.get_axis_from_description(reduce_dim) for reduce_dim in reduce]
        b_reduce_axes = [b.get_axis_from_description(reduce_dim) for reduce_dim in reduce]

        # We assume that dim tags in remaining dims are unique.
        common_dims = [dim for i, dim in enumerate(a_dims) if dim in b_dims and i not in a_reduce_axes]
        a_common_axes = [a_dims.index(common_dim) for common_dim in common_dims]
        b_common_axes = [b_dims.index(common_dim) for common_dim in common_dims]

        a_unique_axes = [i for i in range(len(a_dims)) if i not in a_reduce_axes and i not in a_common_axes]
        b_unique_axes = [i for i in range(len(b_dims)) if i not in b_reduce_axes and i not in b_common_axes]

        a_raw = a.raw_tensor
        b_raw = b.raw_tensor

        a_shape = a_raw.shape
        b_shape = b_raw.shape

        common_axes_shape = tuple(a_shape[i] for i in a_common_axes)
        b_common_axes_shape = tuple(b_shape[i] for i in b_common_axes)
        assert common_axes_shape == b_common_axes_shape, "Tensor shape for common Dims of a and b does not match."

        common_axes_total_dimension = prod(common_axes_shape)

        a_unique_axes_shape = tuple(a_shape[i] for i in a_unique_axes)
        b_unique_axes_shape = tuple(b_shape[i] for i in b_unique_axes)

        a_unique_axes_total_dimension = prod(a_unique_axes_shape)
        b_unique_axes_total_dimension = prod(b_unique_axes_shape)

        reduce_axes_shape = tuple(a_shape[i] for i in a_reduce_axes)
        b_reduce_axes_shape = tuple(b_shape[i] for i in b_reduce_axes)
        assert reduce_axes_shape == b_reduce_axes_shape, "Tensor shape for reduce Dims does not match between a and b."

        reduce_axes_total_dimension = prod(reduce_axes_shape)

        a_raw = torch.permute(a_raw, a_common_axes + a_unique_axes + a_reduce_axes)
        b_raw = torch.permute(b_raw, b_common_axes + b_reduce_axes + b_unique_axes)

        if common_axes_total_dimension == 1:  # standard matrix multiplication
            a_raw = torch.reshape(a_raw, (a_unique_axes_total_dimension, reduce_axes_total_dimension))
            b_raw = torch.reshape(b_raw, (reduce_axes_total_dimension, b_unique_axes_total_dimension))

            raw_result = torch.mm(a_raw, b_raw)

        else:  # batched matrix multiplication
            a_raw = torch.reshape(
                a_raw, (common_axes_total_dimension, a_unique_axes_total_dimension, reduce_axes_total_dimension)
            )
            b_raw = torch.reshape(
                b_raw, (common_axes_total_dimension, reduce_axes_total_dimension, b_unique_axes_total_dimension)
            )

            raw_result = torch.bmm(a_raw, b_raw)

        raw_result = torch.reshape(raw_result, common_axes_shape + a_unique_axes_shape + b_unique_axes_shape)

        a_unique_dims = [a_dims[i] for i in a_unique_axes]
        b_unique_dims = [b_dims[i] for i in b_unique_axes]
        result_dims = common_dims + a_unique_dims + b_unique_dims

        result_tensor = Tensor(name="dot", dims=result_dims, raw_tensor=raw_result, dtype=a.dtype)

        return result_tensor

    @staticmethod
    def range_over_dim(dim: Dim, *, dtype: Optional[str] = None) -> Tensor[torch.Tensor]:
        """
        :param dim:
        :param dtype:
        :return: tensor with shape [dim]
        """
        if not dtype and dim.dyn_size_ext:
            dtype = dim.dyn_size_ext.dtype
        if not dtype:
            dtype = rf.get_default_array_index_dtype()
        out = Tensor(
            "range",
            dims=[dim],
            sparse_dim=dim,
            dtype=dtype,
        )
        out.raw_tensor = torch.arange(dim.get_dim_value(), dtype=TorchBackend.as_dtype_raw(out.dtype))
        return out

    @staticmethod
    def reduce(
        source: Tensor[torch.Tensor],
        *,
        mode: str,
        axis: Union[Dim, Sequence[Dim]],
        use_mask: bool = True,
    ) -> Tensor[torch.Tensor]:
        """reduce"""
        assert mode in Backend._AllowedReduceModes
        if isinstance(axis, Dim):
            axis = [axis]
        assert all(isinstance(dim, Dim) for dim in axis)
        if use_mask and any(dim.need_masking() for dim in axis):
            source = source.copy()
            dtype = source.raw_tensor.dtype
            if mode == "max":
                mask_value = torch.finfo(dtype).min if dtype.is_floating_point else torch.iinfo(dtype).min
            elif mode == "min":
                mask_value = torch.finfo(dtype).max if dtype.is_floating_point else torch.iinfo(dtype).max
            elif mode == "sum":
                mask_value = 0
            else:
                raise NotImplementedError(f"reduce_{mode} not implemented with masking on tensor {source!r}.")
            for i, dim in enumerate(axis):
                if dim.need_masking():
                    mask = source.get_sequence_mask_broadcast(axis=i)
                    source.raw_tensor = torch.where(mask, source.raw_tensor, mask_value)
        func = getattr(torch, mode)
        raw_dims = [source.get_axis_from_description(dim) for dim in axis]
        res_dims = [dim for i, dim in enumerate(source.dims) if i not in raw_dims]
        if not res_dims:
            raw_result = func(source.raw_tensor)
        elif len(raw_dims) == 1:
            raw_result = func(source.raw_tensor, dim=raw_dims[0])
            if mode in ["max", "min"]:
                # result is a tuple (values, indices). https://pytorch.org/docs/stable/generated/torch.max.html
                raw_result, _ = raw_result
        else:
            assert mode == "sum"  # not implemented otherwise for multiple axes
            raw_result = func(source.raw_tensor, dim=raw_dims)
        res = Tensor(
            name=f"reduce_{mode}",
            raw_tensor=raw_result,
            dims=res_dims,
            dtype=TorchBackend.get_dtype_name_raw(raw_result),
            sparse_dim=source.sparse_dim,
        )
        return res

    @staticmethod
    @contextlib.contextmanager
    def random_journal_record() -> List[Dict[str, Any]]:
        """
        :return: the journal
        """
        try:
            TorchBackend._random_journal_record_enabled = True
            TorchBackend._random_journal = []
            yield TorchBackend._random_journal
        finally:
            TorchBackend._random_journal_record_enabled = False
            TorchBackend._random_journal = None

    _random_journal_record_enabled = False
    _random_journal = None  # type: Optional[List[Dict[str, Any]]]

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
        out: Optional[Tensor[torch.Tensor]] = None,
    ) -> Tensor:
        """
        random. See `rf.random` for details.
        """
        shape = [d.get_dim_value() for d in dims]
        dtype_ = TorchBackend.as_dtype_raw(dtype)
        if out is None:
            out = Tensor(name=f"random_{distribution}", dims=dims, dtype=dtype, sparse_dim=sparse_dim)
            out.raw_tensor = torch.empty(shape, dtype=dtype_)
        assert explicit_state is None  # not implemented otherwise
        generator = None  # using the global default from PT
        assert isinstance(static, bool)
        if static:
            assert seed is not None
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            assert seed is None
        assert auto_update_state is None  # not implemented otherwise
        if distribution == "uniform":
            assert mean is None and stddev is None  # not implemented otherwise
            if dtype_.is_floating_point:
                if minval is None:
                    minval = 0
                if maxval is None:
                    maxval = 1
                if isinstance(minval, Tensor):
                    assert minval.dims == (), f"only scalar minval supported, got {minval}"
                    minval = minval.raw_tensor
                if isinstance(maxval, Tensor):
                    assert maxval.dims == (), f"only scalar maxval supported, got {maxval}"
                    maxval = maxval.raw_tensor
                with torch.no_grad():
                    out.raw_tensor.uniform_(minval, maxval, generator=generator)  # noqa
            else:
                if minval is None:
                    minval = 0
                assert maxval is not None, "maxval must be specified for integer random uniform"
                if isinstance(minval, Tensor):
                    assert minval.dims == (), f"only scalar minval supported, got {minval}"
                    minval = minval.raw_tensor
                if isinstance(maxval, Tensor):
                    assert maxval.dims == (), f"only scalar maxval supported, got {maxval}"
                    maxval = maxval.raw_tensor
                with torch.no_grad():
                    out.raw_tensor.random_(minval, maxval, generator=generator)
        elif distribution == "normal":
            assert minval is None and maxval is None
            if mean is None:
                mean = 0
            if stddev is None:
                stddev = 1
            if isinstance(mean, Tensor):
                assert mean.dims == (), f"only scalar mean supported, got {mean}"
                mean = mean.raw_tensor
            if isinstance(stddev, Tensor):
                assert stddev.dims == (), f"only scalar stddev supported, got {stddev}"
                stddev = stddev.raw_tensor
            with torch.no_grad():
                out.raw_tensor.normal_(mean, stddev, generator=generator)
        elif distribution == "truncated_normal":
            if mean is None:
                mean = 0
            if stddev is None:
                stddev = 1
            if minval is None:
                minval = mean - 2 * stddev
            if maxval is None:
                maxval = mean + 2 * stddev

            from . import _rand

            _rand.no_grad_trunc_normal_(out.raw_tensor, mean=mean, std=stddev, a=minval, b=maxval, generator=generator)
        else:
            raise NotImplementedError(f"random distribution {distribution} not implemented")
        if TorchBackend._random_journal_record_enabled:
            out_ = out.copy()
            out_.raw_tensor = out_.raw_tensor.detach().cpu().numpy()
            TorchBackend._random_journal.append(
                {
                    "dims": tuple(dims),
                    "dtype": dtype,
                    "sparse_dim": sparse_dim,
                    "distribution": distribution,
                    "mean": mean,
                    "stddev": stddev,
                    "bound": bound,
                    "minval": minval,
                    "maxval": maxval,
                    "seed": seed,
                    "static": static,
                    "out": out_,
                }
            )
        return out

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
        remaining_dims = [d for d in tensor.dims if d not in mask.dims]
        tensor_templ = tensor.copy_template_new_dim_tags(tuple(dims) + tuple(remaining_dims))
        tensor = tensor.copy_compatible_to(tensor_templ, add_dims=False)
        mask = mask.copy_compatible_to(tensor_templ, check_dtype=False, check_sparse=False)
        in_raw = tensor.raw_tensor
        # We have a very strange problem with the gradient of masked_select,
        # when used together with some specific other operations before that,
        # like convolution.
        # This clone() with contiguous_format seems to fix the problem.
        # https://github.com/pytorch/pytorch/issues/99638
        in_raw = in_raw.clone(memory_format=torch.contiguous_format)
        out_raw = torch.masked_select(in_raw, mask.raw_tensor)
        remaining_shape = [d.get_dim_value() for d in remaining_dims]
        remaining_num_elements = numpy.prod(remaining_shape) if remaining_shape else 1
        assert out_raw.numel() % remaining_num_elements == 0
        flattened_num_elements = out_raw.numel() // remaining_num_elements
        out_raw = torch.reshape(out_raw, [flattened_num_elements] + remaining_shape)
        if not out_dim:
            out_dim = TorchBackend.get_new_dim_raw(out_raw, 0, name="masked_select")
        out = Tensor(
            "masked_select",
            dims=(out_dim,) + tuple(remaining_dims),
            dtype=tensor.dtype,
            sparse_dim=tensor.sparse_dim,
            raw_tensor=out_raw,
        )
        return out, out_dim

    @staticmethod
    def batch_norm(
        source: Tensor[torch.Tensor],
        *,
        in_dim: Union[Dim, Sequence[Dim]],
        running_mean: Optional[Tensor],
        running_variance: Optional[Tensor],
        gamma: Optional[Tensor],
        beta: Optional[Tensor],
        epsilon: float,
        momentum: float,
        affine: bool,
        use_mask: bool,
    ) -> Tensor:
        """batch norm"""
        if use_mask:
            raise NotImplementedError("batch_norm with masking not implemented")
        if (running_mean is None) != (running_variance is None):
            raise ValueError("running_mean and running_variance must be both None or both not None")
        assert isinstance(in_dim, Dim)  # multiple dims not supported yet
        if affine:
            if gamma is None or beta is None:
                raise ValueError("gamma and beta must be given if affine=True")
            if not gamma.dims == beta.dims == (in_dim,):
                raise ValueError(f"gamma and beta must have shape [{in_dim}], got gamma {gamma} and beta {beta}")
        if running_mean is not None:
            if not running_mean.dims == running_variance.dims == (in_dim,):
                raise ValueError(
                    f"running_mean and running_variance must have shape [{in_dim}], got "
                    f"running_mean {running_mean} and running_variance {running_variance}"
                )
        feat_axis = source.get_axis_from_description(in_dim)
        if feat_axis == 0:
            pre_dims = 1
        else:
            pre_dims = numpy.prod(source.raw_tensor.shape[:feat_axis])
        # Torch batch_norm expects (N,C,+) as shape.
        src_raw = torch.reshape(source.raw_tensor, [pre_dims, in_dim.get_dim_value(), -1])
        # https://github.com/pytorch/pytorch/blob/59605811488eb07b3b8bf70a5f0b4b56b34b4a61/aten/src/ATen/native/Normalization.cpp#L546
        out_raw = torch.nn.functional.batch_norm(
            src_raw,
            running_mean=running_mean.raw_tensor if running_mean is not None else None,
            running_var=running_variance.raw_tensor if running_variance is not None else None,
            weight=gamma.raw_tensor if affine else None,
            bias=beta.raw_tensor if affine else None,
            # training: means whether we should use the current batch statistics
            #   + update the running statistics (if given)
            training=rf.get_run_ctx().train_flag or (running_mean is None),
            momentum=momentum,
            eps=epsilon,
        )
        out = source.copy_template()
        out.raw_tensor = torch.reshape(out_raw, source.raw_tensor.shape)
        out.feature_dim = in_dim
        return out

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
        filter_in_dim = in_dim if not groups or groups == 1 else in_dim // groups
        filter_dims = (out_dim, filter_in_dim) + tuple(filter_size)
        filter = filter.copy_transpose(filter_dims)
        batch_dims = [d for d in source.dims if d not in (in_dim,) + tuple(in_spatial_dims)]
        # Torch conv expects (N,C,<spatial dims>) as shape.
        source = source.copy_transpose(batch_dims + [in_dim] + list(in_spatial_dims))
        src_raw = torch.reshape(
            source.raw_tensor,
            # potentially merge batch dims all together
            [-1, in_dim.get_dim_value()] + [d.get_dim_value() for d in in_spatial_dims],
        )
        use_striding = strides and (strides > 1 if isinstance(strides, int) else any(s > 1 for s in strides))
        if padding == "same" and use_striding:
            # padding='same' is not supported for strided convolutions
            padding = "valid"
            pads = []
            for i, s in reversed(list(enumerate(filter_size))):
                if use_striding:
                    stride_ = strides[i] if isinstance(strides, (list, tuple)) else strides
                else:
                    stride_ = 1
                # What is the logic here? You might be aware, in case without striding,
                # we simply have pad_left = (s.dimension - 1) // 2,
                # or the total amount of padding is s.dimension - 1.
                # So we add the same amount of padding on both sides (or one less on the left side if odd).
                # The output seq length in case of "valid" padding is ⌈(in_len - s.dimension + 1) / stride⌉.
                # The output seq length in case of "same" padding with no striding (stride = 1)
                # is simply the same as the input length.
                # What is the output seq length in case of "same" padding with striding?
                # It should be ⌈in_len / stride⌉.
                # So, then we need to add a total amount of padding of s.dimension - 1.
                # However, doing it the same way as without striding is incorrect.
                # Why? Because then the first window might have more padding than the final window.
                # But we want to have an even amount of padding on the first and last window,
                # or maybe one less on the first window if odd.
                # How many frames do the windows cover?
                # in_len_covered = out_len * stride + (s.dimension - stride)
                # We can rewrite out_len as:
                # out_len = ⌊(in_len + stride - 1) / stride⌋ = (in_len + stride - 1 - (in_len - 1) % stride) / stride
                # So we have:
                # in_len_covered = (in_len + stride - 1 - (in_len - 1) % stride) + (s.dimension - stride)
                #                = in_len + s.dimension - 1 - (in_len - 1) % stride
                # Now, the amount of padding which actually matters is:
                # pad = in_len_covered - in_len = s.dimension - 1 - (in_len - 1) % stride.
                pad = s.dimension - 1 - (src_raw.shape[2 + i] - 1) % stride_
                pad_left = pad // 2
                pad_right = pad - pad_left
                pads.extend([pad_left, pad_right])
            src_raw = torch.nn.functional.pad(src_raw, pads)
        if len(filter_size) == 1:
            # There is also conv_tbc, but it's a bit limited (no dilation)
            # and also unclear when exactly it is faster.
            out_raw = torch.nn.functional.conv1d(
                src_raw,
                weight=filter.raw_tensor,
                bias=bias.raw_tensor if bias is not None else None,
                stride=strides or 1,
                padding=padding,
                dilation=dilation_rate or 1,
                groups=groups or 1,
            )
        elif len(filter_size) == 2:
            out_raw = torch.nn.functional.conv2d(
                src_raw,
                weight=filter.raw_tensor,
                bias=bias.raw_tensor if bias is not None else None,
                stride=strides or 1,
                padding=padding,
                dilation=dilation_rate or 1,
                groups=groups or 1,
            )
        elif len(filter_size) == 3:
            out_raw = torch.nn.functional.conv3d(
                src_raw,
                weight=filter.raw_tensor,
                bias=bias.raw_tensor if bias is not None else None,
                stride=strides or 1,
                padding=padding,
                dilation=dilation_rate or 1,
                groups=groups or 1,
            )
        else:
            raise ValueError(f"invalid number of filter dims {filter_size}, expected 1, 2, or 3")
        out = Tensor("conv", dims=batch_dims + [out_dim] + list(out_spatial_dims), dtype=source.dtype)
        out.raw_tensor = torch.reshape(out_raw, [d.get_dim_value() for d in out.dims])
        out.feature_dim = out_dim
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
        batch_dims = [d for d in source.dims if d not in tuple(in_spatial_dims)]
        # Torch conv expects (N,C,<spatial dims>) as shape.
        # batch_dims would actually cover the channel-dim (C) as well,
        # as it does not really matter to differentiate it from other batch dims.
        source = source.copy_transpose(batch_dims + list(in_spatial_dims))
        src_raw = torch.reshape(
            source.raw_tensor,
            # Potentially merge batch dims all together.
            # Keep the last as the channel-dim, but not sure if this is really relevant.
            [-1, batch_dims[-1].get_dim_value() if batch_dims else 1] + [d.get_dim_value() for d in in_spatial_dims],
        )
        assert isinstance(strides, (list, tuple)) and len(strides) == len(in_spatial_dims) == len(pool_size)
        if padding.lower() == "same":
            # padding='same' is not quite the same as ceil_mode=True, so we explicitly pad here.
            padding = []
            for i, s in enumerate(pool_size):
                # See comment in conv.
                pad = s - 1 - (src_raw.shape[2 + i] - 1) % strides[i]
                padding.append(pad // 2)
            ceil_mode = True
        elif padding.lower() == "valid":
            padding = 0
            ceil_mode = False
        else:
            raise ValueError(f"invalid padding {padding!r}")
        if len(in_spatial_dims) == 1:
            out_raw = torch.nn.functional.max_pool1d(
                src_raw,
                kernel_size=pool_size,
                stride=strides,
                dilation=dilation_rate or 1,
                ceil_mode=ceil_mode,
                padding=padding,
            )
        elif len(in_spatial_dims) == 2:
            out_raw = torch.nn.functional.max_pool2d(
                src_raw,
                kernel_size=pool_size,
                stride=strides,
                dilation=dilation_rate or 1,
                ceil_mode=ceil_mode,
                padding=padding,
            )
        elif len(in_spatial_dims) == 3:
            out_raw = torch.nn.functional.max_pool3d(
                src_raw,
                kernel_size=pool_size,
                stride=strides,
                dilation=dilation_rate or 1,
                ceil_mode=ceil_mode,
                padding=padding,
            )
        else:
            raise ValueError(f"invalid number of filter dims {in_spatial_dims}, expected 1, 2, or 3")
        out = Tensor("conv", dims=batch_dims + list(out_spatial_dims), dtype=source.dtype)
        out.raw_tensor = torch.reshape(out_raw, [d.get_dim_value() for d in out.dims])
        if source.feature_dim and source.feature_dim in out.dims:
            out.feature_dim = source.feature_dim
        return out, out_spatial_dims

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
        batch_dims = [d for d in x.dims if d != in_spatial_dim]
        x = x.copy_transpose(batch_dims + [in_spatial_dim])
        x_raw = torch.reshape(x.raw_tensor, [-1, in_spatial_dim.get_dim_value()])

        # TF code: y = tf.signal.stft(x, frame_length=frame_size, frame_step=frame_shift, fft_length=fft_size)
        # This is similar to what SciPy will also return.
        # It is somewhat nontrivial to get the same result in PyTorch, esp for the case frame_length < fft_length,
        # and even more when frame_length % 2 != 0.
        # PyTorch behaves like librosa.
        # https://github.com/pytorch/pytorch/issues/100177
        # https://github.com/albertz/playground/blob/master/tf_pt_stft.py

        if frame_length < fft_length and window_use_frame_length:
            # PyTorch uses windows always of the fft_length,
            # so the output seq len is ⌈(T - fft_length + 1) / frame_step⌉.
            # TF/SciPy use windows of frame_length,
            # so the output seq len is ⌈(T - frame_length + 1) / frame_step⌉.
            # By padding the difference (fft_length - frame_length) to the right,
            # we get the same output seq length in both cases.
            x_raw = torch.nn.functional.pad(x_raw, (0, (fft_length - frame_length)))

        if window_enforce_even:
            frame_length -= frame_length % 2

        window_pt = torch.hann_window(frame_length)
        if frame_length < fft_length:
            if align_window_left:  # this is how TF/SciPy do it
                window_pt = torch.nn.functional.pad(window_pt, (0, (fft_length - frame_length)))
            else:  # this is how PyTorch/librosa do it
                pad_left = (fft_length - frame_length) // 2
                pad_right = fft_length - frame_length - pad_left
                window_pt = torch.nn.functional.pad(window_pt, (pad_left, pad_right))

        y_raw = torch.stft(
            x_raw,
            n_fft=fft_length,
            hop_length=frame_step,
            # PyTorch anyway uses a window with size = n_fft internally.
            # So we just explicitly handle the window logic.
            win_length=fft_length,
            window=window_pt,
            center=False,
            return_complex=True,
        )
        y = Tensor("stft", dims=batch_dims + [out_dim, out_spatial_dim], dtype=TorchBackend.get_dtype_name_raw(y_raw))
        y.feature_dim = out_dim
        y.raw_tensor = torch.reshape(y_raw, [d.get_dim_value() for d in y.dims])
        return y

    @staticmethod
    def lstm(
        source: _TT,
        *,
        state_h: _TT,
        state_c: _TT,
        ff_weight: _TT,
        rec_weight: _TT,
        bias: Optional[_TT],
        spatial_dim: Dim,
        in_dim: Dim,
        out_dim: Dim,
    ) -> Tuple[_TT, Tuple[_TT, _TT]]:
        """
        Wraps the functional LSTM from PyTorch.

        :return: Tuple consisting of two elements: the result as a :class:`Tensor`
            and the new state as a :class:`State` (different from the previous one).
        """
        # The order in the parameters for lstm_params is specified as follows:
        # Feed-forward has priority over recurrent, and weights have priority over biases: (w_ih, w_hh, b_ih, b_hh).
        # See https://github.com/pytorch/pytorch/blob/c263bd43/torch/nn/modules/rnn.py#L107.
        # Alternatively see the implementation of LSTMCell:
        # https://github.com/pytorch/pytorch/blob/4bead64/aten/src/ATen/native/RNN.cpp#L1458.
        if bias is None:
            lstm_params = (ff_weight.raw_tensor, rec_weight.raw_tensor)
            has_biases = False
        else:
            # Note that the mathematical definition of the model does not require two bias terms.
            # PyTorch just follows what CuDNN does here.
            # I don't really understand why CuDNN have two bias terms:
            # https://stackoverflow.com/questions/76051800/why-does-the-cudnn-lstm-requires-two-biases-b-ih-and-b-hh
            # It looks like an oversight by the CuDNN devs.
            # I'm not sure if the two bias terms get the same gradient or not.
            # If they do not, that would be mathematical incorrect, but maybe that's how CuDNN works.
            # To really get both gradients, we don't just set one bias to zero, but we use 0.5 * bias for both.
            bias_raw = bias.raw_tensor * 0.5
            lstm_params = (
                ff_weight.raw_tensor,
                rec_weight.raw_tensor,
                bias_raw,
                bias_raw,
            )
            has_biases = True

        batch_dims = [d for d in source.dims if d != spatial_dim and d != in_dim]
        source = source.copy_transpose([spatial_dim] + batch_dims + [in_dim])
        state_h = state_h.copy_transpose(batch_dims + [out_dim])
        state_c = state_c.copy_transpose(batch_dims + [out_dim])

        source_raw = source.raw_tensor
        state_h_raw = state_h.raw_tensor
        state_c_raw = state_c.raw_tensor
        batch_dim = torch.prod(torch.tensor([d.get_dim_value() for d in batch_dims])) if batch_dims else 1
        if len(batch_dims) != 1:
            # Torch LSTM expects (seq_len, batch, input_size) as shape.
            # We need to merge all batch dims together.
            source_raw = torch.reshape(
                source_raw, [spatial_dim.get_dim_value()] + [batch_dim] + [in_dim.get_dim_value()]
            )
        # Torch LSTM expects (num_layers * num_directions, batch, hidden_size) as shape.
        state_h_raw = torch.reshape(state_h_raw, [1, batch_dim, out_dim.get_dim_value()])
        state_c_raw = torch.reshape(state_c_raw, [1, batch_dim, out_dim.get_dim_value()])

        sizes = spatial_dim.get_size_tensor()
        sizes = sizes.copy_compatible_to(
            Tensor("batch_dims", batch_dims, dtype=sizes.dtype), unbroadcast=True, check_sparse=False
        )
        sizes_raw = torch.reshape(sizes.raw_tensor, [batch_dim])

        # See the code of torch.nn.LSTM for sorting the batch dims.
        # We need pack_padded_sequence because otherwise the LSTM would ignore the padding,
        # and we would get incorrect final states.
        source_packed = torch.nn.utils.rnn.pack_padded_sequence(source_raw, sizes_raw, enforce_sorted=False)
        state_h_raw = state_h_raw.index_select(dim=1, index=source_packed.sorted_indices)
        state_c_raw = state_c_raw.index_select(dim=1, index=source_packed.sorted_indices)

        out_raw, new_state_h_raw, new_state_c_raw = torch.lstm(
            source_packed.data,
            source_packed.batch_sizes,
            (state_h_raw, state_c_raw),
            lstm_params,
            has_biases=has_biases,
            num_layers=1,
            dropout=0.0,
            train=rf.get_run_ctx().train_flag,
            bidirectional=False,
        )

        # Unsort the batch dims.
        new_state_h_raw = new_state_h_raw.index_select(dim=1, index=source_packed.unsorted_indices)
        new_state_c_raw = new_state_c_raw.index_select(dim=1, index=source_packed.unsorted_indices)
        # Unpack the sequence.
        output_packed = torch.nn.utils.rnn.PackedSequence(
            out_raw,
            batch_sizes=source_packed.batch_sizes,
            sorted_indices=source_packed.sorted_indices,
            unsorted_indices=source_packed.unsorted_indices,
        )
        out_raw = torch.nn.utils.rnn.pad_packed_sequence(output_packed)[0]

        if len(batch_dims) != 1:
            out_raw = torch.reshape(
                out_raw,
                [spatial_dim.get_dim_value()] + [d.get_dim_value() for d in batch_dims] + [out_dim.get_dim_value()],
            )
        new_state_h_raw = torch.reshape(new_state_h_raw, [d.get_dim_value() for d in state_h.dims])
        new_state_c_raw = torch.reshape(new_state_c_raw, [d.get_dim_value() for d in state_c.dims])

        out = source.copy_template_replace_dim_tag(axis=-1, new_dim_tag=out_dim, name="lstm")
        out.feature_dim = out_dim
        out.raw_tensor = out_raw

        new_state_h = state_h.copy_template()
        new_state_h.raw_tensor = new_state_h_raw
        new_state_h.feature_dim = out_dim
        new_state_c = state_c.copy_template()
        new_state_c.raw_tensor = new_state_c_raw
        new_state_c.feature_dim = out_dim

        return out, (new_state_h, new_state_c)
