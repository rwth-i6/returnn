"""
Backend for exposing PyTorch-specific functionality.
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, Tuple, List, Dict, Generator
import contextlib
import torch
import numpy

from returnn.tensor import Tensor, Dim, single_step_dim
from returnn.util.basic import prod, get_global_inf_value

# noinspection PyProtectedMember
from returnn.frontend._backend import Backend
from returnn.frontend import RawTensorTypes
import returnn.frontend as rf

# noinspection PyProtectedMember
from returnn.frontend import _random_journal

# noinspection PyProtectedMember
from returnn.frontend import _utils

from . import raw_ops

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
    def get_shape_raw(raw_tensor: torch.Tensor) -> Tuple[int]:
        """shape"""
        return tuple(raw_tensor.shape)

    @staticmethod
    def get_shape_tuple_raw(raw_tensor: torch.Tensor) -> Tuple[int]:
        """
        :return: shape of raw tensor
        """
        return tuple(raw_tensor.shape)

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
        return Dim(int(raw_tensor.size(axis)), name=name)

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
    def expand_raw(raw_tensor: torch.Tensor, axis: int, dim: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        :param raw_tensor:
        :param axis: shape[axis] must be 1
        :param dim: the new dim for shape[axis]
        :return: shape[axis] expands to dim.
            in PyTorch or other frameworks which support custom strides,
            this is an efficient view and not a copy.
        """
        return raw_tensor.expand(*[-1 if i != axis else dim for i in range(raw_tensor.dim())])

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
    def set_requires_gradient(tensor: Tensor[torch.Tensor]):
        """set requires grad"""
        tensor.raw_tensor.requires_grad = True

    @staticmethod
    def gradient(y: Tensor, x: Tensor) -> Tensor:
        """gradient"""
        out = x.copy_template(name="gradient")
        out.raw_tensor = torch.autograd.grad(y.raw_tensor, x.raw_tensor, create_graph=True)[0]
        return out

    @staticmethod
    def stop_gradient(tensor: Tensor) -> Tensor:
        """stop grad"""
        out = tensor.copy()
        out.raw_tensor = out.raw_tensor.detach()
        return out

    @staticmethod
    def scaled_gradient(tensor: Tensor, scale: Union[float, Tensor]) -> Tensor:
        """scaled gradient"""
        from returnn.torch.util.scaled_gradient import scaled_gradient

        out = tensor.copy()
        out.raw_tensor = scaled_gradient(out.raw_tensor, scale=scale)
        return out

    @staticmethod
    def scaled_gradient_ext(
        x: Tensor,
        *,
        scale: Union[float, Tensor] = 1.0,
        shift: Optional[Union[float, Tensor]] = None,
        scale_shift_by_sum_over_axis: Optional[Dim] = None,
    ):
        """scaled gradient ext"""
        from returnn.torch.util.scaled_gradient import scaled_gradient_ext

        out = x.copy()
        out.raw_tensor = scaled_gradient_ext(
            out.raw_tensor,
            scale=scale.raw_tensor if isinstance(scale, Tensor) else scale,
            shift=shift.raw_tensor if isinstance(shift, Tensor) else shift,
            scale_shift_by_sum_over_axis=x.get_axis_from_description(scale_shift_by_sum_over_axis, allow_int=False)
            if scale_shift_by_sum_over_axis is not None
            else None,
        )
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
        if source.feature_dim and source.feature_dim in dims:
            out.feature_dim = out_dim
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
    def squeeze(source: Tensor, axis: Dim) -> Tensor:
        """squeeze"""
        axis = source.get_axis_from_description(axis)
        out = source.copy_template_excluding_axis(axis)
        out.raw_tensor = torch.squeeze(source.raw_tensor, axis)
        return out

    @staticmethod
    def concat(
        *sources: Tuple[Tensor, Dim],
        allow_broadcast: bool = False,
        out_dim: Dim,
    ) -> Tensor:
        """concat"""
        axis = sources[0][0].get_axis_from_description(sources[0][1])
        other_dims = list(sources[0][0].dims)
        other_dims.remove(sources[0][1])
        need_broadcast = False
        if allow_broadcast:
            for source, dim in sources[1:]:
                assert dim in source.dims
                for dim_ in source.dims:
                    if dim_ == dim:
                        continue
                    if dim_ not in other_dims:
                        other_dims.append(dim_)
                        need_broadcast = True
        sources_raw = []
        if allow_broadcast and need_broadcast:
            for source, dim in sources:
                # Maybe extend dims, and also transpose in the right dim order.
                templ = Tensor(
                    source.name,
                    dims=other_dims[:axis] + [dim] + other_dims[axis:],
                    dtype=source.dtype,
                    sparse_dim=source.sparse_dim,
                )
                source_ = source.copy_compatible_to(templ, unbroadcast=True)
                sources_raw.append(source_.raw_tensor)
        else:  # not allow_broadcast
            for source, dim in sources:
                templ_dims = other_dims[:axis] + [dim] + other_dims[axis:]
                assert set(templ_dims) == set(
                    source.dims
                ), f"concat {source} {dim} not allowed with allow_broadcast=False"
                source_ = source.copy_transpose(templ_dims)
                sources_raw.append(source_.raw_tensor)
        out = Tensor(
            "concat",
            dims=other_dims[:axis] + [out_dim] + other_dims[axis:],
            dtype=sources[0][0].dtype,
            sparse_dim=sources[0][0].sparse_dim,
        )
        if sources[0][0].feature_dim and sources[0][0].feature_dim != sources[0][1]:
            out.feature_dim = sources[0][0].feature_dim
        out.raw_tensor = torch.cat([s for s in sources_raw], dim=axis)
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
            [out_dims[axes.index(dim)] if dim in axes else dim for dim in source.dim_tags], keep_special_axes=True
        )
        remaining_dims = set(axes)
        raw_pad = []
        for dim in reversed(source.dims):
            if dim not in remaining_dims:
                raw_pad += [0, 0]
                continue
            remaining_dims.remove(dim)
            pad_ = padding[axes.index(dim)]
            raw_pad += [
                pad_[0].get_dim_value() if isinstance(pad_[0], Dim) else pad_[0],
                pad_[1].get_dim_value() if isinstance(pad_[1], Dim) else pad_[1],
            ]
            if not remaining_dims:
                break
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
        source_raw = source.copy_compatible_to_dims_raw(prev_accum.dims)
        out.raw_tensor = torch.cat((prev_accum.raw_tensor, source_raw), dim=prev_accum.get_axis_from_description(axis))
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
        out_raw = torch.softmax(tensor.raw_tensor, dim=tensor.dims.index(axis))
        out.dtype = TorchBackend.get_dtype_name_raw(out_raw)
        out.raw_tensor = out_raw
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
        out_raw = torch.log_softmax(tensor.raw_tensor, dim=tensor.dims.index(axis))
        out.dtype = TorchBackend.get_dtype_name_raw(out_raw)
        out.raw_tensor = out_raw
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
    def ctc_loss(
        *,
        logits: Tensor,
        targets: Tensor,
        input_spatial_dim: Dim,
        targets_spatial_dim: Dim,
        blank_index: int,
        max_approx: bool = False,
    ) -> Tensor:
        """CTC"""
        if max_approx:
            raise NotImplementedError("ctc_loss: max_approx not implemented for PyTorch")
        assert targets.sparse_dim and targets.sparse_dim.dimension <= logits.feature_dim.dimension
        # PyTorch expects the logits to be of shape (T, B, C) where T is the input spatial dim.
        batch_dims = logits.remaining_dims((input_spatial_dim, logits.feature_dim))
        logits = logits.copy_transpose([input_spatial_dim] + batch_dims + [logits.feature_dim])
        logits_raw: torch.Tensor = logits.raw_tensor
        input_lengths = input_spatial_dim.dyn_size_ext.copy_transpose(batch_dims).raw_tensor
        logits_raw_shape = logits_raw.shape  # [T, B..., C]
        if len(batch_dims) != 1:
            logits_raw = torch.reshape(logits_raw, logits_raw.shape[:1] + (-1,) + logits_raw.shape[-1:])  # [T, B', C]
            input_lengths = torch.reshape(input_lengths, (-1,))  # [B']
        log_probs = torch.nn.functional.log_softmax(logits_raw, dim=-1)
        # PyTorch expects the targets to be of shape (B, S) where S is the targets spatial dim.
        targets = targets.copy_transpose(batch_dims + [targets_spatial_dim])
        targets_raw = targets.raw_tensor  # [B..., S]
        targets_lengths = targets_spatial_dim.dyn_size_ext.copy_transpose(batch_dims).raw_tensor
        if len(batch_dims) != 1:
            targets_raw = torch.reshape(targets_raw, (-1, targets_raw.shape[-1]))  # [B', S]
            targets_lengths = torch.reshape(targets_lengths, (-1,))  # [B']
        loss_raw = torch.nn.functional.ctc_loss(
            log_probs=log_probs,
            targets=targets_raw,
            input_lengths=input_lengths,
            target_lengths=targets_lengths,
            blank=blank_index,
            zero_infinity=True,
            reduction="none",
        )
        if len(batch_dims) != 1:
            loss_raw = torch.reshape(loss_raw, logits_raw_shape[1:-1])
        loss = Tensor(
            name="ctc_loss",
            dims=batch_dims,
            raw_tensor=loss_raw,
            dtype=logits.dtype,
        )
        return loss

    @staticmethod
    def create_parameter_raw(tensor: rf.Parameter, *, device: Optional[str] = None) -> torch.nn.Parameter:
        """
        :return: parameter
        """
        assert all(d.is_static() for d in tensor.dims)
        data = torch.zeros(
            [d.dimension for d in tensor.dims],
            dtype=TorchBackend.as_dtype_raw(tensor.dtype),
            device=device or rf.get_default_device(),
        )
        if tensor.dtype.startswith("int") or tensor.dtype == "bool":
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
                value_raw = value.copy_compatible_to_dims_raw(param.dims)
                raw_param.copy_(value_raw)
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
    def parameter_assign(param: rf.Parameter, value: Tensor, *, op: str = "assign") -> None:
        """param assign"""
        raw_param = param.raw_tensor
        assert isinstance(raw_param, torch.nn.Parameter)
        value_raw = value.copy_compatible_to_dims_raw(param.dims)
        with torch.no_grad():
            if op == "assign":
                raw_param.copy_(value_raw)
            elif op == "add":
                raw_param.add_(value_raw)
            else:
                raise ValueError(f"Parameter {param} assign: Unsupported op: {op}")

    @staticmethod
    def parameter_assign_key(
        param: rf.Parameter,
        key: rf.ItemKeyType,
        value: Tensor,
        *,
        op: str = "assign",
        axis: Optional[Union[Dim, Sequence[Dim]]] = None,
        key_dim: Optional[Union[Dim, Sequence[Dim]]] = None,
    ) -> None:
        """param assign"""
        raw_param = param.raw_tensor
        assert isinstance(raw_param, torch.nn.Parameter)
        key_raw, res_dims = _utils.strided_slice_raw_key(param, axis, key, key_dim)
        value_raw = value.copy_compatible_to_dims_raw(res_dims)
        with torch.no_grad():
            if op == "assign":
                raw_param[key_raw] = value_raw
            elif op == "add":
                raw_param[key_raw] += value_raw
            else:
                raise ValueError(f"Parameter {param} assign: Unsupported op: {op}")

    # keep in sync with native implementation
    @staticmethod
    def compare_raw(a: torch.Tensor, kind: str, b: torch.Tensor) -> torch.Tensor:
        """
        :param a:
        :param kind: "equal", "less", "less_equal", "greater", "greater_equal", "not_equal"
        :param b:
        :return: a `kind` b
        """
        assert a.dim() == b.dim() or a.dim() == 0 or b.dim() == 0
        if kind == "equal":
            kind = "eq"  # eq is different to equal; eq returns a torch Tensor
        op = getattr(torch, kind)  # e.g. torch.equal
        return op(a, b)

    # keep in sync with native implementation
    @staticmethod
    def combine_raw(a: torch.Tensor, kind: str, b: torch.Tensor) -> torch.Tensor:
        """
        :param a:
        :param kind: "add", "sub", "mul", "truediv", "floordiv", "mod", "pow",
            "maximum", "minimum", "logical_and", "logical_or", "squared_difference"
        :param b:
        :return: a `kind` b
        """
        assert a.dim() == b.dim() or a.dim() == 0 or b.dim() == 0
        if kind == "squared_difference":
            return raw_ops.squared_difference(a, b)
        kind = {
            "truediv": "true_divide",
            "floordiv": "floor_divide",
            "mod": "remainder",
        }.get(kind, kind)
        op = getattr(torch, kind)  # e.g. torch.add
        return op(a, b)

    @staticmethod
    def reshape_raw(
        raw_tensor: torch.Tensor, shape: Union[Sequence[Union[int, torch.Tensor]], torch.Tensor]
    ) -> torch.Tensor:
        """
        :param raw_tensor:
        :param shape:
        :return: reshaped raw tensor; wraps torch.reshape
        """
        return torch.reshape(raw_tensor, shape)

    @classmethod
    def squeeze_raw(cls, raw_tensor: torch.Tensor, axes: Sequence[int]) -> torch.Tensor:
        """squeeze"""
        if len(axes) == 1:
            return raw_tensor.squeeze(dim=axes[0])
        elif len(axes) == 0:
            return raw_tensor
        else:
            return super().squeeze_raw(raw_tensor, axes=axes)

    @staticmethod
    def transpose_raw(raw_tensor: torch.Tensor, perm: Sequence[int]) -> torch.Tensor:
        """
        :param raw_tensor:
        :param perm: e.g. [0, 2, 1]
        :return: permuted (transposed) raw tensor; wraps torch.permute
        """
        if all(p == i for i, p in enumerate(perm)):
            return raw_tensor
        return torch.permute(raw_tensor, tuple(perm))

    @staticmethod
    def convert_to_tensor(
        value: Union[Tensor, torch.Tensor, RawTensorTypes],
        *,
        dims: Sequence[Dim],
        dtype: str,
        sparse_dim: Optional[Dim] = None,
        device: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Tensor[torch.Tensor]:
        """
        :param value:
        :param dims:
        :param dtype:
        :param sparse_dim:
        :param device:
        :param name:
        :return: tensor
        """
        if isinstance(value, Tensor):
            return value
        if isinstance(value, torch.Tensor):
            name = name or "raw_tensor"
        else:
            name = name or "const"
            value = torch.tensor(
                value,
                dtype=TorchBackend.as_dtype_raw(dtype),
                device=device or rf.get_default_device(),
            )
        assert isinstance(value, torch.Tensor)
        return Tensor(name, dims=dims, dtype=dtype, sparse_dim=sparse_dim, raw_tensor=value)

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
        shape = [dim.get_dim_value() for dim in dims]
        if isinstance(fill_value, Tensor):
            fill_value = fill_value.raw_tensor
        if torch.onnx.is_in_onnx_export():
            # onnx::ConstantOfShape (via torch.full) must get shape as int64.
            # https://github.com/rwth-i6/returnn/issues/1333#issuecomment-1607236783
            shape = [dim.long() if isinstance(dim, torch.Tensor) else dim for dim in shape]
        raw_tensor = torch.full(
            shape, fill_value, dtype=TorchBackend.as_dtype_raw(dtype), device=device or rf.get_default_device()
        )
        return Tensor(
            "full", dims=dims, sparse_dim=sparse_dim, feature_dim=feature_dim, dtype=dtype, raw_tensor=raw_tensor
        )

    @staticmethod
    def gather(
        source: Tensor,
        *,
        indices: Union[Tensor, int],
        axis: Dim,
        clip_to_valid: bool = False,
    ) -> Tensor:
        """
        Gather.

        There are a few options in PyTorch, all having somewhat different semantics
        and different advantages or disadvantages and different limitations.

        - torch.gather, most generic
        - torch.index_select, similar as tf.gather, but does not support batch axes
        - Tensor.__getitem__
        - torch.embedding
        """
        if isinstance(indices, Tensor):
            indices: Tensor[torch.Tensor]
        elif isinstance(indices, int):
            indices = Tensor(
                "indices_int",
                dims=(),
                dtype=rf.get_default_array_index_dtype(),
                raw_tensor=torch.tensor(indices, dtype=TorchBackend.as_dtype_raw(rf.get_default_array_index_dtype())),
            )
        else:
            raise TypeError(f"Unsupported type for indices: {type(indices)}")
        axis_int = source.get_axis_from_description(axis, allow_int=False)
        if clip_to_valid:
            indices = indices.copy()
            dim: Dim = source.dims[axis_int]
            if dim.dyn_size_ext:
                assert dim.dyn_size_ext.dims_set.issubset(
                    indices.dims_set
                ), f"gather with clip_to_valid: indices ({indices}) dims must be a superset of {dim} dyn-size"
                size = dim.dyn_size_ext.copy_compatible_to(indices, check_sparse=False)
                indices.raw_tensor = torch.clamp(indices.raw_tensor, torch.tensor(0), size.raw_tensor - 1)
            else:
                indices.raw_tensor = torch.clamp(indices.raw_tensor, 0, source.raw_tensor.shape[axis_int] - 1)
        index_own_dims = [dim for dim in indices.dims if dim not in source.dims or dim == axis]
        out = Tensor(
            "gather",
            dims=list(source.dims[:axis_int]) + index_own_dims + list(source.dims[axis_int + 1 :]),
            dtype=source.dtype,
            sparse_dim=source.sparse_dim,
        )
        if source.feature_dim and source.feature_dim in out.dims:
            out.feature_dim = source.feature_dim
        if indices.dims_set.intersection(source.dims_set - {axis}):
            # We cannot use index_select in this case. Need to fallback to gather.
            indices = indices.copy_compatible_to(out, check_dtype=False, check_sparse=False, unbroadcast=True)
            if len(index_own_dims) == 1:
                index_own_dims_flat = index_own_dims[0]  # good
            elif len(index_own_dims) == 0:
                index_own_dims_flat = Dim(1, name="dummy")
                indices = indices.copy_add_dim_by_tag(index_own_dims_flat, unbroadcast=True, axis=axis_int)
            else:
                indices, index_own_dims_flat = rf.merge_dims(indices, dims=index_own_dims)
            index_ext_dims = list(source.dims)
            index_ext_dims[axis_int] = index_own_dims_flat
            assert indices.dims == tuple(index_ext_dims)
            out_raw = torch.gather(source.raw_tensor, dim=axis_int, index=indices.raw_tensor.type(torch.int64))
            if len(index_own_dims) == 1:
                pass  # nothing to do
            elif len(index_own_dims) == 0:
                out_raw = out_raw.squeeze(axis_int)
            else:
                out_raw = out_raw.reshape([d.get_dim_value() for d in out.dims])
            out.raw_tensor = out_raw
        elif axis_int == 0 and indices.batch_ndim == 0:
            out.raw_tensor = source.raw_tensor[indices.raw_tensor]
        elif axis_int == 0 and source.batch_ndim == 2:
            # This is exactly what torch.embedding is intended for. Let's use that.
            out.raw_tensor = torch.embedding(source.raw_tensor, indices.raw_tensor)
        else:
            out_raw = torch.index_select(source.raw_tensor, dim=axis_int, index=indices.raw_tensor.flatten())
            out_shape = (
                source.raw_tensor.shape[:axis_int] + indices.raw_tensor.shape + source.raw_tensor.shape[axis_int + 1 :]
            )
            out.raw_tensor = out_raw.reshape(out_shape)
        return out

    @staticmethod
    def scatter(
        source: Tensor,
        *,
        indices: Tensor,
        indices_dim: Union[Dim, Sequence[Dim]],
        out_dim: Union[Dim, Sequence[Dim]],
    ) -> Tensor:
        """
        Scatters into new zero-tensor.
        If entries in indices are duplicated, the corresponding values in source will be added together
        (scatter_add in PyTorch).
        (TF segment_sum can be implemented via this.)

        :param source: [batch_dims..., indices_dim(s)..., feature_dims...]
        :param indices: [batch_dims..., indices_dim(s)...] -> out_dim
        :param indices_dim:
        :param out_dim:
        :return: [batch_dims..., out_dim, feature_dims...]
        """
        if isinstance(indices_dim, Dim):
            indices_dim = [indices_dim]
        else:
            assert len(indices_dim) >= 1
            indices_dim = list(indices_dim)
        assert indices.dtype.startswith("int")
        if isinstance(out_dim, Dim):
            out_flat_dim = out_dim
            out_dim = [out_dim]
        elif len(out_dim) == 1:
            out_flat_dim = out_dim[0]
            out_dim = [out_flat_dim]
        else:
            assert len(out_dim) > 1
            out_flat_dim = out_dim[0]
            for dim in out_dim[1:]:
                out_flat_dim = out_flat_dim * dim
            out_dim = list(out_dim)
        batch_dims = indices.remaining_dims(indices_dim)
        feature_dims = source.remaining_dims(batch_dims + indices_dim)
        if len(indices_dim) > 1:
            indices, indices_flat_dim = rf.merge_dims(indices, dims=indices_dim)
            source, _ = rf.merge_dims(source, dims=indices_dim, out_dim=indices_flat_dim)
        else:
            indices_flat_dim = indices_dim[0]
        source = source.copy_transpose(batch_dims + [indices_flat_dim] + feature_dims)
        indices = indices.copy_compatible_to(
            # scatter_add_ does not support broadcasting, and expects indices and source to have the same number of dims
            source,
            unbroadcast=True,
            add_dims=True,
            check_sparse=False,
            check_dtype=False,
        )
        out_dims = batch_dims + [out_flat_dim] + feature_dims
        out_shape = [d.get_dim_value() for d in out_dims]
        out_raw = torch.zeros(out_shape, dtype=source.raw_tensor.dtype, device=source.raw_tensor.device)
        out_raw.scatter_add_(dim=len(batch_dims), index=indices.raw_tensor.to(torch.int64), src=source.raw_tensor)
        res = Tensor(
            "scatter",
            dims=out_dims,
            dtype=source.dtype,
            sparse_dim=source.sparse_dim,
            raw_tensor=out_raw,
        )
        if len(out_dim) > 1:
            res = rf.split_dims(res, axis=out_flat_dim, dims=out_dim)
        return res

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
    def where(
        cond: Tensor,
        true_: Union[Tensor, rf.RawTensorTypes],
        false_: Union[Tensor, rf.RawTensorTypes],
        *,
        allow_broadcast_all_sources: bool = False,
    ) -> Tensor:
        """where"""
        true_ = rf.convert_to_tensor(true_, _backend=TorchBackend, device=cond.device)
        false_ = rf.convert_to_tensor(false_, _backend=TorchBackend, device=cond.device)
        out = Tensor.get_common_data(
            [true_, false_, cond], allow_broadcast_all_sources=allow_broadcast_all_sources, name="where"
        )
        out.dtype = true_.dtype
        out.sparse_dim = true_.sparse_dim
        cond_bc_raw = cond.copy_compatible_to_dims_raw(out.dims)
        true_bc_raw = true_.copy_compatible_to_dims_raw(out.dims)
        false_bc_raw = false_.copy_compatible_to_dims_raw(out.dims)
        out.raw_tensor = torch.where(cond_bc_raw, true_bc_raw, false_bc_raw)
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

        # First check whether we can use torch.nn.functional.linear.
        if len(b_common_axes) == 0 and len(b_reduce_axes) == 1 and len(b_unique_axes) == 1:
            a_raw = torch.permute(a_raw, a_unique_axes + a_reduce_axes)
            b_raw = torch.permute(b_raw, b_unique_axes + b_reduce_axes)

            raw_result = torch.nn.functional.linear(a_raw, b_raw)

        else:
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

        result_tensor = Tensor(
            name="dot", dims=result_dims, raw_tensor=raw_result, dtype=TorchBackend.get_dtype_name_raw(raw_result)
        )

        return result_tensor

    @staticmethod
    def range_over_dim(dim: Dim, *, dtype: Optional[str] = None, device: Optional[str] = None) -> Tensor[torch.Tensor]:
        """
        :param dim:
        :param dtype:
        :param device:
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
        out.raw_tensor = torch.arange(
            dim.get_dim_value(), dtype=TorchBackend.as_dtype_raw(out.dtype), device=device or rf.get_default_device()
        )
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
        raw_dims = [source.get_axis_from_description(dim) for dim in axis]
        res_dims = [dim for i, dim in enumerate(source.dims) if i not in raw_dims]
        correction_factor: Optional[torch.Tensor] = None
        if use_mask and any(dim.need_masking() for dim in axis):
            source = source.copy()
            dtype = source.raw_tensor.dtype
            if mode == "max":
                mask_value = torch.finfo(dtype).min if dtype.is_floating_point else torch.iinfo(dtype).min
            elif mode == "min":
                mask_value = torch.finfo(dtype).max if dtype.is_floating_point else torch.iinfo(dtype).max
            elif mode == "sum":
                mask_value = 0
            elif mode == "mean":
                mask_value = 0
                for dim in axis:
                    if dim.need_masking():
                        total_num_el = dim.get_dim_value_tensor()
                        actual_num_el = dim.get_size_tensor()
                        num_el_reduce_dims = [dim_ for dim_ in axis if dim_ in actual_num_el.dims]
                        if num_el_reduce_dims:
                            actual_num_el = rf.reduce_sum(actual_num_el, axis=num_el_reduce_dims)
                            for dim_ in num_el_reduce_dims:
                                total_num_el *= dim_.get_dim_value_tensor()
                        correction_factor_ = rf.cast(total_num_el, source.dtype) / rf.cast(actual_num_el, source.dtype)
                        correction_factor__ = correction_factor_.copy_compatible_to_dims_raw(res_dims)
                        if correction_factor is None:
                            correction_factor = correction_factor__
                        else:
                            correction_factor *= correction_factor__
            else:
                raise NotImplementedError(f"reduce_{mode} not implemented with masking on tensor {source!r}.")
            for dim in axis:
                if dim.need_masking():
                    mask = source.get_sequence_mask_broadcast(dim)
                    source.raw_tensor = torch.where(mask, source.raw_tensor, mask_value)
        func = getattr(torch, mode)
        if not res_dims:
            raw_result = func(source.raw_tensor)
        elif len(raw_dims) == 1:
            raw_result = func(source.raw_tensor, dim=raw_dims[0])
            if mode in ["max", "min"]:
                # result is a tuple (values, indices). https://pytorch.org/docs/stable/generated/torch.max.html
                raw_result, _ = raw_result
        else:
            raw_result = func(source.raw_tensor, dim=raw_dims)
        if correction_factor is not None:
            raw_result *= correction_factor.to(raw_result.device)
        res = Tensor(
            name=f"reduce_{mode}",
            raw_tensor=raw_result,
            dims=res_dims,
            dtype=TorchBackend.get_dtype_name_raw(raw_result),
            sparse_dim=axis[0] if mode.startswith("arg") else source.sparse_dim,
        )
        return res

    # noinspection PyShadowingBuiltins
    @staticmethod
    def top_k(
        source: Tensor[torch.Tensor],
        *,
        axis: Union[Dim, Sequence[Dim]],
        k: Union[int, Tensor],
        k_dim: Optional[Dim] = None,
        sorted: bool = True,
    ) -> Tuple[Tensor, Union[Tensor, Sequence[Tensor]], Dim]:
        """top_k"""
        if not k_dim:
            k_dim = Dim(k, name="top-k-dim")
        axes = [axis] if isinstance(axis, Dim) else axis
        if any(a.need_masking() for a in axes):
            mask_value = (
                torch.finfo(source.raw_tensor.dtype).min
                if source.raw_tensor.dtype.is_floating_point
                else torch.iinfo(source.raw_tensor.dtype).min
            )
            source = source.copy()
            for a in axes:
                if a.need_masking():
                    source = rf.where(
                        a.get_mask(dim_order=source.dims, device=source.device),
                        source,
                        mask_value,
                    )
        if isinstance(axis, (list, tuple)):
            # Move axis to the end, in the right order.
            source = source.copy_transpose([d for d in source.dims if d not in axis] + list(axis))
            source_raw_flat = source.raw_tensor.flatten(start_dim=source.batch_ndim - len(axis))
            values_raw, indices_raw = torch.topk(
                source_raw_flat, k=k_dim.get_dim_value(), dim=-1, largest=True, sorted=sorted
            )
            values = source.copy_template_new_dim_tags(
                new_dim_tags=source.dims[: -len(axis)] + (k_dim,), name="top_k_values"
            )
            if source.feature_dim and source.feature_dim in values.dims:
                values.feature_dim = source.feature_dim
            values.raw_tensor = values_raw
            indices_out = []
            for i, a in reversed(list(enumerate(axis))):
                assert isinstance(a, Dim)
                indices_out_raw = indices_raw % a.dimension
                indices_raw = indices_raw // a.dimension
                indices = values.copy_template(name=f"top_k_indices_{a.name or i}")
                indices.dtype = TorchBackend.get_dtype_name_raw(indices_out_raw)
                indices.sparse_dim = a
                indices.raw_tensor = indices_out_raw
                indices_out.insert(0, indices)
            return values, indices_out, k_dim
        assert isinstance(axis, Dim)
        axis_int = source.get_axis_from_description(axis, allow_int=False)
        values_raw, indices_raw = torch.topk(
            source.raw_tensor, k=k_dim.get_dim_value(), dim=axis_int, largest=True, sorted=sorted
        )
        values = source.copy_template_replace_dim_tag(axis=axis_int, new_dim_tag=k_dim, name="top_k_values")
        values.raw_tensor = values_raw
        indices = source.copy_template_replace_dim_tag(axis=axis_int, new_dim_tag=k_dim, name="top_k_indices")
        indices.dtype = TorchBackend.get_dtype_name_raw(indices_raw)
        indices.sparse_dim = axis
        indices.raw_tensor = indices_raw
        return values, indices, k_dim

    @staticmethod
    @contextlib.contextmanager
    def random_journal_record() -> Generator[_random_journal.RandomJournal]:
        """
        :return: the journal
        """
        prev_journal = TorchBackend._random_journal
        try:
            TorchBackend._random_journal = _random_journal.RandomJournal()
            yield TorchBackend._random_journal
        finally:
            TorchBackend._random_journal = prev_journal

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
        out: Optional[Tensor[torch.Tensor]] = None,
    ) -> Tensor:
        """
        random. See `rf.random` for details.
        """
        shape = [d.get_dim_value() for d in dims]
        dtype_ = TorchBackend.as_dtype_raw(dtype)
        if out is None:
            out = Tensor(
                name=f"random_{distribution}", dims=dims, dtype=dtype, sparse_dim=sparse_dim, feature_dim=feature_dim
            )
            out.raw_tensor = torch.empty(shape, dtype=dtype_, device=device or rf.get_default_device())
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
        if TorchBackend._random_journal:
            out_ = out.copy()
            out_.raw_tensor = out_.raw_tensor.detach().cpu().numpy()
            TorchBackend._random_journal.append(
                distribution=distribution,
                mean=mean,
                stddev=stddev,
                bound=bound,
                minval=minval,
                maxval=maxval,
                seed=seed,
                static=static,
                out=out_,
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
        remaining_dims = [d for d in tensor.dims if d not in mask.dims]
        tensor_templ_dims = tuple(dims) + tuple(remaining_dims)
        in_raw = tensor.copy_compatible_to_dims_raw(tensor_templ_dims)
        mask_raw = mask.copy_compatible_to_dims_raw(tensor_templ_dims)
        # We have a very strange problem with the gradient of masked_select,
        # when used together with some specific other operations before that,
        # like convolution.
        # This clone() with contiguous_format seems to fix the problem.
        # https://github.com/pytorch/pytorch/issues/99638
        in_raw = in_raw.clone(memory_format=torch.contiguous_format)
        if mask_raw.device.type == "meta":
            # This is not supported, but also, we would anyway not know the out shape.
            # However, instead of erroring, just assume some dummy mask.
            # https://github.com/pytorch/pytorch/issues/109871
            out_raw = in_raw.flatten()
        else:
            out_raw = torch.masked_select(in_raw, mask_raw)
        remaining_shape = [d.get_dim_value() for d in remaining_dims]
        remaining_num_elements = numpy.prod(remaining_shape) if remaining_shape else 1
        assert out_raw.numel() % remaining_num_elements == 0
        flattened_num_elements = out_raw.numel() // remaining_num_elements
        out_raw = torch.reshape(out_raw, [flattened_num_elements] + remaining_shape)
        if not out_dim:
            out_dim = Dim(None, name="masked_select")
        if not out_dim.dyn_size_ext:
            out_dim.dyn_size_ext = Tensor("masked_select_size", dims=(), dtype="int64")
        if out_dim.dyn_size_ext.raw_tensor is None:
            out_dim.dyn_size_ext.raw_tensor = torch.tensor(flattened_num_elements, dtype=torch.int64)
        out = Tensor(
            "masked_select",
            dims=(out_dim,) + tuple(remaining_dims),
            dtype=tensor.dtype,
            sparse_dim=tensor.sparse_dim,
            raw_tensor=out_raw,
        )
        return out, out_dim

    @staticmethod
    def masked_scatter(source: Tensor, *, mask: Tensor, dims: Sequence[Dim], in_dim: Dim) -> Tensor:
        """masked scatter"""
        assert mask.dtype == "bool"
        assert set(mask.dims) == set(dims)
        assert in_dim in source.dims
        remaining_dims = [d for d in source.dims if d not in mask.dims and d != in_dim]
        source_templ_dims = (in_dim,) + tuple(remaining_dims)
        tensor_templ_dims = tuple(dims) + tuple(remaining_dims)
        source_raw = source.copy_compatible_to_dims_raw(source_templ_dims)
        mask_raw = mask.copy_compatible_to_dims_raw(tensor_templ_dims)
        out_shape = [d.get_dim_value() for d in tensor_templ_dims]
        out_raw = torch.zeros(out_shape, dtype=source_raw.dtype, device=source_raw.device)
        out_raw.masked_scatter_(mask_raw, source_raw)
        return Tensor(
            "masked_scatter",
            dims=tensor_templ_dims,
            dtype=source.dtype,
            sparse_dim=source.sparse_dim,
            raw_tensor=out_raw,
        )

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
        if len(batch_dims) == 1:
            src_raw = source.raw_tensor
        else:
            src_raw = torch.reshape(
                source.raw_tensor,
                # potentially merge batch dims all together
                [-1, in_dim.get_dim_value()] + [d.get_dim_value() for d in in_spatial_dims],
            )
        use_striding = strides and (strides > 1 if isinstance(strides, int) else any(s > 1 for s in strides))
        if padding == "same" and not use_striding and all(d.dimension % 2 == 1 for d in filter_size):
            if all(filter_size[0].dimension == d.dimension for d in filter_size):  # all same
                padding = (filter_size[0].dimension - 1) // 2
            else:
                padding = tuple(d.dimension // 2 for d in filter_size)
        if padding == "same" and (use_striding or torch.onnx.is_in_onnx_export()):
            # padding='same' is not supported for strided convolutions.
            # Moreover, padding specified as a string isn't supported for ONNX exporting as of 2023/05/19.
            # Manual add the padding, and then do not use any padding in the conv.
            padding = 0
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
        if padding == "valid":
            # padding as string is not supported e.g. in ONNX.
            padding = 0
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
        out = Tensor(
            "conv", dims=batch_dims + [out_dim] + list(out_spatial_dims), dtype=TorchBackend.get_dtype_name_raw(out_raw)
        )
        if len(batch_dims) == 1:
            out.raw_tensor = out_raw
        else:
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
        out = Tensor("pool", dims=batch_dims + list(out_spatial_dims), dtype=source.dtype)
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

        window_pt = torch.hann_window(frame_length, device=x_raw.device)
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
        squeeze_spatial_dim = False
        if spatial_dim == single_step_dim:
            spatial_dim = Dim(1, name="dummy-spatial-dim-single-step")
            source = source.copy_add_dim_by_tag(spatial_dim, unbroadcast=True, axis=0)
            squeeze_spatial_dim = True

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

        if squeeze_spatial_dim:
            out = out.copy_squeeze_axes([out.get_axis_from_description(spatial_dim)])

        return out, (new_state_h, new_state_c)

    TensorArrayType = List[Tensor]

    @staticmethod
    def tensor_array_unstack(tensor: Tensor, *, axis: Dim) -> TensorArrayType:
        """unstack"""
        axis_int = tensor.get_axis_from_description(axis)
        out_tensors_raw = torch.unbind(tensor.raw_tensor, dim=axis_int)
        out_tensor_template = tensor.copy_template().copy_template_excluding_axis(axis_int)
        out_tensors = []
        for out_tensor_raw in out_tensors_raw:
            out_tensor = out_tensor_template.copy_template()
            out_tensor.raw_tensor = out_tensor_raw
            out_tensors.append(out_tensor)
        return out_tensors

    @staticmethod
    def tensor_array_stack(tensor_array: TensorArrayType, *, axis: Dim, tensor_template: Tensor) -> Tensor:
        """stack"""
        if tensor_array:
            # In the actual array, the tensors might be a better template (different dim order).
            # We already checked in TensorArray that they are compatible.
            tensor_template = tensor_array[0].copy_template()
        out_tensor = tensor_template.copy_add_dim_by_tag(axis, unbroadcast=True, axis=0)
        if not tensor_array:
            return rf.zeros_like(out_tensor)
        tensor_array_raw = [tensor.copy_transpose(tensor_template.dims).raw_tensor for tensor in tensor_array]
        out_tensor_raw = torch.stack(tensor_array_raw, dim=0)
        out_tensor.raw_tensor = out_tensor_raw
        return out_tensor
