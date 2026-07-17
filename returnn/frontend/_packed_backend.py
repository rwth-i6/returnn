"""
Packed (ragged) tensor storage behind the normal RF Tensor/Dim API, as an opt-in backend.

A packed tensor stores data without padding:
e.g. dims [batch, time(dyn), feature] are stored as [packed, feature],
where packed = sum of seq lens.
The outer :class:`Tensor` keeps the original (virtual, unpacked) dims,
so model code does not change;
the packed storage and the packing relation live in the raw-tensor wrapper
(:class:`PackedRawTensor`).

Design (see https://github.com/rwth-i6/returnn/issues/1645 and #466):
transparent packed support inside every op was rejected
(per-op storage-type check = overhead also for the normal padded case).
Instead, the packed state is a raw-tensor *type*,
so the existing per-type backend dispatch (:func:`get_backend_by_raw_tensor_type`)
routes packed tensors to :class:`PackedBackend` --
zero overhead for normal tensors.

Three layers:

- Generic, inner-backend-agnostic:
  ops that do not touch the packed dims (elementwise, matmul over feature, softmax over vocab, ...)
  apply the inner backend op directly on the packed raw data.
  Also reductions over *all* packed dims jointly (e.g. mean over (batch, time))
  run directly on the packed data, without any masking.
- Per-backend specializations (TODO, not yet here):
  ops with native varlen kernels,
  e.g. attention via FlashAttention varlen (cu_seqlens) / FlexAttention doc-masking on torch.
- Fallback for ops needing the sequence structure: unpack, then run the op on padded storage.
  Note: the fallback *decays* to padded storage --
  the result is a normal padded tensor, and downstream computation stays padded
  (correct first; keeping more ops packed comes incrementally).

Known limitations (TODO):

- Mixed binary ops only work packed-first (``packed + plain``);
  ``plain + packed`` dispatches on the plain backend which cannot handle the wrapper.
- ``dim_order`` in :func:`combine` is ignored on the packed fast path.

Status: early skeleton. Import this module explicitly to activate the dispatch registration.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple, Union

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from ._backend import Backend, register_backend_by_tensor_type

__all__ = ["PackedRawTensor", "PackedBackend", "pack", "unpack", "is_packed"]


class PackedRawTensor:
    """
    Raw-tensor wrapper marking packed storage.

    The wrapped :attr:`inner` is a normal RF :class:`Tensor` (of the inner backend)
    holding the packed data, dims = [packed_dim] + remaining (non-packed) dims.
    :attr:`orig_dims` are the dims packed into :attr:`packed_dim`, e.g. (batch, time);
    the seq lens live in the dyn dims' dyn_size_ext as usual, nothing is duplicated here.
    """

    def __init__(self, *, inner: Tensor, packed_dim: Dim, orig_dims: Sequence[Dim]):
        self.inner = inner
        self.packed_dim = packed_dim
        self.orig_dims = tuple(orig_dims)

    @property
    def inner_backend(self):
        """:return: backend of the inner (packed data) tensor"""
        # noinspection PyProtectedMember
        return self.inner._raw_backend

    def virtual_ndim(self) -> int:
        """:return: ndim of the virtual (unpacked) view, i.e. of the outer Tensor"""
        return self.inner.batch_ndim - 1 + len(self.orig_dims)

    def virtual_dims(self, inner: Optional[Tensor] = None) -> Tuple[Dim, ...]:
        """:return: dims of the virtual (unpacked) view, for the given inner tensor (default: self.inner)"""
        if inner is None:
            inner = self.inner
        return tuple(self.orig_dims) + tuple(d for d in inner.dims if d != self.packed_dim)

    def rewrap(self, inner_out: Tensor, *, name: Optional[str] = None) -> Tensor:
        """
        :param inner_out: result of an op on :attr:`inner` which kept :attr:`packed_dim`
        :param name:
        :return: outer (virtual-dims) Tensor wrapping inner_out with the same packing relation
        """
        assert self.packed_dim in inner_out.dims
        out = Tensor(
            name=name or inner_out.name,
            dims=self.virtual_dims(inner_out),
            dtype=inner_out.dtype,
            sparse_dim=inner_out.sparse_dim,
            feature_dim=inner_out.feature_dim if inner_out.feature_dim != self.packed_dim else None,
        )
        out.raw_tensor = PackedRawTensor(inner=inner_out, packed_dim=self.packed_dim, orig_dims=self.orig_dims)
        return out


def _raw(tensor: Tensor) -> PackedRawTensor:
    raw = tensor.raw_tensor
    assert isinstance(raw, PackedRawTensor)
    return raw


def _unpack_if_packed(x):
    if isinstance(x, Tensor) and isinstance(x.raw_tensor, PackedRawTensor):
        return unpack(x)
    return x


# We do not expect to ever implement all methods of the Backend interface here --
# only what is profitable on packed data; the rest falls back (unpack, then padded op).
# noinspection PyAbstractClass
class PackedBackend(Backend[PackedRawTensor]):
    """
    Backend for packed tensors, wrapping any inner backend.

    Selected via the normal raw-tensor-type dispatch;
    the actual computation is delegated to the inner backend of the wrapped tensor.
    """

    name = "packed"
    RawTensorType = PackedRawTensor

    @staticmethod
    def executing_eagerly() -> bool:
        """executing eagerly. all inner backends we compose with are eager."""
        return True

    @staticmethod
    def get_dtype_name_raw(raw_tensor: PackedRawTensor) -> str:
        """:return: dtype of the packed data"""
        return raw_tensor.inner.dtype

    @staticmethod
    def get_ndim_raw(raw_tensor: PackedRawTensor) -> int:
        """:return: ndim of the virtual (unpacked) view, matching the outer Tensor dims"""
        return raw_tensor.virtual_ndim()

    @staticmethod
    def get_known_shape_raw(raw_tensor: PackedRawTensor) -> Tuple[Optional[int], ...]:
        """:return: shape of the virtual (unpacked) view; dynamic entries are None"""
        return tuple(d.dimension for d in raw_tensor.virtual_dims())

    @staticmethod
    def get_device(x: Tensor) -> Optional[str]:
        """:return: device of the packed data"""
        return _raw(x).inner.device

    @staticmethod
    def stop_gradient(tensor: Tensor) -> Tensor:
        """stop gradient -- on the packed data"""
        raw = _raw(tensor)
        return raw.rewrap(rf.stop_gradient(raw.inner), name="stop_gradient")

    @staticmethod
    def activation_raw(raw_tensor: PackedRawTensor, func: str) -> PackedRawTensor:
        """elementwise -- applied directly on the packed data"""
        inner_out = raw_tensor.inner.copy_template(name=func)
        inner_out.raw_tensor = raw_tensor.inner_backend.activation_raw(raw_tensor.inner.raw_tensor, func)
        return PackedRawTensor(inner=inner_out, packed_dim=raw_tensor.packed_dim, orig_dims=raw_tensor.orig_dims)

    @classmethod
    def combine(
        cls,
        a: Union[Tensor, Any],
        kind: str,
        b: Union[Tensor, Any],
        *,
        allow_broadcast_all_sources: Optional[bool] = None,
        dim_order: Optional[Sequence[Dim]] = None,
    ) -> Tensor:
        """
        binary op. fast paths:
        both packed with the same packing -> on packed data;
        packed vs plain operand not touching the packed dims (e.g. bias, scale) -> broadcast on packed data.
        Otherwise: unpack fallback.
        """
        a_packed = isinstance(a, Tensor) and isinstance(a.raw_tensor, PackedRawTensor)
        b_packed = isinstance(b, Tensor) and isinstance(b.raw_tensor, PackedRawTensor)
        if a_packed and b_packed:
            a_raw, b_raw = a.raw_tensor, b.raw_tensor
            if a_raw.packed_dim == b_raw.packed_dim and a_raw.orig_dims == b_raw.orig_dims:
                return a_raw.rewrap(
                    rf.combine(a_raw.inner, kind, b_raw.inner, allow_broadcast_all_sources=True), name=kind
                )
        elif a_packed or b_packed:
            packed_t, other = (a, b) if a_packed else (b, a)
            packed_raw = packed_t.raw_tensor
            if not isinstance(other, Tensor) or not (set(other.dims) & set(packed_raw.orig_dims)):
                args = (packed_raw.inner, kind, other) if a_packed else (other, kind, packed_raw.inner)
                return packed_raw.rewrap(rf.combine(*args, allow_broadcast_all_sources=True), name=kind)
        return rf.combine(
            _unpack_if_packed(a),
            kind,
            _unpack_if_packed(b),
            allow_broadcast_all_sources=allow_broadcast_all_sources,
            dim_order=dim_order,
        )

    @staticmethod
    def matmul(a: Tensor, b: Tensor, *, reduce: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> Tensor:
        """
        matmul where a is packed (dispatch is on a).
        Profitable case: reduce and b do not touch the packed dims
        (e.g. the Linear layer / vocab projection on [packed, F]) --
        then it runs directly on the packed data, saving all padding FLOPs.
        """
        a_raw = _raw(a)
        reduce_dims = [reduce] if isinstance(reduce, Dim) else list(reduce)
        b_raw = b.raw_tensor
        if (
            not isinstance(b_raw, PackedRawTensor)
            and not any(d in a_raw.orig_dims for d in reduce_dims)
            and not any(d in b.dims for d in a_raw.orig_dims)
        ):
            inner_out = rf.matmul(a_raw.inner, b, reduce=reduce, use_mask=use_mask)
            return a_raw.rewrap(inner_out, name="matmul")
        return rf.matmul(_unpack_if_packed(a), _unpack_if_packed(b), reduce=reduce, use_mask=use_mask)

    @staticmethod
    def softmax(tensor: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
        """softmax over a non-packed axis (e.g. vocab) -> on packed data. otherwise: fallback."""
        raw = _raw(tensor)
        if axis not in raw.orig_dims:
            return raw.rewrap(rf.softmax(raw.inner, axis=axis, use_mask=use_mask), name="softmax")
        # TODO axis in orig_dims = softmax over the packed spatial dim (e.g. attention energies over time):
        #  implementable as segment softmax directly on packed data
        #  (segment ids from the seq lens, seg-max / seg-sum via scatter-reduce, then elementwise),
        #  which needs no masking at all.
        #  For full attention, the flash-varlen specialization avoids materializing the energies altogether.
        return rf.softmax(unpack(tensor), axis=axis, use_mask=use_mask)

    @staticmethod
    def log_softmax(tensor: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
        """log_softmax over a non-packed axis (e.g. vocab) -> on packed data. otherwise: fallback."""
        raw = _raw(tensor)
        if axis not in raw.orig_dims:
            return raw.rewrap(rf.log_softmax(raw.inner, axis=axis, use_mask=use_mask), name="log_softmax")
        return rf.log_softmax(unpack(tensor), axis=axis, use_mask=use_mask)

    @staticmethod
    def softmax_cross_entropy_with_logits(*, logits: Tensor, targets: Tensor, axis: Dim):
        """
        CE over a non-packed axis (vocab), with targets packed the same way -> on packed data.
        This is the packed output block: logits [packed, vocab], targets [packed] (sparse).
        """
        logits_raw = _raw(logits)
        targets_raw = targets.raw_tensor
        if (
            axis not in logits_raw.orig_dims
            and isinstance(targets_raw, PackedRawTensor)
            and targets_raw.packed_dim == logits_raw.packed_dim
        ):
            inner_out = rf.cross_entropy(
                estimated=logits_raw.inner, target=targets_raw.inner, axis=axis, estimated_type="logits"
            )
            return logits_raw.rewrap(inner_out, name="cross_entropy")
        return rf.cross_entropy(
            estimated=unpack(logits), target=_unpack_if_packed(targets), axis=axis, estimated_type="logits"
        )

    @staticmethod
    def reduce(source: Tensor, *, mode: str, axis: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> Tensor:
        """
        reduce. fast paths:
        over non-packed dims only (e.g. layer-norm statistics over feature) -> on packed data;
        over all packed dims jointly (e.g. mean loss over (batch, time)) -> reduce the packed dim directly,
        which needs no masking at all (there are no padded frames in packed storage).
        Otherwise (partial, e.g. over time only): unpack fallback.
        """
        raw = _raw(source)
        axes = [axis] if isinstance(axis, Dim) else list(axis)
        if not set(axes) & set(raw.orig_dims):
            return raw.rewrap(rf.reduce(raw.inner, mode=mode, axis=axes, use_mask=use_mask), name=mode)
        if set(raw.orig_dims) <= set(axes) and not mode.startswith("arg"):
            inner_axes = [raw.packed_dim] + [d for d in axes if d not in raw.orig_dims]
            # use_mask=False: packed storage has no padded frames.
            return rf.reduce(raw.inner, mode=mode, axis=inner_axes, use_mask=False)
        return rf.reduce(unpack(source), mode=mode, axis=axes, use_mask=use_mask)


def _make_unpack_fallback(name: str):
    def _fallback(*args, **kwargs):
        packed_args = [x for x in list(args) + list(kwargs.values()) if isinstance(x, Tensor) and is_packed(x)]
        assert packed_args, f"PackedBackend.{name} fallback: no packed tensor in args"
        inner_backend = packed_args[0].raw_tensor.inner_backend
        args = [_unpack_if_packed(x) for x in args]
        kwargs = {k: _unpack_if_packed(v) for k, v in kwargs.items()}
        return getattr(inner_backend, name)(*args, **kwargs)

    _fallback.__name__ = name
    _fallback.__qualname__ = f"PackedBackend.{name}"
    _fallback.__doc__ = f"{name}: unpack fallback (needs the sequence structure; result decays to padded storage)"
    return staticmethod(_fallback)


# Ops needing the sequence structure, without a packed implementation (yet):
# unpack, then run on padded storage via the inner backend.
for _name in [
    "concat",
    "conv",
    "cumsum",
    "expand_dim",
    "flip_no_mask",
    "gather",
    "masked_scatter",
    "masked_select",
    "merge_dims",
    "pad",
    "pool",
    "reshape",
    "scatter",
    "search_sorted",
    "slice",
    "sort",
    "split",
    "split_dims",
    "squeeze",
    "stack",
    "stft",
    "top_k",
    "transposed_conv",
]:
    setattr(PackedBackend, _name, _make_unpack_fallback(_name))


def is_packed(source: Tensor) -> bool:
    """:return: whether source has packed storage"""
    return isinstance(source.raw_tensor, PackedRawTensor)


def _auto_pack_dims(source: Tensor) -> Tuple[Dim, ...]:
    """
    :return: all dyn dims of source, plus the dims their sizes depend on (typically batch).
        order: size dims (batch) first, then the dyn dims in source order.
    """
    dims = []
    for d in source.dims:
        if d.dyn_size_ext is not None:
            for size_dim in d.dyn_size_ext.dims:
                if size_dim not in dims:
                    dims.append(size_dim)
    for d in source.dims:
        if d.dyn_size_ext is not None and d not in dims:
            dims.append(d)
    return tuple(dims)


def pack(source: Tensor, *, dims: Optional[Sequence[Dim]] = None, out_dim: Optional[Dim] = None) -> Tensor:
    """
    Pack the given dims of source into packed storage.

    Unlike :func:`rf.pack_padded`, the returned Tensor keeps the *original* (virtual) dims,
    so downstream model code is unchanged;
    only the storage (raw tensor) is packed.

    :param source: e.g. [batch, time, feature]
    :param dims: dims to pack, e.g. (batch, time). the order defines the packed layout.
        If not given: all dyn dims of source plus the dims their sizes depend on (typically batch).
    :param out_dim: the packed dim, created if not given.
        Pass the packed dim of another packed tensor to share the packing (e.g. logits and targets).
    :return: tensor with same dims as source, packed storage
    """
    if dims is None:
        dims = _auto_pack_dims(source)
        assert dims, f"pack: no dims with dynamic length found in {source}"
    inner, packed_dim = rf.pack_padded(source, dims=dims, out_dim=out_dim)
    out = source.copy_template(name=(source.name or "packed") + "_packed")
    out.raw_tensor = PackedRawTensor(inner=inner, packed_dim=packed_dim, orig_dims=dims)
    return out


def unpack(source: Tensor) -> Tensor:
    """
    :param source: tensor with packed storage. if not packed, returned as-is.
    :return: tensor with normal padded storage of the inner backend, same dims
    """
    raw = source.raw_tensor
    if not isinstance(raw, PackedRawTensor):
        return source
    return rf.pad_packed(raw.inner, dims=raw.orig_dims, in_dim=raw.packed_dim)


register_backend_by_tensor_type(PackedRawTensor, PackedBackend)
