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
  This is decided generically per call:
  if no Dim referenced by the call (explicit Dim args, dims of other Tensor args)
  is one of the packed dims, the op cannot see the packed structure,
  so it runs on the packed data as-is.
  Also reductions over *all* packed dims jointly (e.g. mean over (batch, time))
  run directly on the packed data, without any masking.
- Per-backend specializations (TODO, not yet here):
  ops with native varlen kernels,
  e.g. attention via FlashAttention varlen (cu_seqlens) / FlexAttention doc-masking on torch.
- Fallback for ops needing the sequence structure: unpack, run the op on padded storage, repack.
  The result is repacked in the same format/order
  (sharing the packed dim if the packed dims are unchanged;
  following a replaced dim, e.g. the subsampled time dim of a strided conv, if the op created one),
  so downstream computation stays packed.
  Only if the new packing cannot be inferred does the result stay padded.
  The first fallback per op prints a warning (see :func:`_warn_fallback_once`),
  so slow paths are visible during development.

Known limitations (TODO):

- Mixed binary ops only work packed-first (``packed + plain``);
  ``plain + packed`` dispatches on the plain backend which cannot handle the wrapper.
- ``dim_order`` in :func:`combine` is ignored on the packed fast path.

Status: early skeleton. Import this module explicitly to activate the dispatch registration.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, Set, Tuple, Union

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

    def same_packing(self, other: PackedRawTensor) -> bool:
        """:return: whether other uses the identical packing (shared packed dim)"""
        return self.packed_dim == other.packed_dim and self.orig_dims == other.orig_dims

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
        :return: outer (virtual-dims) Tensor wrapping inner_out with the same packing relation
        """
        assert self.packed_dim in inner_out.dims
        opts = {}
        if inner_out.feature_dim is not None and inner_out.feature_dim != self.packed_dim:
            opts["feature_dim"] = inner_out.feature_dim
        # else: no explicit feature_dim, so the outer Tensor applies the usual default heuristic --
        # the inner tensor often loses the implicit feature dim (its heuristic fails without a batch dim),
        # while the virtual (unpacked) dims match what the padded op would produce.
        out = Tensor(
            name=name or inner_out.name,
            dims=self.virtual_dims(inner_out),
            dtype=inner_out.dtype,
            sparse_dim=inner_out.sparse_dim,
            **opts,
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


_warned_fallback_ops: Set[str] = set()


def _warn_fallback_once(op_name: str, reason: str):
    """print a warning on the first unpack fallback per op, so slow paths are visible"""
    if op_name in _warned_fallback_ops:
        return
    _warned_fallback_ops.add(op_name)
    print(
        f"PackedBackend warning: op {op_name!r} has no packed implementation ({reason}), "
        f"using slow unpack -> op -> repack fallback. (Only warned once per op.)"
    )


def _add_dim_with_size_deps(dims: Set[Dim], d: Dim):
    """add d and the dims its seq lens depend on (conservative: those matter for masking)"""
    dims.add(d)
    if d.dyn_size_ext is not None:
        dims.update(d.dyn_size_ext.dims)


def _collect_referenced_dims(*values) -> Set[Dim]:
    """
    :return: all Dims referenced by the given call args:
        explicit Dim args, and the dims of all non-packed Tensor args
        (packed Tensor args are excluded -- their packed dims are only storage, not a reference).
        For each dim, the dims its seq lens depend on are included as well
        (conservative: an op masking over such a dim needs the packed structure,
        e.g. softmax over the attention hist dim whose lens depend on batch).
    """
    dims = set()
    for v in values:
        if isinstance(v, Dim):
            _add_dim_with_size_deps(dims, v)
        elif isinstance(v, Tensor):
            if not isinstance(v.raw_tensor, PackedRawTensor):
                for d in v.dims:
                    _add_dim_with_size_deps(dims, d)
        elif isinstance(v, (list, tuple)):
            dims.update(_collect_referenced_dims(*v))
        elif isinstance(v, dict):
            dims.update(_collect_referenced_dims(*v.values()))
    return dims


def _dim_refs_packed(d: Dim, template: PackedRawTensor) -> bool:
    """:return: whether d is a packed dim, or its seq lens depend on one (so masking over d needs the structure)"""
    if d in template.orig_dims or d == template.packed_dim:
        return True
    if d.dyn_size_ext is not None and set(d.dyn_size_ext.dims) & set(template.orig_dims):
        return True
    return False


def _rewrap_result(out, template: PackedRawTensor):
    """rewrap Tensor results which kept the packed dim; pass through everything else (Dims, tuples, ...)"""
    if isinstance(out, Tensor):
        return template.rewrap(out) if template.packed_dim in out.dims else out
    if isinstance(out, tuple):
        return tuple(_rewrap_result(x, template) for x in out)
    return out


def _map_packed_to_inner(x):
    if isinstance(x, Tensor) and isinstance(x.raw_tensor, PackedRawTensor):
        return x.raw_tensor.inner
    if isinstance(x, (list, tuple)):
        return type(x)(_map_packed_to_inner(v) for v in x)
    return x


def _repack_result(out, template: PackedRawTensor, *, input_dims: frozenset = frozenset()):
    """
    Repack an unpack-fallback result in the same format/order as the template packing.

    If all packed dims are still present, the result shares the template's packed dim.
    If some packed dim was replaced by a new dyn dim created by the op
    (e.g. the subsampled time dim of a strided conv),
    the result is packed over the replacement (in the same order), with a new packed dim.
    If the new packing cannot be inferred (e.g. the packed dims were reduced away),
    the result stays padded.

    :param out: result of the padded op (Tensor, or tuple containing Tensors)
    :param template: the packing of the packed inputs
    :param input_dims: all dims referenced by the call inputs,
        to distinguish op-created dyn dims (replacement candidates) from pre-existing ones
    """
    if isinstance(out, tuple):
        return tuple(_repack_result(x, template, input_dims=input_dims) for x in out)
    if not isinstance(out, Tensor):
        return out
    missing = [d for d in template.orig_dims if d not in out.dims]
    if not missing:
        return pack(out, dims=template.orig_dims, out_dim=template.packed_dim)
    new_dyn_dims = [d for d in out.dims if d not in input_dims and d.dyn_size_ext is not None]
    if len(new_dyn_dims) == len(missing):
        new_it = iter(new_dyn_dims)
        dims = tuple(d if d not in missing else next(new_it) for d in template.orig_dims)
        return pack(out, dims=dims)
    return out  # cannot infer the new packing; stays padded


def _set_feature_dim_like_binop(out: Tensor, a, b):
    """
    Propagate feature_dim from the outer binop operands, like the padded path does
    (see _utils.res_feature_dim) --
    the operands' feature marking often lives only on the outer tensor
    (e.g. rf.Linear sets it on the matmul result before adding the bias),
    so rebuilding the outer from the inner would lose it.
    """
    if out.feature_dim is not None:
        return
    a_fd = a.feature_dim if isinstance(a, Tensor) else None
    b_fd = b.feature_dim if isinstance(b, Tensor) else None
    fd = None
    if a_fd and not b_fd:
        fd = a_fd
    elif b_fd and not a_fd:
        fd = b_fd
    elif a_fd and b_fd and a_fd == b_fd:
        fd = a_fd
    if fd is not None and fd in out.dims:
        out.feature_dim = fd


def _pack_like(x: Tensor, template: PackedRawTensor) -> Optional[Tensor]:
    """
    :param x: plain (non-packed) tensor whose dims include all packed dims (e.g. a seq mask over (batch, time))
    :param template: packing to follow
    :return: inner (packed-storage) tensor for x with the template's packing, or None if not possible directly
    """
    if not set(template.orig_dims) <= set(x.dims):
        return None
    inner, _ = rf.pack_padded(x, dims=template.orig_dims, out_dim=template.packed_dim)
    return inner


def _make_dim_aware_op(name: str):
    """
    Generic wrapper for backend ops without a dedicated packed implementation.

    If the call does not reference any of the packed dims
    (checked over explicit Dim args and the dims of all other Tensor args),
    the op cannot see the packed structure,
    so it runs directly on the packed data (packed args replaced by their inner tensors)
    and the results are rewrapped.
    Otherwise: unpack fallback (with a one-time warning).
    """

    def _op(*args, **kwargs):
        all_values = list(args) + list(kwargs.values())
        packed_args = [x for x in _flatten(all_values) if isinstance(x, Tensor) and is_packed(x)]
        assert packed_args, f"PackedBackend.{name}: no packed tensor in args"
        raw0 = packed_args[0].raw_tensor
        referenced = _collect_referenced_dims(*all_values)
        if all(raw0.same_packing(x.raw_tensor) for x in packed_args[1:]):
            overlap = referenced & (set(raw0.orig_dims) | {raw0.packed_dim})
            if not overlap:
                inner_args = [_map_packed_to_inner(x) for x in args]
                inner_kwargs = {k: _map_packed_to_inner(v) for k, v in kwargs.items()}
                out = getattr(raw0.inner_backend, name)(*inner_args, **inner_kwargs)
                return _rewrap_result(out, raw0)
            _warn_fallback_once(name, f"references packed dims {sorted(overlap, key=lambda d: d.name or '')}")
        else:
            _warn_fallback_once(name, "mixed packings")
        input_dims = frozenset(referenced | set().union(*(set(x.raw_tensor.virtual_dims()) for x in packed_args)))
        args = [_unpack_if_packed(x) for x in args]
        kwargs = {k: _unpack_if_packed(v) for k, v in kwargs.items()}
        out = getattr(raw0.inner_backend, name)(*args, **kwargs)
        return _repack_result(out, raw0, input_dims=input_dims)

    _op.__name__ = name
    _op.__qualname__ = f"PackedBackend.{name}"
    _op.__doc__ = (
        f"{name}: generic packed handling -- "
        f"runs on packed data if the call does not reference the packed dims, otherwise unpack fallback"
    )
    return staticmethod(_op)


def _flatten(values):
    for v in values:
        if isinstance(v, (list, tuple)):
            yield from _flatten(v)
        else:
            yield v


# We do not expect to ever implement all methods of the Backend interface here --
# only what is profitable on packed data; the rest goes through the generic dim-aware wrapper below.
# noinspection PyAbstractClass
class PackedBackend(Backend[PackedRawTensor]):
    """
    Backend for packed tensors, wrapping any inner backend.

    Selected via the normal raw-tensor-type dispatch;
    the actual computation is delegated to the inner backend of the wrapped tensor.
    """

    name = "packed"
    RawTensorType = PackedRawTensor
    # Win mixed-backend dispatch (get_backend_from_tensors):
    # we can handle plain tensors of the inner backend, but not vice versa.
    dispatch_priority = 1

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
            if a_raw.same_packing(b_raw):
                out = a_raw.rewrap(
                    rf.combine(a_raw.inner, kind, b_raw.inner, allow_broadcast_all_sources=True), name=kind
                )
                _set_feature_dim_like_binop(out, a, b)
                return out
            _warn_fallback_once("combine", "mixed packings")
        elif a_packed or b_packed:
            packed_t, other = (a, b) if a_packed else (b, a)
            packed_raw = packed_t.raw_tensor
            if isinstance(other, Tensor) and set(other.dims) & set(packed_raw.orig_dims):
                # operand over the packed dims (e.g. an additive mask over (batch, time)):
                # pack it alike, then combine on packed data.
                other = _pack_like(other, packed_raw)
            if other is not None and not (isinstance(other, Tensor) and set(other.dims) & set(packed_raw.orig_dims)):
                args = (packed_raw.inner, kind, other) if a_packed else (other, kind, packed_raw.inner)
                out = packed_raw.rewrap(rf.combine(*args, allow_broadcast_all_sources=True), name=kind)
                _set_feature_dim_like_binop(out, a, b)
                return out
            _warn_fallback_once("combine", "operand references packed dims and is not packable alike")
        template = a.raw_tensor if a_packed else b.raw_tensor
        out = rf.combine(
            _unpack_if_packed(a),
            kind,
            _unpack_if_packed(b),
            allow_broadcast_all_sources=allow_broadcast_all_sources,
            dim_order=dim_order,
        )
        return _repack_result(out, template)

    @staticmethod
    def where(
        cond: Union[Tensor, Any],
        true_: Union[Tensor, Any],
        false_: Union[Tensor, Any],
        *,
        allow_broadcast_all_sources: bool = False,
    ) -> Tensor:
        """
        where -- on packed data if all Tensor operands can share the packing:
        packed alike, plain operands not touching the packed dims,
        or plain operands packable to the same packing
        (e.g. a seq mask over (batch, time), as used for masking before conv).
        """
        operands = [cond, true_, false_]
        packed_ops = [x for x in operands if isinstance(x, Tensor) and is_packed(x)]
        assert packed_ops, "PackedBackend.where: no packed operand"
        raw0 = packed_ops[0].raw_tensor
        if all(raw0.same_packing(x.raw_tensor) for x in packed_ops[1:]):
            inner_ops = []
            for x in operands:
                if isinstance(x, Tensor) and is_packed(x):
                    inner_ops.append(x.raw_tensor.inner)
                elif isinstance(x, Tensor) and set(x.dims) & set(raw0.orig_dims):
                    x_ = _pack_like(x, raw0)
                    if x_ is None:
                        break
                    inner_ops.append(x_)
                else:
                    inner_ops.append(x)
            else:
                out = raw0.rewrap(rf.where(*inner_ops, allow_broadcast_all_sources=True), name="where")
                _set_feature_dim_like_binop(out, true_, false_)
                return out
        _warn_fallback_once("where", "operands not packable to a common packing")
        out = rf.where(
            _unpack_if_packed(cond),
            _unpack_if_packed(true_),
            _unpack_if_packed(false_),
            allow_broadcast_all_sources=allow_broadcast_all_sources,
        )
        return _repack_result(out, raw0)

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
            and not any(_dim_refs_packed(d, a_raw) for d in reduce_dims)
            and not any(d in b.dims for d in a_raw.orig_dims)
        ):
            inner_out = rf.matmul(a_raw.inner, b, reduce=reduce, use_mask=use_mask)
            return a_raw.rewrap(inner_out, name="matmul")
        _warn_fallback_once("matmul", "reduce dims or other operand reference the packed dims")
        out = rf.matmul(_unpack_if_packed(a), _unpack_if_packed(b), reduce=reduce, use_mask=use_mask)
        return _repack_result(out, a_raw)

    @staticmethod
    def softmax(tensor: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
        """softmax over a non-packed axis (e.g. vocab) -> on packed data. otherwise: fallback."""
        raw = _raw(tensor)
        if not _dim_refs_packed(axis, raw):
            return raw.rewrap(rf.softmax(raw.inner, axis=axis, use_mask=use_mask), name="softmax")
        # TODO axis in orig_dims = softmax over the packed spatial dim (e.g. attention energies over time):
        #  implementable as segment softmax directly on packed data
        #  (segment ids from the seq lens, seg-max / seg-sum via scatter-reduce, then elementwise),
        #  which needs no masking at all.
        #  For full attention, the flash-varlen specialization avoids materializing the energies altogether.
        _warn_fallback_once("softmax", f"axis {axis} is a packed dim (segment softmax TODO)")
        return _repack_result(rf.softmax(unpack(tensor), axis=axis, use_mask=use_mask), raw)

    @staticmethod
    def log_softmax(tensor: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
        """log_softmax over a non-packed axis (e.g. vocab) -> on packed data. otherwise: fallback."""
        raw = _raw(tensor)
        if not _dim_refs_packed(axis, raw):
            return raw.rewrap(rf.log_softmax(raw.inner, axis=axis, use_mask=use_mask), name="log_softmax")
        _warn_fallback_once("log_softmax", f"axis {axis} is a packed dim (segment softmax TODO)")
        return _repack_result(rf.log_softmax(unpack(tensor), axis=axis, use_mask=use_mask), raw)

    @staticmethod
    def softmax_cross_entropy_with_logits(*, logits: Tensor, targets: Tensor, axis: Dim):
        """
        CE over a non-packed axis (vocab), with targets packed the same way -> on packed data.
        This is the packed output block: logits [packed, vocab], targets [packed] (sparse).
        """
        logits_raw = _raw(logits)
        targets_raw = targets.raw_tensor
        if (
            not _dim_refs_packed(axis, logits_raw)
            and isinstance(targets_raw, PackedRawTensor)
            and logits_raw.same_packing(targets_raw)
        ):
            inner_out = rf.cross_entropy(
                estimated=logits_raw.inner, target=targets_raw.inner, axis=axis, estimated_type="logits"
            )
            return logits_raw.rewrap(inner_out, name="cross_entropy")
        _warn_fallback_once("softmax_cross_entropy_with_logits", "axis packed or targets not packed alike")
        out = rf.cross_entropy(
            estimated=unpack(logits), target=_unpack_if_packed(targets), axis=axis, estimated_type="logits"
        )
        return _repack_result(out, logits_raw)

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
        if not any(_dim_refs_packed(d, raw) for d in axes):
            return raw.rewrap(rf.reduce(raw.inner, mode=mode, axis=axes, use_mask=use_mask), name=mode)
        extra_axes = [d for d in axes if d not in raw.orig_dims]
        if (
            set(raw.orig_dims) <= set(axes)
            and not mode.startswith("arg")
            and not any(_dim_refs_packed(d, raw) for d in extra_axes)
        ):
            # use_mask=False: packed storage has no padded frames.
            return rf.reduce(raw.inner, mode=mode, axis=[raw.packed_dim] + extra_axes, use_mask=False)
        _warn_fallback_once("reduce", f"partial reduce over packed dims (axis {axes}, mode {mode})")
        return _repack_result(rf.reduce(unpack(source), mode=mode, axis=axes, use_mask=use_mask), raw)


# All other structural ops go through the generic dim-aware wrapper:
# packed data directly if the call does not reference the packed dims, otherwise unpack fallback.
for _name in [
    "batch_norm",
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
    setattr(PackedBackend, _name, _make_dim_aware_op(_name))


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
    if source.feature_dim is not None and inner.feature_dim is None and source.feature_dim in inner.dims:
        inner.feature_dim = source.feature_dim  # masked_select drops it
    # Note: the returned dims order is the canonical virtual order (packed dims first),
    # which can differ from source.dims order (dims are unordered semantically in RF).
    helper = PackedRawTensor(inner=inner, packed_dim=packed_dim, orig_dims=dims)
    return helper.rewrap(inner, name=(source.name or "packed") + "_packed")


def unpack(source: Tensor) -> Tensor:
    """
    :param source: tensor with packed storage. if not packed, returned as-is.
    :return: tensor with normal padded storage of the inner backend, same dims
    """
    raw = source.raw_tensor
    if not isinstance(raw, PackedRawTensor):
        return source
    out = rf.pad_packed(raw.inner, dims=raw.orig_dims, in_dim=raw.packed_dim)
    if source.feature_dim is not None and out.feature_dim is None and source.feature_dim in out.dims:
        out.feature_dim = source.feature_dim  # masked_scatter drops it
    return out


register_backend_by_tensor_type(PackedRawTensor, PackedBackend)
