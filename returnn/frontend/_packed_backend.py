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

__all__ = ["PackedRawTensor", "PackedBackend", "pack", "unpack", "regap", "is_packed"]


class PackedRawTensor:
    """
    Raw-tensor wrapper marking packed storage.

    The wrapped :attr:`inner` is a normal RF :class:`Tensor` (of the inner backend)
    holding the packed data, dims = [packed_dim] + remaining (non-packed) dims.
    :attr:`orig_dims` are the dims packed into :attr:`packed_dim`, e.g. (batch, time);
    the seq lens live in the dyn dims' dyn_size_ext as usual, nothing is duplicated here.
    """

    def __init__(
        self,
        *,
        inner: Tensor,
        packed_dim: Dim,
        orig_dims: Sequence[Dim],
        dims: Optional[Sequence[Dim]] = None,
        gap: int = 0,
        align: int = 1,
    ):
        self.inner = inner
        self.packed_dim = packed_dim
        self.orig_dims = tuple(orig_dims)
        if dims is None:
            # canonical virtual order: packed dims first, then the remaining dims in inner order
            dims = self.orig_dims + tuple(d for d in inner.dims if d != packed_dim)
        # outer (virtual) dims order; can deviate from canonical after transpose (pure metadata)
        self.dims = tuple(dims)
        # align = per-seq footprint alignment: every seq occupies roundup(len + gap, align) frames,
        # so all seq starts are multiples of align -- required by strided ops (stride must divide align).
        # Convention: align should divide gap, then the layout params stay closed under stride division.
        # 1 = no alignment (the default layout).
        self.align = align
        # gap = extra frames between sequences in the packed buffer, uniform for all seqs.
        # Purpose (planned, packed conv/pool/pad): safety margin so a dense conv kernel
        # cannot mix sequences, and in-place space for pad -- e.g. gap >= dilation*(kernel-1).
        # Specified manually at pack() time for now (derive by hand for the model at hand).
        # Ops that need more gap than available must warn and fall back.
        # 0 = dense layout (the only layout produced so far).
        self.gap = gap

    @property
    def seq_lens(self) -> Tensor:
        """
        :return: frame count per sequence,
            over the packed dims except the innermost (e.g. [batch] for packing (batch, time))
        """
        last = self.orig_dims[-1]
        assert last.dyn_size_ext is not None, f"packed dim {last} needs dyn sizes"
        return last.dyn_size_ext

    def seq_starts(self) -> Tuple[Tensor, Dim]:
        """
        :return: (starts, seqs_dim):
            start offset of every sequence in the packed buffer,
            flattened row-major over the packed dims except the innermost (matching the packed layout),
            and the flat seqs dim.
            Dense layout: exclusive cumsum of the seq lens; the gap layout adds gap frames per preceding seq.
        """
        starts, seqs_dim = _seq_starts_math(self.orig_dims, self.gap, self.align)
        assert starts is not None, f"seq_starts: no outer packed dims in {self.orig_dims}"
        return starts, seqs_dim

    def cu_seqlens(self, *, device: Optional[str] = None) -> Tuple[Tensor, Dim]:
        """
        :return: (cu, cu_dim): cumulative sequence boundaries
            as used by varlen attention kernels (e.g. FlashAttention flash_attn_varlen_func):
            [seqs+1] int32, cu[i] = start of seq i, cu[-1] = total number of frames.
            Dense layout only (gap == 0), since the boundaries imply that seq i ends where seq i+1 starts.
        """
        assert self.gap == 0 and self.align == 1, "cu_seqlens requires the dense layout"
        starts, seqs_dim = self.seq_starts()
        total = rf.cast(self.packed_dim.get_dim_value_tensor(), starts.dtype)
        end_dim = Dim(1, name="cu_seqlens_end")
        cu, cu_dim = rf.concat((starts, seqs_dim), (rf.expand_dim(total, dim=end_dim), end_dim))
        cu = rf.cast(cu, "int32")
        if device:
            cu = rf.copy_to_device(cu, device)
        return cu, cu_dim

    @property
    def inner_backend(self):
        """:return: backend of the inner (packed data) tensor"""
        # noinspection PyProtectedMember
        return self.inner._raw_backend

    def same_packing(self, other: PackedRawTensor) -> bool:
        """:return: whether other uses the identical packing (shared packed dim)"""
        return (
            self.packed_dim == other.packed_dim
            and self.orig_dims == other.orig_dims
            and self.gap == other.gap
            and self.align == other.align
        )

    def virtual_ndim(self) -> int:
        """:return: ndim of the virtual (unpacked) view, i.e. of the outer Tensor"""
        return len(self.dims)

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
        out.raw_tensor = PackedRawTensor(
            inner=inner_out, packed_dim=self.packed_dim, orig_dims=self.orig_dims, gap=self.gap, align=self.align
        )
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


def _warn_fallback_once(op_name: str, reason: str, *, action: str = "using slow unpack -> op -> repack fallback"):
    """print a warning on the first fallback / slow path per op, so these are visible"""
    if op_name in _warned_fallback_ops:
        return
    _warned_fallback_ops.add(op_name)
    print(f"PackedBackend warning: op {op_name!r}: {reason}, {action}. (Only warned once per op.)")


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


def _dim_derivation_bases(d: Dim) -> Set[Dim]:
    """:return: all dims that d is derived from via dim math (e.g. ceildiv, add), recursively, incl. d"""
    bases = set()
    stack = [d]
    while stack:
        x = stack.pop()
        if x in bases:
            continue
        bases.add(x)
        op = getattr(x, "derived_from_op", None)
        if op is not None:
            stack.extend(op.inputs)
        same_as = getattr(x, "same_as", None)
        if same_as is not None:
            stack.append(same_as)
    return bases


def _repack_result(out, template: PackedRawTensor, *, replacements: Optional[dict] = None):
    """
    Repack an unpack-fallback result in the same format/order as the template packing.

    If all packed dims are still present, the result shares the template's packed dim.
    If some packed dim was replaced (e.g. the subsampled time dim of a strided conv),
    the result is packed over the replacement (in the same order), with a new packed dim.
    Replacements come from the op itself where available
    (e.g. conv/pool in_spatial_dims -> returned out_spatial_dims, passed in as `replacements`),
    otherwise from the exact dim-math derivation (a result dim derived from the missing dim,
    e.g. time+1 from pad). Only an unambiguous derivation is trusted --
    a merely related dim (common ancestor) is NOT enough,
    since e.g. attention kv or rel-pos window dims also descend from time but mean something else.
    If the new packing cannot be inferred, the result stays padded.

    :param out: result of the padded op (Tensor, or tuple containing Tensors)
    :param template: the packing of the packed inputs
    :param replacements: explicit dim replacements from the op (in dim -> out dim)
    """
    if isinstance(out, tuple):
        return tuple(_repack_result(x, template, replacements=replacements) for x in out)
    if not isinstance(out, Tensor):
        return out
    missing = [d for d in template.orig_dims if d not in out.dims]
    if not missing:
        return pack(out, dims=template.orig_dims, out_dim=template.packed_dim, gap=template.gap, align=template.align)
    resolved = {}
    for m in missing:
        if replacements and m in replacements and replacements[m] in out.dims:
            resolved[m] = replacements[m]
            continue
        candidates = [
            d
            for d in out.dims
            if d not in template.orig_dims and d.dyn_size_ext is not None and m in _dim_derivation_bases(d)
        ]
        if len(candidates) != 1:
            # cannot infer the new packing (none or ambiguous); stays padded
            _warn_fallback_once(
                f"repack:{m.name}", f"result {out} has {len(candidates)} replacement candidates for packed dim {m}"
            )
            return out
        resolved[m] = candidates[0]
    dims = tuple(resolved.get(d, d) for d in template.orig_dims)
    return pack(out, dims=dims, gap=template.gap, align=template.align)


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


def _same_seq_lens(a: Dim, b: Dim) -> bool:
    """:return: whether a and b have identical seq lens (so packing over one is packing over the other)"""
    if a.dyn_size_ext is None and b.dyn_size_ext is None:
        return a.dimension == b.dimension
    if a.dyn_size_ext is None or b.dyn_size_ext is None:
        return False
    if a.dyn_size_ext is b.dyn_size_ext:
        return True
    if a.dyn_size_ext.dims != b.dyn_size_ext.dims:
        return False
    a_sizes = a.get_dyn_size_ext_for_device("cpu")
    b_sizes = b.get_dyn_size_ext_for_device("cpu")
    return bool(rf.reduce_all(a_sizes == b_sizes, axis=list(a_sizes.dims)).raw_tensor)


def _seq_footprints(orig_dims: Sequence[Dim], gap: int, align: int) -> Tuple[Optional[Tensor], Optional[Dim]]:
    """
    :return: (footprints, seqs_dim): frames occupied per sequence in the packed buffer,
        roundup(len + gap, align), flat row-major over the outer packed dims.
        (None, None) if there are no outer packed dims.
    """
    others = tuple(orig_dims[:-1])
    if not others:
        return None, None
    last = orig_dims[-1]
    assert last.dyn_size_ext is not None, f"packed dim {last} needs dyn sizes"
    lens = last.dyn_size_ext
    for d in others:
        if d not in lens.dims:
            lens = rf.expand_dim(lens, dim=d)
    if len(others) > 1:
        lens, seqs_dim = rf.merge_dims(lens, dims=others)
    else:
        (seqs_dim,) = others
    footprints = lens + gap
    if align > 1:
        footprints = (footprints + (align - 1)) // align * align
    return footprints, seqs_dim


def _seq_starts_math(orig_dims: Sequence[Dim], gap: int, align: int) -> Tuple[Optional[Tensor], Optional[Dim]]:
    """
    :return: (starts, seqs_dim): see :func:`PackedRawTensor.seq_starts`:
        exclusive cumsum of the seq footprints.
        (None, None) if there are no outer packed dims (a single sequence; starts trivially at 0).
    """
    footprints, seqs_dim = _seq_footprints(orig_dims, gap, align)
    if footprints is None:
        return None, None
    cum = rf.cumsum(footprints, spatial_dim=seqs_dim)
    return cum - footprints, seqs_dim


def _padded_positions(orig_dims: Sequence[Dim], gap: int, align: int) -> Tensor:
    """
    :return: [orig_dims...] (int): for every frame of the padded grid,
        its position in the packed buffer (only meaningful on the non-padded frames).
    """
    pos = rf.range_over_dim(orig_dims[-1])
    starts, seqs_dim = _seq_starts_math(orig_dims, gap, align)
    if starts is not None:
        others = tuple(orig_dims[:-1])
        if len(others) > 1:
            starts = rf.split_dims(starts, axis=seqs_dim, dims=others)
        pos = starts + pos
    return pos


def _packed_total(orig_dims: Sequence[Dim], gap: int, align: int) -> Tensor:
    """:return: scalar (int32): total number of frames in the packed buffer (sum of footprints)"""
    footprints, _ = _seq_footprints(orig_dims, gap, align)
    if footprints is None:
        last = orig_dims[-1]
        assert last.dyn_size_ext is not None
        return rf.cast(last.dyn_size_ext, "int32")
    return rf.cast(rf.reduce_sum(footprints, axis=list(footprints.dims)), "int32")


def _frame_coords(template: PackedRawTensor, d: Dim) -> Tensor:
    """
    :param d: one of the template's packed dims
    :return: [packed_dim] (int): for every packed frame, its coordinate in d
        (gap frames get 0 -- only meaningful on sequence frames).
        Cheap: only an int grid over the packed dims is scattered, no feature-sized data.
    """
    grid = rf.range_over_dim(d)
    for o in template.orig_dims:
        if o not in grid.dims:
            grid = rf.expand_dim(grid, dim=o)  # broadcast view, ints only
    pos = _padded_positions(template.orig_dims, template.gap, template.align)
    return rf.scatter(
        grid, indices=pos, indices_dim=list(template.orig_dims), out_dim=template.packed_dim, use_mask=True
    )


def _frame_mask(template: PackedRawTensor) -> Optional[Tensor]:
    """
    :return: [packed_dim] (bool): True on sequence frames, False on gap frames.
        None for the dense layout (all frames are sequence frames).
    """
    if template.gap == 0 and template.align == 1:
        return None
    ones = rf.cast(rf.sequence_mask(list(template.orig_dims)), "int32")
    pos = _padded_positions(template.orig_dims, template.gap, template.align)
    counts = rf.scatter(
        ones, indices=pos, indices_dim=list(template.orig_dims), out_dim=template.packed_dim, use_mask=True
    )
    return counts > 0


def _pack_like(x: Tensor, template: PackedRawTensor) -> Optional[Tensor]:
    """
    :param x: plain (non-packed) tensor over (some of) the packed dims
        (e.g. a seq mask over (batch, time), or a positional encoding over (time,) only)
    :param template: packing to follow
    :return: inner (packed-storage) tensor for x with the template's packing, or None if not possible
    """
    in_dims = [d for d in template.orig_dims if d in x.dims]
    assert in_dims
    # Gather via per-frame coordinates instead of broadcast + pack:
    # avoids materializing the full broadcast tensor
    # (e.g. a pos enc [time, feat] would blow up to [batch, time, feat] first).
    if len(in_dims) == 1:
        return rf.gather(x, indices=_frame_coords(template, in_dims[0]), axis=in_dims[0])
    x_flat, flat_dim = rf.merge_dims(x, dims=in_dims)
    idx = None
    for d in in_dims:
        coords = _frame_coords(template, d)
        idx = coords if idx is None else idx * d.get_dim_value_tensor() + coords
    return rf.gather(x_flat, indices=idx, axis=flat_dim)


# Ops that reinterpret the raw buffer across their axes (flatten-reshape tricks, dim merging):
# after such an op involving a packed dim, that dim no longer indexes sequence frames,
# so the result must NOT be repacked over it
# (e.g. the rel-pos-encoding shift trick reshapes (T, 2T) -> (2T, T);
# repacking in between would zero-fill entries which still belong to valid frames).
# The result stays padded until an op with frame semantics combines it with a packed tensor again.
_NO_REPACK_OPS = {"merge_dims", "reshape", "split_dims"}

# Ops whose inner path implicitly reduces over the packed dim (statistics over "everything"):
# valid on the dense layout (packed = exactly the real frames), but not with gap frames.
_DENSE_ONLY_INNER_OPS = {"batch_norm"}


def _segment_softmax(tensor: Tensor, *, axis: Dim, log: bool) -> Optional[Tensor]:
    """
    (log_)softmax over a packed dim, directly on the packed data -- no masking needed
    (padded frames do not exist in packed storage).

    A packed frame belongs to the segment given by its coordinates in the *other* packed dims
    (e.g. softmax over time with packing (batch, time): segment = batch entry).
    Implemented with generic RF ops: :func:`rf.scatter` mode "logsumexp" + :func:`rf.gather` --
    inner-backend-agnostic, no contiguity requirement on the segments,
    so this stays correct for future layout variants (e.g. gap padding for packed conv).

    :param tensor: packed, axis in orig_dims
    :param axis: packed dim to normalize over
    :param log: log_softmax instead of softmax
    """
    raw = _raw(tensor)
    inner = raw.inner
    other_packed = [d for d in raw.orig_dims if d != axis]
    if not other_packed:
        if raw.gap or raw.align > 1:
            return None  # single segment incl. gap/pad frames; rare, fallback
        lse = rf.reduce_logsumexp(inner, axis=raw.packed_dim, use_mask=False)
        out_inner = inner - lse
    else:
        # gap frames (if any) go to the dump segment and only affect other gap frames
        seg, scatter_dim, _ = _segment_index(raw, other_packed)
        lse = rf.scatter(
            inner, indices=seg, indices_dim=raw.packed_dim, mode="logsumexp", out_dim=scatter_dim, use_mask=False
        )
        out_inner = inner - rf.gather(lse, indices=seg, axis=scatter_dim)
    if not log:
        out_inner = rf.exp(out_inner)
    return raw.rewrap(out_inner, name="log_softmax" if log else "softmax")


def _segment_index(template: PackedRawTensor, seg_dims: Sequence[Dim]) -> Tuple[Tensor, Dim, Dim]:
    """
    :param seg_dims: (some of) the template's packed dims defining the segments
    :return: (seg, scatter_dim, valid_dim): for every packed frame, its flat index over seg_dims
        (row-major, matching the merged-dims layout);
        the dim to scatter into; and the dim holding the valid segments.
        Dense layout: scatter_dim == valid_dim (the merged seg_dims).
        Gapped layout: gap frames are routed to an extra dump segment
        (scatter_dim = valid_dim + 1), so any segment reduction stays correct;
        slice the dump segment off (or simply never gather it).
    """
    seg = None
    merged = None
    for d in seg_dims:
        coords = _frame_coords(template, d)
        seg = coords if seg is None else seg * d.get_dim_value_tensor() + coords
        merged = d if merged is None else merged * d
    if template.gap or template.align > 1:
        mask = _frame_mask(template)
        dump = merged.get_dim_value_tensor()
        if isinstance(dump, Tensor):
            dump = rf.cast(dump, seg.dtype)
        seg = rf.where(mask, seg, dump)
        return seg, merged + 1, merged
    return seg, merged, merged


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
        return _dim_aware_call(name, args, kwargs)

    _op.__name__ = name
    _op.__qualname__ = f"PackedBackend.{name}"
    _op.__doc__ = (
        f"{name}: generic packed handling -- "
        f"runs on packed data if the call does not reference the packed dims, otherwise unpack fallback"
    )
    return staticmethod(_op)


def _dim_aware_call(name: str, args, kwargs):
    """see :func:`_make_dim_aware_op`"""
    all_values = list(args) + list(kwargs.values())
    packed_args = [x for x in _flatten(all_values) if isinstance(x, Tensor) and is_packed(x)]
    assert packed_args, f"PackedBackend.{name}: no packed tensor in args"
    raw0 = packed_args[0].raw_tensor
    referenced = _collect_referenced_dims(*all_values)
    if all(raw0.same_packing(x.raw_tensor) for x in packed_args[1:]):
        overlap = referenced & (set(raw0.orig_dims) | {raw0.packed_dim})
        if overlap:
            _warn_fallback_once(name, f"references packed dims {sorted(overlap, key=lambda d: d.name or '')}")
        elif name in _DENSE_ONLY_INNER_OPS and (raw0.gap or raw0.align > 1):
            # e.g. batch_norm: its statistics implicitly reduce over the packed dim,
            # and gap frames would pollute them.
            _warn_fallback_once(name, "implicitly reduces over the packed dim, gapped layout")
        else:
            inner_args = [_map_packed_to_inner(x) for x in args]
            inner_kwargs = {k: _map_packed_to_inner(v) for k, v in kwargs.items()}
            out = getattr(raw0.inner_backend, name)(*inner_args, **inner_kwargs)
            return _rewrap_result(out, raw0)
    else:
        _warn_fallback_once(name, "mixed packings")
    in_spatial = kwargs.get("in_spatial_dims", kwargs.get("in_spatial_dim"))
    args = [_unpack_if_packed(x) for x in args]
    kwargs = {k: _unpack_if_packed(v) for k, v in kwargs.items()}
    out = getattr(raw0.inner_backend, name)(*args, **kwargs)
    if name in _NO_REPACK_OPS:
        return out  # stays padded, see _NO_REPACK_OPS
    replacements = None
    if in_spatial is not None and isinstance(out, tuple):
        # ops like conv/pool return their out spatial dims: an explicit in -> out dim mapping
        in_list = [in_spatial] if isinstance(in_spatial, Dim) else list(in_spatial)
        for elem in out[1:]:
            out_list = [elem] if isinstance(elem, Dim) else elem if isinstance(elem, (list, tuple)) else None
            if out_list is not None and len(out_list) == len(in_list) and all(isinstance(d, Dim) for d in out_list):
                replacements = {i: o for i, o in zip(in_list, out_list) if i != o}
                break
    return _repack_result(out, raw0, replacements=replacements)


def _flatten(values):
    for v in values:
        if isinstance(v, (list, tuple)):
            yield from _flatten(v)
        else:
            yield v


def _strided_out_wrapper(
    raw: PackedRawTensor, out_inner: Tensor, out_packed_dim: Dim, out_time: Dim, st: int
) -> Optional[Tensor]:
    """
    Build the outer wrapper for the output of a stride-st op over the packed dim.

    The actual per-seq output spacing is footprint/st (starts are starts/st, all exact by st | align).
    The closed form (out lens, gap', align/st) can express it iff the residual footprint/st - out_len
    is uniform over the seqs -- verified here at runtime (one small host sync),
    which is robust against out-length semantic variants. None if not expressible.
    """
    footprints, seqs_dim = _seq_footprints(raw.orig_dims, raw.gap, raw.align)
    align_out = max(raw.align // st, 1)
    if footprints is None:
        helper = PackedRawTensor(
            inner=out_inner, packed_dim=out_packed_dim, orig_dims=(out_time,), gap=0, align=align_out
        )
        return helper.rewrap(out_inner, name="conv")
    lens_out = out_time.dyn_size_ext
    for d in raw.orig_dims[:-1]:
        if d not in lens_out.dims:
            lens_out = rf.expand_dim(lens_out, dim=d)
    if len(raw.orig_dims) > 2:
        lens_out, _ = rf.merge_dims(lens_out, dims=raw.orig_dims[:-1])
    residual = footprints // st - lens_out
    residual0 = int(residual.raw_tensor.flatten()[0])
    if residual0 < 0 or not bool(rf.reduce_all(residual == residual0, axis=list(residual.dims)).raw_tensor):
        return None
    helper = PackedRawTensor(
        inner=out_inner,
        packed_dim=out_packed_dim,
        orig_dims=tuple(raw.orig_dims[:-1]) + (out_time,),
        gap=residual0,
        align=align_out,
    )
    return helper.rewrap(out_inner, name="conv")


def _extract_strided(raw: PackedRawTensor, out_inner: Tensor, out_packed_dim: Dim, out_time: Dim, st: int) -> Tensor:
    """
    Extract the output of a stride-st op over the packed dim to padded storage
    (for the rare case that the closed-form layout cannot express it, see :func:`_strided_out_wrapper`).
    """
    out_orig = tuple(raw.orig_dims[:-1]) + (out_time,)
    pos = rf.range_over_dim(out_time)
    starts, seqs_dim = _seq_starts_math(raw.orig_dims, raw.gap, raw.align)
    if starts is not None:
        starts = starts // st
        others = tuple(raw.orig_dims[:-1])
        if len(others) > 1:
            starts = rf.split_dims(starts, axis=seqs_dim, dims=others)
        pos = starts + pos
    out = rf.gather(out_inner, indices=pos, axis=out_packed_dim, clip_to_valid=True)
    return rf.where(rf.sequence_mask(list(out_orig), device=out.device), out, 0)


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
        return tuple(d.dimension for d in raw_tensor.dims)

    @staticmethod
    def transpose_raw(raw_tensor: PackedRawTensor, perm: Sequence[int]) -> PackedRawTensor:
        """
        transpose. Pure metadata for packed storage:
        the outer dims are semantically unordered, the packed storage order stays canonical.
        No data movement. This makes e.g. copy_compatible_to_dims / copy_transpose work on packed tensors.
        """
        dims = tuple(raw_tensor.dims[i] for i in perm)
        return PackedRawTensor(
            inner=raw_tensor.inner,
            packed_dim=raw_tensor.packed_dim,
            orig_dims=raw_tensor.orig_dims,
            dims=dims,
            gap=raw_tensor.gap,
            align=raw_tensor.align,
        )

    @staticmethod
    def get_device(x: Tensor) -> Optional[str]:
        """:return: device of the packed data"""
        return _raw(x).inner.device

    @staticmethod
    def replace_dim(source: Tensor, *, in_dim: Dim, out_dim: Dim) -> Tensor:
        """
        replace_dim. Pure metadata, but unlike the base implementation,
        the packing relation in the wrapper must be updated as well:
        e.g. attention replaces the kv time dim (a packed dim),
        and with a stale packing relation, a later unpack would silently misalign the data.
        """
        raw = _raw(source)
        if not out_dim.is_dim_known():
            out_dim.copy_from(in_dim)
        if (in_dim in raw.orig_dims or in_dim == raw.packed_dim) and not _same_seq_lens(in_dim, out_dim):
            # e.g. the causal-attention kv dim, whose lens are per-step (2D dyn sizes):
            # not the same packing anymore, the metadata replace would misdescribe the storage.
            _warn_fallback_once("replace_dim", f"replacement dim {out_dim} has different seq lens than {in_dim}")
            x = unpack(source)
            # noinspection PyProtectedMember
            return x._raw_backend.replace_dim(x, in_dim=in_dim, out_dim=out_dim)
        out = source.copy_template_replace_dim_tag(
            axis=source.get_axis_from_description(in_dim), new_dim_tag=out_dim, name="replace_dim"
        )
        if in_dim in raw.orig_dims or in_dim == raw.packed_dim:
            out.raw_tensor = PackedRawTensor(
                inner=raw.inner,
                packed_dim=raw.packed_dim,
                orig_dims=tuple(out_dim if d == in_dim else d for d in raw.orig_dims),
                dims=tuple(out_dim if d == in_dim else d for d in raw.dims),
                gap=raw.gap,
                align=raw.align,
            )
        else:
            inner_out, _ = rf.replace_dim(raw.inner, in_dim=in_dim, out_dim=out_dim)
            out.raw_tensor = PackedRawTensor(
                inner=inner_out,
                packed_dim=raw.packed_dim,
                orig_dims=raw.orig_dims,
                dims=tuple(out_dim if d == in_dim else d for d in raw.dims),
                gap=raw.gap,
                align=raw.align,
            )
        return out

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
        return PackedRawTensor(
            inner=inner_out,
            packed_dim=raw_tensor.packed_dim,
            orig_dims=raw_tensor.orig_dims,
            gap=raw_tensor.gap,
            align=raw_tensor.align,
        )

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
        opts = {}
        # note: the native tensor_combine does not accept None for these
        if allow_broadcast_all_sources is not None:
            opts["allow_broadcast_all_sources"] = allow_broadcast_all_sources
        if dim_order is not None:
            opts["dim_order"] = dim_order
        out = rf.combine(_unpack_if_packed(a), kind, _unpack_if_packed(b), **opts)
        return _repack_result(out, template)

    @staticmethod
    def conv(
        source: Tensor,
        *,
        in_dim: Dim,
        out_dim: Dim,
        in_spatial_dims: Sequence[Dim],
        out_spatial_dims: Optional[Sequence[Dim]] = None,
        filter: Tensor,
        filter_size: Sequence[Dim],
        padding: Union[str, int, Sequence[int]],
        strides: Optional[Union[int, Sequence[int]]] = None,
        dilation_rate: Optional[Union[int, Sequence[int]]] = None,
        groups: Optional[int] = None,
        bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Sequence[Dim]]:
        """
        conv. Packed fast path (spatial dim = innermost packed dim, padding "same"/"valid"/int):
        the conv runs directly over the packed buffer as one long sequence,
        using the inter-seq gap as the boundary safety margin (see PackedRawTensor.gap):
        the gap frames are zeroed first, so the windows crossing a sequence boundary
        see exactly the zero padding that the per-sequence conv would produce.
        Every sequence's outputs land exactly at its input start / stride
        (the global output index of window start w is (w + pad_left) / stride;
        for stride > 1 this needs stride-aligned starts, i.e. stride | align, and align | gap):
        "same" stride 1 keeps the layout unchanged; otherwise seq lens and gap change,
        with the resulting layout verified at runtime against the closed form
        (see :func:`_strided_out_wrapper`).
        """
        raw = source.raw_tensor
        assert isinstance(raw, PackedRawTensor)
        n = len(in_spatial_dims)
        strides_ = list(strides) if isinstance(strides, (list, tuple)) else [strides or 1] * n
        dils_ = list(dilation_rate) if isinstance(dilation_rate, (list, tuple)) else [dilation_rate or 1] * n
        span = (filter_size[0].dimension - 1) * (dils_[0] or 1)  # window span beyond its start, time axis
        st = strides_[0] or 1
        if padding == "same":
            pad_l, pad_r = span // 2, (span + 1) // 2
        elif padding == "valid":
            pad_l = pad_r = 0
        elif isinstance(padding, int):
            pad_l = pad_r = padding
        elif isinstance(padding, (list, tuple)) and padding and isinstance(padding[0], int):
            pad_l = pad_r = padding[0]  # per-spatial-dim symmetric ints (e.g. from consistent-same padding)
        else:
            pad_l = pad_r = None  # unsupported spec
        required_gap = max(pad_l, pad_r) if pad_l is not None else None
        stride_ok = st == 1 or (raw.align % st == 0 and raw.gap % raw.align == 0)
        if (
            pad_l is not None
            and stride_ok
            and (st > 1 or raw.gap + span - pad_l - pad_r >= 0)
            and in_spatial_dims[0] == raw.orig_dims[-1]
            and not any(_dim_refs_packed(d, raw) for d in in_spatial_dims[1:])
            and not out_spatial_dims
            and not is_packed(filter)
            and (bias is None or not is_packed(bias))
        ):
            if raw.gap < required_gap:
                # still keep the conv packed: cheap re-layout with the required gap.
                # The warning stays so this gets noticed and fixed at pack() time.
                # For strided convs, keep the align | gap convention.
                target_gap = required_gap if st == 1 else -(-required_gap // raw.align) * raw.align
                _warn_fallback_once(
                    "conv",
                    f"gap {raw.gap} < required {required_gap} for the packed conv"
                    f" -- specify pack(..., gap=...) to avoid the extra re-layout",
                    action="re-packing with the required gap (conv stays packed)",
                )
                source = regap(source, target_gap)
                raw = source.raw_tensor
            inner = raw.inner
            mask = _frame_mask(raw)
            if mask is not None:
                inner = rf.where(mask, inner, 0.0)  # gaps may contain junk from previous elementwise ops
            out_inner, out_sp = rf.conv(
                inner,
                in_dim=in_dim,
                out_dim=out_dim,
                in_spatial_dims=[raw.packed_dim] + list(in_spatial_dims[1:]),
                filter=filter,
                filter_size=filter_size,
                padding=padding,
                strides=strides,
                dilation_rate=dilation_rate,
                groups=groups,
                bias=bias,
            )
            time_dim = in_spatial_dims[0]
            if st == 1 and out_sp[0] == raw.packed_dim:  # "same": layout unchanged
                out = raw.rewrap(out_inner, name="conv")
                return out, [time_dim] + list(out_sp[1:])
            (out_time,) = rf.make_conv_out_spatial_dims(
                [time_dim], filter_size=filter_size[0], strides=st, dilation_rate=dils_[0] or 1, padding=padding
            )
            if st == 1:
                # "valid"/int: seq lens and gap change by a constant, starts stay in place
                helper = PackedRawTensor(
                    inner=out_inner,
                    packed_dim=out_sp[0],
                    orig_dims=tuple(raw.orig_dims[:-1]) + (out_time,),
                    gap=raw.gap + span - pad_l - pad_r,
                    align=raw.align,
                )
                out = helper.rewrap(out_inner, name="conv")
                return out, [out_time] + list(out_sp[1:])
            out = _strided_out_wrapper(raw, out_inner, out_sp[0], out_time, st)
            if out is not None:
                return out, [out_time] + list(out_sp[1:])
            # out layout not expressible in closed form (unusual out-len semantics): extract to padded
            _warn_fallback_once(
                "conv:strided-out",
                "strided conv output layout not expressible in the (lens, gap, align) form",
                action="extracting the (packed-computed) output to padded storage",
            )
            return (
                _extract_strided(raw, out_inner, out_sp[0], out_time, st),
                [out_time] + list(out_sp[1:]),
            )
        return _dim_aware_call("conv", (source,), dict(
            in_dim=in_dim,
            out_dim=out_dim,
            in_spatial_dims=in_spatial_dims,
            out_spatial_dims=out_spatial_dims,
            filter=filter,
            filter_size=filter_size,
            padding=padding,
            strides=strides,
            dilation_rate=dilation_rate,
            groups=groups,
            bias=bias,
        ))

    @staticmethod
    def pad(
        source: Tensor,
        *,
        axes: Sequence[Dim],
        padding: Sequence[Tuple[Union[Dim, int, Tensor], Union[Dim, int, Tensor]]],
        out_dims: Sequence[Dim],
        handle_dynamic_dims: bool,
        mode: str = "constant",
        value: Optional[Union[Any, Tensor]] = None,
    ) -> Tensor:
        """
        pad. Packed fast path for right-padding the innermost packed dim (constant mode):
        an in-place layout change -- the new frames come out of the gap
        (footprints and seq starts unchanged: per-seq (len, gap) -> (len + p, gap - p)),
        and the pad value is written into the claimed slots.
        This is what e.g. the consistent-"same" padding of strided pools produces.
        """
        raw = source.raw_tensor
        assert isinstance(raw, PackedRawTensor)
        time_dim = raw.orig_dims[-1]
        if time_dim in list(axes) and mode == "constant":
            i = list(axes).index(time_dim)
            pad_l, pad_r = padding[i]
            if (
                isinstance(pad_l, int)
                and pad_l == 0
                and isinstance(pad_r, int)
                and pad_r >= 0
                and not any(_dim_refs_packed(d, raw) for j, d in enumerate(axes) if j != i)
            ):
                if raw.gap < pad_r:
                    target_gap = pad_r if raw.align == 1 else -(-pad_r // raw.align) * raw.align
                    _warn_fallback_once(
                        "pad",
                        f"gap {raw.gap} < pad {pad_r} for the packed in-place pad"
                        f" -- specify pack(..., gap=...) to avoid the extra re-layout",
                        action="re-packing with the required gap (pad stays packed)",
                    )
                    source = regap(source, target_gap)
                    raw = source.raw_tensor
                inner = raw.inner
                rest_axes = [d for j, d in enumerate(axes) if j != i]
                if rest_axes:
                    inner = raw.inner_backend.pad(
                        inner,
                        axes=rest_axes,
                        padding=[p for j, p in enumerate(padding) if j != i],
                        out_dims=[d for j, d in enumerate(out_dims) if j != i],
                        handle_dynamic_dims=handle_dynamic_dims,
                        mode=mode,
                        value=value,
                    )
                out_time = out_dims[i]
                helper = PackedRawTensor(
                    inner=inner,
                    packed_dim=raw.packed_dim,
                    orig_dims=tuple(raw.orig_dims[:-1]) + (out_time,),
                    gap=raw.gap - pad_r,
                    align=raw.align,
                )
                if pad_r:
                    # write the pad value into the newly claimed slots (they may contain junk)
                    old_mask = _frame_mask(raw)
                    new_mask = _frame_mask(helper)
                    claimed = rf.logical_not(old_mask)
                    if new_mask is not None:
                        claimed = rf.logical_and(new_mask, claimed)
                    inner = rf.where(claimed, value if value is not None else 0, inner)
                    helper.inner = inner
                return helper.rewrap(inner, name="pad")
        return _dim_aware_call(
            "pad",
            (source,),
            dict(
                axes=axes,
                padding=padding,
                out_dims=out_dims,
                handle_dynamic_dims=handle_dynamic_dims,
                mode=mode,
                value=value,
            ),
        )

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
        """
        softmax over a non-packed axis (e.g. vocab) -> on packed data.
        softmax over a packed dim (e.g. attention energies over time) -> segment softmax on packed data,
        which needs no masking at all.
        Otherwise: fallback.
        """
        raw = _raw(tensor)
        if not _dim_refs_packed(axis, raw):
            return raw.rewrap(rf.softmax(raw.inner, axis=axis, use_mask=use_mask), name="softmax")
        if axis in raw.orig_dims:
            # use_mask does not matter here: packed storage has no padded frames,
            # so the segment softmax is correct either way.
            out = _segment_softmax(tensor, axis=axis, log=False)
            if out is not None:
                return out
        _warn_fallback_once("softmax", f"axis {axis} (packed-related), no segment impl for this case")
        return _repack_result(rf.softmax(unpack(tensor), axis=axis, use_mask=use_mask), raw)

    @staticmethod
    def log_softmax(tensor: Tensor, *, axis: Dim, use_mask: bool = True) -> Tensor:
        """log_softmax, packed handling like :func:`softmax`."""
        raw = _raw(tensor)
        if not _dim_refs_packed(axis, raw):
            return raw.rewrap(rf.log_softmax(raw.inner, axis=axis, use_mask=use_mask), name="log_softmax")
        if axis in raw.orig_dims:
            # use_mask does not matter here, see softmax
            out = _segment_softmax(tensor, axis=axis, log=True)
            if out is not None:
                return out
        _warn_fallback_once("log_softmax", f"axis {axis} (packed-related), no segment impl for this case")
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
            and raw.gap == 0
            and raw.align == 1  # gap / alignment pad frames would contaminate the maskless reduce
        ):
            # use_mask=False: packed storage has no padded frames.
            return rf.reduce(raw.inner, mode=mode, axis=[raw.packed_dim] + extra_axes, use_mask=False)
        packed_axes = [d for d in axes if d in raw.orig_dims]
        other = [d for d in raw.orig_dims if d not in packed_axes]
        if (
            packed_axes
            and other
            and mode in ("sum", "max", "min", "mean", "logsumexp")
            and not any(_dim_refs_packed(d, raw) for d in extra_axes)
        ):
            # partial reduce over packed dims (e.g. over time, keeping batch):
            # segment reduce via rf.scatter on packed data, maskless.
            # Gap frames (if any) go to the dump segment, which is sliced off.
            # The result has no packed dims left, so it is a normal (plain) tensor.
            seg, scatter_dim, valid_dim = _segment_index(raw, other)
            out = rf.scatter(
                raw.inner, indices=seg, indices_dim=raw.packed_dim, mode=mode, out_dim=scatter_dim, use_mask=False
            )
            if scatter_dim != valid_dim:
                out, _ = rf.slice(out, axis=scatter_dim, size=valid_dim)
            if len(other) > 1:
                out = rf.split_dims(out, axis=valid_dim, dims=other)
            if extra_axes:
                out = rf.reduce(out, mode=mode, axis=extra_axes, use_mask=use_mask)
            return out
        _warn_fallback_once("reduce", f"partial reduce over packed dims (axis {axes}, mode {mode})")
        return _repack_result(rf.reduce(unpack(source), mode=mode, axis=axes, use_mask=use_mask), raw)


# All other structural ops go through the generic dim-aware wrapper:
# packed data directly if the call does not reference the packed dims, otherwise unpack fallback.
for _name in [
    "batch_norm",
    "concat",
    "cumsum",
    "expand_dim",
    "flip_no_mask",
    "gather",
    "masked_scatter",
    "masked_select",
    "merge_dims",
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


def pack(
    source: Tensor,
    *,
    dims: Optional[Sequence[Dim]] = None,
    out_dim: Optional[Dim] = None,
    gap: int = 0,
    align: int = 1,
) -> Tensor:
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
    :param gap: extra frames between sequences in the packed buffer
        (see :class:`PackedRawTensor`; e.g. for packed conv, specify by hand what the model needs).
    :param align: per-seq footprint alignment (see :class:`PackedRawTensor`;
        for strided ops, use the total downsampling factor of the model, and gap a multiple of it).
    :return: tensor with same dims as source, packed storage
    """
    if dims is None:
        dims = _auto_pack_dims(source)
        assert dims, f"pack: no dims with dynamic length found in {source}"
    if gap or align > 1:
        # gapped/aligned layout: scatter frames to their positions, zeros in between the sequences
        last = dims[-1]
        assert last.dyn_size_ext is not None, f"pack: innermost packed dim {last} needs dyn sizes"
        if out_dim is None:
            out_dim = Dim(_packed_total(dims, gap, align), name="packed_gap")
        pos = _padded_positions(dims, gap, align)
        inner = rf.scatter(source, indices=pos, indices_dim=list(dims), out_dim=out_dim, use_mask=True)
        packed_dim = out_dim
    else:
        inner, packed_dim = rf.pack_padded(source, dims=dims, out_dim=out_dim)
    if source.feature_dim is not None and inner.feature_dim is None and source.feature_dim in inner.dims:
        inner.feature_dim = source.feature_dim  # masked_select / scatter drop it
    # Note: the returned dims order is the canonical virtual order (packed dims first),
    # which can differ from source.dims order (dims are unordered semantically in RF).
    helper = PackedRawTensor(inner=inner, packed_dim=packed_dim, orig_dims=dims, gap=gap, align=align)
    return helper.rewrap(inner, name=(source.name or "packed") + "_packed")


def regap(source: Tensor, gap: int, *, align: Optional[int] = None) -> Tensor:
    """
    :return: same content, packed with the given gap (and align, default: keep):
        a cheap packed -> packed re-layout (one scatter over the frames, no padded intermediate).
        Used e.g. by the packed conv when the tensor's gap is too small.
    """
    raw = _raw(source)
    if align is None:
        align = raw.align
    others = raw.orig_dims[:-1]
    if (raw.gap == gap and raw.align == align) or not others:
        return source
    last = raw.orig_dims[-1]
    new_dim = Dim(_packed_total(raw.orig_dims, gap, align), name="packed_regap")
    new_starts, seqs_dim = _seq_starts_math(raw.orig_dims, gap, align)
    t_coords = _frame_coords(raw, last)
    seg, _, _ = _segment_index(raw, others)
    pos = rf.gather(new_starts, indices=seg, axis=seqs_dim, clip_to_valid=True) + t_coords
    mask = _frame_mask(raw)
    if mask is not None:
        # route old gap frames to a dump slot, then slice it off
        ext_dim = new_dim + 1
        pos = rf.where(mask, pos, rf.cast(new_dim.get_dim_value_tensor(), pos.dtype))
        inner_ext = rf.scatter(
            raw.inner, indices=pos, indices_dim=raw.packed_dim, out_dim=ext_dim, use_mask=False
        )
        inner_new, _ = rf.slice(inner_ext, axis=ext_dim, size=new_dim)
    else:
        inner_new = rf.scatter(raw.inner, indices=pos, indices_dim=raw.packed_dim, out_dim=new_dim, use_mask=False)
    if raw.inner.feature_dim is not None and inner_new.feature_dim is None:
        inner_new.feature_dim = raw.inner.feature_dim
    helper = PackedRawTensor(inner=inner_new, packed_dim=new_dim, orig_dims=raw.orig_dims, gap=gap, align=align)
    return helper.rewrap(inner_new, name="regap")


def unpack(source: Tensor) -> Tensor:
    """
    :param source: tensor with packed storage. if not packed, returned as-is.
    :return: tensor with normal padded storage of the inner backend, same dims
    """
    raw = source.raw_tensor
    if not isinstance(raw, PackedRawTensor):
        return source
    if raw.gap or raw.align > 1:
        pos = _padded_positions(raw.orig_dims, raw.gap, raw.align)
        out = rf.gather(raw.inner, indices=pos, axis=raw.packed_dim, clip_to_valid=True)
        # zero the padded frames, like the dense masked_scatter does
        out = rf.where(rf.sequence_mask(list(raw.orig_dims), device=out.device), out, 0)
    else:
        out = rf.pad_packed(raw.inner, dims=raw.orig_dims, in_dim=raw.packed_dim)
    if source.feature_dim is not None and out.feature_dim is None and source.feature_dim in out.dims:
        out.feature_dim = source.feature_dim  # masked_scatter drops it
    return out


register_backend_by_tensor_type(PackedRawTensor, PackedBackend)
