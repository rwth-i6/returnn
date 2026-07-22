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
- Per-backend (torch) specializations, implemented here:
  attention via fused varlen kernels --
  compiled FlexAttention with a document block mask over the flat packed buffer
  (self-/cross-/causal attention, and rel-pos attention via a score_mod bias,
  see :func:`rf.rel_pos_self_attention`),
  or NJT (nested-jagged) SDPA where dropout is needed;
  conv/pool/pad directly on the (gapped) packed buffer;
  masked-stats batch_norm;
  segment softmax / reductions via :func:`rf.scatter`.
  Layout metadata (cu_seqlens, block masks, frame coords/masks)
  is cached per packing, see :data:`_layout_cache`.
- Fallback for ops needing the sequence structure: unpack, run the op on padded storage, repack.
  The result is repacked in the same format/order
  (sharing the packed dim if the packed dims are unchanged;
  following a replaced dim, e.g. the subsampled time dim of a strided conv, if the op created one),
  so downstream computation stays packed.
  Only if the new packing cannot be inferred does the result stay padded.
  The first fallback per op prints a warning (see :func:`_warn_fallback_once`),
  so slow paths are visible during development.

Known limitations (TODO):

- ``dim_order`` in :func:`combine` is ignored on the packed fast path.
- Strided conv/pool output layouts must have uniform per-seq residuals,
  in practice: seq lens multiple of the total stride
  (otherwise the output is extracted to padded storage and repacked, with a warning).
  Would need per-seq output footprints (a layout generalization).
- FlexAttention has no dropout support (torch 2.7 and 2.12):
  attention with att_dropout in training runs the eager NJT SDPA path
  (correct, flash kernels, but heavy per-call python --
  compiled NJT backward is broken in torch 2.7/2.12).

Status: complete for the Conformer (default rel-pos attention + BatchNorm)
and the Transformer AED (incl. cross-attention and the CE output block),
both verified against the padded reference (tests/test_rf_packed.py)
and running without any fallback.
Import this module explicitly to activate the dispatch registration.
"""

from __future__ import annotations
from typing import Any, Optional, Union, Sequence, Set, Tuple, Dict

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from ._backend import Backend, register_backend_by_tensor_type, global_backend
from ._cache import Cache

__all__ = ["PackedRawTensor", "PackedBackend", "pack", "pack_import", "unpack", "regap", "is_packed"]


# Layout metadata (cu_seqlens, flex document mask, frame coords/masks, ...)
# is identical for every tensor sharing a packing
# and constant over the lifetime of the dims,
# so it is computed once and cached, see :data:`_layout_cache`.
# rf._cache.Cache keys eager dyn dims by their size VALUES (see DimWrapper),
# so equal-length dims share the entry
# (e.g. the fresh kv dim which attention creates per layer via replace_dim),
# and it maps dims in cached outputs back to the queried dims.
_layout_cache = Cache(128)


def _packing_cache_key(kind: str, raw: PackedRawTensor, device) -> Tuple[Any, ...]:
    """
    :param kind: metadata kind (part of the key)
    :param raw: the packing
    :param device: torch device (part of the key)
    :return: cache key for layout metadata of the packing.
        The dims go in directly (the Cache handles the value semantics, see above).
    """
    return kind, raw.orig_dims, raw.gap, raw.align, _layout_lens_key(raw.layout_lens), str(device)


def _layout_lens_key(layout_lens):
    """:return: value key for the per-seq layout lens (a small cpu int tensor, cheap to read)"""
    if layout_lens is None:
        return None
    return tuple(int(v) for v in layout_lens.raw_tensor.flatten())


class PackedRawTensor:
    """
    Raw-tensor wrapper marking packed storage.

    The wrapped :attr:`inner` is a normal RF :class:`Tensor` (of the inner backend)
    holding the packed data, dims = [packed_dim] + remaining (non-packed) dims.
    :attr:`orig_dims` are the dims packed into :attr:`packed_dim`, e.g. (batch, time);
    the seq lens live in the dyn dims dyn_size_ext as usual, nothing is duplicated here.
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
        layout_lens: Optional[Tensor] = None,
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
        # layout_lens = optional per-seq layout lens (dims like the seq lens, e.g. [batch]):
        # the footprint becomes roundup(layout_len + gap, align), layout_len >= content len,
        # decoupling the footprint from the content length per seq.
        # Produced by strided ops on mixed seq-len residuals (see _strided_out_wrapper),
        # where the out footprint old_footprint // stride is not a function of the out content length.
        # None = the footprint follows the content lens (the default closed form).
        self.layout_lens = layout_lens

    @property
    def has_gap_frames(self) -> bool:
        """:return: whether the buffer has non-content frames (gap / alignment / footprint junk)"""
        return bool(self.gap) or self.align > 1 or self.layout_lens is not None

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
        starts, seqs_dim = _seq_starts_math(self.orig_dims, self.gap, self.align, layout_lens=self.layout_lens)
        assert starts is not None, f"seq_starts: no outer packed dims in {self.orig_dims}"
        return starts, seqs_dim

    def cu_seqlens(self, *, device: Optional[str] = None) -> Tuple[Tensor, Dim]:
        """
        :return: (cu, cu_dim): cumulative sequence boundaries
            as used by varlen attention kernels (e.g. FlashAttention flash_attn_varlen_func):
            [seqs+1] int32, cu[i] = start of seq i, cu[-1] = total number of frames.
            Dense layout only (gap == 0), since the boundaries imply that seq i ends where seq i+1 starts.
        """
        assert self.gap == 0 and self.align == 1 and self.layout_lens is None, "cu_seqlens requires the dense layout"
        key = _packing_cache_key("cu_seqlens", self, device)
        hit = _layout_cache.get(key)
        if hit is not None:
            return hit
        starts, seqs_dim = self.seq_starts()
        total = self.packed_dim.get_dim_value_tensor()
        if isinstance(total, Tensor):
            total = rf.cast(total, starts.dtype)
        else:
            # a static packed dim (e.g. built by the data pipeline) yields a python int, not a tensor
            total = rf.copy_to_device(rf.constant(int(total), dims=(), dtype=starts.dtype), starts.device)
        end_dim = Dim(1, name="cu_seqlens_end")
        cu, cu_dim = rf.concat((starts, seqs_dim), (rf.expand_dim(total, dim=end_dim), end_dim))
        cu = rf.cast(cu, "int32")
        if device:
            cu = rf.copy_to_device(cu, device)
        _layout_cache.set(key, (cu, cu_dim))
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
            # identity: all tensors of one lineage share the object (rewrap/regap propagate it)
            and self.layout_lens is other.layout_lens
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
        :param name: name for the resulting outer Tensor
        :return: outer (virtual-dims) Tensor wrapping inner_out with the same packing relation
        """
        assert self.packed_dim in inner_out.dims
        opts = {}
        if inner_out.feature_dim is not None and inner_out.feature_dim != self.packed_dim:
            opts["feature_dim"] = inner_out.feature_dim
        # else: no explicit feature_dim, so the outer Tensor applies the usual default heuristic --
        # the inner tensor often loses the implicit feature dim (its heuristic fails without a batch dim),
        # while the virtual (unpacked) dims match what the padded op would produce.
        vdims = self.virtual_dims(inner_out)
        out = Tensor(
            name=name or inner_out.name,
            dims=vdims,
            dtype=inner_out.dtype,
            sparse_dim=inner_out.sparse_dim,
            **opts,
        )
        out.raw_tensor = PackedRawTensor(
            inner=inner_out,
            packed_dim=self.packed_dim,
            orig_dims=self.orig_dims,
            dims=vdims,
            gap=self.gap,
            align=self.align,
            layout_lens=self.layout_lens,
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

# Which attention implementation actually ran, by call count.
# Diagnosis / benchmarking: e.g. a silent fall-through from the direct-flash path
# to eager NJT is functionally correct but 10-20x slower per call,
# and NOT visible via _warned_fallback_ops (both are "fast paths") --
# it once went unnoticed exactly that way.
# Keys: "flash", "njt_compiled", "njt_eager", "flex_doc", "rel_pos_flex".
attention_path_counts: Dict[str, int] = {}


def _count_attention_path(name: str):
    attention_path_counts[name] = attention_path_counts.get(name, 0) + 1


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


def _seq_footprints(
    orig_dims: Sequence[Dim], gap: int, align: int, *, layout_lens: Optional[Tensor] = None
) -> Tuple[Optional[Tensor], Optional[Dim]]:
    """
    :return: (footprints, seqs_dim): frames occupied per sequence in the packed buffer,
        roundup(len + gap, align) with len = layout_lens (if given) or the content lens,
        flat row-major over the outer packed dims.
        (None, None) if there are no outer packed dims.
    """
    others = tuple(orig_dims[:-1])
    if not others:
        return None, None
    if layout_lens is not None:
        lens = layout_lens
    else:
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


def _seq_starts_math(
    orig_dims: Sequence[Dim], gap: int, align: int, *, layout_lens: Optional[Tensor] = None
) -> Tuple[Optional[Tensor], Optional[Dim]]:
    """
    :return: (starts, seqs_dim): see :func:`PackedRawTensor.seq_starts`:
        exclusive cumsum of the seq footprints.
        (None, None) if there are no outer packed dims (a single sequence; starts trivially at 0).
    """
    footprints, seqs_dim = _seq_footprints(orig_dims, gap, align, layout_lens=layout_lens)
    if footprints is None:
        return None, None
    cum = rf.cumsum(footprints, spatial_dim=seqs_dim)
    return cum - footprints, seqs_dim


def _padded_positions(
    orig_dims: Sequence[Dim], gap: int, align: int, *, layout_lens: Optional[Tensor] = None
) -> Tensor:
    """
    :return: [orig_dims...] (int): for every frame of the padded grid,
        its position in the packed buffer (only meaningful on the non-padded frames).
    """
    pos = rf.range_over_dim(orig_dims[-1])
    starts, seqs_dim = _seq_starts_math(orig_dims, gap, align, layout_lens=layout_lens)
    if starts is not None:
        others = tuple(orig_dims[:-1])
        if len(others) > 1:
            starts = rf.split_dims(starts, axis=seqs_dim, dims=others)
        if starts.device != pos.device:
            # starts derive from the dyn sizes (often cpu), pos follows the default device
            starts = rf.copy_to_device(starts, pos.device)
        pos = rf.combine_bc(starts, "add", pos)  # cross-dim (seqs x frames) broadcast
    return pos


def _packed_total(orig_dims: Sequence[Dim], gap: int, align: int, *, layout_lens: Optional[Tensor] = None) -> Tensor:
    """:return: scalar (int32): total number of frames in the packed buffer (sum of footprints)"""
    footprints, _ = _seq_footprints(orig_dims, gap, align, layout_lens=layout_lens)
    if footprints is None:
        last = orig_dims[-1]
        assert last.dyn_size_ext is not None
        return rf.cast(last.dyn_size_ext, "int32")
    return rf.cast(rf.reduce_sum(footprints, axis=list(footprints.dims)), "int32")


def _frame_coords(template: PackedRawTensor, d: Dim) -> Tensor:
    """
    :param template: the packing
    :param d: one of the template's packed dims
    :return: [packed_dim] (int): for every packed frame, its coordinate in d
        (gap frames get 0 -- only meaningful on sequence frames).
        Cheap: only an int grid over the packed dims is scattered, no feature-sized data.
    """
    # The result is Tensor over template.packed_dim,
    # so the key must include that dim (the Cache remaps it on equal-valued hits).
    # The full layout must be in the key too:
    # e.g. the packed pad changes (orig_dims, gap) but keeps the packed dim.
    key = (
        "frame_coords",
        template.packed_dim,
        template.orig_dims,
        template.gap,
        template.align,
        _layout_lens_key(template.layout_lens),
        template.orig_dims.index(d),
        str(template.inner.device),
    )
    hit = _layout_cache.get(key)
    if hit is not None:
        return hit
    grid = rf.range_over_dim(d)
    for o in template.orig_dims:
        if o not in grid.dims:
            grid = rf.expand_dim(grid, dim=o)  # broadcast view, ints only
    pos = _padded_positions(template.orig_dims, template.gap, template.align, layout_lens=template.layout_lens)
    # padding frames (padded grid beyond a seq len) can overshoot the buffer for tight packings;
    # they are masked out of the scatter, so clamping their index keeps it in-bounds without effect.
    total = template.packed_dim.get_dim_value_tensor()
    if isinstance(total, Tensor):
        # dyn sizes live on cpu; move to the coords device so the clamp does not mix devices
        total = rf.copy_to_device(rf.cast(total, pos.dtype), pos.device)
    pos = rf.minimum(pos, total - 1)
    out = rf.scatter(
        grid, indices=pos, indices_dim=list(template.orig_dims), out_dim=template.packed_dim, use_mask=True
    )
    _layout_cache.set(key, out)
    return out


def _frame_mask(template: PackedRawTensor) -> Optional[Tensor]:
    """
    :return: [packed_dim] (bool): True on sequence frames, False on gap frames.
        None for the dense layout (all frames are sequence frames).
    """
    if not template.has_gap_frames:
        return None
    # full layout in the key, see _frame_coords
    key = (
        "frame_mask",
        template.packed_dim,
        template.orig_dims,
        template.gap,
        template.align,
        _layout_lens_key(template.layout_lens),
        str(template.inner.device),
    )
    hit = _layout_cache.get(key)
    if hit is not None:
        return hit
    ones = rf.cast(rf.sequence_mask(list(template.orig_dims)), "int32")
    pos = _padded_positions(template.orig_dims, template.gap, template.align, layout_lens=template.layout_lens)
    counts = rf.scatter(
        ones, indices=pos, indices_dim=list(template.orig_dims), out_dim=template.packed_dim, use_mask=True
    )
    out = counts > 0
    _layout_cache.set(key, out)
    return out


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


def _stats_active(name: str, kwargs) -> bool:
    """
    :param name: op from :data:`_DENSE_ONLY_INNER_OPS`
    :param kwargs: the call kwargs
    :return: whether the op will actually compute statistics over the packed dim in this call.
        E.g. batch_norm in eval mode with running stats is purely elementwise,
        then gap frames are harmless and the inner path is fine even on a gapped layout.
    """
    if name == "batch_norm":
        # mirrors the `training` condition in the torch backend batch_norm
        train_flag = rf.get_run_ctx().is_train_flag_enabled(func=rf.BatchNorm.__call__)
        return train_flag is not False or kwargs.get("running_mean") is None
    return True


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
        if raw.has_gap_frames:
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
    :param template: the packing
    :param seg_dims: (some of) the template's packed dims defining the segments
    :return: (seg, scatter_dim, valid_dim): for every packed frame, its flat index over seg_dims
        (row-major, matching the merged-dims layout);
        the dim to scatter into; and the dim holding the valid segments.
        Dense layout: scatter_dim == valid_dim (the merged seg_dims).
        Gapped layout: gap frames are routed to an extra dump segment
        (scatter_dim = valid_dim + 1), so any segment reduction stays correct;
        slice the dump segment off (or simply never gather it).
    """
    # full layout in the key, see _frame_coords
    key = (
        "segment_index",
        template.packed_dim,
        template.orig_dims,
        template.gap,
        template.align,
        _layout_lens_key(template.layout_lens),
        tuple(template.orig_dims.index(d) for d in seg_dims),
        str(template.inner.device),
    )
    hit = _layout_cache.get(key)
    if hit is not None:
        return hit
    seg = None
    merged = None
    for d in seg_dims:
        coords = _frame_coords(template, d)
        seg = coords if seg is None else seg * d.get_dim_value_tensor() + coords
        merged = d if merged is None else merged * d
    if template.has_gap_frames:
        mask = _frame_mask(template)
        dump = merged.get_dim_value_tensor()
        if isinstance(dump, Tensor):
            dump = rf.cast(dump, seg.dtype)
        seg = rf.where(mask, seg, dump)
        out = (seg, merged + 1, merged)
    else:
        out = (seg, merged, merged)
    _layout_cache.set(key, out)
    return out


def _batch_norm_gapped(source: Tensor, kwargs) -> Optional[Tensor]:
    """
    batch_norm on a gapped packed layout in training,
    with the statistics over the valid frames only:
    masked mean/var directly on the packed buffer, no re-layout.
    Mirrors the torch backend batch_norm training semantics
    (biased var for the normalization;
    unbiased var + in-place no-grad update for the running stats, torch F.batch_norm convention).
    None if not applicable (then the dense re-layout path runs).

    :param source: packed, gapped
    :param kwargs: the batch_norm call kwargs, see :func:`Backend.batch_norm`
    :return: packed result (input packing kept), or None
    """
    raw = _raw(source)
    in_dim = kwargs.get("in_dim")
    running_mean, running_variance = kwargs.get("running_mean"), kwargs.get("running_variance")
    gamma, beta = kwargs.get("gamma"), kwargs.get("beta")
    epsilon, momentum, affine = kwargs.get("epsilon"), kwargs.get("momentum"), kwargs.get("affine")
    inner = raw.inner
    if not isinstance(in_dim, Dim) or set(inner.dims) != {raw.packed_dim, in_dim}:
        return None  # unusual layout, keep the generic path
    if raw.inner_backend.name != "torch":
        return None  # the in-place running-stat update below is raw torch
    mask = _frame_mask(raw)
    n_t = _packed_total(raw.orig_dims, 0, 1)  # number of valid frames (cpu, from the dyn sizes)
    n = rf.cast(rf.copy_to_device(n_t, inner.device), inner.dtype)
    x0 = rf.where(mask, inner, 0.0)
    mean = rf.reduce_sum(x0, axis=raw.packed_dim, use_mask=False) / n
    diff = rf.where(mask, inner - mean, 0.0)
    var = rf.reduce_sum(diff * diff, axis=raw.packed_dim, use_mask=False) / n
    if running_mean is not None:
        import torch

        with torch.no_grad():
            n_f = float(n_t.raw_tensor)
            unbiased = n_f / max(n_f - 1.0, 1.0)
            rm, rv = running_mean.raw_tensor, running_variance.raw_tensor
            rm.mul_(1.0 - momentum).add_(mean.raw_tensor.detach().to(rm.dtype), alpha=momentum)
            rv.mul_(1.0 - momentum).add_(var.raw_tensor.detach().to(rv.dtype) * unbiased, alpha=momentum)
    out_inner = (inner - mean) / rf.sqrt(var + epsilon)
    if affine:
        out_inner = out_inner * gamma + beta
    out_inner.feature_dim = in_dim
    return raw.rewrap(out_inner, name="batch_norm")


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


def _conform_packing(x, target_raw: PackedRawTensor):
    """
    Conform x to target_raw's packing (nesting-aware).
    A packed tensor over the SAME sequences (same orig_dims) but a different layout
    (gap/align/layout_lens, or just a different packed dim object) is regapped to
    target_raw's layout and rewrapped onto its packed dim, so a multi-arg op can run
    on the packed data. Anything else (not packed, already same packing, or different
    sequences) is returned unchanged.
    """
    if isinstance(x, (list, tuple)):
        return type(x)(_conform_packing(e, target_raw) for e in x)
    if isinstance(x, Tensor) and is_packed(x):
        xr = x.raw_tensor
        if not target_raw.same_packing(xr) and xr.orig_dims == target_raw.orig_dims:
            y = regap(x, target_raw.gap, align=target_raw.align, layout_lens=target_raw.layout_lens)
            yr = y.raw_tensor
            if yr.packed_dim != target_raw.packed_dim:
                inner, _ = rf.replace_dim(yr.inner, in_dim=yr.packed_dim, out_dim=target_raw.packed_dim)
                y = target_raw.rewrap(inner, name=x.name)
            return y
    return x


def _dim_aware_call(name: str, args, kwargs):
    """see :func:`_make_dim_aware_op`"""
    all_values = list(args) + list(kwargs.values())
    packed_args = [x for x in _flatten(all_values) if isinstance(x, Tensor) and is_packed(x)]
    assert packed_args, f"PackedBackend.{name}: no packed tensor in args"
    raw0 = packed_args[0].raw_tensor
    # conform other packed args (same seqs, possibly a different layout) to raw0's packing,
    # so multi-arg ops (combine, compare, where, concat, ...) stay packed instead of unpacking.
    args = [_conform_packing(x, raw0) for x in args]
    kwargs = {k: _conform_packing(v, raw0) for k, v in kwargs.items()}
    all_values = list(args) + list(kwargs.values())
    packed_args = [x for x in _flatten(all_values) if isinstance(x, Tensor) and is_packed(x)]
    referenced = _collect_referenced_dims(*all_values)
    if all(raw0.same_packing(x.raw_tensor) for x in packed_args[1:]):
        overlap = referenced & (set(raw0.orig_dims) | {raw0.packed_dim})
        if overlap:
            _warn_fallback_once(name, f"references packed dims {sorted(overlap, key=lambda d: d.name or '')}")
        elif name in _DENSE_ONLY_INNER_OPS and raw0.has_gap_frames and _stats_active(name, kwargs):
            # e.g. batch_norm: its statistics implicitly reduce over the packed dim,
            # and gap frames would pollute them.
            # Re-layout to dense, run there, restore the layout:
            # correct statistics by construction (incl. in-place running-stat updates),
            # no padded intermediate,
            # and the result keeps the original packing.
            src = packed_args[0] if len(packed_args) == 1 else None
            if name == "batch_norm" and src is not None:
                out = _batch_norm_gapped(src, kwargs)
                if out is not None:
                    return out
            if src is not None and (any(x is src for x in args) or any(v is src for v in kwargs.values())):
                _warn_fallback_once(
                    name,
                    "implicitly reduces over the packed dim, gapped layout",
                    action=f"re-layouting to dense and back ({name} stays packed)",
                )
                dense = regap(src, 0, align=1)
                out = _dim_aware_call(
                    name,
                    [dense if x is src else x for x in args],
                    {k: (dense if v is src else v) for k, v in kwargs.items()},
                )
                if isinstance(out, Tensor) and is_packed(out) and out.raw_tensor.same_packing(dense.raw_tensor):
                    back = regap(out, raw0.gap, align=raw0.align)
                    inner, _ = rf.replace_dim(
                        back.raw_tensor.inner, in_dim=back.raw_tensor.packed_dim, out_dim=raw0.packed_dim
                    )
                    return raw0.rewrap(inner, name=name)
                return out  # unusual result structure; correct, just re-layouted
            _warn_fallback_once(name, "implicitly reduces over the packed dim, gapped layout")
        else:
            inner_args = [_map_packed_to_inner(x) for x in args]
            inner_kwargs = {k: _map_packed_to_inner(v) for k, v in kwargs.items()}
            out = getattr(raw0.inner_backend, name)(*inner_args, **inner_kwargs)
            return _rewrap_result(out, raw0)
    else:
        _warn_fallback_once(name, "packed args over different sequences")
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


# Set when NJT SDPA failed with an environment-level error (e.g. CUDA-built torch without a driver):
# then we skip further attempts (the construction cost + exception would repeat on every call).
# Shape-specific failures do not set this.
_njt_sdpa_env_broken = False


def _sdpa_no(reason: str):
    """warn once + None: the packed varlen SDPA is not applicable, the generic path runs"""
    _warn_fallback_once(
        "scaled_dot_product_attention",
        reason,
        action="using the generic implementation (with the packed matmul/softmax handling)",
    )
    return None


def _harmonize_qkv_dtypes(q_t, k_t, v_t):
    """
    Mixed q/k/v dtypes occur under autocast:
    e.g. value in bf16 straight from an autocast matmul,
    while query/key got promoted back to f32 (e.g. by the rotary sin/cos terms).
    F.scaled_dot_product_attention would cast all inputs under autocast,
    but the direct flex/flash/NJT calls validate equal dtypes, so do the same cast here.
    """
    import torch

    if q_t.dtype == k_t.dtype == v_t.dtype:
        return q_t, k_t, v_t
    # noinspection PyArgumentList
    if hasattr(torch, "get_autocast_dtype") and torch.is_autocast_enabled(q_t.device.type):  # torch >= 2.4
        dtype = torch.get_autocast_dtype(q_t.device.type)
    else:
        dtype = torch.promote_types(torch.promote_types(q_t.dtype, k_t.dtype), v_t.dtype)
    return q_t.to(dtype), k_t.to(dtype), v_t.to(dtype)


def _njt_sdpa_raw(q_t, k_t, v_t, cu_q, cu_k, dropout_p: float, is_causal: bool, scale: float):
    """
    torch SDPA over nested (jagged) tensors built from the packed values + offsets.

    :param q_t: [total_q, H, D]
    :param k_t: [total_kv, H, D]
    :param v_t: [total_kv, H, Dv]
    :param cu_q: [n_seqs + 1] (int64), query seq boundaries
    :param cu_k: [n_seqs + 1] (int64), key/value seq boundaries
    :param dropout_p:
    :param is_causal:
    :param scale:
    :return: [total_q, H, Dv]
    """
    import torch

    # [B, T(jagged), H, D] -> [B, H, T(jagged), D]
    q_n = torch.nested.nested_tensor_from_jagged(q_t, offsets=cu_q).transpose(1, 2)
    k_n = torch.nested.nested_tensor_from_jagged(k_t, offsets=cu_k).transpose(1, 2)
    v_n = torch.nested.nested_tensor_from_jagged(v_t, offsets=cu_k).transpose(1, 2)
    # Restrict to the fused kernels (flash / mem-efficient / cudnn), exclude MATH:
    # F.sdpa would otherwise silently fall back to the unfused math path
    # (materialized energies) in unsupported cases; with the restriction it raises,
    # our caller warns visibly and uses the generic packed path instead.
    from torch.nn.attention import sdpa_kernel, SDPBackend

    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    if hasattr(SDPBackend, "CUDNN_ATTENTION"):
        backends.append(SDPBackend.CUDNN_ATTENTION)
    with sdpa_kernel(backends):
        out_n = torch.nn.functional.scaled_dot_product_attention(
            q_n, k_n, v_n, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )
    return out_n.transpose(1, 2).values()


_njt_sdpa_compiled = None
_njt_sdpa_compile_failed = False

_flash_varlen_fn = None
_flash_varlen_broken = False


def _get_flash_varlen_fn():
    """
    :return: autograd function
        (q, k, v, cu_q, cu_k, max_q, max_k, dropout_p, is_causal, scale) -> out,
        calling the aten flash varlen kernels directly
        (torch.ops.aten._flash_attention_forward/_backward):
        the same kernels NJT SDPA dispatches to,
        but without the per-op python subclass overhead,
        with native dropout (the rng state is saved for the backward),
        and with the flash backward
        (measured ~1.65x faster fwd+bwd than compiled flex, ~23x than eager NJT).
        Private aten API: the caller must handle signature drift across torch versions
        (catch TypeError etc. and set :data:`_flash_varlen_broken`).
    """
    global _flash_varlen_fn
    if _flash_varlen_fn is not None:
        return _flash_varlen_fn
    import torch

    # noinspection PyAbstractClass
    class _FlashVarlenAttention(torch.autograd.Function):
        """flash varlen attention, see :func:`_get_flash_varlen_fn`"""

        @staticmethod
        def forward(ctx, q, k, v, cu_q, cu_k, max_q, max_k, dropout_p, is_causal, scale):
            """forward"""
            # noinspection PyProtectedMember
            out, lse, rng0, rng1, _ = torch.ops.aten._flash_attention_forward(
                q, k, v, cu_q, cu_k, max_q, max_k, dropout_p, is_causal, False, scale=scale
            )
            ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_k, rng0, rng1)
            ctx.max_q, ctx.max_k = max_q, max_k
            ctx.dropout_p, ctx.is_causal, ctx.scale = dropout_p, is_causal, scale
            return out

        # noinspection PyMethodOverriding
        @staticmethod
        def backward(ctx, grad_out):
            """backward"""
            q, k, v, out, lse, cu_q, cu_k, rng0, rng1 = ctx.saved_tensors
            # noinspection PyProtectedMember
            gq, gk, gv = torch.ops.aten._flash_attention_backward(
                grad_out.contiguous(),
                q,
                k,
                v,
                out,
                lse,
                cu_q,
                cu_k,
                ctx.max_q,
                ctx.max_k,
                ctx.dropout_p,
                ctx.is_causal,
                rng0,
                rng1,
                scale=ctx.scale,
            )
            return gq, gk, gv, None, None, None, None, None, None, None

    _flash_varlen_fn = _FlashVarlenAttention.apply
    return _flash_varlen_fn


def _get_njt_sdpa_fn(device_type: str, *, needs_grad: bool):
    """
    :param device_type: "cuda" / "cpu" / ...
    :param needs_grad: whether the call needs autograd.
        Compiled NJT backward is broken (torch 2.7: AOT autograd tangent meta assert),
        so we compile only the inference path;
        the training no-dropout case avoids NJT via :func:`_flex_doc_attention` instead.
    :return: (fn to use, eager fn) for :func:`_njt_sdpa_raw`.
        The eager NJT python-subclass dispatch has a large per-call overhead
        (profiled ~5x the whole padded step time for a Transformer AED);
        NJT is designed to run under torch.compile, which removes that entirely,
        so on CUDA we lazily compile (dynamic shapes, shared across calls).
    """
    global _njt_sdpa_compiled
    if needs_grad or device_type != "cuda" or _njt_sdpa_compile_failed:
        return _njt_sdpa_raw, _njt_sdpa_raw
    if _njt_sdpa_compiled is None:
        import torch

        _njt_sdpa_compiled = torch.compile(_njt_sdpa_raw, dynamic=True)
    return _njt_sdpa_compiled, _njt_sdpa_raw


def _sdpa_varlen_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    qk_feat_dim: Dim,
    v_feat_dim: Dim,
    kv_spatial_dim: Dim,
    is_causal: bool,
    dropout_p: float = 0.0,
    scale: Optional[float],
    allow_njt: bool = True,
) -> Optional[Tensor]:
    """
    Varlen SDPA directly on the packed layout (values buffer + cu_seqlens offsets):
    no padding compute, no materialized energies -- pure torch, no external package.
    Preferred impl: the aten flash varlen kernels called directly
    (cuda + fp16/bf16, incl. native dropout, see :func:`_get_flash_varlen_fn`);
    otherwise torch SDPA over nested (jagged) tensors,
    unless ``allow_njt`` is False (then the caller falls back to flex instead --
    eager NJT has a large per-call python overhead).
    None if not applicable (then the flex / generic paths run).
    """
    global _njt_sdpa_env_broken, _njt_sdpa_compile_failed, _flash_varlen_broken

    if _njt_sdpa_env_broken:
        return None  # already warned once when it broke
    if not (is_packed(query) and is_packed(key) and is_packed(value)):
        return _sdpa_no("not all of query/key/value are packed")
    q_raw, k_raw, v_raw = query.raw_tensor, key.raw_tensor, value.raw_tensor
    if q_raw.inner_backend.name != "torch":
        return _sdpa_no("inner backend is not torch")
    if not k_raw.same_packing(v_raw):
        return _sdpa_no("key and value packings differ")
    if is_causal:
        # causal self-attention: query and key/value over the very same seqs
        if kv_spatial_dim != q_raw.orig_dims[-1] or not q_raw.same_packing(k_raw):
            return _sdpa_no("causal, but query and key/value packings differ")
    else:
        if kv_spatial_dim != k_raw.orig_dims[-1]:
            return _sdpa_no(f"kv axis {kv_spatial_dim} is not the innermost packed dim of key")
    if q_raw.orig_dims[:-1] != k_raw.orig_dims[:-1]:
        return _sdpa_no("query vs key/value outer packed dims differ")
    import torch

    if not hasattr(torch.nested, "nested_tensor_from_jagged"):
        return _sdpa_no("torch.nested.nested_tensor_from_jagged not available (torch too old)")
    # nested jagged needs the dense layout for the offsets; regap is cheap
    orig_layout = (q_raw.gap, q_raw.align)
    orig_layout_lens = q_raw.layout_lens
    if q_raw.has_gap_frames:
        query = regap(query, 0, align=1)
        q_raw = query.raw_tensor
    if k_raw.has_gap_frames:
        key = regap(key, 0, align=1)
        value = regap(value, 0, align=1)
        k_raw, v_raw = key.raw_tensor, value.raw_tensor
    q_inner, k_inner, v_inner = q_raw.inner, k_raw.inner, v_raw.inner
    head_dims = [d for d in q_inner.dims if d not in (q_raw.packed_dim, qk_feat_dim)]
    if len(head_dims) != 1 or head_dims[0].dimension is None:
        return _sdpa_no("expect exactly the [packed, heads, qk_feat] layout for query")
    if set(v_inner.dims) != {v_raw.packed_dim, head_dims[0], v_feat_dim}:
        return _sdpa_no("unexpected value dims layout")
    if set(k_inner.dims) != {k_raw.packed_dim, head_dims[0], qk_feat_dim}:
        return _sdpa_no("unexpected key dims layout")
    try:
        q_t = q_inner.copy_transpose([q_raw.packed_dim, head_dims[0], qk_feat_dim]).raw_tensor
        k_t = k_inner.copy_transpose([k_raw.packed_dim, head_dims[0], qk_feat_dim]).raw_tensor
        v_t = v_inner.copy_transpose([v_raw.packed_dim, head_dims[0], v_feat_dim]).raw_tensor
        q_t, k_t, v_t = _harmonize_qkv_dtypes(q_t, k_t, v_t)
        cu_q_key = _packing_cache_key("cu", q_raw, query.device)
        cu_q = _layout_cache.get(cu_q_key)
        if cu_q is None:
            cu_raw = q_raw.cu_seqlens(device=query.device)[0].raw_tensor
            cu_q = (cu_raw.long(), cu_raw.to(torch.int32))
            _layout_cache.set(cu_q_key, cu_q)
        cu_k_key = _packing_cache_key("cu", k_raw, query.device)
        cu_k = _layout_cache.get(cu_k_key)
        if cu_k is None:
            cu_raw = k_raw.cu_seqlens(device=query.device)[0].raw_tensor
            cu_k = (cu_raw.long(), cu_raw.to(torch.int32))
            _layout_cache.set(cu_k_key, cu_k)
        scale_ = scale if scale is not None else qk_feat_dim.dimension**-0.5
        out_t = None
        # Preferred: the flash varlen kernels directly (see _get_flash_varlen_fn):
        # fast flash backward, native dropout, no NJT subclass overhead.
        q_f, k_f, v_f = q_t, k_t, v_t
        # noinspection PyArgumentList
        if (
            q_f.device.type == "cuda"
            and q_f.dtype not in (torch.float16, torch.bfloat16)
            and hasattr(torch, "get_autocast_dtype")  # torch >= 2.4
            and torch.is_autocast_enabled("cuda")
        ):
            # Could be float32.
            # F.scaled_dot_product_attention casts internally under autocast;
            # do the same here for the direct flash call.
            amp_dtype = torch.get_autocast_dtype("cuda")
            if amp_dtype in (torch.float16, torch.bfloat16):
                q_f, k_f, v_f = q_t.to(amp_dtype), k_t.to(amp_dtype), v_t.to(amp_dtype)
        if not _flash_varlen_broken and q_f.device.type == "cuda" and q_f.dtype in (torch.float16, torch.bfloat16):
            if not hasattr(torch.ops.aten, "_flash_attention_forward"):
                _flash_varlen_broken = True
            else:
                max_q = int(q_raw.orig_dims[-1].get_dim_value())
                max_k = int(k_raw.orig_dims[-1].get_dim_value())
                # flash requires a contiguous last dim (transposed views are not)
                q_f = q_f if q_f.stride(-1) == 1 else q_f.contiguous()
                k_f = k_f if k_f.stride(-1) == 1 else k_f.contiguous()
                v_f = v_f if v_f.stride(-1) == 1 else v_f.contiguous()
                try:
                    # noinspection PyArgumentList
                    out_t = _get_flash_varlen_fn()(
                        q_f, k_f, v_f, cu_q[1], cu_k[1], max_q, max_k, dropout_p, is_causal, scale_
                    )
                    _count_attention_path("flash")
                except (RuntimeError, TypeError, NotImplementedError) as exc:
                    # e.g. aten signature drift in another torch version, or head-dim limits
                    _flash_varlen_broken = True
                    _warn_fallback_once(
                        "scaled_dot_product_attention",
                        f"direct flash varlen failed ({type(exc).__name__}: {exc})",
                        action="using the NJT SDPA path",
                    )
        if out_t is None:
            if not allow_njt:
                return None  # caller prefers flex over eager NJT (no dropout needed)
            needs_grad = torch.is_grad_enabled() and (q_t.requires_grad or k_t.requires_grad or v_t.requires_grad)
            sdpa_fn, sdpa_eager = _get_njt_sdpa_fn(q_t.device.type, needs_grad=needs_grad)
            sdpa_args = (q_t, k_t, v_t, cu_q[0], cu_k[0], dropout_p, is_causal, scale_)
            try:
                out_t = sdpa_fn(*sdpa_args)
                _count_attention_path("njt_compiled" if sdpa_fn is not sdpa_eager else "njt_eager")
            except (RuntimeError, NotImplementedError):
                # incl. dynamo/inductor compile errors (TorchDynamoException subclasses RuntimeError)
                if sdpa_fn is sdpa_eager:
                    raise
                _njt_sdpa_compile_failed = True  # compiled NJT SDPA failed; stick to eager from now on
                out_t = sdpa_eager(*sdpa_args)
                _count_attention_path("njt_eager")
    except (RuntimeError, NotImplementedError) as exc:
        if isinstance(exc, RuntimeError) and "CUDA" in str(exc):
            # environment-level (e.g. CUDA-built torch without a driver): do not retry every call
            _njt_sdpa_env_broken = True
        return _sdpa_no(f"torch nested-jagged SDPA failed here ({type(exc).__name__}: {exc})")
    out_inner = Tensor(
        "sdpa_varlen",
        dims=[q_raw.packed_dim, head_dims[0], v_feat_dim],
        # dtype from the raw result: under autocast, SDPA computes in bf16 even for f32 inputs
        dtype=q_raw.inner_backend.get_dtype_name_raw(out_t),
        feature_dim=v_feat_dim,
    )
    out_inner.raw_tensor = out_t
    out = q_raw.rewrap(out_inner, name="sdpa_varlen")
    if orig_layout != (0, 1) or orig_layout_lens is not None:
        out = regap(out, orig_layout[0], align=orig_layout[1], layout_lens=orig_layout_lens)
    return out


_flex_env_broken = False
_flex_attention_compiled = None
_flex_compile_failed = False

# Triton kernel options for the rel-pos flex attention,
# tuned on H100 (scripts: tune_flex_relpos.py): ~13% faster fwd+bwd than the defaults.
_FLEX_REL_POS_KERNEL_OPTIONS = {"num_warps": 4, "num_stages": 2}


def _flex_no(reason: str):
    """warn once that the FlexAttention rel-pos fast path is not used, and return None"""
    _warn_fallback_once(
        "rel_pos_self_attention",
        f"FlexAttention fast path not applicable ({reason})",
        action="using the generic implementation (with the packed op handling)",
    )
    return None


def _get_flex_attention_fn(device_type: str):
    """
    :param device_type: "cuda" / "cpu" / ...
    :return: (flex_attention fn to use, eager flex_attention fn).
        FlexAttention only gets fused kernels via torch.compile; eager is a slow reference impl,
        so on CUDA we lazily compile it (shared across calls).
    """
    global _flex_attention_compiled
    from torch.nn.attention.flex_attention import flex_attention

    if device_type != "cuda" or _flex_compile_failed:
        return flex_attention, flex_attention
    if _flex_attention_compiled is None:
        import torch

        _flex_attention_compiled = torch.compile(flex_attention)
    return _flex_attention_compiled, flex_attention


def _flex_seq_ids(raw: PackedRawTensor, spatial_dim: Dim, device) -> Optional[Tuple[Any, int]]:
    """
    :param raw: the packing
    :param spatial_dim: the innermost packed dim (carries the seq lens)
    :param device: torch device of the data
    :return: (seq_id, max_len):
        flat seq index per buffer frame ([total_buf] int64, on device),
        -1 on gap/alignment frames;
        and the max seq len.
        Cached per packing (see :data:`_layout_cache`). None if not derivable.
    """
    import torch

    key = _packing_cache_key("flex_seq_ids", raw, device)
    hit = _layout_cache.get(key)
    if hit is not None:
        return hit
    if spatial_dim.dyn_size_ext is None:
        return None
    lens_rf = spatial_dim.dyn_size_ext
    for d in raw.orig_dims[:-1]:
        if d not in lens_rf.dims:
            lens_rf = rf.expand_dim(lens_rf, dim=d)
    if len(raw.orig_dims) > 2:
        lens_rf, _ = rf.merge_dims(lens_rf, dims=raw.orig_dims[:-1])
    # max over the cpu-side dyn sizes (no gpu sync)
    max_len = int(lens_rf.raw_tensor.max()) if lens_rf.raw_tensor.numel() > 0 else 0
    lens = rf.copy_to_device(lens_rf, str(device)).raw_tensor.long().flatten()
    starts_rf, _ = _seq_starts_math(raw.orig_dims, raw.gap, raw.align, layout_lens=raw.layout_lens)
    if starts_rf is None:  # single seq
        starts = torch.zeros(1, dtype=torch.int64, device=device)
    else:
        starts = rf.copy_to_device(starts_rf, str(device)).raw_tensor.long().flatten()
    total_buf = int(raw.packed_dim.get_dim_value())
    pos = torch.arange(total_buf, device=device)
    seq = torch.searchsorted(starts, pos, right=True) - 1
    local = pos - starts[seq]
    seq_id = torch.where(local < lens[seq], seq, torch.full_like(seq, -1))
    _layout_cache.set(key, (seq_id, max_len))
    return seq_id, max_len


def _flex_doc_block_mask(q_raw: PackedRawTensor, kv_raw: PackedRawTensor, q_ids, kv_ids, *, is_causal: bool, device):
    """
    :param q_raw: query packing (cache key part)
    :param kv_raw: key/value packing (cache key part)
    :param q_ids: seq_id per query buffer frame, see :func:`_flex_seq_ids`
    :param kv_ids: seq_id per key/value buffer frame
    :param is_causal: also mask kj > qi. Requires the same packing for q and kv
        (then the global buffer order equals the local order within a seq).
    :param device: torch device
    :return: flex BlockMask for the document mask
        (same-seq pairs only; gap/alignment frames fully masked out),
        cached per packing pair (see :data:`_layout_cache`).
    """
    q_key = _packing_cache_key("q", q_raw, device)
    kv_key = _packing_cache_key("kv", kv_raw, device)
    key = ("flex_doc_block_mask", q_key, kv_key, is_causal)
    hit = _layout_cache.get(key)
    if hit is not None:
        return hit
    from torch.nn.attention.flex_attention import create_block_mask

    def _mask_mod(_b, _h, qi, kj):
        ok = (kv_ids[kj] >= 0) & (q_ids[qi] == kv_ids[kj])
        if is_causal:
            ok = ok & (qi >= kj)
        return ok

    block_mask = create_block_mask(
        _mask_mod, B=None, H=None, Q_LEN=q_ids.numel(), KV_LEN=kv_ids.numel(), device=str(device)
    )
    _layout_cache.set(key, block_mask)
    return block_mask


def _flex_doc_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    qk_feat_dim: Dim,
    v_feat_dim: Dim,
    kv_spatial_dim: Dim,
    query_spatial_dim: Dim,
    is_causal: bool,
    scale: Optional[float],
) -> Optional[Tensor]:
    """
    Plain (dot) attention over packed tensors via torch FlexAttention:
    one flat attention over the packed buffers with a document block mask
    (self-, cross- and causal attention; the gapped layout is used as-is).
    Preferred over the NJT SDPA path when there is no dropout:
    compiled flex has none of the eager NJT per-call overhead,
    and its autograd works under torch.compile
    (compiled NJT backward is broken in torch 2.7, AOT autograd tangent meta assert).
    None if not applicable (then the NJT / generic paths run).

    :param query: {..., query_spatial_dim, qk_feat_dim}. not yet scaled.
    :param key: {..., kv_spatial_dim, qk_feat_dim}
    :param value: {..., kv_spatial_dim, v_feat_dim}
    :param qk_feat_dim: embedding dimension of key and query
    :param v_feat_dim: embedding dimension of value
    :param kv_spatial_dim: spatial axis of key/value to attend over
    :param query_spatial_dim: spatial axis of query
    :param is_causal: causal masking (requires the same packing for q and kv)
    :param scale: scaling factor applied prior to softmax (default: qk_feat_dim**-0.5)
    :return: attention output (packed like query), or None
    """
    global _flex_env_broken, _flex_compile_failed

    if _flex_env_broken:
        return None
    if not (is_packed(query) and is_packed(key) and is_packed(value)):
        return None
    q_raw, k_raw, v_raw = query.raw_tensor, key.raw_tensor, value.raw_tensor
    if q_raw.inner_backend.name != "torch":
        return None
    if not k_raw.same_packing(v_raw):
        return None
    if kv_spatial_dim != k_raw.orig_dims[-1] or query_spatial_dim != q_raw.orig_dims[-1]:
        return None
    if q_raw.orig_dims[:-1] != k_raw.orig_dims[:-1]:
        return None
    if is_causal and not (q_raw.same_packing(k_raw) and _same_seq_lens(query_spatial_dim, kv_spatial_dim)):
        return None
    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa
    except ImportError:
        return None
    q_inner, k_inner, v_inner = q_raw.inner, k_raw.inner, v_raw.inner
    head_dims = [d for d in q_inner.dims if d not in (q_raw.packed_dim, qk_feat_dim)]
    if len(head_dims) != 1 or head_dims[0].dimension is None:
        return None
    heads_dim = head_dims[0]
    if set(k_inner.dims) != {k_raw.packed_dim, heads_dim, qk_feat_dim}:
        return None
    if set(v_inner.dims) != {v_raw.packed_dim, heads_dim, v_feat_dim}:
        return None

    try:
        q_t = q_inner.copy_transpose([q_raw.packed_dim, heads_dim, qk_feat_dim]).raw_tensor
        k_t = k_inner.copy_transpose([k_raw.packed_dim, heads_dim, qk_feat_dim]).raw_tensor
        v_t = v_inner.copy_transpose([v_raw.packed_dim, heads_dim, v_feat_dim]).raw_tensor
        q_t, k_t, v_t = _harmonize_qkv_dtypes(q_t, k_t, v_t)
        q_ids = _flex_seq_ids(q_raw, query_spatial_dim, q_t.device)
        kv_ids = _flex_seq_ids(k_raw, kv_spatial_dim, q_t.device)
        if q_ids is None or kv_ids is None:
            return None
        block_mask = _flex_doc_block_mask(q_raw, k_raw, q_ids[0], kv_ids[0], is_causal=is_causal, device=q_t.device)
        flex_fn, flex_eager = _get_flex_attention_fn(q_t.device.type)
        q_f = q_t.transpose(0, 1).unsqueeze(0)  # [1, H, total, D]
        k_f = k_t.transpose(0, 1).unsqueeze(0)
        v_f = v_t.transpose(0, 1).unsqueeze(0)
        flex_kwargs = dict(block_mask=block_mask, scale=scale if scale is not None else qk_feat_dim.dimension**-0.5)
        try:
            out_f = flex_fn(q_f, k_f, v_f, **flex_kwargs)
        except (RuntimeError, NotImplementedError):
            # incl. dynamo/inductor compile errors (TorchDynamoException subclasses RuntimeError)
            if flex_fn is flex_eager:
                raise
            _flex_compile_failed = True  # compiled FlexAttention failed; stick to eager from now on
            out_f = flex_eager(q_f, k_f, v_f, **flex_kwargs)
        out_t = out_f.squeeze(0).transpose(0, 1)  # [total, H, Dv]
    except (RuntimeError, NotImplementedError) as exc:
        if isinstance(exc, RuntimeError) and "CUDA" in str(exc):
            # environment-level (e.g. CUDA-built torch without a driver): do not retry every call
            _flex_env_broken = True
        return None
    out_inner = Tensor(
        "flex_doc_att",
        dims=[q_raw.packed_dim, heads_dim, v_feat_dim],
        # dtype from the raw result, see _sdpa_varlen_attention
        dtype=q_raw.inner_backend.get_dtype_name_raw(out_t),
        feature_dim=v_feat_dim,
    )
    out_inner.raw_tensor = out_t.contiguous()
    _count_attention_path("flex_doc")
    return q_raw.rewrap(out_inner, name="flex_doc_att")


def _flex_rel_pos_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    pos_emb: Tensor,
    *,
    pos_bias_u: Optional[Tensor],
    pos_bias_v: Optional[Tensor],
    v_feat_dim: Dim,
    qk_feat_dim: Dim,
    kv_spatial_dim: Dim,
    query_spatial_dim: Dim,
    pos_emb_spatial_dim: Dim,
) -> Optional[Tensor]:
    """
    Packed self-attention with relative positional encoding via torch FlexAttention:
    one flat attention over the packed buffer with a document block mask (seq membership)
    and a score_mod adding the (precomputed, packed) position-based term (matrix b+d),
    indexed by the relative position --
    within a seq the global offset equals the local one (the seq starts cancel),
    so no per-seq local positions are needed.
    Pure torch, no external package. None if not applicable (then the generic path runs).

    :param query: {..., query_spatial_dim, qk_feat_dim}. not yet scaled.
    :param key: {..., kv_spatial_dim, qk_feat_dim}
    :param value: {..., kv_spatial_dim, v_feat_dim}
    :param pos_emb: {..., pos_emb_spatial_dim, qk_feat_dim}, relative positional encoding (dense)
    :param pos_bias_u: {..., qk_feat_dim}, added to query for the content-based term (matrix a+c)
    :param pos_bias_v: {..., qk_feat_dim}, added to query for the position-based term (matrix b+d)
    :param v_feat_dim: embedding dimension of value
    :param qk_feat_dim: embedding dimension of key and query
    :param kv_spatial_dim: spatial axis of key/value to attend over
    :param query_spatial_dim: spatial axis of query
    :param pos_emb_spatial_dim: relative-position axis of pos_emb (must be 2*max_len-1, centered)
    :return: attention output (packed like query), or None
    """
    global _flex_env_broken, _flex_compile_failed

    if _flex_env_broken:
        return None  # already warned once when it broke
    if not (is_packed(query) and is_packed(key) and is_packed(value)):
        return _flex_no("not all of query/key/value are packed")
    if is_packed(pos_emb):
        return _flex_no("pos_emb is packed (expected dense)")
    q_raw, k_raw, v_raw = query.raw_tensor, key.raw_tensor, value.raw_tensor
    if q_raw.inner_backend.name != "torch":
        return _flex_no("inner backend is not torch")
    if not k_raw.same_packing(v_raw):
        return _flex_no("key and value packings differ")
    if kv_spatial_dim != k_raw.orig_dims[-1] or query_spatial_dim != q_raw.orig_dims[-1]:
        return _flex_no("spatial dims are not the innermost packed dims")
    if q_raw.orig_dims[:-1] != k_raw.orig_dims[:-1] or not _same_seq_lens(query_spatial_dim, kv_spatial_dim):
        return _flex_no("query vs key/value packings differ (not self-attention seqs)")
    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa  # only check availability
    except ImportError:
        return _flex_no("torch.nn.attention.flex_attention not available (torch too old)")

    # The position-based term (matrix b+d, pre-shift), computed packed
    # (no packed-dim reduction, so these stay inner ops): {packed.., heads?, pos_emb_spatial}
    q_with_bias_u = (query + pos_bias_u) if pos_bias_u is not None else query
    q_with_bias_v = (query + pos_bias_v) if pos_bias_v is not None else query
    matrix_bd = rf.matmul(q_with_bias_v, pos_emb, reduce=qk_feat_dim)
    if not is_packed(q_with_bias_u) or not is_packed(matrix_bd):
        return _flex_no("query with bias / matrix b+d did not stay packed")
    if (q_raw.gap, q_raw.align) != (k_raw.gap, k_raw.align) or q_raw.layout_lens is not k_raw.layout_lens:
        return _flex_no("query vs key/value layout (gap/align/layout_lens) differs")
    qu_raw, bd_raw = q_with_bias_u.raw_tensor, matrix_bd.raw_tensor
    q_inner, k_inner, v_inner, bd_inner = qu_raw.inner, k_raw.inner, v_raw.inner, bd_raw.inner
    head_dims = [d for d in q_inner.dims if d not in (qu_raw.packed_dim, qk_feat_dim)]
    if len(head_dims) != 1 or head_dims[0].dimension is None:
        return _flex_no("expect exactly the [packed, heads, qk_feat] layout for query")
    heads_dim = head_dims[0]
    if set(k_inner.dims) != {k_raw.packed_dim, heads_dim, qk_feat_dim}:
        return _flex_no("unexpected key dims layout")
    if set(v_inner.dims) != {v_raw.packed_dim, heads_dim, v_feat_dim}:
        return _flex_no("unexpected value dims layout")
    if set(bd_inner.dims) - {heads_dim} != {bd_raw.packed_dim, pos_emb_spatial_dim}:
        return _flex_no("unexpected matrix b+d dims layout")
    import torch

    try:
        q_t = q_inner.copy_transpose([qu_raw.packed_dim, heads_dim, qk_feat_dim]).raw_tensor
        k_t = k_inner.copy_transpose([k_raw.packed_dim, heads_dim, qk_feat_dim]).raw_tensor
        v_t = v_inner.copy_transpose([v_raw.packed_dim, heads_dim, v_feat_dim]).raw_tensor
        q_t, k_t, v_t = _harmonize_qkv_dtypes(q_t, k_t, v_t)
        # the (possibly gapped) layout is used as-is:
        # gap/alignment frames get seq_id -1 and are masked out, no re-layout needed
        ids = _flex_seq_ids(qu_raw, query_spatial_dim, q_t.device)
        if ids is None:
            return _flex_no("cannot derive the seq ids for the packing")
        seq_id, max_len = ids
        block_mask = _flex_doc_block_mask(qu_raw, qu_raw, seq_id, seq_id, is_causal=False, device=q_t.device)
        r_size = int(pos_emb_spatial_dim.get_dim_value())
        if r_size != 2 * max_len - 1:
            # only the standard centered layout (positions -(max_len-1)..(max_len-1)) is supported
            return _flex_no(f"pos_emb size {r_size} != 2*max_len-1 ({2 * max_len - 1})")
        center = max_len - 1
        if heads_dim in bd_inner.dims:
            bd_t = bd_inner.copy_transpose([bd_raw.packed_dim, heads_dim, pos_emb_spatial_dim]).raw_tensor
        else:
            bd_t = bd_inner.copy_transpose([bd_raw.packed_dim, pos_emb_spatial_dim]).raw_tensor
            bd_t = bd_t.unsqueeze(1).expand(-1, heads_dim.dimension, -1)
        # pre-scale the bias like the content-based term (flex applies its scale to q*k before score_mod)
        bd_t = (bd_t * qk_feat_dim.dimension**-0.5).contiguous()

        def _score_mod(score, _b, h, qi, kj):
            # Relative position of key w.r.t. query (kj - qi),
            # like the shift trick: bd[i, j] = bd_pre_shift[i, center + (j - i)].
            # Within a seq the global kj - qi equals the local j - i.
            # clamp: mask_mod-masked positions inside partial blocks may still evaluate this.
            rel = torch.clamp(kj - qi + center, 0, r_size - 1)
            return score + bd_t[qi, h, rel]

        flex_fn, flex_eager = _get_flex_attention_fn(q_t.device.type)
        q_f = q_t.transpose(0, 1).unsqueeze(0)  # [1, H, total, D]
        k_f = k_t.transpose(0, 1).unsqueeze(0)
        v_f = v_t.transpose(0, 1).unsqueeze(0)
        flex_kwargs = dict(score_mod=_score_mod, block_mask=block_mask)
        if q_t.device.type == "cuda":
            flex_kwargs["kernel_options"] = _FLEX_REL_POS_KERNEL_OPTIONS
        try:
            out_f = flex_fn(q_f, k_f, v_f, **flex_kwargs)
        except (RuntimeError, NotImplementedError):
            # incl. dynamo/inductor compile errors (TorchDynamoException subclasses RuntimeError)
            if flex_fn is flex_eager:
                raise
            _flex_compile_failed = True  # compiled FlexAttention failed; stick to eager from now on
            out_f = flex_eager(q_f, k_f, v_f, **flex_kwargs)
        out_t = out_f.squeeze(0).transpose(0, 1)  # [total, H, Dv]
    except (RuntimeError, NotImplementedError) as exc:
        if isinstance(exc, RuntimeError) and "CUDA" in str(exc):
            # environment-level (e.g. CUDA-built torch without a driver): do not retry every call
            _flex_env_broken = True
        return _flex_no(f"torch FlexAttention failed here ({type(exc).__name__}: {exc})")
    out_inner = Tensor(
        "rel_pos_att_flex",
        dims=[qu_raw.packed_dim, heads_dim, v_feat_dim],
        # dtype from the raw result, see _sdpa_varlen_attention
        dtype=qu_raw.inner_backend.get_dtype_name_raw(out_t),
        feature_dim=v_feat_dim,
    )
    out_inner.raw_tensor = out_t.contiguous()
    _count_attention_path("rel_pos_flex")
    return qu_raw.rewrap(out_inner, name="rel_pos_att_flex")


def _triton_rel_pos_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    pos_emb: Tensor,
    *,
    pos_bias_u: Optional[Tensor],
    pos_bias_v: Optional[Tensor],
    att_dropout: float,
    v_feat_dim: Dim,
    qk_feat_dim: Dim,
    kv_spatial_dim: Dim,
    query_spatial_dim: Dim,
    pos_emb_spatial_dim: Dim,
) -> Optional[Tensor]:
    """
    Packed rel-pos self-attention with post-softmax weight dropout
    via :mod:`returnn.torch.util.rel_pos_att_triton`
    (flash: no bias; FlexAttention: no dropout).
    CUDA only; any layout as-is (explicit per-seq starts + lens).
    None if not applicable (then the per-seq / once-unpack fallbacks run).
    """
    if not (is_packed(query) and is_packed(key) and is_packed(value)) or is_packed(pos_emb):
        return None
    q_raw, k_raw, v_raw = query.raw_tensor, key.raw_tensor, value.raw_tensor
    if q_raw.inner_backend.name != "torch" or query.device is None or not str(query.device).startswith("cuda"):
        return None
    try:
        from returnn.torch.util import rel_pos_att_triton
    except ImportError:
        return None
    if not rel_pos_att_triton.is_available():
        return None
    if not k_raw.same_packing(v_raw):
        return None
    if kv_spatial_dim != k_raw.orig_dims[-1] or query_spatial_dim != q_raw.orig_dims[-1]:
        return None
    if q_raw.orig_dims[:-1] != k_raw.orig_dims[:-1] or not _same_seq_lens(query_spatial_dim, kv_spatial_dim):
        return None
    if (q_raw.gap, q_raw.align) != (k_raw.gap, k_raw.align) or q_raw.layout_lens is not k_raw.layout_lens:
        return None
    if len(q_raw.orig_dims) != 2:
        return None
    # any layout as-is (per-seq starts + lens, junk frames never touched):
    # no re-layout, the output shares the input packing.
    # the position-based term (matrix b+d, pre-shift), computed packed:
    q_with_bias_u = (query + pos_bias_u) if pos_bias_u is not None else query
    q_with_bias_v = (query + pos_bias_v) if pos_bias_v is not None else query
    matrix_bd = rf.matmul(q_with_bias_v, pos_emb, reduce=qk_feat_dim)
    if not is_packed(q_with_bias_u) or not is_packed(matrix_bd):
        return None
    qu_raw, bd_raw = q_with_bias_u.raw_tensor, matrix_bd.raw_tensor
    q_inner, k_inner, v_inner, bd_inner = qu_raw.inner, k_raw.inner, v_raw.inner, bd_raw.inner
    head_dims = [d for d in q_inner.dims if d not in (qu_raw.packed_dim, qk_feat_dim)]
    if len(head_dims) != 1 or head_dims[0].dimension is None:
        return None
    heads_dim = head_dims[0]
    if set(k_inner.dims) != {k_raw.packed_dim, heads_dim, qk_feat_dim}:
        return None
    if set(v_inner.dims) != {v_raw.packed_dim, heads_dim, v_feat_dim}:
        return None
    if set(bd_inner.dims) - {heads_dim} != {bd_raw.packed_dim, pos_emb_spatial_dim}:
        return None
    max_len = int(query_spatial_dim.get_dim_value())
    r_size = int(pos_emb_spatial_dim.get_dim_value())
    if r_size != 2 * max_len - 1:
        return None  # only the standard centered layout
    q_t = q_inner.copy_transpose([qu_raw.packed_dim, heads_dim, qk_feat_dim]).raw_tensor
    k_t = k_inner.copy_transpose([k_raw.packed_dim, heads_dim, qk_feat_dim]).raw_tensor
    v_t = v_inner.copy_transpose([v_raw.packed_dim, heads_dim, v_feat_dim]).raw_tensor
    q_t, k_t, v_t = _harmonize_qkv_dtypes(q_t, k_t, v_t)
    import torch

    # noinspection PyArgumentList
    if (
        q_t.dtype not in (torch.float16, torch.bfloat16)
        and hasattr(torch, "get_autocast_dtype")  # torch >= 2.4
        and torch.is_autocast_enabled("cuda")
    ):
        # like the flash prep: run in the autocast dtype (the RF raws are often still f32)
        amp_dtype = torch.get_autocast_dtype("cuda")
        if amp_dtype in (torch.float16, torch.bfloat16):
            q_t, k_t, v_t = q_t.to(amp_dtype), k_t.to(amp_dtype), v_t.to(amp_dtype)
    if heads_dim in bd_inner.dims:
        bd_t = bd_inner.copy_transpose([bd_raw.packed_dim, heads_dim, pos_emb_spatial_dim]).raw_tensor
    else:
        bd_t = bd_inner.copy_transpose([bd_raw.packed_dim, pos_emb_spatial_dim]).raw_tensor
        bd_t = bd_t.unsqueeze(1).expand(-1, heads_dim.dimension, -1)
    # pre-scale the bias like the content-based term (the kernel scales only q k^T)
    bd_t = (bd_t * qk_feat_dim.dimension**-0.5).to(q_t.dtype)
    starts_key = _packing_cache_key("triton_starts_lens", q_raw, query.device)
    hit = _layout_cache.get(starts_key)
    if hit is not None:
        starts, lens = hit
    else:
        starts_rf, _ = _seq_starts_math(q_raw.orig_dims, q_raw.gap, q_raw.align, layout_lens=q_raw.layout_lens)
        if starts_rf is None:
            return None
        starts = rf.copy_to_device(starts_rf, query.device).raw_tensor.int().flatten()
        lens = rf.copy_to_device(query_spatial_dim.dyn_size_ext, query.device).raw_tensor.int().flatten()
        _layout_cache.set(starts_key, (starts, lens))
    try:
        out_t = rel_pos_att_triton.rel_pos_att_varlen(
            q_t, k_t, v_t, bd_t, starts, lens, max_len, dropout_p=att_dropout, scale=qk_feat_dim.dimension**-0.5
        )
    except (RuntimeError, NotImplementedError) as exc:
        _warn_fallback_once(
            "rel_pos_self_attention",
            f"Triton varlen kernel failed here ({type(exc).__name__}: {exc})",
            action="using the per-seq / once-unpack fallback",
        )
        return None
    out_inner = Tensor(
        "rel_pos_att_triton",
        dims=[qu_raw.packed_dim, heads_dim, v_feat_dim],
        dtype=qu_raw.inner_backend.get_dtype_name_raw(out_t),
        feature_dim=v_feat_dim,
    )
    out_inner.raw_tensor = out_t
    _count_attention_path("rel_pos_triton")
    return qu_raw.rewrap(out_inner, name="rel_pos_att_triton")


def _rel_pos_attention_per_seq(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    pos_emb: Tensor,
    *,
    pos_bias_u: Optional[Tensor],
    pos_bias_v: Optional[Tensor],
    att_dropout: float,
    att_dropout_broadcast: bool,
    v_feat_dim: Dim,
    qk_feat_dim: Dim,
    kv_spatial_dim: Dim,
    query_spatial_dim: Dim,
    pos_emb_spatial_dim: Dim,
) -> Optional[Tensor]:
    """
    Per-sequence rel-pos self-attention on the packed buffer, no unpack:
    every sequence is a contiguous slice of the dense packed buffer,
    so the standard attention runs per seq (static per-seq dims, incl. the centered
    pos_emb window of size 2*len-1) and the outputs are concatenated back.
    Exact semantics incl. real (weight) dropout; memory sum(len^2) instead of B*T_max^2.
    The simple CPU path -- the Triton kernel is the CUDA fast path (planned).
    None if not applicable (then the once-unpack fallback runs).
    """
    if not (is_packed(query) and is_packed(key) and is_packed(value)) or is_packed(pos_emb):
        return None
    q_raw, k_raw, v_raw = query.raw_tensor, key.raw_tensor, value.raw_tensor
    # query/key/value may each carry their own packed dim; equivalent layouts suffice
    # (one shared lineage in the attention module, but they can also be packed independently)
    if k_raw.orig_dims != v_raw.orig_dims or (k_raw.gap, k_raw.align) != (v_raw.gap, v_raw.align):
        return None
    if k_raw.layout_lens is not v_raw.layout_lens:
        return None
    if len(q_raw.orig_dims) != 2 or q_raw.orig_dims[-1] != query_spatial_dim:
        return None
    # kv is packed alike but over its own (copied) spatial dim, cf. _flex_rel_pos_attention
    if kv_spatial_dim != k_raw.orig_dims[-1] or q_raw.orig_dims[:-1] != k_raw.orig_dims[:-1]:
        return None
    if not _same_seq_lens(query_spatial_dim, kv_spatial_dim):
        return None
    if (q_raw.gap, q_raw.align) != (k_raw.gap, k_raw.align) or q_raw.layout_lens is not k_raw.layout_lens:
        return None
    lens_t = query_spatial_dim.dyn_size_ext
    if lens_t is None or lens_t.dims != (q_raw.orig_dims[0],):
        return None
    orig_layout = (q_raw.gap, q_raw.align)
    orig_layout_lens = q_raw.layout_lens
    if q_raw.has_gap_frames:
        query, key, value = regap(query, 0, align=1), regap(key, 0, align=1), regap(value, 0, align=1)
        q_raw = query.raw_tensor
    lens = [int(v) for v in lens_t.raw_tensor.flatten()]
    max_len = max(lens) if lens else 0
    if int(pos_emb_spatial_dim.get_dim_value()) != 2 * max_len - 1:
        return None  # only the standard centered layout is supported
    center = max_len - 1
    outs = []
    off = 0
    for seq_len in lens:
        q_b, q_time = rf.slice(q_raw.inner, axis=q_raw.packed_dim, start=off, end=off + seq_len)
        kv_packed = key.raw_tensor.packed_dim  # kv has its own packed dim (copied spatial dim)
        k_b, kv_time = rf.slice(key.raw_tensor.inner, axis=kv_packed, start=off, end=off + seq_len)
        v_b = rf.slice(
            value.raw_tensor.inner, axis=value.raw_tensor.packed_dim, start=off, end=off + seq_len, out_dim=kv_time
        )[0]
        pos_b, pos_dim = rf.slice(pos_emb, axis=pos_emb_spatial_dim, start=center - (seq_len - 1), end=center + seq_len)
        out_b = Backend.rel_pos_self_attention(
            q_b,
            k_b,
            v_b,
            pos_b,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v,
            att_dropout=att_dropout,
            att_dropout_broadcast=att_dropout_broadcast,
            v_feat_dim=v_feat_dim,
            qk_feat_dim=qk_feat_dim,
            kv_spatial_dim=kv_time,
            query_spatial_dim=q_time,
            pos_emb_spatial_dim=pos_dim,
        )
        outs.append((out_b, q_time))
        off += seq_len
    inner_new, cat_dim = rf.concat(*outs)
    inner_new, _ = rf.replace_dim(inner_new, in_dim=cat_dim, out_dim=q_raw.packed_dim)
    helper = PackedRawTensor(inner=inner_new, packed_dim=q_raw.packed_dim, orig_dims=q_raw.orig_dims, gap=0, align=1)
    out = helper.rewrap(inner_new, name="rel_pos_att_per_seq")
    _count_attention_path("rel_pos_per_seq")
    if orig_layout != (0, 1) or orig_layout_lens is not None:
        out = regap(out, orig_layout[0], align=orig_layout[1], layout_lens=orig_layout_lens)
    return out


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
    footprints, seqs_dim = _seq_footprints(raw.orig_dims, raw.gap, raw.align, layout_lens=raw.layout_lens)
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
    if residual0 >= 0 and bool(rf.reduce_all(residual == residual0, axis=list(residual.dims)).raw_tensor):
        # uniform residual: expressible as the plain (lens, gap, align) form
        helper = PackedRawTensor(
            inner=out_inner,
            packed_dim=out_packed_dim,
            orig_dims=tuple(raw.orig_dims[:-1]) + (out_time,),
            gap=residual0,
            align=align_out,
        )
        return helper.rewrap(out_inner, name="conv")
    gap_out = raw.gap // st
    if len(raw.orig_dims) != 2 or not bool(rf.reduce_all(residual >= gap_out, axis=list(residual.dims)).raw_tensor):
        return None
    # Non-uniform residuals (mixed seq-len residuals mod stride):
    # exact via per-seq layout lens, layout_len + gap_out = footprint // st. No re-layout needed.
    helper = PackedRawTensor(
        inner=out_inner,
        packed_dim=out_packed_dim,
        orig_dims=tuple(raw.orig_dims[:-1]) + (out_time,),
        gap=gap_out,
        align=align_out,
        layout_lens=footprints // st - gap_out,
    )
    return helper.rewrap(out_inner, name="conv")


def _extract_strided(raw: PackedRawTensor, out_inner: Tensor, out_packed_dim: Dim, out_time: Dim, st: int) -> Tensor:
    """
    Re-layout the output of a stride-st op over the packed dim
    into the closed-form (out lens, gap // st, max(align // st, 1)) layout,
    for the case that the raw strided output is not expressible in the closed form
    (the per-seq residuals differ, e.g. mixed seq-len parities at stride 2;
    see :func:`_strided_out_wrapper`).
    The raw strided output has per-seq starts old_start_i // st;
    one gather over the frames moves it into the closed form
    (only int coordinate grids as intermediates, no padded feature-sized data),
    so the result STAYS PACKED and downstream computation continues packed.
    """
    out_orig = tuple(raw.orig_dims[:-1]) + (out_time,)
    others = tuple(out_orig[:-1])
    gap_out = raw.gap // st
    align_out = max(raw.align // st, 1)
    new_dim = Dim(_packed_total(out_orig, gap_out, align_out), name="packed_strided")
    new_pos = _padded_positions(out_orig, gap_out, align_out)
    # per new-buffer frame: its local time coord, and its (flat) seq index
    grid_t = rf.range_over_dim(out_time)
    for d in others:
        if d not in grid_t.dims:
            grid_t = rf.expand_dim(grid_t, dim=d)  # broadcast view, ints only
    t_coords = rf.scatter(grid_t, indices=new_pos, indices_dim=list(out_orig), out_dim=new_dim, use_mask=True)
    src = t_coords
    old_starts, seqs_dim = _seq_starts_math(raw.orig_dims, raw.gap, raw.align, layout_lens=raw.layout_lens)
    if old_starts is not None:
        seg_grid = None
        for d in others:
            coords = rf.range_over_dim(d)
            for o in out_orig:
                if o not in coords.dims:
                    coords = rf.expand_dim(coords, dim=o)
            seg_grid = coords if seg_grid is None else seg_grid * d.get_dim_value_tensor() + coords
        seg = rf.scatter(seg_grid, indices=new_pos, indices_dim=list(out_orig), out_dim=new_dim, use_mask=True)
        old_starts = old_starts // st
        if old_starts.device != seg.device:
            old_starts = rf.copy_to_device(old_starts, seg.device)
        src = rf.gather(old_starts, indices=seg, axis=seqs_dim, clip_to_valid=True) + t_coords
    new_inner = rf.gather(out_inner, indices=src, axis=out_packed_dim, clip_to_valid=True)
    helper = PackedRawTensor(inner=new_inner, packed_dim=new_dim, orig_dims=out_orig, gap=gap_out, align=align_out)
    return helper.rewrap(new_inner, name="conv")


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
    def convert_to_tensor(value, *, dims, dtype, sparse_dim=None, feature_dim=None, device=None, name=None) -> Tensor:
        """
        Delegate to the inner backend: a converted constant (numpy array, scalar, raw tensor) is plain,
        not packed. Reached e.g. when a packed input resolves the backend to this one at a creation site,
        such as the mel filterbank matrix in :func:`rf.audio.mel_filterbank`.
        """
        return global_backend.convert_to_tensor(
            value, dims=dims, dtype=dtype, sparse_dim=sparse_dim, feature_dim=feature_dim, device=device, name=name
        )

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
            layout_lens=raw_tensor.layout_lens,
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
        layout_lens = raw.layout_lens
        if layout_lens is not None and in_dim in layout_lens.dims:
            layout_lens, _ = rf.replace_dim(layout_lens, in_dim=in_dim, out_dim=out_dim)
        if in_dim in raw.orig_dims or in_dim == raw.packed_dim:
            out.raw_tensor = PackedRawTensor(
                inner=raw.inner,
                packed_dim=raw.packed_dim,
                orig_dims=tuple(out_dim if d == in_dim else d for d in raw.orig_dims),
                dims=tuple(out_dim if d == in_dim else d for d in raw.dims),
                gap=raw.gap,
                align=raw.align,
                layout_lens=layout_lens,
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
                layout_lens=layout_lens,
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
        out_raw = raw_tensor.inner_backend.activation_raw(raw_tensor.inner.raw_tensor, func)
        # dtype from the raw result: e.g. square (aten::pow) is on the autocast fp32 list
        inner_out.dtype = raw_tensor.inner_backend.get_dtype_name_raw(out_raw)
        inner_out.raw_tensor = out_raw
        return PackedRawTensor(
            inner=inner_out,
            packed_dim=raw_tensor.packed_dim,
            orig_dims=raw_tensor.orig_dims,
            gap=raw_tensor.gap,
            align=raw_tensor.align,
            layout_lens=raw_tensor.layout_lens,
        )

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
        """
        stft on packed audio. Same idea as the packed conv (:func:`PackedBackend.conv`):
        one inner stft runs over the whole packed buffer as a single long sequence,
        then the strided output (stride = frame_step) is re-layouted into the packed form.
        The framing is valid (no cross-seq padding); every real window lies fully inside its
        own sequence, while the boundary-crossing windows are junk and land in the output gap
        slots (sliced off by the re-layout).
        Requires stride | align (frame_step divides the align, so the seq starts stay on the
        output grid), as for the strided conv. Falls back to the generic (unpack) handling otherwise.
        """
        raw = x.raw_tensor
        opts = dict(
            frame_step=frame_step,
            frame_length=frame_length,
            fft_length=fft_length,
            window_use_frame_length=window_use_frame_length,
            align_window_left=align_window_left,
            window_enforce_even=window_enforce_even,
        )
        st = frame_step
        if (
            isinstance(raw, PackedRawTensor)
            and len(raw.orig_dims) == 2
            and raw.orig_dims[-1] == in_spatial_dim
            and raw.align % st == 0
            and out_spatial_dim.dyn_size_ext is not None
        ):
            out_inner, out_sp, _ = rf.stft(raw.inner, in_spatial_dim=raw.packed_dim, out_dim=out_dim, **opts)
            out = _strided_out_wrapper(raw, out_inner, out_sp, out_spatial_dim, st)
            if out is not None:
                return out
            _warn_fallback_once(
                "stft:strided-out",
                "strided stft output layout not expressible in the (lens, gap, align) form",
                action="re-layouting into the closed form, one extra gather (stft stays packed)",
            )
            return _extract_strided(raw, out_inner, out_sp, out_spatial_dim, st)
        return _dim_aware_call(
            "stft", (x,), dict(in_spatial_dim=in_spatial_dim, out_spatial_dim=out_spatial_dim, out_dim=out_dim, **opts)
        )

    @staticmethod
    def cast_raw(raw_tensor: PackedRawTensor, dtype: str) -> PackedRawTensor:
        """cast -- on the packed data (elementwise)"""
        inner_out = raw_tensor.inner.copy_template(name="cast")
        inner_out.dtype = dtype
        inner_out.raw_tensor = raw_tensor.inner_backend.cast_raw(raw_tensor.inner.raw_tensor, dtype)
        return PackedRawTensor(
            inner=inner_out,
            packed_dim=raw_tensor.packed_dim,
            orig_dims=raw_tensor.orig_dims,
            gap=raw_tensor.gap,
            align=raw_tensor.align,
            layout_lens=raw_tensor.layout_lens,
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
            a_raw = a.raw_tensor
            b = _conform_packing(b, a_raw)  # same seqs, different layout -> conform to a's packing
            b_raw = b.raw_tensor
            if a_raw.same_packing(b_raw):
                out = a_raw.rewrap(
                    rf.combine(a_raw.inner, kind, b_raw.inner, allow_broadcast_all_sources=True), name=kind
                )
                _set_feature_dim_like_binop(out, a, b)
                return out
            _warn_fallback_once("combine", "packed operands over different sequences")
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
        # padding can be a per-spatial-dim list; the time axis entry is the relevant one here
        padding_t = padding[0] if isinstance(padding, (list, tuple)) and padding else padding
        if padding_t == "same":
            pad_l, pad_r = span // 2, (span + 1) // 2
        elif padding_t == "valid":
            pad_l = pad_r = 0
        elif isinstance(padding_t, int):
            pad_l = pad_r = padding_t
        else:
            pad_l = pad_r = None  # unsupported spec
        required_gap = max(pad_l, pad_r) if pad_l is not None else None
        # note: gap % align == 0 is NOT required: starts stay stride-aligned via the footprints;
        # whether the out layout is expressible is verified at runtime (_strided_out_wrapper).
        stride_ok = st == 1 or raw.align % st == 0
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
                [time_dim],
                filter_size=filter_size[0],
                strides=st,
                dilation_rate=dils_[0] or 1,
                # padding can be a per-spatial-dim list; this call is over the time dim only
                padding=padding[0] if isinstance(padding, (list, tuple)) else padding,
            )
            if st == 1:
                # "valid"/int: seq lens and gap change by a constant, starts stay in place
                helper = PackedRawTensor(
                    inner=out_inner,
                    packed_dim=out_sp[0],
                    orig_dims=tuple(raw.orig_dims[:-1]) + (out_time,),
                    gap=raw.gap + span - pad_l - pad_r,
                    align=raw.align,
                    # constant footprints: the layout lens shift by the same content-len delta
                    layout_lens=(raw.layout_lens + pad_l + pad_r - span) if raw.layout_lens is not None else None,
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
                action="re-layouting into the closed form, one extra gather (conv stays packed)",
            )
            return (
                _extract_strided(raw, out_inner, out_sp[0], out_time, st),
                [out_time] + list(out_sp[1:]),
            )
        return _dim_aware_call(
            "conv",
            (source,),
            dict(
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
            ),
        )

    @classmethod
    def scaled_dot_product_attention(
        cls,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        attention_mask: Optional[Tensor] = None,
        att_dropout: float = 0.0,
        att_dropout_broadcast: bool,
        v_feat_dim: Dim,
        qk_feat_dim: Dim,
        kv_spatial_dim: Dim,
        query_spatial_dim: Dim,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ):
        """
        Scaled dot-product attention. Torch specialization:
        torch SDPA over nested (jagged) tensors built directly from the packed layout
        (no external package, flash varlen internally), incl. is_causal;
        see :func:`_sdpa_varlen_attention`.
        Generic fallback otherwise (which then uses the packed matmul/softmax handling).
        """
        dropout_p = 0.0
        if att_dropout:
            if att_dropout_broadcast:
                # Legacy broadcast dropout (behavior <=18):
                # a different mask shape than elementwise SDPA dropout_p.
                dropout_p = None
            else:
                train_flag = rf.get_run_ctx().is_train_flag_enabled(func=rf.dropout)
                if isinstance(train_flag, bool):
                    # rf.dropout here == SDPA dropout_p
                    # (elementwise on att weights, 1/(1-p) scaling, modulo the RNG realization).
                    dropout_p = att_dropout if train_flag else 0.0
                else:
                    dropout_p = None  # dynamic train flag, cannot resolve to a static dropout_p
        if attention_mask is None and dropout_p is not None:
            # Preferred: the direct flash varlen kernels (inside _sdpa_varlen_attention;
            # 2x faster than flex at model level, native dropout).
            # Fallbacks: without dropout, flex over eager NJT
            # (allow_njt=False -> None when flash is not available);
            # with dropout, eager NJT (correct, flex has no dropout).
            out = _sdpa_varlen_attention(
                query,
                key,
                value,
                qk_feat_dim=qk_feat_dim,
                v_feat_dim=v_feat_dim,
                kv_spatial_dim=kv_spatial_dim,
                is_causal=is_causal,
                dropout_p=dropout_p,
                scale=scale,
                allow_njt=bool(dropout_p),
            )
            if out is not None:
                return out
            if not dropout_p:
                out = _flex_doc_attention(
                    query,
                    key,
                    value,
                    qk_feat_dim=qk_feat_dim,
                    v_feat_dim=v_feat_dim,
                    kv_spatial_dim=kv_spatial_dim,
                    query_spatial_dim=query_spatial_dim,
                    is_causal=is_causal,
                    scale=scale,
                )
                if out is not None:
                    return out
        else:
            _sdpa_no(
                "attention_mask given"
                if attention_mask is not None
                else "att_dropout with broadcast (legacy) or dynamic train flag"
            )
        return Backend.scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            att_dropout=att_dropout,
            att_dropout_broadcast=att_dropout_broadcast,
            v_feat_dim=v_feat_dim,
            qk_feat_dim=qk_feat_dim,
            kv_spatial_dim=kv_spatial_dim,
            query_spatial_dim=query_spatial_dim,
            is_causal=is_causal,
            scale=scale,
        )

    @classmethod
    def rel_pos_self_attention(
        cls,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: Tensor,
        *,
        pos_bias_u: Optional[Tensor],
        pos_bias_v: Optional[Tensor],
        att_dropout: float = 0.0,
        att_dropout_broadcast: bool,
        v_feat_dim: Dim,
        qk_feat_dim: Dim,
        kv_spatial_dim: Dim,
        query_spatial_dim: Dim,
        pos_emb_spatial_dim: Dim,
    ):
        """
        Self-attention with relative positional encoding. Packed specialization:
        torch FlexAttention over the flat packed buffer with a document block mask
        and the position-based term (matrix b+d) as score_mod, see :func:`_flex_rel_pos_attention`.
        FlexAttention has no dropout support, so with att_dropout active in training
        the generic fallback runs (which then uses the packed op handling).
        """
        dropout_active = False
        if att_dropout:
            train_flag = rf.get_run_ctx().is_train_flag_enabled(func=rf.dropout)
            # for a dynamic (tensor) train flag, conservatively assume training
            dropout_active = train_flag if isinstance(train_flag, bool) else True
        if not dropout_active:
            out = _flex_rel_pos_attention(
                query,
                key,
                value,
                pos_emb,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                v_feat_dim=v_feat_dim,
                qk_feat_dim=qk_feat_dim,
                kv_spatial_dim=kv_spatial_dim,
                query_spatial_dim=query_spatial_dim,
                pos_emb_spatial_dim=pos_emb_spatial_dim,
            )
            if out is not None:
                return out
        else:
            # the kernel drops per-element, so only the non-broadcast dropout matches
            if not att_dropout_broadcast:
                out = _triton_rel_pos_attention(
                    query,
                    key,
                    value,
                    pos_emb,
                    pos_bias_u=pos_bias_u,
                    pos_bias_v=pos_bias_v,
                    att_dropout=att_dropout,
                    v_feat_dim=v_feat_dim,
                    qk_feat_dim=qk_feat_dim,
                    kv_spatial_dim=kv_spatial_dim,
                    query_spatial_dim=query_spatial_dim,
                    pos_emb_spatial_dim=pos_emb_spatial_dim,
                )
                if out is not None:
                    return out
            _flex_no("att_dropout active in training (FlexAttention has no dropout support)")
        if is_packed(query) and query.raw_tensor.inner.device == "cpu":
            # CPU: per-seq slice loop over the packed buffer, no unpack
            # (the planned Triton kernel is the CUDA fast path)
            out = _rel_pos_attention_per_seq(
                query,
                key,
                value,
                pos_emb,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                att_dropout=att_dropout,
                att_dropout_broadcast=att_dropout_broadcast,
                v_feat_dim=v_feat_dim,
                qk_feat_dim=qk_feat_dim,
                kv_spatial_dim=kv_spatial_dim,
                query_spatial_dim=query_spatial_dim,
                pos_emb_spatial_dim=pos_emb_spatial_dim,
            )
            if out is not None:
                return out
        # Single unpack around the whole attention (instead of per-op unpack round trips inside):
        # the padded intermediates exist only within the attention block,
        # where the O(B * T^2) score matrix is materialized anyway.
        template = query.raw_tensor if is_packed(query) else None
        out = Backend.rel_pos_self_attention(
            unpack(query),
            unpack(key),
            unpack(value),
            unpack(pos_emb),
            pos_bias_u=unpack(pos_bias_u) if pos_bias_u is not None else None,
            pos_bias_v=unpack(pos_bias_v) if pos_bias_v is not None else None,
            att_dropout=att_dropout,
            att_dropout_broadcast=att_dropout_broadcast,
            v_feat_dim=v_feat_dim,
            qk_feat_dim=qk_feat_dim,
            kv_spatial_dim=kv_spatial_dim,
            query_spatial_dim=query_spatial_dim,
            pos_emb_spatial_dim=pos_emb_spatial_dim,
        )
        return _repack_result(out, template) if template is not None else out

    @staticmethod
    def pool(
        source: Tensor,
        *,
        mode: str,
        pool_size: Sequence[int],
        padding: Union[str, int, Sequence[int]] = "valid",
        dilation_rate: Union[Sequence[int], int] = 1,
        strides: Sequence[int],
        in_spatial_dims: Sequence[Dim],
        out_spatial_dims: Optional[Sequence[Dim]] = None,
    ) -> Tuple[Tensor, Sequence[Dim]]:
        """
        pool. Packed fast path, same machinery as :func:`PackedBackend.conv`:
        runs directly over the packed buffer; gap frames are filled with the pool-neutral value
        (-inf for max, 0 otherwise), so boundary windows behave like the per-sequence pool.
        """
        raw = source.raw_tensor
        assert isinstance(raw, PackedRawTensor)
        n = len(in_spatial_dims)
        strides_ = list(strides) if isinstance(strides, (list, tuple)) else [strides] * n
        dils_ = list(dilation_rate) if isinstance(dilation_rate, (list, tuple)) else [dilation_rate or 1] * n
        span = (pool_size[0] - 1) * (dils_[0] or 1)
        st = strides_[0] or 1
        # padding can be a per-spatial-dim list; the time axis entry is the relevant one here
        padding_t = padding[0] if isinstance(padding, (list, tuple)) and padding else padding
        if padding_t == "same":
            pad_l, pad_r = span // 2, (span + 1) // 2
        elif padding_t == "valid":
            pad_l = pad_r = 0
        elif isinstance(padding_t, int):
            pad_l = pad_r = padding_t
        else:
            pad_l = pad_r = None
        required_gap = max(pad_l, pad_r) if pad_l is not None else None
        stride_ok = st == 1 or raw.align % st == 0
        if (
            pad_l is not None
            and stride_ok
            and (st > 1 or raw.gap + span - pad_l - pad_r >= 0)
            and in_spatial_dims[0] == raw.orig_dims[-1]
            and not any(_dim_refs_packed(d, raw) for d in in_spatial_dims[1:])
            and not out_spatial_dims
        ):
            if raw.gap < required_gap:
                target_gap = required_gap if st == 1 else -(-required_gap // raw.align) * raw.align
                _warn_fallback_once(
                    "pool",
                    f"gap {raw.gap} < required {required_gap} for the packed pool"
                    f" -- specify pack(..., gap=...) to avoid the extra re-layout",
                    action="re-packing with the required gap (pool stays packed)",
                )
                source = regap(source, target_gap)
                raw = source.raw_tensor
            inner = raw.inner
            if required_gap:
                # boundary windows read the gap: fill with the pool-neutral value
                mask = _frame_mask(raw)
                if mask is not None:
                    inner = rf.where(mask, inner, float("-inf") if mode == "max" else 0.0)
            out_inner, out_sp = raw.inner_backend.pool(
                inner,
                mode=mode,
                pool_size=pool_size,
                padding=padding,
                dilation_rate=dilation_rate,
                strides=strides,
                in_spatial_dims=[raw.packed_dim] + list(in_spatial_dims[1:]),
            )
            time_dim = in_spatial_dims[0]
            if st == 1 and out_sp[0] == raw.packed_dim:  # "same" stride 1: layout unchanged
                out = raw.rewrap(out_inner, name="pool")
                return out, [time_dim] + list(out_sp[1:])
            (out_time,) = rf.make_conv_out_spatial_dims(
                [time_dim],
                filter_size=pool_size[0],
                strides=st,
                dilation_rate=dils_[0] or 1,
                # padding can be a per-spatial-dim list; this call is over the time dim only
                padding=padding[0] if isinstance(padding, (list, tuple)) else padding,
            )
            if st == 1:
                helper = PackedRawTensor(
                    inner=out_inner,
                    packed_dim=out_sp[0],
                    orig_dims=tuple(raw.orig_dims[:-1]) + (out_time,),
                    gap=raw.gap + span - pad_l - pad_r,
                    align=raw.align,
                    # constant footprints: the layout lens shift by the same content-len delta
                    layout_lens=(raw.layout_lens + pad_l + pad_r - span) if raw.layout_lens is not None else None,
                )
                out = helper.rewrap(out_inner, name="pool")
                return out, [out_time] + list(out_sp[1:])
            out = _strided_out_wrapper(raw, out_inner, out_sp[0], out_time, st)
            if out is not None:
                return out, [out_time] + list(out_sp[1:])
            _warn_fallback_once(
                "pool:strided-out",
                "strided pool output layout not expressible in the (lens, gap, align) form",
                action="re-layouting into the closed form, one extra gather (pool stays packed)",
            )
            return (
                _extract_strided(raw, out_inner, out_sp[0], out_time, st),
                [out_time] + list(out_sp[1:]),
            )
        return _dim_aware_call(
            "pool",
            (source,),
            dict(
                mode=mode,
                pool_size=pool_size,
                padding=padding,
                dilation_rate=dilation_rate,
                strides=strides,
                in_spatial_dims=in_spatial_dims,
                out_spatial_dims=out_spatial_dims,
            ),
        )

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
                    # constant footprints: the pad frames come out of the gap
                    layout_lens=(raw.layout_lens + pad_r) if raw.layout_lens is not None else None,
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
        # conform other packed operands (same seqs, possibly a different layout) to raw0's packing.
        operands = [_conform_packing(x, raw0) for x in operands]
        cond, true_, false_ = operands
        packed_ops = [x for x in operands if isinstance(x, Tensor) and is_packed(x)]
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
        matmul where a (or b) is packed.
        Profitable case: reduce and the other operand do not touch the packed dims
        (e.g. the Linear layer / vocab projection on [packed, F]) --
        then it runs directly on the packed data, saving all padding FLOPs.
        """
        if not is_packed(a) and is_packed(b):
            # matmul is symmetric up to the dim order, and RF dims are semantically unordered
            a, b = b, a
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
            out = raw.rewrap(rf.softmax(raw.inner, axis=axis, use_mask=use_mask), name="softmax")
            if tensor.feature_dim is not None and out.feature_dim is None and tensor.feature_dim in out.dims:
                out.feature_dim = tensor.feature_dim  # the inner op may not have it set
            return out
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
            out = raw.rewrap(rf.log_softmax(raw.inner, axis=axis, use_mask=use_mask), name="log_softmax")
            if tensor.feature_dim is not None and out.feature_dim is None and tensor.feature_dim in out.dims:
                out.feature_dim = tensor.feature_dim  # the inner op may not have it set
            return out
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
        CE over a non-packed axis (vocab), targets over the same sequences -> on packed data.
        logits [packed, vocab], targets [packed] (sparse).
        The targets are matched to the logits packing (regap, a no-op if the layout already agrees,
        e.g. a different per-key gap), then rewrapped onto the logits packed dim.
        Falls back if the axis is packed or the targets are not packed over the same sequences.
        """
        logits_raw = _raw(logits)
        targets_raw = targets.raw_tensor
        if (
            not _dim_refs_packed(axis, logits_raw)
            and isinstance(targets_raw, PackedRawTensor)
            and targets_raw.orig_dims == logits_raw.orig_dims
        ):
            targets = regap(targets, logits_raw.gap, align=logits_raw.align, layout_lens=logits_raw.layout_lens)
            targets_raw = targets.raw_tensor
            target_inner = targets_raw.inner
            if targets_raw.packed_dim != logits_raw.packed_dim:
                target_inner, _ = rf.replace_dim(
                    target_inner, in_dim=targets_raw.packed_dim, out_dim=logits_raw.packed_dim
                )
            inner_out = rf.cross_entropy(
                estimated=logits_raw.inner, target=target_inner, axis=axis, estimated_type="logits"
            )
            return logits_raw.rewrap(inner_out, name="cross_entropy")
        _warn_fallback_once("softmax_cross_entropy_with_logits", "axis packed or targets not over the same seqs")
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
            inner = raw.inner
            if raw.has_gap_frames:
                # gap / alignment pad frames would contaminate the maskless reduce:
                # fill them with the mode-neutral value first (mean: sum, then / valid count)
                mask = _frame_mask(raw)
                if mode == "mean":
                    inner = rf.where(mask, inner, 0.0)
                    total = rf.reduce(inner, mode="sum", axis=[raw.packed_dim] + extra_axes, use_mask=False)
                    n = rf.cast(_packed_total(raw.orig_dims, 0, 1), total.dtype)
                    n = rf.copy_to_device(n, total.device)
                    for d in extra_axes:
                        v = d.get_dim_value_tensor()  # int for static dims
                        n = n * (rf.cast(v, total.dtype) if isinstance(v, Tensor) else float(v))
                    return total / n
                neutral = {"sum": 0.0, "logsumexp": float("-inf"), "max": float("-inf"), "min": float("inf")}.get(mode)
                if neutral is None:
                    _warn_fallback_once("reduce", f"full reduce over gapped layout, mode {mode}")
                    return _repack_result(rf.reduce(unpack(source), mode=mode, axis=axes, use_mask=use_mask), raw)
                inner = rf.where(mask, inner, neutral)
            # use_mask=False: packed storage has no padded frames (gap frames neutralized above).
            return rf.reduce(inner, mode=mode, axis=[raw.packed_dim] + extra_axes, use_mask=False)
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
        use_native_op: Optional[bool] = None,
        label_loop: bool = True,
    ) -> Tensor:
        """
        CTC loss on packed logits, via the packed native fast-baum-welch op
        (see :class:`returnn.native_op.FastBaumWelchPackedOp`):
        the (total, vocab) log-probs buffer is used as-is, no padded intermediate.
        Falls back to the generic (unpack) handling if not applicable.
        """
        targets_ = unpack(targets) if is_packed(targets) else targets
        raw = logits.raw_tensor if is_packed(logits) else None
        if (
            raw is not None
            and raw.inner_backend.name == "torch"
            and not max_approx
            and use_native_op is not False
            and len(raw.orig_dims) == 2
            and raw.orig_dims[-1] == input_spatial_dim
            and logits.feature_dim is not None
            and set(raw.inner.dims) == {raw.packed_dim, logits.feature_dim}
            and targets_.dims_set == {raw.orig_dims[0], targets_spatial_dim}
        ):
            import torch
            from returnn.torch.util import native_op as torch_native_op

            batch_dim = raw.orig_dims[0]
            logits_t = raw.inner.copy_transpose([raw.packed_dim, logits.feature_dim]).raw_tensor
            if logits_t.dtype != torch.float32:
                logits_t = logits_t.to(torch.float32)  # the native op is float32-only
            device = logits_t.device
            # per-seq start offsets into the packed buffer (gapped or dense): the native op reads each
            # seq at [start, start+len), so the gap frames are never touched -- no regap needed.
            starts_rf, _ = raw.seq_starts()
            seq_starts = rf.copy_to_device(starts_rf, device).raw_tensor.to(torch.int32).flatten()
            in_lens = input_spatial_dim.dyn_size_ext.copy_compatible_to_dims_raw([batch_dim]).to(device)
            targets_raw = targets_.copy_compatible_to_dims_raw([batch_dim, targets_spatial_dim]).to(device)
            tgt_lens = targets_spatial_dim.dyn_size_ext.copy_compatible_to_dims_raw([batch_dim]).to(device)
            loss_raw = torch_native_op.ctc_loss_packed(
                logits=logits_t,
                seq_starts=seq_starts,
                logits_seq_lens=in_lens,
                max_seq_len=int(input_spatial_dim.get_dim_value()),
                targets=targets_raw,
                targets_seq_lens=tgt_lens,
                label_loop=label_loop,
                logits_normalize=not logits_normalized,
                blank_index=blank_index,
            )
            loss = Tensor("ctc_loss", dims=[batch_dim], dtype="float32")
            loss.raw_tensor = loss_raw
            return loss
        return _dim_aware_call(
            "ctc_loss",
            (),
            dict(
                logits=logits,
                logits_normalized=logits_normalized,
                targets=targets,
                input_spatial_dim=input_spatial_dim,
                targets_spatial_dim=targets_spatial_dim,
                blank_index=blank_index,
                max_approx=max_approx,
                use_native_op=use_native_op,
                label_loop=label_loop,
            ),
        )


# All other structural ops go through the generic dim-aware wrapper:
# packed data directly if the call does not reference the packed dims, otherwise unpack fallback.
for _name in [
    "batch_norm",
    "compare",
    "concat",
    "cumsum",
    "expand_dim",
    "flip_no_mask",
    "gather",
    "masked_scatter",
    "masked_select",
    "merge_dims",
    "reshape",
    "scatter",
    "search_sorted",
    "slice",
    "sort",
    "split",
    "split_dims",
    "squeeze",
    "stack",
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


def pack_import(
    inner_flat: Tensor,
    *,
    batch_dim: Dim,
    spatial_dim: Dim,
    packed_dim: Dim,
    feature_dim: Optional[Dim] = None,
) -> Tensor:
    """
    Import an already-flat inner tensor into packed storage,
    i.e. the sequences are already concatenated along the packed dim, without padding.
    This is the entry point for the packed data pipeline (collate_batch),
    where the flat buffer never went through a padded intermediate.

    :param inner_flat: dims [packed_dim, feature...], the concatenated dense buffer
    :param batch_dim: virtual batch dim
    :param spatial_dim: virtual spatial (time) dim,
        must carry the per-seq sizes (dyn_size_ext over [batch_dim])
    :param packed_dim: the packed dim (total frames), must match inner_flat's first dim
    :param feature_dim: optional feature dim for the result
    :return: tensor with virtual dims [batch_dim, spatial_dim, feature...], packed (dense) storage
    """
    assert packed_dim in inner_flat.dims
    if feature_dim is not None and inner_flat.feature_dim is None:
        inner_flat.feature_dim = feature_dim
    helper = PackedRawTensor(inner=inner_flat, packed_dim=packed_dim, orig_dims=(batch_dim, spatial_dim))
    return helper.rewrap(inner_flat, name=inner_flat.name)


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

    Unlike :func:`pack_padded`, the returned Tensor keeps the *original* (virtual) dims,
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


def regap(source: Tensor, gap: int, *, align: Optional[int] = None, layout_lens: Optional[Tensor] = None) -> Tensor:
    """
    :return: same content, packed with the given gap (and align, default: keep;
        plus optional target layout_lens, e.g. to restore an exact strided-out layout):
        a cheap packed -> packed re-layout (one scatter over the frames, no padded intermediate).
        Used e.g. by the packed conv when the tensor's gap is too small.
    """
    raw = _raw(source)
    if align is None:
        align = raw.align
    others = raw.orig_dims[:-1]
    if (raw.gap == gap and raw.align == align and raw.layout_lens is layout_lens) or not others:
        return source
    last = raw.orig_dims[-1]
    new_dim = Dim(_packed_total(raw.orig_dims, gap, align, layout_lens=layout_lens), name="packed_regap")
    new_starts, seqs_dim = _seq_starts_math(raw.orig_dims, gap, align, layout_lens=layout_lens)
    t_coords = _frame_coords(raw, last)
    seg, _, _ = _segment_index(raw, others)
    if new_starts.device != seg.device:
        # starts derive from the dyn sizes (often cpu), the coords live on the data device
        new_starts = rf.copy_to_device(new_starts, seg.device)
    pos = rf.gather(new_starts, indices=seg, axis=seqs_dim, clip_to_valid=True) + t_coords
    mask = _frame_mask(raw)
    if mask is not None:
        # route old gap frames to a dump slot, then slice it off
        ext_dim = new_dim + 1
        pos = rf.where(mask, pos, rf.cast(new_dim.get_dim_value_tensor(), pos.dtype))
        inner_ext = rf.scatter(raw.inner, indices=pos, indices_dim=raw.packed_dim, out_dim=ext_dim, use_mask=False)
        inner_new, _ = rf.slice(inner_ext, axis=ext_dim, size=new_dim)
    else:
        inner_new = rf.scatter(raw.inner, indices=pos, indices_dim=raw.packed_dim, out_dim=new_dim, use_mask=False)
    if raw.inner.feature_dim is not None and inner_new.feature_dim is None:
        inner_new.feature_dim = raw.inner.feature_dim
    helper = PackedRawTensor(
        inner=inner_new, packed_dim=new_dim, orig_dims=raw.orig_dims, gap=gap, align=align, layout_lens=layout_lens
    )
    out = helper.rewrap(inner_new, name="regap")
    if source.feature_dim is not None and out.feature_dim is None:
        out.feature_dim = source.feature_dim
    return out


def unpack(source: Tensor) -> Tensor:
    """
    :param source: tensor with packed storage. if not packed, returned as-is.
    :return: tensor with normal padded storage of the inner backend, same dims
    """
    raw = source.raw_tensor
    if not isinstance(raw, PackedRawTensor):
        return source
    if raw.has_gap_frames:
        pos = _padded_positions(raw.orig_dims, raw.gap, raw.align, layout_lens=raw.layout_lens)
        out = rf.gather(raw.inner, indices=pos, axis=raw.packed_dim, clip_to_valid=True)
        # zero the padded frames, like the dense masked_scatter does
        out = rf.where(rf.sequence_mask(list(raw.orig_dims), device=out.device), out, 0)
    else:
        out = rf.pad_packed(raw.inner, dims=raw.orig_dims, in_dim=raw.packed_dim)
    if source.feature_dim is not None and out.feature_dim is None and source.feature_dim in out.dims:
        out.feature_dim = source.feature_dim  # masked_scatter drops it
    return out


register_backend_by_tensor_type(PackedRawTensor, PackedBackend)
