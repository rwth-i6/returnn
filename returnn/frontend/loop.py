"""
While loop.

Why did we choose this API?
To allow for both eager-based and graph-based frameworks,
and also to avoid any magic happening here.
https://github.com/rwth-i6/returnn/issues/1282
https://github.com/rwth-i6/returnn/issues/1324
"""

from __future__ import annotations
from typing import Optional, Union, Any, TypeVar, Callable, Sequence, Tuple, List, Dict, Type
import tree
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from .tensor_array import TensorArray, _BufferTensorArray
from ._backend import Backend, global_backend


__all__ = ["while_loop", "scan"]


S = TypeVar("S")  # any nested structure, can be None
X = TypeVar("X")  # any nested structure, can be None
Y = TypeVar("Y")  # any nested structure, can be None


def while_loop(
    cond: Callable[[S], Union[bool, Tensor]],
    body: Callable[[S], S],
    initial: S,
) -> S:
    """
    It executes::

        while cond(loop_vars):
            loop_vars = body(loop_vars)

    And then it returns the final loop vars.

    If you want to iterate over some axis,
    maybe of an existing tensor,
    or if you want to accumulate frames in each iteration,
    see :func:`scan`.

    :param cond:
    :param body:
    :param initial: initial loop vars
    :return: final loop vars
    """
    init_tensors = [v for v in tree.flatten(initial) if isinstance(v, Tensor)]
    if not init_tensors:
        backend = global_backend
    else:
        v = init_tensors[0]
        # noinspection PyProtectedMember
        backend = v._raw_backend
    # Prefer the backend's graph loop where it has one.
    if backend.while_loop is not Backend.while_loop:
        return backend.while_loop(cond, body, initial)
    if backend.executing_eagerly():
        return _while_loop_eager(cond, body, initial)
    return backend.while_loop(cond, body, initial)


def _while_loop_eager(cond: Callable[[S], Union[bool, Tensor]], body: Callable[[S], S], initial: S) -> S:
    """the Python loop, for eager backends"""
    loop_vars = initial
    loop_var_templates = _templates_for_loop_vars(loop_vars)
    dim_updates = _DimUpdatesEager(initial_state=loop_vars)
    while _get_bool_value_eager(cond(loop_vars)):
        loop_vars = body(loop_vars)
        tree.assert_same_structure(loop_var_templates, loop_vars)
        loop_var_templates = dim_updates.update_template_from_new(loop_var_templates, loop_vars)
        _check_matching_loop_var_templates(loop_var_templates, loop_vars)
    return loop_vars


def _get_bool_value_eager(v: Union[Tensor, bool]) -> bool:
    if isinstance(v, Tensor):
        assert v.dims == () and v.dtype == "bool"
        assert v.device in (None, "cpu"), f"while_loop: cond should be on CPU, got {v} on device {v.device}"
        return bool(v.raw_tensor)
    elif isinstance(v, bool):
        return v
    else:
        raise TypeError(f"while_loop: cond: unexpected return type {type(v)}")


def _graph_while_loop(
    cond: Callable[[S], Union[bool, Tensor]],
    body: Callable[[S], S],
    initial: S,
    *,
    backend: Type[Backend],
    run_loop: Callable[
        [Callable[[Tuple[Any, ...]], Any], Callable[[Tuple[Any, ...]], Tuple[Any, ...]], Tuple[Any, ...]],
        Sequence[Any],
    ],
) -> S:
    """
    The backend-neutral part of a graph loop over a fixed-shape carry
    (JAX ``lax.while_loop``, PT ``while_loop`` op).

    Loop vars are the :class:`Tensor` and :class:`TensorArray` entries of the structure;
    everything else is loop-invariant and must come back unchanged from ``body``.
    A Tensor may not change its dims per iteration,
    and a TensorArray needs a capacity (``TensorArray(capacity=...)``), so it is a buffer, not a list.
    A dim in the state may be replaced per iteration, e.g. a growing history dim:
    it is carried by its dyn size and needs a capacity.

    :param cond: gets the loop vars, returns a scalar bool Tensor
    :param body: gets the loop vars, returns the next loop vars, same structure
    :param initial: initial loop vars
    :param backend: the backend of the loop vars
    :param run_loop: ``(cond_raw, body_raw, init) -> final``, the loop over a flat tuple of raw tensors:
        each Tensor, then buffer and write index of each TensorArray, then the dyn size of each dim.
        It also moves the carry to one device if the backend requires that.
    :return: final loop vars
    """
    flat_initial = list(tree.flatten(initial))
    var_idxs = [i for i, v in enumerate(flat_initial) if isinstance(v, (Tensor, TensorArray))]
    assert var_idxs, f"while_loop: no Tensor/TensorArray among the loop vars {initial}"
    dim_idxs = [i for i, v in enumerate(flat_initial) if isinstance(v, Dim) and v.dyn_size_ext is not None]
    for i in dim_idxs:
        assert flat_initial[i].capacity is not None, (
            f"while_loop: dim {flat_initial[i]} in the state has no capacity."
            f" The graph loop needs a static bound on what it carries."
        )

    def _tensor_array_raws(v: TensorArray) -> Tuple[Any, Any]:
        """:return: buffer, index"""
        # noinspection PyProtectedMember
        raw = v._backend_tensor_array
        if isinstance(raw, _BufferTensorArray) and raw.buffer is not None:
            return raw.buffer, raw.index
        assert v.capacity is not None and (isinstance(raw, _BufferTensorArray) or raw == []), (
            f"while_loop: TensorArray {v.tensor_template} must be created with a capacity"
            f" to cross the graph loop, as the carry cannot grow"
        )
        tmpl = v.tensor_template
        buffer = rf.zeros([Dim(v.capacity, name="tensor_array")] + list(tmpl.dims), dtype=tmpl.dtype).raw_tensor
        if isinstance(raw, _BufferTensorArray):
            return buffer, raw.index
        return buffer, rf.constant(0, dims=(), dtype=rf.get_default_array_index_dtype()).raw_tensor

    def _carry(flat: List[Any]) -> Tuple[Any, ...]:
        """:return: flat raw carry, each Tensor in the dim order of its initial value"""
        out = []
        for i in var_idxs:
            v = flat[i]
            if isinstance(v, Tensor):
                # An RF Tensor is dim-order agnostic, but the raw carry is not:
                # the body may return the same tensor transposed.
                out.append(v.copy_compatible_to_dims_raw(flat_initial[i].dims))
            else:
                out.extend(_tensor_array_raws(v))
        out.extend(flat[i].dyn_size_ext.raw_tensor for i in dim_idxs)
        return tuple(out)

    def _rebuild(carry: Sequence[Any]) -> S:
        """:return: the loop vars as body/cond expect them"""
        out = list(flat_initial)
        carry = list(carry)
        dim_map = {}
        for i, raw in zip(dim_idxs, carry[len(carry) - len(dim_idxs) :]):
            d = flat_initial[i]
            size = d.dyn_size_ext.copy_template()
            size.raw_tensor = raw
            d_new = Dim(size, name=d.name, capacity=d.capacity)
            dim_map[d] = d_new
            out[i] = d_new
        pos = 0
        for i in var_idxs:
            value = flat_initial[i]
            if isinstance(value, Tensor):
                tensor = value.copy_template()
                if dim_map:
                    tensor = tensor.copy_template_new_dim_tags([dim_map.get(d, d) for d in tensor.dims])
                tensor.raw_tensor = carry[pos]
                out[i] = tensor
                pos += 1
            else:
                # noinspection PyProtectedMember
                prev = value._backend_tensor_array
                template = prev.template if isinstance(prev, _BufferTensorArray) else None
                out[i] = TensorArray(
                    tensor_template=value.tensor_template,
                    capacity=value.capacity,
                    _backend_tensor_array=_BufferTensorArray(
                        capacity=value.capacity,
                        buffer=carry[pos],
                        index=carry[pos + 1],
                        template=template if template is not None else value.tensor_template,
                    ),
                    _backend=backend,
                )
                pos += 2
        return tree.unflatten_as(initial, out)

    def _cond(carry: Tuple[Any, ...]) -> Any:
        res = cond(_rebuild(carry))
        assert isinstance(res, Tensor) and res.dims == () and res.dtype == "bool", (
            f"while_loop: cond must return a scalar bool Tensor, got {res}"
        )
        return res.raw_tensor

    def _body(carry: Tuple[Any, ...]) -> Tuple[Any, ...]:
        new = body(_rebuild(carry))
        tree.assert_same_structure(initial, new)
        flat_new = list(tree.flatten(new))
        # the body may hand back replaced dims; map them onto the initial ones so the carry
        # is compared and laid out against a single set of tags
        back_map = {}
        for i in dim_idxs:
            d_new = flat_new[i]
            assert isinstance(d_new, Dim), f"while_loop: loop var {i} was a Dim, got {d_new!r}"
            back_map[d_new] = flat_initial[i]
        for i in var_idxs:
            v = flat_new[i]
            if back_map and isinstance(v, Tensor) and any(d in back_map for d in v.dims):
                flat_new[i] = v.copy_template_new_dim_tags([back_map.get(d, d) for d in v.dims], keep_special_axes=True)
                flat_new[i].raw_tensor = v.raw_tensor
            before, after = flat_initial[i], flat_new[i]
            if isinstance(before, Tensor):
                assert isinstance(after, Tensor) and before.dims_set == after.dims_set, (
                    f"while_loop: loop var {i} changed its dims, {before} -> {after}."
                    f" The graph loop needs one fixed shape across iterations;"
                    f" give the dim a capacity instead of growing it."
                )
        return _carry(flat_new)

    init = _carry(flat_initial)
    # Inside the graph loop every shape is static,
    # so we must take a dim's capacity, not its dyn size.
    with rf.set_static_traceable_ctx(True):
        final = run_loop(_cond, _body, init)
    return _rebuild(final)


def scan(
    *,
    spatial_dim: Optional[Dim] = None,
    cond_dims: Optional[Sequence[Dim]] = None,
    cond_before_body: bool = True,
    initial: S = None,
    xs: X = None,
    ys: Y = None,
    cond: Optional[Callable[[X, S], Tensor]] = None,
    body: Callable[[X, S], Tuple[Y, S]],
    max_seq_len: Optional[Union[int, Tensor]] = None,
    return_tensor_arrays: bool = False,
) -> Tuple[Y, S, Dim]:
    """
    Extended variant of :func:`while_loop`.

    Supports iterating over a given axis (spatial_dim),
    supports iterating over some input tensors (xs: X) on the given axis,
    and supports returning some frame-accumulated output tensors (ys: Y).

    https://github.com/rwth-i6/returnn/issues/1324

    :param spatial_dim: if None or unknown, need to provide cond. must be given if xs is not None.
    :param cond_dims: if spatial_dim is not given, this must be given to know what shape to expect from cond.
        This will also be the shape of the dyn_size_ext of the resulting spatial_dim.
    :param cond_before_body: if True, will execute cond before body, otherwise after.
        If True, corresponds to ``while cond(...): body(...)``,
        otherwise ``while True: body(...); if not cond(...): break``.
        Note that `cond` is executed in any case at the end of the loop
        but with `cond_before_body=True` this final value would be ignored.
        Be careful though that you do not have any side-effects in `cond`.
    :param initial: state/carry
    :param xs: input, will be unstacked over spatial_dim. can be None.
    :param ys: output, as templates, per iteration (excluding spatial_dim)
    :param cond: if spatial_dim is None/unknown, need to provide this.
        The shape will be the same as the dyn_size_ext of the resulting spatial_dim.
        Unlike while_loop cond, does not need to be scalar. E.g. some shape like [B] is normal.
        Once it returns False for all entries, the loop will stop.
        Once it returns False for some entry, further values in further iterations for this entry will be ignored.
        We do not expect any side-effects in `cond`.
    :param body:
    :param max_seq_len: If given, it is checked in addition to `cond`, and when reached, it stops the loop.
    :param return_tensor_arrays: if True, will return TensorArray instead of Tensor for ys.
        Internally, we work with TensorArray anyway, so this avoids the final stack().
        In case of beam search, it might make more sense
        to perform some post-processing on the TensorArray per entry,
        like selecting the right beam entries.
    :return: outputs ys, final state, and the new spatial_dim
    """
    device = None
    if initial is not None:
        vs = [v for v in tree.flatten(initial) if isinstance(v, Tensor) and v.device not in (None, "cpu")]
        if vs:
            device = vs[0].device
    if device is None and xs is not None:
        vs = [v for v in tree.flatten(xs) if isinstance(v, Tensor) and v.device not in (None, "cpu")]
        if vs:
            device = vs[0].device

    if spatial_dim is None or not spatial_dim.is_dim_known():
        assert cond is not None, f"scan: spatial_dim {spatial_dim} is None/unknown, need to provide `cond`"
        assert cond_dims is not None, f"scan: spatial_dim {spatial_dim} is None/unknown, need to provide `cond_dims`"
        assert xs is None, f"scan: spatial_dim {spatial_dim} is None/unknown, cannot use input `xs` {xs}"
        if spatial_dim is None:
            spatial_dim = Dim(None, name="scan_dim")

        def _cond(_s: Tuple[Tensor, Tensor, Tensor, S, Y]) -> Tensor:
            i, _, c, _, _ = _s
            c = rf.reduce_any(c, axis=c.dims)
            if max_seq_len is not None:
                c = rf.logical_and(c, i < max_seq_len)
            return c

        def _body(_s: Tuple[Tensor, Tensor, Tensor, S, Y]) -> Tuple[Tensor, Tensor, Tensor, S, Y]:
            i, seq_len_, prev_cond, s, ys_ = _s
            seq_len_ = seq_len_ + rf.cast(prev_cond, dtype=seq_len_.dtype)
            y, s_ = body(None, s)
            tree.assert_same_structure(ys_, y)
            ys_ = tree.map_structure(lambda ys__, y_: ys__.push_back(y_) if ys__ is not None else None, ys_, y)
            c = cond(None, s_)
            c = rf.logical_and(c, prev_cond)
            s_ = rf.nested.mask_nested(s_, mask=c, mask_value=s, allow_dim_extension=False)
            return i + 1, seq_len_, c, s_, ys_

        if cond_before_body:
            initial_cond = cond(None, initial)
            assert (
                isinstance(initial_cond, Tensor)
                and initial_cond.dtype == "bool"
                and initial_cond.dims_set == set(cond_dims)
            )
        else:
            initial_cond = rf.constant(True, dtype="bool", dims=cond_dims)
        _, seq_len, _, final_s, ys = while_loop(
            _cond,
            _body,
            (
                rf.constant(0, dtype=rf.get_default_array_index_dtype(), dims=()),  # i
                rf.constant(0, dtype=rf.get_default_array_index_dtype(), dims=cond_dims),  # seq_len
                initial_cond,  # initial cond. keep this in state such that we can update seq_len in body
                initial,  # state
                tree.map_structure(lambda y: TensorArray(y) if y is not None else None, ys),
            ),
        )

        spatial_dim.dyn_size_ext = seq_len

    else:
        assert cond is None, f"scan: spatial_dim {spatial_dim} is known, cannot use `cond` {cond}"
        assert max_seq_len is None, f"scan: spatial_dim {spatial_dim} is known, cannot use `max_seq_len` {max_seq_len}"

        xs = tree.map_structure(lambda x: TensorArray.unstack(x, axis=spatial_dim), xs)

        def _cond(_s: Tuple[Tensor, S, Y]) -> Tensor:
            i, *_ = _s
            return i < spatial_dim.get_dim_value_tensor()

        def _body(_s: Tuple[Tensor, S, Y]) -> Tuple[Tensor, S, Y]:
            i, s, ys_ = _s
            y, s_ = body(tree.map_structure(lambda x: x[i], xs), s)
            tree.assert_same_structure(ys_, y)
            ys_ = tree.map_structure(lambda ys__, y_: ys__.push_back(y_) if ys__ is not None else None, ys_, y)
            s_ = rf.nested.mask_nested(
                s_, mask=i < spatial_dim.get_size_tensor(device=device), mask_value=s, allow_dim_extension=False
            )
            return i + 1, s_, ys_

        _, final_s, ys = while_loop(
            _cond,
            _body,
            (
                rf.constant(0, dtype=rf.get_default_array_index_dtype(), dims=(), device="cpu"),  # i
                initial,  # state
                tree.map_structure(lambda y: TensorArray(y) if y is not None else None, ys),
            ),
        )

    if not return_tensor_arrays:
        ys = tree.map_structure(lambda ys_: ys_.stack(axis=spatial_dim) if ys_ is not None else None, ys)
    return ys, final_s, spatial_dim


def _templates_for_loop_vars(loop_vars: S) -> S:
    def _get_template(x):
        if isinstance(x, Tensor):
            return x.copy_template()
        elif isinstance(x, Dim):
            return x
        elif isinstance(x, TensorArray):
            return x  # not really checked, but just leave it for now
        elif x is None:
            return None
        else:
            raise TypeError(f"unexpected type {type(x)} for loop var {x}")

    return tree.map_structure(_get_template, loop_vars)


def _check_matching_loop_var_templates(loop_var_templates: S, loop_vars: S):
    def _check(path, template, x):
        if isinstance(template, Tensor):
            assert isinstance(x, Tensor), f"loop var {path} is not a Tensor but {type(x)}"

            assert template.batch_ndim == x.batch_ndim, (
                f"loop var {path} template {template} does not match var {x}, "
                f"different batch_ndim {template.batch_ndim} vs {x.batch_ndim}"
            )
            assert template.dims_set == x.dims_set, (
                f"loop var {path} template {template} does not match var {x}, "
                f"different dims (no matter the order) {template.dims} vs {x.dims}"
            )
            assert template.sparse_dim == x.sparse_dim, (
                f"loop var {path} template {template} does not match var {x}, "
                f"different sparse_dim {template.sparse_dim} vs {x.sparse_dim}"
            )
            assert template.feature_dim == x.feature_dim, (
                f"loop var {path} template {template} does not match var {x}, "
                f"different feature_dim {template.feature_dim} vs {x.feature_dim}"
            )

        elif isinstance(template, Dim):
            assert isinstance(x, Dim), f"loop var {path} is not a Dim but {type(x)}"
            # Later, we actually want to support having different dims.
            # See https://github.com/rwth-i6/returnn/issues/1327.
            # For now, we don't support this.
            assert template == x, f"loop var {path} template dim {template} does not match var dim {x}"

        elif isinstance(template, TensorArray):
            assert isinstance(x, TensorArray), f"loop var {path} is not a TensorArray but {type(x)}"
            _check(path, template.tensor_template, x.tensor_template)
            # noinspection PyProtectedMember
            x._push_back_delayed_check()

        else:  # other cases: just check same type
            assert type(template) is type(x), (
                f"loop var {path} template type {type(template)} does not match var type {type(x)}"
            )
            assert not isinstance(x, Tensor), f"loop var {path} is a Tensor but should not be"

    tree.map_structure_with_path(_check, loop_var_templates, loop_vars)


class _DimUpdatesEager:
    """
    In case dims are updated in the loop body, we need to keep track of this.

    This implementation is for eager-based backends.

    A graph-based backend would need to distinguish:
    - initial dim
    - dim in loop body (temporarily), input from prev iteration state
    - dim in loop body (temporarily), output for new state
    - final dim

    When collecting tensors with such dims in a :class:`TensorArray`,
    we must later be able to iterate through it again, maybe backwards,
    and then also iterate over the collected dims,
    and be able to match it.

    See here for some discussion and motivation:
    https://github.com/rwth-i6/returnn/issues/1327
    """

    def __init__(self, *, initial_state: S):
        self.dim_variants: Dict[Dim, List[Dim]] = {}  # initial -> variants in each iteration
        self._initial_dim: Dict[Dim, Dim] = {}  # variant -> initial
        self._init_initial(initial_state)

    def _init_initial(self, state: S):
        """init"""
        for x in tree.flatten(state):
            if isinstance(x, Dim):
                assert x not in self.dim_variants, f"dim {x} already in dim_variants, assumed to be unique in state"
                self.dim_variants[x] = []
                self._initial_dim[x] = x
        for x in tree.flatten(state):
            if isinstance(x, TensorArray):
                if any(d in self._initial_dim for d in x.tensor_template.dims):
                    # noinspection PyProtectedMember
                    x._set_enable_delayed_check()

    def update_template_from_new(self, template_state: S, new_state: S) -> S:
        """
        :param template_state: via :func:`_templates_for_loop_vars`
        :param new_state: output from body(). Warning: we update TensorArray.tensor_template inplace here.
        :return: updated template state
        """
        if not self.dim_variants:
            # Fast path: No dims are updated in the loop. No change on the template.
            return template_state

        def _visit_add_dim_variants(template, x):
            if isinstance(x, Dim):
                assert isinstance(template, Dim)
                initial_dim = self._initial_dim[template]
                self.dim_variants[initial_dim].append(x)
                self._initial_dim[x] = initial_dim

        tree.map_structure(_visit_add_dim_variants, template_state, new_state)

        def _visit_update_template(template, x):
            if isinstance(x, Dim):
                assert isinstance(template, Dim)
                return x
            if isinstance(x, Tensor):
                assert isinstance(template, Tensor)
                return self.template_for_current_iteration(template)
            if isinstance(x, TensorArray):
                assert isinstance(template, TensorArray)
                # It's ugly that we replace this inplace,
                # but we don't really have a better solution for now.
                # This assumes that stack() would not be called on the tensor array,
                # i.e. return_tensor_arrays=True for scan().
                x.tensor_template = self.template_for_current_iteration(x.tensor_template)
                template.tensor_template = self.template_for_current_iteration(template.tensor_template)
                return template
            return template

        return tree.map_structure(_visit_update_template, template_state, new_state)

    def template_for_current_iteration(self, template: Tensor) -> Tensor:
        """
        :param template: template for the current iteration
        :return: template for the current iteration, with updated dims
        """
        if any(d in self._initial_dim for d in template.dims):
            return template.copy_template_new_dim_tags(
                [self.dim_variants[self._initial_dim[d]][-1] if d in self._initial_dim else d for d in template.dims]
            )
        return template
