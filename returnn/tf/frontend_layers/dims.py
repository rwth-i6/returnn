"""
Dim helpers, specific to this net dict backend
"""

from __future__ import annotations
from typing import Union, Sequence, List
import itertools
from weakref import WeakKeyDictionary
from returnn.tensor import Tensor, Dim
from .. import frontend_layers as rfl
from . import _utils


# currently assume there is only one root NameCtx at a time, otherwise we would need this to be root NameCtx dependent
_dim_deps = WeakKeyDictionary()  # type: WeakKeyDictionary[Dim, List[Tensor]]


def get_dim_deps(dim: Union[Dim, Sequence[Dim]]) -> List[Tensor]:
    """
    :return: the tensors the dim tag depends on.
      This is needed for some functions (layers) such as `nn.constant` or `nn.random_...`.
      https://github.com/rwth-i6/returnn/issues/1096
    """
    if isinstance(dim, (tuple, list, set)):
        return _utils.unique_tensor_list(itertools.chain(*(get_dim_deps(dim_) for dim_ in dim)))
    if not isinstance(dim, Dim):
        raise TypeError(f"expected nn.Dim, got {type(dim)}")
    dim = dim.get_same_base()
    if dim.is_static():  # static dim -> no deps
        return []
    if dim.special:
        raise ValueError(f"{dim} deps not defined for special tags")
    _register_dim_via_dyn_layer(dim)
    if dim in _dim_deps:
        deps = _dim_deps[dim]
        if _deps_valid_in_cur_name_ctx(deps):
            return deps
        _dim_deps.pop(dim)
    if not dim.is_dim_known() and not dim.derived_from_op:
        raise ValueError(f"{dim} is not defined yet")
    if dim.derived_from_op:
        deps = get_dim_deps(dim.derived_from_op.inputs)
        _dim_deps[dim] = deps
        return deps
    # should not get here
    raise Exception(f"{dim} deps not defined (_register_dim_deps not called?)")


def _register_dim_deps_when_novel(dim: Dim, deps: List[Tensor]):
    if dim.derived_from_op:
        return  # not needed
    dim = dim.get_same_base()
    if dim in _dim_deps:
        # We could just always keep the first dep list.
        # But there are cases where the new dep list might be better:
        old_deps = _dim_deps[dim]
        if not _deps_valid_in_cur_name_ctx(old_deps):
            _dim_deps.pop(dim)  # replace, use new deps list
        elif (
            # For extern_data, when the first dep list for not fully available for inference,
            # but the new dep list is, we take over the new one.
            any(not dep.available_for_inference for dep in old_deps)
            and all(dep.available_for_inference for dep in deps)
        ):
            _dim_deps.pop(dim)  # go on, replace, use the new list
        else:
            return  # discard new list, keep old
    if _register_dim_via_dyn_layer(dim):
        return
    _dim_deps[dim] = deps


def _deps_valid_in_cur_name_ctx(deps: List[Tensor]) -> bool:
    cur_root = rfl.Layer.top().root
    for dep in deps:
        assert isinstance(dep, Tensor)
        assert isinstance(dep.raw_tensor, rfl.Layer)
        if dep.raw_tensor.root != cur_root:
            return False
    return True


def _register_dim_via_dyn_layer(dim: Dim) -> bool:
    """
    :param dim:
    :return: whether we registered the dim
    """
    if dim.is_static():
        return False
    if dim in _dim_deps:
        return False
    assert dim.dyn_size_ext
    if dim.dyn_size_ext.raw_tensor is None:
        return False
    assert isinstance(dim.dyn_size_ext.raw_tensor, rfl.Layer)
    # It means the user probably has created some dynamic dim directly.
    # This is valid.
    # But for the net dict backend, we must handle it differently.
    _dim_deps[dim] = []  # Will be reassigned below. This is just to avoid recursion.
    # Follow the logic as in returnn-common make_dim_from_length.
    # The actual range tensor is never used, but this has the side effect to set up the dim tag.
    layer_with_dim = rfl.make_layer(
        {
            "class": "range_from_length",
            "from": dim.dyn_size_ext.copy(),
            "dtype": dim.dyn_size_ext.dtype,
            "out_spatial_dim": dim,
        },
        name=dim.dyn_size_ext.name,
    )
    _dim_deps[dim] = [layer_with_dim]
    dim.dyn_size_ext.raw_tensor = None
    return True
