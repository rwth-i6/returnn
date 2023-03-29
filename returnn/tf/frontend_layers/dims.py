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
    if dim.dimension is not None:  # static dim -> no deps
        return []
    if dim.special:
        raise ValueError(f"{dim} deps not defined for special tags")
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
            pass  # replace, use new deps list
        elif (
            # For extern_data, when the first dep list for not fully available for inference,
            # but the new dep list is, we take over the new one.
            any(not dep.available_for_inference for dep in old_deps)
            and all(dep.available_for_inference for dep in deps)
        ):
            pass  # go on, replace, use the new list
        else:
            return  # discard new list, keep old
    _dim_deps[dim] = deps


def _deps_valid_in_cur_name_ctx(deps: List[Tensor]) -> bool:
    return all(dep.raw_tensor.root == rfl.Layer.top().root for dep in deps)
