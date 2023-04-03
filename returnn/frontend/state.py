"""
Recurrent state
"""

from __future__ import annotations
from typing import Union, Any, List, Set
from returnn.tensor import Tensor


__all__ = ["State"]


class State(dict):
    """
    Covers all the state of a recurrent module,
    i.e. exactly what needs to be stored and passed into the module
    next time you call it as initial state.

    This behaves somewhat like a namedtuple, although we derive from dict.
    """

    def __init__(self, *args, **kwargs):
        if kwargs:
            assert not args
            super().__init__(**kwargs)
        elif args:
            assert len(args) == 1
            if isinstance(args[0], dict):
                super().__init__(**args[0])
            else:
                super().__init__(state=args[0])
        else:
            super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for (k, v) in self.items())})"

    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError(f"{self}.{item}")

    def __setattr__(self, key, value):
        self[key] = value

    def flatten_tensors(self) -> List[Tensor]:
        """See :func:`cls_deep_tensors`."""
        return self.cls_flatten_tensors(self)

    @classmethod
    def cls_flatten_tensors(cls, obj: Union[State, dict, Any]) -> List[Tensor]:
        """
        Iterates through obj and all its sub-objects, yielding all tensors.
        """
        from returnn.util.basic import RefIdEq

        cache_tensor_refs = set()  # type: Set[RefIdEq[Tensor]]  # RefIdEq because tensors are not hashable
        tensors = []  # type: List[Tensor]
        queue = [obj]

        while queue:
            x = queue.pop()
            if isinstance(x, Tensor):
                if RefIdEq(x) not in cache_tensor_refs:
                    cache_tensor_refs.add(RefIdEq(x))
                    tensors.append(x)
            elif isinstance(x, dict):
                queue.extend(x.values())
            elif isinstance(x, (list, tuple)):
                queue.extend(x)
            else:
                raise TypeError(f"unexpected type {type(x)}")

        return tensors
