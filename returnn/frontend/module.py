"""
Base module class, :class:`Module`.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, List, Tuple, Union, Set, Iterator, Callable, TypeVar
from returnn.util.basic import OptionalNotImplementedError, RefIdEq
from returnn.tensor import Tensor, Dim
from .. import frontend as rf

__all__ = ["Module", "Functional"]


T = TypeVar("T", bound="Module")


class Module:
    """
    This can represent a subnetwork in RETURNN.

    You can write PyTorch-like code here, like::

        class MyModule(rf.Module):

          def __init__(self, dim: Dim, activation=tanh):
            super().__init__()
            self.layer_norm = rf.LayerNorm(dim)
            self.linear = rf.Linear(dim, dim)
            self.activation = activation

          def __call__(self, x: Tensor) -> Tensor:
            x_ = x
            x = self.layer_norm(x)
            x = self.linear(x)
            x = self.activation(x)
            return x_ + x

    A module (here, just like in PyTorch or Keras)
    has params, but getting some output for some input
    requires an additional `forward` or `__call__` call,
    which can be called multiple times.
    Every such call would then share the same module parameters.

    The :func:`__init__` would usually get module-level arguments
    which describe the parameters.
    As a module might be called multiple times,
    any input-specific arguments such as spatial dims
    are usually arguments of :func:`__call__`.
    Other arguments which might vary between calls
    would also be arguments of :func:`__call__`
    such as epsilon
    although there are no strict rules.
    """

    def __init__(self):
        """
        By convention, any options to the module are passed to __init__,
        and potential changing inputs (other tensors)
        are passed to :func:`__call__`.
        """

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> Optional[rf.State]:
        """
        :return: default initial state, to be used if the module has recurrent (hidden) state.
            When a module has recurrent state,
            the convention is to return a tuple with instance :class:`State` as the last item,
            and to accept the ``state`` argument with a :class:`State` with the same nested structure.
            This can be a nested structure and should match the structure of the ``state`` argument and returned value.
        """
        state = rf.State()
        for key, mod in self.named_children():
            sub_state = mod.default_initial_state(batch_dims=batch_dims)
            if sub_state:
                state[key] = sub_state
        if state:
            return state
        return None

    def get_default_name(self) -> str:
        """
        Get a default layer name (used when we do not have a Module attribute pointing to this).
        This is used by :class:`NameCtx` for the RETURNN layer naming
        (but only when the RETURNN layer name is not implied by other the module attribute hierarchy).
        """
        name = self.__class__.__name__
        if name.startswith("_"):
            name = name[1:]
        if name[:1].isupper():
            from returnn.util.basic import camel_case_to_snake_case

            name = camel_case_to_snake_case(name)
        return name

    def __call__(self, *args, **kwargs) -> Union[Tensor, Tuple[Tensor, rf.State], Any]:
        """
        Main module call.

        Note that there is nothing really specific about this method.
        Your module can have other methods as well,
        and you don't necessarily need to define this.
        Only certain other functions or modules like Sequential make use of it.
        """
        raise OptionalNotImplementedError

    def get_deep(self, target: str) -> Any:
        """
        Returns the deep attrib given by ``target`` if it exists, otherwise throws an error.
        """
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod: Module = self

        for item in atoms[:-1]:
            if not hasattr(mod, item):
                raise AttributeError(f"{mod} has no attribute `{item}`")
            mod = getattr(mod, item)
            if not isinstance(mod, Module):
                raise AttributeError(f"`{item}` is not an rf.Module")

        return getattr(mod, atoms[-1])

    def set_deep(self, target: str, value: Any) -> None:
        """
        Sets the deep attrib given by ``target`` to ``value``.
        """
        if target == "":
            raise AttributeError("Cannot set root module")

        if "." in target:
            prefix, target = target.rsplit(".", 2)
            mod = self.get_deep(prefix)
            if not isinstance(mod, Module):
                raise AttributeError(f"{self}: `{prefix}` is not an rf.Module")
        else:
            mod = self

        setattr(mod, target, value)

    def children(self) -> Iterator[rf.Module]:
        """
        Get all immediate children modules, excluding self.
        """
        return self.modules(recurse=False, include_self=False)

    def named_children(self) -> Iterator[Tuple[str, rf.Module]]:
        """
        Get all immediate children modules, excluding self.
        """
        return self.named_modules(recurse=False, include_self=False)

    def modules(self, *, recurse: bool = True, include_self: bool = True) -> Iterator[rf.Module]:
        """
        Get all children modules, optionally recursively, maybe including self.
        """
        for name, child in self.named_modules(recurse=recurse, include_self=include_self):
            yield child

    def named_modules(
        self,
        *,
        recurse: bool = True,
        include_self: bool = True,
        memo: Optional[Set[RefIdEq[rf.Module]]] = None,
        prefix: str = "",
    ) -> Iterator[Tuple[str, rf.Module]]:
        """
        Get all children modules (excluding self)
        """
        if memo is None:
            memo = set()
        if self in memo:
            return
        memo.add(RefIdEq(self))
        if include_self:
            yield prefix, self
        queue = [(prefix, self)]  # type: List[Tuple[str, Module]]
        while queue:
            prefix, mod = queue.pop(0)
            for name, module in vars(mod).items():
                if not isinstance(module, Module):
                    continue
                if RefIdEq(module) in memo:
                    continue
                sub_prefix = prefix + ("." if (prefix and not prefix.endswith(".")) else "") + name
                memo.add(RefIdEq(module))
                yield sub_prefix, module
                if recurse:
                    queue.append((sub_prefix, module))

    def named_parameters(self, *, recurse: bool = True) -> Iterator[Tuple[str, rf.Parameter]]:
        """
        Get all children parameters, together with their names.

        With recurse=True (default), this iterates over all children modules
        and iterates through their parameters as well.

        Note that some modules (e.g. :class:`rf.Linear`) can behave lazy,
        i.e. they only create the parameters on the first call,
        e.g. when the input dimension is unknown and thus the parameter shape is not defined
        before the first call.
        This means you need to first call the module once to get all the parameters.
        https://github.com/rwth-i6/returnn_common/issues/149
        """
        memo: Set[RefIdEq[Tensor]] = set()  # RefIdEq because we cannot hash layer refs
        for prefix, module in self.named_modules() if recurse else [("", self)]:
            for key, value in vars(module).items():
                if isinstance(value, rf.Parameter) and RefIdEq(value) not in memo:
                    sub_prefix = prefix + ("." if prefix else "") + key
                    memo.add(RefIdEq(value))
                    yield sub_prefix, value

    def parameters(self, *, recurse: bool = True) -> Iterator[rf.Parameter]:
        """
        Get all children parameters. Also see :func:`named_parameters` for some more documentation.
        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    @property
    def has_parameters(self):
        """
        Whether this module has variables
        """
        for _, _ in self.named_parameters(recurse=True):
            return True
        return False

    def apply(self: T, fn: Callable[[rf.Module], None]) -> T:
        """
        Applies the function ``fn`` to all children modules and self.

        :return: self
        """
        # Use `children` here and not `modules` to allow any submodule to potentially overwrite the `apply` logic.
        for child in self.children():
            child.apply(fn)
        fn(self)
        return self


class Functional(Module):
    """
    Used for functions (pure functional, i.e. not methods of another module)
    and via :class:`ModuleList` to wrap up any functions or lambdas as modules.

    (This is often not necessary, but sometimes useful.)
    """

    def __init__(self, func):
        super().__init__()
        assert callable(func)
        self.func = func

    def __repr__(self):
        return f"{self.__class__.__name__}({self.func.__qualname__})"

    def get_default_name(self) -> str:
        """default name"""
        import re

        name = self.func.__qualname__
        assert isinstance(name, str)
        if name.startswith("Tensor.__"):
            m = re.match(r"^Tensor\.__(.*)__$", name)
            if m:
                return m.group(1)
        if ".<locals>." in name:
            name = name.replace(".<locals>.", ".")
        return name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
