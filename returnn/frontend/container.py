"""
container functions
"""

from __future__ import annotations
import returnn.frontend as rf
from returnn.tensor import Tensor
from typing import Optional, TypeVar, Generic, Iterable, Iterator, Union, Tuple, Dict, Callable


__all__ = ["ModuleList", "Sequential", "sequential", "ParameterList"]


_UnaryFuncT = Callable[[Tensor], Tensor]
_ModT = Union[rf.Module, _UnaryFuncT]
__ModT = TypeVar("__ModT", bound=rf.Module)


class ModuleList(rf.Module, Generic[__ModT]):
    """
    Module list, getting passed an Iterable of Modules and creates a list of Modules in that order
    """

    def __init__(self, *modules: Union[__ModT, Iterable[__ModT], Dict[str, __ModT], ModuleList]):
        super().__init__()
        if len(modules) == 1 and isinstance(modules[0], dict):
            for key, module in modules[0].items():
                setattr(self, key, _convert_to_module(module))
        elif len(modules) == 1 and isinstance(modules[0], ModuleList):
            for key, module in modules[0]._get_modules().items():
                setattr(self, key, _convert_to_module(module))
        elif len(modules) == 1 and _is_iterable(modules[0]):
            for idx, module in enumerate(modules[0]):
                setattr(self, str(idx), _convert_to_module(module))
        else:
            for idx, module in enumerate(modules):
                setattr(self, str(idx), _convert_to_module(module))

    def _get_modules(self) -> Dict[str, __ModT]:
        return {key: value for (key, value) in vars(self).items() if isinstance(value, rf.Module)}

    def append(self, module: __ModT) -> ModuleList[__ModT]:
        """
        appends one module to the list
        """
        setattr(self, str(len(self)), _convert_to_module(module))
        return self

    def extend(self, modules: Iterable[__ModT]) -> ModuleList[__ModT]:
        """
        appends multiple modules to the list
        """
        for module in modules:
            self.append(module)
        return self

    def __len__(self) -> int:
        return len(self._get_modules())

    def __iter__(self) -> Iterator[__ModT]:
        return iter(self._get_modules().values())

    def items(self) -> Iterable[Tuple[str, __ModT]]:
        """module items"""
        return self._get_modules().items()

    def __getitem__(self, idx) -> Union[ModuleList[__ModT], __ModT]:
        from builtins import slice

        if isinstance(idx, slice):
            return self.__class__(dict(list(self._get_modules().items())[idx]))
        else:
            return list(self._get_modules().values())[idx]

    def __setitem__(self, idx: int, module: __ModT) -> None:
        key = list(self._get_modules().keys())[idx]
        return setattr(self, key, _convert_to_module(module))

    __call__ = rf.Module.__call__  # stays abstract


class Sequential(ModuleList):
    """
    Sequential Module, takes callable of Modules which are then executed in sequence
    """

    def __call__(self, inp, *, collected_outputs: Optional[Dict[str, Tensor]] = None, **kwargs) -> Tensor:
        """
        Forward
        """
        for name, module in self.items():
            inp = module(inp, **kwargs)
            if collected_outputs is not None:
                collected_outputs[name] = inp
        return inp


def sequential(source: Tensor, *modules) -> Tensor:
    """
    Wraps ``Sequential(*modules)(source)``
    """
    return Sequential(*modules)(source)


def _convert_to_module(obj: _ModT) -> rf.Module:
    if isinstance(obj, rf.Module):
        return obj
    elif callable(obj):
        return rf.Functional(obj)
    else:
        raise TypeError(f"did not expect {obj!r}")


def _is_iterable(obj) -> bool:
    # No good generic way, so do this ugly hack.
    # https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class ParameterList(rf.Module):
    """
    Parameter list, getting passed an Iterable of Parameters and creates a list of Parameters in that order
    """

    def __init__(self, *parameters: Union[rf.Parameter, Iterable[rf.Parameter], ParameterList]):
        super().__init__()
        if len(parameters) == 1 and isinstance(parameters[0], ParameterList):
            for key, parameter in parameters[0]._get_parameters().items():
                setattr(self, key, parameter)
        elif len(parameters) == 1 and _is_iterable(parameters[0]):
            for idx, parameter in enumerate(parameters[0]):
                setattr(self, str(idx), parameter)
        else:
            for idx, parameter in enumerate(parameters):
                setattr(self, str(idx), parameter)

    def _get_parameters(self) -> Dict[str, rf.Parameter]:
        return {key: value for (key, value) in vars(self).items() if isinstance(value, rf.Parameter)}

    def append(self, parameter: rf.Parameter) -> ParameterList:
        """
        appends one Parameter to the list
        """
        setattr(self, str(len(self)), parameter)
        return self

    def extend(self, parameters: Iterable[rf.Parameter]) -> ParameterList:
        """
        appends multiple Parameters to the list
        """
        for parameter in parameters:
            self.append(parameter)
        return self

    def __len__(self) -> int:
        return len(self._get_parameters())

    def __iter__(self) -> Iterator[rf.Parameter]:
        return iter(self._get_parameters().values())

    def __getitem__(self, idx) -> Union[ParameterList, rf.Parameter]:
        from builtins import slice

        if isinstance(idx, slice):
            return self.__class__(dict(list(self._get_parameters().items())[idx]))
        else:
            return list(self._get_parameters().values())[idx]

    def __setitem__(self, idx: int, parameter: rf.Parameter) -> None:
        key = list(self._get_parameters().keys())[idx]
        return setattr(self, key, rf.Parameter)

    __call__ = rf.Module.__call__  # stays abstract
