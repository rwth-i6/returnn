"""
Parametrize some parameters, e.g. to implement weight dropout, variational noise, weight norm, etc.

We follow the `PyTorch parametrization API
<https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.register_parametrization.html>`__
and also borrow some code.

https://github.com/rwth-i6/returnn/issues/1518
"""

from __future__ import annotations
from typing import Optional, Union
import copyreg
import weakref
from returnn.util.py_compat import Protocol
import returnn.frontend as rf
from returnn.tensor import Tensor


__all__ = ["register_parametrization", "remove_parametrization", "is_parametrized"]


def register_parametrization(
    module: rf.Module, param_name: str, parametrization: _ParametrizationType, *, keep_existing_param: bool = True
) -> rf.Module:
    """
    Register parametrization for a tensor (parameter) in a module.

    :param module:
    :param param_name:
    :param parametrization:
    :param keep_existing_param:
        True: the original parameter stays in there,
            and parametrization will be called with the original parameter as an argument::
                parametrization(orig_param)
            In this case, parametrization must not have own parameters.
            This is useful for potential optional transformations, e.g. weight dropout or variational noise.
        False: the original parameter will be removed, and this will be a submodule,
            which can have its own parameters.
            It will be called without arguments::
                parametrization()
    """
    if not is_parametrized(module):
        # Sets up a module to be parametrized.
        # This works by substituting the class of the module by a class
        # that extends it to be able to inject a property.
        # We need this because we cannot inject a property into an object instance
        # (see https://docs.python.org/3/howto/descriptor.html)
        # and also we do not want to modify the original class.
        cls = module.__class__
        param_cls = _Metaclass(f"Parametrized{cls.__name__}", (cls,), {})
        module.__class__ = param_cls

    if hasattr(module.__class__, param_name):
        raise ValueError(
            f"register_parametrization: parametrized property {param_name} already exists in module {module}"
        )

    orig_param = getattr(module, param_name)
    if not isinstance(orig_param, rf.Parameter):
        raise TypeError(f"module.{param_name} is not a parameter, got {orig_param!r}")

    if keep_existing_param:
        if isinstance(parametrization, rf.Module):
            if len(list(parametrization.parameters())) > 0:
                raise ValueError(
                    f"register_parametrization: parametrization {parametrization} must not have parameters"
                    f" with keep_existing_param=True"
                )

    else:
        if hasattr(parametrization, "assign"):
            parametrization.assign(orig_param)
        orig_param = None
        # Put the parametrization into the module as a submodule
        # instead of the original parameter.
        # module.named_parameters() will thus find it, even when we install the new property.
        setattr(module, param_name, parametrization)

    # Injects a property into module
    assert isinstance(module.__class__, _Metaclass), "module must be parametrized"
    assert not hasattr(module.__class__, param_name), "property already exists"
    prop = _Property(module, param_name, parametrization, orig_param)
    setattr(module.__class__, param_name, prop)

    return module


def remove_parametrization(module: rf.Module, param_name: str) -> rf.Module:
    """
    Remove parametrization for a tensor (parameter) in a module.
    """
    if not is_parametrized(module):
        raise ValueError(f"module {module} is not parametrized")
    prop = getattr(module.__class__, param_name)
    assert isinstance(prop, _Property)
    delattr(module.__class__, param_name)
    assert not hasattr(module.__class__, param_name)
    if prop.orig_param is None:
        setattr(module, param_name, rf.Parameter(prop.parametrization()))
    # Check if there are any other parametrizations left.
    for k, v in vars(module.__class__).items():
        if isinstance(v, _Property):
            break
    else:  # no break, no other parametrizations
        module.__class__ = module.__class__.__bases__[0]  # revert to original class
    return module


def is_parametrized(module: rf.Module, param_name: Optional[str] = None) -> bool:
    r"""Returns ``True`` if module has an active parametrization.

    If the argument :attr:`tensor_name` is specified, returns ``True`` if
    ``module[tensor_name]`` is parametrized.

    Args:
        module: module to query
        param_name: attribute in the module to query
            Default: ``None``
    """
    if module.__class__.__class__ is not _Metaclass:
        return False
    if param_name is None:
        return True
    return hasattr(module.__class__, param_name)


class _ParametrizationTransform(Protocol):
    def __call__(self, x: Tensor) -> Tensor:
        """Return the parametrized tensor based on the original parameter."""


class _ParametrizationWithAssign(Protocol):
    def __call__(self) -> Tensor:
        """Return the parametrized tensor."""

    def assign(self, x: Tensor):
        """Assign as if it was a single parameter."""


class _ParametrizationWithoutAssign(Protocol):
    def __call__(self) -> Tensor:
        """Return the parametrized tensor."""


_ParametrizationType = Union[
    _ParametrizationTransform,
    _ParametrizationWithAssign,
    _ParametrizationWithoutAssign,
]


class _Metaclass(type):
    """
    https://stackoverflow.com/a/75943813/133374
    """


def _reduce_metaclass(cls):
    metaclass = cls.__class__
    cls_vars = dict(vars(cls))
    cls_vars.pop("__dict__", None)
    cls_vars.pop("__weakref__", None)
    return metaclass, (cls.__name__, cls.__bases__, cls_vars)


copyreg.pickle(_Metaclass, _reduce_metaclass)


class _Property:
    def __init__(
        self,
        module: rf.Module,
        param_name: str,
        parametrization: _ParametrizationType,
        orig_param: Optional[rf.Parameter],
    ):
        self.module_ref = weakref.ref(module)
        self.param_name = param_name
        self.parametrization = parametrization
        self.orig_param = orig_param

    def __get__(self, obj, objtype=None):
        if obj is None:  # called on the class
            return self
        assert obj is self.module_ref(), f"parametrize _Property __get__: {obj!r} vs {self.module_ref()!r}"
        if self.orig_param is not None:
            return self.parametrization(self.orig_param)
        else:
            return self.parametrization()

    def __set__(self, obj, value):
        assert obj is self.module_ref(), f"parametrize _Property __set__: {obj!r} vs {self.module_ref()!r}"
        if self.orig_param is not None:
            self.orig_param.assign(value)
        else:
            if hasattr(self.parametrization, "assign"):
                self.parametrization.assign(value)
            else:
                raise AttributeError(f"Cannot assign to {self.param_name} parametrization {self.parametrization}")
