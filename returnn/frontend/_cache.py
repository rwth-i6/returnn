"""
Cache, to store some data.
See :class:`Cache`.

One use case example is :func:`sinusoidal_positional_encoding` and :func:`relative_positional_encoding`.
"""

from __future__ import annotations
from typing import Optional, Union, Any, Type, Callable, Tuple, Dict, List
from weakref import ref
import tree
from returnn.util.lru_cache import lru_cache
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend._backend import global_backend, get_backend_by_raw_tensor_type, Backend


class Cache:
    """
    Cache, intended for internal use of RF functions.

    One use case example is :func:`sinusoidal_positional_encoding` and :func:`relative_positional_encoding`.

    There are some specific properties we must take care of:

    - Lifetime of values: For graph-based backends, it can only stay alive for the current run ctx.
      (For eager-based backends, there is no such restriction.)
    - Size: Put some limit, use LRU logic.
    - Dims: Use only weakrefs. Some Dim should not stay alive just because of the cache.
    - Scalar dynamic Dims in eager mode, or static dims: Instead of the Dim, use the dim value for the key
      (and map the output to the Dim).
    - Tensor as keys: Use weakrefs. Also don't check by value but by identity.
    """

    def __init__(self, max_size: int):
        # Use lru_cache here, but not via a decorator,
        # as we want custom set/get logic.
        # Also, we want the lru_cache to be local to this Cache instance,
        # not shared over all instances of this class.
        self._lru_cache = lru_cache(max_size)(_lru_cache_dummy_func)

    def __repr__(self):
        return f"<{self.__class__.__module__}.{self.__class__.__qualname__} {self._lru_cache.cache_info()}>"

    def get(self, key, default=None):
        """
        :param key:
        :param default:
        :return: entry in cache or default
        """
        key_transformed = _transform_key(key)
        key_transformed_orig, value = self._lru_cache.cache_peek(key_transformed, fallback=(None, None))
        if key_transformed_orig is None:
            return default

        assert len(key_transformed_orig) == len(key_transformed)
        dim_map = {}  # orig -> new
        for key_item_orig, key_item in zip(key_transformed_orig, key_transformed):
            if isinstance(key_item_orig, DimWrapper):
                assert isinstance(key_item, DimWrapper)
                dim_orig = key_item_orig.dim_ref()
                if dim_orig is None:  # orig dim could be dead. but then it would not be used anyway
                    continue
                dim = key_item.dim_ref()
                assert isinstance(dim_orig, Dim) and isinstance(dim, Dim)
                dim_map[dim_orig] = dim

        # noinspection PyShadowingNames
        def _map_output(output):
            if isinstance(output, Dim):
                return dim_map.get(output, output)
            if isinstance(output, Tensor):
                if any(dim in dim_map for dim in output.dims):
                    out_raw = output.raw_tensor
                    for axis, dim in enumerate(output.dims):
                        if dim in dim_map:
                            output = output.copy_template_replace_dim_tag(axis=axis, new_dim_tag=dim_map[dim])
                    output.raw_tensor = out_raw
            return output

        return tree.map_structure(_map_output, value)

    def set(self, key, value):
        """
        :param key:
        :param value:
        """

        def _finalize_callback(*_args):
            self._lru_cache.cache_pop(key_transformed, fallback=None)

        key_backend = _get_backend(key)
        value_backend = _get_backend(value)
        if key_backend != value_backend:
            raise ValueError(f"key and value have different backends: {key_backend} != {value_backend}")
        key_transformed = _transform_key(key, finalize_callback=_finalize_callback)
        self._lru_cache.cache_set(key_transformed, result=(key_transformed, value))


def _lru_cache_dummy_func(*_args, **_kwargs):
    raise Exception("This should not be called.")


def _transform_key(
    key: Any, *, finalize_callback: Optional[Callable] = None, collected_dim_map: Optional[Dict[Dim, DimWrapper]] = None
) -> Tuple[Union[Type[Backend], ref[rf.RunCtx], _KeyItemType], ...]:
    backend = _get_backend(key)
    keys_flat: List[Any] = [backend]
    if not backend.executing_eagerly():
        # See comment above: If graph-mode, the cached value becomes invalid
        # when the current run ctx goes out of scope.
        keys_flat.append(ref(rf.get_run_ctx(), finalize_callback))
    if collected_dim_map is None:
        collected_dim_map = {}
    keys_flat += [
        _transform_key_item(key, finalize_callback=finalize_callback, collected_dim_map=collected_dim_map)
        for key in tree.flatten(key)
    ]
    return tuple(keys_flat)


def _transform_key_item(
    key: Any, *, finalize_callback: Optional[Callable] = None, collected_dim_map: Dict[Dim, DimWrapper]
) -> _KeyItemType:
    if isinstance(key, Tensor):
        return TensorWrapper(key, finalize_callback=finalize_callback)
    if isinstance(key, Dim):
        if key in collected_dim_map:
            return collected_dim_map[key]
        dim_wrapper = DimWrapper(key, finalize_callback=finalize_callback)
        collected_dim_map[key] = dim_wrapper
        return dim_wrapper
    if not isinstance(key, _RawTypes):
        raise TypeError(f"unexpected type {type(key)}")
    return key


def _get_backend(*args) -> Type[Backend]:
    args_flat = tree.flatten(args)
    for arg in args_flat:
        if isinstance(arg, Tensor) and arg.raw_tensor is not None:
            return get_backend_by_raw_tensor_type(type(arg.raw_tensor))
    return global_backend.__class__


class TensorWrapper:
    """
    Wraps :class:`Tensor`.
    Using weakref for the tensor, including also ``raw_tensor``.
    Equality is given if the identity is the same, for the Tensor itself and the raw_tensor.
    No value of the tensor is checked.
    """

    def __init__(self, value: Tensor, *, finalize_callback):
        self.value_ref = ref(value, finalize_callback)
        self.raw_value_ref = ref(value.raw_tensor, finalize_callback)
        self._hash = id(value)

    def __eq__(self, other):
        if isinstance(other, TensorWrapper):
            return self.value_ref() is other.value_ref() and self.raw_value_ref() is other.raw_value_ref()
        return False

    def __hash__(self):
        return self._hash


class DimWrapper:
    """
    Wraps :class:`Dim`.
    Using weakref for the dim.
    If the size is scalar and known, equality is given when the size is equal (and dim tag is ignored)
    """

    def __init__(self, dim: Dim, *, finalize_callback):
        self.dim_value = _dim_value_for_key(dim)
        # finalize_callback only needed when we don't use the dim value.
        self.dim_ref = ref(dim, finalize_callback if self.dim_value is None else None)
        self.dyn_size_ref = (
            # E.g. consider the batch dim or data spatial dim which would be reset each step.
            # We need some ref to the dyn size, and finalize this key when it goes out of scope.
            # This is only needed when there is no info on the static size (or eager scalar dyn size).
            ref(dim.dyn_size_ext.raw_tensor, finalize_callback)
            if self.dim_value is None and dim.dyn_size_ext is not None and dim.dyn_size_ext.raw_tensor is not None
            else None
        )
        self._hash = hash(dim) if self.dim_value is None else hash(self.dim_value)

    def __eq__(self, other):
        if isinstance(other, DimWrapper):
            if self.dim_value is not None:
                return self.dim_value == other.dim_value
            return self.dim_ref() == other.dim_ref() and self.dyn_size_ref() is other.dyn_size_ref()
        return False

    def __hash__(self):
        return self._hash


def _dim_value_for_key(dim: Dim) -> Optional[int]:
    if dim.size is not None:
        return dim.size
    if dim.dyn_size_ext is not None and not dim.dyn_size_ext.dims:
        if dim.dyn_size_ext.raw_tensor is not None:
            # noinspection PyProtectedMember
            if dim.dyn_size_ext._raw_backend.executing_eagerly():
                return int(dim.get_dim_value())
    return None


# For now... we might extend it by some more types.
_KeyItemType = Union[None, str, bool, int, float, TensorWrapper, DimWrapper]
_RawTypes = (type(None), str, bool, int, float)
