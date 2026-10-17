"""
Cache, to store some data.
See :class:`Cache`.

One use case example is :func:`sinusoidal_positional_encoding` and :func:`relative_positional_encoding`.
"""

from __future__ import annotations
from typing import Optional, Union, Any, Type, Callable, Tuple, Dict, List
from weakref import ref
import tree
from returnn.util.basic import make_hashable
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
    - Tensor as keys: Use weakrefs. Also don't check by value but by identity of the raw tensor,
      so a new Tensor wrapping the same raw tensor matches (e.g. raw tensors wrapped only for the lookup).
      Its dims (and sparse dim, feature axis, version) are part of the key, the dims like Dim keys
      (and the output is mapped to them).
    - :class:`rf.TensorLayout` as keys: its key parts (:func:`rf.TensorLayout.cache_key_parts`) are keys as above.
      The output is mapped also in its storage layout (e.g. the packed dims), via :func:`rf.replace_dim`.
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
        key_items_orig, key_items = _flat_dim_wrappers(key_transformed_orig), _flat_dim_wrappers(key_transformed)
        assert len(key_items_orig) == len(key_items)
        for key_item_orig, key_item in zip(key_items_orig, key_items):
            if isinstance(key_item_orig, DimWrapper):
                assert isinstance(key_item, DimWrapper)
                dim_orig = key_item_orig.dim_ref()
                if dim_orig is None:  # orig dim could be dead. but then it would not be used anyway
                    continue
                dim = key_item.dim_ref()
                assert isinstance(dim_orig, Dim) and isinstance(dim, Dim)
                if dim is not dim_orig:
                    dim_map[dim_orig] = dim
        if not dim_map:
            return value

        # noinspection PyShadowingNames
        def _map_output(output):
            if isinstance(output, Dim):
                return dim_map.get(output, output)
            if isinstance(output, Tensor):
                # the dims of the tensor and of its storage layout (e.g. the packed dims), each once
                dims = {}
                for dim in list(output.dims) + _flatten_key(rf.get_tensor_layout(output)):
                    if isinstance(dim, Dim) and dim in dim_map:
                        dims[id(dim)] = dim
                for dim in dims.values():
                    output, _ = rf.replace_dim(output, in_dim=dim, out_dim=dim_map[dim])
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
        if key_backend.executing_eagerly() and not value_backend.executing_eagerly():
            # the value is only valid in the run ctx, but only a graph-based key backend scopes the key to it
            raise ValueError(f"key backend {key_backend} is eager, value backend {value_backend} is graph-based")
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
    if rf.is_static_traceable():
        # Results computed under static traceable are only valid within the current step
        # (e.g. under CUDA-graph capture they live in the captured graph region,
        # and the regime's static buffers change in place across steps).
        # The generation identifies the step (the ctx is enabled once per step):
        # entries from earlier steps never match again and age out of the LRU.
        keys_flat.append(rf.static_traceable_generation())
    if collected_dim_map is None:
        collected_dim_map = {}
    keys_flat += [
        _transform_key_item(key, finalize_callback=finalize_callback, collected_dim_map=collected_dim_map)
        for key in _flatten_key(key)
    ]
    return tuple(keys_flat)


def _flatten_key(key: Any) -> List[Any]:
    """
    :param key: nested structure, can contain :class:`rf.TensorLayout`
    :return: flat list, each layout replaced by its flattened key parts
    """
    res = []
    for key_item in tree.flatten(key):
        if isinstance(key_item, rf.TensorLayout):
            res += _flatten_key(key_item.cache_key_parts())
        else:
            res.append(key_item)
    return res


def _transform_key_item(
    key: Any, *, finalize_callback: Optional[Callable] = None, collected_dim_map: Dict[Dim, DimWrapper]
) -> _KeyItemType:
    if isinstance(key, Tensor):
        return TensorWrapper(key, finalize_callback=finalize_callback, collected_dim_map=collected_dim_map)
    if isinstance(key, Dim):
        if key in collected_dim_map:
            return collected_dim_map[key]
        dim_wrapper = DimWrapper(key, finalize_callback=finalize_callback)
        collected_dim_map[key] = dim_wrapper
        return dim_wrapper
    if isinstance(key, type) and issubclass(key, Backend):  # e.g. of a layout
        return key
    if not isinstance(key, _RawTypes):
        raise TypeError(f"unexpected type {type(key)}")
    return key


def _flat_dim_wrappers(key_transformed: Tuple[Any, ...]) -> List[Any]:
    """
    :param key_transformed: from :func:`_transform_key`
    :return: the key items, with each :class:`TensorWrapper` replaced by its dim wrappers
    """
    res = []
    for key_item in key_transformed:
        if isinstance(key_item, TensorWrapper):
            res += key_item.get_dim_wrappers()
        else:
            res.append(key_item)
    return res


def _get_backend(*args) -> Type[Backend]:
    args_flat = _flatten_key(args)
    for arg in args_flat:
        if isinstance(arg, Dim) and arg.dyn_size_ext is not None:
            arg = arg.dyn_size_ext
        if isinstance(arg, Tensor) and arg.raw_tensor is not None:
            return get_backend_by_raw_tensor_type(type(arg.raw_tensor))
    return global_backend.__class__


class TensorWrapper:
    """
    Wraps :class:`Tensor`.
    Using weakref for the ``raw_tensor`` only, not for the Tensor itself:
    the entry lives as long as the raw tensor, also when the Tensor was only created for the lookup.
    Equality is given if the identity of the raw_tensor is the same,
    and the dims, the sparse dim, the feature axis and the version are equal (dims via :class:`DimWrapper`).
    (The dtype is given by the raw tensor.)
    No value of the tensor is checked.
    """

    def __init__(self, value: Tensor, *, finalize_callback, collected_dim_map: Dict[Dim, DimWrapper]):
        self.raw_value_ref = _WeakIdRef(value.raw_tensor, finalize_callback)
        self.dims = tuple(
            _transform_key_item(dim, finalize_callback=finalize_callback, collected_dim_map=collected_dim_map)
            for dim in value.dims
        )
        self.sparse_dim = None
        if value.sparse_dim is not None:
            self.sparse_dim = _transform_key_item(
                value.sparse_dim, finalize_callback=finalize_callback, collected_dim_map=collected_dim_map
            )
        self.feature_dim_axis = value.feature_dim_axis
        self.version = value.version

    def _key(self) -> Tuple[Any, ...]:
        return self.raw_value_ref, self.dims, self.sparse_dim, self.feature_dim_axis, self.version

    def __eq__(self, other):
        if isinstance(other, TensorWrapper):
            return self._key() == other._key()
        return False

    def __hash__(self):
        return hash(self._key())

    def get_dim_wrappers(self) -> Tuple[DimWrapper, ...]:
        """
        :return: the dims and the sparse dim, e.g. to map the cached output to the dims of the query
        """
        return self.dims + ((self.sparse_dim,) if self.sparse_dim is not None else ())


class _WeakIdRef:
    """
    Weakref with identity-based eq/hash.
    Dead refs never compare equal -- also makes id() reuse after GC safe
    (a new object at the same address hits the old hash but fails the eq check).
    """

    __slots__ = ("_ref", "_id")

    def __init__(self, obj, callback=None):
        self._ref = ref(obj, callback)
        self._id = id(obj)

    def __eq__(self, other):
        if not isinstance(other, _WeakIdRef):
            return False
        if self._id != other._id:
            return False
        obj = self._ref()
        return obj is not None and obj is other._ref()

    def __hash__(self):
        return self._id


class DimWrapper:
    """
    Wraps :class:`Dim`.
    Using weakref for the dim.
    If the size is known (static, or eager scalar, or eager seq lens),
    equality is given when the size values are equal (and the dim tag is ignored);
    :func:`Cache.get` then maps the cached output back to the queried dim.
    """

    def __init__(self, dim: Dim, *, finalize_callback):
        self.dim_value = _dim_value_for_key(dim, finalize_callback=finalize_callback)
        # dim_value covers every dim with a dyn size (value key or size_ref identity key),
        # incl. finalization via the raw size tensor.
        # The dim ref fallback remains only for dims without any size info
        # (e.g. a batch dim without dyn_size_ext);
        # finalize via the dim itself then.
        self.dim_ref = ref(dim, finalize_callback if self.dim_value is None else None)
        self._hash = hash(dim) if self.dim_value is None else hash(self.dim_value)

    def __eq__(self, other):
        if not isinstance(other, DimWrapper):
            return False
        if self.dim_value is not None or other.dim_value is not None:
            # value key or size_ref identity key.
            # None on one side compares unequal to any key, as it must.
            return self.dim_value == other.dim_value
        # No size info at all on both sides: dim identity is all there is.
        return self.dim_ref() == other.dim_ref()

    def __hash__(self):
        return self._hash


def _dim_value_for_key(dim: Dim, *, finalize_callback=None) -> Optional[Tuple[Any, ...]]:
    # Every key includes the capacity:
    # cached artifacts can be capacity-shaped (e.g. bound-regime range/mask tensors),
    # so dims with equal sizes but different capacities must not share an entry.
    if dim.size is not None:
        return dim.size, dim.capacity
    if dim.dyn_size_ext is None or dim.dyn_size_ext.raw_tensor is None:
        return None
    # noinspection PyProtectedMember
    if (
        not dim.dyn_size_ext._raw_backend.executing_eagerly()
        or dim.dyn_size_ext.device not in (None, "cpu")
        or rf.is_static_traceable()
    ):
        # No value-keying possible here:
        # - non-eager (graph-based) backends: symbolic size tensors have no values at trace time;
        # - device-resident dyn sizes: reading the values means a device-to-host copy
        #   (a sync, illegal under CUDA-graph capture; also, the values change in place there).
        # - static tracing: the size tensors are trace inputs even when on cpu --
        #   values read at trace time would be baked into the key and the cached graph artifact.
        #   Identity-keying within one trace is just common-subexpression reuse, value-free, correct.
        # Key by the IDENTITY of the raw size tensor instead (value-free, so still correct):
        # dims sharing the same size tensor (e.g. fresh kv dims from replace_dim)
        # then share the cache entry, as with value-keying in the eager cpu case.
        return "size_ref", _WeakIdRef(dim.dyn_size_ext.raw_tensor, finalize_callback), dim.capacity
    # Eager dyn sizes (e.g. seq lens [batch], or a scalar size): key by the size values
    # (not get_dim_value, which is the capacity if there is one),
    # so dims with equal lens share the cache entry
    # (e.g. the fresh kv dim which attention creates per layer via replace_dim).
    # The cached output is mapped back to the queried dim in Cache.get (dim_map).
    # get_dyn_size_ext_for_device caches the cpu copy on the dim,
    # so a potential device transfer happens only once per dim.
    size_ext = dim.get_dyn_size_ext_for_device("cpu")
    if size_ext.raw_tensor is None:
        # No cpu values available: identity of the raw size tensor, as above.
        return "size_ref", _WeakIdRef(dim.dyn_size_ext.raw_tensor, finalize_callback), dim.capacity
    # noinspection PyProtectedMember
    values = size_ext._raw_backend.raw_to_numpy(size_ext.raw_tensor)
    return make_hashable(values.tolist()), dim.capacity


# For now... we might extend it by some more types.
_KeyItemType = Union[None, str, bool, int, float, TensorWrapper, DimWrapper, Type[Backend]]
_RawTypes = (type(None), str, bool, int, float)
