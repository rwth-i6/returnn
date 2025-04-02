"""
:func:`lru_cache`, copied from Python functools, slightly adapted,
and extended by functions to check whether some key is cached or not.
"""

from __future__ import annotations
from typing import Dict, Any
from functools import update_wrapper
from threading import RLock
from collections import namedtuple


def lru_cache(maxsize: int = 128, typed: bool = False):
    """Least-recently-used cache decorator.

    If *maxsize* is set to None, the LRU features are disabled and the cache
    can grow without bound.

    If *typed* is True, arguments of different types will be cached separately.
    For example, f(3.0) and f(3) will be treated as distinct calls with
    distinct results.

    Arguments to the cached function must be hashable.

    Use f.cache_len() to see the current size of the cache.
    Use f.cache_set(*args, result, **kwargs) to set a value in the cache directly.
    Use f.cache_peek(*args, update_statistics=False, fallback=None, **kwargs)
    to peek the cache, without ever calling the user function.
    View the cache statistics named tuple (hits, misses, maxsize, currsize)
    with f.cache_info().
    Clear the cache and statistics with f.cache_clear().
    Remove the oldest entry from the cache with f.cache_pop_oldest().
    Take out some entry from the cache with f.cache_pop(*args, fallback=not_specified, **kwargs).
    Set the maximum cache size to a new value with f.cache_set_maxsize(new_maxsize).
    Access the underlying function with f.__wrapped__.

    See:  https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)

    """

    # Users should only access the lru_cache through its public API:
    #       cache_info, cache_clear, and f.__wrapped__
    # The internals of the lru_cache are encapsulated for thread safety and
    # to allow the implementation to change (including a possible C version).

    if isinstance(maxsize, int):
        assert maxsize >= 0
    elif callable(maxsize) and isinstance(typed, bool):
        # The user_function was passed in directly via the maxsize argument
        user_function, maxsize = maxsize, 128
        return _lru_cache_wrapper(user_function, maxsize, typed)
    elif maxsize is not None:
        raise TypeError("Expected first argument to be an integer, a callable, or None")

    # noinspection PyShadowingNames
    def _decorating_function(user_function):
        return _lru_cache_wrapper(user_function, maxsize, typed)

    return _decorating_function


_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


def _lru_cache_wrapper(user_function, maxsize: int, typed: bool):
    # Constants shared by all lru cache instances:
    make_key = _make_key  # build a key from the function arguments
    # noinspection PyPep8Naming
    PREV, NEXT, KEY, RESULT = 0, 1, 2, 3  # names for the link fields

    cache: Dict[Any, list] = {}
    hits = misses = 0
    full = False
    cache_get = cache.get  # bound method to lookup a key or return None
    cache_len = cache.__len__  # get cache size without calling len()
    lock = RLock()  # because linkedlist updates aren't threadsafe
    root = []  # root of the circular doubly linked list
    root[:] = [root, root, None, None]  # initialize by pointing to self

    assert maxsize >= 0

    def wrapper(*args, **kwds):
        """
        User-facing wrapper function.
        """
        # Size limited caching that tracks accesses by recency
        nonlocal root, hits, misses, full
        key = make_key(args, kwds, typed)
        with lock:
            link = cache_get(key)
            if link is not None:
                # Move the link to the front of the circular queue
                link_prev, link_next, _key, result = link
                link_prev[NEXT] = link_next
                link_next[PREV] = link_prev
                last = root[PREV]
                last[NEXT] = root[PREV] = link
                link[PREV] = last
                link[NEXT] = root
                hits += 1
                return result
            misses += 1
        result = user_function(*args, **kwds)
        if maxsize > 0:
            _cache_insert(key, result)
        return result

    def _cache_insert(key, result):
        nonlocal root, full
        with lock:
            if key in cache:
                # Getting here means that this same key was added to the
                # cache while the lock was released.  Since the link
                # update is already done, we need only return the
                # computed result and update the count of misses.
                pass
            elif full:
                # Use the old root to store the new key and result.
                oldroot = root
                oldroot[KEY] = key
                oldroot[RESULT] = result
                # Empty the oldest link and make it the new root.
                # Keep a reference to the old key and old result to
                # prevent their ref counts from going to zero during the
                # update. That will prevent potentially arbitrary object
                # clean-up code (i.e. __del__) from running while we're
                # still adjusting the links.
                root = oldroot[NEXT]
                oldkey = root[KEY]
                root[KEY] = root[RESULT] = None
                # Now update the cache dictionary.
                del cache[oldkey]
                # Save the potentially reentrant cache[key] assignment
                # for last, after the root and links have been put in
                # a consistent state.
                cache[key] = oldroot
            else:
                # Put result in a new link at the front of the queue.
                last = root[PREV]
                link = [last, root, key, result]
                last[NEXT] = root[PREV] = cache[key] = link
                # Use the cache_len bound method instead of the len() function
                # which could potentially be wrapped in an lru_cache itself.
                full = cache_len() >= maxsize

    def cache_info():
        """Report cache statistics"""
        with lock:
            return _CacheInfo(hits, misses, maxsize, cache_len())

    def cache_clear():
        """Clear the cache and cache statistics"""
        nonlocal hits, misses, full
        with lock:
            for link in cache.values():
                link.clear()  # make GC happy
            cache.clear()
            root[:] = [root, root, None, None]
            hits = misses = 0
            full = False

    def cache_parameters():
        """
        :return: parameters (maxsize, typed) of the cache as dict
        """
        return {"maxsize": maxsize, "typed": typed}

    def cache_set(*args, result, **kwargs):
        """
        Sets a value in the cache directly.
        """
        nonlocal root, full
        if maxsize > 0:
            key = make_key(args, kwargs, typed)
            _cache_insert(key, result)

    def cache_peek(*args, update_statistics: bool = True, fallback=None, **kwargs):
        """
        Peeks the cache without ever calling the user function.
        """
        nonlocal hits, misses
        key = make_key(args, kwargs, typed)
        with lock:
            link = cache_get(key)
            if link is not None:
                if update_statistics:
                    hits += 1
                return link[RESULT]
            if update_statistics:
                misses += 1
            return fallback

    not_specified = object()

    def cache_pop(*args, fallback=not_specified, **kwargs):
        """
        Removes the entry from the cache.
        """
        nonlocal hits, misses, full
        key = make_key(args, kwargs, typed)
        with lock:
            link = cache_get(key)
            if link is not None:
                # Take out link.
                link[PREV][NEXT] = link[NEXT]
                link[NEXT][PREV] = link[PREV]
                oldkey = link[KEY]
                oldvalue = link[RESULT]
                link.clear()
                del cache[oldkey]
                full = cache_len() >= maxsize
                return oldvalue
            if fallback is not_specified:
                raise KeyError("key not found")
            return fallback

    def cache_pop_oldest(*, fallback=not_specified):
        """
        Removes the oldest entry from the cache.
        """
        nonlocal root, full
        with lock:
            if not cache:
                if fallback is not_specified:
                    raise KeyError("cache is empty")
                return fallback
            assert cache
            # Take out oldest link.
            link: list = root[NEXT]
            link[NEXT][PREV] = root
            root[NEXT] = link[NEXT]
            oldkey = link[KEY]
            oldvalue = link[RESULT]
            link.clear()
            del cache[oldkey]
            full = cache_len() >= maxsize
            return oldvalue

    def cache_set_maxsize(new_maxsize: int):
        """
        Resets the maxsize.
        If the new maxsize is smaller than the current cache size, the oldest entries are removed.
        """
        nonlocal maxsize, full
        assert new_maxsize >= 0
        with lock:
            maxsize = new_maxsize
            while cache_len() > maxsize:
                cache_pop_oldest()
            full = cache_len() >= maxsize

    wrapper.cache_info = cache_info
    wrapper.cache_clear = cache_clear
    wrapper.cache_parameters = cache_parameters
    wrapper.cache_set = cache_set
    wrapper.cache_peek = cache_peek
    wrapper.cache_pop = cache_pop
    wrapper.cache_len = cache_len
    wrapper.cache_pop_oldest = cache_pop_oldest
    wrapper.cache_set_maxsize = cache_set_maxsize

    update_wrapper(wrapper, user_function)

    return wrapper


def _make_key(args, kwds, typed, *, _kwd_mark=(object(),), _fasttypes=(int, str), _tuple=tuple, _type=type, _len=len):
    """Make a cache key from optionally typed positional and keyword arguments

    The key is constructed in a way that is flat as possible rather than
    as a nested structure that would take more memory.

    If there is only a single argument and its data type is known to cache
    its hash value, then that argument is returned without a wrapper.  This
    saves space and improves lookup speed.

    """
    # All of code below relies on kwds preserving the order input by the user.
    # Formerly, we sorted() the kwds before looping.  The new way is *much*
    # faster; however, it means that f(x=1, y=2) will now be treated as a
    # distinct call from f(y=2, x=1) which will be cached separately.
    key = args
    if kwds:
        key += _kwd_mark
        for item in kwds.items():
            key += item
    if typed:
        key += _tuple(_type(v) for v in args)  # noqa
        if kwds:
            key += _tuple(_type(v) for v in kwds.values())  # noqa
    elif _len(key) == 1 and _type(key[0]) in _fasttypes:
        return key[0]
    return _HashedSeq(key)


class _HashedSeq(list):
    """
    This class guarantees that hash() will be called no more than once
    per element.  This is important because the lru_cache() will hash
    the key multiple times on a cache miss.
    """

    __slots__ = ("hashvalue",)

    def __init__(self, tup, *, _hash=hash):
        super().__init__(tup)
        self.hashvalue = _hash(tup)

    def __hash__(self):
        return self.hashvalue
