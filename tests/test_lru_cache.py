"""
Tests for :mod:`returnn.util.lru_cache`.
"""

from __future__ import annotations

import sys

import _setup_test_env  # noqa
from returnn.util import lru_cache as lru_cache_mod
from returnn.util.lru_cache import lru_cache


def test_lru_cache_keys_survive_module_teardown():
    """
    A weakref finalizer registered by returnn.frontend._cache pops its entry when the key's tensor dies,
    which at interpreter shutdown happens after the module globals were set to None. The key builder then
    saw ``'NoneType' object is not callable`` on every training job's exit.
    """
    cache = lru_cache(4)(lambda *args: None)
    cache.cache_set("a", 1, result="one")
    saved = lru_cache_mod._HashedSeq
    lru_cache_mod._HashedSeq = None
    try:
        assert cache.cache_pop("a", 1) == "one"
        assert cache.cache_peek("a", 1, fallback="gone") == "gone"
    finally:
        lru_cache_mod._HashedSeq = saved


if __name__ == "__main__":
    if len(sys.argv) > 1:
        globals()[sys.argv[1]]()
    else:
        for name, func in sorted(globals().items()):
            if name.startswith("test_"):
                print(f"-- {name}")
                func()
        print("all passed")
