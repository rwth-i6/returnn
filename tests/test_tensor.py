"""
tests for returnn.tensor
"""

import numpy
import _setup_test_env  # noqa
import sys
import unittest
from returnn.util import better_exchook
import returnn.frontend as rf
from returnn.tensor import Dim, Tensor, TensorDict


def test_tensor():
    batch_dim = Dim(name="batch", dimension=None)
    time_dim = Dim(name="time", dimension=None)
    feat_dim = Dim(10)
    x = Tensor("x", (batch_dim, time_dim, feat_dim), "float32")
    print(x)


# Note: Some of the dim math tests are duplicated from test_TFUtil.py, and partly adapted.
# But here they are TF-independent.


def test_dim_math_basics():
    a = Dim(None, name="a")
    b = Dim(None, name="b")
    assert a == a
    assert (a + 2 - 2) == a
    assert a + b == a + b
    assert a + b != b + a  # not commutative
    assert a * b == a * b
    assert a * b != b * a  # not commutative
    assert 2 * a == a + a
    assert 3 * a == a + a + a
    assert a * 2 != 2 * a
    assert 2 * a + b == a + a + b
    assert a + b - b == a
    assert a + 2 * b - b + -b == a
    assert a * b + b == (a + 1) * b
    assert (a + b) * 2 == a * 2 + b * 2
    assert 0 + a + 0 == a
    assert sum([0, a, 0, a, 0]) == 2 * a


def test_dim_math_neq_after_inc():
    a = Dim(0, name="a")
    b = Dim(0, name="a")  # same name, but different instance, intentionally to trigger potential issues
    assert a != b
    assert a + 1 != b + 1
    # Note: We had the bug that _representative_tag selected an auto-generated dim
    # (the unnamed "1" dim is auto-generated),
    # and that was used as derived_from_tag, and that triggered that the resulting dim was also auto-generated,
    # and for auto-generated dims, we allow dim equality also by name.
    # The fix was: _representative_tag will always prefer a non-auto-generated dim if there is any.
    assert a + 1 + 1 != b + 1 + 1


def test_dim_math_double_neg():
    a = Dim(None, name="a")
    assert --a == a


def test_dim_math_mul_rmul():
    a = Dim(None, name="a")
    b = a * 3
    c = 2 * b
    assert c == 2 * (a * 3)
    assert c != a * 6
    assert c == (2 * a) * 3


def test_dim_math_mul_div():
    a = Dim(None, name="a")
    b = Dim(None, name="b")
    assert (a * b) // b == a
    assert (b * a) // b != a
    assert (b * a).div_left(b) == a


def test_dim_math_div_mul():
    a = Dim(None, name="a")
    b = Dim(None, name="b")
    assert a // b == a // b


def test_dim_math_div_div():
    a = Dim(None, name="a")
    b = a.ceildiv_right(2)
    b = b.ceildiv_right(3)
    c = a.ceildiv_right(6)
    print(a, b, c)
    assert b == c


def test_dim_math_pad_conv():
    time = Dim(None, name="time")
    padded = 2 + time + 2
    assert padded == 2 + time + 2
    conv_valid = (-2) + padded + (-2)
    assert conv_valid == time


def test_dim_math_pad_window():
    rf.select_backend("numpy")
    batch_dim = Dim(3, name="batch")
    time1_dim = Dim(rf.convert_to_tensor([5, 6, 7], dims=[batch_dim], name="time1"))
    time2_dim = Dim(rf.convert_to_tensor([20, 21, 22], dims=[batch_dim], name="time2"))
    time3_dim = Dim(rf.convert_to_tensor([10, 5, 3], dims=[batch_dim], name="time3"))
    in_spatial_dim = time1_dim + time2_dim + time3_dim
    filter_size = 17
    # As it would happen with conv/pool/window/... with window size 17.
    out_spatial_dim = in_spatial_dim.sub_left(8).sub_right(8)
    print("out_spatial_dim:", out_spatial_dim)
    sizes = out_spatial_dim.dyn_size.tolist()
    print("sizes:", sizes)
    expected_sizes = (time1_dim.dyn_size + time2_dim.dyn_size + time3_dim.dyn_size - (filter_size - 1)).tolist()
    print("expected_sizes:", expected_sizes)
    assert sizes == expected_sizes


class _EqKey:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return isinstance(other, _EqKey) and other.value == self.value

    def __hash__(self):
        return hash(self.value)


class _AnyValue:
    pass


def test_Dim_math_cache_dict_equal_key_replace():
    import gc

    from returnn.tensor._dim_extra import _WeakKeyWeakValueDict

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        d = _WeakKeyWeakValueDict()
        key1 = _EqKey(1)
        key2 = _EqKey(1)
        value1 = _AnyValue()
        d[key1] = value1
        del value1
        value2 = _AnyValue()
        d[key2] = value2
        del key1
        assert d.get(key2) is value2, "live entry must survive the death of an equal old key"
    finally:
        if gc_was_enabled:
            gc.enable()


def test_Dim_math_cache_dict_prunes_dead():
    import gc

    from returnn.tensor._dim_extra import _WeakKeyWeakValueDict

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        d = _WeakKeyWeakValueDict()
        keys = [_EqKey(i) for i in range(10)]
        for key in keys:
            value = _AnyValue()
            d[key] = value
            del value
        assert len(d) == 0
        key_live = _EqKey("live")
        value_live = _AnyValue()
        d[key_live] = value_live
        assert len(d._entries) == 1, "dead entries must be pruned on insert"
        assert d.get(key_live) is value_live
    finally:
        if gc_was_enabled:
            gc.enable()


def test_Dim_math_cache_hash_change_after_declare_same_as():
    p = Dim(3, name="p")
    b = Dim(4, name="b")
    r = p + b
    del r
    b.declare_same_as(Dim(4, name="base"))
    d = p + Dim(5, name="c")
    assert d.dimension == 8


def test_Dim_math_cache_keys_becoming_equal():
    p = Dim(10, name="p")
    static = Dim(3, name="static")
    dynamic = Dim(None, name="dynamic")
    static_result = p + static
    assert static_result.dimension == 13
    dynamic_result = p + dynamic
    assert dynamic_result.dimension is None
    static.declare_same_as(dynamic)
    assert static == dynamic
    p.reset_raw()
    assert (p + static).dimension == 13


def test_Dim_derived_pickle():
    import pickle

    a = Dim(3, name="a")
    d = a + 1
    d2 = pickle.loads(pickle.dumps(d))
    assert d2.dimension == d.dimension == 4
    assert d2.derived_from_op is not None
    assert d2.derived_from_op.kind == d.derived_from_op.kind


def test_Dim_math_cache_weak():
    import gc
    import weakref

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        a = Dim(3, name="a")
        b = Dim(4, name="b")
        d_static = a + 1
        ref_static = weakref.ref(d_static)
        del d_static
        assert ref_static() is None, "dim + int derived dim must be freed by refcount"
        d_dyn = a + b
        ref_dyn = weakref.ref(d_dyn)
        del d_dyn
        assert ref_dyn() is None, "dim + dim derived dim must be freed by refcount"
        assert (a + 1) == (a + 1)
        assert (a + b) == (a + b)
    finally:
        if gc_was_enabled:
            gc.enable()


def test_Dim_get_mask_cache_identity():
    batch_dim = Dim(2, name="batch")
    time_dim = Dim(
        Tensor("time", [batch_dim], dtype="int32", raw_tensor=numpy.array([3, 2], dtype="int32")),
        name="time",
    )
    mask1 = time_dim.get_mask(device="cpu")
    mask2 = time_dim.get_mask(device="cpu")
    assert mask1 is mask2  # cached wrapper, stable identity while alive
    raw1 = mask1.raw_tensor
    del mask1, mask2
    mask3 = time_dim.get_mask(device="cpu")
    assert set(mask3.dims) == {batch_dim, time_dim}
    assert mask3.raw_tensor is raw1  # raw tensor stays cached across wrapper rebuilds


def test_Dim_get_mask_no_cycle():
    import gc
    import weakref

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        batch_dim = Dim(2, name="batch")
        time_dim = Dim(
            Tensor("time", [batch_dim], dtype="int32", raw_tensor=numpy.array([3, 2], dtype="int32")),
            name="time",
        )
        mask = time_dim.get_mask(device="cpu")
        ref_mask_raw = weakref.ref(mask.raw_tensor)
        ref_dim = weakref.ref(time_dim)
        del mask
        del time_dim
        assert ref_dim() is None, "dim with cached mask must be freed by refcount"
        assert ref_mask_raw() is None, "cached mask raw tensor must die with the dim"
    finally:
        if gc_was_enabled:
            gc.enable()


def test_Tensor_pickle():
    import pickle

    tensor = Tensor("data", dims=[Dim(10)], dtype="int32", raw_tensor=numpy.zeros((10,), dtype="int32"))

    s = pickle.dumps(tensor)
    tensor2: Tensor = pickle.loads(s)

    assert tensor.name == tensor2.name
    assert tensor.dtype == tensor2.dtype
    assert tensor.raw_tensor is not None
    assert tensor2.raw_tensor is not None
    assert numpy.array_equal(tensor.raw_tensor, tensor2.raw_tensor)


def test_TensorDict_pickle():
    import pickle

    tensor_dict = TensorDict(
        {"data": Tensor("data", dims=[Dim(10)], dtype="int32", raw_tensor=numpy.zeros((10,), dtype="int32"))}
    )
    assert all(tensor.raw_tensor is not None for tensor in tensor_dict.data.values())

    s = pickle.dumps(tensor_dict)
    tensor_dict2 = pickle.loads(s)

    assert tensor_dict.data.keys() == tensor_dict2.data.keys()
    assert all(tensor.raw_tensor is not None for tensor in tensor_dict2.data.values())


def test_TensorDict_queue():
    from multiprocessing import Queue

    tensor_dict = TensorDict(
        {"data": Tensor("data", dims=[Dim(10)], dtype="int32", raw_tensor=numpy.zeros((10,), dtype="int32"))}
    )
    assert all(tensor.raw_tensor is not None for tensor in tensor_dict.data.values())

    q = Queue(maxsize=2)
    q.put(tensor_dict)
    tensor_dict2 = q.get()

    assert tensor_dict.data.keys() == tensor_dict2.data.keys()
    assert all(tensor.raw_tensor is not None for tensor in tensor_dict2.data.values())


if __name__ == "__main__":
    better_exchook.install()
    if len(sys.argv) <= 1:
        for k, v in sorted(globals().items()):
            if k.startswith("test_"):
                print("-" * 40)
                print("Executing: %s" % k)
                try:
                    v()
                except unittest.SkipTest as exc:
                    print("SkipTest:", exc)
                print("-" * 40)
        print("Finished all tests.")
    else:
        assert len(sys.argv) >= 2
        for arg in sys.argv[1:]:
            print("Executing: %s" % arg)
            if arg in globals():
                globals()[arg]()  # assume function and execute
            else:
                eval(arg)  # assume Python code and execute
