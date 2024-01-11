# start test like this:  nosetests-2.7  tests/test_TFUtil.py

from __future__ import annotations

import _setup_test_env  # noqa
import tensorflow as tf
from returnn.tf.util.basic import *
from returnn.tf.util.data import SpatialDim, FeatureDim
import returnn.tf.compat as tf_compat
from nose.tools import assert_equal, assert_not_equal, assert_is_instance, assert_is, assert_in, assert_true
from numpy.testing import assert_almost_equal, assert_allclose
from pprint import pprint
import contextlib
import unittest
import numpy.testing
from returnn.util import better_exchook


print("TF version:", tf.__version__)

session = tf_compat.v1.InteractiveSession()


def test_tf_version_tuple():
    print("TF version:", tf.__version__)
    print("TF version tuple:", tf_version_tuple())


def test_Data():
    data = Data(name="my_data", shape=(None, 13))
    assert_equal(data.name, "my_data")
    assert_equal(data.dim, 13)
    assert_equal(data.batch_dim_axis, 0)
    assert_equal(data.time_dim_axis, 1)
    assert_equal(data.feature_dim_axis, 2)
    assert_equal(data.batch_ndim, 3)
    assert_equal(data.batch_shape, (None, None, 13))
    assert_equal(data.dtype, "float32")
    assert_equal(data.sparse, False)


def test_Data_dim():
    data = Data(name="my_data", dim=13)
    assert_equal(data.name, "my_data")
    assert_equal(data.dim, 13)
    assert_equal(data.batch_dim_axis, 0)
    assert_equal(data.time_dim_axis, 1)
    assert_equal(data.feature_dim_axis, 2)
    assert_equal(data.batch_ndim, 3)
    assert_equal(data.batch_shape, (None, None, 13))
    assert_equal(data.dtype, "float32")
    assert_equal(data.sparse, False)


def test_Data_dim_none():
    data = Data(name="my_data", dim=None)
    assert_equal(data.dim, None)
    assert_equal(data.batch_dim_axis, 0)
    assert_equal(data.time_dim_axis, 1)
    assert_equal(data.feature_dim_axis, 2)
    assert_equal(data.batch_ndim, 3)
    assert_equal(data.batch_shape, (None, None, None))
    assert_equal(data.dtype, "float32")
    assert_equal(data.sparse, False)


def test_Data_dim_none_auto_create_placeholders():
    data = Data(name="my_data", dim=None, auto_create_placeholders=True)
    assert_equal(data.dim, None)
    assert_equal(data.batch_dim_axis, 0)
    assert_equal(data.time_dim_axis, 1)
    assert_equal(data.feature_dim_axis, 2)
    data_ = Data(name="my_data", dim=None)
    assert_equal(data.batch_ndim, 3)
    assert_equal(data.batch_shape, (None, None, None))
    assert_equal(data.dtype, "float32")
    assert_equal(data.sparse, False)
    assert (data.batch_dim_axis, data.time_dim_axis, data.feature_dim_axis) == (
        data_.batch_dim_axis,
        data_.time_dim_axis,
        data_.feature_dim_axis,
    )


def test_Data_default_time_no_time():
    # This is new behavior.
    data = Data(name="merge_dims_test_output", shape=(3, 5))
    assert data.time_dim_axis is None and data.feature_dim_axis == 2


def test_Data_copy_time_major():
    data = Data(name="my_data", dim=13)
    assert_equal(data.batch_dim_axis, 0)
    assert_equal(data.time_dim_axis, 1)
    assert_equal(data.feature_dim_axis, 2)
    assert_equal(data.batch_ndim, 3)
    data2 = data.copy_as_time_major()
    assert_equal(data2.time_dim_axis, 0)
    assert_equal(data2.batch_dim_axis, 1)
    assert_equal(data2.feature_dim_axis, 2)
    assert_equal(data2.batch_ndim, 3)


def test_Data_copy_batch_major():
    data = Data(name="my_data", dim=13, time_dim_axis=0, batch_dim_axis=1)
    assert_equal(data.time_dim_axis, 0)
    assert_equal(data.batch_dim_axis, 1)
    assert_equal(data.feature_dim_axis, 2)
    assert_equal(data.batch_ndim, 3)
    data2 = data.copy_as_batch_major()
    assert_equal(data2.batch_dim_axis, 0)
    assert_equal(data2.time_dim_axis, 1)
    assert_equal(data2.feature_dim_axis, 2)
    assert_equal(data2.batch_ndim, 3)


def test_Data_copy_as_batch_major_no_extra_feat():
    data = Data(name="att_weights_output", shape=(None,), batch_dim_axis=1)
    print("data", data, "feat axis:", data.feature_dim_axis_or_unspecified, data.feature_dim_axis)
    assert_equal(data.time_dim_axis, 0)
    data2 = data.copy_as_batch_major()
    assert_equal(data2.batch_dim_axis, 0)
    assert_equal(data2.time_dim_axis, 1)
    # No check for feature_dim_axis, as this behavior does not matter here.


def test_Data_spatial_batch_axes():
    d1 = Data(name="ff_out_prior_output", shape=(1, 9001), dtype="float32", batch_dim_axis=None)
    d2 = Data(name="ff_out_output", shape=(None, 9001), dtype="float32")
    spatial_axes1 = d1.get_spatial_batch_axes()
    spatial_axes2 = d2.get_spatial_batch_axes()
    assert_equal(len(spatial_axes1), len(spatial_axes2))
    spatial_axes1 = d1.get_spatial_axes()
    spatial_axes2 = d2.get_spatial_axes()
    assert_equal(len(spatial_axes1), len(d1.get_spatial_batch_axes()))
    assert_equal(spatial_axes1, spatial_axes2)


def test_Data_spatial_batch_axes_2():
    d = Data(name="data", shape=(None, 9000))
    assert_equal(d.get_spatial_batch_axes(), [1])
    d = Data(name="data", shape=(13, 9000))
    assert_equal(d.get_spatial_batch_axes(), [1])
    d = Data(name="data", shape=(None, 13, 9000))
    assert_equal(d.get_spatial_batch_axes(), [1, 2])


def test_Data_get_bc_spatial_batch_shape():
    d = Data(name="data", shape=(None, 9000))
    assert_equal(d.get_bc_spatial_batch_shape(), (1, 1, 9000))
    d = Data(name="data", shape=(13, 9000))
    assert_equal(d.get_bc_spatial_batch_shape(), (1, 1, 9000))
    d = Data(name="data", shape=(None, 13, 9000))
    assert_equal(d.get_bc_spatial_batch_shape(), (1, 1, 1, 9000))


def test_Data_get_bc_shape():
    d = Data(name="data", shape=(None, 9000))
    assert_equal(d.get_bc_shape(), (1, 1, 9000))
    d = Data(name="data", shape=(13, 9000))
    assert_equal(d.get_bc_shape(), (1, 1, 9000))
    d = Data(name="data", shape=(None, 13, 9000))
    assert_equal(d.get_bc_shape(), (1, 1, 1, 9000))
    d = Data(name="data", shape=(None, 13, 9000))
    assert_equal(d.get_bc_shape({"*": None}), (None, None, 13, 9000))
    assert_equal(d.get_bc_shape({("B", "dim:13"): None}), (None, 1, 13, 9000))


def test_Data_copy_template_adding_time_dim_no_feature():
    d1 = Data(name="d1", shape=(), time_dim_axis=None)
    assert d1.batch_dim_axis == 0 and d1.batch_shape == (None,)
    assert d1.feature_dim_axis is None
    d2 = d1.copy_template_adding_time_dim()
    assert d2.batch_dim_axis == 1 and d2.time_dim_axis == 0 and d2.batch_shape == (None, None)
    # assert d2.feature_dim_axis is None  # not sure what we would want here...


def test_Data_copy_template_adding_time_dim_no_batch():
    d1 = Data(name="d1", shape=(), dtype="int32", batch_dim_axis=None, time_dim_axis=None)
    assert d1.batch_dim_axis is None and d1.batch_shape == ()
    assert d1.feature_dim_axis is None
    d2 = d1.copy_template_adding_time_dim()
    assert d2.batch_dim_axis is None and d2.time_dim_axis == 0 and d2.batch_shape == (None,)


def test_Data_get_axes_from_description_except_time_ext():
    data = Data(name="merge_dims_test_output", shape=(3, None, 5), time_dim_axis=2)
    axes = data.get_axes_from_description("except_time")
    assert axes == [1, 3], "data %r 'except_time' axes %r unexpected" % (data, axes)


def test_Data_get_axes_from_description_except_time_no_time():
    data = Data(name="merge_dims_test_output", shape=(3, 5))
    assert data.time_dim_axis is None
    axes = data.get_axes_from_description("except_time")
    assert axes == [1, 2], "data %r 'except_time' axes %r unexpected" % (data, axes)


def test_Data_copy_template_excluding_time_dim_two_time_dims():
    data = Data(name="ref_att_weights_output", shape=(None, None, 1), auto_create_placeholders=True)
    assert set(data.size_placeholder.keys()) == {0, 1}
    data_wo_time = data.copy_template_excluding_time_dim()
    assert data_wo_time.shape == (None, 1) and data_wo_time.have_time_axis()


def test_Data_time_no_feature():
    d1 = Data(name="d1", shape=(None,), batch_dim_axis=0, time_dim_axis=1, dim=None)
    assert d1.time_dim_axis == 1


def test_Data_unknown_feature_no_time():
    d1 = Data(name="d1", shape=(None,), batch_dim_axis=0, time_dim_axis=None, dim=None)
    assert d1.batch_dim_axis == 0 and d1.time_dim_axis is None and d1.feature_dim_axis == 1
    assert d1.batch_shape == (None, None)


def test_Data_time_end():
    data = Data(name="att_weights_output", shape=(1, None), time_dim_axis=2)
    print("data:", data, "feature axis:", data.feature_dim_axis)
    assert data.shape == (1, None) and data.batch_dim_axis == 0 and data.time_dim_axis == 2
    # No test for feature axis, as it does not really matter.


def test_Data_copy_with_feature_dim_axis_case_1():
    # Case 1: new_feature_dim_axis <= time_dim_axis < old_feature_dim_axis
    import numpy as np

    size_placeholder = tf.constant(np.full((10,), 10), dtype=tf.int32)
    d = Data(
        name="test_data",
        shape=(None, 13, 17),
        dtype="float32",
        size_placeholder={0: size_placeholder},
        batch_dim_axis=0,
        time_dim_axis=1,
        feature_dim_axis=3,
    )
    assert list(d.size_placeholder.keys()) == [0]
    d_copy = d.copy_with_feature_dim_axis(1)
    assert d_copy.shape == (17, None, 13)
    assert d_copy.batch_dim_axis == 0
    assert d_copy.time_dim_axis == 2
    assert d_copy.feature_dim_axis == 1
    assert list(d_copy.size_placeholder.keys()) == [1]


def test_Data_copy_with_feature_dim_axis_case_2():
    # Case 2: time_dim_axis < old_feature_dim_axis < new_feature_dim_axis
    import numpy as np

    size_placeholder = tf.constant(np.full((10,), 10), dtype=tf.int32)
    d = Data(
        name="test_data",
        shape=(None, 17, 13),
        dtype="float32",
        size_placeholder={0: size_placeholder},
        batch_dim_axis=0,
        time_dim_axis=1,
        feature_dim_axis=2,
    )
    d_copy = d.copy_with_feature_dim_axis(-1)
    assert d_copy.shape == (None, 13, 17)
    assert d_copy.batch_dim_axis == 0
    assert d_copy.time_dim_axis == 1
    assert d_copy.feature_dim_axis == 3
    assert list(d_copy.size_placeholder.keys()) == [0]


def test_Data_copy_with_feature_dim_axis_case_3():
    # Case 3: time_dim_axis < new_feature_dim_axis < old_feature_dim_axis
    import numpy as np

    size_placeholder = tf.constant(np.full((10,), 10), dtype=tf.int32)
    d = Data(
        name="test_data",
        shape=(None, 13, 17),
        dtype="float32",
        size_placeholder={0: size_placeholder},
        batch_dim_axis=0,
        time_dim_axis=1,
        feature_dim_axis=3,
    )
    d_copy = d.copy_with_feature_dim_axis(2)
    assert d_copy.shape == (None, 17, 13)
    assert d_copy.batch_dim_axis == 0
    assert d_copy.time_dim_axis == 1
    assert d_copy.feature_dim_axis == 2
    assert list(d_copy.size_placeholder.keys()) == [0]


def test_Data_copy_with_feature_dim_axis_case_4():
    # Case 4: new_feature_dim_axis < old_feature_dim_axis < time_dim_axis
    import numpy as np

    size_placeholder = tf.constant(np.full((10,), 10), dtype=tf.int32)
    d = Data(
        name="test_data",
        shape=(13, 17, None),
        dtype="float32",
        size_placeholder={2: size_placeholder},
        batch_dim_axis=0,
        time_dim_axis=3,
        feature_dim_axis=2,
    )
    d_copy = d.copy_with_feature_dim_axis(1)
    assert d_copy.shape == (17, 13, None)
    assert d_copy.batch_dim_axis == 0
    assert d_copy.time_dim_axis == 3
    assert d_copy.feature_dim_axis == 1
    assert list(d_copy.size_placeholder.keys()) == [2]


def test_Data_copy_with_feature_dim_axis_case_5():
    # Case 5: old_feature_dim_axis < new_feature_dim_axis <= time_dim_axis
    import numpy as np

    size_placeholder = tf.constant(np.full((10,), 10), dtype=tf.int32)
    d = Data(
        name="test_data",
        shape=(17, 13, None),
        dtype="float32",
        size_placeholder={2: size_placeholder},
        batch_dim_axis=0,
        time_dim_axis=3,
        feature_dim_axis=1,
    )
    d_copy = d.copy_with_feature_dim_axis(-1)
    assert d_copy.shape == (13, None, 17)
    assert d_copy.batch_dim_axis == 0
    assert d_copy.time_dim_axis == 2
    assert d_copy.feature_dim_axis == 3
    assert list(d_copy.size_placeholder.keys()) == [1]


def test_Data_copy_with_feature_dim_axis_case_6():
    # Case 6: old_feature_dim_axis < time_dim_axis < new_feature_dim_axis
    import numpy as np

    size_placeholder = tf.constant(np.full((10,), 10), dtype=tf.int32)
    d = Data(
        name="test_data",
        shape=(17, None, 13),
        dtype="float32",
        size_placeholder={1: size_placeholder},
        batch_dim_axis=0,
        time_dim_axis=2,
        feature_dim_axis=1,
    )
    d_copy = d.copy_with_feature_dim_axis(-1)
    assert d_copy.shape == (None, 13, 17)
    assert d_copy.batch_dim_axis == 0
    assert d_copy.time_dim_axis == 1
    assert d_copy.feature_dim_axis == 3
    assert list(d_copy.size_placeholder.keys()) == [0]


def test_Data_copy_with_feature_dim_axis_case_7():
    # Case 7:
    d = Data(name="test_data", shape=(None, 17, 13))
    assert d.feature_dim_axis == 3
    d_copy = d.copy_with_feature_dim_axis(1)
    assert d_copy.shape == (13, None, 17)
    assert d_copy.feature_dim_axis == 1
    assert d_copy.dim == 13


def test_Data_copy_compatible_to_time_major():
    d1 = Data(name="ff_out_output", shape=(None, 9001), dtype="float32", batch_dim_axis=1)
    d2 = Data(name="ff_out_prior_output", shape=(9001,), dtype="float32", batch_dim_axis=None, time_dim_axis=None)
    d2a = d2.copy_compatible_to(d1)
    assert d2a.shape == (1, 9001)
    assert d2a.batch_dim_axis == d1.batch_dim_axis
    assert d2a.time_dim_axis == d1.time_dim_axis
    assert d2a.feature_dim_axis == d1.feature_dim_axis


def test_Data_find_matching_dim_map_different_static_dims():
    d1 = Data(name="p1_output", shape=(5, 5, 3), batch_dim_axis=None, time_dim_axis=None)  # [5,5,F|3]
    d2 = Data(name="p2_output", shape=(5, 1, 1), batch_dim_axis=None, time_dim_axis=None)  # [5,1,F|1]

    # without broadcast_matches=False should not match
    is_equal_opts = dict(allow_same_feature_dim=True, allow_same_spatial_dim=True, treat_feature_as_spatial=True)
    assert_equal(d1.find_matching_dims(d2.get_dim_tag(0), is_equal_opts=is_equal_opts), [0, 1])
    assert_equal(d1.find_matching_dims(d2.get_dim_tag(1), is_equal_opts=is_equal_opts), [])
    assert_equal(d1.find_matching_dims(d2.get_dim_tag(2), is_equal_opts=is_equal_opts), [])

    # with broadcast_matches=True should match
    is_equal_opts = dict(
        allow_same_feature_dim=True, allow_same_spatial_dim=True, treat_feature_as_spatial=True, broadcast_matches=True
    )
    assert_equal(d1.find_matching_dims(d2.get_dim_tag(0), is_equal_opts=is_equal_opts), [0, 1])
    assert_equal(d1.find_matching_dims(d2.get_dim_tag(1), is_equal_opts=is_equal_opts), [0, 1, 2])
    assert_equal(d1.find_matching_dims(d2.get_dim_tag(2), is_equal_opts=is_equal_opts), [0, 1, 2])

    mapping = d1.find_matching_dim_map(d2, list(range(d2.batch_ndim)))  # maps d2 -> d1
    assert len(mapping.values()) == d2.batch_ndim
    assert all(mapping[i] == i for i in range(d2.batch_ndim))

    d2_compatible = d2.copy_compatible_to(d1)
    assert d2_compatible.batch_shape == d2.batch_shape
    d1_compatible = d1.copy_compatible_to(d2)
    assert d1_compatible.batch_shape == d1.batch_shape


def test_Data_find_matching_dim_map_broadcast_matches():
    d1 = Data(name="d1", shape=(5, None), time_dim_axis=2)  # [B,F|5,T]
    d2 = Data(name="d2", shape=(5, 1), batch_dim_axis=None, time_dim_axis=None, feature_dim_axis=0)  # [F|5,1]
    print("d1:", d1)
    print("d2:", d2)

    # default should not match
    is_equal_opts = dict(allow_same_feature_dim=True, allow_same_spatial_dim=True, treat_feature_as_spatial=True)
    assert_equal(d1.find_matching_dims(d2.get_dim_tag(0), is_equal_opts=is_equal_opts), [1])
    assert_equal(d1.find_matching_dims(d2.get_dim_tag(1), is_equal_opts=is_equal_opts), [])

    # with broadcast_matches=True should match
    is_equal_opts_match = dict(
        allow_same_feature_dim=True, allow_same_spatial_dim=True, treat_feature_as_spatial=True, broadcast_matches=True
    )
    assert_equal(d1.find_matching_dims(d2.get_dim_tag(0), is_equal_opts=is_equal_opts_match), [1])
    assert_equal(d1.find_matching_dims(d2.get_dim_tag(1), is_equal_opts=is_equal_opts_match), [1, 2])

    mapping = d1.find_matching_dim_map(d2, list(range(d2.batch_ndim)), is_equal_opts)  # maps d2 -> d1
    assert mapping[0] == 1 and mapping[1] == 2

    copied = d2.copy_compatible_to(d1)
    assert copied.batch_ndim == d1.batch_ndim and copied.get_static_batch_dim() == 1 and copied.batch_shape == (1, 5, 1)
    print("copied compatible:", copied)


def test_Data_get_all_dimension_tags_same_spatial_dim_twice():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    in_dim = FeatureDim("input", 13)
    out_dim = FeatureDim("output", 13)
    source1 = Data("a", dim_tags=(batch_dim, time_dim, in_dim))
    source2 = Data("b", dim_tags=(in_dim, out_dim))
    all_dim_tags_std, _ = Dim.get_all_dimension_tags([source1, source2])
    assert all_dim_tags_std == [batch_dim, time_dim, in_dim, out_dim]
    # DotLayer._auto_var_axes but other code uses also similar is_equal_opts.
    # Also, while these is_equal_opts could be problematic maybe in other cases,
    # this is a case which should still be correct.
    is_equal_opts = dict(
        treat_feature_as_spatial=True, allow_same_spatial_dim=True, undefined_matches=True, derived_matches=True
    )
    all_dim_tags, _ = Dim.get_all_dimension_tags([source1, source2], is_equal_opts=is_equal_opts)
    assert all_dim_tags == [batch_dim, time_dim, in_dim, out_dim]


def test_Dim_get_all_dimension_tags_one_derived_time():
    from returnn.tf.util.data import batch_dim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feature", 3)
    a = Data(name="a", dim_tags=[batch_dim, time_dim, feat_dim])
    print("a:", a)
    derived_time_dim = SpatialDim("derived_time", derived_from_tag=time_dim)
    b = Data(name="b", dim_tags=[batch_dim, derived_time_dim, feat_dim])
    print("b:", b)
    # Data.get_common_data defaults:
    is_equal_opts = dict(
        ignore_feature_dim=False,
        treat_feature_as_spatial=True,
        allow_same_spatial_dim=True,
        undefined_matches=True,
        derived_matches=True,
    )
    all_dim_tags, _ = DimensionTag.get_all_dimension_tags([b, a], is_equal_opts=is_equal_opts)
    print("all_dim_tags:", all_dim_tags)
    # The is_equal_opts with derived_matches should match time_dim with derived_time_dim.
    # We explicitly want that the derived_time_dim is returned here.
    assert time_dim not in all_dim_tags and all_dim_tags == [batch_dim, derived_time_dim, feat_dim]


def test_Data_sparse_int32_with_dim_kwargs_init():
    data = Data(name="classes_with_dim", shape=(None,), dim=10, sparse=True, dtype="int32")
    assert data.sparse and data.have_time_axis() and data.shape == (None,) and data.dim == 10


def test_Data_sparse_with_dims():
    from returnn.tf.util.data import batch_dim, SpatialDim

    time_dim = SpatialDim("time")
    out = Data(name="out", sparse=True, dim=5, batch_dim_axis=0, time_dim_axis=1, dim_tags=(batch_dim, time_dim))
    assert out.sparse and out.dtype == "int32"


def test_Data_int32_no_dim_kwargs_init():
    data = Data(name="classes_with_no_dim", shape=(None,), dtype="int32")
    assert data.have_time_axis() and data.shape == (None,)


def test_Data_copy_template_excluding_spatial_dim():
    att_weights = Data(name="att_weights", shape=(None, None, 1), batch_dim_axis=2)
    rem_enc_time = att_weights.copy_template_excluding_spatial_dim(-1)
    assert rem_enc_time.shape == (None, 1) and rem_enc_time.batch_dim_axis == 1


def test_Data_copy_template_excluding_axis():
    data = Data(name="data", shape=(None, 8), batch_dim_axis=0, time_dim_axis=1, feature_dim_axis=2)
    data_wo_batch = data.copy_template_excluding_axis(data.batch_dim_axis)
    assert data_wo_batch.shape == (None, 8) and data_wo_batch.feature_dim_axis == 1 and data_wo_batch.time_dim_axis == 0
    data_wo_time = data.copy_template_excluding_axis(data.time_dim_axis)
    assert data_wo_time.shape == (8,) and data_wo_time.feature_dim_axis == 1 and data_wo_time.batch_dim_axis == 0
    data_wo_feature = data.copy_template_excluding_axis(data.feature_dim_axis)
    assert (
        data_wo_feature.shape == (None,) and data_wo_feature.time_dim_axis == 1 and data_wo_feature.batch_dim_axis == 0
    )


def test_Data_copy_squeeze_axes():
    weights = Data(name="att_weights_output", shape=(1, None), time_dim_axis=2, auto_create_placeholders=True)
    squeezed = weights.copy_squeeze_axes([1])
    print("orig:", weights, "squeezed:", squeezed)
    assert squeezed.shape == (None,) and squeezed.time_dim_axis == 1
    assert weights.size_placeholder[1] is squeezed.size_placeholder[0]


def test_Data_copy_squeeze_axes_feature_axis():
    weights = Data(name="att_weights_output", shape=(None, 1), auto_create_placeholders=True)
    squeezed = weights.copy_squeeze_axes([2])
    print("orig:", weights, "squeezed:", squeezed)
    assert squeezed.shape == (None,) and squeezed.time_dim_axis == 1
    assert weights.size_placeholder[0] is squeezed.size_placeholder[0]


def test_Data_copy_time_flattened():
    x = Data(name="x", shape=(None, 1031), batch_dim_axis=1, auto_create_placeholders=True)
    y = x.copy_time_flattened()


def test_ExternData_via_config():
    # Like ExternData.init_from_config.
    from returnn.config import Config
    from returnn.tf.network import _extern_data_types_from_config

    config = Config(
        {
            "extern_data": {
                "data": (40, 2),
                "classes": (10025, 1),
                "att_weights": {"shape": (None, None, 1)},
                "att_weights_sizes": {"shape": (None,), "dtype": "int32"},
            }
        }
    )
    data_dims = _extern_data_types_from_config(config)
    data = {}
    for key, init_args in data_dims.items():
        data[key] = Data(name=key, auto_create_placeholders=True, **init_args)
    pprint(data)
    data_data = data["data"]
    assert isinstance(data_data, Data)
    assert data_data.have_time_axis() and not data_data.sparse and data_data.shape == (None, 40)
    att_weights_sizes = data["att_weights_sizes"]
    assert isinstance(att_weights_sizes, Data)
    assert att_weights_sizes.have_time_axis()


def test_4D_Data_get_placeholder_flattened():
    import numpy as np

    size_placeholder = tf.constant(np.full((7,), 9), dtype=tf.int32)
    d = Data(
        name="test_data",
        shape=(None, 13, 17),
        dtype="float32",
        size_placeholder={0: size_placeholder},
        batch_dim_axis=0,
        time_dim_axis=1,
        feature_dim_axis=3,
    )
    placeholder = tf_compat.v1.placeholder(shape=(None, None, 13, 17), dtype=tf.float32)
    d.placeholder = placeholder
    feed_data = np.random.rand(7, 9, 13, 17)
    res = session.run(d.placeholder, feed_dict={placeholder: feed_data})
    print(res.shape)
    flat_placeholder = d.get_placeholder_flattened(keepdims=True)
    res = session.run(flat_placeholder, feed_dict={placeholder: feed_data})
    print(res.shape)
    assert res.shape[0] == 7 * 9 * 13
    assert len(res.shape) == 4
    flat_placeholder = d.get_placeholder_flattened(keepdims=False)
    res = session.run(flat_placeholder, feed_dict={placeholder: feed_data})
    print(res.shape)
    assert res.shape[0] == 7 * 9 * 13
    assert len(res.shape) == 2


def test_2D_Data_get_placeholder_flattened():
    import numpy as np

    d = Data(name="test_data", shape=(17,), dtype="float32", batch_dim_axis=0, feature_dim_axis=1)
    placeholder = tf_compat.v1.placeholder(shape=(None, 17), dtype=tf.float32)
    d.placeholder = placeholder
    feed_data = np.random.rand(7, 17)
    res = session.run(d.placeholder, feed_dict={placeholder: feed_data})
    print(res.shape)
    flat_placeholder = d.get_placeholder_flattened(keepdims=True)
    res = session.run(flat_placeholder, feed_dict={placeholder: feed_data})
    assert res.shape == (7, 17)
    flat_placeholder = d.get_placeholder_flattened(keepdims=False)
    res = session.run(flat_placeholder, feed_dict={placeholder: feed_data})
    assert res.shape == (7, 17)


def test_Data_copy_compatible_to_batch_major():
    d1 = Data(name="ff_out_output", shape=(None, 9001), dtype="float32")
    d2 = Data(name="ff_out_prior_output", shape=(9001,), dtype="float32", batch_dim_axis=None, time_dim_axis=None)
    d2a = d2.copy_compatible_to(d1)
    assert d2a.shape == (1, 9001)
    assert d2a.batch_dim_axis == d1.batch_dim_axis
    assert d2a.time_dim_axis == d1.time_dim_axis
    assert d2a.feature_dim_axis == d1.feature_dim_axis


def test_Data_copy_compatible_to_feature_dim():
    # copy_compatible_to should leave the feature dim as-is.
    d1 = Data(name="d1", shape=(None, 11), dtype="float32")
    d2 = Data(name="d2", shape=(13,), dtype="float32", batch_dim_axis=None, time_dim_axis=None)
    assert d1.dim != d2.dim
    d2a = d2.copy_compatible_to(d1)
    assert d2a.shape == (1, 13)
    assert d2a.batch_dim_axis == d1.batch_dim_axis
    assert d2a.time_dim_axis == d1.time_dim_axis
    assert d2a.feature_dim_axis == d1.feature_dim_axis


def test_Data_copy_compatible_to_src_no_batch():
    d1 = Data(name="d1", shape=(None, 1), time_dim_axis=None)
    d1.placeholder = tf.zeros([d if (d is not None) else 1 for d in d1.batch_shape])
    d2 = Data(name="d2", shape=(), batch_dim_axis=None, time_dim_axis=None)
    d2.placeholder = tf.zeros([d if (d is not None) else 1 for d in d2.batch_shape])
    d3 = d2.copy_compatible_to(d1)
    assert d3.get_static_batch_dim() == 1 and d3.batch_shape == (1, 1, 1)


def test_Data_copy_compatible_to_add_batch_dim():
    common_data = Data(name="accum_att_weights_output", shape=(None, 1))
    d1 = Data(name="att_weights_avg_output", shape=(1,), batch_dim_axis=None)
    d2 = d1.copy_compatible_to(common_data)
    assert d2.have_batch_axis() and d2.get_static_batch_dim() == 1 and d2.batch_shape == (1, 1, 1)


def test_Data_copy_compatible_to_add_feature_dim():
    common_data = Data(name="energy", shape=(None, 3), time_dim_axis=0, batch_dim_axis=None)  # [T,F]
    d1 = Data(name="mask", shape=(None,), batch_dim_axis=None, time_dim_axis=0, feature_dim_axis=None)  # [T]
    d2 = d1.copy_compatible_to(common_data)
    assert d2.batch_dim_axis is None and d2.time_dim_axis is not None and d2.batch_shape == (None, 1)


def test_Data_copy_compatible_to_add_batch_feature_dim():
    common_data = Data(name="energy", shape=(None, 3))  # [B,T,F]
    assert common_data.batch_shape == (None, None, 3)
    d1 = Data(name="mask", shape=(None,), batch_dim_axis=None, time_dim_axis=0, feature_dim_axis=None)  # [T]
    assert d1.batch_shape == (None,)
    d2 = d1.copy_compatible_to(common_data)
    print(d2)
    assert d2.have_batch_axis() and d2.have_time_axis()
    assert d2.get_static_batch_dim() == 1  # batch-dim is broadcasted
    assert d2.batch_shape == (1, None, 1)


def test_Data_copy_compatible_to_add_time_dim():
    common_data = Data(name="energy", shape=(None, 3))  # [B,T,F]
    d1 = Data(name="mask", shape=(3,))  # [B,F]
    d2 = d1.copy_compatible_to(common_data)
    assert d2.time_dim_axis is not None and d2.batch_shape == (None, 1, 3)


def test_Data_copy_compatible_to_time_axis_at_end():
    data = Data(name="att_weights_output", shape=(1, None), time_dim_axis=2, feature_dim_axis=1)
    common_data = Data(name="accum_att_weights_output", shape=(None, 1))
    data2 = data.copy_compatible_to(common_data)
    assert data2.batch_dim_axis == common_data.batch_dim_axis == 0
    assert data2.time_dim_axis == common_data.time_dim_axis == 1
    assert data2.feature_dim_axis == common_data.feature_dim_axis == 2
    assert data2.shape == common_data.shape == (None, 1)
    assert data2.dim == common_data.dim == 1


def test_Data_copy_compatible_to_batch_axis1_time_axis_at_end():
    beam = SearchBeam(beam_size=12)
    data = Data(name="att_weights_output", shape=(1, None), time_dim_axis=2, feature_dim_axis=1, beam=beam)
    common_data = Data(name="accum_att_weights_output", shape=(None, 1), batch_dim_axis=1)
    data2 = data.copy_compatible_to(common_data)
    assert data2.time_dim_axis == common_data.time_dim_axis == 0
    assert data2.batch_dim_axis == common_data.batch_dim_axis == 1
    assert data2.feature_dim_axis == common_data.feature_dim_axis == 2
    assert data2.shape == common_data.shape == (None, 1)
    assert data2.dim == common_data.dim == 1


def test_Data_copy_compatible_to_batch_feature_is_dynamic():
    # Enc/dec for proper time dim tags.
    enc = Data(name="enc", shape=(None, 1), auto_create_placeholders=True)
    dec = Data(name="dec", shape=(None, 1), auto_create_placeholders=True)
    print("enc:", enc)
    print("dec:", dec)
    # start: batch_shape_meta=[T|'time-with-postfix:0_data_target0',B]
    start = Data(name="t_start_output", shape=(None,), dtype="int32", sparse=True, dim=None, batch_dim_axis=1)
    start.size_placeholder = {0: dec.size_placeholder[0]}
    print("start:", start)
    assert_equal(start.get_time_dim_tag(), dec.get_time_dim_tag())
    # energy: batch_shape_meta=[F|'time-with-postfix:0_data_target0',B,T|'time-with-postfix:encoder']
    energy = Data(name="energy2_output", shape=(None, None), batch_dim_axis=1, time_dim_axis=2, feature_dim_axis=0)
    energy.size_placeholder = {0: dec.size_placeholder[0], 1: enc.size_placeholder[0]}
    print("energy:", energy)
    assert_equal(energy.get_size_dim_tag(0), dec.get_time_dim_tag())
    assert_equal(energy.get_size_dim_tag(1), enc.get_time_dim_tag())
    assert_equal(energy.get_time_dim_tag(), enc.get_time_dim_tag())
    t = start.copy_compatible_to(energy, check_sparse=False, check_dtype=False)
    print("t:", t)
    assert t.shape == (None, 1) and t.time_dim_axis == energy.time_dim_axis
    assert t.batch_dim_axis == energy.batch_dim_axis
    assert t.sparse and t.feature_dim_axis is None  # because it is sparse
    assert set(t.size_placeholder.keys()) == {0}
    assert t.size_placeholder[0] is dec.size_placeholder[0]
    assert_equal(t.get_size_dim_tag(0), dec.get_time_dim_tag())


def test_Data_copy_compatible_to_bias_to_batch_time_spatial_feature():
    from returnn.tf.util.data import batch_dim

    time_dim = SpatialDim("time")
    static_dim = FeatureDim("feature-spatial", 10)
    feat_dim = FeatureDim("feature", 14)
    x = Data(name="input", dim_tags=[batch_dim, time_dim, static_dim, feat_dim])
    print("x:", x)
    bias = Data(name="bias", dim_tags=[feat_dim])
    print("bias:", bias)
    b_ = bias.copy_compatible_to(x)
    print("bias copy_compatible_to x:", b_)
    assert b_.batch_shape == (1, 1, 1, 14)


def test_Data_copy_compatible_to_match_priority():
    feat_dim = FeatureDim("feature", 2)
    in_dim = feat_dim.copy(match_priority=1)
    assert in_dim == feat_dim and in_dim.match_priority > feat_dim.match_priority and in_dim is not feat_dim
    with tf.Graph().as_default() as graph, tf_compat.v1.Session(graph=graph) as session:
        raw_np = numpy.arange(0, 2 * 2, dtype=numpy.float32).reshape((2, 2))
        raw = tf.constant(raw_np)
        x = Tensor("x", [in_dim, feat_dim], "float32", raw_tensor=raw)

        # (in,out) -> (in,out) (noop)
        x_ = x.copy_compatible_to(Tensor("y", [in_dim, feat_dim], "float32"))
        assert len(x_.dims) == 2 and x_.dims[0] is in_dim and x_.dims[1] is feat_dim
        x_np = session.run(x_.raw_tensor)
        numpy.testing.assert_equal(x_np, raw_np)

        # (in,out) -> (out,in)
        x_ = x.copy_compatible_to(Tensor("y", [feat_dim, in_dim], "float32"))
        assert len(x_.dims) == 2 and x_.dims[0] is feat_dim and x_.dims[1] is in_dim
        x_np = session.run(x_.raw_tensor)
        numpy.testing.assert_equal(x_np, raw_np.transpose([1, 0]))

        # (out,in) -> (out,in) (noop)
        x_ = x_.copy_compatible_to(Tensor("y", [feat_dim, in_dim], "float32"))
        assert len(x_.dims) == 2 and x_.dims[0] is feat_dim and x_.dims[1] is in_dim
        x_np = session.run(x_.raw_tensor)
        numpy.testing.assert_equal(x_np, raw_np.transpose([1, 0]))

        # (out,in) -> (in,out)
        x_ = x_.copy_compatible_to(Tensor("y", [in_dim, feat_dim], "float32"))
        assert len(x_.dims) == 2 and x_.dims[0] is in_dim and x_.dims[1] is feat_dim
        x_np = session.run(x_.raw_tensor)
        numpy.testing.assert_equal(x_np, raw_np)


def test_Data_copy_compatible_to_dims_match_priority():
    feat_dim = FeatureDim("feature", 2)
    in_dim = feat_dim.copy(match_priority=1)
    assert in_dim == feat_dim and in_dim.match_priority > feat_dim.match_priority and in_dim is not feat_dim
    with tf.Graph().as_default() as graph, tf_compat.v1.Session(graph=graph) as session:
        raw_np = numpy.arange(0, 2 * 2, dtype=numpy.float32).reshape((2, 2))
        raw = tf.constant(raw_np)
        x = Tensor("x", [in_dim, feat_dim], "float32", raw_tensor=raw)

        # (in,out) -> (in,out) (noop)
        x_ = x.copy_compatible_to_dims([in_dim, feat_dim])
        assert len(x_.dims) == 2 and x_.dims[0] is in_dim and x_.dims[1] is feat_dim
        x_np = session.run(x_.raw_tensor)
        numpy.testing.assert_equal(x_np, raw_np)

        # (in,out) -> (out,in)
        x_ = x.copy_compatible_to_dims([feat_dim, in_dim])
        assert len(x_.dims) == 2 and x_.dims[0] is feat_dim and x_.dims[1] is in_dim
        x_np = session.run(x_.raw_tensor)
        numpy.testing.assert_equal(x_np, raw_np.transpose([1, 0]))

        # (out,in) -> (out,in) (noop)
        x_ = x_.copy_compatible_to_dims([feat_dim, in_dim])
        assert len(x_.dims) == 2 and x_.dims[0] is feat_dim and x_.dims[1] is in_dim
        x_np = session.run(x_.raw_tensor)
        numpy.testing.assert_equal(x_np, raw_np.transpose([1, 0]))

        # (out,in) -> (in,out)
        x_ = x_.copy_compatible_to_dims([in_dim, feat_dim])
        assert len(x_.dims) == 2 and x_.dims[0] is in_dim and x_.dims[1] is feat_dim
        x_np = session.run(x_.raw_tensor)
        numpy.testing.assert_equal(x_np, raw_np)


def test_Data_copy_tranpose_match_priority():
    feat_dim = FeatureDim("feature", 2)
    in_dim = feat_dim.copy(match_priority=1)
    assert in_dim == feat_dim and in_dim.match_priority > feat_dim.match_priority and in_dim is not feat_dim
    with tf.Graph().as_default() as graph, tf_compat.v1.Session(graph=graph) as session:
        raw_np = numpy.arange(0, 2 * 2, dtype=numpy.float32).reshape((2, 2))
        raw = tf.constant(raw_np)
        x = Tensor("x", [in_dim, feat_dim], "float32", raw_tensor=raw)

        # (in,out) -> (in,out) (noop)
        x_ = x.copy_transpose([in_dim, feat_dim])
        assert len(x_.dims) == 2 and x_.dims[0] is in_dim and x_.dims[1] is feat_dim
        x_np = session.run(x_.raw_tensor)
        numpy.testing.assert_equal(x_np, raw_np)

        # (in,out) -> (out,in)
        x_ = x.copy_transpose([feat_dim, in_dim])
        assert len(x_.dims) == 2 and x_.dims[0] is feat_dim and x_.dims[1] is in_dim
        x_np = session.run(x_.raw_tensor)
        numpy.testing.assert_equal(x_np, raw_np.transpose([1, 0]))

        # (out,in) -> (out,in) (noop)
        x_ = x_.copy_transpose([feat_dim, in_dim])
        assert len(x_.dims) == 2 and x_.dims[0] is feat_dim and x_.dims[1] is in_dim
        x_np = session.run(x_.raw_tensor)
        numpy.testing.assert_equal(x_np, raw_np.transpose([1, 0]))

        # (out,in) -> (in,out)
        x_ = x_.copy_transpose([in_dim, feat_dim])
        assert len(x_.dims) == 2 and x_.dims[0] is in_dim and x_.dims[1] is feat_dim
        x_np = session.run(x_.raw_tensor)
        numpy.testing.assert_equal(x_np, raw_np)


def test_Data_get_common_data_extra_static_spatial():
    d1 = Data(name="t", shape=(None, 32, 128), dtype="float32", auto_create_placeholders=True)
    d2 = Data(name="r", shape=(None, 32, 128), dtype="float32", auto_create_placeholders=True)
    d2.get_size_dim_tag(0).declare_same_as(d1.get_size_dim_tag(0))
    common = Data.get_common_data([d1, d2])
    assert d1.shape == common.shape


def test_Data_get_common_data_broadcast_multiple():
    d1 = Data(name="d_orig", shape=(5, 5, 3), dtype="float32", batch_dim_axis=None)
    d2 = Data(name="d_bc", shape=(5,), dtype="float32", batch_dim_axis=None)
    common = Data.get_common_data([d1, d2])
    assert d1.shape == common.shape


def test_Data_get_common_data_broadcast_multiple_dim_tags():
    from returnn.tf.util.data import batch_dim

    time_dim = SpatialDim("time")
    input_dim = FeatureDim("input", 3)
    feat_dim = FeatureDim("feat", 3)
    a = Data("a", dim_tags=[batch_dim, time_dim, input_dim])
    b = Data("b", dim_tags=[feat_dim])
    out = Data.get_common_data([a, b], allow_broadcast_all_sources=True)
    assert out.dim_tags == (batch_dim, time_dim, input_dim, feat_dim)


@contextlib.contextmanager
def set_behavior_version(version: int):
    """
    This is a context manager which sets the behavior version to the given value.
    """
    from returnn.util.basic import BehaviorVersion

    # noinspection PyProtectedMember
    old = BehaviorVersion._get_state()
    try:
        # noinspection PyProtectedMember
        BehaviorVersion._reset()
        BehaviorVersion.set(version)
        yield
    finally:
        # noinspection PyProtectedMember
        BehaviorVersion._reset(old)


def test_Data_get_common_data_no_broadcast_for_explicit():
    from returnn.tf.util.data import batch_dim

    time_dim = SpatialDim("time")
    input_dim = FeatureDim("input", 3)
    feat_dim = FeatureDim("feat", 1)
    a = Data("a", dim_tags=[batch_dim, time_dim, input_dim])
    b = Data("b", dim_tags=[feat_dim])
    with set_behavior_version(0):
        out = Data.get_common_data([a, b], allow_broadcast_all_sources=True)
        assert out.dim_tags == (batch_dim, time_dim, input_dim, feat_dim)


def test_Data_get_common_data_extra2_static_spatial():
    d1 = Data(name="t", shape=(None, 32, 32, 128), dtype="float32", auto_create_placeholders=True)
    d2 = Data(name="r", shape=(None, 32, 32, 128), dtype="float32", auto_create_placeholders=True)
    d2.get_size_dim_tag(0).declare_same_as(d1.get_size_dim_tag(0))
    common = Data.get_common_data([d1, d2])
    assert d1.shape == common.shape


def test_Data_get_common_data_one_undefined_time():
    # Data(name='accum_output', shape=(None, 1), batch_shape_meta=[B,T|?,F|1])
    a = Data(name="a", shape=(None, 1))  # undefined time-dim-tag
    print("a:", a)
    assert a.size_placeholder.get(0) is None
    # Data(name='enc0_output', shape=(None,), batch_shape_meta=[B,T|F|'time:var:extern_data:encoder'])
    b = Data(name="b", shape=(None,), auto_create_placeholders=True)
    print("b:", b)
    # Data(name='enc1_output', shape=(None, 1), batch_shape_meta=[B,T|'time:var:extern_data:encoder',F|1])
    c = Data(name="c", shape=(None, 1))
    c.size_placeholder = b.size_placeholder.copy()
    print("c:", c)
    c.sanity_check()
    assert_equal(b.get_time_dim_tag(), c.get_time_dim_tag())

    out = Data.get_common_data([a, b, c])
    print("out:", out)
    assert out.shape == (None, 1) and out.batch_dim_axis == 0
    assert_equal(out.get_time_dim_tag(), b.get_time_dim_tag())


def test_Data_get_common_data_copy_compatible_to_different_time_dim():
    a = Data(name="a", shape=(None, 3, 5), auto_create_placeholders=True)
    b = Data(name="b", shape=(None, 3, 5), auto_create_placeholders=True)
    print("a:", a)
    print("b:", b)
    common_data = Data.get_common_data([a, b], allow_broadcast_all_sources=True)
    print("common:", common_data)
    assert common_data.shape == (None, None, 3, 5) and common_data.batch_dim_axis == 0
    assert_equal(common_data.get_size_dim_tag(0), a.get_time_dim_tag())
    assert_equal(common_data.get_size_dim_tag(1), b.get_time_dim_tag())
    aa = a.copy_compatible_to(common_data)
    bb = b.copy_compatible_to(common_data)
    print("aa:", aa)
    print("bb:", bb)
    assert aa.batch_ndim == bb.batch_ndim
    for i in range(aa.batch_ndim):
        d1 = aa.batch_shape[i]
        d2 = bb.batch_shape[i]
        if d1 == 1 or d2 == 1:
            continue  # it's fine, that will broadcast
        assert d1 == d2, "mismatch in axis %i" % i
    assert_equal(aa.get_dim_tag(axis=1), a.get_time_dim_tag())
    assert aa.batch_shape[2] == 1
    assert_equal(bb.get_dim_tag(axis=2), b.get_time_dim_tag())
    assert bb.batch_shape[1] == 1
    x = aa.placeholder + bb.placeholder
    session.run(
        x,
        feed_dict={
            a.placeholder: numpy.zeros((2, 7, 3, 5), "float32"),
            b.placeholder: numpy.zeros((2, 11, 3, 5), "float32"),
        },
    )


def test_Data_get_common_data_copy_compatible_to_different_time_dim_different_static_order():
    a = Data(name="a", shape=(None, 3, 5), auto_create_placeholders=True)
    b = Data(name="b", shape=(3, None, 5), auto_create_placeholders=True)
    print("a:", a)
    print("b:", b)
    assert_not_equal(a.get_time_dim_tag(), b.get_time_dim_tag())
    common_data = Data.get_common_data([a, b], allow_broadcast_all_sources=True)
    print("common:", common_data)
    assert common_data.shape.count(None) == 2 and 3 in common_data.shape and 5 in common_data.shape
    assert common_data.batch_ndim == 5
    assert_equal(common_data.get_size_dim_tag(0), a.get_time_dim_tag())
    assert_equal(common_data.get_size_dim_tag(1), b.get_time_dim_tag())
    common_tags, _ = Dim.get_all_dimension_tags([common_data])
    print("common dim tags:")
    pprint(common_tags)
    assert len(common_tags) == common_data.batch_ndim  # all unique
    assert_in(a.get_time_dim_tag(), common_tags)
    assert_in(b.get_time_dim_tag(), common_tags)
    aa = a.copy_compatible_to(common_data)
    bb = b.copy_compatible_to(common_data)
    print("aa:", aa)
    print("bb:", bb)
    assert aa.batch_ndim == bb.batch_ndim
    for i in range(aa.batch_ndim):
        d1 = aa.batch_shape[i]
        d2 = bb.batch_shape[i]
        if d1 == 1 or d2 == 1:
            continue  # it's fine, that will broadcast
        assert d1 == d2, "mismatch in axis %i" % i
    assert_equal(aa.get_size_dim_tag(0), a.get_time_dim_tag())
    assert_equal(bb.get_size_dim_tag(0), b.get_time_dim_tag())
    x = aa.placeholder + bb.placeholder
    session.run(
        x,
        feed_dict={
            a.placeholder: numpy.zeros((2, 7, 3, 5), "float32"),
            b.placeholder: numpy.zeros((2, 3, 11, 5), "float32"),
        },
    )


def test_Data_copy_compatible_to_get_common_data_auto_feature_non_sparse():
    d1 = Data(
        name="t",
        shape=(None,),
        dtype="int32",
        batch_dim_axis=None,
        feature_dim_axis=None,
        auto_create_placeholders=True,
    )  # placeholder for specific spatial dim-tag
    d2 = Data(name="r", shape=(6,), dtype="int32", batch_dim_axis=None, time_dim_axis=None)
    common = Data.get_common_data([d1, d2], allow_broadcast_all_sources=True)
    print("common:", common)
    d1a = d1.copy_compatible_to(common)
    print("d1':", d1a)
    d2a = d2.copy_compatible_to(common)
    print("d2':", d2a)
    assert common.feature_dim_axis_or_unspecified is NotSpecified
    assert d1a.feature_dim_axis_or_unspecified is NotSpecified
    assert d2a.feature_dim_axis_or_unspecified is NotSpecified


def test_Data_copy_compatible_to_get_common_data_no_feature_sparse():
    d1 = Data(name="t", shape=(), dtype="int32", sparse=True, dim=None, time_dim_axis=None)
    d2 = Data(name="r", shape=(6,), dtype="int32", sparse=True, dim=6, batch_dim_axis=None, time_dim_axis=None)
    common = Data.get_common_data([d1, d2], allow_broadcast_all_sources=True)
    print("common:", common)
    d1a = d1.copy_compatible_to(common)
    print("d1':", d1a)
    d2a = d2.copy_compatible_to(common)
    print("d2':", d2a)
    assert common.sparse and d1a.sparse and d2a.sparse
    assert common.feature_dim_axis_or_unspecified is NotSpecified
    assert d1a.feature_dim_axis_or_unspecified is NotSpecified
    assert d2a.feature_dim_axis_or_unspecified is NotSpecified
    assert common.feature_dim_axis is None
    assert d1a.feature_dim_axis is None
    assert d2a.feature_dim_axis is None


def test_Data_copy_compatible_to_add_dummy_time_also_feature_dim():
    start = Data(
        name="start",
        shape=(),
        dtype="int32",
        sparse=True,
        dim=None,
        batch_dim_axis=0,
        time_dim_axis=None,
        feature_dim_axis=None,
    )
    print("start:", start)
    assert start.batch_ndim == 1
    energy = Data(
        name="energy",
        shape=(None,),
        dtype="float32",
        sparse=False,
        dim=None,
        batch_dim_axis=0,
        time_dim_axis=1,
        feature_dim_axis=1,
    )
    print("energy:", energy)
    assert energy.batch_ndim == 2
    assert energy.time_dim_axis == energy.feature_dim_axis
    x = start.copy_compatible_to(energy, check_sparse=False, check_dtype=False)
    print("start copy_compatible_to energy result:", x)
    assert x.sparse
    assert x.batch_ndim == energy.batch_ndim
    assert x.batch_dim_axis == energy.batch_dim_axis
    assert x.feature_dim_axis is None  # it's sparse, thus by definition it does not have a feature axis
    assert x.time_dim_axis == energy.time_dim_axis


def test_Data_copy_compatible_to_keep_feature_new_time():
    start = Data(name="start", shape=(1,), time_dim_axis=None)
    print("start:", start)
    assert start.batch_ndim == 2 and start.feature_dim_axis == 1
    energy = Data(name="energy", shape=(7, 4), time_dim_axis=2, feature_dim_axis=1)
    print("energy:", energy)
    assert energy.batch_ndim == 3
    x = start.copy_compatible_to(energy, check_sparse=False, check_dtype=False)
    print("start copy_compatible_to energy result:", x)
    assert x.batch_ndim == energy.batch_ndim
    assert x.batch_dim_axis == energy.batch_dim_axis
    assert x.feature_dim_axis == 1
    assert x.time_dim_axis == energy.time_dim_axis


def test_Data_copy_compatible_to_sparse_to_dense():
    source = Data(name="start", shape=(), dtype="int32", sparse=True, dim=1, time_dim_axis=None)
    print("source:", source)
    target = Data(name="energy", shape=(None,))
    print("target:", target)
    dest = source.copy_compatible_to(target, check_sparse=False, check_dtype=False)
    print("dest:", dest)
    assert dest.shape == (1,) and dest.dtype == "int32" and dest.sparse and dest.dim == 1 and dest.time_dim_axis == 1


def test_Data_copy_compatible_to_move_spatial_axes():
    common = Data(name="common", shape=(None, 3, 5), auto_create_placeholders=True)
    print("common:", common)
    a = Data(name="a", shape=(3, None, 5))
    a.size_placeholder = {1: common.size_placeholder[0]}
    print("a:", a)
    assert_equal(common.get_time_dim_tag(), a.get_time_dim_tag())
    b = a.copy_compatible_to(common)
    print("b:", b)
    assert b.shape == common.shape
    assert_equal(b.get_time_dim_tag(), a.get_time_dim_tag())


def test_Data_copy_add_spatial_dim_added_time_at_end():
    d = Data(name="start", shape=(1,), time_dim_axis=None)
    print("d:", d)
    assert d.batch_shape == (None, 1) and d.feature_dim_axis == 1 and d.time_dim_axis is None
    assert d.feature_dim_axis_or_unspecified is NotSpecified
    d2 = d.copy_add_spatial_dim(2)
    print("d2:", d2)
    assert d2.batch_shape == (None, 1, 1) and d2.feature_dim_axis == 1 and d2.time_dim_axis == 2
    assert d2.feature_dim_axis_or_unspecified is NotSpecified


def test_Data_get_common_data_tbf_and_bf():
    sources = [
        Data(name="target", shape=(None, 13), batch_dim_axis=1, time_dim_axis=0),
        Data(name="encoder", shape=(13,), time_dim_axis=None, batch_dim_axis=0),
    ]
    pprint(sources)
    common = Data.get_common_data(sources=sources)
    print("common:", common)
    assert common.batch_ndim == 3


def test_Data_get_common_data_tbf_and_bf2():
    sources = [
        Data(name="target", shape=(None, 13), batch_dim_axis=1, time_dim_axis=0),
        Data(name="encoder", shape=(11,), time_dim_axis=None, batch_dim_axis=0),
    ]
    pprint(sources)
    common = Data.get_common_data(sources=sources, allow_broadcast_all_sources=True)
    print("common:", common)
    assert common.batch_ndim == 4


def test_Data_get_common_data_btf_and_bf_get_kwargs_copy_compatible_to():
    s0 = Data(name="location_feedback", shape=(None, 6), batch_dim_axis=0, time_dim_axis=1)
    s1 = Data(name="s_transformed", shape=(6,), time_dim_axis=None, batch_dim_axis=0)
    pprint([s0, s1])
    common = Data.get_common_data(sources=[s0, s1])
    print("common:", common)
    assert common.shape == (None, 6)
    assert common.batch_dim_axis == 0
    assert common.get_dim_tag(1) == s0.get_dim_tag(1)
    assert common.time_dim_axis == 1
    common_opts = common.get_kwargs()
    common_opts.pop("batch_dim_axis", None)
    common_opts.pop("feature_dim_axis", None)
    common_opts.pop("time_dim_axis", None)
    common_ = Data(**common_opts)
    assert common_.shape == (None, 6)
    assert common_.batch_dim_axis == 0
    assert common_.get_dim_tag(1) == s0.get_dim_tag(1)
    assert common_.time_dim_axis == 1
    s0c, s1c = [s.copy_compatible_to(common_) for s in [s0, s1]]
    assert isinstance(s0c, Data) and isinstance(s1c, Data)
    assert (s0c.batch_dim_axis, s0c.time_dim_axis, s0c.feature_dim_axis) == (0, 1, 2)
    assert (s1c.batch_dim_axis, s1c.time_dim_axis, s1c.feature_dim_axis) == (0, 1, 2)
    assert s1c.batch_shape == (None, 1, 6)


def test_Data_get_common_data_beam_size():
    condition = Data(name="cond", shape=(), dtype="bool", sparse=True, dim=2, time_dim_axis=None)
    true_from = Data(name="true", shape=(), dtype="int32", sparse=True, dim=19, time_dim_axis=None)
    beam = SearchBeam(beam_size=3)
    false_from = Data(name="false", shape=(), dtype="int32", sparse=True, dim=19, time_dim_axis=None, beam=beam)
    print("cond:", condition)
    print("true:", true_from)
    print("false:", false_from)
    common = Data.get_common_data([true_from, false_from, condition])
    print("common:", common)
    assert common.shape == () and common.sparse
    assert common.beam == beam


def test_Data_get_common_data_tb_bf():
    a = Data(name="a", shape=(None,), time_dim_axis=0, batch_dim_axis=1)
    b = Data(name="b", shape=(5,), batch_dim_axis=0)
    print("a:", a)
    print("b:", b)
    common = Data.get_common_data([a, b], allow_broadcast_all_sources=True)
    print("common:", common)
    assert common.shape == (None, 5) and common.batch_dim_axis == 1


def test_Data_no_feature_dim():
    d = Data(name="x", shape=(6,), dtype="int32", sparse=True, dim=6, batch_dim_axis=None, time_dim_axis=None)
    assert d.feature_dim_axis is None


def test_Data_feature_dim_axis_btd():
    d1 = Data(name="d1", shape=(None, 11), feature_dim_axis=-1)
    d2 = Data(name="d2", shape=(None, 11), feature_dim_axis=2)
    d3 = Data(name="d3", shape=(None, 11))
    d4 = Data(name="d4", feature_dim_axis=2, dim=11)
    assert d1.batch_dim_axis == d2.batch_dim_axis == d3.batch_dim_axis == d4.batch_dim_axis == 0
    assert d1.time_dim_axis == d2.time_dim_axis == d3.time_dim_axis == d4.time_dim_axis == 1
    assert d1.feature_dim_axis == d2.feature_dim_axis == d3.feature_dim_axis == d4.feature_dim_axis == 2
    assert d1.batch_shape == d2.batch_shape == d3.batch_shape == d4.batch_shape == (None, None, 11)
    assert d1.feature_dim_axis_or_unspecified == 2
    assert d3.feature_dim_axis_or_unspecified is NotSpecified


def test_Data_feature_dim_axis_none():
    d1 = Data(name="d1", shape=())
    d2 = Data(name="d2", shape=(), feature_dim_axis=None)
    d3 = Data(name="d3", shape=(None,), sparse=True, dim=7)
    d4 = Data(name="d4", shape=(None,), sparse=True, dim=7, feature_dim_axis=None)
    assert d1.feature_dim_axis == d2.feature_dim_axis == d3.feature_dim_axis == d4.feature_dim_axis == None
    assert d1.feature_dim_axis_or_unspecified is NotSpecified
    assert d2.feature_dim_axis_or_unspecified is None


def test_Data_feature_dim_axis_bdt():
    d1 = Data(name="d1", shape=(11, None), feature_dim_axis=1)
    d2 = Data(name="d2", time_dim_axis=2, feature_dim_axis=1, dim=11)
    d3 = Data(name="d3", dim=11, feature_dim_axis=1)  # will add time-dim by default
    assert d1.batch_ndim == d2.batch_ndim == d3.batch_ndim == 3
    assert d1.batch_dim_axis == d2.batch_dim_axis == d3.batch_dim_axis == 0
    assert d1.feature_dim_axis == d2.feature_dim_axis == d3.feature_dim_axis == 1
    assert d1.time_dim_axis == d2.time_dim_axis == d3.time_dim_axis == 2
    assert d1.dim == d2.dim == d3.dim == 11
    assert d1.batch_shape == d2.batch_shape == d3.batch_shape == (None, 11, None)


def test_Data_feature_dim_axis_bd():
    d1 = Data(name="d1", time_dim_axis=None, dim=11)
    d2 = Data(name="d2", shape=(11,))
    assert d1.batch_dim_axis == d2.batch_dim_axis == 0
    assert d1.time_dim_axis == d2.time_dim_axis == None
    assert d1.feature_dim_axis == d2.feature_dim_axis == 1
    assert d1.dim == d2.dim == 11
    assert d1.batch_shape == d2.batch_shape == (None, 11)


def test_Data_feature_dim_axis_d():
    d1 = Data(name="d1", batch_dim_axis=None, time_dim_axis=None, dim=11)
    d2 = Data(name="d2", batch_dim_axis=None, shape=(11,))
    assert d1.batch_dim_axis == d2.batch_dim_axis == None
    assert d1.time_dim_axis == d2.time_dim_axis == None
    assert d1.feature_dim_axis == d2.feature_dim_axis == 0
    assert d1.dim == d2.dim == 11
    assert d1.batch_shape == d2.batch_shape == (11,)


def test_Data_feature_dim_axis_NHWC():
    d1 = Data(name="d1", shape=(None, None, 11))
    d2 = Data(name="d2", shape=(None, None, 11), feature_dim_axis=-1)
    d3 = Data(name="d3", dim=11, feature_dim_axis=3)
    assert d1.batch_ndim == d2.batch_ndim == d3.batch_ndim == 4
    assert d1.batch_dim_axis == d2.batch_dim_axis == d3.batch_dim_axis == 0
    assert d1.time_dim_axis == d2.time_dim_axis == d3.time_dim_axis == 1
    assert d1.feature_dim_axis == d2.feature_dim_axis == d3.feature_dim_axis == 3
    assert d1.dim == d2.dim == d3.dim == 11
    assert d1.batch_shape == d2.batch_shape == d3.batch_shape == (None, None, None, 11)


def test_Data_feature_dim_axis_NCHW():
    d1 = Data(name="d1", shape=(11, None, None), feature_dim_axis=1)
    d2 = Data(name="d2", shape=(11, None, None), time_dim_axis=2, feature_dim_axis=1, dim=11)
    assert d1.batch_ndim == d2.batch_ndim == 4
    assert d1.batch_dim_axis == d2.batch_dim_axis == 0
    assert d1.feature_dim_axis == d2.feature_dim_axis == 1
    assert d1.time_dim_axis == d2.time_dim_axis == 2
    assert d1.dim == d2.dim == 11
    assert d1.batch_shape == d2.batch_shape == (None, 11, None, None)


def test_Data_scalar():
    d1 = Data(name="d1", batch_dim_axis=None, time_dim_axis=None, feature_dim_axis=None)
    assert d1.batch_dim_axis is None
    assert d1.time_dim_axis is None
    assert d1.feature_dim_axis is None
    assert d1.dim is None
    assert d1.batch_shape == ()


def test_Data_scalar_default():
    d1 = Data(name="d1", shape=(), dtype="int32", batch_dim_axis=None)
    assert not d1.sparse
    assert d1.batch_shape == () and d1.dim is None and d1.feature_dim_axis is None and d1.batch_dim_axis is None


def test_Data_copy_add_feature_dim():
    d1 = Data(name="d1", shape=(None, 11))
    d2 = d1.copy_add_feature_dim()
    assert d2.batch_shape == (None, None, 11, 1)
    assert d2.dim == 1


def test_Data_copy_split_feature_dim():
    d1 = Data(name="d1", shape=(None, 12))
    d2 = d1.copy_split_feature_dim(4)
    assert d2.batch_shape == (None, None, 3, 4)
    assert d2.dim == 4


def test_Data_copy_as_batch_feature_major():
    d1 = Data(name="d1", shape=(None, 12))
    assert d1.batch_shape == (None, None, 12) and d1.time_dim_axis == 1 and d1.feature_dim_axis == 2
    d2 = d1.copy_as_batch_feature_major()
    assert d2.batch_shape == (None, 12, None) and d2.time_dim_axis == 2 and d2.feature_dim_axis == 1
    assert d2.dim == 12


def test_Data_copy_template_excluding_time_dim():
    d1 = Data(name="d1", shape=(None, 12))
    assert d1.batch_shape == (None, None, 12) and d1.time_dim_axis == 1 and d1.feature_dim_axis == 2
    d2 = d1.copy_template_excluding_time_dim()
    assert d2.batch_shape == (None, 12) and d2.time_dim_axis is None and d2.feature_dim_axis == 1


def test_Data_copy_template_excluding_time_dim_explicit_feature():
    d1 = Data(name="d1", shape=(None, 12), feature_dim_axis=2)
    assert d1.batch_shape == (None, None, 12) and d1.time_dim_axis == 1 and d1.feature_dim_axis == 2
    d2 = d1.copy_template_excluding_time_dim()
    assert d2.batch_shape == (None, 12) and d2.time_dim_axis is None and d2.feature_dim_axis == 1


def test_Data_copy_template_excluding_time_dim_multiple_time():
    d = Data(
        name="energy_in_t_rel_var_output",
        shape=(None, None, 13),
        batch_dim_axis=2,
        time_dim_axis=0,
        auto_create_placeholders=True,
    )  # placeholders to have proper dim tags
    print("d:", d)
    print("sizes:", d.size_placeholder)
    assert set(d.size_placeholder.keys()) == {0, 1}
    assert d.feature_dim_axis == 3
    size1tag = d.get_size_dim_tag(0)
    size2tag = d.get_size_dim_tag(1)
    print("size tags:", size1tag, size2tag)
    assert size1tag.dyn_size is not None and size2tag.dyn_size is not None
    d2 = d.copy_template_excluding_time_dim()
    print("excluding time:", d2)
    assert d2.shape == (None, 13) and (d2.time_dim_axis, d2.batch_dim_axis, d2.feature_dim_axis) == (0, 1, 2)
    print("sizes:", d2.size_placeholder)
    assert set(d2.size_placeholder.keys()) == {0}
    assert d2.size_placeholder[0] is d.size_placeholder[1]
    new_size_tag = d2.get_time_dim_tag()
    assert new_size_tag is size2tag


def test_Data_copy_add_spatial_dim_no_batch():
    d1 = Data(name="d1", shape=(3,), batch_dim_axis=None, time_dim_axis=None)
    assert d1.batch_dim_axis is None and d1.time_dim_axis is None and d1.feature_dim_axis == 0
    assert d1.batch_shape == (3,) and d1.dim == 3
    d2 = d1.copy_add_spatial_dim(0)
    assert d2.batch_dim_axis is None and d2.time_dim_axis == 0 and d2.feature_dim_axis == 1
    assert d2.batch_shape == (1, 3) and d2.dim == 3


def test_Data_copy_add_spatial_dim_no_batch_explicit_feature():
    d1 = Data(name="d1", shape=(3,), batch_dim_axis=None, time_dim_axis=None, feature_dim_axis=0)
    assert d1.batch_dim_axis is None and d1.time_dim_axis is None and d1.feature_dim_axis == 0
    assert d1.batch_shape == (3,) and d1.dim == 3
    d2 = d1.copy_add_spatial_dim(0)
    assert d2.batch_dim_axis is None and d2.time_dim_axis == 0 and d2.feature_dim_axis == 1
    assert d2.batch_shape == (1, 3) and d2.dim == 3


def test_Data_copy_add_spatial_dim_becomes_new_feature():
    d1 = Data(name="att_weights_avg_output", shape=(None,), batch_dim_axis=None, time_dim_axis=None)
    d2 = d1.copy_add_spatial_dim(0)


def test_Data_copy_add_spatial_dim_most_right():
    d1 = Data(name="att_weights_avg_output", shape=(1,))
    print(d1, "spatial axes:", d1.get_spatial_batch_axes())
    d2 = d1.copy_add_spatial_dim(1)
    print(d2, "spatial axes:", d2.get_spatial_batch_axes())
    assert_equal(d2.get_spatial_batch_axes(), [1])


def test_Data_copy_add_spatial_dim_no_batch_end():
    d = Data(
        name="t",
        shape=(None,),
        dtype="int32",
        sparse=True,
        dim=None,
        batch_dim_axis=None,
        time_dim_axis=None,
        feature_dim_axis=None,
    )
    assert d.batch_shape == (None,)
    d2 = d.copy_add_spatial_dim(spatial_dim_axis=1, dim=1)
    assert d2.batch_shape == (None, 1)


def test_Data_copy_add_spatial_dim_default_after_last_spatial():
    d1 = Data(name="x", shape=(2, 1337), batch_dim_axis=0, time_dim_axis=1)
    assert d1.batch_shape == (None, 2, 1337)
    d2 = d1.copy_add_spatial_dim(dim=3)
    assert d2.batch_shape == (None, 2, 3, 1337)
    d3 = d2.copy_add_spatial_dim(dim=4)
    assert d3.batch_shape == (None, 2, 3, 4, 1337)


def test_Data_copy_add_spatial_dim_before_time():
    a = Data(name="a", shape=(None, 3, 5), auto_create_placeholders=True)
    print("a:", a)
    b = a.copy_add_spatial_dim(spatial_dim_axis=1, auto_time_dim_axis=False)
    print("b:", b)
    assert b.shape == (1, None, 3, 5) and (b.batch_dim_axis, b.time_dim_axis) == (0, 2)
    assert b.size_placeholder[1] is a.size_placeholder[0]


def test_Data_copy_add_dim_by_tag_unbroadcast_feature_non_specific_feature_dim():
    d = Data(name="t", shape=(None,), dtype="int32", batch_dim_axis=None, time_dim_axis=None, feature_dim_axis=None)
    tag = FeatureDim("feature:r", dimension=6)
    d2 = d.copy_add_dim_by_tag(tag, unbroadcast=True)
    print("d2:", d2)
    assert d2.batch_shape == (None, 6)
    assert d2.feature_dim_axis_or_unspecified is NotSpecified and d2.feature_dim_axis == 1


def test_Data_copy_add_dim_by_tag_unbroadcast_spatial_sparse():
    d = Data(name="t", shape=(None,), dtype="int32", sparse=True, dim=None, batch_dim_axis=None, feature_dim_axis=None)
    tag = SpatialDim("spatial:0:range", dimension=6)
    d2 = d.copy_add_dim_by_tag(tag, unbroadcast=True)
    print("d2:", d2)
    assert d2.batch_shape == (None, 6)
    assert d2.sparse
    assert d2.feature_dim_axis_or_unspecified is d.feature_dim_axis_or_unspecified


def test_Data_copy_add_dim_by_tag_unbroadcast_spatial():
    d = Data(name="ts", shape=(None,), time_dim_axis=None)
    tag = SpatialDim("spatial:0:ts", dimension=6)
    d2 = d.copy_add_dim_by_tag(tag, unbroadcast=True, axis=-1)
    assert d2.shape == (None, 6)


def test_Data_copy_add_dim_by_tag_sparse_unbroadcast_feature():
    d = Data(name="t", shape=(), dtype="int32", sparse=True, dim=None, time_dim_axis=None)
    tag = FeatureDim("feature:t", dimension=6)
    d2 = d.copy_add_dim_by_tag(tag, unbroadcast=True)
    # The feature axis should become a spatial axis in this case.
    assert d2.shape == (6,) and d2.sparse and d2.dim is None and d2.feature_dim_axis is None


def test_Data_copy_move_axis_time_to_end():
    d1 = Data(name="att_weights", shape=(None, None, 4))
    d2 = d1.copy_move_axis(d1.time_dim_axis, -1)
    assert d2.shape == (None, 4, None) and d2.feature_dim_axis == 2 and d2.time_dim_axis == 3


def test_Data_template_from_constant_bool():
    value = False
    out = Data.template_from_constant(
        value, name="bool_const", shape=None, dtype=None, with_batch_dim=False, sparse_dim=None
    )
    assert out.batch_shape == ()
    assert out.dtype == "bool"


def test_Data_copy_feat_with_vocab():
    from returnn.tf.util.data import batch_dim, FeatureDim, SpatialDim
    from returnn.datasets.util.vocabulary import Vocabulary

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feat", dimension=3)
    vocab = Vocabulary(vocab_file=None, labels=["a", "b", "c"], unknown_label=None)
    feat_dim.vocab = vocab
    data = Data("data", dim_tags=[batch_dim, time_dim, feat_dim])
    assert data.vocab is vocab
    data2 = data.copy()
    assert data2.vocab is vocab


def test_Data_verify_out_shape_optional_implicit_dim():
    # https://github.com/rwth-i6/returnn/issues/1153
    from returnn.tf.util.data import batch_dim, FeatureDim, SpatialDim, BatchInfo, VerifyOutShapeException

    batch = BatchInfo.make_global_batch_info(-1)
    time_dim = SpatialDim("time")
    time_dim.batch = batch
    time_dim.dyn_size_ext = Data("dyn_size_ext", dim_tags=[batch_dim], dtype="int32", batch=batch)
    feat_dim = FeatureDim("feat", dimension=3)
    x = Data("x", dim_tags=[time_dim, feat_dim])
    try:
        x.verify_out_shape({time_dim, feat_dim})
    except VerifyOutShapeException as exc:
        print("Got expected exception:", exc)
        assert "Missing dims" in str(exc)
    else:
        raise Exception("did not get expected exception")
    # This should not raise an exception:
    x.verify_out_shape({time_dim, feat_dim}, allow_missing_implicit_dims=True)


def test_Data_auto_create_placeholders_same_dim_tags_as_existing():
    # Came up via: https://github.com/rwth-i6/returnn/pull/1143
    n_out = 3
    time_tag = SpatialDim("time")
    with tf.Graph().as_default() as graph, tf_compat.v1.Session(graph=graph) as session:
        assert isinstance(graph, tf.Graph)
        data = Data("data", dim=n_out, same_dim_tags_as={"t": time_tag}, auto_create_placeholders=True)
        classes = Data(
            "classes", dim=n_out, sparse=True, same_dim_tags_as={"t": time_tag}, auto_create_placeholders=True
        )
        assert time_tag.dyn_size is not None  # this is not so relevant and might change
        seq_len = time_tag.dyn_size
        assert seq_len is data.get_sequence_lengths() is classes.get_sequence_lengths()
        assert seq_len.op.type == "Placeholder"
        placeholder_ops = [op for op in graph.get_operations() if op.type == "Placeholder"]
        assert_equal(set(placeholder_ops), {data.placeholder.op, classes.placeholder.op, time_tag.dyn_size.op})


def test_Data_copy_masked_0():
    x = Tensor("b_out", shape=(None, 3), dtype="float32", auto_create_placeholders=True)
    y = x.copy_masked(0)
    rnd = numpy.random.RandomState(3)
    session.run(
        y.raw_tensor, feed_dict={x.raw_tensor: rnd.normal(size=(2, 5, 3)), x.dims[1].dyn_size_ext.raw_tensor: [5, 4]}
    )


def test_Dim_copy():
    # https://github.com/rwth-i6/returnn/issues/860
    import copy

    a = SpatialDim("a")
    assert a == copy.copy(a)
    assert a == copy.deepcopy(a)


def test_Dim_pickle():
    from returnn.tf.util.data import batch_dim, single_step_dim
    import pickle

    a = SpatialDim("a")
    a_copy = pickle.loads(pickle.dumps(a))
    batch_dim_copy = pickle.loads(pickle.dumps(batch_dim))
    single_step_dim_copy = pickle.loads(pickle.dumps(single_step_dim))
    assert a != a_copy
    assert batch_dim_copy == batch_dim
    assert single_step_dim_copy == single_step_dim


def test_Dim_sorted():
    from returnn.util.basic import obj_diff_str

    a = SpatialDim("a")
    b = SpatialDim("b", 2)
    c = FeatureDim("c", 3)
    print(sorted((c, a, c, b)))
    assert sorted((c, a, c, b)) == [a, b, c, c]  # order defined by creation index (somewhat arbitrary...)
    print(obj_diff_str({"key": [a, b]}, {"key": [b, c]}))


def test_Dim_MarkedDim_sorted():
    from returnn.tf.util.data import ImplicitSparseDim, ImplicitDynSizeDim

    a = SpatialDim("a")
    b = SpatialDim("b", 2)
    a_implicit = ImplicitSparseDim(a)
    a_implicit2 = ImplicitDynSizeDim(a)
    b_implicit = ImplicitSparseDim(b)
    ls = [a, b, a_implicit, a_implicit2, b_implicit]
    print(ls)
    print(sorted(ls))
    # Test current order, but the order itself doesn't really matter for anything.
    assert_equal(sorted(ls), [a, b, a_implicit2, a_implicit, b_implicit])


def test_Dim_find_matching_dim_map_match_priority():
    in_dim = Dim(7, name="in")
    out_dim = in_dim
    filter_in_dim = in_dim.copy(match_priority=1)
    filter_size_dim = Dim(4, name="filter_size")
    filter_ = Tensor("filter", [out_dim, filter_in_dim, filter_size_dim], dtype="float32")
    filter_feat_dim_map = filter_.find_matching_dim_map(
        other=Data("dummy", [filter_in_dim, out_dim], dtype="float32"), other_axes=[0, 1]
    )
    assert_equal(filter_feat_dim_map, {0: 1, 1: 0})


def test_ExternData_ext_Data_batch_info():
    # https://github.com/rwth-i6/returnn_common/issues/193
    # https://github.com/rwth-i6/returnn/issues/975
    from returnn.tf.util.data import Data, BatchInfo, SpatialDim, FeatureDim, batch_dim
    from returnn.tf.network import ExternData

    time_dim = SpatialDim("time")
    in_dim = FeatureDim("in", 3)
    x = Data("x", dim_tags=[batch_dim, time_dim, in_dim])
    # This is how it is done in returnn-common construction, to set a custom dummy batch info.
    # There is no reason why this should not be fine; we want that this is supported.
    x.batch = BatchInfo.make_global_batch_info(-1)
    x.dim_tags[1].dyn_size_ext = Data(
        name="x_default_dyn_size_ext", dim_tags=[batch_dim], dtype=Data.size_dtype, batch=x.batch
    )
    x.sanity_check()  # still fine
    x.dim_tags[1]._maybe_update()  # might trigger some error
    print("(1) x:", x)

    with tf.Graph().as_default() as graph, tf_compat.v1.Session(graph=graph) as session:
        data = Data("x", dim_tags=[batch_dim, time_dim, in_dim], auto_create_placeholders=True)
        extern_data = ExternData()
        extern_data.data["x"] = data
        extern_data.init_batch_info()
        data.sanity_check()
        assert data.batch != x.batch

        # x.sanity_check() might fail now. but this is not really relevant. x.copy() matters.
        y = x.copy()  # failed earlier due to dim tag batch info mismatch
        y.sanity_check()
        assert data.dim_tags[1].dyn_size_ext
        x.dim_tags[1]._maybe_update()  # might trigger some error

        # In returnn_common, when get_network is called again,
        # it will still have the old graph and session active,
        # and then call nn.get_extern_data, ...
        print("(2) x:", x, x.batch, x.dim_tags[0].batch, x.batch == x.dim_tags[0].batch, x.batch == x.dim_tags[1].batch)
        x.batch = x.dim_tags[0].batch
        print(
            "(3) x:",
            x,
            x.batch,
            x.dim_tags[0].batch,
            x.batch == x.dim_tags[0].batch,
            x.batch == x.dim_tags[1].batch,
            x.dim_tags[1].dyn_size_ext,
        )
        if not x.dim_tags[1].dyn_size_ext:
            x.dim_tags[1].dyn_size_ext = Data(
                name="x_default_dyn_size_ext_new", dim_tags=[batch_dim], dtype=Data.size_dtype, batch=x.batch
            )
            print("(3a) x:", x)

        # Check again.
        # x.sanity_check() might fail now. but this is not really relevant. x.copy() matters.
        y = x.copy()  # failed earlier due to dim tag batch info mismatch
        y.sanity_check()
        assert data.dim_tags[1].dyn_size_ext


def test_dim_math_basics():
    a = SpatialDim("a")
    b = SpatialDim("b")
    assert a == a
    assert (a + 2 - 2) == a
    assert a + b == a + b
    assert a + b != b + a  # not commutative
    assert a * b == a * b
    assert a * b != b * a  # not commutative
    assert 2 * a == a + a
    assert a * 2 != 2 * a
    assert 2 * a + b == a + a + b
    assert a + b - b == a
    assert a + 2 * b - b + -b == a
    assert a * b + b == (a + 1) * b
    assert (a + b) * 2 == a * 2 + b * 2
    assert 0 + a + 0 == a
    assert sum([0, a, 0, a, 0]) == 2 * a


def test_dim_math_double_neg():
    a = SpatialDim("a")
    assert --a == a


def test_dim_math_mul_div():
    a = SpatialDim("a")
    b = SpatialDim("b")
    assert (a * b) // b == a
    assert (b * a) // b != a
    assert (b * a).div_left(b) == a


def test_dim_math_div():
    a = SpatialDim("a")
    b = SpatialDim("b")
    c = SpatialDim("c", 14)
    d = SpatialDim("d", 10)
    assert a // 2 + b // 2 != (a + b) // 2  # only allowed when divisible but this is unknown here for dyn dims
    assert c // 2 + d // 2 == (c + d) // 2


def test_dim_math_div_mul():
    a = FeatureDim("a", 10)
    b = FeatureDim("b", 2)
    c = SpatialDim("c")
    assert a // b == a // b
    assert (a // b) * b == a
    assert b * a.div_left(b) == a
    assert (c // b) * b != c


def test_dim_math_div_div():
    a = SpatialDim("a")
    b = a.ceildiv_right(2)
    b = b.ceildiv_right(3)
    c = a.ceildiv_right(6)
    print(a, b, c)
    assert b == c


def test_dim_math_static_self_att_example():
    num_heads = SpatialDim("num_heads", dimension=2)
    key_dim_total = FeatureDim("key_dim_total", dimension=6)
    key_dim_per_head = key_dim_total // num_heads
    assert key_dim_per_head.dimension == 3
    value_dim_total = FeatureDim("value_dim_total", dimension=10)
    value_dim_per_head = value_dim_total // num_heads
    qkv_dim_total = 2 * key_dim_total + value_dim_total
    qkv_dim_per_head = 2 * key_dim_per_head + value_dim_per_head
    assert qkv_dim_total.dimension == 6 * 2 + 10
    assert qkv_dim_per_head.dimension == (6 * 2 + 10) // 2
    assert key_dim_total + key_dim_total + value_dim_total == qkv_dim_total
    assert 2 * key_dim_total + value_dim_total == qkv_dim_total
    assert key_dim_per_head * num_heads == key_dim_total
    assert qkv_dim_per_head * num_heads == qkv_dim_total


def test_dim_math_static_self_att_feat_last():
    num_heads = SpatialDim("num_heads", dimension=2)
    key_dim_total = FeatureDim("key_dim_total", dimension=6)
    key_dim_per_head = key_dim_total.div_left(num_heads)
    assert key_dim_per_head.dimension == 3
    value_dim_total = FeatureDim("value_dim_total", dimension=10)
    value_dim_per_head = value_dim_total.div_left(num_heads)
    qkv_dim_total = 2 * key_dim_total + value_dim_total
    qkv_dim_per_head = 2 * key_dim_per_head + value_dim_per_head
    assert qkv_dim_total.dimension == 6 * 2 + 10
    assert qkv_dim_per_head.dimension == (6 * 2 + 10) // 2
    assert key_dim_total + key_dim_total + value_dim_total == qkv_dim_total
    assert 2 * key_dim_total + value_dim_total == qkv_dim_total
    assert num_heads * key_dim_per_head == key_dim_total
    assert num_heads * qkv_dim_per_head == qkv_dim_total


def test_dim_math_static_add_mul():
    a = FeatureDim("a", dimension=3)
    b = 2 * a
    c = a + a
    assert b == c


def test_dim_math_static_div_mul():
    num_heads = SpatialDim("num_heads", dimension=2)
    key_dim_total = FeatureDim("key_dim_total", dimension=6)
    key_dim_per_head = key_dim_total // num_heads
    key_dim_total_ = key_dim_per_head * num_heads
    assert key_dim_total_ == key_dim_total


def test_dim_math_feature_type():
    feat = FeatureDim("feature", dimension=1)
    feat_sum = feat + feat
    assert feat_sum.dimension == 2 and feat_sum.kind == Dim.Types.Feature


def test_dim_math_feature_type2():
    feat1 = FeatureDim("feature1", dimension=3)
    feat2 = FeatureDim("feature2", dimension=5)
    feat_sum = feat1 + feat1 + feat2
    assert feat_sum.dimension == 11 and feat_sum.kind == Dim.Types.Feature


def test_dim_math_pad_stag_description():
    time = SpatialDim("time:var:extern_data:data")
    pad_right = time + 2
    assert "extern_data:data" in pad_right.description
    data = Data("padded", dim_tags=[pad_right])
    assert data.get_axis_from_description("stag:extern_data:data") == 0


def test_dim_math_pad_conv_valid():
    time = SpatialDim("time:var:extern_data:data")
    padded = 2 + time + 2
    conv_valid = padded.sub_right(2).sub_left(2)
    assert conv_valid == time


def test_dim_math_pad_conv_valid_in_ctx():
    from returnn.tf.util.data import BatchInfo, ControlFlowContext

    time = SpatialDim("time:var:extern_data:data")
    loop_dim = SpatialDim("time:var:extern_data:classes")
    batch_info = BatchInfo.make_global_batch_info(-1)
    ctx = ControlFlowContext(kind=ControlFlowContext.Types.Loop, identifier="loop")
    ctx.loop_spatial_dim = loop_dim
    time_ = time.get_for_batch_ctx(batch=batch_info, ctx=ctx)
    # Note: once time is actually defined (dyn_size_ext is set), the following assert would not be the case,
    # as the base can be used for any control flow context, and the _can_use_in_ctx logic applies.
    # However, in this test case here, it is yet undefined, which is also a valid case during template construction.
    assert time_.control_flow_ctx == ctx
    padded = 2 + time_ + 2
    assert padded == 2 + time + 2
    padded_ = padded.get_for_batch_ctx(batch=batch_info, ctx=ctx)
    # Same here as above.
    assert padded_.control_flow_ctx == ctx
    conv_valid = (-2) + padded_ + (-2)
    assert conv_valid == time


def test_dim_math_pad_conv_valid_in_ctx_derived():
    # Behavior when this is inside a loop.
    # Similar as in test_attention_convolutional_feedback_variant1.
    from returnn.tf.util.data import BatchInfo, ControlFlowContext

    loop_dim = SpatialDim("time:var:extern_data:classes")
    batch_info = BatchInfo.make_global_batch_info(-1)
    ctx = ControlFlowContext(kind=ControlFlowContext.Types.Loop, identifier="loop")
    ctx.loop_spatial_dim = loop_dim
    time_undefined = Dim(kind=Dim.Types.Spatial, description="time_undefined", dimension=None)
    time_ = time_undefined.get_for_batch_ctx(batch=batch_info, ctx=ctx)
    padded = 2 + time_ + 2
    padded_ = padded.get_for_batch_ctx(batch=batch_info, ctx=ctx)
    conv = Dim(kind=Dim.Types.Spatial, description="conv", dimension=None, derived_from_tag=padded_)
    conv_valid = (-2) + padded_ + (-2)
    assert conv_valid is time_undefined  # implementation detail, not the relevant test (but we should have equality)
    conv_valid.declare_same_as(conv)
    conv_derived_base = conv_valid.get_same_derived_base()
    assert conv_derived_base is time_undefined
    assert conv == time_undefined


def test_dim_math_pad_dummy_equal():
    from returnn.tf.util.data import batch_dim

    time = SpatialDim("time")
    pad_dim = SpatialDim("prefix", 1)
    x = Data("x", dim_tags=[batch_dim, pad_dim + time])
    time_ = pad_dim + time
    time_ = x.get_dim_tag_from_description(time_)
    time__ = 1 + time
    assert time_ != time__
    time_.declare_same_as(time__)
    assert time_ == time__
    assert time_ == 1 + time
    assert time__ == 1 + time
    # The following commented-out checks are tricky.
    # But also, it's not so clear whether we really want them.
    # assert time_ == pad_dim + time  # tricky
    # assert time__ == pad_dim + time  # tricky
    # assert 1 + time == pad_dim + time  # more tricky than the others...
    x.verify_out_shape({batch_dim, time__})


def test_dim_math_pad_dummy_equal2():
    # The same as test_dim_math_pad_dummy_equal but with explicit BatchInfo.
    # We have had the case where just this aspect caused the test to fail.
    # Even worse: the declare_same_as really caused different behavior due to this.
    # More specifically, the batch causes that Data._adapt_batch_consistent_dim_tags
    # and thus Dim.get_for_batch_ctx gets called, and this makes the difference.
    # This causes a call to complete_dyn_size, and then the time_ dim is defined.
    # I.e. in the declare_same_as call, `self.is_dim_known() and not other.is_dim_known()` is True
    # in contrast to before. This swaps the order of the declare_same_as calls.
    from returnn.tf.util.data import batch_dim, BatchInfo

    batch = BatchInfo.make_global_batch_info(-1)
    time = SpatialDim("time")
    time.batch = batch
    time.dyn_size_ext = Data("dyn_size_ext", dim_tags=[batch_dim], dtype="int32", batch=batch)
    pad_dim = SpatialDim("prefix", 1)
    x = Data("x", dim_tags=[batch_dim, pad_dim + time], batch=batch)
    time_ = pad_dim + time
    time_ = x.get_dim_tag_from_description(time_)
    time__ = 1 + time
    assert time_ != time__
    time_.declare_same_as(time__)
    assert time_ == time__
    assert time_ == 1 + time
    assert time__ == 1 + time
    # assert time_ == pad_dim + time  # still tricky
    x.verify_out_shape({batch_dim, time__})


def test_dim_math_pad_dummy_equal3():
    # Again very similar to test_dim_math_pad_dummy_equal2
    # but more directly testing with Dim.get_for_batch_ctx.
    # Like test_prev_target_seq from returnn_common.
    from returnn.tf.util.data import batch_dim, BatchInfo

    batch = BatchInfo.make_global_batch_info(-1)
    spatial_dim = SpatialDim("time")
    spatial_dim.batch = batch
    spatial_dim.dyn_size_ext = Data("dyn_size_ext", dim_tags=[batch_dim], dtype="int32", batch=batch)
    pad_dim = SpatialDim("bos-prefix", 1)
    dim__ = sum([pad_dim, spatial_dim])
    dim__ = dim__.get_for_batch_ctx(batch=batch, ctx=None)
    dim__.declare_same_as(1 + spatial_dim)
    print(dim__)
    assert dim__ == 1 + spatial_dim


def test_dim_math_add_dyn_defined():
    from returnn.tf.util.data import batch_dim, BatchInfo

    batch = BatchInfo.make_global_batch_info(-1)
    d = SpatialDim("time")
    print("d=", d)
    d.batch = batch
    d.dyn_size_ext = Data("dyn_size_ext", dim_tags=[batch_dim], batch=batch)
    print("d=", d)
    y = sum([d - 1, d])
    print("y=", y)
    y = y.get_for_batch_ctx(batch=batch, ctx=None)
    print("y=", y)
    assert y.dyn_size_ext and y.dyn_size_ext.dim_tags == (batch_dim,)


def test_dim_math_feat_declare_same_as_circle():
    # We had this logic in returnn_common relative_positional_encoding,
    # where we used concat.
    # The implementation of relative_positional_encoding might change at some point,
    # however, this logic here should still work.
    # At some point, this failed because there was an infinite recursion loop
    # in the Dim.__hash__ function due to the declare_same_as.
    feat_dim = FeatureDim("feat", 12)
    feat2_dim = feat_dim // 2
    feat_dim_ = feat2_dim + feat2_dim
    print(feat_dim_)
    feat_dim_.declare_same_as(feat_dim)
    print(feat_dim_, "==", feat_dim)
    assert feat_dim_ == feat_dim
    s = {feat_dim, feat2_dim, feat_dim_}
    assert s == {feat_dim, feat2_dim}


def test_dim_math_derived():
    from returnn.tf.util.data import SpatialDim

    time_dim = SpatialDim("time")
    time_dim_2 = time_dim * 2
    assert time_dim_2.derived_from_tag == time_dim
    assert time_dim_2.get_same_derived_base() == time_dim


def test_sequence_mask_len_via_loop():
    seq_len = tf.while_loop(
        cond=lambda x: tf.less(x[0], 2), body=lambda x: x + 1, loop_vars=[tf.convert_to_tensor([1, 2])]
    )
    if isinstance(seq_len, list):  # TF 2
        seq_len = seq_len[0]
    assert not has_control_flow_context(seq_len)
    mask = sequence_mask_time_major(seq_len)
    seq_len_v, mask_v = session.run((seq_len, mask))
    print(seq_len_v)
    assert_equal(seq_len_v.tolist(), [2, 3])
    print(mask_v)
    assert_equal(mask_v.tolist(), [[True, True], [True, True], [False, True]])


def test_get_initializer_zero():
    shape = (2, 3)
    initializer = get_initializer(0.0)
    v = initializer(shape)
    assert_almost_equal(session.run(v), numpy.zeros(shape))


def test_get_initializer_const_formula():
    shape = (2, 3)
    initializer = get_initializer("log(1.0 / 4.0)")
    v = initializer(shape)
    assert_almost_equal(session.run(v), numpy.zeros(shape) + numpy.log(1.0 / 4.0))


def test_get_initializer_zeros():
    shape = (2, 3)
    initializer = get_initializer("zeros")
    v = initializer(shape)
    assert_almost_equal(session.run(v), numpy.zeros(shape))


def test_get_initializer_constant():
    shape = (2, 3)
    initializer = get_initializer("constant")
    v = initializer(shape)
    assert_almost_equal(session.run(v), numpy.zeros(shape))


def test_get_initializer_xavier():
    shape = (2, 3)
    initializer = get_initializer("xavier")
    v = initializer(shape)
    assert_equal(session.run(v).shape, shape)  # returns some random matrix


def test_get_initializer_glorot_uniform():
    shape = (2, 3)
    initializer = get_initializer("glorot_uniform")
    v = initializer(shape)
    assert_equal(session.run(v).shape, shape)  # returns some random matrix


def test_get_initializer_glorot_normal_with_scale():
    shape = (2, 3)
    initializer = get_initializer('VarianceScaling(scale=6.0, mode="fan_avg", distribution="normal")')
    v = initializer(shape)
    assert_equal(session.run(v).shape, shape)  # returns some random matrix


def test_get_initializer_uniform():
    shape = (2, 3)
    initializer = get_initializer("RandomUniform(-0.01, 0.01)")
    v = initializer(shape)
    assert_equal(session.run(v).shape, shape)  # returns some random matrix


def test_get_initializer_gauss():
    shape = (2, 3)
    initializer = get_initializer("RandomNormal(0.0, 0.01)")
    v = initializer(shape)
    assert_equal(session.run(v).shape, shape)  # returns some random matrix


def test_wrap_distribution_non_zero():
    assert_almost_equal(session.run(wrap_distribution_non_zero(0.1, zero_limit=0.5, limit=2.0)), 0.575)
    assert_almost_equal(session.run(wrap_distribution_non_zero(-0.1, zero_limit=0.5, limit=2.0)), -0.575)


def test_close_event_writer_thread():
    import threading
    import tempfile
    from tensorflow.python.summary.writer.event_file_writer import EventFileWriter, _EventLoggerThread

    def count_event_logger_threads():
        return len([t for t in threading.enumerate() if isinstance(t, _EventLoggerThread)])

    tmp_dir = tempfile.mkdtemp()
    writer = tf_compat.v1.summary.FileWriter(tmp_dir)
    assert_equal(count_event_logger_threads(), 1)
    assert isinstance(writer.event_writer, EventFileWriter)
    assert isinstance(writer.event_writer._worker, _EventLoggerThread)
    writer.close()

    # https://github.com/tensorflow/tensorflow/issues/4820
    # The _EventLoggerThread is still running (at least in TF 1.1.0).
    stop_event_writer_thread(writer)
    assert_equal(count_event_logger_threads(), 0)


def test_single_strided_slice():
    x = tf.expand_dims(tf.range(10), axis=0)
    assert_equal(list(tf.shape(x).eval()), [1, 10])
    assert_equal(list(single_strided_slice(x, axis=1, begin=3, end=6, step=2)[0].eval()), [3, 5])
    assert_equal(list(single_strided_slice(x, axis=1, begin=4)[0].eval()), list(range(4, 10)))
    assert_equal(list(single_strided_slice(x, axis=1, end=3)[0].eval()), [0, 1, 2])
    assert_equal(list(single_strided_slice(x, axis=tf.constant(1), end=3)[0].eval()), [0, 1, 2])
    assert_equal(list(single_strided_slice(x, axis=tf.constant(-1), end=3)[0].eval()), [0, 1, 2])
    x2 = tf.reshape(tf.range(9), (3, 3))
    assert_equal(list(x2[0].eval()), [0, 1, 2])
    assert_equal(list(tf.squeeze(single_strided_slice(x2, axis=tf.constant(0), end=1), axis=0).eval()), [0, 1, 2])


def test_slice_pad_zeros():
    x = tf.constant([1, 2, 3, 4])
    assert_equal(list(slice_pad_zeros(x, begin=1, end=3).eval()), [2, 3])
    assert_equal(list(slice_pad_zeros(x, begin=-2, end=2).eval()), [0, 0, 1, 2])
    assert_equal(list(slice_pad_zeros(x, begin=-2, end=6).eval()), [0, 0, 1, 2, 3, 4, 0, 0])
    assert_equal(list(slice_pad_zeros(x, begin=2, end=6).eval()), [3, 4, 0, 0])


def test_circular_pad():
    x = tf.reshape(tf.range(9), (3, 3))
    assert_equal(list(x[0].eval()), [0, 1, 2])
    x_ref = numpy.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    numpy.testing.assert_equal(x.eval(), x_ref)
    y = circular_pad(x, paddings=1)
    y_ref = numpy.array([[8, 6, 7, 8, 6], [2, 0, 1, 2, 0], [5, 3, 4, 5, 3], [8, 6, 7, 8, 6], [2, 0, 1, 2, 0]])
    numpy.testing.assert_equal(y.eval(), y_ref)

    x = tf.expand_dims(tf.reshape(tf.range(9), (3, 3)), axis=2)
    assert_equal(list(x[0, :, 0].eval()), [0, 1, 2])
    x_ref = numpy.array([[[0], [1], [2]], [[3], [4], [5]], [[6], [7], [8]]])
    numpy.testing.assert_equal(x.eval(), x_ref)
    y = circular_pad(x, paddings=1, axes=(0, 1))
    y_ref = numpy.array(
        [
            [[8], [6], [7], [8], [6]],
            [[2], [0], [1], [2], [0]],
            [[5], [3], [4], [5], [3]],
            [[8], [6], [7], [8], [6]],
            [[2], [0], [1], [2], [0]],
        ]
    )
    numpy.testing.assert_equal(y.eval(), y_ref)


def test_reuse_name_scope_double():
    with reuse_name_scope("double"):
        assert_equal(tf_compat.v1.get_default_graph()._name_stack, "double")
        with reuse_name_scope("sub"):
            assert_equal(tf_compat.v1.get_default_graph()._name_stack, "double/sub")
            assert_equal(get_current_name_scope(), "double/sub")


def test_reuse_name_scope_mix1():
    with reuse_name_scope("mix1"):
        assert_equal(tf_compat.v1.get_default_graph()._name_stack, "mix1")
        with tf.name_scope("sub"):
            assert_equal(tf_compat.v1.get_default_graph()._name_stack, "mix1/sub")
            # The following is not true because get_current_name_scope is only var-scope:
            # assert_equal(get_current_name_scope(), "mix1/sub")


def test_reuse_name_scope_mix2():
    with tf.name_scope("mix2"):
        with reuse_name_scope("sub"):
            assert_equal(tf_compat.v1.get_default_graph()._name_stack, "mix2/sub")
            # The following is not true because get_current_name_scope is only var-scope:
            # assert_equal(get_current_name_scope(), "mix2/sub")


def test_reuse_name_scope_mix3():
    with reuse_name_scope("mix3"):
        with tf_compat.v1.variable_scope("sub"):
            assert_equal(get_current_name_scope(), "mix3/sub")


def test_reuse_name_scope_mix4():
    with tf_compat.v1.variable_scope("mix4"):
        with reuse_name_scope("sub"):
            assert_equal(get_current_name_scope(), "mix4/sub")


def test_reuse_name_scope_2():
    with reuse_name_scope("lstm2"):
        with reuse_name_scope("rec") as scope:
            assert_is_instance(scope, tf_compat.v1.VariableScope)
            assert_equal(scope.name, "lstm2/rec")
            assert_equal(get_current_name_scope(), "lstm2/rec")
            with tf.name_scope("sub"):
                assert_equal(get_current_name_scope(), "lstm2/rec/sub")


def test_reuse_name_scope():
    with reuse_name_scope("lstm0"):
        with tf_compat.v1.variable_scope("rec"):
            a = tf_compat.v1.get_variable("a", shape=(3, 4))
            assert_is_instance(a, tf.Variable)
            assert_equal(a.name, "lstm0/rec/a:0")

            b = tf.Variable(name="b", initial_value=tf.zeros((2,)))
            assert_equal(b.name, "lstm0/rec/b:0")

    with reuse_name_scope("lstm0"):
        with reuse_name_scope("rec"):
            c = tf.Variable(name="c", initial_value=tf.zeros((2,)))
            assert_equal(c.name, "lstm0/rec/c:0")

            c2 = tf.Variable(name="c", initial_value=tf.zeros((2,)))
            assert_equal(c2.name, "lstm0/rec/c_1:0")


def test_reuse_name_scope_root():
    with reuse_name_scope("", absolute=True):
        pass


def test_reuse_var_scope():
    with tf_compat.v1.variable_scope("v1"):
        assert_equal(get_current_var_scope_name(), "v1")
        assert_equal(get_current_name_scope(), "v1")
        with tf_compat.v1.variable_scope("v2") as scope:
            assert_equal(get_current_var_scope_name(), "v1/v2")
            assert_equal(get_current_name_scope(), "v1/v2")
            with tf.name_scope("v3"):
                assert_equal(get_current_name_scope(), "v1/v2/v3")
                assert_equal(get_current_var_scope_name(), "v1/v2")
                assert_equal(scope.name, "v1/v2")
                # Note: tf.compat.v1.variable_scope(scope) is broken here.
                with reuse_name_scope(scope):
                    assert_equal(get_current_var_scope_name(), "v1/v2")
                    assert_equal(get_current_name_scope(), "v1/v2")


def test_name_var_scope_mixing():
    with tf_compat.v1.variable_scope("mv1"):
        assert_equal(get_current_var_scope_name(), "mv1")
        assert_equal(get_current_name_scope(), "mv1")
        with tf_compat.v1.variable_scope("v2") as scope:
            assert_equal(get_current_var_scope_name(), "mv1/v2")
            assert_equal(get_current_name_scope(), "mv1/v2")
            with tf.name_scope("v3"):
                assert_equal(get_current_name_scope(), "mv1/v2/v3")
                assert_equal(get_current_var_scope_name(), "mv1/v2")
                assert_equal(scope.name, "mv1/v2")
                # Note: tf.compat.v1.variable_scope("v4") is broken here.
                with reuse_name_scope("v4"):
                    assert_equal(get_current_var_scope_name(), "mv1/v2/v3/v4")
                    assert_equal(get_current_name_scope(), "mv1/v2/v3/v4")
                    with reuse_name_scope(scope):
                        assert_equal(get_current_var_scope_name(), "mv1/v2")
                        assert_equal(get_current_name_scope(), "mv1/v2")


def test_reuse_name_scope_of_tensor():
    with tf.name_scope("scope1") as scope1:
        x = tf.constant(42)
    with tf.name_scope("scope2") as scope2:
        assert_equal(get_current_name_scope() + "/", scope2)
        with reuse_name_scope_of_tensor(x):
            assert_equal(get_current_name_scope() + "/", scope1)


def test_reuse_name_scope_of_tensor_root():
    x = tf.constant(42)
    with tf.name_scope("scope2") as scope2:
        assert_equal(get_current_name_scope() + "/", scope2)
        with reuse_name_scope_of_tensor(x):
            assert_equal(get_current_name_scope(), "")


def test_loop_var_creation():
    """
    test_loop_var_creation

    TF error:
    InvalidArgumentError: The node 'while/w/Assign' has inputs from different frames.
    The input 'while/j' is in frame 'while/while/'. The input 'while/w' is in frame ''.

    Related TF bugs:
    https://github.com/tensorflow/tensorflow/issues/3114
    https://github.com/tensorflow/tensorflow/issues/4478
    https://github.com/tensorflow/tensorflow/issues/8604
    """

    # tf.compat.v1.reset_default_graph()  # Strange, this does not work.
    i = tf.constant(0)

    def body(i):
        # None of these works, with error:
        # InvalidArgumentError: The node 'while/w/Assign' has inputs from different frames.
        # The input 'while/j' is in frame 'while/while/'. The input 'while/w' is in frame ''.
        # w = tf.Variable(tf.constant(1))
        # w = tf.Variable(tf.constant_initializer(value=1, dtype=tf.int32)(shape=()))
        # However, resetting the control dependencies will also reset the frame.
        with default_control_flow_ctx():
            # Note: tf.Variable directly will have this problem, as tf.constant() is in the current ctx.
            w1 = tf.Variable(name="w1", initial_value=tf.constant(1))
        # tf.get_variable only works well in TF1 control flow.
        # When we use TF2 control flow, we anyway should use default_control_flow_ctx().
        with default_control_flow_ctx():
            w2 = tf_compat.v1.get_variable("w2", shape=(), dtype=tf.int32, initializer=tf.constant_initializer(2))
        return [i + w1 + w2]

    loop = tf.while_loop(lambda i: tf.less(i, 5), body, [i])
    session.run(tf_compat.v1.global_variables_initializer())
    session.run(loop)


def test_dot_simple():
    n_time, n_batch = 7, 11
    n_in, n_out = 3, 5
    weights = tf_compat.v1.random_normal((n_in, n_out))
    x = tf_compat.v1.random_normal((n_time, n_batch, n_in))
    y = dot(x, weights)
    y.set_shape((n_time, n_batch, n_out))
    session.run(y)


def test_gather_nd_grad():
    # https://github.com/tensorflow/tensorflow/issues/9406
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/gather_nd_op.cc
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/scatter_nd_op.cc
    # Fixed in TF 1.1.0.
    assert_min_tf_version((1, 1), "tf.gather_nd")
    n_base_time = 5
    n_in = 7
    n_beam = 3
    n_batch = 1
    base = tf.ones((n_base_time, n_batch, n_in))  # (base_time,batch,n_in)
    idxs_exp = tf.constant(
        0, shape=(n_beam, n_batch, 2), name="idxs_exp"
    )  # (beam,batch,2), where the 2 stands for (base_time,batch)
    # Thus K == 2. gather_nd out will be idxs_exp.shape[:2] + params.shape[2:] = (beam,batch,n_in).
    gathered = tf.gather_nd(base, idxs_exp)  # (beam,batch,n_in)
    gathered_shape, _ = session.run([tf.shape(gathered), gathered])
    assert_equal(list(gathered_shape), [n_beam, n_batch, n_in])

    base_grad = tf.gradients(gathered, base)
    assert base_grad is not None
    session.run(base_grad)


def test_scatter_nd():
    # https://github.com/tensorflow/tensorflow/issues/9406
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/scatter_nd_op.cc
    # Fixed in TF 1.1.0.
    assert_min_tf_version((1, 1), "tf.scatter_nd")
    n_base_time = 5
    n_in = 7
    n_beam = 3
    n_batch = 1
    ref_grad = tf.scatter_nd(
        indices=tf.zeros((n_beam, n_batch, 2), dtype=tf.int32),
        updates=tf.ones((n_beam, n_batch, n_in)),
        shape=(n_base_time, n_batch, n_in),
    )
    session.run(ref_grad)


def test_nd_indices_scatter_nd_time_major():
    def rel_embed(x, v, t):
        """
        :param Data x: energy_in. (B, T, K) or (T, B, K)
        :param Data v: t_rel_var. (B, Ts, K)
        :param tf.Tensor t: (B,), int32
        :rtype: tf.Tensor
        """
        import tensorflow as tf
        from returnn.tf.util.basic import nd_indices

        v = v.copy_move_axis(v.batch_dim_axis, x.batch_dim_axis)  # t_rel_var. (B, Ts, K) or (Ts, B, K)
        assert v.feature_dim_axis == x.feature_dim_axis and v.dim == x.dim
        t = t + 1  # shift by 1, because we init at -1
        # t = tf.Print(t, ["t:", t])
        time_dim = tf.shape(x.placeholder)[x.time_dim_axis]
        batch_dim = tf.shape(x.placeholder)[x.batch_dim_axis]
        assert len(v.shape) == 2 and all([isinstance(d, int) for d in v.shape])
        ts_dim = v.shape[0]
        assert x.batch_dim_axis in [0, 1]
        indices = tf.expand_dims(tf.range(ts_dim), axis=x.batch_dim_axis)  # (1,Ts) or (Ts,1)
        indices = indices + tf.expand_dims(t, axis=1 - x.batch_dim_axis)  # (B,Ts) or (Ts,B)
        max_t = tf.maximum(tf.reduce_max(indices) + 1, time_dim + 1)
        indices = nd_indices(indices, batch_axis=x.batch_dim_axis)  # (B,Ts,2) or (Ts,B,2)
        x0 = tf.scatter_nd(
            indices=indices,
            updates=v.placeholder,
            shape=[batch_dim, max_t, x.dim] if x.batch_dim_axis == 0 else [max_t, batch_dim, x.dim],
        )  # (B,T,K) or (T,B,K)
        if x.batch_dim_axis == 0:
            x0 = x0[:, 1 : time_dim + 1]  # correct the shift from above
        else:
            x0 = x0[1 : time_dim + 1]  # correct the shift from above
        out = x.placeholder + x0
        # out = tf.Print(out, ["i:", network.get_rec_step_index(), "t:", t], summarize=5)
        return out

    n_batch = 3
    t = tf.convert_to_tensor([4, 3, 2])  # (B,)
    n_time = 7
    seq_len = tf.convert_to_tensor([7, 4, 5])
    n_ts = 2
    n_k = 5
    v = tf_compat.v1.random_normal((n_ts, n_k))
    v = expand_dims_unbroadcast(v, axis=0, dim=n_batch)  # (B,Ts,K)
    v = Data(name="v", shape=(n_ts, n_k), placeholder=v)
    x = tf_compat.v1.random_normal((n_batch, n_time, n_k))
    x = Data(name="x", shape=(None, n_k), placeholder=x, size_placeholder={0: seq_len})
    print(x)
    print(v)
    print(t)
    res1 = rel_embed(x, v, t)
    print("res1 (batch major):", res1)
    res2 = rel_embed(x.copy_as_time_major(), v, t)
    print("res2 (time major):", res2)
    session.run(res1)
    session.run(res2)


def test_dimshuffle():
    x = tf.zeros((2, 3, 5))
    assert_equal(list(session.run(tf.shape(x))), [2, 3, 5])
    assert_equal(list(session.run(tf.shape(dimshuffle(x, (1, 2, 0))))), [3, 5, 2])
    assert_equal(list(session.run(tf.shape(dimshuffle(x, ("x", 1, 2, 0))))), [1, 3, 5, 2])
    assert_equal(list(session.run(tf.shape(dimshuffle(x, ("x", 1, "x", 2, "x", 0, "x"))))), [1, 3, 1, 5, 1, 2, 1])
    x = tf.zeros((2, 1, 3))
    assert_equal(list(session.run(tf.shape(dimshuffle(x, (2, 0))))), [3, 2])
    assert_equal(list(session.run(tf.shape(dimshuffle(x, (2, "x", "x", 0))))), [3, 1, 1, 2])


def test_expand_multiple_dims():
    x = tf.zeros((2, 3, 5))
    assert_equal(list(session.run(tf.shape(x))), [2, 3, 5])
    assert_equal(list(session.run(tf.shape(expand_multiple_dims(x, (1, 2))))), [2, 1, 1, 3, 5])
    assert_equal(list(session.run(tf.shape(expand_multiple_dims(x, (1, 4))))), [2, 1, 3, 5, 1])
    assert_equal(list(session.run(tf.shape(expand_multiple_dims(x, (1, 3, 5))))), [2, 1, 3, 1, 5, 1])


def test_move_axis():
    x = tf.zeros((2, 3, 5))
    assert_equal(list(session.run(tf.shape(x))), [2, 3, 5])
    assert_equal(list(session.run(tf.shape(move_axis(x, old_axis=0, new_axis=1)))), [3, 2, 5])
    assert_equal(list(session.run(tf.shape(move_axis(x, old_axis=0, new_axis=2)))), [3, 5, 2])
    assert_equal(list(session.run(tf.shape(move_axis(x, old_axis=2, new_axis=0)))), [5, 2, 3])
    assert_equal(list(session.run(tf.shape(move_axis(x, old_axis=2, new_axis=1)))), [2, 5, 3])


def test_flatten_with_seq_len_mask():
    n_batch, n_time, n_dim = 3, 4, 2
    seq_lens = tf.convert_to_tensor([4, 3, 2])
    x = tf.convert_to_tensor(numpy.arange(0, n_batch * n_time * n_dim).reshape((n_time, n_batch, n_dim)))
    assert x.shape.ndims == 3
    print("x (time-major):", x.eval().tolist())
    print("x (batch-major):", x.eval().transpose(1, 0, 2).tolist())
    assert_equal(x.eval()[0].tolist(), [[0, 1], [2, 3], [4, 5]])
    assert_equal(x.eval()[:3, 0].tolist(), [[0, 1], [6, 7], [12, 13]])
    flat_bm = flatten_with_seq_len_mask(x, seq_lens=seq_lens, batch_dim_axis=1, time_dim_axis=0)
    assert flat_bm.shape.ndims == 2
    print("flat (batch-major):", flat_bm.eval().tolist())
    assert_equal(
        flat_bm.eval().tolist(), [[0, 1], [6, 7], [12, 13], [18, 19], [2, 3], [8, 9], [14, 15], [4, 5], [10, 11]]
    )
    flat_tm = flatten_with_seq_len_mask_time_major(x, seq_lens=seq_lens, batch_dim_axis=1, time_dim_axis=0)
    assert flat_tm.shape.ndims == 2
    print("flat (time-major):", flat_tm.eval().tolist())
    assert_equal(
        flat_tm.eval().tolist(), [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [18, 19]]
    )


def test_constant_with_shape():
    x = session.run(constant_with_shape(3, [2, 3]))
    assert_equal(x.shape, (2, 3))
    assert_equal(x.dtype, numpy.int32)
    assert_equal(x.flatten().tolist(), [3] * 2 * 3)

    x = session.run(constant_with_shape(7.0, [2, 3]))
    assert_equal(x.shape, (2, 3))
    assert_equal(x.dtype, numpy.float32)
    assert_equal(x.flatten().tolist(), [7.0] * 2 * 3)

    x = session.run(constant_with_shape(False, [2, 3]))
    assert_equal(x.shape, (2, 3))
    assert_equal(x.dtype, numpy.bool_)
    assert_equal(x.flatten().tolist(), [False] * 2 * 3)

    x = session.run(constant_with_shape(True, [2, 3]))
    assert_equal(x.shape, (2, 3))
    assert_equal(x.dtype, numpy.bool_)
    assert_equal(x.flatten().tolist(), [True] * 2 * 3)


def naive_windowed_batch(source, window, padding="same"):
    assert source.ndim == 3  # (time,batch,dim). not sure how to handle other cases
    if padding == "same":
        n_time = source.shape[0]
        w_right = window // 2
        w_left = window - w_right - 1
    elif padding == "valid":
        n_time = source.shape[0] - window + 1
        w_right = 0
        w_left = 0
    else:
        raise Exception("invalid padding %r" % padding)

    n_batch = source.shape[1]
    n_dim = source.shape[2]
    dtype = source.dtype
    pad_left = numpy.zeros((w_left, n_batch, n_dim), dtype=dtype)
    pad_right = numpy.zeros((w_right, n_batch, n_dim), dtype=dtype)
    padded = numpy.concatenate([pad_left, source, pad_right], axis=0)
    final = numpy.zeros((n_time, window, n_batch, n_dim), dtype=dtype)
    for t in range(n_time):
        for w in range(window):
            final[t, w] = padded[t + w]
    return final


def test_windowed_nd_small():
    n_time = 2
    n_batch = 2
    n_dim = 2
    window = 3
    source = numpy.arange(1, n_time * n_batch * n_dim + 1).reshape(n_time, n_batch, n_dim)
    print("source:")
    print(source)
    naive = naive_windowed_batch(source, window=window)
    real = windowed_nd(source, window_size=window, time_axis=0, new_window_axis=1).eval()
    print("naive:")
    print(naive)
    print("real:")
    print(real)
    numpy.testing.assert_almost_equal(naive, real)


def test_windowed_pad_valid_nd_small():
    n_time = 10
    n_batch = 1
    n_dim = 1
    window = 3
    source = numpy.arange(1, n_time * n_batch * n_dim + 1).reshape(n_time, n_batch, n_dim)
    print("source:")
    print(source)
    naive = naive_windowed_batch(source, window=window, padding="valid")
    real = windowed_nd(source, window_size=window, time_axis=0, new_window_axis=1, padding="valid").eval()
    print("naive:")
    print(naive)
    print("real:")
    print(real)
    numpy.testing.assert_almost_equal(naive, real)


def test_windowed_nd_big():
    n_time = 11
    n_batch = 5
    n_dim = 7
    window = 3
    numpy.random.seed(123)
    source = numpy.random.random((n_time, n_batch, n_dim)).astype("float32")
    naive = naive_windowed_batch(source, window=window)
    real = windowed_nd(source, window_size=window, time_axis=0, new_window_axis=1).eval()
    numpy.testing.assert_almost_equal(naive, real)


def test_windowed_nd_big_stride():
    n_time = 11
    n_batch = 5
    n_dim = 7
    window = 3
    stride = 2
    numpy.random.seed(123)
    source = numpy.random.random((n_time, n_batch, n_dim)).astype("float32")
    naive = naive_windowed_batch(source, window=window)[::stride]
    real = windowed_nd(source, window_size=window, time_axis=0, new_window_axis=1, stride=stride).eval()
    numpy.testing.assert_almost_equal(naive, real)


def naive_slice_nd(x, start, size):
    slices_shape = [x.shape[0], size] + list(x.shape)[2:]
    ys = numpy.zeros(shape=slices_shape)
    for i in range(len(start)):
        time_len = len(x[i])
        end = start[i] + size
        if time_len < end:
            end = time_len
        y = x[i][start[i] : end]

        # padding
        if time_len < start[i] + size:
            y = numpy.pad(y, [[0, start[i] + size - time_len], [0, 0]], mode="constant")
        ys[i] = y
    return ys


def test_slice_nd_small():
    n_batch = 3
    n_time = 4
    n_dim = 2
    size = 2
    start = numpy.array([0, 2, 3]).astype("int32")
    source = (
        numpy.arange(1, n_batch * n_time * n_dim + 1, dtype=numpy.float32)
        .reshape(n_batch, n_time, n_dim)
        .astype("float32")
    )
    source_tf = tf.constant(source)
    naive = naive_slice_nd(source, start, size)
    real = slice_nd(source_tf, start=start, size=size).eval()
    print("source:")
    print(source)
    print("naive:")
    print(naive)
    print("real:")
    print(real)
    numpy.testing.assert_almost_equal(naive, real)


def test_slice_nd_big():
    n_batch = 8
    n_time = 12
    n_dim = 4
    size = 4
    numpy.random.seed(123)
    start = numpy.random.randint(low=0, high=12, size=(n_batch,), dtype="int32")
    source = numpy.random.random((n_batch, n_time, n_dim)).astype("float32")
    source_tf = tf.constant(source)
    naive = naive_slice_nd(source, start, size)
    real = slice_nd(source_tf, start=start, size=size).eval()
    print("source:")
    print(source)
    print("naive:")
    print(naive)
    print("real:")
    print(real)
    numpy.testing.assert_almost_equal(naive, real)


def test_CustomGradient_register_new_graph_generic_loss_and_error_signal():
    def check():
        with tf.Graph().as_default() as graph:
            with tf_compat.v1.Session(graph=graph) as session:
                custom_gradient.register_generic_loss_and_error_signal()
                x = tf.constant(2.0)
                session.run(x)  # do some early call, before `generic_loss_and_error_signal` below
                y = custom_gradient.generic_loss_and_error_signal(loss=1.0, x=x, grad_x=3.0)
                assert y.graph is graph
                (grad_y,) = tf.gradients(y, x)
                assert_equal(session.run([y, x, grad_y]), [1.0, 2.0, 3.0])

    check()
    check()
    check()


def test_CustomGradient_generic_loss_and_error_signal_post_func():
    with tf.Graph().as_default() as graph:
        with tf_compat.v1.Session(graph=graph) as session:
            custom_gradient.register_generic_loss_and_error_signal()
            x = tf.constant(5.0)
            y = custom_gradient.generic_loss_and_error_signal(loss=2.0, x=x, grad_x=3.0)
            z = 2.0 * y
            assert y.graph is graph
            (grad_z,) = tf.gradients(z, x)
            assert_equal(session.run([z, x, grad_z]), [4.0, 5.0, 6.0])


def test_global_tensor():
    class C:
        i = 0

    def f():
        C.i += 1
        return tf.constant(42, name="hello")

    x = global_tensor(f, name="hello")
    x2 = global_tensor(f, name="hello")
    x3 = global_tensor(f, name="hello")
    assert_equal(C.i, 1)
    assert_is(x, x2)
    assert_is(x, x3)
    assert_equal(x.eval(), 42)


def test_encode_raw_direct():
    raw = tf_compat.v1.decode_raw(tf.constant("ABC"), tf.uint8)
    assert_equal(list(raw.eval()), [65, 66, 67])


def test_encode_raw_simple():
    raw = tf_compat.v1.decode_raw(tf.constant("hello"), tf.uint8)
    back = encode_raw(raw)
    assert_equal(back.eval(), b"hello")


def test_encode_raw_seq_lens():
    strs = ["hello", "world", "a    "]  # all same lengths for tf.compat.v1.decode_raw
    strs_stripped = [s.strip() for s in strs]
    raw = tf_compat.v1.decode_raw(tf.constant(strs), tf.uint8)
    seq_lens = tf.constant([len(s) for s in strs_stripped])
    back = encode_raw(raw, seq_lens=seq_lens)
    assert_equal(list(back.eval()), [s.encode("utf8") for s in strs_stripped])


@unittest.skip("broken? https://github.com/tensorflow/tensorflow/issues/11240")
def test_sequential_control_dependencies():
    v = tf.Variable(initial_value=2, trainable=False, name="test_sequential_control_dependencies")
    with sequential_control_dependencies(
        [lambda: v.initializer, lambda: tf_compat.v1.assign(v, 3), lambda: tf_compat.v1.assign(v, v.read_value() + 5)]
    ):
        x = v.read_value()
    assert_equal(x.eval(), 3 + 5)


@unittest.skip("broken? https://github.com/tensorflow/tensorflow/issues/11240")
def test_var_init():
    # upstream comment: use resource variables instead
    v = tf.Variable(initial_value=2, trainable=False, name="test_var_init")
    with tf.control_dependencies([v.initializer]):
        x = v.read_value()
    assert_equal(x.eval(), 2)


def test_resource_var_init():
    # https://github.com/tensorflow/tensorflow/issues/11240
    # Will use :class:`ResourceVariable`.
    v = tf_compat.v1.get_variable(
        initializer=tf.constant_initializer(2),
        shape=(),
        trainable=False,
        name="test_resource_var_init",
        use_resource=True,
    )
    with tf.control_dependencies([v.initializer]):
        x = v.read_value()
    assert_equal(x.eval(), 2)


@unittest.skip("broken? see also test_var_init")  # TODO...
def test_true_once():
    x = true_once()
    assert_equal(x.eval(), True)
    assert_equal(x.eval(), False)
    assert_equal(x.eval(), False)
    assert_equal(x.eval(), False)


@unittest.skip("broken?")  # TODO...
def test_raise_OutOfRangeError():
    for j in range(2):
        x = raise_OutOfRangeError()
        for i in range(3):
            try:
                session.run(x)
                assert False, "should have raised OutOfRangeError"
            except tf.errors.OutOfRangeError:
                pass


def test_enforce_copy():
    v = tf.Variable(initial_value=2, trainable=False, name="test_copy")
    # with tf.control_dependencies([v.initializer]) does not work?
    session.run(v.initializer)
    a = tf.identity(v.read_value())
    b = enforce_copy(v.read_value())
    with tf.control_dependencies([a, b]):
        with tf.control_dependencies([tf_compat.v1.assign(v, 3)]):
            # `a` is a ref to v, thus also 3 now.
            # `b` is a copy, thus 2, as initially.
            x = tf.add(0, [a, b, v.read_value()])
    x_eval = list(x.eval())
    assert len(x_eval) == 3
    assert_equal(x_eval[1:], [2, 3])
    # x[0] might depend on the implementation, and TF version.
    # In TF 1, it is 3. In TF 2, it is 2. (2 is actually probably more correct...)
    assert x_eval[0] in [2, 3]


@unittest.skip("does not work")
def test_TensorArray():
    # see https://stackoverflow.com/questions/44418036/
    # Reason is that the TensorArray uses a per-run ("per-step") resource manager,
    # thus it will not remember anything across session.run() calls.
    # This is by design.
    # Our :class:`GlobalTensorArrayOpMaker` could fix this.
    ta = tf.TensorArray(tf.int32, size=3)
    index = tf_compat.v1.placeholder(tf.int32)
    value = tf_compat.v1.placeholder(tf.int32)
    flow = tf_compat.v1.placeholder(tf.float32)
    ta_new = tf.TensorArray(dtype=ta.dtype, handle=ta.handle, flow=flow)
    write = ta_new.write(index, value).flow
    read = ta_new.read(index)
    f = 0
    f = session.run(write, feed_dict={index: 0, value: 1, flow: f})
    f = session.run(write, feed_dict={index: 1, value: 2, flow: f})
    assert_equal(session.run(read, feed_dict={index: 0, flow: f}), 1)
    assert_equal(session.run(read, feed_dict={index: 1, flow: f}), 2)


def test_tfconv1d_evensize():
    filters = tf.constant([[[2.0]], [[3.0]]])  # [filter_width, in_channels, out_channels]
    assert isinstance(filters, tf.Tensor)
    assert_equal(filters.get_shape().as_list(), [2, 1, 1])
    value = tf.constant([[[5.0], [7.0]]])  # (batch, time, dim)
    assert isinstance(value, tf.Tensor)
    assert_equal(value.get_shape().as_list(), [1, 2, 1])
    res = tf.nn.conv1d(value, filters=filters, stride=1, padding="SAME", data_format="NHWC")
    resv = res.eval()
    assert isinstance(resv, numpy.ndarray)
    assert_equal(resv.shape, (1, 2, 1))  # (batch, time, dim)
    # Tests that the kernel-size of 2 is applied on current-frame + right-frame.
    # Note that in the Dataset with context_window = 2, it will do the corresponding thing,
    # i.e. adds one right-frame and no left-frame, such that if you use padding="VALID",
    # it will match the right frames.
    assert_almost_equal(resv, [[[2 * 5.0 + 3 * 7.0], [2 * 7.0]]])


def test_tf_tile():
    batch_size = 3
    beam_size = 5
    v = tf.constant([1, 2, 3])  # (batch,)
    v.set_shape((batch_size,))
    v2 = tf.tile(v, [beam_size])  # (beam*batch,)
    v2.set_shape((beam_size * batch_size,))
    print(v2.eval())
    assert_equal(list(v2.eval()), [1, 2, 3] * 5)
    v3 = tf.reshape(v2, [beam_size, batch_size])  # (beam,batch)
    r = v3.eval()
    print(r)
    assert isinstance(r, numpy.ndarray)
    for beam in range(beam_size):
        assert_equal(list(r[beam]), [1, 2, 3])


def test_tile_transposed():
    batch_size = 3
    beam_size = 5
    v = tf.constant([1, 2, 3])  # (batch,)
    v.set_shape((batch_size,))
    v2 = tile_transposed(v, axis=0, multiples=beam_size)  # (batch*beam,)
    v2.set_shape((batch_size * beam_size,))
    print(v2.eval())
    assert_equal(list(v2.eval()), [1] * 5 + [2] * 5 + [3] * 5)
    v3 = tf.reshape(v2, [batch_size, beam_size])  # (batch,beam)
    r = v3.eval()
    print(r)
    assert isinstance(r, numpy.ndarray)
    for beam in range(beam_size):
        assert_equal(list(r[:, beam]), [1, 2, 3])


def test_expand_dims_unbroadcast_instead_of_tf_tile():
    batch_size = 3
    beam_size = 5
    v = tf.constant([1, 2, 3])  # (batch,)
    v.set_shape((batch_size,))
    v2 = expand_dims_unbroadcast(v, axis=1, dim=beam_size)  # (batch,beam)
    v2.set_shape((batch_size, beam_size))
    r = v2.eval()
    print(r)
    assert isinstance(r, numpy.ndarray)
    for beam in range(beam_size):
        assert_equal(list(r[:, beam]), [1, 2, 3])


def test_expand_dims_unbroadcast_negative_axis():
    batch_size = 3
    n_time = 5
    n_dim = 2
    expand_dim = 6
    v = tf.ones((batch_size, n_time, n_dim))  # (batch, time, dim)
    v2 = expand_dims_unbroadcast(v, axis=-2, dim=expand_dim)  # (batch, time, 6, dim)
    r = v2.eval()
    print(r)
    assert isinstance(r, numpy.ndarray)
    assert_equal(r.shape, (batch_size, n_time, expand_dim, n_dim))  # (batch, time, dim)


def test_where_nan():
    # via: https://stackoverflow.com/a/42497444/133374
    # @ops.RegisterGradient("Select")
    # def _SelectGrad(op, grad):
    #   c = op.inputs[0]
    #   x = op.inputs[1]
    #   zeros = array_ops.zeros_like(x)
    #   return (None, array_ops.where(c, grad, zeros),
    #           array_ops.where(c, zeros, grad))
    # SelectOp, https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/cwise_op_select.cc
    # We later check for nan. assert_equal does not work as-is because (nan == nan) is False.
    # Thus, we resort to this check:
    assert_equal(str(float("nan")), "nan")

    where_0_nan = tf.where(True, 0.0, float("nan"))
    print("where_0_nan:", where_0_nan.eval())
    assert_equal(where_0_nan.eval(), 0.0)

    x = tf.constant(0.0)
    x_equal_0 = tf.equal(x, 0.0)
    f = tf.where(x_equal_0, 0.0, 1.0 / x)
    grad_x = tf.gradients(f, x)[0]
    print("grad_x:", grad_x.eval())  # nan? or 0?
    # This is expected when you look at the resulting computation graph for the gradient.
    # You will have grad(1./x, x) * 0.0 in the graph in the back-propagation of the gradient, which is nan.
    assert_equal(str(grad_x.eval()), "nan")

    safe_x = tf.where(x_equal_0, 2.0, x)
    grad_safe_x = tf.where(x_equal_0, 0.0, 1.0 / safe_x)
    print("grad_safe_x:", grad_safe_x.eval())  # nan? ln(2)? 0?
    # This works, because at no time, there is nan in the back-propagation.
    assert_equal(grad_safe_x.eval(), 0.0)

    f = tf.cond(x_equal_0, lambda: 0.0, lambda: 1.0 / x)
    grad_cond_x = tf.gradients(f, x)[0]
    print("grad_cond_x:", grad_cond_x.eval())  # nan? or 0?
    # This is different than tf.where because really only one branch will go into the gradient.
    assert_equal(grad_cond_x.eval(), 0.0)


def test_variable_summaries():
    v = tf.Variable(initial_value=[[1.0, 2.0], [-4.0, -1.0]], name="test_variable_summaries")
    variable_summaries(v)
    variable_summaries(tf.square(v))
    session.run(v.initializer)
    session.run(tf_compat.v1.summary.merge_all())
    assert_almost_equal(session.run(variable_scalar_summaries_dict(v)["test_variable_summaries_mean"]), -0.5)


def test_get_variable_from_tensor():
    var = tf.Variable(initial_value=[[1.0, 2.0], [-4.0, -1.0]], name="test_get_variable_from_tensor")
    x = tf.identity(var)
    print_graph_output(x)
    var_ = get_variable_from_tensor(x)
    assert var_ is var


def test_VariableAssigner():
    v = tf.Variable(initial_value=1.0)
    session.run(v.initializer)
    assert_equal(session.run(v), 1.0)
    assigner = VariableAssigner(v)
    assigner.assign(value=2.0, session=session)
    assert_equal(session.run(v), 2.0)


def test_VariableAssigner_ResourceVariable():
    v = tf_compat.v1.get_variable(
        initializer=tf.constant_initializer(1.0),
        shape=(),
        name="test_VariableAssigner_ResourceVariable",
        use_resource=True,
    )
    session.run(v.initializer)
    assert_equal(session.run(v), 1.0)
    assigner = VariableAssigner(v)
    assigner.assign(value=2.0, session=session)
    assert_equal(session.run(v), 2.0)


def test_map_labels():
    x = tf.constant([0, 1, 2, 3, 2, 1, 0])
    label_map = {0: 1, 1: 2, 2: 3, 3: 0}
    y = map_labels(x, label_map=label_map)
    assert_equal(session.run(y).tolist(), [1, 2, 3, 0, 3, 2, 1])


def test_map_labels_SparseTensor():
    x = tf.SparseTensor(
        indices=tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.int64, name="x_indices"),
        values=tf.constant([0, 1, 2, 3], name="x_values"),
        dense_shape=tf.constant([3, 3], dtype=tf.int64, name="x_dense_shape"),
    )
    label_map = {0: 1, 1: 2, 2: 3, 3: 0}
    y = map_labels(x, label_map=label_map)
    assert isinstance(y, tf.SparseTensor)
    y_eval = session.run(y)
    assert isinstance(y_eval, tf_compat.v1.SparseTensorValue)
    assert_equal(y_eval.values.tolist(), [1, 2, 3, 0])


def test_sparse_labels():
    x = tf.constant([[0, 1, 2, 3], [4, 5, 0, 0]], name="x")
    seq_lens = tf.constant([4, 2], name="seq_lens")
    y = sparse_labels(x, seq_lens=seq_lens)
    y_eval = session.run(y)
    assert isinstance(y_eval, tf_compat.v1.SparseTensorValue)
    assert isinstance(y_eval.indices, numpy.ndarray)
    assert isinstance(y_eval.values, numpy.ndarray)
    assert isinstance(y_eval.dense_shape, numpy.ndarray)
    assert_equal(y_eval.indices.tolist(), [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1]])
    assert_equal(y_eval.values.tolist(), [0, 1, 2, 3, 4, 5])
    assert_equal(y_eval.dense_shape.tolist(), [2, 4])


def test_remove_labels():
    x = tf.SparseTensor(
        indices=tf.constant([[0, 0], [0, 1], [0, 2], [1, 0]], dtype=tf.int64, name="x_indices"),
        values=tf.constant([0, 1, 2, 3], name="x_values"),
        dense_shape=tf.constant([3, 3], dtype=tf.int64, name="x_dense_shape"),
    )
    labels = {1}
    y = remove_labels(x, labels=labels)
    assert isinstance(y, tf.SparseTensor)
    y_eval = session.run(y)
    assert isinstance(y_eval, tf_compat.v1.SparseTensorValue)
    assert isinstance(y_eval.indices, numpy.ndarray)
    assert isinstance(y_eval.values, numpy.ndarray)
    assert isinstance(y_eval.dense_shape, numpy.ndarray)
    assert_equal(y_eval.indices.tolist(), [[0, 0], [0, 1], [1, 0]])
    assert_equal(y_eval.values.tolist(), [0, 2, 3])
    assert_equal(y_eval.dense_shape.tolist(), [3, 2])


def test_ctc_greedy_decode():
    logits = tf.constant(
        [
            [[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [2.0, 3.0, 1.0], [3.0, 0.0, 0.0]],
            [[-1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [2.0, 1.0, 1.0]],
            [[2.0, 3.0, 4.0], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [3.0, 3.0, 3.0]],
        ],
        name="logits",
    )
    seq_lens = tf.constant([4, 4, 2], name="seq_lens")
    expected_labels = [[1, 0], [1, 1, 0], [0]]
    y1 = ctc_greedy_decode(logits=logits, seq_lens=seq_lens, time_major=False)
    (y2,), _ = tf.nn.ctc_greedy_decoder(inputs=tf.transpose(logits, [1, 0, 2]), sequence_length=seq_lens)
    assert isinstance(y1, tf.SparseTensor)
    assert isinstance(y2, tf.SparseTensor)
    z = tf_compat.v1.sparse_to_dense(
        sparse_indices=y1.indices, sparse_values=y1.values, output_shape=y1.dense_shape, default_value=-1
    )
    z_eval = session.run(z)
    assert isinstance(z_eval, numpy.ndarray)
    assert z_eval.shape == (3, 3)
    for i in range(3):
        assert list(z_eval[i, : len(expected_labels[i])]) == expected_labels[i]
        assert all([x == -1 for x in z_eval[i, len(expected_labels[i]) :]])
    y1_eval = session.run(y1)
    y2_eval = session.run(y2)
    assert isinstance(y1_eval, tf_compat.v1.SparseTensorValue)
    assert isinstance(y1_eval.indices, numpy.ndarray)
    assert isinstance(y1_eval.values, numpy.ndarray)
    assert isinstance(y1_eval.dense_shape, numpy.ndarray)
    print("y indices:", y1_eval.indices.tolist())
    print("y values:", y1_eval.values.tolist())
    assert_equal(y2_eval.indices.tolist(), y1_eval.indices.tolist())
    assert_equal(y2_eval.values.tolist(), y1_eval.values.tolist())
    assert_equal(y2_eval.dense_shape.tolist(), y1_eval.dense_shape.tolist())


def test_supported_devices_for_op():
    op_name = "MatMul"
    devs = supported_devices_for_op(op_name)
    print("Supported devs for op %r: %r" % (op_name, devs))
    assert "CPU" in devs


def test_bleu_score():
    hyp = [1, 2, 3]
    truth = [2, 3]
    from returnn.util.basic import compute_bleu

    res = compute_bleu([truth], [hyp])
    print("res:", res)
    tf_res = session.run(
        bleu_score(hypothesis=[hyp], hyp_seq_lens=[len(hyp)], truth=[truth], truth_seq_lens=[len(truth)])
    )
    print("TF res:", tf_res)
    assert isinstance(tf_res, numpy.ndarray)
    assert tf_res.shape == (1,)
    assert_almost_equal(tf_res, [res])
    assert_almost_equal(tf_res, [0.6389431])


def test_bleu_score_empty():
    hyp = []
    truth = [2, 3]
    from returnn.util.basic import compute_bleu

    res = compute_bleu([truth], [hyp])
    print("res:", res)
    tf_res = session.run(
        bleu_score(hypothesis=[hyp], hyp_seq_lens=[len(hyp)], truth=[truth], truth_seq_lens=[len(truth)])
    )
    print("TF res:", tf_res)
    assert isinstance(tf_res, numpy.ndarray)
    assert tf_res.shape == (1,)
    assert_almost_equal(tf_res, [res])
    assert_almost_equal(tf_res, [0.0])


def test_safe_log_softmax():
    x = tf.constant(0.5, shape=(3, 7))
    x = tf.nn.softmax(x)
    print("x (softmax) op_def:")
    pprint(x.op.op_def)
    assert x.op.type == "Softmax"
    x = safe_log(x)
    print("x (safe_log(softmax)) op_def")
    pprint(x.op.op_def)
    print("x.op.inputs[0] op_def")
    pprint(x.op.inputs[0].op.op_def)
    print("graph:")
    print_graph_output(x)
    assert x.op.type == "LogSoftmax"


def test_safe_log_with_identity_check_numerics():
    x = tf.constant(0.5, shape=(3, 7))
    x = tf.nn.softmax(x)
    print("x (softmax) op_def:")
    pprint(x.op.op_def)
    assert x.op.type == "Softmax"
    x = identity_with_check_numerics(x)
    print("x (identity_with_check_numerics) op_def:")
    pprint(x.op.op_def)
    x_ = x
    assert x.op.type == "Identity"
    x = safe_log(x)
    print("x (safe_log(softmax)) op_def")
    pprint(x.op.op_def)
    print("x.op.inputs[0] op_def")
    pprint(x.op.inputs[0].op.op_def)
    print("graph:")
    print_graph_output(x)
    assert x.op.type == x_.op.type == "Identity"
    x = x.op.inputs[0]
    assert x.op.type == "LogSoftmax"


def test_safe_log_with_softmax_move_axis():
    x = tf.constant(0.5, shape=(3, 7))
    x = tf.nn.softmax(x)
    print("x (softmax) op_def:")
    pprint(x.op.op_def)
    assert x.op.type == "Softmax"
    x = move_axis(x, 1, 0)
    print("x (move_axis) op_def:")
    pprint(x.op.op_def)
    x_ = x
    assert x.op.type == "Transpose"
    x = safe_log(x)
    print("x (safe_log(softmax)) op_def")
    pprint(x.op.op_def)
    print("x.op.inputs[0] op_def")
    pprint(x.op.inputs[0].op.op_def)
    print("graph:")
    print_graph_output(x)
    assert x.op.type == x_.op.type == "Transpose"
    x = x.op.inputs[0]
    assert x.op.type == "LogSoftmax"


def _get_relevant_ops(xs, op_types):
    """
    :param list[tf.Tensor|tf.Operation] xs:
    :param list[str] op_types:
    :return: list of matching ops
    :rtype: list[tf.Operation]
    """
    from returnn.extern import graph_editor

    return [x for x in graph_editor.get_backward_walk_ops(xs, inclusive=True) if x.type in op_types]


def test_safe_log_with_softmax3d_move_axis():
    x = tf.constant(0.5, shape=(3, 7, 5))
    x = tf.nn.softmax(x)
    print("x (softmax):")
    print_graph_output(x)
    x_ops = _get_relevant_ops([x], ["Softmax", "LogSoftmax"])
    assert len(x_ops) == 1 and x_ops[0].type == "Softmax"
    x = move_axis(x, 2, 1)
    print("x (move_axis):")
    print_graph_output(x)
    x_ops = _get_relevant_ops([x], ["Softmax", "LogSoftmax"])
    assert len(x_ops) == 1 and x_ops[0].type == "Softmax"
    x = safe_log(x)
    print("x (safe_log(softmax)):")
    print_graph_output(x)
    x_ops = _get_relevant_ops([x], ["Softmax", "LogSoftmax"])
    assert len(x_ops) == 1 and x_ops[0].type == "LogSoftmax"


def test_simplify_sub():
    x = tf_compat.v1.placeholder(tf.int32, shape=[3], name="x")
    a = x + 3
    b = tf.constant(3)
    a_b = a - b
    print("Normal:")
    print_graph_output(a_b)
    a_b_ = simplify_sub(a, b)
    print("Simplified:")
    print_graph_output(a_b_)
    assert a_b_ is x


def test_simplify_sub2():
    x = tf_compat.v1.placeholder(tf.int32, shape=[3], name="x")
    a = x
    b = 0
    a_b = a - b
    print("Normal:")
    print_graph_output(a_b)
    a_b_ = simplify_sub(a, b)
    print("Simplified:")
    print_graph_output(a_b_)
    assert a_b_ is x


def test_clip_by_value_with_identity_grad():
    err_y = 42.0
    limit = 1.0
    limits = -limit, limit
    with tf.name_scope("test_safe_log_and_grad"):
        x_t = tf_compat.v1.placeholder(tf.float32, shape=(), name="x")
        y_t = clip_by_value_with_identity_grad(x_t, *limits)
        (err_x_t,) = tf.gradients(ys=y_t, xs=x_t, grad_ys=tf.constant(err_y))
        (err2_x_t,) = tf.gradients(ys=tf.clip_by_value(x_t, *limits), xs=x_t, grad_ys=tf.constant(err_y))

    for x in [0.0, -0.5, 0.5, -1.0, 1.0, -2.0, 2.0]:
        x = numpy.array(x, dtype="float32")
        y, err_x, err2_x = session.run([y_t, err_x_t, err2_x_t], feed_dict={x_t: x})
        print("x:", x, "y:", y, "err_x:", err_x, "err2_x:", err2_x)
        assert_equal(err_x, err_y)
        assert -limit <= y <= limit
        if abs(x) > limit:
            assert_equal(err2_x, 0.0)
        if abs(x) < limit:
            assert_equal(err2_x, err_y)


def test_safe_log_and_grad():
    with tf.name_scope("test_safe_log_and_grad"):
        x_t = tf_compat.v1.placeholder(tf.float32, shape=(), name="x")
        y_t = safe_log(x_t)
        (err_x_t,) = tf.gradients(ys=y_t, xs=x_t)
        check_numerics_op = add_check_numerics_ops([y_t, err_x_t])
        # For comparison:
        y2_t = tf_compat.v1.log(x_t)
        (err2_x_t,) = tf.gradients(ys=y2_t, xs=x_t)

    for x in [0.0, 100, 1e30, 1e-30]:
        x = numpy.array(x, dtype="float32")
        print("x:", x)
        assert numpy.isfinite(x).all()
        y, err_x = session.run([y_t, err_x_t], feed_dict={x_t: x})
        print("y:", y, "err_x:", err_x)
        y2, err2_x = session.run([y2_t, err2_x_t], feed_dict={x_t: x})
        print("y2:", y2, "err2_x:", err2_x)
        if not numpy.isfinite(y).all() or not numpy.isfinite(err_x).all():
            print("Warning, some nan or inf!")
            session.run(check_numerics_op, feed_dict={x_t: x})
        assert numpy.isfinite(y).all() and numpy.isfinite(err_x).all()
        assert err_x != 0.0  # there should be some gradient


def test_safe_exp_and_grad():
    with tf.name_scope("test_safe_log_and_grad"):
        x_t = tf_compat.v1.placeholder(tf.float32, shape=(), name="x")
        y_t = safe_exp(x_t)
        (err_x_t,) = tf.gradients(ys=y_t, xs=x_t)
        check_numerics_op = add_check_numerics_ops([y_t, err_x_t])
        # For comparison:
        y2_t = tf.exp(x_t)
        (err2_x_t,) = tf.gradients(ys=y2_t, xs=x_t)

    for x in [0.0, 100, 1e30, 1e-30, -1e30, -1e-30]:
        x = numpy.array(x, dtype="float32")
        print("x:", x)
        assert numpy.isfinite(x).all()
        y, err_x = session.run([y_t, err_x_t], feed_dict={x_t: x})
        print("y:", y, "err_x:", err_x)
        y2, err2_x = session.run([y2_t, err2_x_t], feed_dict={x_t: x})
        print("y2:", y2, "err2_x:", err2_x)
        if not numpy.isfinite(y).all() or not numpy.isfinite(err_x).all():
            print("Warning, some nan or inf!")
            session.run(check_numerics_op, feed_dict={x_t: x})
        assert numpy.isfinite(y).all() and numpy.isfinite(err_x).all()
        assert err_x != 0.0  # there should be some gradient


def test_lin_exp_normed_limits_not_nan():
    with tf.name_scope("test_lin_exp_normed_limits_not_nan"):
        x_t = tf_compat.v1.placeholder(tf.float32, shape=(None,), name="x")
        y_t = lin_exp_normed(x_t)
        # Also see :class:`CrossEntropyLoss`. here score instead of loss.
        score_t = safe_log(y_t[..., -1])
        (err_x_t,) = tf.gradients(ys=score_t, xs=x_t)
        check_numerics_op = add_check_numerics_ops([score_t, y_t, err_x_t])

    for x in [(0, 0), (1, -1), (100, 100), (100, -100), (1e20, 1e-20), (1e30, -1e30), (1e30, 1e30), (-1e30, -1e30)]:
        x = numpy.array(x, dtype="float32")
        print("x:", x)
        assert numpy.isfinite(x).all()
        score, y, err_x = session.run([score_t, y_t, err_x_t], feed_dict={x_t: x})
        print("score:", score, "y:", y, "err_x:", err_x)
        if not numpy.isfinite(y).all() or not numpy.isfinite(err_x).all():
            print("Warning, some nan or inf!")
            session.run(check_numerics_op, feed_dict={x_t: x})
        assert numpy.isfinite(y).all() and numpy.isfinite(err_x).all()
        # We constructed the examples in such a way that there should always be a gradient.
        assert any(err_x != 0.0)


def test_check_base_op_type_and_replace_softmax():
    with tf.name_scope("test_check_base_op_type_and_replace_softmax"):
        z = tf.constant([1.0, 2.0])
        x = tf.nn.softmax(z)
        y = tf_compat.v1.log(x)
        print("x:", x, list(x.op.inputs), "y:", y)
        y2 = check_base_op_type_and_replace(x, "Softmax", "LogSoftmax")
        print("y2:", y2)
        assert y2 is not None
        vy1, vy2 = session.run([y, y2])
        print("eval:", vy1, vy2)
        assert_almost_equal(vy1, vy2)


def test_check_base_op_type_and_replace_sigmoid():
    with tf.name_scope("test_check_base_op_type_and_replace_sigmoid"):
        z = tf.constant([1.0, 2.0])
        x = tf.sigmoid(z)
        y = tf_compat.v1.log(x)
        print("x:", x, list(x.op.inputs), "y:", y)
        y2 = check_base_op_type_and_replace(x, "Sigmoid", "LogSigmoid")
        print("y2:", y2)
        assert y2 is not None
        vy1, vy2 = session.run([y, y2])
        print("eval:", vy1, vy2)
        assert_almost_equal(vy1, vy2)


def test_move_axis_auto_optimize_multiple():
    x0 = tf.constant(numpy.random.normal(size=(3, 4, 2, 5)).astype("float32"))
    x1 = move_axis(x0, 2, 0)
    x2 = move_axis(x1, 1, 3)
    pass  # TODO check that there is only a single transpose....


def test_string_merge():
    strings = [["sub@@", "word", "test"], ["hel@@", "lo", "wo@@", "r@@", "ld"], ["foo"]]
    seq_lens = [len(seq) for seq in strings]
    max_len = max(seq_lens)
    strings = [seq + [""] * (max_len - len(seq)) for seq in strings]

    tf_strings = tf_compat.v1.placeholder(tf.string, [None, None])
    tf_seq_lens = tf_compat.v1.placeholder(tf.int32, [None])
    tf_res = string_merge(tf_strings, tf_seq_lens)
    res = session.run(tf_res, feed_dict={tf_strings: strings, tf_seq_lens: seq_lens})
    print(res)
    assert isinstance(res, numpy.ndarray)
    assert res.shape == (len(seq_lens),)
    res = res.tolist()
    print(res)
    res = [s.decode("utf8") for s in res]
    print(res)
    assert_equal(res, ["sub@@ word test", "hel@@ lo wo@@ r@@ ld", "foo"])


def test_vocab_string_merge():
    vocab = tf.convert_to_tensor(["</s>", "sub@@", "word", "test", "hel@@", "lo", "wo@@", "r@@", "ld", "foo", "bar"])
    labels = tf.convert_to_tensor([[1, 2, 3, 0, 0, 0], [4, 5, 6, 7, 8, 0], [9, 0, 0, 0, 0, 0]])
    seq_lens = tf.convert_to_tensor([4, 6, 2])
    strings = vocab_idx_to_vocab_string(labels, vocab=vocab)
    tf_res = string_merge(strings, seq_lens=seq_lens)
    res = session.run(tf_res)
    print(res)
    assert isinstance(res, numpy.ndarray)
    res = res.tolist()
    print(res)
    res = [s.decode("utf8") for s in res]
    print(res)
    assert_equal(res, ["sub@@ word test </s>", "hel@@ lo wo@@ r@@ ld </s>", "foo </s>"])


def test_string_replace():
    strings = ["sub@@ word test", "hel@@ lo wo@@ r@@ ld", "foo"]
    tf_strings = tf_compat.v1.placeholder(tf.string, [None])
    tf_res = string_replace(tf_strings, old="@@ ", new="")
    res = session.run(tf_res, feed_dict={tf_strings: strings})
    print(res)
    assert isinstance(res, numpy.ndarray)
    assert res.shape == (len(strings),)
    res = res.tolist()
    print(res)
    res = [s.decode("utf8") for s in res]
    print(res)
    assert_equal(res, ["subword test", "hello world", "foo"])


def test_words_split_get_sparse_tensor_length():
    strings = ["subword test", "a b c d", "hello world", "foo"]
    word_lens = [len(s.split(" ")) for s in strings]
    tf_strings = tf_compat.v1.placeholder(tf.string, [None])
    tf_words = words_split(tf_strings)
    tf_dense_words = tf_compat.v1.sparse_to_dense(
        tf_words.indices, tf_words.dense_shape, tf_words.values, default_value=""
    )
    tf_num_words = get_sparse_tensor_length(tf_words)
    words, dense_words, num_words = session.run(
        [tf_words, tf_dense_words, tf_num_words], feed_dict={tf_strings: strings}
    )
    print(words)
    print(dense_words)
    print(num_words)
    assert isinstance(words, tf_compat.v1.SparseTensorValue)
    assert isinstance(dense_words, numpy.ndarray)
    assert isinstance(num_words, numpy.ndarray)
    assert dense_words.shape == (len(word_lens), max(word_lens))
    assert num_words.shape == (len(strings),)
    dense_words = dense_words.tolist()
    print(dense_words)
    assert_equal(
        dense_words,
        [
            [b"subword", b"test", b"", b""],
            [b"a", b"b", b"c", b"d"],
            [b"hello", b"world", b"", b""],
            [b"foo", b"", b"", b""],
        ],
    )
    assert_equal(num_words.tolist(), word_lens)


def test_string_words_calc_wer():
    hyps = ["hello world", "a b c", "how are you", "good"]
    refs = ["hello nice world", "a x c d", "how are we", "good"]
    tf_hyps = tf_compat.v1.placeholder(tf.string, [None])
    tf_refs = tf_compat.v1.placeholder(tf.string, [None])
    tf_wer, tf_ref_num_words = string_words_calc_wer(hyps=tf_hyps, refs=tf_refs)
    wer, ref_num_words = session.run([tf_wer, tf_ref_num_words], {tf_hyps: hyps, tf_refs: refs})
    print(wer, ref_num_words)
    assert isinstance(wer, numpy.ndarray)
    assert isinstance(ref_num_words, numpy.ndarray)
    assert_equal(wer.tolist(), [1, 2, 1, 0])
    assert_equal(ref_num_words.tolist(), [3, 4, 3, 1])


def test_kenlm():
    import returnn.tf.util.ken_lm as tf_ken_lm

    if not tf_ken_lm.kenlm_checked_out():
        raise unittest.SkipTest("KenLM not checked out")
    input_strings = ["beyond immediate concerns </s>"]
    test_lm_file = tf_ken_lm.kenlm_dir + "/lm/test.arpa"
    assert os.path.exists(test_lm_file)
    lm_tf = tf_ken_lm.ken_lm_load(filename=test_lm_file)
    input_strings_tf = tf_compat.v1.placeholder(tf.string, [None])
    output_scores_tf = tf_ken_lm.ken_lm_abs_score_strings(handle=lm_tf, strings=input_strings_tf)
    with tf_compat.v1.Session() as session:
        output_scores = session.run(output_scores_tf, feed_dict={input_strings_tf: input_strings})
    print("input strings:", input_strings)
    print("output scores:", output_scores)
    assert isinstance(output_scores, numpy.ndarray)
    assert_almost_equal(output_scores, [-9.251298])  # +log space, not +log10
    print("Score is as expected.")


def test_kenlm_bpe():
    import returnn.tf.util.ken_lm as tf_ken_lm

    if not tf_ken_lm.kenlm_checked_out():
        raise unittest.SkipTest("KenLM not checked out")
    input_strings = [
        "beyond immediate concerns </s>",
        "be@@ yond imm@@ edi@@ ate conc@@ erns </s>",
        "be@@ yond imm@@",
        "be@@ yond <unk>",
    ]
    test_lm_file = tf_ken_lm.kenlm_dir + "/lm/test.arpa"
    assert os.path.exists(test_lm_file)
    lm_tf = tf_ken_lm.ken_lm_load(filename=test_lm_file)
    input_strings_tf = tf_compat.v1.placeholder(tf.string, [None])
    output_scores_tf = tf_ken_lm.ken_lm_abs_score_bpe_strings(
        handle=lm_tf, strings=input_strings_tf, bpe_merge_symbol="@@"
    )
    with tf_compat.v1.Session() as session:
        output_scores = session.run(output_scores_tf, feed_dict={input_strings_tf: input_strings})
    print("input strings:", input_strings)
    print("output scores:", output_scores)
    assert isinstance(output_scores, numpy.ndarray)
    assert_equal(output_scores.shape, (len(input_strings),))
    assert_almost_equal(output_scores[0], -9.251298)  # example from above
    assert_equal(output_scores[0], output_scores[1])
    assert_equal(output_scores[2], output_scores[3])
    print("Scores are as expected.")


def test_openfst():
    import returnn.tf.util.open_fst as tf_open_fst

    if not tf_open_fst.openfst_checked_out():
        raise unittest.SkipTest("OpenFST not checked out")
    tf_open_fst.get_tf_mod(verbose=True)

    """
  $ fstprint --osymbols=lexicon_opt.osyms --isymbols=lexicon_opt.isyms lexicon_opt.fst
  0	1	M	<epsilon>
  0	2	m	man
  0
  1	3	a	<epsilon>
  2	4	a	<epsilon>
  3	5	r	<epsilon>
  4	6	n	<epsilon>
  5	6	s	Mars
  5	7	t	Martian
  6	0	<space>	<epsilon>
  6	0	!	<epsilon>
  6	0	,	<epsilon>
  6	0	.	<epsilon>
  6	0	?	<epsilon>
  7	2	i	<epsilon>
  """
    fst_fn = tf_open_fst.returnn_dir + "/tests/lexicon_opt.fst"
    assert os.path.exists(fst_fn)
    output_symbols = {"man": 26, "Mars": 111, "Martian": 1530}

    fst_tf = tf_open_fst.get_fst(filename=fst_fn)
    states_tf = tf_compat.v1.placeholder(tf.int32, [None])
    inputs_tf = tf_compat.v1.placeholder(tf.int32, [None])
    output_tf = tf_open_fst.fst_transition(fst_handle=fst_tf, states=states_tf, inputs=inputs_tf)

    def transitions(states, inputs):
        return session.run(output_tf, feed_dict={states_tf: states, inputs_tf: inputs})

    def transition(state, input):
        """
        :param int state:
        :param int|str input:
        :return next_state,output_label,weight
        """
        if isinstance(input, str):
            init_state = state
            out_labels = []
            out_weight = 0.0
            for c in input:
                next_state, out_label, weight = transition(state, ord(c))
                state = next_state
                if out_label > 0:  # 0 is epsilon. -1 is invalid.
                    out_labels.append(out_label)
                out_weight += weight
            print("Input (%i, %r) -> output (%i, %r, weight %f)" % (init_state, input, state, out_labels, out_weight))
            return state, out_labels, out_weight
        assert isinstance(input, int)
        next_states, out_labels, weights = transitions([state], [input])
        return next_states[0], out_labels[0], weights[0]

    assert_equal(transition(0, "Mars "), (0, [output_symbols["Mars"]], 0.0))
    assert_equal(transition(0, "Martian "), (0, [output_symbols["Martian"]], 0.0))
    assert_equal(transition(0, "Mar"), (5, [], 0.0))
    assert_equal(transition(5, "s"), (6, [output_symbols["Mars"]], 0.0))
    assert_equal(transition(0, "Unknown "), (-1, [], float("-inf")))


def test_layer_norms():
    from returnn.tf.native_op import have_blocksparse_requirements

    try:
        from tensorflow.contrib.layers import layer_norm as tf_contrib_layer_norm
    except ImportError as exc:
        raise unittest.SkipTest("%s, but just skipping..." % exc)
    rnd = numpy.random.RandomState(3)
    for ndim in [2, 3, 4]:
        dims = [3] * ndim
        x_np = rnd.rand(*dims).astype("float32")
        print("x:")
        print(x_np)
        with tf.name_scope("test_ndim_%i" % ndim):
            x = tf.constant(x_np, name="x")
            g = tf.ones([3])
            b = tf.zeros([3])
            for axis in range(ndim):
                with tf.name_scope("test_axis_%i" % axis):
                    print("ndim %i, axis %i" % (ndim, axis))
                    ln = layer_norm(x=x, gain=g, bias=b, axis=axis)
                    if not have_blocksparse_requirements():
                        print("  OpenAI cannot be used")
                        ln2 = ln
                    # OpenAI seems to be broken for these cases:
                    elif axis < ndim - 1:
                        print("  ignore OpenAI layer norm for this case")
                        ln2 = ln
                    else:
                        ln2 = openai_layer_norm(x=x, gain=g, bias=b, axis=axis)
                    if axis < ndim - 1:
                        print("  cannot use tf.contrib layer norm for this case")
                        ln3 = ln  # cannot use tf_contrib_layer_norm
                    else:
                        ln3 = tf_contrib_layer_norm(
                            x, center=False, scale=False, begin_norm_axis=axis, begin_params_axis=axis
                        )
                    ln_np, ln2_np, ln3_np = session.run((ln, ln2, ln3))
                    print("layer norm:")
                    print(ln_np)
                    assert isinstance(ln_np, numpy.ndarray)
                    assert isinstance(ln2_np, numpy.ndarray)
                    assert isinstance(ln3_np, numpy.ndarray)
                    assert x_np.shape == ln_np.shape == ln2_np.shape == ln3_np.shape
                    assert_allclose(ln_np, ln2_np, rtol=1e-4)
                    assert_allclose(ln_np, ln3_np, rtol=5e-2)
                    print("ok")


def test_transform_param_axes_split_info_to_new_shape():
    assert_equal(transform_param_axes_split_info_to_new_shape([[7], [7] * 4], [7 * 2, 7 * 8]), [[7 * 2], [7 * 2] * 4])
    assert_equal(
        transform_param_axes_split_info_to_new_shape([[3, 7], [7] * 4], [3 + 7 * 2, 7 * 8]), [[3, 7 * 2], [7 * 2] * 4]
    )
    assert_equal(
        transform_param_axes_split_info_to_new_shape([[3, 7], [7] * 4], [1 + 7 * 2, 7 * 8]), [[1, 7 * 2], [7 * 2] * 4]
    )
    assert_equal(
        transform_param_axes_split_info_to_new_shape([[7, 7], [7] * 4], [3 + 7 * 2, 7 * 8]), [[3, 7 * 2], [7 * 2] * 4]
    )
    assert_equal(
        transform_param_axes_split_info_to_new_shape([[7, 7], [7] * 4], [7 * 2 + 7 * 2, 7 * 8]),
        [[7 * 2, 7 * 2], [7 * 2] * 4],
    )
    assert_equal(transform_param_axes_split_info_to_new_shape([[7], [7] * 4], [7, 7 * 8]), [[7], [7 * 2] * 4])
    assert_equal(
        transform_param_axes_split_info_to_new_shape([[1000, 621, 1280], [1000]], (2645, 1000)),
        [[1000, 621, 1024], [1000]],
    )
    assert_equal(
        transform_param_axes_split_info_to_new_shape([[512, 128, 32], [544]], (512, 544)),
        [[512, 0, 0], [544]],
    )


def test_get_op_attrib_keys():
    x = tf.matmul(a=tf.zeros((3, 4, 5)), b=tf.zeros((3, 5, 7)))
    assert isinstance(x, tf.Tensor)
    assert isinstance(x.op, tf.Operation)
    print("x op:", x.op.type)
    assert_in(x.op.type, ["BatchMatMul", "BatchMatMulV2"])
    assert_equal(x.get_shape().as_list(), [3, 4, 7])
    attrib_keys = get_op_attrib_keys(x)
    print("matmul attrib keys:", attrib_keys)
    assert_equal(sorted(attrib_keys), ["T", "adj_x", "adj_y"])
    dtype = x.op.get_attr("T")
    assert_equal(dtype, tf.float32)


def test_get_op_input_names_MatMul():
    x = tf.matmul(a=tf.zeros((3, 4, 5)), b=tf.zeros((3, 5, 7)))
    assert isinstance(x, tf.Tensor)
    assert isinstance(x.op, tf.Operation)
    print("x op:", x.op.type)
    assert_in(x.op.type, ["BatchMatMul", "BatchMatMulV2"])
    input_names = get_op_input_names(x.op)
    print("matmul input names:", input_names)
    assert_equal(sorted(input_names), ["x", "y"])


def test_get_op_input_names_Constant():
    x = tf.constant(1)
    assert isinstance(x, tf.Tensor)
    assert isinstance(x.op, tf.Operation)
    print("x op:", x.op.type)
    assert_equal(x.op.type, "Const")
    input_names = get_op_input_names(x.op)
    print("constant input names:", input_names)
    assert_equal(sorted(input_names), [])


def test_get_op_attrib_keys__is_variable_initialized():
    with tf_compat.v1.variable_scope("test_get_op_attrib_keys__is_variable_initialized"):
        var = tf_compat.v1.get_variable("var", shape=(3,))
        check = tf_compat.v1.is_variable_initialized(var)
        print("check:", check)
        assert isinstance(check, tf.Tensor)
        print("op:", check.op)
        assert check.op.type in {"IsVariableInitialized", "VarIsInitializedOp"}
        print("attrib keys:", get_op_attrib_keys(check.op))


def test_print_graph_output():
    x = tf.matmul(a=tf.zeros((3, 4, 5)), b=tf.zeros((3, 5, 7)))
    x.set_shape((3, 4, 7))
    x = tf.reshape(x, [3, 4 * 7])
    x = x + tf.constant(3.0)
    x = safe_log(tf.nn.softmax(x))
    print_graph_output(x)


def test_get_var_ops():
    with tf_compat.v1.variable_scope("test_get_var_ops"):
        v = tf_compat.v1.get_variable("v", ())
        assert_equal(find_ops_with_tensor_input(v), [v.initializer])


def test_find_ops_with_tensor_input():
    with tf_compat.v1.variable_scope("test_find_ops_with_tensor_input"):
        x0 = tf.constant(1.0, name="x0")
        v1 = tf_compat.v1.get_variable("v1", ())
        v2 = tf_compat.v1.get_variable("v2", ())
        x1a = tf.add(x0, v1, name="x1a")
        x1b = tf.add(x1a, v2, name="x1b")
        x2a = tf.multiply(v1, v2, name="x2a")
        x2b = tf.multiply(x2a, x0, name="x2b")
        assert_equal(find_ops_with_tensor_input(x0), [x1a.op, x2b.op])
        print("v1 usages:", find_ops_with_tensor_input(v1))
        assert_equal(find_ops_with_tensor_input(v1), [v1.initializer, x1a.op, x2a.op])
        assert_equal(find_ops_with_tensor_input(v2), [v2.initializer, x1b.op, x2a.op])
        assert_equal(find_ops_with_tensor_input(v2, fetches=[x2b]), [x2a.op])


def test_get_var_update_ops():
    with tf_compat.v1.variable_scope("test_get_var_update_ops"):
        v = tf_compat.v1.get_variable("v", ())
        loss = (v - 1.0) ** 2
        opt = tf_compat.v1.train.AdamOptimizer()
        minimize_op = opt.minimize(loss=loss, var_list=[v])
        assert isinstance(minimize_op, tf.Operation)
        print("find_ops_with_tensor_input:", find_ops_with_tensor_input(v, fetches=minimize_op))
        print("get_var_update_ops:", get_var_update_ops(v, fetches=minimize_op))
        update_ops = get_var_update_ops(v, fetches=minimize_op)
        assert len(update_ops) == 1
        assert update_ops[0].type in {"ApplyAdam", "ResourceApplyAdam"}


def test_get_var_update_ops__get_variable_value_copy_before_update_ops():
    with tf_compat.v1.variable_scope("test_get_var_update_ops__get_variable_value_copy_before_update_ops"):
        v = tf_compat.v1.get_variable("v", (), initializer=tf.zeros_initializer())
        assert isinstance(v, tf.Variable)
        loss = (v - 1.0) ** 2
        assert isinstance(loss, tf.Tensor)
        opt = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=1.0)
        minimize_op = opt.minimize(loss=loss, var_list=[v])
        assert isinstance(minimize_op, tf.Operation)
        print("find_ops_with_tensor_input:", find_ops_with_tensor_input(v, fetches=minimize_op))
        print("get_var_update_ops:", get_var_update_ops(v, fetches=minimize_op))
        update_ops = get_var_update_ops(v, fetches=minimize_op)
        assert len(update_ops) == 1
        assert update_ops[0].type in {"ApplyGradientDescent", "ResourceApplyGradientDescent"}
        with tf.control_dependencies(update_ops):
            # v.value() is the last snapshot (no new op), i.e. it points to the actual memory.
            # To make sure we get the value before the update (0), we must do a copy at the right point.
            v_val = get_variable_value_copy_before_update_ops(v, update_ops)
            # v.read_value() is a new read op to the current value.
            # Anyway, make sure that we have the same everywhere below.
            # This should be the value after the update, and the grad is -2, lr 1, thus should be 2.
            v_read_val = tf.identity(v.read_value())
            res = [
                py_print(0, ["loss:", loss]),
                tf.Assert(tf.equal(loss, 1.0), ["loss ", loss, " == 1"]),
                py_print(0, ["v:", v]),
                py_print(0, ["v.value:", v_val]),
                tf.Assert(tf.equal(v_val, 0.0), ["v.value ", v_val, " == 0"]),  # last snapshot
                py_print(0, ["v.read_value:", v_read_val]),
                tf.Assert(tf.equal(v_read_val, 2.0), ["v.read_value ", v_read_val, " == 2"]),  # after update
            ]
        session.run(v.initializer)
        session.run([loss, minimize_op, res])


def test_get_variable_grad_from_update_ops():
    with tf_compat.v1.variable_scope("test_get_variable_grad_from_update_ops"):
        var = tf_compat.v1.get_variable("var", (), initializer=tf.zeros_initializer())
        loss = (var - 1.0) ** 2
        for opt in [
            tf_compat.v1.train.AdamOptimizer(),
            tf_compat.v1.train.GradientDescentOptimizer(learning_rate=1.0),
            tf_compat.v1.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9),
            tf_compat.v1.train.RMSPropOptimizer(learning_rate=0.1),
        ]:
            print("Optimizer:", opt)
            minimize_op = opt.minimize(loss=loss, var_list=[var])
            assert isinstance(minimize_op, tf.Operation)
            update_ops = get_var_update_ops(var, fetches=minimize_op)
            print("update ops:", update_ops)
            print("update op keys:", get_op_attrib_keys(update_ops[0]))
            print("update op inputs by name:", get_op_input_names(update_ops[0]))
            session.run(var.initializer)  # reset
            session.run(tf_compat.v1.global_variables_initializer())  # from Adam or so
            assert_equal(session.run(var), 0.0)
            grad = get_variable_grad_from_update_ops(var, update_ops)
            print("grad:", grad)
            _, grad_np = session.run([minimize_op, grad])
            assert_equal(grad_np, -2.0)


def test_get_variable_grad_from_update_ops_mix_sparse_dense():
    with tf_compat.v1.variable_scope("test_get_variable_grad_from_update_ops_mix_sparse_dense"):
        var = tf_compat.v1.get_variable("var", (3, 5), initializer=tf.ones_initializer())
        loss = tf.reduce_sum((tf.matmul(tf.nn.embedding_lookup(var, [1]) - 1.0, tf.transpose(var)) - 1.0) ** 2)
        (ref_grad,) = tf.gradients(loss, var)
        ref_grad = tf.convert_to_tensor(ref_grad)
        session.run(var.initializer)  # reset
        ref_grad_np = session.run(ref_grad)
        print("ref grad value:")
        print(ref_grad_np)
        for opt in [
            tf_compat.v1.train.AdamOptimizer(),
            tf_compat.v1.train.GradientDescentOptimizer(learning_rate=1.0),
            tf_compat.v1.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9),
            tf_compat.v1.train.RMSPropOptimizer(learning_rate=0.1),
        ]:
            print("Optimizer:", opt)
            if isinstance(opt, (tf_compat.v1.train.MomentumOptimizer, tf_compat.v1.train.RMSPropOptimizer)):
                if is_gpu_available():
                    print("Skipping because SparseApplyMomentum/SparseApplyRMSProp does not support GPU")
                    print("supported_devices_for_op:", supported_devices_for_op("SparseApplyMomentum"))
                    continue  # SparseApplyMomentum only supports CPU currently, and this might break then
            minimize_op = opt.minimize(loss=loss, var_list=[var])
            assert isinstance(minimize_op, tf.Operation)
            print("find_ops_with_tensor_input:", find_ops_with_tensor_input(var, fetches=minimize_op))
            update_ops = get_var_update_ops(var, fetches=minimize_op)
            print("update ops:", update_ops)
            print("update op keys:", get_op_attrib_keys(update_ops[0]))
            print("update op inputs by name:", get_op_input_names(update_ops[0]))
            session.run(var.initializer)  # reset
            session.run(tf_compat.v1.global_variables_initializer())  # from Adam or so
            try:
                grad = get_variable_grad_from_update_ops(var, update_ops)
            except Exception:
                print_graph_output(update_ops)
                raise
            print("grad:", grad)
            _, grad_np, grad_dense_np = session.run([minimize_op, grad, tf.convert_to_tensor(grad)])
            print("grad value:")
            print(grad_np)
            print("grad dense value:")
            print(grad_dense_np)
            assert_almost_equal(ref_grad_np, grad_dense_np)


def test_mixed_dense_sparse_grad():
    with tf_compat.v1.variable_scope("test_mixed_dense_sparse_grad"):
        var = tf_compat.v1.get_variable("var", (3, 5), initializer=tf.ones_initializer())
        session.run(var.initializer)
        loss = tf.reduce_sum(tf.nn.embedding_lookup(var, [1]) ** 2) + tf.reduce_sum(var**2)
        (grad,) = tf.gradients(loss, var)
        print("grad:", grad)
        # It is an IndexedSlices.
        # https://github.com/tensorflow/tensorflow/issues/21243
        grad_dense = tf.convert_to_tensor(grad)
        print("grad dense:", grad_dense)
        print("grad value:")
        print(session.run(grad))
        print("grad dense value:")
        print(session.run(grad_dense))
        opt = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=1.0)
        session.run(opt.minimize(loss=loss, var_list=[var]))
        var_np = session.run(var)
        print("var:")
        print(var_np)
        assert_equal(var_np[0, 0], var_np[2, 0])
        assert_not_equal(var_np[0, 0], var_np[1, 0])


def test_tensor_array_is_dynamic_size():
    ta1 = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    assert_equal(tensor_array_is_dynamic_size(ta1), True)
    ta2 = tf.TensorArray(tf.float32, size=0, dynamic_size=False)
    assert_equal(tensor_array_is_dynamic_size(ta2), False)


def test_tensor_array_like():
    ta1 = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    ta1 = tensor_array_like(ta1)
    assert_equal(tensor_array_is_dynamic_size(ta1), True)


def test_tensor_array_like_elem_shape():
    ta1 = tf.TensorArray(tf.float32, size=0, dynamic_size=True, element_shape=tf.TensorShape([None, 13]))
    ta2 = tensor_array_like(ta1)
    assert_equal(tensor_array_is_dynamic_size(ta2), True)
    assert_equal(tensor_array_element_shape(ta1).as_list(), [None, 13])
    assert_equal(tensor_array_element_shape(ta2).as_list(), [None, 13])


def test_copy_with_new_split_axes():
    old_values = numpy.arange((3 + 5) * 5 * 4).reshape((3 + 5), 5 * 4)
    new_values = copy_with_new_split_axes([[3, 5], [5] * 4], [[5, 7], [7] * 4], old_values)
    for p in range(4):
        assert (new_values[:3, p * 7 : p * 7 + 5] == old_values[:3, p * 5 : p * 5 + 5]).all()
        assert (new_values[5 : 5 + 5, p * 7 : p * 7 + 5] == old_values[3:, p * 5 : p * 5 + 5]).all()


def test_same_context_loop():
    outer_loop_val = tf.constant(3) + tf.constant(4)
    print("outer_loop_var control flow:", outer_loop_val.op._control_flow_context)
    assert not has_control_flow_context(outer_loop_val)

    def body(t, _x):
        assert isinstance(_x, tf.Tensor)
        print("in loop x control flow:", _x.op._control_flow_context)
        assert has_control_flow_context(_x)
        v0 = _x + 1
        assert isinstance(v0, tf.Tensor)
        print("v0 control flow:", v0.op._control_flow_context)
        assert has_control_flow_context(v0)
        v1 = outer_loop_val + 1
        print("v1 control flow:", v1.op._control_flow_context)
        assert has_control_flow_context(v1)  # because +1 is done here in the loop
        with same_control_flow_ctx(outer_loop_val):
            v2 = outer_loop_val + 2
        print("v2 control flow:", v2.op._control_flow_context)
        assert not has_control_flow_context(v2)  # should be done outside now, because `same_context` usage
        return t + 1, v0 + v1 + v2

    x = tf.while_loop(cond=lambda t, _x: tf.less(t, 3), body=body, loop_vars=(0, 0))
    print("magic (totally arbitrary) res:", session.run(x))


def test_softmax_cross_entropy_over_size_batch1():
    energy_np = numpy.array(
        [
            [0.00060279],
            [0.00106305],
            [0.00139351],
            [0.0016565],
            [0.00179641],
            [0.00188511],
            [0.00197855],
            [0.00212687],
            [0.00229054],
            [0.00253187],
            [0.0028633],
            [0.00317292],
            [0.00333956],
            [0.00321618],
            [0.00301804],
            [0.00286921],
            [0.00269102],
            [0.00233986],
            [0.00198704],
            [0.00179809],
            [0.00180432],
            [0.00180019],
            [0.00168032],
            [0.00148445],
            [-0.00021808],
            [0.00024213],
            [0.00057256],
            [0.00083552],
            [0.00097541],
            [0.00106409],
            [0.00115748],
            [0.00130573],
            [0.00146933],
            [0.00171059],
            [0.00204194],
            [0.0023515],
            [0.00251812],
            [0.00239481],
            [0.00219674],
            [0.00204793],
            [0.00186977],
            [0.00151865],
            [0.00116586],
            [0.00097693],
            [0.00098313],
            [0.000979],
            [0.00085914],
            [0.00066328],
        ],
        dtype="float32",
    )
    energy_sizes = [2, 24]
    ref_att_weights_np = numpy.array(
        [
            [8.85698071e-04],
            [3.03550856e-03],
            [4.28047322e-04],
            [2.12707062e-04],
            [1.69979918e-04],
            [1.69104227e-04],
            [1.76708025e-04],
            [2.30993493e-04],
            [1.30674185e-03],
            [1.69335492e-02],
            [5.89773171e-02],
            [1.02726415e-01],
            [1.38920724e-01],
            [3.36773008e-01],
            [3.02626193e-01],
            [3.60515639e-02],
            [1.24421655e-04],
            [1.09412758e-06],
            [7.09717142e-07],
            [7.12369911e-07],
            [1.48852378e-05],
            [1.92153064e-04],
            [1.94299287e-06],
            [3.98656666e-05],
            [1.83324970e-03],
            [9.68748587e-04],
            [7.93990912e-05],
            [3.37559104e-05],
            [2.54511542e-05],
            [2.01851981e-05],
            [1.46051125e-05],
            [8.32758542e-06],
            [1.90258543e-05],
            [1.61443924e-04],
            [7.95573505e-05],
            [6.75756164e-05],
            [1.12831636e-04],
            [4.37621129e-05],
            [5.69019585e-06],
            [5.78170584e-04],
            [2.09521659e-05],
            [1.89785933e-05],
            [1.07380874e-04],
            [1.02525763e-03],
            [4.51886881e-04],
            [1.37639674e-03],
            [9.68037128e-01],
            [2.49103159e-02],
        ],
        dtype="float32",
    )
    n_batch = 1
    n_extra_dim = 1  # number of att heads
    assert energy_np.shape == ref_att_weights_np.shape == (numpy.prod(energy_sizes), n_extra_dim)
    sizes_tf = {
        i: tf.constant(numpy.array(energy_sizes[i], dtype="int32").reshape((n_batch,)))
        for i in range(len(energy_sizes))
    }
    energy_tf = tf.constant(energy_np.reshape([n_batch] + energy_sizes + [n_extra_dim]))
    ref_att_weights_tf = tf.constant(ref_att_weights_np.reshape([n_batch] + energy_sizes + [n_extra_dim]))
    energy_data = Data(
        name="energy",
        shape=(None, None, n_extra_dim),
        batch_dim_axis=0,
        placeholder=energy_tf,
        size_placeholder=sizes_tf,
    )
    ref_att_weights_data = Data(
        name="ref_att_weights",
        shape=(None, None, n_extra_dim),
        batch_dim_axis=0,
        placeholder=ref_att_weights_tf,
        size_placeholder=sizes_tf,
    )
    res_tf = softmax_cross_entropy_over_size(logits=energy_data, labels=ref_att_weights_data)
    res_tf.set_shape((n_batch, energy_sizes[0], n_extra_dim))
    res_np = session.run(res_tf)
    print("res:", res_np)
    assert numpy.alltrue(numpy.isfinite(res_np))


def test_softmax_cross_entropy_over_size_n_batch():
    energy_np = numpy.array(
        [
            [0.00060279],
            [0.00106305],
            [0.00139351],
            [0.0016565],
            [0.00179641],
            [0.00188511],
            [0.00197855],
            [0.00212687],
            [0.00229054],
            [0.00253187],
            [0.0028633],
            [0.00317292],
            [0.00333956],
            [0.00321618],
            [0.00301804],
            [0.00286921],
            [0.00269102],
            [0.00233986],
            [0.00198704],
            [0.00179809],
            [0.00180432],
            [0.00180019],
            [0.00168032],
            [0.00148445],
            [-0.00021808],
            [0.00024213],
            [0.00057256],
            [0.00083552],
            [0.00097541],
            [0.00106409],
            [0.00115748],
            [0.00130573],
            [0.00146933],
            [0.00171059],
            [0.00204194],
            [0.0023515],
            [0.00251812],
            [0.00239481],
            [0.00219674],
            [0.00204793],
            [0.00186977],
            [0.00151865],
            [0.00116586],
            [0.00097693],
            [0.00098313],
            [0.000979],
            [0.00085914],
            [0.00066328],
        ],
        dtype="float32",
    )
    energy_sizes = [2, 24]
    ref_att_weights_np = numpy.array(
        [
            [8.85698071e-04],
            [3.03550856e-03],
            [4.28047322e-04],
            [2.12707062e-04],
            [1.69979918e-04],
            [1.69104227e-04],
            [1.76708025e-04],
            [2.30993493e-04],
            [1.30674185e-03],
            [1.69335492e-02],
            [5.89773171e-02],
            [1.02726415e-01],
            [1.38920724e-01],
            [3.36773008e-01],
            [3.02626193e-01],
            [3.60515639e-02],
            [1.24421655e-04],
            [1.09412758e-06],
            [7.09717142e-07],
            [7.12369911e-07],
            [1.48852378e-05],
            [1.92153064e-04],
            [1.94299287e-06],
            [3.98656666e-05],
            [1.83324970e-03],
            [9.68748587e-04],
            [7.93990912e-05],
            [3.37559104e-05],
            [2.54511542e-05],
            [2.01851981e-05],
            [1.46051125e-05],
            [8.32758542e-06],
            [1.90258543e-05],
            [1.61443924e-04],
            [7.95573505e-05],
            [6.75756164e-05],
            [1.12831636e-04],
            [4.37621129e-05],
            [5.69019585e-06],
            [5.78170584e-04],
            [2.09521659e-05],
            [1.89785933e-05],
            [1.07380874e-04],
            [1.02525763e-03],
            [4.51886881e-04],
            [1.37639674e-03],
            [9.68037128e-01],
            [2.49103159e-02],
        ],
        dtype="float32",
    )
    n_extra_dim = 1  # number of att heads
    assert energy_np.shape == ref_att_weights_np.shape == (numpy.prod(energy_sizes), n_extra_dim)
    n_batch = 5
    energy_np = energy_np.reshape([1] + energy_sizes + [n_extra_dim]).repeat(n_batch, axis=0)
    ref_att_weights_np = ref_att_weights_np.reshape([1] + energy_sizes + [n_extra_dim]).repeat(n_batch, axis=0)
    sizes_tf = {i: tf.constant([energy_sizes[i]] * n_batch, dtype="int32") for i in range(len(energy_sizes))}
    energy_tf = tf.constant(energy_np)
    ref_att_weights_tf = tf.constant(ref_att_weights_np)
    energy_data = Data(
        name="energy",
        shape=(None, None, n_extra_dim),
        batch_dim_axis=0,
        placeholder=energy_tf,
        size_placeholder=sizes_tf,
    )
    ref_att_weights_data = Data(
        name="ref_att_weights",
        shape=(None, None, n_extra_dim),
        batch_dim_axis=0,
        placeholder=ref_att_weights_tf,
        size_placeholder=sizes_tf,
    )
    res_tf = softmax_cross_entropy_over_size(logits=energy_data, labels=ref_att_weights_data)
    res_tf.set_shape((n_batch, energy_sizes[0], n_extra_dim))
    res_np = session.run(res_tf)
    print("res:", res_np)
    assert numpy.alltrue(numpy.isfinite(res_np))


def test_softmax_cross_entropy_over_size_n_batch_real():
    if sys.version_info[0] <= 2:  # gzip.decompress is >=PY3
        raise unittest.SkipTest
    import gzip
    import base64

    # Via HDFDumpLayer of "output/energy", and dump_whole_batches=True.
    # ./returnn/tools/dump-dataset.py data-train/att-kl/energy.hdf --stdout_limit inf --stdout_as_bytes --endseq 0
    energy_np = numpy.frombuffer(
        gzip.decompress(
            base64.decodebytes(
                b"H4sIAK3yb1wC/33SeVSU5xUG8GEyLKIwIChrgOhAIwZEsfh994IiGQwFgqgDoS6AiCBbBKpxsNYE"
                b"WWSRfdMwEQRDAJXFAeR7X1JIDShbwqIISJESIQlERFEMICaenNNzUlv7nPP77/5xz3Oe9HtGEGWT"
                b"BJum6qA7vQ9EDT9Boew5MLMC3OcuxCeGBjg6bY56Q/ZYZuKJ+Z1+mNG4B0UFLviwcjPW+WzA/CJD"
                b"zGYEGMyOQmPVLVA63Q3NiTegZLIWeK9ko9SXc606yioPrIRDhp4wzUTDed0M0Ao8D6v05MCragM3"
                b"jwmQai9FvpcJVgkssOiZKab2r0B2QgXHN86BRkMneP8ogzL/MLiycydYvXCB3W8wsNCiDV0KRuT3"
                b"Mu82k3W6yUSlxZY4LFngHvsWcbx2R86kua7enslm3vGNYSue8uBQnhf0dOWCfE0V5GsWAdMXDWmz"
                b"22EXZwR12Wns3uvf1O9M6uAefaBAPrqiTOJLDUjH/c2kZvvQfxgo/hJKl2lg7UoHvP6nMCxXSMD4"
                b"q7mYda0IB0KvYMYXtahjx6HNz3Js4RfhNkkcHh+RYPc1G2xIscDlhtY4ZmqD2bNWuHaLId6aWvyv"
                b"Dn+f2Qp9zBl3wZAHf8P7qdlYNvE5Nn9ag5KCJnSQtGN/ai8eE93BDSPduDjZiFpbivH9vQl4Jz0S"
                b"kwKC8KuZD7HieQRu7wzFdTpeuKcb8OB7Z+F1yrWTufupL9jp8mIwVZqHrHpjdNWxw1BjDyxJCMas"
                b"xeOYoh2NPReO4Q90Fzoe1Mfd4lY4O3sGBh2OQi4XBzWtyTAcHA1RfCv4pD2EDXArIK+TlNHACgoZ"
                b"yOzPgdRng2Btr4XWnzFoAR74h6wAVLMPwQ56EIMTX/5+wxad/DXxwvF2mOJHgp+ujB2y+JCTfa5F"
                b"NlcaEC/XJs5m3+36/9dntFEGCfj+PRJZV8YtBnmwmftdAcaqIEF/AhL9lTBMYylmJitiV/U08Ndx"
                b"sHdADCbWVxlJsToJdjpHUoXDJHCZIuWrK9NQr2kyKuskhtUFr6U46ETnxRuoScwKeubhGHGK+JS0"
                b"xRoT99SNnEbOV4y7dC1btbWEaVLncaVKQtKx9Ty5Gf+UFC6spmVPXOmox2G66/Ip2nI5hjpKpdTH"
                b"IJDmD255LYF3EPvOsJzt6LnDVqyfZkcfz7BY/x27+61/sn3hS8Dh0Q6oHi2Fk72jcOqCIjpsUMCF"
                b"jDYQTSRCfOEm0DSdY/u+bmHl286xbd0se15wihEP1zAOp7JZ7qIiJCmZQHyoAFwcL7Djd99lW814"
                b"qKamgv3Jy7AlT4jPhRpIVNSx6o9qqPRAC9uXmuGdF46476kfRq49hkdoJAb1eOKTkzYoijRGWcXL"
                b"ez9VvC4XYO/OeeDZTkJ/yBRUGguw5ufluCNDD8WXlqOIU0Rz2QIkThPCrL5ELE2LyEo1GYkwzCcV"
                b"5gVkqquQZB/IIk47vMm03pec7vpy5plVB1t0uZTl+Q/Wp581JHfbpcTpwTnCiyshJ/5VTaQzTcRe"
                b"uYvA7Q4yzpMTuVsO2fpmAkn/IYe4dlSSnqJGMvcwF7y166Ct5VsYnRmDsKYFaFlQRZMYHXR/aoz8"
                b"J0ZY+WcD7HPTw4rkJegubQe3YYAjUWs4+1vbyHBwCrlrKyN2B4pJiXkx2fJy82pj58jIVBT5X/vs"
                b"/36SGU26xkrMhbA3FuEXg0DIvJ4Cf+kqARv5NaiorIWOqAoI8iuHhKEksIpRghreJU7s+RlxS54k"
                b"oV+r0jcsllNx0wr6wcWXm6vTot58TTp+hk+DOyXkVTMvbLibjrEMb+g0O5M2wvoOG8An29zB1TIa"
                b"QoSZMK+aDv29CdBAYiEz2BksZ0JZ/XoTMvfjP0jWGRXamvgmnX8kok5xb9O2oLdp+Akz+vebq2mJ"
                b"ZAUNOOpPXpWvbM0Jj8VwXieOcP73tDn7WYtN4lof9sgXD9hszhZWmyVAhBaB8Ns/QZCZEJ0j9bH/"
                b"sBAPLf0WVHycoTAwddOwqgohkxvJlIsCqZqT1H/n08deZS0h/IAeyHzy2LKUaubffYrWxdFvOk/T"
                b"RfY05dnFUquTJ+ma9RFUXbqf2v2yncYZIBWZraKNLSpUSX+EDMa2kv2J90j3ZR06kedKXTROUtO8"
                b"dGppnUvljWn0ra3RNMXfn35cvIs+7veizeJQmlQcRdMEkb+Z3uNJXc7uobbhuynzkYSWHHemapZ2"
                b"NOmoBY1YNKB1+srU5f0hcjWjjCwThxNfo3eJptlhMuV9g+Tm6dLNas40YU0Ytd8vpYZ/DaG9H7tR"
                b"B38rWnrRmK4aFFG+I0PNih2pyYDVb34Fpl0tleAHAAA="
            )
        ),
        dtype="float32",
    ).reshape((504, 1))
    # ./returnn/tools/dump-dataset.py .../ref_att_weights.hdf --stdout_limit inf --stdout_as_bytes --endseq 0
    ref_att_weights_np = numpy.frombuffer(
        gzip.decompress(
            base64.decodebytes(
                b"H4sIAG/zb1wC/73Qe1SMeRzH8VZrmEpSKUmxOhuRMjXP8/t9f7/neWYT2dCN3CW1RKEoU0MuaVal"
                b"TVKnXeNaI9lKKSnlkpVLRbJKTg0yOOySbnJbuez2h3P2OPjTH69/P+9zPhKHZSSww40mLteCtZ4/"
                b"HOCcINVHAsVTAJrYcOhxyyOkLJFzTl3Nk8B6nnMxFJQrDwv1XbuEulwT3jGqD3Q2pzKLlQJTCDJm"
                b"z8m16Oh6d/Bb3IcNY0ZhyX/7/3fIvZPs695AjJTZeGT6ANy5qREFzshB0f3lqKRUH5Wq9iM3dzvo"
                b"15SNA49vw6KMJ5jdhnBwewXrn2JOwiqPICzdj/Ikd3FnQByhrU+AXXaM/HgtSlZmVMt93PvaShoM"
                b"Se7qcML7m4HCsRk7D87CjiOz8bAFybg16BK+WXmIts0VCTMeuQpcj0J409UsZJzPIs+NRiDbOAW7"
                b"JGcD+8OMG6gr1hhQnhXij/ni3s3PKRPr0jmzY4npXVN8Ex1B72pXoKfVwWjPQ4KMJhigoq4OrNFR"
                b"gXZGLLTPiSde6jVYv+QGja2PJBbzvSmt8iAdWyPhZb6S+KWGy3bqDOS/1PvansUryf4OSzq74A/w"
                b"PeEC7UcNgIm0A9sKHfgTm4A2Po1rrQRh9hVD2c6YcMH0lpoojrx0rtbZ7lwl3uz09x2FdJqjAasd"
                b"5QVo/EFmd9U5tnfzc8QF/30iMSATuMmIv1jHVizeziaHxLEeZt8xoltDpG5RBFXUavHCyJ2s3blV"
                b"KB9i0IrccGnx6YdMgPoJ+7xcTDJTxsBG+RUSOUkuMwkx5r/U+9o2txcSJ6Pd9PzEqaQj+VsywfMJ"
                b"rJSLCdennChfNdKlmWq+QrtdyJPXCHvEwwXDvi7885I7nOM6e5y5Si31dl0gLdbPZm6PT2Jq5m1k"
                b"F/uaMzFB4KT6ZaDT3pt5Es/kECfSqIv/UjU4xYvKmZSadZxNwgm6ticONsSMhhvMAAg7awnWEUq4"
                b"Pf4gcLMcoOJBOboaqIOaZl2DlX7Poe4BS/kj1Zx98BrByjZdcLC148dW9RXE1jbC1PepPJt9g5b7"
                b"1ZISfjMUSRfCqHIzepExoHvc/Ki/YScnjnDh3p0wJdY5J8Fbo4Lu9TmwKOEdXBpTCD7Xq7F77FJE"
                b"rxSwWY7X4MUFFtqaPcm0CxroH13F2g/imfL6VgjJdQVPF4zvJQiQnD+Q/CbaQWbuAhKRn0be6mmo"
                b"fuZS2ab31ryoIIyUK9zpaDmQKf2vw87S3+He3FDIeWwNPx2PpV5OW4UlxvsF77sBQrOOq9B8Wgs/"
                b"6wUyl05ZsFu2qFiTyWekipbXzhpPrdSYfc2mtBVgsxYlk/DSB/Vuf+x2vAe/1uIUPzhMyr2/k0FT"
                b"WuRU98x8mhHnS80ri6l7LaE2ZkfJ7mArKqlcTg/EZPJhxinCq0MThVtvB/D1tuOEgKZC7uzCKXSL"
                b"xwO6zD6R03usptpxDfRTPQt+ECe230qzBhmD+pkaN7okY8MVb1CDpS17PFQXKR3uIY1ePDYNvwwb"
                b"jcpgq2t/bOcvZ3pqpyNlrYjUZafCGsU6yqwPgmn5pmRwURtpLYqQ7brfzH2ql/tqGa1KiuaMyvrR"
                b"sVoVqY6RkwPdm8jUy/uI7Gk0rV6EhBPVNcLbx+ZCtvkGYWFfHRLY5gZWoZfBrWUHDPvGE1bXOiDD"
                b"ufPwpAnZ6NcsK+Rol8CU+uaji0O74WLGaBaFTse9nV4OBts45GPKtdYkQxJxhvQkS3izYhyYWIRC"
                b"4iIR6VprQtJ7LLB0hIhGlqwkh69oeNc5GUK3JEdwUDfwhz2S+PhjiVzUXg/u+wUCiZsuI+461tB4"
                b"K4IUFnTS9qIAGLO0hX7oeT2M4JKCztKZ9YMg4aoKi4qC8H3TYBxdmIDbJBOxJq2E/SfKixk+dDRK"
                b"a9/LOtY1srq3B0rrEpucQ/Y+YidbalBeTTvabjAPol/4oZSRDchmaAioFDZ0zJB79OmkMJlVuz7/"
                b"ofcvjNYYm+AHAAA="
            )
        ),
        dtype="float32",
    ).reshape((504, 1))
    seq_sizes_np = numpy.frombuffer(
        gzip.decompress(base64.decodebytes(b"H4sIAK3yb1wC/2NiYGCQAGImIBZFopmBWAZKi0NpKSAGAN0FJ6owAAAA")), dtype="int32"
    ).reshape((6, 2))
    assert isinstance(seq_sizes_np, numpy.ndarray)
    max_seq_sizes_np = numpy.max(seq_sizes_np, axis=0)
    assert max_seq_sizes_np.shape == (seq_sizes_np.shape[1],)
    print("seq_sizes_np:")
    print(seq_sizes_np)
    n_batch = seq_sizes_np.shape[0]  # 6
    print("total n_batch:", n_batch)
    n_extra_dim = energy_np.shape[-1]
    energy_np = energy_np.reshape([n_batch] + list(max_seq_sizes_np) + [n_extra_dim])
    ref_att_weights_np = ref_att_weights_np.reshape([n_batch] + list(max_seq_sizes_np) + [n_extra_dim])
    for new_n_batch in range(1, n_batch + 1):
        for n_batch_start in range(0, n_batch - new_n_batch + 1):
            # Cut n_batch.
            n_batch_end = n_batch_start + new_n_batch
            print("Try with n_batch %i (from %i to %i)." % (new_n_batch, n_batch_start, n_batch_end))
            _seq_sizes_np = seq_sizes_np[n_batch_start:n_batch_end]
            _max_seq_sizes_np = numpy.max(_seq_sizes_np, axis=0)
            _energy_np = energy_np[n_batch_start:n_batch_end, : _max_seq_sizes_np[0], : _max_seq_sizes_np[1]]
            _ref_att_weights_np = ref_att_weights_np[
                n_batch_start:n_batch_end, : _max_seq_sizes_np[0], : _max_seq_sizes_np[1]
            ]
            seq_sizes_tf = {i: tf.constant(_seq_sizes_np[:, i]) for i in range(_seq_sizes_np.shape[1])}
            energy_tf = tf.constant(_energy_np)
            ref_att_weights_tf = tf.constant(_ref_att_weights_np)
            energy_data = Data(
                name="energy",
                shape=(None, None, n_extra_dim),
                batch_dim_axis=0,
                placeholder=energy_tf,
                size_placeholder=seq_sizes_tf,
            )
            ref_att_weights_data = Data(
                name="ref_att_weights",
                shape=(None, None, n_extra_dim),
                batch_dim_axis=0,
                placeholder=ref_att_weights_tf,
                size_placeholder=seq_sizes_tf,
            )
            res_tf = softmax_cross_entropy_over_size(logits=energy_data, labels=ref_att_weights_data)
            res_tf.set_shape((new_n_batch, _max_seq_sizes_np[0], n_extra_dim))
            res_np = session.run(res_tf)
            print("res:", res_np)
            assert numpy.alltrue(numpy.isfinite(res_np))


def test_softmax_cross_entropy_over_size_small_batch_2():
    import returnn.util.basic as util

    rnd = numpy.random.RandomState(42)
    n_batch = 2
    n_extra_dim = 1
    dec_seq_lens = [2, 2]
    enc_seq_lens = [4, 3]
    energy_np = rnd.normal(size=(n_batch, max(dec_seq_lens), max(enc_seq_lens), n_extra_dim)).astype("float32")
    ref_att_weights_np = rnd.normal(size=(n_batch, max(dec_seq_lens), max(enc_seq_lens), n_extra_dim)).astype("float32")
    for i in range(n_batch):
        ref_att_weights_np[i, : dec_seq_lens[i], : enc_seq_lens[i]] = util.softmax(
            ref_att_weights_np[i, : dec_seq_lens[i], : enc_seq_lens[i]], axis=1
        )
        ref_att_weights_np[i, dec_seq_lens[i] :] = 0
        ref_att_weights_np[i, : dec_seq_lens[i], enc_seq_lens[i] :] = 0
    sizes_tf = {0: tf.constant(dec_seq_lens), 1: tf.constant(enc_seq_lens)}
    energy_tf = tf.constant(energy_np)
    ref_att_weights_tf = tf.constant(ref_att_weights_np)
    energy_data = Data(
        name="energy",
        shape=(None, None, n_extra_dim),
        batch_dim_axis=0,
        placeholder=energy_tf,
        size_placeholder=sizes_tf,
    )
    ref_att_weights_data = Data(
        name="ref_att_weights",
        shape=(None, None, n_extra_dim),
        batch_dim_axis=0,
        placeholder=ref_att_weights_tf,
        size_placeholder=sizes_tf,
    )
    res_tf = softmax_cross_entropy_over_size(logits=energy_data, labels=ref_att_weights_data)
    res_tf.set_shape((n_batch, max(dec_seq_lens), n_extra_dim))
    res_np = session.run(res_tf)
    print("res:", res_np)
    assert numpy.alltrue(numpy.isfinite(res_np))


def test_softmax_cross_entropy_over_size_gradient():
    n_batch = 2
    n_dec_time = n_enc_time = 10
    n_extra_dim = 1
    tf_compat.v1.set_random_seed(42)
    energy_tf = tf_compat.v1.get_variable(
        "test_softmax_cross_entropy_over_size_gradient_var",
        shape=(n_batch, n_dec_time, n_enc_time, n_extra_dim),
        initializer=tf_compat.v1.random_normal_initializer(seed=23),
    )
    ref_att_weights_tf = tf.reshape(
        tf.one_hot(tf.range(n_dec_time, dtype=tf.int32), n_enc_time, dtype=tf.float32),
        (1, n_dec_time, n_enc_time, n_extra_dim),
    )
    ref_att_weights_tf = tf.tile(ref_att_weights_tf, [n_batch, 1, 1, 1])
    ref_att_weights_tf.set_shape((n_batch, n_dec_time, n_enc_time, n_extra_dim))
    sizes = {0: [n_dec_time, n_dec_time - 1], 1: [n_enc_time, n_enc_time - 1]}
    sizes_tf = {i: tf.constant(size) for (i, size) in sizes.items()}
    energy_data = Data(
        name="energy",
        shape=(None, None, n_extra_dim),
        batch_dim_axis=0,
        placeholder=energy_tf,
        size_placeholder=sizes_tf,
    )
    ref_att_weights_data = Data(
        name="ref_att_weights",
        shape=(None, None, n_extra_dim),
        batch_dim_axis=0,
        placeholder=ref_att_weights_tf,
        size_placeholder=sizes_tf,
    )
    for stable_gradient in [False, True]:
        res_tf = softmax_cross_entropy_over_size(
            logits=energy_data, labels=ref_att_weights_data, stable_gradient=stable_gradient
        )
        res_tf.set_shape((n_batch, n_dec_time, n_extra_dim))
        res_flat_tf = flatten_with_seq_len_mask(res_tf, sizes_tf[0], batch_dim_axis=0, time_dim_axis=1)
        res_flat_tf.set_shape((sum(sizes[0]), n_extra_dim))
        loss_tf = tf.reduce_mean(res_tf)
        optimizer = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=1e2)
        optim_op = optimizer.minimize(loss=loss_tf, var_list=[energy_tf])
        session.run(energy_tf.initializer)  # Note: the second time this is called, it will get a different init
        last_loss = float("inf")
        for i in range(10):
            loss, _ = session.run([loss_tf, optim_op])
            print("step %i, loss %f" % (i, loss))
            if numpy.isnan(loss):
                print("WARNING: got nan")
                print("lr:", session.run(optimizer._learning_rate_tensor))
                print("var:", session.run(energy_tf))
                raise Exception("got nan")
            assert loss < last_loss or 0.0 == loss == last_loss  # this must always improve
            last_loss = loss


def test_FetchHelper_simple():
    y = tf.sqrt(42.0)
    v = session.run(y)
    numpy.testing.assert_almost_equal(v, numpy.sqrt(42.0), decimal=5)

    another_debug_test = tf_compat.v1.Print(y.op.inputs[0], ["debug print:"] + list(y.op.inputs))
    # https://stackoverflow.com/questions/57707445/how-to-add-control-input-to-an-op-after-it-was-run-by-a-session
    # add_control_input(y.op, another_debug_test.op)
    # y.op._add_control_input(another_debug_test.op)
    from returnn.extern import graph_editor

    y = graph_editor.graph_replace(
        target_ts=y, replacement_ts={y.op.inputs[0]: another_debug_test}, reuse_dst_scope=True
    )

    fetch_helper = FetchHelper(y.op.inputs[0], verbose_stream=sys.stdout)
    y = FetchHelper.copy_graph_replace_tensors(y, [fetch_helper])
    session.run(y)
    assert fetch_helper.callback_count == 1
    numpy.testing.assert_almost_equal(fetch_helper.most_recent_value, 42.0)


def test_FetchHelper_loop():
    if hasattr(tf_compat.v1, "control_flow_v2_enabled") and tf_compat.v1.control_flow_v2_enabled():
        raise unittest.SkipTest("TensorFlow control flow v2 not supported")
    N = 3

    class Loop:
        def body(self, i, x):
            self.y = x / 2.0
            return i + 1, self.y

        def cond(self, i, x):
            return tf.less(i, N)

    loop = Loop()
    _, y = tf.while_loop(cond=loop.cond, body=loop.body, loop_vars=(0, 42.0))
    session.run(y)  # first run, to trigger https://stackoverflow.com/questions/57707445/

    from returnn.extern import graph_editor

    ops = graph_editor.get_backward_walk_ops([y.op], inclusive=True, control_inputs=True)
    _, info = graph_editor.copy(ops, reuse_dst_scope=True)
    assert isinstance(info, graph_editor.TransformerInfo)
    y = info.transformed(y)

    # Note: This only works correct with tf.compat.v1.disable_control_flow_v2() currently...
    # Maybe we need to extend graph_editor?
    print(ops)
    x = loop.y.op.inputs[0]
    print(x)
    print(info.transformed(x))
    transformed_map = info._get_transformed_map(x)
    print(transformed_map)

    fetch_helper = FetchHelper(info.transformed(loop.y.op.inputs[0]), verbose_stream=sys.stdout)
    fetch_helper.add_to_control_inputs(info.transformed(loop.y.op))

    v = session.run(y)
    numpy.testing.assert_almost_equal(v, 42.0 / (2.0**N), decimal=5)
    assert fetch_helper.callback_count == N
    numpy.testing.assert_almost_equal(fetch_helper.most_recent_value, v * 2.0)


def test_FetchHelper_loop_invalid():
    if hasattr(tf_compat.v1, "control_flow_v2_enabled") and tf_compat.v1.control_flow_v2_enabled():
        raise unittest.SkipTest("TensorFlow control flow v2 not supported")
    from returnn.tf.network import help_on_tf_exception  # not needed for the test, but helpful for further debug output

    have_gpu = is_gpu_available()
    print("Have GPU:", have_gpu)
    graph = tf.Graph()
    with graph.as_default():
        session = tf_compat.v1.Session()
        with session:
            with tf.device("/gpu:0" if have_gpu else "/cpu:0"):
                N = 3

                class Loop:
                    def body(self, i, x):
                        target_shape = tf.convert_to_tensor([i + 1, 2])
                        with tf.device("/cpu:0"):
                            target_shape = tf_compat.v1.Print(target_shape, ["target shape:", target_shape])
                        self.y = tf.reshape(x / 2.0, target_shape)
                        with tf.device("/cpu:0"):
                            y = tf_compat.v1.Print(self.y, ["i:", i, "y:", self.y, "shape:", tf.shape(self.y)])
                        return i + 1, y

                    def cond(self, i, x):
                        return tf.less(i, N)

                loop = Loop()
                _, y = tf.while_loop(
                    cond=loop.cond,
                    body=loop.body,
                    loop_vars=(0, tf.convert_to_tensor([[42.0, 42.0]])),
                    shape_invariants=(tf.TensorShape(()), tf.TensorShape((None, None))),
                )
                assert isinstance(y, tf.Tensor)

            print("Run a first time now.")
            try:
                v = session.run(y)
            except tf.errors.OpError as exc:
                print("Got TF exception (kind of expected).")
                print(exc)
                help_on_tf_exception(session=session, exception=exc, fetches=y)
                if exc.op is not loop.y.op:
                    print("Error, unexpected: %r vs %r" % (exc.op, loop.y.op))
                    raise
            else:
                assert False, "we should have gotten a TF exception, but we got: %r" % (v,)

            y, fetch_helpers, target_op_transformed = FetchHelper.copy_graph(
                y, target_op=loop.y.op, fetch_helper_tensors=list(loop.y.op.inputs), verbose_stream=sys.stdout
            )

            print("Now run a second time, but now with fetch helpers added.")
            try:
                v = session.run(y)
            except tf.errors.OpError as exc:
                print("Got TF exception (kind of expected).")
                print(exc)
                # help_on_tf_exception(session=session, exception=exc, fetches=y)  # Broken now?
                if exc.op is not target_op_transformed:
                    print("Error, unexpected: %r vs %r" % (exc.op, loop.y.op))
                    raise
            else:
                assert False, "we should have gotten a TF exception, but we got: %r" % (v,)

            print("Fetches:")
            for input_t, fetch_helper in zip(loop.y.op.inputs, fetch_helpers):
                print("  %r: %r" % (input_t, fetch_helper.most_recent_value))
                assert fetch_helper.callback_count >= 1


def test_FetchHelper_loop_invalid_vars_switch():
    if hasattr(tf_compat.v1, "control_flow_v2_enabled") and tf_compat.v1.control_flow_v2_enabled():
        raise unittest.SkipTest("TensorFlow control flow v2 not supported")
    step = tf_compat.v1.get_variable(
        "step", shape=(), dtype=tf.int64, initializer=tf.zeros_initializer(), trainable=False
    )
    v = tf_compat.v1.get_variable(
        name="var_accum_grad", shape=(), dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False
    )
    session.run(tf_compat.v1.global_variables_initializer())

    v = tf.cond(
        tf.less_equal(tf_compat.v1.mod(step, 2), 0),
        lambda: tf_compat.v1.assign(v, 2.0),
        lambda: tf_compat.v1.assign_add(v, 3.0),
    )
    v = tf.identity(v)
    print("v:", v, v.dtype, v.op._control_flow_context)

    N = 3

    class Loop:
        def body(self, i, x):
            target_shape = tf.convert_to_tensor([i + 1, 2 + tf.cast(v, tf.int32)])
            with tf.device("/cpu:0"):
                target_shape = tf_compat.v1.Print(target_shape, ["target shape:", target_shape])
            self.y = tf.reshape(x / 2.0, target_shape)
            with tf.device("/cpu:0"):
                y = tf_compat.v1.Print(self.y, ["i:", i, "y:", self.y, "shape:", tf.shape(self.y)])
            return i + 1, y

        def cond(self, i, x):
            return tf.less(i, N)

    loop = Loop()
    _, y = tf.while_loop(
        cond=loop.cond,
        body=loop.body,
        loop_vars=(0, tf.convert_to_tensor([[42.0, 42.0]])),
        shape_invariants=(tf.TensorShape(()), tf.TensorShape((None, None))),
    )
    assert isinstance(y, tf.Tensor)

    try:
        v = session.run(y)
    except tf.errors.OpError as exc:
        print("Got TF exception (kind of expected).")
        if exc.op is not loop.y.op:
            print("Error, unexpected: %r vs %r" % (exc.op, loop.y.op))
            raise
    else:
        assert False, "we should have gotten a TF exception, but we got: %r" % (v,)

    op = loop.y.op
    stop_at_ts = []
    for op_ in op.graph.get_operations():
        assert isinstance(op_, tf.Operation)
        if has_control_flow_context(op_):
            continue
        for x in list(op_.inputs) + list(op_.outputs) + list(op.control_inputs):
            assert isinstance(x, tf.Tensor)
            # noinspection PyProtectedMember
            if x.dtype._is_ref_dtype:
                print("add stop:", x)
                stop_at_ts.append(x)  # and also should not copy any variables/refs

    y, fetch_helpers, target_op_transformed = FetchHelper.copy_graph(
        y,
        target_op=loop.y.op,
        fetch_helper_tensors=list(loop.y.op.inputs),
        stop_at_ts=stop_at_ts,
        verbose_stream=sys.stdout,
    )

    print("Now run a second time, but now with fetch helpers added.")
    try:
        v = session.run(y)
    except tf.errors.OpError as exc:
        print("Got TF exception (kind of expected).")
        if exc.op is not target_op_transformed:
            print("Error, unexpected: %r vs %r" % (exc.op, loop.y.op))
            raise
    else:
        assert False, "we should have gotten a TF exception, but we got: %r" % (v,)

    print("Fetches:")
    for input_t, fetch_helper in zip(loop.y.op.inputs, fetch_helpers):
        print("  %r: %r" % (input_t, fetch_helper.most_recent_value))
        assert fetch_helper.callback_count >= 1


def test_mem_usage_for_dev_via_tf_log_memory_usage():
    if not OpCodeCompiler.cuda_available():
        raise unittest.SkipTest("CUDA not available")
    d = {}
    gpu_dev = None
    for dev in get_tf_list_local_devices():
        if dev.device_type != "GPU":
            # mem_usage_for_dev currently only works for GPU
            continue
        d[dev.name] = mem_usage_for_dev(dev.name)
        gpu_dev = dev.name
    if not d:
        print("No GPU devices, nothing to do.")
        return  # nothing to do
    res1 = session.run(d)
    print(res1)
    with tf.device(gpu_dev):
        v = tf.Variable(name="c", initial_value=tf.zeros((100, 100)))
    session.run(v.initializer)
    res = session.run(d)
    print(res)
    assert res[gpu_dev] > res1[gpu_dev]


def test_get_positional_encoding_batch_position():
    # Test `get_positional_encoding` with `position` with a batch dimension.
    num_channels = 8
    position0 = tf.range(5)
    position1 = tf.range(5, 10)
    position2 = tf.range(5)
    position = tf.stack([position0, position1, position2])  # (3, 5).
    signal = get_positional_encoding(num_channels=num_channels, position=position)  # (3, 5, 8).
    signal1 = get_positional_encoding(num_channels=num_channels, position=position1)  # (5, 8).
    signal_np, signal1_np = session.run([signal, signal1])
    assert signal_np.shape == (3, 5, 8)
    numpy.array_equal(signal1_np, signal_np[1])


def test_get_position_encoding():
    y = get_positional_encoding(num_channels=4, position=tf.range(3))
    values = session.run(y)
    numpy.testing.assert_almost_equal(
        values,
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.8414709568023682, 9.99999901978299e-05, 0.5403022766113281, 1.0],
            [0.9092974066734314, 0.0001999999803956598, -0.416146844625473, 1.0],
        ],
    )


def test_get_linear_alignment_out_to_in_indices():
    #     Examples:
    #       * input_len=7, output_len=3, resulting indices [1,3,5].
    #       * input_len=3, output_len=3, resulting indices [0,1,2].
    #       * input_len=2, output_len=4, resulting indices [0,0,1,1].
    assert_equal(
        session.run(get_linear_alignment_out_to_in_indices(input_lens=[7], output_lens=[3])).tolist(), [[1, 3, 5]]
    )
    assert_equal(
        session.run(get_linear_alignment_out_to_in_indices(input_lens=[3], output_lens=[3])).tolist(), [[0, 1, 2]]
    )
    assert_equal(
        session.run(get_linear_alignment_out_to_in_indices(input_lens=[2], output_lens=[4])).tolist(), [[0, 0, 1, 1]]
    )
    assert_equal(
        session.run(get_linear_alignment_out_to_in_indices(input_lens=[7, 3, 1], output_lens=[3, 3, 3])).tolist(),
        [[1, 3, 5], [0, 1, 2], [0, 0, 0]],
    )
    assert_equal(
        session.run(
            get_linear_alignment_out_to_in_indices(input_lens=[7, 4, 2, 1], output_lens=[3, 4, 4, 2], pad_value=-1)
        ).tolist(),
        [[1, 3, 5, -1], [0, 1, 2, 3], [0, 0, 1, 1], [0, 0, -1, -1]],
    )
    assert_equal(
        session.run(get_linear_alignment_out_to_in_indices(input_lens=[2], output_lens=[3])).tolist(), [[0, 1, 1]]
    )


def test_get_rnnt_linear_aligned_output():
    #   Examples: (B is blank.)
    #     * input_len=4, targets=[a,b,c] (len 3), output=[B,a,B,b,B,c,B] (len 7).
    #     * input_len=0, targets=[a,b,c] (len 3), output=[a,b,c] (len 3).
    #     * input_len=4, targets=[a] (len 1), output=[B,B,a,B,B] (len 5).
    #     * input_len=3, targets=[a,b] (len 2), output=[B,a,B,b,B] (len 5)
    assert_equal(
        session.run(
            get_rnnt_linear_aligned_output(input_lens=[4], targets=[[1, 2, 3]], target_lens=[3], blank_label_idx=4)[0]
        ).tolist(),
        [[4, 1, 4, 2, 4, 3, 4]],
    )
    assert_equal(
        session.run(
            get_rnnt_linear_aligned_output(input_lens=[0], targets=[[1, 2, 3]], target_lens=[3], blank_label_idx=4)[0]
        ).tolist(),
        [[1, 2, 3]],
    )
    assert_equal(
        session.run(
            get_rnnt_linear_aligned_output(input_lens=[4], targets=[[1]], target_lens=[1], blank_label_idx=4)[0]
        ).tolist(),
        [[4, 4, 1, 4, 4]],
    )
    assert_equal(
        session.run(
            get_rnnt_linear_aligned_output(input_lens=[3], targets=[[1, 2]], target_lens=[2], blank_label_idx=4)[0]
        ).tolist(),
        [[4, 1, 4, 2, 4]],
    )
    assert_equal(
        session.run(
            get_rnnt_linear_aligned_output(
                input_lens=[2], targets=tf.zeros((1, 0), dtype=tf.int32), target_lens=[0], blank_label_idx=4
            )[0]
        ).tolist(),
        [[4, 4]],
    )
    assert_equal(
        session.run(
            get_rnnt_linear_aligned_output(
                input_lens=[4, 3, 2, 0],
                targets=[[1, 2, 3], [1, 2, -1], [-1, -1, -1], [1, 2, 3]],
                target_lens=[3, 2, 0, 3],
                blank_label_idx=4,
            )[0]
        ).tolist(),
        [[4, 1, 4, 2, 4, 3, 4], [4, 1, 4, 2, 4, 0, 0], [4, 4, 0, 0, 0, 0, 0], [1, 2, 3, 0, 0, 0, 0]],
    )
    # RNA test
    assert_equal(
        session.run(
            get_rnnt_linear_aligned_output(
                input_lens=[7], targets=[[1, 2, 3]], target_lens=[3], blank_label_idx=4, targets_consume_time=True
            )[0]
        ).tolist(),
        [[4, 1, 4, 2, 4, 3, 4]],
    )
    assert_equal(
        session.run(
            get_rnnt_linear_aligned_output(
                input_lens=[3], targets=[[1, 2, 3]], target_lens=[3], blank_label_idx=4, targets_consume_time=True
            )[0]
        ).tolist(),
        [[1, 2, 3]],
    )
    assert_equal(
        session.run(
            get_rnnt_linear_aligned_output(
                input_lens=[2], targets=[[1, 2, 3]], target_lens=[3], blank_label_idx=4, targets_consume_time=True
            )[0]
        ).tolist(),
        [[1, 2]],
    )


if __name__ == "__main__":
    try:
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
    finally:
        import threading

        # if len(list(threading.enumerate())) > 1:
        #  print("Warning, more than one thread at exit:")
        #  better_exchook.dump_all_thread_tracebacks()
