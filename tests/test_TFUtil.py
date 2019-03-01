
# start test like this:  nosetests-2.7  tests/test_TFUtil.py

from __future__ import print_function


import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import sys
sys.path += ["."]  # Python 3 hack
from TFUtil import *
from nose.tools import assert_equal, assert_not_equal, assert_is_instance, assert_is, assert_in
from numpy.testing.utils import assert_almost_equal, assert_allclose
from pprint import pprint
import unittest
import numpy.testing
import better_exchook
better_exchook.replace_traceback_format_tb()


print("TF version:", tf.__version__)

session = tf.InteractiveSession()


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


def test_Data_default_time_no_time():
  # This is new behavior.
  data = Data(name='merge_dims_test_output', shape=(3, 5))
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
  data = Data(name='att_weights_output', shape=(None,), batch_dim_axis=1)
  print("data", data, "feat axis:", data.feature_dim_axis_or_unspecified, data.feature_dim_axis)
  assert_equal(data.time_dim_axis, 0)
  data2 = data.copy_as_batch_major()
  assert_equal(data2.batch_dim_axis, 0)
  assert_equal(data2.time_dim_axis, 1)
  # No check for feature_dim_axis, as this behavior does not matter here.


def test_Data_spatial_batch_axes():
  d1 = Data(name='ff_out_prior_output', shape=(1, 9001), dtype='float32', batch_dim_axis=None)
  d2 = Data(name='ff_out_output', shape=(None, 9001), dtype='float32')
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
  assert_equal(d.get_bc_shape({("B", "s:1"): None}), (None, 1, 13, 9000))


def test_Data_copy_template_adding_time_dim_no_feature():
  d1 = Data(name="d1", shape=(), time_dim_axis=None)
  assert d1.batch_dim_axis == 0 and d1.batch_shape == (None,)
  assert d1.feature_dim_axis is None
  d2 = d1.copy_template_adding_time_dim()
  assert d2.batch_dim_axis == 1 and d2.time_dim_axis == 0 and d2.batch_shape == (None, None)
  # assert d2.feature_dim_axis is None  # not sure what we would want here...


def test_Data_get_axes_from_description_except_time_ext():
  data = Data(name='merge_dims_test_output', shape=(3, None, 5), time_dim_axis=2)
  axes = data.get_axes_from_description("except_time")
  assert axes == [1, 3], "data %r 'except_time' axes %r unexpected" % (data, axes)


def test_Data_get_axes_from_description_except_time_no_time():
  data = Data(name='merge_dims_test_output', shape=(3, 5))
  assert data.time_dim_axis is None
  axes = data.get_axes_from_description("except_time")
  assert axes == [1, 2], "data %r 'except_time' axes %r unexpected" % (data, axes)


def test_Data_copy_template_excluding_time_dim_two_time_dims():
  data = Data(name='ref_att_weights_output', shape=(None, None, 1), auto_create_placeholders=True)
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
  data = Data(name='att_weights_output', shape=(1, None), time_dim_axis=2)
  print("data:", data, "feature axis:", data.feature_dim_axis)
  assert data.shape == (1, None) and data.batch_dim_axis == 0 and data.time_dim_axis == 2
  # No test for feature axis, as it does not really matter.


def test_Data_copy_with_feature_dim_axis_case_1():
  # Case 1: new_feature_dim_axis <= time_dim_axis < old_feature_dim_axis
  import numpy as np
  size_placeholder = tf.constant(np.full((10,), 10), dtype=tf.int32)
  d = Data(name='test_data', shape=(None, 13, 17), dtype='float32',
           size_placeholder={0: size_placeholder}, batch_dim_axis=0,
           time_dim_axis=1, feature_dim_axis=3)
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
  d = Data(name='test_data', shape=(None, 17, 13), dtype='float32',
           size_placeholder={0: size_placeholder}, batch_dim_axis=0,
           time_dim_axis=1, feature_dim_axis=2)
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
  d = Data(name='test_data', shape=(None, 13, 17), dtype='float32',
           size_placeholder={0: size_placeholder}, batch_dim_axis=0,
           time_dim_axis=1, feature_dim_axis=3)
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
  d = Data(name='test_data', shape=(13, 17, None), dtype='float32',
           size_placeholder={2: size_placeholder}, batch_dim_axis=0,
           time_dim_axis=3, feature_dim_axis=2)
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
  d = Data(name='test_data', shape=(17, 13, None), dtype='float32',
           size_placeholder={2: size_placeholder}, batch_dim_axis=0,
           time_dim_axis=3, feature_dim_axis=1)
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
  d = Data(name='test_data', shape=(17, None, 13), dtype='float32',
           size_placeholder={1: size_placeholder}, batch_dim_axis=0,
           time_dim_axis=2, feature_dim_axis=1)
  d_copy = d.copy_with_feature_dim_axis(-1)
  assert d_copy.shape == (None, 13, 17)
  assert d_copy.batch_dim_axis == 0
  assert d_copy.time_dim_axis == 1
  assert d_copy.feature_dim_axis == 3
  assert list(d_copy.size_placeholder.keys()) == [0]


def test_Data_copy_compatible_to_time_major():
  d1 = Data(name='ff_out_output', shape=(None, 9001), dtype='float32', batch_dim_axis=1)
  d2 = Data(name='ff_out_prior_output', shape=(9001,), dtype='float32', batch_dim_axis=None, time_dim_axis=None)
  d2a = d2.copy_compatible_to(d1)
  assert d2a.shape == (1, 9001)
  assert d2a.batch_dim_axis == d1.batch_dim_axis
  assert d2a.time_dim_axis == d1.time_dim_axis
  assert d2a.feature_dim_axis == d1.feature_dim_axis


def test_Data_sparse_int32_with_dim_kwargs_init():
  data = Data(name="classes_with_dim", shape=(None,), dim=10, sparse=True, dtype="int32")
  assert data.sparse and data.have_time_axis() and data.shape == (None,) and data.dim == 10


def test_Data_int32_no_dim_kwargs_init():
  data = Data(name="classes_with_no_dim", shape=(None,), dtype="int32")
  assert data.have_time_axis() and data.shape == (None,)


def test_Data_copy_template_excluding_spatial_dim():
  att_weights = Data(name="att_weights", shape=(None, None, 1), batch_dim_axis=2)
  rem_enc_time = att_weights.copy_template_excluding_spatial_dim(-1)
  assert rem_enc_time.shape == (None, 1) and rem_enc_time.batch_dim_axis == 1


def test_Data_copy_squeeze_axes():
  weights = Data(name='att_weights_output', shape=(1, None), time_dim_axis=2, auto_create_placeholders=True)
  squeezed = weights.copy_squeeze_axes([1])
  print("orig:", weights, "squeezed:", squeezed)
  assert squeezed.shape == (None,) and squeezed.time_dim_axis == 1
  assert weights.size_placeholder[1] is squeezed.size_placeholder[0]


def test_Data_copy_squeeze_axes_feature_axis():
  weights = Data(name='att_weights_output', shape=(None, 1), auto_create_placeholders=True)
  squeezed = weights.copy_squeeze_axes([2])
  print("orig:", weights, "squeezed:", squeezed)
  assert squeezed.shape == (None,) and squeezed.time_dim_axis == 1
  assert weights.size_placeholder[0] is squeezed.size_placeholder[0]


def test_ExternData_via_config():
  # Like ExternData.init_from_config.
  from Config import Config
  config = Config({
    "extern_data": {
      "data": (40, 2),
      "classes": (10025, 1),
      "att_weights": {"shape": (None, None, 1)},
      "att_weights_sizes": {"shape": (None,), "dtype": "int32"}
    }})
  from NetworkDescription import LayerNetworkDescription
  data_dims = LayerNetworkDescription.tf_extern_data_types_from_config(config)
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
  d = Data(name='test_data', shape=(None, 13, 17), dtype='float32',
           size_placeholder={0: size_placeholder}, batch_dim_axis=0,
           time_dim_axis=1, feature_dim_axis=3)
  placeholder = tf.placeholder(shape=(None, None, 13, 17), dtype=tf.float32)
  d.placeholder = placeholder
  feed_data = np.random.rand(7, 9, 13, 17)
  res = session.run(d.placeholder, feed_dict={placeholder: feed_data})
  print(res.shape)
  flat_placeholder = d.get_placeholder_flattened(keep_dims=True)
  res = session.run(flat_placeholder, feed_dict={placeholder: feed_data})
  print(res.shape)
  assert res.shape[0] == 7 * 9 * 13
  assert len(res.shape) == 4
  flat_placeholder = d.get_placeholder_flattened(keep_dims=False)
  res = session.run(flat_placeholder, feed_dict={placeholder: feed_data})
  print(res.shape)
  assert res.shape[0] == 7 * 9 * 13
  assert len(res.shape) == 2

def test_2D_Data_get_placeholder_flattened():
  import numpy as np
  d = Data(name='test_data', shape=(17,), dtype='float32',
           batch_dim_axis=0, feature_dim_axis=1)
  placeholder = tf.placeholder(shape=(None, 17), dtype=tf.float32)
  d.placeholder = placeholder
  feed_data = np.random.rand(7, 17)
  res = session.run(d.placeholder, feed_dict={placeholder: feed_data})
  print(res.shape)
  flat_placeholder = d.get_placeholder_flattened(keep_dims=True)
  res = session.run(flat_placeholder, feed_dict={placeholder: feed_data})
  assert res.shape == (7, 17)
  flat_placeholder = d.get_placeholder_flattened(keep_dims=False)
  res = session.run(flat_placeholder, feed_dict={placeholder: feed_data})
  assert res.shape == (7, 17)


def test_Data_copy_compatible_to_batch_major():
  d1 = Data(name='ff_out_output', shape=(None, 9001), dtype='float32')
  d2 = Data(name='ff_out_prior_output', shape=(9001,), dtype='float32', batch_dim_axis=None, time_dim_axis=None)
  d2a = d2.copy_compatible_to(d1)
  assert d2a.shape == (1, 9001)
  assert d2a.batch_dim_axis == d1.batch_dim_axis
  assert d2a.time_dim_axis == d1.time_dim_axis
  assert d2a.feature_dim_axis == d1.feature_dim_axis


def test_Data_copy_compatible_to_feature_dim():
  # copy_compatible_to should leave the feature dim as-is.
  d1 = Data(name='d1', shape=(None, 11), dtype='float32')
  d2 = Data(name='d2', shape=(13,), dtype='float32', batch_dim_axis=None, time_dim_axis=None)
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
  assert d3.batch_shape == (None, 1, 1)


def test_Data_feature_dim_axis_btd():
  d1 = Data(name="d1", shape=(None, 11), feature_dim_axis=-1)
  d2 = Data(name="d2", shape=(None, 11), feature_dim_axis=2)
  d3 = Data(name="d3", shape=(None, 11))
  d4 = Data(name="d4", feature_dim_axis=2, dim=11)
  assert d1.batch_dim_axis == d2.batch_dim_axis == d3.batch_dim_axis == d4.batch_dim_axis == 0
  assert d1.time_dim_axis == d2.time_dim_axis == d3.time_dim_axis == d4.time_dim_axis == 1
  assert d1.feature_dim_axis == d2.feature_dim_axis == d3.feature_dim_axis == d4.feature_dim_axis == 2
  assert d1.batch_shape == d2.batch_shape == d3.batch_shape == d4.batch_shape == (None, None, 11)
  assert d1._feature_dim_axis == 2
  assert d3._feature_dim_axis is NotSpecified


def test_Data_feature_dim_axis_none():
  d1 = Data(name="d1", shape=())
  d2 = Data(name="d2", shape=(), feature_dim_axis=None)
  d3 = Data(name="d3", shape=(None,), sparse=True, dim=7)
  d4 = Data(name="d4", shape=(None,), sparse=True, dim=7, feature_dim_axis=None)
  assert d1.feature_dim_axis == d2.feature_dim_axis == d3.feature_dim_axis == d4.feature_dim_axis == None
  assert d1._feature_dim_axis is NotSpecified
  assert d2._feature_dim_axis is None


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
  d1 = Data(name='d1', shape=(None, 12))
  assert d1.batch_shape == (None, None, 12) and d1.time_dim_axis == 1 and d1.feature_dim_axis == 2
  d2 = d1.copy_template_excluding_time_dim()
  assert d2.batch_shape == (None, 12) and d2.time_dim_axis is None and d2.feature_dim_axis == 1


def test_Data_copy_template_excluding_time_dim_explicit_feature():
  d1 = Data(name='d1', shape=(None, 12), feature_dim_axis=2)
  assert d1.batch_shape == (None, None, 12) and d1.time_dim_axis == 1 and d1.feature_dim_axis == 2
  d2 = d1.copy_template_excluding_time_dim()
  assert d2.batch_shape == (None, 12) and d2.time_dim_axis is None and d2.feature_dim_axis == 1


def test_Data_copy_add_spatial_dim_no_batch():
  d1 = Data(name='d1', shape=(3,), batch_dim_axis=None, time_dim_axis=None)
  assert d1.batch_dim_axis is None and d1.time_dim_axis is None and d1.feature_dim_axis == 0
  assert d1.batch_shape == (3,) and d1.dim == 3
  d2 = d1.copy_add_spatial_dim(0)
  assert d2.batch_dim_axis is None and d2.time_dim_axis == 0 and d2.feature_dim_axis == 1
  assert d2.batch_shape == (1, 3) and d2.dim == 3


def test_Data_copy_add_spatial_dim_no_batch_explicit_feature():
  d1 = Data(name='d1', shape=(3,), batch_dim_axis=None, time_dim_axis=None, feature_dim_axis=0)
  assert d1.batch_dim_axis is None and d1.time_dim_axis is None and d1.feature_dim_axis == 0
  assert d1.batch_shape == (3,) and d1.dim == 3
  d2 = d1.copy_add_spatial_dim(0)
  assert d2.batch_dim_axis is None and d2.time_dim_axis == 0 and d2.feature_dim_axis == 1
  assert d2.batch_shape == (1, 3) and d2.dim == 3


def test_Data_copy_add_spatial_dim_becomes_new_feature():
  d1 = Data(name='att_weights_avg_output', shape=(None,), batch_dim_axis=None, time_dim_axis=None)
  d2 = d1.copy_add_spatial_dim(0)


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
  assert_almost_equal(session.run(wrap_distribution_non_zero(0.1, zero_limit=0.5, limit=2.)), 0.575)
  assert_almost_equal(session.run(wrap_distribution_non_zero(-0.1, zero_limit=0.5, limit=2.)), -0.575)


def test_close_event_writer_thread():
  import threading
  import tempfile
  from tensorflow.python.summary.writer.event_file_writer import EventFileWriter, _EventLoggerThread

  def count_event_logger_threads():
    return len([t for t in threading.enumerate() if isinstance(t, _EventLoggerThread)])

  tmp_dir = tempfile.mkdtemp()
  writer = tf.summary.FileWriter(tmp_dir)
  assert_equal(count_event_logger_threads(), 1)
  assert isinstance(writer.event_writer, EventFileWriter)
  assert isinstance(writer.event_writer._worker, _EventLoggerThread)
  writer.close()

  # https://github.com/tensorflow/tensorflow/issues/4820
  # The _EventLoggerThread is still running (at least in TF 1.1.0).
  if writer and writer.event_writer and writer.event_writer._worker.is_alive():
    stop_event_writer_thread(writer.event_writer)
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
  x_ref = numpy.array(
    [[0, 1, 2],
     [3, 4, 5],
     [6, 7, 8]])
  numpy.testing.assert_equal(x.eval(), x_ref)
  y = circular_pad(x, paddings=1)
  y_ref = numpy.array(
    [[8, 6, 7, 8, 6],
     [2, 0, 1, 2, 0],
     [5, 3, 4, 5, 3],
     [8, 6, 7, 8, 6],
     [2, 0, 1, 2, 0]])
  numpy.testing.assert_equal(y.eval(), y_ref)

  x = tf.expand_dims(tf.reshape(tf.range(9), (3, 3)), axis=2)
  assert_equal(list(x[0, :, 0].eval()), [0, 1, 2])
  x_ref = numpy.array(
    [[[0], [1], [2]],
     [[3], [4], [5]],
     [[6], [7], [8]]])
  numpy.testing.assert_equal(x.eval(), x_ref)
  y = circular_pad(x, paddings=1, axes=(0, 1))
  y_ref = numpy.array(
    [[[8], [6], [7], [8], [6]],
     [[2], [0], [1], [2], [0]],
     [[5], [3], [4], [5], [3]],
     [[8], [6], [7], [8], [6]],
     [[2], [0], [1], [2], [0]]])
  numpy.testing.assert_equal(y.eval(), y_ref)


def test_reuse_name_scope_double():
  with reuse_name_scope("double"):
    assert_equal(tf.get_default_graph()._name_stack, "double")
    with reuse_name_scope("sub"):
      assert_equal(tf.get_default_graph()._name_stack, "double/sub")
      assert_equal(get_current_name_scope(), "double/sub")


def test_reuse_name_scope_mix1():
  with reuse_name_scope("mix1"):
    assert_equal(tf.get_default_graph()._name_stack, "mix1")
    with tf.name_scope("sub"):
      assert_equal(tf.get_default_graph()._name_stack, "mix1/sub")
      # The following is not true because get_current_name_scope is only var-scope:
      # assert_equal(get_current_name_scope(), "mix1/sub")


def test_reuse_name_scope_mix2():
  with tf.name_scope("mix2"):
    with reuse_name_scope("sub"):
      assert_equal(tf.get_default_graph()._name_stack, "mix2/sub")
      # The following is not true because get_current_name_scope is only var-scope:
      # assert_equal(get_current_name_scope(), "mix2/sub")


def test_reuse_name_scope_mix3():
  with reuse_name_scope("mix3"):
    with tf.variable_scope("sub"):
      assert_equal(get_current_name_scope(), "mix3/sub")


def test_reuse_name_scope_mix4():
  with tf.variable_scope("mix4"):
    with reuse_name_scope("sub"):
      assert_equal(get_current_name_scope(), "mix4/sub")


def test_reuse_name_scope_2():
  with reuse_name_scope("lstm2"):
    with reuse_name_scope("rec") as scope:
      assert_is_instance(scope, tf.VariableScope)
      assert_equal(scope.name, "lstm2/rec")
      assert_equal(get_current_name_scope(), "lstm2/rec")
      with tf.name_scope("sub"):
        assert_equal(get_current_name_scope(), "lstm2/rec/sub")


def test_reuse_name_scope():
  with reuse_name_scope("lstm0"):
    with tf.variable_scope("rec"):
      a = tf.get_variable("a", shape=(3, 4))
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
  with tf.variable_scope("v1"):
    assert_equal(get_current_var_scope_name(), "v1")
    assert_equal(get_current_name_scope(), "v1")
    with tf.variable_scope("v2") as scope:
      assert_equal(get_current_var_scope_name(), "v1/v2")
      assert_equal(get_current_name_scope(), "v1/v2")
      with tf.name_scope("v3"):
        assert_equal(get_current_name_scope(), "v1/v2/v3")
        assert_equal(get_current_var_scope_name(), "v1/v2")
        assert_equal(scope.name, "v1/v2")
        # Note: tf.variable_scope(scope) is broken here.
        with reuse_name_scope(scope):
          assert_equal(get_current_var_scope_name(), "v1/v2")
          assert_equal(get_current_name_scope(), "v1/v2")


def test_name_var_scope_mixing():
  with tf.variable_scope("mv1"):
    assert_equal(get_current_var_scope_name(), "mv1")
    assert_equal(get_current_name_scope(), "mv1")
    with tf.variable_scope("v2") as scope:
      assert_equal(get_current_var_scope_name(), "mv1/v2")
      assert_equal(get_current_name_scope(), "mv1/v2")
      with tf.name_scope("v3"):
        assert_equal(get_current_name_scope(), "mv1/v2/v3")
        assert_equal(get_current_var_scope_name(), "mv1/v2")
        assert_equal(scope.name, "mv1/v2")
        # Note: tf.variable_scope("v4") is broken here.
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
  # Related TF bugs:
  # https://github.com/tensorflow/tensorflow/issues/3114
  # https://github.com/tensorflow/tensorflow/issues/4478
  # https://github.com/tensorflow/tensorflow/issues/8604

  # tf.reset_default_graph()  # Strange, this does not work.
  i = tf.constant(0)

  def body(i):
    # None of these works, with error:
    # InvalidArgumentError: The node 'while/w/Assign' has inputs from different frames.
    # The input 'while/j' is in frame 'while/while/'. The input 'while/w' is in frame ''.
    # w = tf.Variable(tf.constant(1))
    # w = tf.Variable(tf.constant_initializer(value=1, dtype=tf.int32)(shape=()))
    # However, resetting the control dependencies will also reset the frame.
    with var_creation_scope():
      w = tf.Variable(tf.constant(1))
    return [i + w]

  loop = tf.while_loop(lambda i: tf.less(i, 5), body, [i])
  session.run(tf.global_variables_initializer())


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
  idxs_exp = tf.constant(0, shape=(n_beam, n_batch, 2), name="idxs_exp")  # (beam,batch,2), where the 2 stands for (base_time,batch)
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
    shape=(n_base_time, n_batch, n_in))
  session.run(ref_grad)


def test_dimshuffle():
  x = tf.zeros((2, 3, 5))
  assert_equal(list(session.run(tf.shape(x))), [2, 3, 5])
  assert_equal(list(session.run(tf.shape(dimshuffle(x, (1, 2, 0))))), [3, 5, 2])
  assert_equal(list(session.run(tf.shape(dimshuffle(x, ('x', 1, 2, 0))))), [1, 3, 5, 2])
  assert_equal(list(session.run(tf.shape(dimshuffle(x, ('x', 1, 'x', 2, 'x', 0, 'x'))))), [1, 3, 1, 5, 1, 2, 1])
  x = tf.zeros((2, 1, 3))
  assert_equal(list(session.run(tf.shape(dimshuffle(x, (2, 0))))), [3, 2])
  assert_equal(list(session.run(tf.shape(dimshuffle(x, (2, 'x', 'x', 0))))), [3, 1, 1, 2])


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


def naive_windowed_batch(source, window, padding='same'):
  assert source.ndim == 3  # (time,batch,dim). not sure how to handle other cases
  if padding == 'same':
    n_time = source.shape[0]
    w_right = window // 2
    w_left = window - w_right - 1
  elif padding == 'valid':
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
  source = numpy.arange(1, n_time*n_batch*n_dim + 1).reshape(n_time, n_batch, n_dim)
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
  source = numpy.arange(1, n_time*n_batch*n_dim + 1).reshape(n_time, n_batch, n_dim)
  print("source:")
  print(source)
  naive = naive_windowed_batch(source, window=window, padding='valid')
  real = windowed_nd(source, window_size=window, time_axis=0, new_window_axis=1, padding='valid').eval()
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


def naive_slice_nd(x, start, size):
  slices_shape = [x.shape[0], size] + list(x.shape)[2:]
  ys = numpy.zeros(shape=slices_shape)
  for i in range(len(start)):
    time_len = len(x[i])
    end = start[i] + size
    if time_len < end:
      end = time_len
    y = x[i][start[i]:end]

    # padding
    if time_len < start[i] + size:
       y = numpy.pad(y, [[0,start[i]+size-time_len], [0,0]], mode='constant')
    ys[i] = y
  return ys


def test_slice_nd_small():
  n_batch = 3
  n_time = 4
  n_dim = 2
  size = 2
  start = numpy.array([0,2,3]).astype("int32")
  source = numpy.arange(1, n_batch*n_time*n_dim + 1, dtype=numpy.float32).reshape(n_batch, n_time, n_dim).astype("float32")
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
      with tf.Session(graph=graph) as session:
        custom_gradient.register_generic_loss_and_error_signal()
        x = tf.constant(2.)
        session.run(x)  # do some early call, before `generic_loss_and_error_signal` below
        y = custom_gradient.generic_loss_and_error_signal(loss=1., x=x, grad_x=3.)
        assert y.graph is graph
        grad_y, = tf.gradients(y, x)
        assert_equal(session.run([y, x, grad_y]), [1., 2., 3.])
  check()
  check()
  check()


def test_CustomGradient_generic_loss_and_error_signal_post_func():
  with tf.Graph().as_default() as graph:
    with tf.Session(graph=graph) as session:
      custom_gradient.register_generic_loss_and_error_signal()
      x = tf.constant(5.)
      y = custom_gradient.generic_loss_and_error_signal(loss=2., x=x, grad_x=3.)
      z = 2. * y
      assert y.graph is graph
      grad_z, = tf.gradients(z, x)
      assert_equal(session.run([z, x, grad_z]), [4., 5., 6.])


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
  raw = tf.decode_raw(tf.constant("ABC"), tf.uint8)
  assert_equal(list(raw.eval()), [65, 66, 67])


def test_encode_raw_simple():
  raw = tf.decode_raw(tf.constant("hello"), tf.uint8)
  back = encode_raw(raw)
  assert_equal(back.eval(), b"hello")


def test_encode_raw_seq_lens():
  strs = ["hello", "world", "a    "]  # all same lengths for tf.decode_raw
  strs_stripped = [s.strip() for s in strs]
  raw = tf.decode_raw(tf.constant(strs), tf.uint8)
  seq_lens = tf.constant([len(s) for s in strs_stripped])
  back = encode_raw(raw, seq_lens=seq_lens)
  assert_equal(list(back.eval()), [s.encode("utf8") for s in strs_stripped])


@unittest.skip("broken? https://github.com/tensorflow/tensorflow/issues/11240")
def test_sequential_control_dependencies():
  v = tf.Variable(initial_value=2, trainable=False, name="test_sequential_control_dependencies")
  with sequential_control_dependencies([
    lambda: v.initializer,
    lambda: tf.assign(v, 3),
    lambda: tf.assign(v, v.read_value() + 5)
  ]):
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
  v = tf.get_variable(
    initializer=tf.constant_initializer(2), shape=(),
    trainable=False, name="test_resource_var_init", use_resource=True)
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
    with tf.control_dependencies([tf.assign(v, 3)]):
      # `a` is a ref to v, thus also 3 now.
      # `b` is a copy, thus 2, as initially.
      x = tf.add(0, [a, b, v.read_value()])
  assert_equal(list(x.eval()), [3, 2, 3])


def test_Lock():
  lock = Lock()
  session.run(lock.init())
  v = tf.Variable(initial_value=0, trainable=False, name="test_Lock")
  session.run(v.initializer)
  with tf.control_dependencies([lock.lock()]):
    with tf.control_dependencies([v.assign_add(1)]):
      x = enforce_copy(v)
      with tf.control_dependencies([x, lock.unlock()]):
        x = tf.identity(x)
  # Just checking lock + unlock, not really the behavior.
  for i in range(5):
    assert_equal(x.eval(), i + 1)
    assert_equal(v.eval(), i + 1)


def test_Condition():
  cond = Condition()
  v = tf.Variable(initial_value=0, trainable=False, name="test_Condition")
  session.run([cond.init(), v.initializer])
  with sequential_control_dependencies([
    lambda: cond.lock.lock(),
    lambda: v.assign_add(2),
    lambda: cond.signal(),
    lambda: cond.lock.unlock()
  ]):
    s = tf.no_op()
  session.run(cond.lock.lock())
  from threading import Thread
  t = Thread(target=lambda: session.run(s))
  t.start()
  session.run(cond.wait())
  assert_equal(v.eval(), 2)
  t.join()
  session.run(cond.lock.unlock())


@unittest.skip("needs tensor_array.h, see https://github.com/tensorflow/tensorflow/issues/10527")
def test_GlobalTensorArray():
  GlobalTensorArrayOpMaker().get_op()


def test_TFArrayContainer():
  # Bug #10950 is fixed upstream, should be in TF 1.2.2.
  # https://stackoverflow.com/questions/44455722/create-my-own-resource-types-tf-resource
  # https://github.com/tensorflow/tensorflow/issues/1419
  ta = TFArrayContainer(dtype=tf.int32)
  print(ta._mod)
  print(ta._mod.array_container_create.__doc__)
  assert_equal(ta.get_size().eval(), 0)
  session.run(ta.set_size(3))
  assert_equal(ta.get_size().eval(), 3)
  session.run(ta.set(1, [1, 2, 3]))
  assert_equal(list(ta.get(1).eval()), [1, 2, 3])


@unittest.skip("does not work")
def test_TensorArray():
  # see https://stackoverflow.com/questions/44418036/
  # Reason is that the TensorArray uses a per-run ("per-step") resource manager,
  # thus it will not remember anything across session.run() calls.
  # This is by design.
  # Our :class:`GlobalTensorArrayOpMaker` could fix this.
  ta = tf.TensorArray(tf.int32, size=3)
  index = tf.placeholder(tf.int32)
  value = tf.placeholder(tf.int32)
  flow = tf.placeholder(tf.float32)
  ta_new = tf.TensorArray(dtype=ta.dtype, handle=ta.handle, flow=flow)
  write = ta_new.write(index, value).flow
  read = ta_new.read(index)
  f = 0
  f = session.run(write, feed_dict={index: 0, value: 1, flow: f})
  f = session.run(write, feed_dict={index: 1, value: 2, flow: f})
  assert_equal(session.run(read, feed_dict={index: 0, flow: f}), 1)
  assert_equal(session.run(read, feed_dict={index: 1, flow: f}), 2)


@unittest.skip("does not work")
def test_ExplicitRandomShuffleQueue():
  # see test_TensorArray, which is internally used by ExplicitRandomShuffleQueue
  queue = ExplicitRandomShuffleQueue(capacity=3, min_after_dequeue=2, dtypes=[tf.int32])
  placeholder = tf.placeholder(tf.int32, shape=())
  session.run(queue.init())
  enqueue = queue.enqueue(placeholder)
  dequeue = queue.dequeue()
  size = queue.size()
  session.run(enqueue, feed_dict={placeholder: 1})
  session.run(enqueue, feed_dict={placeholder: 2})
  session.run(enqueue, feed_dict={placeholder: 3})
  pool = {1, 2, 3}
  for i in range(3):
    d = session.run(dequeue)
    assert_in(d, pool)
    pool.remove(d)
    session.run(enqueue, feed_dict={placeholder: i + 4})
    pool.add(i + 4)
    assert_equal(session.run(size), len(pool))
  session.run(queue.min_after_dequeue_assign(0))
  while pool:
    d = session.run(dequeue)
    assert_in(d, pool)
    pool.remove(d)
  assert_equal(session.run(size), 0)
  session.run(enqueue, feed_dict={placeholder: 17})
  assert_equal(session.run(dequeue), 17)


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
  assert_almost_equal(resv, [[[2*5.0+3*7.0], [2*7.0]]])


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

  where_0_nan = tf.where(True, 0., float("nan"))
  print("where_0_nan:", where_0_nan.eval())
  assert_equal(where_0_nan.eval(), 0.0)

  x = tf.constant(0.)
  x_equal_0 = tf.equal(x, 0.)
  f = tf.where(x_equal_0, 0., 1. / x)
  grad_x = tf.gradients(f, x)[0]
  print("grad_x:", grad_x.eval())  # nan? or 0?
  # This is expected when you look at the resulting computation graph for the gradient.
  # You will have grad(1./x, x) * 0.0 in the graph in the back-propagation of the gradient, which is nan.
  assert_equal(str(grad_x.eval()), "nan")

  safe_x = tf.where(x_equal_0, 2., x)
  grad_safe_x = tf.where(x_equal_0, 0., 1. / safe_x)
  print("grad_safe_x:", grad_safe_x.eval())  # nan? ln(2)? 0?
  # This works, because at no time, there is nan in the back-propagation.
  assert_equal(grad_safe_x.eval(), 0.0)

  f = tf.cond(x_equal_0, lambda: 0., lambda: 1. / x)
  grad_cond_x = tf.gradients(f, x)[0]
  print("grad_cond_x:", grad_cond_x.eval())  # nan? or 0?
  # This is different than tf.where because really only one branch will go into the gradient.
  assert_equal(grad_cond_x.eval(), 0.0)


def test_variable_summaries():
  v = tf.Variable(initial_value=[[1.0, 2.0], [-4.0, -1.0]], name="test_variable_summaries")
  variable_summaries(v)
  variable_summaries(tf.square(v))
  session.run(v.initializer)
  session.run(tf.summary.merge_all())
  assert_almost_equal(session.run(variable_scalar_summaries_dict(v)["test_variable_summaries_mean"]), -0.5)


def test_VariableAssigner():
  v = tf.Variable(initial_value=1.)
  session.run(v.initializer)
  assert_equal(session.run(v), 1.)
  assigner = VariableAssigner(v)
  assigner.assign(value=2., session=session)
  assert_equal(session.run(v), 2.)


def test_VariableAssigner_ResourceVariable():
  v = tf.get_variable(
    initializer=tf.constant_initializer(1.), shape=(),
    name="test_VariableAssigner_ResourceVariable", use_resource=True)
  session.run(v.initializer)
  assert_equal(session.run(v), 1.)
  assigner = VariableAssigner(v)
  assigner.assign(value=2., session=session)
  assert_equal(session.run(v), 2.)


def test_map_labels():
  x = tf.constant([0, 1, 2, 3, 2, 1, 0])
  label_map = {0: 1, 1: 2, 2: 3, 3: 0}
  y = map_labels(x, label_map=label_map)
  assert_equal(session.run(y).tolist(), [1, 2, 3, 0, 3, 2, 1])


def test_map_labels_SparseTensor():
  x = tf.SparseTensor(
    indices=tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.int64, name="x_indices"),
    values=tf.constant([0, 1, 2, 3], name="x_values"),
    dense_shape=tf.constant([3, 3], dtype=tf.int64, name="x_dense_shape"))
  label_map = {0: 1, 1: 2, 2: 3, 3: 0}
  y = map_labels(x, label_map=label_map)
  assert isinstance(y, tf.SparseTensor)
  y_eval = session.run(y)
  assert isinstance(y_eval, tf.SparseTensorValue)
  assert_equal(y_eval.values.tolist(), [1, 2, 3, 0])


def test_sparse_labels():
  x = tf.constant([[0, 1, 2, 3], [4, 5, 0, 0]], name="x")
  seq_lens = tf.constant([4, 2], name="seq_lens")
  y = sparse_labels(x, seq_lens=seq_lens)
  y_eval = session.run(y)
  assert isinstance(y_eval, tf.SparseTensorValue)
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
    dense_shape=tf.constant([3, 3], dtype=tf.int64, name="x_dense_shape"))
  labels = {1}
  y = remove_labels(x, labels=labels)
  assert isinstance(y, tf.SparseTensor)
  y_eval = session.run(y)
  assert isinstance(y_eval, tf.SparseTensorValue)
  assert isinstance(y_eval.indices, numpy.ndarray)
  assert isinstance(y_eval.values, numpy.ndarray)
  assert isinstance(y_eval.dense_shape, numpy.ndarray)
  assert_equal(y_eval.indices.tolist(), [[0, 0], [0, 1], [1, 0]])
  assert_equal(y_eval.values.tolist(), [0, 2, 3])
  assert_equal(y_eval.dense_shape.tolist(), [3, 2])


def test_supported_devices_for_op():
  op_name = "MatMul"
  devs = supported_devices_for_op(op_name)
  print("Supported devs for op %r: %r" % (op_name, devs))
  assert "CPU" in devs


def test_bleu_score():
  hyp = [1, 2, 3]
  truth = [2, 3]
  from Util import compute_bleu
  res = compute_bleu([truth], [hyp])
  print("res:", res)
  tf_res = session.run(bleu_score(
    hypothesis=[hyp], hyp_seq_lens=[len(hyp)],
    truth=[truth], truth_seq_lens=[len(truth)]
  ))
  print("TF res:", tf_res)
  assert isinstance(tf_res, numpy.ndarray)
  assert tf_res.shape == (1,)
  assert_almost_equal(tf_res, [res])
  assert_almost_equal(tf_res, [0.6389431])


def test_bleu_score_empty():
  hyp = []
  truth = [2, 3]
  from Util import compute_bleu
  res = compute_bleu([truth], [hyp])
  print("res:", res)
  tf_res = session.run(bleu_score(
    hypothesis=[hyp], hyp_seq_lens=[len(hyp)],
    truth=[truth], truth_seq_lens=[len(truth)]
  ))
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
  from tensorflow.contrib import graph_editor
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


def test_clip_by_value_with_identity_grad():
  err_y = 42.0
  limit = 1.0
  limits = -limit, limit
  with tf.name_scope("test_safe_log_and_grad"):
    x_t = tf.placeholder(tf.float32, shape=(), name="x")
    y_t = clip_by_value_with_identity_grad(x_t, *limits)
    err_x_t, = tf.gradients(ys=y_t, xs=x_t, grad_ys=tf.constant(err_y))
    err2_x_t, = tf.gradients(ys=tf.clip_by_value(x_t, *limits), xs=x_t, grad_ys=tf.constant(err_y))

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
    x_t = tf.placeholder(tf.float32, shape=(), name="x")
    y_t = safe_log(x_t)
    err_x_t, = tf.gradients(ys=y_t, xs=x_t)
    check_numerics_op = add_check_numerics_ops([y_t, err_x_t])
    # For comparison:
    y2_t = tf.log(x_t)
    err2_x_t, = tf.gradients(ys=y2_t, xs=x_t)

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
    x_t = tf.placeholder(tf.float32, shape=(), name="x")
    y_t = safe_exp(x_t)
    err_x_t, = tf.gradients(ys=y_t, xs=x_t)
    check_numerics_op = add_check_numerics_ops([y_t, err_x_t])
    # For comparison:
    y2_t = tf.exp(x_t)
    err2_x_t, = tf.gradients(ys=y2_t, xs=x_t)

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
    x_t = tf.placeholder(tf.float32, shape=(None,), name="x")
    y_t = lin_exp_normed(x_t)
    # Also see :class:`CrossEntropyLoss`. here score instead of loss.
    score_t = safe_log(y_t[..., -1])
    err_x_t, = tf.gradients(ys=score_t, xs=x_t)
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
    y = tf.log(x)
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
    y = tf.log(x)
    print("x:", x, list(x.op.inputs), "y:", y)
    y2 = check_base_op_type_and_replace(x, "Sigmoid", "LogSigmoid")
    print("y2:", y2)
    assert y2 is not None
    vy1, vy2 = session.run([y, y2])
    print("eval:", vy1, vy2)
    assert_almost_equal(vy1, vy2)


def test_string_merge():
  strings = [
    ["sub@@", "word", "test"],
    ["hel@@", "lo", "wo@@", "r@@", "ld"],
    ["foo"]]
  seq_lens = [len(seq) for seq in strings]
  max_len = max(seq_lens)
  strings = [seq + [""] * (max_len - len(seq)) for seq in strings]

  tf_strings = tf.placeholder(tf.string, [None, None])
  tf_seq_lens = tf.placeholder(tf.int32, [None])
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


def test_string_replace():
  strings = ["sub@@ word test", "hel@@ lo wo@@ r@@ ld", "foo"]
  tf_strings = tf.placeholder(tf.string, [None])
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
  tf_strings = tf.placeholder(tf.string, [None])
  tf_words = words_split(tf_strings)
  tf_dense_words = tf.sparse_to_dense(
    tf_words.indices, tf_words.dense_shape, tf_words.values, default_value="")
  tf_num_words = get_sparse_tensor_length(tf_words)
  words, dense_words, num_words = session.run(
    [tf_words, tf_dense_words, tf_num_words], feed_dict={tf_strings: strings})
  print(words)
  print(dense_words)
  print(num_words)
  assert isinstance(words, tf.SparseTensorValue)
  assert isinstance(dense_words, numpy.ndarray)
  assert isinstance(num_words, numpy.ndarray)
  assert dense_words.shape == (len(word_lens), max(word_lens))
  assert num_words.shape == (len(strings),)
  dense_words = dense_words.tolist()
  print(dense_words)
  assert_equal(dense_words, [
    [b"subword", b"test", b"", b""], [b"a", b"b", b"c", b"d"],
    [b"hello", b"world", b"", b""], [b"foo", b"", b"", b""]])
  assert_equal(num_words.tolist(), word_lens)


def test_string_words_calc_wer():
  hyps = ["hello world", "a b c", "how are you", "good"]
  refs = ["hello nice world", "a x c d", "how are we", "good"]
  tf_hyps = tf.placeholder(tf.string, [None])
  tf_refs = tf.placeholder(tf.string, [None])
  tf_wer, tf_ref_num_words = string_words_calc_wer(hyps=tf_hyps, refs=tf_refs)
  wer, ref_num_words = session.run([tf_wer, tf_ref_num_words], {tf_hyps: hyps, tf_refs: refs})
  print(wer, ref_num_words)
  assert isinstance(wer, numpy.ndarray)
  assert isinstance(ref_num_words, numpy.ndarray)
  assert_equal(wer.tolist(), [1, 2, 1, 0])
  assert_equal(ref_num_words.tolist(), [3, 4, 3, 1])


def test_kenlm():
  import TFKenLM
  input_strings = ["beyond immediate concerns </s>"]
  test_lm_file = TFKenLM.kenlm_dir + "/lm/test.arpa"
  assert os.path.exists(test_lm_file)
  lm_tf = TFKenLM.ken_lm_load(filename=test_lm_file)
  input_strings_tf = tf.placeholder(tf.string, [None])
  output_scores_tf = TFKenLM.ken_lm_abs_score_strings(handle=lm_tf, strings=input_strings_tf)
  with tf.Session() as session:
    output_scores = session.run(output_scores_tf, feed_dict={input_strings_tf: input_strings})
  print("input strings:", input_strings)
  print("output scores:", output_scores)
  assert isinstance(output_scores, numpy.ndarray)
  assert_almost_equal(output_scores, [-9.251298])  # +log space, not +log10
  print("Score is as expected.")


def test_kenlm_bpe():
  import TFKenLM
  input_strings = [
    "beyond immediate concerns </s>",
    "be@@ yond imm@@ edi@@ ate conc@@ erns </s>",
    "be@@ yond imm@@",
    "be@@ yond <unk>"
    ]
  test_lm_file = TFKenLM.kenlm_dir + "/lm/test.arpa"
  assert os.path.exists(test_lm_file)
  lm_tf = TFKenLM.ken_lm_load(filename=test_lm_file)
  input_strings_tf = tf.placeholder(tf.string, [None])
  output_scores_tf = TFKenLM.ken_lm_abs_score_bpe_strings(handle=lm_tf, strings=input_strings_tf, bpe_merge_symbol="@@")
  with tf.Session() as session:
    output_scores = session.run(output_scores_tf, feed_dict={input_strings_tf: input_strings})
  print("input strings:", input_strings)
  print("output scores:", output_scores)
  assert isinstance(output_scores, numpy.ndarray)
  assert_equal(output_scores.shape, (len(input_strings),))
  assert_almost_equal(output_scores[0], -9.251298)  # example from above
  assert_equal(output_scores[0], output_scores[1])
  assert_equal(output_scores[2], output_scores[3])
  print("Scores are as expected.")


def test_layer_norms():
  from TFNativeOp import have_blocksparse_requirements
  from tensorflow.contrib.layers import layer_norm as tf_contrib_layer_norm
  rnd = numpy.random.RandomState(3)
  for ndim in [2, 3, 4]:
    dims = [3] * ndim
    x_np = rnd.rand(*dims).astype('float32')
    print('x:')
    print(x_np)
    with tf.name_scope("test_ndim_%i" % ndim):
      x = tf.constant(x_np, name='x')
      g = tf.ones([3])
      b = tf.zeros([3])
      for axis in range(ndim):
        with tf.name_scope('test_axis_%i' % axis):
          print('ndim %i, axis %i' % (ndim, axis))
          ln = layer_norm(x=x, gain=g, bias=b, axis=axis)
          if not have_blocksparse_requirements():
            print('  OpenAI cannot be used')
            ln2 = ln
          # OpenAI seems to be broken for these cases:
          elif axis < ndim - 1:
            print('  ignore OpenAI layer norm for this case')
            ln2 = ln
          else:
            ln2 = openai_layer_norm(x=x, gain=g, bias=b, axis=axis)
          if axis < ndim - 1:
            print('  cannot use tf.contrib layer norm for this case')
            ln3 = ln  # cannot use tf_contrib_layer_norm
          else:
            ln3 = tf_contrib_layer_norm(x, center=False, scale=False, begin_norm_axis=axis, begin_params_axis=axis)
          ln_np, ln2_np, ln3_np = session.run((ln, ln2, ln3))
          print('layer norm:')
          print(ln_np)
          assert isinstance(ln_np, numpy.ndarray)
          assert isinstance(ln2_np, numpy.ndarray)
          assert isinstance(ln3_np, numpy.ndarray)
          assert x_np.shape == ln_np.shape == ln2_np.shape == ln3_np.shape
          assert_allclose(ln_np, ln2_np, rtol=1e-4)
          assert_allclose(ln_np, ln3_np, rtol=5e-2)
          print('ok')


def test_transform_param_axes_split_info_to_new_shape():
  assert_equal(transform_param_axes_split_info_to_new_shape([[7],[7]*4], [7*2,7*8]), [[7*2],[7*2]*4])
  assert_equal(transform_param_axes_split_info_to_new_shape([[3,7],[7]*4], [3+7*2,7*8]), [[3,7*2],[7*2]*4])
  assert_equal(transform_param_axes_split_info_to_new_shape([[3,7],[7]*4], [1+7*2,7*8]), [[1,7*2],[7*2]*4])
  assert_equal(transform_param_axes_split_info_to_new_shape([[7,7],[7]*4], [3+7*2,7*8]), [[3,7*2],[7*2]*4])
  assert_equal(transform_param_axes_split_info_to_new_shape([[7,7],[7]*4], [7*2+7*2,7*8]), [[7*2,7*2],[7*2]*4])
  assert_equal(transform_param_axes_split_info_to_new_shape([[7],[7]*4], [7,7*8]), [[7],[7*2]*4])


def test_get_op_attrib_keys():
  x = tf.matmul(a=tf.zeros((3, 4, 5)), b=tf.zeros((3, 5, 7)))
  assert isinstance(x, tf.Tensor)
  assert isinstance(x.op, tf.Operation)
  print("x op:", x.op.type)
  assert_equal(x.op.type, "BatchMatMul")
  assert_equal(x.get_shape().as_list(), [3, 4, 7])
  attrib_keys = get_op_attrib_keys(x)
  print("matmul attrib keys:", attrib_keys)
  assert_equal(sorted(attrib_keys), ['T', 'adj_x', 'adj_y'])
  dtype = x.op.get_attr("T")
  assert_equal(dtype, tf.float32)


def test_get_op_input_names_MatMul():
  x = tf.matmul(a=tf.zeros((3, 4, 5)), b=tf.zeros((3, 5, 7)))
  assert isinstance(x, tf.Tensor)
  assert isinstance(x.op, tf.Operation)
  print("x op:", x.op.type)
  assert_equal(x.op.type, "BatchMatMul")
  input_names = get_op_input_names(x.op)
  print("matmul input names:", input_names)
  assert_equal(sorted(input_names), ['x', 'y'])


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
  with tf.variable_scope("test_get_op_attrib_keys__is_variable_initialized"):
    var = tf.get_variable("var", shape=(3,))
    check = tf.is_variable_initialized(var)
    print("check:", check)
    assert isinstance(check, tf.Tensor)
    print("op:", check.op)
    assert_equal(check.op.type, "IsVariableInitialized")
    print("attrib keys:", get_op_attrib_keys(check.op))


def test_print_graph_output():
  x = tf.matmul(a=tf.zeros((3, 4, 5)), b=tf.zeros((3, 5, 7)))
  x.set_shape((3, 4, 7))
  x = tf.reshape(x, [3, 4 * 7])
  x = x + tf.constant(3.0)
  x = safe_log(tf.nn.softmax(x))
  print_graph_output(x)


def test_get_var_ops():
  with tf.variable_scope("test_get_var_ops"):
    v = tf.get_variable("v", ())
    assert_equal(find_ops_with_tensor_input(v), [v.initializer])


def test_find_ops_with_tensor_input():
  with tf.variable_scope("test_find_ops_with_tensor_input"):
    x0 = tf.constant(1.0, name="x0")
    v1 = tf.get_variable("v1", ())
    v2 = tf.get_variable("v2", ())
    x1a = tf.add(x0, v1, name="x1a")
    x1b = tf.add(x1a, v2, name="x1b")
    x2a = tf.multiply(v1, v2, name="x2a")
    x2b = tf.multiply(x2a, x0, name="x2b")
    assert_equal(find_ops_with_tensor_input(x0), [x1a.op, x2b.op])
    assert_equal(find_ops_with_tensor_input(v1), [v1.initializer, x1a.op, x2a.op])
    assert_equal(find_ops_with_tensor_input(v2), [v2.initializer, x1b.op, x2a.op])
    assert_equal(find_ops_with_tensor_input(v2, fetches=[x2b]), [x2a.op])


def test_get_var_update_ops():
  with tf.variable_scope("test_get_var_update_ops"):
    v = tf.get_variable("v", ())
    loss = (v - 1.0) ** 2
    opt = tf.train.AdamOptimizer()
    minimize_op = opt.minimize(loss=loss, var_list=[v])
    assert isinstance(minimize_op, tf.Operation)
    print("find_ops_with_tensor_input:", find_ops_with_tensor_input(v, fetches=minimize_op))
    print("get_var_update_ops:", get_var_update_ops(v, fetches=minimize_op))
    update_ops = get_var_update_ops(v, fetches=minimize_op)
    assert len(update_ops) == 1
    assert update_ops[0].type == "ApplyAdam"


def test_get_var_update_ops__get_variable_value_copy_before_update_ops():
  with tf.variable_scope("test_get_var_update_ops__get_variable_value_copy_before_update_ops"):
    v = tf.get_variable("v", (), initializer=tf.zeros_initializer())
    assert isinstance(v, tf.Variable)
    loss = (v - 1.0) ** 2
    assert isinstance(loss, tf.Tensor)
    opt = tf.train.GradientDescentOptimizer(learning_rate=1.0)
    minimize_op = opt.minimize(loss=loss, var_list=[v])
    assert isinstance(minimize_op, tf.Operation)
    print("find_ops_with_tensor_input:", find_ops_with_tensor_input(v, fetches=minimize_op))
    print("get_var_update_ops:", get_var_update_ops(v, fetches=minimize_op))
    update_ops = get_var_update_ops(v, fetches=minimize_op)
    assert len(update_ops) == 1
    assert update_ops[0].type == "ApplyGradientDescent"
    with tf.control_dependencies(update_ops):
      # v.value() is the last snapshot (no new op), i.e. it points to the actual memory.
      # To make sure we get the value before the update (0), we must do a copy at the right point.
      v_val = get_variable_value_copy_before_update_ops(v, update_ops)
      # v.read_value() is a new read op to the current value.
      # Anyway, make sure that we have the same everywhere below.
      # This should be the value after the update, and the grad is -2, lr 1, thus should be 2.
      v_read_val = tf.identity(v.read_value())
      res = [
        tf.Print(0, ["loss:", loss]), tf.Assert(tf.equal(loss, 1.0), ["loss ", loss, " == 1"]),
        tf.Print(0, ["v:", v]),
        tf.Print(0, ["v.value:", v_val]),
        tf.Assert(tf.equal(v_val, 0.0), ["v.value ", v_val, " == 0"]),  # last snapshot
        tf.Print(0, ["v.read_value:", v_read_val]),
        tf.Assert(tf.equal(v_read_val, 2.0), ["v.read_value ", v_read_val, " == 2"])  # after update
      ]
    session.run(v.initializer)
    session.run([loss, minimize_op, res])


def test_get_variable_grad_from_update_ops():
  with tf.variable_scope("test_get_variable_grad_from_update_ops"):
    var = tf.get_variable("var", (), initializer=tf.zeros_initializer())
    loss = (var - 1.0) ** 2
    for opt in [
      tf.train.AdamOptimizer(),
      tf.train.GradientDescentOptimizer(learning_rate=1.0),
      tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9),
      tf.train.RMSPropOptimizer(learning_rate=0.1),
    ]:
      print("Optimizer:", opt)
      minimize_op = opt.minimize(loss=loss, var_list=[var])
      assert isinstance(minimize_op, tf.Operation)
      update_ops = get_var_update_ops(var, fetches=minimize_op)
      print("update ops:", update_ops)
      print("update op keys:", get_op_attrib_keys(update_ops[0]))
      print("update op inputs by name:", get_op_input_names(update_ops[0]))
      session.run(var.initializer)  # reset
      session.run(tf.global_variables_initializer())  # from Adam or so
      assert_equal(session.run(var), 0.0)
      grad = get_variable_grad_from_update_ops(var, update_ops)
      print("grad:", grad)
      _, grad_np = session.run([minimize_op, grad])
      assert_equal(grad_np, -2.0)


def test_get_variable_grad_from_update_ops_mix_sparse_dense():
  with tf.variable_scope("test_get_variable_grad_from_update_ops_mix_sparse_dense"):
    var = tf.get_variable("var", (3, 5), initializer=tf.ones_initializer())
    loss = tf.reduce_sum((tf.matmul(tf.nn.embedding_lookup(var, [1]) - 1.0, tf.transpose(var)) - 1.0) ** 2)
    ref_grad, = tf.gradients(loss, var)
    ref_grad = tf.convert_to_tensor(ref_grad)
    session.run(var.initializer)  # reset
    ref_grad_np = session.run(ref_grad)
    print("ref grad value:")
    print(ref_grad_np)
    for opt in [
      tf.train.AdamOptimizer(),
      tf.train.GradientDescentOptimizer(learning_rate=1.0),
      tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9),
      tf.train.RMSPropOptimizer(learning_rate=0.1),
    ]:
      print("Optimizer:", opt)
      if isinstance(opt, (tf.train.MomentumOptimizer, tf.train.RMSPropOptimizer)):
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
      session.run(tf.global_variables_initializer())  # from Adam or so
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
  with tf.variable_scope("test_mixed_dense_sparse_grad"):
    var = tf.get_variable("var", (3, 5), initializer=tf.ones_initializer())
    session.run(var.initializer)
    loss = tf.reduce_sum(tf.nn.embedding_lookup(var, [1]) ** 2) + tf.reduce_sum(var ** 2)
    grad, = tf.gradients(loss, var)
    print("grad:", grad)
    # It is an IndexedSlices.
    # https://github.com/tensorflow/tensorflow/issues/21243
    grad_dense = tf.convert_to_tensor(grad)
    print("grad dense:", grad_dense)
    print("grad value:")
    print(session.run(grad))
    print("grad dense value:")
    print(session.run(grad_dense))
    opt = tf.train.GradientDescentOptimizer(learning_rate=1.0)
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
  old_values = numpy.arange((3+5)*5*4).reshape((3+5),5*4)
  new_values = copy_with_new_split_axes([[3,5],[5]*4], [[5,7],[7]*4], old_values)
  for p in range(4):
    assert (new_values[:3,p*7:p*7+5] == old_values[:3,p*5:p*5+5]).all()
    assert (new_values[5:5+5,p*7:p*7+5] == old_values[3:,p*5:p*5+5]).all()


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
    with same_context(outer_loop_val):
      v2 = outer_loop_val + 2
    print("v2 control flow:", v2.op._control_flow_context)
    assert not has_control_flow_context(v2)  # should be done outside now, because `same_context` usage
    return t + 1, v0 + v1 + v2

  x = tf.while_loop(
    cond=lambda t, _x: tf.less(t, 3),
    body=body,
    loop_vars=(0, 0))
  print("magic (totally arbitrary) res:", session.run(x))


def test_softmax_cross_entropy_over_size_batch1():
  energy_np = numpy.array([
    [0.00060279], [0.00106305], [0.00139351], [0.0016565], [0.00179641], [0.00188511], [0.00197855],
    [0.00212687], [0.00229054], [0.00253187], [0.0028633], [0.00317292], [0.00333956], [0.00321618],
    [0.00301804], [0.00286921], [0.00269102], [0.00233986], [0.00198704], [0.00179809], [0.00180432],
    [0.00180019], [0.00168032], [0.00148445], [-0.00021808], [0.00024213], [0.00057256], [0.00083552],
    [0.00097541], [0.00106409], [0.00115748], [0.00130573], [0.00146933], [0.00171059], [0.00204194],
    [0.0023515], [0.00251812], [0.00239481], [0.00219674], [0.00204793], [0.00186977], [0.00151865],
    [0.00116586], [0.00097693], [0.00098313], [0.000979], [0.00085914], [0.00066328]], dtype="float32")
  energy_sizes = [2, 24]
  ref_att_weights_np = numpy.array([
    [8.85698071e-04], [3.03550856e-03], [4.28047322e-04], [2.12707062e-04], [1.69979918e-04], [1.69104227e-04],
    [1.76708025e-04], [2.30993493e-04], [1.30674185e-03], [1.69335492e-02], [5.89773171e-02], [1.02726415e-01],
    [1.38920724e-01], [3.36773008e-01], [3.02626193e-01], [3.60515639e-02], [1.24421655e-04], [1.09412758e-06],
    [7.09717142e-07], [7.12369911e-07], [1.48852378e-05], [1.92153064e-04], [1.94299287e-06], [3.98656666e-05],
    [1.83324970e-03], [9.68748587e-04], [7.93990912e-05], [3.37559104e-05], [2.54511542e-05], [2.01851981e-05],
    [1.46051125e-05], [8.32758542e-06], [1.90258543e-05], [1.61443924e-04], [7.95573505e-05], [6.75756164e-05],
    [1.12831636e-04], [4.37621129e-05], [5.69019585e-06], [5.78170584e-04], [2.09521659e-05], [1.89785933e-05],
    [1.07380874e-04], [1.02525763e-03], [4.51886881e-04], [1.37639674e-03], [9.68037128e-01], [2.49103159e-02]],
    dtype="float32")
  n_batch = 1
  n_extra_dim = 1  # number of att heads
  assert energy_np.shape == ref_att_weights_np.shape == (numpy.prod(energy_sizes), n_extra_dim)
  sizes_tf = {
    i: tf.constant(numpy.array(energy_sizes[i], dtype="int32").reshape((n_batch,))) for i in range(len(energy_sizes))}
  energy_tf = tf.constant(energy_np.reshape([n_batch] + energy_sizes + [n_extra_dim]))
  ref_att_weights_tf = tf.constant(ref_att_weights_np.reshape([n_batch] + energy_sizes + [n_extra_dim]))
  energy_data = Data(
    name="energy", shape=(None, None, n_extra_dim), batch_dim_axis=0,
    placeholder=energy_tf, size_placeholder=sizes_tf)
  ref_att_weights_data = Data(
    name="ref_att_weights", shape=(None, None, n_extra_dim), batch_dim_axis=0,
    placeholder=ref_att_weights_tf, size_placeholder=sizes_tf)
  res_tf = softmax_cross_entropy_over_size(logits=energy_data, labels=ref_att_weights_data)
  res_tf.set_shape((n_batch, energy_sizes[0], n_extra_dim))
  res_np = session.run(res_tf)
  print("res:", res_np)
  assert numpy.alltrue(numpy.isfinite(res_np))


def test_softmax_cross_entropy_over_size_n_batch():
  energy_np = numpy.array([
    [0.00060279], [0.00106305], [0.00139351], [0.0016565], [0.00179641], [0.00188511], [0.00197855],
    [0.00212687], [0.00229054], [0.00253187], [0.0028633], [0.00317292], [0.00333956], [0.00321618],
    [0.00301804], [0.00286921], [0.00269102], [0.00233986], [0.00198704], [0.00179809], [0.00180432],
    [0.00180019], [0.00168032], [0.00148445], [-0.00021808], [0.00024213], [0.00057256], [0.00083552],
    [0.00097541], [0.00106409], [0.00115748], [0.00130573], [0.00146933], [0.00171059], [0.00204194],
    [0.0023515], [0.00251812], [0.00239481], [0.00219674], [0.00204793], [0.00186977], [0.00151865],
    [0.00116586], [0.00097693], [0.00098313], [0.000979], [0.00085914], [0.00066328]], dtype="float32")
  energy_sizes = [2, 24]
  ref_att_weights_np = numpy.array([
    [8.85698071e-04], [3.03550856e-03], [4.28047322e-04], [2.12707062e-04], [1.69979918e-04], [1.69104227e-04],
    [1.76708025e-04], [2.30993493e-04], [1.30674185e-03], [1.69335492e-02], [5.89773171e-02], [1.02726415e-01],
    [1.38920724e-01], [3.36773008e-01], [3.02626193e-01], [3.60515639e-02], [1.24421655e-04], [1.09412758e-06],
    [7.09717142e-07], [7.12369911e-07], [1.48852378e-05], [1.92153064e-04], [1.94299287e-06], [3.98656666e-05],
    [1.83324970e-03], [9.68748587e-04], [7.93990912e-05], [3.37559104e-05], [2.54511542e-05], [2.01851981e-05],
    [1.46051125e-05], [8.32758542e-06], [1.90258543e-05], [1.61443924e-04], [7.95573505e-05], [6.75756164e-05],
    [1.12831636e-04], [4.37621129e-05], [5.69019585e-06], [5.78170584e-04], [2.09521659e-05], [1.89785933e-05],
    [1.07380874e-04], [1.02525763e-03], [4.51886881e-04], [1.37639674e-03], [9.68037128e-01], [2.49103159e-02]],
    dtype="float32")
  n_extra_dim = 1  # number of att heads
  assert energy_np.shape == ref_att_weights_np.shape == (numpy.prod(energy_sizes), n_extra_dim)
  n_batch = 5
  energy_np = energy_np.reshape([1] + energy_sizes + [n_extra_dim]).repeat(n_batch, axis=0)
  ref_att_weights_np = ref_att_weights_np.reshape([1] + energy_sizes + [n_extra_dim]).repeat(n_batch, axis=0)
  sizes_tf = {
    i: tf.constant([energy_sizes[i]] * n_batch, dtype="int32") for i in range(len(energy_sizes))}
  energy_tf = tf.constant(energy_np)
  ref_att_weights_tf = tf.constant(ref_att_weights_np)
  energy_data = Data(
    name="energy", shape=(None, None, n_extra_dim), batch_dim_axis=0,
    placeholder=energy_tf, size_placeholder=sizes_tf)
  ref_att_weights_data = Data(
    name="ref_att_weights", shape=(None, None, n_extra_dim), batch_dim_axis=0,
    placeholder=ref_att_weights_tf, size_placeholder=sizes_tf)
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
  energy_np = numpy.frombuffer(gzip.decompress(base64.decodebytes(
    b'H4sIAK3yb1wC/33SeVSU5xUG8GEyLKIwIChrgOhAIwZEsfh994IiGQwFgqgDoS6AiCBbBKpxsNYE'
    b'WWSRfdMwEQRDAJXFAeR7X1JIDShbwqIISJESIQlERFEMICaenNNzUlv7nPP77/5xz3Oe9HtGEGWT'
    b'BJum6qA7vQ9EDT9Boew5MLMC3OcuxCeGBjg6bY56Q/ZYZuKJ+Z1+mNG4B0UFLviwcjPW+WzA/CJD'
    b'zGYEGMyOQmPVLVA63Q3NiTegZLIWeK9ko9SXc606yioPrIRDhp4wzUTDed0M0Ao8D6v05MCragM3'
    b'jwmQai9FvpcJVgkssOiZKab2r0B2QgXHN86BRkMneP8ogzL/MLiycydYvXCB3W8wsNCiDV0KRuT3'
    b'Mu82k3W6yUSlxZY4LFngHvsWcbx2R86kua7enslm3vGNYSue8uBQnhf0dOWCfE0V5GsWAdMXDWmz'
    b'22EXZwR12Wns3uvf1O9M6uAefaBAPrqiTOJLDUjH/c2kZvvQfxgo/hJKl2lg7UoHvP6nMCxXSMD4'
    b'q7mYda0IB0KvYMYXtahjx6HNz3Js4RfhNkkcHh+RYPc1G2xIscDlhtY4ZmqD2bNWuHaLId6aWvyv'
    b'Dn+f2Qp9zBl3wZAHf8P7qdlYNvE5Nn9ag5KCJnSQtGN/ai8eE93BDSPduDjZiFpbivH9vQl4Jz0S'
    b'kwKC8KuZD7HieQRu7wzFdTpeuKcb8OB7Z+F1yrWTufupL9jp8mIwVZqHrHpjdNWxw1BjDyxJCMas'
    b'xeOYoh2NPReO4Q90Fzoe1Mfd4lY4O3sGBh2OQi4XBzWtyTAcHA1RfCv4pD2EDXArIK+TlNHACgoZ'
    b'yOzPgdRng2Btr4XWnzFoAR74h6wAVLMPwQ56EIMTX/5+wxad/DXxwvF2mOJHgp+ujB2y+JCTfa5F'
    b'NlcaEC/XJs5m3+36/9dntFEGCfj+PRJZV8YtBnmwmftdAcaqIEF/AhL9lTBMYylmJitiV/U08Ndx'
    b'sHdADCbWVxlJsToJdjpHUoXDJHCZIuWrK9NQr2kyKuskhtUFr6U46ETnxRuoScwKeubhGHGK+JS0'
    b'xRoT99SNnEbOV4y7dC1btbWEaVLncaVKQtKx9Ty5Gf+UFC6spmVPXOmox2G66/Ip2nI5hjpKpdTH'
    b'IJDmD255LYF3EPvOsJzt6LnDVqyfZkcfz7BY/x27+61/sn3hS8Dh0Q6oHi2Fk72jcOqCIjpsUMCF'
    b'jDYQTSRCfOEm0DSdY/u+bmHl286xbd0se15wihEP1zAOp7JZ7qIiJCmZQHyoAFwcL7Djd99lW814'
    b'qKamgv3Jy7AlT4jPhRpIVNSx6o9qqPRAC9uXmuGdF46476kfRq49hkdoJAb1eOKTkzYoijRGWcXL'
    b'ez9VvC4XYO/OeeDZTkJ/yBRUGguw5ufluCNDD8WXlqOIU0Rz2QIkThPCrL5ELE2LyEo1GYkwzCcV'
    b'5gVkqquQZB/IIk47vMm03pec7vpy5plVB1t0uZTl+Q/Wp581JHfbpcTpwTnCiyshJ/5VTaQzTcRe'
    b'uYvA7Q4yzpMTuVsO2fpmAkn/IYe4dlSSnqJGMvcwF7y166Ct5VsYnRmDsKYFaFlQRZMYHXR/aoz8'
    b'J0ZY+WcD7HPTw4rkJegubQe3YYAjUWs4+1vbyHBwCrlrKyN2B4pJiXkx2fJy82pj58jIVBT5X/vs'
    b'/36SGU26xkrMhbA3FuEXg0DIvJ4Cf+kqARv5NaiorIWOqAoI8iuHhKEksIpRghreJU7s+RlxS54k'
    b'oV+r0jcsllNx0wr6wcWXm6vTot58TTp+hk+DOyXkVTMvbLibjrEMb+g0O5M2wvoOG8An29zB1TIa'
    b'QoSZMK+aDv29CdBAYiEz2BksZ0JZ/XoTMvfjP0jWGRXamvgmnX8kok5xb9O2oLdp+Akz+vebq2mJ'
    b'ZAUNOOpPXpWvbM0Jj8VwXieOcP73tDn7WYtN4lof9sgXD9hszhZWmyVAhBaB8Ns/QZCZEJ0j9bH/'
    b'sBAPLf0WVHycoTAwddOwqgohkxvJlIsCqZqT1H/n08deZS0h/IAeyHzy2LKUaubffYrWxdFvOk/T'
    b'RfY05dnFUquTJ+ma9RFUXbqf2v2yncYZIBWZraKNLSpUSX+EDMa2kv2J90j3ZR06kedKXTROUtO8'
    b'dGppnUvljWn0ra3RNMXfn35cvIs+7veizeJQmlQcRdMEkb+Z3uNJXc7uobbhuynzkYSWHHemapZ2'
    b'NOmoBY1YNKB1+srU5f0hcjWjjCwThxNfo3eJptlhMuV9g+Tm6dLNas40YU0Ytd8vpYZ/DaG9H7tR'
    b'B38rWnrRmK4aFFG+I0PNih2pyYDVb34Fpl0tleAHAAA='
    )), dtype='float32').reshape((504, 1))
  # ./returnn/tools/dump-dataset.py .../ref_att_weights.hdf --stdout_limit inf --stdout_as_bytes --endseq 0
  ref_att_weights_np = numpy.frombuffer(gzip.decompress(base64.decodebytes(
    b'H4sIAG/zb1wC/73Qe1SMeRzH8VZrmEpSKUmxOhuRMjXP8/t9f7/neWYT2dCN3CW1RKEoU0MuaVal'
    b'TVKnXeNaI9lKKSnlkpVLRbJKTg0yOOySbnJbuez2h3P2OPjTH69/P+9zPhKHZSSww40mLteCtZ4/'
    b'HOCcINVHAsVTAJrYcOhxyyOkLJFzTl3Nk8B6nnMxFJQrDwv1XbuEulwT3jGqD3Q2pzKLlQJTCDJm'
    b'z8m16Oh6d/Bb3IcNY0ZhyX/7/3fIvZPs695AjJTZeGT6ANy5qREFzshB0f3lqKRUH5Wq9iM3dzvo'
    b'15SNA49vw6KMJ5jdhnBwewXrn2JOwiqPICzdj/Ikd3FnQByhrU+AXXaM/HgtSlZmVMt93PvaShoM'
    b'Se7qcML7m4HCsRk7D87CjiOz8bAFybg16BK+WXmIts0VCTMeuQpcj0J409UsZJzPIs+NRiDbOAW7'
    b'JGcD+8OMG6gr1hhQnhXij/ni3s3PKRPr0jmzY4npXVN8Ex1B72pXoKfVwWjPQ4KMJhigoq4OrNFR'
    b'gXZGLLTPiSde6jVYv+QGja2PJBbzvSmt8iAdWyPhZb6S+KWGy3bqDOS/1PvansUryf4OSzq74A/w'
    b'PeEC7UcNgIm0A9sKHfgTm4A2Po1rrQRh9hVD2c6YcMH0lpoojrx0rtbZ7lwl3uz09x2FdJqjAasd'
    b'5QVo/EFmd9U5tnfzc8QF/30iMSATuMmIv1jHVizeziaHxLEeZt8xoltDpG5RBFXUavHCyJ2s3blV'
    b'KB9i0IrccGnx6YdMgPoJ+7xcTDJTxsBG+RUSOUkuMwkx5r/U+9o2txcSJ6Pd9PzEqaQj+VsywfMJ'
    b'rJSLCdennChfNdKlmWq+QrtdyJPXCHvEwwXDvi7885I7nOM6e5y5Si31dl0gLdbPZm6PT2Jq5m1k'
    b'F/uaMzFB4KT6ZaDT3pt5Es/kECfSqIv/UjU4xYvKmZSadZxNwgm6ticONsSMhhvMAAg7awnWEUq4'
    b'Pf4gcLMcoOJBOboaqIOaZl2DlX7Poe4BS/kj1Zx98BrByjZdcLC148dW9RXE1jbC1PepPJt9g5b7'
    b'1ZISfjMUSRfCqHIzepExoHvc/Ki/YScnjnDh3p0wJdY5J8Fbo4Lu9TmwKOEdXBpTCD7Xq7F77FJE'
    b'rxSwWY7X4MUFFtqaPcm0CxroH13F2g/imfL6VgjJdQVPF4zvJQiQnD+Q/CbaQWbuAhKRn0be6mmo'
    b'fuZS2ab31ryoIIyUK9zpaDmQKf2vw87S3+He3FDIeWwNPx2PpV5OW4UlxvsF77sBQrOOq9B8Wgs/'
    b'6wUyl05ZsFu2qFiTyWekipbXzhpPrdSYfc2mtBVgsxYlk/DSB/Vuf+x2vAe/1uIUPzhMyr2/k0FT'
    b'WuRU98x8mhHnS80ri6l7LaE2ZkfJ7mArKqlcTg/EZPJhxinCq0MThVtvB/D1tuOEgKZC7uzCKXSL'
    b'xwO6zD6R03usptpxDfRTPQt+ECe230qzBhmD+pkaN7okY8MVb1CDpS17PFQXKR3uIY1ePDYNvwwb'
    b'jcpgq2t/bOcvZ3pqpyNlrYjUZafCGsU6yqwPgmn5pmRwURtpLYqQ7brfzH2ql/tqGa1KiuaMyvrR'
    b'sVoVqY6RkwPdm8jUy/uI7Gk0rV6EhBPVNcLbx+ZCtvkGYWFfHRLY5gZWoZfBrWUHDPvGE1bXOiDD'
    b'ufPwpAnZ6NcsK+Rol8CU+uaji0O74WLGaBaFTse9nV4OBts45GPKtdYkQxJxhvQkS3izYhyYWIRC'
    b'4iIR6VprQtJ7LLB0hIhGlqwkh69oeNc5GUK3JEdwUDfwhz2S+PhjiVzUXg/u+wUCiZsuI+461tB4'
    b'K4IUFnTS9qIAGLO0hX7oeT2M4JKCztKZ9YMg4aoKi4qC8H3TYBxdmIDbJBOxJq2E/SfKixk+dDRK'
    b'a9/LOtY1srq3B0rrEpucQ/Y+YidbalBeTTvabjAPol/4oZSRDchmaAioFDZ0zJB79OmkMJlVuz7/'
    b'ofcvjNYYm+AHAAA='
    )), dtype='float32').reshape((504, 1))
  seq_sizes_np = numpy.frombuffer(gzip.decompress(base64.decodebytes(
    b'H4sIAK3yb1wC/2NiYGCQAGImIBZFopmBWAZKi0NpKSAGAN0FJ6owAAAA')), dtype='int32').reshape((6, 2))
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
      _energy_np = energy_np[n_batch_start:n_batch_end, :_max_seq_sizes_np[0], :_max_seq_sizes_np[1]]
      _ref_att_weights_np = ref_att_weights_np[n_batch_start:n_batch_end, :_max_seq_sizes_np[0], :_max_seq_sizes_np[1]]
      seq_sizes_tf = {i: tf.constant(_seq_sizes_np[:, i]) for i in range(_seq_sizes_np.shape[1])}
      energy_tf = tf.constant(_energy_np)
      ref_att_weights_tf = tf.constant(_ref_att_weights_np)
      energy_data = Data(
        name="energy", shape=(None, None, n_extra_dim), batch_dim_axis=0,
        placeholder=energy_tf, size_placeholder=seq_sizes_tf)
      ref_att_weights_data = Data(
        name="ref_att_weights", shape=(None, None, n_extra_dim), batch_dim_axis=0,
        placeholder=ref_att_weights_tf, size_placeholder=seq_sizes_tf)
      res_tf = softmax_cross_entropy_over_size(logits=energy_data, labels=ref_att_weights_data)
      res_tf.set_shape((new_n_batch, _max_seq_sizes_np[0], n_extra_dim))
      res_np = session.run(res_tf)
      print("res:", res_np)
      assert numpy.alltrue(numpy.isfinite(res_np))


def test_softmax_cross_entropy_over_size_small_batch_2():
  import Util
  rnd = numpy.random.RandomState(42)
  n_batch = 2
  n_extra_dim = 1
  dec_seq_lens = [2, 2]
  enc_seq_lens = [4, 3]
  energy_np = rnd.normal(size=(n_batch, max(dec_seq_lens), max(enc_seq_lens), n_extra_dim)).astype("float32")
  ref_att_weights_np = rnd.normal(size=(n_batch, max(dec_seq_lens), max(enc_seq_lens), n_extra_dim)).astype("float32")
  for i in range(n_batch):
    ref_att_weights_np[i, :dec_seq_lens[i], :enc_seq_lens[i]] = Util.softmax(
      ref_att_weights_np[i, :dec_seq_lens[i], :enc_seq_lens[i]], axis=1)
    ref_att_weights_np[i, dec_seq_lens[i]:] = 0
    ref_att_weights_np[i, :dec_seq_lens[i], enc_seq_lens[i]:] = 0
  sizes_tf = {0: tf.constant(dec_seq_lens), 1: tf.constant(enc_seq_lens)}
  energy_tf = tf.constant(energy_np)
  ref_att_weights_tf = tf.constant(ref_att_weights_np)
  energy_data = Data(
    name="energy", shape=(None, None, n_extra_dim), batch_dim_axis=0,
    placeholder=energy_tf, size_placeholder=sizes_tf)
  ref_att_weights_data = Data(
    name="ref_att_weights", shape=(None, None, n_extra_dim), batch_dim_axis=0,
    placeholder=ref_att_weights_tf, size_placeholder=sizes_tf)
  res_tf = softmax_cross_entropy_over_size(logits=energy_data, labels=ref_att_weights_data)
  res_tf.set_shape((n_batch, max(dec_seq_lens), n_extra_dim))
  res_np = session.run(res_tf)
  print("res:", res_np)
  assert numpy.alltrue(numpy.isfinite(res_np))


def test_softmax_cross_entropy_over_size_gradient():
  n_batch = 2
  n_dec_time = n_enc_time = 10
  n_extra_dim = 1
  tf.set_random_seed(42)
  energy_tf = tf.get_variable(
    "test_softmax_cross_entropy_over_size_gradient_var",
    shape=(n_batch, n_dec_time, n_enc_time, n_extra_dim),
    initializer=tf.random_normal_initializer(seed=23))
  ref_att_weights_tf = tf.reshape(
    tf.one_hot(tf.range(n_dec_time, dtype=tf.int32), n_enc_time, dtype=tf.float32),
    (1, n_dec_time, n_enc_time, n_extra_dim))
  ref_att_weights_tf = tf.tile(ref_att_weights_tf, [n_batch, 1, 1, 1])
  ref_att_weights_tf.set_shape((n_batch, n_dec_time, n_enc_time, n_extra_dim))
  sizes = {0: [n_dec_time, n_dec_time - 1], 1: [n_enc_time, n_enc_time - 1]}
  sizes_tf = {i: tf.constant(size) for (i, size) in sizes.items()}
  energy_data = Data(
    name="energy", shape=(None, None, n_extra_dim), batch_dim_axis=0,
    placeholder=energy_tf, size_placeholder=sizes_tf)
  ref_att_weights_data = Data(
    name="ref_att_weights", shape=(None, None, n_extra_dim), batch_dim_axis=0,
    placeholder=ref_att_weights_tf, size_placeholder=sizes_tf)
  for stable_gradient in [False, True]:
    res_tf = softmax_cross_entropy_over_size(
      logits=energy_data, labels=ref_att_weights_data, stable_gradient=stable_gradient)
    res_tf.set_shape((n_batch, n_dec_time, n_extra_dim))
    res_flat_tf = flatten_with_seq_len_mask(res_tf, sizes_tf[0], batch_dim_axis=0, time_dim_axis=1)
    res_flat_tf.set_shape((sum(sizes[0]), n_extra_dim))
    loss_tf = tf.reduce_mean(res_tf)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e2)
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
    #if len(list(threading.enumerate())) > 1:
    #  print("Warning, more than one thread at exit:")
    #  better_exchook.dump_all_thread_tracebacks()
