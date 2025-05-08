# start: nosetests $this_file --nologcapture
from __future__ import annotations

import _setup_test_env  # noqa
import tensorflow as tf
import unittest
import numpy.testing
import tempfile
from pprint import pprint
from returnn.util import better_exchook
from returnn.config import Config
from returnn.tf.network import *
from returnn.tf.layers.basic import *
from returnn.tf.layers.variable import *
import returnn.tf.compat as tf_compat
import returnn.tf.util.basic as tf_util
from returnn.tf.util.data import Dim, SpatialDim, FeatureDim, BatchInfo

print("TF version:", tf.__version__)
print("Numpy version:", numpy.__version__)


@contextlib.contextmanager
def make_scope():
    """
    :rtype: tf.compat.v1.Session
    """
    with tf.Graph().as_default() as graph:
        with tf_compat.v1.Session(graph=graph) as session:
            yield session


def make_feed_dict(data_list, same_time=False, n_batch=3, n_time=7):
    """
    :param list[returnn.tensor.Tensor]|ExternData data_list:
    :param bool same_time:
    :param int n_batch:
    :param int n_time:
    :rtype: dict[tf.Tensor,numpy.ndarray]
    """
    from returnn.tensor import batch_dim
    from returnn.util.basic import RefIdEq

    if isinstance(data_list, ExternData):
        data_list = [value for (key, value) in sorted(data_list.data.items())]
    assert n_time > 0 and n_batch > 0
    rnd = numpy.random.RandomState(42)
    existing_sizes = {}  # type: typing.Dict[RefIdEq[tf.Tensor],int]
    d = {}
    batch_info = None
    for data in data_list:
        if data.batch and not batch_info:
            batch_info = data.batch
    for data in data_list:
        if data.batch and not batch_info:
            batch_info = data.batch
        shape = list(data.batch_shape)
        if data.batch_dim_axis is not None:
            shape[data.batch_dim_axis] = n_batch
        for axis, dim in enumerate(shape):
            if dim is None:
                tag: Dim = data.dims[axis]
                dyn_size = tag.dyn_size
                if RefIdEq(dyn_size) in existing_sizes:
                    shape[axis] = existing_sizes[RefIdEq(dyn_size)]
                    continue
                existing_sizes[RefIdEq(dyn_size)] = n_time
                shape[axis] = n_time
                if tag.dyn_size_ext.dims == (batch_dim,):
                    dyn_size_v = numpy.array([n_time, max(n_time - 2, 1), max(n_time - 3, 1)])
                    if dyn_size_v.shape[0] > n_batch:
                        dyn_size_v = dyn_size_v[:n_batch]
                    elif dyn_size_v.shape[0] < n_batch:
                        dyn_size_v = numpy.concatenate(
                            [dyn_size_v, rnd.randint(1, n_time + 1, size=(n_batch - dyn_size_v.shape[0],))], axis=0
                        )
                elif tag.dyn_size_ext.dims == ():  # scalar
                    dyn_size_v = numpy.array(n_time)
                else:
                    raise NotImplementedError(f"tag {tag} with dyn_size_ext: {tag.dyn_size_ext}")
                d[dyn_size] = dyn_size_v
                if not same_time:
                    n_time += 1
        print("%r %r: shape %r" % (data, data.placeholder, shape))
        if data.sparse:
            d[data.placeholder] = rnd.randint(0, data.dim or 13, size=shape, dtype=data.dtype)
        else:
            d[data.placeholder] = rnd.normal(size=shape).astype(data.dtype)
    if batch_info:
        batch_dim = batch_info.dim
        if isinstance(batch_dim, int):
            assert batch_dim == n_batch, "invalid batch info %r" % batch_info
        else:
            assert isinstance(batch_dim, tf.Tensor)
            d[batch_dim] = n_batch
    return d


def test_ExternData_init_from_config():
    config = Config(
        {
            "extern_data": {"data": {"dim": 42}},
        }
    )
    extern_data = ExternData()
    with make_scope() as session:
        extern_data.init_from_config(config)
        data = extern_data.data["data"]
        assert data.batch_shape == (None, None, 42)
        assert (data.batch_dim_axis, data.time_dim_axis, data.feature_dim_axis) == (0, 1, 2)


def test_ExternData_init_from_config_dim_none():
    config = Config(
        {
            "extern_data": {"data": {"dim": None}},
        }
    )
    extern_data = ExternData()
    with make_scope() as session:
        extern_data.init_from_config(config)
        data = extern_data.data["data"]
        assert data.batch_shape == (None, None, None)
        assert (data.batch_dim_axis, data.time_dim_axis, data.feature_dim_axis) == (0, 1, 2)


def test_ExternData_init_twice_existing_dim_tags():
    from returnn.tf.util.data import batch_dim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feature", dimension=10)
    config = Config({"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}}})  # [B,T,D]
    for _ in range(2):
        with make_scope() as session:
            net = TFNetwork(config=config)
            net.construct_from_dict({"output": {"class": "softmax_over_spatial", "from": "data"}})
            session.run(net.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_LinearLayer():
    from returnn.tf.util.data import batch_dim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feature", dimension=5)
    config = Config({"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}}})  # [B,T,D]
    for _ in range(2):
        with make_scope() as session:
            net = TFNetwork(config=config)
            net.construct_from_dict({"output": {"class": "linear", "from": "data", "n_out": 3}})
            session.run(tf_compat.v1.global_variables_initializer())
            session.run(net.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_LinearLayer_in_dim_spatial():
    from returnn.tf.util.data import batch_dim

    time_dim = SpatialDim("time")
    static_spatial_dim = FeatureDim("static-spatial", dimension=3)
    feat_dim = FeatureDim("in-feature", dimension=5)
    out_dim = FeatureDim("out-feature", dimension=7)
    config = Config(
        {"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, static_spatial_dim, feat_dim]}}}  # [B,T,D1,D2]
    )
    for _ in range(2):
        with make_scope() as session:
            net = TFNetwork(config=config)
            net.construct_from_dict(
                {"output": {"class": "linear", "from": "data", "in_dim": static_spatial_dim, "out_dim": out_dim}}
            )
            layer = net.get_default_output_layer()
            print("Output:", layer.output)
            assert layer.output.dim_tags_set_implicit == {batch_dim, time_dim, out_dim, feat_dim}
            param = layer.params["W"]
            assert isinstance(param, tf.Variable)
            assert param.shape.as_list() == [static_spatial_dim.dimension, out_dim.dimension]
            session.run(tf_compat.v1.global_variables_initializer())
            session.run(layer.output.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_LinearLayer_two_time_dims_allow_broadcast_all_sources():
    from returnn.tf.util.data import batch_dim

    with make_scope() as session:
        time1_dim = SpatialDim("time1")
        time2_dim = SpatialDim("time2")
        feat_dim = FeatureDim("feature", dimension=5)
        out_dim = FeatureDim("feature", dimension=3)
        config = Config(
            {
                "extern_data": {
                    "in1": {"dim_tags": [batch_dim, time1_dim, feat_dim]},
                    "in2": {"dim_tags": [batch_dim, time2_dim, feat_dim]},
                },
            }
        )
        network = TFNetwork(config=config)
        try:
            network.construct_from_dict({"output": {"class": "linear", "from": ["data:in1", "data:in2"], "n_out": 3}})
        except Exception as exc:
            # https://github.com/rwth-i6/returnn/issues/691
            print("Expected exception:", exc)
            assert "require broadcasting" in str(exc)
        else:
            raise Exception(
                "Expect allow_broadcast_all_sources exception, but layer constructed: %s"
                % network.get_default_output_layer()
            )
        network.construct_from_dict(
            {
                "output": {
                    "class": "linear",
                    "from": ["data:in1", "data:in2"],
                    "out_dim": out_dim,
                    "out_shape": {batch_dim, time1_dim, time2_dim, out_dim},
                }
            }
        )
        output = network.get_default_output_layer().output
        assert output.shape == (None, None, 3)
        session.run(tf_compat.v1.global_variables_initializer())
        session.run(fetches=output.placeholder, feed_dict=make_feed_dict(network.extern_data))


def test_LinearLayer_generic_dim_tags():
    from returnn.tf.util.data import batch_dim

    with make_scope() as session:
        time1_dim = SpatialDim("time1")
        time2_dim = SpatialDim("time2", dimension=7)
        feat_dim = FeatureDim("feature", dimension=5)
        out_dim = FeatureDim("feature", dimension=3)
        config = Config(
            {
                "extern_data": {
                    "in1": {"dim_tags": [batch_dim, time1_dim, time2_dim, feat_dim]},
                    "in2": {"dim_tags": [batch_dim, time2_dim, feat_dim]},
                },
            }
        )
        network = TFNetwork(config=config)
        network.construct_from_dict(
            {
                "output1": {
                    "class": "linear",
                    "from": "data:in1",
                    "out_dim": out_dim,
                    "out_shape": {batch_dim, time1_dim, time2_dim, out_dim},
                    "is_output_layer": True,
                }
            }
        )
        network.construct_from_dict(
            {
                "output2": {
                    "class": "linear",
                    "from": "data:in1",
                    "in_dim": time2_dim,
                    "out_dim": out_dim,
                    "out_shape": {batch_dim, time1_dim, out_dim, feat_dim},
                    "is_output_layer": True,
                }
            }
        )
        network.construct_from_dict(
            {
                "output4": {
                    "class": "linear",
                    "from": "data:in2",
                    "in_dim": time2_dim,
                    "out_dim": out_dim,
                    "out_shape": {batch_dim, out_dim, feat_dim},
                    "is_output_layer": True,
                }
            }
        )
        session.run(tf_compat.v1.global_variables_initializer())
        session.run(
            fetches=[layer.output.placeholder for layer in network.get_output_layers()],
            feed_dict=make_feed_dict(network.extern_data),
        )


def test_LinearLayer_reuse_params_layer_output():
    from returnn.tf.util.data import batch_dim

    with make_scope() as session:
        time_dim = SpatialDim("time")
        data_feat_dim = FeatureDim("feature", dimension=5)
        out_feat_dim = FeatureDim("feature", dimension=7)
        config = Config(
            {
                "extern_data": {
                    "data": {"dim_tags": [batch_dim, time_dim, data_feat_dim]},
                },
            }
        )
        network = TFNetwork(config=config)
        network.construct_from_dict(
            {
                "weights": {"class": "variable", "shape": [data_feat_dim, out_feat_dim]},
                "bias": {"class": "variable", "shape": [out_feat_dim]},
                "out1": {
                    "class": "linear",
                    "from": "data",
                    "out_dim": out_feat_dim,
                    "is_output_layer": True,
                    "reuse_params": {"map": {"W": {"layer_output": "weights"}, "b": {"layer_output": "bias"}}},
                },
                "out2_": {
                    "class": "dot",
                    "from": ["data", "weights"],
                    "red1": data_feat_dim,
                    "red2": data_feat_dim,
                    "var1": [batch_dim, time_dim],
                    "var2": out_feat_dim,
                },
                "out2": {"class": "combine", "kind": "add", "from": ["out2_", "bias"], "is_output_layer": True},
            }
        )
        out1 = network.get_layer("out1").output
        out2 = network.get_layer("out2").output
        params = network.get_params_list()
        assert len(params) == 2  # weights and bias
        session.run(tf_compat.v1.global_variables_initializer())
        out1_np, out2_np = session.run(
            fetches=(out1.placeholder, out2.placeholder), feed_dict=make_feed_dict(network.extern_data)
        )
        numpy.testing.assert_array_equal(out1_np, out2_np)


def test_CopyLayer_target():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feature", dimension=5)
    config = Config(
        {
            "extern_data": {
                "data": {"dim_tags": [batch_dim, time_dim], "sparse_dim": feat_dim},
            },
        }
    )
    net_dict = {"output": {"class": "copy", "from": "data", "target": "data"}}
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        out = net.get_default_output_layer().output
        assert out.sparse_dim == feat_dim
        session.run(out.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_PadLayer_time():
    n_batch, n_time, n_in = 7, 3, 20
    config = Config(
        {
            "extern_data": {"data": {"dim": n_in}},
            "debug_print_layer_output_template": True,
        }
    )
    with make_scope() as session:
        padding = (2, 3)
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {
                "output": {
                    "class": "pad",
                    "axes": "T",
                    "padding": padding,
                    "handle_dynamic_dims": False,  # our test below does not handle dyn seq lens
                    "mode": "replication",
                    "from": "data:data",
                }
            }
        )
        out_t = net.get_default_output_layer().output.placeholder
        assert out_t.shape.as_list() == [None, None, n_in]
        in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_time, n_in))
        out_v = session.run(out_t, feed_dict={net.extern_data.data["data"].placeholder: in_v})
        assert isinstance(out_v, numpy.ndarray)
        assert out_v.shape == (n_batch, n_time + sum(padding), n_in)
        assert (out_v[:, 0, :] == out_v[:, padding[0], :]).all()
        assert (out_v[:, -1, :] == out_v[:, -1 - padding[1], :]).all()
        numpy.testing.assert_array_equal(in_v, out_v[:, padding[0] : (-padding[1] or None), :])
        # check padding on left
        if padding[0] > 0:
            padded_left_ref = numpy.resize(in_v[:, 0, :], (padding[0], n_batch, n_in)).transpose(1, 0, 2)
            numpy.testing.assert_array_equal(padded_left_ref, out_v[:, : padding[0], :])
        # check padding on right
        if padding[1] > 0:
            padded_right_ref = numpy.resize(in_v[:, -1, :], (padding[1], n_batch, n_in)).transpose(1, 0, 2)
            numpy.testing.assert_array_equal(padded_right_ref, out_v[:, -padding[1] :, :])


def test_PadLayer_feature():
    n_batch, n_time, n_in = 7, 3, 20
    config = Config(
        {
            "extern_data": {"data": {"dim": None}},
            "debug_print_layer_output_template": True,
        }
    )
    with make_scope() as session:
        padding = (2, 3)
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {
                "output": {
                    "class": "pad",
                    "axes": "F",
                    "padding": padding,
                    "handle_dynamic_dims": False,  # our test below does not handle dyn seq lens
                    "mode": "replication",
                    "from": "data:data",
                }
            }
        )
        out_t = net.get_default_output_layer().output.placeholder
        assert out_t.shape.as_list() == [None, None, None]
        in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_time, n_in))
        out_v = session.run(out_t, feed_dict={net.extern_data.data["data"].placeholder: in_v})
        assert isinstance(out_v, numpy.ndarray)
        assert out_v.shape == (n_batch, n_time, n_in + sum(padding))
        assert (out_v[:, :, 0] == out_v[:, :, padding[0]]).all()
        assert (out_v[:, :, -1] == out_v[:, :, -1 - padding[1]]).all()
        numpy.testing.assert_array_equal(in_v, out_v[:, :, padding[0] : (-padding[1] or None)])
        # check padding on left
        if padding[0] > 0:
            padded_left_ref = numpy.resize(in_v[:, :, 0], (n_batch, n_time, 1))
            numpy.testing.assert_array_equal(padded_left_ref - out_v[:, :, : padding[0]], 0)
        # check padding on right
        if padding[1] > 0:
            padded_left_ref = numpy.resize(in_v[:, :, -1], (n_batch, n_time, 1))
            numpy.testing.assert_array_equal(padded_left_ref - out_v[:, :, -padding[1] :], 0)


def test_PadLayer_no_op():
    # https://github.com/rwth-i6/returnn/issues/687
    n_batch, n_time, n_in = 7, 3, 5
    config = Config(
        {
            "extern_data": {"data": {"shape": (n_in, None)}},  # [B,D,T]
            "debug_print_layer_output_template": True,
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {
                "output": {
                    "class": "pad",
                    "mode": "constant",
                    "axes": "spatial",
                    "padding": [(0, 0)],
                    "from": "data",
                    "value": 0,
                }
            }
        )
        out = net.get_default_output_layer().output
        out_t = out.placeholder
        assert out_t.shape.as_list() == [None, n_in, None]
        in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_in, n_time))
        out_v = session.run(out_t, feed_dict={net.extern_data.data["data"].placeholder: in_v})
        assert isinstance(out_v, numpy.ndarray)
        assert out_v.shape == (n_batch, n_in, n_time)
        numpy.testing.assert_array_equal(in_v, out_v)


def test_PadLayer_window():
    # https://github.com/rwth-i6/returnn/issues/1224
    from returnn.config import Config
    from returnn.tf.engine import Engine
    from returnn.datasets import init_dataset
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    in_dim = FeatureDim("in", 3)
    out_dim = FeatureDim("out", 4)

    def _config_get_network(epoch, **_kwargs):
        window_dim = SpatialDim("window", 3)
        time_dim_ = (window_dim // 2) + time_dim + window_dim.ceildiv_right(2) + (-1)
        flat_dim = window_dim * time_dim_
        flat_dim_ = flat_dim + window_dim
        time_window_dim = time_dim + window_dim

        # This is what PadLayer.get_out_data_from_opts() does.
        # In returnn-common, we would execute that.
        # This should be fine. But this triggers the bug.
        time_dim__ = 1 + time_dim + 1
        time_dim__.declare_same_as(time_dim_)
        flat_dim__ = 0 + flat_dim + 3
        flat_dim__.declare_same_as(flat_dim_)

        net_dict = {
            "#epoch": epoch,  # trigger reinit
            "window": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "pad": {
                        "class": "pad",
                        "from": "base:data:data",
                        "axes": time_dim,
                        "padding": (1, 1),
                        "out_dims": time_dim_,
                        "out_shape": {batch_dim, in_dim, time_dim_},
                    },
                    "expand_dim": {
                        "class": "expand_dims",
                        "from": "pad",
                        "axis": "spatial",
                        "dim": window_dim,
                        "out_shape": {batch_dim, in_dim, window_dim, time_dim_},
                    },
                    "merge_dims": {
                        "class": "merge_dims",
                        "from": "expand_dim",
                        "axes": (window_dim, time_dim_),
                        "out_dim": flat_dim,
                        "out_shape": {batch_dim, in_dim, flat_dim},
                    },
                    "pad_0": {
                        "class": "pad",
                        "from": "merge_dims",
                        "axes": flat_dim,
                        "padding": (0, 3),
                        "out_dims": flat_dim_,
                        "out_shape": {batch_dim, in_dim, flat_dim_},
                    },
                    "reshape": {
                        "class": "reshape",
                        "from": "pad_0",
                        "in_dims": [flat_dim_],
                        "out_dims": [window_dim, time_window_dim],
                        "extra_deps": ["base:data:data"],
                        "out_shape": {batch_dim, in_dim, window_dim, time_window_dim},
                    },
                    "slice_nd": {
                        "class": "slice_nd",
                        "from": "reshape",
                        "size": time_dim,
                        "axis": time_window_dim,
                        "out_spatial_dim": time_dim,
                        "out_shape": {batch_dim, time_dim, in_dim, window_dim},
                    },
                    "output": {
                        "class": "copy",
                        "from": "slice_nd",
                        "out_shape": {batch_dim, time_dim, in_dim, window_dim},
                    },
                },
            },
            "reduce": {
                "class": "reduce",
                "from": "window",
                "mode": "mean",
                "axis": (window_dim, in_dim),
                "out_shape": {batch_dim, time_dim},
            },
            "add": {
                "class": "combine",
                "from": ["dot", "reduce"],
                "kind": "add",
                "is_output_layer": True,
                "out_shape": {batch_dim, time_dim, out_dim},
            },
            "reduce_0": {
                "class": "reduce",
                "from": "add",
                "mode": "mean",
                "axis": out_dim,
                "out_shape": {batch_dim, time_dim},
            },
            "dummy": {"class": "copy", "from": "reduce_0", "loss": "as_is", "out_shape": {batch_dim, time_dim}},
            "weight": {
                "class": "variable",
                "shape": [in_dim, out_dim],
                "param_name": "param",
            },
            "dot": {
                "class": "dot",
                "from": ["data:data", "weight"],
                "reduce": in_dim,
                "out_shape": {batch_dim, time_dim, out_dim},
            },
        }
        return net_dict

    config = Config(
        {
            "task": "train",
            "num_epochs": 2,
            "start_epoch": 1,
            "get_network": _config_get_network,
            "extern_data": {"data": {"dim_tags": (batch_dim, time_dim, in_dim)}},
        }
    )
    train_dataset = init_dataset(
        {"class": "DummyDataset", "input_dim": in_dim.dimension, "output_dim": 5, "num_seqs": 3}
    )
    engine = Engine(config)
    engine.init_train_from_config(config, train_data=train_dataset)
    engine.train()


def test_concat_sources():
    with make_scope() as session:
        network = TFNetwork(train_flag=True, extern_data=ExternData())
        n_batch = 5
        n_time = 3
        size_placeholder = {0: tf.constant(n_time, dtype=tf.int32, shape=(n_batch,))}
        src1 = InternalLayer(
            name="src1",
            network=network,
            output=Data(
                name="src1_output",
                shape=(None, 11),
                placeholder=tf.zeros((n_batch, n_time, 11)),
                size_placeholder=size_placeholder,
            ),
        )
        print("src1 output:", src1.output)
        src2 = InternalLayer(
            name="src2",
            network=network,
            output=Data(
                name="src2_output",
                shape=(None, 13),
                placeholder=tf.zeros((n_batch, n_time, 13)),
                size_placeholder=size_placeholder,
            ),
        )
        print("src2 output:", src2.output)
        out_kwargs = dict(name="out", sources=[src1, src2], network=network)
        out_output = CopyLayer.get_out_data_from_opts(**out_kwargs)
        print("out output:", out_output)
        assert out_output.dim == 11 + 13
        out = CopyLayer(output=out_output, **out_kwargs)
        session.run(out.output.placeholder)


def test_concat_sources_batch_dim():
    with make_scope() as session:
        network = TFNetwork(train_flag=True, extern_data=ExternData())
        n_batch = 5
        n_time = 3
        size_placeholder = {0: tf.constant(n_time, dtype=tf.int32, shape=(n_batch,))}
        src1 = InternalLayer(
            name="src1",
            network=network,
            output=Data(
                name="src1_output",
                shape=(None, 11),
                placeholder=tf.zeros((n_batch, n_time, 11)),
                size_placeholder=size_placeholder,
            ),
        )
        print("src1 output:", src1.output)
        src2 = InternalLayer(
            name="src2",
            network=network,
            output=Data(
                name="src2_output",
                shape=(None, 13),
                time_dim_axis=0,
                batch_dim_axis=1,
                placeholder=tf.zeros((n_time, n_batch, 13)),
                size_placeholder=size_placeholder,
            ),
        )
        print("src2 output:", src2.output)
        out_kwargs = dict(name="out", sources=[src1, src2], network=network)
        out_output = CopyLayer.get_out_data_from_opts(**out_kwargs)
        print("out output:", out_output)
        assert out_output.dim == 11 + 13
        assert out_output.batch_dim_axis == 0 and out_output.time_dim_axis == 1
        out = CopyLayer(output=out_output, **out_kwargs)
        session.run(out.output.placeholder)


def test_concat_sources_missing_dim():
    with make_scope() as session:
        network = TFNetwork(train_flag=True, extern_data=ExternData())
        n_batch = 5
        n_time = 3
        size_placeholder = {0: tf.constant(n_time, dtype=tf.int32, shape=(n_batch,))}
        src1 = InternalLayer(
            name="src1",
            network=network,
            output=Data(
                name="src1_output",
                shape=(None, 11),
                placeholder=tf.zeros((n_batch, n_time, 11)),
                size_placeholder=size_placeholder,
            ),
        )
        print("src1 output:", src1.output)
        src2 = InternalLayer(
            name="src2",
            network=network,
            output=Data(
                name="src2_output",
                shape=(13,),
                time_dim_axis=None,
                batch_dim_axis=0,
                placeholder=tf.zeros((n_batch, 13)),
                size_placeholder={},
            ),
        )
        print("src2 output:", src2.output)
        out_kwargs = dict(name="out", sources=[src1, src2], network=network)
        out_output = CopyLayer.get_out_data_from_opts(**out_kwargs)
        print("out output:", out_output)
        assert out_output.dim == 11 + 13
        assert out_output.batch_dim_axis == 0 and out_output.time_dim_axis == 1
        out = CopyLayer(output=out_output, **out_kwargs)
        session.run(out.output.placeholder)


def test_concat_sources_dim1():
    with make_scope() as session:
        net_dict = {
            "lin1": {"class": "linear", "activation": "sigmoid", "n_out": 5, "from": "data:data"},
            "lin2": {"class": "linear", "activation": "sigmoid", "n_out": 1, "from": "data:data"},
            "concat": {"class": "copy", "from": ["lin1", "lin2"]},
            "output": {"class": "softmax", "loss": "ce", "from": "concat"},
        }
        config = Config({"debug_print_layer_output_template": True})
        config.update(dict(num_inputs=4, num_outputs=9))
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        assert network.get_layer("concat").output.shape == (None, 6)
        out = network.get_default_output_layer()
        assert out.output.shape == (None, 9)
        feed_dict = make_feed_dict(network.extern_data.data.values(), same_time=True)
        session.run(tf_compat.v1.global_variables_initializer())
        session.run(out.output.placeholder, feed_dict=feed_dict)


def test_concat_new_dim_tag():
    from returnn.tf.util.data import Dim

    with make_scope():
        n_out = 5
        time_tag = Dim(kind=Dim.Types.Spatial, description="time", dimension=None)
        new_time_tag = Dim(kind=Dim.Types.Spatial, description="new-time", dimension=None)
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "data": {"dim": n_out, "same_dim_tags_as": {"t": time_tag}},
                    "classes": {"dim": n_out, "sparse": True, "same_dim_tags_as": {"t": time_tag}},
                },
            }
        )
        net = TFNetwork(config=config, search_flag=True)
        net.construct_from_dict(
            {
                "data_new": {"class": "reinterpret_data", "from": "data", "set_dim_tags": {"t": new_time_tag}},
                "output": {
                    "class": "rec",
                    "from": "data",
                    "unit": {
                        "prev_out0": {"class": "reinterpret_data", "from": "prev:output", "set_sparse": False},
                        "prev_out1": {"class": "cast", "from": "prev_out0", "dtype": "float32"},
                        "prev_out": {"class": "expand_dims", "from": "prev_out1", "axis": "f"},
                        "data_concat": {"class": "copy", "from": ["base:data_new", "prev_out"]},
                        "data_red": {"class": "reduce", "from": "data_concat", "axis": "stag:new-time", "mode": "max"},
                        "output_prob": {"class": "softmax", "from": "data_red", "target": "classes", "loss": "ce"},
                        "output": {
                            "class": "choice",
                            "from": "output_prob",
                            "beam_size": 3,
                            "target": "classes",
                            "input_type": "prob",
                            "initial_output": 0,
                        },
                    },
                },
            }
        )


def test_ConcatLayer():
    with make_scope() as session:
        net_dict = {
            "lin1": {"class": "linear", "activation": "sigmoid", "n_out": 5, "from": "data:data"},
            "lin2": {"class": "linear", "activation": "sigmoid", "n_out": 3, "from": "data:data"},
            "output": {"class": "concat", "from": [("lin1", "F"), ("lin2", "F"), ("data", "F")]},
        }
        config = Config({"extern_data": {"data": {"dim": 2}}})
        network = TFNetwork(config=config)
        network.construct_from_dict(net_dict)
        out = network.get_default_output_layer()
        assert out.output.shape == (None, 10)
        feed_dict = make_feed_dict(network.extern_data, same_time=True)
        session.run(tf_compat.v1.global_variables_initializer())
        session.run(out.output.placeholder, feed_dict=feed_dict)


def test_ConcatLayer_range_dyn():
    with make_scope() as session:
        net_dict = {
            "range": {"class": "range_in_axis", "from": "data:data", "axis": "T"},
            "output": {"class": "concat", "from": [("range", "T"), ("range", "T")]},
        }
        config = Config({"extern_data": {"data": {"dim": 2}}})
        network = TFNetwork(config=config)
        network.construct_from_dict(net_dict)
        out = network.get_default_output_layer().output
        assert out.batch_shape == (None,)
        feed_dict = make_feed_dict(network.extern_data, n_time=7)
        session.run(out.placeholder, feed_dict=feed_dict)
        assert session.run(out.dim_tags[0].get_dim_value(), feed_dict=feed_dict) == 14


def test_LinearLayer_batch_feature_major():
    with make_scope() as session:
        network = TFNetwork(config=Config(), extern_data=ExternData(), train_flag=True)
        n_in = 3
        n_out = 7
        source = InternalLayer(
            name="source",
            network=network,
            output=Data(name="source", shape=(n_in, None), time_dim_axis=2, auto_create_placeholders=True),
        )
        assert source.output.feature_dim_axis == 1
        assert source.output.is_batch_feature_major
        out_template = LinearLayer.get_out_data_from_opts(
            name="lin", network=network, n_out=n_out, activation=None, sources=[source]
        )
        out_template.sanity_check()
        assert out_template.shape == (n_out, None) and (out_template.feature_dim_axis, out_template.time_dim_axis) == (
            1,
            2,
        )
        assert out_template.is_batch_feature_major
        with tf_compat.v1.variable_scope("lin"):
            layer = LinearLayer(
                name="lin", network=network, n_out=n_out, activation=None, sources=[source], output=out_template
            )
        layer.output.sanity_check()
        n_batch = 5
        n_times = [13, 13, 11, 7, 5]
        assert len(n_times) == n_batch
        n_time = max(n_times)
        feed_dict = {
            source.output.placeholder: numpy.random.normal(size=(n_batch, n_in, n_time)).astype("float32"),
            source.output.size_placeholder[1]: numpy.array(n_times, dtype="int32"),
        }
        session.run(tf_compat.v1.global_variables_initializer())
        session.run(layer.output.placeholder, feed_dict=feed_dict)


def test_batch_norm_vars():
    with make_scope() as session:
        n_in, n_out = 2, 3
        config = Config()
        layer_name = "layer1"
        config.update(
            {
                "num_outputs": n_out,
                "num_inputs": n_in,
                "network": {
                    layer_name: {
                        "class": "linear",
                        "activation": "relu",
                        "batch_norm": {"masked_time": True},
                        "n_out": n_out,
                        "is_output_layer": True,
                        "from": "data:data",
                    }
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_value("network"))
        layer = network.layers[layer_name]
        print("layer:", layer)
        print("layer vars:")
        pprint(layer.params)
        assert layer.use_batch_norm
        bn_prefix = "batch_norm/v2_"
        assert set(layer.params.keys()) == {
            "W",
            "b",
            bn_prefix + "beta",
            bn_prefix + "mean",
            bn_prefix + "gamma",
            bn_prefix + "variance",
        }
        assert layer.params["W"].get_shape().as_list() == [n_in, n_out]
        assert layer.params["b"].get_shape().as_list() == [n_out]
        assert layer.params[bn_prefix + "beta"].get_shape().as_list() == [n_out]
        assert layer.params[bn_prefix + "gamma"].get_shape().as_list() == [n_out]
        assert layer.params[bn_prefix + "mean"].get_shape().as_list() == [n_out]
        assert layer.params[bn_prefix + "variance"].get_shape().as_list() == [n_out]


def _test_batch_norm_param_old_to_new_import(old_version, new_version):
    import tempfile

    model_tmp_dir = tempfile.mkdtemp("tmp-checkpoint")
    model_filename = model_tmp_dir + "/model"
    layer_name = "layer1"
    n_in = 3

    def _make_net_dict(param_version):
        return {
            layer_name: {
                "class": "batch_norm",
                "from": "data:data",
                "is_output_layer": True,
                "param_version": param_version,
                "masked_time": True,
            }
        }

    with make_scope() as session:
        config = Config({"extern_data": {"data": {"dim": n_in}}})
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(_make_net_dict(param_version=old_version))
        network.initialize_params(session)
        network.save_params_to_file(filename=model_filename, session=session)
        out_ref = session.run(
            network.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(network.extern_data)
        )
        assert isinstance(out_ref, numpy.ndarray)
        assert not numpy.allclose(out_ref, 0.0)

    with make_scope() as session:
        config = Config({"extern_data": {"data": {"dim": n_in}}})
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(_make_net_dict(param_version=new_version))
        network.load_params_from_file(filename=model_filename, session=session)
        out_new = session.run(
            network.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(network.extern_data)
        )
        assert isinstance(out_new, numpy.ndarray)
        assert not numpy.allclose(out_new, 0.0)
        numpy.testing.assert_allclose(out_ref, out_new)


def test_batch_norm_param_v0_to_v1_import():
    _test_batch_norm_param_old_to_new_import(old_version=0, new_version=1)


def test_batch_norm_param_v0_to_v2_import():
    _test_batch_norm_param_old_to_new_import(old_version=0, new_version=2)


def test_batch_norm_param_v1_to_v2_import():
    _test_batch_norm_param_old_to_new_import(old_version=1, new_version=2)


def test_batch_norm_fused():
    n_in = 3
    net_dict = {
        "output": {
            "class": "batch_norm",
            "from": "data:data",
            "masked_time": False,
            "param_version": 2,
        }
    }

    def _find_fused_bn_op(session_):
        for op in session_.graph.get_operations():
            assert isinstance(op, tf.Operation)
            if "FusedBatchNorm" in op.type:
                return op

    with make_scope() as session:
        config = Config({"extern_data": {"data": {"dim": n_in}}})
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        out = network.get_default_output_layer().output.placeholder
        assert _find_fused_bn_op(session)
        network.initialize_params(session)
        out_np = session.run(out, feed_dict=make_feed_dict(network.extern_data))
        assert isinstance(out_np, numpy.ndarray)
        assert not numpy.allclose(out_np, 0.0)

    with make_scope() as session:
        config = Config({"extern_data": {"data": {"dim": n_in}}})
        network = TFNetwork(config=config, train_flag=False)
        network.construct_from_dict(net_dict)
        out = network.get_default_output_layer().output.placeholder
        assert _find_fused_bn_op(session)
        network.initialize_params(session)
        out_np = session.run(out, feed_dict=make_feed_dict(network.extern_data))
        assert isinstance(out_np, numpy.ndarray)
        assert not numpy.allclose(out_np, 0.0)


def test_batch_norm():
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData(), train_flag=True)
        with tf_compat.v1.variable_scope("src_nchw"):
            src_nhwc = InternalLayer(
                name="src_nchw",
                network=net,
                output=Data(
                    **{
                        "name": "src_nchw_output",
                        "dim": 16,
                        "shape": (None, 16, 16),
                        "batch_dim_axis": 0,
                        "time_dim_axis": 1,
                        "feature_dim_axis": 3,
                        "sparse": False,
                    }
                ),
            )
            src_nhwc.output.placeholder = tf_compat.v1.placeholder(shape=(None, None, 16, 16), dtype=tf.float32)
            src_nhwc.output.size_placeholder = {0: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}

        rnd = numpy.random.RandomState(42)
        input_data = rnd.rand(10, 11, 16, 16)
        seq_lens = numpy.array([11] * 10)

        with tf_compat.v1.variable_scope("batch_norm_masked_nchw"):
            batch_norm_1 = BatchNormLayer(
                name="batch_norm_masked_nchw",
                network=net,
                masked_time=True,
                sources=[src_nhwc],
                output=BatchNormLayer.get_out_data_from_opts(
                    name="batch_norm_masked_nchw", sources=[src_nhwc], network=net
                ),
            )
            batch_norm_1.post_init(layer_desc={"output": batch_norm_1.output})
        with tf_compat.v1.variable_scope("batch_norm_nonmasked_nchw"):
            batch_norm_2 = BatchNormLayer(
                name="batch_norm_nonmasked_nchw",
                network=net,
                masked_time=False,
                sources=[src_nhwc],
                output=BatchNormLayer.get_out_data_from_opts(
                    name="batch_norm_nonmasked_nchw", sources=[src_nhwc], network=net
                ),
            )
            batch_norm_2.post_init(layer_desc={"output": batch_norm_2.output})
        tf_compat.v1.global_variables_initializer().run(session=session)
        out_1, seq_lens_1 = session.run(
            [batch_norm_1.output.placeholder, batch_norm_1.output.size_placeholder[0]],
            feed_dict={src_nhwc.output.placeholder: input_data, src_nhwc.output.size_placeholder[0]: seq_lens},
        )
        out_2, seq_lens_2 = session.run(
            [batch_norm_2.output.placeholder, batch_norm_2.output.size_placeholder[0]],
            feed_dict={src_nhwc.output.placeholder: input_data, src_nhwc.output.size_placeholder[0]: seq_lens},
        )
        numpy.testing.assert_array_almost_equal(out_1, out_2)
        print(numpy.sum(out_1 - out_2))


def test_batch_norm_unequal_seq_len():
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData(), train_flag=True)
        with tf_compat.v1.variable_scope("src_nhwc"):
            src_nhwc = InternalLayer(
                name="src_nhwc",
                network=net,
                output=Data(
                    **{
                        "name": "src_nhwc_output",
                        "dim": 16,
                        "shape": (None, 16, 16),
                        "batch_dim_axis": 0,
                        "time_dim_axis": 1,
                        "feature_dim_axis": 3,
                        "sparse": False,
                    }
                ),
            )
            src_nhwc.output.placeholder = tf_compat.v1.placeholder(shape=(None, None, 16, 16), dtype=tf.float32)
            src_nhwc.output.size_placeholder = {0: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}

        rnd = numpy.random.RandomState(42)
        input_data = rnd.rand(10, 11, 16, 16).astype("f")
        input_data[2, 5:, :, :] = 0
        input_data_masked = numpy.copy(input_data)
        seq_lens = numpy.array([11, 11, 5, 11, 11, 11, 11, 11, 11, 11], dtype=numpy.float32)
        n1 = 9 * 11 * 16 + 5 * 16
        n2 = 10 * 11 * 16

        with tf_compat.v1.variable_scope("batch_norm_masked_nchw"):
            batch_norm_1 = BatchNormLayer(
                name="batch_norm_masked_nchw",
                network=net,
                masked_time=True,
                use_shift=False,
                use_std=False,
                epsilon=0.0,
                sources=[src_nhwc],
                output=BatchNormLayer.get_out_data_from_opts(
                    name="batch_norm_masked_nchw", sources=[src_nhwc], network=net
                ),
            )
            batch_norm_1.post_init(layer_desc={"output": batch_norm_1.output})
        with tf_compat.v1.variable_scope("batch_norm_nonmasked_nchw"):
            batch_norm_2 = BatchNormLayer(
                name="batch_norm_nonmasked_nchw",
                network=net,
                masked_time=False,
                use_shift=False,
                use_std=False,
                epsilon=0,
                sources=[src_nhwc],
                output=BatchNormLayer.get_out_data_from_opts(
                    name="batch_norm_nonmasked_nchw", sources=[src_nhwc], network=net
                ),
            )
            batch_norm_2.post_init(layer_desc={"output": batch_norm_2.output})
        tf_compat.v1.global_variables_initializer().run(session=session)
        out_1, seq_lens_1 = session.run(
            [batch_norm_1.output.placeholder, batch_norm_1.output.size_placeholder[0]],
            feed_dict={src_nhwc.output.placeholder: input_data, src_nhwc.output.size_placeholder[0]: seq_lens},
        )
        out_2, seq_lens_2 = session.run(
            [batch_norm_2.output.placeholder, batch_norm_2.output.size_placeholder[0]],
            feed_dict={src_nhwc.output.placeholder: input_data_masked, src_nhwc.output.size_placeholder[0]: seq_lens},
        )

        # Manually calculating batch_norm and compare to the tf output
        data_mean = numpy.mean(input_data, axis=(0, 1, 2), keepdims=True, dtype=numpy.float32)
        data_var = numpy.var(input_data, axis=(0, 1, 2), keepdims=True, dtype=numpy.float32)
        np_bn2 = (input_data - data_mean) * (1.0 / numpy.sqrt(data_var))
        numpy.testing.assert_array_almost_equal(np_bn2, out_2, decimal=5)
        # Manually calculating batch_norm with different seq_lens, having:
        # Mean_1 = n2 / n1 * Mean_2
        # Var_1 = n2 / n1 * (Var_2 + Mean_2 ^ 2 (1 - n2 / n1))
        # bn_1 = (x - Mean_1) * 1 / sqrt(Var_1)
        # Substituting Mean_1 and Var_1:
        np_bn1 = (input_data - n2 / n1 * data_mean) * (
            1.0 / numpy.sqrt(n2 / n1 * (data_var + data_mean**2 * (1 - n2 / n1)))
        )
        # Check with tf output.
        numpy.testing.assert_array_almost_equal(np_bn1, out_1, decimal=5)


def test_BatchNormLayer_static_time():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time", 13)
    in_dim = FeatureDim("in", 5)

    config = Config(
        dict(
            extern_data={
                "data": {
                    "dim_tags": (batch_dim, time_dim, in_dim),
                    "time_dim_axis": 1,
                }
            }
        )
    )

    net_dict = {
        "output": {"class": "batch_norm", "from": "data", "masked_time": True},
    }

    with make_scope() as session:
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        out = net.get_default_output_layer().output
        fetches = net.get_fetches_dict()
        fetches["out"] = out.placeholder
        session.run(fetches, feed_dict=make_feed_dict(net.extern_data))


def test_BatchNormLayer_dyn_time_scalar():
    from returnn.tensor import batch_dim, Dim, Tensor

    time_dim = Dim(Tensor("time", dims=(), dtype="int32"))  # scalar dyn size
    in_dim = Dim(5, name="in")

    config = Config(
        dict(
            extern_data={
                "data": {
                    "dim_tags": (batch_dim, time_dim, in_dim),
                    "time_dim_axis": 1,
                }
            }
        )
    )

    net_dict = {
        "output": {"class": "batch_norm", "from": "data", "masked_time": True},
    }

    with make_scope() as session:
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        out = net.get_default_output_layer().output
        fetches = net.get_fetches_dict()
        fetches["out"] = out.placeholder
        session.run(fetches, feed_dict=make_feed_dict(net.extern_data))


def test_BatchNormLayer_CondLayer():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    in_dim = FeatureDim("in", 12)

    config = Config(
        dict(
            extern_data={
                "data": {
                    "dim_tags": (batch_dim, time_dim, in_dim),
                }
            }
        )
    )

    net_dict = {
        "length": {"class": "length", "from": ["data:data"], "axis": batch_dim, "out_shape": {}},
        "mod": {"class": "eval", "from": "length", "eval": "source(0) % 2", "out_shape": {}},
        "compare": {"class": "compare", "from": "mod", "kind": "equal", "value": 0, "out_shape": {}},
        "cond": {
            "class": "cond",
            "from": [],
            "condition": "compare",
            "true_layer": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "batch_norm": {
                        "class": "subnetwork",
                        "from": [],
                        "subnetwork": {
                            "batch_norm": {
                                "class": "batch_norm",
                                "from": "base:base:data:data",
                                "in_dim": in_dim,
                                "use_std": True,
                                "use_shift": True,
                                "param_version": 2,
                                "momentum": 0.1,
                                "epsilon": 0.001,
                                "masked_time": True,
                                "out_shape": {batch_dim, time_dim, in_dim},
                            },
                            "output": {
                                "class": "copy",
                                "from": "batch_norm",
                                "out_shape": {batch_dim, time_dim, in_dim},
                            },
                        },
                        "name_scope": "",
                    },
                    "output": {"class": "copy", "from": "batch_norm", "out_shape": {batch_dim, time_dim, in_dim}},
                },
            },
            "false_layer": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "output": {"class": "copy", "from": "base:data:data", "out_shape": {batch_dim, time_dim, in_dim}}
                },
            },
            "out_shape": {batch_dim, time_dim, in_dim},
            "name_scope": "",
        },
        "output": {"class": "copy", "from": "cond", "out_shape": {batch_dim, time_dim, in_dim}},
    }

    with make_scope() as session:
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        out = net.get_default_output_layer().output
        fetches = net.get_fetches_dict()
        fetches["out"] = out.placeholder
        session.run(fetches, feed_dict=make_feed_dict(net.extern_data))


def test_activation_layer_net_construct():
    with make_scope() as session:
        num_inputs = 2
        config = Config()
        config.update(
            {
                "num_outputs": 3,
                "num_inputs": num_inputs,
                "network": {"output": {"class": "activation", "activation": "relu", "from": ["data"]}},
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_value("network"))
        out = network.get_default_output_layer().output.placeholder
        n_batch = 1
        seq_len = 3
        feed = {
            network.extern_data.get_default_input_data().placeholder: numpy.array(
                [[[0, 0], [-1, -1], [2, 2]]], dtype="float32"
            )
        }
        assert feed[network.extern_data.get_default_input_data().placeholder].shape == (n_batch, seq_len, num_inputs)
        v = session.run(out, feed_dict=feed)
        assert v.shape == (n_batch, seq_len, num_inputs)
        assert v.tolist() == [[[0, 0], [0, 0], [2, 2]]]


def test_activation_layer_abs_for_stft():
    with make_scope() as session:
        num_inputs = 1
        config = Config()
        frame_shift = 1
        frame_size = 3
        fft_size = 3
        config.update(
            {
                "num_outputs": 3,
                "num_inputs": num_inputs,
                "network": {
                    "stft": {
                        "class": "multichannel_stft_layer",
                        "from": "data",
                        "frame_shift": frame_shift,
                        "frame_size": frame_size,
                        "window": "hanning",
                        "fft_size": fft_size,
                        "use_rfft": True,
                        "pad_last_frame": False,
                        "nr_of_channels": 1,
                    },
                    "output": {"class": "activation", "activation": "abs", "from": ["stft"]},
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_value("network"))
        out = network.get_default_output_layer().output.placeholder
        n_batch = 1
        seq_len = 6
        feed = {
            network.extern_data.get_default_input_data().placeholder: numpy.array(
                [[[0], [0], [2], [1], [4], [3]]], dtype=numpy.float32
            )
        }
        assert feed[network.extern_data.get_default_input_data().placeholder].shape == (n_batch, seq_len, num_inputs)
        v = session.run(out, feed_dict=feed)
        assert v.shape == (n_batch, seq_len - (frame_size - 1), fft_size // 2 + 1)

        input_stft = tf.signal.stft(
            numpy.array([[0, 0, 2, 1, 4, 3]], dtype=numpy.float32),
            frame_length=frame_size,
            frame_step=frame_shift,
            fft_length=fft_size,
            window_fn=tf.signal.hann_window,
        )
        exp_output = tf.math.abs(input_stft)
        assert v.tolist() == exp_output.eval().tolist()


def test_activation_layer_net_construct_two_out():
    with make_scope() as session:
        num_inputs = 2
        config = Config()
        config.update(
            {
                "num_outputs": 3,
                "num_inputs": num_inputs,
                "network": {
                    "0out": {
                        "class": "linear",
                        "n_out": 1,
                        "activation": "relu",
                        "from": ["data"],
                        "is_output_layer": True,
                    },
                    "output": {"class": "activation", "activation": "relu", "from": ["data"]},
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_value("network"))
        session.run(tf_compat.v1.global_variables_initializer())
        out = network.layers["output"].output.placeholder
        out2 = network.layers["0out"].output.placeholder
        n_batch = 1
        seq_len = 3
        feed = {
            network.extern_data.get_default_input_data().placeholder: numpy.array(
                [[[0, 0], [-1, -1], [2, 2]]], dtype="float32"
            )
        }
        assert feed[network.extern_data.get_default_input_data().placeholder].shape == (n_batch, seq_len, num_inputs)
        v, v2 = session.run([out, out2], feed_dict=feed)
        assert v.shape == (n_batch, seq_len, num_inputs)
        assert v.tolist() == [[[0, 0], [0, 0], [2, 2]]]


def _test_simple_eval_func(s):
    with make_scope() as session:
        num_inputs = 2
        config = Config()
        config.update(
            {
                "extern_data": {"data": {"dim": num_inputs}},
                "network": {"output": {"class": "eval", "eval": "%s(source(0))" % s, "from": "data"}},
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_value("network"))
        feed = make_feed_dict(network.extern_data)
        out = network.get_default_output_layer().output.placeholder
        session.run(out, feed_dict=feed)


def test_simple_eval_tanh():
    _test_simple_eval_func("tf.tanh")


def test_simple_eval_sigmoid():
    _test_simple_eval_func("tf.sigmoid")


def _test_simple_activation(s):
    with make_scope() as session:
        num_inputs = 2
        config = Config()
        config.update(
            {
                "extern_data": {"data": {"dim": num_inputs}},
                "network": {"output": {"class": "activation", "activation": s, "from": "data"}},
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_value("network"))
        feed = make_feed_dict(network.extern_data)
        out = network.get_default_output_layer().output.placeholder
        session.run(out, feed_dict=feed)


def test_simple_activation_tanh():
    _test_simple_activation("tanh")


def test_simple_activation_log_sigmoid():
    _test_simple_activation("log_sigmoid")


def test_cnn_building_block():
    with make_scope() as session:
        num_inputs = 192
        channel_num = 32
        feature_dim = 6
        filters = 32
        filter_size = (3, 3)
        config = Config()
        config.update(
            {
                "num_inputs": num_inputs,
                "num_outputs": filters,
                "network": {
                    "split": {"class": "split_dims", "axis": "f", "dims": (channel_num, feature_dim), "from": ["data"]},
                    "swap_axes": {
                        "class": "swap_axes",
                        "axis1": "dim:%i" % channel_num,
                        "axis2": "dim:%i" % feature_dim,
                        "from": "split",
                    },
                    "c1": {
                        "class": "conv",
                        "n_out": filters,
                        "filter_size": filter_size,
                        "auto_use_channel_first": False,
                        "in_spatial_dims": ("T", "dim:6"),
                        "strides": (1, 1),
                        "dilation_rate": (1, 1),
                        "padding": "SAME",
                        "activation": None,
                        "with_bias": False,
                        "from": "swap_axes",
                    },
                    "bn1": {"class": "batch_norm", "from": "c1", "masked_time": True},
                    "y1": {"class": "activation", "activation": "relu", "batch_norm": False, "from": "bn1"},
                    "c2": {
                        "class": "conv",
                        "n_out": filters,
                        "filter_size": filter_size,
                        "auto_use_channel_first": False,
                        "in_spatial_dims": ("T", "dim:6"),
                        "strides": (1, 1),
                        "dilation_rate": (1, 1),
                        "padding": "SAME",
                        "activation": None,
                        "with_bias": False,
                        "from": "y1",
                    },
                    "p": {"class": "combine", "kind": "add", "from": ["c2", "swap_axes"]},
                    "bn2": {"class": "batch_norm", "from": "p", "masked_time": True},
                    "y2": {"class": "activation", "activation": "relu", "batch_norm": False, "from": "bn2"},
                    "out_pool": {
                        "class": "reduce",
                        "mode": "avg",
                        "axes": "dim:%i" % feature_dim,
                        "keep_dims": False,
                        "from": "y2",
                    },
                    "output": {"class": "copy", "from": ["out_pool"], "is_output_layer": True},
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_value("network"))
        swap_layer = network.layers["swap_axes"]
        assert swap_layer.output.dim == channel_num
        session.run(tf_compat.v1.global_variables_initializer())
        out = network.layers["output"].output.placeholder
        n_batch = 5
        seq_len = 10
        seq_lens = numpy.array([10, 10, 10, 10, 10], dtype=numpy.int32)
        feed = {
            network.extern_data.get_default_input_data().placeholder: numpy.random.rand(
                n_batch, seq_len, num_inputs
            ).astype("f"),
            network.extern_data.get_default_input_data().size_placeholder[0]: seq_lens,
        }
        v = session.run(out, feed_dict=feed)


def test_combine_layer_net_construct():
    with make_scope() as session:
        net_dict = {
            "lstm0_fw": {"class": "rec", "unit": "lstm", "n_out": 5, "direction": 1, "from": "data:data"},
            "lstm0_bw": {"class": "rec", "unit": "lstm", "n_out": 5, "direction": -1, "from": "data:data"},
            "lstm0_avg": {"class": "combine", "kind": "average", "from": ["lstm0_fw", "lstm0_bw"]},
            "output": {"class": "softmax", "loss": "ce", "from": ["lstm0_avg"]},
        }
        config = Config()
        config.update(dict(num_inputs=4, num_outputs=9))
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)


def test_CombineLayer_simple_add():
    with make_scope() as session:
        net_dict = {
            "lin1": {"class": "linear", "activation": "sigmoid", "n_out": 5, "from": "data:data"},
            "lin2": {"class": "linear", "activation": "sigmoid", "n_out": 5, "from": "data:data"},
            "combine": {"class": "combine", "kind": "add", "from": ["lin1", "lin2"]},
            "output": {"class": "softmax", "loss": "ce", "from": "combine"},
        }
        config = Config({"debug_print_layer_output_template": True})
        config.update(dict(num_inputs=4, num_outputs=9))
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        out = network.get_default_output_layer()
        feed_dict = make_feed_dict(network.extern_data.data.values(), same_time=True)
        session.run(tf_compat.v1.global_variables_initializer())
        session.run(out.output.placeholder, feed_dict=feed_dict)


def test_CombineLayer_broadcast():
    with make_scope() as session:
        net_dict = {
            "lin1": {"class": "linear", "activation": "sigmoid", "n_out": 5, "from": "data:data"},
            "lin2": {"class": "linear", "activation": "sigmoid", "n_out": 1, "from": "data:data"},
            "lin2_squeeze": {"class": "squeeze", "from": "lin2", "axis": "f"},
            "combine": {"class": "combine", "kind": "add", "from": ["lin1", "lin2_squeeze"]},
            "output": {"class": "softmax", "loss": "ce", "from": "combine"},
        }
        config = Config({"debug_print_layer_output_template": True})
        config.update(dict(num_inputs=4, num_outputs=9))
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        assert network.get_layer("combine").output.shape == (None, 5)
        out = network.get_default_output_layer()
        assert out.output.shape == (None, 9)
        feed_dict = make_feed_dict(network.extern_data.data.values(), same_time=True)
        session.run(tf_compat.v1.global_variables_initializer())
        session.run(out.output.placeholder, feed_dict=feed_dict)


def test_CombineLayer_broadcast_multiple():
    with make_scope() as session:
        net_dict = {
            "p1": {"class": "variable", "shape": (5, 5, 3), "add_batch_axis": False},
            "p2": {"class": "variable", "shape": (5,), "add_batch_axis": False},
            "combine": {"class": "combine", "kind": "add", "from": ["p1", "p2"]},
            "output": {"class": "softmax", "loss": "ce", "from": "combine"},
        }
        config = Config({"debug_print_layer_output_template": True})
        config.update(dict(num_inputs=4, num_outputs=9))
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        assert network.get_layer("combine").output.batch_shape == (5, 5, 3)
        out = network.get_default_output_layer()
        assert out.output.batch_shape == (5, 5, 9) and not out.output.have_batch_axis()
        feed_dict = make_feed_dict(network.extern_data.data.values(), same_time=True)
        session.run(tf_compat.v1.global_variables_initializer())
        out_v = session.run(out.output.placeholder, feed_dict=feed_dict)
        assert out_v.shape == out.output.batch_shape


def test_CombineLayer_broadcast_same_dim_diff_tag():
    from returnn.tf.util.data import batch_dim, FeatureDim, SpatialDim

    time_dim = SpatialDim("time")
    input_dim = FeatureDim("input", 3)
    feat_dim = FeatureDim("feat", 3)

    template_net = TFNetwork(config=Config(), extern_data=ExternData())
    out = CombineLayer.get_out_data_from_opts(
        network=template_net,
        name="template_combine",
        kind="add",
        sources=[
            InternalLayer(name="a", network=template_net, output=Data("a", dim_tags=[batch_dim, time_dim, input_dim])),
            InternalLayer(name="b", network=template_net, output=Data("b", dim_tags=[feat_dim])),
        ],
        allow_broadcast_all_sources=True,
    )
    assert out.dim_tags == (batch_dim, time_dim, input_dim, feat_dim)

    with make_scope() as session:
        net_dict = {
            "p": {"class": "variable", "shape": [feat_dim], "add_batch_axis": False},
            "output": {"class": "combine", "kind": "add", "from": ["data", "p"], "allow_broadcast_all_sources": True},
        }
        config = Config({"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, input_dim]}}})
        network = TFNetwork(config=config)
        network.construct_from_dict(net_dict)
        out = network.get_default_output_layer()
        assert out.output.dim_tags == (batch_dim, time_dim, input_dim, feat_dim)
        feed_dict = make_feed_dict(network.extern_data, n_batch=2, n_time=5)
        session.run(tf_compat.v1.global_variables_initializer())
        out_v = session.run(out.output.placeholder, feed_dict=feed_dict)
        assert out_v.shape == (2, 5, 3, 3)


def test_CombineLayer_match_unknown():
    with make_scope() as session:
        dat1 = Data(name="undefined", shape=(None, 3))
        assert dat1.dim_tags[1].undefined
        # Create placeholders to have this dyn size clearly defined.
        dat2 = Data(name="defined", shape=(None, 3), auto_create_placeholders=True)
        net = TFNetwork(extern_data=ExternData())
        layer1 = InternalLayer(name="layer1_undefined", network=net, output=dat1)
        layer2 = InternalLayer(name="layer2_defined", network=net, output=dat2)
        out = CombineLayer.get_out_data_from_opts(name="combine", network=net, sources=[layer1, layer2])
        assert out.dim_tags[:2] == dat2.dim_tags[:2] and out.batch_shape == dat2.batch_shape


def test_CombineLayer_match_unknown_derived():
    with make_scope() as session:
        dat1 = Data(name="undefined", shape=(None, 3))
        assert dat1.dim_tags[1].undefined
        dat1_derived_dim_tags = list(dat1.dim_tags)
        dat1_derived_dim_tags[1] = Dim(
            kind=Dim.Types.Spatial,
            description="undefined_derived_dim",
            derived_from_tag=dat1.dim_tags[1],
            dimension=None,
        )
        dat1_derived = Data(name="undefined_derived", dim_tags=dat1_derived_dim_tags)
        assert dat1_derived.dim_tags[1].undefined
        # Create placeholders to have this dyn size clearly defined.
        dat2 = Data(name="defined", shape=(None, 3), auto_create_placeholders=True)
        net = TFNetwork(extern_data=ExternData())
        layer1 = InternalLayer(name="layer1_undefined_derived", network=net, output=dat1_derived)
        layer2 = InternalLayer(name="layer2_defined", network=net, output=dat2)
        out = CombineLayer.get_out_data_from_opts(name="combine", network=net, sources=[layer1, layer2])
        assert out.dim_tags[:2] == dat2.dim_tags[:2] and out.batch_shape == dat2.batch_shape


def test_CombineLayer_match_unknown_batch_feature_major():
    with make_scope() as session:
        dat1 = Data(name="undefined", shape=(None, 1))
        assert dat1.dim_tags[1].undefined
        # Create placeholders to have this dyn size clearly defined.
        dat2 = Data(name="defined", shape=(None, 1), auto_create_placeholders=True)
        dat2_bf_major = dat2.copy_as_batch_feature_major()
        net = TFNetwork(extern_data=ExternData())
        layer1 = InternalLayer(name="layer1_undefined", network=net, output=dat1)
        layer2 = InternalLayer(name="layer2_defined", network=net, output=dat2_bf_major)
        out = CombineLayer.get_out_data_from_opts(name="combine", network=net, sources=[layer1, layer2])
        assert out.dim_tags[:2] == dat2.dim_tags[:2] and out.batch_shape == dat2.batch_shape


def test_CombineLayer_match_unknown_batch_feature_major_with_out_type():
    with make_scope() as session:
        dat1 = Data(name="undefined", shape=(None, 1))
        assert dat1.dim_tags[1].undefined
        # Create placeholders to have this dyn size clearly defined.
        dat2 = Data(name="defined", shape=(None, 1), auto_create_placeholders=True)
        dat2_bf_major = dat2.copy_as_batch_feature_major()
        net = TFNetwork(extern_data=ExternData())
        layer1 = InternalLayer(name="layer1_undefined", network=net, output=dat1)
        layer2 = InternalLayer(name="layer2_defined", network=net, output=dat2_bf_major)
        out = CombineLayer.get_out_data_from_opts(
            name="combine", network=net, sources=[layer1, layer2], out_type={"dim": 1, "shape": (None, 1)}
        )
        assert out.dim_tags[:2] == dat2.dim_tags[:2] and out.batch_shape == dat2.batch_shape


def test_CombineLayer_different_batch_axis():
    # ["base:enc_ctx", "weight_feedback", "s_transformed"]
    # base:enc_ctx: Data(name='enc_ctx_output', shape=(None, 14), batch_dim_axis=1)
    # weight_feedback: Data(name='weight_feedback_output', shape=(None, 14), batch_dim_axis=1)
    # s_transformed: Data(name='s_transformed_output', shape=(14,), time_dim_axis=None)
    # out: Data(name='energy_in_output', shape=(None, 14), beam_size=3)
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    feature_dim = FeatureDim("feature", 5)
    time_dim = SpatialDim("time")
    with make_scope() as session:
        config = Config(
            {
                "extern_data": {
                    "enc_ctx": {"dim_tags": (batch_dim, time_dim, feature_dim), "available_for_inference": True},
                    "weight_feedback": {
                        "dim_tags": (time_dim, batch_dim, feature_dim),
                        "available_for_inference": True,
                    },
                    "s_transformed": {"dim_tags": (batch_dim, feature_dim), "available_for_inference": True},
                }
            }
        )
        net = TFNetwork(config=config, train_flag=True)
        l1 = net.get_layer("data:enc_ctx")
        l2 = net.get_layer("data:weight_feedback")
        l3 = net.get_layer("data:s_transformed")
        out = net.add_layer(name="energy_in", layer_class=CombineLayer, kind="add", sources=[l1, l2, l3])
        print("out:", out)
        assert out.output.dim_tags == (batch_dim, time_dim, feature_dim)
        session.run(out.output.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_CombineLayer_two_time_dims():
    with make_scope() as session:
        n_dim = 5
        n_batch = 3
        n_time1 = 7
        n_time2 = 11
        rnd = numpy.random.RandomState(42)
        net_dict = {"output": {"class": "combine", "kind": "add", "from": ["data:in0", "data:in1", "data:in2"]}}
        config = Config({"debug_print_layer_output_template": True})
        extern_data = ExternData()
        in0 = Data(name="in0", shape=(None, None, n_dim), batch_dim_axis=1, auto_create_placeholders=True)
        in1 = Data(
            # same time as first in in0
            name="in1",
            shape=(None, n_dim),
            auto_create_placeholders=True,
        )
        in2 = Data(
            # same time as in second in in0
            name="in2",
            shape=(None, n_dim),
            batch_dim_axis=1,
            auto_create_placeholders=True,
        )
        extern_data.register_data(in0)
        extern_data.register_data(in1)
        extern_data.register_data(in2)
        in1.get_size_dim_tag(0).declare_same_as(in0.get_size_dim_tag(0))
        in2.get_size_dim_tag(0).declare_same_as(in0.get_size_dim_tag(1))
        print("ExternData all dimension tags (allow_same_feature_dim=True):")
        pprint(extern_data.get_all_dimension_tags(allow_same_feature_dim=True))
        network = TFNetwork(config=config, extern_data=extern_data, train_flag=True)
        network.construct_from_dict(net_dict)
        output = network.get_default_output_layer().output
        assert output.shape == (None, None, n_dim) and set(output.size_placeholder.keys()) == {0, 1}
        assert output.batch_dim_axis == 1 and output.time_dim_axis == 0
        time1_np = numpy.array([n_time1, n_time1 - 3, n_time1 - 2])
        assert min(time1_np) > 0 and max(time1_np) == n_time1 and len(time1_np) == n_batch
        time2_np = numpy.array([n_time2, n_time2 - 2, n_time2 - 5])
        assert min(time2_np) > 0 and max(time2_np) == n_time2 and len(time2_np) == n_batch
        in0_np = rnd.normal(size=(n_time1, n_batch, n_time2, n_dim)).astype("float32")
        in1_np = rnd.normal(size=(n_batch, n_time1, n_dim)).astype("float32")
        in2_np = rnd.normal(size=(n_time2, n_batch, n_dim)).astype("float32")
        out_np, out_sizes_np = session.run(
            fetches=(output.placeholder, output.size_placeholder.as_dict()),
            feed_dict={
                in0.placeholder: in0_np,
                in0.size_placeholder[0]: time1_np,
                in0.size_placeholder[1]: time2_np,
                in1.placeholder: in1_np,
                in1.size_placeholder[0]: time1_np,
                in2.placeholder: in2_np,
                in2.size_placeholder[0]: time2_np,
            },
        )
        assert isinstance(out_np, numpy.ndarray)
        assert isinstance(out_sizes_np, dict) and set(out_sizes_np.keys()) == {0, 1}
        out_time0_np, out_time1_np = out_sizes_np[0], out_sizes_np[1]
        assert isinstance(out_time0_np, numpy.ndarray) and isinstance(out_time1_np, numpy.ndarray)
        assert out_np.shape == (n_time1, n_batch, n_time2, n_dim)


def test_CombineLayer_two_time_dims_first_not_most_generic():
    with make_scope() as session:
        n_dim = 5
        n_batch = 3
        n_time1 = 7
        n_time2 = 11
        rnd = numpy.random.RandomState(42)
        net_dict = {"output": {"class": "combine", "kind": "add", "from": ["data:in1", "data:in0", "data:in2"]}}
        config = Config({"debug_print_layer_output_template": True})
        extern_data = ExternData()
        in0 = Data(name="in0", shape=(None, None, n_dim), batch_dim_axis=1, auto_create_placeholders=True)
        in1 = Data(
            # same time as first in in0
            name="in1",
            shape=(None, n_dim),
            auto_create_placeholders=True,
        )
        in2 = Data(
            # same time as in second in in0
            name="in2",
            shape=(None, n_dim),
            batch_dim_axis=1,
            auto_create_placeholders=True,
        )
        extern_data.register_data(in0)
        extern_data.register_data(in1)
        extern_data.register_data(in2)
        in1.get_size_dim_tag(0).declare_same_as(in0.get_size_dim_tag(0))
        in2.get_size_dim_tag(0).declare_same_as(in0.get_size_dim_tag(1))
        print("ExternData all dimension tags (allow_same_feature_dim=True):")
        pprint(extern_data.get_all_dimension_tags(allow_same_feature_dim=True))
        network = TFNetwork(config=config, extern_data=extern_data, train_flag=True)
        network.construct_from_dict(net_dict)
        output = network.get_default_output_layer().output
        assert output.shape == (None, None, n_dim) and set(output.size_placeholder.keys()) == {0, 1}
        assert output.batch_dim_axis == 1 and output.time_dim_axis == 0
        time1_np = numpy.array([n_time1, n_time1 - 3, n_time1 - 2])
        assert min(time1_np) > 0 and max(time1_np) == n_time1 and len(time1_np) == n_batch
        time2_np = numpy.array([n_time2, n_time2 - 2, n_time2 - 5])
        assert min(time2_np) > 0 and max(time2_np) == n_time2 and len(time2_np) == n_batch
        in0_np = rnd.normal(size=(n_time1, n_batch, n_time2, n_dim)).astype("float32")
        in1_np = rnd.normal(size=(n_batch, n_time1, n_dim)).astype("float32")
        in2_np = rnd.normal(size=(n_time2, n_batch, n_dim)).astype("float32")
        out_np, out_sizes_np = session.run(
            fetches=(output.placeholder, output.size_placeholder.as_dict()),
            feed_dict={
                in0.placeholder: in0_np,
                in0.size_placeholder[0]: time1_np,
                in0.size_placeholder[1]: time2_np,
                in1.placeholder: in1_np,
                in1.size_placeholder[0]: time1_np,
                in2.placeholder: in2_np,
                in2.size_placeholder[0]: time2_np,
            },
        )
        assert isinstance(out_np, numpy.ndarray)
        assert isinstance(out_sizes_np, dict) and set(out_sizes_np.keys()) == {0, 1}
        out_time0_np, out_time1_np = out_sizes_np[0], out_sizes_np[1]
        assert isinstance(out_time0_np, numpy.ndarray) and isinstance(out_time1_np, numpy.ndarray)
        assert out_np.shape == (n_time1, n_batch, n_time2, n_dim)


def test_CombineLayer_two_time_dims_first_not_most_generic_with_n_out():
    with make_scope() as session:
        n_dim = 5
        n_batch = 3
        n_time1 = 7
        n_time2 = 11
        rnd = numpy.random.RandomState(42)
        net_dict = {
            "output": {"class": "combine", "kind": "add", "n_out": n_dim, "from": ["data:in1", "data:in0", "data:in2"]}
        }
        config = Config({"debug_print_layer_output_template": True})
        extern_data = ExternData()
        in0 = Data(name="in0", shape=(None, None, n_dim), batch_dim_axis=1, auto_create_placeholders=True)
        in1 = Data(
            # same time as first in in0
            name="in1",
            shape=(None, n_dim),
            auto_create_placeholders=True,
        )
        in2 = Data(
            # same time as in second in in0
            name="in2",
            shape=(None, n_dim),
            batch_dim_axis=1,
            auto_create_placeholders=True,
        )
        extern_data.register_data(in0)
        extern_data.register_data(in1)
        extern_data.register_data(in2)
        in1.get_size_dim_tag(0).declare_same_as(in0.get_size_dim_tag(0))
        in2.get_size_dim_tag(0).declare_same_as(in0.get_size_dim_tag(1))
        print("ExternData all dimension tags (allow_same_feature_dim=True):")
        pprint(extern_data.get_all_dimension_tags(allow_same_feature_dim=True))
        network = TFNetwork(config=config, extern_data=extern_data, train_flag=True)
        network.construct_from_dict(net_dict)
        output = network.get_default_output_layer().output
        assert output.shape == (None, None, n_dim) and set(output.size_placeholder.keys()) == {0, 1}
        assert output.batch_dim_axis == 1 and output.time_dim_axis == 0
        time1_np = numpy.array([n_time1, n_time1 - 3, n_time1 - 2])
        assert min(time1_np) > 0 and max(time1_np) == n_time1 and len(time1_np) == n_batch
        time2_np = numpy.array([n_time2, n_time2 - 2, n_time2 - 5])
        assert min(time2_np) > 0 and max(time2_np) == n_time2 and len(time2_np) == n_batch
        in0_np = rnd.normal(size=(n_time1, n_batch, n_time2, n_dim)).astype("float32")
        in1_np = rnd.normal(size=(n_batch, n_time1, n_dim)).astype("float32")
        in2_np = rnd.normal(size=(n_time2, n_batch, n_dim)).astype("float32")
        out_np, out_sizes_np = session.run(
            fetches=(output.placeholder, output.size_placeholder.as_dict()),
            feed_dict={
                in0.placeholder: in0_np,
                in0.size_placeholder[0]: time1_np,
                in0.size_placeholder[1]: time2_np,
                in1.placeholder: in1_np,
                in1.size_placeholder[0]: time1_np,
                in2.placeholder: in2_np,
                in2.size_placeholder[0]: time2_np,
            },
        )
        assert isinstance(out_np, numpy.ndarray)
        assert isinstance(out_sizes_np, dict) and set(out_sizes_np.keys()) == {0, 1}
        out_time0_np, out_time1_np = out_sizes_np[0], out_sizes_np[1]
        assert isinstance(out_time0_np, numpy.ndarray) and isinstance(out_time1_np, numpy.ndarray)
        assert out_np.shape == (n_time1, n_batch, n_time2, n_dim)


def test_CombineLayer_two_time_dims_allow_broadcast_all_sources():
    from returnn.tf.util.data import batch_dim

    with make_scope() as session:
        n_dim = 5
        n_batch = 3
        n_time1 = 7
        n_time2 = 11
        time1_dim = SpatialDim("time1")
        time2_dim = SpatialDim("time2")
        feat_dim = FeatureDim("feature", dimension=n_dim)
        rnd = numpy.random.RandomState(42)
        config = Config({"debug_print_layer_output_template": True})
        extern_data = ExternData()
        in1 = Data(name="in1", dim_tags=[batch_dim, time1_dim, feat_dim], auto_create_placeholders=True)
        in2 = Data(name="in2", dim_tags=[batch_dim, time2_dim, feat_dim], auto_create_placeholders=True)
        extern_data.register_data(in1)
        extern_data.register_data(in2)
        print("ExternData all dimension tags (allow_same_feature_dim=True):")
        pprint(extern_data.get_all_dimension_tags(allow_same_feature_dim=True))
        network = TFNetwork(config=config, extern_data=extern_data, train_flag=True)
        try:
            network.construct_from_dict(
                {"output": {"class": "combine", "kind": "add", "from": ["data:in1", "data:in2"]}}
            )
        except Exception as exc:
            # https://github.com/rwth-i6/returnn/issues/691
            print("Expected exception:", exc)
            assert "require broadcasting" in str(exc)
        else:
            raise Exception(
                "Expect allow_broadcast_all_sources exception, but layer constructed: %s"
                % network.get_default_output_layer()
            )
        network.construct_from_dict(
            {
                "output": {
                    "class": "combine",
                    "kind": "add",
                    "from": ["data:in1", "data:in2"],
                    "out_shape": {batch_dim, time1_dim, time2_dim, feat_dim},
                }
            }
        )
        output = network.get_default_output_layer().output
        assert output.shape == (None, None, n_dim) and set(output.size_placeholder.keys()) == {0, 1}
        time1_np = numpy.array([n_time1, n_time1 - 3, n_time1 - 2])
        assert min(time1_np) > 0 and max(time1_np) == n_time1 and len(time1_np) == n_batch
        time2_np = numpy.array([n_time2, n_time2 - 2, n_time2 - 5])
        assert min(time2_np) > 0 and max(time2_np) == n_time2 and len(time2_np) == n_batch
        in1_np = rnd.normal(size=(n_batch, n_time1, n_dim)).astype("float32")
        in2_np = rnd.normal(size=(n_batch, n_time2, n_dim)).astype("float32")
        out_np, out_sizes_np = session.run(
            fetches=(output.placeholder, output.size_placeholder.as_dict()),
            feed_dict={
                in1.placeholder: in1_np,
                in1.size_placeholder[0]: time1_np,
                in2.placeholder: in2_np,
                in2.size_placeholder[0]: time2_np,
            },
        )
        assert isinstance(out_np, numpy.ndarray)
        assert isinstance(out_sizes_np, dict) and set(out_sizes_np.keys()) == {0, 1}
        out_time0_np, out_time1_np = out_sizes_np[0], out_sizes_np[1]
        assert isinstance(out_time0_np, numpy.ndarray) and isinstance(out_time1_np, numpy.ndarray)
        assert out_np.shape == (n_batch, n_time1, n_time2, n_dim)


def test_CombineLayer_time_broadcast():
    with make_scope() as session:
        n_batch, n_time, n_features = 3, 7, 5
        net_dict = {
            "output": {"class": "combine", "kind": "add", "from": ["data:in1", "data:in2"]},
        }
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "in1": {
                        "shape": (n_features,),
                        "batch_dim_axis": None,
                        "time_dim_axis": None,
                        "feature_dim_axis": 0,
                    },
                    "in2": {"shape": (n_features, None), "batch_dim_axis": 0, "time_dim_axis": 2},
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        out = network.get_default_output_layer()
        assert out.output.batch_shape == (None, n_features, None)
        feed_dict = make_feed_dict(network.extern_data, n_batch=n_batch, n_time=n_time)
        session.run(tf_compat.v1.global_variables_initializer())
        out_v = session.run(out.output.placeholder, feed_dict=feed_dict)
        assert out_v.shape == (n_batch, n_features, n_time)


def test_CombineLayer_time_broadcast_swapped():
    with make_scope() as session:
        n_batch, n_time, n_features = 3, 7, 5
        net_dict = {
            "output": {"class": "combine", "kind": "add", "from": ["data:in1", "data:in2"]},
        }
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "in1": {"shape": (n_features, None), "batch_dim_axis": 0, "time_dim_axis": 2},
                    "in2": {
                        "shape": (n_features,),
                        "batch_dim_axis": None,
                        "time_dim_axis": None,
                        "feature_dim_axis": 0,
                    },
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        out = network.get_default_output_layer()
        assert out.output.batch_shape == (None, n_features, None)
        feed_dict = make_feed_dict(network.extern_data, n_batch=n_batch, n_time=n_time)
        session.run(tf_compat.v1.global_variables_initializer())
        out_v = session.run(out.output.placeholder, feed_dict=feed_dict)
        assert out_v.shape == (n_batch, n_features, n_time)


def test_CombineLayer_RangeFromLengthLayer():
    from returnn.tf.util.basic import py_print
    from returnn.tf.util.data import batch_dim, Dim, ImplicitDynSizeDim

    def _eval_seq_lens(source, **_kwargs):
        # Get some random varying seq lens.
        res = tf.cast(6.3 * tf.maximum(source(0), 0.0), tf.int32) + 1
        res = py_print(res, ["seq lens", res])
        return res

    time_dim = SpatialDim("T")
    new_time_dim = SpatialDim("T_new")
    feat_dim = FeatureDim("F", dimension=13)
    net_dict = {
        "data_red1": {
            "class": "reduce",
            "from": "data",
            "axis": "T",
            "mode": "mean",
            "out_shape": {batch_dim, feat_dim},
        },
        "data_red2": {"class": "reduce", "from": "data_red1", "axis": "F", "mode": "sum", "out_shape": {batch_dim}},
        "seq_lens": {
            "class": "eval",
            "from": "data_red2",
            "eval": _eval_seq_lens,
            "out_type": {"dtype": "int32"},
            "out_shape": {batch_dim},
        },
        "range": {
            "class": "range_from_length",
            "from": "seq_lens",
            "out_spatial_dim": new_time_dim,
            "out_shape": {new_time_dim, ImplicitDynSizeDim(batch_dim)},
        },
        "combine": {
            "class": "eval",
            "from": ["data_red1", "range"],
            "eval": "source(0) + 0.1 * tf.cast(source(1), tf.float32)",
            "out_shape": {batch_dim, new_time_dim, feat_dim},
        },
        "output": {"class": "copy", "from": "combine"},
    }

    config = Config(
        {
            "debug_print_layer_output_template": True,
            "extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}},
        }
    )

    with make_scope() as session:
        net = TFNetwork(config=config)
        in_data = net.extern_data.get_default_input_data()
        net.construct_from_dict(net_dict)
        out_data = net.get_default_output_layer().output
        seq_lens_data = net.get_layer("seq_lens").output
        assert out_data.batch_ndim == in_data.batch_ndim == 3
        assert out_data.dim_tags[0] == in_data.dim_tags[0] and out_data.dim_tags[0].is_batch_dim()
        out_time = out_data.get_time_dim_tag()
        assert out_time.dyn_size_ext.placeholder is seq_lens_data.placeholder
        assert out_time not in in_data.dim_tags
        feed_dict = make_feed_dict(net.extern_data)
        session.run((out_data.placeholder, out_data.get_sequence_lengths()), feed_dict=feed_dict)


def test_CompareLayer_allow_broadcast_all_sources():
    from returnn.tf.util.data import batch_dim, Dim

    time_tag = Dim(kind=Dim.Types.Spatial, description="time", dimension=None)
    with make_scope():
        n_out = 5
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {"data": {"dim": n_out, "same_dim_tags_as": {"t": time_tag}}},
            }
        )
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "range0": {"class": "range_in_axis", "from": "data", "axis": "b"},
                "range1": {"class": "range_in_axis", "from": "data", "axis": "t"},
                "compare": {
                    "class": "compare",
                    "from": ["range0", "range1"],
                    "kind": "equal",
                    "is_output_layer": True,
                    "out_shape": {batch_dim, time_tag},
                },
            }
        )


def test_CombineLayer_time_dim_no_session_after_session():
    from returnn.tf.util.data import batch_dim, FeatureDim, SpatialDim

    input_dim = FeatureDim("input", 3)

    with make_scope():
        time_dim = SpatialDim("time")
        net = TFNetwork(
            config=Config(), extern_data=ExternData({"data": {"dim_tags": [batch_dim, time_dim, input_dim]}})
        )
        net.construct_from_dict({"output": {"class": "copy", "from": "data"}})

    # This has messed up the time dim or batch info at some point.
    # It is only triggered by executing the code above,
    # otherwise it ran fine in isolation.
    test_CombineLayer_broadcast_same_dim_diff_tag()


def test_RangeFromLength_over_batch():
    # https://github.com/rwth-i6/pytorch-to-returnn/issues/100
    net_dict = {
        "batch_len": {"class": "length", "axis": "B", "from": "data"},
        "output": {"class": "range_from_length", "from": "batch_len"},
    }
    with make_scope() as session:
        config = Config({"extern_data": {"data": {"dim": 3}}})
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        in_ = net.extern_data.get_default_input_data()
        out = net.get_default_output_layer().output
        assert out.dim_tags[0].is_batch_dim()
        in_v, out_v = session.run((in_.placeholder, out.placeholder), feed_dict=make_feed_dict(net.extern_data))
        n_batch, _, _ = in_v.shape
        assert out_v.shape == (n_batch,)
        assert list(out_v) == list(range(n_batch))


def test_RangeInAxisLayer():
    net_dict = {
        "output": {"class": "range_in_axis", "from": "data", "axis": "T"},
    }
    with make_scope() as session:
        config = Config({"extern_data": {"data": {"dim": 3}}})
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        in_ = net.extern_data.get_default_input_data()
        out = net.get_default_output_layer().output
        assert out.get_time_dim_tag() == in_.get_time_dim_tag()
        in_v, out_v = session.run((in_.placeholder, out.placeholder), feed_dict=make_feed_dict(net.extern_data))
        n_batch, n_time, n_feat = in_v.shape
        assert out_v.shape == (n_time,)
        assert list(out_v) == list(range(n_time))


def test_RangeInAxisLayer_generic_dim():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    out_time_dim = time_dim + 2
    feat_dim = FeatureDim("feat", 3)
    config = Config({"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}}})
    net_dict = {
        "output": {"class": "range_in_axis", "from": "data", "axis": out_time_dim},
    }
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        in_ = net.extern_data.get_default_input_data()
        out = net.get_default_output_layer().output
        assert in_.get_time_dim_tag() == time_dim != out_time_dim == out.get_time_dim_tag()
        in_v, out_v = session.run((in_.placeholder, out.placeholder), feed_dict=make_feed_dict(net.extern_data))
        n_batch, n_time, n_feat = in_v.shape
        assert out_v.shape == (n_time + 2,)
        assert list(out_v) == list(range(n_time + 2))


def test_SwitchLayer_sanity_check():
    """
    https://github.com/rwth-i6/returnn/issues/800
    """
    from returnn.tf.util.data import Dim

    with make_scope():
        n_out = 5
        time_tag = Dim(kind=Dim.Types.Spatial, description="time", dimension=None)
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "data": {"dim": n_out, "same_dim_tags_as": {"t": time_tag}},
                    "classes": {"dim": n_out, "sparse": True, "same_dim_tags_as": {"t": time_tag}},
                },
            }
        )
        net = TFNetwork(config=config, search_flag=True)
        net.construct_from_dict(
            {
                "data_int": {"class": "cast", "from": "data", "dtype": "int32"},
                "output": {
                    "class": "rec",
                    "from": "data",
                    "unit": {
                        "prev_out": {"class": "reinterpret_data", "from": "prev:output", "set_sparse": False},
                        "cond": {"class": "compare", "from": ["prev_out", "base:data_int"], "kind": "less_equal"},
                        "switch": {"class": "switch", "condition": "cond", "true_from": 0.0, "false_from": 0.0},
                        "switch_red": {"class": "reduce", "from": "switch", "axis": "t", "mode": "max"},
                        "output_prob": {"class": "softmax", "from": "switch_red", "target": "classes", "loss": "ce"},
                        "output": {
                            "class": "choice",
                            "from": "output_prob",
                            "beam_size": 3,
                            "target": "classes",
                            "input_type": "prob",
                            "initial_output": 0,
                        },
                    },
                },
            }
        )


def test_dot_layer_shuffled_remaining_dims_static():
    with make_scope() as session:
        import numpy as np

        net_dict = {
            "a": {"class": "split_dims", "axis": "F", "dims": (2, 3, 5), "from": "data:data"},
            "b": {"class": "transpose", "from": ["a"], "perm": {"dim:2": "dim:3", "dim:3": "dim:2"}},
            "dot": {
                "class": "dot",
                "from": ["a", "b"],
                "red1": "dim:5",
                "red2": "dim:5",
                "var1": None,
                "var2": None,
                "debug": True,
            },
            "output": {"class": "merge_dims", "axes": ["dim:2", "dim:3"], "from": "dot"},
        }
        config = Config()
        config.update({"extern_data": {"data": {"shape": (30,)}}, "network": net_dict})
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        out = network.get_default_output_layer(must_exist=True)
        input_data = np.ones(shape=(17, 30))
        feed_dict = {network.extern_data.data["data"].placeholder: input_data}

        # just check that it runs
        session.run(out.output.placeholder, feed_dict)


def test_dot_layer_shuffled_remaining_dims_dynamic():
    with make_scope() as session:
        import numpy as np

        batch_size = 8
        time_size = 20
        feat_size = 10
        net_dict = {
            "a": {"class": "copy", "from": "data"},
            "b": {"class": "transpose", "from": ["a"], "perm": {"B": "T", "T": "B"}},
            "dot": {
                "class": "dot",
                "from": ["a", "b"],
                "red1": "F",
                "red2": "F",
                "var1": None,
                "var2": None,
                "debug": True,
            },
            "output": {"class": "copy", "from": ["dot"]},
        }
        config = Config()
        config.update({"num_outputs": 1, "num_inputs": feat_size, "network": net_dict})
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        out = network.get_default_output_layer(must_exist=True)
        input_data = np.ones(shape=(batch_size, time_size, feat_size))
        feed_dict = {network.layers["data"].output.placeholder: input_data}

        # just check that it runs
        session.run(out.output.placeholder, feed_dict)


def test_dropout_layer_net_construct():
    with make_scope() as session:
        net_dict = {
            "drop": {"class": "dropout", "dropout": 0.3, "dropout_noise_shape": {"*": None}, "from": "data:data"},
            "output": {"class": "softmax", "loss": "ce", "from": ["drop"]},
        }
        config = Config({"num_inputs": 4, "num_outputs": 9, "debug_print_layer_output_template": True})
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)


def test_DropoutLayer_axis():
    with make_scope() as session:
        net_dict = {
            "drop": {"class": "dropout", "dropout": 0.3, "dropout_axis": ["T", "F"], "from": "data:data"},
            "output": {"class": "softmax", "loss": "ce", "from": "drop"},
        }
        config = Config({"num_inputs": 4, "num_outputs": 9})
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        network.initialize_params(session)
        session.run(
            network.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(network.extern_data)
        )


def test_ScaledGradientLayer_tensor():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    input_dim = FeatureDim("input", 3)
    out_dim = FeatureDim("out", 5)

    config = Config(
        dict(
            extern_data={
                "data": {"dim_tags": (batch_dim, time_dim, input_dim)},
                "classes": {"dim_tags": (batch_dim, time_dim), "sparse_dim": out_dim},
            }
        )
    )

    net_dict = {
        "weight_raw": {
            "class": "variable",
            "shape": [input_dim, out_dim],
            "param_name": "param",
        },
        "scales": {
            "class": "random",
            "from": [],
            "shape": [out_dim],
            "distribution": "uniform",
            "minval": 0.1,
            "maxval": 1.5,
        },
        "weight": {
            "class": "scaled_grad",
            "from": "weight_raw",
            "scale": "scales",
        },
        "bias_raw": {
            "class": "variable",
            "shape": [out_dim],
            "param_name": "param",
        },
        "bias": {
            "class": "scaled_grad",
            "from": "bias_raw",
            "scale": "scales",
        },
        "dot": {
            "class": "dot",
            "from": ["data:data", "weight"],
            "reduce": input_dim,
            "out_shape": {batch_dim, time_dim, out_dim},
        },
        "add": {
            "class": "combine",
            "from": ["dot", "bias"],
            "kind": "add",
            "out_shape": {batch_dim, time_dim, out_dim},
        },
        "ce": {
            "class": "sparse_softmax_cross_entropy_with_logits",
            "logits": "add",
            "targets": "data:classes",
            "axis": out_dim,
            "loss": "as_is",
            "out_shape": {batch_dim, time_dim},
        },
    }

    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        optimizer = tf_compat.v1.train.AdamOptimizer(0.1)
        loss = network.get_total_loss()
        optim_op = optimizer.minimize(network.get_objective())
        session.run(tf_compat.v1.global_variables_initializer())
        feed_dict = make_feed_dict(network.extern_data)
        last_loss_v = None
        for step in range(10):
            loss_v, _ = session.run((loss, optim_op), feed_dict=feed_dict)
            print("Step %i, loss %f" % (step, loss_v))
            if last_loss_v is not None:
                assert loss_v < last_loss_v
            last_loss_v = loss_v


def test_subnetwork_layer_net_construct():
    with make_scope() as session:
        net_dict = {
            "ff0": {"class": "forward", "activation": "tanh", "n_out": 3, "from": "data:data"},
            "sub": {
                "class": "subnetwork",
                "from": "ff0",
                "subnetwork": {
                    "ff1": {"class": "forward", "activation": "relu", "n_out": 2, "from": "data"},  # unused
                    "output": {"class": "forward", "activation": "relu", "n_out": 2, "from": "data"},
                },
            },
            "output": {"class": "softmax", "loss": "ce", "from": "sub"},
        }
        config = Config()
        config.update(dict(num_inputs=4, num_outputs=3))
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        assert network.layers["sub"].output.dim == 2
        sub_layer = network.layers["sub"]
        assert isinstance(sub_layer, SubnetworkLayer)
        sub_layer_deps = sub_layer.get_dep_layers()
        assert sub_layer_deps, "%r no deps" % sub_layer
        all_deps = set()
        queue = [sub_layer]
        while queue:
            layer = queue.pop(0)
            if layer in all_deps:
                continue
            all_deps.add(layer)
            for dep in layer.get_dep_layers():
                if dep not in all_deps:
                    queue.append(dep)
        assert network.layers["ff0"] in all_deps


def test_subnet_loss():
    with make_scope() as session:
        config = Config({"extern_data": {"data": {"dim": 1}}, "debug_print_layer_output_template": True})
        net_dict = {
            "sub": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "var": {"class": "variable", "shape": [1]},
                    "loss": {"class": "copy", "from": "var", "loss": "as_is"},
                    "output": {"class": "copy", "from": "var"},  # no dep on "loss"
                },
            },
            # Output dep on "sub" to trigger subnet creation.
            # In theory, it would be nice if the loss is also constructed without that,
            # but this doesn't work currently as "sub" is never constructed by the current heuristics.
            "output": {"class": "copy", "from": "sub"},
        }
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        losses_dict, total_loss, total_constraints = network.get_losses_initialized(with_total=True)
        print("losses:")
        pprint(losses_dict)
        assert len(losses_dict) == 1 and set(losses_dict.keys()) == {"sub/loss"}


def test_subnet2_loss():
    with make_scope() as session:
        config = Config({"extern_data": {"data": {"dim": 1}}, "debug_print_layer_output_template": True})
        net_dict = {
            "sub": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "var": {"class": "variable", "shape": [1]},
                    "loss": {"class": "copy", "from": "var", "loss": "as_is"},
                    "output": {"class": "copy", "from": "var"},  # no dep on "loss"
                },
            },
            # Output dep on "sub" to trigger subnet creation.
            # In theory, it would be nice if the loss is also constructed without that,
            # but this doesn't work currently as "sub" is never constructed by the current heuristics.
            # Specifically depend on "sub/output", because in that case,
            # the SubnetworkLayer itself might not be created with the new subnet logic,
            # which is sth we want to test.
            "output": {"class": "copy", "from": "sub/output"},
        }
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        losses_dict, total_loss, total_constraints = network.get_losses_initialized(with_total=True)
        print("losses:")
        pprint(losses_dict)
        assert len(losses_dict) == 1 and set(losses_dict.keys()) == {"sub/loss"}


def test_constant_layer():
    with make_scope() as session:
        config = Config()
        config.update({"num_outputs": 3, "num_inputs": 2, "network": {"output": {"class": "constant", "value": 42}}})
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        out = network.get_default_output_layer(must_exist=True)
        v = session.run(out.output.placeholder)
        assert v.shape == ()  # (batch,), where batch==1 for broadcasting
        assert v == 42


def test_compare_layer():
    with make_scope() as session:
        config = Config()
        config.update(
            {
                "model": "/tmp/test-compare-layer-model",
                "num_outputs": 3,
                "num_inputs": 2,
                "network": {
                    "const": {"class": "constant", "value": 3},
                    "output": {"class": "compare", "from": "const", "value": 3},
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        out = network.get_default_output_layer(must_exist=True)
        v = session.run(out.output.placeholder)
        assert v.shape == ()  # (batch,), where batch==1 for broadcasting
        assert v.dtype == numpy.dtype("bool")
        assert v == True


def test_ShiftAxisLayer():
    with make_scope() as session:
        import numpy as np

        batch_size = 8
        time_size = 20
        feat_size = 10
        shift_amount = 5  # right-shift of 5 elements
        config = Config()
        config.update(
            {
                "num_outputs": feat_size,
                "num_inputs": feat_size,
                "network": {
                    "output": {
                        "class": "shift_axis",
                        "from": ["data"],
                        "amount": shift_amount,
                        "pad": True,
                        "axis": "T",
                        "adjust_size_info": False,
                    }
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        out = network.get_default_output_layer(must_exist=True)
        input_np = np.ones(shape=(batch_size, time_size, feat_size))
        input_np[0, :, 0] = np.arange(time_size)  # 0..time_size in time-axis
        feed_dict = {network.layers["data"].output.placeholder: input_np}
        v = session.run(out.output.placeholder, feed_dict)

        assert v.shape == (batch_size, time_size, feat_size)
        assert np.equal(v[0, shift_amount:, 0], np.arange(time_size - shift_amount)).all() == True
        assert (v[:, :shift_amount, :] == 0).all() == True  # padding
        assert (v[1:, shift_amount:, :] == 1).all() == True


def test_ShiftAxisLayer_small_time():
    with make_scope() as session:
        import numpy as np

        batch_size = 3
        time_size = 4
        feat_size = 2
        shift_amount = 5  # right-shift of 5 elements, more than time_size
        config = Config()
        config.update(
            {
                "num_outputs": feat_size,
                "num_inputs": feat_size,
                "network": {
                    "output": {
                        "class": "shift_axis",
                        "from": ["data"],
                        "amount": shift_amount,
                        "pad": True,
                        "axis": "T",
                        "adjust_size_info": False,
                    }
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        out = network.get_default_output_layer(must_exist=True)
        input_np = np.ones(shape=(batch_size, time_size, feat_size))
        feed_dict = {network.layers["data"].output.placeholder: input_np}
        v = session.run(out.output.placeholder, feed_dict)

        assert v.shape == (batch_size, time_size, feat_size)
        assert (v == 0).all() == True  # padding


def test_ReinterpretDataLayer_change_batch_to_spatial():
    new_spatial_dim = SpatialDim("new-spatial")
    net_dict = {"output": {"class": "reinterpret_data", "from": "data", "set_dim_tags": {"B": new_spatial_dim}}}
    config = Config({"extern_data": {"data": {"dim": 7}}})
    with make_scope():
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)


def test_layer_base_get_out_data_from_opts():
    with make_scope() as session:
        config = Config()
        config.update({"num_inputs": 4, "num_outputs": 3})
        network = TFNetwork(config=config)
        input_data = network.extern_data.data["data"]
        target_data = network.extern_data.data["classes"]
        assert input_data.dim == 4
        assert input_data.shape == (None, 4)
        assert not input_data.sparse
        assert input_data.dtype == "float32"
        assert target_data.dim == 3
        assert target_data.shape == (None,)
        assert target_data.sparse
        assert target_data.dtype == "int32"
        out = LayerBase._base_get_out_data_from_opts(network=network, name="output", target="classes")
        # Output data type is a non-sparse version of the targets by default.
        assert out.dim == target_data.dim
        assert out.shape == target_data.shape_dense
        assert not out.sparse
        assert out.dtype == "float32"


def test_ReduceLayer_reduce4d():
    config = Config()
    config.update({"num_inputs": 4, "num_outputs": 3, "debug_print_layer_output_template": True})
    network = TFNetwork(config=config)
    src_layer = InternalLayer(
        name="src", network=network, output=Data(name="src", shape=(None, 4, 512), auto_create_placeholders=True)
    )
    print("src:", src_layer)
    opts = {
        "axes": "dim:4",
        "keep_dims": True,
        "mode": "mean",
        "name": "c_out_reduce",
        "network": network,
        "sources": [src_layer],
    }
    out = ReduceLayer.get_out_data_from_opts(**opts)
    layer = ReduceLayer(output=out, **opts)
    print("layer:", layer)


def test_ReduceLayer_mean():
    net_dict = {"output": {"class": "reduce", "mode": "mean", "from": "data", "axis": "T"}}
    config = Config(dict(extern_data={"data": {"shape": (None, 4)}}))
    with make_scope() as session:
        network = TFNetwork(config=config)
        network.construct_from_dict(net_dict)
        in_ = network.extern_data.get_default_input_data()
        out = network.get_default_output_layer().output
        assert out.batch_dim_axis == 0 and out.shape == (4,)
        in_v, seq_len, out_v = session.run(
            (in_.placeholder, in_.get_sequence_lengths(), out.placeholder),
            feed_dict=make_feed_dict(network.extern_data),
        )
        n_batch = in_v.shape[0]
        assert n_batch == seq_len.shape[0] == out_v.shape[0]
        for b in range(n_batch):
            numpy.testing.assert_almost_equal(out_v[b], in_v[b, : seq_len[b]].mean(axis=0))


def test_reduce_repeat_1102():
    # https://github.com/rwth-i6/returnn/issues/1102
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    F_dim = FeatureDim("F", 1)
    speech_dim = SpatialDim("speech")
    speech_feat_dim = FeatureDim("speech-feat", 3)

    config = Config(
        dict(
            extern_data={
                "emb": {"dim_tags": (batch_dim, time_dim, F_dim), "dtype": "float32", "available_for_inference": True},
                "durations": {"dim_tags": (batch_dim, time_dim), "dtype": "int32", "available_for_inference": True},
                "target_speech": {
                    "dim_tags": (batch_dim, speech_dim, speech_feat_dim),
                    "dtype": "float32",
                    "available_for_inference": True,
                },
            }
        )
    )

    net_dict = {
        "nartts_model_reduce": {
            "class": "copy",
            "from": "reduce",
            "loss": "as_is",
            "out_shape": {batch_dim, speech_feat_dim},
        },
        "output": {"class": "copy", "from": "repeat", "out_shape": {batch_dim, F_dim, speech_dim}},
        "reduce": {
            "class": "reduce",
            "from": "data:target_speech",
            "mode": "mean",
            "axis": speech_dim,
            "out_shape": {batch_dim, speech_feat_dim},
        },
        "repeat": {
            "class": "repeat",
            "from": "data:emb",
            "repetitions": "data:durations",
            "axis": time_dim,
            "out_dim": speech_dim,
            "out_shape": {batch_dim, F_dim, speech_dim},
        },
    }

    with make_scope() as session:
        net = TFNetwork(config=config, eval_flag=True)
        net.construct_from_dict(net_dict)

        d = net.extern_data.data
        feed_dict = {
            d["emb"].placeholder: [[[1.0], [2.0], [0.0]]],
            d["emb"].size_placeholder[0]: [3],
            d["durations"].placeholder: [[1, 2, 1]],
            d["target_speech"].placeholder: [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]],
            d["target_speech"].size_placeholder[0]: [4],
        }
        # TODO ERROR this is not extern_data/placeholders/target_speech/target_speech_dim0_size
        #   but Tensor("repeat/Sum:0", shape=(?,), dtype=int32)
        print(d["target_speech"].size_placeholder[0])
        fetches = net.get_fetches_dict()
        session.run(fetches, feed_dict=feed_dict)


def test_SoftmaxOverSpatialLayer_start():
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData({"data": {"shape": (None, 5)}}), config=Config())
        rnd = numpy.random.RandomState(42)
        n_batch = 3
        n_time = 4
        n_dim = 7
        start_idxs = numpy.array([[3], [0], [1]]).astype("int32")  # (B, 1)
        input_np = rnd.normal(size=(n_batch, n_time, n_dim)).astype("float32")  # (B, T, D)
        src = InternalLayer(
            name="src", network=net, output=Data(**{"name": "src", "shape": (None, n_dim), "time_dim_axis": 1})
        )
        start = InternalLayer(
            name="start", network=net, output=Data(**{"name": "start", "shape": (1,), "dtype": "int32"})
        )
        start.output.placeholder = tf.constant(start_idxs)
        start.output.size_placeholder = {}
        print("input:", src.output)
        src.output.placeholder = tf.constant(input_np, dtype=tf.float32)
        src.output.size_placeholder = {0: tf.constant([n_time] * n_batch)}  # not sure if enough...
        opts = {
            "network": net,
            "name": "softmax_over_spatial_test",
            "sources": [src],
            "start": start,
            "use_time_mask": True,
        }
        out_data = SoftmaxOverSpatialLayer.get_out_data_from_opts(**opts)
        print("output:", out_data)
        out_data.sanity_check(ignore_placeholder=True)  # placeholder might be overwritten later
        assert out_data.shape == (n_dim, None)  # layer moves time-dim to back
        layer = SoftmaxOverSpatialLayer(output=out_data, **opts)
        assert layer.output.shape == (n_dim, None)
        try:
            out_np = session.run(layer.output.placeholder, feed_dict={net.extern_data.get_batch_info().dim: n_batch})
        except Exception as exc:
            from returnn.tf.network import help_on_tf_exception

            help_on_tf_exception(session=session, exception=exc, fetches=layer.output.placeholder)
            raise
        assert out_np.shape == (n_batch, n_dim, n_time)
        # check if masking worked
        range_idxs = numpy.ones_like(start_idxs) * numpy.expand_dims(numpy.arange(n_time), axis=0)
        cond = range_idxs < numpy.broadcast_to(start_idxs, [n_batch, n_time])  # (B, T)
        cond = numpy.expand_dims(cond, axis=1)
        cond = numpy.broadcast_to(cond, [n_batch, n_dim, n_time])  # (B, D, T)
        assert cond.sum() == n_dim * start_idxs.sum()  # check num of conds
        numpy.testing.assert_array_equal(out_np[cond], 0)


def test_SoftmaxOverSpatialLayer_window():
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData({"data": {"shape": (None, 5)}}), config=Config())
        rnd = numpy.random.RandomState(42)
        n_batch = 4
        n_time = 9
        n_dim = 1
        window_size = 5
        window_start_idxs = numpy.array([3, 0, 1, 7]).astype("int32")  # (B,)
        seqlens = numpy.array([5, 7, 3, 9]).astype("int32")
        input_np = rnd.normal(size=(n_batch, n_time, n_dim)).astype("float32")  # (B, T, D)
        src = InternalLayer(
            name="src", network=net, output=Data(**{"name": "src", "shape": (None, n_dim), "time_dim_axis": 1})
        )
        window_start = InternalLayer(
            name="window_start", network=net, output=Data(**{"name": "window_start", "shape": (), "dtype": "int32"})
        )
        window_start.output.placeholder = tf.constant(window_start_idxs)  # (B,)
        window_start.output.size_placeholder = {}
        print("input:", src.output)
        src.output.placeholder = tf.constant(input_np, dtype=tf.float32)
        src.output.size_placeholder = {0: tf.constant(seqlens, dtype=tf.int32)}
        opts = {
            "network": net,
            "name": "softmax_over_spatial_test",
            "sources": [src],
            "window_start": window_start,
            "window_size": window_size,
        }
        out_data = SoftmaxOverSpatialLayer.get_out_data_from_opts(**opts)
        print("output:", out_data)
        out_data.sanity_check(ignore_placeholder=True)  # placeholder might be overwritten later
        assert out_data.shape == (n_dim, None)  # layer moves time-dim to back
        layer = SoftmaxOverSpatialLayer(output=out_data, **opts)
        layer.output.sanity_check()
        assert layer.output.shape == (n_dim, None)
        out_np = session.run(layer.output.placeholder, feed_dict=make_feed_dict(net.extern_data, n_batch=n_batch))
        assert out_np.shape == (n_batch, n_dim, n_time)
        # check if window masking worked:
        # handle edge cases correctly: (start is 0-based)
        # 1. if the energy time-dim is less than `window_size`, we adjust the window size.
        # 2. for each seq, we adjust the window so that no elements after the seq-len are indexed.
        # seq[0]: start=3, seqlen=5 -> [1, 1, 1, 1, 1, 0, 0, 0, 0]
        # seq[1]: start=0, seqlen=7 -> [1, 1, 1, 1, 1, 0, 0, 0, 0]
        # seq[2]: start=1, seqlen=3 -> [1, 1, 1, 0, 0, 0, 0, 0, 0]
        # seq[3]: start=7, seqlen=9 -> [0, 0, 0, 0, 1, 1, 1, 1, 1]
        mask = numpy.array(
            [
                [0, 0, 0, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )  # (B, T)
        print("mask", mask)
        mask = numpy.expand_dims(mask, axis=1)
        mask = numpy.broadcast_to(mask, [n_batch, n_dim, n_time])  # (B, D, T)
        # check if layer output sums to one for each seq:
        out_sum = numpy.sum(out_np, axis=(1, 2))
        numpy.testing.assert_allclose(out_sum, [1] * n_batch, rtol=1e-5)
        numpy.testing.assert_allclose(out_np[~mask], 0, rtol=1e-5)  # check if masking worked


def test_SplitLayer_after_SplitDimsLayer():
    n_batch, n_time, n_in = 7, 3, 40
    config = Config(
        {
            "extern_data": {"data": {"dim": n_in}},
            "debug_print_layer_output_template": True,
        }
    )
    with make_scope():
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {
                "split_heads": {
                    "class": "split_dims",
                    "dims": (2, -1),
                    "axis": "F",
                    "from": "data:data",
                },  # [B,T,2,F|20]
                "split_qkv": {"class": "split", "size_splits": (5, 5, 10), "axis": "F", "from": "split_heads"},
                "output": {"class": "copy", "from": "split_qkv/0"},
            }
        )  # [B,T,2,F|5]
        out_t = net.get_default_output_layer().output.placeholder
        assert out_t.shape.as_list() == [None, None, 2, 5]


def test_SplitLayer_search():
    n_batch, n_time, n_in, n_out = 7, 3, 10, 10
    beam_size = 4
    config = Config(
        {
            "extern_data": {
                "data": {"dim": n_in},
                "classes": {"dim": n_out, "sparse": True, "available_for_inference": False},
            },
            "debug_print_layer_output_template": True,
        }
    )
    with make_scope():
        net = TFNetwork(config=config, search_flag=True, train_flag=False, eval_flag=True)
        net.construct_from_dict(
            {
                "encoder_seq": {"class": "linear", "activation": "tanh", "n_out": 5, "from": "data:data"},
                "encoder": {"class": "reduce", "mode": "sum", "from": ["encoder_seq"], "axis": "T"},
                "output": {
                    "class": "rec",
                    "from": [],
                    "target": "classes",
                    "max_seq_len": 20,
                    "unit": {
                        "embed": {"class": "linear", "from": ["prev:output"], "activation": None, "n_out": 10},
                        "split": {"class": "split", "size_splits": (5, 5), "axis": "F", "from": ["embed"]},
                        "output_prob": {
                            "class": "softmax",
                            "from": ["split/0", "base:encoder"],
                            "target": "classes",
                            "loss": "ce",
                        },
                        "output": {
                            "class": "choice",
                            "target": "classes",
                            "beam_size": beam_size,
                            "from": ["output_prob"],
                            "initial_output": 0,
                        },
                        "end": {"class": "compare", "from": ["output"], "value": 0},
                    },
                },
                "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"},
            }
        )


def test_SplitDimsLayer_simple_feat():
    n_batch, n_time, n_in = 7, 3, 20
    config = Config(
        {
            "extern_data": {"data": {"dim": n_in}},
            "debug_print_layer_output_template": True,
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict({"output": {"class": "split_dims", "axis": "f", "dims": (-1, 5), "from": "data:data"}})
        out_t = net.get_default_output_layer().output.placeholder
        assert out_t.shape.as_list() == [None, None, 4, 5]
        in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_time, n_in))
        out_v = session.run(out_t, feed_dict={net.extern_data.data["data"].placeholder: in_v})
        assert isinstance(out_v, numpy.ndarray)
        assert out_v.shape == (n_batch, n_time, 4, 5)
        numpy.testing.assert_almost_equal(out_v, in_v.reshape(out_v.shape))


def test_SplitDimsLayer_simple_time():
    n_batch, n_time, n_in = 7, 3, 20
    config = Config(
        {
            "extern_data": {"data": {"dim": n_in}},
            "debug_print_layer_output_template": True,
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict({"output": {"class": "split_dims", "axis": "t", "dims": (-1, 1), "from": "data:data"}})
        assert net.get_default_output_layer().output.get_dim_tag(
            1
        ) == net.extern_data.get_default_input_data().get_dim_tag(1)
        out_t = net.get_default_output_layer().output.placeholder
        assert out_t.shape.as_list() == [None, None, 1, 20]
        in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_time, n_in))
        out_v = session.run(out_t, feed_dict={net.extern_data.data["data"].placeholder: in_v})
        assert isinstance(out_v, numpy.ndarray)
        assert out_v.shape == (n_batch, n_time, 1, n_in)
        numpy.testing.assert_almost_equal(out_v, in_v.reshape(out_v.shape))


def test_SplitDimsLayer_simple_time2():
    n_batch, n_time, n_in = 7, 3, 20
    config = Config(
        {
            "extern_data": {"data": {"dim": n_in}},
            "debug_print_layer_output_template": True,
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict({"output": {"class": "split_dims", "axis": "t", "dims": (1, -1), "from": "data:data"}})
        in_ = net.extern_data.get_default_input_data()
        out = net.get_default_output_layer().output
        print(in_)
        print(out)
        assert out.get_dim_tag(2) == in_.get_dim_tag(1)
        assert out.time_dim_axis == 2
        out_t = out.placeholder
        assert out_t.shape.as_list() == [None, 1, None, 20]
        in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_time, n_in))
        out_v = session.run(out_t, feed_dict={in_.placeholder: in_v})
        assert isinstance(out_v, numpy.ndarray)
        assert out_v.shape == (n_batch, 1, n_time, n_in)
        numpy.testing.assert_almost_equal(out_v, in_v.reshape(out_v.shape))


def test_SplitDimsLayer_resolve_dims():
    assert SplitDimsLayer._resolve_dims(old_dim=3 * 5, new_dims=(3, -1)) == (3, 5)
    assert SplitDimsLayer._resolve_dims(old_dim=3 * 5, new_dims=(3, 5)) == (3, 5)
    assert SplitDimsLayer._resolve_dims(old_dim=3 * 5, new_dims=(5, -1)) == (5, 3)
    assert SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(-1, 3, 5)) == (2, 3, 5)
    assert SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(2, -1, 5)) == (2, 3, 5)
    assert SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(2, 3, -1)) == (2, 3, 5)
    assert SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(2, 3, -1, 1)) == (2, 3, 5, 1)

    assert SplitDimsLayer._resolve_dims(old_dim=3 * 5, new_dims=(3, -1), pad_to_multiples=True) == (3, 5)
    assert SplitDimsLayer._resolve_dims(old_dim=3 * 5 + 1, new_dims=(3, -1), pad_to_multiples=True) == (3, 6)
    assert SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(2, 3, -1), pad_to_multiples=True) == (2, 3, 5)
    assert SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5, new_dims=(2, 3, -1, 1), pad_to_multiples=True) == (
        2,
        3,
        5,
        1,
    )
    assert SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5 + 2, new_dims=(2, 3, -1), pad_to_multiples=True) == (2, 3, 6)
    assert SplitDimsLayer._resolve_dims(old_dim=2 * 3 * 5 + 2, new_dims=(2, 3, -1, 1), pad_to_multiples=True) == (
        2,
        3,
        6,
        1,
    )


def test_SplitDimsLayer_batch_feature_major_keep_feature():
    n_batch, n_time, n_in = 7, 3, 5
    config = Config(
        {"extern_data": {"data": {"dim": n_in, "shape": (n_in, None)}}}  # [B,D,T], i.e. batch-feature-major
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        assert net.extern_data.get_default_input_data().is_batch_feature_major
        net.construct_from_dict(
            {"output": {"class": "split_dims", "from": "data", "axis": "T", "dims": [-1, 1]}}  # [B,D,T,1]
        )
        out = net.get_default_output_layer().output
        assert out.get_dim_tag(2) == net.extern_data.get_default_input_data().get_time_dim_tag()
        assert out.dim_tags[1].dimension == n_in and out.dim_tags[3].dimension == 1
        assert out.placeholder.shape.as_list() == [None, n_in, None, 1]
        assert out.feature_dim_axis == 1  # https://github.com/rwth-i6/returnn/issues/596
        assert out.time_dim_axis == 2
        in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_in, n_time))
        out_v = session.run(out.placeholder, feed_dict={net.extern_data.data["data"].placeholder: in_v})
        assert isinstance(out_v, numpy.ndarray)
        assert out_v.shape == (n_batch, n_in, n_time, 1)
        numpy.testing.assert_almost_equal(out_v, in_v.reshape(out_v.shape))


def test_SplitDimsLayer_split_feature():
    n_batch, n_time, n_in = 7, 3, 5
    config = Config({"extern_data": {"data": {"dim": n_in}}})  # [B,T,D]
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict({"output": {"class": "split_dims", "from": "data", "axis": "F", "dims": [-1, 1]}})
        out = net.get_default_output_layer().output
        assert out.batch_shape == (None, None, n_in, 1)
        assert out.feature_dim_axis == 3  # https://github.com/rwth-i6/returnn/issues/704
        in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_time, n_in))
        out_v = session.run(out.placeholder, feed_dict={net.extern_data.data["data"].placeholder: in_v})
        assert isinstance(out_v, numpy.ndarray)
        assert out_v.shape == (n_batch, n_time, n_in, 1)
        numpy.testing.assert_almost_equal(out_v, in_v.reshape(out_v.shape))


def test_SplitDimsLayer_dim_tags():
    from returnn.tf.util.data import batch_dim

    for window_static_dim in [2, 1]:
        time_dim = SpatialDim("in-time")
        rem_dim = SpatialDim("rem-time")
        window_dim = FeatureDim("window", window_static_dim)
        feat_dim = FeatureDim("feat", 3)
        config = Config({"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}}})
        with make_scope():
            net = TFNetwork(config=config)
            net.construct_from_dict(
                {
                    "output": {
                        "class": "split_dims",
                        "from": "data",
                        "axis": time_dim,
                        "dims": [rem_dim, window_dim],
                        "out_shape": {batch_dim, rem_dim, window_dim, feat_dim},
                    }
                }
            )


def test_SplitDimsLayer_dim_tags_expand():
    from returnn.tf.util.data import batch_dim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feat", 3)
    expand_dim = SpatialDim("expand_dim", 1)
    config = Config({"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}}})
    with make_scope():
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {
                "output": {
                    "class": "split_dims",
                    "from": "data",
                    "axis": feat_dim,
                    "dims": [feat_dim, expand_dim],
                    "out_shape": {batch_dim, time_dim, feat_dim, expand_dim},
                }
            }
        )


def test_SplitDimsLayer_dim_tags_split_batch_simple():
    # https://github.com/rwth-i6/returnn/issues/908
    # https://github.com/rwth-i6/pytorch-to-returnn/pull/78
    from returnn.tf.util.data import batch_dim

    time_dim = SpatialDim("in-time")
    feat_dim = FeatureDim("feat", 3)
    config = Config({"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}}})
    with make_scope():
        net = TFNetwork(config=config)
        net.construct_from_dict({"output": {"class": "split_dims", "from": "data", "axis": batch_dim, "dims": [1, -1]}})


def test_SplitDimsLayer_dim_tags_split_batch():
    from returnn.tf.util.data import batch_dim

    time_dim = SpatialDim("in-time")
    feat_dim = FeatureDim("feat", 3)
    config = Config({"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}}})
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {
                "merge": {"class": "merge_dims", "from": "data", "axes": [batch_dim, feat_dim]},
                "output": {"class": "split_dims", "from": "merge", "axis": batch_dim, "dims": [batch_dim, feat_dim]},
            }
        )
        in_ = net.extern_data.get_default_input_data()
        assert in_.batch.is_global_batch()
        merged = net.layers["merge"].output
        out = net.layers["output"].output
        print(merged)
        assert not merged.batch.is_global_batch()
        assert len(merged.batch.virtual_dims) == 2
        b1, b2 = merged.batch.virtual_dims
        assert isinstance(b2, BatchInfo.FixedDim)
        assert b2.dim_tag == feat_dim
        print(out)
        assert in_.batch == out.batch and out.batch.is_global_batch()
        session.run(out.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_SplitDimsLayer_dyn_dim_tags_with_batch():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feat", 3)
    config = Config({"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}}})
    time2_dim = time_dim * 2
    print("time:", time_dim)
    print("time * 2:", time2_dim)
    net_dict = {
        "Flatten": {"class": "merge_dims", "from": "data", "axes": ["B", "T"], "keep_order": True},
        "Range_Length": {"class": "length", "axis": "B", "from": "data"},
        "Range_Length_1": {"class": "length", "axis": "T", "from": "data"},
        "Range_Reduce": {"class": "reduce", "mode": "max", "axes": ["B"], "from": "Range_Length_1"},
        "Range_mul": {"class": "combine", "kind": "mul", "out_shape": set(), "from": ["Range_Length", "Range_Reduce"]},
        "Range": {"class": "range_from_length", "from": "Range_mul"},  # [T]
        "Cat": {"class": "concat", "from": [("Range", "T"), ("Range", "T")]},
        "Unflatten__Unflatten_mul_unnamed_const": {"class": "constant", "value": 2},
        "Unflatten__Unflatten_mul": {
            "class": "combine",
            "kind": "mul",
            "out_shape": set(),
            "from": ["Range_Reduce", "Unflatten__Unflatten_mul_unnamed_const"],
        },
        "output": {"class": "split_dims", "from": "Cat", "axis": "T", "dims": [batch_dim, time2_dim]},
    }
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        in_ = net.extern_data.get_default_input_data()
        out = net.get_default_output_layer().output
        assert out.dim_tags == (batch_dim, time2_dim)
        x, x_size, y, y_size = session.run(
            (in_.placeholder, in_.size_placeholder.as_dict(), out.placeholder, out.size_placeholder.as_dict()),
            feed_dict=make_feed_dict(net.extern_data),
        )
        assert isinstance(x, numpy.ndarray)
        assert isinstance(x_size, dict) and set(x_size.keys()) == {0}
        x_size = x_size[0]
        assert isinstance(x_size, numpy.ndarray)
        n_batch, n_time, n_feat = x.shape
        assert x_size.shape == (n_batch,) and max(x_size) == n_time
        assert isinstance(y, numpy.ndarray)
        assert isinstance(y_size, dict) and set(y_size.keys()) == {0}
        y_size = y_size[0]
        assert isinstance(y_size, numpy.ndarray)
        assert y.shape == (n_batch, n_time * 2)
        assert y_size.shape == (n_batch,) and max(y_size) == n_time * 2


def test_ReshapeLayer():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    n_batch, n_time, n_dim = 2, 4, 3
    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feature", dimension=n_dim)
    config = Config({"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}}})  # [B,T,D]
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {
                "output": {
                    "class": "reshape",
                    "from": "data",
                    "in_dims": [time_dim, feat_dim],
                    "out_dims": [feat_dim, time_dim],
                }
            }
        )
        out = net.get_default_output_layer().output
        assert out.dim_tags == (batch_dim, feat_dim, time_dim)
        in_v = numpy.arange(n_time, dtype="float32")[None, :, None] + numpy.zeros(
            (n_batch, n_time, n_dim), dtype="float32"
        )
        feed_dict = {
            net.extern_data.data["data"].placeholder: in_v,
            net.extern_data.data["data"].size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
            net.extern_data.get_batch_info().dim: n_batch,
        }
        out_v = session.run(out.placeholder, feed_dict=feed_dict)
        print(out_v.shape)
        print(out_v)
        assert out_v.shape == (n_batch, n_dim, n_time)
        numpy.testing.assert_array_equal(in_v.reshape((n_batch, n_dim, n_time)), out_v)


def test_out_shape():
    # https://github.com/rwth-i6/returnn/issues/706
    # Note: Using SplitDimsLayer would also be nice to test out_shape. Or any layer which creates a new dim.
    # However, for that, we need https://github.com/rwth-i6/returnn/issues/597 first.
    from returnn.tf.util.data import batch_dim, VerifyOutShapeException

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feature", dimension=10)
    config = Config({"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}}})  # [B,T,D]
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {"output": {"class": "softmax_over_spatial", "from": "data", "out_shape": {batch_dim, time_dim, feat_dim}}}
        )
        out = net.get_default_output_layer().output
        session.run(out.placeholder, feed_dict=make_feed_dict(net.extern_data))
    with make_scope():
        other_feat_dim = FeatureDim("other-feature", dimension=10)
        net = TFNetwork(config=config)
        # noinspection PyBroadException
        try:
            net.construct_from_dict(
                {
                    "output": {
                        "class": "softmax_over_spatial",
                        "from": "data",
                        "out_shape": {batch_dim, time_dim, other_feat_dim},
                    }
                }
            )
        except VerifyOutShapeException as exc:
            print("Got expected exception: %r" % exc)
        else:
            raise Exception("Expected an exception but did not get any.")


def _check_MergeDimsLayer(
    session: tf.compat.v1.Session,
    in_data_opts: Dict[str, Any],
    in_static_shape: Tuple[int, ...],
    opts: Dict[str, Any],
    out_data_shape: Tuple[Optional[int], ...],
    out_static_shape: Tuple[int, ...],
    in_sizes: Optional[Dict[int, Tuple[int, ...]]] = None,
    out_sizes: Optional[Dict[int, Tuple[int, ...]]] = None,
) -> MergeDimsLayer:
    """
    :param session:
    :param in_data_opts:
    :param in_static_shape:
    :param opts: for MergeDimsLayer
    :param out_data_shape:
    :param out_static_shape:
    :param in_sizes:
    :param out_sizes:
    :return: layer
    """
    net = TFNetwork(extern_data=ExternData())
    rnd = numpy.random.RandomState(42)
    src = InternalLayer(name="src", network=net, output=Data(name="src", **in_data_opts))
    print("input:", src.output)
    src.output.placeholder = tf.constant(rnd.normal(size=in_static_shape).astype("float32"), dtype=tf.float32)
    src.output.size_placeholder = {}
    if src.output.batch_dim_axis is not None:
        n_batch = in_static_shape[src.output.batch_dim_axis]
        for axis, dim in enumerate(src.output.batch_shape):
            axis_wo_b = src.output.get_batch_axis_excluding_batch(axis)
            if dim is None and axis_wo_b is not None:
                if in_sizes and axis_wo_b in in_sizes:
                    src.output.size_placeholder[axis_wo_b] = tf.constant(in_sizes[axis_wo_b])
                else:
                    src.output.size_placeholder[axis_wo_b] = tf.fill([n_batch], in_static_shape[axis])

    opts = opts.copy()
    print("opts:", opts)
    opts.update({"network": net, "name": "merge_dims_test", "sources": [src]})
    out_data = MergeDimsLayer.get_out_data_from_opts(**opts)
    out_data.sanity_check(ignore_placeholder=True)  # placeholder might be overwritten later
    assert out_data.shape == out_data_shape
    layer = MergeDimsLayer(output=out_data, **opts)
    assert layer.output.shape == out_data_shape
    out_np, size_placeholder = session.run([layer.output.placeholder, layer.output.size_placeholder.as_dict()])
    print("output:", out_data)
    assert out_np.shape == out_static_shape

    if out_sizes:
        assert sorted(size_placeholder.keys()) == sorted(out_sizes)
        for k in size_placeholder.keys():
            numpy.testing.assert_array_equal(size_placeholder[k], out_sizes[k])

    return layer


def test_MergeDimsLayer_basic():
    with make_scope() as session:
        _check_MergeDimsLayer(
            session,
            {"shape": (4, 7), "time_dim_axis": None},
            (2, 4, 7),
            {"axes": ["dim:4", "dim:7"]},
            (4 * 7,),
            (2, 4 * 7),
        )
        _check_MergeDimsLayer(
            session,
            {"shape": (4, None, 7), "time_dim_axis": None},
            (2, 4, 3, 7),
            {"axes": ["dim:4", "dim:7"]},
            (None, 4 * 7),
            (2, 3, 4 * 7),
        )
        _check_MergeDimsLayer(
            session,
            {"shape": (4, None, 7), "time_dim_axis": 2},
            (2, 4, 3, 7),
            {"axes": ["dim:4", "dim:7"]},
            (None, 4 * 7),
            (2, 3, 4 * 7),
        )
        _check_MergeDimsLayer(
            session,
            {"shape": (1, None), "time_dim_axis": 2, "feature_dim_axis": 1},
            (2, 1, 4),
            {"axes": ["F", "T"]},
            (None,),
            (2, 4),
        )


def test_MergeDimsLayer_size_placeholder():
    with make_scope() as session:
        _check_MergeDimsLayer(
            session,
            {"shape": (None, 2), "time_dim_axis": 1, "feature_dim_axis": 2},
            (3, 4, 2),
            {"axes": ["T", "F"]},
            (None,),
            (3, 8),
            in_sizes={0: (4, 2, 1)},
            out_sizes={0: (8, 4, 2)},
        )


def test_MergeDimsLayer_batch_time_ext():
    with make_scope() as session:
        n_batch = 11
        n_time = 13
        _check_MergeDimsLayer(
            session,
            {"shape": (None, 5, 3)},
            (n_batch, n_time, 5, 3),
            {"axes": ["B", "T"]},
            (5, 3),
            (n_batch * n_time, 5, 3),
        )


def test_MergeDimsLayer_batch_time_time_major():
    with make_scope() as session:
        n_batch = 11
        n_time = 13
        layer = _check_MergeDimsLayer(
            session,
            {"shape": (None, 5), "time_dim_axis": 0, "batch_dim_axis": 1},
            (n_time, n_batch, 5),
            {"axes": ["B", "T"]},
            (5,),
            (n_time * n_batch, 5),
        )
        assert layer.output.batch_dim_axis == 0
        assert layer.output.time_dim_axis is None


def test_MergeDimsLayer_batch_time_time_major_ext():
    with make_scope() as session:
        n_batch = 11
        n_time = 13
        layer = _check_MergeDimsLayer(
            session,
            {"shape": (None, 5, 3), "time_dim_axis": 0, "batch_dim_axis": 1},
            (n_time, n_batch, 5, 3),
            {"axes": ["B", "T"]},
            (5, 3),
            (n_time * n_batch, 5, 3),
        )
        assert layer.output.batch_dim_axis == 0
        assert layer.output.time_dim_axis is None  # Note: This behavior was changed.


def test_MergeDimsLayer_except_time_ext():
    with make_scope() as session:
        n_batch = 11
        n_time = 13
        layer = _check_MergeDimsLayer(
            session,
            {"shape": (3, None, 5), "time_dim_axis": 2},
            (n_batch, 3, n_time, 5),
            {"axes": ["dim:3", "dim:5"]},
            (None, 15),
            (n_batch, n_time, 15),
        )
        assert layer.output.batch_dim_axis == 0 and layer.output.time_dim_axis == 1


def test_MergeDimsLayer_static_time():
    with make_scope() as session:
        n_batch = 11
        layer = _check_MergeDimsLayer(
            session,
            {"shape": (3, 5), "time_dim_axis": 1},
            (n_batch, 3, 5),
            {"axes": ["dim:3", "dim:5"]},
            (15,),
            (n_batch, 15),
        )
        assert layer.output.batch_dim_axis == 0 and layer.output.feature_dim_axis == 1
        assert layer.output.time_dim_axis is None


def test_MergeDimsLayer_feat_static_static():
    with make_scope() as session:
        n_batch = 11
        layer = _check_MergeDimsLayer(
            session,
            {"shape": (None, 8, 2, 3), "feature_dim_axis": 2},
            (n_batch, 7, 8, 2, 3),
            {"axes": ["F", "dim:2"]},
            (None, 16, 3),
            (n_batch, 7, 16, 3),
        )
        assert (layer.output.batch_dim_axis, layer.output.time_dim_axis, layer.output.feature_dim_axis) == (0, 1, 2)


def test_MergeDimsLayer_dim_tags():
    n_batch = 3
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData())
        rnd = numpy.random.RandomState(42)

        src_data = Data("input", shape=(None, None, None, 1), feature_dim_axis=None)
        input_static_shape = (n_batch, 7, 1, 2, 1)
        src_data.placeholder = tf.constant(rnd.normal(size=input_static_shape).astype("float32"), dtype=tf.float32)
        src_data.size_placeholder = {}
        from returnn.tf.util.basic import Dim

        # map axis_wo_batch -> (tag description, dyn_size)
        tag_names_with_dyn_size = {
            0: ("key-chunk", [4, 2, 3]),
            1: ("key-window", [1, 1, 1]),
            2: ("att-heads", [2, 2, 2]),
        }
        for axis_wo_batch, (description, dyn_size) in tag_names_with_dyn_size.items():
            tag = Dim(description=description, kind=Dim.Types.Spatial, dimension=None)
            dyn_size = tf.constant(dyn_size)
            tag.set_tag_on_size_tensor(dyn_size)
            src_data.size_placeholder[axis_wo_batch] = dyn_size
        print("in data:", src_data)  # should be [B,T|'key-chunk',1|'key-window',2|'att-heads',1]
        assert (
            src_data.get_axis_by_tag_name("key-chunk") == 1
            and src_data.get_axis_by_tag_name("key-window") == 2
            and src_data.get_axis_by_tag_name("att-heads") == 3
        )

        merge_axes = ["stag:key-window", "dim:1"]
        print("merge axes:", merge_axes)

        src = InternalLayer(name="src", network=net, output=src_data)
        opts = {"network": net, "name": "merge_dims_test", "sources": [src], "axes": merge_axes}
        out_data = MergeDimsLayer.get_out_data_from_opts(**opts)
        out_data.sanity_check(ignore_placeholder=True)  # placeholder might be overwritten later
        print("template out data:", out_data)  # should be [B,T|'key-chunk',1|<anything>,2|'att-heads',1]
        assert out_data.shape == src_data.shape[:-1]
        assert out_data.get_axis_by_tag_name("key-chunk") == 1 and out_data.get_axis_by_tag_name("att-heads") == 3

        layer = MergeDimsLayer(output=out_data, **opts)
        layer.output.sanity_check()
        out_data = layer.output
        print("layer out data:", out_data)
        assert out_data.shape == src_data.shape[:-1]
        assert out_data.get_axis_by_tag_name("key-chunk") == 1 and out_data.get_axis_by_tag_name("att-heads") == 3


def test_MergeDimsLayer_SplitBatchTimeLayer_time_major():
    n_batch = 3
    n_time = 4
    n_input_dim = 5
    # Time major
    input_data = numpy.arange(n_time * n_batch * n_input_dim).reshape((n_time, n_batch, n_input_dim)).astype("float32")
    with make_scope() as session:
        seq_lens = [n_time, n_time - 1, n_time - 2]
        assert len(seq_lens) == n_batch and all([s > 0 for s in seq_lens])
        net = TFNetwork(extern_data=ExternData(), config=Config({"debug_print_layer_output_template": True}))
        input_layer = net.add_layer(
            "input",
            InternalLayer,
            output=Data(
                name="input",
                shape=(None, n_input_dim),
                time_dim_axis=0,
                batch_dim_axis=1,
                placeholder=tf.constant(input_data),
                size_placeholder={0: tf.constant(seq_lens)},
            ),
        )
        assert input_layer.output.is_time_major
        net.construct_from_dict(
            {
                "merge_dims": {"class": "merge_dims", "from": "input", "axes": ["B", "T"]},
                "split_dims": {"class": "split_batch_time", "from": "merge_dims", "base": "input"},
                "output": {"class": "copy", "from": "split_dims"},
            }
        )
        output = net.get_default_output_layer().output
        # Depending on implementation, output could be batch-major or time-major.
        output = output.copy_as_time_major()  # such that we can compare easily to input_data
        assert output.is_time_major and output.shape == (None, n_input_dim)
        output_data = session.run(output.placeholder)
        numpy.testing.assert_almost_equal(input_data, output_data)


def test_MergeDimsLayer_SplitBatchTimeLayer_two_time_axes():
    n_dim = 11
    with make_scope() as session:
        net = TFNetwork(
            config=Config(
                {"extern_data": {"data": {"shape": (None, None, n_dim)}}, "debug_print_layer_output_template": True}
            )
        )
        feed_dict = make_feed_dict(net.extern_data)
        net.construct_from_dict(
            {
                "merge_dims": {"class": "merge_dims", "from": "data", "axes": ["B", "T"]},
                "split_dims": {"class": "split_batch_time", "from": "merge_dims", "base": "data"},
                "output": {"class": "copy", "from": "split_dims"},
            }
        )
        input_data = net.extern_data.get_default_input_data()
        print("input_data:", input_data)
        assert set(input_data.size_placeholder.keys()) == {0, 1}
        assert input_data.size_placeholder[0].name != input_data.size_placeholder[1].name
        assert input_data.get_size_dim_tag(0) != input_data.get_size_dim_tag(1)
        merged_data = net.layers["merge_dims"].output
        print("merged_data:", merged_data)
        assert set(merged_data.size_placeholder.keys()) == {0}
        assert merged_data.get_size_dim_tag(0) != input_data.get_size_dim_tag(0)
        assert merged_data.get_size_dim_tag(0) == input_data.get_size_dim_tag(1)  # like beam-search, still same dim-tag
        assert merged_data.size_placeholder[0] is not input_data.size_placeholder[1]  # but different sizes
        output_data = net.get_default_output_layer().output
        output_data = output_data.copy_as_batch_major()
        print("output_data:", output_data)
        assert output_data.shape == (None, None, n_dim)
        assert output_data.get_size_dim_tag(0) == input_data.get_size_dim_tag(0)
        assert output_data.get_size_dim_tag(1) == input_data.get_size_dim_tag(1)
        input_value = session.run(input_data.placeholder, feed_dict=feed_dict)
        merged_value = session.run(merged_data.placeholder, feed_dict=feed_dict)
        output_value = session.run(output_data.placeholder, feed_dict=feed_dict)
        assert input_value.shape == output_value.shape
        assert input_value.shape[-1] == n_dim
        print("input_value.shape:", input_value.shape)
        n_batch, n_time0, n_time1, _ = input_value.shape
        numpy.testing.assert_almost_equal(input_value, output_value)
        print("merged_value.shape:", merged_value.shape)
        assert merged_value.shape == (n_batch * n_time0, n_time1, n_dim)
        numpy.testing.assert_almost_equal(input_value, merged_value.reshape(input_value.shape))
        merged_size = session.run(merged_data.size_placeholder[0], feed_dict=feed_dict)
        input_size0, input_size1 = session.run(
            (input_data.size_placeholder[0], input_data.size_placeholder[1]), feed_dict=feed_dict
        )
        assert input_size0.shape == input_size1.shape == (n_batch,)
        assert merged_size.shape == (n_batch * n_time0,)
        merged_size = merged_size.reshape(n_batch, n_time0)
        assert (merged_size == input_size1[:, None]).all()


def test_MergeDimsLayer_simple_feat():
    n_batch, n_time, n_in1, n_in2 = 7, 3, 10, 32
    config = Config(
        {
            "extern_data": {"data": {"shape": (None, n_in1, n_in2)}},
            "debug_print_layer_output_template": True,
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {"output": {"class": "merge_dims", "axes": ["dim:%i" % n_in1, "dim:%i" % n_in2], "from": "data:data"}}
        )
        out_t = net.get_default_output_layer().output.placeholder
        assert out_t.shape.as_list() == [None, None, n_in1 * n_in2]
        in_v = (
            numpy.arange(0, n_batch * n_time * n_in1 * n_in2).astype("float32").reshape((n_batch, n_time, n_in1, n_in2))
        )
        out_v = session.run(out_t, feed_dict={net.extern_data.data["data"].placeholder: in_v})
        assert isinstance(out_v, numpy.ndarray)
        assert out_v.shape == (n_batch, n_time, n_in1 * n_in2)
        numpy.testing.assert_almost_equal(out_v, in_v.reshape(out_v.shape))


def test_MergeDimsLayer_2d_dynamic_merge_axis():
    # https://github.com/rwth-i6/returnn/issues/662
    from returnn.tf.util.data import batch_dim, Dim, ImplicitDynSizeDim

    time_dim = SpatialDim("T")
    feat_dim = FeatureDim("F", dimension=5)
    with make_scope() as session:
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}},
            }
        )
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "start0": {"class": "range_in_axis", "from": "data", "axis": "b", "out_shape": {batch_dim}},
                "start1": {
                    "class": "range_in_axis",
                    "from": "data",
                    "axis": "t",
                    "out_shape": {time_dim, ImplicitDynSizeDim(batch_dim)},
                },
                "start": {
                    "class": "combine",
                    "from": ["start0", "start1"],
                    "kind": "add",
                    "out_shape": {batch_dim, time_dim},
                },
                "slices": {
                    "class": "slice_nd",
                    "from": "data",
                    "start": "start",
                    "size": None,
                },  # [B,T[B],slice[B,T],D]
                "output": {"class": "merge_dims", "from": "slices", "axes": ["f", "stag:slice"]},  # [B,T[B],merge[B,T]]
            }
        )
        slices_layer = net.get_layer("slices")
        assert isinstance(slices_layer, SliceNdLayer)
        merge_layer = net.get_layer("output")
        assert isinstance(merge_layer, MergeDimsLayer)
        assert slices_layer.output.dim_tags[:2] == merge_layer.output.dim_tags[:2]
        assert merge_layer.output.dim_tags[0].is_batch_dim()
        assert merge_layer.output.dim_tags[1] == net.extern_data.get_default_input_data().get_time_dim_tag()
        assert merge_layer.output.dim_tags[2].dyn_size_ext.dim_tags == merge_layer.output.dim_tags[:2]  # [B,T]
        assert slices_layer.output.dim_tags[2].dyn_size_ext.dim_tags == slices_layer.output.dim_tags[:2]  # [B,T]
        assert slices_layer.output.dim_tags[1].dyn_size_ext.dim_tags == slices_layer.output.dim_tags[:1]  # [B]
        out, merged_time, sliced_time = session.run(
            (
                merge_layer.output.placeholder,
                merge_layer.output.dim_tags[2].dyn_size_ext.placeholder,
                slices_layer.output.dim_tags[2].dyn_size_ext.placeholder,
            ),
            make_feed_dict(net.extern_data),
        )
        assert isinstance(out, numpy.ndarray)
        assert isinstance(merged_time, numpy.ndarray) and isinstance(sliced_time, numpy.ndarray)
        assert out.shape == merged_time.shape + (numpy.max(merged_time),)
        assert sliced_time.shape == merged_time.shape
        assert numpy.all(sliced_time * feat_dim.dimension == merged_time)


def test_MergeDimsLayer_modified_time_dim():
    n_batch, n_time, n_in = 3, 7, 2
    config = Config(
        {
            "extern_data": {"data": {"dim": n_in}},
            "debug_print_layer_output_template": True,
            "behavior_version": 12,
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {
                "conv": {
                    "class": "conv",
                    "n_out": n_in,
                    "filter_size": (3,),
                    "strides": (2,),
                    "padding": "valid",
                    "from": "data:data",
                },
                "output": {"class": "merge_dims", "from": "conv", "axes": ["B", "T"], "keep_order": True},
            }
        )
        out = net.get_default_output_layer().output
        net.initialize_params(session)
        session.run(out.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_MergeDimsLayer_unspecified_out_dim():
    # https://github.com/rwth-i6/returnn/issues/955
    # https://github.com/rwth-i6/returnn_common/issues/117
    config = Config(
        {
            "extern_data": {"data": {"shape": (None, 3, 5)}},
        }
    )
    out_dim = SpatialDim("out")
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {
                "output": {
                    "class": "merge_dims",
                    "from": "data",
                    "axes": ["dim:3", "dim:5"],
                    "keep_order": True,
                    "out_dim": out_dim,
                },
            }
        )


def test_FlattenBatchLayer():
    from returnn.tf.util.data import BatchInfo

    n_batch, n_time, n_in = 3, 4, 2
    config = Config(
        {
            "extern_data": {"data": {"dim": n_in, "dtype": "int32"}},
            "debug_print_layer_output_template": True,
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict({"output": {"class": "flatten_batch", "batch_major": False, "from": "data:data"}})
        in_data = net.extern_data.data["data"]
        out_data = net.get_default_output_layer().output
        assert out_data.batch_shape == (None, n_in) and not out_data.size_placeholder
        assert len(out_data.batch.virtual_dims) == 2
        batch_flat_dim0, batch_flat_dim1 = out_data.batch.virtual_dims
        assert isinstance(batch_flat_dim0, BatchInfo.PackedDim)
        assert isinstance(batch_flat_dim1, BatchInfo.GlobalBatchDim)
        assert batch_flat_dim0.sizes is in_data.size_placeholder[0]
        out_t = net.get_default_output_layer().output.placeholder
        assert out_t.shape.as_list() == [None, n_in]
        in_v = numpy.arange(0, n_batch * n_time * n_in).reshape((n_time, n_batch, n_in)).transpose(1, 0, 2)
        in_seq_lens = [4, 3, 2]
        out_v = session.run(out_t, feed_dict={in_data.placeholder: in_v, in_data.size_placeholder[0]: in_seq_lens})
        assert isinstance(out_v, numpy.ndarray)
        assert out_v.shape == (sum(in_seq_lens), n_in)
        assert out_v.tolist() == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [18, 19]]


def test_UnflattenBatchLayer():
    n_batch, n_time, n_in = 3, 4, 2
    config = Config(
        {
            "extern_data": {"data": {"dim": n_in, "dtype": "int32"}},
            "debug_print_layer_output_template": True,
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {
                "flatten": {"class": "flatten_batch", "from": "data:data"},
                "output": {"class": "unflatten_batch", "from": "flatten"},
            }
        )
        in_data = net.extern_data.data["data"]
        out_data = net.get_default_output_layer().output
        assert out_data.batch_shape == (None, None, n_in) and out_data.size_placeholder is not None
        out_t = net.get_default_output_layer().output.placeholder
        assert out_t.shape.as_list() == [None, None, n_in]
        in_v = numpy.arange(0, n_batch * n_time * n_in).reshape((n_time, n_batch, n_in)).transpose(1, 0, 2)
        in_seq_lens = [4, 3, 2]
        out_v = session.run(out_t, feed_dict={in_data.placeholder: in_v, in_data.size_placeholder[0]: in_seq_lens})
        for b, seq_len in enumerate(in_seq_lens):
            in_v[b, seq_len:, :] = 0
        assert isinstance(out_v, numpy.ndarray)
        numpy.testing.assert_equal(out_v, in_v)


def test_UnflattenBatchLayer_time_major():
    n_batch, n_time, n_in = 3, 4, 2
    config = Config(
        {
            "extern_data": {"data": {"dim": n_in, "dtype": "int32"}},
            "debug_print_layer_output_template": True,
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {
                "flatten": {"class": "flatten_batch", "batch_major": False, "from": "data:data"},
                "output": {"class": "unflatten_batch", "from": "flatten"},
            }
        )
        in_data = net.extern_data.data["data"]
        out_data = net.get_default_output_layer().output
        assert out_data.batch_shape == (None, None, n_in) and out_data.size_placeholder is not None
        out_t = net.get_default_output_layer().output.placeholder
        assert out_t.shape.as_list() == [None, None, n_in]
        in_v = numpy.arange(0, n_batch * n_time * n_in).reshape((n_time, n_batch, n_in)).transpose(1, 0, 2)
        in_seq_lens = [4, 3, 2]
        out_v = session.run(out_t, feed_dict={in_data.placeholder: in_v, in_data.size_placeholder[0]: in_seq_lens})
        for b, seq_len in enumerate(in_seq_lens):
            in_v[b, seq_len:, :] = 0
        assert isinstance(out_v, numpy.ndarray)
        numpy.testing.assert_equal(out_v, in_v.transpose(1, 0, 2))


def test_SwitchLayer_const_no_time():
    config = Config(
        {
            "extern_data": {
                "data": {"dim": 3, "sparse": True, "shape": ()},
            },
            "debug_print_layer_output_template": True,
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "input_eq_0": {"class": "compare", "from": "data", "value": 0},  # (B,)
                "const0": {"class": "constant", "value": 0},
                "const1": {"class": "constant", "value": 1},
                "switch": {"class": "switch", "condition": "input_eq_0", "true_from": "const1", "false_from": "const0"},
                "output": {"class": "copy", "from": "switch"},
            }
        )
        net.print_network_info()
        feed_dict = make_feed_dict(net.extern_data.data.values())
        out = session.run(net.get_default_output_layer().output.placeholder, feed_dict=feed_dict)
        print(out)


def test_SwitchLayer_const():
    config = Config(
        {
            "extern_data": {
                "data": {"dim": 3, "sparse": True},
            },
            "debug_print_layer_output_template": True,
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "input_eq_0": {"class": "compare", "from": "data", "value": 0},  # (B,T)
                "const0": {"class": "constant", "value": 0},
                "const1": {"class": "constant", "value": 1},
                "switch": {"class": "switch", "condition": "input_eq_0", "true_from": "const1", "false_from": "const0"},
                "output": {"class": "copy", "from": "switch"},
            }
        )
        net.print_network_info()
        feed_dict = make_feed_dict(net.extern_data.data.values())
        out = session.run(net.get_default_output_layer().output.placeholder, feed_dict=feed_dict)
        print(out)


def test_SwitchLayer_masking():
    config = Config(
        {
            "extern_data": {
                "data": {"dim": 3, "sparse": False},
            },
            "debug_print_layer_output_template": True,
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "projected": {"class": "gather", "from": "data", "axis": "F", "position": 0},  # (B,T)
                "mask": {"class": "compare", "from": "projected", "value": 0, "kind": "greater"},  # (B,T)
                "switch": {"class": "switch", "condition": "mask", "true_from": "data", "false_from": float("-inf")},
                "output": {"class": "copy", "from": "switch"},
            }
        )
        net.print_network_info()
        feed_dict = make_feed_dict(net.extern_data.data.values())
        out = session.run(net.get_default_output_layer().output.placeholder, feed_dict=feed_dict)
        print(out)


def test_SwitchLayer_template_const_from():
    net = TFNetwork(extern_data=ExternData())
    batch_dim = Dim(kind=Dim.Types.Batch, description="batch", dimension=None)
    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feature", dimension=2)
    # [T]
    condition = InternalLayer(
        network=net,
        name="condition",
        output=Data("condition_output", time_dim_axis=0, feature_dim_axis=None, dim_tags=[time_dim]),
    )
    true_from = 42
    # [B,F|2,T]
    false_from = InternalLayer(
        network=net,
        name="false_from",
        output=Data(
            "false_from_output",
            batch_dim_axis=0,
            time_dim_axis=2,
            feature_dim_axis=1,
            dim_tags=[batch_dim, feat_dim, time_dim],
        ),
    )

    # should be [B,F|2,T]
    switch = SwitchLayer.get_out_data_from_opts(
        "switch", condition=condition, true_from=true_from, false_from=false_from
    )
    assert switch.batch_ndim == 3
    assert switch.batch_dim_axis == 0 and switch.time_dim_axis == 2 and switch.feature_dim_axis == 1
    assert switch.dim == 2


def test_TopKLayer_single_axis():
    config = Config(
        {
            "extern_data": {"data": {"shape": (None, 5)}},
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "output": {"class": "top_k", "from": "data", "axis": "F", "k": 2},  # (B,T)
            }
        )
        in_data = net.extern_data.data["data"]
        in_np = numpy.arange(2 * 3 * 5).reshape((2, 3, 5)).astype(in_data.dtype)
        in_sizes = [3, 2]
        values = net.get_layer("output").output
        indices = net.get_layer("output/indices").output
        values_np, indices_np = session.run(
            (values.placeholder, indices.placeholder),
            feed_dict={in_data.placeholder: in_np, in_data.get_sequence_lengths(): in_sizes},
        )
        print("inputs:\n", in_np)
        print("values:\n", values_np)
        print("indices:\n", indices_np)
        assert (indices_np == numpy.array([4, 3])[None, None]).all()


def test_TopKLayer_two_axes():
    config = Config(
        {
            "extern_data": {"data": {"shape": (3, 5)}},
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "output": {"class": "top_k", "from": "data", "axis": ("dim:3", "dim:5"), "k": 2},  # (B,T)
            }
        )
        in_data = net.extern_data.data["data"]
        in_np = numpy.arange(2 * 3 * 5).reshape((2, 3, 5)).astype(in_data.dtype)
        values = net.get_layer("output").output
        indices0 = net.get_layer("output/indices0").output
        indices1 = net.get_layer("output/indices1").output
        values_np, indices0_np, indices1_np = session.run(
            (values.placeholder, indices0.placeholder, indices1.placeholder), feed_dict={in_data.placeholder: in_np}
        )
        print("inputs:\n", in_np)
        print("values:\n", values_np)
        print("indices0:\n", indices0_np)
        print("indices1:\n", indices1_np)
        assert (indices0_np == 2).all()
        assert (indices1_np == numpy.array([4, 3])[None]).all()


def test_TopKLayer_in_cond_kdim():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feat", 5)

    config = Config(
        dict(
            extern_data={
                "data": {
                    "dim_tags": (batch_dim, time_dim, feat_dim),
                    "dtype": "float32",
                    "available_for_inference": True,
                }
            },
            debug_runtime_sanity_checks=True,
            debug_print_layer_output_shape=True,
            debug_print_layer_output=True,
        )
    )

    k_dim = SpatialDim("feature_masking:num")

    net_dict = {
        "specaugment_v2": {
            "class": "subnetwork",
            "from": [],
            "subnetwork": {
                "train_flag": {"class": "train_flag"},
                "cond": {
                    "class": "cond",
                    "from": [],
                    "condition": "train_flag",
                    "true_layer": {
                        "class": "subnetwork",
                        "from": [],
                        "subnetwork": {
                            "k": {
                                "class": "random",
                                "shape": (),
                                "distribution": "uniform",
                                "dtype": "int32",
                                "minval": 2,
                                "maxval": 4,
                            },
                            "scores": {
                                "class": "reduce",
                                "from": "base:base:data",
                                "mode": "max",
                                "axis": time_dim,
                                "out_shape": {batch_dim, feat_dim},
                            },
                            "top_k": {
                                "class": "top_k",
                                "from": "scores",
                                "axis": feat_dim,
                                "k": "k",
                                "k_dim": k_dim,
                                "sorted": True,
                                "out_shape": {batch_dim, k_dim},
                            },
                            "range_in_axis": {
                                "class": "range_in_axis",
                                "from": "top_k/indices",
                                "axis": k_dim,
                                "out_shape": {k_dim},
                            },
                            "output": {"class": "reduce", "from": "range_in_axis", "mode": "max", "axis": k_dim},
                        },
                    },
                    "false_layer": {
                        "class": "subnetwork",
                        "from": [],
                        "subnetwork": {"output": {"class": "constant", "value": -1, "dtype": "int32", "out_shape": {}}},
                    },
                    "name_scope": "",
                },
                "output": {"class": "copy", "from": "cond"},
            },
        },
        "output": {
            "class": "copy",
            "from": "specaugment_v2",
        },
    }

    with make_scope() as session:
        train_flag = tf_util.get_global_train_flag_placeholder()
        net = TFNetwork(config=config, train_flag=train_flag)
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        feed_dict = make_feed_dict(net.extern_data)
        feed_dict[train_flag] = False
        res_eval = session.run(net.get_default_output_layer().output.placeholder, feed_dict=feed_dict)
        print("eval:", res_eval)
        feed_dict[train_flag] = True
        res_train = session.run(net.get_default_output_layer().output.placeholder, feed_dict=feed_dict)
        print("train:", res_train)


def test_specaugment_pure_returnn_reduced_with_cond():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim, ImplicitSparseDim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feat", 50)

    config = Config(
        dict(
            extern_data={
                "data": {
                    "dim_tags": (batch_dim, time_dim, feat_dim),
                    "dtype": "float32",
                    "available_for_inference": True,
                }
            },
            debug_runtime_sanity_checks=True,
            debug_print_layer_output_shape=True,
        )
    )

    specaugment_v2_cond_true_feature_masking_top_k_k_dim = SpatialDim("feature_masking:num")

    net_dict = {
        "specaugment_v2": {
            "class": "subnetwork",
            "from": [],
            "subnetwork": {
                "ones": {"class": "constant", "value": True, "shape": (), "dtype": "bool"},
                "cond": {
                    "class": "cond",
                    "from": [],
                    "condition": "ones",
                    "true_layer": {
                        "class": "subnetwork",
                        "from": [],
                        "subnetwork": {
                            "dim_value": {
                                "class": "subnetwork",
                                "from": [],
                                "subnetwork": {
                                    "length": {
                                        "class": "length",
                                        "from": "base:base:base:data:data",
                                        "axis": time_dim,
                                        "out_shape": {batch_dim},
                                    },
                                    "reduce": {
                                        "class": "reduce",
                                        "from": "length",
                                        "mode": "max",
                                        "axis": (batch_dim,),
                                        "out_shape": {},
                                    },
                                    "output": {"class": "copy", "from": "reduce", "out_shape": {}},
                                },
                                "out_shape": {},
                            },
                            "constant": {"class": "constant", "value": 2},
                            "minimum": {
                                "class": "combine",
                                "from": ["constant", "dim_value"],
                                "kind": "minimum",
                                "out_shape": {},
                            },
                            "constant_0": {"class": "constant", "value": 100},
                            "floordiv": {
                                "class": "combine",
                                "from": ["dim_value", "constant_0"],
                                "kind": "floordiv",
                                "out_shape": {},
                            },
                            "constant_1": {"class": "constant", "value": 2},
                            "maximum": {
                                "class": "combine",
                                "from": ["floordiv", "constant_1"],
                                "kind": "maximum",
                                "out_shape": {},
                            },
                            "constant_2": {"class": "constant", "value": 4},
                            "mul": {
                                "class": "combine",
                                "from": ["maximum", "constant_2"],
                                "kind": "mul",
                                "out_shape": {},
                            },
                            "minimum_0": {
                                "class": "combine",
                                "from": ["mul", "dim_value"],
                                "kind": "minimum",
                                "out_shape": {},
                            },
                            "feature_masking": {
                                "class": "subnetwork",
                                "from": [],
                                "subnetwork": {
                                    "random": {
                                        "class": "subnetwork",
                                        "from": [],
                                        "subnetwork": {
                                            "random": {
                                                "class": "random",
                                                "shape": [batch_dim],
                                                "distribution": "uniform",
                                                "minval": 2,
                                                "maxval": 6,
                                                "dtype": "int32",
                                            },
                                            "output": {"class": "copy", "from": "random", "out_shape": {batch_dim}},
                                        },
                                        "out_shape": {batch_dim},
                                    },
                                    "random_0": {
                                        "class": "subnetwork",
                                        "from": [],
                                        "subnetwork": {
                                            "random": {
                                                "class": "random",
                                                "shape": [batch_dim, feat_dim],
                                                "distribution": "uniform",
                                                "minval": 0.0,
                                                "maxval": 1.0,
                                            },
                                            "output": {
                                                "class": "copy",
                                                "from": "random",
                                                "out_shape": {batch_dim, feat_dim},
                                            },
                                        },
                                        "out_shape": {batch_dim, feat_dim},
                                    },
                                    "log": {
                                        "class": "activation",
                                        "from": "random_0",
                                        "activation": "log",
                                        "out_shape": {batch_dim, feat_dim},
                                    },
                                    "negative": {
                                        "class": "activation",
                                        "from": "log",
                                        "activation": "negative",
                                        "out_shape": {batch_dim, feat_dim},
                                    },
                                    "log_0": {
                                        "class": "activation",
                                        "from": "negative",
                                        "activation": "log",
                                        "out_shape": {batch_dim, feat_dim},
                                    },
                                    "negative_0": {
                                        "class": "activation",
                                        "from": "log_0",
                                        "activation": "negative",
                                        "out_shape": {batch_dim, feat_dim},
                                    },
                                    "reduce": {
                                        "class": "reduce",
                                        "from": "random",
                                        "mode": "max",
                                        "axis": (batch_dim,),
                                        "out_shape": {},
                                    },
                                    "top_k": {
                                        "class": "top_k",
                                        "from": "negative_0",
                                        "axis": feat_dim,
                                        "k": "reduce",
                                        "k_dim": specaugment_v2_cond_true_feature_masking_top_k_k_dim,
                                        "sorted": True,
                                        "out_shape": {batch_dim, specaugment_v2_cond_true_feature_masking_top_k_k_dim},
                                    },
                                    "loop": {
                                        "class": "rec",
                                        "from": [],
                                        "unit": {
                                            "rec_unstack": {
                                                "class": "rec_unstack",
                                                "from": "base:range_in_axis",
                                                "axis": specaugment_v2_cond_true_feature_masking_top_k_k_dim,
                                                "out_shape": {},
                                            },
                                            "gather": {
                                                "class": "gather",
                                                "from": "base:top_k/indices",
                                                "position": "rec_unstack",
                                                "axis": specaugment_v2_cond_true_feature_masking_top_k_k_dim,
                                                "out_shape": {batch_dim, ImplicitSparseDim(feat_dim)},
                                            },
                                            "_mask_v2": {
                                                "class": "subnetwork",
                                                "from": [],
                                                "subnetwork": {
                                                    "length": {
                                                        "class": "length",
                                                        "from": "base:prev:_mask_v2",
                                                        "axis": feat_dim,
                                                        "out_shape": {},
                                                    },
                                                    "random": {
                                                        "class": "subnetwork",
                                                        "from": [],
                                                        "subnetwork": {
                                                            "random": {
                                                                "class": "random",
                                                                "shape": (batch_dim,),
                                                                "distribution": "uniform",
                                                                "minval": 1,
                                                                "maxval": 11,
                                                                "dtype": "int32",
                                                            },
                                                            "output": {
                                                                "class": "copy",
                                                                "from": "random",
                                                                "out_shape": {batch_dim},
                                                            },
                                                        },
                                                        "out_shape": {batch_dim},
                                                    },
                                                    "add": {
                                                        "class": "combine",
                                                        "from": ["base:gather", "random"],
                                                        "kind": "add",
                                                        "out_shape": {batch_dim, ImplicitSparseDim(feat_dim)},
                                                    },
                                                    "minimum": {
                                                        "class": "combine",
                                                        "from": ["add", "length"],
                                                        "kind": "minimum",
                                                        "out_shape": {batch_dim, ImplicitSparseDim(feat_dim)},
                                                    },
                                                    "range_in_axis": {
                                                        "class": "range_in_axis",
                                                        "from": "base:prev:_mask_v2",
                                                        "axis": feat_dim,
                                                        "out_shape": {feat_dim},
                                                    },
                                                    "greater_equal": {
                                                        "class": "compare",
                                                        "from": ["range_in_axis", "base:gather"],
                                                        "kind": "greater_equal",
                                                        "allow_broadcast_all_sources": True,
                                                        "out_shape": {batch_dim, feat_dim},
                                                    },
                                                    "less": {
                                                        "class": "compare",
                                                        "from": ["range_in_axis", "minimum"],
                                                        "kind": "less",
                                                        "allow_broadcast_all_sources": True,
                                                        "out_shape": {batch_dim, feat_dim},
                                                    },
                                                    "logical_and": {
                                                        "class": "combine",
                                                        "from": ["greater_equal", "less"],
                                                        "kind": "logical_and",
                                                        "out_shape": {batch_dim, feat_dim},
                                                    },
                                                    "where": {
                                                        "class": "switch",
                                                        "condition": "logical_and",
                                                        "true_from": 0.0,
                                                        "false_from": "base:prev:_mask_v2",
                                                        "out_shape": {batch_dim, time_dim, feat_dim},
                                                    },
                                                    "output": {
                                                        "class": "copy",
                                                        "from": "where",
                                                        "out_shape": {batch_dim, time_dim, feat_dim},
                                                    },
                                                },
                                                "initial_output": "base:base:base:base:data",
                                                "need_last": True,
                                                "out_shape": {batch_dim, time_dim, feat_dim},
                                            },
                                            "output": {"class": "copy", "from": "rec_unstack", "out_shape": {}},
                                        },
                                        "axis": specaugment_v2_cond_true_feature_masking_top_k_k_dim,
                                        "out_shape": {specaugment_v2_cond_true_feature_masking_top_k_k_dim},
                                        "name_scope": "",
                                    },
                                    "range_in_axis": {
                                        "class": "range_in_axis",
                                        "from": "top_k/indices",
                                        "axis": specaugment_v2_cond_true_feature_masking_top_k_k_dim,
                                        "out_shape": {specaugment_v2_cond_true_feature_masking_top_k_k_dim},
                                    },
                                    "_mask_v2": {
                                        "class": "rec_last_output",
                                        "rec_layer": "loop",
                                        "sub_layer_name": "_mask_v2",
                                        "out_shape": {batch_dim, time_dim, feat_dim},
                                    },
                                    "output": {
                                        "class": "copy",
                                        "from": "_mask_v2",
                                        "out_shape": {batch_dim, time_dim, feat_dim},
                                    },
                                },
                                "out_shape": {batch_dim, time_dim, feat_dim},
                            },
                            "output": {
                                "class": "copy",
                                "from": "feature_masking",
                                "out_shape": {batch_dim, time_dim, feat_dim},
                            },
                        },
                    },
                    "false_layer": {
                        "class": "subnetwork",
                        "from": [],
                        "subnetwork": {
                            "output": {
                                "class": "copy",
                                "from": "base:base:data:data",
                                "out_shape": {batch_dim, time_dim, feat_dim},
                            }
                        },
                    },
                    "out_shape": {batch_dim, time_dim, feat_dim},
                    "name_scope": "",
                },
                "output": {"class": "copy", "from": "cond"},
            },
            "out_shape": {batch_dim, time_dim, feat_dim},
        },
        "output": {"class": "copy", "from": "specaugment_v2", "out_shape": {batch_dim, time_dim, feat_dim}},
    }

    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        session.run(net.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_specaugment_pure_returnn_reduced():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim, ImplicitSparseDim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feat", 50)

    config = Config(
        dict(
            extern_data={
                "data": {
                    "dim_tags": (batch_dim, time_dim, feat_dim),
                    "dtype": "float32",
                    "available_for_inference": True,
                }
            },
            debug_runtime_sanity_checks=True,
        )
    )

    specaugment_v2_cond_true_feature_masking_top_k_k_dim = SpatialDim("feature_masking:num")

    net_dict = {
        "specaugment_v2": {
            "class": "subnetwork",
            "from": [],
            "subnetwork": {
                "dim_value": {
                    "class": "subnetwork",
                    "from": [],
                    "subnetwork": {
                        "length": {
                            "class": "length",
                            "from": "base:base:data:data",
                            "axis": time_dim,
                            "out_shape": {batch_dim},
                        },
                        "reduce": {
                            "class": "reduce",
                            "from": "length",
                            "mode": "max",
                            "axis": (batch_dim,),
                            "out_shape": {},
                        },
                        "output": {"class": "copy", "from": "reduce", "out_shape": {}},
                    },
                    "out_shape": {},
                },
                "constant": {"class": "constant", "value": 2},
                "minimum": {"class": "combine", "from": ["constant", "dim_value"], "kind": "minimum", "out_shape": {}},
                "constant_0": {"class": "constant", "value": 100},
                "floordiv": {
                    "class": "combine",
                    "from": ["dim_value", "constant_0"],
                    "kind": "floordiv",
                    "out_shape": {},
                },
                "constant_1": {"class": "constant", "value": 2},
                "maximum": {"class": "combine", "from": ["floordiv", "constant_1"], "kind": "maximum", "out_shape": {}},
                "constant_2": {"class": "constant", "value": 4},
                "mul": {"class": "combine", "from": ["maximum", "constant_2"], "kind": "mul", "out_shape": {}},
                "minimum_0": {"class": "combine", "from": ["mul", "dim_value"], "kind": "minimum", "out_shape": {}},
                "feature_masking": {
                    "class": "subnetwork",
                    "from": [],
                    "subnetwork": {
                        "random": {
                            "class": "subnetwork",
                            "from": [],
                            "subnetwork": {
                                "random": {
                                    "class": "random",
                                    "shape": [batch_dim],
                                    "distribution": "uniform",
                                    "minval": 2,
                                    "maxval": 6,
                                    "dtype": "int32",
                                },
                                "output": {"class": "copy", "from": "random", "out_shape": {batch_dim}},
                            },
                            "out_shape": {batch_dim},
                        },
                        "random_0": {
                            "class": "subnetwork",
                            "from": [],
                            "subnetwork": {
                                "random": {
                                    "class": "random",
                                    "shape": [batch_dim, feat_dim],
                                    "distribution": "uniform",
                                    "minval": 0.0,
                                    "maxval": 1.0,
                                },
                                "output": {"class": "copy", "from": "random", "out_shape": {batch_dim, feat_dim}},
                            },
                            "out_shape": {batch_dim, feat_dim},
                        },
                        "log": {
                            "class": "activation",
                            "from": "random_0",
                            "activation": "log",
                            "out_shape": {batch_dim, feat_dim},
                        },
                        "negative": {
                            "class": "activation",
                            "from": "log",
                            "activation": "negative",
                            "out_shape": {batch_dim, feat_dim},
                        },
                        "log_0": {
                            "class": "activation",
                            "from": "negative",
                            "activation": "log",
                            "out_shape": {batch_dim, feat_dim},
                        },
                        "negative_0": {
                            "class": "activation",
                            "from": "log_0",
                            "activation": "negative",
                            "out_shape": {batch_dim, feat_dim},
                        },
                        "reduce": {
                            "class": "reduce",
                            "from": "random",
                            "mode": "max",
                            "axis": (batch_dim,),
                            "out_shape": {},
                        },
                        "top_k": {
                            "class": "top_k",
                            "from": "negative_0",
                            "axis": feat_dim,
                            "k": "reduce",
                            "k_dim": specaugment_v2_cond_true_feature_masking_top_k_k_dim,
                            "sorted": True,
                            "out_shape": {batch_dim, specaugment_v2_cond_true_feature_masking_top_k_k_dim},
                        },
                        "loop": {
                            "class": "rec",
                            "from": [],
                            "unit": {
                                "rec_unstack": {
                                    "class": "rec_unstack",
                                    "from": "base:range_in_axis",
                                    "axis": specaugment_v2_cond_true_feature_masking_top_k_k_dim,
                                    "out_shape": {},
                                },
                                "gather": {
                                    "class": "gather",
                                    "from": "base:top_k/indices",
                                    "position": "rec_unstack",
                                    "axis": specaugment_v2_cond_true_feature_masking_top_k_k_dim,
                                    "out_shape": {batch_dim, ImplicitSparseDim(feat_dim)},
                                },
                                "_mask_v2": {
                                    "class": "subnetwork",
                                    "from": [],
                                    "subnetwork": {
                                        "length": {
                                            "class": "length",
                                            "from": "base:prev:_mask_v2",
                                            "axis": feat_dim,
                                            "out_shape": {},
                                        },
                                        "random": {
                                            "class": "subnetwork",
                                            "from": [],
                                            "subnetwork": {
                                                "random": {
                                                    "class": "random",
                                                    "shape": (batch_dim,),
                                                    "distribution": "uniform",
                                                    "minval": 1,
                                                    "maxval": 11,
                                                    "dtype": "int32",
                                                },
                                                "output": {"class": "copy", "from": "random", "out_shape": {batch_dim}},
                                            },
                                            "out_shape": {batch_dim},
                                        },
                                        "add": {
                                            "class": "combine",
                                            "from": ["base:gather", "random"],
                                            "kind": "add",
                                            "out_shape": {batch_dim, ImplicitSparseDim(feat_dim)},
                                        },
                                        "minimum": {
                                            "class": "combine",
                                            "from": ["add", "length"],
                                            "kind": "minimum",
                                            "out_shape": {batch_dim, ImplicitSparseDim(feat_dim)},
                                        },
                                        "range_in_axis": {
                                            "class": "range_in_axis",
                                            "from": "base:prev:_mask_v2",
                                            "axis": feat_dim,
                                            "out_shape": {feat_dim},
                                        },
                                        "greater_equal": {
                                            "class": "compare",
                                            "from": ["range_in_axis", "base:gather"],
                                            "kind": "greater_equal",
                                            "allow_broadcast_all_sources": True,
                                            "out_shape": {batch_dim, feat_dim},
                                        },
                                        "less": {
                                            "class": "compare",
                                            "from": ["range_in_axis", "minimum"],
                                            "kind": "less",
                                            "allow_broadcast_all_sources": True,
                                            "out_shape": {batch_dim, feat_dim},
                                        },
                                        "logical_and": {
                                            "class": "combine",
                                            "from": ["greater_equal", "less"],
                                            "kind": "logical_and",
                                            "out_shape": {batch_dim, feat_dim},
                                        },
                                        "where": {
                                            "class": "switch",
                                            "condition": "logical_and",
                                            "true_from": 0.0,
                                            "false_from": "base:prev:_mask_v2",
                                            "out_shape": {batch_dim, time_dim, feat_dim},
                                        },
                                        "output": {
                                            "class": "copy",
                                            "from": "where",
                                            "out_shape": {batch_dim, time_dim, feat_dim},
                                        },
                                    },
                                    "initial_output": "base:base:base:data",
                                    "need_last": True,
                                    "out_shape": {batch_dim, time_dim, feat_dim},
                                },
                                "output": {"class": "copy", "from": "rec_unstack", "out_shape": {}},
                            },
                            "axis": specaugment_v2_cond_true_feature_masking_top_k_k_dim,
                            "out_shape": {specaugment_v2_cond_true_feature_masking_top_k_k_dim},
                            "name_scope": "",
                        },
                        "range_in_axis": {
                            "class": "range_in_axis",
                            "from": "top_k/indices",
                            "axis": specaugment_v2_cond_true_feature_masking_top_k_k_dim,
                            "out_shape": {specaugment_v2_cond_true_feature_masking_top_k_k_dim},
                        },
                        "_mask_v2": {
                            "class": "rec_last_output",
                            "rec_layer": "loop",
                            "sub_layer_name": "_mask_v2",
                            "out_shape": {batch_dim, time_dim, feat_dim},
                        },
                        "output": {"class": "copy", "from": "_mask_v2", "out_shape": {batch_dim, time_dim, feat_dim}},
                    },
                    "out_shape": {batch_dim, time_dim, feat_dim},
                },
                "output": {"class": "copy", "from": "feature_masking", "out_shape": {batch_dim, time_dim, feat_dim}},
            },
            "out_shape": {batch_dim, time_dim, feat_dim},
        },
        "output": {"class": "copy", "from": "specaugment_v2", "out_shape": {batch_dim, time_dim, feat_dim}},
    }

    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        session.run(net.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_specaugment_pure_returnn():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim, ImplicitDynSizeDim, ImplicitSparseDim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feat", 50)

    config = Config(
        dict(
            extern_data={
                "data": {
                    "dim_tags": (batch_dim, time_dim, feat_dim),
                    "dtype": "float32",
                    "available_for_inference": True,
                }
            },
            debug_runtime_sanity_checks=True,
        )
    )

    specaugment_v2_cond_true_time_masking_top_k_k_dim = SpatialDim("specaugment_v2/cond/true/time_masking:top_k:k_dim")
    specaugment_v2_cond_true_feature_masking_top_k_k_dim = SpatialDim(
        "specaugment_v2/cond/true/feature_masking:top_k:k_dim"
    )
    random_state_dim = FeatureDim("random-state", 3)

    net_dict = {
        "specaugment_v2": {
            "class": "subnetwork",
            "from": [],
            "subnetwork": {
                "ones": {"class": "constant", "value": True, "shape": (), "dtype": "bool"},
                "cond": {
                    "class": "cond",
                    "from": [],
                    "condition": "ones",
                    "true_layer": {
                        "class": "subnetwork",
                        "from": [],
                        "subnetwork": {
                            "dim_value": {
                                "class": "subnetwork",
                                "from": [],
                                "subnetwork": {
                                    "length": {
                                        "class": "length",
                                        "from": "base:base:base:data:data",
                                        "axis": time_dim,
                                        "out_shape": {batch_dim},
                                    },
                                    "reduce": {
                                        "class": "reduce",
                                        "from": "length",
                                        "mode": "max",
                                        "axis": (batch_dim,),
                                        "out_shape": {},
                                    },
                                    "output": {"class": "copy", "from": "reduce", "out_shape": {}},
                                },
                                "out_shape": {},
                            },
                            "constant": {"class": "constant", "value": 2},
                            "minimum": {
                                "class": "combine",
                                "from": ["constant", "dim_value"],
                                "kind": "minimum",
                                "out_shape": {},
                            },
                            "constant_0": {"class": "constant", "value": 100},
                            "floordiv": {
                                "class": "combine",
                                "from": ["dim_value", "constant_0"],
                                "kind": "floordiv",
                                "out_shape": {},
                            },
                            "constant_1": {"class": "constant", "value": 2},
                            "maximum": {
                                "class": "combine",
                                "from": ["floordiv", "constant_1"],
                                "kind": "maximum",
                                "out_shape": {},
                            },
                            "constant_2": {"class": "constant", "value": 4},
                            "mul": {
                                "class": "combine",
                                "from": ["maximum", "constant_2"],
                                "kind": "mul",
                                "out_shape": {},
                            },
                            "minimum_0": {
                                "class": "combine",
                                "from": ["mul", "dim_value"],
                                "kind": "minimum",
                                "out_shape": {},
                            },
                            "time_masking": {
                                "class": "subnetwork",
                                "from": [],
                                "subnetwork": {
                                    "constant": {"class": "constant", "value": 1},
                                    "add": {
                                        "class": "combine",
                                        "from": ["base:minimum_0", "constant"],
                                        "kind": "add",
                                        "out_shape": {},
                                    },
                                    "random": {
                                        "class": "subnetwork",
                                        "from": [],
                                        "subnetwork": {
                                            "random": {
                                                "class": "random",
                                                "shape": [batch_dim],
                                                "distribution": "uniform",
                                                "minval": "base:base:minimum",
                                                "maxval": "base:add",
                                                "dtype": "int32",
                                                "explicit_state": "base:base:base:random_state_var0_1",
                                                "auto_update_state": True,
                                            },
                                            "output": {"class": "copy", "from": "random", "out_shape": {batch_dim}},
                                        },
                                        "out_shape": {batch_dim},
                                    },
                                    "random_0": {
                                        "class": "subnetwork",
                                        "from": [],
                                        "subnetwork": {
                                            "random": {
                                                "class": "random",
                                                "shape": [batch_dim, time_dim],
                                                "distribution": "uniform",
                                                "minval": 0.0,
                                                "maxval": 1.0,
                                                "explicit_state": "base:base:base:random_0_state_var0_0",
                                                "auto_update_state": True,
                                            },
                                            "output": {
                                                "class": "copy",
                                                "from": "random",
                                                "out_shape": {batch_dim, time_dim},
                                            },
                                        },
                                        "out_shape": {batch_dim, time_dim},
                                    },
                                    "log": {
                                        "class": "activation",
                                        "from": "random_0",
                                        "activation": "log",
                                        "out_shape": {batch_dim, time_dim},
                                    },
                                    "negative": {
                                        "class": "activation",
                                        "from": "log",
                                        "activation": "negative",
                                        "out_shape": {batch_dim, time_dim},
                                    },
                                    "log_0": {
                                        "class": "activation",
                                        "from": "negative",
                                        "activation": "log",
                                        "out_shape": {batch_dim, time_dim},
                                    },
                                    "negative_0": {
                                        "class": "activation",
                                        "from": "log_0",
                                        "activation": "negative",
                                        "out_shape": {batch_dim, time_dim},
                                    },
                                    "reduce": {
                                        "class": "reduce",
                                        "from": "random",
                                        "mode": "max",
                                        "axis": (batch_dim,),
                                        "out_shape": {},
                                    },
                                    "top_k": {
                                        "class": "top_k",
                                        "from": "negative_0",
                                        "axis": time_dim,
                                        "k": "reduce",
                                        "k_dim": specaugment_v2_cond_true_time_masking_top_k_k_dim,
                                        "sorted": True,
                                        "out_shape": {batch_dim, specaugment_v2_cond_true_time_masking_top_k_k_dim},
                                    },
                                    "loop": {
                                        "class": "rec",
                                        "from": [],
                                        "unit": {
                                            "rec_unstack": {
                                                "class": "rec_unstack",
                                                "from": "base:range_in_axis",
                                                "axis": specaugment_v2_cond_true_time_masking_top_k_k_dim,
                                                "out_shape": {},
                                            },
                                            "gather": {
                                                "class": "gather",
                                                "from": "base:top_k/indices",
                                                "position": "rec_unstack",
                                                "axis": specaugment_v2_cond_true_time_masking_top_k_k_dim,
                                                "out_shape": {batch_dim, ImplicitSparseDim(time_dim)},
                                            },
                                            "_mask_v2": {
                                                "class": "subnetwork",
                                                "from": [],
                                                "subnetwork": {
                                                    "length": {
                                                        "class": "length",
                                                        "from": "base:prev:_mask_v2",
                                                        "axis": time_dim,
                                                        "out_shape": {batch_dim},
                                                    },
                                                    "random": {
                                                        "class": "subnetwork",
                                                        "from": [],
                                                        "subnetwork": {
                                                            "random": {
                                                                "class": "random",
                                                                "shape": (batch_dim,),
                                                                "distribution": "uniform",
                                                                "minval": 1,
                                                                "maxval": 21,
                                                                "dtype": "int32",
                                                                "explicit_state": "base:base:base:base:base:random_state_var0_2",
                                                                "auto_update_state": True,
                                                            },
                                                            "output": {
                                                                "class": "copy",
                                                                "from": "random",
                                                                "out_shape": {batch_dim},
                                                            },
                                                        },
                                                        "out_shape": {batch_dim},
                                                    },
                                                    "add": {
                                                        "class": "combine",
                                                        "from": ["base:gather", "random"],
                                                        "kind": "add",
                                                        "out_shape": {batch_dim, ImplicitSparseDim(time_dim)},
                                                    },
                                                    "minimum": {
                                                        "class": "combine",
                                                        "from": ["add", "length"],
                                                        "kind": "minimum",
                                                        "out_shape": {batch_dim, ImplicitSparseDim(time_dim)},
                                                    },
                                                    "range_in_axis": {
                                                        "class": "range_in_axis",
                                                        "from": "base:prev:_mask_v2",
                                                        "axis": time_dim,
                                                        "out_shape": {ImplicitDynSizeDim(batch_dim), time_dim},
                                                    },
                                                    "greater_equal": {
                                                        "class": "compare",
                                                        "from": ["range_in_axis", "base:gather"],
                                                        "kind": "greater_equal",
                                                        "allow_broadcast_all_sources": True,
                                                        "out_shape": {batch_dim, time_dim},
                                                    },
                                                    "less": {
                                                        "class": "compare",
                                                        "from": ["range_in_axis", "minimum"],
                                                        "kind": "less",
                                                        "allow_broadcast_all_sources": True,
                                                        "out_shape": {batch_dim, time_dim},
                                                    },
                                                    "logical_and": {
                                                        "class": "combine",
                                                        "from": ["greater_equal", "less"],
                                                        "kind": "logical_and",
                                                        "out_shape": {batch_dim, time_dim},
                                                    },
                                                    "where": {
                                                        "class": "switch",
                                                        "condition": "logical_and",
                                                        "true_from": 0.0,
                                                        "false_from": "base:prev:_mask_v2",
                                                        "out_shape": {batch_dim, time_dim, feat_dim},
                                                    },
                                                    "output": {
                                                        "class": "copy",
                                                        "from": "where",
                                                        "out_shape": {batch_dim, time_dim, feat_dim},
                                                    },
                                                },
                                                "initial_output": "base:base:base:base:data:data",
                                                "need_last": True,
                                                "out_shape": {batch_dim, time_dim, feat_dim},
                                            },
                                            "output": {"class": "copy", "from": "rec_unstack", "out_shape": {}},
                                        },
                                        "axis": specaugment_v2_cond_true_time_masking_top_k_k_dim,
                                        "out_shape": {specaugment_v2_cond_true_time_masking_top_k_k_dim},
                                        "name_scope": "",
                                    },
                                    "range_in_axis": {
                                        "class": "range_in_axis",
                                        "from": "top_k/indices",
                                        "axis": specaugment_v2_cond_true_time_masking_top_k_k_dim,
                                        "out_shape": {specaugment_v2_cond_true_time_masking_top_k_k_dim},
                                    },
                                    "_mask_v2": {
                                        "class": "rec_last_output",
                                        "rec_layer": "loop",
                                        "sub_layer_name": "_mask_v2",
                                        "out_shape": {batch_dim, time_dim, feat_dim},
                                    },
                                    "output": {
                                        "class": "copy",
                                        "from": "_mask_v2",
                                        "out_shape": {batch_dim, time_dim, feat_dim},
                                    },
                                },
                                "out_shape": {batch_dim, time_dim, feat_dim},
                            },
                            "feature_masking": {
                                "class": "subnetwork",
                                "from": [],
                                "subnetwork": {
                                    "random": {
                                        "class": "subnetwork",
                                        "from": [],
                                        "subnetwork": {
                                            "random": {
                                                "class": "random",
                                                "shape": [batch_dim],
                                                "distribution": "uniform",
                                                "minval": 2,
                                                "maxval": 6,
                                                "dtype": "int32",
                                                "explicit_state": "base:base:base:random_state_var0",
                                                "auto_update_state": True,
                                            },
                                            "output": {"class": "copy", "from": "random", "out_shape": {batch_dim}},
                                        },
                                        "out_shape": {batch_dim},
                                    },
                                    "random_0": {
                                        "class": "subnetwork",
                                        "from": [],
                                        "subnetwork": {
                                            "random": {
                                                "class": "random",
                                                "shape": [batch_dim, feat_dim],
                                                "distribution": "uniform",
                                                "minval": 0.0,
                                                "maxval": 1.0,
                                                "explicit_state": "base:base:base:random_0_state_var0",
                                                "auto_update_state": True,
                                            },
                                            "output": {
                                                "class": "copy",
                                                "from": "random",
                                                "out_shape": {batch_dim, feat_dim},
                                            },
                                        },
                                        "out_shape": {batch_dim, feat_dim},
                                    },
                                    "log": {
                                        "class": "activation",
                                        "from": "random_0",
                                        "activation": "log",
                                        "out_shape": {batch_dim, feat_dim},
                                    },
                                    "negative": {
                                        "class": "activation",
                                        "from": "log",
                                        "activation": "negative",
                                        "out_shape": {batch_dim, feat_dim},
                                    },
                                    "log_0": {
                                        "class": "activation",
                                        "from": "negative",
                                        "activation": "log",
                                        "out_shape": {batch_dim, feat_dim},
                                    },
                                    "negative_0": {
                                        "class": "activation",
                                        "from": "log_0",
                                        "activation": "negative",
                                        "out_shape": {batch_dim, feat_dim},
                                    },
                                    "reduce": {
                                        "class": "reduce",
                                        "from": "random",
                                        "mode": "max",
                                        "axis": (batch_dim,),
                                        "out_shape": {},
                                    },
                                    "top_k": {
                                        "class": "top_k",
                                        "from": "negative_0",
                                        "axis": feat_dim,
                                        "k": "reduce",
                                        "k_dim": specaugment_v2_cond_true_feature_masking_top_k_k_dim,
                                        "sorted": True,
                                        "out_shape": {batch_dim, specaugment_v2_cond_true_feature_masking_top_k_k_dim},
                                    },
                                    "loop": {
                                        "class": "rec",
                                        "from": [],
                                        "unit": {
                                            "rec_unstack": {
                                                "class": "rec_unstack",
                                                "from": "base:range_in_axis",
                                                "axis": specaugment_v2_cond_true_feature_masking_top_k_k_dim,
                                                "out_shape": {},
                                            },
                                            "gather": {
                                                "class": "gather",
                                                "from": "base:top_k/indices",
                                                "position": "rec_unstack",
                                                "axis": specaugment_v2_cond_true_feature_masking_top_k_k_dim,
                                                "out_shape": {batch_dim, ImplicitSparseDim(feat_dim)},
                                            },
                                            "_mask_v2": {
                                                "class": "subnetwork",
                                                "from": [],
                                                "subnetwork": {
                                                    "length": {
                                                        "class": "length",
                                                        "from": "base:prev:_mask_v2",
                                                        "axis": feat_dim,
                                                        "out_shape": {},
                                                    },
                                                    "random": {
                                                        "class": "subnetwork",
                                                        "from": [],
                                                        "subnetwork": {
                                                            "random": {
                                                                "class": "random",
                                                                "shape": (batch_dim,),
                                                                "distribution": "uniform",
                                                                "minval": 1,
                                                                "maxval": 11,
                                                                "dtype": "int32",
                                                                "explicit_state": "base:base:base:base:base:random_state_var0_0",
                                                                "auto_update_state": True,
                                                            },
                                                            "output": {
                                                                "class": "copy",
                                                                "from": "random",
                                                                "out_shape": {batch_dim},
                                                            },
                                                        },
                                                        "out_shape": {batch_dim},
                                                    },
                                                    "add": {
                                                        "class": "combine",
                                                        "from": ["base:gather", "random"],
                                                        "kind": "add",
                                                        "out_shape": {batch_dim, ImplicitSparseDim(feat_dim)},
                                                    },
                                                    "minimum": {
                                                        "class": "combine",
                                                        "from": ["add", "length"],
                                                        "kind": "minimum",
                                                        "out_shape": {batch_dim, ImplicitSparseDim(feat_dim)},
                                                    },
                                                    "range_in_axis": {
                                                        "class": "range_in_axis",
                                                        "from": "base:prev:_mask_v2",
                                                        "axis": feat_dim,
                                                        "out_shape": {feat_dim},
                                                    },
                                                    "greater_equal": {
                                                        "class": "compare",
                                                        "from": ["range_in_axis", "base:gather"],
                                                        "kind": "greater_equal",
                                                        "allow_broadcast_all_sources": True,
                                                        "out_shape": {batch_dim, feat_dim},
                                                    },
                                                    "less": {
                                                        "class": "compare",
                                                        "from": ["range_in_axis", "minimum"],
                                                        "kind": "less",
                                                        "allow_broadcast_all_sources": True,
                                                        "out_shape": {batch_dim, feat_dim},
                                                    },
                                                    "logical_and": {
                                                        "class": "combine",
                                                        "from": ["greater_equal", "less"],
                                                        "kind": "logical_and",
                                                        "out_shape": {batch_dim, feat_dim},
                                                    },
                                                    "where": {
                                                        "class": "switch",
                                                        "condition": "logical_and",
                                                        "true_from": 0.0,
                                                        "false_from": "base:prev:_mask_v2",
                                                        "out_shape": {batch_dim, time_dim, feat_dim},
                                                    },
                                                    "output": {
                                                        "class": "copy",
                                                        "from": "where",
                                                        "out_shape": {batch_dim, time_dim, feat_dim},
                                                    },
                                                },
                                                "initial_output": "base:base:time_masking",
                                                "need_last": True,
                                                "out_shape": {batch_dim, time_dim, feat_dim},
                                            },
                                            "output": {"class": "copy", "from": "rec_unstack", "out_shape": {}},
                                        },
                                        "axis": specaugment_v2_cond_true_feature_masking_top_k_k_dim,
                                        "out_shape": {specaugment_v2_cond_true_feature_masking_top_k_k_dim},
                                        "name_scope": "",
                                    },
                                    "range_in_axis": {
                                        "class": "range_in_axis",
                                        "from": "top_k/indices",
                                        "axis": specaugment_v2_cond_true_feature_masking_top_k_k_dim,
                                        "out_shape": {specaugment_v2_cond_true_feature_masking_top_k_k_dim},
                                    },
                                    "_mask_v2": {
                                        "class": "rec_last_output",
                                        "rec_layer": "loop",
                                        "sub_layer_name": "_mask_v2",
                                        "out_shape": {batch_dim, time_dim, feat_dim},
                                    },
                                    "output": {
                                        "class": "copy",
                                        "from": "_mask_v2",
                                        "out_shape": {batch_dim, time_dim, feat_dim},
                                    },
                                },
                                "out_shape": {batch_dim, time_dim, feat_dim},
                            },
                            "output": {
                                "class": "copy",
                                "from": "feature_masking",
                                "out_shape": {batch_dim, time_dim, feat_dim},
                            },
                        },
                    },
                    "false_layer": {
                        "class": "subnetwork",
                        "from": [],
                        "subnetwork": {
                            "output": {
                                "class": "copy",
                                "from": "base:base:data:data",
                                "out_shape": {batch_dim, time_dim, feat_dim},
                            }
                        },
                    },
                    "out_shape": {batch_dim, time_dim, feat_dim},
                    "name_scope": "",
                },
                "random_state_init": {
                    "class": "random_state_init",
                    "out_dim": random_state_dim,
                    "out_shape": {random_state_dim},
                },
                "random_state_init_0": {
                    "class": "random_state_init",
                    "out_dim": random_state_dim,
                    "out_shape": {random_state_dim},
                },
                "random_state_init_1": {
                    "class": "random_state_init",
                    "out_dim": random_state_dim,
                    "out_shape": {random_state_dim},
                },
                "random_state_init_2": {
                    "class": "random_state_init",
                    "out_dim": random_state_dim,
                    "out_shape": {random_state_dim},
                },
                "random_state_init_3": {
                    "class": "random_state_init",
                    "out_dim": random_state_dim,
                    "out_shape": {random_state_dim},
                },
                "random_state_init_4": {
                    "class": "random_state_init",
                    "out_dim": random_state_dim,
                    "out_shape": {random_state_dim},
                },
                "output": {"class": "copy", "from": "cond", "out_shape": {batch_dim, time_dim, feat_dim}},
                "random_state_var0": {
                    "class": "variable",
                    "shape": [random_state_dim],
                    "param_name": "param",
                    "dtype": "int64",
                    "init_by_layer": "random_state_init_2",
                },
                "random_state_var0_0": {
                    "class": "variable",
                    "shape": [random_state_dim],
                    "param_name": "param",
                    "dtype": "int64",
                    "init_by_layer": "random_state_init_4",
                },
                "random_0_state_var0": {
                    "class": "variable",
                    "shape": [random_state_dim],
                    "param_name": "param",
                    "dtype": "int64",
                    "init_by_layer": "random_state_init_3",
                },
                "random_state_var0_1": {
                    "class": "variable",
                    "shape": [random_state_dim],
                    "param_name": "param",
                    "dtype": "int64",
                    "init_by_layer": "random_state_init",
                },
                "random_state_var0_2": {
                    "class": "variable",
                    "shape": [random_state_dim],
                    "param_name": "param",
                    "dtype": "int64",
                    "init_by_layer": "random_state_init_1",
                },
                "random_0_state_var0_0": {
                    "class": "variable",
                    "shape": [random_state_dim],
                    "param_name": "param",
                    "dtype": "int64",
                    "init_by_layer": "random_state_init_0",
                },
            },
            "out_shape": {batch_dim, time_dim, feat_dim},
        },
        "output": {"class": "copy", "from": "specaugment_v2", "out_shape": {batch_dim, time_dim, feat_dim}},
    }

    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        session.run(net.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_SearchSortedLayer():
    n_batch, n_time, n_in, n_out = 2, 10, 3, 5
    random = numpy.random.RandomState(seed=1)
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData({"data": {"shape": (None, 3)}}))
        sorted_layer = InternalLayer(
            name="sorted_sequence", network=net, output=Data(name="sorted_sequence", shape=(None, n_in))
        )  # [B,T,F]
        sorted = numpy.sort(random.uniform(0.0, 10.0, size=(n_batch, n_time, n_in)), axis=1)
        sorted_lens = [10, 7]
        sorted_layer.output.placeholder = tf.constant(sorted, dtype="float32")
        sorted_layer.output.size_placeholder = {0: tf.constant(sorted_lens, dtype="int32")}  # [B]
        values_layer = InternalLayer(
            name="values",
            network=net,
            output=Data(**{"name": "values", "shape": (n_in, n_out), "time_dim_axis": None, "feature_dim_axis": 1}),
        )  # [B,F,F']
        values = random.uniform(0.0, 10.0, size=(n_batch, n_in, n_out))
        values_layer.output.placeholder = tf.constant(values, dtype="float32")

        for side in ["left", "right"]:
            print("Testing side=%r" % side)
            opts = {
                "network": net,
                "name": "search_sorted_test",
                "sources": [],
                "sorted_sequence": sorted_layer,
                "values": values_layer,
                "axis": "T",
                "side": side,
            }
            out_data = SearchSortedLayer.get_out_data_from_opts(**opts)
            out_data.sanity_check(ignore_placeholder=True)
            search_layer = SearchSortedLayer(output=out_data, **opts)  # should be [B,F]
            out_data = search_layer.output
            out_data.sanity_check()
            print(search_layer.output)

            output = session.run(
                search_layer.output.placeholder, feed_dict=make_feed_dict(net.extern_data, n_batch=n_batch)
            )
            assert output.dtype == "int32"
            assert out_data.batch_shape == (None, n_in, n_out)
            assert out_data.batch_dim_axis == 0
            assert out_data.time_dim_axis is None
            assert out_data.feature_dim_axis == 1
            for b, t_max in enumerate(sorted_lens):
                for f in range(n_in):
                    expected = numpy.searchsorted(a=sorted[b, :t_max, f], v=values[b, f, :], side=side)
                    actual = output[b, f, :]
                    numpy.testing.assert_equal(expected, actual)


def test_CondLayer_subnetwork_train():
    n_batch, n_time, n_in, n_out = 3, 7, 11, 13
    config = Config(
        {
            "extern_data": {
                "data": {"dim": n_in},
                "classes": {"dim": n_out, "sparse": True},
            },
            "debug_print_layer_output_template": True,
        }
    )
    rnd = numpy.random.RandomState(42)
    with make_scope() as session:
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "src": {"class": "linear", "activation": "tanh", "n_out": 10, "from": "data"},
                "cond": {
                    "class": "cond",
                    "from": [],
                    "condition": {
                        "class": "eval",
                        "from": [],
                        "out_type": {"batch_dim_axis": None, "shape": (), "dtype": "bool"},
                        "eval": "tf.equal(self.network.global_train_step % 2, 0)",
                    },
                    "true_layer": {
                        "class": "subnetwork",
                        "from": "src",
                        "subnetwork": {
                            "lin": {"class": "linear", "activation": "tanh", "n_out": 10, "from": "data"},
                            "res": {"class": "combine", "kind": "add", "from": ["data", "lin"]},
                            "output": {
                                "class": "print",
                                "from": "res",
                                "extra_print_args": ["true_layer"],
                                "summarize": 1,
                            },
                        },
                    },
                    "false_layer": {"class": "copy", "from": "src"},
                },
                "output": {"class": "softmax", "from": "cond", "loss": "ce", "target": "classes"},
            }
        )
        net.print_network_info()
        cond_layer = net.get_layer("cond")
        assert isinstance(cond_layer, CondLayer)
        assert not tf_util.has_control_flow_context(cond_layer.output.placeholder)
        cond_true_layer = cond_layer.true_layer
        assert isinstance(cond_true_layer, SubnetworkLayer)
        # Check whether the execution of the true branch is actually conditionally.
        assert tf_util.has_control_flow_context(cond_true_layer.output.placeholder)
        trainable_vars = net.get_trainable_params()
        print("Trainable vars:")
        pprint(trainable_vars)
        cond_var = net.layers["cond"].params["lin/W"]
        assert cond_var in trainable_vars
        from returnn.tf.updater import Updater

        updater = Updater(config=config, network=net, initial_learning_rate=0.1)
        updater.set_trainable_vars(trainable_vars)
        updater.init_optimizer_vars(session)
        updater.set_learning_rate(value=updater.initial_learning_rate, session=session)
        net.initialize_params(session)
        in_v = rnd.normal(size=(n_batch, n_time, n_in)).astype("float32")
        targets_v = rnd.randint(0, n_out, size=(n_batch, n_time)).astype("int32")
        seq_lens_v = numpy.array([n_time, n_time - 1, n_time - 2])
        assert len(seq_lens_v) == n_batch
        feed_dict = {
            net.extern_data.data["data"].placeholder: in_v,
            net.extern_data.data["data"].size_placeholder[0]: seq_lens_v,
            net.extern_data.data["classes"].placeholder: targets_v,
            net.extern_data.data["classes"].size_placeholder[0]: seq_lens_v,
        }
        fetches = net.get_fetches_dict(with_summary=True, with_size=True)
        fetches["optim_op"] = updater.get_optim_op()
        try:
            loss = None
            initial_loss = float("inf")
            for i in range(10):
                step = session.run(net.global_train_step)
                print("step: %i" % step)
                assert i == step
                old_var_value = session.run(cond_var)
                result = session.run(feed_dict=feed_dict, fetches=fetches)
                loss = result["loss"]
                print("loss:", loss)
                if i == 0:
                    initial_loss = loss
                new_var_value = session.run(cond_var)
                var_changed = (old_var_value != new_var_value).any()
                print("var changed:", var_changed)
                if i % 2 == 0:  # See cond layer, condition. Use true_layer every second iteration, starting with 0.
                    # We used true_layer, thus the params should have been updated.
                    assert var_changed
                else:
                    # We did not use true_layer, thus the params should not have been updated.
                    assert not var_changed
            assert loss is not None and loss < initial_loss and numpy.isfinite(initial_loss)
        except tf.errors.OpError as exc:
            print("TF exception:", type(exc).__name__, ":", exc)
            from returnn.tf.network import help_on_tf_exception

            help_on_tf_exception(session=session, exception=exc, fetches=fetches, feed_dict=feed_dict)
            raise


def test_CondLayer_subnet_template_construct():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feat", 5)

    config = Config(
        dict(
            extern_data={
                "data": {
                    "dim_tags": (batch_dim, time_dim, feat_dim),
                    "dtype": "float32",
                    "available_for_inference": True,
                }
            },
            debug_runtime_sanity_checks=True,
            debug_print_layer_output_shape=True,
            debug_print_layer_output=True,
        )
    )

    class _EvalFuncLocals:
        graph_call_count = 0
        session_call_count = 0

    def _py_func(x):
        _EvalFuncLocals.session_call_count += 1
        return x

    def _eval_func(source, **_kwargs):
        _EvalFuncLocals.graph_call_count += 1
        x = source(0)
        (y,) = tf_compat.v1.py_func(_py_func, [x], [x.dtype], stateful=True)
        y.set_shape(x.get_shape())
        return y

    net_dict = {
        "specaugment_v2": {
            "class": "subnetwork",
            "from": [],
            "subnetwork": {
                "train_flag": {"class": "train_flag"},
                "cond": {
                    "class": "cond",
                    "from": [],
                    "condition": "train_flag",
                    "true_layer": {
                        "class": "subnetwork",
                        "from": [],
                        "subnetwork": {
                            "scores": {
                                "class": "reduce",
                                "from": "base:base:data",
                                "mode": "sum",
                                "axis": [batch_dim, time_dim, feat_dim],
                                "out_shape": {},
                            },
                            "output": {"class": "eval", "from": "scores", "eval": _eval_func},
                        },
                    },
                    "false_layer": {
                        "class": "subnetwork",
                        "from": [],
                        "subnetwork": {"output": {"class": "constant", "value": -1.0, "out_shape": {}}},
                    },
                    "name_scope": "",
                },
                "output": {"class": "copy", "from": "cond"},
            },
        },
        "output": {
            "class": "copy",
            "from": "specaugment_v2",
        },
    }

    with make_scope() as session:
        train_flag = tf_util.get_global_train_flag_placeholder()
        net = TFNetwork(config=config, train_flag=train_flag)
        net.construct_from_dict(net_dict)
        assert _EvalFuncLocals.graph_call_count == 1  # if more often, set a breakpoint above
        net.initialize_params(session)
        feed_dict = make_feed_dict(net.extern_data)
        feed_dict[train_flag] = False
        res_eval = session.run(net.get_default_output_layer().output.placeholder, feed_dict=feed_dict)
        print("eval:", res_eval)
        assert _EvalFuncLocals.session_call_count == 0
        feed_dict[train_flag] = True
        res_train = session.run(net.get_default_output_layer().output.placeholder, feed_dict=feed_dict)
        print("train:", res_train)
        assert _EvalFuncLocals.session_call_count == 1


def test_CondLayer_data_access():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    input_dim = FeatureDim("input", 13)

    config = Config(dict(extern_data={"data": {"dim_tags": (batch_dim, time_dim, input_dim)}}))
    net_dict = {
        "output": {"class": "copy", "from": "cond", "out_shape": {batch_dim, time_dim, input_dim}},
        "length": {"class": "length", "from": "data:data", "axis": batch_dim, "out_shape": {}},
        "mod": {"class": "eval", "from": "length", "eval": "source(0) % 2", "out_shape": {}},
        "eq": {"class": "compare", "from": "mod", "kind": "equal", "value": 0, "out_shape": {}},
        "cond": {
            "class": "cond",
            "from": [],
            "condition": "eq",
            "true_layer": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "const": {
                        "class": "constant",
                        "value": 1.0,
                        "shape": (batch_dim, time_dim, input_dim),
                        "shape_deps": ["base:data:data"],
                    },
                    "output": {
                        "class": "combine",
                        "from": ["base:data:data", "const"],
                        "kind": "add",
                        "out_shape": {batch_dim, time_dim, input_dim},
                    },
                },
            },
            "false_layer": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "output": {"class": "copy", "from": "base:data:data", "out_shape": {batch_dim, time_dim, input_dim}}
                },
            },
            "out_shape": {batch_dim, time_dim, input_dim},
            "name_scope": "",
        },
    }
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        network.initialize_params(session)
        session.run(
            network.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(network.extern_data)
        )


def test_CondLayer_mult_dyn_axes():
    # https://github.com/rwth-i6/returnn/issues/1207
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    input_dim = FeatureDim("input", 13)
    extra_time_dim = SpatialDim("extra_time")

    config = Config(dict(extern_data={"data": {"dim_tags": (batch_dim, time_dim, input_dim)}}))
    net_dict = {
        "cond": {
            "class": "eval",
            "from": "data",
            "eval": "tf.equal(tf.shape(source(0, auto_convert=False))[0] % 2, 0)",
            "out_type": {"shape": (), "dtype": "bool", "batch_dim_axis": None, "time_dim_axis": None},
        },
        "new_dim": {
            "class": "reinterpret_data",
            "from": "data",
            "set_dim_tags": {time_dim: extra_time_dim},
        },
        "pool": {
            "class": "pool",
            "from": "new_dim",
            "mode": "max",
            "padding": "same",
            "pool_size": [2],
        },
        "combine": {
            "class": "combine",
            "from": ["data", "pool"],
            "kind": "mul",
            "out_shape": {batch_dim, time_dim, extra_time_dim.ceildiv_right(2), input_dim},
        },
        "conv": {
            "class": "conv",
            "from": "combine",
            "filter_size": [3, 3],
            "strides": 2,
            "padding": "same",
            "in_spatial_dims": (time_dim, extra_time_dim.ceildiv_right(2)),
            "n_out": 7,
        },
        "output": {
            "class": "cond",
            "from": [],
            "condition": "cond",
            "true_layer": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "output": {
                        "class": "eval",
                        "from": "base:conv",
                        "eval": "source(0) * 1.5",
                    },
                },
            },
            "false_layer": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "output": {
                        "class": "copy",
                        "from": "base:conv",
                    },
                },
            },
        },
    }
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        network.initialize_params(session)
        in_ = network.extern_data.get_default_input_data()
        out = network.get_default_output_layer().output
        t1, _, t2, t3 = session.run(
            (
                in_.get_time_dim_tag().dyn_size_ext.placeholder,
                out.placeholder,
                out.get_dyn_size_tags()[0].dyn_size_ext.placeholder,
                out.get_dyn_size_tags()[1].dyn_size_ext.placeholder,
            ),
            feed_dict=make_feed_dict(network.extern_data),
        )

        def _ceildiv(a, b):
            return -(a // -b)

        numpy.testing.assert_array_equal(t2, _ceildiv(t1, 2))
        numpy.testing.assert_array_equal(t3, _ceildiv(t1, 4))


def test_CondLayer_dyn_dim_replace():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    in_dim = FeatureDim("in", 12)

    config = Config(
        dict(
            extern_data={
                "data": {"dim_tags": (batch_dim, time_dim, in_dim), "dtype": "float32", "available_for_inference": True}
            }
        )
    )

    # For the test, it is crucial to have two different dim tags here
    # which can be replaced by each other though.
    _2_time_dim = 2 * time_dim
    time_2_dim = time_dim * 2

    net_dict = {
        "length": {"class": "length", "from": ["data:data"], "axis": time_dim, "out_shape": {batch_dim}},
        "dim_value": {"class": "reduce", "from": "length", "mode": "max", "axis": (batch_dim,), "out_shape": {}},
        "mod": {"class": "eval", "from": "dim_value", "eval": "source(0) % 2", "out_shape": {}},
        "compare": {"class": "compare", "from": "mod", "kind": "equal", "value": 0, "out_shape": {}},
        "cond": {
            "class": "cond",
            "from": [],
            "condition": "compare",
            "true_layer": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "concat": {
                        "class": "concat",
                        "from": (("base:data:data", time_dim), ("base:data:data", time_dim)),
                        "out_dim": _2_time_dim,
                        "out_shape": {batch_dim, in_dim, _2_time_dim},
                    },
                    "new_dim": {
                        "class": "reinterpret_data",
                        "set_dim_tags": {_2_time_dim: time_2_dim},
                        "from": "concat",
                        "out_shape": {batch_dim, in_dim, time_2_dim},
                    },
                    "output": {"class": "copy", "from": "new_dim", "out_shape": {batch_dim, in_dim, time_2_dim}},
                },
            },
            "false_layer": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "random": {
                        "class": "random",
                        "shape": (batch_dim, time_2_dim, in_dim),
                        "distribution": "uniform",
                        "minval": 0,
                        "maxval": 1.0,
                        "shape_deps": ["base:data:data"],
                    },
                    "output": {"class": "copy", "from": "random", "out_shape": {batch_dim, in_dim, time_2_dim}},
                },
            },
            "out_shape": {batch_dim, in_dim, time_2_dim},
            "name_scope": "",
        },
        "output": {"class": "copy", "from": "cond", "out_shape": {batch_dim, in_dim, time_2_dim}},
    }

    with make_scope() as session:
        network = TFNetwork(config=config)
        network.construct_from_dict(net_dict)
        network.initialize_params(session)
        out = network.get_default_output_layer().output
        print("out:", out)
        out_seq_len = out.get_sequence_lengths()
        # Before the fix, the seq len tensor had the wrong control flow context inside the condition.
        # This caused the error:
        # tensorflow.python.framework.errors_impl.InvalidArgumentError: Retval[0] does not have value
        # Adding the same_control_flow_ctx at the place where it is created fixes this.
        print("out_seq_len:", out_seq_len)
        # tf_util.print_graph_output(out_seq_len) -- not really relevant
        # print(out_seq_len.op._traceback) -- not always available?
        fetch = out.placeholder
        session.run(fetch, feed_dict=make_feed_dict(network.extern_data, n_time=1))
        session.run(fetch, feed_dict=make_feed_dict(network.extern_data, n_time=2))


def test_CondLayer_variational_weight_noise():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    in_dim = FeatureDim("in", 3)

    config = Config(
        dict(
            extern_data={
                "data": {
                    "dim_tags": (batch_dim, time_dim, in_dim),
                }
            }
        )
    )

    out_dim = FeatureDim("out", 5)
    random_state_dim = FeatureDim("random-state", 3)

    net_dict = {
        "output": {"class": "copy", "from": "add", "out_shape": {batch_dim, time_dim, out_dim}},
        "bias": {"class": "variable", "shape": [out_dim], "param_name": "param", "init": 0.0},
        "weight_raw": {
            "class": "variable",
            "shape": [in_dim, out_dim],
            "param_name": "param",
        },
        "train_flag": {"class": "train_flag", "out_shape": {}},
        "weight": {
            "class": "cond",
            "from": [],
            "condition": "train_flag",
            "true_layer": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "random": {
                        "class": "random",
                        "shape": (in_dim, out_dim),
                        "distribution": "normal",
                        "mean": 0.0,
                        "stddev": 0.075,
                        "dtype": "float32",
                    },
                    "output": {
                        "class": "combine",
                        "from": ["base:weight_raw", "random"],
                        "kind": "add",
                        "out_shape": {in_dim, out_dim},
                    },
                },
            },
            "false_layer": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {"output": {"class": "copy", "from": "base:weight_raw", "out_shape": {in_dim, out_dim}}},
            },
            "out_shape": {in_dim, out_dim},
            "name_scope": "",
        },
        "dot": {
            "class": "dot",
            "from": ["data:data", "weight"],
            "reduce": in_dim,
            "out_shape": {batch_dim, time_dim, out_dim},
        },
        "add": {
            "class": "combine",
            "from": ["dot", "bias"],
            "kind": "add",
            "out_shape": {batch_dim, time_dim, out_dim},
        },
    }

    print("* Test with dynamic train flag")
    with make_scope() as session:
        train_flag = tf_compat.v1.placeholder(tf.bool, shape=(), name="train_flag")
        network = TFNetwork(config=config)
        network.construct_from_dict(net_dict)
        network.initialize_params(session)
        feed_dict = make_feed_dict(network.extern_data)
        feed_dict[train_flag] = True
        session.run(network.get_default_output_layer().output.placeholder, feed_dict=feed_dict)
        feed_dict[train_flag] = False
        session.run(network.get_default_output_layer().output.placeholder, feed_dict=feed_dict)

    for train_flag in [True, False]:
        print("* Test with static train flag", train_flag)
        with make_scope() as session:
            network = TFNetwork(config=config, train_flag=train_flag)
            network.construct_from_dict(net_dict)
            network.initialize_params(session)
            feed_dict = make_feed_dict(network.extern_data)
            session.run(network.get_default_output_layer().output.placeholder, feed_dict=feed_dict)


def test_CondLayer_outside_layer_access():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim, ImplicitDynSizeDim, ImplicitSparseDim

    time_dim = SpatialDim("time")
    input_dim = FeatureDim("input", 10)

    config = Config(
        dict(
            extern_data={
                "data": {
                    "dim_tags": (batch_dim, time_dim, input_dim),
                    "dtype": "float32",
                    "available_for_inference": True,
                }
            }
        )
    )

    labels_dim = FeatureDim("labels", 5)
    labels_1_dim = labels_dim + 1
    keys_dim = FeatureDim("keys", 2)
    qkv_dim_ = (2 * keys_dim) + input_dim
    num_heads_dim = SpatialDim("num_heads", 1)
    keys_h_dim = keys_dim.div_left(num_heads_dim)
    value_h_dim = input_dim.div_left(num_heads_dim)
    qkv_dim = 2 * keys_h_dim + value_h_dim
    time_kv_dim = SpatialDim("time:kv")
    time_kv_0_dim = SpatialDim("time:kv")

    net_dict = {
        "output": {
            "class": "copy",
            "from": "loop/output",
            "out_shape": {batch_dim, time_dim, ImplicitSparseDim(labels_1_dim)},
        },
        "encoder": {
            "class": "subnetwork",
            "from": [],
            "subnetwork": {
                "self_att": {
                    "class": "subnetwork",
                    "from": [],
                    "subnetwork": {
                        "qkv": {
                            "class": "subnetwork",
                            "from": [],
                            "subnetwork": {
                                "weight": {
                                    "class": "variable",
                                    "shape": [input_dim, qkv_dim_],
                                    "param_name": "param",
                                },
                                "bias": {"class": "variable", "shape": [qkv_dim_], "param_name": "param", "init": 0.0},
                                "output": {"class": "copy", "from": "bias", "out_shape": {qkv_dim_}},
                            },
                        },
                        "output": {
                            "class": "copy",
                            "from": "qkv",
                        },
                    },
                },
                "output": {
                    "class": "copy",
                    "from": "self_att",
                },
            },
        },
        "out_label_logits": {
            "class": "subnetwork",
            "from": [],
            "subnetwork": {
                "weight": {
                    "class": "variable",
                    "shape": [input_dim, labels_1_dim],
                    "param_name": "param",
                },
                "bias": {"class": "variable", "shape": [labels_1_dim], "param_name": "param", "init": 0.0},
                "output": {"class": "copy", "from": "bias", "out_shape": {labels_1_dim}},
            },
        },
        "train_flag": {"class": "train_flag", "out_shape": {}},
        "cond": {
            "class": "cond",
            "from": [],
            "condition": "train_flag",
            "true_layer": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "encoder": {
                        "class": "subnetwork",
                        "from": [],
                        "subnetwork": {
                            "self_att": {
                                "class": "subnetwork",
                                "from": [],
                                "subnetwork": {
                                    "qkv": {
                                        "class": "subnetwork",
                                        "from": [],
                                        "subnetwork": {
                                            "dot": {
                                                "class": "dot",
                                                "from": [
                                                    "base:base:base:base:data:data",
                                                    "base:base:base:base:encoder/self_att/qkv/weight",
                                                ],
                                                "reduce": input_dim,
                                                "out_shape": {batch_dim, time_dim, qkv_dim_},
                                            },
                                            "add": {
                                                "class": "combine",
                                                "from": ["dot", "base:base:base:base:encoder/self_att/qkv"],
                                                "kind": "add",
                                                "out_shape": {batch_dim, time_dim, qkv_dim_},
                                            },
                                            "output": {
                                                "class": "copy",
                                                "from": "add",
                                                "out_shape": {batch_dim, time_dim, qkv_dim_},
                                            },
                                        },
                                    },
                                    "qkv_split_dims": {
                                        "class": "split_dims",
                                        "from": "qkv",
                                        "axis": qkv_dim_,
                                        "dims": (num_heads_dim, qkv_dim),
                                        "out_shape": {batch_dim, time_dim, num_heads_dim, qkv_dim},
                                    },
                                    "split": {
                                        "class": "split",
                                        "from": "qkv_split_dims",
                                        "axis": qkv_dim,
                                        "out_dims": (keys_h_dim, keys_h_dim, value_h_dim),
                                        "out_shape": {batch_dim, time_dim, num_heads_dim, qkv_dim},
                                    },
                                    "reinterpret_new_dim": {
                                        "class": "reinterpret_data",
                                        "set_dim_tags": {time_dim: time_kv_dim},
                                        "from": "split/1",
                                        "out_shape": {batch_dim, num_heads_dim, keys_h_dim, time_kv_dim},
                                    },
                                    "reinterpret_new_dim_0": {
                                        "class": "reinterpret_data",
                                        "set_dim_tags": {time_dim: time_kv_dim},
                                        "from": "split/2",
                                        "out_shape": {batch_dim, num_heads_dim, value_h_dim, time_kv_dim},
                                    },
                                    "add": {
                                        "class": "copy",
                                        "from": "split/0",
                                    },
                                    "add_0": {
                                        "class": "copy",
                                        "from": "split/0",
                                    },
                                    "dot": {
                                        "class": "dot",
                                        "from": ["add", "reinterpret_new_dim"],
                                        "reduce": keys_h_dim,
                                        "out_shape": {batch_dim, time_dim, num_heads_dim, time_kv_dim},
                                    },
                                    "add_1": {
                                        "class": "copy",
                                        "from": "dot",
                                    },
                                    "mul": {
                                        "class": "eval",
                                        "from": "add_1",
                                        "eval": "source(0) * 0.7071067811865476",
                                        "out_shape": {batch_dim, time_dim, num_heads_dim, time_kv_dim},
                                    },
                                    "att_weights": {
                                        "class": "softmax_over_spatial",
                                        "from": "mul",
                                        "axis": time_kv_dim,
                                        "out_shape": {batch_dim, time_dim, num_heads_dim, time_kv_dim},
                                    },
                                    "dropout": {
                                        "class": "dropout",
                                        "from": "att_weights",
                                        "dropout": 0.1,
                                        "dropout_axis": time_kv_dim,
                                        "out_shape": {batch_dim, time_dim, num_heads_dim, time_kv_dim},
                                    },
                                    "att": {
                                        "class": "dot",
                                        "from": ["dropout", "reinterpret_new_dim_0"],
                                        "reduce": time_kv_dim,
                                        "out_shape": {batch_dim, time_dim, num_heads_dim, value_h_dim},
                                    },
                                    "output_0": {
                                        "class": "merge_dims",
                                        "from": "att",
                                        "axes": (num_heads_dim, value_h_dim),
                                        "out_dim": input_dim,
                                        "out_shape": {batch_dim, time_dim, input_dim},
                                    },
                                    "output": {
                                        "class": "copy",
                                        "from": "output_0",
                                        "out_shape": {batch_dim, time_dim, input_dim},
                                    },
                                },
                            },
                            "output": {
                                "class": "copy",
                                "from": "self_att",
                                "out_shape": {batch_dim, time_dim, input_dim},
                            },
                        },
                    },
                    "output": {"class": "copy", "from": "encoder", "out_shape": {batch_dim, time_dim, input_dim}},
                },
            },
            "false_layer": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "encoder": {
                        "class": "subnetwork",
                        "from": [],
                        "subnetwork": {
                            "self_att": {
                                "class": "subnetwork",
                                "from": [],
                                "subnetwork": {
                                    "qkv": {
                                        "class": "subnetwork",
                                        "from": [],
                                        "subnetwork": {
                                            "dot": {
                                                "class": "dot",
                                                "from": [
                                                    "base:base:base:base:data:data",
                                                    "base:base:base:base:encoder/self_att/qkv/weight",
                                                ],
                                                "reduce": input_dim,
                                                "out_shape": {batch_dim, time_dim, qkv_dim_},
                                            },
                                            "add": {
                                                "class": "combine",
                                                "from": ["dot", "base:base:base:base:encoder/self_att/qkv"],
                                                "kind": "add",
                                                "out_shape": {batch_dim, time_dim, qkv_dim_},
                                            },
                                            "output": {
                                                "class": "copy",
                                                "from": "add",
                                                "out_shape": {batch_dim, time_dim, qkv_dim_},
                                            },
                                        },
                                    },
                                    "qkv_split_dims": {
                                        "class": "split_dims",
                                        "from": "qkv",
                                        "axis": qkv_dim_,
                                        "dims": (num_heads_dim, qkv_dim),
                                        "out_shape": {batch_dim, time_dim, num_heads_dim, qkv_dim},
                                    },
                                    "split": {
                                        "class": "split",
                                        "from": "qkv_split_dims",
                                        "axis": qkv_dim,
                                        "out_dims": (keys_h_dim, keys_h_dim, value_h_dim),
                                        "out_shape": {batch_dim, time_dim, num_heads_dim, qkv_dim},
                                    },
                                    "reinterpret_new_dim": {
                                        "class": "reinterpret_data",
                                        "set_dim_tags": {time_dim: time_kv_0_dim},
                                        "from": "split/1",
                                        "out_shape": {batch_dim, num_heads_dim, keys_h_dim, time_kv_0_dim},
                                    },
                                    "reinterpret_new_dim_0": {
                                        "class": "reinterpret_data",
                                        "set_dim_tags": {time_dim: time_kv_0_dim},
                                        "from": "split/2",
                                        "out_shape": {batch_dim, num_heads_dim, value_h_dim, time_kv_0_dim},
                                    },
                                    "add": {
                                        "class": "copy",
                                        "from": "split/0",
                                    },
                                    "add_0": {
                                        "class": "copy",
                                        "from": "split/0",
                                    },
                                    "dot": {
                                        "class": "dot",
                                        "from": ["add", "reinterpret_new_dim"],
                                        "reduce": keys_h_dim,
                                        "out_shape": {batch_dim, time_dim, num_heads_dim, time_kv_0_dim},
                                    },
                                    "add_1": {
                                        "class": "copy",
                                        "from": "dot",
                                    },
                                    "mul": {
                                        "class": "eval",
                                        "from": "add_1",
                                        "eval": "source(0) * 0.7071067811865476",
                                        "out_shape": {batch_dim, time_dim, num_heads_dim, time_kv_0_dim},
                                    },
                                    "att_weights": {
                                        "class": "softmax_over_spatial",
                                        "from": "mul",
                                        "axis": time_kv_0_dim,
                                        "out_shape": {batch_dim, time_dim, num_heads_dim, time_kv_0_dim},
                                    },
                                    "dropout": {
                                        "class": "dropout",
                                        "from": "att_weights",
                                        "dropout": 0.1,
                                        "dropout_axis": time_kv_0_dim,
                                        "out_shape": {batch_dim, time_dim, num_heads_dim, time_kv_0_dim},
                                    },
                                    "att": {
                                        "class": "dot",
                                        "from": ["dropout", "reinterpret_new_dim_0"],
                                        "reduce": time_kv_0_dim,
                                        "out_shape": {batch_dim, time_dim, num_heads_dim, value_h_dim},
                                    },
                                    "output_0": {
                                        "class": "merge_dims",
                                        "from": "att",
                                        "axes": (num_heads_dim, value_h_dim),
                                        "out_dim": input_dim,
                                        "out_shape": {batch_dim, time_dim, input_dim},
                                    },
                                    "output": {
                                        "class": "copy",
                                        "from": "output_0",
                                        "out_shape": {batch_dim, time_dim, input_dim},
                                    },
                                },
                            },
                            "output": {
                                "class": "copy",
                                "from": "self_att",
                                "out_shape": {batch_dim, time_dim, input_dim},
                            },
                        },
                    },
                    "output": {"class": "copy", "from": "encoder", "out_shape": {batch_dim, time_dim, input_dim}},
                },
            },
            "out_shape": {batch_dim, time_dim, input_dim},
            "name_scope": "",
        },
        "loop": {
            "class": "rec",
            "from": [],
            "unit": {
                "rec_unstack": {
                    "class": "rec_unstack",
                    "from": "base:cond",
                    "axis": time_dim,
                    "out_shape": {batch_dim, input_dim},
                },
                "model": {
                    "class": "subnetwork",
                    "from": [],
                    "subnetwork": {
                        "out_label_logits": {
                            "class": "subnetwork",
                            "from": [],
                            "subnetwork": {
                                "dot": {
                                    "class": "dot",
                                    "from": ["base:base:rec_unstack", "base:base:base:out_label_logits/weight"],
                                    "reduce": input_dim,
                                    "out_shape": {batch_dim, labels_1_dim},
                                },
                                "add": {
                                    "class": "combine",
                                    "from": ["dot", "base:base:base:out_label_logits"],
                                    "kind": "add",
                                    "out_shape": {batch_dim, labels_1_dim},
                                },
                                "output": {"class": "copy", "from": "add", "out_shape": {batch_dim, labels_1_dim}},
                            },
                        },
                        "output": {"class": "copy", "from": "out_label_logits", "out_shape": {batch_dim, labels_1_dim}},
                    },
                    "name_scope": "",
                },
                "softmax": {
                    "class": "softmax_over_spatial",
                    "from": "model",
                    "axis": labels_1_dim,
                    "log_space": True,
                    "out_shape": {batch_dim, labels_1_dim},
                },
                "choice": {
                    "class": "choice",
                    "from": "softmax",
                    "target": None,
                    "beam_size": 3,
                    "search": True,
                    "input_type": "log_prob",
                    "length_normalization": False,
                    "initial_output": "base:constant",
                    "out_shape": {batch_dim, ImplicitSparseDim(labels_1_dim)},
                },
                "output": {
                    "class": "copy",
                    "from": "choice",
                    "out_shape": {batch_dim, ImplicitSparseDim(labels_1_dim)},
                },
            },
            "axis": time_dim,
            "out_shape": {batch_dim, time_dim, ImplicitSparseDim(labels_1_dim)},
            "name_scope": "",
        },
        "length": {"class": "length", "from": ["data:data"], "axis": time_dim, "out_shape": {batch_dim}},
        "mul": {"class": "eval", "from": "encoder", "eval": "source(0) * 2", "out_shape": {}},
        "constant": {
            "class": "constant",
            "value": 5,
            "shape": [batch_dim],
            "sparse_dim": labels_1_dim,
            "shape_deps": ["data:data"],
        },
    }
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        net.initialize_params(session=session)
        out = net.get_default_output_layer()
        session.run(out.output.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_CondLayer_multiple_outputs():
    net_dict = {
        "cond": {
            "class": "cond",
            "from": [],
            "condition": "data:cond",
            "true_layer": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "a": {"class": "constant", "value": 2, "is_output_layer": True},
                    "b": {"class": "constant", "value": 3, "is_output_layer": True},
                    "output": {"class": "copy", "from": "a"},
                },
            },
            "false_layer": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "a": {"class": "constant", "value": 5, "is_output_layer": True},
                    "b": {"class": "constant", "value": 7, "is_output_layer": True},
                    "output": {"class": "copy", "from": "a"},
                },
            },
        },
        "output": {"class": "combine", "kind": "mul", "from": ["cond/a", "cond/b"]},
    }
    config = Config(dict(extern_data={"cond": {"dim_tags": (), "dtype": "bool", "available_for_inference": True}}))
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        out_t = net.get_default_output_layer().output.placeholder
        assert session.run(out_t, feed_dict={net.extern_data.data["cond"].placeholder: True}) == 6
        assert session.run(out_t, feed_dict={net.extern_data.data["cond"].placeholder: False}) == 35


def test_ScatterNdLayer_RangeLayer():
    from returnn.tf.util.data import batch_dim, Dim

    n_batch, n_time, n_ts, n_out = 2, 3, 6, 11
    time_dim = SpatialDim("T")
    feat_dim = FeatureDim("F", dimension=7)
    ts_dim = SpatialDim("ts", dimension=n_ts)
    rnd = numpy.random.RandomState(42)
    config = Config(
        {
            "debug_print_layer_output_template": True,
            "extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}},
        }
    )
    net_dict = {
        "t": {
            "class": "eval",
            "from": [],
            "eval": "tf.convert_to_tensor([1, 2])",
            "out_type": {"shape": (), "dtype": "int32"},
        },  # (B,)
        "range": {"class": "range", "limit": n_ts, "out_spatial_dim": ts_dim},  # (Ts,)
        "add_t": {
            "class": "combine",
            "kind": "add",
            "from": ["t", "range"],
            "out_shape": {batch_dim, ts_dim},
        },  # (B,Ts)
        "t_rel_var": {"class": "variable", "shape": (ts_dim, n_out), "init": "glorot_uniform"},  # (B,Ts,D)
        "output": {
            "class": "scatter_nd",
            "from": "t_rel_var",
            "position": "add_t",
            "position_axis": ts_dim,
            "output_dim_via_time_from": "data",
            "filter_invalid_indices": True,
        },
    }
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)

        fetches = network.get_fetches_dict()
        data_input = network.extern_data.data["data"]
        out_layer = network.get_default_output_layer()
        assert isinstance(out_layer, ScatterNdLayer)
        assert out_layer.output.shape == (None, 11)
        assert (
            out_layer.output.feature_dim_axis_or_unspecified is NotSpecified and out_layer.output.feature_dim_axis == 2
        )
        assert out_layer.output.time_dim_axis == 1

        session.run(
            tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step_var])
        )
        info, out = session.run(
            (fetches, out_layer.output.placeholder),
            feed_dict={
                data_input.batch.dim: n_batch,
                data_input.placeholder: rnd.normal(size=(n_batch, n_time, feat_dim.dimension)).astype("float32"),
                data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
            },
        )
        print(info)
        print(out)  # random...


def _run_repeat_layer(session, net, input_data_layer):
    repetitions_data_layer = InternalLayer(
        name="repetitions",
        network=net,
        output=Data(**{"name": "repetitions", "shape": (None,), "feature_dim_axis": None, "dtype": "int32"}),
    )
    repetitions_data_layer.output.placeholder = tf.constant(
        [[1, 3, 2, 1, 3, 4, 1, 1, 2, 1], [3, 2, 1, 3, 0, 1, 1, 0, 0, 1]], dtype="int32"
    )  # [B, T] (sparse), additional 1 in second sequence to test masking
    repetitions_data_layer.output.size_placeholder = {0: tf.constant([10, 7], dtype="int32")}  # [B]

    opts = {
        "network": net,
        "name": "repeat_layer_test",
        "sources": [input_data_layer],
        "repetitions": repetitions_data_layer,
        "axis": "T",
    }
    out_data = RepeatLayer.get_out_data_from_opts(**opts)
    out_data.sanity_check()
    print(out_data)
    repeat_layer = RepeatLayer(output=out_data, **opts)
    print(repeat_layer.output)

    output, size_placeholder = session.run(
        [repeat_layer.output.placeholder, repeat_layer.output.size_placeholder.as_dict()],
        feed_dict=make_feed_dict(net.extern_data, n_batch=2),
    )
    assert numpy.all(numpy.equal(size_placeholder[0], numpy.asarray([19, 11])))
    assert numpy.all(numpy.equal(output.shape, numpy.asarray([2, 19, 5])))
    # the 6 last positions of the second sequence need to be padded with zeros
    assert numpy.all(numpy.equal(output[1, 11:], 0))
    assert out_data.shape == (None, 5)
    assert out_data.batch_dim_axis == 0
    assert out_data.time_dim_axis == 1


def test_RepeatLayerBTF():
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData({"data": {"shape": (None, 3)}}))
        input_data_layer = InternalLayer(name="src", network=net, output=Data(name="src", shape=(None, 5), dim=5))
        input_data_layer.output.size_placeholder = {0: tf.constant([10, 7])}  # [B]
        input_data_layer.output.placeholder = tf_compat.v1.random_uniform((2, 10, 5))  # [B, T, F]

        _run_repeat_layer(session, net, input_data_layer)


def test_RepeatLayerTBF():
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData({"data": {"shape": (None, 3)}}))
        input_data_layer = InternalLayer(
            name="src",
            network=net,
            output=Data(**{"name": "src", "shape": (None, 5), "dim": 5, "batch_dim_axis": 1, "time_dim_axis": 0}),
        )
        input_data_layer.output.size_placeholder = {0: tf.constant([10, 7])}  # [B]
        input_data_layer.output.placeholder = tf_compat.v1.random_uniform((10, 2, 5))  # [T, B, F]

        _run_repeat_layer(session, net, input_data_layer)


def test_RepeatLayerBFT():
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData({"data": {"shape": (None, 3)}}))
        input_data_layer = InternalLayer(
            name="src",
            network=net,
            output=Data(**{"name": "src", "shape": (5, None), "dim": 5, "time_dim_axis": 2, "feature_dim_axis": 1}),
        )
        input_data_layer.output.size_placeholder = {1: tf.constant([10, 7])}  # [B]
        input_data_layer.output.placeholder = tf_compat.v1.random_uniform((2, 5, 10))  # [B, F, T]

        _run_repeat_layer(session, net, input_data_layer)


def test_RepeatLayerTF():
    with make_scope() as session:
        # need to provide extern_data here to provide batch info
        net = TFNetwork(extern_data=ExternData(data={"data": {"dim": 2, "sparse": True}}), train_flag=True)
        input_data_layer = InternalLayer(
            name="src",
            network=net,
            output=Data(**{"name": "src", "shape": (None, 5), "dim": 5, "batch_dim_axis": None, "time_dim_axis": 0}),
        )
        input_data_layer.output.size_placeholder = {0: tf.constant([10])}  # []
        input_data_layer.output.placeholder = tf_compat.v1.random_uniform((10, 5))  # [T, F]

        _run_repeat_layer(session, net, input_data_layer)


def test_RepeatLayer_int_repetitions():
    with make_scope() as session:
        n_out = 5
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "data": {"dim": n_out},
                },
            }
        )
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict({"output": {"class": "repeat", "repetitions": 3, "axis": "F", "from": ["data"]}})
        session.run(tf_compat.v1.global_variables_initializer())
        out = net.layers["output"].output
        n_batch = 3
        max_seq_len = 10
        feed = make_feed_dict(net.extern_data.data.values(), n_batch=n_batch, n_time=max_seq_len, same_time=True)
        v = session.run(out.placeholder, feed_dict=feed)
        input_data = feed[net.extern_data.data["data"].placeholder]

        assert out.batch_dim_axis == 0
        ref = numpy.swapaxes(numpy.repeat(input_data, 3, axis=-1), -1, out.feature_dim_axis)
        numpy.testing.assert_allclose(ref, v, rtol=1e-5)


def test_RepeatLayer_int():
    # https://github.com/rwth-i6/returnn_common/issues/162
    from returnn.tf.util.data import batch_dim, SpatialDim

    time_dim = SpatialDim("time")
    config = Config({"extern_data": {"data": {"dim_tags": (batch_dim, time_dim), "dtype": "float32"}}})
    _repeat_out_dim = time_dim * 5
    net_dict = {
        "output": {
            "class": "repeat",
            "from": "data:data",
            "repetitions": 5,
            "axis": time_dim,
            "out_dim": _repeat_out_dim,
            "out_shape": {batch_dim, _repeat_out_dim},
        },
    }
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        in_ = net.extern_data.data["data"]
        out = net.get_default_output_layer().output
        in_time = in_.get_time_dim_tag()
        out_time = out.get_time_dim_tag()
        assert in_time * 5 == out_time, "in %r vs out %r" % (in_, out)
        in_seq_len_ = in_time.dyn_size_ext.placeholder
        out_seq_len_ = out_time.dyn_size_ext.placeholder
        _, in_seq_len, _, out_seq_len = session.run(
            (in_.placeholder, in_seq_len_, out.placeholder, out_seq_len_), feed_dict=make_feed_dict(net.extern_data)
        )
        assert (in_seq_len * 5 == out_seq_len).all()


def test_RepeatLayer_as_loss_flatten():
    # https://github.com/rwth-i6/returnn_common/issues/201
    # https://github.com/rwth-i6/returnn/issues/1115
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    in_dim = FeatureDim("in", 3)
    config = Config(dict(extern_data={"data": {"dim_tags": (batch_dim, time_dim, in_dim)}}))
    net_dict = {
        "output": {
            "class": "repeat",
            "from": "data:data",
            "repetitions": 1,
            "axis": time_dim,
            "out_dim": time_dim,
            "loss": "as_is",
            "out_shape": {batch_dim, time_dim, in_dim},
        },
    }
    with make_scope() as session:
        net = TFNetwork(config=config, eval_flag=True)
        net.construct_from_dict(net_dict)
        feed_dict = make_feed_dict(net.extern_data)
        fetches = net.get_fetches_dict(should_eval=True)
        session.run(fetches, feed_dict=feed_dict)


def test_TileLayer():
    with make_scope() as session:
        n_out = 5
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "data": {"dim": n_out},
                },
            }
        )
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict({"output": {"class": "tile", "multiples": {"F": 3}, "from": ["data"]}})
        session.run(tf_compat.v1.global_variables_initializer())
        out = net.layers["output"].output.placeholder
        n_batch = 3
        max_seq_len = 10
        feed = make_feed_dict(net.extern_data.data.values(), n_batch=n_batch, n_time=max_seq_len, same_time=True)
        v = session.run(out, feed_dict=feed)
        input_len = feed[net.extern_data.data["data"].size_placeholder[0]]
        input_data = feed[net.extern_data.data["data"].placeholder]

        ref = numpy.tile(input_data, [1, 1, 3])

        numpy.testing.assert_allclose(ref, v, rtol=1e-5)


def test_ScatterNdLayer_RangeLayer_RangeInAxisLayer():
    from returnn.tf.util.data import batch_dim, Dim, ImplicitDynSizeDim

    n_batch, n_time, n_ts, n_in, n_out = 2, 3, 6, 7, 11
    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("in-feature", dimension=n_in)
    ts_dim = SpatialDim("ts", dimension=n_ts)
    rnd = numpy.random.RandomState(42)
    config = Config(
        {
            "debug_print_layer_output_template": True,
            "extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}},
        }
    )
    net_dict = {
        "t": {
            "class": "range_in_axis",
            "axis": "t",
            "from": "data",
            "out_shape": {time_dim, ImplicitDynSizeDim(batch_dim)},
        },  # (T,)
        "range": {"class": "range", "limit": n_ts, "out_spatial_dim": ts_dim},  # (Ts,)
        "add_t": {
            "class": "combine",
            "kind": "add",
            "from": ["t", "range"],
            "out_shape": {time_dim, ts_dim, ImplicitDynSizeDim(batch_dim)},
        },  # (T,Ts)
        "t_rel_var": {"class": "variable", "shape": (ts_dim, n_out), "init": "glorot_uniform"},  # (Ts,D)
        "output": {
            "class": "scatter_nd",
            "from": "t_rel_var",
            "position": "add_t",
            "position_axis": ts_dim,
            "output_dim_via_time_from": "data",
            "filter_invalid_indices": True,
        },  # (T,T,D)
    }
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)

        t_layer = network.layers["t"]
        assert isinstance(t_layer, RangeInAxisLayer)
        assert t_layer.output.time_dim_axis == 0

        fetches = network.get_fetches_dict()
        data_input = network.extern_data.data["data"]
        out_layer = network.get_default_output_layer()
        assert isinstance(out_layer, ScatterNdLayer)
        assert out_layer.output.shape == (None, None, 11)
        assert (
            out_layer.output.feature_dim_axis_or_unspecified is NotSpecified and out_layer.output.feature_dim_axis == 2
        )
        assert out_layer.output.time_dim_axis == 0

        session.run(
            tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step_var])
        )
        info, out = session.run(
            (fetches, out_layer.output.placeholder),
            feed_dict={
                data_input.placeholder: rnd.normal(size=(n_batch, n_time, n_in)).astype("float32"),
                data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
            },
        )
        print(info)
        print(out)  # random...


def test_ScatterNdLayer_pos_batch_last_dim():
    config = Config({"debug_print_layer_output_template": True, "extern_data": {"data": {"dim": 13}}})
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        data = network.construct_layer({}, "data")
        pos = InternalLayer(
            name="pos",
            network=network,
            output=Data(
                name="pos",
                shape=(None, 6),
                dtype="int32",
                sparse=True,
                dim=None,
                batch_dim_axis=2,
                auto_create_placeholders=True,
            ),
        )
        val = InternalLayer(
            name="val",
            network=network,
            output=Data(name="var", shape=(6, 11), time_dim_axis=None, auto_create_placeholders=True),
        )
        scatter_opts = dict(
            name="scatter",
            network=network,
            sources=[val],
            position=pos,
            position_axis="dim:6",
            output_dim_via_time_from=data,
            filter_invalid_indices=True,
        )
        scatter_out_template = ScatterNdLayer.get_out_data_from_opts(**scatter_opts)
        print("scatter out:", scatter_out_template)
        assert scatter_out_template.shape == (None, None, 11) and scatter_out_template.batch_ndim == 4
        scatter = ScatterNdLayer(output=scatter_out_template, **scatter_opts)
        print("scatter out dim tags:")
        pprint(scatter.output.get_batch_shape_dim_tags())
        assert scatter.output.get_size_dim_tag(0) == pos.output.get_time_dim_tag()
        assert scatter.output.get_size_dim_tag(1) == data.output.get_time_dim_tag()
        session.run(scatter.output.placeholder, feed_dict=make_feed_dict([data.output, pos.output, val.output]))


def test_ConvLayer_get_valid_out_dim():
    assert ConvLayer.calc_out_dim(in_dim=10, stride=1, filter_size=2, padding="same") == 10
    assert ConvLayer.calc_out_dim(in_dim=10, stride=1, filter_size=3, padding="same") == 10
    assert ConvLayer.calc_out_dim(in_dim=10, stride=1, filter_size=2, padding="valid") == 9
    assert ConvLayer.calc_out_dim(in_dim=10, stride=1, filter_size=3, padding="valid") == 8
    assert ConvLayer.calc_out_dim(in_dim=10, stride=2, filter_size=2, padding="valid") == 5
    assert ConvLayer.calc_out_dim(in_dim=10, stride=3, filter_size=2, padding="valid") == 3
    assert ConvLayer.calc_out_dim(in_dim=10, stride=3, filter_size=1, padding="valid") == 4
    assert ConvLayer.calc_out_dim(in_dim=10, stride=3, filter_size=2, padding="same") == 4
    assert ConvLayer.calc_out_dim(in_dim=41, stride=1, filter_size=2, padding="valid") == 40
    assert ConvLayer.calc_out_dim(in_dim=40, stride=2, filter_size=2, padding="valid") == 20
    assert ConvLayer.calc_out_dim(in_dim=2, stride=1, filter_size=3, padding="valid") == 0


def test_LengthLayer():
    net_dict = {
        "output": {"class": "length", "from": "data", "axis": "T"},
    }
    with make_scope() as session:
        config = Config({"extern_data": {"data": {"dim": 10}}})
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        in_ = net.extern_data.get_default_input_data()
        out = net.get_default_output_layer().output
        in_v, in_size_v, out_v = session.run(
            (in_.placeholder, in_.size_placeholder[0], out.placeholder), feed_dict=make_feed_dict(net.extern_data)
        )
        n_batch, n_time, n_feat = in_v.shape
        assert out_v.shape == in_size_v.shape == (n_batch,)
        assert list(out_v) == list(in_size_v)


def test_LengthLayer_batch():
    net_dict = {
        "input_flat": {"class": "flatten_batch", "from": "data"},  # [B_T,F]
        "output": {"class": "length", "from": "input_flat", "axis": "B"},  # scalar -> B_T
    }
    with make_scope() as session:
        config = Config({"extern_data": {"data": {"dim": 10}}})
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        in_ = net.extern_data.get_default_input_data()
        flat = net.layers["input_flat"].output
        out = net.get_default_output_layer().output
        in_shape_v, in_lens_v, flat_shape_v, out_v = session.run(
            (tf.shape(in_.placeholder), in_.get_sequence_lengths(), tf.shape(flat.placeholder), out.placeholder),
            feed_dict=make_feed_dict(net.extern_data),
        )
        assert in_lens_v.shape == (in_shape_v[0],)
        assert max(in_lens_v) == in_shape_v[1]
        assert sum(in_lens_v) == flat_shape_v[0] == out_v


def test_LengthLayer_generic_dim():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    out_time_dim = time_dim + 2
    feat_dim = FeatureDim("feat", 3)
    config = Config({"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}}})
    net_dict = {
        "output": {"class": "length", "from": "data", "axis": out_time_dim},
    }
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        in_ = net.extern_data.get_default_input_data()
        out = net.get_default_output_layer().output
        in_v, in_size_v, out_v = session.run(
            (in_.placeholder, in_.size_placeholder[0], out.placeholder), feed_dict=make_feed_dict(net.extern_data)
        )
        n_batch, n_time, n_feat = in_v.shape
        assert out_v.shape == in_size_v.shape == (n_batch,)
        assert list(out_v) == list(in_size_v + 2)


def test_LengthLayer_static_dim_sparse():
    from returnn.tf.util.data import batch_dim, FeatureDim

    feat_dim = FeatureDim("feat", 3)
    config = Config({"extern_data": {"data": {"dim_tags": [batch_dim], "sparse_dim": feat_dim}}})
    net_dict = {
        "output": {"class": "length", "from": "data", "axis": feat_dim, "sparse": True},
    }
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        out = net.get_default_output_layer().output
        print("out:", out)
        assert out.sparse
        out_v = session.run(out.placeholder, feed_dict=make_feed_dict(net.extern_data))
        assert out_v.shape == ()
        assert out_v == feat_dim.dimension


def test_RandIntLayer():
    with make_scope() as session:
        from returnn.tf.util.data import Dim

        n_out = 5
        config = Config({"debug_print_layer_output_template": True, "extern_data": {"data": {"dim": n_out}}})
        net = TFNetwork(config=config, train_flag=True)
        n_batch = 3
        max_seq_len = 10
        feed = make_feed_dict(net.extern_data.data.values(), n_batch=n_batch, n_time=max_seq_len, same_time=True)
        size_placeholder = net.extern_data.data["data"].size_placeholder[0]
        input_len = feed[size_placeholder]
        sz = (
            Dim(description="feature", kind=Dim.Types.Feature, dimension=5),
            Dim(kind=Dim.Types.Batch, dimension=None),
            net.extern_data.data["data"].get_size_dim_tag(0),
            3,
        )
        net.construct_from_dict({"output": {"class": "rand_int", "shape": sz, "minval": 3, "maxval": 10, "seed": 42}})
        session.run(tf_compat.v1.global_variables_initializer())
        out = net.layers["output"].output.placeholder
        v = session.run(out, feed_dict=feed)

        assert v.shape == (5, n_batch, max(input_len), 3)


def test_rand_indices():
    with make_scope() as session:
        from returnn.tf.util.data import FeatureDim, SpatialDim, batch_dim

        feature_dim = FeatureDim("feature", 5)
        time_dim = SpatialDim("time")
        config = Config({"extern_data": {"data": {"dim_tags": (batch_dim, time_dim, feature_dim)}}})
        net = TFNetwork(config=config, train_flag=True)
        sz = (batch_dim, time_dim, SpatialDim("other-spatial", 7))
        net.construct_from_dict(
            {
                "flat": {"class": "flatten_batch", "from": "data"},
                "length_flat": {"class": "length", "from": "flat", "axis": batch_dim},
                "indices_flat": {"class": "rand_int", "shape": sz, "minval": 0, "maxval": "length_flat", "seed": 42},
                "output": {"class": "gather", "from": "flat", "axis": batch_dim, "position": "indices_flat"},
            }
        )

        n_batch, n_time = 3, 11
        indices_flat, output = session.run(
            (net.layers["indices_flat"].output.placeholder, net.layers["output"].output.placeholder),
            feed_dict=make_feed_dict(net.extern_data, n_batch=n_batch, n_time=n_time),
        )
        assert indices_flat.shape == (n_batch, n_time, sz[-1].dimension)
        assert output.shape == (n_batch, n_time, sz[-1].dimension, feature_dim.dimension)


def test_RandomLayer():
    # https://github.com/rwth-i6/returnn_common/issues/197
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    input_dim = FeatureDim("input", 3)
    config = Config({"extern_data": {"data": {"dim_tags": (batch_dim, time_dim, input_dim)}}})
    net_dict = {
        "random": {
            "class": "random",
            "shape": [batch_dim, input_dim],
            "distribution": "normal",
            "mean": 0.0,
            "stddev": 1.0,
        },
        "output": {
            "class": "combine",
            "kind": "add",
            "from": ["data:data", "random"],
            "out_shape": {batch_dim, time_dim, input_dim},
        },
    }
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        out = net.get_default_output_layer().output.copy_as_time_major()
        out_np = session.run(out.placeholder, feed_dict=make_feed_dict(net.extern_data))
        print(out_np)


def test_RandomLayer_shape_deps():
    # https://github.com/rwth-i6/returnn_common/issues/197
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    time_pool_dim = SpatialDim("time-pool")
    input_dim = FeatureDim("input", 3)
    config = Config({"extern_data": {"data": {"dim_tags": (batch_dim, time_dim, input_dim)}}})
    net_dict = {
        "pool": {"class": "pool", "pool_size": [2], "mode": "max", "from": "data", "out_spatial_dims": [time_pool_dim]},
        "random": {
            "class": "random",
            "shape": [batch_dim, time_pool_dim, input_dim],
            "shape_deps": ["pool"],
            "distribution": "normal",
            "mean": 0.0,
            "stddev": 1.0,
        },
        "output": {
            "class": "combine",
            "kind": "add",
            "from": ["random", "pool"],  # the order is relevant to test shape_deps
            "out_shape": {batch_dim, time_pool_dim, input_dim},
        },
    }
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        out = net.get_default_output_layer().output.copy_as_time_major()
        out_np = session.run(out.placeholder, feed_dict=make_feed_dict(net.extern_data))
        print(out_np)


def test_RandomLayer_in_loop():
    # https://github.com/rwth-i6/returnn/issues/1044
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    input_dim = FeatureDim("input", 3)
    config = Config({"extern_data": {"data": {"dim_tags": (batch_dim, time_dim, input_dim)}}})
    random_state_dim = FeatureDim("random-state", 3)
    net_dict = {
        "output": {
            "class": "rec",
            "from": "data",
            "unit": {
                "rnd": {
                    "class": "subnetwork",
                    "from": [],
                    "subnetwork": {
                        "random": {
                            "class": "random",
                            "shape": [batch_dim, input_dim],
                            "distribution": "normal",
                            "mean": 0.0,
                            "stddev": 1.0,
                            "explicit_state": "base:base:rnd_state_var0",
                            "auto_update_state": True,
                        },
                        "output": {"class": "copy", "from": "random", "out_shape": {batch_dim, input_dim}},
                    },
                    "out_shape": {batch_dim, input_dim},
                },
                "output": {
                    "class": "eval",
                    "from": ["data:source", "rnd"],
                    "eval": "source(0) * 0.0 + source(1)",
                    "out_shape": {batch_dim, input_dim},
                },
            },
            "axis": time_dim,
            "out_shape": {batch_dim, time_dim, input_dim},
            "name_scope": "",
        },
        "random_state_init": {
            "class": "random_state_init",
            "out_dim": random_state_dim,
            "out_shape": {random_state_dim},
        },
        "rnd_state_var0": {
            "class": "variable",
            "shape": [random_state_dim],
            "param_name": "param",
            "dtype": "int64",
            "init_by_layer": "random_state_init",
            "name_scope": "rnd/state_var0",
        },
    }
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        out = net.get_default_output_layer().output.copy_as_time_major()
        out_np = session.run(out.placeholder, feed_dict=make_feed_dict(net.extern_data))
        print(out_np)
        out0_np = out_np[:1]  # [1,B,D]
        print((out0_np == out_np))
        assert not (out0_np == out_np).all()  # not all the same


def test_RandomLayer_zero_shape():
    # https://github.com/rwth-i6/returnn/issues/1190
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    input_dim = FeatureDim("input", 3)
    config = Config({"extern_data": {"data": {"dim_tags": (batch_dim, time_dim, input_dim)}}})
    net_dict = {
        "random": {
            "class": "random",
            "shape": [batch_dim, time_dim, FeatureDim("zero", 0)],
            "shape_deps": ["data"],
            "distribution": "normal",
            "mean": 0.0,
            "stddev": 1.0,
        },
        "output": {
            "class": "copy",
            "from": ["data", "random"],
        },
    }
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        in_ = net.extern_data.get_default_input_data()
        out = net.get_default_output_layer().output
        in_np, out_np = session.run((in_.placeholder, out.placeholder), feed_dict=make_feed_dict(net.extern_data))
        print(in_np.shape, out_np.shape)
        print(out_np)
        assert in_np.shape == out_np.shape


def test_untrainable_params():
    with make_scope() as session:
        config = Config()
        n_in, n_out = 2, 3
        config.update(
            {
                "num_outputs": n_out,
                "num_inputs": n_in,
                "network": {
                    "l1": {"class": "linear", "activation": None, "n_out": n_out, "from": "data:data"},
                    "output": {
                        "class": "linear",
                        "activation": None,
                        "from": ["l1"],
                        "n_out": n_out,
                        "trainable": False,
                    },
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        l1 = network.layers["l1"]
        l2 = network.layers["output"]
        assert set(network.get_trainable_params()) == {l1.params["W"], l1.params["b"]}


def test_reuse_params():
    with make_scope() as session:
        config = Config()
        n_in, n_out = 2, 3
        config.update(
            {
                "num_outputs": n_out,
                "num_inputs": n_in,
                "network": {
                    "l1": {"class": "linear", "activation": None, "n_out": n_out, "from": "data:data"},
                    "output": {
                        "class": "linear",
                        "activation": None,
                        "n_out": n_out,
                        "from": "data:data",
                        "reuse_params": "l1",
                    },
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        l1 = network.layers["l1"]
        l2 = network.layers["output"]
        assert set(l1.params.keys()) == {"W", "b"}
        assert set(l2.params.keys()) == set()
        assert set(network.get_trainable_params()) == {l1.params["W"], l1.params["b"]}


def test_reuse_params_map_custom():
    with make_scope() as session:
        config = Config()
        n_in, n_out = 2, 3
        config.update(
            {
                "num_outputs": n_out,
                "num_inputs": n_in,
                "network": {
                    "l1": {
                        "class": "linear",
                        "activation": "tanh",
                        "with_bias": False,
                        "n_out": 5,
                        "from": "data:data",
                    },
                    "output": {
                        "class": "linear",
                        "activation": None,
                        "n_out": n_in,
                        "from": ["l1"],
                        "target": "data",
                        "reuse_params": {
                            "map": {
                                "W": {
                                    "reuse_layer": "l1",
                                    "custom": (lambda reuse_layer, **kwargs: tf.transpose(reuse_layer.params["W"])),
                                },
                                "b": None,
                            }
                        },
                    },
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        l1 = network.layers["l1"]
        l2 = network.layers["output"]
        assert set(l1.params.keys()) == {"W"}
        assert set(l2.params.keys()) == {"b"}
        assert set(network.get_trainable_params()) == {l1.params["W"], l2.params["b"]}


def test_reuse_params_map_custom_rev():
    with make_scope() as session:
        config = Config()
        n_in, n_out = 2, 3
        config.update(
            {
                "num_outputs": n_out,
                "num_inputs": n_in,
                "network": {
                    "output": {
                        "class": "linear",
                        "activation": "tanh",
                        "with_bias": False,
                        "from": ["l1"],
                        "n_out": n_in,
                    },
                    "l1": {
                        "class": "linear",
                        "activation": None,
                        "n_out": 5,
                        "from": ["data"],
                        "target": "data",
                        "reuse_params": {
                            "map": {
                                "W": {
                                    "reuse_layer": "output",
                                    "custom": (lambda reuse_layer, **kwargs: tf.transpose(reuse_layer.params["W"])),
                                },
                                "b": None,
                            }
                        },
                    },
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        l1 = network.layers["l1"]
        l2 = network.layers["output"]
        assert set(l1.params.keys()) == {"b"}
        assert set(l2.params.keys()) == {"W"}
        assert set(network.get_trainable_params()) == {l2.params["W"], l1.params["b"]}


def test_reuse_params_map_custom_dep_loop():
    config = Config()
    n_in, n_out = 2, 3
    config.update(
        {
            "num_outputs": n_out,
            "num_inputs": n_in,
            "network": {
                "encoder": {"class": "copy", "from": ["data"]},
                "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
                "inv_fertility": {
                    "class": "linear",
                    "activation": "sigmoid",
                    "with_bias": False,
                    "from": ["encoder"],
                    "n_out": 1,
                },
                "output": {
                    "class": "rec",
                    "from": [],
                    "unit": {
                        "output": {
                            "class": "choice",
                            "target": "classes",
                            "beam_size": 5,
                            "from": ["output_prob"],
                            "initial_output": 0,
                        },
                        "end": {"class": "compare", "from": ["output"], "value": 0},
                        "target_embed": {
                            "class": "linear",
                            "activation": None,
                            "with_bias": False,
                            "from": ["output"],
                            "n_out": 6,
                            "initial_output": 0,
                        },
                        "weight_feedback": {
                            "class": "linear",
                            "activation": None,
                            "with_bias": False,
                            "from": ["prev:accum_att_weights"],
                            "n_out": 10,
                        },
                        "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
                        "prev_s_transformed": {
                            "class": "linear",
                            "activation": None,
                            "with_bias": False,
                            "from": ["prev_s_state"],
                            "n_out": 10,
                        },
                        "energy_in": {
                            "class": "combine",
                            "kind": "add",
                            "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"],
                            "n_out": 10,
                        },
                        "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
                        "energy": {
                            "class": "linear",
                            "activation": None,
                            "with_bias": False,
                            "from": ["energy_tanh"],
                            "n_out": 1,
                        },
                        "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},
                        "accum_att_weights": {
                            "class": "eval",
                            "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
                            "eval": "source(0) + source(1) * source(2) * 0.5",
                            "out_type": {"dim": 1, "shape": (None, 1)},
                        },
                        "att": {
                            "class": "generic_attention",
                            "weights": "att_weights",
                            "base": "base:encoder",
                            "auto_squeeze": True,
                        },
                        "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["target_embed", "att"], "n_out": 10},
                        "readout_in": {
                            "class": "linear",
                            "from": ["prev:s", "prev:target_embed", "att"],
                            "activation": None,
                            "n_out": 2 * 6,
                        },
                        "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
                        "output_prob": {
                            "class": "softmax",
                            "from": ["readout"],
                            "dropout": 0.3,
                            "reuse_params": {
                                "map": {
                                    "W": {
                                        "reuse_layer": "target_embed",
                                        "custom": (lambda reuse_layer, **kwargs: tf.transpose(reuse_layer.params["W"])),
                                    },
                                    "b": None,
                                }
                            },
                            "target": "classes",
                            "loss": "ce",
                            "loss_opts": {"label_smoothing": 0.1},
                        },
                    },
                    "target": "classes",
                    "max_seq_len": "max_len_from('base:encoder')",
                },
            },
        }
    )
    with make_scope() as session:
        print("Construct for training")
        from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell

        train_net = TFNetwork(config=config, train_flag=True)
        train_net.construct_from_dict(config.typed_dict["network"])
        train_rec_layer = train_net.layers["output"]
        assert isinstance(train_rec_layer, RecLayer)
        assert isinstance(train_rec_layer.cell, _SubnetworkRecCell)
        assert set(train_rec_layer.cell.input_layers_moved_out) == {"output", "target_embed"}
        assert set(train_rec_layer.cell.output_layers_moved_out) == {"output_prob", "readout", "readout_in"}
        assert isinstance(train_rec_layer.cell.output_layers_net, TFNetwork)
        assert set(train_rec_layer.cell.output_layers_net.layers["output_prob"].params.keys()) == {"b"}
    with make_scope() as session:
        print("Construct for search")
        search_net = TFNetwork(config=config, train_flag=False, eval_flag=True, search_flag=True)
        search_net.construct_from_dict(config.typed_dict["network"])


def test_name_scope():
    with make_scope() as session:
        n_in, n_out = 2, 3
        config = Config({"extern_data": {"data": {"dim": n_in}}})
        network = TFNetwork(config=config)
        net_dict = {
            "output": {"class": "linear", "n_out": n_out, "from": "data:data", "name_scope": "custom_name"},
        }
        network.construct_from_dict(net_dict)
        layer = network.layers["output"]
        param = layer.params["W"]
        assert param.name == "custom_name/W:0"


def test_name_scope_abs():
    with make_scope() as session:
        n_in, n_out = 2, 3
        config = Config({"extern_data": {"data": {"dim": n_in}}})
        network = TFNetwork(config=config)
        net_dict = {
            "output": {
                "class": "subnetwork",
                "subnetwork": {
                    "layer1": {"class": "linear", "n_out": n_out, "from": "base:data:data"},
                    "layer2": {
                        "class": "linear",
                        "n_out": n_out,
                        "from": "base:data:data",
                        "name_scope": "/custom_name",
                    },
                    "output": {"class": "combine", "kind": "add", "from": ["layer1", "layer2"]},
                },
            }
        }
        network.construct_from_dict(net_dict)
        layer1 = network.get_layer("output/layer1")
        layer2 = network.get_layer("output/layer2")
        param1 = layer1.params["W"]
        param2 = layer2.params["W"]
        assert param1.name == "output/layer1/W:0" and param2.name == "custom_name/W:0"


def test_name_scope_sub_empty():
    with make_scope() as session:
        n_in, n_out = 2, 3
        config = Config({"extern_data": {"data": {"dim": n_in}}})
        network = TFNetwork(config=config)
        net_dict = {
            "output": {
                "class": "subnetwork",
                "name_scope": "",
                "subnetwork": {
                    "layer1": {"class": "linear", "n_out": n_out, "from": "base:data:data"},
                    "layer2": {"class": "linear", "n_out": n_out, "from": "base:data:data", "name_scope": "layer1"},
                    "output": {"class": "combine", "kind": "add", "from": ["layer1", "layer2"]},
                },
            }
        }
        network.construct_from_dict(net_dict)
        layer1 = network.get_layer("output/layer1")
        layer2 = network.get_layer("output/layer2")
        param1 = layer1.params["W"]
        param2 = layer2.params["W"]
        assert param1.name == "layer1/W:0" and param2.name == "layer1/W:0"


def test_name_scope_rec_sub_empty():
    with make_scope() as session:
        n_in, n_out = 2, 3
        config = Config({"extern_data": {"data": {"dim": n_in}}})
        network = TFNetwork(config=config)
        net_dict = {
            "output": {
                "class": "rec",
                "name_scope": "",
                "from": "data:data",
                "optimize_move_layers_out": False,
                "unit": {
                    "layer1": {"class": "linear", "n_out": n_out, "from": "data:source", "is_output_layer": True},
                    "layer2": {
                        "class": "linear",
                        "n_out": n_out,
                        "from": "data:source",
                        "is_output_layer": True,
                        "name_scope": "layer1",
                    },
                    "output": {"class": "combine", "kind": "add", "from": ["layer1", "layer2"]},
                },
            }
        }
        network.construct_from_dict(net_dict)
        layer1 = network.get_layer("output/layer1")
        layer2 = network.get_layer("output/layer2")
        param1 = layer1.params["W"]
        param2 = layer2.params["W"]
        assert param1.name == "layer1/W:0" and param2.name == "layer1/W:0"


def test_name_scope_share_params():
    with make_scope() as session:
        n_in, n_out = 3, 3
        config = Config({"extern_data": {"data": {"dim": n_in}}})
        network = TFNetwork(config=config)
        net_dict = {
            "layer1": {"class": "linear", "n_out": n_out, "from": "data:data"},
            "output": {"class": "linear", "n_out": n_out, "from": "layer1", "name_scope": "layer1"},
        }
        network.construct_from_dict(net_dict)
        l1 = network.layers["layer1"]
        l2 = network.layers["output"]
        assert set(l1.params.keys()) == {"W", "b"}
        assert set(l2.params.keys()) == {"W", "b"}
        assert l1.params["W"] is l2.params["W"]
        assert l1.params["b"] is l2.params["b"]
        assert set(network.get_trainable_params()) == {l1.params["W"], l1.params["b"]}


def test_SliceLayer_output_placeholder():
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData())
        src = InternalLayer(name="src", network=net, output=Data(**{"name": "src", "dim": 20, "sparse": True}))
        src.output.placeholder = tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], dtype=tf.int32)
        src.output.size_placeholder = {0: tf.constant([5, 3, 2], dtype=tf.int32)}
        layer = SliceLayer(
            name="slice",
            network=net,
            axis="T",
            slice_step=2,
            slice_start=1,
            sources=[src],
            output=SliceLayer.get_out_data_from_opts(
                name="slice", network=net, axis="T", slice_step=2, slice_start=1, sources=[src]
            ),
        )
        out, seq_lens = session.run([layer.output.placeholder, layer.output.size_placeholder[0]])
        print(out)
        print(seq_lens)
        assert isinstance(out, numpy.ndarray)
        assert isinstance(seq_lens, numpy.ndarray)
        assert out.tolist() == [[2, 4], [7, 9], [12, 14]]
        assert seq_lens.tolist() == [2, 1, 1]


def test_SliceLayer_NCHW():
    with make_scope() as session:
        import numpy as np

        net = TFNetwork(extern_data=ExternData())
        with tf_compat.v1.variable_scope("src_nchw"):
            src_nchw = InternalLayer(
                name="src_nchw",
                network=net,
                output=Data(
                    **{
                        "name": "src_nchw_output",
                        "dim": 16,
                        "shape": (16, None, 16),
                        "batch_dim_axis": 0,
                        "time_dim_axis": 2,
                        "feature_dim_axis": 1,
                        "sparse": False,
                    }
                ),
            )
            src_nchw.output.placeholder = tf_compat.v1.placeholder(shape=(None, 16, None, 16), dtype=tf.float32)
            src_nchw.output.size_placeholder = {1: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}
        with tf_compat.v1.variable_scope("src_nchw_feature_unspecified"):
            src_nchw_no_f = InternalLayer(
                name="src_nchw_feature_unspecified",
                network=net,
                output=Data(
                    **{
                        "name": "src_nchw_feature_unspecified_output",
                        "dim": 16,
                        "shape": (16, None, 16),
                        "batch_dim_axis": 0,
                        "time_dim_axis": 2,
                        "feature_dim_axis": NotSpecified,
                        "sparse": False,
                    }
                ),
            )
            src_nchw_no_f.output.placeholder = tf_compat.v1.placeholder(shape=(None, 16, None, 16), dtype=tf.float32)
            src_nchw_no_f.output.size_placeholder = {1: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}
        with tf_compat.v1.variable_scope("slice1"):
            slice1 = SliceLayer(
                name="slice1",
                network=net,
                axis="f",
                slice_step=2,
                sources=[src_nchw],
                output=SliceLayer.get_out_data_from_opts(name="slice1", axis="f", slice_step=2, sources=[src_nchw]),
            )
        with tf_compat.v1.variable_scope("slice2"):
            slice2 = SliceLayer(
                name="slice2",
                network=net,
                axis="f",
                slice_step=2,
                sources=[src_nchw_no_f],
                output=SliceLayer.get_out_data_from_opts(
                    name="slice2", axis="f", slice_step=2, sources=[src_nchw_no_f]
                ),
            )
        out1, out2 = session.run(
            [slice1.output.placeholder, slice2.output.placeholder],
            feed_dict={
                src_nchw.output.placeholder: np.random.rand(10, 16, 11, 16),
                src_nchw.output.size_placeholder[1]: np.full(shape=(10,), fill_value=11),
                src_nchw_no_f.output.placeholder: np.random.rand(10, 16, 11, 16),
                src_nchw_no_f.output.size_placeholder[1]: np.full(shape=(10,), fill_value=11),
            },
        )
        assert out1.shape == (10, 8, 11, 16)
        assert slice1.output.dim == 8 and slice1.output.feature_dim_axis == 1
        assert out2.shape == (10, 16, 11, 8)
        assert slice2.output.dim == 8 and slice2.output.feature_dim_axis == 3


def test_pad_conv_slice():
    # https://github.com/rwth-i6/returnn/issues/1017
    with make_scope() as session:
        net = TFNetwork(config=Config({"extern_data": {"data": {"dim": 5}}}))
        net.construct_from_dict(
            {
                "padding": {
                    "class": "pad",
                    "mode": "constant",
                    "value": 0,
                    "axes": ["T"],
                    "padding": [(1, 1)],
                    "from": "data",
                },
                "conv": {
                    "class": "conv",
                    "n_out": 5,
                    "filter_size": (2,),
                    "padding": "valid",
                    "in_spatial_dims": ["T"],
                    "from": "padding",
                },
                "output": {
                    "class": "slice",
                    "axis": "T",
                    "slice_start": None,
                    "slice_end": -1,
                    "slice_step": None,
                    "from": "conv",
                },
            }
        )
        out = net.get_default_output_layer().output
        in_ = net.extern_data.get_default_input_data()
        assert in_.get_time_dim_tag() != out.get_time_dim_tag()
        net.initialize_params(session)
        session.run((out.placeholder, out.get_sequence_lengths()), feed_dict=make_feed_dict(net.extern_data))


def test_GatherLayer():
    with make_scope() as session:
        import numpy as np

        net = TFNetwork(extern_data=ExternData())
        batch_dim, gather_dim, time_dim, feature_dim1, feature_dim2 = 3, 4, 2, 1, 2
        # [B, D, T, F1]
        values = InternalLayer(
            name="values",
            network=net,
            output=Data(
                **{
                    "name": "values",
                    "batch_dim_axis": 0,
                    "time_dim_axis": 2,
                    "feature_dim_axis": 3,
                    "shape": [gather_dim, None, feature_dim1],
                    "sparse": False,
                }
            ),
        )
        # [B, T, F2]
        position = InternalLayer(
            name="position",
            network=net,
            output=Data(
                **{
                    "name": "position",
                    "batch_dim_axis": 0,
                    "time_dim_axis": 1,
                    "shape": [None, feature_dim2],
                    "sparse": True,
                    "dim": gather_dim,
                }
            ),
        )

        random = np.random.RandomState(42)
        values_seqs = random.rand(batch_dim, gather_dim, time_dim, feature_dim1).astype("float32")
        seq_lens = random.randint(1, time_dim, size=[batch_dim])
        seq_lens_tensor = tf.constant(seq_lens, dtype=tf.int32)
        values.output.placeholder = tf.constant(values_seqs, dtype=tf.float32)
        values.output.size_placeholder = {1: seq_lens_tensor}
        position_seqs = random.randint(low=0, high=gather_dim, size=[batch_dim, time_dim, feature_dim2])
        position.output.placeholder = tf.constant(position_seqs, dtype=tf.int32)
        position.output.size_placeholder = {0: seq_lens_tensor}
        position.output.sanity_check()
        values.output.sanity_check()

        # should become [B, T, F2, F1]
        layer = GatherLayer(
            name="gather",
            network=net,
            sources=[values],
            position=position,
            axis="dim:%i" % gather_dim,
            output=GatherLayer.get_out_data_from_opts(
                name="gather", sources=[values], position=position, axis="dim:%i" % gather_dim
            ),
        )
        layer.output.sanity_check()
        out_seqs, size = session.run([layer.output.placeholder, layer.output.size_placeholder.as_dict()])
        assert isinstance(out_seqs, numpy.ndarray)

        # test shapes
        print("shapes: values", values.output, "position", position.output, "output", layer.output)
        assert layer.output.batch_dim_axis == 0 and layer.output.time_dim_axis == 1
        assert layer.output.batch_shape == (None, None, feature_dim2, feature_dim1)
        assert np.shape(out_seqs) == (batch_dim, time_dim, feature_dim2, feature_dim1)
        assert layer.output.dtype == values.output.dtype
        assert np.array_equal(size[0], seq_lens)

        print("values [B, D, T, F1]:", values_seqs)
        print("position [B, T, F2]:", position_seqs)
        print("produced output [B, T, F2, F1]:", out_seqs)

        # test values
        for b in range(batch_dim):
            for t in range(seq_lens[b]):
                for f2 in range(feature_dim2):
                    for f1 in range(feature_dim1):
                        np.testing.assert_almost_equal(
                            out_seqs[b, t, f2, f1], values_seqs[b, position_seqs[b, t, f2], t, f1]
                        )


def test_GatherLayer_constant_position():
    with make_scope() as session:
        import numpy as np

        net = TFNetwork(extern_data=ExternData())
        batch_dim, gather_dim, feature_dim1, feature_dim2 = 3, 4, 1, 2
        # [B, F1, D, F2]
        values = InternalLayer(
            name="values",
            network=net,
            output=Data(
                **{
                    "name": "values",
                    "batch_dim_axis": 0,
                    "feature_dim_axis": 3,
                    "time_dim_axis": None,
                    "shape": [feature_dim1, gather_dim, feature_dim2],
                }
            ),
        )
        position = 3

        random = np.random.RandomState(42)
        values_seqs = random.rand(batch_dim, feature_dim1, gather_dim, feature_dim2).astype("float32")
        values.output.placeholder = tf.constant(values_seqs, dtype=tf.float32)
        values.output.sanity_check()

        # should become [B, F1, F2]
        layer = GatherLayer(
            name="gather",
            network=net,
            sources=[values],
            position=position,
            axis="dim:%i" % gather_dim,
            output=GatherLayer.get_out_data_from_opts(
                name="gather", sources=[values], position=position, axis="dim:%i" % gather_dim
            ),
        )
        layer.output.sanity_check()
        out_seqs = session.run(layer.output.placeholder)
        assert isinstance(out_seqs, numpy.ndarray)

        # test shapes
        print("shapes: values", values.output, "position", position, "output", layer.output)
        assert layer.output.batch_dim_axis == 0 and layer.output.feature_dim_axis == 2
        assert layer.output.batch_shape == (None, feature_dim1, feature_dim2)
        assert np.shape(out_seqs) == (batch_dim, feature_dim1, feature_dim2)
        assert layer.output.dtype == values.output.dtype

        print("values [B, F1, D, F2]:", values_seqs)
        print("position:", position)
        print("produced output [B, F1, F2]:", out_seqs)

        # test values
        for b in range(batch_dim):
            for f1 in range(feature_dim1):
                for f2 in range(feature_dim2):
                    np.testing.assert_almost_equal(out_seqs[b, f1, f2], values_seqs[b, f1, position, f2])


def test_GatherLayer_search_beam():
    from returnn.tf.network import TFNetwork
    from returnn.config import Config

    with make_scope() as session:
        n_out = 5
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {"data": {"dim": n_out}, "classes": {"dim": n_out, "sparse": True}},
            }
        )
        net = TFNetwork(config=config, search_flag=True)
        net.construct_from_dict(
            {
                "output": {
                    "class": "rec",
                    "from": "data:data",
                    "unit": {
                        "position": {"class": "reinterpret_data", "from": "prev:output", "set_sparse": False},
                        "gather": {
                            "class": "gather",
                            "from": "base:data:data",
                            "position": "position",
                            "axis": "t",
                        },  # [B,T,slice,D]
                        "prob": {"class": "softmax", "from": "gather", "target": "classes", "loss": "ce"},
                        "output": {
                            "class": "choice",
                            "target": "classes",
                            "beam_size": 3,
                            "from": "prob",
                            "input_type": "prob",
                            "initial_output": 0,
                        },
                    },
                }
            }
        )


def test_GatherLayer_broadcast_dim():
    from returnn.tf.util.data import batch_dim

    head_dim = SpatialDim("head", 1)  # previously, this dim would match all others and therefore fail.
    round_dim = SpatialDim("round", 2)
    chunk_dim = SpatialDim("chunk")
    time_dim = SpatialDim("time")
    config = Config(
        {
            "extern_data": {
                "source": {"dim_tags": [batch_dim, head_dim, time_dim]},
                "position": {"dim_tags": [batch_dim, head_dim, round_dim, chunk_dim], "dtype": "int32"},
            },
            "debug_print_layer_output_template": True,
        }
    )
    net = TFNetwork(config=config)
    net.construct_from_dict(
        {
            "output": {
                "class": "gather",
                "from": "data:source",
                "position": "data:position",
                "axis": time_dim,
                "out_shape": {batch_dim, head_dim, round_dim, chunk_dim},
            }
        }
    )


def test_GatherLayer_different_static_dims():
    # https://github.com/rwth-i6/returnn/issues/1219
    from returnn.tf.util.data import batch_dim

    head_dim = SpatialDim("head", 2)
    round_dim = SpatialDim("round", 2)
    chunk_dim = SpatialDim("chunk")
    time_dim = SpatialDim("time")
    config = Config(
        {
            "extern_data": {
                "source": {"dim_tags": [batch_dim, head_dim, time_dim]},
                "position": {"dim_tags": [batch_dim, round_dim, chunk_dim], "dtype": "int32"},
            },
            "debug_print_layer_output_template": True,
        }
    )
    net = TFNetwork(config=config)
    net.construct_from_dict(
        {
            "output": {
                "class": "gather",
                "from": "data:source",
                "position": "data:position",
                "axis": time_dim,
                "out_shape": {batch_dim, head_dim, round_dim, chunk_dim},
            }
        }
    )


def test_SliceNdLayer():
    n_batch = 5
    n_time = 7
    n_dim = 11
    rnd = numpy.random.RandomState(42)
    seqs = rnd.randint(1, 100, (n_batch, n_time, n_dim)).astype("float32")  # all != 0
    seq_lens = numpy.array([n_time, n_time - 2, n_time - 3, n_time - 1, n_time - 4], dtype="int32")
    starts = numpy.array([2, 1, 3, n_time + 1, -1], dtype="int32")
    size = 5
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData())
        batch = BatchInfo.make_global_batch_info(tf.constant(n_batch))
        net.extern_data.set_batch_info(batch)
        src = InternalLayer(name="src", network=net, output=Data(name="src", dim=n_dim))
        src.output.placeholder = tf.constant(seqs)
        src.output.size_placeholder = {0: tf.constant(seq_lens)}
        src.output.sanity_check()
        start = InternalLayer(
            name="start",
            network=net,
            output=Data(**{"name": "start", "dim": None, "sparse": True, "time_dim_axis": None}),
        )
        start.output.placeholder = tf.constant(starts)
        start.output.sanity_check()
        kwargs = dict(name="slice", network=net, sources=[src], start=start, size=size)
        kwargs["output"] = SliceNdLayer.get_out_data_from_opts(**kwargs)
        layer = SliceNdLayer(**kwargs)
        print(layer)
        assert not layer.output.size_placeholder
        assert layer.output.batch_shape == (None, size, n_dim)
        out = session.run(layer.output.placeholder)
        print(out)
        assert isinstance(out, numpy.ndarray)
        assert out.shape == (n_batch, size, n_dim)
        for b in range(n_batch):
            s = starts[b]
            if s < 0:
                assert s + size > 0
                orig_seq = numpy.pad(seqs[b, : s + size], [(-s, 0), (0, 0)], "constant")
            else:
                orig_seq = seqs[b, s : s + size]
            if len(orig_seq) < size:
                orig_seq = numpy.pad(orig_seq, [(0, size - len(orig_seq)), (0, 0)], "constant")
            assert orig_seq.shape == (size, n_dim)
            orig_seq = numpy.where((numpy.arange(s, s + size) >= seq_lens[b])[:, None], 0.0, orig_seq)
            for t in range(size):
                numpy.testing.assert_equal(orig_seq[t], out[b, t])


def test_SliceNdLayer_dyn_size():
    n_batch = 4
    n_time = 7
    n_dim = 11
    rnd = numpy.random.RandomState(42)
    seqs = rnd.randint(1, 100, (n_batch, n_time, n_dim)).astype("float32")  # all != 0
    seq_lens = numpy.array([n_time, n_time - 2, n_time - 3, n_time - 1], dtype="int32")
    starts = numpy.array([2, 1, 3, n_time + 1], dtype="int32")
    size = None
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData())
        src = InternalLayer(name="src", network=net, output=Data(name="src", dim=n_dim))
        src.output.placeholder = tf.constant(seqs)
        src.output.size_placeholder = {0: tf.constant(seq_lens)}
        src.output.sanity_check()
        start = InternalLayer(
            name="start",
            network=net,
            output=Data(**{"name": "start", "dim": None, "sparse": True, "time_dim_axis": None}),
        )
        start.output.placeholder = tf.constant(starts)
        start.output.sanity_check()
        kwargs = dict(name="slice", network=net, sources=[src], start=start, size=size)
        kwargs["output"] = SliceNdLayer.get_out_data_from_opts(**kwargs)
        layer = SliceNdLayer(**kwargs)
        print(layer)
        assert 0 in layer.output.size_placeholder
        assert layer.output.batch_shape == (None, size, n_dim)
        out = session.run(layer.output.placeholder)
        print(out)
        assert isinstance(out, numpy.ndarray)
        max_size = max(list(seq_lens - starts) + [0])
        assert out.shape == (n_batch, max_size, n_dim)
        for b in range(n_batch):
            s = starts[b]
            orig_seq = seqs[b, s:]
            if len(orig_seq) < max_size:
                orig_seq = numpy.pad(orig_seq, [(0, max_size - len(orig_seq)), (0, 0)], "constant")
            elif len(orig_seq) > max_size:
                orig_seq = orig_seq[:max_size]
            assert orig_seq.shape == (max_size, n_dim)
            orig_seq = numpy.where((numpy.arange(s, s + max_size) >= seq_lens[b])[:, None], 0.0, orig_seq)
            for t in range(max_size):
                numpy.testing.assert_equal(orig_seq[t], out[b, t])


def test_SliceNdLayer_multidimensional_start():
    with make_scope() as session:
        from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

        out_dim = FeatureDim("feat", 3)
        time_dim = SpatialDim("time")
        n_batch = 3
        max_seq_len = 10
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "data": {"dim_tags": [batch_dim, time_dim, out_dim]},
                    "classes": {"dim_tags": [batch_dim, time_dim], "sparse_dim": out_dim},
                },
            }
        )
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "output": {
                    "class": "rec",
                    "from": "data:data",
                    "unit": {
                        "start": {"class": "copy", "from": "prev:choice"},
                        "slices": {"class": "slice_nd", "from": "base:data:data", "start": "start", "size": None},
                        "output": {"class": "reduce", "from": "slices", "mode": "max", "axes": "stag:slice"},
                        "prob": {"class": "softmax", "from": "data:source", "target": "classes", "loss": "ce"},
                        "choice": {
                            "class": "choice",
                            "target": "classes",
                            "beam_size": 3,
                            "from": "prob",
                            "input_type": "prob",
                            "initial_output": 0,
                        },
                    },
                }
            }
        )
        session.run(tf_compat.v1.global_variables_initializer())
        output_layer = net.layers["output"]
        starts = output_layer.cell.output_layers_net.layers["start"].output.get_placeholder_as_batch_major()
        segments = output_layer.cell.output_layers_net.layers["slices"].output.get_placeholder_as_batch_major()
        feed = make_feed_dict(net.extern_data.data.values(), n_batch=n_batch, n_time=max_seq_len, same_time=True)
        starts = session.run(starts, feed_dict=feed)
        segments = session.run(segments, feed_dict=feed)
        seq_lens = feed[net.extern_data.data["data"].size_placeholder[0]]
        input_data = feed[net.extern_data.data["data"].placeholder]
        max_size = numpy.amax(seq_lens[:, None] - starts)
        max_size = max(max_size, 0)
        assert segments.shape == (n_batch, max_seq_len, max_size, out_dim.dimension)
        for b in range(n_batch):
            for t in range(max_seq_len):
                s = starts[b, t]
                orig_seq = input_data[b, s:]
                if len(orig_seq) < max_size:
                    orig_seq = numpy.pad(orig_seq, [(0, max_size - len(orig_seq)), (0, 0)], "constant")
                elif len(orig_seq) > max_size:
                    orig_seq = orig_seq[:max_size]
                assert orig_seq.shape == (max_size, out_dim.dimension)
                orig_seq = numpy.where((numpy.arange(s, s + max_size) >= seq_lens[b])[:, None], 0.0, orig_seq)
                for t2 in range(max_size):
                    numpy.testing.assert_equal(orig_seq[t2], segments[b, t, t2])


def test_SliceNdLayer_multidimensional_size():
    with make_scope() as session:
        from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

        out_dim = FeatureDim("feat", 3)
        time_dim = SpatialDim("time")
        n_batch = 3
        max_seq_len = 10
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "data": {"dim_tags": [batch_dim, time_dim, out_dim]},
                    "classes": {"dim_tags": [batch_dim, time_dim], "sparse_dim": out_dim},
                },
            }
        )
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "output": {
                    "class": "rec",
                    "from": "data:data",
                    "unit": {
                        "const1": {"class": "constant", "value": 1},
                        "start": {"class": "reinterpret_data", "from": "prev:choice", "set_sparse": False},
                        "size": {"class": "combine", "from": ["const1", "start"], "kind": "add"},
                        "slices": {"class": "slice_nd", "from": "base:data:data", "start": "start", "size": "size"},
                        "output": {"class": "reduce", "from": "slices", "mode": "max", "axes": "stag:slice"},
                        "prob": {"class": "softmax", "from": "data:source", "target": "classes", "loss": "ce"},
                        "choice": {
                            "class": "choice",
                            "target": "classes",
                            "beam_size": 3,
                            "from": "prob",
                            "input_type": "prob",
                            "initial_output": 0,
                        },
                    },
                }
            }
        )
        session.run(tf_compat.v1.global_variables_initializer())
        output_layer = net.layers["output"]
        starts = output_layer.cell.output_layers_net.layers["start"].output.get_placeholder_as_batch_major()
        sizes = output_layer.cell.output_layers_net.layers["size"].output.get_placeholder_as_batch_major()
        segments = output_layer.cell.output_layers_net.layers["slices"].output.get_placeholder_as_batch_major()
        feed = make_feed_dict(net.extern_data.data.values(), n_batch=n_batch, n_time=max_seq_len, same_time=True)
        starts = session.run(starts, feed_dict=feed)
        sizes = session.run(sizes, feed_dict=feed)
        segments = session.run(segments, feed_dict=feed)
        seq_lens = feed[net.extern_data.data["data"].size_placeholder[0]]
        input_data = feed[net.extern_data.data["data"].placeholder]
        max_size = numpy.amax(sizes)
        max_size = max(max_size, 0)
        assert segments.shape == (n_batch, max_seq_len, max_size, out_dim.dimension)
        for b in range(n_batch):
            for t in range(max_seq_len):
                s = starts[b, t]
                size = sizes[b, t]
                end = min(s + size, seq_lens[b])
                orig_seq = input_data[b, s:end]
                if len(orig_seq) < max_size:
                    orig_seq = numpy.pad(orig_seq, [(0, max_size - len(orig_seq)), (0, 0)], "constant")
                elif len(orig_seq) > max_size:
                    orig_seq = orig_seq[:max_size]
                assert orig_seq.shape == (max_size, out_dim.dimension)
                orig_seq = numpy.where((numpy.arange(s, s + max_size) >= seq_lens[b])[:, None], 0.0, orig_seq)
                for t2 in range(max_size):
                    numpy.testing.assert_equal(orig_seq[t2], segments[b, t, t2])


def test_SliceNdLayer_set_tag_on_size_tensor():
    with make_scope():
        n_out = 5
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {"data": {"dim": n_out}, "classes": {"dim": n_out, "sparse": True}},
            }
        )
        net = TFNetwork(config=config, train_flag=True)
        # the construction of the "compare" layer will fail if set_tag_on_size_tensor is not called on the slice axis
        # inside of the SliceNdLayer
        net.construct_from_dict(
            {
                "start": {"class": "range_in_axis", "from": "data", "axis": "b"},
                "slices": {"class": "slice_nd", "from": "data", "start": "start", "size": None},
                "output": {"class": "compare", "from": ["slices", "slices"], "kind": "equal"},
            }
        )


def test_SliceNdLayer_start0():
    with make_scope() as session:
        from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

        time_dim = SpatialDim("time")
        feature_dim = FeatureDim("feature", 5)
        config = Config({"extern_data": {"data": {"dim_tags": (batch_dim, time_dim, feature_dim)}}})
        net = TFNetwork(config=config, train_flag=True)
        # the construction of the "compare" layer will fail if set_tag_on_size_tensor is not called on the slice axis
        # inside of the SliceNdLayer
        net.construct_from_dict(
            {
                "downsample": {"class": "pool", "mode": "avg", "pool_size": 2, "from": "data"},
                "upsample": {"class": "resize", "axis": "T", "factor": 2, "from": "downsample"},
                "cutoff": {"class": "slice_nd", "from": "data", "size": time_dim},
                "output": {"class": "combine", "from": ["cutoff", "data"], "kind": "sub"},
            }
        )
        session.run(net.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_SliceNdLayer_ReinterpretDataLayer():
    """
    https://github.com/rwth-i6/returnn/issues/851
    """
    from returnn.tf.util.data import DimensionTag

    new_slice_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description="new-slice", dimension=None)
    with make_scope():
        n_out = 5
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {"data": {"dim": n_out}, "classes": {"dim": n_out, "sparse": True}},
            }
        )
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "start": {"class": "reinterpret_data", "from": "data:classes", "set_sparse": False},
                "slices": {"class": "slice_nd", "from": "data", "start": "start", "size": None},
                "output": {
                    "class": "reinterpret_data",
                    "from": "slices",
                    "set_dim_tags": {"stag:sliced-time:slices": new_slice_tag},
                },
            }
        )


def test_WindowLayer_output_placeholder():
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData())
        src = InternalLayer(name="src", network=net, output=Data(name="src", dim=20, sparse=True))  # [B,T]
        src.output.placeholder = tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], dtype=tf.int32)
        src.output.size_placeholder = {0: tf.constant([5, 3, 1], dtype=tf.int32)}
        layer = WindowLayer(
            name="window",
            network=net,
            axis="T",
            window_size=3,
            padding="valid",
            sources=[src],
            output=WindowLayer.get_out_data_from_opts(
                name="window", network=net, axis="T", window_size=3, padding="valid", sources=[src]
            ),
        )
        print("layer:", layer)
        out, seq_lens = session.run([layer.output.placeholder, layer.output.get_sequence_lengths()])
        print(out)
        print(seq_lens)
        assert isinstance(out, numpy.ndarray)
        assert isinstance(seq_lens, numpy.ndarray)
        out = out.transpose([2, 1, 0])  # [W, T', B] -> [B, T', W]
        assert out.tolist() == [
            [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
            [[6, 7, 8], [7, 8, 9], [8, 9, 10]],
            [[11, 12, 13], [12, 13, 14], [13, 14, 15]],
        ]
        assert seq_lens.tolist() == [3, 1, 0]


def test_FoldLayer_unchunk():
    from returnn.tensor import Tensor, Dim, batch_dim

    in_spatial_dim = Dim(Tensor("in_spatial", [batch_dim], dtype="int32"))
    out_spatial_dim = Dim(None, name="out_spatial")
    window_dim = Dim(3, name="win")

    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData())
        net.extern_data.set_batch_info(BatchInfo.make_global_batch_info(tf.constant(3)))
        src = InternalLayer(
            name="src",
            network=net,
            output=Data(name="src", dims=[batch_dim, in_spatial_dim, window_dim], dtype="float32"),
        )
        src.output.dims[1].dyn_size_ext.raw_tensor = tf.constant([3, 2, 1])
        src.output.raw_tensor = tf.constant(
            [
                [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
                [[6, 7, 8], [7, 8, 9], [8, 9, 10]],
                [[11, 12, 13], [12, 13, 14], [13, 14, 15]],
            ],
            dtype=tf.float32,
        )
        opts = dict(
            name="fold",
            network=net,
            sources=[src],
            in_spatial_dim=in_spatial_dim,
            window_dim=window_dim,
            out_spatial_dim=out_spatial_dim,
            padding="valid",
            mode="mean",
        )
        layer = FoldLayer(**opts, output=FoldLayer.get_out_data_from_opts(**opts))
        print(layer)
        assert layer.output.dims == (out_spatial_dim, batch_dim)
        output = layer.output.copy_transpose((batch_dim, out_spatial_dim))
        out, seq_lens = session.run([output.raw_tensor, output.dims[1].dyn_size_ext.raw_tensor])
        print(out)
        print(seq_lens)
        assert out.tolist() == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], [11, 12, 13, 0, 0]]
        assert seq_lens.tolist() == [5, 4, 3]


def test_conv_window_merge_dims():
    n_in = 1
    n_out = 13
    net_dict = {
        "conv_1": {
            "activation": "abs",
            "class": "conv",
            "from": "data:data",
            "filter_size": (4,),
            "n_out": 64,
            "padding": "valid",
            "strides": 10,
        },
        "pad_conv_1_time_dim": {"axes": "time", "class": "pad", "from": ["conv_1"], "padding": 20},
        "conv_2": {
            "activation": "abs",
            "class": "conv",
            "filter_size": (2, 6),
            "from": ["pad_conv_1_time_dim"],
            "input_add_feature_dim": True,
            "n_out": 12,
            "padding": "valid",
            "strides": 16,
        },
        "flatten_conv": {"axes": ["dim:4", "dim:12"], "class": "merge_dims", "from": ["conv_2"], "n_out": 48},
        "window_1": {"class": "window", "from": ["flatten_conv"], "window_size": 17},
        "flatten_window": {"axes": ["dim:17", "dim:48"], "class": "merge_dims", "from": ["window_1"]},
        "output": {"activation": None, "class": "linear", "from": ["flatten_window"], "n_out": n_out},
    }
    config = Config({"num_outputs": n_out, "num_inputs": n_in, "debug_print_layer_output_template": True})
    with make_scope() as session:
        net = TFNetwork(config=config)
        print("extern data:")
        print(net.extern_data)
        # The construction itself is also the test.
        net.construct_from_dict(net_dict)
        out = net.get_default_output_layer()
        # Maybe this will not be the case in the future anymore;
        # however, if this test runs on CPU, currently the feature_dim_axis should always stay the default.
        # See also test_ConvLayer_feature_dim_unspecified.
        assert out.output.feature_dim_axis_or_unspecified in (NotSpecified, 1)


def test_ConvLayer_feature_dim_unspecified():
    n_in = 1
    n_out = 13
    net_dict = {
        "output": {
            "activation": "abs",
            "class": "conv",
            "from": "data:data",
            "filter_size": (4,),
            "n_out": 64,
            "padding": "valid",
            "strides": 10,
        }
    }
    config = Config({"num_outputs": n_out, "num_inputs": n_in, "debug_print_layer_output_template": True})
    with make_scope() as session:
        net = TFNetwork(config=config)
        print("extern data:")
        print(net.extern_data)
        net.construct_from_dict(net_dict)
        out = net.get_default_output_layer()
        # Maybe this will not be the case in the future anymore;
        # however, if this test runs on CPU, currently the feature_dim_axis should always stay the default.
        assert out.output.feature_dim_axis_or_unspecified is NotSpecified


def test_StftLayer():
    from returnn.tf.layers.signal_processing import StftLayer

    config = Config({"extern_data": {"data": {"dim": 1}}})
    with make_scope() as session:
        net = TFNetwork(config=config)
        in_layer = SourceLayer(
            name="input", network=net, data_key="data", output=net.extern_data.get_default_input_data()
        )
        layer_desc = {
            "name": "stft",
            "network": net,
            "sources": [in_layer],
            "frame_size": 32,
            "frame_shift": 8,
        }
        stft_out = StftLayer.get_out_data_from_opts(**layer_desc)
        print("stft out:", stft_out)
        out_time = stft_out.get_time_dim_tag()
        assert in_layer.output.dim_tags[1].is_spatial_dim()
        assert out_time != in_layer.output.dim_tags[1]
        layer_desc["output"] = stft_out
        with tf_compat.v1.variable_scope("stft"):
            stft_layer = StftLayer(**layer_desc)
        net.layers["stft"] = stft_layer
        net.initialize_params(session)
        session.run(
            (stft_layer.output.placeholder, stft_layer.output.get_sequence_lengths()),
            feed_dict=make_feed_dict(net.extern_data, n_time=1024),
        )


def test_IstftLayer():
    with make_scope() as session:
        frame_size = 32
        frame_shift = 8
        config = Config()
        config.update(
            {
                "extern_data": {"data": {"dim": 1}},
                "network": {
                    "stft": {"class": "stft", "frame_size": frame_size, "frame_shift": frame_shift, "from": "data"},
                    "output": {"class": "istft", "frame_size": frame_size, "frame_shift": frame_shift, "from": "stft"},
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_value("network"))
        out = network.get_default_output_layer().output.placeholder
        feed_dict = make_feed_dict(network.extern_data, n_time=1024)
        v_in = feed_dict[network.extern_data.data["data"].placeholder]
        v_out = session.run(out, feed_dict=feed_dict)
        # ignore border effects, see https://github.com/tensorflow/tensorflow/issues/16465
        numpy.testing.assert_allclose(
            v_in[:, frame_size:-frame_size, :], v_out[:, frame_size:-frame_size, :], rtol=1e-3
        )


def test_ConvLayer_time_dim_out():
    config = Config({"extern_data": {"data": {"dim": 7}}})
    with make_scope() as session:
        net = TFNetwork(config=config)
        in_layer = SourceLayer(
            name="input", network=net, data_key="data", output=net.extern_data.get_default_input_data()
        )
        layer_desc = {
            "name": "conv",
            "network": net,
            "sources": [in_layer],
            "filter_size": (4,),
            "strides": 10,
            "padding": "valid",
            "n_out": 64,
            "activation": "abs",
        }
        conv_out = ConvLayer.get_out_data_from_opts(**layer_desc)
        print("conv out:", conv_out)
        out_time = conv_out.get_time_dim_tag()
        assert in_layer.output.dim_tags[1].is_spatial_dim()
        assert out_time != in_layer.output.dim_tags[1]
        layer_desc["output"] = conv_out
        with tf_compat.v1.variable_scope("conv"):
            conv_layer = ConvLayer(**layer_desc)
        net.layers["conv"] = conv_layer
        net.initialize_params(session)
        session.run(
            (conv_layer.output.placeholder, conv_layer.output.get_sequence_lengths()),
            feed_dict=make_feed_dict(net.extern_data),
        )


def test_ConvLayer_get_out_data_from_opts_out_spatial_dims():
    from returnn.tf.util.data import batch_dim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("input", 7)
    conv_dim = SpatialDim("conv")
    config = Config({"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}}})
    with make_scope() as session:
        net = TFNetwork(config=config)
        layer_desc = {
            "name": "conv",
            "_name": "conv",
            "network": net,
            "_network": net,
            "from": "data",
            "filter_size": [4],
            "strides": 3,
            "padding": "valid",
            "n_out": 13,
            "out_spatial_dims": [conv_dim],
        }
        ConvLayer.transform_config_dict(layer_desc, network=net, get_layer=net.get_layer)
        conv_out = ConvLayer.get_out_data_from_opts(**layer_desc)
        print("conv out:", conv_out)
        print("conv dim:", conv_dim, "time dim:", time_dim)
        assert conv_dim == time_dim.sub_left(1).sub_right(2).ceildiv_right(3)
        assert conv_dim.get_same_base().derived_from_op
        with tf_compat.v1.variable_scope("conv"):
            conv_layer = ConvLayer(output=conv_out, **layer_desc)
        net.layers["conv"] = conv_layer
        net.initialize_params(session)
        session.run(
            (conv_layer.output.placeholder, conv_layer.output.get_sequence_lengths()),
            feed_dict=make_feed_dict(net.extern_data),
        )


def test_ConvLayer_static_time_get_out_data_from_opts_out_spatial_dims():
    from returnn.tf.util.data import batch_dim

    time_dim = SpatialDim("time", 13)
    feat_dim = FeatureDim("input", 7)
    conv_dim = SpatialDim("conv")
    config = Config({"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}}})
    with make_scope() as session:
        net = TFNetwork(config=config)
        layer_desc = {
            "name": "conv",
            "_name": "conv",
            "network": net,
            "_network": net,
            "from": "data",
            "filter_size": [4],
            "strides": 3,
            "padding": "valid",
            "n_out": 11,
            "in_spatial_dims": [time_dim],
            "out_spatial_dims": [conv_dim],
        }
        ConvLayer.transform_config_dict(layer_desc, network=net, get_layer=net.get_layer)
        conv_out = ConvLayer.get_out_data_from_opts(**layer_desc)
        print("conv out:", conv_out)
        print("conv dim:", conv_dim, "time dim:", time_dim)
        assert conv_dim == time_dim.sub_left(1).sub_right(2).ceildiv_right(3)
        assert conv_dim.get_same_base().derived_from_op
        assert not conv_dim.is_dynamic()
        with tf_compat.v1.variable_scope("conv"):
            conv_layer = ConvLayer(output=conv_out, **layer_desc)
        net.layers["conv"] = conv_layer
        net.initialize_params(session)
        session.run(conv_layer.output.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_ConvLayer_2d_device_based_opt():
    from returnn.tf.util.data import batch_dim

    for dev in ["cpu", "gpu"]:
        time_dim = SpatialDim("time")
        feat_dim = FeatureDim("input", 50)
        extra_dim = FeatureDim("extra", 1)
        config = Config(
            {
                "extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim, extra_dim]}},
                "device": dev,
            }
        )
        with tf.Graph().as_default() as graph:
            # We don't use make_scope() or enter the session scope because this is consistent
            # to how it usually would be the case for RETURNN.
            net = TFNetwork(config=config)
            gpu_available = tf_util.is_gpu_available_in_session()
            print("GPU available:", gpu_available)
            assert gpu_available == (dev == "gpu")
            net_dict = {
                "output": {
                    "class": "conv",
                    "from": "data",
                    "filter_size": (3, 3),
                    "in_spatial_dims": ["T", "dim:50"],
                    "n_out": 32,
                    "padding": "same",
                    "activation": None,
                    "with_bias": True,
                },
            }
            net.construct_from_dict(net_dict)
            conv_layer = net.get_default_output_layer()
            print("conv layer:", conv_layer)
            assert conv_layer.output.batch_dim_axis == 0
            if gpu_available:
                assert conv_layer.output.dim_tags[2:] == (time_dim, feat_dim)
                assert conv_layer.output.feature_dim_axis == 1 and conv_layer.output.dim == 32
            else:
                assert conv_layer.output.dim_tags[1:3] == (time_dim, feat_dim)
                assert conv_layer.output.feature_dim_axis == 3 and conv_layer.output.dim == 32


def test_ConvLayer_unrelated_dim():
    from returnn.tf.util.data import batch_dim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("input", 7)
    other_dim = SpatialDim("other")
    config = Config({"extern_data": {"data": {"dim_tags": [batch_dim, time_dim, other_dim, feat_dim]}}})
    with make_scope() as session:
        net = TFNetwork(config=config)
        layer_desc = {
            "name": "conv",
            "_name": "conv",
            "network": net,
            "_network": net,
            "from": "data",
            "filter_size": [4],
            "in_spatial_dims": [time_dim],
            "strides": 3,
            "padding": "valid",
            "n_out": 13,
        }
        ConvLayer.transform_config_dict(layer_desc, network=net, get_layer=net.get_layer)
        conv_out = ConvLayer.get_out_data_from_opts(**layer_desc)
        print("conv out:", conv_out)
        dyn_axes = conv_out.get_dynamic_axes()
        assert len(dyn_axes) == 2, "conv out: %r" % conv_out
        assert conv_out.get_axis_from_description(other_dim) in dyn_axes
        dyn_axes.remove(conv_out.get_axis_from_description(other_dim))
        out_spatial_dim = conv_out.dim_tags[dyn_axes[0]]
        assert out_spatial_dim not in net.extern_data.get_default_input_data().dim_tags
        assert conv_out.time_dim_axis == dyn_axes[0]
        with tf_compat.v1.variable_scope("conv"):
            conv_layer = ConvLayer(output=conv_out, **layer_desc)
        net.layers["conv"] = conv_layer
        net.initialize_params(session)
        session.run(
            (conv_layer.output.placeholder, conv_layer.output.get_sequence_lengths()),
            feed_dict=make_feed_dict(net.extern_data),
        )


def test_conv_layer_NCHW():
    with make_scope() as session:
        import numpy as np

        net = TFNetwork(extern_data=ExternData())
        with tf_compat.v1.variable_scope("src_nhwc"):
            src_nhwc = InternalLayer(
                name="src_nhwc",
                network=net,
                output=Data(
                    **{
                        "name": "src_nhwc_output",
                        "placeholder": tf_compat.v1.placeholder(shape=(None, None, 16, 17), dtype=tf.float32),
                        "size_placeholder": {0: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)},
                        "dim": 17,
                        "shape": (None, 16, 17),
                        "batch_dim_axis": 0,
                        "time_dim_axis": 1,
                        "feature_dim_axis": 3,
                        "sparse": False,
                    }
                ),
            )
        with tf_compat.v1.variable_scope("src_nchw"):
            src_nchw = InternalLayer(
                name="src_nchw",
                network=net,
                output=Data(
                    **{
                        "name": "src_nchw_output",
                        "placeholder": tf_compat.v1.placeholder(shape=(None, 17, None, 16), dtype=tf.float32),
                        "size_placeholder": {1: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)},
                        "dim": 17,
                        "shape": (17, None, 16),
                        "batch_dim_axis": 0,
                        "time_dim_axis": 2,
                        "feature_dim_axis": 1,
                        "sparse": False,
                    }
                ),
            )

        filters = 64
        filter_size = (5, 5)
        strides = (1, 2)
        padding = "VALID"

        with tf_compat.v1.variable_scope("conv_nhwc_from_nhwc"):
            conv_nhwc_from_nhwc = ConvLayer(
                name="conv_nhwc_from_nhwc",
                network=net,
                n_out=filters,
                filter_size=filter_size,
                in_spatial_dims=["T", "dim:16"],
                padding=padding,
                strides=strides,
                auto_use_channel_first=False,
                sources=[src_nhwc],
                output=ConvLayer.get_out_data_from_opts(
                    name="conv_nhwc_from_nhwc",
                    n_out=filters,
                    in_spatial_dims=["T", "dim:16"],
                    filter_size=filter_size,
                    padding=padding,
                    strides=strides,
                    auto_use_channel_first=False,
                    network=net,
                    sources=[src_nhwc],
                ),
            )
        with tf_compat.v1.variable_scope("conv_nchw_from_nhwc"):
            conv_nchw_from_nhwc = ConvLayer(
                name="conv_nchw_from_nhwc",
                network=net,
                n_out=filters,
                filter_size=filter_size,
                in_spatial_dims=["T", "dim:16"],
                padding=padding,
                strides=strides,
                auto_use_channel_first=True,
                sources=[src_nhwc],
                output=ConvLayer.get_out_data_from_opts(
                    name="conv_nchw_from_nhwc",
                    n_out=filters,
                    in_spatial_dims=["T", "dim:16"],
                    filter_size=filter_size,
                    padding=padding,
                    strides=strides,
                    auto_use_channel_first=True,
                    network=net,
                    sources=[src_nhwc],
                ),
            )
        with tf_compat.v1.variable_scope("conv_nchw_from_nchw"):
            conv_nchw_from_nchw = ConvLayer(
                name="conv_nchw_from_nchw",
                network=net,
                n_out=filters,
                filter_size=filter_size,
                in_spatial_dims=["T", "dim:16"],
                padding=padding,
                strides=strides,
                auto_use_channel_first=True,
                sources=[src_nchw],
                output=ConvLayer.get_out_data_from_opts(
                    name="conv_nchw_from_nchw",
                    n_out=filters,
                    in_spatial_dims=["T", "dim:16"],
                    filter_size=filter_size,
                    padding=padding,
                    strides=strides,
                    auto_use_channel_first=True,
                    network=net,
                    sources=[src_nchw],
                ),
            )
        tf_compat.v1.global_variables_initializer().run()
        out, seq_lens = session.run(
            [conv_nhwc_from_nhwc.output.placeholder, conv_nhwc_from_nhwc.output.size_placeholder[0]],
            feed_dict={
                src_nhwc.output.placeholder: np.random.rand(10, 10, 16, 17),
                src_nhwc.output.size_placeholder[0]: np.full(shape=(10,), fill_value=10),
            },
        )
        print(out.shape)
        assert out.shape == (10, 6, 6, 64)
        print(seq_lens)
        time_dim_axis = 1 if tf_util.is_gpu_available() else 0
        out, seq_lens = session.run(
            [conv_nchw_from_nhwc.output.placeholder, conv_nchw_from_nhwc.output.size_placeholder[time_dim_axis]],
            feed_dict={
                src_nhwc.output.placeholder: np.random.rand(10, 10, 16, 17),
                src_nhwc.output.size_placeholder[0]: np.full(shape=(10,), fill_value=10),
            },
        )
        print(out.shape)
        if time_dim_axis == 1:
            assert out.shape == (10, 64, 6, 6)
        else:
            assert out.shape == (10, 6, 6, 64)
        print(seq_lens)
        if tf_util.is_gpu_available():
            out, seq_lens = session.run(
                [conv_nchw_from_nchw.output.placeholder, conv_nchw_from_nchw.output.size_placeholder[1]],
                feed_dict={
                    src_nchw.output.placeholder: np.random.rand(10, 17, 10, 16),
                    src_nchw.output.size_placeholder[1]: np.full(shape=(10,), fill_value=10),
                },
            )
            print(out.shape)
            assert out.shape == (10, 64, 6, 6)
            print(seq_lens)


def test_ConvLayer_empty_out():
    with make_scope() as session:
        net = TFNetwork(config=Config({"extern_data": {"data": {"dim": 5}}}))
        net.construct_from_dict(
            {
                # Use filter_size 2 and T=1 to get 0 size out.
                # Using filter_size 3 would result in negative size according to the formula.
                # Actually I would have expected that TF also deals with this but this is not the case.
                "output": {"class": "conv", "n_out": 7, "filter_size": [2], "padding": "valid", "from": "data"},
            }
        )
        out_ = net.layers["output"].output.copy_as_batch_spatial_major()
        print(out_)
        net.initialize_params(session)
        out, seq_lens = session.run(
            [out_.placeholder, out_.size_placeholder[0]], feed_dict=make_feed_dict(net.extern_data, n_time=1, n_batch=1)
        )
        print(out)
        print(seq_lens)
        assert isinstance(out, numpy.ndarray)
        assert isinstance(seq_lens, numpy.ndarray)
        assert seq_lens.tolist() == [0]
        assert out.shape == (1, 0, 7)


def test_ConvLayer_stride_dilation():
    with make_scope() as session:
        net = TFNetwork(config=Config({"extern_data": {"data": {"dim": 5}}}))
        net.construct_from_dict(
            {
                "output": {
                    "class": "conv",
                    "n_out": 7,
                    "with_bias": False,
                    "filter_size": (2,),
                    "strides": (2,),
                    "dilation_rate": (2,),
                    "padding": "valid",
                    "from": "data",
                    "forward_weights_init": 1.0,
                },
            }
        )
        input = numpy.random.randint(low=0, high=10, size=(3, 20, 5))
        out_ = net.layers["output"].output.copy_as_batch_spatial_major()
        print(out_)
        net.initialize_params(session)
        out, seq_lens = session.run(
            [out_.placeholder, out_.size_placeholder[0]],
            feed_dict={
                net.extern_data.data["data"].size_placeholder[0]: numpy.array([20, 18, 17]),
                net.extern_data.data["data"].placeholder: input,
            },
        )
        print(out.shape)
        print(seq_lens)
        assert isinstance(out, numpy.ndarray)
        assert isinstance(seq_lens, numpy.ndarray)
        assert out.shape == (3, 9, 7)
        assert (seq_lens == [9, 8, 8]).all()


def test_ConvLayer_custom_filter_no_dim_tags():
    # https://github.com/rwth-i6/returnn/issues/1340
    # By intention, do not use dim tags.
    config = Config({"extern_data": {"data": {"shape": (None, 1)}}})
    with make_scope() as session:
        net = TFNetwork(config=config)
        net_dict = {
            "conv_h_filter": {
                "class": "variable",
                "shape": (6, 1, 15),
                "init": "glorot_uniform",
            },
            "conv_h": {
                "class": "conv",
                "filter_size": (6,),
                "strides": 2,
                "n_out": 15,
                "padding": "valid",
                "filter": "conv_h_filter",
                "from": "data",
                "is_output_layer": True,
            },
        }
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        out = net.layers["conv_h"].output
        session.run(out.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_pool_layer_NCHW():
    with make_scope() as session:
        import numpy as np

        net = TFNetwork(extern_data=ExternData())
        with tf_compat.v1.variable_scope("src_nhwc"):
            src_nhwc = InternalLayer(
                name="src_nhwc",
                network=net,
                output=Data(
                    **{
                        "name": "src_nhwc_output",
                        "dim": 17,
                        "shape": (None, 16, 17),
                        "batch_dim_axis": 0,
                        "time_dim_axis": 1,
                        "feature_dim_axis": 3,
                        "sparse": False,
                    }
                ),
            )
            src_nhwc.output.placeholder = tf_compat.v1.placeholder(shape=(None, None, 16, 17), dtype=tf.float32)
            src_nhwc.output.size_placeholder = {0: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}
        with tf_compat.v1.variable_scope("src_nchw"):
            src_nchw = InternalLayer(
                name="src_nchw",
                network=net,
                output=Data(
                    **{
                        "name": "src_nchw_output",
                        "dim": 17,
                        "shape": (17, None, 16),
                        "batch_dim_axis": 0,
                        "time_dim_axis": 2,
                        "feature_dim_axis": 1,
                        "sparse": False,
                    }
                ),
            )
            src_nchw.output.placeholder = tf_compat.v1.placeholder(shape=(None, 17, None, 16), dtype=tf.float32)
            src_nchw.output.size_placeholder = {1: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}

        pool_size = (5, 5)
        strides = (1, 2)
        padding = "VALID"

        with tf_compat.v1.variable_scope("pool_nhwc_from_nhwc"):
            pool_nhwc_from_nhwc = PoolLayer(
                name="pool_nhwc_from_nhwc",
                network=net,
                mode="max",
                pool_size=pool_size,
                in_spatial_dims=["T", "dim:16"],
                padding=padding,
                strides=strides,
                use_channel_first=False,
                sources=[src_nhwc],
                output=PoolLayer.get_out_data_from_opts(
                    name="pool_nhwc_from_nhwc",
                    pool_size=pool_size,
                    padding=padding,
                    strides=strides,
                    in_spatial_dims=["T", "dim:16"],
                    use_channel_first=False,
                    network=net,
                    sources=[src_nhwc],
                ),
            )
        with tf_compat.v1.variable_scope("pool_nchw_from_nhwc"):
            pool_nchw_from_nhwc = PoolLayer(
                name="pool_nchw_from_nhwc",
                network=net,
                mode="max",
                pool_size=pool_size,
                in_spatial_dims=["T", "dim:16"],
                padding=padding,
                strides=strides,
                use_channel_first=True,
                sources=[src_nhwc],
                output=PoolLayer.get_out_data_from_opts(
                    name="pool_nchw_from_nhwc",
                    pool_size=pool_size,
                    padding=padding,
                    strides=strides,
                    in_spatial_dims=["T", "dim:16"],
                    use_channel_first=True,
                    network=net,
                    sources=[src_nhwc],
                ),
            )
        with tf_compat.v1.variable_scope("pool_nchw_from_nchw"):
            pool_nchw_from_nchw = PoolLayer(
                name="pool_nchw_from_nchw",
                network=net,
                mode="max",
                pool_size=pool_size,
                in_spatial_dims=["T", "dim:16"],
                padding=padding,
                strides=strides,
                use_channel_first=True,
                sources=[src_nchw],
                output=PoolLayer.get_out_data_from_opts(
                    name="pool_nchw_from_nchw",
                    pool_size=pool_size,
                    padding=padding,
                    strides=strides,
                    in_spatial_dims=["T", "dim:16"],
                    use_channel_first=True,
                    network=net,
                    sources=[src_nchw],
                ),
            )
        tf_compat.v1.global_variables_initializer().run()
        out, seq_lens = session.run(
            [pool_nhwc_from_nhwc.output.placeholder, pool_nhwc_from_nhwc.output.get_sequence_lengths()],
            feed_dict={
                src_nhwc.output.placeholder: np.random.rand(10, 11, 16, 17),
                src_nhwc.output.get_sequence_lengths(): np.full(shape=(10,), fill_value=11),
            },
        )
        print(out.shape)
        assert out.shape == (10, 7, 6, 17)
        print(seq_lens)
        out, seq_lens = session.run(
            [pool_nchw_from_nhwc.output.placeholder, pool_nchw_from_nhwc.output.get_sequence_lengths()],
            feed_dict={
                src_nhwc.output.placeholder: np.random.rand(10, 11, 16, 17),
                src_nhwc.output.get_sequence_lengths(): np.full(shape=(10,), fill_value=11),
            },
        )
        print(pool_nchw_from_nhwc.output, out.shape)
        if pool_nchw_from_nhwc.output.feature_dim_axis == 1:
            assert out.shape == (10, 17, 7, 6)
        else:
            assert out.shape == (10, 7, 6, 17)
        print(seq_lens)
        if tf_util.is_gpu_available():
            out, seq_lens = session.run(
                [pool_nchw_from_nchw.output.placeholder, pool_nchw_from_nchw.output.get_sequence_lengths()],
                feed_dict={
                    src_nchw.output.placeholder: np.random.rand(10, 17, 11, 16),
                    src_nchw.output.get_sequence_lengths(): np.full(shape=(10,), fill_value=11),
                },
            )
            print(out.shape)
            assert out.shape == (10, 17, 7, 6)
            print(seq_lens)


def test_TransposedConvLayer_1d_time_major():
    # https://github.com/rwth-i6/returnn/issues/949
    from returnn.tf.util.data import batch_dim

    time_dim = SpatialDim("time")
    in_feat_dim = FeatureDim("in_feat", 5)
    config = Config({"extern_data": {"data": {"dim_tags": [time_dim, batch_dim, in_feat_dim]}}})
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {
                "output": {
                    "class": "transposed_conv",
                    "activation": None,
                    "filter_size": [3],
                    "from": "data",
                    "n_out": 7,
                    "strides": [2],
                }
            }
        )
        out = net.get_default_output_layer().output
        session.run(tf_compat.v1.global_variables_initializer())
        session.run(out.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_TransposedConvLayer_2d_simple():
    # https://github.com/rwth-i6/returnn/issues/595
    n_batch, n_time, n_in, n_out = 7, 3, 5, 13
    config = Config(
        {"extern_data": {"data": {"dim": n_in, "shape": (n_in, None)}}}  # [B,D,T], i.e. batch-feature-major
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        assert net.extern_data.get_default_input_data().is_batch_feature_major
        net.construct_from_dict(
            {
                "Unflatten": {"class": "split_dims", "from": "data", "axis": "T", "dims": [-1, 1]},  # [B,D,T,1]
                "output": {  # [B,D',T,2]
                    "class": "transposed_conv",
                    "from": "Unflatten",
                    "activation": None,
                    "with_bias": True,
                    "in_spatial_dims": ("T", "dim:1"),
                    "n_out": n_out,
                    "filter_size": (1, 2),
                    "strides": (1, 1),
                    "padding": "valid",
                    "output_padding": (0, 0),
                    "remove_padding": (0, 0),
                },
            }
        )
        out = net.get_default_output_layer().output.copy_as_batch_feature_major()
        assert out.batch_shape == (None, 13, None, 2)
        assert out.get_dim_tag(2) == net.extern_data.get_default_input_data().get_time_dim_tag()
        assert out.dim_tags[1].dimension == n_out and out.dim_tags[3].dimension == 2
        in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_in, n_time))
        session.run(tf_compat.v1.global_variables_initializer())
        out_v = session.run(out.placeholder, feed_dict={net.extern_data.data["data"].placeholder: in_v})
        assert isinstance(out_v, numpy.ndarray)
        assert out_v.shape == (n_batch, n_out, n_time, 2)


def test_TransposedConvLayer_2d_2x2():
    n_batch, n_time, n_in, n_out = 7, 3, 5, 13
    config = Config(
        {"extern_data": {"data": {"dim": n_in, "shape": (n_in, None)}}}  # [B,D,T], i.e. batch-feature-major
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        assert net.extern_data.get_default_input_data().is_batch_feature_major
        net.construct_from_dict(
            {
                "Unflatten": {"class": "split_dims", "from": "data", "axis": "T", "dims": [-1, 1]},  # [B,D,T,1]
                "output": {  # [B,D',T,2]
                    "class": "transposed_conv",
                    "from": "Unflatten",
                    "activation": None,
                    "with_bias": True,
                    "in_spatial_dims": ("T", "dim:1"),
                    "n_out": n_out,
                    "filter_size": (2, 2),
                    "strides": (2, 2),
                    "padding": "valid",
                    "output_padding": (0, 0),
                    "remove_padding": (0, 0),
                },
            }
        )
        out = net.get_default_output_layer().output.copy_as_batch_feature_major()
        assert out.batch_shape == (None, 13, None, 2)
        assert out.get_dim_tag(2) != net.extern_data.get_default_input_data().get_time_dim_tag()
        assert out.dim_tags[1].dimension == n_out and out.dim_tags[3].dimension == 2
        in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_in, n_time))
        session.run(tf_compat.v1.global_variables_initializer())
        out_v = session.run(out.placeholder, feed_dict={net.extern_data.data["data"].placeholder: in_v})
        assert isinstance(out_v, numpy.ndarray)
        assert out_v.shape == (n_batch, n_out, n_time * 2, 2)


def test_TransposedConvLayer_out_size_pool_pad_same():
    with make_scope() as session:
        from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

        time_dim = SpatialDim("time")
        feature_dim = FeatureDim("feature", 5)
        config = Config({"extern_data": {"data": {"dim_tags": (batch_dim, time_dim, feature_dim)}}})
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "downsample": {
                    "class": "pool",
                    "mode": "avg",
                    "pool_size": [2],
                    "from": "data",
                    "padding": "same",
                },  # ceildiv with same
                "upsample": {
                    "class": "transposed_conv",
                    "filter_size": [2],
                    "from": "downsample",
                    "padding": "same",
                    "out_spatial_dims": [time_dim],
                    "out_dim": feature_dim,
                },
                "output": {"class": "combine", "from": ["upsample", "data"], "kind": "sub"},
            }
        )
        net.initialize_params(session)
        session.run(net.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_TransposedConvLayer_out_size_pool_pad_valid():
    with make_scope() as session:
        from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

        time_dim = SpatialDim("time")
        feature_dim = FeatureDim("feature", 5)
        config = Config({"extern_data": {"data": {"dim_tags": (batch_dim, time_dim, feature_dim)}}})
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "downsample": {
                    "class": "pool",
                    "mode": "avg",
                    "pool_size": [2],
                    "from": "data",
                    "padding": "valid",
                },  # floordiv with valid
                "upsample": {
                    "class": "transposed_conv",
                    "filter_size": [2],
                    "from": "downsample",
                    "padding": "valid",
                    "out_spatial_dims": [time_dim],
                    "out_dim": feature_dim,
                },
                "output": {"class": "combine", "from": ["upsample", "data"], "kind": "sub"},
            }
        )
        net.initialize_params(session)
        session.run(net.get_default_output_layer().output.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_ReduceLayer_NCHW():
    with make_scope() as session:
        import numpy as np

        net = TFNetwork(extern_data=ExternData())
        with tf_compat.v1.variable_scope("src_nchw"):
            src_nchw = InternalLayer(
                name="src_nchw",
                network=net,
                output=Data(
                    **{
                        "name": "src_nchw_output",
                        "dim": 16,
                        "shape": (16, None, 16),
                        "batch_dim_axis": 0,
                        "time_dim_axis": 2,
                        "feature_dim_axis": 1,
                        "sparse": False,
                    }
                ),
            )
            src_nchw.output.placeholder = tf_compat.v1.placeholder(shape=(None, 16, None, 16), dtype=tf.float32)
            src_nchw.output.size_placeholder = {1: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}
        with tf_compat.v1.variable_scope("reduce1"):
            reduce1 = ReduceLayer(
                name="reduce1",
                network=net,
                mode="max",
                axis="f",
                sources=[src_nchw],
                output=ReduceLayer.get_out_data_from_opts(name="reduce1", mode="max", axis="f", sources=[src_nchw]),
            )
        with tf_compat.v1.variable_scope("reduce2"):
            reduce2 = ReduceLayer(
                name="reduce2",
                network=net,
                mode="max",
                axis="b",
                sources=[src_nchw],
                output=ReduceLayer.get_out_data_from_opts(name="reduce2", mode="max", axis="b", sources=[src_nchw]),
            )
        out1, out2 = session.run(
            [reduce1.output.placeholder, reduce2.output.placeholder],
            feed_dict={
                src_nchw.output.placeholder: np.random.rand(10, 16, 11, 16),
                src_nchw.output.size_placeholder[1]: np.full(shape=(10,), fill_value=11),
            },
        )
        assert out1.shape == (10, 11, 16)
        assert out2.shape == (16, 11, 16)
        assert reduce1.output.time_dim_axis == 1
        assert reduce2.output.feature_dim_axis == 0 and reduce2.output.dim == 16
        assert reduce2.output.batch_dim_axis is None


def test_Loss_NCHW():
    with make_scope() as session:
        import numpy as np

        net = TFNetwork(extern_data=ExternData())
        with tf_compat.v1.variable_scope("src_nchw"):
            src_nchw = InternalLayer(
                name="src_nchw",
                network=net,
                output=Data(
                    **{
                        "name": "src_nchw_output",
                        "dim": 16,
                        "shape": (16, None),
                        "batch_dim_axis": 0,
                        "time_dim_axis": 2,
                        "feature_dim_axis": 1,
                        "sparse": False,
                    }
                ),
            )
            src_nchw.output.placeholder = tf_compat.v1.placeholder(shape=(None, 16, None), dtype=tf.float32)
            src_nchw.output.size_placeholder = {1: tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)}

        with tf_compat.v1.variable_scope("activation"):
            activation = ActivationLayer(
                name="activation",
                activation="softmax",
                network=net,
                sources=[src_nchw],
                output=ActivationLayer.get_out_data_from_opts(
                    name="activation", activation="softmax", network=net, sources=[src_nchw]
                ),
            )

        target_placeholder = tf_compat.v1.placeholder(shape=(None, None, 16), dtype=tf.float32)
        target_size_placeholder = tf_compat.v1.placeholder(shape=(None,), dtype=tf.int32)
        target_data = Data(
            name="target",
            shape=(None, 16),
            placeholder=target_placeholder,
            size_placeholder={0: target_size_placeholder},
            time_dim_axis=1,
            feature_dim_axis=2,
        )

        with tf_compat.v1.variable_scope("loss"):
            loss = CrossEntropyLoss(base_network=net)
            loss.init(
                output=activation.output,
                output_with_activation=activation.output_before_activation,
                target=target_data,
                layer=activation,
            )

        random_input = np.random.rand(10, 16, 32)
        loss_out, out_flat = session.run(
            [loss.get_value(), loss.output_before_softmax_flat],
            feed_dict={
                src_nchw.output.placeholder: random_input,
                src_nchw.output.size_placeholder[1]: np.full(shape=(10,), fill_value=32),
                target_placeholder: np.random.rand(10, 32, 16),
                target_size_placeholder: np.full(shape=(10,), fill_value=32),
            },
        )
        print(loss_out)
        assert loss.output.feature_dim_axis == 2
        assert out_flat.shape == (320, 16)


def test_ResizeLayer_fill_value():
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData())
        src = InternalLayer(name="src", network=net, output=Data(name="src", dim=20, sparse=True))
        src.output.placeholder = tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=tf.int32)
        src.output.size_placeholder = {0: tf.constant([5, 3], dtype=tf.int32)}
        layer = ResizeLayer(
            name="resize",
            network=net,
            factor=3,
            axis="T",
            kind="fill",
            fill_value=19,
            sources=[src],
            output=ResizeLayer.get_out_data_from_opts(
                name="resize", network=net, factor=3, axis="T", kind="fill", sources=[src]
            ),
        )
        out, seq_lens = session.run([layer.output.placeholder, layer.output.size_placeholder[0]])
        print(out)
        print(seq_lens)
        assert isinstance(out, numpy.ndarray)
        assert isinstance(seq_lens, numpy.ndarray)
        assert out.tolist() == [
            [
                1,
                19,
                19,
                2,
                19,
                19,
                3,
                19,
                19,
                4,
                19,
                19,
                5,
                19,
                19,
            ],
            [6, 19, 19, 7, 19, 19, 8, 19, 19, 9, 19, 19, 10, 19, 19],
        ]
        assert seq_lens.tolist() == [15, 9]


def test_ResizeLayer_fill_dropout():
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData())
        src = InternalLayer(name="src", network=net, output=Data(name="src", dim=20, sparse=True))
        src_seqs = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        src_seq_lens = [5, 3]
        factor = 3
        fill_value = 19
        src.output.placeholder = tf.constant(src_seqs, dtype=tf.int32)
        src.output.size_placeholder = {0: tf.constant(src_seq_lens, dtype=tf.int32)}
        layer = ResizeLayer(
            name="resize",
            network=net,
            factor=factor,
            axis="T",
            kind="fill",
            fill_value=fill_value,
            fill_dropout=0.5,
            sources=[src],
            output=ResizeLayer.get_out_data_from_opts(
                name="resize",
                network=net,
                factor=factor,
                axis="T",
                kind="fill",
                fill_value=fill_value,
                fill_dropout=0.5,
                sources=[src],
            ),
        )
        out, seq_lens = session.run([layer.output.placeholder, layer.output.size_placeholder[0]])
        print(out)
        print(seq_lens)
        assert isinstance(out, numpy.ndarray)
        assert isinstance(seq_lens, numpy.ndarray)
        # Non-deterministic output. But we can check some constraints.
        for i in range(len(src_seq_lens)):
            assert src_seq_lens[i] <= seq_lens[i] <= src_seq_lens[i] * factor
            assert [s for s in out[i] if s != fill_value] == src_seqs[i]


def test_ResizeLayer_BFT():
    n_batch, n_time, n_in = 7, 3, 5
    config = Config(
        {"extern_data": {"data": {"dim": n_in, "shape": (n_in, None)}}}  # [B,F,T], i.e. batch-feature-major
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        assert net.extern_data.get_default_input_data().is_batch_feature_major
        net.construct_from_dict({"output": {"class": "resize", "axis": "T", "factor": 2, "kind": "nn", "from": "data"}})
        out = net.get_default_output_layer().output.copy_as_batch_feature_major()
        in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_in, n_time))
        session.run(tf_compat.v1.global_variables_initializer())
        out_v = session.run(out.placeholder, feed_dict={net.extern_data.data["data"].placeholder: in_v})
        assert isinstance(out_v, numpy.ndarray)
        assert out_v.shape == (n_batch, n_in, n_time * 2)


def test_ResizeLayer_dynamic():
    n_batch, n_time, n_in = 2, 5, 3
    in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_time, n_in))
    in_seq_lens = numpy.array([5, 4])
    config = Config(
        {
            "extern_data": {
                "data": {"shape": (None, n_in)},
                "factor": {"shape": (), "batch_dim_axis": None, "dtype": "float32"},
            }
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {"output": {"class": "resize", "axis": "T", "factor": "data:factor", "kind": "nn", "from": "data"}}
        )
        out = net.get_default_output_layer().output
        for factor in [0.5, 2.0]:
            out_v, out_lens = session.run(
                (out.placeholder, out.get_sequence_lengths()),
                feed_dict={
                    net.extern_data.get_batch_info().dim: n_batch,
                    net.extern_data.data["data"].placeholder: in_v,
                    net.extern_data.data["data"].get_sequence_lengths(): in_seq_lens,
                    net.extern_data.data["factor"].placeholder: factor,
                },
            )
            assert isinstance(out_v, numpy.ndarray)
            assert out_v.shape == (n_batch, numpy.ceil(n_time * factor), n_in)
            numpy.testing.assert_equal(out_lens, numpy.ceil(in_seq_lens * factor).astype("int32"))


def test_PostfixInTimeLayer():
    with make_scope() as session:
        import numpy as np

        net = TFNetwork(extern_data=ExternData())
        batch = BatchInfo.make_global_batch_info(tf.constant(2))
        net.extern_data.set_batch_info(batch)
        src = InternalLayer(name="src", network=net, output=Data(name="src", dim=2, dtype="int32", batch=batch))
        src_seqs = np.array([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], [[6, 6], [7, 7], [8, 8], [0, 0], [0, 0]]])
        src_seq_lens = [5, 3]
        src.output.placeholder = tf.constant(src_seqs, dtype=tf.int32)
        src.output.size_placeholder = {0: tf.constant(src_seq_lens, dtype=tf.int32)}

        static_postfix = -7
        layer_postfix = InternalLayer(
            name="postfix", network=net, output=Data(name="postfix", dim=2, time_dim_axis=None, dtype="int32")
        )
        layer_postfix.output.placeholder = tf.constant([[-7, -8], [-9, -10]], dtype=tf.int32)
        layer_postfix_wo_batch = InternalLayer(
            name="postfix_wo_batch",
            network=net,
            output=Data(name="postfix_wo_batch", dim=2, time_dim_axis=None, batch_dim_axis=None, dtype="int32"),
        )
        layer_postfix_wo_batch.output.placeholder = tf.constant([-7, -8], dtype=tf.int32)

        for postfix in [static_postfix, layer_postfix, layer_postfix_wo_batch]:
            for repeat in (1, 3):
                layer = PostfixInTimeLayer(
                    name="postfix_in_time",
                    network=net,
                    sources=[src],
                    postfix=postfix,
                    repeat=repeat,
                    output=PostfixInTimeLayer.get_out_data_from_opts(
                        name="postfix_in_time", network=net, sources=[src], postfix=postfix, repeat=repeat
                    ),
                )
                out, seq_lens = session.run([layer.output.placeholder, layer.output.size_placeholder[0]])
                print(out)
                print(seq_lens)
                assert isinstance(out, numpy.ndarray)
                assert isinstance(seq_lens, numpy.ndarray)
                assert out.shape == (2, 5 + repeat, 2)
                assert all(new_len == src_len + repeat for new_len, src_len in zip(seq_lens, src_seq_lens))
                assert out[0, src_seq_lens[0] - 1, 0] == src_seqs[0, src_seq_lens[0] - 1, 0]
                assert out[1, src_seq_lens[1] - 1, 0] == src_seqs[1, src_seq_lens[1] - 1, 0]
                assert out[0, src_seq_lens[0], 0] == -7
                assert out[0, src_seq_lens[0] + repeat - 1, 0] == -7


def test_TimeChunkingLayer():
    n_batch, n_time, n_in = 2, 11, 3
    in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_time, n_in))
    in_seq_lens = numpy.array([11, 9])
    config = Config({"extern_data": {"data": {"shape": (None, n_in)}}})
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {"output": {"class": "time_chunking", "chunk_size": 5, "chunk_step": 5, "from": "data"}}
        )
        out = net.get_default_output_layer().output
        print("out:", out)
        out_v, out_lens = session.run(
            (out.placeholder, out.get_sequence_lengths()),
            feed_dict={
                net.extern_data.get_batch_info().dim: n_batch,
                net.extern_data.data["data"].placeholder: in_v,
                net.extern_data.data["data"].get_sequence_lengths(): in_seq_lens,
            },
        )
        assert isinstance(out_v, numpy.ndarray)
        print(out_v.shape)


def test_TimeChunkingLayer_TimeUnchunkingLayer():
    n_batch, n_time, n_in = 2, 11, 3
    in_v = numpy.arange(0, n_batch * n_time * n_in).astype("float32").reshape((n_batch, n_time, n_in))
    in_seq_lens = numpy.array([11, 9])
    for b in range(n_batch):
        in_v[b, in_seq_lens[b] :] = 0
    config = Config({"extern_data": {"data": {"shape": (None, n_in)}}})
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(
            {
                "chunked": {"class": "time_chunking", "chunk_size": 5, "chunk_step": 5, "from": "data"},
                "output": {"class": "time_unchunking", "chunking_layer": "chunked", "from": "chunked"},
            }
        )
        in_ = net.get_layer("data").output
        out = net.get_default_output_layer().output
        print("out:", out)
        out = out.copy_transpose(in_.dims).copy_masked(0.0)
        out_v, out_lens = session.run(
            (out.placeholder, out.get_sequence_lengths()),
            feed_dict={
                net.extern_data.get_batch_info().dim: n_batch,
                net.extern_data.data["data"].placeholder: in_v,
                net.extern_data.data["data"].get_sequence_lengths(): in_seq_lens,
            },
        )
        assert isinstance(out_v, numpy.ndarray)
        assert out_v.shape == in_v.shape
        print(out_v)
        numpy.testing.assert_equal(out_v, in_v)


def test_DotLayer():
    with make_scope() as session:
        B = 2
        H = 3
        D = H * 5
        net = TFNetwork(extern_data=ExternData())
        a = InternalLayer(name="A", network=net, output=Data(name="A", shape=(None, H, D // H)))
        assert a.output.batch_dim_axis == 0
        assert a.output.time_dim_axis == 1
        assert a.output.shape == (None, H, D // H)
        assert a.output.dim == D // H
        a_seq_lens = [7, 3]
        assert len(a_seq_lens) == B
        a.output.placeholder = tf.reshape(
            tf.range(B * max(a_seq_lens) * D, dtype=tf.float32), (B, max(a_seq_lens), H, D // H)
        )
        a.output.size_placeholder = {0: tf.constant(a_seq_lens, dtype=tf.int32)}
        b = InternalLayer(name="B", network=net, output=Data(name="B", shape=(H, D // H)))
        assert b.output.batch_dim_axis == 0
        assert b.output.shape == (H, D // H)
        assert b.output.dim == D // H
        b.output.placeholder = tf.reshape(tf.add(tf.range(B * D, dtype=tf.float32), 0.5), (B, H, D // H))
        kwargs = dict(
            name="dot",
            network=net,
            sources=[a, b],
            debug=True,
            red1="F",
            red2="F",
            var1="T",
            var2=None,
            add_var2_if_empty=False,
        )
        layer = DotLayer(output=DotLayer.get_out_data_from_opts(**kwargs), **kwargs)
        print(layer, layer.output)
        assert layer.output.batch_dim_axis == 0
        assert layer.output.time_dim_axis == 2
        assert layer.output.shape == (H, None)
        out, seq_lens = session.run([layer.output.placeholder, layer.output.size_placeholder[1]])
        print(out)
        print(seq_lens)
        assert isinstance(out, numpy.ndarray)
        assert isinstance(seq_lens, numpy.ndarray)
        assert seq_lens.tolist() == a_seq_lens
        assert out.shape == (B, H, max(a_seq_lens))


def test_DotLayer2():
    """Test if DotLayer can handle inputs which dont have a batch-dim"""
    with make_scope() as session:
        B = 3
        S1, S2, R, V = 2, 4, 8, 16
        net = TFNetwork(extern_data=ExternData())

        a = InternalLayer(
            name="A", network=net, output=Data(name="A", shape=(S1, S2, R), batch_dim_axis=0, time_dim_axis=None)
        )
        assert a.output.batch_dim_axis == 0
        assert a.output.time_dim_axis is None
        assert a.output.shape == (S1, S2, R)
        assert a.output.dim == R
        a.output.placeholder = tf.reshape(tf.range(B * S1 * S2 * R, dtype=tf.float32), (B, S1, S2, R))
        a.output.size_placeholder = {}

        b = InternalLayer(
            name="B", network=net, output=Data(name="B", shape=(S1, S2, R, V), batch_dim_axis=None, time_dim_axis=None)
        )
        assert b.output.batch_dim_axis is None
        assert b.output.time_dim_axis is None
        assert b.output.shape == (S1, S2, R, V)
        assert b.output.dim == V
        b.output.placeholder = tf.reshape(tf.range(S1 * S2 * R * V, dtype=tf.float32), (S1, S2, R, V))
        b.output.size_placeholder = {}

        kwargs = dict(
            name="dot", network=net, sources=[a, b], debug=True, red1="F", red2="dim:%i" % R, var1="B", var2="F"
        )
        layer = DotLayer(output=DotLayer.get_out_data_from_opts(**kwargs), **kwargs)
        print(layer, layer.output)
        assert layer.output.batch_dim_axis == 2
        assert layer.output.time_dim_axis is None
        assert layer.output.shape == (S1, S2, V)
        assert layer.output.batch_shape == (S1, S2, None, V)
        assert layer.output.dim == V
        out = session.run(layer.output.placeholder)
        assert isinstance(out, numpy.ndarray)
        assert out.shape == (S1, S2, B, V)


def test_DotLayer_linear_square_matrix():
    from returnn.tf.util.data import batch_dim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feature", dimension=3)
    config = Config(
        {
            "extern_data": {
                "data": {"dim_tags": [batch_dim, time_dim, feat_dim]},
                "matrix_ambiguous": {"dim_tags": [feat_dim, feat_dim], "available_for_inference": True},
                "matrix_non_ambiguous": {
                    "dim_tags": [feat_dim.copy(match_priority=1), feat_dim],
                    "available_for_inference": True,
                },
            },
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        try:
            net.construct_from_dict(
                {
                    "output": {"class": "dot", "from": ["data:data", "data:matrix_ambiguous"], "reduce": feat_dim},
                }
            )
        except Exception as exc:
            print("Expected exception: %r" % exc)
            assert "must be unique" in str(exc)
        else:
            raise Exception("Expected exception but constructed layer: %s" % net.get_default_output_layer())
        net.construct_from_dict(
            {
                "output": {"class": "dot", "from": ["data:data", "data:matrix_non_ambiguous"], "reduce": feat_dim},
            }
        )
        out = net.get_default_output_layer().output
        assert out.dim_tags == (batch_dim, time_dim, feat_dim)
        session.run(out.placeholder, feed_dict=make_feed_dict(net.extern_data))


def test_DotLayer_mask_dyn_seq():
    batch = Dim(kind=Dim.Types.Batch, description="batch", dimension=None)
    time = SpatialDim("time")
    feat1 = FeatureDim("feature 1", dimension=3)
    feat2 = FeatureDim("feature 2", dimension=5)
    config = Config(
        {
            "extern_data": {
                "src1": {"dim_tags": [batch, time, feat1]},
                "src2": {"dim_tags": [batch, time, feat2]},
            },
            "network": {
                "dot": {
                    "class": "dot",
                    "from": ["data:src1", "data:src2"],
                    "is_output_layer": True,
                    "red1": time,
                    "red2": time,
                    "var1": feat1,
                    "var2": feat2,
                },
            },
            "debug_print_layer_output_template": True,
        }
    )

    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(config.typed_dict["network"])
        layer = net.layers["dot"]
        assert isinstance(layer, DotLayer)
        assert layer.output.dim_tags == (batch, feat1, feat2)
        assert layer._info_reduce_mask == "mask-source-1"

        feed_dict = make_feed_dict(net.extern_data)
        session.run(layer.output.placeholder, feed_dict=feed_dict)


def test_DotLayer_mask_dyn_seq_after_softmax():
    batch = Dim(kind=Dim.Types.Batch, description="batch", dimension=None)
    time = SpatialDim("time")
    feat1 = FeatureDim("feature 1", dimension=3)
    feat2 = FeatureDim("feature 2", dimension=5)
    config = Config(
        {
            "extern_data": {
                "src1": {"dim_tags": [batch, time, feat1]},
                "src2": {"dim_tags": [batch, time, feat2]},
            },
            "network": {
                "sm1": {"class": "softmax_over_spatial", "from": "data:src1"},
                "dot": {
                    "class": "dot",
                    "from": ["sm1", "data:src2"],
                    "is_output_layer": True,
                    "red1": time,
                    "red2": time,
                    "var1": feat1,
                    "var2": feat2,
                },
            },
            "debug_print_layer_output_template": True,
        }
    )

    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(config.typed_dict["network"])
        layer = net.layers["dot"]
        assert isinstance(layer, DotLayer)
        assert layer.output.dim_tags == (batch, feat1, feat2)
        assert layer._info_reduce_mask == "source-0-already-masked"

        feed_dict = make_feed_dict(net.extern_data)
        session.run(layer.output.placeholder, feed_dict=feed_dict)


def test_DotLayer_self_att_dyn_size_ext():
    batch_dim = Dim(kind=Dim.Types.Batch, dimension=None)
    heads_dim = SpatialDim("heads", dimension=8)
    classes_dim = Dim(kind=Dim.Types.Time, description="classes", dimension=None)
    keys_dim = Dim(
        kind=Dim.Types.Spatial,
        description="keys",
        dyn_size_ext=Data(name="keys_dyn_size", dim_tags=[classes_dim], dtype="int32", auto_create_placeholders=True),
        dimension=None,
    )
    feature_dim = FeatureDim("feature", dimension=64)

    a = Data(name="att_weights", dim_tags=[batch_dim, heads_dim, classes_dim, keys_dim], auto_create_placeholders=True)
    b = Data(name="att_value", dim_tags=[keys_dim, batch_dim, heads_dim, feature_dim], auto_create_placeholders=True)
    print("a:", a)
    print("b:", b)

    config = Config({"debug_print_layer_output_template": True})
    config.update(dict(num_inputs=64 * 8, num_outputs=64 * 8))
    net = TFNetwork(config=config, train_flag=True, search_flag=False, eval_flag=False)

    a_lay = InternalLayer(name="att_weights", network=net, output=a)
    b_lay = InternalLayer(name="att_value", network=net, output=b)
    dot_kwargs = {"red1": keys_dim, "red2": keys_dim, "var1": classes_dim, "var2": feature_dim}
    dot = DotLayer.get_out_data_from_opts(name="dot", sources=[a_lay, b_lay], **dot_kwargs)
    DotLayer(network=net, name="dot", sources=[a_lay, b_lay], output=dot, **dot_kwargs)  # just check that it builds.


def test_DotLayer_sparse_input():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    classes_dim = FeatureDim("classes", dimension=5)
    embed_dim = FeatureDim("embed", dimension=3)
    config = Config(
        {
            "extern_data": {
                "src": {"dim_tags": [batch_dim, time_dim], "sparse_dim": classes_dim},
                "embed": {"dim_tags": [classes_dim, embed_dim]},
            },
            "network": {
                "output": {"class": "dot", "from": ["data:src", "data:embed"], "reduce": classes_dim},
            },
            "debug_runtime_sanity_checks": True,
        }
    )

    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(config.typed_dict["network"])
        layer = net.get_default_output_layer()
        assert layer.output.dim_tags == (batch_dim, time_dim, embed_dim)

        feed_dict = make_feed_dict(net.extern_data)
        session.run(layer.output.placeholder, feed_dict=feed_dict)


def test_DotLayer_dim_wrong_matching_same_dim_value():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("feat", dimension=5)
    feat2_dim = FeatureDim("other-feat", dimension=5)

    # First directly check DotLayer.get_out_data_from_opts.
    # This is more similar like we have it in returnn_common
    # and might trigger different errors due to the dim matching logic of DotLayer,
    # which behaves slightly different when there are no size_placeholders set yet,
    # see Dim.is_equal with unknown_spatial_matches.
    a = Data("a", dim_tags=[batch_dim, time_dim, feat_dim])
    b = Data("b", dim_tags=[batch_dim, time_dim, feat2_dim])
    net = TFNetwork(config=Config(), extern_data=ExternData())
    out = DotLayer.get_out_data_from_opts(
        name="dot",
        sources=[InternalLayer(name="a", network=net, output=a), InternalLayer(name="b", network=net, output=b)],
        reduce=time_dim,
    )
    assert out.dim_tags == (batch_dim, feat_dim, feat2_dim)

    # Now full config.
    config = Config(
        {
            "extern_data": {
                "a": {"dim_tags": [batch_dim, time_dim, feat_dim]},
                "b": {"dim_tags": [batch_dim, time_dim, feat2_dim]},
            },
            "network": {
                "output": {"class": "dot", "from": ["data:a", "data:b"], "reduce": time_dim},
            },
            "debug_runtime_sanity_checks": True,
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(config.typed_dict["network"])
        layer = net.get_default_output_layer()
        assert layer.output.dim_tags == (batch_dim, feat_dim, feat2_dim)
        feed_dict = make_feed_dict(net.extern_data)
        session.run(layer.output.placeholder, feed_dict=feed_dict)


def test_DotLayer_dim_wrong_matching_derived():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    time_dim_2 = time_dim * 2
    assert time_dim_2.derived_from_tag == time_dim
    assert time_dim_2.get_same_derived_base() == time_dim
    feat_dim = FeatureDim("feat", dimension=5)

    # First directly check DotLayer.get_out_data_from_opts.
    # This is more similar like we have it in returnn_common
    # and might trigger different errors due to the dim matching logic of DotLayer,
    # which behaves slightly different when there are no size_placeholders set yet,
    # see Dim.is_equal with unknown_spatial_matches.
    a = Data("a", dim_tags=[batch_dim, time_dim, feat_dim])
    b = Data("b", dim_tags=[batch_dim, time_dim_2, feat_dim])
    net = TFNetwork(config=Config(), extern_data=ExternData())
    out = DotLayer.get_out_data_from_opts(
        name="dot",
        sources=[InternalLayer(name="a", network=net, output=a), InternalLayer(name="b", network=net, output=b)],
        reduce=feat_dim,
    )
    assert out.dim_tags == (batch_dim, time_dim, time_dim_2)

    # Now full config.
    config = Config(
        {
            "extern_data": {
                "a": {"dim_tags": [batch_dim, time_dim, feat_dim]},
                "b": {"dim_tags": [batch_dim, time_dim_2, feat_dim]},
            },
            "network": {
                "output": {"class": "dot", "from": ["data:a", "data:b"], "reduce": feat_dim},
            },
            "debug_runtime_sanity_checks": True,
        }
    )
    with make_scope() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(config.typed_dict["network"])
        layer = net.get_default_output_layer()
        assert layer.output.dim_tags == (batch_dim, time_dim, time_dim_2)
        feed_dict = make_feed_dict(net.extern_data)
        session.run(layer.output.placeholder, feed_dict=feed_dict)


def test_subnet_load_on_init():
    import tempfile

    model_tmp_dir = tempfile.mkdtemp("tmp-checkpoint")
    model_filename = model_tmp_dir + "/model"
    with make_scope() as session:
        config = Config()
        n_in, n_hidden, n_out = 2, 5, 3
        config.update(
            {
                "num_outputs": n_out,
                "num_inputs": n_in,
                "network": {
                    "l1": {"class": "linear", "activation": None, "n_out": n_hidden, "from": "data:data"},
                    "output": {"class": "linear", "activation": None, "n_out": n_out, "from": ["l1"]},
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        network.initialize_params(session)
        params_orig_dump = network.get_params_serialized(session)
        print("l1:")
        print(params_orig_dump.values_dict["l1"]["W"])
        print("output:")
        print(params_orig_dump.values_dict["output"]["W"])
        assert params_orig_dump.values_dict["l1"]["W"].any()
        assert params_orig_dump.values_dict["output"]["W"].any()
        network.save_params_to_file(filename=model_filename, session=session)

    with make_scope() as session:
        config = Config()
        config.update(
            {
                "num_outputs": n_out,
                "num_inputs": n_in,
                "network": {
                    "l0": {"class": "linear", "activation": None, "n_out": n_in, "from": "data:data"},
                    "subnet": {
                        "class": "subnetwork",
                        "from": ["l0"],
                        "load_on_init": model_filename,
                        "subnetwork": {
                            "l1": {"class": "linear", "activation": None, "n_out": n_hidden, "from": "data"},
                            "output": {"class": "linear", "activation": None, "n_out": n_out, "from": ["l1"]},
                        },
                    },
                    "output": {"class": "linear", "activation": None, "n_out": n_out, "from": ["subnet"]},
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        network.initialize_params(session)
        params_dump = network.get_params_serialized(session)
        params_dump_subnet = params_dump.values_dict["subnet"]
        for layer_name in ["l1", "output"]:
            layer_orig = params_orig_dump.values_dict[layer_name]
            for param_name in ["W", "b"]:
                param_orig = layer_orig[param_name]
                param_subnet = params_dump_subnet[layer_name + "/" + param_name]
                numpy.testing.assert_array_equal(param_orig, param_subnet)


def test_ReuseParams_rec():
    print("test_ReuseParams_rec()")
    numpy.set_printoptions(precision=15)
    num_inputs = 100
    num_outputs = 15
    config = Config()
    config.update(
        {
            "num_inputs": num_inputs,
            "num_outputs": {"data": [num_inputs, 2], "classes": [num_outputs, 1]},  # dense output
            "network": {
                "out1": {"class": "softmax", "from": ["rec_fwd"], "loss": "ce", "n_out": num_outputs},
                "out2": {"class": "softmax", "from": ["rec_fwd_copy"], "loss": "ce", "n_out": num_outputs},
                "rec_fwd": {"class": "rec", "direction": 1, "from": ["data"], "n_out": 300, "unit": "lstmp"},
                "rec_fwd_copy": {
                    "class": "rec",
                    "direction": 1,
                    "from": ["data"],
                    "n_out": 300,
                    "unit": "lstmp",
                    "reuse_params": "rec_fwd",
                },
            },
            "optimizer": {"class": "adam"},
            "target": "classes",
            "debug_grad_summaries": True,
            "debug_save_updater_vars": True,
            "debug_add_check_numerics_ops": True,
        }
    )
    print("Creating network...")
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    random = numpy.random.RandomState(seed=1)

    def make_feed_dict(seq_len=10):
        return {
            network.extern_data.get_batch_info().dim: 1,
            network.extern_data.data["data"].placeholder: random.uniform(-1, 1, (1, seq_len, num_inputs)),
            network.extern_data.data["data"].size_placeholder[0]: numpy.array([seq_len]),
            network.extern_data.data["classes"].placeholder: random.randint(low=0, high=num_outputs, size=(1, seq_len)),
            network.extern_data.data["classes"].size_placeholder[0]: numpy.array([seq_len]),
        }

    print("Creating session...")
    with tf_compat.v1.Session() as session:
        print("Init params...")
        network.initialize_params(session=session)
        print("Testing reuse_params ...")
        feed = make_feed_dict(10)
        fwd_out, fwd_out_copy = session.run(
            [network.layers["rec_fwd"].output.placeholder, network.layers["rec_fwd_copy"].output.placeholder],
            feed_dict=feed,
        )
        numpy.testing.assert_array_equal(fwd_out, fwd_out_copy)


def test_ReuseParams_dep_loop():
    num_inputs = 10
    num_outputs = 15
    config = Config()
    config.update(
        {
            "num_inputs": num_inputs,
            "num_outputs": {"data": [num_inputs, 2], "classes": [num_outputs, 1]},  # dense output
            "network": {
                "layer1": {
                    "class": "rec",
                    "from": "data",
                    "n_out": 10,
                    "unit": {
                        "sub1": {
                            "class": "linear",
                            "from": ["data:source", "prev:output"],
                            "activation": "relu",
                            "n_out": 10,
                        },
                        "sub2": {
                            "class": "linear",
                            "from": "sub1",
                            "activation": "relu",
                            "n_out": 10,
                            "reuse_params": "base:layer2",  # circular dependency!
                        },
                        "output": {
                            "class": "linear",
                            "from": ["sub1", "sub2", "prev:output"],
                            "activation": "relu",
                            "n_out": 10,
                        },
                    },
                },
                "layer2": {"class": "linear", "from": "layer1", "activation": "relu", "n_out": 10},
                "out": {"class": "softmax", "from": "layer2", "loss": "ce", "n_out": num_outputs},
            },
            "optimizer": {"class": "adam"},
            "target": "classes",
            "debug_print_layer_output_template": True,
        }
    )
    print("Creating network...")
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])

    params = network.get_params_list()
    pprint(params)
    l1 = network.get_layer("layer1")
    from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell

    assert isinstance(l1, RecLayer)
    cell = l1.cell
    assert isinstance(cell, _SubnetworkRecCell)
    l1 = cell.net.layers["sub2"]
    assert isinstance(l1, LinearLayer)
    assert tf_util.has_control_flow_context(l1.output.placeholder)  # this should be in the loop
    l2 = network.get_layer("layer2")
    assert not tf_util.has_control_flow_context(l2.output.placeholder)  # outside the loop
    assert isinstance(l2, LinearLayer)
    assert set(l1.params.keys()) == set() and set(l2.params.keys()) == {"W", "b"}

    def make_feed_dict(seq_len=10):
        random = numpy.random.RandomState(seed=1)
        return {
            network.extern_data.get_batch_info().dim: 1,
            network.extern_data.data["data"].placeholder: random.uniform(-1, 1, (1, seq_len, num_inputs)),
            network.extern_data.data["data"].size_placeholder[0]: numpy.array([seq_len]),
            network.extern_data.data["classes"].placeholder: random.randint(low=0, high=num_outputs, size=(1, seq_len)),
            network.extern_data.data["classes"].size_placeholder[0]: numpy.array([seq_len]),
        }

    with tf_compat.v1.Session() as session:
        network.initialize_params(session=session)
        feed = make_feed_dict(10)
        # Not really needed (for testing reuse_params), but just test anyway.
        session.run(network.get_default_output_layer().output.placeholder, feed_dict=feed)


def test_ReuseParams_dep_loop_2():
    num_inputs = 10
    num_outputs = 15
    config = Config()
    config.update(
        {
            "num_inputs": num_inputs,
            "num_outputs": {"data": [num_inputs, 2], "classes": [num_outputs, 1]},  # dense output
            "network": {
                "layer1": {
                    "class": "rec",
                    "from": "data",
                    "n_out": 10,
                    "unit": {
                        "sub1": {
                            "class": "linear",
                            "from": ["data:source", "prev:output"],
                            "activation": "relu",
                            "n_out": 10,
                        },
                        "sub2": {
                            "class": "linear",
                            "from": "sub1",
                            "activation": "relu",
                            "n_out": 10,
                            "reuse_params": "base:layer2",  # circular dependency!
                            "is_output_layer": True,
                        },
                        "output": {
                            "class": "linear",
                            "from": ["sub1", "sub2", "prev:output"],
                            "activation": "relu",
                            "n_out": 10,
                        },
                    },
                },
                "layer2": {"class": "linear", "from": "layer1/sub2", "activation": "relu", "n_out": 10},
                "out": {"class": "softmax", "from": "layer2", "loss": "ce", "n_out": num_outputs},
            },
            "optimizer": {"class": "adam"},
            "target": "classes",
            "debug_print_layer_output_template": True,
        }
    )
    print("Creating network...")
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])

    params = network.get_params_list()
    pprint(params)
    l1 = network.get_layer("layer1")
    from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell

    assert isinstance(l1, RecLayer)
    cell = l1.cell
    assert isinstance(cell, _SubnetworkRecCell)
    l1 = cell.net.layers["sub2"]
    assert isinstance(l1, LinearLayer)
    assert tf_util.has_control_flow_context(l1.output.placeholder)  # this should be in the loop
    l2 = network.get_layer("layer2")
    assert not tf_util.has_control_flow_context(l2.output.placeholder)  # outside the loop
    assert isinstance(l2, LinearLayer)
    assert set(l1.params.keys()) == set() and set(l2.params.keys()) == {"W", "b"}

    def make_feed_dict(seq_len=10):
        random = numpy.random.RandomState(seed=1)
        return {
            network.extern_data.get_batch_info().dim: 1,
            network.extern_data.data["data"].placeholder: random.uniform(-1, 1, (1, seq_len, num_inputs)),
            network.extern_data.data["data"].size_placeholder[0]: numpy.array([seq_len]),
            network.extern_data.data["classes"].placeholder: random.randint(low=0, high=num_outputs, size=(1, seq_len)),
            network.extern_data.data["classes"].size_placeholder[0]: numpy.array([seq_len]),
        }

    with tf_compat.v1.Session() as session:
        network.initialize_params(session=session)
        feed = make_feed_dict(10)
        # Not really needed (for testing reuse_params), but just test anyway.
        session.run(network.get_default_output_layer().output.placeholder, feed_dict=feed)


def test_ReuseParams_dep_loop_3():
    num_inputs = 10
    num_outputs = 15
    config = Config()
    config.update(
        {
            "num_inputs": num_inputs,
            "num_outputs": {"data": [num_inputs, 2], "classes": [num_outputs, 1]},  # dense output
            "network": {
                "layer1": {
                    "class": "rec",
                    "from": "data",
                    "n_out": 10,
                    "unit": {
                        "sub1": {
                            "class": "linear",
                            "from": ["data:source", "prev:output"],
                            "activation": "relu",
                            "n_out": 10,
                            "is_output_layer": True,
                        },
                        "sub2": {
                            "class": "linear",
                            "from": "sub1",
                            "activation": "relu",
                            "n_out": 10,
                            "reuse_params": "base:layer2",  # circular dependency!
                        },
                        "output": {
                            "class": "linear",
                            "from": ["sub1", "sub2", "prev:output"],
                            "activation": "relu",
                            "n_out": 10,
                        },
                    },
                },
                "layer2": {"class": "linear", "from": "layer1/sub1", "activation": "relu", "n_out": 10},
                "out": {"class": "softmax", "from": "layer2", "loss": "ce", "n_out": num_outputs},
            },
            "optimizer": {"class": "adam"},
            "target": "classes",
            "debug_print_layer_output_template": True,
        }
    )
    print("Creating network...")
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])

    params = network.get_params_list()
    pprint(params)
    l1 = network.get_layer("layer1")
    from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell

    assert isinstance(l1, RecLayer)
    cell = l1.cell
    assert isinstance(cell, _SubnetworkRecCell)
    l1 = cell.net.layers["sub2"]
    assert isinstance(l1, LinearLayer)
    assert tf_util.has_control_flow_context(l1.output.placeholder)  # this should be in the loop
    l2 = network.get_layer("layer2")
    assert not tf_util.has_control_flow_context(l2.output.placeholder)  # outside the loop
    assert isinstance(l2, LinearLayer)
    assert set(l1.params.keys()) == set() and set(l2.params.keys()) == {"W", "b"}

    def make_feed_dict(seq_len=10):
        random = numpy.random.RandomState(seed=1)
        return {
            network.extern_data.get_batch_info().dim: 1,
            network.extern_data.data["data"].placeholder: random.uniform(-1, 1, (1, seq_len, num_inputs)),
            network.extern_data.data["data"].size_placeholder[0]: numpy.array([seq_len]),
            network.extern_data.data["classes"].placeholder: random.randint(low=0, high=num_outputs, size=(1, seq_len)),
            network.extern_data.data["classes"].size_placeholder[0]: numpy.array([seq_len]),
        }

    with tf_compat.v1.Session() as session:
        network.initialize_params(session=session)
        feed = make_feed_dict(10)
        # Not really needed (for testing reuse_params), but just test anyway.
        session.run(network.get_default_output_layer().output.placeholder, feed_dict=feed)


def test_LossAsIs_custom_dim():
    config = Config()
    config.update(
        {
            "extern_data": {
                "data": (40, 2),
                "classes": (10025, 1),
                "att_weights": {"shape": (None, None, 1)},
                "att_weights_sizes": {"shape": (None,), "dtype": "int32"},
            },
            "debug_print_layer_output_template": True,
        }
    )
    print("Creating network...")
    network = TFNetwork(config=config, train_flag=True)
    net_dict = {
        "att_distill_loss": {
            "class": "eval",
            "from": ["energy", "att_weights"],
            "out_type": (lambda sources, **kwargs: sources[0].output.copy_template_excluding_spatial_dim(-1)),
            "eval": "softmax_cross_entropy_over_size("
            + "logits=source(0, as_data=True, auto_convert=False),"
            + "labels=source(1, as_data=True, auto_convert=False))",
            "loss": "as_is",
        },
    }
    n_batch = 5
    n_enc_time = 11
    n_dec_time = 7
    with tf_compat.v1.Session() as session:
        enc_time = tf.constant([n_enc_time] * n_batch)
        dec_time = tf.constant([n_dec_time] * n_batch)
        network.add_layer(
            name="energy",
            layer_class=InternalLayer,
            output=Data(
                name="energy",
                shape=(None, None, 1),
                dim=1,
                batch_dim_axis=2,
                size_placeholder={0: dec_time, 1: enc_time},
                placeholder=tf.constant(
                    numpy.random.normal(size=(n_dec_time, n_enc_time, n_batch, 1)).astype("float32")
                ),
            ),
        )
        network.add_layer(
            name="att_weights",
            layer_class=InternalLayer,
            output=Data(
                name="att_weights",
                shape=(None, None, 1),
                dim=1,
                batch_dim_axis=0,
                size_placeholder={0: dec_time, 1: enc_time},
                placeholder=tf.expand_dims(
                    tf.nn.softmax(
                        tf.constant(numpy.random.normal(size=(n_batch, n_dec_time, n_enc_time)).astype("float32"))
                    ),
                    -1,
                ),
            ),
        )
        network.construct_from_dict(net_dict)
        loss = session.run(network.get_total_loss())
        assert loss


def test_loss_cross_entropy():
    from returnn.tf.util.data import Dim, batch_dim, single_step_dim, SpatialDim, FeatureDim

    extern_data_data_dim_tags_1_time_dim = SpatialDim("time")
    extern_data_data_dim_tags_2_input_dim = FeatureDim("input", 3)
    extern_data_classes_sparse_dim_out_dim = FeatureDim("out", 5)

    extern_data_opts = {
        "data": {
            "dim_tags": (batch_dim, extern_data_data_dim_tags_1_time_dim, extern_data_data_dim_tags_2_input_dim),
            "dtype": "float32",
            "available_for_inference": True,
        },
        "classes": {
            "dim_tags": (batch_dim, extern_data_data_dim_tags_1_time_dim),
            "dtype": "int32",
            "sparse_dim": extern_data_classes_sparse_dim_out_dim,
            "available_for_inference": True,
        },
    }

    net_dict = {
        "dot": {
            "class": "dot",
            "from": ["data:data", "weight"],
            "reduce": extern_data_data_dim_tags_2_input_dim,
            "out_shape": {batch_dim, extern_data_data_dim_tags_1_time_dim, extern_data_classes_sparse_dim_out_dim},
        },
        "add": {
            "class": "combine",
            "from": ["dot", "bias"],
            "kind": "add",
            "out_shape": {batch_dim, extern_data_data_dim_tags_1_time_dim, extern_data_classes_sparse_dim_out_dim},
        },
        "cross_entropy": {
            "class": "subnetwork",
            "from": [],
            "subnetwork": {
                "sparse_softmax_cross_entropy_with_logits": {
                    "class": "sparse_softmax_cross_entropy_with_logits",
                    "logits": "base:add",
                    "targets": "base:data:classes",
                    "axis": extern_data_classes_sparse_dim_out_dim,
                    "out_shape": {batch_dim, extern_data_data_dim_tags_1_time_dim},
                },
                "output": {
                    "class": "copy",
                    "from": "sparse_softmax_cross_entropy_with_logits",
                    "out_shape": {batch_dim, extern_data_data_dim_tags_1_time_dim},
                },
            },
            "out_shape": {batch_dim, extern_data_data_dim_tags_1_time_dim},
            "loss": "as_is",
            "loss_scale": 1.0,
        },
        "weight": {
            "class": "variable",
            "shape": [extern_data_data_dim_tags_2_input_dim, extern_data_classes_sparse_dim_out_dim],
        },
        "bias": {"class": "variable", "shape": [extern_data_classes_sparse_dim_out_dim]},
    }

    # First with train flag False. Only construct.
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData(extern_data_opts), train_flag=False)
        net.construct_from_dict(net_dict)

    # Now with train flag True. Try to calc loss.
    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData(extern_data_opts), train_flag=True)
        net.construct_from_dict(net_dict)
        loss = net.get_total_loss()
        print("loss tensor:", loss)
        assert isinstance(loss, tf.Tensor)
        tf_util.print_graph_output(loss, max_depth=4)
        from tensorflow.python.framework import tensor_util

        loss_const = tensor_util.constant_value(loss)
        assert loss_const is None
        net.initialize_params(session)
        session.run(loss, feed_dict=make_feed_dict(net.extern_data))


def test_loss_cross_entropy_as_is_optimize_flatten():
    from returnn.tf.util.data import Dim, batch_dim, single_step_dim, SpatialDim, FeatureDim

    extern_data_data_dim_tags_1_time_dim = SpatialDim("time")
    extern_data_data_dim_tags_2_input_dim = FeatureDim("input", 3)
    extern_data_classes_sparse_dim_out_dim = FeatureDim("out", 5)

    extern_data_opts = {
        "data": {
            "dim_tags": (batch_dim, extern_data_data_dim_tags_1_time_dim, extern_data_data_dim_tags_2_input_dim),
            "dtype": "float32",
            "available_for_inference": True,
        },
        "classes": {
            "dim_tags": (batch_dim, extern_data_data_dim_tags_1_time_dim),
            "dtype": "int32",
            "sparse_dim": extern_data_classes_sparse_dim_out_dim,
            "available_for_inference": True,
        },
    }

    net_dict = {
        "dot": {
            "class": "dot",
            "from": ["data:data", "weight"],
            "reduce": extern_data_data_dim_tags_2_input_dim,
            "out_shape": {batch_dim, extern_data_data_dim_tags_1_time_dim, extern_data_classes_sparse_dim_out_dim},
        },
        "add": {
            "class": "combine",
            "from": ["dot", "bias"],
            "kind": "add",
            "out_shape": {batch_dim, extern_data_data_dim_tags_1_time_dim, extern_data_classes_sparse_dim_out_dim},
        },
        "cross_entropy": {
            "class": "subnetwork",
            "from": [],
            "subnetwork": {
                "sparse_softmax_cross_entropy_with_logits": {
                    "class": "sparse_softmax_cross_entropy_with_logits",
                    "logits": "base:add",
                    "targets": "base:data:classes",
                    "axis": extern_data_classes_sparse_dim_out_dim,
                    "out_shape": {batch_dim, extern_data_data_dim_tags_1_time_dim},
                },
                "output": {
                    "class": "copy",
                    "from": "sparse_softmax_cross_entropy_with_logits",
                    "out_shape": {batch_dim, extern_data_data_dim_tags_1_time_dim},
                },
            },
            "out_shape": {batch_dim, extern_data_data_dim_tags_1_time_dim},
            "loss": "as_is",
            "loss_scale": 1.0,
        },
        "weight": {
            "class": "variable",
            "shape": [extern_data_data_dim_tags_2_input_dim, extern_data_classes_sparse_dim_out_dim],
        },
        "bias": {"class": "variable", "shape": [extern_data_classes_sparse_dim_out_dim]},
    }

    with make_scope() as session:
        net = TFNetwork(extern_data=ExternData(extern_data_opts), train_flag=True)
        net.construct_from_dict(net_dict)
        loss = net.get_total_loss()
        print("loss tensor:", loss)
        assert isinstance(loss, tf.Tensor)
        tf_util.print_graph_output(loss, max_depth=4)

        assert loss.op.type == "Sum"  # reduce over flattened frames
        loss_ = loss.op.inputs[0]
        assert isinstance(loss_, tf.Tensor)
        assert loss_.op.type == "SparseSoftmaxCrossEntropyWithLogits"

        net.initialize_params(session)
        session.run(loss, feed_dict=make_feed_dict(net.extern_data))


def test_reduce_with_flatten():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    feature_dim = FeatureDim("feat", 1)
    config = Config({"extern_data": {"data": {"dim_tags": (batch_dim, time_dim, feature_dim), "dtype": "float32"}}})
    net_dict = {
        "exp": {
            "class": "activation",
            "from": "data:data",
            "activation": "exp",
            "out_shape": {batch_dim, time_dim, feature_dim},
        },
        "mean_absolute_difference": {
            "class": "subnetwork",
            "from": [],
            "subnetwork": {
                "sub": {
                    "class": "combine",
                    "from": ["base:exp", "base:data:data"],
                    "kind": "sub",
                    "out_shape": {batch_dim, time_dim, feature_dim},
                },
                "abs": {
                    "class": "activation",
                    "from": "sub",
                    "activation": "abs",
                    "out_shape": {batch_dim, time_dim, feature_dim},
                },
                "reduce": {
                    "class": "reduce",
                    "from": "abs",
                    "mode": "mean",
                    "axis": feature_dim,
                    "out_shape": {batch_dim, time_dim},
                },
                "output": {"class": "copy", "from": "reduce", "out_shape": {batch_dim, time_dim}},
            },
            "loss": "as_is",
            "out_shape": {batch_dim, time_dim},
        },
    }
    with make_scope():
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(net_dict)
        net.get_total_loss()


def test_double_flatten_loss():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    _repeat_out_dim = time_dim * 5
    feature_dim = FeatureDim("feat", 1)
    config = Config({"extern_data": {"data": {"dim_tags": (batch_dim, time_dim, feature_dim), "dtype": "int32"}}})
    network = {
        "sub": {
            "class": "combine",
            "from": ["data:data", "data:data"],
            "kind": "sub",
            "loss": "as_is",
            "out_shape": {batch_dim, time_dim, feature_dim},
        },
        "repeat": {
            "class": "repeat",
            "from": "sub",
            "repetitions": 5,
            "axis": time_dim,
            "out_dim": _repeat_out_dim,
            "out_shape": {batch_dim, feature_dim, _repeat_out_dim},
        },
        "sub_0": {
            "class": "combine",
            "from": ["repeat", "repeat"],
            "kind": "sub",
            "loss": "as_is",
            "out_shape": {batch_dim, feature_dim, _repeat_out_dim},
        },
        "output": {"class": "copy", "from": "sub_0", "out_shape": {batch_dim, feature_dim, _repeat_out_dim}},
    }
    with make_scope():
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(network)
        net.get_total_loss()


def test_double_flatten_loss_1079():
    # https://github.com/rwth-i6/returnn/issues/1079
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    time_dim = SpatialDim("time")
    _repeat_out_dim = time_dim * 5
    feature_dim = FeatureDim("feat", 1)
    config = Config({"extern_data": {"data": {"dim_tags": (batch_dim, time_dim, feature_dim), "dtype": "int32"}}})
    network = {
        "sub": {
            "class": "combine",
            "from": ["data:data", "data:data"],
            "kind": "sub",
            "loss": "as_is",
            "out_shape": {batch_dim, time_dim, feature_dim},
        },
        "repeat": {
            "class": "repeat",
            "from": "data:data",
            "repetitions": 5,
            "axis": time_dim,
            "out_dim": _repeat_out_dim,
            "out_shape": {batch_dim, feature_dim, _repeat_out_dim},
        },
        "sub_0": {
            "class": "combine",
            "from": ["repeat", "repeat"],
            "kind": "sub",
            "loss": "as_is",
            "out_shape": {batch_dim, feature_dim, _repeat_out_dim},
        },
        "output": {"class": "copy", "from": "sub_0", "out_shape": {batch_dim, feature_dim, _repeat_out_dim}},
    }
    with make_scope():
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(network)
        net.get_total_loss()


def test_LossLayer_sublayers():
    from returnn.tf.util.basic import Dim

    n_in, n_out = 7, 11
    time_tag = SpatialDim("time")

    config = Config(
        {
            "extern_data": {
                "data": {"dim": n_in, "same_dim_tags_as": {"t": time_tag}},
                "classes": {"dim": n_out, "dtype": "int64", "sparse": True, "same_dim_tags_as": {"t": time_tag}},
                "prev-classes": {"dim": n_out, "dtype": "int64", "sparse": True, "same_dim_tags_as": {"t": time_tag}},
            },
            "debug_print_layer_output_template": True,
        }
    )
    net_dict = {
        "encoder-output": {"class": "linear", "activation": "relu", "n_out": 10, "from": "data:data"},
        "left-output": {"class": "softmax", "from": "encoder-output", "n_out": n_out},
        "left-output-ce": {
            "class": "loss",
            "from": "left-output",
            "loss": "as_is",
            "loss_": "ce",
            "loss_scale": 0,
            "target_": "prev-classes",
        },
        "left-err": {"class": "copy", "from": "left-output-ce/error", "loss": "as_is", "loss_scale": 0},
        "left-loss": {"class": "copy", "from": "left-output-ce", "loss": "as_is"},
        "past-embed": {"activation": None, "class": "linear", "from": ["data:prev-classes"], "n_out": 10},
        "center-output": {"class": "softmax", "from": ["encoder-output", "past-embed"], "n_out": n_out},
        "center-output-ce": {
            "class": "loss",
            "from": "center-output",
            "loss": "as_is",
            "loss_": "ce",
            "loss_scale": 0,
            "target_": "classes",
        },
        "center-err": {"class": "copy", "from": "center-output-ce/error", "loss": "as_is", "loss_scale": 0},
        "center-loss": {"class": "copy", "from": "center-output-ce", "loss": "as_is"},
    }
    print("Layers:", sorted(net_dict.keys()))

    print("Training")
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        # Check defaults available for inference.
        assert network.extern_data.data["data"].available_for_inference
        assert not network.extern_data.data["classes"].available_for_inference
        assert not network.extern_data.data["prev-classes"].available_for_inference
        network.construct_from_dict(net_dict)
        optimizer = tf_compat.v1.train.AdamOptimizer()
        fetches_dict = network.get_fetches_dict(should_train=True, should_eval=True, with_size=True)
        fetches_dict["optim_op"] = optimizer.minimize(network.get_objective())
        feed_dict = make_feed_dict(network.extern_data, same_time=True)
        session.run(tf_compat.v1.global_variables_initializer())
        for step in range(3):
            try:
                results = session.run(fetches_dict, feed_dict=feed_dict)
            except tf.errors.OpError as exc:
                help_on_tf_exception(
                    session=session,
                    exception=exc,
                    fetches=fetches_dict,
                    feed_dict=feed_dict,
                    extern_data=network.extern_data,
                )
                raise
            pprint(results)

    print("Forwarding")
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=False)
        network.construct_from_dict(net_dict)
        # Not sure if we would always expect that all layers with losses are constructed in this case,
        # but this is current behavior.
        # In any case, if we do that, they should infer the available_for_inference correctly.
        assert network.layers["encoder-output"].output.available_for_inference
        assert network.layers["left-output"].output.available_for_inference
        assert not network.layers["past-embed"].output.available_for_inference
        assert not network.layers["center-output"].output.available_for_inference
        assert not network.layers["center-output-ce"].output.available_for_inference
        assert not network.layers["center-loss"].output.available_for_inference
        assert not network.layers["center-err"].output.available_for_inference
        fetches_dict = network.get_fetches_dict(should_train=False, should_eval=False, with_size=True)
        feed_dict = make_feed_dict(network.extern_data, same_time=True)
        session.run(tf_compat.v1.global_variables_initializer())
        results = session.run(fetches_dict, feed_dict=feed_dict)
        pprint(results)

    print("Forwarding with fixed available-for-inference")
    with make_scope() as session:
        config.typed_dict["extern_data"]["prev-classes"]["available_for_inference"] = True
        network = TFNetwork(config=config, train_flag=False)
        network.construct_from_dict(net_dict)
        assert network.layers["encoder-output"].output.available_for_inference
        assert network.layers["left-output"].output.available_for_inference
        assert network.layers["past-embed"].output.available_for_inference
        assert network.layers["center-output"].output.available_for_inference
        assert not network.layers["center-output-ce"].output.available_for_inference
        assert not network.layers["center-loss"].output.available_for_inference
        assert not network.layers["center-err"].output.available_for_inference
        fetches_dict = network.get_fetches_dict(should_train=False, should_eval=False, with_size=True)
        feed_dict = make_feed_dict(network.extern_data, same_time=True)
        session.run(tf_compat.v1.global_variables_initializer())
        results = session.run(fetches_dict, feed_dict=feed_dict)
        pprint(results)


def test_EditDistanceLayer_greedy_ctc_decode():
    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

    input_spatial_dim = SpatialDim("input-spatial")
    targets_spatial_dim = SpatialDim("targets-spatial")
    classes_dim = FeatureDim("classes", 10)
    blank_idx = classes_dim.dimension
    config = Config(
        {
            "extern_data": {
                "logits": {"dim_tags": [batch_dim, input_spatial_dim, classes_dim + 1]},
                "targets": {"dim_tags": [batch_dim, targets_spatial_dim], "sparse_dim": classes_dim},
            },
        }
    )
    net_dict = {
        "argmax": {"class": "reduce", "from": "data:logits", "axis": "F", "mode": "argmax"},
        "ctc_decode": {
            "class": "subnetwork",
            "from": "argmax",
            "subnetwork": {
                # tf_util.sparse_labels_with_seq_lens
                "shift_right": {
                    "class": "shift_axis",
                    "from": "data",
                    "axis": "T",
                    "amount": 1,
                    "pad_value": -1,
                    "adjust_size_info": False,
                },
                "unique_mask": {"class": "compare", "from": ["data", "shift_right"], "kind": "not_equal"},
                "non_blank_mask": {"class": "compare", "from": "data", "kind": "not_equal", "value": blank_idx},
                "mask": {"class": "combine", "kind": "logical_and", "from": ["unique_mask", "non_blank_mask"]},
                "output": {
                    "class": "masked_computation",
                    "from": "data",
                    "mask": "mask",
                    "unit": {"class": "copy", "from": "data"},
                },
            },
        },
        "edit_dist": {"class": "edit_distance", "a": "ctc_decode", "b": "data:targets", "is_output_layer": True},
    }

    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)

        logits = network.extern_data.data["logits"]
        targets = network.extern_data.data["targets"]
        decoded_sparse = tf_util.ctc_greedy_decode(
            logits=logits.get_placeholder_as_time_major(), seq_lens=logits.get_sequence_lengths(), time_major=True
        )
        decoded_dense = tf_compat.v1.sparse_to_dense(
            sparse_indices=decoded_sparse.indices,
            sparse_values=decoded_sparse.values,
            output_shape=decoded_sparse.dense_shape,
        )
        targets_sparse = tf_util.sparse_labels(targets.placeholder, seq_lens=targets.get_sequence_lengths())
        error = tf.edit_distance(
            hypothesis=tf.cast(decoded_sparse, targets_sparse.dtype), truth=targets_sparse, normalize=False
        )

        argmax = network.layers["argmax"].output
        decoded_net = network.layers["ctc_decode"].output
        print("decoded out:", decoded_net)
        error_net = network.layers["edit_dist"].output
        print("error out:", error_net)

        n_batch = 5
        n_logits_time = 20
        n_targets_time = 7
        rnd = numpy.random.RandomState(42)
        targets_np = rnd.randint(low=0, high=classes_dim.dimension, size=(n_batch, n_targets_time))
        targets_sizes = [7, 6, 5, 4, 3]
        logits_np = rnd.uniform(size=(n_batch, n_logits_time, classes_dim.dimension + 1)).astype("float32")
        for b in range(n_batch):
            # Random correct.
            for _ in range(rnd.randint(1, 7)):
                size = rnd.randint(1, 6)
                t = rnd.randint(0, n_logits_time - size + 1)
                logits_np[b, t : t + size, targets_np[b, rnd.randint(0, targets_sizes[b])]] = rnd.uniform(1.0, 2.0)
            # Random repeats.
            for _ in range(rnd.randint(1, 5)):
                size = rnd.randint(1, 6)
                t = rnd.randint(0, n_logits_time - size + 1)
                logits_np[b, t : t + size] = logits_np[b, t : t + 1]
            # Random blanks.
            for _ in range(rnd.randint(1, 5)):
                size = rnd.randint(1, 6)
                t = rnd.randint(0, n_logits_time - size + 1)
                logits_np[b, t : t + size, blank_idx] = rnd.uniform(1.0, 2.0)
        logits_sizes = [20, 19, 18, 17, 16]
        feed_dict = {
            logits.placeholder: logits_np,
            logits.get_sequence_lengths(): logits_sizes,
            targets.placeholder: targets_np,
            targets.get_sequence_lengths(): targets_sizes,
        }
        argmax_np, dec1_np, err1_np, dec2_np, dec2_sizes, err2_np = session.run(
            (
                argmax.placeholder,
                decoded_dense,
                error,
                decoded_net.get_placeholder_as_batch_major(),
                decoded_net.get_sequence_lengths(),
                error_net.placeholder,
            ),
            feed_dict=feed_dict,
        )
        print("targets:", targets_np)
        print("targets sizes:", targets_sizes)
        print("argmax:", argmax_np)
        print("decoded ref:", dec1_np)
        print("error ref:", err1_np)
        print("decoded:", dec2_np)
        print("decoded sizes:", dec2_sizes)
        print("error:", err2_np)
        numpy.testing.assert_array_equal(dec1_np, dec2_np)
        numpy.testing.assert_array_equal(err1_np, err2_np)


def test_param_variational_noise():
    from returnn.tf.util.basic import print_graph_output, find_ops_with_tensor_input

    config = Config(
        {
            "debug_print_layer_output_template": True,
            "param_variational_noise": 0.075,
            "extern_data": {"data": {"dim": 7}},
        }
    )
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        # Do subnetwork by intention, to test when we have multiple variable scopes.
        network.construct_from_dict(
            {
                "output": {
                    "class": "subnetwork",
                    "from": "data:data",
                    "subnetwork": {"output": {"class": "linear", "n_out": 13, "activation": "tanh", "from": "data"}},
                }
            }
        )
        out = network.get_default_output_layer().output.placeholder
        print("output:")
        print_graph_output(out)
        params = network.get_params_list()
        print("params:", params)
        assert len(params) == 2  # weights and bias
        for param in params:
            print("param:", param)
            ops = find_ops_with_tensor_input(param, fetches=out)
            print("param graph:")
            print_graph_output(ops)
            # There can be multiple ops due to gradient checkpointing.
            assert 1 <= len(ops) and all("_variational_noise/" in op.name for op in ops)


def test_param_weight_dropout():
    from returnn.tensor import Dim, batch_dim
    from returnn.tf.util.basic import print_graph_output, find_ops_with_tensor_input
    from returnn.tf.util.gradient_checkpoint import prepare_gradient_checkpointing

    time_dim = Dim(None, name="time")
    feature_dim = Dim(7, name="feature")
    classes_dim = Dim(13, name="classes")

    config = Config(
        {
            "param_dropout": 0.1,
            "extern_data": {
                "data": {
                    "dim_tags": [batch_dim, time_dim, feature_dim],
                    "time_dim_axis": 1,
                    "feature_dim": feature_dim,
                    "dtype": "float32",
                },
                "classes": {"dim_tags": [batch_dim, time_dim], "sparse_dim": classes_dim, "dtype": "int32"},
            },
        }
    )
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        # Do subnetwork by intention, to test when we have multiple variable scopes.
        network.construct_from_dict(
            {
                "output": {
                    "class": "linear",
                    "out_dim": classes_dim,
                    "activation": "softmax",
                    "from": "data",
                    "loss": "ce",
                    "target": "classes",
                }
            }
        )
        loss = network.get_total_loss()

        prepare_gradient_checkpointing()
        opt = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
        opt_op = opt.minimize(loss)
        print("optimizer:")
        print_graph_output(opt_op)

        tf_log_dir = tempfile.mkdtemp()
        print("TF log dir:", tf_log_dir)
        writer = tf_compat.v1.summary.FileWriter(logdir=tf_log_dir, graph=session.graph, session=session)
        params = network.get_params_list()
        print("params:", params)
        assert len(params) == 2  # weights and bias
        for param in params:
            print("param:", param)
            ops = find_ops_with_tensor_input(param, fetches=opt_op)
            print("param graph:")
            print_graph_output(ops)
            # There can be multiple ops due to gradient checkpointing.
            assert (
                1 <= len(ops)
                and all("_weight_dropout/" in op.name or "/ResourceApply" in op.name for op in ops)
                and any("_weight_dropout/" in op.name for op in ops)
            )

        network.initialize_params(session=session)

        run_metadata = tf_compat.v1.RunMetadata()
        run_options = tf_compat.v1.RunOptions(trace_level=tf_compat.v1.RunOptions.FULL_TRACE)
        session.run(
            opt_op, feed_dict=make_feed_dict(network.extern_data), options=run_options, run_metadata=run_metadata
        )
        writer.add_run_metadata(run_metadata, tag="step_0")
        writer.close()
        print("TF log dir:", tf_log_dir)


def test_param_weight_dropout_and_reuse_params():
    from returnn.tensor import Dim, batch_dim
    from returnn.tf.util.basic import print_graph_output, find_ops_with_tensor_input
    from returnn.tf.util.gradient_checkpoint import prepare_gradient_checkpointing

    time_dim = Dim(None, name="time")
    feature_dim = Dim(7, name="feature")
    classes_dim = Dim(13, name="classes")

    config = Config(
        {
            "extern_data": {
                "data": {
                    "dim_tags": [batch_dim, time_dim, feature_dim],
                    "time_dim_axis": 1,
                    "feature_dim": feature_dim,
                    "dtype": "float32",
                },
                "classes": {"dim_tags": [batch_dim, time_dim], "sparse_dim": classes_dim, "dtype": "int32"},
            },
        }
    )
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        # Do subnetwork by intention, to test when we have multiple variable scopes.
        network.construct_from_dict(
            {
                "a": {
                    "class": "linear",
                    "out_dim": classes_dim,
                    "activation": "softmax",
                    "from": "data",
                    "loss": "ce",
                    "target": "classes",
                    "param_dropout": 0.1,
                },
                "b": {
                    "class": "linear",
                    "out_dim": classes_dim,
                    "activation": "softmax",
                    "from": "data",
                    "reuse_params": "a",
                    "loss": "ce",
                    "target": "classes",
                    "param_dropout": 0.1,  # test that it works to use it again
                },
                "c": {
                    "class": "linear",
                    "out_dim": classes_dim,
                    "activation": "softmax",
                    "from": "data",
                    "reuse_params": "a",
                    "loss": "ce",
                    "target": "classes",
                    # test that it works without
                },
            }
        )
        loss = network.get_total_loss()

        prepare_gradient_checkpointing()
        opt = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
        opt_op = opt.minimize(loss)
        print("optimizer:")
        print_graph_output(opt_op)

        params = network.get_params_list()
        print("params:", params)
        assert len(params) == 2  # weights and bias
        for param in params:
            print("param:", param)
            ops = find_ops_with_tensor_input(param, fetches=opt_op)
            print("param graph:")
            print_graph_output(ops)
            # There can be multiple ops due to gradient checkpointing.
            assert (
                1 <= len(ops)
                and all(
                    "_weight_dropout/" in op.name
                    or "/ResourceApply" in op.name
                    or op.name.startswith("c/")
                    or "/c/" in op.name
                    for op in ops
                )
                and any("_weight_dropout/" in op.name for op in ops)
            )

        network.initialize_params(session=session)
        session.run(opt_op, feed_dict=make_feed_dict(network.extern_data))


def test_param_weight_dropout_and_variational_noise():
    from returnn.tensor import Dim, batch_dim
    from returnn.tf.util.basic import print_graph_output, find_ops_with_tensor_input
    from returnn.tf.util.gradient_checkpoint import prepare_gradient_checkpointing

    time_dim = Dim(None, name="time")
    feature_dim = Dim(7, name="feature")
    classes_dim = Dim(13, name="classes")

    config = Config(
        {
            "param_dropout": 0.1,
            "param_variational_noise": 0.075,
            "extern_data": {
                "data": {
                    "dim_tags": [batch_dim, time_dim, feature_dim],
                    "time_dim_axis": 1,
                    "feature_dim": feature_dim,
                    "dtype": "float32",
                },
                "classes": {"dim_tags": [batch_dim, time_dim], "sparse_dim": classes_dim, "dtype": "int32"},
            },
        }
    )
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        # Do subnetwork by intention, to test when we have multiple variable scopes.
        network.construct_from_dict(
            {
                "output": {
                    "class": "linear",
                    "out_dim": classes_dim,
                    "activation": "softmax",
                    "from": "data",
                    "loss": "ce",
                    "target": "classes",
                }
            }
        )
        loss = network.get_total_loss()

        prepare_gradient_checkpointing()
        opt = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
        opt_op = opt.minimize(loss)
        print("optimizer:")
        print_graph_output(opt_op)

        tf_log_dir = tempfile.mkdtemp()
        print("TF log dir:", tf_log_dir)
        writer = tf_compat.v1.summary.FileWriter(logdir=tf_log_dir, graph=session.graph, session=session)
        params = network.get_params_list()
        print("params:", params)
        assert len(params) == 2  # weights and bias
        for param in params:
            print("param:", param)
            ops = find_ops_with_tensor_input(param, fetches=opt_op)
            print("param graph:")
            print_graph_output(ops)
            # There can be multiple ops due to gradient checkpointing.
            assert (
                1 <= len(ops)
                and all("_variational_noise/" in op.name or "/ResourceApply" in op.name for op in ops)
                and any("_variational_noise/" in op.name for op in ops)
            ), f"ops: {ops}"

        layer = network.layers["output"]
        assert isinstance(layer, LinearLayer)
        print("weights:", layer.weights)
        assert layer.weights.name.startswith("output/W_weight_dropout/")

        network.initialize_params(session=session)

        run_metadata = tf_compat.v1.RunMetadata()
        run_options = tf_compat.v1.RunOptions(trace_level=tf_compat.v1.RunOptions.FULL_TRACE)
        session.run(
            opt_op, feed_dict=make_feed_dict(network.extern_data), options=run_options, run_metadata=run_metadata
        )
        writer.add_run_metadata(run_metadata, tag="step_0")
        writer.close()
        print("TF log dir:", tf_log_dir)


def test_LinearLayer_simple_train():
    config = Config()
    n_in, n_out = 7, 3
    config.update(
        {
            "extern_data": {
                "data": (n_in, 2),
                "classes": (n_out, 1),
            },
            "debug_print_layer_output_template": True,
        }
    )
    print("Creating network...")
    with tf.Graph().as_default():
        network = TFNetwork(config=config, train_flag=True)

        net_dict = {}
        layer_n_out = 10
        layer_common_args = {"class": "linear", "activation": "relu", "n_out": layer_n_out, "L2": 0.01}

        def layer(sources, **kwargs):
            args = kwargs.copy()
            for k, v in layer_common_args.items():
                args.setdefault(k, v)
            args.setdefault("from", sources)
            return args

        def make_network(num_layers):
            sources = ["data"]
            for i in range(num_layers):
                net_dict["layer%i" % i] = layer(sources=sources)
                sources = ["layer%i" % i]
            net_dict["output"] = {"class": "softmax", "loss": "ce", "from": sources}

        make_network(num_layers=3)
        network.construct_from_dict(net_dict)
        data_input = network.extern_data.get_default_input_data()
        data_target = network.extern_data.get_default_target_data()
        optimizer = tf_compat.v1.train.AdamOptimizer()
        network.maybe_construct_objective()
        update_op = optimizer.minimize(network.get_objective())
        n_batch = 5
        n_time = 11
        rnd = numpy.random.RandomState(42)
        with tf_compat.v1.Session() as session:
            session.run(tf_compat.v1.global_variables_initializer())
            for step in range(5):
                info, _ = session.run(
                    (network.get_fetches_dict(), update_op),
                    feed_dict={
                        data_input.placeholder: rnd.normal(size=(n_batch, n_time, n_in)).astype("float32"),
                        data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
                        data_target.placeholder: rnd.randint(0, n_out, size=(n_batch, n_time), dtype="int32"),
                        data_target.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
                    },
                )
                print("step:", step, "info:", info)


def test_flat_net_construction():
    config = Config()
    n_in, n_out = 7, 3
    config.update(
        {
            "extern_data": {
                "data": (n_in, 2),
                "classes": (n_out, 1),
            },
            "flat_net_construction": True,
            "debug_print_layer_output_template": True,
        }
    )
    print("Creating network...")
    with tf.Graph().as_default():
        network = TFNetwork(config=config, train_flag=True)

        net_dict = {
            "pre0": {"class": "linear", "activation": "tanh", "from": "data", "n_out": 10},
            "pre1": {"class": "linear", "activation": "tanh", "from": "pre0", "n_out": 10},
            "pre2": {"class": "linear", "activation": "tanh", "from": "pre1", "n_out": 10},
        }
        layer_common_args = {"class": "copy"}

        def layer(sources, **kwargs):
            args = kwargs.copy()
            for k, v in layer_common_args.items():
                args.setdefault(k, v)
            args.setdefault("from", sources)
            return args

        def make_network(num_layers):
            sources = ["pre2"]
            for i in range(num_layers):
                net_dict["layer%i" % i] = layer(sources=sources)
                sources = ["layer%i" % i]
            net_dict["output"] = {"class": "softmax", "loss": "ce", "from": sources}

        make_network(num_layers=5000)
        network.construct_from_dict(net_dict)
        data_input = network.extern_data.get_default_input_data()
        data_target = network.extern_data.get_default_target_data()
        optimizer = tf_compat.v1.train.AdamOptimizer()
        network.maybe_construct_objective()
        update_op = optimizer.minimize(network.get_objective())
        n_batch = 5
        n_time = 11
        rnd = numpy.random.RandomState(42)
        with tf_compat.v1.Session() as session:
            session.run(tf_compat.v1.global_variables_initializer())
            for step in range(5):
                info, _ = session.run(
                    (network.get_fetches_dict(), update_op),
                    feed_dict={
                        data_input.placeholder: rnd.normal(size=(n_batch, n_time, n_in)).astype("float32"),
                        data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
                        data_target.placeholder: rnd.randint(0, n_out, size=(n_batch, n_time), dtype="int32"),
                        data_target.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
                    },
                )
                print("step:", step, "info:", info)


def test_SyntheticGradientLayer():
    """
    Tests :class:`SyntheticGradientLayer`.
    """
    config = Config()
    n_in, n_out = 7, 3
    config.update(
        {
            "extern_data": {
                "data": (n_in, 2),
                "classes": (n_out, 1),
            },
            "debug_print_layer_output_template": True,
        }
    )
    print("Creating network...")
    with tf.Graph().as_default():
        network = TFNetwork(config=config, train_flag=True)

        net_dict = {}
        layer_n_out = 10
        layer_common_args = {"class": "linear", "activation": "relu", "n_out": layer_n_out, "L2": 0.01}

        def layer(sources, **kwargs):
            args = kwargs.copy()
            for k, v in layer_common_args.items():
                args.setdefault(k, v)
            args.setdefault("from", sources)
            return args

        def make_network(num_layers):
            sources = ["data"]
            for i in range(num_layers):
                net_dict["layer%i" % i] = layer(sources=sources)
                sources = ["layer%i" % i]
                net_dict["predict_grad%i" % i] = layer(sources=sources)
                net_dict["syn_grad%i" % i] = {
                    "class": "synthetic_gradient",
                    "gradient": "predict_grad%i" % i,
                    "from": sources,
                }
                sources = ["syn_grad%i" % i]
            net_dict["output"] = {"class": "softmax", "loss": "ce", "from": sources}

        make_network(num_layers=3)
        network.construct_from_dict(net_dict)
        data_input = network.extern_data.get_default_input_data()
        data_target = network.extern_data.get_default_target_data()
        from returnn.tf.updater import Updater

        updater = Updater(config=config, network=network, initial_learning_rate=0.001)
        updater.set_trainable_vars(tf_compat.v1.trainable_variables())
        update_op = updater.get_optim_op()
        assert updater.optim_meta_losses_dict
        fetches = network.get_fetches_dict()
        fetches.update(updater.optim_meta_losses_dict)

        n_batch = 5
        n_time = 11
        rnd = numpy.random.RandomState(42)
        with tf_compat.v1.Session() as session:
            session.run(
                tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step_var])
            )
            for step in range(5):
                info, _ = session.run(
                    (fetches, update_op),
                    feed_dict={
                        data_input.placeholder: rnd.normal(size=(n_batch, n_time, n_in)).astype("float32"),
                        data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
                        data_target.placeholder: rnd.randint(0, n_out, size=(n_batch, n_time), dtype="int32"),
                        data_target.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
                    },
                )
                print("step:", step, "info:", info)


def test_TikhonovRegularizationLayer():
    """
    Tests :class:`TikhonovRegularizationLayer`.
    """
    config = Config()
    n_in, n_out = 7, 3
    config.update(
        {
            "extern_data": {
                "data": (n_in, 2),
                "classes": (n_out, 1),
            },
            "debug_print_layer_output_template": True,
        }
    )
    print("Creating network...")
    with tf.Graph().as_default():
        network = TFNetwork(config=config, train_flag=True)

        net_dict = {}
        layer_n_out = 10
        layer_common_args = {"class": "linear", "activation": "relu", "n_out": layer_n_out, "L2": 0.01}

        def layer(sources, **kwargs):
            args = kwargs.copy()
            for k, v in layer_common_args.items():
                args.setdefault(k, v)
            args.setdefault("from", sources)
            return args

        def make_network(num_layers):
            net_dict["input"] = {"class": "tikhonov_regularization", "meta_loss_scale": 0.1, "from": "data"}
            sources = ["input"]
            for i in range(num_layers):
                net_dict["layer%i" % i] = layer(sources=sources)
                sources = ["layer%i" % i]
            net_dict["output"] = {"class": "softmax", "loss": "ce", "loss_opts": {"use_fused": False}, "from": sources}

        make_network(num_layers=3)
        network.construct_from_dict(net_dict)
        data_input = network.extern_data.get_default_input_data()
        data_target = network.extern_data.get_default_target_data()
        from returnn.tf.updater import Updater

        updater = Updater(config=config, network=network, initial_learning_rate=0.001)
        updater.set_trainable_vars(tf_compat.v1.trainable_variables())
        update_op = updater.get_optim_op()
        assert updater.optim_meta_losses_dict
        fetches = network.get_fetches_dict()
        fetches.update(updater.optim_meta_losses_dict)

        n_batch = 5
        n_time = 11
        rnd = numpy.random.RandomState(42)
        with tf_compat.v1.Session() as session:
            session.run(
                tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step_var])
            )
            for step in range(5):
                info, _ = session.run(
                    (fetches, update_op),
                    feed_dict={
                        data_input.placeholder: rnd.normal(size=(n_batch, n_time, n_in)).astype("float32"),
                        data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
                        data_target.placeholder: rnd.randint(0, n_out, size=(n_batch, n_time), dtype="int32"),
                        data_target.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
                    },
                )
                print("step:", step, "info:", info)


def test_split_info_input():
    from returnn.tf.util.basic import print_graph_output, find_ops_with_tensor_input

    config = Config({"debug_print_layer_output_template": True, "extern_data": {"data": {"dim": 7}}})
    net_dict = {
        "a": {"class": "linear", "activation": "tanh", "n_out": 11, "from": "data:data"},
        "b": {"class": "linear", "activation": "tanh", "n_out": 13, "from": "data:data"},
        "concat": {"class": "copy", "from": ["a", "b"]},
        "output": {"class": "linear", "activation": None, "with_bias": True, "from": ["concat"], "n_out": 17},
    }
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        out_weights = network.get_default_output_layer().params["W"]
        print("out_weights:", out_weights)
        assert isinstance(out_weights, tf.Variable)
        assert out_weights.get_shape().as_list() == [11 + 13, 17]
        # TODO: multiple checks:
        # the split info itself
        # the param init handling...
        # actually, for param init handling, input dim splits do not matter.
        # they matter just for copying/growing-pretrain.
        # for param init handling, output dim split do matter.


def test_VariableLayer_split_info():
    feat1 = FeatureDim("feature1", 3)
    feat2 = FeatureDim("feature2", 5)
    net_dict = {"output": {"class": "variable", "shape": [2 * feat1 + feat2, 3 * feat1]}}
    with make_scope() as session:
        network = TFNetwork(extern_data=ExternData())
        network.construct_from_dict(net_dict)
        layer = network.layers["output"]
        assert isinstance(layer, VariableLayer)
        assert tf_util.get_param_axes_split_info(layer.output.placeholder) == [
            2 * [feat1.dimension] + [feat2.dimension],
            3 * [feat1.dimension],
        ]


def test_VariableLayer_init_by_layer():
    # https://github.com/rwth-i6/returnn/wiki/Parameter-initialization
    # https://pytorch.org/docs/1.9.1/_modules/torch/nn/modules/linear.html#Linear
    # https://pytorch.org/docs/1.9.1/nn.init.html
    # https://github.com/pytorch/pytorch/blob/d665097cad3207795a655bbdde7a4123c0adc1c3/torch/nn/modules/linear.py#L96
    # https://github.com/pytorch/pytorch/issues/57109
    # https://github.com/pytorch/pytorch/blob/77721ee318d6785010144aa4569efb98199e7162/torch/nn/init.py#L390-L395
    # https://github.com/pytorch/pytorch/pull/41638
    # https://github.com/pytorch/pytorch/issues/18182
    """
    PyTorch new / proposed Linear weight:
      kaiming_normal_(mode='fan_out')  # default nonlinearity='leaky_relu', default neg slope 0.01
      (ignoring neg slope) -> std = sqrt(2 / fan_out)

    PyTorch old / current Linear weight:
      kaiming_uniform_(self.weight, a=math.sqrt(5))  # default nonlinearity='leaky_relu', mode='fan_in'
      -> std = sqrt(1 / (3 * fan_in))  -> bound = sqrt(1/fan_in)

    For uniform distribution, bound = sqrt(3) * std.
    For truncated normal distribution: stddev /= .87962566103423978  # TF VarianceScaling, scipy a=-2, b=2 ...
    """
    shape = (3, 4)
    net_dict = {
        "random": {"class": "random", "shape": shape, "distribution": "truncated_normal", "static": True},
        "var": {"class": "variable", "shape": shape, "init_by_layer": "random"},
        "output": {"class": "copy", "from": "var"},
    }
    config = Config(
        {
            "extern_data": {"data": {"dim": 4}},  # not actually used...
        }
    )
    tf_rnd_seed = 42
    with make_scope() as session:
        tf_compat.v1.set_random_seed(tf_rnd_seed)
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        assert net.layers["random"].params == {}
        assert net.get_params_list() == [next(iter(net.layers["var"].params.values()))]
        net.initialize_params(session)
        var_v = session.run(net.layers["var"].output.placeholder)
    # Run again to check that it is deterministic.
    with make_scope() as session:
        tf_compat.v1.set_random_seed(tf_rnd_seed)
        net = TFNetwork(config=config)
        net.construct_from_dict(net_dict)
        net.initialize_params(session)
        var_v_ = session.run(net.layers["var"].output.placeholder)
    numpy.testing.assert_array_equal(var_v, var_v_)


def test_extra1():
    n_in, n_out = 2, 3
    config = Config(
        {
            "extern_data": {"data": {"dim": n_in}},
            "debug_print_layer_output_template": True,
        }
    )
    net_dict = {
        "input": {"class": "linear", "activation": "relu", "n_out": n_out, "from": "data"},
        "extra.2:input": {"class": "linear", "activation": None, "n_out": n_out, "from": "data"},
        # "extra.3:input automatically ...
        "output1": {"class": "copy", "from": "input", "is_output_layer": True},
        "output2": {"class": "activation", "from": "extra.2:input", "activation": "relu", "is_output_layer": True},
        "output3": {"class": "copy", "from": "extra.3:input", "is_output_layer": True},
    }
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)

        assert "extra.2" in network.extra_nets
        assert "extra.3" in network.extra_nets
        params = network.get_params_list()
        print("Params:", params)
        assert len(params) == 2  # W + b

        feed_dict = make_feed_dict(network.extern_data)
        session.run(
            tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step_var])
        )
        out1 = session.run(network.layers["output1"].output.placeholder, feed_dict=feed_dict)
        out2 = session.run(network.layers["output2"].output.placeholder, feed_dict=feed_dict)
        out3 = session.run(network.layers["output3"].output.placeholder, feed_dict=feed_dict)
        numpy.testing.assert_almost_equal(out1, out2)
        numpy.testing.assert_almost_equal(out2, out3)


def test_extra_subnet():
    n_in, n_out = 3, 3
    config = Config(
        {
            "extern_data": {"data": {"dim": n_in}},
            "debug_print_layer_output_template": True,
        }
    )
    net_dict = {
        "subnet": {
            "class": "subnetwork",
            "from": "data:data",
            "subnetwork": {
                "output": {"class": "linear", "activation": "relu", "n_out": n_out, "from": "data"},
                "output2": {
                    "class": "linear",
                    "activation": "relu",
                    "n_out": n_out,
                    "from": "data",
                    "is_output_layer": True,
                },
            },
        },
        "extra.2:subnet": {
            "class": "subnetwork",
            "from": "data:data",
            "subnetwork": {
                "output": {"class": "copy", "from": "data"},
                "output2": {
                    "class": "linear",
                    "from": "data:data",
                    "activation": None,
                    "n_out": n_out,
                    "is_output_layer": True,
                },
            },
        },
        # extra.3:subnet automatically
        "sub1_output1": {"class": "copy", "from": "subnet/output", "is_output_layer": True},
        "sub1_output2": {"class": "copy", "from": "subnet/output2", "is_output_layer": True},
        "sub2_output1": {"class": "copy", "from": "extra.2:subnet/output", "is_output_layer": True},
        "sub2_output2": {
            "class": "activation",
            "activation": "relu",
            "from": "extra.2:subnet/output2",
            "is_output_layer": True,
        },
        "sub3_output1": {"class": "copy", "from": "extra.3:subnet/output", "is_output_layer": True},
        "sub3_output2": {"class": "copy", "from": "extra.3:subnet/output2", "is_output_layer": True},
    }
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)
        assert "extra.2" in network.extra_nets
        assert "extra.3" in network.extra_nets
        params = network.get_params_list()
        print("Params:", params)
        assert len(params) == 4

        feed_dict = make_feed_dict(network.extern_data)
        session.run(
            tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step_var])
        )
        in_ = feed_dict[network.extern_data.data["data"].placeholder]
        sub1_out1 = session.run(network.layers["sub1_output1"].output.placeholder, feed_dict=feed_dict)
        sub1_out2 = session.run(network.layers["sub1_output2"].output.placeholder, feed_dict=feed_dict)
        sub2_out1 = session.run(network.layers["sub2_output1"].output.placeholder, feed_dict=feed_dict)
        sub2_out2 = session.run(network.layers["sub2_output2"].output.placeholder, feed_dict=feed_dict)
        sub3_out1 = session.run(network.layers["sub3_output1"].output.placeholder, feed_dict=feed_dict)
        sub3_out2 = session.run(network.layers["sub3_output2"].output.placeholder, feed_dict=feed_dict)
        numpy.testing.assert_almost_equal(sub1_out1, sub3_out1)
        numpy.testing.assert_almost_equal(sub1_out2, sub3_out2)
        numpy.testing.assert_almost_equal(sub1_out2, sub2_out2)
        numpy.testing.assert_almost_equal(in_, sub2_out1)


def test_subnetwork_unused_output():
    # https://github.com/rwth-i6/returnn/issues/580
    with make_scope() as session:
        net_dict = {
            "sub": {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "linear": {"class": "linear", "from": "base:data:data", "n_out": 1},
                    "linear_0": {"class": "linear", "from": "linear", "n_out": 1},
                    "output": {"class": "copy", "from": "linear"},
                },
            },
            "linear": {"class": "linear", "from": ["sub/linear", "sub/linear_0"], "n_out": 1},
            "output": {"class": "copy", "from": "linear"},
        }
        config = Config()
        config.update(dict(num_inputs=1, num_outputs=1))
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)


def test_subnetwork_deep_stack():
    # https://github.com/rwth-i6/returnn/issues/993
    with make_scope() as session:
        import better_exchook
        import traceback
        import types
        import collections

        max_depth = 5

        def _find_net_construct_from_dict_in_stack(stack):
            found_last = None
            for i, frame in enumerate(stack):
                assert isinstance(frame, types.FrameType)
                func = better_exchook.get_func_from_code_object(frame.f_code)
                if func == TFNetwork.construct_from_dict:
                    found_last = i
            if found_last is not None:
                return found_last
            raise Exception("construct_from_dict not found in stack")

        def _extract_stack():
            stack = [frame for frame, _ in traceback.walk_stack(None)]
            stack = stack[: _find_net_construct_from_dict_in_stack(stack) + 1]
            return stack

        def _top_eval(source, **_):
            print("Top eval func stack:")
            stack = _extract_stack()
            # LayerBase.transform_config_dict would only increase if this is not flat net construction,
            # see https://github.com/rwth-i6/returnn/issues/992.
            # For reference, output without fixes: https://gist.github.com/albertz/a1a20710e4d08292e35ba6910ca132e8
            func_counter = collections.Counter()
            for frame in stack:
                assert isinstance(frame, types.FrameType)
                func = better_exchook.get_func_from_code_object(frame.f_code)
                if not func:
                    print("  Warning: unexpected code object: %s" % frame.f_code.co_name)
                    continue
                func_counter[func] += 1
                print(" ", func.__qualname__)
            print("Stack depth:", len(stack))
            print("Num functions:", len(func_counter))
            for func, count in func_counter.most_common(10):
                print("Most common func %s: count %i" % (func.__qualname__, count))
            """
      For reference, without flat construction (#992), before the fix of #993, I get:
        Most common func Subnetwork.get_sub_layer_func.<locals>.wrapped_get_layer: count 30
        Most common func Subnetwork.get_layer_func.<locals>.wrapped_get_layer: count 30
        Most common func TFNetwork.construct_layer: count 12
        Most common func TFNetwork.construct_layer.<locals>.get_layer: count 11
        Most common func TFNetwork.add_layer: count 6
        Most common func LayerBase.transform_config_dict.<locals>.<listcomp>: count 6
        Most common func LayerBase.transform_config_dict: count 6
        Most common func CopyLayer.transform_config_dict: count 6
        Most common func Subnetwork.construct_layer: count 5
        Most common func SubnetworkLayer.transform_config_dict: count 5
      """
            print("Num TFNetwork.construct_layer:", func_counter[TFNetwork.construct_layer])
            print("Num LayerBase.transform_config_dict:", func_counter[LayerBase.transform_config_dict.__func__])
            # Before the fix, the stack depth was 125; with the improved variant (even not flat) it is 65.
            # We don't check for the exact number to allow for some variation of future changes.
            # However, we want to avoid that it becomes too deep in any case.
            assert len(stack) <= 70
            return source(0)

        def _create_subnet_layer_dict(depth):
            if depth >= max_depth:
                return {"class": "eval", "from": "base:" * depth + "data", "eval": _top_eval}
            return {
                "class": "subnetwork",
                "from": [],
                "subnetwork": {
                    "sub%i" % depth: _create_subnet_layer_dict(depth + 1),
                    "output": {"class": "copy", "from": "sub%i" % depth},
                },
            }

        net_dict = {"sub": _create_subnet_layer_dict(0), "output": {"class": "copy", "from": "sub"}}
        config = Config({"extern_data": {"data": {"dim": 3}}})
        network = TFNetwork(config=config)
        network.construct_from_dict(net_dict)


def test_subnet_construct_layer():
    # https://github.com/rwth-i6/returnn/issues/1014
    with make_scope() as session:
        net = TFNetwork(config=Config({"extern_data": {"data": {"dim": 1}}}))
        subnet_layer = net.construct_layer(
            name="subnet",
            net_dict={
                "subnet": {
                    "class": "subnetwork",
                    "from": "data",
                    "subnetwork": {"output": {"class": "copy", "from": "data"}},
                }
            },
        )
        assert isinstance(subnet_layer, SubnetworkLayer)
        assert subnet_layer.output.placeholder is net.extern_data.get_default_input_data().placeholder


def test_extra_search():
    class Callbacks:
        history = []

        @classmethod
        def callback(cls, self, source, **kwargs):
            """
            :param LayerBase self:
            :param (int)->tf.Tensor source:
            :rtype: tf.Tensor
            """
            print(
                "test_extra_search, callback: %r, %r; search flag %r"
                % (self.network.name, self, self.network.search_flag)
            )
            cls.history.append(self)
            return source(0)

    n_batch, n_time, n_in, n_out = 2, 3, 7, 11
    rnd = numpy.random.RandomState(42)
    config = Config({"debug_print_layer_output_template": True, "extern_data": {"data": {"dim": n_in}}})
    net_dict = {
        "input": {"class": "eval", "eval": Callbacks.callback, "from": "search_post_output"},
        "extra.search:input": {"class": "eval", "eval": Callbacks.callback, "from": "data"},
        # Note: This 'output' layer is created twice: Once in main net, once in extra-net.
        "output": {
            "class": "subnetwork",
            "from": "input",
            "subnetwork": {
                "inner": {"class": "linear", "from": "data", "activation": "relu", "n_out": n_out},
                "output": {"class": "eval", "from": "inner", "eval": Callbacks.callback},
            },
        },
        "search_post_output": {"class": "linear", "from": "extra.search:output", "activation": "relu", "n_out": n_in},
    }
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(net_dict)

        assert not network.search_flag
        assert "extra.search" in network.extra_nets
        extra_net = network.extra_nets["extra.search"]
        assert extra_net.search_flag
        assert "input" in network.layers
        assert "input" in extra_net.layers
        assert "output" in network.layers
        assert "output" in extra_net.layers
        layer_input = network.layers["input"]
        assert isinstance(layer_input, EvalLayer)
        assert layer_input in Callbacks.history
        assert layer_input.network is network
        layer_extra_input = extra_net.layers["input"]
        assert isinstance(layer_extra_input, EvalLayer)
        assert layer_extra_input in Callbacks.history
        assert layer_extra_input.network is extra_net
        layer_output = network.layers["output"]
        assert isinstance(layer_output, SubnetworkLayer)
        assert layer_output.network is network
        layer_output_output = layer_output.subnetwork.layers["output"]
        assert layer_output_output in Callbacks.history
        layer_extra_output = extra_net.layers["output"]
        assert isinstance(layer_extra_output, SubnetworkLayer)
        assert layer_extra_output.network is extra_net
        layer_extra_output_output = layer_extra_output.subnetwork.layers["output"]
        assert layer_extra_output_output in Callbacks.history

        fetches = network.get_fetches_dict()
        data_input = network.extern_data.data["data"]

        session.run(
            tf_compat.v1.variables_initializer(tf_compat.v1.global_variables() + [network.global_train_step_var])
        )
        info, out = session.run(
            (fetches, layer_output.output.placeholder),
            feed_dict={
                data_input.placeholder: rnd.normal(size=(n_batch, n_time, n_in)).astype("float32"),
                data_input.size_placeholder[0]: numpy.array([n_time] * n_batch, dtype="int32"),
            },
        )
        print(info)
        print(out)  # random...


def test_HDFDumpLayer():
    import os
    from test_HDFDataset import get_test_tmp_file, DatasetTestReader, HDFDataset

    hdf_filename = get_test_tmp_file(".hdf")
    os.remove(hdf_filename)  # HDFDumpLayer expects that the file does not exist

    with make_scope() as session:
        n_in, n_out = 4, 3
        config = Config()
        config.update(
            {
                "num_outputs": n_out,
                "num_inputs": n_in,
                "network": {
                    "lstm": {"class": "rec", "unit": "LSTMBlock", "from": ["data"], "n_out": n_out},
                    "dump": {"class": "hdf_dump", "filename": hdf_filename, "from": ["lstm"]},
                    "output": {"class": "copy", "from": ["dump"]},
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_value("network"))

        session.run(tf_compat.v1.global_variables_initializer())
        out = network.layers["output"].output.placeholder
        n_batch = 1
        seq_len = 4
        input_data = numpy.array(
            [[[1, -0.2, 0.3, -4], [2, -0.6, 0.7, -1.8], [1, 0.3, -0.1, -0.8], [0.1, -0.2, 0.2, 0.8]]], dtype="float32"
        )
        input_tags = numpy.array([b"seq-0"], dtype="S5")
        seq_lens = numpy.array([seq_len], dtype="int32")
        assert input_data.shape == (n_batch, seq_lens[0], n_in)
        feed = {
            network.extern_data.data["data"].placeholder: input_data,
            network.extern_data.data["data"].size_placeholder[0]: seq_lens,
            network.extern_data.data["seq_tag"].placeholder: input_tags,
        }
        assert feed[network.extern_data.get_default_input_data().placeholder].shape == (n_batch, seq_len, n_in)
        session.run([out, network.get_post_control_dependencies()], feed_dict=feed)

        network.call_graph_reset_callbacks()

    assert os.path.exists(hdf_filename)
    reader = DatasetTestReader(HDFDataset([hdf_filename]))
    reader.read_all()
    assert reader.num_seqs == 1
    assert reader.seq_tags == ["seq-0"]
    assert reader.seq_lens[0]["data"] == seq_lens[0]
    assert reader.data["data"][0].shape == (seq_lens[0], n_out)


def test_HDFDumpLayer_sparse():
    import os
    from test_HDFDataset import get_test_tmp_file, DatasetTestReader, HDFDataset

    hdf_filename = get_test_tmp_file(".hdf")
    os.remove(hdf_filename)  # HDFDumpLayer expects that the file does not exist

    with make_scope() as session:
        n_in, n_out = 4, 5
        config = Config()
        config.update(
            {
                "num_inputs": n_in,
                "num_outputs": n_out,
                "network": {
                    "dump": {
                        "class": "hdf_dump",
                        "filename": hdf_filename,
                        "from": "data:classes",
                        "is_output_layer": True,
                    },
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_value("network"))

        session.run(tf_compat.v1.global_variables_initializer())
        n_batch = 1
        classes_data = numpy.array([[2, 5, 6]], dtype="int32")
        classes_seq_lens = [classes_data.shape[1]]
        assert classes_data.shape == (n_batch, classes_seq_lens[0])
        input_tags = numpy.array([b"seq-0"], dtype="S5")
        feed = {
            network.extern_data.data["classes"].placeholder: classes_data,
            network.extern_data.data["classes"].size_placeholder[0]: classes_seq_lens,
            network.extern_data.data["seq_tag"].placeholder: input_tags,
        }
        session.run(network.get_fetches_dict(), feed_dict=feed)

        network.call_graph_reset_callbacks()

    assert os.path.exists(hdf_filename)
    reader = DatasetTestReader(HDFDataset([hdf_filename]))
    reader.read_all()
    assert reader.num_seqs == 1
    assert reader.seq_tags == ["seq-0"]
    assert reader.seq_lens[0]["data"] == classes_seq_lens[0]
    assert reader.data["data"][0].shape == (classes_seq_lens[0],)
    assert reader.data_sparse["data"] == True
    assert reader.dataset.get_data_dim("data") == n_out


def test_HDFDumpLayer_fixed_length():
    import os
    from test_HDFDataset import get_test_tmp_file, DatasetTestReader, HDFDataset

    hdf_filename = get_test_tmp_file(".hdf")
    os.remove(hdf_filename)  # HDFDumpLayer expects that the file does not exist

    with make_scope() as session:
        n_in, n_out = 4, 3
        config = Config()
        config.update(
            {
                "num_outputs": n_out,
                "num_inputs": n_in,
                "network": {
                    "lstm": {"class": "rec", "unit": "LSTMBlock", "from": ["data"], "n_out": n_out},
                    "last_state": {"class": "get_last_hidden_state", "from": ["lstm"], "key": "h", "n_out": n_out},
                    "last_state_expanded": {"class": "expand_dims", "from": ["last_state"], "axis": "T"},
                    "dump": {"class": "hdf_dump", "filename": hdf_filename, "from": ["last_state_expanded"]},
                    "output": {"class": "copy", "from": ["dump"]},
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_value("network"))

        session.run(tf_compat.v1.global_variables_initializer())
        out = network.layers["output"].output.placeholder
        n_batch = 1
        seq_len = 4
        input_data = numpy.array(
            [[[1, -0.2, 0.3, -4], [2, -0.6, 0.7, -1.8], [1, 0.3, -0.1, -0.8], [0.1, -0.2, 0.2, 0.8]]], dtype="float32"
        )
        input_tags = numpy.array([b"seq-0"], dtype="S5")
        seq_lens = numpy.array([seq_len], dtype="int32")
        assert input_data.shape == (n_batch, seq_lens[0], n_in)
        feed = {
            network.extern_data.data["data"].placeholder: input_data,
            network.extern_data.data["data"].size_placeholder[0]: seq_lens,
            network.extern_data.data["seq_tag"].placeholder: input_tags,
        }
        session.run([out, network.get_post_control_dependencies()], feed_dict=feed)

        network.call_graph_reset_callbacks()

    assert os.path.exists(hdf_filename)
    reader = DatasetTestReader(HDFDataset([hdf_filename]))
    reader.read_all()
    assert reader.num_seqs == 1
    assert reader.seq_tags == ["seq-0"]
    assert reader.seq_lens[0]["data"] == 1
    assert reader.data["data"][0].shape == (1, n_out)


def test_HDFDumpLayer_extra():
    import os
    from test_HDFDataset import get_test_tmp_file, DatasetTestReader, HDFDataset

    hdf_filename = get_test_tmp_file(".hdf")
    os.remove(hdf_filename)  # HDFDumpLayer expects that the file does not exist

    with make_scope() as session:
        n_in = 5
        n_out1 = 7
        config = Config()
        config.update(
            {
                "extern_data": {
                    "data": {"dim": n_in},
                    "classes1": {"dim": n_out1, "sparse": True},
                    "classes2": {"dim": None, "dtype": "float32", "shape": ()},
                },
                "network": {
                    "dump": {
                        "class": "hdf_dump",
                        "filename": hdf_filename,
                        "from": "data",
                        "extra": {"classes1": "data:classes1", "classes2": "data:classes2"},
                        "is_output_layer": True,
                    },
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_value("network"))
        network.print_network_info()

        session.run(tf_compat.v1.global_variables_initializer())
        n_batch = 1
        input_data = numpy.array(
            [
                [
                    [1, -0.2, 0.3, -4, 5],
                    [2, -0.6, 0.7, -1.8, 2.9],
                    [1, 0.3, -0.1, -0.8, 0.5],
                    [0.1, -0.2, 0.2, 0.8, -0.3],
                ]
            ],
            dtype="float32",
        )
        input_seq_lens = [input_data.shape[1]]
        assert input_data.shape == (n_batch, input_seq_lens[0], n_in)
        classes1_data = numpy.array([[2, 5, 6]], dtype="int32")
        classes1_seq_lens = [classes1_data.shape[1]]
        assert classes1_data.shape == (n_batch, classes1_seq_lens[0])
        classes2_data = numpy.array([-7.89], dtype="float32")
        assert classes2_data.shape == (n_batch,)
        seq_tags = numpy.array([b"seq-0"], dtype="S5")
        feed = {
            network.extern_data.data["data"].placeholder: input_data,
            network.extern_data.data["data"].size_placeholder[0]: input_seq_lens,
            network.extern_data.data["classes1"].placeholder: classes1_data,
            network.extern_data.data["classes1"].size_placeholder[0]: classes1_seq_lens,
            network.extern_data.data["classes2"].placeholder: classes2_data,
            network.extern_data.data["seq_tag"].placeholder: seq_tags,
        }
        fetches = network.get_fetches_dict()
        result = session.run(fetches, feed_dict=feed)
        pprint(result)

        network.call_graph_reset_callbacks()

    assert os.path.exists(hdf_filename)
    reader = DatasetTestReader(HDFDataset([hdf_filename]))
    reader.read_all()
    assert reader.num_seqs == 1
    assert reader.seq_tags == ["seq-0"]
    assert reader.seq_lens[0]["data"] == input_seq_lens[0]
    assert reader.data["data"][0].shape == (input_seq_lens[0], n_in)
    assert reader.data["classes1"][0].shape == (classes1_seq_lens[0],)
    assert reader.data["classes2"][0].shape == (1,)
    numpy.testing.assert_almost_equal(reader.data["data"][0], input_data[0])
    numpy.testing.assert_equal(reader.data["classes1"][0], classes1_data[0])
    numpy.testing.assert_equal(reader.data["classes2"][0], [classes2_data[0]])


def test_HDFDumpLayer_dump_whole_batch_extra_sm():
    import os
    from test_HDFDataset import get_test_tmp_file, DatasetTestReader, HDFDataset

    hdf_filename = get_test_tmp_file(".hdf")
    os.remove(hdf_filename)  # HDFDumpLayer expects that the file does not exist
    rnd = numpy.random.RandomState(42)

    with make_scope() as session:
        n_in = 5
        config = Config()
        config.update(
            {
                "extern_data": {
                    "data": {"dim": n_in},
                    "sm": dict(shape=(None, None)),
                },
                "network": {
                    "dump": {
                        "class": "hdf_dump",
                        "filename": hdf_filename,
                        "from": "data",
                        "extra": {"sm": "data:sm"},
                        "is_output_layer": True,
                        "dump_whole_batches": True,
                    },
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_value("network"))
        network.print_network_info()

        session.run(tf_compat.v1.global_variables_initializer())
        n_batch = 1
        input_data = numpy.array(
            [
                [
                    [1, -0.2, 0.3, -4, 5],
                    [2, -0.6, 0.7, -1.8, 2.9],
                    [1, 0.3, -0.1, -0.8, 0.5],
                    [0.1, -0.2, 0.2, 0.8, -0.3],
                ]
            ],
            dtype="float32",
        )
        input_seq_lens = [input_data.shape[1]]
        assert input_data.shape == (n_batch, input_seq_lens[0], n_in)
        sm_seq_lens1 = [13]
        sm_seq_lens2 = [17]
        sm_data = rnd.normal(size=(n_batch, sm_seq_lens1[0], sm_seq_lens2[0])).astype(dtype="float32")
        seq_tags = numpy.array([b"seq-0"], dtype="S5")
        feed = {
            network.extern_data.data["data"].placeholder: input_data,
            network.extern_data.data["data"].size_placeholder[0]: input_seq_lens,
            network.extern_data.data["sm"].placeholder: sm_data,
            network.extern_data.data["sm"].size_placeholder[0]: sm_seq_lens1,
            network.extern_data.data["sm"].size_placeholder[1]: sm_seq_lens2,
            network.extern_data.data["seq_tag"].placeholder: seq_tags,
        }
        fetches = network.get_fetches_dict()
        result = session.run(fetches, feed_dict=feed)
        pprint(result)

        network.call_graph_reset_callbacks()

    assert os.path.exists(hdf_filename)
    reader = DatasetTestReader(HDFDataset([hdf_filename]))
    reader.read_all()
    assert reader.num_seqs == 1
    assert reader.seq_tags == ["seq-0"]
    assert reader.seq_lens[0]["data"] == input_seq_lens[0]
    assert reader.data["data"][0].shape == (input_seq_lens[0], n_in)
    numpy.testing.assert_almost_equal(reader.data["data"][0], input_data[0])
    assert reader.data["sm"][0].shape == (sm_seq_lens1[0] * sm_seq_lens2[0],)
    numpy.testing.assert_equal(numpy.reshape(reader.data["sm"][0], sm_data[0].shape), sm_data[0])


def test_HDFDumpLayer_dump_whole_batch_extra_sm1():
    import os
    from test_HDFDataset import get_test_tmp_file, DatasetTestReader, HDFDataset

    hdf_filename = get_test_tmp_file(".hdf")
    os.remove(hdf_filename)  # HDFDumpLayer expects that the file does not exist
    rnd = numpy.random.RandomState(42)

    with make_scope() as session:
        n_in = 5
        config = Config()
        config.update(
            {
                "extern_data": {
                    "data": {"dim": n_in},
                    "sm": dict(shape=(None, 1, None), batch_dim_axis=1, feature_dim_axis=2),
                },
                "network": {
                    "dump": {
                        "class": "hdf_dump",
                        "filename": hdf_filename,
                        "from": "data",
                        "extra": {"sm": "data:sm"},
                        "is_output_layer": True,
                        "dump_whole_batches": True,
                    },
                },
            }
        )
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_value("network"))
        network.print_network_info()

        session.run(tf_compat.v1.global_variables_initializer())
        n_batch = 1
        input_data = numpy.array(
            [
                [
                    [1, -0.2, 0.3, -4, 5],
                    [2, -0.6, 0.7, -1.8, 2.9],
                    [1, 0.3, -0.1, -0.8, 0.5],
                    [0.1, -0.2, 0.2, 0.8, -0.3],
                ]
            ],
            dtype="float32",
        )
        input_seq_lens = [input_data.shape[1]]
        assert input_data.shape == (n_batch, input_seq_lens[0], n_in)
        sm_seq_lens1 = [13]
        sm_seq_lens2 = [17]
        sm_data = rnd.normal(size=(sm_seq_lens1[0], n_batch, 1, sm_seq_lens2[0])).astype(dtype="float32")
        seq_tags = numpy.array([b"seq-0"], dtype="S5")
        feed = {
            network.extern_data.data["data"].placeholder: input_data,
            network.extern_data.data["data"].size_placeholder[0]: input_seq_lens,
            network.extern_data.data["sm"].placeholder: sm_data,
            network.extern_data.data["sm"].size_placeholder[0]: sm_seq_lens1,
            network.extern_data.data["sm"].size_placeholder[2]: sm_seq_lens2,
            network.extern_data.data["seq_tag"].placeholder: seq_tags,
        }
        fetches = network.get_fetches_dict()
        result = session.run(fetches, feed_dict=feed)
        pprint(result)

        network.call_graph_reset_callbacks()

    assert os.path.exists(hdf_filename)
    reader = DatasetTestReader(HDFDataset([hdf_filename]))
    reader.read_all()
    assert reader.num_seqs == 1
    assert reader.seq_tags == ["seq-0"]
    assert reader.data["data"][0].shape == (input_seq_lens[0], n_in)
    numpy.testing.assert_almost_equal(reader.data["data"][0], input_data[0])
    assert reader.data["sm"][0].shape == (sm_seq_lens1[0] * sm_seq_lens2[0],)
    sm_data_ = numpy.transpose(sm_data, (1, 0, 3, 2))
    numpy.testing.assert_equal(numpy.reshape(reader.data["sm"][0], sm_data_[0].shape), sm_data_[0])


def test_CrossEntropyLoss():
    with make_scope() as session:
        n_out = 13
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "data": {"dim": n_out},
                    "classes": {"dim": n_out, "sparse": True},
                },
            }
        )
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "var": {"class": "variable", "shape": (n_out,)},
                "add": {"class": "combine", "kind": "add", "from": ["data", "var"]},
                "output": {"class": "activation", "from": "add", "activation": "softmax", "loss": "ce"},
            }
        )
        losses_dict, total_loss, total_constraints = net.get_losses_initialized()
        print("Losses:")
        pprint(losses_dict)
        assert set(losses_dict.keys()) == {"output"}
        loss_holder = losses_dict["output"]
        assert isinstance(loss_holder, LossHolder)
        assert isinstance(loss_holder.loss, CrossEntropyLoss)
        session.run(tf_compat.v1.global_variables_initializer())
        print("Get loss:")
        feed_dict = make_feed_dict(net.extern_data.data.values(), same_time=True)
        print("random classes:", feed_dict[net.extern_data.data["classes"].placeholder])
        loss_t = loss_holder.get_loss_value()
        opt = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
        minimize_op = opt.minimize(loss_t)
        last_loss_v = float("inf")
        for step in range(3):
            loss_v, _ = session.run((loss_t, minimize_op), feed_dict=feed_dict)
            print("step %i, loss %f" % (step, loss_v))
            assert numpy.isfinite(loss_v) and numpy.isscalar(loss_v)
            assert loss_v < last_loss_v  # it's convex and we cannot overshoot
            last_loss_v = loss_v


def test_CrossEntropyLoss_masked_inf():
    with make_scope() as session:
        n_out = 13
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "data": {"dim": n_out},
                    "classes": {"dim": n_out, "sparse": True},
                },
            }
        )
        mask_t = tf_compat.v1.placeholder(tf.bool, (n_out,), name="mask")

        def mask_func(source, **kwargs):
            x = source(0)
            assert x.shape.ndims == 3  # (B,T,n_out)
            from returnn.tf.util.basic import where_bc

            mask_bc = mask_t[None, None, :]  # (1,1,n_out)
            return where_bc(mask_bc, x, float("-inf"))

        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "var": {
                    "class": "variable",
                    "shape": (n_out,),
                },  # such that we can check that there are no nan/inf grads
                "add": {"class": "combine", "kind": "add", "from": ["data", "var"]},
                "mask": {"class": "eval", "from": "add", "eval": mask_func},
                "output": {"class": "activation", "from": "mask", "activation": "softmax", "loss": "ce"},
            }
        )
        losses_dict, total_loss, total_constraints = net.get_losses_initialized()
        print("Losses:")
        pprint(losses_dict)
        assert set(losses_dict.keys()) == {"output"}
        loss_holder = losses_dict["output"]
        assert isinstance(loss_holder, LossHolder)
        assert isinstance(loss_holder.loss, CrossEntropyLoss)
        session.run(tf_compat.v1.global_variables_initializer())
        print("Get loss:")
        feed_dict = make_feed_dict(net.extern_data.data.values(), same_time=True)
        mask_v = numpy.array([True] * n_out)
        feed_dict[mask_t] = mask_v
        loss_t = loss_holder.get_loss_value()
        opt = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
        minimize_op = opt.minimize(loss_t)
        last_loss_v = float("inf")
        for step in range(3):
            loss_v, _ = session.run((loss_t, minimize_op), feed_dict=feed_dict)
            print("step %i, loss %f" % (step, loss_v))
            assert numpy.isfinite(loss_v) and numpy.isscalar(loss_v)
            assert loss_v < last_loss_v  # it's convex and we cannot overshoot
            last_loss_v = loss_v
        print("Now mask.")
        feed_dict = make_feed_dict(net.extern_data.data.values(), same_time=True, n_batch=1, n_time=1)
        feed_dict[mask_t] = mask_v
        rnd_classes = feed_dict[net.extern_data.data["classes"].placeholder]
        print("random classes:", rnd_classes)
        mask_v[rnd_classes[0, 0]] = False
        (var_t,) = tf_compat.v1.trainable_variables()
        last_var_v = session.run(var_t)
        for step in range(3, 6):
            loss_v, _ = session.run((loss_t, minimize_op), feed_dict=feed_dict)
            print("step %i, loss %f" % (step, loss_v))
            assert numpy.isinf(loss_v) and numpy.isscalar(loss_v)
            var_v = session.run(var_t)
            assert numpy.isfinite(var_v).all()  # while the loss is inf, the gradients should be finite!
            assert not (var_v == last_var_v).all()  # and there also was some non-zero gradient!
            last_var_v = var_v


def test_CrossEntropyLoss_masked_inf_fake_upper_bound():
    # Almost the same as test_CrossEntropyLoss_masked_inf, but we use fake_upper_bound.
    with make_scope() as session:
        n_out = 13
        fake_upper_bound = 10.0
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "data": {"dim": n_out},
                    "classes": {"dim": n_out, "sparse": True},
                },
            }
        )
        mask_t = tf_compat.v1.placeholder(tf.bool, (n_out,), name="mask")

        def mask_func(source, **kwargs):
            x = source(0)
            assert x.shape.ndims == 3  # (B,T,n_out)
            from returnn.tf.util.basic import where_bc

            mask_bc = mask_t[None, None, :]  # (1,1,n_out)
            return where_bc(mask_bc, x, float("-inf"))

        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "var": {
                    "class": "variable",
                    "shape": (n_out,),
                },  # such that we can check that there are no nan/inf grads
                "add": {"class": "combine", "kind": "add", "from": ["data", "var"]},
                "mask": {"class": "eval", "from": "add", "eval": mask_func},
                "output": {
                    "class": "activation",
                    "from": "mask",
                    "activation": "softmax",
                    "loss": "ce",
                    "loss_opts": {"fake_upper_bound": fake_upper_bound},
                },
            }
        )
        losses_dict, total_loss, total_constraints = net.get_losses_initialized()
        print("Losses:")
        pprint(losses_dict)
        assert set(losses_dict.keys()) == {"output"}
        loss_holder = losses_dict["output"]
        assert isinstance(loss_holder, LossHolder)
        assert isinstance(loss_holder.loss, CrossEntropyLoss)
        session.run(tf_compat.v1.global_variables_initializer())
        print("Get loss:")
        feed_dict = make_feed_dict(net.extern_data.data.values(), same_time=True)
        mask_v = numpy.array([True] * n_out)
        feed_dict[mask_t] = mask_v
        loss_t = loss_holder.get_loss_value()
        opt = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
        minimize_op = opt.minimize(loss_t)
        last_loss_v = float("inf")
        for step in range(3):
            loss_v, _ = session.run((loss_t, minimize_op), feed_dict=feed_dict)
            print("step %i, loss %f" % (step, loss_v))
            assert numpy.isfinite(loss_v) and numpy.isscalar(loss_v)
            assert loss_v < last_loss_v  # it's convex and we cannot overshoot
            last_loss_v = loss_v
        print("Now mask.")
        feed_dict = make_feed_dict(net.extern_data.data.values(), same_time=True, n_batch=1, n_time=1)
        feed_dict[mask_t] = mask_v
        rnd_classes = feed_dict[net.extern_data.data["classes"].placeholder]
        print("random classes:", rnd_classes)
        mask_v[rnd_classes[0, 0]] = False
        (var_t,) = tf_compat.v1.trainable_variables()
        last_var_v = session.run(var_t)
        for step in range(3, 6):
            loss_v, _ = session.run((loss_t, minimize_op), feed_dict=feed_dict)
            print("step %i, loss %f" % (step, loss_v))
            assert loss_v == fake_upper_bound and numpy.isscalar(loss_v)
            var_v = session.run(var_t)
            assert numpy.isfinite(var_v).all()  # while the loss is bounded, the gradients should be finite!
            assert not (var_v == last_var_v).all()  # and there also was some non-zero gradient!
            last_var_v = var_v


def test_MeanSquaredError():
    with make_scope() as session:
        n_out = 13
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "data": {"dim": n_out},
                    "classes": {"dim": n_out, "sparse": False, "shape": (1, n_out), "time_dim_axis": 1},
                },
            }
        )
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "var": {"class": "variable", "shape": (n_out,)},
                "add": {"class": "combine", "kind": "add", "from": ["data", "var"]},
                "reduce": {"class": "reduce", "mode": "mean", "axes": "T", "from": "add", "keep_dims": True},
                "output": {"class": "activation", "from": "reduce", "activation": "sigmoid", "loss": "mse"},
            }
        )
        losses_dict, total_loss, total_constraints = net.get_losses_initialized()
        print("Losses:")
        pprint(losses_dict)
        assert set(losses_dict.keys()) == {"output"}
        loss_holder = losses_dict["output"]
        assert isinstance(loss_holder, LossHolder)
        assert isinstance(loss_holder.loss, MeanSquaredError)
        session.run(tf_compat.v1.global_variables_initializer())
        print("Get loss:")
        feed_dict = make_feed_dict(net.extern_data.data.values(), same_time=True, n_time=1)
        print("random classes:", feed_dict[net.extern_data.data["classes"].placeholder])
        loss_t = loss_holder.get_loss_value()
        error_t = loss_holder.get_error_value()
        opt = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
        minimize_op = opt.minimize(loss_t)
        last_loss_v = float("inf")
        for step in range(3):
            loss_v, error_v, _ = session.run((loss_t, error_t, minimize_op), feed_dict=feed_dict)
            print("step %i, loss %f, error %f" % (step, loss_v, error_v))
            assert numpy.isfinite(loss_v) and numpy.isscalar(loss_v)
            assert loss_v <= last_loss_v
            last_loss_v = loss_v


def test_reduce_mean_in_time():
    with make_scope() as session:
        n_out = 5
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "data": {"dim": n_out},
                },
            }
        )
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict({"output": {"class": "reduce", "mode": "mean", "axis": "T", "from": ["data"]}})
        session.run(tf_compat.v1.global_variables_initializer())
        out = net.layers["output"].output.placeholder
        n_batch = 3
        max_seq_len = 10
        feed = make_feed_dict(net.extern_data.data.values(), n_batch=n_batch, n_time=max_seq_len, same_time=True)
        v = session.run(out, feed_dict=feed)
        input_len = feed[net.extern_data.data["data"].size_placeholder[0]]
        input_data = feed[net.extern_data.data["data"].placeholder]

        ref = numpy.zeros([n_batch, n_out])
        for batch, seq_len in enumerate(input_len):
            ref[batch, :] = numpy.mean(input_data[batch, :seq_len, :], axis=0)

        numpy.testing.assert_allclose(ref, v, rtol=1e-5)


def test_reduce_mean_batch_time():
    with make_scope() as session:
        n_out = 5
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "data": {"dim": n_out},
                },
            }
        )
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict({"output": {"class": "reduce", "mode": "mean", "axis": ["B", "T"], "from": ["data"]}})
        session.run(tf_compat.v1.global_variables_initializer())
        out = net.layers["output"].output.placeholder
        n_batch = 3
        max_seq_len = 10
        feed = make_feed_dict(net.extern_data.data.values(), n_batch=n_batch, n_time=max_seq_len, same_time=True)
        v = session.run(out, feed_dict=feed)
        input_len = feed[net.extern_data.data["data"].size_placeholder[0]]
        input_data = feed[net.extern_data.data["data"].placeholder]

        input_data_masked = numpy.copy(input_data)
        for batch, seq_len in enumerate(input_len):
            input_data_masked[batch, seq_len:, :] = numpy.nan
        ref = numpy.nanmean(input_data_masked, axis=(0, 1))

        numpy.testing.assert_allclose(ref, v, rtol=1e-5)


def test_ReduceLayer_mean_btf():
    # https://github.com/rwth-i6/returnn/issues/1242
    net_dict = {"output": {"class": "reduce", "mode": "mean", "from": "data", "axis": ["B", "T", "F"]}}
    config = Config(dict(extern_data={"data": {"shape": (None, 4)}}))
    with make_scope() as session:
        network = TFNetwork(config=config)
        network.construct_from_dict(net_dict)
        in_ = network.extern_data.get_default_input_data()
        out = network.get_default_output_layer().output
        in_v, seq_len, out_v = session.run(
            (in_.placeholder, in_.get_sequence_lengths(), out.placeholder),
            feed_dict=make_feed_dict(network.extern_data),
        )
        n_batch = in_v.shape[0]
        assert n_batch == seq_len.shape[0]
        for b in range(n_batch):
            in_v[b, seq_len[b] :, :] = numpy.nan
        numpy.testing.assert_almost_equal(out_v, numpy.nanmean(in_v))


def test_automatic_seq_lengths():
    with make_scope() as session:
        n_out = 5
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "data": {"dim": n_out},
                },
            }
        )
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "layer0": {
                    "class": "pad",
                    "mode": "reflect",
                    "axes": "spatial",
                    "padding": (3, 3),
                    "handle_dynamic_dims": False,  # not supported yet otherwise
                    "from": "data",
                },  # len+6
                "layer1": {
                    "class": "conv",
                    "from": "layer0",
                    "activation": None,
                    "with_bias": True,
                    "n_out": n_out,
                    "filter_size": (7,),
                    "padding": "valid",
                    "strides": (1,),
                    "dilation_rate": (1,),
                },  # max(len+6-6,0)
                "output": {"class": "copy", "from": "layer1"},
            }
        )
        session.run(tf_compat.v1.global_variables_initializer())
        in_data = net.extern_data.get_default_input_data()
        out_data = net.layers["output"].output.copy_as_batch_spatial_major()
        assert out_data.shape == in_data.shape
        n_batch = 3
        max_seq_len = 10
        feed = make_feed_dict([in_data], n_batch=n_batch, n_time=max_seq_len)
        out_lens = out_data.get_sequence_lengths()
        out_v, out_lens_v = session.run((out_data.placeholder, out_lens), feed_dict=feed)
        in_v = feed[in_data.placeholder]
        in_lens_v = feed[in_data.size_placeholder[0]]
        assert in_v.shape == out_v.shape
        assert in_lens_v.tolist() == out_lens_v.tolist()
        # So far, everything should always be true, unless we have messed some op really up.
        # Now we want to do the main test, i.e. whether we get the same tensor.
        from returnn.tf.util.basic import print_graph_output

        print_graph_output(out_lens)
        assert out_lens is in_data.size_placeholder[0]


def test_automatic_seq_lengths2():
    with make_scope() as session:
        n_out = 5
        config = Config(
            {
                "debug_print_layer_output_template": True,
                "extern_data": {
                    "data": {"dim": n_out},
                },
            }
        )
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(
            {
                "layer0": {
                    "class": "conv",
                    "from": "data",
                    "activation": None,
                    "with_bias": True,
                    "n_out": n_out,
                    "filter_size": (1,),
                    "padding": "valid",
                },
                "output": {"class": "copy", "from": "layer0"},
            }
        )
        session.run(tf_compat.v1.global_variables_initializer())
        in_data = net.extern_data.get_default_input_data()
        out_data = net.layers["output"].output.copy_as_batch_spatial_major()
        assert out_data.shape == in_data.shape
        n_batch = 3
        max_seq_len = 10
        feed = make_feed_dict([in_data], n_batch=n_batch, n_time=max_seq_len)
        out_lens = out_data.get_sequence_lengths()
        out_v, out_lens_v = session.run((out_data.placeholder, out_lens), feed_dict=feed)
        in_v = feed[in_data.placeholder]
        in_lens_v = feed[in_data.size_placeholder[0]]
        assert in_v.shape == out_v.shape
        assert in_lens_v.tolist() == out_lens_v.tolist()
        # So far, everything should always be true, unless we have messed some op really up.
        # Now we want to do the main test, i.e. whether we get the same tensor.
        from returnn.tf.util.basic import print_graph_output

        print_graph_output(out_lens)
        assert out_lens is in_data.size_placeholder[0]


def test_batch_norm_args():
    from returnn.tf.layers.basic import BatchNormLayer
    from returnn.util.basic import getargspec

    batch_norm_args = getargspec(BatchNormLayer.batch_norm).args[2:]  # drop self and data
    layer_args = getargspec(BatchNormLayer.__init__).args[2:]  # drop self and in_dim
    assert batch_norm_args == layer_args  # different arguments in BatchNormLayer and LayerBase.batch_norm()


def test_pickle_dim_tags():
    # Test for pickling net dict and extern data.
    # https://github.com/rwth-i6/returnn_common/issues/104
    # What kind of network we pickle here is not really relevant.
    # I took this from some other test.
    # It should just involve some dim tags.
    from returnn.tf.util.data import batch_dim, ImplicitDynSizeDim

    # noinspection PyUnresolvedReferences,PyProtectedMember
    from pickle import _dumps as pickle_dumps  # Use pickle._dumps for easier debugging.
    import pickle

    dim = FeatureDim("feat-demo-dim", dimension=5)
    dim_ = pickle.loads(pickle_dumps(dim))
    print(dim)
    print(dim_)
    assert dim.dimension == dim_.dimension and dim != dim_ and dim._extra is not dim_._extra

    data = Data(name="data-demo", dim_tags=[dim])
    data_ = pickle.loads(pickle_dumps(data))
    print(data)
    print(data_)
    assert data.shape == data_.shape and data.dim_tags != data_.dim_tags

    n_batch, n_time, n_ts, n_in, n_out = 2, 3, 6, 7, 11
    time_dim = SpatialDim("time")
    feat_dim = FeatureDim("in-feature", dimension=n_in)
    ts_dim = SpatialDim("ts", dimension=n_ts)
    out_dim = FeatureDim("out-feature", dimension=n_out)
    config = Config(
        {
            "extern_data": {"data": {"dim_tags": [batch_dim, time_dim, feat_dim]}},
            "network": {
                "t": {
                    "class": "range_in_axis",
                    "axis": "t",
                    "from": "data",
                    "out_shape": {time_dim, ImplicitDynSizeDim(batch_dim)},
                },  # (T,)
                "range": {"class": "range", "limit": n_ts, "out_spatial_dim": ts_dim},  # (Ts,)
                "add_t": {
                    "class": "combine",
                    "kind": "add",
                    "from": ["t", "range"],
                    "out_shape": {time_dim, ts_dim, ImplicitDynSizeDim(batch_dim)},
                },  # (T,Ts)
                "t_rel_var": {"class": "variable", "shape": (ts_dim, out_dim), "init": "glorot_uniform"},  # (Ts,D)
                "output": {
                    "class": "scatter_nd",
                    "from": "t_rel_var",
                    "position": "add_t",
                    "position_axis": ts_dim,
                    "out_spatial_dim": time_dim,
                    "filter_invalid_indices": True,
                },  # (T,T,D)
            },
        }
    )

    # First test as-is.
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        network.initialize_params(session)
        fetches = network.get_fetches_dict()
        out_layer = network.get_default_output_layer()
        print("out layer:", out_layer)
        session.run((out_layer.output.placeholder, fetches), feed_dict=make_feed_dict(network.extern_data))

    def _debug_dump(tuple_path, obj):
        print("%s: %s" % (tuple_path, type(obj)))
        assert isinstance(obj, (Dim, str, set, str, bool, int))

    from tensorflow.python.util import nest

    nest.map_structure_with_tuple_paths(_debug_dump, config.typed_dict)

    # Now pickle, unpickle and test again.
    s = pickle_dumps(config.typed_dict)
    config_dict = pickle.loads(s)
    new_dim_tags = config_dict["extern_data"]["data"]["dim_tags"]
    new_batch, new_time, new_feat = new_dim_tags
    assert isinstance(new_batch, Dim) and new_batch == batch_dim and new_batch.is_batch_dim()
    assert isinstance(new_time, Dim) and new_time.is_spatial_dim() and new_time.dimension is None
    assert isinstance(new_feat, Dim) and new_feat.is_feature_dim() and new_feat.dimension == n_in
    config = Config(config_dict)
    with make_scope() as session:
        network = TFNetwork(config=config, train_flag=True)
        network.construct_from_dict(config.typed_dict["network"])
        network.initialize_params(session)
        fetches = network.get_fetches_dict()
        out_layer = network.get_default_output_layer()
        print("out layer:", out_layer)
        session.run((out_layer.output.placeholder, fetches), feed_dict=make_feed_dict(network.extern_data))


def test_contrastive_loss():
    from returnn.tf.util.data import batch_dim

    masked_time_dim = SpatialDim("masked_time")
    input_dim = FeatureDim("input", 10)
    enc_feat_dim = FeatureDim("encoder_dim", 20)
    project_dim = FeatureDim("project_dim", 15)

    dim_neg_samples = SpatialDim("neg_samples", 10)  # 100
    dim_expand = SpatialDim("expand_dim", 1)
    contrastive_loss_temp = 0.1
    contrastive_loss_factor = 1.0
    seed = 1

    def _mask(x, axis, pos, max_amount):
        """
        :param tf.Tensor x: (batch,time,feature)
        :param int axis:
        :param tf.Tensor pos: (batch,)
        :param int max_amount: inclusive
        :return: (batch,dim)
        """
        from returnn.tf.compat import v1 as tf

        n_batch = tf.shape(x)[0]
        dim = tf.shape(x)[axis]
        amount = tf.random_uniform(shape=(n_batch,), minval=1, maxval=max_amount + 1, seed=seed, dtype=tf.int32)
        pos2 = tf.minimum(pos + amount, dim)
        idxs = tf.expand_dims(tf.range(0, dim), 0)  # (1,dim)
        pos_bc = tf.expand_dims(pos, 1)  # (batch,1)
        pos2_bc = tf.expand_dims(pos2, 1)  # (batch,1)
        cond = tf.logical_and(tf.greater_equal(idxs, pos_bc), tf.less(idxs, pos2_bc))  # (batch,dim)
        return cond

    def _random_mask(x, axis, min_num, max_num, max_dims):
        """
        :param tf.Tensor x: (batch,time,feature)
        :param int axis:
        :param int|tf.Tensor min_num:
        :param int|tf.Tensor max_num: inclusive
        :param int max_dims: inclusive
        :return: (batch,dim)
        """
        from returnn.tf.compat import v1 as tf

        n_batch = tf.shape(x)[0]
        num = tf.random_uniform(shape=(n_batch,), minval=min_num, maxval=max_num + 1, seed=seed, dtype=tf.int32)
        # https://github.com/tensorflow/tensorflow/issues/9260
        # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
        z = -tf.log(-tf.log(tf.random_uniform((n_batch, tf.shape(x)[axis]), 0, 1, seed=seed)))
        _, indices = tf.nn.top_k(z, tf.reduce_max(num))
        return _mask(x, axis=axis, pos=indices[:, 0], max_amount=max_dims)

    def _get_mask_eval_layer(source, **_kwargs):
        data = source(0, as_data=True, auto_convert=False)
        assert (data.batch_dim_axis, data.time_dim_axis) == (0, 1)
        x = data.placeholder
        mask = _random_mask(x, axis=data.time_dim_axis, min_num=1, max_num=1, max_dims=3)
        return mask

    net_dict = {
        "input": {"class": "linear", "from": "data", "activation": "relu", "out_dim": input_dim},
        # True -> should be masked out by mask embed vector, False -> keep
        "input_mask": {
            "class": "eval",
            "from": "input",
            "eval": _get_mask_eval_layer,
            "out_type": {"dtype": "bool", "shape": (None,)},
        },  # [B,T]
        "mask_emb": {"class": "variable", "shape": [input_dim], "init": "RandomUniform(0., 1.)"},
        "input_masked": {
            "class": "switch",
            "condition": "input_mask",
            "true_from": "mask_emb",
            "false_from": "input",
        },  # [B,T,F]
        # For this loss, the model must be able to look at context, otherwise it does not make sense.
        # https://github.com/rwth-i6/returnn/pull/918
        "conv": {
            "class": "conv",
            "from": "input_masked",
            "filter_size": [5],
            "padding": "same",
            "n_out": 10,
            "activation": "relu",
        },
        "encoder": {"class": "linear", "from": "conv", "out_dim": enc_feat_dim},  # [B,T,F]
        "contrastive_loss": {
            "class": "subnetwork",
            "from": [],
            "subnetwork": {
                "enc_masked_frames_": {
                    "class": "masked_computation",
                    "mask": "base:input_mask",
                    "from": "base:encoder",
                    "unit": {"class": "copy", "from": "data"},
                },  # [B, T_M, F]
                "enc_masked_frames": {
                    "class": "reinterpret_data",
                    "from": "enc_masked_frames_",
                    "set_dim_tags": {"T": masked_time_dim},
                },
                "c": {"class": "linear", "from": "enc_masked_frames", "out_dim": project_dim},
                # We take the non-masked input of the masked frames -> q_t in the paper.
                "input_masked_frames": {
                    "class": "masked_computation",
                    "mask": "base:input_mask",
                    "from": "base:input",
                    "out_spatial_dim": masked_time_dim,
                    "unit": {"class": "copy", "from": "data"},
                },  # [B, T_M, F]
                "q": {"class": "linear", "from": "input_masked_frames", "out_dim": project_dim},
                "q_len": {"class": "length", "from": "input_masked_frames", "axis": "T"},  # [B]
                # Candidate samples
                "q_samples_rand_indices__": {
                    "class": "rand_int",
                    "maxval": 2**30,
                    "from": "input_masked_frames",  # only for masked_time_dim
                    "shape": [batch_dim, masked_time_dim, dim_neg_samples],
                },  # [B, T_M, K] -> 0..BIG
                "q_samples_rand_indices_": {
                    "class": "eval",
                    "from": ["q_samples_rand_indices__", "q_len"],
                    "eval": "source(0) % tf.maximum(source(1) - 1, 1)",
                },  # [B, T_M, K] -> 0..T_M-1
                "_range": {"class": "range_in_axis", "from": "input_masked_frames", "axis": masked_time_dim},  # [T_M]
                "_range_ge_indices": {
                    "class": "compare",
                    "kind": "greater_equal",
                    "from": ["q_samples_rand_indices_", "_range"],
                },  # [B, T_M, K]
                "_indices_offsets": {
                    "class": "switch",
                    "condition": "_range_ge_indices",
                    "true_from": 1,
                    "false_from": 0,
                },
                "q_samples_rand_indices": {
                    "class": "combine",
                    "kind": "add",
                    "from": ["q_samples_rand_indices_", "_indices_offsets"],
                },  # [B, T_M, K]
                "q_sampled_frames": {
                    "class": "gather",
                    "from": "q",
                    "position": "q_samples_rand_indices",
                    "axis": masked_time_dim,
                },  # [B, T_M, K, F]
                "q_expand": {
                    "class": "expand_dims",
                    "axis": "spatial",
                    "dim": dim_expand,
                    "from": "q",
                },  # [B, T_M ,1, F]
                "Q": {
                    "class": "concat",
                    "from": [("q_expand", dim_expand), ("q_sampled_frames", dim_neg_samples)],
                },  # [B, T_M, K+1, F]
                # Cosine similarity between sampled frames and masked encoder frames
                "cos_similarity": {
                    "class": "subnetwork",
                    "from": ["Q", "c"],
                    "concat_sources": False,
                    "subnetwork": {
                        # [B_M, K+1, F] * [B_M, F] -> [B_M, K+1]
                        "dot": {"class": "dot", "from": ["data:0", "data:1"], "reduce": project_dim},
                        "norm_a_sq_": {"class": "eval", "from": "data:0", "eval": "source(0) ** 2"},
                        "norm_a_sq": {
                            "class": "reduce",
                            "mode": "sum",
                            "from": "norm_a_sq_",
                            "axes": project_dim,
                        },  # [B, T_M, K+1]
                        "norm_b_sq_": {"class": "eval", "from": "data:1", "eval": "source(0) ** 2"},
                        "norm_b_sq": {
                            "class": "reduce",
                            "mode": "sum",
                            "from": "norm_b_sq_",
                            "axes": project_dim,
                        },  # [B, T_M]
                        "output": {
                            "class": "eval",
                            "from": ["dot", "norm_a_sq", "norm_b_sq"],
                            "eval": "source(0) * tf.minimum(tf.math.rsqrt(source(1) * source(2)), 1./1e-8)",
                        },  # [B, T_M, K+1]
                    },
                },
                # The contrastive loss is the negative log-likelihood of the softmax of the cosine similarity
                "log_sm_cos_sim": {
                    "class": "softmax_over_spatial",
                    "from": "cos_similarity",
                    "axis": dim_expand + dim_neg_samples,
                    "log_space": True,
                    "energy_factor": contrastive_loss_temp,
                },  # [B, T_M, K+1]
                "log_likelihood": {
                    "class": "gather",
                    "from": "log_sm_cos_sim",
                    "axis": dim_expand + dim_neg_samples,
                    "position": 0,
                },  # [B, T_M]
                "neg_los_likelihood": {"class": "eval", "from": "log_likelihood", "eval": "-source(0)"},  # [B, T_M]
                "output": {"class": "copy", "from": "neg_los_likelihood"},
            },
            "loss": "as_is",
            "loss_scale": contrastive_loss_factor,
        },
        "output": {"class": "softmax", "from": "encoder", "loss": "ce"},
    }

    with make_scope() as session:
        config = Config(
            {
                "extern_data": {"data": {"dim": 7}, "classes": {"dim": 3, "sparse": True}},
                "debug_print_layer_output": True,
                "debug_print_layer_output_shape": True,
            }
        )
        net = TFNetwork(config=config, train_flag=True)
        net.construct_from_dict(net_dict)
        loss = net.get_total_loss()

        tf_compat.v1.set_random_seed(1)
        net.initialize_params(session)
        loss_v = session.run(loss, feed_dict=make_feed_dict(net.extern_data, same_time=True))
        print("loss:", loss_v)
        assert numpy.isfinite(loss_v)


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
