import _setup_test_env  # noqa
import tensorflow as tf
import sys
import os
from nose.tools import assert_equal, assert_is_instance
import contextlib
import unittest
import numpy.testing
from pprint import pprint
from returnn.util import better_exchook
from returnn.config import Config
from returnn.tf.network import *
from returnn.tf.layers.basic import *
from returnn.tf.engine import *
from returnn.log import log
import returnn.tf.compat as tf_compat
import returnn.tf.util.basic as tf_util


@contextlib.contextmanager
def make_scope():
    with tf.Graph().as_default() as graph:
        with tf_compat.v1.Session(graph=graph) as session:
            yield session


network = {}
_last = "data"


def build_resnet(conv_time_dim):
    # network
    # (also defined by num_inputs & num_outputs)
    dropout = 0
    L2 = 0.1
    filter_size = (3, 3)  # for 2D conv on (window, feature) axes

    # data
    context_window = 1

    window = 1
    feature_dim = 64  # Gammatone 40-dim
    channel_num = 3
    num_inputs = feature_dim * channel_num * window
    num_outputs = 9001  # CART labels
    EpochSplit = 6

    cur_feat_dim = feature_dim

    global _last, network
    network = {}
    _last = "data"

    def add_sequential_layer(name, d, from_=None):
        global _last, network
        assert "from" not in d
        if from_ is not None:
            d["from"] = from_
        else:
            d["from"] = [_last]
        assert name not in network
        network[name] = d
        _last = name

    def fixed_padding(prefix, kernel_size, data_format, conv_time_dim):
        """Pads the input along the spatial dimensions independently of input size."""
        pad_total = kernel_size - 1
        feature_pad_beg = pad_total // 2
        feature_pad_end = pad_total - feature_pad_beg

        time_pad_beg = 0
        time_pad_end = 0

        return add_sequential_layer(
            "%s_pad" % prefix,
            {
                "class": "pad",
                "axes": ("s:0", "s:1"),
                "padding": [(time_pad_beg, time_pad_end), (feature_pad_end, feature_pad_end)],
            },
        )

    def conv2d_fixed_padding(
        prefix, filters, kernel_size, strides, dilation_rate, data_format, conv_time_dim, source=None
    ):
        """Strided 2-D convolution with explicit padding."""
        fixed_padding("%s_pad" % prefix, kernel_size, data_format, conv_time_dim)

        padding = "VALID"
        strides = (1, strides) if conv_time_dim else strides
        filter_size = (kernel_size, kernel_size)
        dilation_rate = (dilation_rate, 1) if conv_time_dim else (1, 1)

        if data_format == "channels_first":
            NCHW = True
        else:
            NCHW = False
        add_sequential_layer(
            "%s_conv" % prefix,
            {
                "class": "conv",
                "n_out": filters,
                "filter_size": filter_size,
                "auto_use_channel_first": NCHW,
                "strides": strides,
                "dilation_rate": dilation_rate,
                "padding": padding,
                "activation": None,
                "with_bias": False,
                "dropout": 0,
                "forward_weights_init": "xavier",
                "L2": L2,
            },
            from_=source,
        )
        return "%s_conv" % prefix

    def _building_block_v2(
        prefix,
        filters,
        projection_shortcut,
        strides,
        dilation_rate,
        dilation_rate_multiplier,
        kernel_size,
        data_format,
        conv_time_dim,
    ):
        """A single block for ResNet v2, without a bottleneck.

        Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
        """
        add_sequential_layer("%s_in" % prefix, {"class": "copy"})
        # add_sequential_layer("%s_bn1" % prefix, {"class": "batch_norm", "masked_time": False, "fused_bn": True})
        add_sequential_layer("%s_relu1" % prefix, {"class": "activation", "activation": "relu", "batch_norm": False})
        if conv_time_dim:
            # Workaround (conv can't work with strides > 1 and dilation > 1)
            # => do striding before via slicing.
            conv2d_fixed_padding(
                prefix=("%s_conv_1" % prefix),
                filters=filters,
                kernel_size=kernel_size,
                strides=1,
                dilation_rate=dilation_rate,
                data_format=data_format,
                conv_time_dim=conv_time_dim,
            )
            add_sequential_layer("%s_stride" % prefix, {"class": "slice", "axis": "s:1", "slice_step": strides})
            dilation_rate *= dilation_rate_multiplier
        else:
            conv2d_fixed_padding(
                prefix=("%s_conv_1" % prefix),
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                dilation_rate=dilation_rate,
                data_format=data_format,
                conv_time_dim=conv_time_dim,
            )
        # add_sequential_layer("%s_bn2" % prefix, {"class": "batch_norm", "masked_time": False, "fused_bn": True})
        add_sequential_layer("%s_relu2" % prefix, {"class": "activation", "activation": "relu", "batch_norm": False})

        conv = conv2d_fixed_padding(
            prefix=("%s_conv_2" % prefix),
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            dilation_rate=dilation_rate,
            data_format=data_format,
            conv_time_dim=conv_time_dim,
        )
        result = "%s_conv_2" % prefix
        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        crop_lr = filter_size[0] - 1
        crop_left = crop_lr // 2
        crop_right = crop_lr - crop_left

        if conv_time_dim:
            if dilation_rate_multiplier > 1:
                crop = int(crop_left * (dilation_rate / dilation_rate_multiplier + dilation_rate))
            else:
                crop = int(crop_left * 2 * dilation_rate)
            add_sequential_layer(
                "%s_crop" % prefix,
                {"class": "slice", "axis": "T", "slice_start": crop, "slice_end": -crop},
                from_=("%s_relu1" % prefix),
            )
            shortcut = "%s_crop" % prefix
            if projection_shortcut is not None:
                shortcut = projection_shortcut(source=shortcut)
        else:
            crop = crop_left
            add_sequential_layer(
                "%s_crop_1" % prefix,
                {"class": "slice", "axis": "T", "slice_start": crop, "slice_end": -crop},
                from_=("%s_relu1" % prefix),
            )
            shortcut = "%s_crop_1" % prefix

            if projection_shortcut is not None:
                shortcut = projection_shortcut(source=shortcut)

            add_sequential_layer(
                "%s_crop_2" % prefix,
                {"class": "slice", "axis": "T", "slice_start": crop, "slice_end": -crop},
                from_=shortcut,
            )
            shortcut = "%s_crop_2" % prefix

        add_sequential_layer("%s_out" % prefix, {"class": "combine", "kind": "add"}, from_=[conv, shortcut])
        return

    def block_layer(
        prefix,
        filters,
        bottleneck,
        block_fn,
        blocks,
        strides,
        dilation_rate,
        dilation_rate_multiplier,
        kernel_size,
        data_format,
        conv_time_dim,
    ):
        """Creates one layer of blocks for the ResNet model."""
        # Bottleneck blocks end with 4x the number of filters as they start with
        filters_out = filters * 4 if bottleneck else filters
        if not conv_time_dim:
            strides = (dilation_rate_multiplier, strides)

        def projection_shortcut(source=None):
            return conv2d_fixed_padding(
                prefix=("%s_sc" % prefix),
                filters=filters_out,
                kernel_size=1,
                strides=strides,
                dilation_rate=1,
                data_format=data_format,
                conv_time_dim=conv_time_dim,
                source=source,
            )

        # Only the first block per block_layer uses projection_shortcut and strides
        block_fn(
            "%s_0" % prefix,
            filters,
            projection_shortcut,
            strides,
            dilation_rate,
            dilation_rate_multiplier,
            kernel_size,
            data_format,
            conv_time_dim,
        )
        dilation_rate *= dilation_rate_multiplier
        for i in range(1, blocks):
            block_fn("%s_%i" % (prefix, i), filters, None, 1, dilation_rate, 1, kernel_size, data_format, conv_time_dim)

        return add_sequential_layer("%s_out" % prefix, {"class": "copy"})

    # Params for
    # ResNet Version (d): https://arxiv.org/pdf/1703.02136.pdf
    resnet_version = 2
    conv_time_dim = conv_time_dim
    bottleneck = False
    num_filters = 64
    first_kernel_size = 5
    kernel_size = 3
    conv_stride = 2 if conv_time_dim else (1, 2)
    first_pool_size = (1, 2)
    first_pool_stride = (1, 1)
    last_pool_size = (2, 2)
    last_pool_stride = (1, 2) if conv_time_dim else (2, 2)
    block_sizes = [2, 2, 2, 2]
    block_strides = [1, 2, 2, 2]
    block_dilations = [1, 1, 1, 2]
    block_fn = _building_block_v2
    data_format = "channels_first"
    pre_activation = resnet_version == 2

    if data_format == "channels_first":
        NCHW = True
    else:
        NCHW = False
    # Calculating the reduction of the time dim
    if conv_time_dim:
        multiplier = 1 if bottleneck else 2
        building_block_reduction = multiplier * 2 * (kernel_size // 2)
        total_reduction = first_kernel_size - 1

        dilation_rate_multiplier = 1
        total_reduction += dilation_rate_multiplier * (first_pool_size[0] - 1)

        for i, bs in enumerate(block_sizes):
            total_reduction += building_block_reduction / multiplier * dilation_rate_multiplier
            dilation_rate_multiplier *= block_dilations[i]
            total_reduction += building_block_reduction / multiplier * dilation_rate_multiplier
            total_reduction += building_block_reduction * (bs - 1) * dilation_rate_multiplier

        total_reduction += dilation_rate_multiplier * (last_pool_size[0] - 1)
        dilation_rate_multiplier *= 2
        print(total_reduction, dilation_rate_multiplier)

        total_reduction += dilation_rate_multiplier * 2
        print(total_reduction, dilation_rate_multiplier)

        time_dim_reduction = total_reduction
        context_window = int(2 * (total_reduction // 2) + 1)
    else:
        time_dim_reduction = 0

    print("time_dim_reduction: ", time_dim_reduction)
    print("context_window: ", context_window)

    # Building the ResNet
    conv2d_fixed_padding(
        prefix="c_init",
        filters=num_filters,
        kernel_size=first_kernel_size,
        strides=conv_stride,
        dilation_rate=1,
        data_format=data_format,
        conv_time_dim=conv_time_dim,
    )

    dilation_rate = 1

    if resnet_version == 1:
        # add_sequential_layer("c_init_bn", {"class": "batch_norm", "masked_time": False, "fused_bn": True})
        add_sequential_layer("c_init_relu", {"class": "activation", "activation": "relu", "batch_norm": False})

    if first_pool_size:
        if conv_time_dim:
            dr = (dilation_rate, 1)
        else:
            dr = (1, 1)

        pad_total = first_pool_size[1] - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        add_sequential_layer("c_init_pool_pad", {"class": "pad", "axes": "s:1", "padding": (pad_beg, pad_end)})
        add_sequential_layer(
            "c_init_pool",
            {
                "class": "pool",
                "mode": "max",
                "padding": "VALID",
                "pool_size": first_pool_size,
                "strides": first_pool_stride,
                "dilation_rate": dr,
                "use_channel_first": NCHW,
            },
        )
    print("dr: ", dilation_rate)
    for i, num_blocks in enumerate(block_sizes):
        filters = num_filters * (2**i)
        block_layer(
            prefix="c_%i" % i,
            filters=filters,
            bottleneck=bottleneck,
            block_fn=block_fn,
            blocks=num_blocks,
            strides=block_strides[i],
            dilation_rate=dilation_rate,
            dilation_rate_multiplier=block_dilations[i],
            kernel_size=kernel_size,
            data_format=data_format,
            conv_time_dim=conv_time_dim,
        )
        dilation_rate *= block_dilations[i]
        print("dr: ", dilation_rate)

    if pre_activation:
        # add_sequential_layer("c_out_bn", {"class": "batch_norm", "masked_time": False, "fused_bn": True})
        add_sequential_layer("c_out_relu", {"class": "activation", "activation": "relu", "batch_norm": False})

    if last_pool_size:
        pad_total = last_pool_size[1] - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        add_sequential_layer("c_last_pool_pad", {"class": "pad", "axes": "s:1", "padding": (pad_beg, pad_end)})

        if conv_time_dim:
            dr = (dilation_rate, 1)
            add_sequential_layer(
                "c_last_pool",
                {
                    "class": "pool",
                    "mode": "max",
                    "padding": "VALID",
                    "pool_size": last_pool_size,
                    "strides": (1, 1),
                    "dilation_rate": dr,
                    "use_channel_first": NCHW,
                },
            )
            add_sequential_layer("c_last_stride", {"class": "slice", "axis": "s:1", "slice_step": last_pool_stride[1]})
            dilation_rate *= 2
        else:
            dr = (1, 1)
            add_sequential_layer(
                "c_last_pool",
                {
                    "class": "pool",
                    "mode": "max",
                    "padding": "VALID",
                    "pool_size": last_pool_size,
                    "strides": last_pool_stride,
                    "dilation_rate": dr,
                    "use_channel_first": NCHW,
                },
            )
    if conv_time_dim:
        dr = (dilation_rate, 1)
    else:
        dr = (1, 1)

    """
  See https://arxiv.org/pdf/1611.09288.pdf
  Fully connected layers are equivalent to, and can be trivially replaced by,
  convolutional layers with kernel (1×1) (except the first convolution which
  has kernel size matching the output of the conv stack before being flattened
  for the fully connected layers).
  """
    add_sequential_layer(
        "fc1",
        {
            "class": "conv",
            "n_out": 2048,
            "filter_size": (3, 2),
            "auto_use_channel_first": NCHW,
            "strides": (1, 1),
            "dilation_rate": dr,
            "padding": "VALID",
            "activation": None,
            "with_bias": False,
            "dropout": 0,
            "forward_weights_init": "xavier",
            "L2": L2,
        },
    )
    add_sequential_layer(
        "fc2",
        {
            "class": "conv",
            "n_out": 2048,
            "filter_size": (1, 1),
            "auto_use_channel_first": NCHW,
            "strides": (1, 1),
            "dilation_rate": (1, 1),
            "padding": "VALID",
            "activation": None,
            "with_bias": False,
            "dropout": 0,
            "forward_weights_init": "xavier",
            "L2": L2,
        },
    )
    add_sequential_layer(
        "fc3",
        {
            "class": "conv",
            "n_out": 2048,
            "filter_size": (1, 1),
            "auto_use_channel_first": NCHW,
            "strides": (1, 1),
            "dilation_rate": (1, 1),
            "padding": "VALID",
            "activation": None,
            "with_bias": False,
            "dropout": 0,
            "forward_weights_init": "xavier",
            "L2": L2,
        },
    )
    add_sequential_layer(
        "fc4",
        {
            "class": "conv",
            "n_out": 1024,
            "filter_size": (1, 1),
            "auto_use_channel_first": NCHW,
            "strides": (1, 1),
            "dilation_rate": (1, 1),
            "padding": "VALID",
            "activation": None,
            "with_bias": False,
            "dropout": 0,
            "forward_weights_init": "xavier",
            "L2": L2,
        },
    )
    add_sequential_layer(
        "fc5",
        {
            "class": "conv",
            "n_out": num_outputs,
            "filter_size": (1, 1),
            "auto_use_channel_first": NCHW,
            "strides": (1, 1),
            "dilation_rate": (1, 1),
            "padding": "VALID",
            "activation": None,
            "with_bias": False,
            "dropout": 0,
            "forward_weights_init": "xavier",
            "L2": L2,
        },
    )

    add_sequential_layer("merge", {"class": "merge_dims", "axes": ("s:0", "s:1")})
    add_sequential_layer("swap", {"class": "swap_axes", "axis1": "s:0", "axis2": "f"})
    add_sequential_layer("output", {"class": "activation", "activation": "softmax", "loss": "ce"})
    return network, context_window


def test_ResNet():
    """Test to compare Resnet convolving (window x frequency) vs (time x frequency).
    Batch_norm layers are turned off in oder to compare, since the statistics over the
    windowed input data is a bit different from the plain input (when convolving directing
    over the time dim).
    """

    def sliding_window(seq, window_size):
        import numpy as np
        import copy

        it = iter(seq)
        win = [it.__next__() for cnt in range(window_size)]  # First window
        res_arr = []
        res_arr.append(copy.deepcopy(win))
        for e in it:  # Subsequent windows
            win[:-1] = win[1:]
            win[-1] = e
            res_arr.append(copy.deepcopy(win))
        return np.array(res_arr)

    with make_scope() as session:
        import numpy as np
        import math
        from tensorflow.python.client import timeline

        net_dict_conv_td, window_size = build_resnet(conv_time_dim=True)
        net_dict_windowed, _ = build_resnet(conv_time_dim=False)

        # Making two time-steps
        time_size = window_size + 1
        data_layer_win = Data(name="win", shape=(window_size, 64, 3), dim=3, batch_dim_axis=0, sparse=False)
        data_layer_win.placeholder = tf_compat.v1.placeholder(shape=(None, window_size, 64, 3), dtype=tf.float32)

        data_layer_nowin = Data(
            name="nowin", shape=(time_size, 64, 3), dim=3, batch_dim_axis=0, time_dim_axis=1, sparse=False
        )
        data_layer_nowin.placeholder = tf_compat.v1.placeholder(shape=(None, time_size, 64, 3), dtype=tf.float32)

        extern_data_nowin = ExternData()
        extern_data_nowin.data["data"] = data_layer_nowin
        extern_data_win = ExternData()
        extern_data_win.data["data"] = data_layer_win

        net_conv_td = TFNetwork(extern_data=extern_data_nowin)
        net_conv_td.train_flag = True

        net_conv_td.construct_from_dict(net_dict_conv_td)
        net_conv_td.initialize_params(session)

        net_windowed = TFNetwork(extern_data=extern_data_win)
        net_windowed.train_flag = True

        net_windowed.construct_from_dict(net_dict_windowed)
        net_windowed.initialize_params(session)

        data = np.random.rand(time_size, 64, 3)
        data_win = sliding_window(data, window_size)
        data = np.array([data])

        feed_dict = {data_layer_nowin.placeholder: data, data_layer_win.placeholder: data_win}

        res1, res2 = session.run(
            [net_conv_td.layers["output"].output.placeholder, net_windowed.layers["output"].output.placeholder],
            feed_dict=feed_dict,
        )

        print(res1[0][0] - res2[0][0])
        print(res1[0][1] - res2[1][0])
        assert math.isclose(np.sum(res1[0][0] - res2[0][0]), 0.0, abs_tol=1e-07)
        assert math.isclose(np.sum(res1[0][1] - res2[1][0]), 0.0, abs_tol=1e-07)


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
