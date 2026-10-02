#!/usr/bin/env python3

"""
We need to create a simple LSTM model to run in web browser with javascript and tensorflow js.
Therefore, we cannot use NativeLSTM2 here and have to use StandardLSTM. At the same time, the
inference will run in a frame-by-frame base. The goal of this script to create a compatible graph.
network = {
    "source": {
        "class": "eval",
        "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True),
        network=self.network)"
    },
    "dropout1": {
        "class": "dropout",
        "dropout": 0.2,
        "from": "source"
    },
    "lstm_fw": {
        "class": "rec",
        "unit": "StandardLSTM",
        "n_out": 64,
        "dropout": 0.2,
        "direction": 1,
        "from": "dropout1",
        "unit_opts": {"forget_bias": 0.0}
    },
    "dropout2": {
        "class": "dropout",
        "dropout": 0.2,
        "from": "lstm_fw"
    },
    "output": {
        "class": "softmax",
        "n_out": 2,
        "target": "classes",
        "loss": "ce",
        "dropout": 0.2,
        "from": "dropout2"
    }
}
"""

import os
import sys
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_io

try:
    from tensorflow.python.ops.nn import rnn_cell
except ImportError:
    from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn

tf.compat.v1.disable_eager_execution()


def main(argv):
    """
    Main entry.
    """
    # noinspection SpellCheckingInspection
    argparser = argparse.ArgumentParser(description="Compile some StandardLSTM ops")
    argparser.add_argument("--output_file", help="allowed extensions: meta")
    argparser.add_argument("--hidden_size", default=64, type=int, help="LSTM hidden size")
    argparser.add_argument("--input_size", default=75, type=int, help="input feature size")
    argparser.add_argument("--output_classes", default=2, type=int, help="number of output classes")
    args = argparser.parse_args(argv[1:])

    tf.compat.v1.reset_default_graph()

    inputs = tf.compat.v1.placeholder(tf.float32, [1, None, args.input_size], name="inputs")
    initial_c = tf.compat.v1.placeholder(tf.float32, [1, args.hidden_size], name="initial_c")
    initial_h = tf.compat.v1.placeholder(tf.float32, [1, args.hidden_size], name="initial_h")
    lstm_cell = rnn_cell.LSTMCell(num_units=args.hidden_size, forget_bias=0.0, name="lstm_cell")
    initial_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(initial_c, initial_h)

    with tf.compat.v1.variable_scope("lstm_fw"):
        lstm_outputs, lstm_final_state = rnn.dynamic_rnn(
            cell=lstm_cell, inputs=inputs, initial_state=initial_state, dtype=tf.float32, scope="rec"
        )

    with tf.compat.v1.variable_scope("output"):
        # noinspection PyPep8Naming
        W = tf.compat.v1.get_variable(
            name="W",
            shape=[args.hidden_size, args.output_classes],
            initializer=tf.compat.v1.glorot_uniform_initializer(),
        )
        b = tf.compat.v1.get_variable(
            name="b", shape=[args.output_classes], initializer=tf.compat.v1.glorot_uniform_initializer()
        )
        matmul_output = tf.einsum("bth,hk->btk", lstm_outputs, W)
        dense_logits = tf.add(matmul_output, b)
        tf.nn.softmax(dense_logits, axis=-1, name="softmax_output")

    tf.identity(lstm_final_state.c, name="final_state_c")
    tf.identity(lstm_final_state.h, name="final_state_h")

    saver = tf.compat.v1.train.Saver()
    graph_def = saver.export_meta_graph()

    if args.output_file:
        filename = args.output_file
        _, ext = os.path.splitext(filename)
        assert ext in [".meta"], "filename %r extension invalid" % filename
        print("Write graph to file:", filename)
        graph_io.write_graph(
            graph_def,
            logdir=os.path.dirname(filename),
            name=os.path.basename(filename),
            as_text=ext.endswith("txt"),
        )


if __name__ == "__main__":
    main(sys.argv)
