import os
from returnn.tf.layers.rec import RecLayer
from returnn.tf.native_op import RecSeqCellOp
from tensorflow.python.ops.rnn_cell_impl import RNNCell

header_text = """
.. _rec_units:

===============
Recurrent Units
===============

These are the units that can be used in a :class:`TFNetworkRecLayer.RecLayer` type of layer.
Common units are:

   * BasicLSTM (the cell), via official TF, pure TF implementation
   * LSTMBlock (the cell), via tf.contrib.rnn.
   * LSTMBlockFused, via tf.contrib.rnn. should be much faster than BasicLSTM
   * CudnnLSTM, via tf.contrib.cudnn_rnn. This is experimental yet.
   * NativeLSTM, our own native LSTM. should be faster than LSTMBlockFused.
   * NativeLstm2, improved own native LSTM, should be the fastest and most powerful

Note that the native implementations can not be in a recurrent subnetwork, as they process the whole sequence at once.
A performance comparison of the different LSTM Layers is available :ref:`here <tf_lstm_benchmark>`.
"""


def generate():
  RecLayer._create_rnn_cells_dict()
  layer_names = sorted(list(RecLayer._rnn_cells_dict.keys()))

  rst_file = open("layer_reference/units.rst", "w")
  rst_file.write(header_text)

  for layer_name in layer_names:
    unit_class = RecLayer.get_rnn_cell_class(layer_name)

    if issubclass(unit_class, RNNCell) or issubclass(unit_class, RecSeqCellOp):
      module = unit_class.__module__
      name = unit_class.__name__

      if name.endswith("Cell") and not name.startswith("_"):
        rst_file.write("\n")
        rst_file.write("%s\n" % name)
        rst_file.write("%s\n" % ("-" * len(name)))
        rst_file.write("\n")
        rst_file.write(".. autoclass:: %s.%s\n" % (module, name))
        rst_file.write("    :members:\n")
        rst_file.write("    :undoc-members:\n")
        rst_file.write("\n")

  rst_file.close()

