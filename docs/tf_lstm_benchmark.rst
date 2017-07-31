.. _tf_lstm_benchmark:

=========================
TensorFlow LSTM benchmark
=========================

There are multiple LSTM implementations/kernels available in TensorFlow, and we also have our own kernel.
In this benchmark, we try to compare the runtime performance during training for each of the kernels.
We try to measure in a way that it should be generic and not be specific for our Returnn framework.
You can run this benchmark yourself with `this script <https://github.com/rwth-i6/returnn/blob/master/demos/demo-tf-lstm-benchmark.py>`_.

In Returnn with the TensorFlow backend, the ``rec`` layer (:class:`TFNetworkRecLayer.RecLayer`)
you can use these LSTM kernels via the ``unit`` argument:

* ``BasicLSTM`` (GPU and CPU).
    Uses `tf.contrib.rnn.BasicLSTMCell <https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell>`_
    via `dynamic_rnn <https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn>`_.
    I.e. the cell itself is pure TensorFlow, and the loop over time is done via ``tf.while_loop``.
* ``StandardLSTM`` (GPU and CPU).
    Uses `tf.contrib.rnn.LSTMCell <https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell>`_
    via `dynamic_rnn <https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn>`_.
    I.e. the cell itself is pure TensorFlow, and the loop over time is done via ``tf.while_loop``.
    This has some more options compared to ``BasicLSTM``.
* ``LSTMBlock`` (GPU and CPU).
    Uses `tf.contrib.rnn.LSTMBlockCell <https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMBlockCell>`_
    via `dynamic_rnn <https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn>`_.
    The time-step operation is implemented as a single TF operation,
    and the loop over time is done via ``tf.while_loop``.
    Thus this should be faster than ``BasicLSTM`` and ``StandardLSTM``.
* ``LSTMBlockFused`` (GPU and CPU).
    Uses `tf.contrib.rnn.LSTMBlockFusedCell <https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMBlockFusedCell>`_.
    The loop over time is part of the op ("fused" in TF terminology),
    thus this is like ``NativeLSTM`` and ``CudnnLSTM`` a single op for the whole calculation.
    This is based on ``LSTMBlock`` and should thus be faster than ``LSTMBlock``.
* ``CudnnLSTM`` (GPU only).
    Uses the LSTM kernel from cuDNN,
    via `tf.contrib.cudnn_rnn.CudnnLSTM <https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnLSTM>`_.
    The loop over time is done internally ("fused").
    If you import such a model on CPU, it will automatically convert it into a ``LSTMBlockFused``.
* ``NativeLSTM`` (GPU and CPU).
    Uses our own CUDA kernel which can also be compiled on CPU.
    The loop over time is also done via C++ code inside the op ("fused").
    See :mod:`TFNativeOp`.

If you just use ``LSTM``, it will currently use ``NativeLSTM`` by default.
Except of ``NativeLSTM``, all of these kernels are part of the official TensorFlow framework.
Note that these kernels are always use for a single direction in time and a single layer.

The cuDNN LSTM kernel can also work bidirectional and do multiple layers at once
but ``tf.contrib.cudnn_rnn.CudnnLSTM`` currently does not support batches with sequences of different length,
thus this is normally not an option to use.
Note that most frameworks with cuDNN bindings do not support this correctly
(see `here <https://stackoverflow.com/questions/41461670/cudnnrnnforwardtraining-seqlength-xdesc-usage>`_),
where CNTK is currently the only exception.
In TensorFlow, this is `issue 6633 <https://github.com/tensorflow/tensorflow/issues/6633>`_.
Note that you still can use the cuDNN kernel in the way we do in Returnn,
i.e. for a single layer in one time-direction.

For the benchmark, we build a multi-layer bidirectional network.
Example of a 3 layer bidirectional LSTM:

.. code-block:: python

    network = {
    "lstm1_fwd" : { "class": "rec", "unit": "lstm", "n_out" : 500, "direction": 1 },
    "lstm1_bwd" : { "class": "rec", "unit": "lstm", "n_out" : 500, "direction": -1 },

    "lstm2_fwd" : { "class": "rec", "unit": "lstm", "n_out" : 500, "direction": 1, "from" : ["lstm1_fwd", "lstm1_bwd"] },
    "lstm2_bwd" : { "class": "rec", "unit": "lstm", "n_out" : 500, "direction": -1, "from" : ["lstm1_fwd", "lstm1_bwd"] },

    "lstm3_fwd" : { "class": "rec", "unit": "lstm", "n_out" : 500, "direction": 1, "from" : ["lstm2_fwd", "lstm2_bwd"] },
    "lstm3_bwd" : { "class": "rec", "unit": "lstm", "n_out" : 500, "direction": -1, "from" : ["lstm2_fwd", "lstm2_bwd"] },

    "output" :   { "class" : "softmax", "loss" : "ce", "from" : ["lstm3_fwd", "lstm3_bwd"] }
    }

We use framewise cross entropy as a loss for training,
and we use a very simple artificial dataset (:class:`GeneratingDataset.Task12AXDataset`)
with dense input with a very low number of dimensions (9)
and single output class indices (sparse) with a very low number of class labels (2),
so that the overhead of the final softmax layer should be minimal, as well as the whole input pipeline.
We are not interested in the error performance on this task in this benchmark,
as in theory the results should all be the same. In practice, they are not due to different implementations,
and also the initialization is currently not the same in all cases.
However, that has no effect on the runtime performance.

By default, we use chunking, i.e. not the full sequences but only slices of it of fixed size (50 frames),
to reduce the amount of padding in a mini-batch and also to keep the maximum sequence length of a batch fixed,
and also to be able to increase the amount of sequences in a batch to allow more parallelism (40 sequences).
See `our paper <https://arxiv.org/abs/1608.00895>`_ for more details about chunking.
Thus, our mini-batch has in total 2000 frames.

----------
Comparison
----------

For a 5 layer bidirectional LSTM with dimension 500 in each time direction, on a GeForce GTX 980,
using 8 CPU threads, we got these results::

    GPU:CudnnLSTM: 0:00:08.8151
    GPU:NativeLSTM: 0:00:08.8440
    GPU:LSTMBlockFused: 0:00:16.9765
    GPU:LSTMBlock: 0:00:33.4895
    GPU:StandardLSTM: 0:00:39.5170
    GPU:BasicLSTM: 0:00:41.7282
    CPU:NativeLSTM: 0:04:05.4365
    CPU:LSTMBlockFused: 0:04:35.1702
    CPU:StandardLSTM: 0:04:57.7977
    CPU:BasicLSTM: 0:05:00.5334
    CPU:LSTMBlock: 0:05:07.5613

On a GeForce GTX 1080 Ti, using 8 CPU threads, for the same experiment we got::

    GPU:NativeLSTM: 0:00:05.2728
    GPU:CudnnLSTM: 0:00:05.3645
    GPU:LSTMBlockFused: 0:00:09.3915
    GPU:LSTMBlock: 0:00:15.3071
    GPU:StandardLSTM: 0:00:17.8279
    GPU:BasicLSTM: 0:00:22.3976
    CPU:NativeLSTM: 0:05:09.6268
    CPU:LSTMBlockFused: 0:07:45.5984
    CPU:StandardLSTM: 0:08:02.5465
    CPU:BasicLSTM: 0:08:16.3543
    CPU:LSTMBlock: 0:08:18.1589

And on a GeForce GTX 1070, with 4 CPU threads, we got::

    GPU:NativeLSTM: 0:00:03.9989
    GPU:CudnnLSTM: 0:00:05.4496
    GPU:LSTMBlockFused: 0:00:07.5233
    GPU:LSTMBlock: 0:00:11.1515
    GPU:StandardLSTM: 0:00:12.0605
    GPU:BasicLSTM: 0:00:12.0833
    CPU:LSTMBlockFused: 0:02:53.6482
    CPU:BasicLSTM: 0:03:00.8289
    CPU:StandardLSTM: 0:03:01.6320
    CPU:LSTMBlock: 0:03:04.8836
    CPU:NativeLSTM: 0:03:18.5375

On a CPU-only system with a single CPU thread, we got::

    CPU:NativeLSTM: 0:15:55.7625
    CPU:LSTMBlockFused: 0:24:53.1451
    CPU:BasicLSTM: 0:26:28.2804
    CPU:StandardLSTM: 0:27:10.0493
    CPU:LSTMBlock: 0:27:58.8870

Each of those are executed on different hardware, so there might be small other differences due to that.
Also the number of available CPU threads differs.
Each of those were run on Ubuntu 16.04 with TensorFlow 1.2 (installed via ``pip``), CUDA 8.0 and cuDNN 5.1.

-----------------------
Analysis and discussion
-----------------------

We are quite proud that our own LSTM kernel (``NativeLSTM``)
has a similar runtime than the cuDNN LSTM kernel (``CudnnLSTM``),
sometimes even better.
The implementation of it is quite straight-forward.

As expected, on GPU, both ``NativeLSTM`` and ``CudnnLSTM`` are faster than ``LSTMBlockFused`` (sometimes twice as fast).

Also as expected, on GPU, ``LSTMBlockFused`` is faster than ``LSTMBlock`` (up to 50%).

On GPU, ``LSTMBlock`` seems slightly faster than ``BasicLSTM``/``StandardLSTM`` but the difference is not so big.

Interestingly, on all experiments, on GPU, ``StandardLSTM`` seems to be slightly faster than ``BasicLSTM``,
which is not expected, as the ``BasicLSTM`` is simpler and also recommended by TensorFlow
if you don't need the extended options which are available for ``StandardLSTM``.

On CPU, it again looks different, and not as clear.
This depends also on how much CPU threads will be used, and on the hardware.
For example, ``NativeLSTM`` is currently not well optimized to use multiple threads (intra op parallelism).
See also :func:`TFUtil.setup_tf_thread_pools` about intra and inter op parallelism.

We see that with a very low number of threads, on CPU, ``NativeLSTM`` can be the fastest, but not necessarily.
Increasing the number of threads, ``NativeLSTM`` can become the slowest.

On CPU, ``LSTMBlockFused`` seems to be the fastest despite ``NativeLSTM``, no matter the number of threads.

On CPU, interestingly, ``BasicLSTM`` and ``StandardLSTM`` seem to be slightly faster than ``LSTMBlock``.
