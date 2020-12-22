.. _native_ops:

=================
Native operations
=================

Motivation:

* **Speed up** some important common calculations,
  and potentially **reduce memory requirements**.
  Examples:

  * LSTM
  * CTC loss

* Pure TensorFlow implementations can be suboptimal

  * TF ops almost always create copies, even SplitOp etc

    * Not a memory problem, as input tensor will get freed if not used further
    * Performance problem

  * Gradient might be suboptimal

    * Require too much memory (see automatic gradient checkpointing for a solution)
    * No automatic optimization
    * (Could be solved by custom TF gradient)

  * Memory can be too much distributed (``tf.TensorArray``, TF ``Stack``)

    * Esp. problematic in loop: Separate tensor for every iteration
    * Much better to allocate it as consecutive / contiguous block

  * Overhead (calling individual TF ops, etc)
    (minor compared to the other points)
    (XLA can partially also solve this)

Solution: Write native (C++/CUDA) code

Why is native code faster?

* Operate inplace on tensors

  * Solves all problems mentioned, no unnecessary copies
  * Can use consecutive tensor / memory

* Enforces custom gradient implementation

Problems with native code:

* Can be difficult, memory unsafe, needs more debugging
* Need multiple implementations: CPU (C++), GPU (CUDA)

Our Approach in RETURNN:

The **NativeOp framework**.
See :mod:`returnn.native_op`, :mod:`returnn.tf.native_op`, :mod:`returnn.theano.native_op`.

* Some wrapper / helper code to simplify writing custom native op
* Abstractions to allow single code for CPU & GPU

  * Write kernel CUDA style, using ``threadIdx``, ``blockIdx``, etc

    * Kernel code must be flexible for amount of threads
    * Example, LSTM kernel, loop over dimensions, executed per time-frame:

      .. code-block:: c

          int idx = threadIdx.x + blockDim.x ∗ blockIdx.x;
          while (idx < n_cells ∗ n_batch) {
              int batch_idx = idx / n_cells;
              int cell_idx = idx % n_cells;
              ...
              idx += gridDim.x ∗ blockDim.x;
          }

  * On CPU

    * Custom ``gridDim``, ``blockDim``
    * Other CUDA-like wrappers

History:

* Already available for the Theano backend
* Ported to TensorFlow

  * Directly support for all already prev. implemented ops (LSTM, Baum Welch aligner, ...)

* Easy to port to other frameworks

Examples:

* ``NativeLstm`` (``LstmGenericBase``)
* ``NativeLstm2``
* ``TwoDLSTM``
* ``FastBaumWelch``
* ``FastViterbi``
* ``OptimalCompletionEditDistance``
* ``EditDistance``
* ``Chunking``, ``UnChunking``

See also :ref:`tf_lstm_benchmark`.
