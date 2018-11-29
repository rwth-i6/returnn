.. _deterministic_training:

======================
Deterministic training
======================

There are couple of TF operations which have a non-deterministic GPU implementation (for efficiency reasons), i.e. the result when executed on the GPU is non-deterministic.
See also `here <https://www.twosigma.com/insights/article/a-workaround-for-non-determinism-in-tensorflow/>`_.

Non-deterministic ops:

* ``reduce_mean``, ``reduce_sum`` (`see here <https://github.com/tensorflow/tensorflow/issues/3103>`_).
  Or now deterministic? (`see here <https://github.com/tensorflow/tensorflow/issues/2732>`_)
* convolutional ops (via cuDNN) can be (`see here <https://github.com/tensorflow/tensorflow/issues/18096>`_)
* ``BiasAddGrad`` (`see here <https://github.com/tensorflow/tensorflow/issues/22398>`_)
* ...

E.g. however ``matmul`` is deterministic. From the CUDA doc:

  By design, all CUBLAS API routines from a given toolkit version, generate the same bit-wise results at every run when executed on GPUs with the same architecture and the same number of SMs. However, bit-wise reproducibility is not guaranteed across toolkit version because the implementation might differ due to some implementation changes.


The option ``deterministic_train`` controls whether Returnn should use deterministic ops as far as possible.
So far this uses e.g. ``aggregation_method = tf.AggregationMethod.ADD_N``
and not ``aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N`` for the TF optimizer.
We plan to extend this by replacing some of the non-deterministic ops by deterministic ones.
