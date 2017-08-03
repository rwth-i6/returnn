.. _index:

==================
Welcome to RETURNN
==================

`RETURNN paper <https://arxiv.org/abs/1608.00895>`_.

RETURNN - RWTH extensible training framework for universal recurrent neural networks,
is a Theano/TensorFlow-based implementation of modern recurrent neural network architectures.
It is optimized for fast and reliable training of recurrent neural networks in a multi-GPU environment.

Features include:

- Mini-batch training of feed-forward neural networks
- Sequence-chunking based batch training for recurrent neural networks
- Long short-term memory recurrent neural networks
  including our own fast CUDA kernel
- Multidimensional LSTM (GPU only, there is no CPU version)
- Memory management for large data sets
- Work distribution across multiple devices

See :ref:`basic_usage` and :ref:`tech_overview`.

There are `many example demos <https://github.com/rwth-i6/returnn/blob/master/demos/>`_
which work on artificially generated data,
i.e. they should work as-is.

There are `some real-world examples <https://github.com/rwth-i6/returnn-experiments>`_
such as setups for speech recognition on the Switchboard corpus.

Some benchmark setups against other frameworks
can be found `here <https://github.com/rwth-i6/returnn-benchmarks>`_.
The results are in the `RETURNN paper <https://arxiv.org/abs/1608.00895>`_.

There is also `a wiki <https://github.com/rwth-i6/returnn/wiki>`_.


User guide
----------

.. toctree::
    :maxdepth: 2

    installation
    basic_usage
    tech_overview
    tf_lstm_benchmark


API Reference
-------------

.. toctree::
    :maxdepth: 2
    :glob:

    api


Refs
----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/rwth-i6/returnn/
