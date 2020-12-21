.. _index:

==================
Welcome to RETURNN
==================

`RETURNN paper 2016 <https://arxiv.org/abs/1608.00895>`_,
`RETURNN paper 2018 <https://arxiv.org/abs/1805.05225>`_.

RETURNN - RWTH extensible training framework for universal recurrent neural networks,
is a Theano/TensorFlow-based implementation of modern recurrent neural network architectures.
It is optimized for fast and reliable training of recurrent neural networks in a multi-GPU environment.

The high-level features and goals of RETURNN are:

* **Simplicity**

    – Writing config / code is simple & straight-forward (setting up experiment, defining model)
    – Debugging in case of problems is simple
    – Reading config / code is simple (defined model, training, decoding all becomes clear)

* **Flexibility**

    – Allow for many different kinds of experiments / models

* **Efficiency**

    – Training speed
    – Decoding speed

All items are important for research, decoding speed is esp. important for production.

Here are the `slides of the Interspeech 2020 tutorial "Efficient and Flexible Implementation of Machine Learning for ASR and MT" with an introduction of the core concepts <https://www-i6.informatik.rwth-aachen.de/publications/download/1154/Zeyer--2020.pdf>`__.

More specific features include:

- Mini-batch training of feed-forward neural networks
- Sequence-chunking based batch training for recurrent neural networks
- Long short-term memory recurrent neural networks
  including our own fast CUDA kernel
- Multidimensional LSTM (GPU only, there is no CPU version)
- Memory management for large data sets
- Work distribution across multiple devices
- Flexible and fast architecture which allows all kinds of encoder-attention-decoder models

See :ref:`basic_usage` and :ref:`tech_overview`.

`Here is the video recording of a RETURNN overview talk <https://www-i6.informatik.rwth-aachen.de/web/Software/returnn/downloads/workshop-2019-01-29/01.recording.cut.mp4>`_
(`slides <https://www-i6.informatik.rwth-aachen.de/web/Software/returnn/downloads/workshop-2019-01-29/01.returnn-overview.session1.handout.v1.pdf>`__,
`exercise sheet <https://www-i6.informatik.rwth-aachen.de/web/Software/returnn/downloads/workshop-2019-01-29/01.exercise_sheet.pdf>`__;
hosted by eBay).

There are `many example demos <https://github.com/rwth-i6/returnn/blob/master/demos/>`_
which work on artificially generated data,
i.e. they should work as-is.

There are `some real-world examples <https://github.com/rwth-i6/returnn-experiments>`_
such as setups for speech recognition on the Switchboard or LibriSpeech corpus.

Some benchmark setups against other frameworks
can be found `here <https://github.com/rwth-i6/returnn-benchmarks>`_.
The results are in the `RETURNN paper 2016 <https://arxiv.org/abs/1608.00895>`_.
Performance benchmarks of our LSTM kernel vs CuDNN and other TensorFlow kernels
are in :ref:`tf_lstm_benchmark`.

There is also `a wiki <https://github.com/rwth-i6/returnn/wiki>`_.
Questions can also be asked on
`StackOverflow using the RETURNN tag <https://stackoverflow.com/questions/tagged/returnn>`_.

Some recent development changelog can be seen `here <https://github.com/rwth-i6/returnn/blob/master/CHANGELOG.md>`__.


.. Getting Started
.. ---------------

.. toctree::
    :hidden:
    :caption: Getting Started
    :maxdepth: 2

    getting_started/tech_overview.rst
    getting_started/data.rst
    getting_started/installation.rst
    getting_started/basic_usage.rst
    getting_started/framework.rst
    getting_started/faq.rst
    getting_started/tf_lstm_benchmark.rst

.. toctree::
    :hidden:
    :caption: User Guide
    :maxdepth: 2

    user_guide/network.rst
    user_guide/dataset.rst
    user_guide/recurrent_subnet.rst


.. toctree::
    :hidden:
    :caption: Reference
    :maxdepth: 2

    configuration_reference/index.rst
    dataset_reference/index.rst
    layer_reference/index.rst
    optimizer.rst

.. toctree::
    :hidden:
    :caption: Advanced Topics
    :maxdepth: 2

    advanced/pretraining.rst
    advanced/multi_gpu.rst
    advanced/debugging.rst
    advanced/profiling.rst
    advanced/deterministic_training.rst

.. toctree::
    :hidden:
    :caption: Applications

    applications/asr.rst
    applications/lm.rst
    applications/mt.rst

.. toctree::
    :hidden:
    :caption: Internals

    api.rst
    internals/search.rst


Refs
----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/rwth-i6/returnn/
