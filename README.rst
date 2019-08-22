==================
Welcome to RETURNN
==================

`GitHub repository <https://github.com/rwth-i6/returnn>`__.
`RETURNN paper 2016 <https://arxiv.org/abs/1608.00895>`_,
`RETURNN paper 2018 <https://arxiv.org/abs/1805.05225>`_.

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
- Flexible and fast architecture which allows all kinds of encoder-attention-decoder models

See `documentation <http://returnn.readthedocs.io/>`__.
See `basic usage <https://returnn.readthedocs.io/en/latest/basic_usage.html>`__
and `technological overview <https://returnn.readthedocs.io/en/latest/tech_overview.html>`__.

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
are in `TensorFlow LSTM benchmark <https://returnn.readthedocs.io/en/latest/tf_lstm_benchmark.html>`__.

There is also `a wiki <https://github.com/rwth-i6/returnn/wiki>`_.
Questions can also be asked on
`StackOverflow using the RETURNN tag <https://stackoverflow.com/questions/tagged/returnn>`_.

.. image:: https://travis-ci.org/rwth-i6/returnn.svg?branch=master
    :target: https://travis-ci.org/rwth-i6/returnn
