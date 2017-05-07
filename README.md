RETURNN development tree
========================

[RETURNN paper](https://arxiv.org/abs/1608.00895).

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

[Please read the documentation for more information](http://returnn.readthedocs.io/).

There are some example demos in `/demos`
which work on artifically generated data,
i.e. they should work as-is.

There are some real-world examples [here](https://github.com/rwth-i6/returnn-experiments).

Some benchmark setups against other frameworks
can be found [here](https://github.com/rwth-i6/returnn-benchmarks).
The results are in the [RETURNN paper](https://arxiv.org/abs/1608.00895).

[![Test Status](https://travis-ci.org/rwth-i6/returnn.svg?branch=master)](https://travis-ci.org/rwth-i6/returnn)

