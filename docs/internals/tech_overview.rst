.. _tech_overview:

======================
Technological overview
======================

RETURNN is mostly used as a tool where ``rnn.py`` (:mod:`rnn`) is the main entry point
but you can also use it as a framework / Python module to use in your own Python code.
To get an idea about how it works, it helps to follow roughly the execution path
starting in :mod:`rnn`, esp. in :py:func:`rnn.main`.
In all cases, the code itself should be checked for details and comments.

So far, there are two calculation backends: Theano and TensorFlow,
where Theano was the first backend, thus Theano-specific code files are currently not prefix
but TensorFlow specific files are prefixed with ``TF``.
This is implemented separately for both Theano and TensorFlow:

- The engine for high-level logic, although a bit is shared.
  :mod:`Engine`, :mod:`EngineTask` for Theano and :mod:`TFEngine` for TensorFlow.
- Network topology construction which constructs the computation graph
  for training or just forwarding.
  :mod:`Network`, :mod:`TFNetwork`.
- Network model update code for training, i.e. SGD etc.
  :mod:`Updater`, :mod:`TFUpdater`.
- All the individual layer implementations.
  :mod:`NetworkLayer`, :mod:`NetworkBaseLayer`, :mod:`NetworkHiddenLayer`, :mod:`NetworkRecurrentLayer` etc for Theano
  and :mod:`TFNetworkLayer`, :mod:`TFNetworkRecLayer` for TensorFlow.
  This also means that Theano and TensorFlow don't support the same layers and
  even parameters can be different.
- Some utilities :mod:`TheanoUtil` and :mod:`TFUtil`.
- Multi-GPU logic. :mod:`Device`, :mod:`EngineTask` for Theano and not yet implemented for TensorFlow.

All the rest is shared for all backends, which mostly is:

- The main entry point :mod:`rnn`.
- Config handling :mod:`Config`.
- Logging :mod:`Log`.
- Utilities :mod:`Util`.
- Dataset reading :mod:`Dataset` including all the different dataset implementations
  :mod:`HDFDataset`, :mod:`SprintDataset`, :mod:`LmDataset`, :mod:`GeneratingDataset`, :mod:`MetaDataset`, etc.
- Learning rate scheduling logic such as Newbob :mod:`LearningRateControl`.
- Pretrain network structure construction :mod:`Pretrain`.
- The native op code which generates code for ops for both CUDA and CPU shares a common base.
  :mod:`NativeOp`, where TensorFlow-specific code is in :mod:`TFNativeOp`.


Execution guide
---------------

- :py:func:`rnn.main` will parse command line arguments and read in a config.
- Then logging :mod:`Log` is initialized, based on verbosity and other settings.
- Then it initializes the datasets (``train``, ``dev``, ``eval`` in config),
  i.e. :py:class:`Dataset` instances.
- Theano-only: :py:class:`Device` instances.
- The engine, i.e. a :py:class:`Engine` or :py:class:`TFEngine` instance.
- Depending on the ``task`` option, some engine initialization
  which also initializes the network computation graph, :ref:`tech_net_construct`.
- Then, depending on the ``task`` option, it might start ``engine.train``, ``engine.forward`` etc.
  (:py:func:`Engine.Engine.train` or :py:func:`TFEngine.Engine.train`), :ref:`tech_engine_train`.


.. _tech_net_construct:

put" :   { "class" : "softmax", "loss" : "ce", "from" : ["lstm2_fw", "lstm2_bw"] }
    }



.. _tech_engine_train:

Training
--------

The engine will loop over the epochs and the individual batches / steps and loads and saves the model.
The specific implementation is different in Theano and TensorFlow.
See the code for more details, i.e. :mod:`Engine`, :mod:`EngineTask` for Theano and :mod:`TFEngine` for TensorFlow.

