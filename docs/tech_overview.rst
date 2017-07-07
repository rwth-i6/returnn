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

Network structure construction
------------------------------

The network structure which defines the model topology is defined by the config ``network`` option,
which is a dict, where each entry is a layer specification, which itself is a dict containing
the kwargs for the specific layer class. E.g.:

.. code-block:: python

    network = {
        "fw1": {"class": "linear", "activation": "relu", "dropout": 0.1, "n_out": 500},
        "fw2": {"class": "linear", "activation": "relu", "dropout": 0.1, "n_out": 500, "from": ["fw1"]},
        "output": {"class": "softmax", "loss": "ce", "from": ["fw2"]}
    }

The ``"class"`` key will get extracted from the layer arguments and the specific layer class will be used.
For Theano, the base layer class is :py:class:`NetworkBaseLayer.Container` and :py:class:`NetworkBaseLayer.Layer`;
for TensorFlow, it is :py:class:`TFNetworkLayer.LayerBase`.
E.g. that would use the :py:class:`TFNetworkLayer.LinearLayer` class,
and the ``LinearLayer.__init__`` will accepts arguments like ``activation``.
In the given example, all the remaining arguments will get handled by the base layer.

The construction itself can be found for TensorFlow in :py:func:`TFNetwork.TFNetwork.construct_from_dict`,
which starts from the output layers goes over the sources of a layer, which are defined by ``"from"``.
If a layer does not define ``"from"``, it will automatically get the input from the dataset data.

Here is a 2 layer unidirectional LSTM network:

.. code-block:: python

    network = {
        "lstm1": {"class": "rec", "unit": "lstm", "dropout": 0.1, "n_out": 500},
        "lstm2": {"class": "rec", "unit": "lstm", "dropout": 0.1, "n_out": 500, "from": ["lstm1"]},
        "output": {"class": "softmax", "loss": "ce", "from": ["lstm2"]}
    }

In TensorFlow, that would use the layer class :py:class:`TFNetworkRecLayer.RecLayer`
which will handle the argument ``unit``.

And here is a 3 layer bidirectional LSTM network:

.. code-block:: python

    network = {
    "lstm0_fw" : { "class": "rec", "unit": "lstm", "n_out" : 500, "dropout": 0.1, "L2": 0.01, "direction": 1 },
    "lstm0_bw" : { "class": "rec", "unit": "lstm", "n_out" : 500, "dropout": 0.1, "L2": 0.01, "direction": -1 },

    "lstm1_fw" : { "class": "rec", "unit": "lstm", "n_out" : 500, "dropout": 0.1, "L2": 0.01, "direction": 1, "from" : ["lstm0_fw", "lstm0_bw"] },
    "lstm1_bw" : { "class": "rec", "unit": "lstm", "n_out" : 500, "dropout": 0.1, "L2": 0.01, "direction": -1, "from" : ["lstm0_fw", "lstm0_bw"] },

    "lstm2_fw" : { "class": "rec", "unit": "lstm", "n_out" : 500, "dropout": 0.1, "L2": 0.01, "direction": 1, "from" : ["lstm1_fw", "lstm1_bw"] },
    "lstm2_bw" : { "class": "rec", "unit": "lstm", "n_out" : 500, "dropout": 0.1, "L2": 0.01, "direction": -1, "from" : ["lstm1_fw", "lstm1_bw"] },

    "output" :   { "class" : "softmax", "loss" : "ce", "from" : ["lstm2_fw", "lstm2_bw"] }
    }



.. _tech_engine_train:

Training
--------

The engine will loop over the epochs and the individual batches / steps and loads and saves the model.
The specific implementation is different in Theano and TensorFlow.
See the code for more details, i.e. :mod:`Engine`, :mod:`EngineTask` for Theano and :mod:`TFEngine` for TensorFlow.

