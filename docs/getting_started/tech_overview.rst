.. _tech_overview:

======================
Technological Overview
======================

RETURNN is a machine learning toolkit that can be used as standalone application
or Python framework for training and running
sequential neural network architectures.

For an overview of the core concepts behind RETURNN,
see the slides of `our Interspeech 2020 tutorial about machine learning frameworks including RETURNN <https://www-i6.informatik.rwth-aachen.de/publications/download/1154/Zeyer--2020.pdf>`__.

The main tasks of RETURNN are:

    - Network construction, i.e. definition of the computation graph
    - Dataset loading with predefined and extendable :class:`returnn.datasets.Dataset` objects
    - Automatic management of layer outputs (such as tensor axes and time dimensions)
      with a :ref:`Data <data>` object
    - Support of dynamic training schemes that allow for network structure and parameter changes during training
    - Managing the losses and optimizer functions
    - Learning rate scheduling based on training scores

RETURNN supports two calculation backends: TensorFlow and Theano.
It is recommended to stick to the TensorFlow backend, as Theano is deprecated.

RETURNN is mostly used as a tool where ``rnn.py`` is the main entry point
(see :ref:`basic_usage`)
but you can also use it as a framework / Python module to use in your own Python code
(see :ref:`framework`).
To get an idea about how it works, it helps to follow roughly the execution path
starting in :mod:`returnn.__main__`, esp. in :py:func:`returnn.__main__.main`.
In all cases, the code itself should be checked for details and comments.


Structure
---------

Many components are implemented separately for both Theano and TensorFlow:

- The engine for high-level logic, although a bit is shared.
  :mod:`returnn.theano.engine` for Theano
  and :mod:`returnn.tf.engine` for TensorFlow.
  For TensorFlow the engine contains the high level methods for training, forward pass, and other
  executed tasks. It keeps track of the network, devices, models and the updater function, and is the main connection
  between all these components. :mod:`returnn.tf.engine` also contains
  the :class:`returnn.tf.engine.Runner` which is responsible for
  managing the TensorFlow session.
- Network topology construction which constructs the computation graph
  for training or just forwarding.
  :mod:`returnn.theano.network`, :mod:`returnn.tf.network`.
- Network model update code for training, i.e. SGD etc.
  :mod:`returnn.theano.updater`, :mod:`returnn.tf.updater`.
- All the individual layer implementations.
  :mod:`returnn.theano.layers` for Theano
  and :mod:`returnn.tf.layers` for TensorFlow.
  This also means that Theano and TensorFlow don't support the same layers and
  even parameters can be different.
- Some utilities :mod:`returnn.theano.util` and :mod:`returnn.tf.util`,
  which contains the :class:`returnn.tf.util.data.Data` class.
- Multi-GPU logic.
  :mod:`returnn.theano.device`, :mod:`returnn.theano.engine_task` for Theano,
  :mod:`returnn.tf.distributed`, :mod:`returnn.tf.horovod` for TensorFlow.


All the rest is shared for all backends, which mostly is:

- The main entry point :mod:`returnn.__main__`.
- Config handling :mod:`returnn.config`.
- Logging :mod:`returnn.log`.
- Utilities :mod:`returnn.util`.
- Dataset reading :mod:`returnn.datasets` including all the different dataset implementations
  :class:`HDFDataset`, :class:`SprintDataset`,
  :class:`LmDataset`, :class:`GeneratingDataset`, :class:`MetaDataset`, etc.
- Learning rate scheduling logic such as Newbob :mod:`returnn.learning_rate_control`.
- Pretrain network structure construction :mod:`returnn.pretrain`.
- The native op code which generates code for ops for both CUDA and CPU shares a common base.
  :mod:`returnn.native_op`, where TensorFlow-specific code is in :mod:`returnn.tf.native_op`.


Execution guide
---------------

- :py:func:`returnn.__main__.main` will parse command line arguments and read in a config
  (:class:`returnn.config.Config`).
- Then logging (:mod:`returnn.log`, :class:`returnn.log.Log`)
  is initialized, based on verbosity and other settings.
- Then it initializes the datasets (``train``, ``dev``, ``eval`` in config),
  i.e. :py:class:`returnn.datasets.Dataset` instances.
  See :ref:`dataset` and :ref:`dataset_reference`.
- Theano-only: :py:class:`returnn.theano.device.Device` instances.
- The engine, i.e. a :py:class:`returnn.tf.engine.Engine` instance.
- Depending on the ``task`` option, some engine initialization
  which also initializes the network computation graph, :ref:`tech_net_construct`.
- Then, depending on the ``task`` option, it might start ``engine.train``, ``engine.forward`` etc.
  (:py:func:`returnn.tf.engine.Engine.train`), :ref:`tech_engine_train`.


.. _tech_net_construct:

Network Construction
--------------------

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
for TensorFlow, it is :py:class:`returnn.tf.layers.base.LayerBase`.
E.g. that would use the :py:class:`returnn.tf.layers.basic.LinearLayer` class,
and the ``LinearLayer.__init__`` will accepts arguments like ``activation``.
In the given example, all the remaining arguments will get handled by the base layer.

The construction itself can be found for TensorFlow in :py:func:`returnn.tf.network.TFNetwork.construct_from_dict`,
which starts from the output layers goes over the sources of a layer, which are defined by ``"from"``.
If a layer does not define ``"from"``, it will automatically get the input from the dataset data.

The network itself is stored in a :class:`returnn.tf.network.TFNetwork`.

The network, layers, and the dataset make heavy use of :class:`returnn.tf.util.data.Data`,
see :ref:`data`.

Here is a 2 layer unidirectional LSTM network:

.. code-block:: python

    network = {
        "lstm1": {"class": "rec", "unit": "lstm", "dropout": 0.1, "n_out": 500, "from": "data"},
        "lstm2": {"class": "rec", "unit": "lstm", "dropout": 0.1, "n_out": 500, "from": "lstm1"},
        "output": {"class": "softmax", "loss": "ce", "from": "lstm2"}
    }

In TensorFlow, that would use the layer class :py:class:`returnn.tf.layers.rec.RecLayer`
which will handle the argument ``unit``.

See :ref:`network` for more about the network construction and layer declarations.

See also the next section specifically about :ref:`recurrency <tech_overview_recurrency>`.


.. _tech_overview_recurrency:

Recurrency
----------

Recurrency :=
Anything which is defined by step-by-step execution,
where current step depends on previous step, such as RNN, beam search, etc.

This is all covered by :class:`returnn.tf.layers.rec.RecLayer`,
which is a generic wrapper around ``tf.while_loop``.
It covers:

* Definition of stochastic variables (the output classes itself but also latent variables)
  for either beam search or training (e.g. using ground truth values)
* Automatic optimizations

See :ref:`recurrency` for more details how this works.


.. _tech_engine_train:

Training
--------

The engine will loop over the epochs and the individual batches / steps and loads and saves the model.
The specific implementation is different in Theano and TensorFlow.
See the code for more details, i.e. :mod:`returnn.theano.engine`,
:mod:`returnn.theano.engine_task` for Theano
and :mod:`returnn.tf.engine` for TensorFlow.

