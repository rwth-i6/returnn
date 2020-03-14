.. _network:

=================
Network Structure
=================

Construction
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


Defining Layers
-------------------

Every usable layer with the tensorflow backend inherits from :class:`TFNetworkLayer.LayerBase`.
This class provides most of the parameters that can be set for each layer.

Every layer accepts the following dictionary entries:

**class** [:class:`str`] specifies the type of the layer. Each layer class defines a ``layer_class`` attribute which
defines the layer name.

**from** [:class:`list[str]`] specifies the inputs of a layer, usually refering to the layer name. Many layers automatically concatenate their inputs, as provided by
:class:`TFNetworkLayer._ConcatInputLayer`. For more details on how to connect layers, see :ref:`connecting`.

**n_out** [:class:`int`] specifies the output feature dimension, and is usually set for every layer, but the argument is not strictly required.
If ``n_out`` is not specified or set to :class:`None`, it will try to determine the output size by a provided ``target``.
If a loss is given, it will set ``n_out`` to the value provided by :func:`TFNetworkLayer.Loss.get_auto_output_layer_dim`.

**out_type** [:class:`dict[str]`] specifies the output shape in more details. The keys are ``dim`` and ``shape``.
If ``output`` is specified, the values are used to check if the output matches the given dimension and shape. Otherwise, it
is passed to :func:`TFNetworkLayer.LayerBase.get_out_data_from_opts`.

**loss** [:class:`str`] every layer can have its output connected to a loss function. For available loss functions,
see :ref:`loss`. When specifying a loss, also ``target`` has to be set (see below). In addition, ``loss_scale`` (defaults to 1)
and ``loss_opts`` can be specified.

**target** [:class:`str`] specifies the loss target in the dataset. If the target is not part of extern_data,
but another layer in the network, add 'layer:' as prefix.

**loss_scale** [:class:`float`] specifies a loss scale. Before adding all losses, this factor will be used as scaling.

**loss_opts** [:class:`dict`] specifies additional loss arguments. For details, see the documentation of the loss functions :ref:`loss`

**loss_only_on_non_search** [:class:`bool`] specifies that the loss should not be calculated during search.

**trainable** [:class:`bool`] (default ``True``) if set to ``False``, the layer parameters will not be updated during training (parameter freezing).

**L2** [:class:`float`] if specified, add the L2 norm of the parameters with the given factor to the total constraints.

**darc1** [:class:`float`] if specified, add darc1 loss of the parameters with the given factor to the total constraints.

**spatial_smoothing** [:class:`float`] if specified, add spatial-smoothing loss of the layer output with the given factor to the total constraints.

**register_as_extern_data** [:class:`str`] register the output of the layer as an accessable entry of extern_data.

.. _connecting:

Connecting Layers
-----------------

In most cases it is sufficient to just specify a list of layer names for the **from** attribute. When no input is specified,
it will automatically fallback to ``"data"``, which is the default input-data of the provided dataset. Depending on the
definition of the ``feature`` and ``target`` keys (see :class:`Dataset.DatasetSeq`), the data can be accessed
via ``from["data:DATA_KEY"]``. When specifying layers inside a recurrent unit (see :ref:`recurrent_layers`), two additional
input prefixes are available, ``base`` and ``prev``. When trying to access layers from outside the recurrent unit, the prefix
``base`` as to be used. Otherwise, only other layers inside the recurrent unit are recognised. ``prev`` can be used to access
the layer output from the previous recurrent step (e.g. for target embedding feedback).

Layer Initialization
--------------------

RETURNN offers multiple methods of initializing layers. This is usually done by setting the parameter
``"forward_weights_init"`` in layers that have trainable parameters.
The methods for initializations include, but are not limited to:

  * providing a single value (will map to ``tf.initializers.constant``)
  * providing the (lowercase) name of a given tensorflow `intializer <https://www.tensorflow.org/api_docs/python/tf/keras/initializers>`_,
    which can be e.g.:

    * ``"glorot_normal"``
    * ``"glorot_uniform"``
    * ``"orthogonal"``

  * providing a dictionary for the initializer classes:

    * Example: :code:`"forward_weights_init": {'class': 'VarianceScaling', 'scale': 0.5, 'mode': 'fan_out'}`

The initialization is performed in :func:`TFUtil.get_initializer`.

*Note:* the initalizers can be accessed both as e.g. ``"glorot_normal"`` or ``"glorot_normal_initializer"``.
