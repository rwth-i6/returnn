.. _network:

=================
Network Structure
=================

Construction
------------

The network structure which defines the model topology is defined by the config ``network`` option,
which is a dict, where each entry is a layer specification, which itself is a dict containing
the kwargs for the specific layer class. E.g.:

.. code-block:: python

    network = {
        "fw1": {"class": "linear", "activation": "relu", "dropout": 0.1, "n_out": 500, "from": ["data"]},
        "fw2": {"class": "linear", "activation": "relu", "dropout": 0.1, "n_out": 500, "from": ["fw1"]},
        "output": {"class": "softmax", "loss": "ce", "from": ["fw2"], "target": "classes"}
    }

The ``"class"`` key will get extracted from the layer arguments and the specific layer class will be used.
Some arguments are available for all layer classes, such as ``dropout``.
A list of all general arguments can be found below in :ref:`network_define_layers`.
For the layer specific arguments such as ``activation``for the linear layer
please have a look at the :ref:`layer_reference`.
The ``from`` argument, which is also available for all layers, is a list of all input layers or datasets.
``"data"`` denotes the default data input.
More details on how to connect layers and datasets can be found below at :ref:`connecting`.



For Theano, the base layer class is
:py:class:`returnn.theano.layers.base.Container` and :py:class:`returnn.theano.layers.base.Layer`;
for TensorFlow, it is :py:class:`returnn.tf.layers.base.LayerBase`.
E.g. that would use the :py:class:`returnn.tf.layers.basic.LinearLayer` class,
and the ``LinearLayer.__init__`` will accepts arguments like ``activation``.
In the given example, all the remaining arguments will get handled by the base layer.

The construction itself can be found for TensorFlow in :py:func:`returnn.tf.network.TFNetwork.construct_from_dict`,
which starts from the output layers goes over the sources of a layer, which are defined by ``"from"``.
If a layer does not define ``"from"``, it will automatically get the input from the dataset data.

Here is a 2 layer unidirectional LSTM network:

.. code-block:: python

    network = {
        "lstm1": {"class": "rec", "unit": "lstm", "dropout": 0.1, "n_out": 500, "from": "data"},
        "lstm2": {"class": "rec", "unit": "lstm", "dropout": 0.1, "n_out": 500, "from": "lstm1"},
        "output": {"class": "softmax", "loss": "ce", "from": "lstm2", "target": "classes"}
    }

In TensorFlow, that would use the layer class :py:class:`returnn.tf.layers.rec.RecLayer`
which will handle the argument ``unit``.

And here is a 3 layer bidirectional LSTM network:

.. code-block:: python

    network = {
    "lstm0_fw" : { "class": "rec", "unit": "lstm", "n_out" : 500, "dropout": 0.1, "L2": 0.01, "direction": 1, "from": "data" },
    "lstm0_bw" : { "class": "rec", "unit": "lstm", "n_out" : 500, "dropout": 0.1, "L2": 0.01, "direction": -1, "from": "data" },

    "lstm1_fw" : { "class": "rec", "unit": "lstm", "n_out" : 500, "dropout": 0.1, "L2": 0.01, "direction": 1, "from" : ["lstm0_fw", "lstm0_bw"] },
    "lstm1_bw" : { "class": "rec", "unit": "lstm", "n_out" : 500, "dropout": 0.1, "L2": 0.01, "direction": -1, "from" : ["lstm0_fw", "lstm0_bw"] },

    "lstm2_fw" : { "class": "rec", "unit": "lstm", "n_out" : 500, "dropout": 0.1, "L2": 0.01, "direction": 1, "from" : ["lstm1_fw", "lstm1_bw"] },
    "lstm2_bw" : { "class": "rec", "unit": "lstm", "n_out" : 500, "dropout": 0.1, "L2": 0.01, "direction": -1, "from" : ["lstm1_fw", "lstm1_bw"] },

    "output" :   { "class" : "softmax", "loss" : "ce", "from" : ["lstm2_fw", "lstm2_bw"] }
    }

.. _network_define_layers:

Defining Layers
---------------

Every usable layer with the TensorFlow backend inherits from :class:`returnn.tf.layers.base.LayerBase`.
This class provides most of the parameters that can be set for each layer.

Every layer accepts the following dictionary entries:

**class** [:class:`str`] specifies the type of the layer.
Each layer class defines a ``layer_class`` attribute which
defines the layer name.

**from** [:class:`list[str]`] specifies the inputs of a layer, usually refering to the layer name.
Many layers automatically concatenate their inputs, as provided by
:class:`TFNetworkLayer._ConcatInputLayer`. For more details on how to connect layers, see :ref:`connecting`.

**n_out** [:class:`int`] specifies the output feature dimension, but the argument is usually not strictly required,
except if there is some transformation like for :class:`returnn.tf.layers.basic.LinearLayer`.
Otherwise the output dimension is predefined
(determined by :func:`returnn.tf.layers.base.LayerBase.get_out_data_from_opts`).
If an explicit output feature dimension is required (like for :class:`returnn.tf.layers.basic.LinearLayer`)
and if ``n_out`` is not specified or set to :class:`None`,
it will try to determine the output size by a provided ``target``.
If a loss is given, it will set ``n_out`` to the value
provided by :func:`returnn.tf.layers.base.Loss.get_auto_output_layer_dim`.
See **out_type** for a more generic parameter.

**out_type** [:class:`dict[str]`] specifies the output shape in more details
(i.e. a more generic version than **n_out**).
The keys are ``dim`` and ``shape`` and others from :class:`returnn.tf.util.data.Data`.
Usually it is automatically derived via :func:`returnn.tf.layers.base.LayerBase.get_out_data_from_opts`.

**loss** [:class:`str`] every layer can have its output connected to a loss function.
For available loss functions, see :ref:`loss`.
When specifying a loss, also ``target`` has to be set (see below).
In addition, ``loss_scale`` (defaults to 1) and ``loss_opts`` can be specified.

**target** [:class:`str`] specifies the loss target in the dataset.
If the target is not part of extern_data,
but another layer in the network, add 'layer:' as prefix.

**loss_scale** [:class:`float`] specifies a loss scale.
Before adding all losses, this factor will be used as scaling.

**loss_opts** [:class:`dict`] specifies additional loss arguments.
For details, see the documentation of the loss functions :ref:`loss`

**loss_only_on_non_search** [:class:`bool`] specifies that the loss should not be calculated during search.

**trainable** [:class:`bool`] (default ``True``) if set to ``False``,
the layer parameters will not be updated during training (parameter freezing).

**L2** [:class:`float`] if specified, add the L2 norm of the parameters
with the given factor to the total constraints.

**darc1** [:class:`float`] if specified, add darc1 loss of the parameters
with the given factor to the total constraints.

**dropout** [:class:`float`] if specified, applies dropout in the input of the layer.

**dropout_noise_shape** [:class:`None` | :class:`dict` | :class:`list` | :class:`tuple`]
Specify for which axes the dropout
mask will be broadcasted (= re-used).
Use `1` for broadcasting and `None` otherwise.
When using a `dict`, the default
axis labels can be used (see :ref:`Managing Axes <managing_axes>` below).
To disable broadcasting for all axes ``{"*": None}`` can be used.
Note that the the dropout mask will always be shared inside a recurrent layer for all recurrent steps.

**dropout_on_forward** [:class:`bool`] if set to true, will also apply dropout during all tasks,
and not only during training.

**spatial_smoothing** [:class:`float`] if specified,
add spatial-smoothing loss of the layer output with the given factor to the total constraints.

**register_as_extern_data** [:class:`str`] register the output of the layer as an accessable entry of ``extern_data``.

.. _connecting:

Connecting Layers
-----------------

In most cases it is sufficient to just specify a list of layer names for the **from** attribute.
When no input is specified,
it will automatically fallback to ``"data"``, which is the default input-data of the provided dataset.
Depending on the
definition of the ``feature`` and ``target`` keys (see :class:`Dataset.DatasetSeq`),
the data can be accessed
via ``from["data:DATA_KEY"]``.
When specifying layers inside a recurrent unit (see :ref:`recurrent_layers`),
two additional
input prefixes are available, ``base`` and ``prev``.
When trying to access layers from outside the recurrent unit, the prefix
``base`` as to be used. Otherwise, only other layers inside the recurrent unit are recognised.
``prev`` can be used to access
the layer output from the previous recurrent step (e.g. for target embedding feedback).

Layer Initialization
--------------------

RETURNN offers multiple methods of initializing layers. This is usually done by setting the parameter
``"forward_weights_init"`` in layers that have trainable parameters.
The methods for initializations include, but are not limited to:

  * providing a single value (will map to ``tf.initializers.constant``)
  * providing the (lowercase) name of a given tensorflow
    initializer <https://www.tensorflow.org/api_docs/python/tf/keras/initializers>`_,
    which can be e.g.:

    * ``"glorot_normal"``
    * ``"glorot_uniform"``
    * ``"orthogonal"``

  * providing a dictionary for the initializer classes:

    * Example: :code:`"forward_weights_init": {'class': 'VarianceScaling', 'scale': 0.5, 'mode': 'fan_out'}`

The initialization is performed in :func:`TFUtil.get_initializer`.

*Note:* the initalizers can be accessed both as e.g. ``"glorot_normal"`` or ``"glorot_normal_initializer"``.

.. _managing_axes:

Managing Axes
-------------

In the default case, the axes of data that is passed between layers (such as batch, time, spatial and feature)
are not visible to the user, and handled by RETURNN internally
with the help of :class:`returnn.tf.util.data.Data` objects.
For layers that operate on specific axes, meaning they have an ``axis`` or ``axes`` parameter, different identifier
(strings) can be used to select the correct axes. These identifier are e.g.

    - ``*:`` select all axes
    - ``B|batch:`` select the batch axis
    - ``T|time:`` select the time axis
    - ``F|feature`` select the feature axis
    - ``spatial`` select all spatial axes (not batch and not feature)
    - ``S:<int>|spatial:<int>`` select a single spatial axis
      from the list of all spatial axes (zero-based, can be negative)
    - ``dyn|dynamic`` select all dynamic axes
      (all spacial axes with dynamic time and time even if it has no dynamic length)
    - ``D:<int>|dyn:<int>|dynamic:<int>`` select a specific dynamic axis (zero-based, can be negative)
    - ``static`` select all static axes (not batch, and has a fixed dimension)
    - ``static:<int>`` select a specific static axis
    - ``T?`` select time axis if existing, none otherwise
    - ``spatial_except_time`` select all spatial axes but also not the time axis
    - ``except_time`` select all axes except time and batch axis
    - ``except_batch`` select all axes except batch axis


Note that all identifier can be used case-insensitive.
For ``axes`` parameter it is also possible to provide a tuple or list of the above identifiers.
If something is unclear, or not working as intended, please refer to
:func:`Data.get_axes_from_description() <returnn.tf.util.data.Data.get_axes_from_description()>`.
