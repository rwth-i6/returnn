.. _layer_reference:

==========================
Tensorflow Layer Reference
==========================


General Information
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
is passed to :func:TFNetworkLayer.LayerBase.get_out_data_from_opts`.

**loss** [:class:`str`] every layer can have its output connected to a loss function. For available loss functions,
see :ref:`loss`. When specifying a loss, also ``target`` has to be set (see below). In addition, ``loss_scale`` (defaults to 1)
and ``loss_opts`` can be specified.

**target** [:class:`str`] specifies the loss target in the dataset.

**loss_scale** [:class:`float`] specifies a loss scale. Before adding all losses, this factor will be used as scaling.

**loss_opts** [:class:`dict`] specifies additional loss arguments. For details, see the documentation of the loss functions :ref:`loss`

**trainable** [:class:`bool`] (default ``True``) if set to ``False``, the layer parameters will not be updated during training (parameter freezing).

**L2** [:class:`float`] if specified, add the L2 norm of the parameters with the given factor to the total constraints.

**darc1** [:class:`float`] if specified, add darc1 loss of the parameters with the given factor to the total constraints.

**spatial_smoothing** [:class:`float`] if specified, add spatial-smoothing loss of the layer output with the given factor to the total constraints.

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

RETURNN offers multiple methods of intializing layers. This is usually done by setting the parameter
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

*Note: the initalizers can be accessed both as e.g. ``"glorot_normal"`` or ``"glorot_normal_initializer"``.

Layer Types
-----------

.. toctree::
    :maxdepth: 2

    basic.rst
    shape.rst
    recurrent.rst
    attention.rst
    norm.rst
    custom.rst
    utility.rst
    loss.rst
    softmax.rst


