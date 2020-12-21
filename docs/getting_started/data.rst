.. _data:

========
``Data``
========

This wraps a ``tf.Tensor``
by adding a lot of meta information about it
and its axes.
This is all in the :class:`returnn.tf.util.data.Data` class.

This was introduced with the TF backend in 2016.

It is conceptually similar to named tensors / named axes
in other frameworks,
but goes much beyond that by having many other meta information
about a tensor and its axes.
Also, an axis name is not simply a string like in other frameworks,
but a :class:`returnn.tf.util.data.DimensionTag` object.

Specifically, the information :class:`returnn.tf.util.data.Data` covers:

* **Shape**

    - Dimension tags for each axis (:class:`returnn.tf.util.data.DimensionTag`)
    - Specific handling of batch axis
    - Default spatial/time axis
    - Default feature axis
    - Shape itself

* **Sequence lengths**
  (tensor of shape [Batch]) for each variable-length axis
  (can have multiple variable-length axes)

* **Data type** (float, int, string, ...)

* **Categorical data** flag,
  i.e. data represents class indices
  (implies ``int`` data type)

    - Number of classes
    - Vocabulary for classes

* **Beam search** information (beam scores, beam source indices for traceback)
  (:class:`returnn.tf.util.data.SearchBeam`)

* Flag whether data is available at decoding/inference time

:class:`returnn.tf.util.data.Data` is used **everywhere** in the TF backend of RETURNN.
Specifically, the inputs/outputs of **layers** are :class:`returnn.tf.util.data.Data`.

Layers are flexible w.r.t. the input format:

* Order of axis should not matter.
  The specific operation will be done on the logical axis
  (e.g. :class:`returnn.tf.layers.basic.LinearLayer` operates on the feature dimension).

* A layer potentially changes the order of axes for efficiency.

    - [Time,Batch,Feature] is more efficient for RNNs
    - [Batch,Feature,Time] is more efficient for CNNs
    - [Batch,Time,Feature] is the default


Example usages
--------------

See :ref:`managing_axes`.

:class:`returnn.tf.layers.basic.SoftmaxOverSpatial`
could be used like

.. code-block::

    "att_weights": {"class": "softmax_over_spatial", "from": "energy"}

This would use the default time axis of the energy.

Or:

.. code-block::

    "att_weights": {"class": "softmax_over_spatial", "from": "energy", "axis": "stag:encoder"}

This would use the dimension tag called "encoder".

:class:`returnn.tf.layers.basic.DotLayer`.


Current shortcomings
--------------------

* Currently the matching / identification of dimension tags is by partial string matching,
  which is hacky, and could potentially also lead to bugs.
  See :ref:`managing_axes`.
  In the future, we probably should make this more explicit
  by using the :class:`returnn.tf.util.data.DimensionTag` object instance explicitly.

* The logic to define the default time/feature axes can be ambiguous in some (rare, exotic) cases.
  Thus, when you use ``"axis": "T"`` in your code, and the tensor has multiple time/spatial axes,
  it sometimes can lead to unexpected behavior.
  This might be a problem also for all layers which operate on the feature dim axis,
  such as :class:`returnn.tf.layers.basic.LinearLayer` and many others.
  (Although in most cases, there is no ambiguity about it...)

* There are sometimes cases where layers are dependent on the order of the axis.
  Examples:

    - :class:`returnn.tf.layers.ConvLayer`:
      The order of the spatial axes matters.
      You define a kernel shape, and the first entry corresponds to the first spatial axis, etc.

    - :class:`returnn.tf.layers.MergeDimsLayer`:
      The order of the merged axes matters.
      (Unless you specify the option ``keep_order``, in which cases the input order does not matter,
      and just the order of what is specified in the config matters.)

* New dim tags are currently created in the ``__init__`` of a layer,
  but they should be created (uniquely) by ``get_out_data_from_opts``.


Related work
------------

* `Pandas for Python (2008) <https://pandas.pydata.org/>`__,
  ``DataFrame``, labelled tabular data
* `xarray for Python (2014) <http://xarray.pydata.org/en/stable/>`__,
  N-D labelled arrays
* `AxisArrays.jl for Julia (2015) <https://github.com/JuliaArrays/AxisArrays.jl>`__,
  each dimension can have a named axis
* `LabeledTensor for TensorFlow (2016) <https://github.com/tensorflow/tensorflow/tree/v1.15.4/tensorflow/contrib/labeled_tensor>`__,
  semantically meaningful dimensions
* `Tensor Shape Annotation Library (tsalib) for TF/PyTorch/NumPy (2018) <https://github.com/ofnote/tsalib>`__,
  named dimensions (e.g. ``'btd'``)
* `NamedTensor for PyTorch (2019) <https://github.com/harvardnlp/NamedTensor>`__
* `PyTorch official support for named tensors (2019) <https://pytorch.org/docs/stable/named_tensor.html>`__,
  e.g. ``torch.zeros(2, 3, names=('N', 'C'))``
* `DeepMind TensorAnnotations (2020) <https://github.com/deepmind/tensor_annotations>`__

In most cases,
this introduces names to axes.
The name is simply a string
(and identification is by string matching).
There usually is no other meta information attached to it (e.g. sequence lengths).
