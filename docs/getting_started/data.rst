.. _data:

====================
``Data`` and ``Dim``
====================

``Data``
--------

This wraps a ``tf.Tensor``
by adding a lot of meta information about it
and its axes.
This is all in the :class:`returnn.tf.util.data.Data` class.

This was introduced with the TF backend in 2016.
The idea and concept is also explained in the slides of
`our Interspeech 2020 tutorial about machine learning frameworks including RETURNN <https://www-i6.informatik.rwth-aachen.de/publications/download/1154/Zeyer--2020.pdf>`__.

It is conceptually similar to *named tensors / named axes*
in other frameworks,
but goes much beyond that by having lots of other meta information
about a tensor and its axes.
Also, an axis name is not simply a string like in other frameworks,
but a :class:`returnn.tf.util.data.Dim` object.

Specifically, the information :class:`returnn.tf.util.data.Data` covers:

* **Shape**

  * Dimension tags for each axis (:class:`returnn.tf.util.data.Dim`), see below
  * Specific handling of batch axis
  * Default spatial/time axis
  * Default feature axis
  * Shape itself

* **Sequence lengths**
  (tensor of shape [Batch]) for each variable-length axis
  (can have multiple variable-length axes)

* **Data type** (float, int, string, ...)

* **Categorical data** flag,
  i.e. data represents class indices
  (implies ``int`` data type)

  * Number of classes
  * Vocabulary for classes

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

  * [Time,Batch,Feature] is more efficient for RNNs
  * [Batch,Feature,Time] is more efficient for CNNs
  * [Batch,Time,Feature] is the default


``Dim``
-------

A :class:`returnn.tf.util.data.Dim` object,
representing a dimension (axis) of a :class:`returnn.tf.util.data.Data` object.
We also refer to this as dimension tag,
as it covers more meta information than just the size.

It stores:

- Static size, or ``None`` representing dynamic sizes
- (Sequence) lengths in case of dynamic sizes.
  Usually, these are per batch entry, i.e. of shape [Batch].
  However, this is not a requirement, and they can also have any shape.
  In fact, the dynamic size is again another :class:`returnn.tf.util.data.Data` object.
- Optional some vocabulary
- Its kind: batch, spatial or feature
  (although in most cases there is no real difference between spatial or feature)

Many layers allow to specify a custom dimension tag as output,
via ``out_dim`` or similar options.
See `#597 <https://github.com/rwth-i6/returnn/issues/597>`__.

To make it easier for the user to create custom new dimension tags
(and then set them in the network via ``out_dim`` or related options),
there are the helper functions:

* ``SpatialDim(...)``
* ``FeatureDim(...)``

Further, it is possible to perform elementary algebra on dimension tags
such as addition, subtraction, multiplication and division.
These operations are not commutative,
i.e. ``a + b != b + a`` and ``a * b != b * a``,
because the order of concatenation and merging dimensions matters
and vice versa for splitting features and splitting dimensions.
We support equality for simple identities
like ``2 * a == a + a`` (but ``2 * a != a * 2``),
``(a + b) * c == a * c + b * c``,
``a * b // b == a``.
See `#853 <https://github.com/rwth-i6/returnn/pull/853>`__.
See ``test_dim_math_...`` functions for examples.

We provide a global batch dim object (``returnn.tf.util.data.batch_dim``)
which can be used to avoid creating a new batch dim object every time,
although it does not matter as we treat all batch dims as equal.
Any logic regarding the batch dim (such as beam search) is handled separately.

In a user config, the dim tags are usually introduced already for ``extern_data``.
Example::

    from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim
    input_seq_dim = SpatialDim("input-seq-len")
    input_feat_dim = FeatureDim("input-feature", 40)
    target_seq_dim = SpatialDim("target-seq-len")
    target_classes_dim = FeatureDim("target-classes", 1000)

    extern_data = {
        "data": {
            "dim_tags": [batch_dim, input_seq_dim, input_feat_dim]},
        "classes": {
            "dim_tags": [batch_dim, target_seq_dim],
            "sparse_dim": target_classes_dim},
    }

All layers which accept some ``axis`` or ``in_dim`` argument also can be given some dim object
instead of using some text description (like ``"T"`` or ``"F"``).
A dimension tag object is usually more robust than relying on such textual description
and is the recommended way.

You can specify ``out_shape`` for any layer to verify the output shape
via dimension tags.
See `#706 <https://github.com/rwth-i6/returnn/issues/706>`__.


Example usages
--------------

See :ref:`managing_axes`.

:class:`returnn.tf.layers.basic.SoftmaxOverSpatialLayer`
could be used like

.. code-block:: python

    "att_weights": {"class": "softmax_over_spatial", "from": "energy"}

This would use the default time axis of the energy.

Or:

.. code-block:: python

    "att_weights": {"class": "softmax_over_spatial", "from": "energy", "axis": "stag:encoder"}

This would use the dimension tag called "encoder".

:class:`returnn.tf.layers.basic.ReduceLayer`, example doing max over the encoder time axis:

.. code-block:: python

    "output": {"class": "reduce", "axis": "stag:encoder", "mode": "max", "from": "encoder"}

:class:`returnn.tf.layers.basic.DotLayer`.


Current shortcomings
--------------------

* The logic to define the default time/feature axes can be ambiguous in some (rare, exotic) cases.
  Thus, when you use ``"axis": "T"`` in your code, and the tensor has multiple time/spatial axes,
  it sometimes can lead to unexpected behavior.
  This might be a problem also for all layers which operate on the feature dim axis,
  such as :class:`returnn.tf.layers.basic.LinearLayer` and many others.
  (Although in most cases, there is no ambiguity about it...)


Related work
------------

* `Pandas for Python (2008) <https://pandas.pydata.org/>`__,
  ``DataFrame``, labelled tabular data
* `xarray for Python (2014) <https://xarray.pydata.org/en/stable/>`__,
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
