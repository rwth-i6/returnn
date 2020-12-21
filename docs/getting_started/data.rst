.. _data:

========
``Data``
========

This wraps a ``tf.Tensor``
by adding a lot of meta information about it
and its axes.
This is all in the :class:`Data` class.

This was introduced with the TF backend in 2016.

It is conceptually similar to named tensors / named axes
in other frameworks,
but goes much beyond that by having many other meta information
about a tensor and its axes.
Also, an axis name is not simply a string like in other frameworks,
but a :class:`DimensionTag` object.

Specifically, the information :class:`Data` covers:

* **Shape**
    - Dimension tags for each axis (:class:`DimensionTag`)
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
* Flag whether data is available at decoding/inference time

:class:`Data` is used **everywhere** in the TF backend of RETURNN.
Specifically, the inputs/outputs of **layers** are :class:`Data`.

Layers are flexible w.r.t. the input format:

* Order of axis should not matter.
  The specific operation will be done on the logical axis
  (e.g. :class:`LinearLayer` operates on the feature dimension).
* A layer potentially changes the order of axes for efficiency.
    - [Time,Batch,Feature] is more efficient for RNNs
    - [Batch,Feature,Time] is more efficient for CNNs
    - [Batch,Time,Feature] is the default


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
