.. _data:

=================
Data Input/Output
=================

The parameters that are used to correctly define the data inputs are the three dataset variables ``train``, ``dev`` and
``eval``, as well as the parameter ``extern_data`` to define the data shapes.
The dataset variables are set to a dictionary structure,
where the key "``class``" defines which class implementation to load, and the other entries
are passed as parameters to the constructor of the respective dataset implementation.
A list of the available datasets can be found :ref:`here <dataset_reference>`.

A very simple example would be:

.. code-block:: python

    train = {'class': 'HDFDataset', 'files': ['path/to/training_data.hdf']}

Most datasets follow the convention that the input data is sequential and has the label "``data``", and the target data
is sparse and has the label "``classes``".
In the case of the hdf file it could be that the input data are 100-dimensional MFCCs
and the target data are 5,000 word classes.

The parameter ``extern_data`` can be used to give an explicit definition of the shapes.
All constructor parameters to :class:`returnn.tf.util.data.Data` can be provided as dictionary for each data stream.

For the above example, ``extern_data`` could be defined as:

.. code-block:: python

    extern_data = {
      'data': {'dim': 100, 'shape': (None, 100)},
      'classes': {'dim': 5000, 'shape': (None,), 'sparse': True}
    }

The ``None`` in the "``shape``" parameter tuple defines that the axis has a dynamic shape.
For sequence tasks there is usually only one dynamic axis, which is specified to be the time axis.
In the case of multiple dynamic axes or spatial axes it is helpful to define the time axis explicitely.
For the example case of two dynamic axes, the time axis could be set to be the first axis:

.. code-block:: python

    extern_data = {
      'data': {'dim': 100, 'shape': (None, None, 100), 'time_dim_axis': 1},
      [...]
    }

Note that while the "``shape``" parameter tuple is always defined without the batch axis,
the axis labels for the time, feature or the batch axis itself are counted including the batch axis.
This means that "``time_dim_axis: 1``" corresponds to the first ``None`` of the "``shape``" tuple.
For the general case (non-sparse data), only ``dim`` and ``shape`` are required, the other parameters are optional.


Using Layer Outputs as Data
---------------------------

In case you want to specify data by using layers, it is possible to add ``register_as_extern_data`` to the layer dictionary.
The provided string is the key to access the data.
It is not required to also add the key manually to the ``extern_data`` dictionary.


Using Multiple Data Inputs
--------------------------

For cases where a single dataset is not sufficient, it is possible to combine multiple datasets by using the
:class:`MetaDataset.MetaDataset`.
Details on how to use the MetaDataset can be found :ref:`here <dataset_combination>`.

Synchronizing Dynamic Axes
--------------------------

In the case that there are multiple data streams that have exactly the same length,
RETURNN does not automatically match those axis while broadcasting.
The dynamic axes of different datastreams can be synchronized by using :class:`returnn.tf.util.data.DimensionTag`.

.. code-block:: python

    dynamic_time_dimension = DimensionTag(name="dynamic_time")

    extern_data = {
      'data1': {'dim': 100, 'shape': (None, 100), 'time_dim_axis': 1, 'same_time_dim_as': {'T': dynamic_time_dimension}},
      'data2': {'dim': 10, 'shape': (None, 10), 'time_dim_axis': 1, 'same_time_dim_as': {'T': dynamic_time_dimension}},
      [...]
    }

The parameter "``same_time_dims_as``" takes a dictionary with axes indices or axes labels (see :ref:`managing_axes`)
as key and the `DimensionTag` as value.
For the above example, there is no difference in using `'T'` or `1` as key.

In case you want to synchronize layer outputs, a "``reinterpret_data``"
layer (:class:`ReinterpretDataLayer <returnn.tf.layers.basic.ReinterpretDataLayer>`) can be used.



