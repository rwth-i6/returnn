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
Then, ``extern_data`` would have to be defined as:

.. code-block:: python

    extern_data = {'data': (100, 2), 'classes': (5000, 1)}

The first entry defines the dimension of the data, or the number of integer indices for sparse data.
The second entry specifies whether your data is dense or sparse (i.e. it is just the index),
which is specified by the number of dimensions, i.e. 2 (time-dim + feature-dim) or 1 (just time-dim).
When using no explicit definition, it is assumed that the data contains a time axis.

For a more explicit definition of the shapes, you can provide a dict instead of a list or tuple.
This dict may contain information to create "Data" objects.
For extern_data, only ``dim`` and ``shape`` are required.
Example: :code:`'speaker_classes': {'dim': 1172, 'shape': (), 'sparse': True}`
This defines a sparse input for e.g. speaker classes that do not have a time axis.

In general, all input parameters to :class:`returnn.tf.util.data.Data` can be provided.


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



