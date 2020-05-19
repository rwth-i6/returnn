.. _basic_usage:

===========
Basic Usage
===========

Install RETURNN, :ref:`installation`.

Now ``rnn.py`` is the main entry point. Usage::

    ./rnn.py <config-file> [other-params]

where ``config-file`` is a config file for RETURNN.
See `here for an example <https://github.com/rwth-i6/returnn/blob/master/demos/demo-tf-native-lstm2.12ax.config>`_,
and `many more examples from the demos <https://github.com/rwth-i6/returnn/blob/master/demos/>`_.
We support multiple config syntax, such as simple line-based ``key value``,
JSON based which is determined by a "``{``" at the beginning of the file
or Python code which is determined by a "``#!``" at the beginning of the file.
There is a variety of config params.

``rnn.py`` will execute some task, such as ``train`` or ``forward``. You must define the task
and the train and dev datasets which it will use,
the mini-batch construction variants,
as well as the neural network topology,
as well as some training parameters.

Some common config parameters:

task
    The task, such as ``train``, ``forward`` or ``search``.

device
    E.g. ``gpu`` or ``cpu``.
    Although RETURNN will automatically detect and use a GPU if available,
    a specific device can be enforced by setting this parameter.

use_tensorflow
    If you set this to ``True``, TensorFlow will be used.

train / dev / eval
    The datasets. This can be a filename to a hdf-file.
    Or it can be a dict with an entry ``class`` where you can choose a from a variety
    of other dataset implementations, including many synthetic generated data.
    ``train`` and ``dev`` are used during training, while ``eval`` is usually used to define the dataset for the
    ``forward`` or ``search`` task.

extern_data
    Defines the source/target dimensions of the data. Both can be integers.
    extern_data can also be a dict if your dataset has other data streams.
    The standard source data is called "``data``" by default,
    and the standard target data is called "``classes``" by default.
    You can also specify whether your data is dense or sparse (i.e. it is just the index),
    which is specified by the number of dimensions, i.e. 2 (time-dim + feature-dim) or 1 (just time-dim).
    When using no explicit definition, it is assumed that the data contains a time axis.

    Example: :code:`extern_data = {"data": [100, 2], "classes": [5000, 1]}`.
    This defines an input dimension of 100, and the input is dense (2),
    and an output dimension of 5000, and the output provided by the dataset is sparse (1).

    For a more explicit definition of the shapes, you can provide a dict instead of a list or tuple. This dict may
    contain information to create "Data" objects. For extern_data, only ``dim`` and ``shape`` are required.
    Example: :code:`'feature_data': {'dim': 80, 'shape': (None, 80)}`
    This defines 80 dimensional features with a time axis of arbitrary length.
    Example: :code:`'speaker_classes': {'dim': 1172, 'shape': (), 'sparse': True}`
    This defines a sparse input for e.g. speaker classes that do not have a time axis.

    In general, all input parameters to :class:`TFUtil.Data` can be provided


network
    This is a dict which defines the network topology.
    It consists of layer-names as strings, mapped on dicts, which defines the layers.
    The layer dict consists of keys as strings and the value type depends on the key.
    The layer dict should contain the key ``class`` which defines the class or type of the layer,
    such as ``hidden`` for a feed-forward layer, ``rec`` for a recurrent layer (including LSTM)
    or ``softmax`` for the output layer (doesn't need to have the softmax activation).
    Usually it also contains the key ``n_out`` which defines the feature-dimension of the output of this layer,
    and the key ``from`` which defines the inputs to this layer, which is a list of other layers.
    If you omit ``from``, it will automatically pass in the input data from the dataset.
    All layer dict keys are passed to the layer class ``__init__``,
    so you have to refer to the code for all details.

    Example of a 3 layer bidirectional LSTM:

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

    See :ref:`api` or the code itself for documentation of the arguments for each layer class type.
    The ``rec`` layer class in particular supports a wide range of arguments, and several units which can be used,
    e.g. you can choose between different LSTM implementations, or GRU, or standard RNN, etc.
    See :class:`TFNetworkRecLayer.RecLayer` or :class:`NetworkRecurrentLayer.RecurrentUnitLayer`.
    See also :ref:`tf_lstm_benchmark`.

batching
    Defines the default value for ``seq_ordering`` across all datasets.
    It is recommended to not use this parameter,
    but rather define ``seq_ordering`` explicitely in the datasets for better readability.
    Possible values are:

        - ``default``: Keep the sequences as is
        - ``reverse``: Use the default sequences in reversed order
        - ``random``: Shuffle the data with a predefined fixed seed
        - ``random:<seed>``: Shuffle the data with the seed given
        - ``sorted``: Sort by length (only if available), beginning with shortest sequences
        - ``sorted_reverse``: Sort by length, beginning with longest sequences
        - ``laplace:<n_buckets>``: Sort by length with n laplacian buckets (one bucket means going from shortest to longest and back with 1/n of the data).
        - ``laplace:.<n_sequences>``: sort by length with n sequences per laplacian bucket.

    Note that not all sequence order modes are available for all datasets,
    and some datasets may provide additional modes.

batch_size
    The total number of frames. A mini-batch has at least a time-dimension
    and a batch-dimension (or sequence-dimension), and depending on dense or sparse,
    also a feature-dimension.
    ``batch_size`` is the upper limit for ``time * sequences`` during creation of the mini-batches.

max_seqs
    The maximum number of sequences in one mini-batch.

learning_rate
    The learning rate during training, e.g. ``0.01``.

adam / nadam / ...
    E.g. set :code:`adam = True` to enable the Adam optimization during training.
    See in `Updater.py` for many more.

model
    Defines the model file where RETURNN will save all model params after an epoch of training.
    For each epoch, it will suffix the filename by the epoch number.
    When running ``forward`` or ``search``, the specified model will be loaded.
    The epoch can then be selected with the paramter ``load_epoch``.

num_epochs
    The number of epochs to train.

log_verbosity
    An integer. Common values are 3 or 4. Starting with 5, you will get an output per mini-batch.


There are much more params, and more details to many of the listed ones.
Details on the parameters can be found in the :ref:`parameter reference <parameter_reference>`.
As the reference is still incomplete, please watch out for additional parameters that can be found in the code.
All config params can also be passed as command line params.
The generic form is ``++param value``, but more options are available.
Please See the code for some usage.

See :ref:`tech_overview` for more details and an overview how it all works.

.. toctree::
    :hidden:

    network.rst
    data.rst
    recurrent_subnet.rst
