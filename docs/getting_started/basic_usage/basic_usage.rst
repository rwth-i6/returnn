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
    E.g. ``gpu`` or ``cpu``. Can also be ``gpu0,gpu1`` for multi-GPU training.

use_tensorflow
    If you set this to ``True``, TensorFlow will be used.

train / dev / eval
    The datasets. This can be a filename to a hdf-file.
    Or it can be a dict with an entry ``class`` where you can choose a from a variety
    of other dataset implementations, including many synthetic generated data.
    ``train`` and ``dev`` are used during training, while ``eval`` is usually used to define the dataset for the
    ``forward`` or ``search`` task.

extern_data
    A dictionary
    Defines the source/target dimensions of the data. Both can be integers.
    extern_data can also be a dict if your dataset has other data streams.


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
..
    batching
        The sorting variant when the mini-batches are created. E.g. ``random``.

batch_size
    The total number of frames. A mini-batch has at least a time-dimension
    and a batch-dimension (or sequence-dimension), and depending on dense or sparse,
    also a feature-dimension.
    ``batch_size`` is the upper limit for ``time * sequences`` during creation of the mini-batches.

max_seqs
    The maximum number of sequences in one mini-batch.

..
    chunking
        You can chunk sequences of your data into parts, which will greatly reduce the amount of needed zero-padding.
        This option is a string of two numbers, separated by a comma, i.e. ``chunk_size:chunk_step``,
        where ``chunk_size`` is the size of a chunk,
        and ``chunk_step`` is the step after which we create the next chunk.
        I.e. the chunks will overlap by ``chunk_size - chunk_step`` frames.
        Set this to ``0`` to disable it, or for example ``100:75`` to enable it.

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


There are much more params, and more details to many of the listed ones. Details on the parameters can be found in the
:ref:`parameter reference <parameter_reference>`. As the reference is still incomplete, please watch out for additional
parameters that can be found in the code. All config params can also be passed as command line params.
The generic form is ``++param value``, but more options are available. Please See the code for some usage.

See :ref:`tech_overview` for more details and an overview how it all works.

.. toctree::
    :hidden:

    network.rst
    data.rst
    recurrent_subnet.rst
