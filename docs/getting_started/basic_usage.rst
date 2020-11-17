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
The configuration syntax can be in three different forms:

  - a simple line-based file with ``key value`` pairs
  - a JSON file (determined by a "``{``" at the beginning of the file)
  - executable python code (determined by a "``#!``" at the beginning of the file)

Config files using the python code syntax are the de-facto standard for all current examples and setups.
The parameters can be set by defining global variables, but it is possible to use any form of python code such
as functions and classes to construct your network or fill in global variables based on more complex decisions.
The python syntax config files may also contain additional code such as layer or dataset definitions.

When calling ``rnn.py`` will execute some ``task``, such as ``train``, ``forward`` or ``search``.

The task ``train`` will train a model specified by a given network structure.
After training each epoch on provided `training data, the current parameters will be stored to a model checkpoint file.
Besides the training data, a development dataset is used to evaluate the current model, and store the evaluation
results in a separate file.

The task ``forward`` will run a forward pass of the network, given an evaluation dataset, and store the results in
an HDF file.

The task ``search`` is used to run the network with the beam-search algorithm.
The results are serialized into text form and stored in a plain text file python dictionary format file.

The following parameters are very common, and are used in most RETURNN config files:

task
    The task, such as ``train``, ``forward`` or ``search``.

device
    E.g. ``gpu`` or ``cpu``.
    Although RETURNN will automatically detect and use a GPU if available,
    a specific device can be enforced by setting this parameter.

use_tensorflow
    If you set this to ``True``, the TensorFlow will be used.
    Otherwise, the installed backend is used.
    If both backends are installed (TensorFlow and Theano), RETURNN will use Theano as default for legacy reasons.

train / dev / eval
    The datasets parameters are set to a python dict with a mandatory entry ``class``.
    The ``class`` attribute needs to be set to the class name of the dataset that should be used.
    An overview over available datasets can be found :ref:`here <dataset_reference>`.
    ``train`` and ``dev`` are used during training, while ``eval`` is usually used to define the dataset for the
    ``forward`` or ``search`` task.

    Beside passing the constructor parameters to the specficic Dataset, there are some common parameters such as:

    ``seq_ordering``: This defines the order of the sequences provided by the dataset.
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

extern_data
    Defines the source/target dimensions of the data as a dictionary of dictionaries describing data streams.
    The standard source data is called "``data``" by default,
    and the standard target data is called "``classes``" by default.

    A common example for an ASR system would be:
    .. code-block:: python

        extern_data = {
          "data": {"dim": 100, "shape": (None, 100)}
          "classes": {"dim": 5000, "shape": (None,), "sparse": True}
        }

    In this case the ``data`` entry defines 80 dimensional features with a time axis of arbitrary length.
    ``classes`` defines sparse target labels, and the dimension then defines the number of labels.
    The shape entries ``None`` indicate a dynamic length of an axis.

    In general, all input parameters to :class:`returnn.tf.util.data.Data` can be provided
    The parameters ``dim`` and ``shape`` should always be used, the other parameters are optional.
    Note that only for ``data`` the parameter ``available_for_inference`` is per default `True``.


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


There are much more parameters, and more details to many of the listed ones.
Details on the parameters can be found in the :ref:`parameter reference <parameter_reference>`.
As the reference is still incomplete, please watch out for additional parameters that can be found in the code.

All configuration params can also be passed as command line parameters.
The generic form is ``++param value``, but more options are available.
Please See the code for some usage.

See :ref:`tech_overview` for more details and an overview how it all works.

