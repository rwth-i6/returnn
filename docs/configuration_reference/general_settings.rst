.. _general_settings:

================
General Settings
================

dev
    A dictionary specifying the developement set. For details on datasets, see :ref:`dataset_reference`

device
    E.g. ``gpu`` or ``cpu``.
    Although RETURNN will automatically detect and use a GPU if available,
    a specific device can be enforced by setting this parameter.

extern_data (former num_outputs)
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

log
    path to the log, or list of paths for multiple logs.

log_batch_size
    If set to ``True``, for each batch the number of sequences and maximal sequence length is displayed

log_verbosity
    An integer or list of integer. Common values are 3 or 4. Starting with 5, you will get an output per mini-batch.
    If a list is proved for logs, log_verbosity can be specified for each log.

model
    Defines the model file where RETURNN will save all model params after an epoch of training.
    For each epoch, it will suffix the filename by the epoch number.
    If ``load_from`` is not set, the model will also be loaded from this path.

network
    This is a nested dict which defines the network topology.
    It consists of layer-names as strings, mapped on dicts, which defines the layers.
    The layer dict consists of keys as strings and the value type depends on the key.
    The layer dict should contain the key ``class`` which defines the class or type of the layer,
    such as ``linear`` for a feed-forward layer, ``rec`` for a recurrent layer (including LSTM)
    or ``softmax`` for the output layer (doesn't need to have the softmax activation).
    Usually it also contains the key ``n_out`` which defines the feature-dimension of the output of this layer,
    and the key ``from`` which defines the inputs to this layer, which is a list of other layers.
    For details sett :ref:`layer_reference`.

num_inputs
    Input feature dimension of the network, related to the 'data' tag.
    Deprecated for the TensorFlow backend, see ``extern_data``

num_outputs
    Output feature dimension of the network, related to the 'classes' tag.
    Deprecated for the TensorFlow backend, see ``extern_data``

task
    The task to run. Common cases are ``train``, ``forward`` or ``search``.

tf_log_memory_usage
    If set to ``True``, will display the current GPU memory usage when using the tensorflow backend.

tf_log_dir
    Defines the folder where the tensorflow/tensorboard logs are writting. Per default, the logs are written next to the models.
    .. note::
        has to be set specifically when loading a model from a folder without write permission

train
    A dictionary specifying the training dataset. For details on datasets, see :ref:`dataset_reference`

use_tensorflow
    If you set this to ``True``, TensorFlow will be used.
