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
