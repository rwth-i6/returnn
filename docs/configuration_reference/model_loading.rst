.. _model_loading:

=============
Model Loading
=============

.. note::

    This documentation does not cover all possible combinations of parameters for loading models.
    For more details, please refer to
    `EngineBase <https://github.com/rwth-i6/returnn/blob/master/returnn/engine/base.py>`_
    and also check
    :class:`CustomCheckpointLoader <returnn.tf.network.CustomCheckpointLoader>`
    and the
    `"Different ways to import parameters" <https://github.com/rwth-i6/returnn/wiki/Different-ways-to-import-parameters>`_
    page.

allow_random_model_init
    Initialize a model randomly.
    This can be useful if you want to use only ``preload_from_files`` to load
    multiple different models into one config for decoding without using ``load``.

import_model_train_epoch1
    If a path to a valid model is provided
    (for TF models paths with or without ``.meta`` or ``.index`` extension are possible),
    use this to initialize the weights for training.
    If you do not want to start a new training, see ``load``.

load
    If a path to a valid model is provided
    (for TF models paths with or without ``.meta`` or ``.index`` extension are possible),
    use this to load the specified model and training state.
    The training is continued from the last position.

load_epoch
    Specifies the epoch index, and selects the checkpoint based on the prefix given in ``model``.
    If not set, RETURNN will determine the epoch from the filename or use the latest epoch in case
    of providing only ``model``.

preload_from_files
    A dictionary that contains a ``filename`` entry and optional parameters to define specific model loading.
    If ``prefix`` is defined, it will load the parameters from the checkpoint but only replace the layers that start
    with the given prefix. The layer name in the checkpoint should match the name of the layer without the prefix
    (e.g. the parameters of "submodel1_layer1" in the network would be set to the parameters of "layer1" in the
    checkpoint).
    Example (containing all possible parameters)::

        preload_from_files = {
          "existing-model": {
            "filename": ".../net-model/network.163",  # your checkpoint file, mandatory
            "init_for_train": True,  # only load the checkpoint at the start of training epoch 1, default is False
            "ignore_missing": True,  # if the checkpoint only partly covers your model, default is False
            "ignore_params": ["some_parameter", ...],  # list of parameter names that should not be loaded
            "ignore_params_prefixes": ["some_prefix_", ...],  # list of parameter prefixes that should not be loaded
            "var_name_mapping": {"name_in_graph": "name_in_checkpoint", ...},  # map non-matching parameter names
            "prefix": "submodel1_",  # only load parameters for layers starting with the given prefix
          }
        }

load_ignore_missing_vars
    If enabled, it will ignore missing variables when loading a checkpoint.
    Otherwise it will error on missing variables.
    Non-loaded variables are using the standard variable initialization (e.g. random init).
    By default, this is disabled.
