.. _model_loading:

=============
Model Loading
=============

import_model_train_epoch1
    If a path to a valid model is provided (for TF models without ``.meta`` extension),
    use this to initialize the weights for training.
    If you do not want to start a new training, see ``load``.

load
    If a path to a valid model is provided (for TF models without ``.meta`` extension),
    use this to load the specified model and training state.
    The training is continued from the last position.

load_epoch
    Specifies the epoch index to load, based on the prefix given in ``model``.
    If not set, RETURNN will use the latest epoch.

preload_from_files
    A dictionary that should contain ``filename`` and ``prefix``.
    For all layers in your network whose layer name starts with prefix, it will load the parameters from
    the checkpoint specified by filename
    (It will look for the corresponding layer name without the prefix in the given checkpoint).
    Example (without using a specific prefix)::

        preload_from_files = {
          "existing-model": {
            "init_for_train": True,
            "ignore_missing": True,  # if the checkpoint only partly covers your model
            "filename": ".../net-model/network.163",  # your checkpoint file
          }
        }

