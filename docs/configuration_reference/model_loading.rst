.. _model_loading:

=============
Model Loading
=============

load_epoch
    Specifies the epoch index to load, based on the prefix given in ``model``.
    If not set, RETURNN will use the latest epoch.

preload_from_files
    A dictionary that should contain ``filename`` and ``prefix``.
    For all layers in your network whose layer name starts with prefix, it will load the parameters from
    the checkpoint specified by filename
    (It will look for the corresponding layer name without the prefix in the given checkpoint).
