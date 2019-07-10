.. _generation_search:

=====================
Generation and Search
=====================

.. note::

    There is no ``beam_size`` parameter for the config, as ``beam_size`` is a parameter for the ``choice`` layer.
    For further details, see :class:`TFNetworkRecLayer.ChoiceLayer`


forward_override_hdf_output
    Per default, Returnn will give an error when trying to overwrite an existing output. If this flag is set to true,
    the check is disabled.

search_output_layer
    TODO...

