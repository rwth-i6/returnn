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

search_output_file
    Defines where the search output is written to.

search_output_file_format
    The supported file formats are `txt` and `py`.

