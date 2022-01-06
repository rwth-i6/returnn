.. _behavior_version:

====================
``behavior_version``
====================

Instead of maintaining different versions explicitly,
RETURNN has a ``behavior_version`` parameter to specify
the default behavior of the configuration,
as well as restricting the usage of legacy parameters/options.

This "version" number is not related anyhow to new features or bugfixes,
which are always added for all behavior versions.
Still, it may change the default parameters of certain features such as of available layers.

Each version change will be explicitly document here,
and the documentation is indented to always reflect
the recommended usage with the latest ``behavior_version``,
and not listing legacy/deprecated parameters.


Version History
---------------

Behavior version 12 (2022-01-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The batch norm default settings have been changed.
The old settings did not make much sense
and almost always lead to unwanted behavior.

Specifically, the changes are:

* ``momentum``: 0.99 → 0.1
* ``update_sample_only_in_training``: False → True
* ``delay_sample_update``: False → True
* ``param_version``: 0 → 2 (see `#898 <https://github.com/rwth-i6/returnn/issues/898>`__)
* ``masked_time``: True → must be specified explicitly

See issue `#522 <https://github.com/rwth-i6/returnn/issues/522>`__.

Behavior version 11 (2021-12-16)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Broadcasting dims no longer match in :class:`CombineLayer` and others.
This was never needed, instead broadcasting happens in RETURNN automatically to non-existing dims.
To fix this, do not add any broadcasting dims.

See issue `#666 <https://github.com/rwth-i6/returnn/issues/666>`__.

Behavior version 10 (2021-12-07)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`ConvLayer` use ``with_bias=True`` by default.

See issue `#787 <https://github.com/rwth-i6/returnn/issues/787>`__.

Behavior version 9 (2021-12-03)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`ConvLayer`, :class:`PoolLayer` use ``auto_use_channel_first=True`` by default.

In principle, nothing should ever change due to this
when a config is correct in that nothing depends on the order of axes.
However, this is now introduced as a new behavior version
because older configs might depend on the order of axes.
With the other behavior changes, this is mostly disallowed though,
so when you make use of a higher behavior version anyway,
this should be safe.

Behavior version 8 (2021-11-30)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`ConvLayer`, :class:`PoolLayer` and :class:`TransposedConvLayer`
require ``in_spatial_dims`` to be specified
when the input has more than one spatial dimension
(which implies that you perform 2D or 3D convolution or pooling).

This is required to make the order of the spatial axes well defined
because the input axes could have been reordered in any way before.
See issue `#594 <https://github.com/rwth-i6/returnn/issues/594>`__.

Usually, you would use :class:`DimensionTag` to specify ``in_spatial_dims``.
However, to make the transition easier for this specific new behavior,
you can also use a string description for a dimension.
So example usages look like::

    enc_dim = DimensionTag(...)
    dec_dim = DimensionTag(...)

    in_spatial_dims = (enc_dim, dec_tim)
    in_spatial_dims = ("T", "dim:16")
    in_spatial_dims = ("stag:encoder", "stag:decoder")

Behavior version 7 (2021-11-29)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do not allow to specify ``axes`` or ``axis`` arguments in a way that depends on the order of the axes.
E.g. things like ``axis="spatial:1"`` would not be allowed.

To fix this, use dimension tags, i.e. :class:`DimensionTag` instances.
To fix older configs without too much effort,
you might also want to use ``"stag:<name>"`` or ``"stag-single:<idx>:<name>"``
or ``"dim:<static-dim>"``.

Behavior version 6 (2021-11-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`MergeDimsLayer` uses ``keep_order=True`` and does not allow ``keep_order=False``.
There never should be a reason to use ``keep_order=False`` anyway.
If you have that, just remove it.
If that causes any problems, there is probably some other issue in your config.

See issue `#654 <https://github.com/rwth-i6/returnn/issues/654>`__.

Behavior version 5 (2021-11-26)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any ``axis`` or ``axes`` argument in layers does not allow int values anymore.
Instead, use either a str like ``"F"`` or ``"stag:..."``
or use a :class:`DimensionTag` instance.

See issue `#773 <https://github.com/rwth-i6/returnn/issues/773>`__.

Behavior version 4 (2021-11-23)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Broadcasting in all inputs simultaneously in layers and other ops
is not allowed anymore by default.
In all inputs simultaneously means that there is no input which has all common dimensions.

Layers can explicitly allow this by specifying ``out_shape``.
In case you stumble upon this, specify ``out_shape`` in the layer.

See :func:`validate_broadcast_all_sources`
and issue `#691 <https://github.com/rwth-i6/returnn/issues/691>`__.

Behavior version 3 (2021-11-08)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DotLayer``: disallow ``int`` axes descriptions, remove and change defaults.

Change ``-1`` to e.g. ``"static:-1"`` or ``"F"``.
Change ``-2`` to e.g. ``"dynamic:0"`` or ``"T"`` or ``"stag:..."`` or ``dim_tag``.

See issue `#627 <https://github.com/rwth-i6/returnn/issues/627>`__.

Behavior version 2 (2021-08-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Disallow boolean optimizer specifications such as ``adam = True``
in favor of using ``optimizer = {"class": "adam", ...}``

See issue `#512 <https://github.com/rwth-i6/returnn/issues/514>`__.

Behavior version 1 (2021-05-28)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Disallow not specifying ``"from"`` in layer definition dictionaries,
thus making use of the hidden default ``"data"`` as layer input.

``"from"`` needs to be set explicitly now.
Set it to ``"data"`` or ``"data:data"`` or some other layer or ``()`` (empty).

See issue `#519 <https://github.com/rwth-i6/returnn/issues/519>`__.

