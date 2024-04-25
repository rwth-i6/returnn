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

Behavior version 21 (2024-04-25)

RF ``pad`` and TF ``PadLayer`` defaults changed:

* ``handle_dynamic_dims``: False → True

Behavior version 20 (2024-01-05)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RF ``TransformerDecoder`` defaults changed:

* ``share_embedding``: False → True
* ``input_embedding_scale``: 1.0 → sqrt(model_dim)
* ``input_dropout``: 0.0 → dropout

Behavior version 19 (2023-11-13)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RF ``SelfAttention`` (and derived classes)
and RF ``dot_attention``:
The attention dropout (via ``att_dropout`` option)
broadcasted over all but the reduced-time dimension before,
which is very likely not what you want.
Now with behavior version 19, the attention dropout does not broadcast.
You can also explicitly get the new behavior via the global config option
``rf_att_dropout_broadcast = False``.

Behavior version 18 (2023-09-02)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TF ``WindowLayer`` returns an optimized dimension order by default.
This is the dimension order which is used anyway internally.
The old behavior was to reshuffle the dim order to the original input order.
There should not be any reason to use the old behavior
(please report it if you think otherwise),
so the flag to control this is considered internal (``_use_opt_dim_order``).

Behavior version 17 (2023-04-19)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TF ``ZoneoutLSTMCell`` used the wrong output,
which was different from ``h``
(it was actually the original output without zoneout),
so it was not as specified in the Zoneout paper,
and likely suboptimal.

A new flag ``use_zoneout_output`` was introduced
to switch between both behaviors.
Setting it to ``True`` enables the correct behavior
and makes it consistent with the paper.
Setting it to ``False`` enables the old incorrect behavior.

With behaviour version 17,
the default changed to ``use_zoneout_output=True``.
If you want to get the old behavior with a new behavior version,
just set ``use_zoneout_output=False``.

See issue `#1313 <https://github.com/rwth-i6/returnn/issues/1313>`__.

Behavior version 16 (2022-11-11)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Dim.is_equal`` is more restrictive and does not give equality
for different user-generated tags,
or also when comparing user-generated to auto-generated tags.
This should rarely have an effect for you.

For TF layers:
It might break when you mix ``n_out`` and then later also have a different
own dim tag for the same dim.
In that case, they will not match because the tag is different.
In such cases, just make use of dim tags consistently, i.e. use ``out_dim``,
or specify the dim tag directly in a shape.

See issue `#865 <https://github.com/rwth-i6/returnn/issues/865>`__
and issue `#1219 <https://github.com/rwth-i6/returnn/issues/1219>`__.

Behavior version 15 (2022-11-08)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``TFNetwork.global_train_step`` should deterministically
always return the initial value of the current step.
Before, it could non-deterministically return the current step or the next step.
This has an influence on any code which makes use of it.
The biggest effect is on gradient accumulation
where the old non-deterministic behavior likely was wrong.

See issue `#1205 <https://github.com/rwth-i6/returnn/issues/1205>`__.

Behavior version 14 (2022-10-19)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dim matching in TF :class:`DotLayer` is now more strict
for the case that ``var1`` and ``var2`` are not provided,
to figure out the common dims.

If this causes problems to you,
e.g. because you are not using dim tags consistently,
then just specify ``var1`` and ``var2`` explicitly.

See issue `#1154 <https://github.com/rwth-i6/returnn/issues/1154>`__.

Behavior version 13 (2022-10-13)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This enables some extra checks in the TF :class:`RecLayer` which break some old configs,
where the old configs where actually broken,
but those broken parts did not play a role for the training
and thus it did not matter.
However, we don't want to allow such broken configs anymore.
More specifically, an optimized-out ``output`` sub-layer of a :class:`RecLayer`
must have the same time dim as the :class:`RecLayer` itself.
For some specific transducer configs, we have this problem
(`example <https://github.com/rwth-i6/returnn-experiments/blob/264d13aef3321d48f685cc9750fd277fb70cc74e/2020-rnn-transducer/configs/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.config#L778>`__).

This behavior version might also require
that the dim tags of ``extern_data`` are properly defined.

See issue `#1140 <https://github.com/rwth-i6/returnn/issues/1140>`__.

Behavior version 12 (2022-01-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The TF batch norm default settings have been changed.
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

Broadcasting dims no longer match in TF :class:`CombineLayer` and others.
This was never needed, instead broadcasting happens in RETURNN automatically to non-existing dims.
To fix this, do not add any broadcasting dims.

See issue `#666 <https://github.com/rwth-i6/returnn/issues/666>`__.

Behavior version 10 (2021-12-07)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TF :class:`ConvLayer` use ``with_bias=True`` by default.

See issue `#787 <https://github.com/rwth-i6/returnn/issues/787>`__.

Behavior version 9 (2021-12-03)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TF :class:`ConvLayer`, :class:`PoolLayer` use ``auto_use_channel_first=True`` by default.

In principle, nothing should ever change due to this
when a config is correct in that nothing depends on the order of axes.
However, this is now introduced as a new behavior version
because older configs might depend on the order of axes.
With the other behavior changes, this is mostly disallowed though,
so when you make use of a higher behavior version anyway,
this should be safe.

Behavior version 8 (2021-11-30)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TF :class:`ConvLayer`, :class:`PoolLayer` and :class:`TransposedConvLayer`
require ``in_spatial_dims`` to be specified
when the input has more than one spatial dimension
(which implies that you perform 2D or 3D convolution or pooling).

This is required to make the order of the spatial axes well defined
because the input axes could have been reordered in any way before.
See issue `#594 <https://github.com/rwth-i6/returnn/issues/594>`__.

Usually, you would use :class:`Dim` to specify ``in_spatial_dims``.
However, to make the transition easier for this specific new behavior,
you can also use a string description for a dimension.
So example usages look like:

.. code-block:: python

    enc_dim = Dim(...)
    dec_dim = Dim(...)

    in_spatial_dims = (enc_dim, dec_tim)
    in_spatial_dims = ("T", "dim:16")
    in_spatial_dims = ("stag:encoder", "stag:decoder")

Behavior version 7 (2021-11-29)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For TF layers:
Do not allow to specify ``axes`` or ``axis`` arguments in a way that depends on the order of the axes.
E.g. things like ``axis="spatial:1"`` would not be allowed.

To fix this, use dimension tags, i.e. :class:`DimensionTag` instances.
To fix older configs without too much effort,
you might also want to use ``"stag:<name>"`` or ``"stag-single:<idx>:<name>"``
or ``"dim:<static-dim>"``.

Behavior version 6 (2021-11-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TF :class:`MergeDimsLayer` uses ``keep_order=True`` and does not allow ``keep_order=False``.
There never should be a reason to use ``keep_order=False`` anyway.
If you have that, just remove it.
If that causes any problems, there is probably some other issue in your config.

See issue `#654 <https://github.com/rwth-i6/returnn/issues/654>`__.

Behavior version 5 (2021-11-26)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For TF layers:
Any ``axis`` or ``axes`` argument in layers does not allow int values anymore.
Instead, use either a str like ``"F"`` or ``"stag:..."``
or use a :class:`DimensionTag` instance.

See issue `#773 <https://github.com/rwth-i6/returnn/issues/773>`__.

Behavior version 4 (2021-11-23)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For TF layers:
Broadcasting in all inputs simultaneously in layers and other ops
is not allowed anymore by default.
In all inputs simultaneously means that there is no input which has all common dimensions.

Layers can explicitly allow this by specifying ``out_shape``.
In case you stumble upon this, specify ``out_shape`` in the layer.

See :func:`validate_broadcast_all_sources`
and issue `#691 <https://github.com/rwth-i6/returnn/issues/691>`__.

Behavior version 3 (2021-11-08)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TF ``DotLayer``: disallow ``int`` axes descriptions, remove and change defaults.

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

For TF layers:
Disallow not specifying ``"from"`` in layer definition dictionaries,
thus making use of the hidden default ``"data"`` as layer input.

``"from"`` needs to be set explicitly now.
Set it to ``"data"`` or ``"data:data"`` or some other layer or ``()`` (empty).

See issue `#519 <https://github.com/rwth-i6/returnn/issues/519>`__.

