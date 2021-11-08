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

