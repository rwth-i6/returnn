.. _api:

===
API
===

The main entry point is in :mod:`rnn`.
This will initialize an instance of :class:`Engine`
which has the high level logic to iterate through epochs.

For each GPU (or CPU), an instance of :class:`Device`
will be created which does all the calculation and the model update.

The model is an instance of :class:`Network.LayerNetwork`
and consists of multiple hidden layers of class :class:`NetworkBaseLayer.Layer`
and one or more output layers of class :class:`NetworkOutputLayer.OutputLayer`.

The model update code, i.e. the optimization methods such as SGD
are implemented in :class:`Updater`.


.. toctree::
	:glob:

	api/*
