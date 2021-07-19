.. _loss:

==============
Loss Functions
==============

This is a list of all loss functions that can be used by adding ``"loss": "<class_name_of_loss>"`` to a layer.
Additional input parameters to the respective loss classes can be given via ``loss_opts``.
A scale for a loss can be set via ``loss_scale`` (also see :ref:`network_define_layers`)

If the output of a loss function is needed as a part of the network,
the ``LossLayer`` can be used in combination with one of the losses.

LossLayer
---------

.. autoclass:: returnn.tf.layers.basic.LossLayer
    :members:
    :undoc-members:

As-Is Loss
----------

.. autoclass:: returnn.tf.layers.basic.AsIsLoss
    :members:
    :undoc-members:

Binary Cross-Entropy Loss
-------------------------

.. autoclass:: returnn.tf.layers.basic.BinaryCrossEntropyLoss
    :members:
    :undoc-members:

Bleu Loss
---------

.. autoclass:: returnn.tf.layers.basic.BleuLoss
    :members:
    :undoc-members:

Cross-Entropy Loss
------------------

.. autoclass:: returnn.tf.layers.basic.CrossEntropyLoss
    :members:
    :undoc-members:

CTC Loss
--------

.. autoclass:: returnn.tf.layers.basic.CtcLoss
    :members:
    :undoc-members:

Deep Clustering Loss
--------------------

.. autoclass:: returnn.tf.layers.basic.DeepClusteringLoss
    :members:
    :undoc-members:

Edit Distance Loss
------------------

.. autoclass:: returnn.tf.layers.basic.EditDistanceLoss
    :members:
    :undoc-members:

Expected Loss
-------------

.. autoclass:: returnn.tf.layers.basic.ExpectedLoss
    :members:
    :undoc-members:

Extern Sprint Loss
------------------

.. autoclass:: returnn.tf.layers.basic.ExternSprintLoss
    :members:
    :undoc-members:

Fast Baum Welch Loss
--------------------

.. autoclass:: returnn.tf.layers.basic.FastBaumWelchLoss
    :members:
    :undoc-members:

Generic Cross-Entropy Loss
--------------------------

.. autoclass:: returnn.tf.layers.basic.GenericCELoss
    :members:
    :undoc-members:

Mean-L1 Loss
-----------------------

.. autoclass:: returnn.tf.layers.basic.MeanL1Loss
    :members:
    :undoc-members:

Mean-Squared-Error Loss
-----------------------

.. autoclass:: returnn.tf.layers.basic.MeanSquaredError
    :members:
    :undoc-members:

L1 Loss
-------

.. autoclass:: returnn.tf.layers.basic.L1Loss
    :members:
    :undoc-members:

Sampling-Based Loss
--------------------

.. autoclass:: returnn.tf.layers.basic.SamplingBasedLoss
    :members:
    :undoc-members:

Triplet Loss
------------

.. autoclass:: returnn.tf.layers.basic.TripletLoss
    :members:
    :undoc-members:

Via Layer Loss
--------------

.. autoclass:: returnn.tf.layers.basic.ViaLayerLoss
    :members:
    :undoc-members:





