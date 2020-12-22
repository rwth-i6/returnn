.. _training:

========
Training
========

This is a summary and overview of all relevant training aspects.


.. _training_loss:

Loss
----

Training a neural network is usually done by gradient descent on a differentiable loss function
(differentiable w.r.t. the model parameters),
e.g. cross entropy, mean squared error or CTC.

You can define one or multiple losses in your network.
(See :ref:`network` and :ref:`loss` about how to define them.)

You can define any custom calculation as a loss.
A loss could be used for supervised training (i.e. you have some ground truth)
or unsupervised training,
or anything else
(just some auxiliary loss, or just for regularization,
or not used for the training itself but just for evaluation).

A loss would only be used for training or evaluation
but usually not in recognition.


.. _training_regularization:

Regularization
--------------

* Extra loss terms

  * L2
  * Other auxiliary losses (supervised or unsupervised)

* Model variations

  * Dropout
  * Variational param noise
  * Stochastic depth
  * Data augmentation (e.g. SpecAugment) ≡ extra layers on input

* Directly in the network definition
* Used in training only, or flexible / configurable

See also :ref:`regularization_layers`.


.. _training_optimizer:

Optimizer
---------

The default is stochastic gradient descent (SGD),
but Adam is also very common.

See :ref:`optimizer` and :ref:`optimizer_settings` for more.

Also very relevant is the learning rate,
and also the learning rate scheduling (see :ref:`lr_scheduling`).


.. _training_param_init:

Parameter initialization
------------------------

Most layers support to configure this via options like ``forward_weights_init``
(e.g. :class:`returnn.tf.layers.basic.LinearLayer`).
This will use the generic function :func:`returnn.tf.util.basic.get_initializer`.

You can also preload some existing weights, e.g. via ``preload_from_files``.
See :ref:`model_loading`.


.. _training_lr_scheduling:

Learning rate scheduling
------------------------

Common features:

* Learning rate warmup (start small, increase, either linearly or exponentially)
* Constant phase
* Decay

  * Exponentially or inverse square root, or other variation
  * With constant rate, or depending on cross-validation score (sometimes called "Newbob")

See :ref:`learning_rate_scheduling_settings`.


.. _training_scheduling:

Generic Scheduling
------------------

Not only the learning rate can be scheduled, but many other aspects as well, such as:

* Regularization (e.g. disable dropout initially, or have lower values)
* Curriculum learning (i.e. take only an "easy" subset of training data initially, e.g. only the short sequences)


.. _pretraining:

Pretraining
-----------

Pretraining can be understood as a phase before the main training,
just to get the model parameters to a good starting point
(despite parameter initialization).

* Maybe a different loss during pretraining (e.g. unsupervised or custom)

* Maybe train only a subset of the parameters

* Different network topology every epoch, e.g. start with one layer, add more and more

* Automatically copies over parameters from one epoch to the next as far as possible

  * Configurable
  * New weights are newly initialized (e.g. randomly, see :ref:`training_param_init`)
  * If dimension increased, can copy over existing weights (grow in width / dim.)

See also :ref:`advanced_pretraining` or :ref:`configuration_pretraining`.

Pretraining can be generalized to any custom training pipeline.
See :ref:`custom_train_pipeline`.


.. _custom_train_pipeline:

Custom training pipeline
------------------------

This can be seen as a generalization of pretraining (see :ref:`pretraining`).

Example:

1. Train small NN using frame-wise cross-entropy with linear alignment
2. Calculate new alignment
3. Train NN using frame-wise cross-entropy with new alignment
4. Repeat with calculating new alignment (maybe increase NN size)

Example:

1. Train CTC model with CTC loss
2. Calculate new alignment
3. Train NN (e.g. transducer) using frame-wise cross-entropy with new alignment

You define ``def get_network(epoch: int, **kwargs): ...`` in your config.


Multi-GPU training
------------------

See :ref:`multi_gpu`.
