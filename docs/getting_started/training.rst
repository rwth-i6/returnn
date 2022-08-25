.. _training:

========
Training
========

This is a summary and overview of all relevant training aspects.
See also :ref:`configuration_training`.


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

  * L2 (can be added to layer options, see :class:`returnn.tf.layers.base.LayerBase`)
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
E.g. ``optimizer = "adam"`` in your config.

Also very relevant is the learning rate,
and also the learning rate scheduling (see :ref:`training_lr_scheduling`).


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

* Reset learning rate on certain epochs, or increase again

Set ``learning_rate_control`` in your config.
Predefine certain learning rates (``learning_rates`` in config) for resets or warmup.
See :ref:`learning_rate_scheduling_settings`.


.. _training_scheduling:

Generic Scheduling
------------------

Not only the learning rate can be scheduled, but many other aspects as well, such as:

* Regularization (e.g. disable dropout initially, or have lower values)
* Curriculum learning (i.e. take only an "easy" subset of training data initially, e.g. only the short sequences)
* Apply gradient clipping only at the beginning of the training (see example below)

This can be done by overwriting config parameters using the :ref:`_pretraining` logic or
``get_network()`` (see :ref:`_custom_train_pipeline`).
In either case, parameters set under ``net_dict["#config"]`` will be used to overwrite existing config parameters.
Example::

    gradient_clip = 0


    def get_network(epoch: int, **kwargs):
        net_dict = ...
        if epoch < 5:
            net_dict["#config"]["gradient_clip"] = 10
        return net_dict


Batching and dataset shuffling
------------------------------

* How to build up individual mini-batches (their size, and the logic for that)

  * Batch size (``batch_size``, ``max_seqs``, ``max_seq_length``)
  * Chunking (``chunking``)

* How to shuffle the dataset (the sequences), or how to iterate through it

  * E.g. shuffle seqs, and sort buckets (``"laplace"``) to reduce padding

See :ref:`configuration_training`.

See :ref:`dataset` about how the dataset is loaded,
and how you can implement your own custom dataset.


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


Deterministic training
----------------------

See :ref:`deterministic_training`.
