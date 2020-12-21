.. _recurrency:

==========
Recurrency
==========

Also see the slides of `our Interspeech 2020 tutorial about machine learning frameworks including RETURNN <https://www-i6.informatik.rwth-aachen.de/publications/download/1154/Zeyer--2020.pdf>`__
which explains the **recurrency** concept as well.

**Recurrency** :=
Anything which is defined by step-by-step execution,
where current step depends on previous step, such as RNN, beam search, etc.

This is all covered by :class:`returnn.tf.layers.rec.RecLayer`,
which is a generic wrapper around ``tf.while_loop``.
It covers:

* Definition of stochastic variables (the output classes itself but also latent variables)
  for either beam search or training (e.g. using ground truth values)
* Automatic optimizations

The recurrent formula is defined in a way as it would be used for recognition.
I.e. specifically you would define your output labels as stochastic variables,
and their probability distribution.
The automatic optimization will make this efficient for the case of training.


.. _recurrency_stochastic_vars:

Stochastic variables
--------------------

The layer to define them is :class:`returnn.tf.layers.rec.ChoiceLayer`.
The default behavior is:

* In training, it will just return the ground truth values.
* With search enabled (in recognition), it will do beam search.

For details on how beam search is implemented,
see :ref:`search`.


.. _recurrency_automatic_optimization:

Automatic optimization
----------------------

The definition of the recurrent formula can have parts
which are actually independent from the loop
-- maybe depending on the mode, e.g. in training.
**Automatic optimization** will find parts of the formula (i.e. sub layers)
which can be calculated independently from the loop,
i.e. outside of the loop.

All layers are implemented in a way that they perform the same mathematical calculation
whether they are inside the loop or outside.
