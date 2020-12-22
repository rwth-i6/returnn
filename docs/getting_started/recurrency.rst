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

Note that there can be multiple stochastic variables.
Usually the output classes are one stochastic variable.
But there can be additional stochastic variables,
e.g. for the segment boundaries or time position in a hard attention model.

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

Example:

.. code-block:: python

    network = {
        "input": {"class": "rec", "unit": "nativelstm2", "n_out": 20},  # encoder
        "input_last": {"class": "get_last_hidden_state", "from": "input", "n_out":40},

        "output": {"class": "rec", "from": [], "target": "classes", "unit": {  # decoder
            "embed": {"class":"linear", "activation":None, "from":"output", "n_out":10},
            "s": {"class": "rec", "unit": "nativelstm2", "n_out": 20, "from": "prev:embed", "initial_state": "base:input_last"},
            "p": {"class":"softmax", "from":"s", "target": "classes", "loss": "ce"},
            "output": {"class":"choice", "from":"p", "target":"classes", "beam_size":8}
        }}
    }

In this example, in training:

- ``output`` is using the ground truth values, i.e. independent of anything in the loop, and can be moved out.
- ``embed`` depends on ``output``, which is moved out, so it can also be calculated outside the loop.
- ``s`` depends on ``embed``, which is moved out, so it can also be calculated outside the loop.
  Note that ``s`` has some internal state, and in fact needs to be calculated recurrently.
  But because it can be calculated independently from the loop, it can make use of **very efficient** kernels:
  In this case, it uses our ``NativeLstm2`` implementation.
- ``p`` depends on ``s``, and its loss calculation depends on the ground truth values,
  so it also can be calculated outside.
  This will result in a **very efficient** and parallel ``tf.matmul``.

With search enabled, in recognition:

``output`` depends on the probability distribution ``p``.
Effectively nothing can be moved out, because everything depends on each other.
This is still **as efficient as it possible can be**.
The ``output`` :class:`returnn.tf.layers.rec.ChoiceLayer` will use ``tf.nn.top_k`` internally.

This example also shows how one single definition of the network
can be used for both training and recognition,
and in a **very efficient** way.

Consider the `Transformer <https://arxiv.org/abs/1706.03762>`__ as another example.
The Transformer can be defined in a similar straight-forward way,
using ``output`` for the output labels with :class:`returnn.tf.layers.rec.ChoiceLayer`.
In training, it will result naturally in the standard fully parallel training.
In decoding, it is also as efficient as it possible can be.
