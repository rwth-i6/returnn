.. _recurrent_subnet:

======================
Recurrent Sub-Networks
======================

For many task it will be necessary to define multiple layers that are applied as recurrent network over a sequential input,
especially when running a search over sequences.
While basic recurrent layers such as LSTM variants are defined by using the ``rec`` layer and selecting the desired
``unit``, custom sub-networks can be defined by passing a network dictionary for the ``unit`` attribute.
The defined structure will then be applied for each position of the sequence.
As for the global network, an ``output`` layer is required to define which values will be the output of the subnet.
The layer outputs of the previous timesteps can be accessed by adding the prefix ``prev:`` to the layer names.
Static data from outside the subnet can be accessed via the layer prefix ``base:``.

Recurrent Nets with Fixed Step Count
====================================

Example of a recurrent "relu" layer:

.. code-block:: python

      {
          "class": "rec",
          "from": ["input"],
          "unit": {
            # Recurrent subnet here, operate on a single time-step:
            "output": {
              "class": "linear",
              "from": ["prev:output", "data:source"],
              "activation": "relu",
              "n_out": n_out},
          },
          "n_out": n_out,
      }
The number of steps is determined by the time axis of the input.
If multiple inputs are given, they will be concatenated on the feature axis.
Layers with recurrent dependencies and hidden states (e.g. LSTMs) can be added as ``rnn_cell`` layer.
For available cell units see :ref:`here <rec_units>`.
Currently there is no support to access two layer outputs directly.
The concatenated data can be split by using a :class`SliceLayer <returnn.tf.layers.basic.SliceLayer>` on the feature axis.

.. _recurrent_subnet_independent:

Recurrent Nets with Independent Step Count
==========================================

If the number of steps in the recurrent net should be determined by a condition
Example of an MLP-style attention mechanism with an LSTM layer:

.. code-block:: python

      {
          "class": "rec",
          "from": [],
          "unit": {
              "state_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["output"], "n_out": 128},
              "energy_in": {"class": "combine", "kind": "add", "from": ["base:enc_ctx", "s_transformed"], "n_out": 128},
              "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
              "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 128},
              "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},  # (B, enc-T, H)
              "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, H, V)
              "decoder": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:att"], "n_out": 256, 'target': 'data'},
              'stop_token': {'class': 'linear', 'activation': None, 'n_out': 1, 'loss': 'bin_ce',
                             'target': 'stop_token_target', 'from': ['output']},
              'stop_token_sigmoid': {'class': 'activation', 'activation': 'sigmoid', 'from': ['stop_token']},
              "end": {"class": "compare", "from": ["output"], "value": 0}
              "output_prob": {"class": "softmax", "from": ["decoder"], "target": "classes", "loss": "ce"},
              'output': {'class': 'choice', 'target': "classes", 'beam_size': 12, 'from': ["output_prob"], "initial_output": 0},
          }
          "n_out": n_out
      }

The ``from`` attribute can be empty when using the output as a target.
The sequence length will then be determined by this target during training,
and with an ``end``-layer during inference. The output of the ``end``-layer needs to be of shape ``[B]``
and of type boolean.
If the end-layer is not based on sparse data it is often required to use a
:class:`SqueezeLayer <returnn.tf.layers.basic.SqueezeLayer>` to remove the feature axis.

To enable beam-search, a :class:`ChoiceLayer <returnn.tf.layers.rec.ChoiceLayer>` must be used.
The choice layer will provide true labels as output during training.
During search, it will extend the batch dimension by the ``beam_size`` and manage the sequences in the beam.
For further information about the internals of beam-search, please have a look at :ref:`search`.

Additional Information
======================

**Using Multiple Outputs**

Besides the default ``output`` layer, additional layers can be flaged as output layer.
When adding the parameter ``is_output_layer`` and setting it to ``True``,
the output of a sublayer can be accessed by using the pattern ``recurrent_layer/sublayer``.

**Accessing Previous Time Steps**

By using the ``"prev:"``-prefix it is only possible to acces the layer outputs from previous time steps.
If a larger history needs to be accessed it is necessesary to use a
:class:`WindowLayer <returnn.tf.layers.basic.WindowLayer>`.
The parameter ``"window_size"`` can then be used to determine the number of previous steps
that need to be accessed.
The output will be of shape ``[B,window_size,D]``.
For steps outside the recurrency the layer will return zeros.



