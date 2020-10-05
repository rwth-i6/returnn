.. _recurrent_subnet:

======================
Recurrent Sub-Networks
======================

For many task it will be necessary to define multiple layers that are applied as recurrent network over a sequential input,
especially when running a search over sequences.
While basic recurrent layers such as LSTM variants are defined by using the "``rec``" layer and selecting the desired
"``unit``", custom sub-networks can be defined by passing a network dictionary for the "``unit``" attribute.
The defined structure will then be applied for each position of the sequence.
As for the global network, an "``output``" layer is required to define which values will be the output of the subnet.
The layer outputs of the previous timesteps can be accessed by adding the prefix "``prev:``" to the layer names.
Static data from outside the subnet can be accessed via the layer prefix "``base:``".

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

Layers with recurrent dependencies and hidden states (e.g. LSTMs) can be added as "``rnn_cell``" layer.
For available cell units see :ref:`here <rec_units>`.
The number of steps is determined by the time axis of the input.
If multiple inputs are given, they will be concatenated on the feature axis.
Currently there is no support to access two layer outputs directly.
The concatenated data can be split by using a :class`SliceLayer <returnn.tf.layers.basic.SliceLayer>` on the feature axis.

**Recurrent net with independent step count:**

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
              "output_prob": {"class": "softmax", "from": ["decoder"], "target": target, "loss": "ce"}
          }
          "n_out": n_out
      }

The ``from`` attribute can be empty when using the output as a target.
The sequence length will then be determined by this target.

**Using Multiple Outputs**

Besides the default ``output`` layer, additional layers can be flaged as output layer.
When adding the parameter ``is_output_layer`` and setting it to ``True``,
the output of a sublayer can be accessed by using the pattern "``recurrent_layer/sublayer``".


