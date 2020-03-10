.. _pretraining:

===============================
Pre-Training / Dynamic Networks
===============================

RETURNN offers the possibility to dynamically change the network structure or configuration parameters during training.
This feature can be activated by setting the ``pretrain`` parameter to a dictionary containing the options.
The common use case is to use a function which edits the network as desired. An example would be:

.. code-block:: python

    def custom_construction_algo(idx, net_dict):
      """

      :param int idx: current index of the construction
      :param dict net_dict: network dictionary
      :return: the modified network
      """
      if idx == 5:
        return None

      stop_token_loss_scale = min(idx/5, 1.0)
      net_dict['decoder']['unit']['stop_token']['loss_scale'] = stop_token_loss_scale

      return net_dict

    pretrain = {"repetitions": 2, "construction_algo": custom_construction_algo}

This example assumes there is some arbitrary layer in a decoder recurrent network which defines a loss.
``repetitions`` defines how many epochs the same scheme (with the same index) is used.
The custom construction function sets the loss scale according to the current index. In this example, in epoch 1 and 2
the loss would be 0.0, in epoch 3 and 4 it would be 0.2, and so on.
Beginning from epoch 11, the default network would be used.
It is very important to return ``None`` for some index, otherwise the construction will be stuck in an infinite loop.

The custom construction function can also remove or add layers, or change layer options.
It is possible to train layers with smaller number of hidden units, and expand them during the training.

It is also possible to access the config parameters itself, which can be done with :code:`net_dict['#config']`.
The number of epoch repetitions can be made dynamic by setting :code:`net_dict['#repetitions']` for a given index.

For complex pre-training schemes, please have a look at the :ref:`ASR setups <asr>`
