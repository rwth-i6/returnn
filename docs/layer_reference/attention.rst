.. _attention_layers:

================
Attention Layers
================

Note that more specific attention layers are deprecated.
It is recommend to define the attention energy explicitly,
and then use :class:`returnn.tf.layers.rec.GenericAttentionLayer`.

Generic Attention Layer
-----------------------

.. autoclass:: returnn.tf.layers.rec.GenericAttentionLayer
    :members:
    :undoc-members:


Self-Attention Layer
--------------------

.. autoclass:: returnn.tf.layers.rec.SelfAttentionLayer
    :members:
    :undoc-members:


Concatenative Attention Layer
-----------------------------

**Deprecated**

.. autoclass:: returnn.tf.layers.rec.ConcatAttentionLayer
    :members:
    :undoc-members:

Dot-Product Attention Layer
---------------------------

**Deprecated**

.. autoclass:: returnn.tf.layers.rec.DotAttentionLayer
    :members:
    :undoc-members:

Gauss Window Attention Layer
----------------------------

**Deprecated**

.. autoclass:: returnn.tf.layers.rec.GaussWindowAttentionLayer
    :members:
    :undoc-members:

