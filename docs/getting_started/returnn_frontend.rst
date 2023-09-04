.. _returnn_frontend:

================
RETURNN frontend
================

This is a common interface to define your models using Python code,
very similar as in PyTorch.

This common interface supports multiple backends, namely:

* PyTorch
* TensorFlow layer dictionaries
* TensorFlow directly

Code:

.. code-block:: python

    import returnn.frontend as rf

    ...


Related work
------------

`Ivy <https://github.com/unifyai/ivy>`__
is both an ML transpiler and a framework,
currently supporting JAX, TensorFlow, PyTorch, and Numpy.

`Keras Core <https://keras.io/keras_core/>`__:
Keras for TensorFlow, JAX and PyTorch.
Also can wrap pure PyTorch modules directly in ``keras.Model``
(`example <https://twitter.com/fchollet/status/1697381832164290754>`__).


History
-------

https://github.com/rwth-i6/returnn/issues/1120
