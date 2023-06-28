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

Code::

    import returnn.frontend as rf

    ...

Related work:

https://github.com/unifyai/ivy
