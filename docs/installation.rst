.. _installation:

============
Installation
============

Installation is easy.
Checkout the Git repository of RETURNN (https://github.com/rwth-i6/returnn/).
Install all dependencies, which are just Theano and h5py (:code:`pip install -r requirements.txt`).
For Theano usage, make sure that you have this in your ``~/.theanorc``::

    [global]
    device = cpu
    floatX = float32

For TensorFlow, use :code:`pip install tensorflow-gpu` and make sure that CUDA and cuDNN can be found.

See :ref:`basic_usage` for the basic usage.
