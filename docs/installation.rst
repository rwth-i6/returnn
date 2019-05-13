.. _installation:

============
Installation
============

Installation is easy.
Checkout the Git repository of RETURNN (https://github.com/rwth-i6/returnn/).
Install all dependencies, which are just numpy, h5py,
and the backend you want to use (TensorFlow or Theano).
You can do so via::

    pip install -r requirements.txt

You probably want to use :code:`pip3` instead of :code:`pip`,
and you also might want to add the option :code:`--user`
(if you are not using ``virtualenv``).

For TensorFlow, use :code:`pip install tensorflow-gpu`
(:code:`pip3 install --user tensorflow-gpu`)
if you have a Nvidia GPU,
and make sure that CUDA and cuDNN can be found.

For Theano, only version 0.9 works (:code:`pip install theano==0.9`).
For Theano usage, make sure that you have this in your ``~/.theanorc``::

    [global]
    device = cpu
    floatX = float32

For some specific datasets or special layers, additional dependencies might be needed,
such as ``librosa``.
For running the tests, you need ``nose``.

You can also install RETURNN as a framework, via ``pip``, like::

    pip install returnn

See :ref:`basic_usage` for the basic usage of RETURNN.
