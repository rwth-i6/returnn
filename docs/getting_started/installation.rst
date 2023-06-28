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

You might want to use :code:`pip3` instead of :code:`pip`,
and you also might want to add the option :code:`--user`
(if you are not using ``virtualenv``).

For TensorFlow, use :code:`pip install tensorflow`,
and for PyTorch, use :code:`pip install torch torchdata torchaudio`.

For some specific datasets or special layers, additional dependencies might be needed,
such as ``librosa``.
For running the tests, you need ``pytest`` and ``nose``.

You can also install RETURNN as a framework, via ``pip`` (`PyPI entry <https://pypi.org/project/returnn/>`__),
like::

    pip install returnn

See :ref:`basic_usage` for the basic usage of RETURNN.
