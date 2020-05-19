.. _debugging:

=========
Debugging
=========

Debugging neural networks is hard.
The worst situation is that training and inference works but the scores are just not good.
It's simpler if it does not even start because of some error.
Here I will outline several useful options, and methods in general.


.. _debug_interactive:

Interactive debugging
---------------------

Often just looking at the stack trace already makes it clear what the problem is,
esp with the additional information added via :mod:`better_exchook`.
However, sometimes it can be more helpful to interactively debug it,
i.e. to use an interactive shell.

There is the option ``debug_shell_in_runner`` to get a Python shell
directly in the main loop (over the steps / mini batches),
with all local variables available, and the ``feed_dict`` already prepared,
such that you can interactively run ``session.run`` on tensors, etc.

You can run RETURNN via::

  ipython --pdb rnn.py your-config.py

That will give you the IPython debugger shell once you hit an unhandled exception.
You can summon the interactive shell by explicitly calling the following from the
source code or from the config::

  import Debug
  Debug.debug_shell(user_ns=locals(), user_global_ns=globals(), exit_afterwards=False)


Shapes and :class:`Data`
------------------------

There is ``debug_print_layer_output_template`` which can always be enabled,
as it only prints additional information about the shape and :class:`Data`
for every layer at startup time, so it does not add any cost at runtime.
This is very helpful, as you can go through that information to double check
whether the output shape/type of each layer is as expected.
Most errors can be localized this way.

There is also ``debug_print_layer_output_shape`` which is only useful for debugging,
as it will print the output shape at runtime for every single step.


Runtime performance
-------------------

See :ref:`profiling`.


Getting nan/inf
---------------

There are various possible sources.
In general, you get these for calculations like x/0.0, log(0.0), ...

Use ``debug_add_check_numerics_on_output`` to enable runtime checks
after every layer. That will help you localize where it occurs.
This adds slightly to the memory requirements and also makes it slightly slower,
but it is still reasonably fast.

``debug_add_check_numerics_ops`` does the same, but for every single tensor.
This is usually too expensive.

Options like ``debug_grad_summaries`` or ``debug_save_updater_vars``
can also be helpful to localize e.g. a variable which explodes during training.
See monitoring.


Monitoring
----------

By default, RETURNN will dump all the losses and error information
to a TensorFlow event file.
This can be watched live (but also afterwards) via TensorBoard.
The default directory of this log dir is the same as the model dir,
but you can also configure it via ``tf_log_dir``.

You would go into this log dir, and then::

  tensorboard --logdir .


Bad scores
----------

There is no crash, no nan/inf, but you just get bad scores.
This is the hardest to debug case.
Maybe you have a bug somewhere but you don't know.

If you are reproducing some existing research,
and there is another existing implementation of it, this is a very good starting point.
You can try to reproduce the exact same model in RETURNN,
and write a model importer script which imports a trained model
from the existing other implementation over to your RETURNN model.
Now you can write a script where you feed in exactly the same input to both,
and compare hidden activations of each layer (or do some binary search).
That is a systematic way to verify that you have exactly the same.
You find a few such example scripts under ``tools/import-*``.

If you are playing with a new type of model,
it helps to first try it on some toy dataset, where you know that it must work
in principle.
If it does not, you can design the toy samples in a way that helps you
understand where it fails.
In the extreme case, in theory, you should even be able to set the neural network
weights by hand to solve the toy task.
If you don't know how, then maybe your model is actually not powerful enough.
If that works, you can make the toy task successively harder and more similar
to the real task.
If all the toy tasks work, but the real task still does not,
maybe you need some sort of curriculum learning or pretraining.

Think about ways to visualize some of the internals of your model.
E.g. for attention models, it helps to visualize the attention weights.
In many other cases, this can be hard, though.

Measure things. Whatever you think is in some way useful, or gives you a hint
whether it is doing the correct thing or not.


Python exception
----------------

RETURNN uses :mod:`better_exchook`
which will automatically provide an extended Python stack trace
which normally should provide enough information
to understand the problem and to fix it.
Maybe interactively debugging this can be helpful:
See :ref:`debug_interactive`.

If there is a bug in RETURNN itself (or might be):
In principle, a good way to work on a fix in a systematic way
is to create a simple test case which reproduces the problem.
Simplify further as much as possible
to identify and understand the real problem.
Then fix it.
Commit both the test case and the fix (pull request).


Crash
-----

E.g. segmentation fault (segfault, SIGSEGV).

RETURNN uses the :mod:`faulthandler` Python module
to provide a stack trace of the Python calls.

You can set the env var ``DEBUG_SIGNAL_HANDLER``,
which will load libSegFault.so.
