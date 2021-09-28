.. _profiling:

=========
Profiling
=========

Your model training (or inference) is too slow, or takes too much memory?
You should profile where the computing time or memory is spend.
This is less specific about RETURNN but more about TensorFlow,
so please refer to the TensorFlow documentation for more recent details.

Since Tensorflow 2.2, you can use the TensorFlow Profiler integrated in
TensorBoard.
Set the option ``store_tf_profile=True`` to record performance data for all `session.run` calls.
Then open TensorBoard in the model directory, where a "Profile" tab should appear.
The `official guide <https://www.tensorflow.org/guide/profiler#profiler_tools>`__
describes all available tools that are available here.

.. note::
    There is
    `a bug in TF 2.3 and earlier <https://github.com/tensorflow/tensorflow/issues/42123#issuecomment-675047711>`__
    where the "Memory Profile" view does not show a breakdown of tensors at peak memory usage.
    This was fixed in TF 2.4.


There is also the option ``store_metadata_mod_step`` which has the effect that
every Nth step, it will do the ``session.run`` with these additional options::

    run_metadata = tf.RunMetadata()
    session.run(
      ...,
      options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
      run_metadata=run_metadata)

That will be written to the TF event file,
so you can see additional information about runtime and memory usage in TensorBoard.
Also, it will write a timeline in Google Chrome trace format
(visit `chrome://tracing <chrome://tracing>`__ in Chrome and open that trace file).

See also this for further information:

* `Optimize TensorFlow performance using the Profiler <https://www.tensorflow.org/guide/profiler#profiler_tools>`__
* `TensorFlow Profiler and Advisor <https://github.com/tensorflow/tensorflow/blob/b2edbd5a640fb2f50989c5579a4cfe87d1fc675e/tensorflow/core/profiler/README.md>`__
* `TensorFlow Profile Model Architecture <https://github.com/tensorflow/tensorflow/blob/9590c4c32dd4346ea5c35673336f5912c6072bf2/tensorflow/core/profiler/g3doc/profile_model_architecture.md>`__
