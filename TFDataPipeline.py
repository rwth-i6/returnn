"""
TensorFlow data pipeline
========================

This module covers all related code to handle the data loading, preprocessing, chunking, batching
and related things in TensorFlow, i.e. the TensorFlow data pipeline from the Dataset.

Some related documents:

https://www.tensorflow.org/programmers_guide/reading_data
https://www.tensorflow.org/programmers_guide/threading_and_queues
https://www.tensorflow.org/performance/performance_models
https://www.tensorflow.org/api_guides/python/io_ops
https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/contrib/data
https://github.com/tensorflow/tensorflow/issues/4535
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/data/README.md
https://stackoverflow.com/questions/41187745/tensorflow-how-can-i-evaluate-a-validation-data-queue


Data shuffling
--------------

#. The sequence shuffling is implemented as part of the Dataset, although we could
   also use a tf.RandomShuffleQueue on sequence level for training.

#. Chunk shuffling can be done with another tf.RandomShuffleQueue for training.

#. Frame shuffling only makes sense for non-recurrent networks.
   It also only makes sense if we already did any windowing beforehand.
   It also only makes sense in training.
   In that case, we could do chunking just with chunk size 1 and chunk step 1,
   or maybe chunk size = context window size, and then the frame shuffling is just chunk shuffling,
   thus we do not need any separate frame shuffling logic.


Generic pipeline
----------------

#. We initialize the Dataset for some epoch.
   The Dataset could shuffle the sequence order based on the epoch.
   Then we iterate over the sequences/entries of the dataset (seq_idx),
   starting with seq_idx = 0, checking Dataset.is_less_than_num_seqs.
   We get the data for each data key (e.g. "data" or "classes") as Numpy arrays
   with any shape. They don't need to have the same number of time frames,
   although in the common case where we have a frame-wise alignment of class labels, they would have.
   In any case, we basically have dict[str,numpy.ndarray], the seq_idx and also a seq_tag.

#. We could implement another sequence shuffling with tf.RandomShuffleQueue in training.

#. We do chunking of the data of each sequence, i.e. selecting chunks of chunk size frames,
   iterating over the frames of a sequence with stride = chunk step.
   While doing chunking, we could add more context around each chunk (zero padding at the borders) if needed,
   e.g. if we use windowing or convolution with padding="valid",
   such that the output of the net will match that of the targets.
   However, this depends on where the data is used in the network; maybe it is used at multiple points?

#. We can do chunk shuffling with another tf.RandomShuffleQueue in training.

#. We build up batches from the chunks.


First the simple method via feed_dict and placeholders
------------------------------------------------------

This is implemented in :class:`FeedDictDataProvider`.

The input data which (and optionally the targets) can be represented with tf.placeholder
and feed via feed_dict from tf.Session.run which does one train/eval/forward step.
In this case, any preprocessing such as chunking and batching must be done beforehand via Numpy.
This was the initial implementation and is also the standard implementation for the Theano backend.

This is not optimal because the tf.Session.run first has to copy the data from CPU to GPU
and then can do whatever it is supposed to do (e.g. one train step).
So copying and then the calculation is done in serial but it could be done in parallel
with some other method which we will discuss below.
Also the preprocessing could involve some more complex operations which could be slow
with Python + Numpy.
Also the chunk shuffling is more difficult to implement and would be slower compared to a pure TF solution.


Implementation via TF queues
----------------------------

In :class:`QueueDataProvider`.
This is currently incomplete.
Also, instead of finishing this, probably using the tf.dataset is the better approach.


Implementation via new tf.dataset API
-------------------------------------

This is planned, but not yet started.


Some use case
-------------

Conv net training. For every sequence, window around every frame for context.
Window must belong together, no unnecessary zero padding should be done introduced by chunking.
Thus, windowing must be done before chunking, or additional zero-padding must be added before chunking.
Then formulated differently: Chunking with step 1, output for a chunk is a single frame.
It also means that the windowing can not be part of the network because we will get only chunks there,
or the windowing makes only sense with padding="valid", otherwise we would get way too much zero-padding
also at the border of every chunk.
The same is true for convolution, pooling and others. I.e. padding in time should always be in "valid" mode.
If we feed in a whole sequence, must return the whole sequence, in recog, forwarding, search or eval.
With padding="valid", the output has less time frames, exactly context-size less frames.
Conv should use padding="valid" anyway to save computation time, and only explicitly pad zeros where needed.
In recog, the input-format is (batch, time + context_size, ...) which is zero-padded by context_size additional frames.
So, have context_size as an additional global option for the network
(could be set explicitly in config, or calculated automatically at construction).
When chunking for such case, we also should have chunks with such zero-paddings so that recog matches.
So, basically, a preprocessing step, before chunking, in both training and recog, is to add
zero-padding of size context_size to the input, then we get the output of the expected size.


Pipeline implementation
-----------------------

#. One thread which goes over the Dataset.
   No need for different training/eval queue, no random-shuffle-queue, seq-shuffling is done by the Dataset.
   Here we can also handle the logic to add the context_size padding to the input.
   Maybe use Dataset.iterate_seqs which gets us the offsets for each chunk.
   We can then just add the context_size to each.
   After that, chunking can be done (can be done in the same thread just at the final step).

#. Another thread TFBatchingQueue, which collects seqs or chunks and prepares batches.

It depends on whether the full network is recurrent or not.

"""

from __future__ import print_function

import sys
import typing
try:
  # noinspection PyCompatibility
  from Queue import Queue
except ImportError:
  # noinspection PyCompatibility,PyUnresolvedReferences
  from queue import Queue
from threading import Thread, Condition

import numpy
import tensorflow as tf
from tensorflow.python.ops.data_flow_ops import StagingArea

from Dataset import Dataset, BatchSetGenerator
from TFNetwork import ExternData, Data
from Util import NumbersDict
from Log import log


class PipeBase(object):
  """
  Abstract base class for a pipe.
  """

  def have_data_for_dequeue(self):
    """
    :return: if we can dequeue from us now without blocking
    :rtype: bool
    """
    raise NotImplementedError

  def have_incoming_data(self, dep_pipe_connector):
    """
    :param PipeConnectorBase dep_pipe_connector: will queue data to us
    :return: whether we have now or in the future data ready for dequeue
    :rtype: bool
    """
    if self.have_data_for_dequeue():
      return True
    while dep_pipe_connector.is_running():
      if self.have_data_for_dequeue():  # maybe in the mean-while we got some data
        return True
      # We don't know if the data from the pipe connector is enough so that we can ensure
      # that we will have data for dequeue.
      # We can only wait now and let the pipe connector do some more work until we know this.
      import time
      time.sleep(0.01)
    # The pipe connector does not have any data any more and also will not in the future.
    # Again. Maybe the dep_pipe_connector has pushed/queued it to us in the mean-while.
    if self.have_data_for_dequeue():
      return True
    # We now know for sure that there will be no data anymore, unless we restart some of the input connectors.
    return False


class PipeConnectorBase(object):
  """
  Base class for pipe connector.
  """

  def is_running(self):
    """
    E.g. for pipe_in/pipe_out model:
    If the pipe_in has data, we increase our counter by 1, then dequeue from pipe_in, do sth and queue to pipe_out,
    and only then decrease the counter again.
    Thus, if we return False, we have ensured that the pipe_out already has the data, or there is no data anymore.
    If we return True, we will ensure that we will push more data to pipe_out at some point.

    :return: counter > 0
    :rtype: bool
    """
    raise NotImplementedError


class DatasetReader(PipeConnectorBase):
  """
  Reads from Dataset into a queue.
  """

  SpecialKeys = ("seq_tag", "seq_idx", "epoch_end")

  def __init__(self, extern_data, dataset, coord, feed_callback,
               with_seq_tag=False, with_seq_idx=False, with_epoch_end=False):
    """
    :param ExternData extern_data:
    :param Dataset dataset:
    :param tf.train.Coordinator coord:
    :param (dict[str,numpy.ndarray|str|int])->None feed_callback:
    :param bool with_seq_tag:
    :param bool with_seq_idx:
    :param bool with_epoch_end:
    """
    self.extern_data = extern_data
    self.dataset = dataset
    self.coord = coord
    self.feed_callback = feed_callback
    self.with_seq_tag = with_seq_tag
    self.with_seq_idx = with_seq_idx
    self.with_epoch_end = with_epoch_end
    self.dict_keys = self._get_keys()
    self._is_running = None
    self.finished_iterating_seqs = False
    self.final_num_seqs = None

  def _get_keys(self):
    keys = set()
    for key, data in self.extern_data.data.items():
      if key in self.SpecialKeys:
        continue  # handled below
      keys.add(keys)
      for axis in data.get_axes_with_size():
        keys.add("%s/size%i" % (key, axis))
    if self.with_seq_tag:
      keys.add("seq_tag")
    if self.with_seq_idx:
      keys.add("seq_idx")
    if self.with_epoch_end:
      keys.add("epoch_end")
    return keys

  def get_dtype_for_key(self, key):
    """
    :param str key:
    :rtype: str
    """
    if key in self.extern_data.data:
      return self.extern_data.data[key].dtype
    if key == "seq_tag":
      return "string"
    if key == "seq_idx":
      return "int32"
    if key == "epoch_end":
      return "bool"
    if "/size" in key:
      return self.extern_data.data[key.split("/")[0]].size_dtype
    raise Exception("invalid key %r" % key)

  def get_shape_for_key(self, key):
    """
    :param str key:
    :return: shape without batch-dim
    :rtype: tuple[int | None]
    """
    if key in self.extern_data.data:
      return self.extern_data.data[key].shape
    if key in self.SpecialKeys:
      return ()
    if "/size" in key:
      return ()
    raise Exception("invalid key %r" % key)

  def get_queue_kwargs(self):
    """
    :rtype: dict[str,list[str]|list[tuple[int|None]]]
    """
    names = sorted(self.dict_keys)
    return {
      "names": names,
      "dtypes": [self.get_dtype_for_key(name) for name in names],
      "shapes": [self.get_shape_for_key(name) for name in names]}

  def _make_end_of_epoch_dict(self, seq_idx):
    d = {}
    for key in self.dict_keys:
      dtype = self.get_dtype_for_key(key)
      shape = self.get_shape_for_key(key)
      if dtype == "string":
        d[key] = ""
      else:
        d[key] = numpy.zeros(shape, dtype=dtype)
    if self.with_seq_idx:
      d["seq_idx"] = seq_idx  # type: int
    d["epoch_end"] = True
    return d

  def loop(self):
    """
    Main loop
    """
    with self.coord.stop_on_exception():
      try:
        self._is_running = True
        seq_idx = 0
        while True:
          if not self.dataset.is_less_than_num_seqs(seq_idx):
            self.finished_iterating_seqs = True
            self.final_num_seqs = seq_idx
            if self.with_epoch_end:
              self.feed_callback(self._make_end_of_epoch_dict(seq_idx=seq_idx))
            break
          if self.coord.should_stop():
            break
          d = {}  # type: typing.Dict[str, typing.Union[numpy.ndarray,int,str]]
          for key, data in self.extern_data.data.items():
            if key in self.SpecialKeys:
              continue  # handled below
            d[key] = self.dataset.get_data(seq_idx, key=key)  # type: numpy.ndarray
            for axis in data.get_axes_with_size():
              d["%s/size%i" % (key, axis)] = d[key].shape[axis]  # type: int
          if self.with_seq_tag:
            d["seq_tag"] = self.dataset.get_tag(seq_idx)  # type: str
          if self.with_seq_idx:
            d["seq_idx"] = seq_idx  # type: int
          if self.with_epoch_end:
            d["epoch_end"] = False
          self.feed_callback(d)
          seq_idx += 1
      finally:
        self._is_running = False

  def is_running(self):
    """
    :rtype: bool
    """
    import time
    while self._is_running is None:
      time.sleep(0.01)
    return self._is_running is True


class MakePlaceholders(object):
  """
  Helper to create TF placeholders.
  """

  def __init__(self, data_keys, extern_data, with_batch):
    """
    :param list[str] data_keys:
    :param ExternData extern_data:
    :param bool with_batch:
    """
    self.data_keys = data_keys
    self.extern_data = extern_data
    self.with_batch = with_batch
    self.names = list(data_keys)
    self.dtypes = [self.extern_data.data[key].dtype for key in self.names]
    self.placeholders = {
      key: tf.placeholder(**self.extern_data.data[key].get_placeholder_kwargs(with_batch=with_batch))
      for key in self.data_keys}
    for key in self.data_keys:
      for axis in self.extern_data.data[key].get_axes_with_size():
        name = "%s/size%i" % (key, axis)
        self.names += [name]
        self.dtypes += [self.extern_data.data[key].size_dtype]
        self.placeholders[name] = tf.placeholder(
          **self.extern_data.data[key].get_size_placeholder_kwargs(axis, with_batch=with_batch))

  def data_placeholders(self):
    """
    :rtype: dict[str,tf.Tensor]
    """
    return {key: self.placeholders[key] for key in self.names}

  def feed_dict(self, d):
    """
    :param dict[str,numpy.ndarray|str|int] d:
    :return: keys replaced by placeholders
    :rtype: dict[tf.placeholder,numpy.ndarray|str|int]
    """
    return {self.placeholders[key]: d[key] for key in self.names}


class TFDataQueues(PipeBase):
  """
  Generic queues which differ between train/eval queues.
  """

  def __init__(self, extern_data, capacity=100, seed=1, with_batch=False, enqueue_data=None):
    """
    :param ExternData extern_data: this specifies the data keys
    :param int capacity:
    :param int seed:
    :param bool with_batch: whether we have the batch-dim in input/output
    :param dict[str,tf.Tensor] enqueue_data: if provided, will be the input
    """
    self.extern_data = extern_data
    self.data_keys = extern_data.data.keys()
    self.with_batch = with_batch
    self.enqueue_data = enqueue_data

    # http://stackoverflow.com/questions/41187745/tensorflow-how-can-i-evaluate-a-validation-data-queue-multiple-times-during-tra/44067467#44067467
    # I.e. we need two separate queues, one for training (RandomShuffleQueue) and one for eval (FIFOQueue),
    # and switch between the dequeue via tf.cond.
    from TFUtil import cond, get_global_train_flag_placeholder
    self.train_flag = get_global_train_flag_placeholder()
    self.names = list(self.data_keys)
    self.dtypes = [self.extern_data.data[key].dtype for key in self.data_keys]
    self.shapes = {
      key: data.batch_shape if with_batch else data.shape
      for (key, data) in self.extern_data.data.items()}
    for key, data in self.extern_data.data.items():
      for axis in data.get_axes_with_size():
        self.shapes["%s/size%i" % (key, axis)] = (None,) if with_batch else ()

    self.enqueue_placeholders = None
    if not self.enqueue_data:
      self.enqueue_placeholders = {
        key: tf.placeholder(**self.extern_data.data[key].get_placeholder_kwargs(with_batch=with_batch))
        for key in self.data_keys}
      for key in self.data_keys:
        for axis in self.extern_data.data[key].get_axes_with_size():
          name = "%s/size%i" % (key, axis)
          self.names += [name]
          self.dtypes += [self.extern_data.data[key].size_dtype]
          self.enqueue_placeholders[name] = tf.placeholder(
            **self.extern_data.data[key].get_size_placeholder_kwargs(axis, with_batch=with_batch))
      self.enqueue_data = self.enqueue_placeholders

    # TF recommendation: capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
    self.capacity = capacity
    self.train_queue_min_after_dequeue = int(capacity * 0.8)
    self.train_queue = tf.RandomShuffleQueue(
      capacity=self.capacity, min_after_dequeue=self.train_queue_min_after_dequeue,
      names=self.names, dtypes=self.dtypes,
      seed=seed, name="train_queue")
    self.eval_queue = tf.FIFOQueue(
      capacity=self.capacity, names=self.names, dtypes=self.dtypes, name="eval_queue")
    self.train_queue_size = self.train_queue.size()
    self.eval_queue_size = self.eval_queue.size()
    self.dequeue_size_op = cond(
      self.train_flag,
      lambda: self.train_queue.size() - self.train_queue_min_after_dequeue,
      lambda: self.eval_queue.size())
    self.have_more_op = tf.greater(self.dequeue_size_op, 0, name="queue_have_more")
    self.one_more_enqueue_is_enough_op = tf.greater(self.dequeue_size_op, -1, name="queue_have_more")
    self.enqueue_op = cond(
      self.train_flag,
      lambda: self.train_queue.enqueue(self.enqueue_data),
      lambda: self.eval_queue.enqueue(self.enqueue_data),
      name="queue_enqueue")

  def _as_list(self, x):
    assert len(x) == len(self.names)
    return [x[key] for key in self.names]

  def _as_dict(self, x):
    """
    :param list[T] x:
    :rtype: dict[str,T]
    """
    assert len(x) == len(self.names)
    return dict(zip(self.names, x))

  def make_dequeue_op(self):
    """
    :rtype: dict[str,tf.Tensor]
    """
    from TFUtil import cond
    return self._as_dict(cond(
      self.train_flag,
      lambda: self._as_list(self.train_queue.dequeue()),
      lambda: self._as_list(self.eval_queue.dequeue()),
      name="queue_dequeue"))

  def have_more(self, tf_session):
    """
    :param tf.Session tf_session:
    """
    return tf_session.run(self.have_more_op)

  def have_data_for_dequeue(self):
    """
    :rtype: bool
    """
    return self.have_more(tf.get_default_session())

  def one_more_enqueue_is_enough(self):
    """
    :rtype: bool
    """
    tf_session = tf.get_default_session()
    return tf_session.run(self.one_more_enqueue_is_enough_op)

  def enqueue(self, tf_session, data=None):
    """
    :param tf.Session tf_session:
    :param dict[str,numpy.ndarray]|None data: needed iff self.with_feed_input
    """
    if self.enqueue_placeholders:
      assert data is not None
      tf_session.run(self.enqueue_op, feed_dict={
        self.enqueue_placeholders[key]: v
        for (key, v) in data.items()})
    else:
      assert data is None
      tf_session.run(self.enqueue_op)


class TFChunkingQueueRunner(PipeConnectorBase):
  """
  Implements chunking in pure TF.
  I.e. we get full sequences of varying lengths as input (from a queue),
  and we go over it with stride = chunk step,
  and extract a window of chunk size at each position,
  which we feed into the target queue.
  Optionally, for each chunk, we can add more frames (context window) around the chunk.
  """

  def __init__(self, extern_data, make_dequeue_op, target_queue,
               chunk_size=None, chunk_step=None, context_window=None,
               source_has_epoch_end_signal=False):
    """
    :param ExternData extern_data:
    :param ()->dict[str,tf.Tensor] make_dequeue_op:
    :param tf.QueueBase target_queue:
    :param int|None chunk_size:
    :param int|None chunk_step:
    :param int|NumbersDict|None context_window:
    :param bool source_has_epoch_end_signal:
    """
    from TFUtil import slice_pad_zeros
    default_key = extern_data.default_input
    if context_window is None:
      context_window = NumbersDict(0)
    elif isinstance(context_window, int):
      context_window = NumbersDict(broadcast_value=0, numbers_dict={default_key: context_window})
    assert isinstance(context_window, NumbersDict)
    if chunk_step is None:
      chunk_step = 1

    # This is basically a pure TF implementation of Dataset.iterate_seqs.
    # noinspection PyUnusedLocal
    def seq_loop_body(last_stop, last_op):
      """
      :param tf.Tensor last_stop: bool scalar
      :param tf.Operation|tf.Tensor last_op:
      :return: (cont, seq_idx, op), cont is whether to continue, checked by the condition
      :rtype: (tf.Tensor, tf.Operation)
      """
      with tf.control_dependencies([last_op]):
        seq_item = make_dequeue_op()
      assert default_key in seq_item
      assert default_key in extern_data.data
      assert extern_data.data[default_key].time_dim_axis_excluding_batch == 0
      default_data_seq_len = tf.shape(seq_item[default_key])[0]

      def get_context_window_chunk(seq_start=0):
        """
        :param int seq_start:
        :rtype: dict[str,tf.Tensor]
        """
        chunk = {}
        for key, data in extern_data.data.items():
          if "/size" in key:
            # will be corrected maybe below, copy here
            chunk[key] = seq_item[key]
        for key, data in extern_data.data.items():
          if "/size" in key:
            continue  # will be corrected maybe below
          assert key in seq_item
          if extern_data.data[key].time_dim_axis is None:
            chunk[key] = seq_item[key]
          else:
            assert extern_data.data[key].time_dim_axis == 0
            seq_end = tf.shape(seq_item[key])[0]
            if chunk_size is not None:
              seq_end = tf.minimum(seq_start + chunk_size, seq_end)
            chunk[key] = slice_pad_zeros(
              seq_item[key],
              begin=seq_start - context_window[key],
              end=seq_end + context_window[key])
            chunk["%s/size0" % key] = seq_end - seq_start + 2 * context_window[key]
        return chunk

      if chunk_size is None:
        # Chunking is not enabled. Just forward the whole sequence.
        return target_queue.enqueue(get_context_window_chunk())

      def chunk_loop_body(seq_start):
        """
        :param int seq_start:
        :rtype: int
        """
        target_queue.enqueue(get_context_window_chunk(seq_start=seq_start))
        return seq_start + chunk_step

      op = tf.while_loop(
        cond=lambda seq_start: tf.less(seq_start, default_data_seq_len),
        body=chunk_loop_body,
        loop_vars=[0],  # initial seq_start
        parallel_iterations=1, back_prop=False)
      if source_has_epoch_end_signal:
        stop = seq_item["epoch_end"]
      else:
        stop = False
      return stop, op

    _, self.loop_op = tf.while_loop(
      cond=(lambda stop, *args: tf.logical_not(stop)),
      body=seq_loop_body,
      loop_vars=[False, 0],  # stop, op
      parallel_iterations=1, back_prop=False)

  def is_running(self):
    """
    :rtype: bool
    """
    pass  # TODO...


class TFBatchingQueue(object):
  """
  Wrapper around tf.PaddingFIFOQueue with more control.
  Gets in data via TFDataQueues without batch-dim, and adds the batch-dim,
  according to batch_size and max_seqs.
  Output can be accessed via self.output_as_extern_data.
  This will represent the final output used by the network, controlled by QueueDataProvider.
  """

  def __init__(self, data_queues, batch_size, max_seqs, capacity=10):
    """
    :param TFDataQueues data_queues:
    :param int batch_size:
    :param int max_seqs:
    :param int capacity:
    """
    assert not data_queues.with_batch
    self.data_queues = data_queues
    self.batch_size = batch_size
    self.max_seqs = max_seqs
    self.shapes = {key: data.batch_shape for (key, data) in data_queues.extern_data.data.items()}
    for key, data in data_queues.extern_data.data.items():
      assert data.batch_dim_axis == 0, "batch-dim currently is always added at axis 0"
      for axis in data.get_axes_with_size():
        self.shapes["%s/size%i" % (key, axis)] = (None,)  # (batch,)
    self._tf_out_queue = tf.PaddingFIFOQueue(
      capacity=capacity, name="TFBatchingQueue",
      names=data_queues.names, dtypes=data_queues.dtypes,
      shapes=[self.data_queues.shapes[key] for key in data_queues.names])
    self._tf_batch_nums = tf.FIFOQueue(
      capacity=capacity, dtypes=[tf.int32], shapes=[()])
    self._cur_batch_num = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False, name="batch_num")
    self._cur_max_seq_len = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False, name="max_seq_len")
    self._tf_enqueue_loop_op = self._make_enqueue_loop_op()
    from TFUtil import Data
    self.output_as_extern_data = ExternData(
      default_input=data_queues.extern_data.default_input,
      default_target=data_queues.extern_data.default_target,
      data={key: Data(**data.get_kwargs()) for (key, data) in data_queues.extern_data.data.items()})
    self.batch_queue_size = self._tf_batch_nums.size()
    batch_dequeue_op = self._tf_out_queue.dequeue_up_to(n=self._tf_batch_nums.dequeue())
    self.last_seq_idx = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False, name="last_seq_idx")
    self.seq_counter = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False, name="seq_counter")
    self.batch_counter = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False, name="batch_counter")
    default_input_key = self.output_as_extern_data.default_input
    default_input_data = self.output_as_extern_data.data[default_input_key]
    last_batch_size = tf.shape(self.batch_dequeue_op[default_input_key])[default_input_data.batch_dim_axis]
    updates = [
      tf.assign(self.last_seq_idx, tf.maximum(self.last_seq_idx, tf.reduce_max(self.batch_dequeue_op["seq_idx"]))),
      tf.assign_add(self.seq_counter, last_batch_size),
      tf.assign_add(self.batch_counter, 1)]
    self._init_op = tf.variables_initializer([
      self._cur_batch_num, self._cur_max_seq_len,
      self.last_seq_idx, self.seq_counter, self.batch_counter])
    from TFUtil import post_control_dependencies
    self.batch_dequeue_op = post_control_dependencies(batch_dequeue_op, updates=updates)

  def _make_enqueue_op(self):
    """
    :return: (stop, op)
    :rtype: (tf.Tensor, tf.Operation)
    """
    default_key = self.data_queues.extern_data.default_input
    v = self.data_queues.make_dequeue_op()
    stop = v["epoch_end"]
    seq_len = tf.shape(v[default_key])[self.data_queues.extern_data.data[default_key].time_dim_axis_excluding_batch]
    cur_batch_num = self._cur_batch_num
    cur_max_seq_len = self._cur_max_seq_len
    cur_batch_num = tf.assign_add(cur_batch_num, tf.where(stop, 0, 1))
    cur_max_seq_len = tf.assign(cur_max_seq_len, tf.maximum(cur_max_seq_len, seq_len))

    def enqueue_cur():
      """
      :rtype: tf.Operation
      """
      return tf.group([
        self._tf_batch_nums.enqueue(cur_batch_num),
        tf.assign(cur_batch_num, 0),
        tf.assign(cur_max_seq_len, 0)])

    maybe_enqueue_batch_num_op = tf.cond(
      # if
      tf.greater_equal(cur_batch_num, self.max_seqs),
      # then
      enqueue_cur,
      # else
      lambda: tf.cond(
        # if
        tf.logical_and(
          tf.greater_equal(cur_batch_num, 2),
          tf.greater(cur_max_seq_len * cur_batch_num, self.batch_size)),
        # then
        lambda: tf.group([
          self._tf_batch_nums.enqueue(cur_batch_num - 1),
          tf.assign(cur_batch_num, 1),
          tf.assign(cur_max_seq_len, seq_len)]),
        # else
        lambda: tf.no_op()))
    maybe_enqueue_batch_num_op = tf.cond(stop, enqueue_cur, lambda: maybe_enqueue_batch_num_op)
    op = tf.group([
      cur_batch_num, cur_max_seq_len,
      maybe_enqueue_batch_num_op,
      self._tf_out_queue.enqueue(v)])
    return stop, op

  def _make_enqueue_loop_op(self):
    """
    This will be an endless loop as a TF op.
    """
    def loop_cond(last_stop, last_op):
      """
      :param tf.Tensor last_stop: bool
      :param tf.Tensor last_op: int
      :rtype: tf.Tensor
      """
      with tf.control_dependencies([last_op]):
        return tf.logical_not(last_stop)

    # noinspection PyUnusedLocal
    def body(last_stop, last_op):
      """
      :param tf.Tensor last_stop: bool
      :param tf.Tensor last_op: int
      :return: stop, op
      :rtype: (tf.Tensor,tf.Tensor)
      """
      with tf.control_dependencies([last_op]):
        stop, op = self._make_enqueue_op()
        with tf.control_dependencies([op]):
          return stop, tf.identity(last_op)

    return tf.while_loop(
      name="enqueue_loop",
      cond=loop_cond,
      body=body,
      loop_vars=[False, 0],  # stop, op
      parallel_iterations=1, back_prop=False)

  def loop(self, tf_session, coord):
    """
    :param tf.Session tf_session:
    :param tf.train.Coordinator coord:
    """
    with coord.stop_on_exception():
      tf_session.run(self._init_op)
      while True:
        tf_session.run(self._tf_enqueue_loop_op)


class QueueOutput(object):
  """
  Queue output
  """

  def get_data(self):
    """
    :rtype: dict[str,tf.Tensor]
    """
    pass  # TODO

  def have_data(self):
    """
    :rtype: bool
    """
    pass  # TODO


class CpuToDefaultDevStage(object):
  """
  Copy from CPU to the device (e.g. GPU) (if needed).
  """

  def __init__(self, input_data, names, dtypes, extern_data, data_keys):
    """
    :param dict[str,tf.Tensor] input_data:
    :param list[str] names: data_keys + extra info
    :param list[tf.DType|str] dtypes: corresponds to names
    :param ExternData extern_data:
    :param list[str] data_keys:
    """
    from TFUtil import post_control_dependencies

    # The device-scope when this gets called is the default device,
    # so everywhere where we want to do it on CPU, we have to specify it explicitly.
    # StagingArea can be used for async CPU->GPU transfer.
    # It will live on the current device by the current device scope, e.g. the GPU.
    self._tf_staging_area = StagingArea(names=names, dtypes=dtypes)

    with tf.device("/cpu:0"):
      self.staging_size = tf.Variable(0, trainable=False, dtype=tf.int32, name="staging_size")
      self._staging_size_init = tf.variables_initializer([self.staging_size])
      with tf.control_dependencies([tf.assign_add(self.staging_size, 1)]):
        self.stage_put_op = self._tf_staging_area.put(input_data)
      get_updates = [tf.assign_sub(self.staging_size, 1)]
    # This should run on the default device (GPU).
    self.stage_get_op = post_control_dependencies(self._tf_staging_area.get(), updates=get_updates)

    self.output_as_extern_data = ExternData(
      default_input=extern_data.default_input,
      default_target=extern_data.default_target,
      data={key: Data(**data.get_kwargs()) for (key, data) in data_keys})
    for key, data in self.output_as_extern_data.data.items():
      data.placeholder = self.stage_get_op[key]
      data.size_placeholder = {
        axis: self.stage_get_op["%s/size%i" % (key, axis)]
        for axis in data.get_axes_with_size()}

  def loop(self, parent, coord, session):
    """
    :param QueueDataProvider parent:
    :param tf.train.Coordinator coord:
    :param tf.Session session:
    """
    with coord.stop_on_exception():
      while parent.have_more_data(session=session):
        session.run(self.stage_put_op)


class DataProviderBase(object):
  """
  Base class which wraps up the logic in this class. See derived classes.
  """

  def __init__(self, extern_data, data_keys):
    """
    :param ExternData extern_data:
    :param set(str)|None data_keys:
    """
    self.coord = tf.train.Coordinator()
    self.extern_data = extern_data
    if data_keys is None:
      data_keys = extern_data.data.keys()
    self.data_keys = sorted(data_keys)  # type: typing.List[str]

  def start_threads(self):
    """
    Start threads.
    """
    raise NotImplementedError

  def stop_threads(self):
    """
    Stop threads.
    """
    raise NotImplementedError

  def have_more_data(self, session):
    """
    It is supposed to return True as long as we want to continue with the current epoch
    in the current dataset (train or eval).
    This is called from the same thread which runs the main computation graph (e.g. train steps).

    :param tf.Session session:
    :return: whether the next session.run() can run in the current epoch & dataset
    :rtype: bool
    """
    raise NotImplementedError

  def get_feed_dict(self, single_threaded=False):
    """
    Gets the feed dict for TF session run().
    Note that this will block if there is nothing in the queue.
    The queue gets filled by the other thread, via self.thread_main().

    :param bool single_threaded: whether to not use the queue
    :returns: We dequeue one batch from the queue and provide the data for all placeholders of our external data.
      Additionally, there can be some meta information.
    :rtype: dict[tf.Tensor,tf.Tensor],dict[str]
    """
    raise NotImplementedError

  def have_reached_end(self):
    """
    :returns: whether the current dataset says that we reached the end.
    :rtype: bool
    """
    raise NotImplementedError

  def get_dataset_name(self):
    """
    :return: current dataset name, e.g. "train" or "dev"
    :rtype: str
    """
    raise NotImplementedError

  def get_complete_frac(self):
    """
    :return: by how much we are through the current dataset, number between 0 and 1, for visual feedback
    :rtype: float
    """
    raise NotImplementedError


class FeedDictDataProvider(DataProviderBase):
  """
  This class will fill all the placeholders used for training or forwarding or evaluation etc.
  of a `TFNetwork.Network`.
  It will run a background thread which reads the data from a dataset and puts it into a queue.
  """

  def __init__(self, tf_session, dataset, batches, enforce_min_len1=False, capacity=10, tf_queue=None,
               batch_slice=None, **kwargs):
    """
    :param tf.Session|tf.InteractiveSession tf_session:
    :param Dataset dataset:
    :param BatchSetGenerator batches:
    :param bool enforce_min_len1:
    :param ExternData extern_data:
    :param set(str)|None data_keys:
    :param int capacity:
    :param TFDataQueues|None tf_queue:
    :param slice|None batch_slice: select a subset of the batches
    """
    super(FeedDictDataProvider, self).__init__(**kwargs)
    self.tf_session = tf_session
    self.dataset = dataset
    self.batches = batches
    self.enforce_min_len1 = enforce_min_len1
    self.batch_slice = batch_slice
    self.state_change_cond = Condition()
    self.queue = None  # type: typing.Optional[Queue]
    self.tf_queue = tf_queue
    if not self.tf_queue:
      self.queue = Queue(maxsize=capacity)
    self.thread = None  # type: typing.Optional[Thread]
    self.thread_finished = False
    self.cur_batch_idx = 0
    self.reached_end = False

  def start_threads(self):
    """
    Start the thread.
    """
    thread = Thread(target=self._thread_main, name="DataProvider thread")
    thread.daemon = True  # Thread will close when parent quits.
    thread.start()
    self.thread = thread

  def stop_threads(self):
    """
    Stop the thread.
    """
    if not self.thread:
      return
    self.coord.request_stop()
    self._flush_all_data()
    self.thread.join()

  def get_next_batch(self, consider_batch_slice):
    """
    This assumes that we have more data, i.e. self.batches.has_more().

    :param bool consider_batch_slice:
    :returns: batch-data-value-dict or None. if not consider_batch_slice, will never be None
    :rtype: dict[str,numpy.ndarray]|None
    """
    # See EngineUtil.assign_dev_data() for reference.
    cur_batch_idx = self.cur_batch_idx
    batch, = self.batches.peek_next_n(1)
    self.cur_batch_idx += 1
    if consider_batch_slice and self.batch_slice is not None:
      assert (self.batch_slice.start or 0) >= 0
      start = self.batch_slice.start or 0
      assert (self.batch_slice.step or 1) >= 1
      step = self.batch_slice.step or 1
      if cur_batch_idx < start:
        return None
      if self.batch_slice.stop is not None and cur_batch_idx >= self.batch_slice.stop:
        return None
      if step > 1 and (cur_batch_idx - start) % step != 0:
        return None
    from Dataset import Batch, shapes_for_batches
    assert isinstance(batch, Batch)
    # In Returnn with Theano, we usually have the shape (time,batch,feature).
    # In TensorFlow, the default is (batch,time,feature).
    # This is also what we use here, i.e. batch_dim_first=True.
    # This must match the Data specification in TFNetwork.ExternData.init_from_config().
    shapes = shapes_for_batches(
      [batch], data_keys=self.data_keys, extern_data=self.extern_data, enforce_min_len1=self.enforce_min_len1)
    data = {k: numpy.zeros(shape=shapes[k], dtype=self.extern_data.data[k].dtype)
            for k in self.data_keys if self.extern_data.data[k].dtype != "string"}
    # Numpy cannot handle "string" dtype. Just make it a list[str], which is what TF can handle.
    data.update({k: [""] * batch.num_slices
                 for k in self.data_keys if self.extern_data.data[k].dtype == "string"})
    data.update({"seq_idx": [-1] * batch.num_slices, "seq_tag": [""] * batch.num_slices})
    seq_lens = {k: numpy.zeros(shape=(shapes[k][0],), dtype=self.extern_data.data[k].size_dtype)
                for k in self.data_keys if self.extern_data.data[k].have_time_axis()}
    self.dataset.load_seqs(batch.start_seq, batch.end_seq)
    from Util import slice_pad_zeros
    with self.dataset.lock:
      for seq in batch.seqs:
        o = seq.batch_frame_offset
        q = seq.batch_slice
        length = seq.frame_length
        # input-data, input-index will also be set in this loop. That is data-key "data".
        for k in self.data_keys:
          # Some special cases first, such as "seq_idx" and "seq_tag".
          # See also :func:`TFNetwork.get_extern_data`.
          if k in ["seq_idx", "seq_tag"]:
            continue  # handled below. will always be added
          if k in self.extern_data.extra_added_keys:
            continue
          if self.extern_data.data[k].have_time_axis():
            if length.get(k) in [0, None]:
              continue
          v = self.dataset.get_data(seq.seq_idx, k)
          if self.extern_data.data[k].have_time_axis():
            v = slice_pad_zeros(v, begin=seq.seq_start_frame[k], end=seq.seq_end_frame[k])
            ls = v.shape[0]
            if ls != length[k]:
              raise Exception("got shape[0]: %i, expected: %i, start/end: %r/%r, seq_idx: %i, seq len: %r" % (
                ls, length[k], seq.seq_start_frame, seq.seq_end_frame, seq.seq_idx,
                self.dataset.get_seq_length(seq.seq_idx)))
            data[k][q, o[k]:o[k] + ls] = v
            seq_lens[k][q] = max(seq_lens[k][q], o[k] + ls)
          else:  # no time-axis
            data[k][q] = v
        data["seq_idx"][q] = seq.seq_idx
        data["seq_tag"][q] = self.dataset.get_tag(seq.seq_idx)
    for k in seq_lens.keys():
      data["%s_seq_lens" % k] = seq_lens[k]
    return data

  def _thread_main(self):
    try:
      import better_exchook
      better_exchook.install()

      while self.batches.has_more() and not self.coord.should_stop():
        enqueue_args = self.get_next_batch(consider_batch_slice=True)
        if enqueue_args is not None:
          if self.queue:
            self.queue.put(enqueue_args)
          else:
            self.tf_queue.enqueue(tf_session=self.tf_session, data=enqueue_args)
        with self.state_change_cond:
          self.state_change_cond.notifyAll()
        self.batches.advance(1)

      self.reached_end = not self.batches.has_more()

    except Exception as exc:
      print("Exception in DataProvider thread: %r" % exc, file=log.v1)
      sys.excepthook(*sys.exc_info())

    finally:
      with self.state_change_cond:
        self.thread_finished = True
        self.state_change_cond.notifyAll()

  def have_more_data(self, session):
    """
    :param tf.Session|None session:
    :rtype: bool
    :return: when we go through an epoch and finished reading, this will return False
    If this returns True, you can definitely read another item from the queue.
    Threading safety: This assumes that there is no other consumer thread for the queue.
    """
    with self.state_change_cond:
      while True:
        # First check if there is still data in the queue to be processed.
        if self.queue and not self.queue.empty():
          return True
        if self.tf_queue and self.tf_queue.have_more(self.tf_session):
          return True
        if self.thread_finished:
          return False
        if not self.thread.is_alive:
          return False
        # The thread is alive and working. Wait for a change.
        self.state_change_cond.wait()

  def _flush_all_data(self):
    """
    This is supposed to be called by the consumer thread after a call to coord.request_stop().
    The data provider thread (self.thread_main()) could currently block in the queue put if it was full.
    """
    while self.have_more_data(None):
      if self.queue:
        self.queue.get()
      else:
        raise NotImplementedError

  def get_feed_dict(self, single_threaded=False):
    """
    Gets the feed dict for TF session run().
    Note that this will block if there is nothing in the queue.
    The queue gets filled by the other thread, via self.thread_main().

    :param bool single_threaded: whether to not use the queue
    :returns: we dequeue one batch from the queue and provide it for all placeholders of our external data,
      and additionally return some meta information.
    :rtype: (dict[tf.Tensor,numpy.ndarray],dict[str])
    """
    if self.tf_queue:
      return {}  # not needed to feed anything, it gets it via the queues
    if single_threaded:
      assert self.batches.has_more()
      assert self.batch_slice is None
      output = self.get_next_batch(consider_batch_slice=False)
    else:
      output = self.queue.get()
    assert isinstance(output, dict)
    # The data itself.
    d = {
      self.extern_data.get_data(k).placeholder: output[k]
      for k in self.data_keys
      if k not in self.extern_data.extra_added_keys}
    # And seq lengths info.
    for k in self.data_keys:
      if k in self.extern_data.extra_added_keys:
        continue
      data = self.extern_data.get_data(k)
      for dim, len_placeholder in data.size_placeholder.items():
        if dim == 0:  # time-dim
          d[len_placeholder] = output["%s_seq_lens" % k]
        else:
          raise Exception(
            "dataset currently does not support variable shape in other dimensions than the first. "
            "dim=%i, placeholder=%r" % (dim, len_placeholder))
    return d, {"seq_idx": output["seq_idx"], "seq_tag": output["seq_tag"]}

  def get_dataset_name(self):
    """
    :rtype: str
    """
    return self.dataset.name

  def have_reached_end(self):
    """
    :rtype: bool
    """
    return self.reached_end

  def get_complete_frac(self):
    """
    :rtype: float
    """
    return self.batches.completed_frac()


class QueueDataProvider(DataProviderBase):
  """
  This class is supposed to encapsulate all the logic of this module and to be used by the TF engine.
  It gets the train and dev dataset instances.

  High-level (not differentiating between train/eval) queues:
  1. sequence queue (filled by the data from Dataset)
  2. chunk queue (filled by chunking, and maybe context window)
  3. batch queue (constructed batches from the chunks)
  4. staging area (e.g. copy to GPU)
  """

  def __init__(self, shuffle_train_seqs=False, **kwargs):
    """
    Creates the queues and connector instances (which will be the queue runners).
    Thus this will be created in the current TF graph,
    and you need to create a new instance of this class for a new TF graph.
    This is also only intended to be recreated when we create a new TF graph,
    so all other things must be created while it exists.
    """
    super(QueueDataProvider, self).__init__(**kwargs)

    # First we need to create the queues on each level because they can be created independently from each other.
    # Then we create the connectors, the queue runners, which do some operation and connect the queues.
    # We treat train and eval dataset separately.
    # The train dataset will use random-shuffling-queue at some places and also is supposed to run indefinitely.
    # There can be multiple eval datasets and it is supposed to go exactly over all sequences from one epoch,
    # thus we use only fifo-queues and count exactly when we are at the end.

    with tf.device("/cpu:0"):  # all on CPU, except the very last

      self.input_placeholders = MakePlaceholders(
        data_keys=self.data_keys, extern_data=self.extern_data, with_batch=False)

      # Queues for sequences, separated for train/eval dataset.
      if shuffle_train_seqs:
        self.train_seq_queue_min_after_dequeue = 8
        self.train_seq_queue = tf.RandomShuffleQueue(
          name="train_seq_queue",
          capacity=10,
          min_after_dequeue=self.train_seq_queue_min_after_dequeue,
          **self.extern_data.get_queue_args(with_batch_dim=False))
      else:
        self.train_seq_queue_min_after_dequeue = 0
        self.train_seq_queue = tf.FIFOQueue(
          name="train_seq_queue",
          capacity=10,
          **self.extern_data.get_queue_args(with_batch_dim=False))
      self.eval_seq_queue = tf.FIFOQueue(
        name="eval_seq_queue",
        capacity=10,
        **self.extern_data.get_queue_args(with_batch_dim=False))
      self.train_seq_queue_enqueue_op = self.train_seq_queue.enqueue(self.input_placeholders.data_placeholders())
      self.eval_seq_queue_enqueue_op = self.eval_seq_queue.enqueue(self.input_placeholders.data_placeholders())

      # Will have separate queues for train and eval.
      self.chunk_queue = TFDataQueues(
        extern_data=self.extern_data, with_batch=False,
        capacity=100)

      # This is both the final queue (tf.PaddingFIFOQueue) as well as the connector to it.
      # This also chooses between the train and eval queue.
      self.batch_queue = TFBatchingQueue(
        data_queues=self.chunk_queue,
        batch_size=5000, max_seqs=40,
        capacity=10)

      # Now we create the remaining connectors.
      self.train_chunker = TFChunkingQueueRunner(
        extern_data=self.extern_data,
        make_dequeue_op=self.train_seq_queue.dequeue,
        target_queue=self.chunk_queue.train_queue,
        chunk_size=50, chunk_step=25,
        context_window=5)
      self.eval_chunker = TFChunkingQueueRunner(
        extern_data=self.extern_data,
        make_dequeue_op=self.eval_seq_queue.dequeue,
        target_queue=self.chunk_queue.eval_queue,
        context_window=5)
      # The dataset reader will be created by self.init_dataset().

    # This does the async CPU->GPU copy. This prepares the final data used for the network.
    self.final_stage = CpuToDefaultDevStage(
      input_data=self.batch_queue.batch_dequeue_op,
      names=self.chunk_queue.names, dtypes=self.chunk_queue.dtypes,
      extern_data=self.extern_data, data_keys=self.data_keys)
    self.output = self.final_stage.output_as_extern_data

    self.cur_dataset_is_train = None  # type: typing.Optional[bool]
    self.cur_dataset_reader = None  # type: typing.Optional[DatasetReader]
    self._last_seq_idx = None

  def get_feed_dict(self, single_threaded=False):
    """
    :param bool single_threaded:
    :rtype: (dict,dict)
    """
    return {}, {}

  def _update_last_seq_idx(self, session):
    """
    :param tf.Session session:
    """
    self._last_seq_idx = session.run(self.batch_queue.last_seq_idx)

  def _last_seq_idx_at_end(self, session):
    """
    :param tf.Session session:
    """
    if self.cur_dataset_reader.final_num_seqs is None:
      return False
    self._update_last_seq_idx(session=session)
    return self._last_seq_idx + 1 >= self.cur_dataset_reader.final_num_seqs

  def have_more_data(self, session):
    """
    It is supposed to return True as long as we want to continue with the current epoch
    in the current dataset (train or eval).
    We want to continue if we still can do a next `self.final_stage.dequeue` op with the current dataset.
    This is called from the same thread which runs the main computation graph (e.g. train steps),
    as well as from the final stage thread.

    :param tf.Session session:
    :return: whether the next session.run() can run in the current epoch & dataset
    :rtype: bool
    """
    if not self.cur_dataset_reader:
      return False
    if not self.cur_dataset_is_train:  # only if not train, we can rely on the last_seq_idx
      is_at_end = self._last_seq_idx_at_end(session=session)
    else:
      is_at_end = False  # seqs/chunks can be shuffled in train
    if is_at_end:
      return False
    if self.cur_dataset_is_train:
      # There can be multiple RandomShuffleQueues which all need a certain buffer of min_after_dequeue.
      while True:
        if session.run(self.final_stage.staging_size) > 0:
          return True
        if session.run(self.batch_queue.batch_queue_size) > 0:
          return True
        if session.run(self.chunk_queue.train_queue_size) > self.chunk_queue.train_queue_min_after_dequeue:
          return True
        # if session.run() ?
      pass
    return True

  def have_reached_end(self):
    """
    :rtype: bool
    """
    pass  # TODO

  def get_complete_frac(self):
    """
    :rtype: float
    """
    pass  # TODO

  def init_dataset(self, session, dataset, is_train_dataset):
    """
    :param tf.Session session:
    :param Dataset dataset:
    :param bool is_train_dataset:
    """
    def feed_callback(d):
      """
      :param dict[str,numpy.ndarray|str|int] d:
      """
      if is_train_dataset:
        session.run(self.train_seq_queue_enqueue_op, feed_dict=self.input_placeholders.feed_dict(d))
      else:
        session.run(self.eval_seq_queue_enqueue_op, feed_dict=self.input_placeholders.feed_dict(d))

    assert not self.cur_dataset_reader
    self.cur_dataset_is_train = is_train_dataset
    self.cur_dataset_reader = DatasetReader(
      extern_data=self.extern_data, dataset=dataset, coord=self.coord, feed_callback=feed_callback)

  def get_dataset_name(self):
    """
    :rtype: str
    """
    assert self.cur_dataset_reader
    return self.cur_dataset_reader.dataset.name

  def get_threads(self):
    """
    TODO

    :return: threads
    """
    pass  # TODO

  def start_threads(self):
    """
    TODO
    """
    pass  # TODO

  def stop_threads(self):
    """
    TODO
    """
    pass  # TODO
