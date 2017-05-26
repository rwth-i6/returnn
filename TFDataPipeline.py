"""
TensorFlow data pipeline
========================

This module covers all related code to handle the data loading, preprocessing, chunking, batching
and related things in TensorFlow, i.e. the TensorFlow data pipeline from the Dataset.

Some related documents:

https://www.tensorflow.org/programmers_guide/reading_data
https://www.tensorflow.org/programmers_guide/threading_and_queues
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
try:
  # noinspection PyCompatibility
  from Queue import Queue
except ImportError:
  # noinspection PyCompatibility
  from queue import Queue
from threading import Thread, Condition

import numpy
import tensorflow as tf

from Dataset import Dataset, Batch, BatchSetGenerator
from Log import log
from TFNetwork import ExternData
from Util import hms, NumbersDict


class DatasetReader(object):
  """
  Reads from Dataset into a queue.
  """
  def __init__(self, extern_data, dataset, coord, feed_callback, with_seq_tag=False, with_seq_idx=False):
    """
    :param ExternData extern_data:
    :param Dataset dataset:
    :param tf.train.Coordinator coord:
    :param feed_callback:
    :param bool with_seq_tag:
    :param bool with_seq_idx:
    """
    self.extern_data = extern_data
    self.dataset = dataset
    self.coord = coord
    self.feed_callback = feed_callback
    self.with_seq_tag = with_seq_tag
    self.with_seq_idx = with_seq_idx

  def loop(self):
    with self.coord.stop_on_exception():
      seq_idx = 0
      while True:
        if not self.dataset.is_less_than_num_seqs(seq_idx):
          break
        if self.coord.should_stop():
          break
        d = {}  # type: dict[str,numpy.ndarray|int|str]
        for key, data in self.extern_data.data.items():
          if key in ["seq_tag", "seq_idx"]:
            continue  # handled below
          d[key] = self.dataset.get_data(seq_idx, key=key)  # type: numpy.ndarray
          for axis in data.get_axes_with_size():
            d["%s/size%i" % (key, axis)] = d[key].shape[axis]  # type: int
        if self.with_seq_tag:
          d["seq_tag"] = self.dataset.get_tag(seq_idx)  # type: str
        if self.with_seq_idx:
          d["seq_idx"] = seq_idx  # type: int
        self.feed_callback(d)
        seq_idx += 1


class TFDataQueues(object):
  """
  Provides the data. Use this instead of feed_dict.
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
    self.train_queue = tf.RandomShuffleQueue(
      capacity=capacity, min_after_dequeue=int(capacity * 0.8),
      names=self.names, dtypes=self.dtypes,
      seed=seed, name="train_queue")
    self.eval_queue = tf.FIFOQueue(
      capacity=capacity, names=self.names, dtypes=self.dtypes, name="eval_queue")
    self.have_more_op = tf.greater(
      cond(self.train_flag, lambda: self.train_queue.size(), lambda: self.eval_queue.size()), 0,
      name="queue_have_more")
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

  def enqueue_end_epoch_signal(self, tf_session):
    """
    :param tf.Session tf_session:
    """
    dtypes = dict(zip(self.names, self.dtypes))
    feed_dict = {
      self.enqueue_data[key]: tf.zeros([(d or 0) for d in self.shapes[key]], dtype=dtypes[key])
      for key in self.names}


class TFChunkingQueue(object):
  """
  Implements chunking in pure TF.
  I.e. we get full sequences of varying lengths as input (from a queue),
  and we go over it with stride = chunk step,
  and extract a window of chunk size at each position,
  which we feed into the target queue.
  Optionally, for each chunk, we can add more frames (context window) around the chunk.
  """
  def __init__(self, extern_data, make_dequeue_op, target_queue, chunk_size, chunk_step, context_window):
    """
    :param ExternData extern_data:
    :param ()->dict[str,tf.Tensor] make_dequeue_op:
    :param tf.QueueBase target_queue:
    :param int|None chunk_size:
    :param int|None chunk_step:
    :param int|NumbersDict|None context_window:
    """
    default_key = "data"
    if context_window is None:
      context_window = NumbersDict(0)
    elif isinstance(context_window, int):
      context_window = NumbersDict(broadcast_value=0, numbers_dict={default_key: context_window})
    assert isinstance(context_window, NumbersDict)
    if chunk_step is None:
      chunk_step = 1

    # This is basically a pure TF implementation of Dataset.iterate_seqs.
    def seq_loop_body(last):
      seq_item = make_dequeue_op()
      assert default_key in seq_item
      assert default_key in extern_data.data
      assert extern_data.data[default_key].time_dim_axis_excluding_batch == 0
      default_data = seq_item[default_key]

      if chunk_size is None:
        # Chunking is not enabled. Just forward the whole sequence.
        return target_queue.enqueue(seq_item)

      def chunk_loop_body(seq_start):
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
            from TFUtil import slice_pad_zeros
            chunk[key] = slice_pad_zeros(
              seq_item[key],
              begin=seq_start - context_window[key],
              end=seq_start + chunk_size + context_window[key])
            chunk["%s/size0" % key] = chunk_size + 2 * context_window[key]

        target_queue.enqueue(chunk)
        return seq_start + chunk_step

      return tf.while_loop(
        cond=lambda seq_start: tf.less(seq_start - context_window[default_data], tf.shape(default_data)[0]),
        body=chunk_loop_body,
        loop_vars=[0],  # initial seq_start
        parallel_iterations=1, back_prop=False)

    self.loop_op = tf.while_loop(
      cond=lambda _: tf.constant(True),
      body=seq_loop_body,
      loop_vars=[0],  # initial seq_start
      parallel_iterations=1, back_prop=False)


class TFBatchingQueue(object):
  """
  Wrapper around tf.PaddingFIFOQueue with more control.
  Gets in data via TFDataQueues without batch-dim, and adds the batch-dim,
  according to batch_size and max_seqs.
  Output can be accessed via self.output_as_extern_data.
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
    self._init_op = tf.variables_initializer([self._cur_batch_num, self._cur_max_seq_len])
    self._tf_enqueue_loop_op = self._make_enqueue_loop_op()
    from TFUtil import Data
    self.output_as_extern_data = ExternData(
      default_input=data_queues.extern_data.default_input,
      default_target=data_queues.extern_data.default_target,
      data={key: Data(**data.get_kwargs()) for (key, data) in data_queues.extern_data.data.items()})
    dequeue_op = self._tf_out_queue.dequeue_up_to(n=self._tf_batch_nums.dequeue())
    for key, data in self.output_as_extern_data.data.items():
      data.placeholder = dequeue_op[key]
      data.size_placeholder = {
        axis: dequeue_op["%s/size%i" % (key, axis)]
        for axis in data.get_axes_with_size()}

  def _make_enqueue_op(self):
    default_key = self.data_queues.extern_data.default_input
    v = self.data_queues.make_dequeue_op()
    seq_len = tf.shape(v[default_key])[self.data_queues.extern_data.data[default_key].time_dim_axis_excluding_batch]
    cur_batch_num = self._cur_batch_num
    cur_max_seq_len = self._cur_max_seq_len
    cur_batch_num = tf.assign_add(cur_batch_num, 1)
    cur_max_seq_len = tf.assign(cur_max_seq_len, tf.maximum(cur_max_seq_len, seq_len))
    maybe_enqueue_batch_num_op = tf.cond(
      tf.greater_equal(cur_batch_num, self.max_seqs),
      lambda: tf.group([
        self._tf_batch_nums.enqueue(cur_batch_num),
        tf.assign(cur_batch_num, 0),
        tf.assign(cur_max_seq_len, 0)]),
      tf.cond(
        tf.logical_and(
          tf.greater_equal(cur_batch_num, 2),
          tf.greater(cur_max_seq_len * cur_batch_num, self.batch_size)),
        lambda: tf.group([
          self._tf_batch_nums.enqueue(cur_batch_num - 1),
          tf.assign(cur_batch_num, 1),
          tf.assign(cur_max_seq_len, seq_len)]),
        lambda: tf.no_op()))
    op = tf.group([
      maybe_enqueue_batch_num_op,
      self._tf_out_queue.enqueue(v)])
    return op

  def _make_enqueue_loop_op(self):
    """
    This will be an endless loop as a TF op.
    """
    def body(last):
      with tf.control_dependencies([last]):
        return self._make_enqueue_op()
    return tf.while_loop(
      cond=lambda _: tf.constant(True),
      body=body,
      loop_vars=[tf.no_op()],
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


class FeedDictDataProvider(object):
  """
  This class will fill all the placeholders used for training or forwarding or evaluation etc.
  of a `TFNetwork.Network`.
  It will run a background thread which reads the data from a dataset and puts it into a queue.
  """

  def __init__(self, tf_session, dataset, batches, extern_data, data_keys=None, capacity=10, tf_queue=None):
    """
    :param tf.Session|tf.InteractiveSession tf_session:
    :param Dataset.Dataset dataset:
    :param BatchSetGenerator batches:
    :param ExternData extern_data:
    :param set(str)|None data_keys:
    :param int capacity:
    :param TFDataQueues|None tf_queue:
    """
    self.tf_session = tf_session
    self.coord = tf.train.Coordinator()
    self.dataset = dataset
    self.batches = batches
    self.extern_data = extern_data
    if data_keys is None:
      data_keys = extern_data.data.keys()
    self.data_keys = sorted(data_keys)  # type: list[str]
    self.state_change_cond = Condition()
    self.queue = None  # type: Queue
    self.tf_queue = tf_queue
    if not self.tf_queue:
      self.queue = Queue(maxsize=capacity)
    self.thread = None  # type: Thread
    self.num_frames = NumbersDict(0)
    self.thread_finished = False
    self.reached_end = False

  def start_thread(self):
    thread = Thread(target=self.thread_main, name="DataProvider thread")
    thread.daemon = True  # Thread will close when parent quits.
    thread.start()
    self.thread = thread

  def stop_thread(self):
    if not self.thread:
      return
    self.coord.request_stop()
    self._flush_all_data()
    self.thread.join()

  def _get_next_batch(self):
    """
    :returns (batch-data-value-dict, batch-seq-lens)
    :rtype: (dict[str,numpy.ndarray], dict[str,numpy.ndarray])
    """
    # See EngineUtil.assign_dev_data() for reference.
    batch, = self.batches.peek_next_n(1)
    # In Returnn with Theano, we usually have the shape (time,batch,feature).
    # In TensorFlow, the default is (batch,time,feature).
    # This is also what we use here, i.e. batch_dim_first=True.
    # This must match the Data specification in TFNetwork.ExternData.init_from_config().
    shapes = self.dataset.shapes_for_batches([batch], data_keys=self.data_keys, batch_dim_first=True)
    data = {k: numpy.zeros(shape=shapes[k], dtype=self.extern_data.get_data(k).dtype)
            for k in self.data_keys}
    seq_lens = {k: numpy.zeros(shape=(shapes[k][0],), dtype=self.extern_data.get_data(k).size_dtype)
                for k in self.data_keys}
    self.dataset.load_seqs(batch.start_seq, batch.end_seq)
    self.num_frames += batch.get_total_num_frames()
    with self.dataset.lock:
      for seq in batch.seqs:
        o = seq.batch_frame_offset
        q = seq.batch_slice
        l = seq.frame_length
        # input-data, input-index will also be set in this loop. That is data-key "data".
        for k in self.data_keys:
          if l[k] == 0: continue
          v = self.dataset.get_data_slice(seq.seq_idx, k, seq.seq_start_frame[k], seq.seq_end_frame[k])
          ls = v.shape[0]
          if ls != l[k]:
            raise Exception("got shape[0]: %i, expected: %i, start/end: %r/%r, seq_idx: %i, seq len: %r" % (
              ls, l[k], seq.seq_start_frame, seq.seq_end_frame, seq.seq_idx, self.dataset.get_seq_length(seq.seq_idx)))
          data[k][q, o[k]:o[k] + ls] = v
          seq_lens[k][q] = max(seq_lens[k][q], o[k] + ls)
    return data, seq_lens

  def get_next_batch(self):
    data, seq_lens = self._get_next_batch()
    enqueue_args = data.copy()
    for k in data.keys():
      enqueue_args["%s_seq_lens" % k] = seq_lens[k]
    return enqueue_args

  def thread_main(self):
    try:
      import better_exchook
      better_exchook.install()

      while self.batches.has_more() and not self.coord.should_stop():
        enqueue_args = self.get_next_batch()
        if self.queue:
          self.queue.put(enqueue_args)
        else:
          self.tf_queue.enqueue(tf_session=self.tf_session, data=enqueue_args)
        with self.state_change_cond:
          self.state_change_cond.notifyAll()
        self.batches.advance(1)

      self.reached_end = not self.batches.has_more()

    except Exception as exc:
      print("Exception in DataProvider thread: %r" % exc)
      sys.excepthook(*sys.exc_info())

    finally:
      with self.state_change_cond:
        self.thread_finished = True
        self.state_change_cond.notifyAll()

  def have_more_data(self):
    """
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
    while self.have_more_data():
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
    :returns: we dequeue one batch from the queue and provide it for all placeholders of our external data
    :rtype: dict[tf.Tensor,tf.Tensor]
    """
    if self.tf_queue:
      return {}  # not needed to feed anything, it gets it via the queues
    if single_threaded:
      assert self.batches.has_more()
      output = self.get_next_batch()
    else:
      output = self.queue.get()
    assert isinstance(output, dict)
    # The data itself.
    d = {self.extern_data.get_data(k).placeholder: output[k] for k in self.data_keys}
    # And seq lengths info.
    for k in self.data_keys:
      data = self.extern_data.get_data(k)
      for dim, len_placeholder in data.size_placeholder.items():
        if dim == 0:  # time-dim
          d[len_placeholder] = output["%s_seq_lens" % k]
        else:
          raise Exception(
            "dataset currently does not support variable shape in other dimensions than the first. "
            "dim=%i, placeholder=%r" % (dim, len_placeholder))
    return d


class QueueDataProvider(object):
  def __init__(self, extern_data, coord, train_dataset):
    """
    :param ExternData extern_data:
    :param tf.train.Coordinator coord:
    :param Dataset train_dataset:
    """
    self.extern_data = extern_data
    self.coord = coord
    self.seq_queue = TFDataQueues(extern_data=extern_data, with_batch=False)
    self.train_dataset_reader = None

  def init_train_dataset(self, session, train_dataset):
    self.train_dataset_reader = DatasetReader(
      extern_data=self.extern_data, dataset=train_dataset, coord=self.coord,
      feed_callback=lambda d: self.seq_queue.enqueue(session, d))
