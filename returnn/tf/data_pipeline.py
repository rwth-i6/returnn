"""
TensorFlow data pipeline
========================

This module covers all related code to handle the data loading, preprocessing, chunking, batching
and related things in TensorFlow, i.e. the TensorFlow data pipeline from the Dataset.

Some related documents:

https://github.com/rwth-i6/returnn/issues/292 (new dataset pipeline)
https://www.tensorflow.org/guide/datasets
https://www.tensorflow.org/versions/r1.12/api_guides/python/threading_and_queues
https://www.tensorflow.org/guide/performance/overview
https://www.tensorflow.org/guide/performance/datasets
https://github.com/tensorflow/docs/tree/r1.10/site/en/api_docs/python/tf/contrib/data
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

The input data which (and optionally the targets) can be represented with tf.compat.v1.placeholder
and feed via feed_dict from TFCompat.v1.Session.run which does one train/eval/forward step.
In this case, any preprocessing such as chunking and batching must be done beforehand via Numpy.
This was the initial implementation and is also the standard implementation for the Theano backend.

This is not optimal because the tf.compat.v1.Session.run first has to copy the data from CPU to GPU
and then can do whatever it is supposed to do (e.g. one train step).
So copying and then the calculation is done in serial but it could be done in parallel
with some other method which we will discuss below.
Also the preprocessing could involve some more complex operations which could be slow
with Python + Numpy.
Also the chunk shuffling is more difficult to implement and would be slower compared to a pure TF solution.


Implementation via new tf.dataset API
-------------------------------------

Define ``def dataset_pipeline(context: InputContext) -> tf.data.Dataset``
in your config.
See :class:`DatasetDataProvider`.


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

from returnn.datasets.basic import Dataset, BatchSetGenerator
from returnn.tf.network import ExternData
import returnn.tf.compat as tf_compat
import returnn.tf.horovod as tf_horovod
from returnn.log import log


class DataProviderBase(object):
  """
  Base class which wraps up the logic in this class. See derived classes.
  """

  def __init__(self, extern_data, data_keys=None):
    """
    :param ExternData extern_data:
    :param set(str)|None data_keys:
    """
    self.coord = tf.train.Coordinator()
    self.extern_data = extern_data
    if data_keys is None:
      data_keys = extern_data.data.keys()
    self.data_keys = sorted(data_keys)  # type: typing.List[str]

  def start_threads(self, session):
    """
    Start threads.

    :param tf.compat.v1.Session session:
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

    :param tf.compat.v1.Session session:
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
  of a :class:`returnn.tf.network.TFNetwork`.
  It will run a background thread which reads the data from a dataset and puts it into a queue.
  """

  def __init__(self, tf_session, dataset, batches, enforce_min_len1=False, capacity=10, tf_queue=None,
               batch_slice=None, **kwargs):
    """
    :param tf.compat.v1.Session|tf.compat.v1.InteractiveSession tf_session:
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

  def start_threads(self, session):
    """
    Start the thread.

    :param tf.compat.v1.Session session:
    """
    thread = Thread(target=self._thread_main, name="DataProvider thread")
    thread.daemon = True  # Thread will close when parent quits.
    thread.start()
    self.thread = thread

  def stop_threads(self):
    """
    Stop the thread.
    """
    self.coord.request_stop()
    if self.thread:
      self._flush_all_data()
      self.thread.join()
      self.thread = None
    self.dataset.finish_epoch()

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
    from returnn.datasets.basic import Batch, shapes_for_batches
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
    from returnn.util.basic import slice_pad_zeros
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
      from returnn.util import better_exchook
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
    :param tf.compat.v1.Session|None session:
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


class InputContext(object):
  """
  This object will be passed to the dataset pipeline function
  (``dataset_pipeline`` in the config)
  and provides all relevant information, functions, dataset transformations.

  The initial design of this class was discussed here:
  https://github.com/rwth-i6/returnn/issues/292
  """

  def __init__(self, parent, extern_data, config, dataset_name, returnn_dataset, iterator):
    """
    :param DatasetDataProvider parent:
    :param ExternData extern_data:
    :param Config.Config config:
    :param str dataset_name: e.g. "train" or "dev"
    :param Dataset returnn_dataset:
    :param tensorflow.data.Iterator iterator:
    """
    self.parent = parent
    self.extern_data = extern_data
    self.config = config  # TODO do we need that? maybe pass horovod and other context explicitly?
    self.dataset_name = dataset_name
    self.returnn_dataset = returnn_dataset  # TODO this might be unset later (if the dataset lives in a separate proc)
    self.iterator = iterator

    self.num_dataset_producers = 1
    self.num_dataset_consumers = 1

    self.horovod_enabled = False
    self.horovod_rank = None
    self.horovod_size = None
    if tf_horovod.get_ctx(config=config):
      self.horovod_enabled = True
      self.horovod_rank = tf_horovod.get_ctx().rank()  # rank 0 is the chief
      self.horovod_size = tf_horovod.get_ctx().size()
      self.num_dataset_consumers = self.horovod_size
      raise NotImplementedError  # TODO...

    self.distributed_tf_enabled = False
    if config.is_true("distributed_tf"):
      self.distributed_tf_enabled = True
      raise NotImplementedError  # TODO...

    # These will be set after init.
    self.final_dataset = None  # type: typing.Optional[tf.data.Dataset]
    self.final_dataset_init_iterator_op = None  # type: typing.Optional[tf.Operation]

  def get_returnn_dataset(self, **kwargs):
    """
    :return: The RETURNN :class:`Dataset` instances wrapped in a :class:`tf.data.Dataset`.
      Note that in distributed TF, this dataset would only be used in the dataset loader worker.
      However, in all cases this will return some dataset. You are not allowed to read from it, though.
      A follow-up call to :func:`map_producer_to_consumer` will take care of this logic.
    :rtype: tensorflow.data.Dataset
    """
    assert not kwargs
    import os

    def generator():
      """
      :rtype: dict[str,numpy.ndarray]
      """
      assert self.parent.current_dataset_name, "current dataset name not set"
      returnn_dataset = self.parent.datasets[self.parent.current_dataset_name]
      assert returnn_dataset, "RETURNN dataset not loaded in this proc (pid %i)" % os.getpid()

      seq_idx = 0
      while returnn_dataset.is_less_than_num_seqs(seq_idx):
        self.parent.current_dataset_complete_frac = returnn_dataset.get_complete_frac(seq_idx)
        returnn_dataset.load_seqs(seq_idx, seq_idx + 1)

        res = {}  # type: typing.Dict[str,numpy.ndarray]
        for key_ in self.parent.data_keys:
          data_ = self.extern_data.data[key_]
          value = returnn_dataset.get_data(seq_idx, key_)
          res[key_] = value
          for axis_wo_b_, dim_ in enumerate(data_.shape):
            if dim_ is None:  # dynamic length -- need size info for it
              size_key_ = "size:%s:%i" % (key_, axis_wo_b_)
              res[size_key_] = value.shape[axis_wo_b_]
        yield res
        seq_idx += 1

      returnn_dataset.finish_epoch()

    output_types = {}  # type: typing.Dict[str,tf.DType]
    output_shapes = {}  # type: typing.Dict[str,tf.TensorShape]
    for key, data in self.extern_data.data.items():
      output_types[key] = tf.as_dtype(data.dtype)
      output_shapes[key] = tf.TensorShape(data.shape)  # not batch-shape
      for axis_wo_b, dim in enumerate(data.shape):
        if dim is None:  # dynamic length -- need size info for it
          size_key = "size:%s:%i" % (key, axis_wo_b)
          output_types[size_key] = tf.as_dtype(data.size_dtype)
          output_shapes[size_key] = tf.TensorShape([])  # scalar. will get batched later

    return tf.data.Dataset.from_generator(
      generator=generator,
      output_types=output_types,
      output_shapes=output_shapes)

  def get_default_max_seqs(self):
    """
    :return: batch size in number of seqs, used e.g. for padded_batch
    :rtype: int
    """
    assert self.config.has("max_seqs") and self.config.int("max_seqs", 0) > 0
    return self.config.int("max_seqs", 0)

  def padded_batch_dataset(self, dataset, drop_remainder=False):
    """
    :param tensorflow.data.Dataset dataset:
    :param bool drop_remainder: if True, we would have a static batch size
    :rtype: tensorflow.data.Dataset
    """
    # We could also use bucket_by_sequence_length.
    return dataset.padded_batch(
      batch_size=self.get_default_max_seqs(),
      padded_shapes=tf.compat.v1.data.get_output_shapes(dataset),
      drop_remainder=drop_remainder)

  def map_producer_to_consumer(self, dataset):
    """
    :param tensorflow.data.Dataset dataset:
    :rtype: tensorflow.data.Dataset
    """
    if self.horovod_enabled:
      raise NotImplementedError  # TODO
    if self.distributed_tf_enabled:
      raise NotImplementedError  # TODO
    # Otherwise this is a no-op.
    return dataset

  # noinspection PyMethodMayBeStatic
  def get_consumer_device(self):
    """
    :return: e.g. "/device:GPU:0"
    :rtype: str
    """
    # TODO this is probably incomplete
    import returnn.tf.util.basic
    if returnn.tf.util.basic.is_gpu_available_in_session():
      return "/device:GPU:0"
    return "/device:CPU:0"

  def prefetch_to_consumer_device(self, dataset):
    """
    This must be called on the consumer (trainer) worker,
    i.e. after :func:`map_producer_to_consumer`.

    :param tensorflow.data.Dataset dataset:
    :rtype: tensorflow.data.Dataset
    """
    from tensorflow.python.data.experimental import prefetch_to_device
    return prefetch_to_device(self.get_consumer_device())(dataset)

  def get_dataset_name(self):
    """
    :return: e.g. "train" or "dev"
    :rtype: str
    """
    return self.dataset_name

  def make_iterator_initializer(self, iterator):
    """
    :param tensorflow.data.Iterator iterator:
    :rtype: tf.Operation
    """
    assert self.final_dataset
    return iterator.make_initializer(self.final_dataset)


class DatasetDataProvider(DataProviderBase):
  """
  Use a :class:`tf.data.Dataset` as input.
  This will be used if ``dataset_pipeline`` is set in the config.
  See the discussion about the new dataset pipeline (https://github.com/rwth-i6/returnn/issues/292).

  Note that this has also a state: the current active dataset.
  """

  def __init__(self, extern_data, config, datasets=None):
    """
    :param ExternData extern_data:
    :param list[str]|dict[str,Dataset|None]|None datasets: e.g. ["train", "dev"]
    :param Config.Config config:
    """
    super(DatasetDataProvider, self).__init__(extern_data=extern_data)
    output_types = {}  # type: typing.Dict[str,tf.DType]
    output_shapes = {}  # type: typing.Dict[str,tf.TensorShape]
    for key, data in extern_data.data.items():
      output_types[key] = tf.as_dtype(data.dtype)
      output_shapes[key] = tf.TensorShape(data.batch_shape)
      for axis_wo_b, dim in enumerate(data.shape):
        if dim is None:  # dynamic length -- need size info for it
          size_key = "size:%s:%i" % (key, axis_wo_b)
          output_types[size_key] = tf.as_dtype(data.size_dtype)
          output_shapes[size_key] = tf.TensorShape([None])  # [Batch]
    self.iterator = tf_compat.v1.data.Iterator.from_structure(output_types=output_types, output_shapes=output_shapes)
    self.iterator_next_element = self.iterator.get_next()
    for key, data in extern_data.data.items():
      assert data.placeholder is None
      assert not data.size_placeholder
      data.placeholder = self.iterator_next_element[key]
      assert isinstance(data.placeholder, tf.Tensor), "next: %r" % (self.iterator_next_element,)
      data.size_placeholder = {}
      for axis_wo_b, dim in enumerate(data.shape):
        if dim is None:  # dynamic length
          size_key = "size:%s:%i" % (key, axis_wo_b)
          data.size_placeholder[axis_wo_b] = self.iterator_next_element[size_key]
          assert isinstance(data.size_placeholder[axis_wo_b], tf.Tensor), "next: %r" % (self.iterator_next_element,)
    extern_data.init_batch_info()

    dataset_pipeline_func = config.typed_value("dataset_pipeline")
    if dataset_pipeline_func in [None, True, 1]:  # allow None here, if this class is used explicitly
      dataset_pipeline_func = self._dataset_pipeline_default
    assert callable(dataset_pipeline_func), "dataset_pipeline in config is invalid"

    if datasets is None or not datasets:  # e.g. in distributed TF
      # We don't use them here. These will be used by the dataset loader producer workers.
      datasets = []
      if config.is_true("train"):
        datasets.append("train")
      if config.is_true("dev"):
        datasets.append("dev")
      if config.is_true("eval"):
        datasets.append("eval")
      if config.has("eval_datasets"):
        datasets.append(sorted(config.typed_value("eval_datasets", {}).keys()))
    if isinstance(datasets, (list, tuple)):
      datasets = {name: None for name in datasets}
    self.datasets = datasets  # type: typing.Dict[str,typing.Optional[Dataset]]
    self.contexts = {}  # type: typing.Dict[str,InputContext]
    for dataset_name, returnn_dataset in datasets.items():
      context = InputContext(
        dataset_name=dataset_name, returnn_dataset=returnn_dataset,
        iterator=self.iterator,
        config=config, parent=self, extern_data=extern_data)
      dataset = dataset_pipeline_func(context)
      assert isinstance(dataset, tf.data.Dataset)
      context.final_dataset = dataset
      context.final_dataset_init_iterator_op = context.make_iterator_initializer(self.iterator)
      self.contexts[dataset_name] = context

    self.current_dataset_reached_end = False
    self.current_dataset_complete_frac = 0.
    self.current_dataset_name = None  # type: typing.Optional[str]

  def set_current_dataset(self, dataset_name):
    """
    :param str dataset_name:
    """
    assert dataset_name in self.contexts
    self.current_dataset_name = dataset_name
    self.current_dataset_complete_frac = 0.
    self.current_dataset_reached_end = False

  def start_threads(self, session):
    """
    Start background threads.

    Currently this wil not actually start the background threads.
    All/any background threads of tf.data are started automatically when needed.

    However, this will initialize the TF dataset iterator.

    :param tf.compat.v1.Session session:
    """
    assert self.current_dataset_name
    init_op = self.contexts[self.current_dataset_name].final_dataset_init_iterator_op
    session.run(init_op)
    # TODO actually it might be nice to start them explicitly in advance...
    #  I think this is currently not possible though.
    #  With a custom final prefetcher (see comment in have_more_data), this would be possible.

  def stop_threads(self):
    """
    Stop background threads (e.g. prefetching).
    (Currently a no-op.)
    """
    # I don't think this is currently possible. See e.g.:
    # https://stackoverflow.com/questions/62148052/how-to-stop-background-thread-of-prefetchdataset
    # Anyway, maybe not relevant.
    # We just should make sure that any access on the RETURNN dataset is save.

  def have_more_data(self, session):
    """
    :param tf.compat.v1.Session session:
    :return: whether the next session.run() can run in the current epoch & dataset
    :rtype: bool
    """
    # we will just raise tf.errors.OutOfRangeError otherwise
    # TODO: horovod sync on this is likely broken then...
    # TODO we could also have sth like an own custom PrefetchDataset in between,
    #  which runs a background thread which always prefetches elements,
    #  and an extra function to check whether we reached the end
    #  (would block if not the case, and not prefetched yet).
    # See also have_reached_end.
    assert self.current_dataset_name
    return True

  def get_feed_dict(self, single_threaded=False):
    """
    :param bool single_threaded: whether to not use the queue (arg name is slightly misleading)
    :returns: batch,meta
    :rtype: dict[tf.Tensor,tf.Tensor],dict[str]
    """
    assert self.current_dataset_name
    assert not single_threaded
    return {}, {}

  def have_reached_end(self):
    """
    :rtype: bool
    """
    # If the last read on the iterator raised OutOfRange, this should return True.
    # In addition, we could check the dataset itself (might need IPC if it lives in another process...).
    assert self.current_dataset_name
    return self.current_dataset_reached_end

  def get_dataset_name(self):
    """
    :return: current dataset name, e.g. "train" or "dev"
    :rtype: str
    """
    assert self.current_dataset_name
    return self.current_dataset_name

  def get_complete_frac(self):
    """
    :return: by how much we are through the current dataset, number between 0 and 1, for visual feedback
    :rtype: float
    """
    # TODO ... this is somewhat tricky...
    #  we would need some IPC to the original RETURNN dataset if it lives in another process...
    #  we could also feed complete_frac as part of the data itself...
    # self.current_dataset_complete_frac can be set if the dataset lives in the same process
    return self.current_dataset_complete_frac

  # noinspection PyMethodMayBeStatic
  def _dataset_pipeline_default(self, context):
    """
    :param InputContext context:
    :rtype: tensorflow.data.Dataset
    """
    dataset = context.get_returnn_dataset()
    dataset = context.padded_batch_dataset(dataset)
    dataset = context.map_producer_to_consumer(dataset)
    dataset = context.prefetch_to_consumer_device(dataset)
    return dataset
