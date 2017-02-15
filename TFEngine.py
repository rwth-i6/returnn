
from __future__ import print_function

import os
import sys
import time
from Queue import Queue
from threading import Thread, Condition

import numpy
import tensorflow as tf
from tensorflow.python.client import timeline

from Dataset import Batch, BatchSetGenerator
from Engine import Engine as TheanoEngine
from LearningRateControl import loadLearningRateControlFromConfig
from Log import log
from Network import LayerNetwork
from Pretrain import pretrainFromConfig
from TFNetwork import TFNetwork, ExternData
from TFUpdater import Updater
from Util import hms, NumbersDict


class DataProvider(object):
  """
  This class will fill all the placeholders used for training or forwarding or evaluation etc.
  of a `TFNetwork.Network`.
  It will run a background thread which reads the data from a dataset and puts it into a queue.
  """

  def __init__(self, tf_session, dataset, batches, extern_data, data_keys=None, capacity=10, have_fixed_batch_size=False):
    """
    :param tf.Session tf_session:
    :param Dataset.Dataset dataset:
    :param BatchSetGenerator batches:
    :param ExternData extern_data:
    :param set(str)|None data_keys:
    :param int capacity:
    """
    self.tf_session = tf_session
    self.coord = tf.train.Coordinator()
    self.dataset = dataset
    self.batches = batches
    self.extern_data = extern_data
    if data_keys is None:
      data_keys = extern_data.data.keys()
    self.data_keys = sorted(data_keys)
    self.state_change_cond = Condition()
    self.queue = None  # type: Queue
    self.tf_queue = None  # type: tf.FIFOQueue
    self._have_fixed_batch_size = have_fixed_batch_size
    if have_fixed_batch_size:
      # TODO... also cache this ....
      self.tf_queue = tf.FIFOQueue(capacity=capacity, **extern_data.get_queue_args(with_batch_dim=True))
    else:
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
    :param Dataset.Batch batch:
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
          self.tf_session.run(self.tf_queue.enqueue(enqueue_args))  # TODO cache op
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
        if self.tf_queue and self.tf_queue.size().eval(session=self.tf_session) > 0:  # TODO cache op
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

  def get_feed_dict(self, previous_feed_dict, single_threaded=False):
    """
    Gets the feed dict for TF session run().
    Note that this will block if there is nothing in the queue.
    The queue gets filled by the other thread, via self.thread_main().

    :param dict[tf.Tensor,tf.Tensor]|None previous_feed_dict:
    :param bool single_threaded: whether to not use the queue
    :returns: we dequeue one batch from the queue and provide it for all placeholders of our external data
    :rtype: dict[tf.Tensor,tf.Tensor]
    """
    if previous_feed_dict and self._have_fixed_batch_size:
      # We can reuse the same feed dict every time.
      # It is based on the tf_queue.dequeue() which is symbolic.
      assert self.tf_queue and not self.queue
      return previous_feed_dict
    if single_threaded:
      assert self.batches.has_more()
      output = self.get_next_batch()
    else:
      output = self.queue.get() if self.queue else self.tf_queue.dequeue()  # TODO cache dequeue op
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


class Runner(object):
  def __init__(self, engine, dataset, batches, train, eval=True):
    """
    :param Engine engine:
    :param Dataset.Dataset dataset:
    :param BatchSetGenerator batches:
    :param bool train: whether to do updates on the model
    :param bool eval: whether to evaluate (i.e. calculate loss/error)
    """
    self.engine = engine
    self.data_provider = DataProvider(
      tf_session=engine.tf_session, extern_data=engine.network.extern_data,
      data_keys=engine.network.used_data_keys,
      dataset=dataset, batches=batches)
    self._should_train = train
    self._should_eval = eval
    self.store_metadata_mod_step = False  # 500
    self.finalized = False
    self.num_steps = None
    self.device_crash_batch = None
    self.start_time = None
    self.elapsed = None
    self._results_accumulated = {}  # type: dict[str,float]  # entries like "cost:output" or "loss"
    self.results = {}  # type: dict[str,float]  # entries like "cost:output" or "loss"
    self.score = {}  # type: dict[str,float]  # entries like "cost:output"
    self.error = {}  # type: dict[str,float]  # entries like "error:output"
    self.stats = {}  # type: dict[str,float]  # entries like "stats:..."

    from Util import terminal_size
    terminal_width, _ = terminal_size()
    self._show_interactive_process_bar = (log.verbose[3] and (not log.verbose[5]) and terminal_width >= 0)

  def _get_fetches_dict(self):
    """
    :return: values and actions which should be calculated and executed in self.run() by the TF session for each step
    :rtype: dict[str,tf.Tensor|tf.Operation]
    """
    # Note that it is important that we do not recreate graph nodes for every call to this function.
    # Thus everything which we access here should be cached.
    d = {}
    for key in self.data_provider.data_keys:
      data = self.data_provider.extern_data.get_data(key)
      for dim, v in data.size_placeholder.items():
        d["size:%s:%i" % (key, dim)] = v
    if self._should_train or self._should_eval:
      # These values are cached internally and the graph nodes are created on the first call.
      d["loss"] = self.engine.network.get_objective()
      for layer_name, loss in self.engine.network.loss_by_layer.items():
        d["cost:%s" % layer_name] = loss
      for layer_name, error in self.engine.network.error_by_layer.items():
        d["error:%s" % layer_name] = error
    for layer in self.engine.network.layers.values():
      for k, v in layer.stats.items():
        d["stats:%s:%s" % (layer.name, k)] = v
    if self._should_train:
      assert self.engine.updater
      def callback_on_new():
        # Force a new check.
        self.engine._checked_uninitialized_vars = False
      d["optim_op"] = self.engine.updater.get_optim_op(callback_on_new=callback_on_new)
    d["summary"] = self.engine.get_all_merged_summaries()
    return d

  def _print_process(self, report_prefix, step, step_duration, eval_info):
    if not self._show_interactive_process_bar and not log.v[5]:
      return
    start_elapsed = time.time() - self.start_time
    complete = self.data_provider.batches.completed_frac()
    assert complete > 0
    total_time_estimated = start_elapsed / complete
    remaining_estimated = total_time_estimated - start_elapsed
    if log.verbose[5]:
      info = [
        report_prefix,
        "step %i" % step]
      if eval_info:  # Such as score.
        info += ["%s %s" % item for item in sorted(eval_info.items())]
      info += [
        "%.3f sec/step" % step_duration,
        "elapsed %s" % hms(start_elapsed),
        "exp. remaining %s" % hms(remaining_estimated),
        "complete %.02f%%" % (complete * 100)]
      print(", ".join(filter(None, info)), file=log.v5)
    elif self._show_interactive_process_bar:
      from Util import progress_bar
      progress_bar(complete, hms(remaining_estimated))

  def _print_finish_process(self):
    if self._show_interactive_process_bar:
      from Util import progress_bar
      progress_bar()

  def _get_target_for_key(self, key):
    """
    :param str key: e.g. "cost:output" where the last part is the layer name. or "loss"
    :return: target name which is the data-key in the dataset, e.g. "classes"
    :rtype: str
    """
    if ":" in key:
      layer = self.engine.network.layers[key.split(':')[-1]]
      if layer.target:
        return layer.target
    return self.engine.network.get_default_target()  # e.g. "classes"

  def _epoch_norm_factor_for_result(self, key):
    """
    :param str key: e.g. "cost:output"
    :return: factor to multiply with such accumulated values for the final epoch stats
    :rtype: float
    """
    target = self._get_target_for_key(key)
    # Default: Normalize by number of frames.
    return 1.0 / float(self.data_provider.num_frames[target])

  def _finalize(self, num_steps):
    """
    Called at the end of an epoch.
    :param int num_steps: number of steps we did for this epoch
    """
    assert not self.data_provider.have_more_data()
    assert self.data_provider.num_frames["data"] > 0
    results = {key: value * self._epoch_norm_factor_for_result(key)
               for (key, value) in self._results_accumulated.items()}
    self.results = results
    self.score = dict([(key,value) for (key, value) in results.items() if key.startswith("cost:")])
    self.error = dict([(key,value) for (key, value) in results.items() if key.startswith("error:")])
    self.num_steps = num_steps
    self.finalized = True

  def _step_seq_len(self, fetches_results, data_key):
    """
    :param dict[str,numpy.ndarray|None] fetches_results: results of calculations, see self._get_fetches_dict()
    :param str data_key: e.g. "classes"
    :return: the seq length of this batch
    :rtype: int
    """
    num_frames = numpy.sum(fetches_results["size:%s:0" % data_key])
    return num_frames

  def _collect_eval_info(self, fetches_results):
    """
    :param dict[str,numpy.ndarray|None] fetches_results: results of calculations, see self._get_fetches_dict()
    :return: dict for printing the step stats, see self._print_process(), e.g. {"cost:output": 2.3}
    :rtype: dict[str,float]
    """
    # See see self._get_fetches_dict() for the keys.
    keys = [k for k in fetches_results.keys() if k.startswith("cost:") or k.startswith("error:") or k == "loss"]

    # Accumulate for epoch stats.
    for key in keys:
      value = fetches_results[key]
      if key not in self._results_accumulated:
        self._results_accumulated[key] = value
      else:
        self._results_accumulated[key] += value

    # Prepare eval info stats for this batch run.
    eval_info = {}
    for key in keys:
      value = fetches_results[key]
      target = self._get_target_for_key(key)
      eval_info[key] = value / float(self._step_seq_len(fetches_results=fetches_results, data_key=target))

    # Add raw stats.
    for k, v in fetches_results.items():
      if k.startswith("stats:"):
        if v.ndim == 1:
          v = list(v)  # looks nicer in logs
        eval_info[k] = v
        self.stats[k] = v  # Always just store latest value.

    return eval_info

  def run(self, report_prefix):
    """
    :param str report_prefix: prefix for logging
    """
    sess = self.engine.tf_session
    logdir = os.path.dirname(self.engine.model_filename) or os.getcwd()
    logdir += "/%s" % self.data_provider.dataset.name
    if not self._should_train:  # like eval
      logdir += "-%i" % self.engine.epoch
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(sess.graph)
    run_metadata = tf.RunMetadata()

    # Not sure if this is the best thing to do for an evaluation but it's ok for now.
    # We could also set it to 0 for non train epochs.
    step_offset = self.engine.network.get_global_train_step(session=sess)

    coord = self.data_provider.coord

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    self.data_provider.start_thread()
    self.start_time = time.time()
    step = None
    try:
      # step is like mini-batch in our usual terminology
      step = 0
      fetches_dict = self._get_fetches_dict()
      # After get_fetches_dict, maybe some new uninitialized vars. Last check.
      self.engine.check_uninitialized_vars()
      feed_dict = None
      while self.data_provider.have_more_data():
        feed_dict = self.data_provider.get_feed_dict(previous_feed_dict=feed_dict)
        start_time = time.time()
        if self.store_metadata_mod_step and step % self.store_metadata_mod_step == 0:
          # Slow run that stores extra information for debugging.
          print('Storing metadata', file=log.v5)
          run_options = tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE)
          fetches_results = sess.run(
            fetches_dict,
            feed_dict=feed_dict,
            options=run_options,
            run_metadata=run_metadata)
          writer.add_summary(fetches_results["summary"], step + step_offset)
          writer.add_run_metadata(run_metadata, 'step_{:04d}'.format(step + step_offset))
          tl = timeline.Timeline(run_metadata.step_stats)
          timeline_path = os.path.join(logdir, 'timeline.trace')
          with open(timeline_path, 'w') as f:
            f.write(tl.generate_chrome_trace_format(show_memory=True))
        else:
          fetches_results = sess.run(fetches_dict, feed_dict=feed_dict)
          writer.add_summary(fetches_results["summary"], step + step_offset)

        eval_info = self._collect_eval_info(fetches_results=fetches_results)
        duration = time.time() - start_time
        self._print_process(report_prefix=report_prefix, step=step, step_duration=duration,
                            eval_info=eval_info)
        step += 1

      self._print_finish_process()

      if not self.data_provider.reached_end:
        raise Exception("Did not successfully reached the end of the dataset.")

      if self._should_train:
        final_global_train_step = self.engine.network.get_global_train_step(session=sess)
        assert step + step_offset == final_global_train_step

      self._finalize(num_steps=step)

    except KeyboardInterrupt:
      print("KeyboardInterrupt in step %r." % step)

    except BaseException as exc:
      print("Exception %r in step %r." % (exc, step), file=log.v1)
      sys.excepthook(*sys.exc_info())
      self.device_crash_batch = step

    finally:
      coord.request_stop()
      coord.join(threads)
      self.data_provider.stop_thread()
      self.elapsed = time.time() - self.start_time


class Engine(object):
  def __init__(self, config=None):
    """
    :param Config.Config|None config:
    """
    if config is None:
      from Config import get_global_config
      config = get_global_config()
    self.config = config
    self.devices_config = self._get_devices_config()
    self._check_devices()
    self.tf_session = None  # type: tf.Session
    self.updater = None  # type: Updater
    self._checked_uninitialized_vars = False
    self._merge_all_summaries = None
    self.dataset_batches = {}  # type: dict[str,BatchSetGenerator]
    self.train_data = None; " :type: Dataset.Dataset "
    self.start_epoch = None

  def finalize(self):
    self._close_tf_session()

  def _get_devices_config(self):
    """
    :rtype: list[dict[str]]
    """
    from Device import getDevicesInitArgs
    if not self.config.value("device", None):
      # Better default: Use GPU if available.
      from TFUtil import is_gpu_available
      if is_gpu_available():
        print("Device not set explicitly, and we found a GPU, which we will use.", file=log.v2)
        self.config.set("device", "gpu")
      else:
        print("Device not set explicitly, and no GPU found.", file=log.v2)
    return getDevicesInitArgs(self.config)

  def is_requesting_for_gpu(self):
    return any([d["device"].startswith("gpu") for d in self.devices_config])

  def _check_devices(self):
    from TFUtil import print_available_devices, is_gpu_available
    print_available_devices()
    assert len(self.devices_config) == 1, "multiple devices not supported yet for TF"
    if self.is_requesting_for_gpu():
      assert is_gpu_available(), "no GPU available"
    else:
      if is_gpu_available():
        print("Note: There is a GPU available but you have set device=cpu.", file=log.v2)

  @classmethod
  def _guess_requested_max_num_threads(cls):
    omp_num_threads = int(os.environ.get("OMP_NUM_THREADS") or 0)
    if omp_num_threads:
      # Minimum of 2 threads, should not hurt.
      return max(omp_num_threads, 2)
    return None

  def _close_tf_session(self):
    if self.tf_session:
      self.tf_session.close()
    self.tf_session = None

  def _make_tf_session(self):
    self._close_tf_session()
    opts = self.config.typed_value("tf_session_opts", {})
    assert isinstance(opts, dict)
    opts = opts.copy()
    # See options here:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto
    opts.setdefault("log_device_placement", False)
    opts.setdefault("device_count", {}).setdefault("GPU", 1 if self.is_requesting_for_gpu() else 0)
    num_threads = self._guess_requested_max_num_threads()
    if num_threads:
      opts.setdefault("intra_op_parallelism_threads", num_threads)
      opts.setdefault("inter_op_parallelism_threads", num_threads)
    print("Setup tf.Session with options %r ..." % opts, file=log.v2)
    config = tf.ConfigProto(**opts)
    # config.gpu_options.allow_growth=True
    self.tf_session = tf.Session(config=config)

  def _reset_graph(self):
    tf.reset_default_graph()
    self._checked_uninitialized_vars = False
    self._merge_all_summaries = None

  get_train_start_epoch_batch = TheanoEngine.get_train_start_epoch_batch
  config_get_final_epoch = TheanoEngine.config_get_final_epoch
  get_epoch_model = TheanoEngine.get_epoch_model
  epoch_model_filename = TheanoEngine.epoch_model_filename

  def get_epoch_model_filename(self):
    return self.epoch_model_filename(self.model_filename, self.epoch, self.is_pretrain_epoch())

  def get_epoch_str(self):
    return ("pretrain " if self.is_pretrain_epoch() else "") + "epoch %s" % self.epoch

  def is_pretrain_epoch(self):
    return self.pretrain and self.epoch <= self.pretrain.get_train_num_epochs()

  def is_first_epoch_after_pretrain(self):
    return self.pretrain and self.epoch == self.pretrain.get_train_num_epochs() + 1

  def get_eval_datasets(self):
    eval_datasets = {}; """ :type: dict[str,Dataset.Dataset] """
    for name, dataset in [("dev", self.dev_data), ("eval", self.eval_data)]:
      if not dataset: continue
      eval_datasets[name] = dataset
    return eval_datasets

  def save_model(self, filename):
    """
    :param str filename: full filename for model
    """
    print("Save model under %s" % (filename,), file=log.v4)
    self.network.save_params_to_file(filename, session=self.tf_session)

  def init_train_from_config(self, config, train_data, dev_data, eval_data):
    """
    :type config: Config.Config
    :type train_data: Dataset.Dataset
    :type dev_data: Dataset.Dataset | None
    :type eval_data: Dataset.Dataset | None
    """
    self.train_data = train_data
    self.dev_data = dev_data
    self.eval_data = eval_data
    self.start_epoch, self.start_batch = self.get_train_start_epoch_batch(config)
    self.batch_size = config.int('batch_size', 1)
    self.shuffle_batches = config.bool('shuffle_batches', True)
    self.update_batch_size = config.int('update_batch_size', 0)
    self.save_model_epoch_interval = config.int('save_interval', 1)
    self.save_epoch1_initial_model = config.bool('save_epoch1_initial_model', False)
    self.learning_rate_control = loadLearningRateControlFromConfig(config)
    self.learning_rate = self.learning_rate_control.defaultLearningRate
    self.initial_learning_rate = self.learning_rate
    self.pretrain_learning_rate = config.float('pretrain_learning_rate', self.learning_rate)
    self.final_epoch = self.config_get_final_epoch(config)  # Inclusive.
    self.max_seqs = config.int('max_seqs', -1)
    self.ctc_prior_file = config.value('ctc_prior_file', None)
    self.exclude = config.int_list('exclude', [])
    self.init_train_epoch_posthook = config.value('init_train_epoch_posthook', None)
    self.share_batches = config.bool('share_batches', False)
    self.seq_drop = config.float('seq_drop', 0.0)
    self.seq_drop_freq = config.float('seq_drop_freq', 10)
    self.max_seq_length = config.float('max_seq_length', 0)
    self.inc_seq_length = config.float('inc_seq_length', 0)
    if self.max_seq_length == 0:
      self.max_seq_length = sys.maxsize
    # And also initialize the network. That depends on some vars here such as pretrain.
    self.init_network_from_config(config)

  def init_network_from_config(self, config):
    """
    :param Config.Config config:
    """
    self.model_filename = config.value('model', None)
    self.pretrain = pretrainFromConfig(config)
    self.max_seqs = config.int('max_seqs', -1)

    epoch, model_epoch_filename = self.get_epoch_model(config)
    assert model_epoch_filename or self.start_epoch
    self.epoch = epoch or self.start_epoch

    if self.pretrain:
      # This would be obsolete if we don't want to load an existing model.
      # In self.init_train_epoch(), we initialize a new model.
      net_dict = self.pretrain.get_network_json_for_epoch(self.epoch)
    else:
      net_dict = LayerNetwork.json_from_config(config)

    self._init_network(net_desc=net_dict, epoch=self.epoch)

    if model_epoch_filename:
      print("loading weights from", model_epoch_filename, file=log.v2)
      self.network.load_params_from_file(model_epoch_filename, session=self.tf_session)

  def _init_network(self, net_desc, epoch=None):
    if epoch is None:
      epoch = self.epoch
    self._close_tf_session()
    self._reset_graph()
    # The new session will by default use the newly created default graph.
    self._make_tf_session()
    tf.set_random_seed(42)
    network = TFNetwork(rnd_seed=epoch)
    network.construct_from_dict(net_desc)
    network.initialize_params(session=self.tf_session)
    network.layers_desc = net_desc
    self.network = network
    if self.train_data:
      # Need to create new Updater because it has the learning_rate var which must be in the current graph.
      self.updater = Updater(config=self.config, tf_session=self.tf_session, network=network)
      self.updater.set_trainable_vars(network.get_trainable_params())
    network.print_network_info()

  def maybe_init_new_network(self, net_desc):
    if self.network.layers_desc == net_desc:
      return
    from Util import dict_diff_str
    print("reinit because network description differs. Diff:",
          dict_diff_str(self.network.layers_desc, net_desc), file=log.v3)
    old_network_params = self.network.get_params_serialized(self.tf_session)
    self._init_network(net_desc)
    # Otherwise it's initialized randomly which is fine.
    # This copy will copy the old params over and leave the rest randomly initialized.
    # This also works if the old network has just the same topology,
    # e.g. if it is the initial model from self.init_network_from_config().
    self.network.set_params_by_serialized(old_network_params, session=self.tf_session)

  def train(self):
    if self.start_epoch:
      print("start training at epoch %i and step %i" % (self.start_epoch, self.start_batch), file=log.v3)
    print("using batch size: %i, max seqs: %i" % (self.batch_size, self.max_seqs), file=log.v4)
    print("learning rate control:", self.learning_rate_control, file=log.v4)
    print("pretrain:", self.pretrain, file=log.v4)

    assert self.start_epoch >= 1, "Epochs start at 1."
    final_epoch = self.final_epoch if self.final_epoch != 0 else sys.maxsize
    if self.start_epoch > final_epoch:
      print("No epochs to train, start_epoch: %i, final_epoch: %i" %
            (self.start_epoch, self.final_epoch), file=log.v1)

    self.check_last_epoch()
    self.max_seq_length += (self.start_epoch - 1) * self.inc_seq_length

    epoch = self.start_epoch  # Epochs start at 1.
    rebatch = True
    while epoch <= final_epoch:
      if self.max_seq_length != sys.maxsize:
        if int(self.max_seq_length + self.inc_seq_length) != int(self.max_seq_length):
          print("increasing sequence lengths to", int(self.max_seq_length + self.inc_seq_length), file=log.v3)
          rebatch = True
        self.max_seq_length += self.inc_seq_length
      # In case of random seq ordering, we want to reorder each epoch.
      if self.train_data.init_seq_order(epoch=epoch):
        rebatch = True
      if epoch % self.seq_drop_freq == 0:
        if self.seq_drop > 0.0:
          rebatch = True
      self.epoch = epoch

      if 'train' in self.dataset_batches:
        if rebatch:
          del self.dataset_batches['train']
        else:
          print("keeping previous dataset batch order for 'train' dataset", file=log.v4)
      for dataset_name, dataset in self.get_eval_datasets().items():
        if dataset.init_seq_order(epoch=self.epoch):
          if dataset_name in self.dataset_batches:
            del self.dataset_batches[dataset_name]
        else:
          if dataset_name in self.dataset_batches:
            print("keeping previous dataset batch order for %r dataset" % dataset_name, file=log.v4)

      self.init_train_epoch()
      self.train_epoch()

      rebatch = False
      epoch += 1

    if self.start_epoch <= self.final_epoch:  # We did train at least one epoch.
      assert self.epoch
      # Save last model, in case it was not saved yet (depends on save_model_epoch_interval).
      if self.model_filename:
        self.save_model(self.get_epoch_model_filename())

      if self.epoch != self.final_epoch:
        print("Stopped after epoch %i and not %i as planned." % (self.epoch, self.final_epoch), file=log.v3)

  def init_train_epoch(self):
    if self.is_pretrain_epoch():
      new_network_desc = self.pretrain.get_network_json_for_epoch(self.epoch)
      self.maybe_init_new_network(new_network_desc)
      self.network.declare_train_params(**self.pretrain.get_train_param_args_for_epoch(self.epoch))
      # Use constant learning rate.
      self.learning_rate = self.pretrain_learning_rate
      self.learning_rate_control.setDefaultLearningRateForEpoch(self.epoch, self.learning_rate)
    elif self.is_first_epoch_after_pretrain():
      # Use constant learning rate.
      self.learning_rate = self.initial_learning_rate
      self.learning_rate_control.setDefaultLearningRateForEpoch(self.epoch, self.learning_rate)
    else:
      self.learning_rate = self.learning_rate_control.getLearningRateForEpoch(self.epoch)

    if not self.is_pretrain_epoch():
      # Train the whole network.
      self.network.declare_train_params()

    self.updater.set_trainable_vars(self.network.get_trainable_params())

  def train_epoch(self):
    print("start", self.get_epoch_str(), "with learning rate", self.learning_rate, "...", file=log.v4)

    if self.epoch == 1 and self.save_epoch1_initial_model:
      epoch0_model_filename = self.epoch_model_filename(self.model_filename, 0, self.is_pretrain_epoch())
      print("save initial epoch1 model", epoch0_model_filename, file=log.v4)
      self.save_model(epoch0_model_filename)

    if 'train' not in self.dataset_batches or not self.train_data.batch_set_generator_cache_whole_epoch():
      self.dataset_batches['train'] = self.train_data.generate_batches(recurrent_net=self.network.recurrent,
                                                                       batch_size=self.batch_size,
                                                                       max_seqs=self.max_seqs,
                                                                       max_seq_length=int(self.max_seq_length),
                                                                       seq_drop=self.seq_drop,
                                                                       shuffle_batches=self.shuffle_batches,
                                                                       used_data_keys=self.network.used_data_keys)
    else:
      self.dataset_batches['train'].reset()
    train_batches = self.dataset_batches['train']

    self.updater.set_learning_rate(self.learning_rate)
    trainer = Runner(engine=self, dataset=self.train_data, batches=train_batches, train=True)
    trainer.run(report_prefix=("pre" if self.is_pretrain_epoch() else "") + "train epoch %s" % self.epoch)

    if not trainer.finalized:
      if trainer.device_crash_batch is not None:  # Otherwise we got an unexpected exception - a bug in our code.
        self.save_model(self.get_epoch_model_filename() + ".crash_%i" % trainer.device_crash_batch)
      sys.exit(1)

    assert not any(numpy.isinf(trainer.score.values())) or any(numpy.isnan(trainer.score.values())), \
      "Model is broken, got inf or nan final score: %s" % trainer.score

    if self.model_filename and (self.epoch % self.save_model_epoch_interval == 0):
      self.save_model(self.get_epoch_model_filename())
    self.learning_rate_control.setEpochError(self.epoch, {"train_score": trainer.score})
    self.learning_rate_control.save()

    print(self.get_epoch_str(), "score:", self.format_score(trainer.score), "elapsed:", hms(trainer.elapsed), end=" ", file=log.v1)
    self.eval_model()

  def format_score(self, score):
    if not score:
      return "None"
    if len(score) == 1:
      return str(list(score.values())[0])
    return " ".join(["%s %s" % (key.split(':')[-1], str(score[key]))
                     for key in sorted(score.keys())])

  def eval_model(self):
    # It's constructed lazily and it will set used_data_keys, so make sure that we have it now.
    self.network.get_all_errors()
    eval_dump_str = []
    for dataset_name, dataset in self.get_eval_datasets().items():
      if dataset_name not in self.dataset_batches or not dataset.batch_set_generator_cache_whole_epoch():
        self.dataset_batches[dataset_name] = dataset.generate_batches(
          recurrent_net=self.network.recurrent,
          batch_size=self.batch_size,
          max_seqs=self.max_seqs,
          max_seq_length=(int(self.max_seq_length) if dataset_name == 'dev' else sys.maxsize),
          used_data_keys=self.network.used_data_keys)
      else:
        self.dataset_batches[dataset_name].reset()
      tester = Runner(engine=self, dataset=dataset, batches=self.dataset_batches[dataset_name], train=False)
      tester.run(report_prefix=self.get_epoch_str() + " eval")
      assert tester.finalized
      eval_dump_str += [" %s: score %s error %s" % (
                        dataset_name, self.format_score(tester.score), self.format_score(tester.error))]
      if dataset_name == "dev":
        self.learning_rate_control.setEpochError(self.epoch, {"dev_score": tester.score, "dev_error": tester.error})
        self.learning_rate_control.save()
    print(" ".join(eval_dump_str).strip(), file=log.v1)

  def check_last_epoch(self):
    if self.start_epoch == 1:
      return
    self.epoch = self.start_epoch - 1
    if self.learning_rate_control.need_error_info:
      if self.dev_data:
        if "dev_score" not in self.learning_rate_control.getEpochErrorDict(self.epoch):
          # This can happen when we have a previous model but did not test it yet.
          print("Last epoch model not yet evaluated on dev. Doing that now.", file=log.v4)
          self.eval_model()

  def get_all_merged_summaries(self):
    # Note: This assumes that the summaries never change.
    # Both both training and evaluation on the CV dataset, this is the case.
    if self._merge_all_summaries is None:
      self._merge_all_summaries = tf.summary.merge_all()
    return self._merge_all_summaries

  def check_uninitialized_vars(self):
    """
    All vars in TF which are controlled by us should also have been initialized by us.
    We also take care about the optimizer slot variables.
    However, TF can still create other vars which we do not know about.
    E.g. the Adam optimizer creates the beta1_power/beta2_power vars (which are no slot vars).
    Here, we find all remaining uninitialized vars, report about them and initialize them.
    """
    if self._checked_uninitialized_vars:
      return
    with tf.name_scope("check_uninitialized_vars"):
      # Like tf.report_uninitialized_variables().
      var_list = tf.global_variables() + tf.local_variables()
      # Get a 1-D boolean tensor listing whether each variable is initialized.
      var_mask = tf.logical_not(tf.pack(
        [tf.is_variable_initialized(v) for v in var_list])).eval(session=self.tf_session)
      assert len(var_mask) == len(var_list)
      uninitialized_vars = [v for (v, mask) in zip(var_list, var_mask) if mask]
      if uninitialized_vars:
        print("Note: There are still these uninitialized variables: %s" % [v.name for v in uninitialized_vars], file=log.v3)
        self.tf_session.run(tf.variables_initializer(uninitialized_vars))
      self._checked_uninitialized_vars = True

  def forward_single(self, dataset, seq_idx, output_layer_name=None):
    """
    :param Dataset.Dataset dataset:
    :param int seq_idx:
    :param str|None output_layer_name: e.g. "output". if not set, will read from config "forward_output_layer"
    :return: numpy array, output in time major format (time,batch,dim)
    :rtype: numpy.ndarray
    """
    if not output_layer_name:
      output_layer_name = self.config.value("forward_output_layer", self.network.get_default_output_layer_name())
      assert output_layer_name, "output layer not defined. set forward_output_layer in config"
    assert output_layer_name in self.network.layers, "output layer %r not found" % output_layer_name

    # No Runner instance here but a very simplified version of Runner.run().
    # First we need a custom DataProvider with a custom BatchSetGenerator
    # which will yield only one single batch for the provided sequence idx.
    batch = Batch()
    batch.add_frames(seq_idx=seq_idx, seq_start_frame=0, length=dataset.get_seq_length(seq_idx))
    batch_generator = iter([batch])
    batches = BatchSetGenerator(dataset, generator=batch_generator)
    data_provider = DataProvider(
      tf_session=self.tf_session, extern_data=self.network.extern_data,
      data_keys=self.network.used_data_keys,
      dataset=dataset, batches=batches)

    # Maybe some new uninitialized vars. Last check.
    self.check_uninitialized_vars()

    feed_dict = data_provider.get_feed_dict(previous_feed_dict=None, single_threaded=True)
    output_data = self.network.layers[output_layer_name].output
    output_value = self.tf_session.run(output_data.get_placeholder_as_time_major(), feed_dict=feed_dict)
    return output_value

  def analyze(self, data, statistics):
    """
    :param Dataset.Dataset data:
    :param list[str]|None statistics: ignored at the moment
    :return: nothing, will print everything to log.v1
    """
    print("Analyze with network on %r." % data, file=log.v1)

    if "analyze" not in self.network.layers:
      from TFNetworkLayer import FramewiseStatisticsLayer
      assert self.config.has("sil_label_idx")
      self.network.add_layer(
        name="analyze", layer_class=FramewiseStatisticsLayer,
        sil_label_idx=self.config.int("sil_label_idx", 0),
        sources=self.network.get_output_layers())

    # It's constructed lazily and it will set used_data_keys, so make sure that we have it now.
    self.network.get_all_errors()

    batch_size = self.config.int('batch_size', 1)
    max_seqs = self.config.int('max_seqs', -1)
    max_seq_length = self.config.float('max_seq_length', 0)
    if max_seq_length <= 0:
      max_seq_length = sys.maxsize

    batches = data.generate_batches(
      recurrent_net=self.network.recurrent,
      batch_size=batch_size,
      max_seqs=max_seqs,
      max_seq_length=max_seq_length,
      used_data_keys=self.network.used_data_keys)
    analyzer = Runner(engine=self, dataset=data, batches=batches, train=False)
    analyzer.run(report_prefix=self.get_epoch_str() + " analyze")

    print("Finished analyzing of the dataset %r." % data, file=log.v1)
    print("elapsed:", hms(analyzer.elapsed), file=log.v1)
    print("num mini-batches:", analyzer.num_steps, file=log.v1)
    print("total num_frames:", analyzer.data_provider.num_frames, file=log.v1)
    print("score:", self.format_score(analyzer.score), file=log.v1)
    print("error:", self.format_score(analyzer.error), file=log.v1)
    for k, v in sorted(analyzer.stats.items()):
      if k.startswith("stats:"):
        print("%s:" % k, v, file=log.v1)
    print("That are all collected stats.", file=log.v1)

    if not analyzer.finalized:
      print("WARNING: Did not finished through the whole epoch.", file=log.v1)
      sys.exit(1)
