
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.client import timeline
import numpy
import sys
import os
import time
from Log import log
from Engine import Engine as TheanoEngine
from LearningRateControl import loadLearningRateControlFromConfig
from Pretrain import pretrainFromConfig
from Network import LayerNetwork
from TFNetwork import TFNetwork, ExternData
from Util import hms
from threading import Thread
from Dataset import BatchSetGenerator


class DataProvider(object):
  """
  This class will fill all the placeholders used for training or forwarding or evaluation etc.
  of a `TFNetwork.Network`.
  It will run a background thread which reads the data from a dataset and puts it into a queue.
  """

  def __init__(self, tf_session, dataset, batches, extern_data, capacity=10):
    """
    :param tf.Session tf_session:
    :param Dataset.Dataset dataset:
    :param BatchSetGenerator batches:
    :param ExternData extern_data:
    :param int capacity:
    """
    self.tf_session = tf_session
    self.coord = tf.train.Coordinator()
    self.dataset = dataset
    self.batches = batches
    self.extern_data = extern_data
    self.data_keys = sorted(extern_data.data.keys())
    self.queue = tf.FIFOQueue(capacity=capacity, **extern_data.get_queue_args(with_batch_dim=True))
    self.thread = None  # type: Thread
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
    self.thread.join()

  def _get_next_batch(self):
    """
    :param Dataset.Batch batch:
    :returns (batch-data-value-dict, batch-seq-lens)
    :rtype: (dict[str,numpy.ndarray], dict[str,numpy.ndarray])
    """
    # See EngineUtil.assign_dev_data() for reference.
    batch, = self.batches.peek_next_n(1)
    shapes = self.dataset.shapes_for_batches([batch], data_keys=self.data_keys, batch_dim_first=True)
    data = {k: numpy.zeros(shape=shapes[k], dtype=self.extern_data.get_data(k).dtype)
            for k in self.data_keys}
    seq_lens = {k: numpy.zeros(shape=(shapes[k][0],), dtype="int64")
                for k in self.data_keys}
    self.dataset.load_seqs(batch.start_seq, batch.end_seq)
    #device.num_frames += batch.get_total_num_frames()
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
          seq_lens[k][q] = max(seq_lens[q], o[k] + ls)
    return data, seq_lens

  def thread_main(self):
    try:
      import better_exchook
      better_exchook.install()

      while self.batches.has_more() and not self.coord.should_stop():
        data, seq_lens = self._get_next_batch()
        enqueue_args = data.copy()
        for k in data.keys():
          enqueue_args["%s_seq_lens" % k] = seq_lens[k]
        self.tf_session.run(self.queue.enqueue(enqueue_args))

      self.reached_end = not self.batches.has_more()

    except Exception as exc:
      print("Exception in DataProvider thread: %r" % exc)
      sys.excepthook(*sys.exc_info())

  def is_data_finished(self):
    """
    :return: when we go through an epoch and finished reading, this will return True
    """
    if not self.reached_end:
      return False
    return self.queue.size().eval(self.tf_session) == 0

  def get_feed_dict(self):
    """
    :returns: we dequeue one batch from the queue and provide it for all placeholders of our external data
    :rtype: dict[tf.Tensor,tf.Tensor]
    """
    output = self.queue.dequeue()
    assert isinstance(output, dict)
    return {self.extern_data.get_data(k).placeholder: output[k] for k in self.data_keys}


class Runner(object):
  def __init__(self, engine, dataset, batches):
    """
    :param Engine engine:
    :param Dataset.Dataset dataset:
    :param BatchSetGenerator batches:
    """
    self.engine = engine
    self.data_provider = DataProvider(
      tf_session=engine.tf_session, extern_data=engine.network.extern_data,
      dataset=dataset, batches=batches)
    self.store_metadata_mod_step = False  # 500

  def get_feed_dict(self):
    return self.data_provider.get_feed_dict()

  def get_fetches_dict(self):
    d = {
      "summary": tf.merge_all_summaries(),
      "queue_size": self.data_provider.queue.size()
    }
    for layer_name, loss in self.engine.network.loss_by_layer.items():
      d["loss_%s" % layer_name] = loss
    for layer_name, error in self.engine.network.error_by_layer.items():
      d["error_%s" % layer_name] = error
    if self.engine.updater:
      d["loss"] = self.engine.updater.loss
      d["optim_op"] = self.engine.updater.get_optim_op()
    return d

  def run(self):
    sess = self.engine.tf_session
    logdir = os.path.dirname(self.engine.model_filename) or os.getcwd()
    writer = tf.train.SummaryWriter(logdir)
    writer.add_graph(sess.graph)
    run_metadata = tf.RunMetadata()

    coord = self.data_provider.coord

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    self.data_provider.start_thread()

    step = None
    try:
      # step is like mini-batch in our usual terminology
      step = 0
      while True:
        start_time = time.time()
        fetches_dict = self.get_fetches_dict()
        feed_dict = self.data_provider.get_feed_dict()
        if self.store_metadata_mod_step and step % self.store_metadata_mod_step == 0:
          # Slow run that stores extra information for debugging.
          print('Storing metadata', file=log.v4)
          run_options = tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE)
          fetches_results = sess.run(
            fetches_dict,
            feed_dict=feed_dict,
            options=run_options,
            run_metadata=run_metadata)
          writer.add_summary(fetches_results["summary"], step)
          writer.add_run_metadata(run_metadata,
                                  'step_{:04d}'.format(step))
          tl = timeline.Timeline(run_metadata.step_stats)
          timeline_path = os.path.join(logdir, 'timeline.trace')
          with open(timeline_path, 'w') as f:
            f.write(tl.generate_chrome_trace_format(show_memory=True))
        else:
          fetches_results = sess.run(fetches_dict, feed_dict=feed_dict)
          writer.add_summary(fetches_results["summary"], step)

        duration = time.time() - start_time
        formated_losses = " - ".join(
          ["{} = {:.3f}".format(k, v)
           for (k, v) in fetches_results.items()
           if k.startswith("loss") or k.startswith("error")])
        print('step {:d} - {}, ({:.3f} sec/step)'
              .format(step, formated_losses or "no loss", duration), file=log.v4)

        step += 1

    except BaseException as exc:
      print("Exception %r in step %r." % (exc, step), file=log.v1)

    finally:
      coord.request_stop()
      coord.join(threads)


_OptimizerClassesDict = {}  # type: dict[str,()->tf.train.Optimizer]

def get_optimizer_class(class_name):
  """
  :param str class_name: e.g. "adam"
  :return:
  """
  if not _OptimizerClassesDict:
    for name, v in vars(tf.train).items():
      if name.endswith("Optimizer"):
        name = name[:-len("Optimizer")]
      else:
        continue
      if not issubclass(v, tf.train.Optimizer):
        continue
      name = name.lower()
      assert name not in _OptimizerClassesDict
      _OptimizerClassesDict[name] = v
  return _OptimizerClassesDict[class_name.lower()]


class Updater(object):
  def __init__(self, config):
    """
    :param Config.Config config:
    """
    self.config = config
    self.learning_rate_var = tf.Variable(initial_value=0.0, trainable=False, dtype="float32")
    self.trainable_vars = []  # type: list[tf.Variable]
    self.loss = None  # type: tf.Tensor
    self.optimizer = None  # type: tf.train.Optimizer
    self.optim_op = None  # type: tf.Operation

  def reset_optim_op(self):
    """
    Call this if sth is changed which the optim_op depends on.
    See self.create_optim_op().
    """
    self.optim_op = None  # type: tf.Operation

  def set_loss(self, loss):
    """
    :param tf.Tensor loss:
    """
    self.loss = loss
    self.reset_optim_op()

  def set_trainable_vars(self, trainable_vars):
    """
    :param list[tf.Variable] trainable_vars:
    """
    if trainable_vars == self.trainable_vars:
      return
    self.trainable_vars = trainable_vars
    self.reset_optim_op()

  def create_optimizer(self):
    lr = self.learning_rate_var
    epsilon = 1e-16
    momentum = self.config.float("momentum", 0.0)
    optim_config = self.config.typed_value("optimizer")
    if optim_config:
      assert isinstance(optim_config, dict)
      optim_config = optim_config.copy()
      optim_class_name = optim_config.pop("class")
      optim_class = get_optimizer_class(optim_class_name)
      optim_config.setdefault("epsilon", epsilon)
      if momentum:
        optim_config.setdefault("momentum", momentum)
      optim_config["learning_rate"] = lr
      print("Create optimizer %s with options %r." % (optim_class, optim_config), file=log.v2)
      optimizer = optim_class(**optim_config)
      assert isinstance(optimizer, tf.train.Optimizer)
    elif self.config.bool("adam", False):
      assert not momentum
      print("Create Adam optimizer.", file=log.v2)
      optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
    elif self.config.bool("rmsprop", False):
      print("Create RMSProp optimizer.", file=log.v2)
      optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum, epsilon=epsilon)
    elif momentum:
      print("Create Momentum optimizer.", file=log.v2)
      optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum)
    else:
      print("Create SGD optimizer.", file=log.v2)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    self.optimizer = optimizer
    self.reset_optim_op()

  def create_optim_op(self):
    if not self.optimizer:
      self.create_optimizer()

    aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N

    self.optim_op = self.optimizer.minimize(
      self.loss, var_list=self.trainable_vars,
      aggregation_method=aggregation_method)

  def get_optim_op(self):
    """
    :rtype: tf.Operation
    """
    if self.optim_op is None:
      self.create_optim_op()
    return self.optim_op


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
    self.dataset_batches = {}  # type: dict[str,BatchSetGenerator]

  def _get_devices_config(self):
    """
    :rtype: list[dict[str]]
    """
    from Device import getDevicesInitArgs
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

  def _make_tf_session(self):
    if self.tf_session:
      self.tf_session.close()
    opts = self.config.typed_value("tf_session_opts", {})
    assert isinstance(opts, dict)
    opts = opts.copy()
    opts.setdefault("log_device_placement", False)
    opts.setdefault("device_count", {}).setdefault("GPU", 1 if self.is_requesting_for_gpu() else 0)
    print("Setup tf.Session with options %r ..." % opts, file=log.v2)
    config = tf.ConfigProto(**opts)
    # config.gpu_options.allow_growth=True
    self.tf_session = tf.Session(config=config)

  def _reset_graph(self):
    tf.reset_default_graph()
    # The new session will by default use the newly created default graph.
    self._make_tf_session()

  get_train_start_epoch_batch = TheanoEngine.get_train_start_epoch_batch
  config_get_final_epoch = TheanoEngine.config_get_final_epoch
  get_epoch_model = TheanoEngine.get_epoch_model

  @classmethod
  def epoch_model_filename(cls, model_filename, epoch, is_pretrain):
    """
    :type model_filename: str
    :type epoch: int
    :type is_pretrain: bool
    :rtype: str
    """
    if sys.platform == "win32" and model_filename.startswith("/tmp/"):
      import tempfile
      model_filename = tempfile.gettempdir() + model_filename[len("/tmp")]
    return model_filename + (".pretrain" if is_pretrain else "") + ".%03d" % epoch

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

  def print_network_info(self):
    self.network.print_network_info()

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
    self.model_filename = config.value('model', None)
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
    self.updater = Updater(config=config)

  def init_network_from_config(self, config):
    self.pretrain = pretrainFromConfig(config)
    self.max_seqs = config.int('max_seqs', -1)

    epoch, model_epoch_filename = self.get_epoch_model(config)
    assert model_epoch_filename or self.start_epoch

    if self.pretrain:
      # This would be obsolete if we don't want to load an existing model.
      # In self.init_train_epoch(), we initialize a new model.
      net_dict = self.pretrain.get_network_json_for_epoch(epoch or self.start_epoch)
    else:
      net_dict = LayerNetwork.json_from_config(config)

    self._init_network(net_desc=net_dict, epoch=epoch or self.start_epoch)

    if model_epoch_filename:
      print("loading weights from", model_epoch_filename, file=log.v2)
      self.network.load_params_from_file(model_epoch_filename, session=self.tf_session)

  def _init_network(self, net_desc, epoch=None):
    if epoch is None:
      epoch = self.epoch
    self._reset_graph()
    tf.set_random_seed(42)
    network = TFNetwork(rnd_seed=epoch)
    network.construct_from_dict(net_desc)
    network.initialize_params(session=self.tf_session)
    network.layers_desc = net_desc
    network.print_network_info()
    self.network = network
    if self.updater:
      self.updater.set_loss(network.get_objective())
      self.updater.set_trainable_vars(network.get_trainable_params())

  def maybe_init_new_network(self, net_desc):
    if self.network.layers_desc == net_desc:
      return
    from Util import dict_diff_str
    print("reinit because network description differs. Diff:",
          dict_diff_str(self.network.layers_desc, net_desc), file=log.v3)
    old_network_params = self.network.get_param_values_dict(self.tf_session)
    self._init_network(net_desc)
    # Otherwise it's initialized randomly which is fine.
    # This copy will copy the old params over and leave the rest randomly initialized.
    # This also works if the old network has just the same topology,
    # e.g. if it is the initial model from self.init_network_from_config().
    self.network.set_param_values_by_dict(old_network_params, session=self.tf_session)

  def train(self):
    if self.start_epoch:
      print("start training at epoch %i and batch %i" % (self.start_epoch, self.start_batch), file=log.v3)
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
      rebatch = self.train_data.init_seq_order(epoch=epoch) or rebatch
      if epoch % self.seq_drop_freq == 0:
        rebatch = self.seq_drop > 0.0 or rebatch
      self.epoch = epoch

      for dataset_name, dataset in self.get_eval_datasets().items():
        if dataset.init_seq_order(self.epoch) and dataset_name in self.dataset_batches:
          del self.dataset_batches[dataset_name]

      if rebatch and 'train' in self.dataset_batches:
        del self.dataset_batches['train']

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

    if self.is_pretrain_epoch():
      self.print_network_info()

    if not 'train' in self.dataset_batches:
      self.dataset_batches['train'] = self.train_data.generate_batches(recurrent_net=self.network.recurrent,
                                                                       batch_size=self.batch_size,
                                                                       max_seqs=self.max_seqs,
                                                                       max_seq_length=int(self.max_seq_length),
                                                                       seq_drop=self.seq_drop,
                                                                       shuffle_batches=self.shuffle_batches)
    else:
      self.dataset_batches['train'].reset()
    train_batches = self.dataset_batches['train']

    self.tf_session.run(self.updater.learning_rate_var.assign(self.learning_rate))
    runner = Runner(engine=self, dataset=self.train_data, batches=train_batches)
    runner.run()

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
    if self.ctc_prior_file is not None:
      trainer.save_ctc_priors(self.ctc_prior_file, self.get_epoch_str())

    print(self.get_epoch_str(), "score:", self.format_score(trainer.score), "elapsed:", hms(trainer.elapsed), end=" ", file=log.v1)
    self.eval_model()

  def format_score(self, score):
    if len(score) == 1:
      return str(list(score.values())[0])
    return " ".join(["%s %s" % (key.split(':')[-1], str(score[key]))
                     for key in sorted(score.keys())])

  def _run_data_provider(self):
    pass

  def _eval_model(self, dataset_name):
    batches = self.dataset_batches[dataset_name]
    report_prefix = self.get_epoch_str() + " eval"
    epoch = self.epoch

  def eval_model(self):
    eval_dump_str = []
    for dataset_name, dataset in self.get_eval_datasets().items():
      if not dataset_name in self.dataset_batches:
        self.dataset_batches[dataset_name] = dataset.generate_batches(recurrent_net=self.network.recurrent,
                                                                      batch_size=self.batch_size,
                                                                      max_seqs=self.max_seqs,
                                                                      max_seq_length=(int(self.max_seq_length) if dataset_name == 'dev' else sys.maxsize))
      else:
        self.dataset_batches[dataset_name].reset()
      tester = EvalTaskThread(self.network, self.devices, data=dataset, batches=self.dataset_batches[dataset_name],
                              report_prefix=self.get_epoch_str() + " eval", epoch=self.epoch)
      tester.join()
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



