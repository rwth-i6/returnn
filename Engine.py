#! /usr/bin/python2.7

import numpy
import sys
import os
from collections import OrderedDict
import h5py
import json
from Network import LayerNetwork
from EngineTask import TrainTaskThread, EvalTaskThread, HDFForwardTaskThread, ClassificationTaskThread, PriorEstimationTaskThread
import SprintCache
from Log import log
from Updater import Updater
import Device
from LearningRateControl import loadLearningRateControlFromConfig
from Pretrain import pretrainFromConfig
import EngineUtil
from Util import hms, hdf5_dimension, BackendEngine, model_epoch_from_filename
import errno
import time
try:
  import SimpleHTTPServer
  import SocketServer
  import BaseHTTPServer
except ImportError:  # Python3
  import http.server as SimpleHTTPServer
  import socketserver as SocketServer
  BaseHTTPServer = SimpleHTTPServer
import json
import cgi
from GeneratingDataset import StaticDataset
import hashlib

class Engine:

  _epoch_model = None; """ :type: (int|None,str|None) """  # See get_epoch_model().

  def __init__(self, devices):
    """
    :type devices: list[Device.Device]
    """
    self.devices = devices
    self.train_data = None; " :type: Dataset.Dataset "
    self.is_training = False
    self.start_epoch = None
    self.training_finished = False
    self.stop_train_after_epoch_request = False
    self.dataset_batches = {}
    self.pretrain = None; " :type: Pretrain.Pretrain "
    self.init_train_epoch_posthook = None

  @classmethod
  def config_get_final_epoch(cls, config):
    """ :type config: Config.Config """
    return config.int('num_epochs', 5)

  @classmethod
  def get_existing_models(cls, config):
    model_filename = config.value('model', '')
    if not model_filename:
      return []
    # Automatically search the filesystem for existing models.
    file_list = []
    for epoch in range(1, cls.config_get_final_epoch(config) + 1):
      for is_pretrain in [False, True]:
        fn = cls.epoch_model_filename(model_filename, epoch, is_pretrain)
        if os.path.exists(fn):
          file_list += [(epoch, fn)]  # epoch, fn
          break
        if BackendEngine.is_tensorflow_selected():
          if os.path.exists(fn + ".index"):
            file_list += [(epoch, fn)]  # epoch, fn
            break
    file_list.sort()
    return file_list

  @classmethod
  def get_epoch_model(cls, config):
    """
    :type config: Config.Config
    :returns (epoch, modelFilename)
    :rtype: (int|None, str|None)
    """
    # XXX: We cache it, although this is wrong if we have changed the config.
    if cls._epoch_model:
      return cls._epoch_model

    start_epoch_mode = config.value('start_epoch', 'auto')
    if start_epoch_mode == 'auto':
      start_epoch = None
    else:
      start_epoch = int(start_epoch_mode)
      assert start_epoch >= 1

    load_model_epoch_filename = config.value('load', '')
    if load_model_epoch_filename:
      fn_postfix = ""
      if BackendEngine.is_tensorflow_selected():
        fn_postfix = ".meta"
      assert os.path.exists(load_model_epoch_filename + fn_postfix)

    import_model_train_epoch1 = config.value('import_model_train_epoch1', '')
    if import_model_train_epoch1:
      assert os.path.exists(import_model_train_epoch1)

    existing_models = cls.get_existing_models(config)

    # Only use this when we don't train.
    # For training, we first consider existing models before we take the 'load' into account when in auto epoch mode.
    # In all other cases, we use the model specified by 'load'.
    if load_model_epoch_filename and (config.value('task', 'train') != 'train' or start_epoch is not None):
      epoch = model_epoch_from_filename(load_model_epoch_filename)
      if config.value('task', 'train') == 'train' and start_epoch is not None:
        # Ignore the epoch. To keep it consistent with the case below.
        epoch = None
      epoch_model = (epoch, load_model_epoch_filename)

    # In case of training, always first consider existing models.
    # This is because we reran CRNN training, we usually don't want to train from scratch
    # but resume where we stopped last time.
    elif existing_models:
      epoch_model = existing_models[-1]
      if load_model_epoch_filename:
        print >> log.v4, "note: there is a 'load' which we ignore because of existing model"

    elif config.value('task', 'train') == 'train' and import_model_train_epoch1 and start_epoch in [None, 1]:
      epoch_model = (0, import_model_train_epoch1)

    # Now, consider this also in the case when we train, as an initial model import.
    elif load_model_epoch_filename:
      # Don't use the model epoch as the start epoch in training.
      # We use this as an import for training.
      epoch_model = (model_epoch_from_filename(load_model_epoch_filename), load_model_epoch_filename)

    else:
      epoch_model = (None, None)

    if start_epoch == 1:
      if epoch_model[0]:  # existing model
        print >> log.v4, "warning: there is an existing model: %s" % (epoch_model,)
        epoch_model = (None, None)
    elif start_epoch > 1:
      if epoch_model[0]:
        if epoch_model[0] != start_epoch - 1:
          print >> log.v4, "warning: start_epoch %i but there is %s" % (start_epoch, epoch_model)
        epoch_model = existing_models[start_epoch-1]

    cls._epoch_model = epoch_model
    return epoch_model

  @classmethod
  def get_train_start_epoch_batch(cls, config):
    """
    We will always automatically determine the best start (epoch,batch) tuple
    based on existing model files.
    This ensures that the files are present and enforces that there are
    no old outdated files which should be ignored.
    Note that epochs start at idx 1 and batches at idx 0.
    :type config: Config.Config
    :returns (epoch,batch)
    :rtype (int,int)
    """
    start_batch_mode = config.value('start_batch', 'auto')
    if start_batch_mode == 'auto':
      start_batch_config = None
    else:
      start_batch_config = int(start_batch_mode)
    last_epoch, _ = cls.get_epoch_model(config)
    if last_epoch is None:
      start_epoch = 1
      start_batch = start_batch_config or 0
    elif start_batch_config is not None:
      # We specified a start batch. Stay in the same epoch, use that start batch.
      start_epoch = last_epoch
      start_batch = start_batch_config
    else:
      # Start with next epoch.
      start_epoch = last_epoch + 1
      start_batch = 0
    return start_epoch, start_batch

  def init_train_from_config(self, config, train_data, dev_data=None, eval_data=None):
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
    self.updater = Updater.initFromConfig(config)
    self.ctc_prior_file = config.value('ctc_prior_file', None)
    self.exclude = config.int_list('exclude', [])
    self.init_train_epoch_posthook = config.value('init_train_epoch_posthook', None)
    self.share_batches = config.bool('share_batches', False)
    self.seq_drop = config.float('seq_drop', 0.0)
    self.seq_drop_freq = config.float('seq_drop_freq', 10)
    self.max_seq_length = config.float('max_seq_length', 0)
    self.inc_seq_length = config.float('inc_seq_length', 0)
    self.output_precision = config.int('output_precision', 12)
    if self.max_seq_length == 0:
      self.max_seq_length = sys.maxsize
    if config.is_typed("seq_train_parallel"):
      self.seq_train_parallel = SeqTrainParallelControl(engine=self, config=config, **config.typed_value("seq_train_parallel"))
    else:
      self.seq_train_parallel = None
    # And also initialize the network. That depends on some vars here such as pretrain.
    self.init_network_from_config(config)

  def init_network_from_config(self, config):
    self.pretrain = pretrainFromConfig(config)
    self.max_seqs = config.int('max_seqs', -1)

    epoch, model_epoch_filename = self.get_epoch_model(config)
    assert model_epoch_filename or self.start_epoch

    if model_epoch_filename:
      print >> log.v2, "loading weights from", model_epoch_filename
      last_model_hdf = h5py.File(model_epoch_filename, "r")
    else:
      last_model_hdf = None

    if config.bool('initialize_from_model', False):
      # That's only about the topology, not the params.
      print >> log.v5, "initializing network topology from model"
      assert last_model_hdf, "last model not specified. use 'load' in config. or don't use 'initialize_from_model'"
      network = LayerNetwork.from_hdf_model_topology(last_model_hdf)
    else:
      if self.pretrain:
        # This would be obsolete if we don't want to load an existing model.
        # In self.init_train_epoch(), we initialize a new model.
        network = self.pretrain.get_network_for_epoch(epoch or self.start_epoch)
      else:
        network = LayerNetwork.from_config_topology(config)

    # We have the parameters randomly initialized at this point.
    # In training, as an initialization, we can copy over the params of an imported model,
    # where our topology might slightly differ from the imported model.
    if config.value('import_model_train_epoch1', '') and self.start_epoch == 1:
      assert last_model_hdf
      old_network = LayerNetwork.from_hdf_model_topology(last_model_hdf)
      old_network.load_hdf(last_model_hdf)
      last_model_hdf.close()
      # Copy params to new network.
      from NetworkCopyUtils import intelli_copy_layer
      # network.hidden are the input + all hidden layers.
      for layer_name, layer in sorted(old_network.hidden.items()):
        print >> log.v3, "Copy hidden layer %s" % layer_name
        intelli_copy_layer(layer, network.hidden[layer_name])
      for layer_name, layer in sorted(old_network.output.items()):
        print >> log.v3, "Copy output layer %s" % layer_name
        intelli_copy_layer(layer, network.output[layer_name])
      print >> log.v3, "Not copied hidden: %s" % sorted(set(network.hidden.keys()).difference(old_network.hidden.keys()))
      print >> log.v3, "Not copied output: %s" % sorted(set(network.output.keys()).difference(old_network.output.keys()))

    # Maybe load existing model parameters.
    elif last_model_hdf:
      network.load_hdf(last_model_hdf)
      last_model_hdf.close()
      EngineUtil.maybe_subtract_priors(network, self.train_data, config)

    self.network = network

    if config.has('dump_json'):
      self.network_dump_json(config.value('dump_json', ''))

    self.print_network_info()

  def network_dump_json(self, json_filename):
    fout = open(json_filename, 'w')
    try:
      json_content = self.network.to_json()
      print(json_content)
      print("---------------")
      json_data = json.loads(json_content)
      print(json_data)
      print("---------------")
      print(json.dumps(json_data, indent=2, sort_keys=True))
      print("---------------")
      print >> fout, json.dumps(json_data, indent=2, sort_keys=True)
    except ValueError as e:
      print >> log.v5, self.network.to_json()
      assert False, "JSON parsing failed: %s" % e
    fout.close()

  def print_network_info(self):
    self.network.print_network_info()

  def check_last_epoch(self):
    if self.start_epoch == 1:
      return
    self.epoch = self.start_epoch - 1
    if self.learning_rate_control.need_error_info:
      if self.dev_data:
        if "dev_score" not in self.learning_rate_control.getEpochErrorDict(self.epoch):
          # This can happen when we have a previous model but did not test it yet.
          print >> log.v4, "Last epoch model not yet evaluated on dev. Doing that now."
          self.eval_model()

  def train(self):
    if self.start_epoch:
      print >> log.v3, "start training at epoch %i and batch %i" % (self.start_epoch, self.start_batch)
    print >> log.v4, "using batch size: %i, max seqs: %i" % (self.batch_size, self.max_seqs)
    print >> log.v4, "learning rate control:", self.learning_rate_control
    print >> log.v4, "pretrain:", self.pretrain
    if self.network.loss == 'priori':
      prior = self.train_data.calculate_priori()
      self.network.output["output"].priori.set_value(prior)
      self.network.output["output"].initialize()

    if self.network.recurrent:
      assert not self.train_data.shuffle_frames_of_nseqs, "Frames must not be shuffled in recurrent net."

    self.is_training = True
    self.training_finished = False

    assert self.start_epoch >= 1, "Epochs start at 1."
    final_epoch = self.final_epoch if self.final_epoch != 0 else sys.maxsize
    if self.start_epoch > final_epoch:
      print >> log.v1, "No epochs to train, start_epoch: %i, final_epoch: %i" % \
                       (self.start_epoch, self.final_epoch)

    self.check_last_epoch()
    self.max_seq_length += (self.start_epoch - 1) * self.inc_seq_length

    epoch = self.start_epoch # Epochs start at 1.
    rebatch = True
    while epoch <= final_epoch:
      if self.max_seq_length != sys.maxsize:
        if int(self.max_seq_length + self.inc_seq_length) != int(self.max_seq_length):
          print >> log.v3, "increasing sequence lengths to", int(self.max_seq_length + self.inc_seq_length)
          rebatch = True
        self.max_seq_length += self.inc_seq_length
      # In case of random seq ordering, we want to reorder each epoch.
      rebatch = self.train_data.init_seq_order(epoch=epoch) or rebatch
      if epoch % self.seq_drop_freq == 0:
        rebatch = self.seq_drop > 0.0 or rebatch
      self.epoch = epoch

      for dataset_name,dataset in self.get_eval_datasets().items():
        if dataset.init_seq_order(self.epoch) and dataset_name in self.dataset_batches:
          del self.dataset_batches[dataset_name]

      if rebatch and 'train' in self.dataset_batches:
        del self.dataset_batches['train']

      self.init_train_epoch()
      self.train_epoch()

      rebatch = False

      if self.stop_train_after_epoch_request:
        self.stop_train_after_epoch_request = False
        break

      epoch += 1

    if self.start_epoch <= self.final_epoch:  # We did train at least one epoch.
      assert self.epoch
      # Save last model, in case it was not saved yet (depends on save_model_epoch_interval).
      if self.model_filename:
        self.save_model(self.get_epoch_model_filename(), self.epoch)

      if self.epoch != self.final_epoch:
        print >> log.v3, "Stopped after epoch %i and not %i as planned." % (self.epoch, self.final_epoch)

    self.is_training = False
    self.training_finished = True

  def get_eval_datasets(self):
    eval_datasets = {}; """ :type: dict[str,Dataset.Dataset] """
    for name, dataset in [("dev", self.dev_data), ("eval", self.eval_data)]:
      if not dataset: continue
      eval_datasets[name] = dataset
    return eval_datasets

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

  def init_train_epoch(self):
    if self.is_pretrain_epoch():
      new_network = self.pretrain.get_network_for_epoch(self.epoch)
      old_network = self.network
      if old_network:
        # Otherwise it's initialized randomly which is fine.
        # This copy will copy the old params over and leave the rest randomly initialized.
        # This also works if the old network has just the same topology,
        # e.g. if it is the initial model from self.init_network_from_config().
        self.pretrain.copy_params_from_old_network(new_network, old_network)
      self.network = new_network
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

    if self.init_train_epoch_posthook:
      print >> log.v5, "execute init_train_epoch_posthook:", self.init_train_epoch_posthook
      exec(self.init_train_epoch_posthook)

  def train_epoch(self):
    print >> log.v4, "start", self.get_epoch_str(), "with learning rate", self.learning_rate, "..."

    if self.epoch == 1 and self.save_epoch1_initial_model:
      epoch0_model_filename = self.epoch_model_filename(self.model_filename, 0, self.is_pretrain_epoch())
      print >> log.v4, "save initial epoch1 model", epoch0_model_filename
      self.save_model(epoch0_model_filename, 0)

    if self.is_pretrain_epoch():
      self.print_network_info()

    training_devices = self.devices
    if 'train' not in self.dataset_batches or not self.train_data.batch_set_generator_cache_whole_epoch():
      self.dataset_batches['train'] = self.train_data.generate_batches(recurrent_net=self.network.recurrent,
                                                                       batch_size=self.batch_size,
                                                                       max_seqs=self.max_seqs,
                                                                       max_seq_length=int(self.max_seq_length),
                                                                       seq_drop=self.seq_drop,
                                                                       shuffle_batches=self.shuffle_batches,
                                                                       used_data_keys=self.network.get_used_data_keys())
    else:
      self.dataset_batches['train'].reset()
    train_batches = self.dataset_batches['train']
    start_batch = self.start_batch if self.epoch == self.start_epoch else 0
    trainer = TrainTaskThread(self.network, training_devices, data=self.train_data, batches=train_batches,
                              learning_rate=self.learning_rate, updater=self.updater,
                              eval_batch_size=self.update_batch_size,
                              start_batch=start_batch, share_batches=self.share_batches,
                              exclude=self.exclude,
                              seq_train_parallel=self.seq_train_parallel,
                              report_prefix=("pre" if self.is_pretrain_epoch() else "") + "train epoch %s" % self.epoch,
                              epoch=self.epoch)
    trainer.join()
    if not trainer.finalized:
      if trainer.device_crash_batch is not None:  # Otherwise we got an unexpected exception - a bug in our code.
        self.save_model(self.get_epoch_model_filename() + ".crash_%i" % trainer.device_crash_batch, self.epoch - 1)
      sys.exit(1)

    assert not any(numpy.isinf(trainer.score.values())) or any(numpy.isnan(trainer.score.values())), \
      "Model is broken, got inf or nan final score: %s" % trainer.score

    if self.model_filename and (self.epoch % self.save_model_epoch_interval == 0):
      self.save_model(self.get_epoch_model_filename(), self.epoch)
    self.learning_rate_control.setEpochError(self.epoch, {"train_score": trainer.score})
    self.learning_rate_control.save()
    if self.ctc_prior_file is not None:
      trainer.save_ctc_priors(self.ctc_prior_file, self.get_epoch_str())

    print >> log.v1, self.get_epoch_str(), "score:", self.format_score(trainer.score), "elapsed:", hms(trainer.elapsed),
    self.eval_model()

  def format_score(self, score):
    if len(score) == 1:
      if self.output_precision < 12:
        return ("%." + str(self.output_precision) + "g") % float(list(score.values())[0])
      else:
        return str(list(score.values())[0])
    if self.output_precision < 12:
      return " ".join([("%s %." + str(self.output_precision) + "g") % (key.split(':')[-1], float(score[key]))
                for key in sorted(score.keys())])
    else:
      return " ".join(["%s %s" % (key.split(':')[-1], str(score[key]))
                       for key in sorted(score.keys())])

  def eval_model(self):
    eval_dump_str = []
    for dataset_name, dataset in self.get_eval_datasets().items():
      if dataset_name not in self.dataset_batches or not dataset.batch_set_generator_cache_whole_epoch():
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
    print >> log.v1, " ".join(eval_dump_str).strip()

  def save_model(self, filename, epoch):
    """
    :param str filename: full filename for model
    :param int epoch: save epoch idx
    """
    print >> log.v4, "Save model from epoch %i under %s" % (epoch, filename)
    # We add some extra logic to try again for DiskQuota and other errors.
    # This could save us multiple hours of computation.
    try_again_wait_time = 10
    while True:
      try:
        model = h5py.File(filename, "w")
        self.network.save_hdf(model, epoch)
        model.close()
        break
      except IOError as e:
        if e.errno in [errno.EBUSY, errno.EDQUOT, errno.EIO, errno.ENOSPC]:
          print >> log.v3, "Exception while saving:", e
          print >> log.v3, "Trying again in %s secs." % try_again_wait_time
          time.sleep(try_again_wait_time)
          continue
        raise

  def forward_to_hdf(self, data, output_file, combine_labels='', batch_size=0):
    """
    :type data: Dataset.Dataset
    :type output_file: str
    :type combine_labels: str
    """
    cache = h5py.File(output_file, "w")
    batches = data.generate_batches(recurrent_net=self.network.recurrent,
                                    batch_size=batch_size,
                                    max_seqs=self.max_seqs)
    merge = {}
    forwarder = HDFForwardTaskThread(self.network, self.devices, data, batches, cache, merge)
    forwarder.join()
    cache.close()

  # example (http):
  # classify: curl -X POST http://localhost:3333/classify -H "Content-Type: application/json" -d '{"data":[[-0.7, 0.98],[0.62, 1.3]], "classes" : [0,0]}'
  # result: (GET) http://localhost:3333/hash
  #
  # example (rpc/python):
  # import jsonrpclib
  # rpc = jsonrpclib.Server('http://localhost:3334')
  # ret = rpc.classify({"data":[[23],[0]], "classes" : [0,0], "classes-1" : [0,0], "classes-2" : [0,0], "classes-3" : [0,0], "classes-4" : [0,0]})
  # print rpc.result(ret['result']['hash'])

  def daemon(self):
    network = self.network
    devices = self.devices
    classifiers = {}

    def _classify(params):
      ret = { }
      output_dim = {}
      hash = hashlib.new('ripemd160')
      hash.update(json.dumps(params))
      hash = hash.hexdigest()
      for k in params:
        try:
          params[k] = numpy.asarray(params[k], dtype='float32')
          if k != 'data':
            output_dim[k] = network.n_out[k] # = [network.n_in,2] if k == 'data' else network.n_out[k]
        except Exception:
          if k != 'data' and not k in network.n_out:
            ret['error'] = 'unknown target: %s' % k
          else:
            ret['error'] = 'unable to convert %s to an array from value %s' % (k,str(params[k]))
          break
      if not 'error' in ret:
        data = StaticDataset(data=[params], output_dim=output_dim)
        data.init_seq_order()
        try:
          data = StaticDataset(data=[params], output_dim=output_dim)
          data.init_seq_order()
        except Exception:
          ret['error'] = "invalid data: %s" % params
        else:
          batches = data.generate_batches(recurrent_net=network.recurrent,
                                          batch_size=sys.maxsize, max_seqs=1)
          if not hash in classifiers:
            classifiers[hash] = ClassificationTaskThread(network, devices, data, batches)
            classifiers[hash].json_params = params
            print >> log.v3, "classifier started:", hash
          ret['result'] = { 'hash' : hash }
      return ret

    def _result(hash):
      if not classifiers[hash].isAlive():
        return { 'result' : { k : classifiers[hash].result[k].tolist() for k in classifiers[hash].result } }
      else:
        return { 'error' : "classification in progress"}


    class RequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
      def do_POST(self):
        if len(self.path) == 0:
          self.send_response(404)
          return
        self.path = self.path[1:]
        ret = {}
        if self.path in ['classify']:
          ctype, pdict = cgi.parse_header(self.headers.getheader('content-type'))
          if ctype == 'application/json':
            length = int(self.headers.getheader('content-length'))
            params = cgi.parse_qs(self.rfile.read(length),keep_blank_values=1)
            try:
              content = params.keys()[0].decode('utf-8') # this is weird
              params = json.loads(content)
            except Exception:
              ret['error'] = 'unable to decode object'
            else:
              ret.update(_classify(params))
          else:
            ret['error'] = 'invalid header: %s' % ctype
        else:
          ret['error'] = 'invalid command: %s' % self.path
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.wfile.write("\n")
        self.wfile.write(json.dumps(ret))
        self.end_headers()

      def do_GET(self):
        if len(self.path.replace('/', '')) == 0:
          self.send_response(200)
        else:
          if len(self.path) == 0:
            self.send_response(404)
            return
          ret = { 'error' : "" }
          self.path = self.path[1:].split('/')
          if self.path[0] in ['result']:
            if self.path[1] in classifiers:
              if not classifiers[self.path[1]].isAlive():
                ret['result'] = { k : classifiers[self.path[1]].result[k].tolist() for k in classifiers[self.path[1]].result }
              else:
                ret['error'] = "classification in progress"
            else:
              ret['error'] = "unknown hash: " % self.path[1]
          else:
            ret['error'] = "invalid command: %s" % self.path[0]
          self.send_response(200)
          self.send_header('Content-Type', 'application/json')
          self.wfile.write("\n")
          self.wfile.write(json.dumps(ret))
          self.end_headers()

      def log_message(self, format, *args): pass
    class ThreadingServer(SocketServer.ThreadingMixIn, BaseHTTPServer.HTTPServer):
      pass

    httpd = ThreadingServer(("", 3333), RequestHandler)
    print >> log.v3, "httpd listening on port", 3333
    try:
      from jsonrpclib.SimpleJSONRPCServer import SimpleJSONRPCServer # https://pypi.python.org/pypi/jsonrpclib/0.1.6
    except Exception:
      httpd.serve_forever()
    else:
      from thread import start_new_thread
      start_new_thread(httpd.serve_forever, ())
      server = SimpleJSONRPCServer(('localhost', 3334))
      server.register_function(_classify, 'classify')
      server.register_function(_result, 'result')
      print >> log.v3, "json-rpc listening on port", 3334
      server.serve_forever()

  def classify(self, data, output_file):
    out = open(output_file, 'w')
    batches = data.generate_batches(recurrent_net=self.network.recurrent,
                                    batch_size=data.num_timesteps, max_seqs=1)
    forwarder = ClassificationTaskThread(self.network, self.devices, data, batches)
    forwarder.join()
    print >> out, forwarder.output
    out.close()

  def analyze(self, data, statistics):
    """
    :param Dataset.Dataset data:
    :param list[str]|None statistics:
    :return: nothing, will print everything to log.v1
    """
    if statistics is None:
      statistics = ["confusion_matrix"]
    num_labels = len(data.labels)
    if "mle" in statistics:
      mle_labels = list(OrderedDict.fromkeys([ label.split('_')[0] for label in data.labels ]))
      mle_map = [mle_labels.index(label.split('_')[0]) for label in data.labels]
      num_mle_labels = len(mle_labels)
      confusion_matrix = numpy.zeros((num_mle_labels, num_mle_labels), dtype = 'int32')
    else:
      confusion_matrix = numpy.zeros((num_labels, num_labels), dtype = 'int32')
    batches = data.generate_batches(self.network.recurrent, data.num_timesteps, 1)
    num_data_batches = len(batches)
    num_batches = 0
    # TODO: This code is broken.
    while num_batches < num_data_batches:
      alloc_devices = self.allocate_devices(data, batches, num_batches)
      for batch, device in enumerate(alloc_devices):
        device.run('analyze', batch, self.network)
        result, _ = device.result()
        max_c = numpy.argmax(result[0], axis=1)
        if self.network.recurrent:
          real_c = device.targets[:,device.batch_start[batch] : device.batch_start[batch + 1]].flatten()
        else:
          real_c = device.targets[device.batch_start[batch] : device.batch_start[batch + 1]].flatten()
        for i in range(max_c.shape[0]):
          #print real_c[i], max_c[i], len(confusion_matrix[0])
          if "mle" in statistics:
            confusion_matrix[mle_map[int(real_c[i])], mle_map[int(max_c[i])]] += 1
          else:
            confusion_matrix[real_c[i], max_c[i]] += 1
      num_batches += len(alloc_devices)
    if "confusion_matrix" in statistics:
      print >> log.v1, "confusion matrix:"
      for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
          print >> log.v1, str(confusion_matrix[i,j]).rjust(3),
        print >> log.v1, ''
    if "confusion_list" in statistics:
      n = 30
      print >> log.v1, "confusion top" + str(n) + ":"
      top = []
      for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
          if i != j:
            if "mle" in statistics:
              top.append([mle_labels[i] + " -> " + mle_labels[j], confusion_matrix[i,j]])
            else:
              top.append([data.labels[i] + " -> " + data.labels[j], confusion_matrix[i,j]])
      top.sort(key = lambda x: x[1], reverse = True)
      for i in range(n):
        print >> log.v1, top[i][0], top[i][1], str(100 * top[i][1] / float(data.num_timesteps)) + "%"
    if "error" in statistics:
      print >> log.v1, "error:", 1.0 - sum([confusion_matrix[i,i] for i in range(confusion_matrix.shape[0])]) / float(data.num_timesteps)

  def compute_priors(self, dataset, config):
    from Dataset import Dataset
    assert isinstance(dataset, Dataset)
    from Config import Config
    assert isinstance(config, Config)

    assert config.has('output_file'), 'output_file for priors numbers should be provided'
    output_file = config.value('output_file', '')
    target = config.value('target', 'classes')
    extract_type = config.list('extract', ['log-posteriors'])
    assert len(extract_type) == 1
    extract_type = extract_type[0]
    batch_size = config.int('batch_size', 1)
    max_seqs = config.int('max_seqs', -1)
    epoch = config.int('epoch', 1)
    max_seq_length = config.float('max_seq_length', 0)
    if max_seq_length <= 0:
      max_seq_length = sys.maxsize

    dataset.init_seq_order(epoch=epoch)
    batches = dataset.generate_batches(recurrent_net=self.network.recurrent,
                                       batch_size=batch_size,
                                       max_seq_length=max_seq_length,
                                       max_seqs=max_seqs)
    priori_file = open(output_file, 'w')
    forwarder = PriorEstimationTaskThread(network=self.network, devices=self.devices, data=dataset, batches=batches,
                                          priori_file=priori_file, target=target, extract_type=extract_type)
    forwarder.join()
    priori_file.close()


class SeqTrainParallelControl:
  """
  Idea: Parallelize some stuff in seq training (e.g. sprint loss). Can use chunked training.
  We have these steps:
    (1) (forward:GPU) forward only, remember output
    (2) (calc_loss:CPU) calculate loss based on data from (1), error signal. store hat_y = y - grad_L (for stability).
    (3) (train:GPU) forward again, and backprop with data from (2).
  (1) and (3) are on the same GPU, use the same shared params.
  (2) is on CPU.
  (3) is done via the usual loop via EngineTask.TrainTaskThread.
    It calls self.train_wait_for_seqs().

  This class `SeqTrainParallelControl` is instantiated by the Engine and it has the these callbacks which are called
  by the engine (TrainTaskThread):
    train_start_epoch()
    train_finish_epoch()
    train_wait_for_seqs()
  Thus, this instance lives in the main proc and this code is executed in the main proc.

  There is a counterpart of this code living in the device proc and we are calling it via
  Device.seq_train_parallel_control which is an instance of `SeqTrainParallelControlDevHost`.
  Most things are actually happening there.
  """

  def __init__(self, engine, config, **kwargs):
    """
    :type engine: Engine
    :type config: Config.Config
    """
    self.engine = engine
    self.config = config
    self.train_started = None
    self.train_device = None
    self.train_start_seq = 0
    self.forward_current_seq = 0
    self.is_forwarding_finished = False

  def forward_fill_queue(self):
    """
    Full sequence forwarding, no chunking (at the moment).
    """
    assert self.train_started
    if self.is_forwarding_finished: return

    # We will ignore max_seq_length.
    batch_size = self.config.int('batch_size', 1)
    max_seqs = self.config.int('max_seqs', -1)
    if max_seqs <= 0: max_seqs = float('inf')
    dataset = self.engine.train_data
    from EngineBatch import Batch

    # Collect all batches.
    forward_batches = []; ":type: list[EngineBatch.Batch]"
    num_seqs = 0
    while self._device_exec("have_space_in_forward_data_queue", num_seqs=num_seqs):
      # Load next sequence for forwarding, keep all which are still needed for training.
      if not dataset.is_less_than_num_seqs(self.forward_current_seq):
        self.is_forwarding_finished = True
        break
      dataset.load_seqs(self.train_start_seq, self.forward_current_seq + 1)
      seq_len = dataset.get_seq_length(self.forward_current_seq)

      if not forward_batches:
        forward_batches.append(Batch())
      batch = forward_batches[-1]
      dt, ds = batch.try_sequence_as_slice(seq_len)
      if ds > 1 and ((dt * ds).max_value() > batch_size or ds > max_seqs):
        batch = Batch()
        forward_batches.append(batch)
      batch.add_sequence_as_slice(seq_idx=self.forward_current_seq, seq_start_frame=0, length=seq_len)
      num_seqs += 1
      self.forward_current_seq += 1

    # Forward the batches.
    from EngineUtil import assign_dev_data
    for batch in forward_batches:
      print >> log.v4, "SeqTrainParallelControl, forward %r" % batch
      success = assign_dev_data(self.train_device, dataset, [batch], load_seqs=False)
      assert success, "failed to allocate & assign data"
      self.train_device.update_data()
      self._device_exec("do_forward", batch=batch)
      self._device_exec("train_check_calc_loss")

  def train_wait_for_seqs(self, device, batches):
    """
    Called from TrainTaskThread while doing training (forward + backprop).
    This will tell the device what set of batches we want to train next.
    :type device: Device.Device
    :type batches: list[EngineBatch.Batch]
    """
    assert self.train_started
    self.train_device = device
    if not batches: return
    start_seq, end_seq = float("inf"), 0
    for batch in batches:
      start_seq = min(start_seq, batch.start_seq)
      end_seq = max(end_seq, batch.end_seq)
    print >>log.v5, "SeqTrainParallelControl, train_wait_for_seqs start_seq:%i, end_seq:%i" % (start_seq, end_seq)
    assert start_seq < end_seq
    assert start_seq >= self.train_start_seq, "non monotonic seq idx increase"
    while device.wait_for_result_call:
      time.sleep(0.1)  # TaskThread.DeviceRun will call device.result()
    self._device_exec("train_set_cur_batches", batches=batches)
    self.train_start_seq = start_seq
    self.forward_fill_queue()
    while not self._device_exec("train_have_loss_for_cur_batches"):
      if not self._device_exec("train_check_calc_loss"):
        time.sleep(0.1)  # wait until we have the data we need
    self._device_exec("train_set_loss_vars_for_cur_batches")

  def train_start_epoch(self):
    """
    Called from TrainTaskThread at the beginning of a new epoch.
    """
    assert not self.train_started
    self.train_started = True
    self.train_start_seq = 0
    self.train_device = None
    self.forward_current_seq = 0
    self.is_forwarding_finished = False
    self._all_devices_exec("train_start_epoch")

  def train_finish_epoch(self):
    """
    Called from TrainTaskThread at the end of an epoch.
    """
    assert self.train_started
    assert self.is_forwarding_finished, "Forwarding not finished?"
    self._all_devices_exec("train_finish_epoch")
    self.train_started = False

  def _all_devices_exec(self, func, **kwargs):
    for dev in self.engine.devices:
      self.train_device = dev
      self._device_exec(func, **kwargs)

  def _device_exec(self, func, **kwargs):
    assert isinstance(self.train_device, Device.Device)
    return self.train_device._generic_exec_on_dev(("seq_train_parallel_control", func), **kwargs)
