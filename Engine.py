#! /usr/bin/python2.7
import numpy
import sys
import os
from collections import OrderedDict
import threading

import h5py
import json

from Network import LayerNetwork
from EngineBatch import Batch
from EngineTask import TrainTaskThread, EvalTaskThread, SprintCacheForwardTaskThread, HDFForwardTaskThread
import SprintCache
from Log import log
from Updater import Updater
import Device
from LearningRateControl import loadLearningRateControlFromConfig
from Pretrain import pretrainFromConfig


class Engine:

  _last_epoch_batch_model = None; """ :type: (int,int|None,str|None) """  # See get_last_epoch_batch().

  def __init__(self, devices):
    """
    :type devices: list[Device.Device]
    """
    self.devices = devices
    self.is_training = False
    self.training_finished = False
    self.lock = threading.RLock()
    self.cond = threading.Condition(lock=self.lock)
    self.pretrain = None; " :type: Pretrain.Pretrain "

  def set_batch_size(self, data, batch_size, batch_step, max_seqs=-1):
    return data.generate_batches(self.network.recurrent, batch_size, batch_step, max_seqs)

  @classmethod
  def config_get_final_epoch(cls, config):
    """ :type config: Config.Config """
    return config.int('num_epochs', 5)

  @classmethod
  def get_last_epoch_batch_model(cls, config):
    """
    :type config: Config.Config
    :returns (epoch,batch,modelFilename)
    :rtype: (int, int|None, str|None)
    """
    # XXX: We cache it, although this is wrong if we have changed the config.
    if cls._last_epoch_batch_model:
      return cls._last_epoch_batch_model

    model_filename = config.value('model', '')
    assert model_filename, "need 'model' in config"
    load_model_epoch_filename = config.value('load', '')
    start_epoch_mode = config.value('start_epoch', 'auto')
    if start_epoch_mode == 'auto':
      start_epoch = None
    else:
      start_epoch = int(start_epoch_mode)
      assert start_epoch >= 1

    if start_epoch == 1:
      # That is an explicit request to start with the first epoch.
      # Thus, there is no previous epoch.
      # XXX: last batch?
      lastEpochBatchModel = (None, None, None)

    elif start_epoch > 1:
      # If we start with a higher epoch, we must have a previous existing model.
      # Check for it.
      last_epoch = start_epoch - 1
      fns = [cls.epoch_model_filename(model_filename, last_epoch, is_pretrain) for is_pretrain in [False, True]]
      fns_existing = [fn for fn in fns if os.path.exists(fn)]
      assert fns_existing, "start_epoch = %i but model of last epoch not found: %s" % \
                           (start_epoch, cls.epoch_model_filename(model_filename, last_epoch, False))
      # XXX: last batch?
      lastEpochBatchModel = (last_epoch, None, fns_existing[0])

    elif load_model_epoch_filename:
      assert os.path.exists(load_model_epoch_filename)
      last_epoch = LayerNetwork.epoch_from_hdf_model_filename(load_model_epoch_filename)
      # XXX: last batch?
      lastEpochBatchModel = (last_epoch, None, load_model_epoch_filename)

    else:
      # Automatically search the filesystem for existing models.
      file_list = []
      for epoch in range(1, cls.config_get_final_epoch(config) + 1):
        for is_pretrain in [False, True]:
          fn = cls.epoch_model_filename(model_filename, epoch, is_pretrain)
          # XXX: batches, e.g. fn.* ...
          if os.path.exists(fn):
            file_list += [(epoch, None, fn)]
            break
      if len(file_list) == 0:
        lastEpochBatchModel = (None, None, None)
      else:
        file_list.sort()
        lastEpochBatchModel = file_list[-1]

    cls._last_epoch_batch_model = lastEpochBatchModel
    return lastEpochBatchModel

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
    last_epoch, last_batch, _ = cls.get_last_epoch_batch_model(config)
    if last_epoch is None:
      start_epoch = 1
      start_batch = start_batch_config or 0
    elif start_batch_config is not None:
      # We specified a start batch. Stay in the same epoch, use that start batch.
      start_epoch = last_epoch
      start_batch = start_batch_config
    elif last_batch is None:
      # No batch -> start with next epoch.
      start_epoch = last_epoch + 1
      start_batch = 0
    else:
      # Stay in last epoch and start with next batch.
      start_epoch = last_epoch
      start_batch = last_batch + 1
    return start_epoch, start_batch

  def init_train_from_config(self, config, train_data, dev_data=None, eval_data=None):
    """
    :type config: Config.Config
    :type train_data: Dataset.Dataset
    :type dev_data: Dataset.Dataset | None
    :type eval_data: Dataset.Dataset | None
    :type start_epoch: int | None
    """
    self.train_data = train_data
    self.dev_data = dev_data
    self.eval_data = eval_data
    self.start_epoch, self.start_batch = self.get_train_start_epoch_batch(config)
    self.batch_size, self.batch_step = config.int_pair('batch_size', (1,1))
    self.model_filename = config.value('model', None)
    self.save_model_epoch_interval = config.int('save_interval', 1)
    self.learning_rate_control = loadLearningRateControlFromConfig(config)
    self.learning_rate = self.learning_rate_control.initialLearningRate
    self.final_epoch = self.config_get_final_epoch(config)  # Inclusive.
    self.max_seqs = config.int('max_seqs', -1)
    self.updater = Updater.initFromConfig(config)
    self.pretrain = pretrainFromConfig(config)
    self.pad_batches = config.bool("pad", False)
    self.ctc_prior_file = config.value('ctc_prior_file', None)
    # And also initialize the network. That depends on some vars here such as pretrain.
    self.init_network_from_config(config)

  def init_network_from_config(self, config):
    last_epoch, _, last_model_epoch_filename = self.get_last_epoch_batch_model(config)

    if last_model_epoch_filename:
      print >> log.v1, "loading weights from", last_model_epoch_filename
      model = h5py.File(last_model_epoch_filename, "r")
    else:
      model = None

    if config.bool('initialize_from_model', False):
      # That's only about the topology, not the params.
      print >> log.v5, "initializing network topology from model"
      assert model, "last model not specified. use 'load' in config. or don't use 'initialize_from_model'"
      network = LayerNetwork.from_hdf_model_topology(model)
    else:
      if self.pretrain:  # set via self.init_train_from_config()
        # This would be obsolete if we don't want to load an existing model.
        # In self.init_train_epoch(), we initialize a new model.
        network = self.pretrain.get_network_for_epoch(last_epoch or self.start_epoch)
      else:
        network = LayerNetwork.from_config_topology(config)

    # We have the parameters randomly initialized at this point.
    # Maybe load existing model parameters.
    if model:
      last_epoch_model = network.load_hdf(model)
      assert last_epoch == last_epoch_model
      model.close()

    self.network = network

    if config.has('dump_json'):
      self.network_dump_json(config.value('dump_json', ''))

    self.print_network_info()

  def network_dump_json(self, json_filename):
    fout = open(json_filename, 'w')
    try:
      json_content = self.network.to_json()
      print json_content
      print "---------------"
      json_data = json.loads(json_content)
      print json_data
      print "---------------"
      print json.dumps(json_data, indent = 2)
      print "---------------"
      print >> fout, json.dumps(json_data, indent = 2)
    except ValueError:
      print >> log.v5, self.network.to_json()
      assert False, "JSON parsing failed"
    fout.close()

  def print_network_info(self):
    network = self.network
    print >> log.v2, "Network layer topology:"
    print >> log.v2, "  input #:", network.n_in
    if network.description:
      for i in xrange(len(network.description.hidden_info)):
        print >> log.v2, "  " + network.description.hidden_info[i][0] + " #:", network.description.hidden_info[i][1]
    print >> log.v2, "  output #:", network.n_out
    print >> log.v2, "net params #:", network.num_params()
    print >> log.v2, "net trainable params:", network.train_params
    print >> log.v5, "net output:", network.output

  def train(self):
    print >> log.v3, "start training at epoch %i and batch %i" % (self.start_epoch, self.start_batch)
    print >> log.v3, "using batch size/step: %i, %i, max seqs: %i" % (self.batch_size, self.batch_step, self.max_seqs)
    print >> log.v3, "learning rate control:", self.learning_rate_control
    print >> log.v3, "pretrain:", self.pretrain
    self.eval_datasets = {}; """ :type: dict[str,(Dataset.Dataset,list[Batch])] """
    for name, dataset in [("eval", self.dev_data), ("eval", self.eval_data)]:
      if not dataset: continue
      if dataset.num_seqs == 0: continue
      self.eval_datasets[name] = (dataset, self.set_batch_size(dataset, self.batch_size, self.batch_step))
    if self.network.loss == 'priori':
      prior = self.train_data.calculate_priori()
      self.network.output.priori.set_value(prior)
      self.network.output.initialize()

    with self.lock:
      self.is_training = True
      self.epoch = 0  # Not yet started. Will be >=1 later.
      self.training_finished = False
      self.cond.notify_all()

    assert self.start_epoch >= 1, "Epochs start at 1."
    assert self.start_epoch <= self.final_epoch, "No epochs to train, start_epoch: %i, final_epoch: %i" % \
                                                 (self.start_epoch, self.final_epoch)

    for epoch in xrange(self.start_epoch, self.final_epoch + 1):  # Epochs start at 1.
      # In case of random seq ordering, we want to reorder each epoch.
      self.train_data.init_seq_order(epoch=epoch)
      with self.lock:
        # Notify about current epoch after we initialized the dataset seq order.
        self.epoch = epoch
        self.cond.notify_all()

      self.init_train_epoch()
      self.train_epoch()

    # Save final model.
    if self.model_filename:
      self.save_model(self.model_filename + ".%03d" % self.final_epoch, self.final_epoch)

    with self.lock:
      self.is_training = False
      self.training_finished = True
      self.final_epoch = None
      self.epoch = None
      self.cond.notify_all()

  @classmethod
  def epoch_model_filename(cls, model_filename, epoch, is_pretrain):
    """
    :type model_filename: str
    :type epoch: int
    :type is_pretrain: bool
    :rtype: str
    """
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
      self.learning_rate_control.setLearningRateForEpoch(self.epoch, self.learning_rate)
    elif self.is_first_epoch_after_pretrain():
      # Use constant learning rate.
      self.learning_rate_control.setLearningRateForEpoch(self.epoch, self.learning_rate)
    else:
      self.learning_rate = self.learning_rate_control.getLearningRateForEpoch(self.epoch)

    if not self.is_pretrain_epoch():
      # Train the whole network.
      self.network.declare_train_params()

  def train_epoch(self):
    print >> log.v1, "start", self.get_epoch_str(), "with learning rate", self.learning_rate, "..."

    if self.is_pretrain_epoch():
      self.print_network_info()

    training_devices = self.devices
    train_batches = self.set_batch_size(self.train_data, self.batch_size, self.batch_step, self.max_seqs)

    start_batch = self.start_batch if self.epoch == self.start_epoch else 0
    trainer = TrainTaskThread(self.network, training_devices, self.train_data, train_batches,
                              self.learning_rate, self.updater, start_batch, self.pad_batches)
    trainer.join()
    if not trainer.finalized:
      if trainer.device_crash_batch is not None:  # Otherwise we got an unexpected exception - a bug in our code.
        self.save_model(self.get_epoch_model_filename() + ".crash_%i" % trainer.device_crash_batch, self.epoch - 1)
      sys.exit(1)

    assert not (numpy.isinf(trainer.score) or numpy.isnan(trainer.score)), \
      "Model is broken, got inf or nan final score: %s" % trainer.score

    if self.model_filename and (self.epoch % self.save_model_epoch_interval == 0):
      self.save_model(self.get_epoch_model_filename(), self.epoch)
    self.learning_rate_control.setEpochError(self.epoch, trainer.score)
    self.learning_rate_control.save()

    if log.verbose[1]:
      eval_dump_str = []
      testing_device = self.devices[-1]
      for name in self.eval_datasets.keys():
        data, num_batches = self.eval_datasets[name]
        tester = EvalTaskThread(self.network, [testing_device], data, num_batches, self.pad_batches)
        tester.join()
        trainer.elapsed += tester.elapsed
        eval_dump_str += ["  %s: score %s error %s" % (name, tester.score, tester.error)]
      print >> log.v1, self.get_epoch_str(), "elapsed:", trainer.elapsed, "score:", trainer.score
      print >> log.v1, "\n".join(eval_dump_str)
      if self.ctc_prior_file is not None:
        trainer.save_ctc_priors(self.ctc_prior_file, self.get_epoch_str())

  def save_model(self, filename, epoch):
    """
    :param str filename: full filename for model
    :param int epoch: save epoch idx
    """
    model = h5py.File(filename, "w")
    self.network.save_hdf(model, epoch)
    model.close()

  def forward_to_sprint(self, data, cache_file, combine_labels=''):
    """
    :type data: Dataset.Dataset
    :type cache_file: str
    :type combine_labels: str
    """
    cache = SprintCache.FileArchive(cache_file)
    batches = self.set_batch_size(data, data.num_timesteps, data.num_timesteps, 1)
    merge = {}
    if combine_labels != '':
      for index, label in enumerate(data.labels):
        merged = combine_labels.join(label.split(combine_labels)[:-1])
        if merged == '': merged = label
        if not merged in merge.keys():
          merge[merged] = []
        merge[merged].append(index)
      import codecs
      label_file = codecs.open(cache_file + ".labels", encoding = 'utf-8', mode = 'w')
      for key in merge.keys():
        label_file.write(key + "\n")
      label_file.close()
    forwarder = SprintCacheForwardTaskThread(self.network, self.devices, data, batches, cache, merge)
    forwarder.join()
    cache.finalize()

  def forward_to_hdf(self, data, output_file, combine_labels=''):
    """
    :type data: Dataset.Dataset
    :type output_file: str
    :type combine_labels: str
    """
    cache = h5py.File(output_file, "w")
    batches = self.set_batch_size(data, data.num_timesteps, data.num_timesteps, 1)
    merge = {}
    if combine_labels != '':
      for index, label in enumerate(data.labels):
        merged = combine_labels.join(label.split(combine_labels)[:-1])
        if merged == '': merged = label
        if not merged in merge.keys():
          merge[merged] = []
        merge[merged].append(index)
      import codecs
      label_file = codecs.open(output_file + ".labels", encoding = 'utf-8', mode = 'w')
      for key in merge.keys():
        label_file.write(key + "\n")
      label_file.close()
    forwarder = HDFForwardTaskThread(self.network, self.devices, data, batches, cache, merge)
    forwarder.join()
    cache.close()

  def classify(self, device, data, label_file):
    batches = self.set_batch_size(data, data.num_timesteps, data.num_timesteps, 1)
    num_data_batches = len(batches)
    num_batches = 0
    out = open(label_file, 'w')
    while num_batches < num_data_batches:
      alloc_devices = self.allocate_devices(data, batches, num_batches)
      for batch, device in enumerate(alloc_devices):
        device.run('classify', self.network)
        labels = numpy.concatenate(device.result(), axis = 1)
        print >> log.v5, "labeling", len(labels), "time steps for sequence", data.tags[num_batches + batch]
        print >> out, data.tags[num_batches + batch],
        for label in labels: print >> out, data.labels[label],
        print >> out, ''
      num_batches += len(alloc_devices)
    out.close()

  def analyze(self, device, data, statistics):
    num_labels = len(data.labels)
    if "mle" in statistics:
      mle_labels = list(OrderedDict.fromkeys([ label.split('_')[0] for label in data.labels ]))
      mle_map = [mle_labels.index(label.split('_')[0]) for label in data.labels]
      num_mle_labels = len(mle_labels)
      confusion_matrix = numpy.zeros((num_mle_labels, num_mle_labels), dtype = 'int32')
    else:
      confusion_matrix = numpy.zeros((num_labels, num_labels), dtype = 'int32')
    batches = self.set_batch_size(data, data.num_timesteps, 1)
    num_data_batches = len(batches)
    num_batches = 0
    while num_batches < num_data_batches:
      alloc_devices = self.allocate_devices(data, batches, num_batches)
      for batch, device in enumerate(alloc_devices):
        device.run('analyze', batch, self.network)
        result = device.result()
        max_c = numpy.argmax(result[0], axis=1)
        if self.network.recurrent:
          real_c = device.targets[:,device.batch_start[batch] : device.batch_start[batch + 1]].flatten()
        else:
          real_c = device.targets[device.batch_start[batch] : device.batch_start[batch + 1]].flatten()
        for i in xrange(max_c.shape[0]):
          #print real_c[i], max_c[i], len(confusion_matrix[0])
          if "mle" in statistics:
            confusion_matrix[mle_map[int(real_c[i])], mle_map[int(max_c[i])]] += 1
          else:
            confusion_matrix[real_c[i], max_c[i]] += 1
      num_batches += len(alloc_devices)
    if "confusion_matrix" in statistics:
      print >> log.v1, "confusion matrix:"
      for i in xrange(confusion_matrix.shape[0]):
        for j in xrange(confusion_matrix.shape[1]):
          print >> log.v1, str(confusion_matrix[i,j]).rjust(3),
        print >> log.v1, ''
    if "confusion_list" in statistics:
      n = 30
      print >> log.v1, "confusion top" + str(n) + ":"
      top = []
      for i in xrange(confusion_matrix.shape[0]):
        for j in xrange(confusion_matrix.shape[1]):
          if i != j:
            if "mle" in statistics:
              top.append([mle_labels[i] + " -> " + mle_labels[j], confusion_matrix[i,j]])
            else:
              top.append([data.labels[i] + " -> " + data.labels[j], confusion_matrix[i,j]])
      top.sort(key = lambda x: x[1], reverse = True)
      for i in xrange(n):
        print >> log.v1, top[i][0], top[i][1], str(100 * top[i][1] / float(data.num_timesteps)) + "%"
    if "error" in statistics:
      print >> log.v1, "error:", 1.0 - sum([confusion_matrix[i,i] for i in xrange(confusion_matrix.shape[0])]) / float(data.num_timesteps)
