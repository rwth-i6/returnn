#! /usr/bin/python2.7

import numpy
import sys
import os
from collections import OrderedDict
import threading
import h5py
import json
from Network import LayerNetwork
from EngineTask import TrainTaskThread, EvalTaskThread, SprintCacheForwardTaskThread, HDFForwardTaskThread
import SprintCache
from Log import log
from Updater import Updater
import Device
from LearningRateControl import loadLearningRateControlFromConfig
from Pretrain import pretrainFromConfig
import EngineUtil
from Util import hms
import errno
import time


class Engine:

  _last_epoch_model = None; """ :type: (int|None,str|None) """  # See get_last_epoch_model().

  def __init__(self, devices):
    """
    :type devices: list[Device.Device]
    """
    self.devices = devices
    self.train_data = None; " :type: Dataset.Dataset "
    self.is_training = False
    self.training_finished = False
    self.stop_train_after_epoch_request = False
    self.pretrain = None; " :type: Pretrain.Pretrain "
    self.init_train_epoch_posthook = None

  @classmethod
  def config_get_final_epoch(cls, config):
    """ :type config: Config.Config """
    return config.int('num_epochs', 5)

  @classmethod
  def get_existing_models(cls, config):
    model_filename = config.value('model', '')
    assert model_filename, "need 'model' in config"
    # Automatically search the filesystem for existing models.
    file_list = []
    for epoch in range(1, cls.config_get_final_epoch(config) + 1):
      for is_pretrain in [False, True]:
        fn = cls.epoch_model_filename(model_filename, epoch, is_pretrain)
        if os.path.exists(fn):
          file_list += [(epoch, fn)]  # epoch, fn
          break
    file_list.sort()
    return file_list

  @classmethod
  def get_last_epoch_model(cls, config):
    """
    :type config: Config.Config
    :returns (epoch, modelFilename)
    :rtype: (int|None, str|None)
    """
    # XXX: We cache it, although this is wrong if we have changed the config.
    if cls._last_epoch_model:
      return cls._last_epoch_model

    load_model_epoch_filename = config.value('load', '')
    start_epoch_mode = config.value('start_epoch', 'auto')
    if start_epoch_mode == 'auto':
      start_epoch = None
    else:
      start_epoch = int(start_epoch_mode)
      assert start_epoch >= 1

    existing_models = cls.get_existing_models(config)

    if existing_models:
      last = existing_models[-1]

    elif load_model_epoch_filename:
      assert os.path.exists(load_model_epoch_filename)
      last_epoch = LayerNetwork.epoch_from_hdf_model_filename(load_model_epoch_filename)
      last = (last_epoch, load_model_epoch_filename)

    else:
      last = (None, None)

    if start_epoch == 1:
      assert last[0] is None, "There is an existing model: %s" % (last,)
    elif start_epoch > 1:
      assert last[0] == start_epoch - 1, "start_epoch %i but there is %s" % (start_epoch, last)

    cls._last_epoch_model = last
    return last

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
    last_epoch, _ = cls.get_last_epoch_model(config)
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
    :type start_epoch: int | None
    """
    self.train_data = train_data
    self.dev_data = dev_data
    self.eval_data = eval_data
    self.start_epoch, self.start_batch = self.get_train_start_epoch_batch(config)
    self.batch_size = config.int('batch_size', 1)
    self.update_batch_size = config.int('update_batch_size', 0)
    self.model_filename = config.value('model', None)
    self.save_model_epoch_interval = config.int('save_interval', 1)
    self.learning_rate_control = loadLearningRateControlFromConfig(config)
    self.learning_rate = self.learning_rate_control.initialLearningRate
    self.initial_learning_rate = self.learning_rate
    self.pretrain_learning_rate = config.float('pretrain_learning_rate', self.learning_rate)
    self.final_epoch = self.config_get_final_epoch(config)  # Inclusive.
    self.max_seqs = config.int('max_seqs', -1)
    self.updater = Updater.initFromConfig(config)
    self.pad_batches = config.bool("pad", False)
    self.ctc_prior_file = config.value('ctc_prior_file', None)
    self.exclude = config.int_list('exclude', [])
    self.init_train_epoch_posthook = config.value('init_train_epoch_posthook', None)
    # And also initialize the network. That depends on some vars here such as pretrain.
    self.init_network_from_config(config)

  def init_network_from_config(self, config):
    self.pretrain = pretrainFromConfig(config)

    last_epoch, last_model_epoch_filename = self.get_last_epoch_model(config)

    if last_model_epoch_filename:
      print >> log.v1, "loading weights from", last_model_epoch_filename
      last_model_hdf = h5py.File(last_model_epoch_filename, "r")
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
        network = self.pretrain.get_network_for_epoch(last_epoch or self.start_epoch)
      else:
        network = LayerNetwork.from_config_topology(config)

    # We have the parameters randomly initialized at this point.
    # Maybe load existing model parameters.
    if last_model_hdf:
      _last_epoch = network.load_hdf(last_model_hdf)
      assert last_epoch == _last_epoch
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
      for info in network.description.hidden_info:
        print >> log.v2, "  " + info["layer_class"] + " #:", info["n_out"]
    else:
      print >> log.v2, "  ..."
    print >> log.v2, "  output #:", network.n_out
    print >> log.v2, "net params #:", network.num_params()
    print >> log.v2, "net trainable params:", network.train_params_vars
    print >> log.v5, "net output:", network.output

  def check_last_epoch(self):
    if self.start_epoch == 1:
      return
    self.epoch = self.start_epoch - 1
    if self.learning_rate_control:
      if self.dev_data:
        if "dev_score" not in self.learning_rate_control.getEpochErrorDict(self.epoch):
          # This can happen when we have a previous model but did not test it yet.
          print >> log.v3, "Last epoch model not yet evaluated on dev. Doing that now."
          self.eval_model()

  def train(self):
    if self.start_epoch:
      print >> log.v3, "start training at epoch %i and batch %i" % (self.start_epoch, self.start_batch)
    print >> log.v4, "using batch size: %i, max seqs: %i" % (self.batch_size, self.max_seqs)
    print >> log.v4, "learning rate control:", self.learning_rate_control
    print >> log.v4, "pretrain:", self.pretrain
    if self.network.loss == 'priori':
      prior = self.train_data.calculate_priori()
      self.network.output.priori.set_value(prior)
      self.network.output.initialize()

    if self.network.recurrent:
      assert not self.train_data.shuffle_frames_of_nseqs, "Frames must not be shuffled in recurrent net."

    self.is_training = True
    self.training_finished = False

    assert self.start_epoch >= 1, "Epochs start at 1."
    if self.start_epoch > self.final_epoch:
      print >> log.v1, "No epochs to train, start_epoch: %i, final_epoch: %i" % \
                       (self.start_epoch, self.final_epoch)

    self.check_last_epoch()

    for epoch in xrange(self.start_epoch, self.final_epoch + 1):  # Epochs start at 1.
      # In case of random seq ordering, we want to reorder each epoch.
      self.train_data.init_seq_order(epoch=epoch)
      self.epoch = epoch

      for dataset in self.get_eval_datasets().values():
        dataset.init_seq_order(self.epoch)

      self.init_train_epoch()
      self.train_epoch()

      if self.stop_train_after_epoch_request:
        self.stop_train_after_epoch_request = False
        break

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
      self.learning_rate_control.setLearningRateForEpoch(self.epoch, self.learning_rate)
    elif self.is_first_epoch_after_pretrain():
      # Use constant learning rate.
      self.learning_rate = self.initial_learning_rate
      self.learning_rate_control.setLearningRateForEpoch(self.epoch, self.learning_rate)
    else:
      self.learning_rate = self.learning_rate_control.getLearningRateForEpoch(self.epoch)

    if not self.is_pretrain_epoch():
      # Train the whole network.
      self.network.declare_train_params()

    if self.init_train_epoch_posthook:
      print >> log.v5, "execute init_train_epoch_posthook:", self.init_train_epoch_posthook
      exec self.init_train_epoch_posthook

  def train_epoch(self):
    print >> log.v4, "start", self.get_epoch_str(), "with learning rate", self.learning_rate, "..."

    if self.is_pretrain_epoch():
      self.print_network_info()

    training_devices = self.devices
    train_batches = self.train_data.generate_batches(recurrent_net=self.network.recurrent,
                                                     batch_size=self.batch_size,
                                                     max_seqs=self.max_seqs)

    start_batch = self.start_batch if self.epoch == self.start_epoch else 0
    trainer = TrainTaskThread(self.network, training_devices, data=self.train_data, batches=train_batches,
                              learning_rate=self.learning_rate, updater=self.updater,
                              update_batch_size=self.update_batch_size,
                              start_batch=start_batch, pad_batches=self.pad_batches,
                              exclude=self.exclude,
                              report_prefix=("pre" if self.is_pretrain_epoch() else "") + "train epoch %s" % self.epoch)
    trainer.join()
    if not trainer.finalized:
      if trainer.device_crash_batch is not None:  # Otherwise we got an unexpected exception - a bug in our code.
        self.save_model(self.get_epoch_model_filename() + ".crash_%i" % trainer.device_crash_batch, self.epoch - 1)
      sys.exit(1)

    assert not (numpy.isinf(trainer.score) or numpy.isnan(trainer.score)), \
      "Model is broken, got inf or nan final score: %s" % trainer.score

    if self.model_filename and (self.epoch % self.save_model_epoch_interval == 0):
      self.save_model(self.get_epoch_model_filename(), self.epoch)
    self.learning_rate_control.setEpochError(self.epoch, {"train_score": trainer.score})
    self.learning_rate_control.save()
    if self.ctc_prior_file is not None:
      trainer.save_ctc_priors(self.ctc_prior_file, self.get_epoch_str())

    print >> log.v1, self.get_epoch_str(), "score:", trainer.score, "elapsed:", hms(trainer.elapsed)
    self.eval_model()

  def eval_model(self):
    eval_dump_str = []
    for dataset_name, dataset in self.get_eval_datasets().items():
      batches = dataset.generate_batches(recurrent_net=self.network.recurrent, batch_size=self.batch_size)
      tester = EvalTaskThread(self.network, self.devices, data=dataset, batches=batches,
                              pad_batches=self.pad_batches,
                              report_prefix=self.get_epoch_str() + " eval")
      tester.join()
      eval_dump_str += ["  %s: score %s error %s" % (dataset_name, tester.score, tester.error)]
      if dataset_name == "dev":
        self.learning_rate_control.setEpochError(self.epoch, {"dev_score": tester.score, "dev_error": tester.error})
        self.learning_rate_control.save()
    print >> log.v1, "\n".join(eval_dump_str)

  def save_model(self, filename, epoch):
    """
    :param str filename: full filename for model
    :param int epoch: save epoch idx
    """
    print >> log.v3, "Save model from epoch %i under %s" % (epoch, filename)
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

  def forward_to_sprint(self, data, cache_file, combine_labels=''):
    """
    :type data: Dataset.Dataset
    :type cache_file: str
    :type combine_labels: str
    """
    cache = SprintCache.FileArchive(cache_file)
    batches = data.generate_batches(recurrent_net=self.network.recurrent,
                                    batch_size=data.get_num_timesteps(),
                                    max_seqs=1)
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
    batches = data.generate_batches(recurrent_net=self.network.recurrent,
                                    batch_size=data.get_num_timesteps(), max_seqs=1)
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
    batches = data.generate_batches(recurrent_net=self.network.recurrent,
                                    batch_size=data.num_timesteps, max_seqs=1)
    num_data_batches = len(batches)
    num_batches = 0
    # TODO: This code is broken.
    out = open(label_file, 'w')
    while num_batches < num_data_batches:
      alloc_devices = self.allocate_devices(data, batches, num_batches)
      for batch, device in enumerate(alloc_devices):
        device.run('classify', self.network)
        labels = numpy.concatenate(device.result()[0], axis = 1)
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
