#! /usr/bin/python2.7

import SprintCache
import numpy
import theano
import h5py
import time
import sys
from Log import log
from Updater import Updater
from Util import hdf5_strings, terminal_size, progress_bar, hms
from collections import OrderedDict
import threading, thread
import atexit
import Device
from LearningRateControl import loadLearningRateControlFromConfig
from Pretrain import pretrainFromConfig


class Batch:
  """
  A batch can consists of several sequences (= segments).
  Note that self.shape[1] is a different kind of batch - related to the data-batch-idx (= seq-idx).
  """

  def __init__(self, start = (0, 0)):
    """
    :type start: list[int]
    """
    self.shape = [0, 0]  # format (time,batch)
    self.start = list(start)  # format (start seq idx in data, start frame idx in seq)
    self.nseqs = 1
    """
    nseqs is the number of sequences which we cover (not data-batches self.shape[1]).
    For recurrent NN training, shape[1] == nseqs.
    For FF NN training, we concatenate all seqs, so shape[1] == 1 but nseqs >= 1.
    """

  def __repr__(self):
    return "<Batch %r %r>" % (self.shape, self.start)

  def try_sequence(self, length):
    """
    :param int length: number of (time) frames
    :return: new shape which covers the old shape and one more data-batch
    :rtype: list[int]
    """
    return [max(self.shape[0], length), self.shape[1] + 1]

  def add_sequence(self, length):
    """
    Adds one data-batch.
    :param int length: number of (time) frames
    """
    self.shape = self.try_sequence(length)

  def add_frames(self, length):
    """
    Adds frames to all data-batches.
    Will add one data-batch if we don't have one yet.
    :param int length: number of (time) frames
    """
    self.shape = [self.shape[0] + length, max(self.shape[1], 1)]

  def size(self):
    return self.shape[0] * self.shape[1]

  def get_end_seq(self):
    return self.start[0] + max(self.nseqs, self.shape[1])


def assign_dev_data(device, dataset, batches, recurrent=False, pad_batches=False):
  """
  :type device: Device.Device
  :type dataset: Dataset.Dataset
  :type batches: list[Batch]
  :type recurrent: bool
  :type pad_batches: bool
  :returns successful and how much batch idx to advance.
  :rtype: (bool,int)
  """
  # The final device.data.shape is in format (time,batch,feature).
  shape = [0, 0]
  for batch in batches:
    shape = [max(shape[0], batch.shape[0]), shape[1] + batch.shape[1]]
  if shape[1] == 0:
    return False, len(batches)

  device.alloc_data(shape + [dataset.num_inputs * dataset.window], dataset.max_ctc_length, pad=pad_batches)
  offset = 0
  for i, batch in enumerate(batches):
    if not dataset.have_seqs(batch.start[0], batch.get_end_seq()):
      # We could also just skip those seqs. However, we might want to keep all batches
      # of similar sizes to have more stable training. Thus, we skip this batch.
      return False, i + 1

    dataset.load_seqs(batch.start[0], batch.get_end_seq())
    idi = dataset.alloc_interval_index(batch.start[0])
    if recurrent:
      for s in xrange(batch.start[0], batch.start[0] + batch.shape[1]):
        ids = dataset.seq_index[s]  # the real seq idx after sorting
        l = dataset.seq_lengths[ids]
        o = dataset.seq_start[s] + batch.start[1] - dataset.seq_start[dataset.alloc_intervals[idi][0]]
        q = s - batch.start[0] + offset
        device.data[:l, q] = dataset.alloc_intervals[idi][2][o:o + l]
        device.targets[:l, q] = dataset.targets[dataset.seq_start[s] + batch.start[1]:dataset.seq_start[s] + batch.start[1] + l]
        if pad_batches:
          #pad with equivalent to 0
          #these are the hardcoded values for IAM
          #TODO load this from somewhere
          pad_data = [-1.46374, -0.151816, -0.161173, 0.0686325, 0.0231148, -0.154613,
                      -0.105614, 0.00550198, 0.0911985, 0.00502809, 0.0512826, -0.0181915,
                      0.0225053, -0.00149681, 0.0782062, 0.0412163, 0.0526166, -0.0722563,
                      0.0268245, -0.0277465, 0.258805, -0.187777, -2.3835, -1.42065]
          device.data[l:, q] = pad_data
          #also pad targets
          #hardcoded si for IAM
          #TODO load this from somewhere
          pad_target = 189
          device.targets[l:, q] = pad_target
        #only copy ctc targets if chunking is inactive to avoid out of range access (ctc is not comaptible with chunking anyway)
        chunking_active = dataset.chunk_size > 0
        if dataset.ctc_targets is not None and not chunking_active:
          device.ctc_targets[q] = dataset.ctc_targets[ids]
        device.tags[q] = dataset.tags[ids] #TODO
        device.index[:l, q] = numpy.ones((l,), dtype = 'int8')
      offset += batch.shape[1]
    else:
      o = dataset.seq_start[batch.start[0]] + batch.start[1] - dataset.seq_start[dataset.alloc_intervals[idi][0]]
      l = batch.shape[0]
      device.data[offset:offset + l, 0] = dataset.alloc_intervals[idi][2][o:o + l]
      device.targets[offset:offset + l, 0] = dataset.targets[dataset.seq_start[batch.start[0]] + batch.start[1]:dataset.seq_start[batch.start[0]] + batch.start[1] + l] #data.targets[o:o + l]
      device.index[offset:offset + l, 0] = numpy.ones((l,), dtype = 'int8')
      offset += l

  return True, len(batches)


def assign_dev_data_single_seq(device, dataset, seq):
  """
  :type device: Device.Device
  :type dataset: Dataset.Dataset
  :param int seq: sorted seq idx
  :return: whether we succeeded
  :rtype: bool
  """
  if not dataset.have_seqs(seq, seq + 1):
    return False
  batch = Batch([seq, 0])
  batch.shape = (dataset.get_seq_length(seq), 1)
  success, _ = assign_dev_data(device, dataset, [batch])
  return success


class TaskThread(threading.Thread):
    def __init__(self, task, network, devices, data, batches, start_batch=0, pad_batches=False):
      """
      :type task: str
      :type network: Network.LayerNetwork
      :type devices: list[Device.Device]
      :type data: Dataset.Dataset
      :type batches: list[Batch]
      :type start_batch: int
      :type pad_batches: bool
      """
      threading.Thread.__init__(self, name="TaskThread %s" % task)
      self.start_batch = start_batch
      self.pad_batches = pad_batches
      self.devices = devices
      self.network = network
      self.batches = batches
      self.task = task
      self.data = data
      self.daemon = True
      self.elapsed = 0
      self.finalized = False
      self.score = None
      self.device_crash_batch = None
      # There is no generic way to see whether Python is exiting.
      # This is our workaround. We check for it in self.run_inner().
      self.stopped = False
      atexit.register(self.stop)
      self.start()

    def stop(self):
      self.stopped = True

    def allocate_devices(self, start_batch):
      """
      Sets the device data, i.e. the next batches, via self.batches.
      This calls Dataset.load_seqs() to get the data.
      This sets:
        device.data
        device.targets
        device.ctc_targets
        device.tags
        device.index
      :param int start_batch: start batch index, index of self.batches
      :rtype: (list[Device.Device], int)
      :return list of used devices, and number of batches which were handled
      Number of batches will always be positive, but devices could be empty on skipped seqs.
      """
      devices = []; """ :type: list[Device.Device] """
      batch_idx = start_batch
      for device in self.devices:
        batches = self.batches[batch_idx:batch_idx + device.num_batches]
        success, batch_adv_idx = assign_dev_data(device, self.data, batches, self.network.recurrent, self.pad_batches)
        if success:
          devices.append(device)
        else:
          # We expect that there was a problem with batch_idx + batch_adv_idx - 1.
          assert batch_adv_idx > 0
          print >> log.v3, "Skipping batches %s because some seqs at %i are missing" % \
                           (range(batch_idx, batch_idx + batch_adv_idx),
                            batches[batch_adv_idx - 1].start[0])
        batch_idx += batch_adv_idx
      batch_adv_idx = batch_idx - start_batch
      assert batch_adv_idx > 0
      return devices, batch_adv_idx

    def prepare_device_for_batch(self, device):
      """ :type device: Device.Device """
      pass
    def get_device_prepare_args(self):
      return {"network": self.network, "updater": None}
    def evaluate(self, batch, results, num_frames):
      """
      :param int batch: start batch
      :param list[list[numpy.ndarray]] result: results from devices
      :type num_frames: int
      :returns some score or None
      """
      pass
    def initialize(self):
      pass
    def finalize(self):
      self.finalized = True

    def run(self):
      # Wrap run_inner() for better exception printing.
      # Thread.__bootstrap_inner() ignores sys.excepthook.
      try:
        self.run_inner()
      except IOError, e:  # Such as broken pipe.
        print >> log.v2, "%s. Some device proc crashed unexpectedly. Maybe just SIGINT." % e
        # Just pass on. We have self.finalized == False which indicates the problem.
      except Exception:
        # Catch all standard exceptions.
        # These are not device errors. We should have caught them in the code
        # and we would leave self.finalized == False.
        # Don't catch KeyboardInterrupt here because that will get send by the main thread
        # when it is exiting. It's never by the user because SIGINT will always
        # trigger KeyboardInterrupt in the main thread only.
        try:
          print("%s failed" % self)
          sys.excepthook(*sys.exc_info())
        finally:
          # Exceptions are fatal. If we can recover, we should handle it in run_inner().
          thread.interrupt_main()

    class DeviceBatchRun:
      def __init__(self, parent, batch_idx):
        """
        :type parent: TaskThread
        """
        self.parent = parent
        self.batch_idx = batch_idx
        self.score = None
        self.num_frames = 0

      def finish(self):
        """
        :returns whether everything is fine.
        """
        if not self.alloc_devices:
          # We skipped segments. That's fine.
          return True

        device_results = self.device_collect_results(self.alloc_devices)
        if device_results is None:
          print >> log.v2, "device crashed on batch", self.batch_idx
          self.parent.device_crash_batch = self.batch_idx
          return False
        assert len(device_results) == len(self.alloc_devices)

        self.score = self.parent.evaluate(self.batch_idx, device_results, self.num_frames)

        self.print_process()
        return True

      def start(self):
        self.batch_start_time = time.time()
        self.alloc_devices, self.num_alloc_batches = self.parent.allocate_devices(start_batch=self.batch_idx)
        assert self.num_alloc_batches > 0
        # Note that alloc_devices could be empty if we skipped seqs.
        if not self.alloc_devices:
          return
        self.device_run()

      def device_run(self):
        batch = self.batch_idx
        for device in self.alloc_devices:
          if self.parent.network.recurrent:
            print >> log.v5, "running", device.data.shape[1], \
                             "sequences (%i nts)" % (device.data.shape[0] * device.data.shape[1]),
          else:
            print >> log.v5, "running", device.data.shape[0], "frames",
          if device.num_batches == 1:
            print >> log.v5, "of batch %i" % batch,
          else:
            print >> log.v5, "of batches %i-%i" % (batch, batch + device.num_batches - 1),
          print >> log.v5, "/", len(self.parent.batches), "on device", device.name
          #if SprintCommunicator.instance is not None:
          #  SprintCommunicator.instance.segments = device.tags #TODO
          self.num_frames += device.data.shape[0] * device.data.shape[1]
          self.parent.prepare_device_for_batch(device)
          device.run(self.parent.task)
          batch += device.num_batches

      def device_collect_results(self, alloc_devices):
        device_results = []
        for device in alloc_devices:
          try:
            result = device.result()
          except RuntimeError:
            result = None
          if result is None:
            return None
          assert isinstance(result, list)
          assert len(result) >= 1  # The first entry is expected to be the score as a scalar.
          device_results.append(result)
        return device_results

      def device_mem_usage_str(self, devices):
        """
        :type devices: list[Device.Device]
        :rtype: str | None
        """
        if not devices:
          return None
        mem_info = [device.get_memory_info() for device in devices]
        if len(mem_info) == 1 and mem_info[0] is None:
          return None
        mem_usage = [info.used if info else None for info in mem_info]
        s = ["%s MB" % (mem / (1024*1024)) if mem is not None else "unknown" for mem in mem_usage]
        return "/".join(s)

      def print_process(self):
        if not self.parent.interactive and not log.v[5]:
          return
        start_elapsed = time.time() - self.parent.start_time
        run_elapsed = time.time() - self.batch_start_time
        self.parent.run_times.append(run_elapsed)
        if len(self.parent.run_times) * run_elapsed > 60: self.parent.run_times = self.parent.run_times[1:]
        time_domain = len(self.parent.run_times) * sum([d.num_batches for d in self.alloc_devices])
        time_factor = 0.0 if time_domain == 0.0 else float(sum(self.parent.run_times)) / time_domain
        complete = float(self.batch_idx + self.num_alloc_batches) / len(self.parent.batches)
        remaining = hms(int(time_factor * (len(self.parent.batches) - self.batch_idx - self.num_alloc_batches)))
        if log.verbose[5]:
          mem_usage = self.device_mem_usage_str(self.alloc_devices)
          info = [
            "batch %i" % self.batch_idx,
            "score %f" % self.score if self.score is not None else None,
            "elapsed %s" % hms(start_elapsed),
            "exp. remaining %s" % remaining,
            "complete %.02f%%" % (complete * 100),
            "memory %s" % mem_usage if mem_usage else None
          ]
          print >> log.v5, ", ".join(filter(None, info))
        if self.parent.interactive:
          progress_bar(complete, remaining)

    def device_can_run_async(self):
      if len(self.devices) != 1:
        return False
      if self.devices[0].blocking:
        # If we are in the same proc (= blocking), nothing can be async.
        return False
      if self.devices[0].updater is None:
        # If nothing needs to be updated, we can run async.
        return True
      # We can run async iff we do the updates online.
      return self.devices[0].updater.updateOnDevice

    def run_inner(self):
      self.start_time = time.time()
      for device in self.devices:
        device.prepare(**self.get_device_prepare_args())
      self.initialize()
      terminal_width, _ = terminal_size()
      self.interactive = (log.v[3] and terminal_width >= 0)
      print >> log.v5, "starting task", self.task
      self.run_times = []

      batch_idx = self.start_batch
      canRunAsync = self.device_can_run_async()
      remainingDeviceRun = None; " :type: DeviceBatchRun "

      while True:
        # Note about the async logic:
        # We start device.run() twice before we do the first device.result() call.
        # That works because the device proc will push the results on the queue
        # and device.result() reads it from there without sending another command.

        if batch_idx < len(self.batches):
          deviceRun = self.DeviceBatchRun(self, batch_idx)
          deviceRun.start()
          batch_idx += deviceRun.num_alloc_batches
        else:
          deviceRun = None

        if remainingDeviceRun:  # Set when canRunAsync.
          if not remainingDeviceRun.finish():
            return

        if not deviceRun:  # Finished loop.
          break

        if canRunAsync:
          remainingDeviceRun = deviceRun
        else:
          if not deviceRun.finish():
            # We leave self.finalized == False. That way, the engine can see that the device crashed.
            return

        if self.stopped:
          # This happens when we exit Python.
          # Without this check, this thread would keep running until all exit handlers of Python are done.
          print >> log.v5, "%s stopped" % self
          return

      self.finalize()
      self.elapsed = (time.time() - self.start_time)


class TrainTaskThread(TaskThread):
  def __init__(self, network, devices, data, batches, learning_rate, updater, start_batch, pad_batches):
    """
    :type network: Network.LayerNetwork
    :type devices: list[Device.Device]
    :type data: Dataset.Dataset
    :type batches: list[Batch]
    :type learning_rate: float
    :type updater: Updater
    :type train_param_args: dict | None
    :type start_batch: int
    :type pad_batches: bool
    """
    self.updater = updater
    self.learning_rate = learning_rate
    # The task is passed to Device.run().
    if self.updater.updateOnDevice:
      task = "train_and_update"
    else:
      task = "train_distributed"
    super(TrainTaskThread, self).__init__(task, network, devices, data, batches, start_batch, pad_batches)

  def initialize(self):
    self.score = 0
    if self.updater.updateOnDevice:
      assert len(self.devices) == 1
      self.devices[0].set_learning_rate(self.learning_rate)
    else:
      self.updater.initVars(self.network, None)
      self.updater.setLearningRate(self.learning_rate)
      self.updater_func = self.updater.getUpdateFunction()

  def prepare_device_for_batch(self, device):
    """ :type device: Device.Device """
    device.maybe_update_network(self.network)

  def get_device_prepare_args(self):
    kwargs = super(TrainTaskThread, self).get_device_prepare_args()
    kwargs["updater"] = self.updater
    kwargs["train_param_args"] = self.network.train_param_args
    return kwargs

  def evaluate(self, batch, results, num_frames):
    """
    :param int batch: starting batch idx
    :param list[(float,params...)] results: result[i] is result for batch + i, result[i][0] is score
    :type num_frames: int
    """
    assert results
    score = sum([res[0] for res in results])
    self.score += score
    if not self.updater.updateOnDevice:
      gparams = {}
      for p in self.network.train_params:
        gparams[p] = numpy.zeros(p.get_value(borrow=True, return_internal_type=True).shape, dtype=theano.config.floatX)
      for res in results:
        for p, q in zip(self.network.train_params, res[1:]):
          gparams[p] += q
      self.updater.setNetParamDeltas(gparams)
      self.updater_func()
    return score / num_frames

  def finalize(self):
    if self.updater.updateOnDevice:
      # Copy over params at the very end. Also only if we did training.
      assert len(self.devices) == 1
      params = self.devices[0].get_net_train_params()
      our_params = self.network.train_params
      assert len(params) == len(our_params)
      for i in range(len(params)):
        our_params[i].set_value(params[i])
    if self.data.num_timesteps > 0:
      self.score /= float(self.data.num_timesteps)
    super(TrainTaskThread, self).finalize()


class EvalTaskThread(TaskThread):
    def __init__(self, network, devices, data, batches, start_batch = 0, pad_batches=False):
      super(EvalTaskThread, self).__init__('eval', network, devices, data, batches, start_batch, pad_batches)
    def initialize(self):
      self.score = 0
      self.error = 0
      for device in self.devices:
        device.set_net_params(self.network)
    def evaluate(self, batch, results, num_frames):
      assert results
      score = sum([res[0] for res in results])
      self.score += score
      self.error += sum([res[1] for res in results])
      return score / num_frames
    def finalize(self):
      self.score /= float(self.data.num_timesteps)
      if self.network.loss in ('ctc','ce_ctc'):
        self.error /= float(self.data.num_running_chars)
      else:
        self.error /= float(self.data.num_timesteps)


class SprintCacheForwardTaskThread(TaskThread):
    def __init__(self, network, devices, data, batches, cache, merge = {}, start_batch = 0):
      super(SprintCacheForwardTaskThread, self).__init__('extract', network, devices, data, batches, start_batch)
      self.cache = cache
      self.merge = merge
    def initialize(self):
      self.toffset = 0
    def evaluate(self, batch, result, num_frames):
      features = numpy.concatenate(result, axis = 1) #reduce(operator.add, device.result())
      if self.merge.keys():
        merged = numpy.zeros((len(features), len(self.merge.keys())), dtype = theano.config.floatX)
        for i in xrange(len(features)):
          for j, label in enumerate(self.merge.keys()):
            for k in self.merge[label]:
              merged[i, j] += numpy.exp(features[i, k])
          z = max(numpy.sum(merged[i]), 0.000001)
          merged[i] = numpy.log(merged[i] / z)
        features = merged
      print >> log.v5, "extracting", len(features[0]), "features over", len(features), "time steps for sequence", self.data.tags[self.data.seq_index[batch]]
      times = zip(range(0, len(features)), range(1, len(features) + 1)) if not self.data.timestamps else self.data.timestamps[self.toffset : self.toffset + len(features)]
      #times = zip(range(0, len(features)), range(1, len(features) + 1))
      self.toffset += len(features)
      self.cache.addFeatureCache(self.data.tags[self.data.seq_index[batch]], numpy.asarray(features), numpy.asarray(times))


class HDFForwardTaskThread(TaskThread):
    def __init__(self, network, devices, data, batches, cache, merge = {}, start_batch = 0):
      super(HDFForwardTaskThread, self).__init__('extract', network, devices, data, batches, start_batch)
      self.tags = []
      self.merge = merge
      self.cache = cache
      cache.attrs['numSeqs'] = data.num_seqs
      cache.attrs['numTimesteps'] = data.num_timesteps
      cache.attrs['inputPattSize'] = data.num_inputs
      cache.attrs['numDims'] = 1
      cache.attrs['numLabels'] = data.num_outputs
      hdf5_strings(cache, 'labels', data.labels)
      self.targets = cache.create_dataset("targetClasses", (data.num_timesteps,), dtype='i')
      self.seq_lengths = cache.create_dataset("seqLengths", (data.num_seqs,), dtype='i')
      self.seq_dims = cache.create_dataset("seqDims", (data.num_seqs, 1), dtype='i')
      if data.timestamps:
        times = cache.create_dataset("times", data.timestamps.shape, dtype='i')
        times[...] = data.timestamps

    def initialize(self):
      self.toffset = 0
    def finalize(self):
      hdf5_strings(self.cache, 'seqTags', self.tags)

    def evaluate(self, batch, result, num_frames):
      features = numpy.concatenate(result, axis = 1)
      if not "inputs" in self.cache:
        self.inputs = self.cache.create_dataset("inputs", (self.cache.attrs['numTimesteps'], features.shape[2]), dtype='f')
      if self.merge.keys():
        merged = numpy.zeros((len(features), len(self.merge.keys())), dtype = theano.config.floatX)
        for i in xrange(len(features)):
          for j, label in enumerate(self.merge.keys()):
            for k in self.merge[label]:
              merged[i, j] += numpy.exp(features[i, k])
          z = max(numpy.sum(merged[i]), 0.000001)
          merged[i] = numpy.log(merged[i] / z)
        features = merged
      print >> log.v5, "extracting", features.shape[2], "features over", features.shape[1], "time steps for sequence", self.data.tags[self.data.seq_index[batch]]
      self.seq_dims[batch] = [features.shape[1]]
      self.seq_lengths[batch] = features.shape[1]
      self.inputs[self.toffset:self.toffset + features.shape[1]] = numpy.asarray(features)
      self.toffset += features.shape[1]
      self.tags.append(self.data.tags[self.data.seq_index[batch]])


class Engine:
  def __init__(self, devices, network):
    """
    :type devices: list[Device.Device]
    :type network: Network.LayerNetwork
    """
    self.network = network
    self.devices = devices
    self.is_training = False
    self.training_finished = False
    self.lock = threading.RLock()
    self.cond = threading.Condition(lock=self.lock)

  def set_batch_size(self, data, batch_size, batch_step, max_seqs = -1):
    """
    :type data: Dataset.Dataset
    :type batch_size: int
    :type batch_step: int
    :type max_seqs: int
    :rtype: list[Batch]
    """
    batches = []
    batch = Batch([0,0])
    if max_seqs == -1: max_seqs = data.num_seqs
    if batch_step == -1: batch_step = batch_size
    s = 0
    while s < data.num_seqs:
      length = data.seq_lengths[data.seq_index[s]]
      if self.network.recurrent:
        if length > batch_size:
          print >> log.v4, "warning: sequence length (" + str(length) + ") larger than limit (" + str(batch_size) + ")"
        dt, ds = batch.try_sequence(length)
        if ds == 1:
          batch.add_sequence(length)
        else:
          if dt * ds > batch_size or ds > max_seqs:
            batches.append(batch)
            s = s - ds + min(batch_step, ds)
            batch = Batch([s, 0])
            length = data.seq_lengths[data.seq_index[s]]
          batch.add_sequence(length)
      else:
        while length > 0:
          num_frames = min(length, batch_size - batch.shape[0])
          if num_frames == 0 or batch.nseqs > max_seqs:
            batches.append(batch)
            batch = Batch([s, data.seq_lengths[data.seq_index[s]] - length])
            num_frames = min(length, batch_size)
          batch.add_frames(num_frames)
          length -= min(num_frames, batch_step)
        if s != data.num_seqs - 1: batch.nseqs += 1
      s += 1
    batches.append(batch)
    return batches

  @classmethod
  def config_get_final_epoch(cls, config):
    """ :type config: Config.Config """
    return config.int('num_epochs', 5)

  def init_train_config(self, config, train_data, dev_data=None, eval_data=None, start_epoch=1, start_batch=0):
    """
    :type config: Config.Config
    :type train_data: Dataset.Dataset
    :type dev_data: Dataset.Dataset | None
    :type eval_data: Dataset.Dataset | None
    """
    self.train_data = train_data
    self.dev_data = dev_data
    self.eval_data = eval_data
    self.start_epoch = start_epoch
    self.start_batch = start_batch or config.int('start_batch', 0)
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

  def init_train_epoch(self):
    if self.is_pretrain_epoch():
      new_network = self.pretrain.get_network_for_epoch(self.epoch)
      old_network = self.network
      if self.epoch >= 2:
        # Otherwise it's initialized randomly which is fine.
        # This copy will copy the old params over and leave the rest randomly initialized.
        self.pretrain.copy_params_from_old_network(new_network, old_network)
      self.network = new_network
      self.network.set_train_params(**self.pretrain.get_train_param_args_for_epoch(self.epoch))
      # Use constant learning rate.
      self.learning_rate_control.setLearningRateForEpoch(self.epoch, self.learning_rate)
    else:
      self.network.set_train_params()  # Whole network.
      self.learning_rate = self.learning_rate_control.getLearningRateForEpoch(self.epoch)

  def train_epoch(self):
    self.init_train_epoch()
    print >> log.v1, "start", self.get_epoch_str(), "with learning rate", self.learning_rate, "..."

    if self.is_pretrain_epoch():
      print >> log.v2, "pretrain train params:", self.network.train_params

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

  def save_model(self, filename, epoch):
    """
    :param str filename: full filename for model
    :param int epoch: save epoch idx
    """
    model = h5py.File(filename, "w")
    self.network.save(model, epoch)
    model.close()

  def forward_to_sprint(self, device, data, cache_file, combine_labels = ''):
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

  def forward(self, device, data, output_file, combine_labels = ''):
    cache =  h5py.File(output_file, "w")
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
