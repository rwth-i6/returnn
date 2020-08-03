from __future__ import print_function

import collections
from math import ceil
import sys
import threading
import time

import numpy
import theano

from Device import Device
from EngineUtil import assign_dev_data
from Log import log
from TaskSystem import ProcConnectionDied
from Util import hms, progress_bar, terminal_size, hdf5_strings, interrupt_main, NumbersDict


class TaskThread(threading.Thread):
    def __init__(self, task, network, devices, data, batches, eval_batch_size=0, start_batch=0, share_batches = False, reduction_rate=1.0, report_prefix=None, exclude=None, epoch=None):
      """
      :type task: str
      :type network: Network.LayerNetwork
      :type devices: list[Device.Device]
      :type data: Dataset.Dataset
      :type batches: EngineBatch.BatchSetGenerator
      :type start_batch: int
      :param str report_prefix: such as epoch or so. only for reporting
      """
      threading.Thread.__init__(self, name="TaskThread %s" % task)
      assert len(devices) > 0
      if eval_batch_size == 0:
        eval_batch_size = sys.maxsize
      self.share_batches = share_batches
      self.eval_batch_size = eval_batch_size
      self.eval_batch_idx = 0
      self.start_batch = start_batch
      self.reduction_rate = reduction_rate
      self.devices = devices
      self.network = network
      self.batches = batches
      self.exclude = exclude
      self.task = task
      self.data = data
      self.daemon = True
      self.elapsed = 0
      self.finalized = False
      self.score = {}
      self.error = {}
      self.results = {}
      self.num_frames = NumbersDict(0)
      self.batch_idx = None; " :type: int | None "
      self.device_crash_batch = None; " :type: int | None "
      self.report_prefix = report_prefix or self.task
      self.epoch = epoch
      self.lock = threading.Lock()
      self.start()

    def assign_dev_data(self, device, batches):
      return assign_dev_data(device, self.data, batches)

    def maybe_wait_for_batches(self, device, batches):
      """
      :type device: Device
      :type batches: list[Batch]
      """
      pass

    def allocate_devices(self, selected_devices = None):
      """
      Sets the device data, i.e. the next batches, via self.batches.
      This calls Dataset.load_seqs() to get the data.
      This sets:
        device.targets
        device.ctc_targets
        device.tags
        device.index
      :rtype: list[list[EngineBatch.Batch]]
      :returns list of batches per device
      """
      if not selected_devices:
        selected_devices = self.devices
      devices_batches = []; " :type: list[list[EngineBatch.Batch]] "
      if self.share_batches:
        batches = self.batches.peek_next_n(1)
      for device in selected_devices:
        if not self.share_batches:
          batches = self.batches.peek_next_n(device.num_batches)
        self.maybe_wait_for_batches(device=device, batches=batches)
        success, batch_adv_idx = self.assign_dev_data(device, batches)
        batch_idx = self.batches.get_current_batch_idx()
        assert success, "batches %s with seqs at %i failed to load" % \
                        (range(batch_idx, batch_idx + batch_adv_idx), batches[batch_adv_idx - 1].start_seq)
        devices_batches.append(batches)
        if not self.share_batches:
          self.batches.advance(batch_adv_idx)
      if self.share_batches:
        self.batches.advance(batch_adv_idx)
      return devices_batches

    def prepare_device_for_batch(self, device):
      """ :type device: Device.Device """
      pass

    def get_device_prepare_args(self):
      return {"network": self.network, "updater": None}

    def evaluate(self, batchess, results, result_format, num_frames):
      """
      :param list[list[EngineBatch.Batch]] batchess: batches per device
      :param list[list[numpy.ndarray]] results: results per device
      :param list[str]|None result_format: describes what we have in a result list
      :type num_frames: NumbersDict
      :returns some score or None
      :rtype: dict[str] | None
      """
      assert results
      assert result_format  # train should always have the format
      assert num_frames["data"] > 0

      # We can get info such as "cost:..." and more info such as gradient_norm.
      # See Device.initialize().
      # We might also get gparams or ctc_priors or so. We will filter them out below when not needed.
      results = [Device.make_result_dict(res, result_format) for res in results]
      if 'weights' in results[0]:
        for batch, result in zip(batchess, results):
          self.batches.dataset.update_weights(batch[0].seqs, result['weights'])
          del result['weights']

      batch_norm_fact = 1 if not self.share_batches else 1.0 / len(self.devices)
      summed_results = {}
      for key in results[0].keys():
        summed_results[key] = sum([res[key] for res in results]) * batch_norm_fact

      # Accumulate for epoch stats.
      for key, value in summed_results.items():
        if key.startswith("gparam:"): continue
        if key not in self.results:
          self.results[key] = value # / float(num_frames[target])
        else:
          self.results[key] += value # / float(num_frames[target])

      # Prepare eval info stats for this (multiple-)batch run.
      eval_info = {}
      for key, value in summed_results.items():
        if key.startswith("gparam:"): continue
        if key == "ctc_priors": continue
        target = self._get_target_for_key(key)
        eval_info[key] = value / float(num_frames[target])

      return eval_info

    def initialize(self):
      """
      Called at the beginning of an epoch.
      """
      pass

    def reduce(self, num_frames):
      pass

    def _get_target_for_key(self, key):
      try:
        target = self.network.output[key.split(':')[-1]].attrs['target']
      except Exception:
        try:
          target = self.network.hidden[key.split(':')[-1]].attrs['target']
        except Exception:
          target = 'classes'
      available_data_keys = self.data.get_data_keys()
      if target not in available_data_keys:
        target = available_data_keys[0]
      return target

    def epoch_norm_factor_for_result(self, key):
      target = self._get_target_for_key(key)
      # Check for key specific behavior
      if key.split(':')[-1] in self.network.output:
        attrs = self.network.output[key.split(':')[-1]].attrs
        if attrs.get('normalize_length', False):
          return 1.0 / float(self.data.num_seqs)
      # Default: Normalize by number of frames.
      return 1.0 / float(self.num_frames[target])

    def finalize(self):
      """
      Called at the end of an epoch.
      """
      assert self.num_frames["data"] > 0
      # Note: self.num_frames could be greater than self.data.get_num_timesteps() in case of chunking.
      for key, value in self.results.items():
        if key != "ctc_priors":
          self.results[key] *= self.epoch_norm_factor_for_result(key)
      self.score = dict([(key,value) for (key, value) in self.results.items() if key.startswith("cost:")])
      self.error = dict([(key,value) for (key, value) in self.results.items() if key.startswith("error:")])
      self.finalized = True

    class DeviceBatchRun(threading.Thread):
      def __init__(self, parent, devices):
        """
        :type parent: TaskThread
        """
        threading.Thread.__init__(self, name="DeviceThread %s" % " ".join([dev.name for dev in devices]))
        self.alloc_devices = devices
        self.parent = parent
        self.devices_batches_idx = None
        self.run_start_batch_idx = None
        self.eval_info = None; " :type: dict[str] | None "
        self.allocated = False
        self.processing = False
        self.finished = True
        self.crashed = False
        self.num_frames = NumbersDict(0)
        self.run_frames = NumbersDict(0)
        self.daemon = True
        self.active = True
        self.result = { 'batchess': [], 'results': [], 'result_format': None, 'num_frames': 0 }
        if self.alloc_devices:
          self.start()

      def allocate(self):
        self.devices_batches_idx = self.parent.batches.get_current_batch_idx()
        self.allocated_devices_batches = self.parent.allocate_devices(self.alloc_devices)
        self.run_frames = NumbersDict(0)
        for batches, device in zip(self.allocated_devices_batches, self.alloc_devices):
          assert batches
          assert batches[0].seqs
          #assert batches[0].seqs[0].frame_length[1] > 0
          device.num_updates += 1 if not device.update_specs['block_size'] else int(ceil(sum([len(batch.seqs) for batch in batches]) / float(device.update_specs['block_size'])))
          self.run_frames += sum([batch.get_total_num_frames() for batch in batches])
        if self.parent.share_batches:
          self.run_frames /= len(self.alloc_devices)
        assert self.run_frames.max_value() > 0
        self.allocated = True

      def finish(self):
        """
        :returns whether everything is fine.
        """
        device_results, outputs_format = self.device_collect_results()
        if device_results is None:
          if not getattr(sys, "exited", False):
            print("device crashed on batch", self.run_start_batch_idx, file=log.v3)
          self.parent.device_crash_batch = self.run_start_batch_idx
          self.crashed = True
          return False
        assert len(device_results) == len(self.alloc_devices) == len(self.running_devices_batches)

        if outputs_format and any([k.startswith("gparam:") for k in outputs_format]):
          # WARNING: this code is untested and likely broken!
          for i in range(len(self.alloc_devices)):
            res = Device.make_result_dict(device_results[i], outputs_format)
            self.alloc_devices[i].sync_net_train_params()
            devnet = self.alloc_devices[i].get_net_train_params(self.parent.network)
            vars = self.parent.network.get_all_params_vars()
            for p, q in zip(vars, devnet):
              p.set_value(q)
            gparams = {}
            for p in vars:
              gparams[p] = numpy.zeros(p.get_value(borrow=True, return_internal_type=True).shape, dtype=theano.config.floatX)
            for p in vars:
              q = res["gparam:%s" % p.name]
              if q.shape == p.get_value().shape:
                gparams[p] = q
              elif q.shape:
                print("warning: shape for gradient does not match:", p.get_value().shape, q.shape, file=log.v2)
            self.parent.updater.setNetParamDeltas(gparams)
            self.parent.updater.update()
            self.alloc_devices[i].set_net_params(self.parent.network)

        self.result = { 'batchess': self.running_devices_batches,
                        'results': device_results,
                        'result_format': outputs_format,
                        'num_frames': self.num_frames }
        self.eval_info = self.parent.evaluate(**self.result)
        self.parent.lock.acquire()
        self.print_process()
        self.parent.lock.release()
        return True

      def run(self):
        try:
          while self.active and not getattr(sys, "exited", False):
            if self.allocated and not self.finished:
              self.device_run()
              self.num_frames = self.run_frames
              self.processing = True
              self.allocated = False
              self.finish()
              self.finished = True
              self.processing = False
            else:
              time.sleep(0.01)
        except BaseException:
          self.crashed = True
          sys.excepthook(*sys.exc_info())
        finally:
          self.finished = True

      def stop(self):
        self.active = False

      def device_run(self):
        batch_idx = self.run_start_batch_idx = self.devices_batches_idx
        assert len(self.alloc_devices) == len(self.allocated_devices_batches)
        self.running_devices_batches = self.allocated_devices_batches
        for device, batches in zip(self.alloc_devices, self.running_devices_batches):
          if self.parent.network.recurrent:
            print("running", device.targets["data"].shape[1], \
                             "sequence slices (%i nts)" % (device.targets["data"].shape[0] * device.targets["data"].shape[1]), end=' ', file=log.v5)
          else:
            print("running", device.targets["data"].shape[0] * device.targets["data"].shape[1], "frames", end=' ', file=log.v5)
          if device.num_batches == 1:
            print("of batch %i" % batch_idx, end=' ', file=log.v5)
          else:
            print("of batches %i-%i" % (batch_idx, batch_idx + device.num_batches - 1), end=' ', file=log.v5)
          print("on device", device.name, file=log.v5)
          device.run(self.parent.task)
      #if not self.share batch_idx += device.num_batches

      def device_collect_results(self):
        device_results = []
        outputs_format = None
        for i, device in enumerate(self.alloc_devices):
          try:
            result, outputs_format_new = device.result()
          except RuntimeError:
            return None, None
          if result is None:
            return None, None
          assert isinstance(result, list)
          assert len(result) > 0  # we always expect to get some result
          if i >= 1:
            assert outputs_format == outputs_format_new, "We expect to always get the same output format."
          outputs_format = outputs_format_new
          device_results.append(result)
        return device_results, outputs_format

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
        complete = self.parent.batches.completed_frac()
        assert complete > 0
        total_time_estimated = start_elapsed / complete
        remaining_estimated = total_time_estimated - start_elapsed
        if log.verbose[5]:
          mem_usage = self.device_mem_usage_str(self.alloc_devices)
          info = [
            self.parent.report_prefix,
            "batch %i" % self.run_start_batch_idx]
          if self.eval_info:  # Such as score.
            info += ["%s %s" % item for item in sorted(self.eval_info.items())]
          info += [
            "elapsed %s" % hms(start_elapsed),
            "exp. remaining %s" % hms(remaining_estimated),
            "complete %.02f%%" % (complete * 100)]
          if mem_usage:
            info += ["memory %s" % mem_usage]
          print(", ".join(filter(None, info)), file=log.v5)
        if self.parent.interactive:
          progress_bar(complete, hms(remaining_estimated))

    def run(self):
      # Wrap run_inner() for better exception printing.
      # Thread.__bootstrap_inner() ignores sys.excepthook.
      try:
        self.run_inner()
      except ProcConnectionDied:
        if not getattr(sys, "exited", False):
          # Normally we should have caught that in run_inner(), so somewhat unexpected.
          print("%s. Some device proc crashed unexpectedly." % self, file=log.v4)
        # Just pass on. We have self.finalized == False which indicates the problem.
      except Exception:
        # Catch all standard exceptions.
        # These are not device errors. We should have caught them in the code
        # and we would leave self.finalized == False.
        # Don't catch KeyboardInterrupt here because that will get send by the main thread
        # when it is exiting. It's never by the user because SIGINT will always
        # trigger KeyboardInterrupt in the main thread only.
        try:
          print("%s failed" % self.name, file=log.v1)
          if log.v[4]:
            sys.excepthook(*sys.exc_info())
            print("")
        finally:
          # Exceptions are fatal. If we can recover, we should handle it in run_inner().
          interrupt_main()

    def run_inner(self):
      self.start_time = time.time()
      for device in self.devices:
        device.prepare(epoch=self.epoch, **self.get_device_prepare_args())
      self.initialize()
      terminal_width, _ = terminal_size()
      self.interactive = (log.v[3] and terminal_width >= 0)
      print("starting task", self.task, file=log.v5)

      for device in self.devices:
        device.eval_batch_idx = -1
        device.start_epoch_stats()
        device.num_frames = 0
        device.num_updates = 0
        device.tot = 0

      num_device_runs = 1 if self.share_batches else len(self.devices)
      deviceRuns = [ self.DeviceBatchRun(self, [self.devices[i]] if not self.share_batches else self.devices) for i in range(num_device_runs) ]

      results = { 'batchess': [], 'results': [], 'num_frames' : NumbersDict(0) }
      run_frames = NumbersDict(0)
      cost_result_format = -1

      crashed = False
      assert num_device_runs > 0

      while True:
        if getattr(sys, "exited", False):
          # This happens when we exit Python.
          # Without this check, this thread would keep running until all exit handlers of Python are done.
          print("%s stopped" % self, file=log.v5)
          crashed = True
          break

        for i in range(num_device_runs):
          if deviceRuns[i].crashed or not deviceRuns[i].is_alive():
            crashed = True
            break
          if deviceRuns[i].finished:
            results['batchess'] += deviceRuns[i].result['batchess'][:]
            results['results'] += deviceRuns[i].result['results'][:]
            results['result_format'] = deviceRuns[i].result['result_format']
            deviceRuns[i].finished = False
        if crashed:
          break

        if cost_result_format < 0 and deviceRuns[i].result['result_format']:
          for idx,fmt in enumerate(deviceRuns[i].result['result_format']):
            if fmt and fmt.startswith('cost:'):
              cost_result_format = idx
        total_cost = 0
        if results['results'] and cost_result_format >= 0:
          total_cost = numpy.asarray(results['results'])[:,cost_result_format].sum()
        if total_cost >= self.eval_batch_size or not self.batches.has_more():
          if all(not (dev.finished or dev.allocated or dev.processing) for dev in deviceRuns):
            results['num_frames'] = run_frames
            self.num_frames += run_frames
            if self.share_batches: run_frames *= len(self.devices)
            self.reduce(run_frames)
            self.eval_batch_idx += 1
            run_frames = NumbersDict(0)
            results['batchess'] = []
            results['results'] = []
            for device in self.devices:
              device.num_frames = 0
              device.num_updates = 0
            if not self.batches.has_more():
              break
          else:
            time.sleep(0.01)


        match = True
        while self.batches.has_more() and total_cost < self.eval_batch_size and match:
          self.batch_idx = self.batches.get_current_batch_idx()
          if self.batch_idx < self.start_batch:
            self.batches.advance(1)
            break
          match = False
          for i in range(num_device_runs):
            if not deviceRuns[i].allocated:
              deviceRuns[i].allocate()
              run_frames += deviceRuns[i].run_frames
              match = True
              break
        if not match:
          time.sleep(0.01)

      for run in deviceRuns:
        run.stop()
      if crashed: return
      for device in self.devices:
        device.finish_epoch_stats()
      self.finalize()
      if self.interactive: progress_bar()
      self.elapsed = (time.time() - self.start_time)


class ModelBrokenError(Exception):
  """
  We got a nan/inf at the result somewhere. This means that something is broken.
  """
  def __init__(self, msg, batches):
    """
    :type msg: str
    :type batches: list[EngineBatch.Batch]
    """
    assert len(batches) > 0
    msg = "%s Starting at seq %i." % (msg, batches[0].start_seq)
    super(ModelBrokenError, self).__init__(msg)
    self.batches = batches


class TrainTaskThread(TaskThread):
  def __init__(self, network, devices, data, batches, learning_rate, updater, seq_train_parallel=None, **kwargs):
    """
    :type network: Network.LayerNetwork
    :type devices: list[Device.Device]
    :type data: Dataset.Dataset
    :type batches: EngineBatch.BatchSetGenerator
    :type learning_rate: float
    :type updater: Updater.Updater
    :type seq_train_parallel: Engine.SeqTrainParallelControl | None
    """
    self.updater = updater
    self.learning_rate = learning_rate
    self.seq_train_parallel = seq_train_parallel
    self.do_ctc_priors = network.ctc_priors is not None
    self.ctc_priors = None
    super(TrainTaskThread, self).__init__("train", network, devices, data=data, batches=batches, **kwargs)

  def initialize(self):
    super(TrainTaskThread, self).initialize()
    self.score = 0
    for device in self.devices:
      device.set_learning_rate(self.learning_rate)
    if not self.updater.isInitialized:
      self.updater.initVars(self.network, None)
      self.updater.setLearningRate(self.learning_rate)
    if self.seq_train_parallel:
      self.seq_train_parallel.train_start_epoch()

  def prepare_device_for_batch(self, device):
    """ :type device: Device.Device """
    return

  def get_device_prepare_args(self):
    kwargs = super(TrainTaskThread, self).get_device_prepare_args()
    kwargs["updater"] = self.updater
    kwargs["train_param_args"] = self.network.train_param_args
    return kwargs

  def maybe_wait_for_batches(self, device, batches):
    """
    :type device: Device
    :type batches: list[Batch]
    """
    if self.seq_train_parallel:
      self.seq_train_parallel.train_wait_for_seqs(device=device, batches=batches)

  def save_ctc_priors(self, filename, epoch_str):
    assert self.ctc_priors is not None
    return # this should be done using compute_priors
    with open(filename, 'a') as f:
      print(epoch_str, file=f)
      numpy.savetxt(f, self.ctc_priors, newline=" ")
      print(file=f)

  class CopyManager():
    class CopyThread(threading.Thread):
      def __init__(self, device, network, copy_to_device):
        threading.Thread.__init__(self, name="CopyThread %s" % device.name)
        self.copy_to_device = copy_to_device
        self.device = device
        self.network = network
        self.active = True
        self.start()

      def run(self):
        if self.copy_to_device:
          self.device.set_net_params(self.network)
          self.result = True
        else:
          self.result = self.device.get_net_train_params(self.network)
        self.active = False

    def __init__(self, devices):
      self.devices = devices
      self.network = None

    def _copy(self, copy_to_device):
      threads = []
      for device in self.devices:
        threads.append(self.CopyThread(device, self.network, copy_to_device))
      result = []
      for thread in threads:
        if thread.active:
          thread.join()
        result.append(thread.result)
      return result

    def copy_to_device(self, network):
      self.network = network
      return self._copy(True)

    def copy_from_device(self):
      return self._copy(False)

  def reduce(self, num_frames):
    for device in self.devices:
      device.sync_net_train_params()
    basenet = self.network.get_all_params_vars()
    consnet = [numpy.zeros(p.get_value().shape, dtype='float32') for p in basenet]
    hypnets = []
    nparams = len(basenet)
    encoded = []
    for device in self.devices:
      hypnets.append([ p for p in device.get_net_train_params(self.network) ])
      assert len(hypnets[-1]) == len(basenet)
    if len(hypnets) == 0:
      consnet = basenet
    elif len(hypnets) == 1:
      consnet = hypnets[0]
    else:
      # consensus via average
      for i in range(nparams):
        num_updates = { dev.name : dev.get_total_cost() for net,dev in zip(hypnets,self.devices) if numpy.sum(abs(net[i] - basenet[i].get_value())) > numpy.float32(0) }
        tot_updates = sum(num_updates.values()) / self.reduction_rate
        if tot_updates:
          consnet[i] = basenet[i].get_value() + numpy.sum([ (net[i] - basenet[i].get_value()) * float(num_updates[dev.name]) / tot_updates for net,dev in zip(hypnets,self.devices) if dev.name in num_updates ], axis = 0)
        else:
          print("warning: no update available for parameter", basenet[i], file=log.v3)
          consnet[i] = basenet[i].get_value()
    self.network.update_step = max([ dev.get_num_updates() for dev in self.devices ])
    for p, q in zip(self.network.get_all_params_vars(), consnet):
      p_shape = p.get_value(borrow=True, return_internal_type=True).shape
      assert p_shape == q.shape
      p.set_value(q)
      encoded.append(q)
    if len(hypnets) > 1:
      for device in self.devices:
        device.set_net_encoded_params(encoded)
    return
    try:
      basenet = self.network.get_all_params_vars()
      consnet = [numpy.zeros(p.get_value().shape, dtype='float32') for p in basenet]
      hypnets = []
      nparams = len(basenet)
      encoded = []
      #pipe = self.CopyManager(self.devices)
      #hypnets = pipe.copy_from_device()
      for device in self.devices:
        hypnets.append([ p for p in device.get_net_train_params(self.network) ])
        assert len(hypnets[-1]) == len(basenet)
      if len(hypnets) == 0:
        consnet = basenet
      elif len(hypnets) == 1:
        consnet = hypnets[0]
      else:
        # consensus via average
        for i in range(nparams):
          num_updates = { dev.name : dev.get_total_cost() for net,dev in zip(hypnets,self.devices) if numpy.sum(abs(net[i] - basenet[i].get_value())) > numpy.float32(0) }
          tot_updates = sum(num_updates.values()) / self.reduction_rate
          #num_updates = numpy.sum([ dev.num_updates for net,dev in zip(hypnets,self.devices) ])
          #ndevs = len([ dev for dev in self.devices if abs(numpy.sum(net[i] - basenet[i].get_value())) > 0.0001 ])
          #consnet[i] = basenet[i].get_value() + numpy.sum([(net[i] - basenet[i].get_value()) * (float(device.num_frames) / num_frames) for net,dev in zip(hypnets,self.devices) if basenet[i].layer.name in dev.update_specs['layers']], axis = 0)
          if tot_updates:
            consnet[i] = basenet[i].get_value() + numpy.sum([ (net[i] - basenet[i].get_value()) * grads[dev.name] * float(num_updates[dev.name]) / tot_updates for net,dev in zip(hypnets,self.devices) if dev.name in num_updates ], axis = 0)
          else:
            print("warning: no update available for parameter", basenet[i], file=log.v3)
            consnet[i] = basenet[i].get_value()
          #consnet[i] = basenet[i].get_value() + ndevs * numpy.sum([ (net[i] - basenet[i].get_value()) * (float(device.num_frames) / nframes) for net,dev in zip(hypnets,self.devices) ], axis = 0)
      self.network.update_step = max([ dev.get_num_updates() for dev in self.devices ])
      for p, q in zip(self.network.get_all_params_vars(), consnet):
        p_shape = p.get_value(borrow=True, return_internal_type=True).shape
        assert p_shape == q.shape
        p.set_value(q)
        encoded.append(q)
      if len(hypnets) > 1:
        for device in self.devices:
          device.set_net_encoded_params(encoded)
    except Exception as e:
      print("network synchronization failed: ", e.message, file=log.v3)
      if log.v4:
        sys.excepthook(*sys.exc_info())

    #pipe.copy_to_device(self.network)

  def finalize(self):
    super(TrainTaskThread, self).finalize()
    if self.do_ctc_priors:
      self.ctc_priors = self.results["ctc_priors"] / float(self.num_frames["data"])
    if self.seq_train_parallel:
      self.seq_train_parallel.train_finish_epoch()


class EvalTaskThread(TaskThread):
    def __init__(self, network, devices, data, batches, **kwargs):
      super(EvalTaskThread, self).__init__('eval', network, devices, data=data, batches=batches, **kwargs)

    def initialize(self):
      super(EvalTaskThread, self).initialize()
      for device in self.devices:
        device.set_net_params(self.network)

class ForwardTaskThread(TaskThread):
    def __init__(self, network, devices, data, batches, eval_batch_size=0):
      super(ForwardTaskThread, self).__init__('extract', network, devices, data, batches, eval_batch_size=eval_batch_size)
      self.result = {}

    def evaluate(self, batchess, results, result_format, num_frames):
      fragments = collections.defaultdict(list)
      for device_idx, batches in enumerate(batchess):
        for batch_idx, batch in enumerate(batches):
          for seq_idx, seq in enumerate(batch.seqs):
            fragments[seq.seq_idx].append((seq.seq_start_frame['data'], seq, results[device_idx][batch_idx]))
      for seq, parts in fragments.items():
        prev_end_frame = -1
        seq_idx = None
        for part in sorted(parts):
          assert part[0] == prev_end_frame + 1
          prev_end_frame = part[1].seq_end_frame
        self.result[seq] = numpy.concatenate([r[s.batch_frame_offset['data']:(s.seq_end_frame['data'] - s.seq_start_frame['data']),s.batch_slice,:]
                                             for _, s, r in parts], axis=0)


class HDFForwardTaskThread(TaskThread):
    def __init__(self, network, devices, data, batches, cache, compression="none"):
      super(HDFForwardTaskThread, self).__init__('extract', network, devices, data, batches, eval_batch_size=1)
      self.tags = []
      self.cache = cache
      self.network = network
      self.num_seqs = 0
      if network.get_layer('output'):
        target = network.get_layer('output').attrs['target']
      else:
        target = 'classes'
      cache.attrs['numTimesteps'] = 0
      cache.attrs['inputPattSize'] = data.num_inputs
      cache.attrs['numDims'] = 1
      cache.attrs['numLabels'] = data.num_outputs[target]
      self.compression=compression
      if target in data.labels:
        hdf5_strings(cache, 'labels', data.labels[target])
      try:
        cache.attrs['numSeqs'] = data.num_seqs
      except Exception:
        cache.attrs['numSeqs'] = 1
        self.seq_lengths = cache.create_dataset("seqLengths", (cache.attrs['numSeqs'],), dtype='i', maxshape=(None,), compression=compression)
      else:
        self.seq_lengths = cache.create_dataset("seqLengths", (cache.attrs['numSeqs'],), dtype='i', compression=compression)
        self.seq_dims = cache.create_dataset("seqDims", (cache.attrs['numSeqs'], 1), dtype='i', compression=compression)
      try:
        self.targets = { k: cache.create_dataset("targets/data/" + k, (data.get_num_timesteps(),), dtype='i', compression=compression) for k in data.get_target_list() }
      except Exception:
        self.targets = None
      self.times = []

    def initialize(self):
      self.toffset = 0

    def finalize(self):
      hdf5_strings(self.cache, 'seqTags', self.tags)
      if self.times:
        times = self.cache.create_dataset("times", (len(self.times), 2), dtype='f')
        times[...] = self.times
      self.cache.attrs['numSeqs'] = self.num_seqs

    def evaluate(self, batchess, results, result_format, num_frames):
      """
      :param list[list[Batch]] batchess: batches per device
      :param list[list[numpy.ndarray]] results: results per device
      :param list[str]|None result_format: describes what we have in a result list
      :type num_frames: NumbersDict
      :returns some score or None
      :rtype: dict[str] | None
      """
      # Currently we support just a single dev with a single batch.
      assert len(batchess) == 1
      assert len(batchess[0]) == 1
      assert len(results) == 1
      assert len(results[0]) == 1
      features = results[0][0]
      batch = batchess[0][0]
      from EngineBatch import Batch
      assert isinstance(batch, Batch)
      if "inputs" not in self.cache:
        self.inputs = self.cache.create_dataset("inputs", (self.cache.attrs['numSeqs'], features.shape[-1]), dtype='f', maxshape=(None, None), compression=self.compression)
      if features.shape[-1] > self.inputs.shape[1]:
        self.inputs.resize(features.shape[-1],axis=1)
      tt = 0
      feats = []
      self.num_seqs += batch.get_num_seqs()
      for seq_idx in range(batch.start_seq, batch.end_seq):
        if self.network.recurrent:
          seqfeats = features[:, seq_idx - batch.start_seq]
          if batch.end_seq - batch.start_seq > 1:
            seqfeats = seqfeats[~numpy.all(seqfeats == 0,axis=1)]
          if seqfeats.shape[0] == 0:
            seqfeats = features[:, seq_idx - batch.start_seq]
        else:
          seq = batch.seqs[seq_idx - batch.start_seq]
          seqfeats = features[
                       seq.batch_frame_offset["data"]:seq.batch_frame_offset["data"] + seq.frame_length["data"],
                       seq.batch_slice]
        print("extracting", seqfeats.shape[-1], "features over", seqfeats.shape[0], "time steps for sequence", self.data.get_tag(seq_idx), file=log.v5)
        self.cache.attrs['numTimesteps'] += seqfeats.shape[0]
        tt += seqfeats.shape[0]
        #self.seq_dims[seq_idx] = [seqfeats.shape[1]]
        if self.seq_lengths.shape[0] <= seq_idx:
          self.seq_lengths.resize(seq_idx+1,axis=0)
        self.seq_lengths[seq_idx] = seqfeats.shape[0]
        #self.inputs[self.toffset:self.toffset + seqfeats.shape[0]] = numpy.asarray(seqfeats)
        feats.append(seqfeats)
        self.tags.append(self.data.get_tag(seq_idx))
        try:
          times = self.data.get_times(seq_idx)
          self.times.extend(times)
        except Exception:
          pass
        if self.inputs.shape[1] < seqfeats.shape[1]:
          self.inputs.resize(seqfeats.shape[1], axis=1)
      if self.inputs.shape[0] < self.toffset + tt:
        self.inputs.resize(self.toffset + tt, axis = 0)
      self.inputs[self.toffset:self.toffset + tt,:feats[0].shape[1]] = numpy.concatenate(feats,axis=0)
      self.cache.attrs['inputPattSize'] = self.inputs.shape[1]
      self.toffset += tt


class ClassificationTaskThread(TaskThread):
    def __init__(self, network, devices, data, batches):
      super(ClassificationTaskThread, self).__init__('extract', network, devices, data, batches, eval_batch_size=1)
      self.result = {}

    def evaluate(self, batchess, results, result_format, num_frames):
      assert len(batchess) == 1
      assert len(batchess[0]) == 1
      assert batchess[0][0].get_num_seqs() == 1
      self.result[self.data.get_tag(batchess[0][0].start_seq)] = numpy.concatenate(results, axis=1)


class PriorEstimationTaskThread(TaskThread):
    def __init__(self, network, devices, data, batches, priori_file, target, extract_type):
      from Network import LayerNetwork
      assert isinstance(network, LayerNetwork)
      super(PriorEstimationTaskThread, self).__init__('extract', network=network, devices=devices, data=data, batches=batches)
      self.priori_file = priori_file
      self.target = target  # e.g. "classes"
      self.extract_type = extract_type
      assert extract_type in ["log-posteriors", "log-posteriors-sum", "posteriors", "posteriors-sum"]
      self.num_outputs = network.n_out[target][0]
      self.sum_posteriors = numpy.zeros(int(self.num_outputs))
      print("Prior estimation via posteriors of %r. output dimension = %i" % (target, self.num_outputs), file=log.v1)
      if not extract_type.endswith("-sum"):
        print("HINT: You can set extract=posteriors-sum in your config to speed up the estimation.", file=log.v1)
      if extract_type.startswith("log-"):
        print("NOTE: Posteriors are averaged in log-space. std-space might be better. Set extract=posteriors-sum.", file=log.v1)
      if data.chunk_size != 0:
        print("WARNING: Dataset uses chunking. You might want to disable that.", file=log.v1)

    def evaluate(self, batchess, results, result_format, num_frames):
      if self.extract_type.endswith("-sum"):
        for ress in results:
          for res in ress:
            assert isinstance(res, numpy.ndarray)
            assert res.ndim == 1
            # Index-masked frames are zero, so this sum works.
            self.sum_posteriors += res
      else:
        for ress in results:
          for res in ress:
            assert isinstance(res, numpy.ndarray)
            assert res.ndim == 3
            # Index-masked frames are zero, so this sum works.
            self.sum_posteriors += numpy.sum(res, axis=(0, 1))

    def finalize(self):
      print("Dumping priors in +log-space to file", self.priori_file, file=log.v1)
      print("Frames in total:", self.num_frames, file=log.v1)
      average_posterior = self.sum_posteriors / self.num_frames[self.target]
      if self.extract_type.startswith("log-"):
        print("Posterior average was calculated in log-space", file=log.v1)
        # We need to renormalize.
        average_posterior -= numpy.log(numpy.sum(numpy.exp(average_posterior)))
      else:
        average_posterior = numpy.log(average_posterior)
        print("Posterior average was calculated in std-space", file=log.v1)
      numpy.savetxt(self.priori_file, average_posterior, delimiter=' ')
      avg_sum = numpy.sum(numpy.exp(average_posterior))
      print("Prior sum in std-space (should be close to 1.0):", avg_sum, file=log.v1)
