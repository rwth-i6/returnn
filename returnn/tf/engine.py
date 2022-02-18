"""
TensorFlow engine
=================

The basic engine for the TensorFlow backend is implemented here,
i.e. the high-level logic to train, i.e. looping over epochs,
holding the network instance, creating the TensorFlow session,
managing the data pipeline, etc.

See :ref:`tech_overview` for an overview how it fits all together.
"""

from __future__ import print_function

import os
import sys
import time
import typing
try:
  # noinspection PyCompatibility
  from Queue import Queue
except ImportError:
  # noinspection PyCompatibility
  from queue import Queue

import numpy
import tensorflow as tf
from tensorflow.python.client import timeline

from returnn.engine.base import EngineBase
from returnn.datasets.basic import Dataset, Batch, BatchSetGenerator, init_dataset
from returnn.datasets.util.vocabulary import Vocabulary
from returnn.learning_rate_control import load_learning_rate_control_from_config, LearningRateControl
from returnn.log import log
from returnn.pretrain import pretrain_from_config
import returnn.tf.compat as tf_compat
from returnn.tf.network import TFNetwork, ExternData, help_on_tf_exception
from returnn.tf.util.data import Data
from returnn.tf.layers.base import LayerBase
from returnn.tf.updater import Updater
from returnn.tf.data_pipeline import FeedDictDataProvider, DatasetDataProvider
import returnn.tf.horovod as tf_horovod
from returnn.util.basic import hms, NumbersDict, BackendEngine, BehaviorVersion
from pprint import pprint


class CancelTrainingException(Exception):
  """
  Training was cancelled.
  """


class Runner(object):
  """
  This encapsulates the logic around TF ``session.run``, i.e. iterating over the dataset.
  """

  # noinspection PyShadowingBuiltins
  def __init__(self, engine,
               dataset_name=None, dataset=None, batches=None,
               train=False, eval=True, train_flag=None,
               extra_fetches=None, extra_fetches_callback=None):
    """
    :param Engine engine:
    :param str|None dataset_name: "train", "dev" or so
    :param Dataset.Dataset|None dataset:
    :param BatchSetGenerator|None batches:
    :param bool train: whether to do updates on the model
    :param bool|None train_flag: normally just as train. but e.g. maybe you want to have the train_flag but not train
    :param bool eval: whether to evaluate (i.e. calculate loss/error)
    :param dict[str,tf.Tensor|Data|LayerBase|(()->tf.Tensor)]|None extra_fetches:
      additional fetches per step.
      `extra_fetches_callback` will be called with these. In case of Data/LayerBase, it will return a list,
      where each item corresponds to the batch-seq.
      It might also be useful to add `network.get_extern_data("seq_idx")` and `network.get_extern_data("seq_tag")`.
    :param (**dict[str,numpy.ndarray|str|list[numpy.ndarray|str])->None extra_fetches_callback: called if extra_fetches
    """
    from returnn.tf.data_pipeline import DataProviderBase
    engine.network.extern_data.check_matched_dataset(
      dataset=dataset, used_data_keys=engine.network.get_used_data_keys())
    self.engine = engine
    self.dataset_name = dataset_name
    # noinspection PyProtectedMember
    self.data_provider = self.engine._get_data_provider(dataset_name=dataset_name, dataset=dataset, batches=batches)
    assert isinstance(self.data_provider, DataProviderBase)
    if train_flag is None:
      train_flag = train
    self._train_flag = train_flag
    self._should_train = train
    self._should_eval = eval
    self.store_tf_profile = engine.config.bool("store_tf_profile", False)
    self.store_metadata_mod_step = engine.config.int("store_metadata_mod_step", 0)
    self.reset_updater_vars_mod_step = engine.config.int("reset_updater_vars_mod_step", 0)
    assert not (self.store_tf_profile and self.store_metadata_mod_step), (
      "Cannot use store_tf_profile and store_metadata_mod_step at the same time")
    self.finalized = False
    self.cancel_flag = False
    self.run_exception = None
    self.num_steps = None
    self.device_crash_batch = None  # type: typing.Optional[int]
    self.start_time = None
    self.elapsed = None
    self.report_prefix = None  # type: typing.Optional[str]
    self._results_accumulated = NumbersDict()  # entries like "cost:output" or "loss"
    self._inv_norm_accumulated = NumbersDict()  # entries like "output"
    self.num_frames_accumulated = NumbersDict()  # for each data key (eg. "classes"), corresponding number of frames
    self.results = {}  # type: typing.Dict[str,float]  # entries like "cost:output" or "loss"
    self.score = {}  # type: typing.Dict[str,float]  # entries like "cost:output"
    self.error = {}  # type: typing.Dict[str,float]  # entries like "error:output"
    self.stats = {}  # type: typing.Dict[str,typing.Union[float,numpy.ndarray,'Util.Stats']]  # entries like "stats:..."
    self.extra_fetches = extra_fetches
    if extra_fetches is not None:
      assert extra_fetches_callback
    self.extra_fetches_callback = extra_fetches_callback
    self._step_start_time = None  # type: typing.Optional[float]
    self._horovod_last_param_sync_time = time.time()  # we assume it is synced right now
    self._horovod_stopped_runner = False
    self._horovod_finish_all = False
    if engine.network.layers_desc.get("#finish_all_data", False):
      self._horovod_finish_all = True
    # With Horovod, during the main session.run, if reduce_type != grad or not training,
    # the following tensors are enough to ensure that we are in sync.
    self._horovod_collected_reduce_inputs = {}  # type: typing.Dict[str,(tf.Tensor,tf.Tensor)]  # name -> (input,output)

    from returnn.util.basic import terminal_size
    terminal_width, _ = terminal_size()
    self._show_interactive_process_bar = (log.verbose[3] and (not log.verbose[5]) and terminal_width >= 0)

  def _get_fetches_dict(self):
    """
    :return: values and actions which should be calculated and executed in self.run() by the TF session for each step
    :rtype: dict[str,tf.Tensor|tf.Operation]
    """
    d = {}
    optim_op = None
    if self._should_train:
      assert self.engine.updater

      def callback_on_new():
        """
        Called when we have something new.
        """
        # Force a new check.
        self.engine._checked_uninitialized_vars = False
        self.engine.updater.init_optimizer_vars(session=self.engine.tf_session)
      # This function has to be called before `get_fetches_dict(..., with_summary=True)`,
      # because it may introduce new summaries to the graph which are later collected.
      optim_op = self.engine.updater.get_optim_op(callback_on_new=callback_on_new)

    # Note that it is important that we do not recreate graph nodes for every call to this function.
    # Thus everything which we access here should be cached.
    d_fetches = self.engine.network.get_fetches_dict(
      config=self.engine.config,
      should_train=self._should_train, should_eval=self._should_eval,
      with_summary=True, with_size=True,
      horovod_collected_reduce_inputs=self._horovod_collected_reduce_inputs)
    d.update(d_fetches)
    if optim_op is not None:
      d["optim_op"] = optim_op
      if self.engine.updater.optim_meta_losses_dict:
        d.update(self.engine.updater.optim_meta_losses_dict)

    if self.extra_fetches is not None:
      for k, v in self.extra_fetches.items():
        if v is None:
          continue
        if isinstance(v, tf.Tensor):
          d["extra:%s" % k] = v
          continue
        if isinstance(v, LayerBase):
          v = v.output
        if callable(v):
          v = v()
          assert isinstance(v, tf.Tensor)
          d["extra:%s" % k] = v
          continue
        assert isinstance(v, Data)
        d["extra:%s" % k] = v.placeholder  # see _maybe_handle_extra_fetches, it will transform to batch-major there
        for i, s in v.size_placeholder.items():
          d["extra:%s:size_%i" % (k, i)] = s

    return d

  def _print_process(self, report_prefix, step, step_duration, eval_info):
    """
    :param str report_prefix:
    :param int step:
    :param float step_duration: in secs
    :param dict[str] eval_info: via :func:`_collect_eval_info`
    :return: nothing, will be printed to log
    """
    if not self._show_interactive_process_bar and not log.verbose[5]:
      return
    start_elapsed = time.time() - self.start_time
    complete = self.data_provider.get_complete_frac()
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
      from returnn.util.basic import progress_bar
      progress_bar(complete, hms(remaining_estimated))

  def _print_finish_process(self):
    if self._show_interactive_process_bar:
      from returnn.util.basic import progress_bar
      progress_bar()

  def _get_target_for_key(self, key):
    """
    :param str key: e.g. "cost:output" where the last part is the layer name. or "loss"
    :return: target name which is the data-key in the dataset, e.g. "classes"
    :rtype: str
    """
    if ":" in key:
      layer = self.engine.network.get_layer(key[key.find(":") + 1:])
      if layer.target:
        return layer.target
    return self.engine.network.extern_data.default_target

  def _finalize(self, num_steps):
    """
    Called at the end of an epoch.

    :param int num_steps: number of steps we did for this epoch
    """
    results = {key: self._normalize_loss(value, key, self._inv_norm_accumulated)
               for (key, value) in self._results_accumulated.items()}
    self.results = results
    self.score = {key: float(value) for (key, value) in results.items() if key.startswith("cost:")}
    if self.engine.config.bool("calculate_exp_loss", False):
      self.score.update({
        key + ":exp": float(numpy.exp(value)) for (key, value) in results.items() if key.startswith("cost:")})
    self.error = {key: float(value) for (key, value) in results.items() if key.startswith("error:")}
    self.num_steps = num_steps
    self.finalized = True

  def _get_batch_dim_from_fetches(self, fetches_results):
    """
    :param dict[str,numpy.ndarray|None] fetches_results: results of calculations, see self._get_fetches_dict()
    :rtype: int
    """
    default_target = self.engine.network.extern_data.default_target
    if "size:%s:0" % default_target in fetches_results:
      return len(fetches_results["size:%s:0" % default_target])
    for k, v in sorted(fetches_results.items()):
      if not k.startswith("size:"):
        continue
      if not k.endswith(":0"):
        continue
      return len(v)
    assert False, "batch-dim not found in %r" % fetches_results

  def _step_seq_len(self, fetches_results, data_key):
    """
    :param dict[str,numpy.ndarray|None] fetches_results: results of calculations, see self._get_fetches_dict()
    :param str data_key: e.g. "classes"
    :return: the seq length of this batch
    :rtype: int
    """
    seq_len_key = "size:%s:0" % data_key
    if seq_len_key in fetches_results:
      return numpy.sum(fetches_results[seq_len_key])
    else:
      # We assume that this data-key has no time axis. Use the batch-dim instead.
      return self._get_batch_dim_from_fetches(fetches_results)

  # noinspection PyMethodMayBeStatic
  def _normalize_loss(self, value, key, inv_loss_norm_factors):
    """
    :param T value:
    :param str key: e.g. "cost:output", "error:output" or "loss"
    :param NumbersDict inv_loss_norm_factors: keys e.g. e.g. "output" (layer names)
    :return: normalized value
    :rtype: T
    """
    if not value:
      return value
    if key == "loss":
      # This is a special case. This is the total loss.
      # Do not normalize this, as it is also used as-is for the gradient.
      # You can use the `use_normalized_loss` for a flag if you want to have this normalized.
      return value
    loss_norm_keys = inv_loss_norm_factors.keys()
    assert len(loss_norm_keys) > 0
    # Assume "cost:output" or "error:output" or so.
    assert ":" in key
    loss_norm_key = key[key.find(":") + 1:]
    assert loss_norm_key in loss_norm_keys, "unexpected key %r" % key
    value = value / inv_loss_norm_factors[loss_norm_key]
    return value

  def _collect_eval_info(self, fetches_results):
    """
    :param dict[str,numpy.ndarray|None] fetches_results: results of calculations, see self._get_fetches_dict()
    :return: dict for printing the step stats, see self._print_process(), e.g. {"cost:output": 2.3}
    :rtype: dict[str,float]
    """
    # See see self._get_fetches_dict() for the keys.
    # keys are e.g. "cost:output", "error:output" or "loss".
    keys = [k for k in fetches_results.keys() if k.startswith("cost:") or k.startswith("error:") or k == "loss"]
    # step_seq_lens keys are e.g. "data" or "classes".
    step_seq_lens = {
      k[len("size:"):-2]: numpy.sum(v)
      for (k, v) in fetches_results.items()
      if k.startswith("size:") and k.endswith(":0")}
    # loss_norm_factors keys are e.g. "output" (layer names).
    loss_norm_factors = {
      k[len("loss_norm_factor:"):]: v for (k, v) in fetches_results.items() if k.startswith("loss_norm_factor:")}
    inv_loss_norm_factors = NumbersDict({k: 1.0 / v for (k, v) in loss_norm_factors.items()})

    # Accumulate for epoch stats.
    self._results_accumulated += NumbersDict({key: fetches_results[key] for key in keys})
    self._inv_norm_accumulated += inv_loss_norm_factors
    self.num_frames_accumulated += NumbersDict(step_seq_lens)

    # Prepare eval info stats for this batch run.
    eval_info = {}
    for key in keys:
      value = fetches_results[key]
      value = self._normalize_loss(value, key, inv_loss_norm_factors)
      eval_info[key] = value
      if self.engine.config.bool("calculate_exp_loss", False) and key.startswith("cost:"):
        eval_info[key + ":exp"] = numpy.exp(value)

    # Add batch size info.
    if self.engine.config.bool("log_batch_size", False):
      for k, v in sorted(fetches_results.items()):
        if not k.startswith("size:"):
          continue
        if not k.endswith(":0"):
          continue
        eval_info["num_seqs"] = len(v)
        eval_info["max_size:%s" % k[len("size:"):-len(":0")]] = max(v)

    # Add raw stats.
    for k, v in fetches_results.items():
      if k.startswith("stats:"):
        if v.ndim == 1:
          v = list(v)  # looks nicer in logs
        eval_info[k] = v
        self.stats[k] = v  # Always just store latest value.
      if k.startswith("mem_usage:"):
        from returnn.util.basic import human_bytes_size, Stats
        self.stats.setdefault(k, Stats(format_str=human_bytes_size))
        self.stats[k].collect([v])
        eval_info[k] = human_bytes_size(int(v))

    return eval_info

  def _maybe_handle_extra_fetches(self, fetches_results):
    """
    :param dict[str,numpy.ndarray|str] fetches_results: results of calculations, see self._get_fetches_dict()
    """
    if self.extra_fetches is None:
      return
    d = {}
    for k, v in self.extra_fetches.items():
      if v is None:
        d[k] = None
        continue
      r = fetches_results["extra:%s" % k]
      if isinstance(v, tf.Tensor):
        d[k] = r
        continue
      if isinstance(v, LayerBase):
        v = v.output
      if callable(v):
        d[k] = fetches_results["extra:%s" % k]
        continue
      assert isinstance(v, Data)
      if v.batch_dim_axis != 0:
        r = numpy.moveaxis(r, v.batch_dim_axis, 0)
      if v.have_time_axis() and v.is_time_axis_dynamic():
        assert v.time_dim_axis_excluding_batch == 0
        assert list(v.size_placeholder.keys()) == [0]
        seq_lens = fetches_results["extra:%s:size_0" % k]  # shape: (batch,)
        assert seq_lens.shape == (r.shape[0],)
        d[k] = [r[i, :seq_lens[i]] for i in range(seq_lens.shape[0])]
      else:
        d[k] = list(r)
    self.extra_fetches_callback(**d)

  def _horovod_finish_data(self, local_step):
    """
    :param int local_step:
    """
    if not self.engine.config.is_true("use_horovod"):
      return
    while True:
      hvd_stop, hvd_error = self._horovod_signal_broadcast(have_more_data=False)
      if hvd_error:
        print("WARNING: Horovod error just after finishing the epoch... (pid %i)" % os.getpid(), file=log.v2)
        return  # or raise? but we just finished. let's first save etc
      if hvd_stop:  # we expect this
        return
      # E.g. horovod_finish_all is enabled. Some other peer is still working.
      start_time = time.time()
      self._horovod_empty_step(local_step=local_step)
      self._print_process(
        report_prefix="%s (empty Horovod step, pid %i)" % (self.report_prefix, os.getpid()),
        step=local_step, step_duration=time.time() - start_time, eval_info={})
      local_step += 1

  def _horovod_signal_error(self):
    self._horovod_signal_broadcast(have_more_data=False, error=True)

  def _horovod_signal_have_more_data(self, local_step):
    """
    :param int local_step:
    :return: whether to stop (because some other instance stopped), whether an error occurred
    :rtype: (bool, bool)
    """
    if not tf_horovod.get_ctx():
      return False, False
    if not tf_horovod.get_ctx().should_sync_every_step():
      # We only need to sync for the param sync.
      if not self._horovod_should_sync_params_now(local_step=local_step):
        return False, False
    return self._horovod_signal_broadcast(have_more_data=True)

  def _horovod_signal_broadcast(self, have_more_data=True, error=False):
    """
    :param bool have_more_data: whether we have more data in this instance
    :param bool error: whether some error occurred here
    :return: whether to stop (because some other instance stopped), whether an error occurred
    :rtype: (bool, bool)
    """
    if not self.engine.config.is_true("use_horovod"):
      return False, False
    # Stopped before? Keep in sync -> Don't send anything anymore, other peers do not expect it.
    if self._horovod_stopped_runner:
      return True, False
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    import horovod.tensorflow as hvd
    from returnn.tf.util.basic import global_tensor
    have_more_data_placeholder = global_tensor(
      lambda: tf_compat.v1.placeholder(tf.int32, shape=(), name="horovod_have_more_data_placeholder"),
      name="horovod_have_more_data_placeholder")  # 0 or 1
    sum_have_data_t = global_tensor(
      lambda: hvd.allreduce(have_more_data_placeholder, average=False),
      name="horovod_sum_have_data")  # 0..size
    have_error_placeholder = global_tensor(
      lambda: tf_compat.v1.placeholder(tf.int32, shape=(), name="horovod_have_error_placeholder"),
      name="horovod_have_error_placeholder")  # 0 or 1
    sum_have_error_t = global_tensor(
      lambda: hvd.allreduce(have_error_placeholder, average=False),
      name="horovod_sum_have_error")  # 0..size
    sum_have_data, sum_have_error = self.engine.tf_session.run(
      (sum_have_data_t, sum_have_error_t),
      feed_dict={
        have_more_data_placeholder: 1 if have_more_data else 0,
        have_error_placeholder: 1 if error else 0})
    stop = False
    error_occurred = sum_have_error > 0
    if error_occurred:  # some peer had an error
      stop = True
    if self._horovod_finish_all:
      if sum_have_data == 0:  # no peer has data anymore
        stop = True
    else:
      if sum_have_data < hvd.size():  # some of the peers do not have data anymore
        stop = True
    if stop:
      # Other peers will not expect further signals.
      self._horovod_stopped_runner = True
    return stop, error_occurred

  def _horovod_empty_step(self, local_step):
    """
    Call this if you want to proceed one step without doing anything.
    E.g. when this local rank has finished the dataset but some other rank has not yet.

    We assume that _horovod_signal_broadcast was called just before
    (in any case, even if should_sync_every_step is False).

    :param int local_step:
    """
    hvd_ctx = tf_horovod.get_ctx()
    if self._horovod_collected_reduce_inputs:
      assert hvd_ctx.should_sync_every_step()
      assert not self._should_train or not hvd_ctx.is_reduce_type_grad()
      feed_dict = {}
      fetches = {}
      for key, (tensor_in, tensor_out) in self._horovod_collected_reduce_inputs.items():
        feed_dict[tensor_in] = 0.0  # should be scalar
        fetches[key] = tensor_out
      self.engine.tf_session.run(fetches)
    else:
      assert not hvd_ctx.should_sync_every_step()
    # Need to call this to keep communication in sync.
    # Note: If used, sync always (via is_final=True), even if should_sync_every_step is False,
    # because we always called _horovod_signal_broadcast before this.
    self._horovod_sync_params(local_step=local_step, is_final=True)

  def _horovod_should_sync_params_now(self, local_step, is_final=False):
    """
    :param int local_step:
    :param bool is_final:
    :rtype: bool
    """
    hvd_ctx = tf_horovod.get_ctx()
    if not hvd_ctx:
      return False
    if not hvd_ctx.is_reduce_type_param():
      return False
    if not self._should_train:
      return False
    if is_final:
      return True
    sync_time_diff = hvd_ctx.get_param_sync_time_diff()
    if sync_time_diff is not None:
      dt = self._step_start_time - self._horovod_last_param_sync_time
      assert dt >= 0.
      assert not hvd_ctx.should_sync_every_step()  # we would be out-of-sync otherwise
      return dt >= sync_time_diff
    sync_step = hvd_ctx.get_param_sync_step()
    assert sync_step >= 1
    return local_step % sync_step == sync_step - 1

  def _horovod_sync_params(self, local_step, is_final=False):
    """
    Horovod reduce type 'param', i.e. each node (rank) does update independently,
    but after N steps, we average params.

    :param int local_step: step of this epoch
    :param bool is_final:
    :return: TF runtime
    :rtype: float
    """
    if not self._horovod_should_sync_params_now(local_step=local_step, is_final=is_final):
      return 0.0
    from returnn.tf.util.basic import global_tensor
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    import horovod.tensorflow as hvd

    # noinspection PyShadowingNames
    def assign_avg_var(var):
      """
      :param tf.Variable var:
      :rtype: tf.Tensor
      """
      return tf_compat.v1.assign(var, hvd.allreduce(var.read_value(), average=True))

    assign_ops = []
    for var in self.engine.updater.trainable_vars:
      assign_ops.append(global_tensor(
        lambda: assign_avg_var(var),
        name="horovod_sync_params__var_%s" % var.name[:-2].replace("/", "_")).op)
    start_time = time.time()
    self.engine.tf_session.run(assign_ops)
    self._horovod_last_param_sync_time = time.time()
    return self._horovod_last_param_sync_time - start_time

  def run(self, report_prefix):
    """
    :param str report_prefix: prefix for logging, e.g. "train"
    """
    self.report_prefix = report_prefix
    sess = self.engine.tf_session
    if self.engine.config.has("tf_log_dir"):
      logdir = self.engine.config.value("tf_log_dir", None)
    elif self.engine.model_filename:
      logdir = os.path.dirname(self.engine.model_filename)
    elif log.filename:
      logdir = os.path.dirname(log.filename)
    else:
      logdir = None
    if logdir:
      from returnn.util.basic import log_runtime_info_to_dir, get_utc_start_time_filename_part
      logdir += "/%s" % self.data_provider.get_dataset_name()
      if not self._should_train:  # like eval
        logdir += "-%i" % self.engine.epoch
      if self.engine.use_search_flag:
        logdir += "-search"
      logdir += "-%s" % get_utc_start_time_filename_part()
      if self.engine.config.is_true("dry_run"):
        logdir += "-dryrun"
      # noinspection PyProtectedMember
      if self.engine._do_save():
        log_runtime_info_to_dir(logdir, config=self.engine.config)
      writer = tf_compat.v1.summary.FileWriter(logdir)
    else:
      writer = None
    print("TF: log_dir: %s" % logdir, file=log.v5)
    run_metadata = tf_compat.v1.RunMetadata()
    debug_shell_in_runner = self.engine.config.bool("debug_shell_in_runner", False)
    debug_shell_in_runner_step = self.engine.config.int("debug_shell_in_runner_step", 1)

    # Not sure if this is the best thing to do for an evaluation but it's ok for now.
    # We could also set it to 0 for non train epochs.
    step_offset = self.engine.network.get_global_train_step(session=sess)

    coord = self.data_provider.coord

    threads = tf_compat.v1.train.start_queue_runners(sess=sess, coord=coord)
    self.data_provider.start_threads(session=sess)
    self.start_time = time.time()
    elapsed_time_tf = 0.0
    step = None
    fetches_dict = None
    feed_dict = None
    meta_step_info = None
    try:
      # step is like mini-batch in our usual terminology
      step = 0
      self.engine.network.set_run_opts(epoch=self.engine.epoch, dataset_name=self.dataset_name)
      fetches_dict = self._get_fetches_dict()
      # After get_fetches_dict, maybe some new uninitialized vars. Last check.
      self.engine.check_uninitialized_vars()
      # Also, add graph to summary here because the updater/optimizer might not have been created before.
      if writer:
        writer.add_graph(sess.graph)
      if self.store_tf_profile:
        tf.profiler.experimental.start(logdir)
      hvd_stop = hvd_error = False
      while self.data_provider.have_more_data(session=sess):
        self._step_start_time = time.time()
        hvd_stop, hvd_error = self._horovod_signal_have_more_data(local_step=step)
        if hvd_error:
          raise Exception("Some other Horovod peer failed.")
        if hvd_stop:
          # Some other peer does not have data anymore, but no error occurred.
          break
        feed_dict, meta_step_info = self.data_provider.get_feed_dict()
        if isinstance(self.engine.network.train_flag, tf.Tensor):
          feed_dict[self.engine.network.train_flag] = self._train_flag
        if isinstance(self.engine.network.epoch_step, tf.Tensor):
          feed_dict[self.engine.network.epoch_step] = step
        start_time = time.time()
        if self._should_train and self.reset_updater_vars_mod_step and step % self.reset_updater_vars_mod_step == 0:
          print("Reset updater vars in step %i." % step, file=log.v5)
          self.engine.updater.init_optimizer_vars(session=sess)

        if step == 0:
          if self.engine.config.bool("check_unsupported_device", False) and self.engine.is_requesting_for_gpu():
            from returnn.tf.util.basic import find_unsupported_devices_in_graph
            ops = find_unsupported_devices_in_graph(graph=sess.graph, dev_name="GPU")
            if not ops:
              print("All ops in graph can be run on GPU.")
            else:
              print("The following ops do not have a GPU kernel:")
              pprint(ops)

        if debug_shell_in_runner and debug_shell_in_runner_step == step:
          print("debug_shell_in_runner, step %i" % step, file=log.v1)
          from returnn.util.debug import debug_shell
          debug_shell(user_ns=locals(), user_global_ns=globals(), exit_afterwards=False)

        # Now do one calculation step. Optionally with metadata.
        try:
          if self.store_metadata_mod_step and step % self.store_metadata_mod_step == 0:
            # Slow run that stores extra information for debugging.
            print('Storing metadata', file=log.v5)
            run_options = tf_compat.v1.RunOptions(
              trace_level=tf_compat.v1.RunOptions.FULL_TRACE)
            # We could use tfdbg.add_debug_tensor_watch here.
            session_run_start_time = time.time()
            fetches_results = sess.run(
              fetches_dict,
              feed_dict=feed_dict,
              options=run_options,
              run_metadata=run_metadata)  # type: typing.Dict[str,typing.Union[numpy.ndarray,str]]
            elapsed_time_tf += time.time() - session_run_start_time
            writer.add_summary(fetches_results["summary"], step + step_offset)
            writer.add_run_metadata(run_metadata, 'step_{:04d}'.format(step + step_offset))
            tl = timeline.Timeline(run_metadata.step_stats)
            timeline_path = os.path.join(logdir, 'timeline.trace')
            with open(timeline_path, 'w') as f:
              f.write(tl.generate_chrome_trace_format(show_memory=True))
          else:
            session_run_start_time = time.time()
            if self.store_tf_profile:
              with tf.profiler.experimental.Trace(name=report_prefix, step_num=step + step_offset):
                fetches_results = sess.run(
                  fetches_dict, feed_dict=feed_dict)  # type: typing.Dict[str,typing.Union[numpy.ndarray,str]]
            else:
              fetches_results = sess.run(
                fetches_dict, feed_dict=feed_dict)  # type: typing.Dict[str,typing.Union[numpy.ndarray,str]]
            elapsed_time_tf += time.time() - session_run_start_time
            if writer and "summary" in fetches_results:
              writer.add_summary(fetches_results["summary"], step + step_offset)
        except tf.errors.OpError as exc:
          if isinstance(exc, tf.errors.OutOfRangeError) and isinstance(self.data_provider, DatasetDataProvider):
            # This means that we got end-of-sequence from the dataset iterator.
            self.data_provider.current_dataset_reached_end = True
            break
          print("TensorFlow exception:", exc, file=log.v1)
          # With Horovod: We are likely out-of-sync now. Avoid any further communication.
          self._horovod_stopped_runner = True
          # Extra info will be printed below.
          raise

        eval_info = self._collect_eval_info(fetches_results=fetches_results)
        self._maybe_handle_extra_fetches(fetches_results)
        elapsed_time_tf += self._horovod_sync_params(local_step=step)
        duration = time.time() - start_time
        self._print_process(report_prefix=report_prefix, step=step, step_duration=duration, eval_info=eval_info)

        if self.engine.config.bool("stop_on_nonfinite_train_score", True):
          score_values = self._results_accumulated.values()
          if any(numpy.isinf(score_values)) or any(numpy.isnan(score_values)):
            print("Model seems broken, got inf or nan score.", file=log.v1)
            print("Accumulated scores:", self._results_accumulated, file=log.v1)
            raise Exception("Inf/nan score in step %i." % step)

        step += 1
        if self.cancel_flag:
          raise CancelTrainingException("cancel_flag is set")

      self._print_finish_process()

      if not hvd_stop and not self.data_provider.have_reached_end():
        raise Exception("Did not successfully reached the end of the dataset.")

      if self._should_train:
        final_global_train_step = self.engine.network.get_global_train_step(session=sess)
        assert step + step_offset == final_global_train_step

      self._horovod_finish_data(local_step=step)
      self._horovod_sync_params(local_step=step, is_final=True)
      self.engine.network.set_run_finished()
      self._finalize(num_steps=step)

      if self.stats:
        print("Stats:", file=log.v1)
        for k, v in sorted(self.stats.items()):
          print("  %s:" % k, v, file=log.v1)
      elapsed = time.time() - self.start_time
      elapsed_tf_percentage = (elapsed_time_tf / elapsed) if (elapsed > 0) else 0.0
      print("%s, finished after %i steps, %s elapsed (%.1f%% computing time)" % (
        report_prefix, step, hms(elapsed), (elapsed_tf_percentage * 100.)), file=log.v3)

    except KeyboardInterrupt as exc:
      print("KeyboardInterrupt in step %r." % step)
      self.run_exception = exc

    except BaseException as exc:
      print("Exception %r in step %r. (pid %i)" % (exc, step, os.getpid()), file=log.v1)
      if not isinstance(exc, CancelTrainingException):
        help_on_tf_exception(
          session=sess,
          exception=exc, fetches=fetches_dict, feed_dict=feed_dict, meta_step_info=meta_step_info,
          extern_data=self.data_provider.extern_data, file=log.v2)
        sys.excepthook(*sys.exc_info())
      self.device_crash_batch = step
      self.run_exception = exc

    finally:
      # Try and ignore certain exceptions as we anyway should try to clean up as much as possible.
      from returnn.util.basic import try_and_ignore_exception
      from returnn.tf.util.basic import stop_event_writer_thread
      try_and_ignore_exception(self._horovod_signal_error)  # ignored if _horovod_finish_data was called before
      if writer:
        try_and_ignore_exception(writer.close)
        try_and_ignore_exception(lambda: stop_event_writer_thread(writer.event_writer))
      try_and_ignore_exception(coord.request_stop)
      try_and_ignore_exception(lambda: coord.join(threads))
      try_and_ignore_exception(self.data_provider.stop_threads)
      if self.store_tf_profile:
        try_and_ignore_exception(tf.profiler.experimental.stop)
      # ignored if called before
      try_and_ignore_exception(lambda: self.engine.network.set_run_finished(error_occurred=True))
      self.elapsed = time.time() - self.start_time

  def exit_due_to_error(self):
    """
    Exit due to an previous error.
    """
    assert not self.finalized
    if self.run_exception:
      # If this is run inside a debugger, reraise the exception.
      get_trace = getattr(sys, "gettrace", None)
      if get_trace and get_trace() is not None:
        raise self.run_exception
      # We do not handle the exception otherwise anymore as this was already handled in Runner.run().
    sys.exit(1)


class Engine(EngineBase):
  """
  TF backend engine.
  """

  def __init__(self, config=None):
    """
    :param Config.Config|None config:
    """
    super(Engine, self).__init__()
    if config is None:
      from returnn.config import get_global_config
      config = get_global_config(auto_create=True)
    if not log.initialized:
      log.init_by_config(config)
    if not BehaviorVersion.is_set():
      BehaviorVersion.set(config.int('behavior_version', None))
    if BackendEngine.selectedEngine is None:
      BackendEngine.select_engine(engine=BackendEngine.TensorFlow)
    assert BackendEngine.is_tensorflow_selected()
    self.config = config
    self.orig_config = {}  # see _maybe_update_config
    self.custom_get_net_dict = None  # type: typing.Optional[typing.Callable]
    self._check_devices()
    self.tf_session = None  # type: typing.Optional[tf.compat.v1.Session]
    self.network = None  # type: typing.Optional[TFNetwork]
    self.updater = None  # type: typing.Optional[Updater]
    self.learning_rate_control = None  # type: typing.Optional[LearningRateControl]
    self._checked_uninitialized_vars = False
    self._merge_all_summaries = None
    self.dataset_batches = {}  # type: typing.Dict[str,BatchSetGenerator]
    self.dataset_provider = None  # type: typing.Optional[DatasetDataProvider]
    self.train_data = None  # type: typing.Optional[Dataset]
    self.eval_datasets = {}  # type: typing.Dict[str,Dataset]
    self.start_epoch = None  # type: typing.Optional[int]
    self.use_dynamic_train_flag = False
    self.use_search_flag = config.value("task", None) == "search"
    self.use_eval_flag = config.value("task", None) != "forward"
    self.learning_rate = 0.0  # set in init_train_epoch
    self._const_cache = {}  # type: typing.Dict[str,tf.Tensor]
    self.preload_from_files = None  # type: typing.Optional[typing.Dict[str,typing.Dict[str]]]
    self.max_seqs = None  # type: typing.Optional[int]

  def finalize(self, error_occurred=False):
    """
    Finalizes the TF session, network, graph.
    """
    self._close_tf_session()
    self._reset_graph(error_occurred=error_occurred)

  def get_const_tensor(self, key, value):
    """
    :param key:
    :param value:
    :return: tf.constant(value)
    :rtype: tf.Tensor
    """
    if key not in self._const_cache:
      self._const_cache[key] = tf.constant(value=value, name="const_%s" % key)
    return self._const_cache[key]

  def is_requesting_for_gpu(self):
    """
    :rtype: bool
    """
    from returnn.config import tf_should_use_gpu
    return tf_should_use_gpu(self.config)

  def _check_devices(self):
    from returnn.tf.util.basic import is_gpu_available
    if self.is_requesting_for_gpu():
      assert is_gpu_available(), "no GPU available"
    else:
      if is_gpu_available():
        print("Note: There is a GPU available but you have set device=cpu.", file=log.v2)

  def _close_tf_session(self):
    if self.tf_session:
      self.tf_session.close()
    self.tf_session = None

  def make_tf_session(self):
    """
    Initializes self.tf_session.
    """
    self._close_tf_session()
    opts = self.config.typed_value("tf_session_opts", {})
    assert isinstance(opts, dict)
    opts = opts.copy()
    # See options here:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto
    opts.setdefault("log_device_placement", False)
    opts.setdefault("device_count", {})
    if self.is_requesting_for_gpu():
      opts["device_count"].setdefault("GPU", 1)
    else:
      opts["device_count"].setdefault("GPU", 0)
    # Note: We don't set intra_op_parallelism_threads and inter_op_parallelism_threads here anymore
    # because it is safer to do it via setup_tf_thread_pools() which we call very early.
    print("Setup TF session with options %r ..." % opts, file=log.v2)
    config = tf_compat.v1.ConfigProto(**opts)
    # config.gpu_options.allow_growth=True
    session_opts = dict(config=config)
    if self.config.is_true("distributed_tf"):
      import returnn.tf.distributed
      session_opts["target"] = returnn.tf.distributed.get_session_target()
    # For debugging, see tfdbg.LocalCLIDebugWrapperSession.
    self.tf_session = tf_compat.v1.Session(**session_opts)

  def _reset_graph(self, error_occurred=False):
    """
    Resets the default graph (of the current thread),
    and clears up any cached tensors created in it.

    :param bool error_occurred:
    """
    if self.network and not error_occurred:
      self.network.call_graph_reset_callbacks()
    tf_compat.v1.reset_default_graph()
    self._checked_uninitialized_vars = False
    self._merge_all_summaries = None
    self._const_cache.clear()
    self.network = None
    self.updater = None

  def get_eval_datasets(self):
    """
    :return: dict of datasets used for eval (dev, eval)
    :rtype: dict[str,Dataset]
    """
    return self.eval_datasets

  @property
  def dev_data(self):
    """
    :rtype: Dataset|None
    """
    return self.eval_datasets.get("dev", None)

  @dev_data.setter
  def dev_data(self, value):
    """
    :param Dataset|None value:
    """
    self.eval_datasets.pop("dev", None)
    if value:
      self.eval_datasets["dev"] = value

  @property
  def eval_data(self):
    """
    :rtype: Dataset|None
    """
    return self.eval_datasets.get("eval", None)

  @eval_data.setter
  def eval_data(self, value):
    """
    :param Dataset|None value:
    """
    self.eval_datasets.pop("eval", None)
    if value:
      self.eval_datasets["eval"] = value

  def load_model(self, epoch=None, filename=None):
    """
    :param int epoch:
    :param str filename:
    """
    assert epoch or filename
    if epoch:
      assert not filename
      filename = self.get_epoch_model_filename(epoch=epoch)
    print("Load model %s" % (filename,), file=log.v4)
    self.network.load_params_from_file(filename, session=self.tf_session)

  def save_model(self, filename=None):
    """
    :param str filename: full filename for model
    """
    if not self._do_save():
      return
    if not filename:
      filename = self.get_epoch_model_filename()
    print("Save model under %s" % (filename,), file=log.v4)
    self.network.save_params_to_file(filename, session=self.tf_session)

  @staticmethod
  def delete_model(filename):
    """
    :param str filename:
    :return: accumulated file-size in bytes of deleted files
    :rtype: int
    """
    # This assumes TensorFlow models here.
    # They consists of multiple files with the extensions ".index", ".meta" and ".data*".
    from glob import glob
    count_bytes = 0
    assert os.path.exists(filename + ".index")
    for fn in glob(filename + "*"):
      fn_ext = os.path.splitext(fn)[1]
      if fn_ext not in [".index", ".meta"] and not fn_ext.startswith(".data"):
        continue
      count_bytes += os.stat(fn).st_size
      os.remove(fn)
    assert count_bytes > 0
    return count_bytes

  # noinspection PyAttributeOutsideInit
  def init_train_from_config(self, config=None, train_data=None, dev_data=None, eval_data=None):
    """
    :param Config.Config|None config:
    :param Dataset|None train_data:
    :param Dataset|None dev_data:
    :param Dataset|None eval_data:
    """
    if not config:
      config = self.config
    if not config.has("num_inputs") and not config.has("num_outputs") and not config.has("extern_data") and (
          train_data or dev_data or eval_data):
      from returnn.datasets.basic import set_config_num_inputs_outputs_from_dataset
      set_config_num_inputs_outputs_from_dataset(config=config, dataset=train_data or dev_data or eval_data)
    self.use_dynamic_train_flag = True
    self.train_data = train_data
    self.eval_datasets.clear()
    if dev_data:
      self.eval_datasets["dev"] = dev_data
    if eval_data:
      self.eval_datasets["eval"] = eval_data
    if config.has("eval_datasets"):
      for dataset_name, dataset_opts in config.typed_value("eval_datasets", {}).items():
        self.eval_datasets[dataset_name] = init_dataset(dataset_opts, default_kwargs={"name": dataset_name})
    self.start_epoch, self.start_batch = self.get_train_start_epoch_batch(config)
    self.batch_size = config.typed_value('batch_size', 1)
    self.shuffle_batches = config.bool('shuffle_batches', False)
    self.update_batch_size = config.int('update_batch_size', 0)
    self.save_model_epoch_interval = config.int('save_interval', 1)
    self.save_epoch1_initial_model = config.bool('save_epoch1_initial_model', False)
    self.learning_rate_control = load_learning_rate_control_from_config(config)
    self.learning_rate = self.learning_rate_control.default_learning_rate
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
    self.max_seq_length = config.typed_value('max_seq_length', None) or config.float('max_seq_length', 0)
    self.min_seq_length = config.typed_value('min_seq_length', None) or config.float('min_seq_length', 0)
    self.inc_seq_length = config.float('inc_seq_length', 0)
    if not self.max_seq_length:
      self.max_seq_length = sys.maxsize  # type: typing.Union[int,float,typing.Dict[str,int],NumbersDict]
    if isinstance(self.max_seq_length, dict):
      self.max_seq_length = NumbersDict(self.max_seq_length)
    assert isinstance(self.max_seq_length, (int, float, NumbersDict))
    if not self.min_seq_length:
      self.min_seq_length = 0
    if isinstance(self.min_seq_length, dict):
      self.min_seq_length = NumbersDict(self.min_seq_length)
    assert isinstance(self.min_seq_length, (int, float, NumbersDict))
    self.max_pad_size = config.typed_value("max_pad_size", None)
    # And also initialize the network. That depends on some vars here such as pretrain.
    self.init_network_from_config(config)

  def get_net_dict_for_epoch(self, epoch, config=None):
    """
    :param int epoch:
    :param Config.Config|None config:
    :rtype: dict[str]
    """
    if not config:
      config = self.config
    if self.is_pretrain_epoch(epoch=epoch):
      # This would be obsolete if we don't want to load an existing model.
      # In self.init_train_epoch(), we initialize a new model.
      net_dict = self.pretrain.get_network_json_for_epoch(epoch)
    elif self.custom_get_net_dict:
      net_dict = self.custom_get_net_dict(epoch=epoch)
      assert isinstance(net_dict, dict), "%s should return dict but returned %s" % (
        self.custom_get_net_dict, type(net_dict))
    elif self.pretrain:
      # Use the net from pretrain. This might resolve things like WrapEpochValue.
      net_dict = self.pretrain.get_final_network_json()
    else:
      from returnn.config import network_json_from_config
      net_dict = network_json_from_config(config)
    return net_dict

  def init_network_from_config(self, config=None, net_dict_post_proc=None):
    """
    :param Config.Config|None config:
    :param ((dict)->dict)|None net_dict_post_proc:
    """
    if not config:
      config = self.config
    self.model_filename = config.value('model', None)
    self.preload_from_files = config.typed_value('preload_from_files', {})
    self.pretrain = pretrain_from_config(config)
    self.custom_get_net_dict = config.typed_value("get_network")
    self.max_seqs = config.int('max_seqs', -1)

    epoch, model_epoch_filename = self.get_epoch_model(config)
    # Note that model_epoch_filename could be set but epoch could be None or 0.
    if not model_epoch_filename and not self.start_epoch:
      if self.config.bool("allow_random_model_init", False):
        print("No model will be loaded. Randomly initializing model.", file=log.v2)
        epoch = 1
      else:
        raise Exception(
          "You are not using training, otherwise start_epoch would be set via self.init_train_from_config(). "
          "There was also no model found which we could load. Set one via 'load'.")
    # self.start_epoch is used as the start epoch in training.
    # If there is an existing model, it might be higher than 1.
    # In that case, epoch == self.start_epoch - 1.
    is_training = config.value('task', 'train') == 'train'
    is_first_train_epoch = not epoch and (is_training or config.value('task', 'train') == 'initialize_model')
    self.epoch = epoch
    if is_first_train_epoch:
      assert self.start_epoch >= 1
      self.epoch = self.start_epoch
    assert self.epoch, "task %r" % config.value("task", "train")

    net_dict = self.get_net_dict_for_epoch(epoch=self.epoch, config=config)
    if net_dict_post_proc:
      net_dict = net_dict_post_proc(net_dict)

    self._maybe_update_config(net_desc=net_dict, epoch=self.epoch)
    self._init_network(net_desc=net_dict, epoch=self.epoch)

    if self.preload_from_files:
      # Notes for related options:
      # - import_model_train_epoch1. This however requires all params to exist in the checkpoint.
      # - SubnetworkLayer also has a load_on_init option.
      # - LayerBase has custom_param_importer which is quite flexible.
      print("Start pre-loading weights...", file=log.v2)
      # model_name will not be used directly, but it defines the order in which we apply the preloading.
      # Variables are initialized by the first preload.
      for model_name, opts in sorted(self.preload_from_files.items()):
        assert isinstance(opts, dict)
        if opts.get("init_for_train", False):
          if not is_first_train_epoch:
            continue
        else:  # default: init for recog
          if is_training:
            continue
        model_filename = opts['filename']
        print("loading weights from", model_filename, file=log.v2)
        self_prefix = self.network.get_absolute_name_scope_prefix()  # "" if root, otherwise with "/" at end
        load_if_prefix = opts.get('prefix', '')  # prefix to identify the variables to be restored from the file
        from returnn.tf.network import CustomCheckpointLoader
        loader = CustomCheckpointLoader(
          filename=model_filename,
          saveable_params=self.network.get_params_list(),
          params_prefix=self_prefix, load_if_prefix=load_if_prefix,
          ignore_missing=opts.get("ignore_missing", False),
          ignore_params=opts.get("ignore_params", ()),
          ignore_params_prefixes=opts.get("ignore_params_prefixes", ()),
          var_name_mapping=opts.get("var_name_mapping", {}))
        # `set_as_custom_init` is also a marker for the vars, that they are preloaded,
        # such that further checkpoint loaders will not load them again.
        loader.set_as_custom_init()
        loader.load_now(session=self.tf_session)

    if model_epoch_filename:
      print("loading weights from", model_epoch_filename, file=log.v2)
      try:
        self.network.load_params_from_file(model_epoch_filename, session=self.tf_session)
      except tf.errors.NotFoundError:
        print("Exiting now because model cannot be loaded.", file=log.v1)
        sys.exit(1)

  def _maybe_update_config(self, net_desc, epoch):
    """
    This is a slightly hacky way to overwrite entries in the config, via the network description.
    This can e.g. be used in pretraining to overwrite certain settings such as batch_size.

    :param dict[str,dict[str]] net_desc:
    :param int epoch:
    """
    updated_datasets = {}  # type: typing.Dict[str,Dataset]

    # noinspection PyShadowingNames
    def set_value(key, value):
      """
      :param str key:
      :param value:
      """
      assert key in self.config.typed_dict
      self.config.typed_dict[key] = value
      # Some entries need specific handling, e.g. to update our attribs.
      if key == "max_seq_length":
        # See init_train_from_config.
        if not value:
          value = sys.maxsize
        if isinstance(value, dict):
          value = NumbersDict(value)
        assert isinstance(value, (int, float, NumbersDict))
      if key in ["batch_size", "max_seq_length", "max_seqs", "inc_seq_length", "seq_drop", "seq_drop_freq"]:
        # To be sure, never keep the batch order.
        self.dataset_batches.clear()
        setattr(self, key, value)
      if key == "chunking" and self.train_data:
        self.dataset_batches.pop("train", None)
        # Note that this might not be 100% correct:
        # E.g. if the dataset explicitly overwrites chunking.
        # However, we assume, if the user explicitly specify to overwrite chunking now, that it should be applied.
        # Also note, if we overwrite the dataset later, this would be ignored anyway,
        # but in that case, the newly initialized dataset
        # would use the right chunking option from the config overwrite.
        # noinspection PyProtectedMember
        self.train_data.chunk_size, self.train_data.chunk_step = Dataset._parse_chunking(value)
      if key in ["train", "dev", "eval"]:
        # `train` actually gets some special treatment, but unify nevertheless now.
        key, value = "eval_datasets", {key: value}
      if key == "eval_datasets":
        assert isinstance(value, dict)
        for key_, value_ in value.items():
          self.dataset_batches.pop(key_, None)
          from returnn.datasets.basic import init_dataset
          dataset_kwargs = {"name": key_}
          if key_ != "train":
            dataset_kwargs.update(Dataset.get_default_kwargs_eval(config=self.config))
          Dataset.kwargs_update_from_config(config=self.config, kwargs=dataset_kwargs)
          dataset = init_dataset(value_, default_kwargs=dataset_kwargs)
          old_dataset = self.train_data if key_ == "train" else self.eval_datasets.get(key_)
          if old_dataset:
            assert isinstance(old_dataset, Dataset)
            old_dataset.finish_epoch()
          if key_ == "train":
            self.train_data = dataset
          else:
            self.eval_datasets[key_] = dataset
          updated_datasets[key_] = dataset

    config_overwrites = net_desc.get("#config", {})
    old_orig_config = self.orig_config.copy()
    self.orig_config.clear()
    keys = sorted(set(config_overwrites.keys()).union(set(old_orig_config.keys())))
    # See Dataset.kwargs_update_from_config.
    for ds_args in ["chunking", "min_chunk_size", "chunking_variance", "batching", "window", "context_window"]:
      if ds_args in keys:
        # Add at the beginning. If we later overwrite the dataset, it would use the overwritten config value.
        keys.remove(ds_args)
        keys.insert(0, ds_args)

    for key in keys:
      if key in config_overwrites:
        value = config_overwrites[key]
      else:
        value = old_orig_config[key]

      if key == "learning_rate":
        if not self.learning_rate_control:
          print("No lr control, ignore learning rate %r for epoch %i" % (value, epoch), file=log.v3)
          continue
        old_lr = self.learning_rate_control.get_learning_rate_for_epoch(epoch)
        print("Overwrite learning rate for epoch %i: %r -> %r" % (epoch, old_lr, value), file=log.v3)
        assert self.config.is_true("use_learning_rate_control_always") or not self.pretrain
        self.learning_rate_control.epoch_data[epoch].learning_rate = value
        continue

      if key in old_orig_config:
        orig_value = old_orig_config.pop(key)
      else:
        assert key in self.config.typed_dict, "config update key %r -> %r expected to be in orig. config" % (key, value)
        orig_value = self.config.typed_dict[key]
      if key in config_overwrites:
        print("Update config key %r for epoch %i: %r -> %r" % (key, epoch, orig_value, value), file=log.v3)
        self.orig_config[key] = orig_value
      set_value(key, value)

    for dataset in updated_datasets.values():
      dataset.init_seq_order(epoch=epoch)

  def _init_network(self, net_desc, epoch=None):
    """
    :param dict[str,dict[str]] net_desc: layer name -> layer description dict
    :param int|None epoch: if not given, uses self.epoch. used for the random seed
    """
    if epoch is None:
      epoch = self.epoch
    self._close_tf_session()
    self._reset_graph()
    # The new session will by default use the newly created default graph.
    self.make_tf_session()
    tf_random_seed = 42
    net_random_seed = epoch
    if self.config.opt_typed_value("random_seed", None):
      seed = self.config.int("random_seed", None)
      net_random_seed = (epoch * 3 + seed * 5 + 7) % (2 ** 31)
      tf_random_seed = (net_random_seed * 2 + 3) % (2 ** 31)
    tf_compat.v1.set_random_seed(tf_random_seed)
    from returnn.tf.util.basic import get_global_train_flag_placeholder
    if self.use_dynamic_train_flag:
      train_flag = get_global_train_flag_placeholder()
    else:
      train_flag = False
    use_dataset_pipeline = False
    if self.config.is_true("dataset_pipeline"):
      use_dataset_pipeline = True
    extern_data = ExternData()
    extern_data.init_from_config(config=self.config, auto_create_placeholders=not use_dataset_pipeline)
    if use_dataset_pipeline:
      datasets = self.eval_datasets.copy()
      if self.train_data:
        datasets["train"] = self.train_data
      self.dataset_provider = DatasetDataProvider(extern_data=extern_data, datasets=datasets, config=self.config)
    self.network, self.updater = self.create_network(
      config=self.config,
      extern_data=extern_data,
      rnd_seed=net_random_seed,
      train_flag=train_flag, eval_flag=self.use_eval_flag, search_flag=self.use_search_flag,
      initial_learning_rate=getattr(self, "initial_learning_rate", None),
      net_dict=net_desc)
    self.network.initialize_params(session=self.tf_session)
    if self.config.is_true("use_horovod"):
      # Note: Might not be needed as it should be deterministic. But just to be sure...
      # noinspection PyPackageRequirements,PyUnresolvedReferences
      import horovod.tensorflow as hvd
      # like hvd.broadcast_global_variables but selected vars only:
      bcast_op = tf.group(*[
        tf_compat.v1.assign(var, hvd.broadcast(var, root_rank=0))
        for var in self.network.get_params_list() + self.network.get_auxiliary_params()])
      self.tf_session.run(bcast_op)

  @classmethod
  def create_network(cls, config, rnd_seed, train_flag, eval_flag, search_flag, net_dict,
                     extern_data=None, initial_learning_rate=1.0):
    """
    :param Config.Config config:
    :param int rnd_seed:
    :param bool|tf.Tensor train_flag:
    :param float initial_learning_rate:
    :param bool eval_flag:
    :param bool search_flag:
    :param ExternData|None extern_data:
    :param dict[str,dict[str]] net_dict:
    :return: network, updater
    :rtype: (TFNetwork, Updater|None)
    """
    network = TFNetwork(
      name="",
      config=config,
      extern_data=extern_data,
      rnd_seed=rnd_seed,
      train_flag=train_flag,
      eval_flag=eval_flag,
      search_flag=search_flag)
    network.construct_from_dict(net_dict)
    if train_flag is not False and config.list("search_train_network_layers"):
      network.construct_extra_net(
        net_dict, layer_list=config.list("search_train_network_layers"), search_flag=True,
        dep_layers_in_extra=True,
        net_name="search train extra net")
    updater = None
    if train_flag is not False:
      # Need to create new Updater because it has the learning_rate var which must be in the current graph.
      updater = Updater(
        config=config, network=network,
        initial_learning_rate=initial_learning_rate)
      updater.set_trainable_vars(network.get_trainable_params())
    network.print_network_info()
    return network, updater

  def need_init_new_network(self, net_desc=None):
    """
    :param dict[str,dict[str]]|None net_desc: layer name -> layer description dict
    :rtype: bool
    """
    if self.config.is_true("reinit_network_each_epoch"):
      return True
    if net_desc is None:
      return False
    return self.network.layers_desc != net_desc

  def init_new_network(self, net_desc=None):
    """
    Reinitializes the network, and copies over the parameter from the current network.

    :param dict[str,dict[str]]|None net_desc: layer name -> layer description dict. use existing by default
    """
    assert self.network
    if net_desc is None:
      net_desc = self.network.layers_desc
      print("reinit network", file=log.v3)
    else:
      from returnn.util.basic import dict_diff_str
      print("reinit because network description differs. Diff:",
            dict_diff_str(self.network.layers_desc, net_desc), file=log.v3)
    old_network_params = self.network.get_params_serialized(self.tf_session)
    self._init_network(net_desc)
    if self.is_pretrain_epoch() and not self.pretrain.copy_output_layer:
      # "ifpossible" logic handled below. copy_output_layer=True is currently not enforced.
      for layer in self.network.get_output_layers():
        if layer.name in old_network_params.values_dict:
          print("suspend copying of output layer: " + layer.name, file=log.v2)
          old_network_params.values_dict.pop(layer.name)
    # Optionally call some callback from config.
    # Do this before we call network.set_params_by_serialized, to allow to remove entries from old_network_params.
    if self.config.has("init_new_network_callback"):
      self.config.typed_value("init_new_network_callback")(
        session=self.tf_session, new_epoch=self.epoch,
        new_network=self.network, old_network_params=old_network_params)
    # This copy will copy the old params over and leave the rest randomly initialized.
    # This also works if the old network has just the same topology,
    # e.g. if it is the initial model from self.init_network_from_config().
    # In pretraining it can happen, that the dimension of output parameters of the previous epoch is
    # not equal to the dimension in the current epoch, due to difference in layer size.
    # In that case initialize output parameters randomly.
    self.network.set_params_by_serialized(
      old_network_params, session=self.tf_session,
      ignore_wrong_shape=self.is_pretrain_epoch() or self.custom_get_net_dict,
      copy_param_mode=net_desc.get(
        "#copy_param_mode", self.pretrain.copy_param_mode if self.is_pretrain_epoch() else None),
      ignore_non_existing=self.is_pretrain_epoch() or self.custom_get_net_dict)

  def train(self):
    """
    Does the whole training, i.e. the loop over all the epochs.
    """
    print("start training at epoch %i and step %i" % (self.start_epoch, self.start_batch), file=log.v3)
    print("using batch size: %r, max seqs: %i" % (self.batch_size, self.max_seqs), file=log.v4)
    print("learning rate control:", self.learning_rate_control, file=log.v4)
    print("pretrain:", self.pretrain, file=log.v4)
    self.dataset_batches.clear()

    assert self.start_epoch >= 1, "Epochs start at 1."
    final_epoch = self.final_epoch if self.final_epoch != 0 else sys.maxsize
    if self.start_epoch > final_epoch:
      print("No epochs to train, start_epoch: %i, final_epoch: %i" %
            (self.start_epoch, self.final_epoch), file=log.v1)

    self.check_last_epoch()
    if isinstance(self.max_seq_length, (int, float)):
      self.max_seq_length += (self.start_epoch - 1) * self.inc_seq_length

    assert isinstance(self.start_epoch, int)
    epoch = self.start_epoch  # Epochs start at 1.
    while epoch <= final_epoch:
      self.epoch = epoch  # type: int
      if isinstance(self.max_seq_length, int) and self.max_seq_length != sys.maxsize:
        if int(self.max_seq_length + self.inc_seq_length) != int(self.max_seq_length):
          print("increasing sequence lengths to", int(self.max_seq_length + self.inc_seq_length), file=log.v3)
          self.dataset_batches.pop("train", None)
          self.max_seq_length += self.inc_seq_length
      if self.epoch % self.seq_drop_freq == 0:
        if self.seq_drop > 0.0:
          self.dataset_batches.pop("train", None)
      # In case of random seq ordering, we want to reorder each epoch.
      if self.train_data.init_seq_order(epoch=self.epoch):
        self.dataset_batches.pop("train", None)
      for dataset_name, dataset in self.get_eval_datasets().items():
        if dataset.init_seq_order(epoch=self.epoch):
          self.dataset_batches.pop(dataset_name, None)

      self.init_train_epoch()
      self.train_epoch()
      epoch += 1

    if self.start_epoch <= self.final_epoch:  # We did train at least one epoch.
      assert self.epoch
      # Save last model, in case it was not saved yet (depends on save_model_epoch_interval).
      if self.model_filename:
        self.save_model(self.get_epoch_model_filename())

      if self.epoch != self.final_epoch:
        print("Stopped after epoch %i and not %i as planned." % (self.epoch, self.final_epoch), file=log.v3)

    print("Finished training in epoch %i." % self.epoch, file=log.v3)  # noqa

  def init_train_epoch(self):
    """
    Init for the current train epoch.
    """
    if self.is_pretrain_epoch() or self.custom_get_net_dict:
      new_network_desc = self.get_net_dict_for_epoch(epoch=self.epoch)
      # Always update config, if needed, even if nothing changed.
      # This might trigger enforcing some learning rate, or so.
      self._maybe_update_config(net_desc=new_network_desc, epoch=self.epoch)
      if self.need_init_new_network(new_network_desc):
        self.init_new_network(new_network_desc)
      if self.is_pretrain_epoch():
        self.network.declare_train_params(**self.pretrain.get_train_param_args_for_epoch(self.epoch))
    else:
      if self.need_init_new_network():
        self.init_new_network()
    if self.config.is_true("use_learning_rate_control_always"):
      self.learning_rate = self.learning_rate_control.get_learning_rate_for_epoch(self.epoch)
    elif self.is_pretrain_epoch():
      # Use constant learning rate.
      self.learning_rate = self.pretrain_learning_rate
      self.learning_rate_control.set_default_learning_rate_for_epoch(self.epoch, self.learning_rate)
    elif self.is_first_epoch_after_pretrain():
      # Use constant learning rate.
      self.learning_rate = self.initial_learning_rate
      self.learning_rate_control.set_default_learning_rate_for_epoch(self.epoch, self.learning_rate)
    else:
      self.learning_rate = self.learning_rate_control.get_learning_rate_for_epoch(self.epoch)

    if not self.is_pretrain_epoch():
      # Train the whole network.
      self.network.declare_train_params()

    self.updater.set_trainable_vars(self.network.get_trainable_params())

    self._maybe_use_better_last_model()

  def _maybe_use_better_last_model(self):
    if not self.config.is_true("use_last_best_model"):
      return
    if self.is_pretrain_epoch():
      return
    opts = self.config.get_of_type("use_last_best_model", dict, default={}).copy()
    if self.epoch % opts.pop("modulo", 1) != 0:
      # Normally we would filter those out. One maybe sensible exception is if the last score was really bad.
      if (self.learning_rate_control.get_epoch_error_value(self.epoch - 1) or 0) \
           <= opts.get("filter_score", float("inf")):
        return
    # Check if the previous epoch model is the best and otherwise take the best last model params.
    last_best_epoch = self.learning_rate_control.get_last_best_epoch(
      last_epoch=self.epoch - 1,
      first_epoch=self.pretrain.get_train_num_epochs() if self.pretrain else 1,
      **opts)
    if last_best_epoch and last_best_epoch != self.epoch - 1:
      print("Last epoch %i (score: %f) is not the optimal model" %
            (self.epoch - 1, self.learning_rate_control.get_epoch_error_value(self.epoch - 1))
            + " but epoch %i has better score %f (%r), will use that model." %
            (last_best_epoch, self.learning_rate_control.get_epoch_error_value(last_best_epoch),
             self.learning_rate_control.get_epoch_error_dict(last_best_epoch)),
            file=log.v2)
      self.load_model(epoch=last_best_epoch)
      self.updater.init_optimizer_vars(session=self.tf_session)  # reset the optimizer vars

  def train_epoch(self):
    """
    Train a single epoch (self.epoch).
    """
    print("start", self.get_epoch_str(), "with learning rate", self.learning_rate, "...", file=log.v4)

    if self.epoch == 1 and self.save_epoch1_initial_model:
      epoch0_model_filename = self.epoch_model_filename(self.model_filename, 0, self.is_pretrain_epoch())
      print("save initial epoch1 model", epoch0_model_filename, file=log.v4)
      self.save_model(epoch0_model_filename)

    if 'train' not in self.dataset_batches or not self.train_data.batch_set_generator_cache_whole_epoch():
      self.dataset_batches['train'] = self.train_data.generate_batches(
        recurrent_net=self.network.recurrent,
        batch_size=self.batch_size,
        max_seqs=self.max_seqs,
        max_seq_length=self.max_seq_length,
        min_seq_length=self.min_seq_length,
        max_pad_size=self.max_pad_size,
        seq_drop=self.seq_drop,
        shuffle_batches=self.shuffle_batches,
        used_data_keys=self.network.get_used_data_keys())
    else:
      print("reusing previous dataset batch order for 'train' dataset", file=log.v4)
      self.dataset_batches['train'].reset()
    train_batches = self.dataset_batches['train']

    self.updater.set_learning_rate(self.learning_rate, session=self.tf_session)
    trainer = Runner(
      engine=self,
      dataset_name="train", dataset=self.train_data, batches=train_batches,
      train=self.network.layers_desc.get("#trainable", True))
    trainer.run(report_prefix=("pre" if self.is_pretrain_epoch() else "") + "train epoch %s" % self.epoch)

    if not trainer.finalized:
      if trainer.device_crash_batch is not None:  # Otherwise we got an unexpected exception - a bug in our code.
        if self.model_filename:
          self.save_model(self.get_epoch_model_filename() + ".crash_%i" % trainer.device_crash_batch)
      print("Trainer not finalized, quitting. (pid %i)" % os.getpid(), file=log.v1)
      trainer.exit_due_to_error()

    if any(numpy.isinf(list(trainer.score.values()))) or any(numpy.isnan(list(trainer.score.values()))):
      print("Model seems broken, got inf or nan final score: %s" % trainer.score, file=log.v1)
      if self.config.bool("stop_on_nonfinite_train_score", True):
        if self.model_filename:
          self.save_model(self.get_epoch_model_filename() + ".broken")
        sys.exit(1)

    should_call_graph_reset_callbacks = False
    should_save_model_after_eval = False
    if (self.network.get_graph_reset_callbacks() and
        # See also init_train_epoch().
        self.need_init_new_network(
          net_desc=(
            self.get_net_dict_for_epoch(epoch=self.epoch + 1)
            if (self.is_pretrain_epoch(epoch=self.epoch + 1) or self.custom_get_net_dict)
            else None))):
      # Do not call it right now, but after eval model. See below.
      should_call_graph_reset_callbacks = True
      # In case that Returnn crashes now during eval (e.g. job timelimit exceeded),
      # e.g. some HDF dump would be incomplete, but Returnn would load the saved model
      # and continue with the next epoch anyway. We want to avoid this incompleteness.
      should_save_model_after_eval = True

    if self.model_filename and (self.epoch % self.save_model_epoch_interval == 0):
      if not should_save_model_after_eval:
        self.save_model(self.get_epoch_model_filename())
    else:
      should_save_model_after_eval = False
    self.learning_rate_control.set_epoch_error(self.epoch, {"train_score": trainer.score, "train_error": trainer.error})
    if self._do_save():
      self.learning_rate_control.save()

    print(
      self.get_epoch_str(),
      "score:", self.format_score(trainer.score),
      "error:", self.format_score(trainer.error),
      "elapsed:", hms(trainer.elapsed), file=log.v1)
    self.eval_model()

    if should_call_graph_reset_callbacks:
      # Early call of reset callbacks, which might trigger some HDF dump or other things.
      # Do this after eval model, such that e.g. the HDF dump contains information about both train and dev.
      self.network.call_graph_reset_callbacks()
    if should_save_model_after_eval:
      self.save_model(self.get_epoch_model_filename())

    if self.config.bool_or_other("cleanup_old_models", None):
      self.cleanup_old_models()

  # noinspection PyMethodMayBeStatic
  def format_score(self, score):
    """
    :param dict[str,float] score:
    :return: score(s) as str
    :rtype: str
    """
    if not score:
      return "None"
    if len(score) == 1:
      return str(list(score.values())[0])
    return " ".join(["%s %s" % (key.split(':', 2)[-1], str(score[key]))
                     for key in sorted(score.keys())])

  def _maybe_prepare_train_in_eval(self, targets_via_search=False):
    """
    :param bool targets_via_search:
    :return: whether train in eval should be used
    :rtype: bool
    """
    if not self.config.get_of_type("train_in_eval", bool, False):
      return False
    if targets_via_search:
      # TODO. This will require a new network.
      # TFNetwork construct_extra_net also does not quite work for this.
      # We need to create a new net, where we set the search as the targets.
      raise NotImplementedError
    # We update the model params in-place.
    # In training, we don't want that, because it should not use the validation data.
    # We could reset it later when continuing the training, but it's not implemented.
    assert self.config.value('task', 'train') != 'train', (
      "task %r should be just 'eval' or so. training will break." % self.config.value('task', None))
    if not self.updater:
      self.updater = Updater(
        config=self.config, network=self.network,
        initial_learning_rate=self.initial_learning_rate)
      self.updater.set_trainable_vars(self.network.get_trainable_params())
      self.updater.init_optimizer_vars(session=self.tf_session)
    eval_learning_rate = self.config.get_of_type(
      'eval_learning_rate', float, default=self.config.float('learning_rate', 1.0))
    print("train in eval, learning rate %f" % eval_learning_rate, file=log.v2)
    self.updater.set_learning_rate(eval_learning_rate, session=self.tf_session)
    return True

  def _do_save(self):
    """
    :return: whether to perform save on disk in this process. e.g. for Horovod rank != 0, do not save.
    :rtype: bool
    """
    import returnn.util.basic
    return returnn.util.basic.should_write_to_disk(config=self.config)

  def _is_dataset_evaluated(self, name):
    """
    Check via self.learning_rate_control.

    :param str name:
    :rtype: bool
    """
    assert self.learning_rate_control.filename  # otherwise we would not have stored it
    error_dict = self.learning_rate_control.get_epoch_error_dict(self.epoch)
    if not error_dict:
      return False
    return any([k.startswith("%s_score" % name) for k in error_dict.keys()])

  def eval_model(self, output_file=None, output_per_seq_file=None, loss_name=None,
                 output_per_seq_format=None, output_per_seq_file_format="txt", skip_already_evaluated=False,
                 lr_control_update_scores=True):
    """
    Eval the current model on the eval datasets (dev + eval, whatever is set).
    See also :func:`self.search` for performing beam search.

    :param str|None output_file: if given, will save the results to this file (total err/score for each dataset)
    :param str|None output_per_seq_file: if given, will save the err/score for each sequence
    :param str|None loss_name: specifies the loss which will be written to output_file
    :param list[str]|tuple[str]|None output_per_seq_format:
      which properties of `loss_name` should be written to `output_per_seq_file`.
      allowed_outputs = {"seq_tag", "seq_len", "score", "error", "pos_score", "pos_error"}.
    :param bool skip_already_evaluated:
    :param str output_per_seq_file_format: "txt" or "py"
    :param bool lr_control_update_scores: update and save scores in learning rate control
    :return: nothing
    """
    extra_fetches = None

    if output_per_seq_file:
      assert output_per_seq_file_format in {"txt", "py"}
      allowed_outputs = {"seq_tag", "seq_len", "score", "error", "pos_score", "pos_error"}

      assert isinstance(output_per_seq_format, (tuple, list)), "provide output_per_seq_format"
      assert set(output_per_seq_format) - allowed_outputs == set(), (
        "Only %r are allowed in function eval_model as output_per_seq_format, but got: %r " % (
          allowed_outputs, output_per_seq_format))

      # always fetch seq_tag to map loss values to the corresponding line
      extra_fetches = {"seq_idx": self.network.get_extern_data("seq_idx", mark_data_key_as_used=True),
                       "seq_tags": self.network.get_seq_tags(mark_data_key_as_used=True)}

      from returnn.tf.util.basic import identity
      losses_dict, _, _ = self.network.get_losses_initialized(reduce_func=identity, with_total=False)
      assert loss_name in losses_dict, (
        "Unknown loss defined. Got %r. Possible losses are %r" % (loss_name, losses_dict.keys()))

      loss_holder = losses_dict[loss_name]
      # enforce reinit, otherwise the new value of 'loss_holder.loss.use_flatten_frames' will be ignored
      loss_holder.loss.layer = None
      loss_holder.loss.use_flatten_frames = False  # we need that such that we get (B*T,...) unreduced values

      # we need sequence lengths for positional fetches
      has_positional_fetch = ("pos_score" in output_per_seq_format) or ("pos_error" in output_per_seq_format)

      if "seq_len" in output_per_seq_format or has_positional_fetch:
        extra_fetches["seq_len"] = loss_holder.loss.output.get_sequence_lengths()
      if "score" in output_per_seq_format:
        extra_fetches["score"] = loss_holder.get_normalized_loss_value_per_seq()
      if "error" in output_per_seq_format:
        extra_fetches["error"] = loss_holder.get_normalized_error_value_per_seq()
      if "pos_score" in output_per_seq_format:
        extra_fetches["pos_score"] = loss_holder.get_loss_value_per_pos()
      if "pos_error" in output_per_seq_format:
        extra_fetches["pos_error"] = loss_holder.get_error_value_per_pos()

    seq_idx_to_tag = {}  # type: typing.Dict[int,str]  # we need this in order to write the results in the correct order later  # nopep8
    results_per_seq = {}  # type: typing.Dict[str,typing.Dict[str,typing.Union[float,str,int]]]  # seq_tag -> dict. Results of fetches will be written in this dict  # nopep8

    # function to save the return values of each callback to the dict `results_per_seq`
    # noinspection PyShadowingNames
    def extra_fetches_callback(seq_idx, seq_tags, **extra_fetches_out):
      """
      :param list[int] seq_idx:
      :param list[str] seq_tags:
      :param dict[str,numpy.ndarray] extra_fetches_out: see extra_fetches
      """

      for batch_idx in range(len(seq_idx)):
        corpus_seq_idx = dataset.get_corpus_seq_idx(seq_idx[batch_idx])
        seq_idx_to_tag[corpus_seq_idx] = seq_tags[batch_idx]

      for name, value in extra_fetches_out.items():
        assert name in allowed_outputs
        assert isinstance(value, numpy.ndarray)

        # in case of positional values, we have to handle a 2-dim ndarray
        if name[:4] == "pos_":
          assert 'seq_len' in extra_fetches_out
          seq_lens = extra_fetches_out['seq_len']
          shorted_scores = [ps[:l] for ps, l in zip(value, seq_lens)]
          for i, seq_tag_ in enumerate(seq_tags):
            results_per_seq.setdefault(seq_tag_, {'seq_tag': seq_tag_})[name] = shorted_scores[i]
        else:
          for i, seq_tag_ in enumerate(seq_tags):
            results_per_seq.setdefault(seq_tag_, {'seq_tag': seq_tag_})[name] = value[i]

    # It's constructed lazily and it will set used_data_keys, so make sure that we have it now.
    self.network.maybe_construct_objective()
    results = {}
    eval_dump_str = []
    train = self._maybe_prepare_train_in_eval()
    train_flag = self.config.bool("eval_use_train_flag", None)

    if output_per_seq_file:
      assert len(self.get_eval_datasets()) == 1, (
        ("output per sequence is only supported for one dataset (dev or eval),"
         "provided datasets are %r") % list(self.get_eval_datasets().keys()))
      # try to sort dataset to minimize zero-padding
      dataset = list(self.get_eval_datasets().values())[0]
      if dataset.have_corpus_seq_idx():
        # We can sort it. Sort it in reverse to make sure that we have enough memory right at the beginning.
        print("Dataset have_corpus_seq_idx == True, i.e. it will be sorted for optimal performance.", file=log.v3)
        dataset.seq_ordering = "sorted_reverse"
      else:
        print("Dataset have_corpus_seq_idx == False, i.e. it will not be sorted for optimal performance.", file=log.v3)
        dataset.seq_ordering = "default"  # enforce order as-is, so that the order in the written file corresponds
      dataset.init_seq_order(epoch=self.epoch)

    for dataset_name, dataset in self.get_eval_datasets().items():
      if skip_already_evaluated and self._is_dataset_evaluated(name=dataset_name):
        continue
      if dataset_name not in self.dataset_batches or not dataset.batch_set_generator_cache_whole_epoch():
        self.dataset_batches[dataset_name] = dataset.generate_batches(
          recurrent_net=self.network.recurrent,
          batch_size=self.batch_size,
          max_seqs=self.max_seqs,
          max_seq_length=(self.max_seq_length if dataset_name == 'dev' else sys.maxsize),
          used_data_keys=self.network.get_used_data_keys())
      else:
        print("reusing previous dataset batch order for %r dataset" % dataset_name, file=log.v4)
        self.dataset_batches[dataset_name].reset()
      tester = Runner(
        engine=self, dataset_name=dataset_name, dataset=dataset, batches=self.dataset_batches[dataset_name],
        train=train, train_flag=train_flag,
        extra_fetches=extra_fetches, extra_fetches_callback=extra_fetches_callback)
      tester.run(report_prefix=self.get_epoch_str() + " %r eval" % dataset_name)
      if not tester.finalized:
        print("Tester not finalized, quitting.", file=log.v1)
        tester.exit_due_to_error()
      eval_dump_str += ["%s: score %s error %s" % (
                        dataset_name, self.format_score(tester.score), self.format_score(tester.error))]
      results[dataset_name] = {"score": tester.score, "error": tester.error}
      if lr_control_update_scores:
        self.learning_rate_control.set_epoch_error(
          self.epoch, {"%s_score" % dataset_name: tester.score, "%s_error" % dataset_name: tester.error})
        if self._do_save():
          self.learning_rate_control.save()
    print(" ".join(eval_dump_str), file=log.v1)
    if output_file:
      print('Write eval results to %r' % output_file, file=log.v3)
      from returnn.util.basic import better_repr
      with open(output_file, 'w') as f:
        f.write(better_repr(results) + '\n')
    if output_per_seq_file:
      print('Write eval results per seq to %r' % output_per_seq_file, file=log.v3)
      with open(output_per_seq_file, 'w') as f:
        if output_per_seq_file_format == "txt":
          for seq_idx in range(len(results_per_seq)):
            seq_tag = seq_idx_to_tag[seq_idx]
            value_list = [results_per_seq[seq_tag][req_out] for req_out in output_per_seq_format]
            value_list = [' '.join(map(str, v)) if isinstance(v, numpy.ndarray) else str(v) for v in value_list]
            assert all([all([c not in "\n;" for c in v]) for v in value_list])
            res = ';'.join(value_list)
            f.write(res + '\n')
        elif output_per_seq_file_format == "py":
          f.write("{\n")
          for seq_idx in range(len(results_per_seq)):
            seq_tag = seq_idx_to_tag[seq_idx]
            f.write("%r: {" % seq_tag)
            with numpy.printoptions(threshold=sys.maxsize):
              f.write(", ".join([
                "%r: %r" % (req_out, results_per_seq[seq_tag][req_out]) for req_out in output_per_seq_format]))
            f.write("},\n")
          f.write("}\n")
        else:
          assert False, output_per_seq_file_format

  def check_last_epoch(self):
    """
    Checks if there are outstanding tasks (eval_model) for the last epoch,
    and executes them.
    """
    if self.start_epoch == 1:
      return
    # noinspection PyAttributeOutsideInit
    self.epoch = self.start_epoch - 1
    if self.learning_rate_control.filename:
      for name, dataset in self.get_eval_datasets().items():
        if not self._is_dataset_evaluated(name=name):
          # This can happen when we have a previous model but did not test it yet.
          print("Last epoch model not yet evaluated on dev. Doing that now.", file=log.v4)
          self.eval_model(skip_already_evaluated=True)
          break

  def cleanup_old_models(self, ask_for_confirmation=False):
    """
    :param bool ask_for_confirmation: if True, will ask the user interactively to confirm
    """
    if not self._do_save():
      return
    from returnn.util.basic import CollectionReadCheckCovered, human_bytes_size, confirm
    from itertools import count
    opts = CollectionReadCheckCovered(self.config.get_of_type("cleanup_old_models", dict, {}))
    existing_models = self.get_existing_models(config=self.config)
    if hasattr(self, "learning_rate_control"):
      lr_control = self.learning_rate_control
    else:
      lr_control = load_learning_rate_control_from_config(self.config)
    epochs = sorted(existing_models.keys())
    if not epochs:
      print("Cannot cleanup models, no models found.", file=log.v2)
      return
    keep_last_n = opts.get("keep_last_n", 2)
    keep_best_n = opts.get("keep_best_n", 4)
    assert keep_last_n >= 1 and keep_best_n >= 0
    if max(keep_last_n, keep_best_n) >= len(epochs):
      print(
        ("Only %i epochs stored so far and keeping last %i epochs and best %i epochs,"
         " thus not cleaning up any epochs yet.") % (
          len(epochs), keep_last_n, keep_best_n), file=log.v2)
      return
    keep_epochs = set()  # type: typing.Set[int]
    default_keep_pattern = set()
    if epochs[-1] <= 10:
      keep_every = 4
      keep_doubles_of = 5
    elif epochs[-1] <= 50:
      keep_every = 20
      keep_doubles_of = 5
    elif epochs[-1] <= 100:
      keep_every = 40
      keep_doubles_of = 10
    else:
      keep_every = 80
      keep_doubles_of = 20
    for i in count(1):
      n = keep_every * i
      if n > epochs[-1]:
        break
      default_keep_pattern.add(n)
    for i in count():
      n = keep_doubles_of * (2 ** i)
      if n > epochs[-1]:
        break
      default_keep_pattern.add(n)
    keep_epochs.update(opts.get("keep", default_keep_pattern))
    keep_epochs.update(epochs[-keep_last_n:])
    score_keys = set()  # e.g. "dev_error", "dev_score", etc.
    # Collect all possible score keys. Note that we could have different ones for different epochs.
    for data in lr_control.epoch_data.values():
      score_keys.update(data.error.keys())
    assert score_keys
    score_keys = sorted(score_keys)
    score_values = {key: [] for key in score_keys}
    for epoch in epochs:
      epoch_scores = lr_control.epoch_data[epoch].error
      for key in epoch_scores.keys():
        score_values[key].append(epoch_scores[key])
    for key in list(score_keys):
      scores = score_values[key]
      if min(scores) == max(scores):
        print("Ignoring score key %r because all epochs have the same value %r." % (key, scores[0]), file=log.v3)
        score_keys.remove(key)
        score_values.pop(key)
    # Actually, terminology is a bit confusing. We call it "score" here (and elsewhere), but it's a loss,
    # so the maximum value is the worst possible value.
    worst_score_values = {key: max(scores) for (key, scores) in score_values.items()}
    for key in score_keys:
      scores = sorted([
        (lr_control.epoch_data[epoch].error.get(key, worst_score_values[key]), epoch) for epoch in epochs])
      scores = scores[:keep_best_n]
      keep_epochs.update([v[1] for v in scores])
    keep_epochs.intersection_update(epochs)
    if len(keep_epochs) == len(epochs):
      print("%i epochs stored so far and keeping all." % len(epochs), file=log.v2)
      return
    remove_epochs = sorted(set(epochs).difference(keep_epochs))
    assert remove_epochs
    if len(epochs) > 6:
      epoch_summary = "[%s, ..., %s]" % (", ".join(map(str, epochs[:3])), ", ".join(map(str, epochs[-3:])))
    else:
      epoch_summary = str(epochs)
    print("We have stored models for epochs %s and keep epochs %s." % (epoch_summary, sorted(keep_epochs)), file=log.v3)
    print("We will delete the models of epochs %s." % (remove_epochs,), file=log.v3)
    opts.assert_all_read()
    if self.config.bool("dry_run", False):
      print("Dry-run, will not delete models.", file=log.v2)
      return
    if ask_for_confirmation:
      confirm("Delete those models?", exit_on_false=True)
    count_bytes = 0
    for epoch in remove_epochs:
      count_bytes += self.delete_model(existing_models[epoch])
    print("Deleted %s." % human_bytes_size(count_bytes), file=log.v2)

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
    with tf.name_scope("check_uninitialized_vars"), self.tf_session.graph.as_default():
      # Like tf.report_uninitialized_variables().
      var_list = tf_compat.v1.global_variables() + tf_compat.v1.local_variables()
      if not var_list:
        return
      # Get a 1-D boolean tensor listing whether each variable is initialized.
      var_mask = tf.logical_not(tf.stack(
        [tf_compat.v1.is_variable_initialized(v) for v in var_list])).eval(session=self.tf_session)
      assert len(var_mask) == len(var_list)
      uninitialized_vars = [v for (v, mask) in zip(var_list, var_mask) if mask]
      if uninitialized_vars:
        print("Note: There are still these uninitialized variables: %s" % [v.name for v in uninitialized_vars],
              file=log.v3)
        self.tf_session.run(tf_compat.v1.variables_initializer(uninitialized_vars))
      self._checked_uninitialized_vars = True

  def _get_data_provider(self, dataset_name=None, dataset=None, batches=None, feed_dict=None):
    """
    :param str|None dataset_name:
    :param Dataset.Dataset|None dataset:
    :param BatchSetGenerator|None batches:
    :param bool|None feed_dict:
    :rtype: FeedDictDataProvider|DatasetDataProvider
    """
    if self.dataset_provider and feed_dict is not True and dataset_name:
      self.dataset_provider.set_current_dataset(dataset_name=dataset_name)
      return self.dataset_provider
    else:
      if self.dataset_provider and feed_dict is not False:
        print("WARNING: dataset_provider is set (via dataset_pipeline) but not used", file=log.v2)
      batch_slice = None
      if tf_horovod.get_ctx() and tf_horovod.get_ctx().is_dataset_distribution_shard():
        batch_slice = tf_horovod.get_ctx().get_dataset_shard_batch_slice()
      data_provider = FeedDictDataProvider(
        tf_session=self.tf_session, extern_data=self.network.extern_data,
        data_keys=self.network.get_used_data_keys(),
        dataset=dataset, batches=batches,
        batch_slice=batch_slice,
        enforce_min_len1=self.config.is_true("enforce_min_len1", False))
      return data_provider

  def get_specific_feed_dict(self, dataset, seq_idx):
    """
    :param Dataset.Dataset dataset:
    :param int seq_idx: index of sequence, -1 for all sequences in dataset
    :return: feed_dict for self.tf_session.run()
    :rtype: dict[tf.Tensor,numpy.ndarray]
    """
    # No Runner instance here but a very simplified version of Runner.run().
    # First we need a custom DataProvider with a custom BatchSetGenerator
    # which will yield only one single batch for the provided sequence idx.
    batch = Batch()
    if seq_idx == -1:  # load all sequences in dataset
      for seq_idx_loop in range(dataset.num_seqs):
        batch.add_sequence_as_slice(
          seq_idx=seq_idx_loop, seq_start_frame=0, length=dataset.get_seq_length(seq_idx_loop))
    else:
      batch.init_with_one_full_sequence(seq_idx=seq_idx, dataset=dataset)
    batch_generator = iter([batch])
    batches = BatchSetGenerator(dataset, generator=batch_generator)
    data_provider = self._get_data_provider(dataset=dataset, batches=batches, feed_dict=True)
    feed_dict, _ = data_provider.get_feed_dict(single_threaded=True)
    if isinstance(self.network.train_flag, tf.Tensor):
      feed_dict[self.network.train_flag] = False
    if isinstance(self.network.epoch_step, tf.Tensor):
      feed_dict[self.network.epoch_step] = 0
    return feed_dict

  def run_single(self, dataset, seq_idx, output_dict, ext_feed_dict=None):
    """
    :param Dataset dataset:
    :param int seq_idx: index of sequence, -1 for all sequences in dataset
    :param dict[str,tf.Tensor] output_dict: key -> tf.Tensor
    :param dict[tf.Tensor,numpy.ndarray] ext_feed_dict:
    :return: output_dict but values evaluated
    :rtype: dict[str,numpy.ndarray]
    """
    feed_dict = self.get_specific_feed_dict(dataset=dataset, seq_idx=seq_idx)
    if ext_feed_dict:
      feed_dict.update(ext_feed_dict)
    self.check_uninitialized_vars()  # Maybe some new uninitialized vars. Last check.
    none_output_values = {k: v for (k, v) in output_dict.items() if v is None}
    output_dict = {k: v for (k, v) in output_dict.items() if v is not None}
    output_values = self.tf_session.run(output_dict, feed_dict=feed_dict)
    output_values.update(none_output_values)
    return output_values

  def _get_output_layer(self, output_layer_name=None):
    """
    :param str|None output_layer_name: e.g. "output". if not set, will read from config "forward_output_layer"
    :rtype: returnn.tf.layers.base.LayerBase
    """
    if not output_layer_name:
      output_layer_name = self.config.value("forward_output_layer", self.network.get_default_output_layer_name())
      assert output_layer_name, "output layer not defined. set forward_output_layer in config"
    assert output_layer_name in self.network.layers, "output layer %r not found, available layers: %s" % (
      output_layer_name, ','.join(self.network.layers.keys()))
    return self.network.layers[output_layer_name]

  def forward_single(self, dataset, seq_idx, output_layer_name=None):
    """
    Forwards a single sequence.
    If you want to perform search, and get a number of hyps out, use :func:`search_single`.

    :param Dataset.Dataset dataset:
    :param int seq_idx:
    :param str|None output_layer_name: e.g. "output". if not set, will read from config "forward_output_layer"
    :return: numpy array, output in time major format (time,dim)
    :rtype: numpy.ndarray
    """
    output_data = self._get_output_layer(output_layer_name).output
    out = output_data.get_placeholder_as_time_major()
    out_d = self.run_single(dataset=dataset, seq_idx=seq_idx, output_dict={"out": out})
    output_value = out_d["out"]
    assert output_value.shape[1] == 1  # batch-dim
    return output_value[:, 0]  # remove batch-dim

  # noinspection PyUnusedLocal
  def forward_to_hdf(self, data, output_file, combine_labels='', batch_size=0, output_layer=None):
    """
    Is aiming at recreating the same interface and output as :func:`Engine.forward_to_hdf`.
    See also :func:`EngineTask.HDFForwardTaskThread` and :func:`hdf_dump_from_dataset` in the hdf_dump.py tool.

    :param Dataset data:
    :param str output_file:
    :param str combine_labels: ignored at the moment
    :param int batch_size:
    :param LayerBase output_layer:
    """
    from returnn.datasets.hdf import SimpleHDFWriter

    if not output_layer:
      output_layer = self._get_output_layer()
    output = output_layer.output.copy_as_batch_spatial_major()
    # Note: We kind of assume that the target, and its class labels, matches this output layer.
    # Of course this is not true in general.
    # Instead of introducing some more involved logic here, we just check whether the dim matches.
    target = self.network.get_default_target()
    labels = data.labels.get(target, None)
    del target
    if labels:
      if len(labels) != output.dim:
        labels = None

    assert output_file
    print("Forwarding to HDF file: %s" % output_file, file=log.v2)
    if self.config.is_true("forward_override_hdf_output"):
      if os.path.exists(output_file):
        print("HDF file exists, delete now (forward_override_hdf_output).", file=log.v2)
        os.remove(output_file)
    else:
      assert not os.path.exists(output_file)
    print("Forward output:", output, file=log.v3)
    writer = SimpleHDFWriter(filename=output_file, dim=output.dim, ndim=output.ndim, labels=labels)

    def extra_fetches_cb(inputs, seq_tag, **kwargs):
      """
      Insert each batch into the output_file (hdf).

      :param numpy.ndarray inputs: shape=(n_batch,time,data) (or whatever the output layer is...)
      :param list[str] seq_tag: sequence tags of length n_batch
      :param kwargs: e.g. seq_len_i (list[int])
      """
      n_batch = len(seq_tag)
      assert n_batch == inputs.shape[0]
      # noinspection PyShadowingNames
      seq_len = {i: kwargs["seq_len_%i" % i] for i in output.size_placeholder.keys()}
      assert all([len(v) == n_batch for v in seq_len.values()])
      writer.insert_batch(inputs=inputs, seq_len=seq_len, seq_tag=seq_tag)

    extra_fetches = {
      'inputs': output.placeholder,
      "seq_tag": self.network.get_seq_tags(),
    }
    for i, seq_len in output.size_placeholder.items():
      extra_fetches["seq_len_%i" % i] = seq_len
    batches = data.generate_batches(
      recurrent_net=True,  # Using non-recurrent batch construction leads to incorrect seqLengths in the HDF
      batch_size=batch_size,
      max_seqs=self.max_seqs,
      used_data_keys=self.network.get_used_data_keys())
    forwarder = Runner(
      engine=self, dataset=data, batches=batches,
      train=False, eval=False,
      extra_fetches=extra_fetches,
      extra_fetches_callback=extra_fetches_cb)
    forwarder.run(report_prefix=self.get_epoch_str() + " forward")
    if not forwarder.finalized:
      print("Error happened. Exit now.")
      forwarder.exit_due_to_error()

    writer.close()

  # noinspection PyUnusedLocal
  def analyze(self, data, statistics):
    """
    :param Dataset.Dataset data:
    :param list[str]|None statistics: ignored at the moment
    :return: print everything to log.v1, and return the Runner instance to get access to all the stats
    :rtype: Runner
    """
    print("Analyze with network on %r." % data, file=log.v1)

    if "analyze" not in self.network.layers:
      from returnn.tf.layers.basic import FramewiseStatisticsLayer
      assert self.config.has("sil_label_idx")
      self.network.add_layer(
        name="analyze", layer_class=FramewiseStatisticsLayer,
        sil_label_idx=self.config.int("sil_label_idx", 0),
        sources=self.network.get_output_layers())

    # It's constructed lazily and it will set used_data_keys, so make sure that we have it now.
    self.network.maybe_construct_objective()

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
      used_data_keys=self.network.get_used_data_keys())
    analyzer = Runner(engine=self, dataset=data, batches=batches, train=False)
    analyzer.run(report_prefix=self.get_epoch_str() + " analyze")

    print("Finished analyzing of the dataset %r." % data, file=log.v1)
    print("elapsed:", hms(analyzer.elapsed), file=log.v1)
    print("num mini-batches:", analyzer.num_steps, file=log.v1)
    print("total num_frames:", analyzer.num_frames_accumulated, file=log.v1)
    print("score:", self.format_score(analyzer.score), file=log.v1)
    print("error:", self.format_score(analyzer.error), file=log.v1)
    for k, v in sorted(analyzer.stats.items()):
      if k.startswith("stats:"):
        print("%s:" % k, v, file=log.v1)
    print("That are all collected stats.", file=log.v1)

    if not analyzer.finalized:
      print("WARNING: Did not finished through the whole epoch.", file=log.v1)
      analyzer.exit_due_to_error()
    return analyzer

  def search(self, dataset, do_eval=True, output_layer_names="output", output_file=None, output_file_format="txt"):
    """
    :param Dataset dataset:
    :param bool do_eval: calculate errors and print reference. can only be done if we have the reference target
    :param str|list[str] output_layer_names:
    :param str output_file:
    :param str output_file_format: "txt" or "py"
    """
    print("Search with network on %r." % dataset, file=log.v1)
    if not self.use_search_flag or not self.network or self.use_dynamic_train_flag:
      self.use_search_flag = True
      # At the moment this is probably not intended to use search with train flag.
      # Also see LayerBase._post_init_output() about setting size_placeholder to the target seq len,
      # so you would have have_known_seq_len=True in the RecLayer, with the given target seq len.
      self.use_dynamic_train_flag = False
      if self.network:
        print("Reinit network with search flag.", file=log.v3)
      self.init_network_from_config(self.config)
    if do_eval:
      # It's constructed lazily and it will set used_data_keys, so make sure that we have it now.
      self.network.maybe_construct_objective()
    if output_file:
      if dataset.have_corpus_seq_idx():
        # We can sort it. Sort it in reverse to make sure that we have enough memory right at the beginning.
        print("Dataset have_corpus_seq_idx == True, i.e. it will be sorted for optimal performance.", file=log.v3)
        dataset.seq_ordering = "sorted_reverse"
      else:
        print("Dataset have_corpus_seq_idx == False, i.e. it will not be sorted for optimal performance.", file=log.v3)
        dataset.seq_ordering = "default"  # enforce order as-is, so that the order in the written file corresponds

    max_seq_length = self.config.typed_value('max_seq_length', None) or self.config.float('max_seq_length', 0)
    assert not max_seq_length, (
      "Set max_seq_length = 0 for search (i.e. no maximal length). We want to keep all source sentences.")

    dataset.init_seq_order(epoch=self.epoch)
    batches = dataset.generate_batches(
      recurrent_net=self.network.recurrent,
      batch_size=self.config.int('batch_size', 1),
      max_seqs=self.config.int('max_seqs', -1),
      max_seq_length=max_seq_length,
      used_data_keys=self.network.get_used_data_keys())

    output_is_dict = isinstance(output_layer_names, list)
    if not output_is_dict:
      output_layer_names = [output_layer_names]
    num_output_layers = len(output_layer_names)

    # Create lists with information about the output layers. All of length num_output_layers.
    output_layers = []  # type: typing.List[LayerBase]
    out_beam_sizes = []  # type: typing.List[typing.Optional[int]]
    output_layer_beam_scores = []  # type: typing.List[typing.Optional[tf.Tensor]]
    target_keys = []  # type: typing.List[typing.Optional[str]]

    for output_layer_name in output_layer_names:
      output_layer = self.network.layers[output_layer_name]
      output_layers.append(output_layer)
      out_beam = output_layer.output.beam
      if out_beam is None:
        print("Given output %r is after decision (no beam)." % output_layer, file=log.v1)
        output_layer_beam_scores.append(None)
      else:
        print("Given output %r has beam %s." % (output_layer, out_beam), file=log.v1)
        output_layer_beam_scores.append(output_layer.get_search_choices().beam_scores)
      out_beam_sizes.append(out_beam.beam_size if out_beam else None)
      target_key = output_layer.target
      if output_layer.output.sparse and not target_key:
        # Use the default target key for sparse outputs
        target_key = self.network.extern_data.default_target
      target_keys.append(target_key)

    out_cache = None
    seq_idx_to_tag = {}
    if output_file:
      assert output_file_format in {"txt", "py"}
      if output_is_dict:
        assert output_file_format == "py", "Text format not supported in the case of multiple output layers."
      assert not os.path.exists(output_file)
      print("Will write outputs to: %s" % output_file, file=log.v2)
      output_file = open(output_file, "w")
      # corpus-seq-idx -> str|list[(float,str)]|dict[str -> str|list[(float,str)]],
      # depending on output_is_dict and whether output is after decision
      out_cache = {}
    if not log.verbose[4]:
      print("Set log_verbosity to level 4 or higher to see seq info on stdout.", file=log.v2)

    def extra_fetches_callback(seq_idx, seq_tag, **kwargs):
      """
      :param list[int] seq_idx: of length batch (without beam)
      :param list[str] seq_tag: of length batch (without beam)

      In addition, for each output layer, we expect the following parameters in kwargs:
        list[numpy.ndarray] output_<layer name>
        list[numpy.ndarray] beam_scores_<layer name>
        list[numpy.ndarray] target_<target key>
      """

      outputs, beam_scores, targets = [], [], []
      # noinspection PyShadowingNames
      for output_layer_idx in range(num_output_layers):
        outputs.append(kwargs["output_" + output_layer_names[output_layer_idx]])
        beam_scores.append(kwargs["beam_scores_" + output_layer_names[output_layer_idx]])
        if do_eval and target_keys[output_layer_idx] is not None:
          targets.append(kwargs["target_" + target_keys[output_layer_idx]])
        else:
          targets.append(None)

      n_batch = len(seq_idx)  # without beam
      assert n_batch == len(seq_tag)

      # noinspection PyShadowingNames
      for output_layer_idx in range(num_output_layers):
        if beam_scores[output_layer_idx] is not None:
          assert beam_scores[output_layer_idx].shape == (n_batch, out_beam_sizes[output_layer_idx])

        assert n_batch * (out_beam_sizes[output_layer_idx] or 1) == len(outputs[output_layer_idx])
        if do_eval and targets[output_layer_idx] is not None:
          assert n_batch == len(targets[output_layer_idx])

        if output_layers[output_layer_idx].output.dim == 256 and output_layers[output_layer_idx].output.sparse:
          # Interpret output as bytes/utf8-string.
          outputs[output_layer_idx] = bytearray(outputs[output_layer_idx]).decode("utf8")

      # Create lists with serialized data. All of length num_output_layers.
      serialized_outputs = []  # type: typing.List[typing.Optional[typing.Union[str,numpy.ndarray]]]
      serialized_targets = []  # type: typing.List[typing.Optional[typing.Union[str,numpy.ndarray]]]
      # noinspection PyShadowingNames
      for output_layer_idx in range(num_output_layers):
        if output_layers[output_layer_idx].output.sparse:
          # Try to get the vocab corresponding to the sparse dim
          vocab = output_layers[output_layer_idx].output.sparse_dim.vocab
          if not vocab and target_keys[output_layer_idx]:
            vocab = self.network.get_extern_data(target_keys[output_layer_idx]).vocab
            if not vocab and target_keys[output_layer_idx] in dataset.labels:
              vocab = Vocabulary.create_vocab_from_labels(dataset.labels[target_keys[output_layer_idx]])

          if vocab is not None:
            layer_outputs = outputs[output_layer_idx]
            serialized_output = [
              vocab.get_seq_labels(output) if output.ndim == 1 else vocab.labels[output] for output in layer_outputs]
          else:
            serialized_output = None
            assert not output_file, "Unable to serialize sparse output of layer '%s'." % (
              output_layer_names[output_layer_idx])
        else:
          # Output dense layers as-is
          serialized_output = outputs[output_layer_idx]
        serialized_outputs.append(serialized_output)

        if target_keys[output_layer_idx] and do_eval:
          target_extern_data = self.network.get_extern_data(target_keys[output_layer_idx])
          if target_extern_data.sparse:
            vocab = target_extern_data.vocab
            if not vocab and target_keys[output_layer_idx] in dataset.labels:
              vocab = Vocabulary.create_vocab_from_labels(dataset.labels[target_keys[output_layer_idx]])

            if vocab is not None:
              serialized_target = [
                vocab.get_seq_labels(target) if target.ndim == 1 else vocab.labels[target]
                for target in targets[output_layer_idx]]
            else:
              serialized_target = None
              assert not output_file, "Unable to serialize sparse target '%s'." % (target_keys[output_layer_idx])
          else:
            serialized_target = targets[output_layer_idx]
        else:
          serialized_target = None
        serialized_targets.append(serialized_target)

      for batch_idx in range(len(seq_idx)):
        corpus_seq_idx = None
        if out_cache is not None:
          corpus_seq_idx = dataset.get_corpus_seq_idx(seq_idx[batch_idx])
          assert corpus_seq_idx not in out_cache
          seq_idx_to_tag[corpus_seq_idx] = seq_tag[batch_idx]
          if output_is_dict:
            out_cache[corpus_seq_idx] = {}

        # noinspection PyShadowingNames
        for output_layer_idx in range(num_output_layers):
          if out_beam_sizes[output_layer_idx] is None:
            print("seq_idx: %i, seq_tag: %r, output %r: %r" % (
              seq_idx[batch_idx], seq_tag[batch_idx],
              output_layer_names[output_layer_idx], outputs[output_layer_idx][batch_idx]), file=log.v4)
            out_idx = batch_idx
          else:
            beam_start_idx = batch_idx * out_beam_sizes[output_layer_idx]
            beam_end_idx = (batch_idx + 1) * out_beam_sizes[output_layer_idx]
            print("seq_idx: %i, seq_tag: %r, outputs %r: %r" % (
              seq_idx[batch_idx], seq_tag[batch_idx], output_layer_names[output_layer_idx],
              outputs[output_layer_idx][beam_start_idx:beam_end_idx]), file=log.v4)
            out_idx = batch_idx * out_beam_sizes[output_layer_idx]
          if serialized_outputs[output_layer_idx]:
            if serialized_targets[output_layer_idx]:
              print("  ref:", serialized_targets[output_layer_idx][batch_idx], file=log.v4)
            if out_beam_sizes[output_layer_idx] is None:
              print("  hyp:", serialized_outputs[output_layer_idx][out_idx],
                    file=log.v4)
            else:
              assert beam_scores[output_layer_idx] is not None
              for beam_idx in range(out_beam_sizes[output_layer_idx]):
                print(
                  "  hyp %i, score %f:" % (beam_idx, beam_scores[output_layer_idx][batch_idx][beam_idx]),
                  serialized_outputs[output_layer_idx][out_idx + beam_idx],
                  file=log.v4)

          if out_cache is not None:
            if out_beam_sizes[output_layer_idx] is None:
              out_data = serialized_outputs[output_layer_idx][out_idx]
            else:
              assert beam_scores[output_layer_idx] is not None
              out_data = [
                  (beam_scores[output_layer_idx][batch_idx][beam_idx],
                   serialized_outputs[output_layer_idx][out_idx + beam_idx])
                  for beam_idx in range(out_beam_sizes[output_layer_idx])]

            if output_is_dict:
              assert output_layer_names[output_layer_idx] not in out_cache[corpus_seq_idx]
              out_cache[corpus_seq_idx][output_layer_names[output_layer_idx]] = out_data
            else:
              assert corpus_seq_idx not in out_cache
              out_cache[corpus_seq_idx] = out_data

    train = self._maybe_prepare_train_in_eval(targets_via_search=True)

    extra_fetches = {
      "seq_idx": self.network.get_extern_data("seq_idx", mark_data_key_as_used=True),
      "seq_tag": self.network.get_extern_data("seq_tag", mark_data_key_as_used=True)}

    for output_layer_idx in range(num_output_layers):
      extra_fetches["output_" + output_layer_names[output_layer_idx]] = output_layers[output_layer_idx]
      extra_fetches["beam_scores_" + output_layer_names[output_layer_idx]] = output_layer_beam_scores[output_layer_idx]
      # We use target_keys[output_layer_idx] and not output_layer_names[output_layer_idx]
      # for the key to avoid fetching the same target multiple times.
      if do_eval and target_keys[output_layer_idx] is not None:
        extra_fetches["target_" + target_keys[output_layer_idx]] = self.network.get_extern_data(
          target_keys[output_layer_idx], mark_data_key_as_used=True)

    runner = Runner(
      engine=self, dataset=dataset, batches=batches, train=train, eval=do_eval,
      extra_fetches=extra_fetches,
      extra_fetches_callback=extra_fetches_callback)
    runner.run(report_prefix=self.get_epoch_str() + " search")
    if not runner.finalized:
      print("Error happened. Exit now.")
      runner.exit_due_to_error()
    print("Search done. Num steps %i, Final: score %s error %s" % (
      runner.num_steps, self.format_score(runner.score), self.format_score(runner.error)), file=log.v1)
    if output_file:
      assert out_cache
      assert 0 in out_cache
      assert len(out_cache) - 1 in out_cache
      if output_file_format == "txt":
        for i in range(len(out_cache)):
          output_file.write("%s\n" % out_cache[i])
      elif output_file_format == "py":
        from returnn.util.basic import better_repr
        output_file.write("{\n")
        with numpy.printoptions(threshold=sys.maxsize):
          for i in range(len(out_cache)):
            output_file.write("%r: %s,\n" % (seq_idx_to_tag[i], better_repr(out_cache[i])))
        output_file.write("}\n")
      else:
        raise Exception("invalid output_file_format %r" % output_file_format)
      output_file.close()

  def search_single(self, dataset, seq_idx, output_layer_name=None):
    """
    Performs search.
    See also :func:`forward_single`.

    :param Dataset.Dataset dataset:
    :param int seq_idx: index of sequence, -1 for all sequences in dataset
    :param str|None output_layer_name: e.g. "output". if not set, will read from config "search_output_layer"
    :return: list of score and numpy array, each numpy arry in format (time,dim)
    :rtype: list[(float,numpy.ndarray)]
    """
    output_layer_name = output_layer_name or self.config.value("search_output_layer", "output")
    output_layer = self.network.layers[output_layer_name]
    output_t = output_layer.output.get_placeholder_as_batch_major()
    output_seq_lens_t = output_layer.output.get_sequence_lengths()
    out_beam_size = output_layer.output.beam.beam_size
    output_layer_beam_scores_t = None
    if out_beam_size is None:
      print("Given output %r is after decision (no beam)." % output_layer, file=log.v4)
    else:
      print("Given output %r has beam size %i." % (output_layer, out_beam_size), file=log.v4)
      output_layer_beam_scores_t = output_layer.get_search_choices().beam_scores

    output_d = self.run_single(dataset=dataset, seq_idx=seq_idx, output_dict={
      "output": output_t,
      "seq_lens": output_seq_lens_t,
      "beam_scores": output_layer_beam_scores_t})
    output = output_d["output"]
    seq_lens = output_d["seq_lens"]
    beam_scores = output_d["beam_scores"]
    assert len(output) == len(seq_lens) == (out_beam_size or 1) * dataset.num_seqs
    if out_beam_size:
      assert beam_scores.shape == (dataset.num_seqs, out_beam_size)  # (batch,beam)

    results = []
    for i in range(len(output)):
      hyp_seq = output[i][:seq_lens[i]]
      # txt = " ".join(map(labels["classes"].__getitem__, output[i][:seq_lens[i]]))
      score = beam_scores[i // out_beam_size][i % out_beam_size] if beam_scores is not None else 0
      results += [(score, hyp_seq)]
    return results

  def search_single_seq(self, sources, output_layer_name=None):
    """
    :param list[numpy.ndarray|list[int]] sources: source sequences as a list of indices
    :param str|None output_layer_name: e.g. "output". if not set, will read from config "search_output_layer"
    :return: list of all hyps, which is a tuple of score and string
    :rtype: list[(float,str)]
    """
    num_outputs = {
      "data": [self.network.extern_data.data["data"].dim, 1],
      "classes": [self.network.extern_data.data["classes"].dim, 1]}
    source_seqs = [numpy.array(s, dtype="int32") for s in sources]
    assert source_seqs[0].ndim == 1
    targets_empty_seq = numpy.array([], dtype="int32")  # empty...
    from returnn.datasets.generating import StaticDataset
    dataset = StaticDataset(
      data=[{"data": source_seq, "classes": targets_empty_seq} for source_seq in source_seqs], output_dim=num_outputs)
    dataset.init_seq_order(epoch=1)
    seq_idx = 0 if len(sources) == 1 else -1
    return self.search_single(dataset=dataset, seq_idx=seq_idx, output_layer_name=output_layer_name)

  def search_single_string_to_string_seq(self, sources, output_layer_name=None):
    """
    :param str|list[str] sources: source text as a string (list for batch translation)
    :param str|None output_layer_name: e.g. "output". if not set, will read from config "search_output_layer"
    :return: list of all hyps, which is a tuple of score and string
    :rtype: list[(float,str)]
    """
    source_voc = self.network.extern_data.data["data"].vocab
    target_voc = self.network.extern_data.data["targets"].vocab
    assert source_voc.num_labels == self.network.extern_data.data["data"].dim
    assert target_voc.num_labels == self.network.extern_data.data["classes"].dim
    if not isinstance(sources, list):
      sources = [sources]
    source_seq_lists = [source_voc.get_seq(s) for s in sources]
    results_raw = self.search_single_seq(sources=source_seq_lists, output_layer_name=output_layer_name)
    results = []
    for (score, raw) in results_raw:
      txt = target_voc.get_seq_labels(raw)
      results += [(score, txt)]
    return results

  def compute_priors(self, dataset, config=None):
    """
    :param Dataset dataset:
    :param Config.Config config:
    """
    assert isinstance(dataset, Dataset)
    if config:
      assert config is self.config
    else:
      config = self.config

    output_layer = self._get_output_layer()
    assert config.has('output_file'), 'output_file for priors numbers should be provided'
    output_file = config.value('output_file', '')
    assert not os.path.exists(output_file), "Already existing output file %r." % output_file
    print("Compute priors, using output layer %r, writing to %r." % (output_layer, output_file), file=log.v2)

    class Accumulator(object):
      """
      Also see PriorEstimationTaskThread for reference.
      """

      def __init__(self):
        self.sum_posteriors = numpy.zeros(int(output_layer.output.dim))
        self.seq_len = 0

      def __call__(self, outputs):
        """
        Called via extra_fetches_callback from the Runner.

        :param numpy.ndarray outputs: shape=(time,data)|(time,), depending if dense or sparse, flattened over batches
        """
        seq_len = outputs.shape[0]
        if output_layer.output.sparse:
          assert outputs.shape == (seq_len,)
        else:
          assert outputs.shape == (seq_len, output_layer.output.dim)
        if output_layer.output.sparse:
          from returnn.util.basic import class_idx_seq_to_1_of_k
          outputs = class_idx_seq_to_1_of_k(outputs, num_classes=output_layer.output.dim)
        self.sum_posteriors += numpy.sum(outputs, axis=0)
        self.seq_len += seq_len

    accumulator = Accumulator()
    batch_size = config.int('batch_size', 1)
    max_seqs = config.int('max_seqs', -1)
    epoch = config.int('epoch', 1)
    max_seq_length = config.float('max_seq_length', 0)
    if max_seq_length <= 0:
      max_seq_length = sys.maxsize
    dataset.init_seq_order(epoch=epoch)
    batches = dataset.generate_batches(
      recurrent_net=self.network.recurrent,
      batch_size=batch_size,
      max_seq_length=max_seq_length,
      max_seqs=max_seqs,
      used_data_keys=self.network.get_used_data_keys())
    forwarder = Runner(
      engine=self, dataset=dataset, batches=batches,
      train=False, eval=False,
      extra_fetches={
        'outputs': output_layer.output.get_placeholder_flattened()
      },
      extra_fetches_callback=accumulator)
    forwarder.run(report_prefix=self.get_epoch_str() + " forward")
    if not forwarder.finalized:
      print("Error happened. Exit now.")
      forwarder.exit_due_to_error()

    average_posterior = accumulator.sum_posteriors / accumulator.seq_len
    avg_sum = numpy.sum(average_posterior)
    assert numpy.isfinite(avg_sum)
    print("Prior sum in std-space (should be close to 1.0):", avg_sum, file=log.v1)
    log_average_posterior = numpy.log(average_posterior)
    with open(output_file, 'w') as f:
      numpy.savetxt(f, log_average_posterior, delimiter=' ')
    print("Saved prior in %r in +log space." % output_file, file=log.v1)

  def web_server(self, port):
    """
    Starts a web-server with a simple API to forward data through the network
    (or search if the flag is set).

    :param int port: for the http server
    :return:
    """
    assert sys.version_info[0] >= 3, "only Python 3 supported"
    # noinspection PyCompatibility
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from returnn.datasets.generating import StaticDataset
    from returnn.datasets.util.feature_extraction import ExtractAudioFeatures
    from returnn.datasets.util.vocabulary import Vocabulary, BytePairEncoding

    if not self.use_search_flag or not self.network or self.use_dynamic_train_flag:
      self.use_search_flag = True
      # At the moment this is probably not intended to use search with train flag.
      # Also see LayerBase._post_init_output() about setting size_placeholder to the target seq len,
      # so you would have have_known_seq_len=True in the RecLayer, with the given target seq len.
      self.use_dynamic_train_flag = False
      if self.network:
        print("Reinit network with search flag.", file=log.v3)
      self.init_network_from_config(self.config)

    engine = self
    soundfile = None
    input_data = self.network.extern_data.get_default_input_data()
    input_vocab = input_data.vocab
    input_audio_feature_extractor = None
    output_data = self.network.extern_data.get_default_target_data()
    output_vocab = output_data.vocab
    if (
          isinstance(self.config.typed_dict.get("dev", None), dict)
          and self.config.typed_dict["dev"]["class"] == "LibriSpeechCorpus"):
      # A bit hacky. Assumes that this is a dataset description for e.g. LibriSpeechCorpus.
      # noinspection PyPackageRequirements,PyUnresolvedReferences
      import soundfile  # pip install pysoundfile
      bpe_opts = self.config.typed_dict["dev"]["bpe"]
      audio_opts = self.config.typed_dict["dev"]["audio"]
      bpe = BytePairEncoding(**bpe_opts)
      assert output_data.sparse
      assert bpe.num_labels == output_data.dim
      output_vocab = bpe
      input_audio_feature_extractor = ExtractAudioFeatures(**audio_opts)
    else:
      assert isinstance(input_vocab, Vocabulary)
    assert isinstance(output_vocab, Vocabulary)
    num_outputs = {
      input_data.name: [input_data.dim, input_data.ndim],
      output_data.name: [output_data.dim, output_data.ndim]}

    output_layer_name = self.config.value("search_output_layer", "output")
    output_layer = self.network.layers[output_layer_name]
    output_t = output_layer.output.get_placeholder_as_batch_major()
    output_seq_lens_t = output_layer.output.get_sequence_lengths()
    out_beam_size = output_layer.output.beam.beam_size
    output_layer_beam_scores_t = None
    if out_beam_size is None:
      print("Given output %r is after decision (no beam)." % output_layer, file=log.v1)
    else:
      print("Given output %r has beam size %i." % (output_layer, out_beam_size), file=log.v1)
      output_layer_beam_scores_t = output_layer.get_search_choices().beam_scores

    class Handler(BaseHTTPRequestHandler):
      """
      Handle POST requests.
      """
      # noinspection PyPep8Naming
      def do_POST(self):
        """
        Handle POST request.
        """
        try:
          self._do_post()
        except Exception:
          sys.excepthook(*sys.exc_info())
          raise

      def _do_post(self):
        import cgi
        form = cgi.FieldStorage(
          fp=self.rfile,
          headers=self.headers,
          environ={'REQUEST_METHOD': 'POST'})
        print("HTTP server, got POST.", file=log.v3)
        from io import BytesIO
        f = BytesIO(form["file"].file.read())
        print("Input file size:", len(f.getbuffer().tobytes()), "bytes", file=log.v4)
        audio_len = None
        if input_audio_feature_extractor:
          try:
            audio, sample_rate = soundfile.read(f)
          except Exception as exc:
            print("Error reading audio (%s). Invalid format? Size %i, first few bytes %r." % (
              exc, len(f.getbuffer().tobytes()), f.getbuffer().tobytes()[:20]), file=log.v2)
            raise
          audio_len = float(len(audio)) / sample_rate
          print("audio len %i (%.1f secs), sample rate %i" % (len(audio), audio_len, sample_rate), file=log.v4)
          if audio.ndim == 2:  # multiple channels:
            audio = numpy.mean(audio, axis=1)  # mix together
          features = input_audio_feature_extractor.get_audio_features(audio=audio, sample_rate=sample_rate)
        else:
          sentence = f.read().decode("utf8").strip()
          print("Input:", sentence, file=log.v4)
          seq = input_vocab.get_seq(sentence)
          print("Input seq:", input_vocab.get_seq_labels(seq), file=log.v4)
          features = numpy.array(seq, dtype="int32")
        targets = numpy.array([], dtype="int32")  # empty...
        dataset = StaticDataset(
          data=[{input_data.name: features, output_data.name: targets}], output_dim=num_outputs)
        dataset.init_seq_order(epoch=1)

        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        start_time = time.time()
        output_d = engine.run_single(dataset=dataset, seq_idx=0, output_dict={
          "output": output_t,
          "seq_lens": output_seq_lens_t,
          "beam_scores": output_layer_beam_scores_t})
        delta_time = time.time() - start_time
        print("Took %.3f secs for decoding." % delta_time, file=log.v4)
        if audio_len:
          print("Real-time-factor: %.3f" % (delta_time / audio_len), file=log.v4)
        output = output_d["output"]
        seq_lens = output_d["seq_lens"]
        beam_scores = output_d["beam_scores"]
        assert len(output) == len(seq_lens) == (out_beam_size or 1)
        if out_beam_size:
          assert beam_scores.shape == (1, out_beam_size)  # (batch, beam)

        first_best_txt = output_vocab.get_seq_labels(output[0][:seq_lens[0]])
        print("Best output: %s" % first_best_txt, file=log.v4)

        if out_beam_size:
          self.wfile.write(b"[\n")
          for i in range(out_beam_size):
            txt = output_vocab.get_seq_labels(output[i][:seq_lens[i]])
            score = beam_scores[0][i]
            self.wfile.write(("(%r, %r)\n" % (score, txt)).encode("utf8"))
          self.wfile.write(b"]\n")

        else:
          self.wfile.write(("%r\n" % first_best_txt).encode("utf8"))

    print("Simple search web server, listening on port %i." % port, file=log.v2)
    server_address = ('', port)
    # noinspection PyAttributeOutsideInit
    self.httpd = HTTPServer(server_address, Handler)
    self.httpd.serve_forever()


def get_global_engine():
  """
  Similar to :func:`Config.get_global_config`.

  :rtype: Engine
  """

  import sys
  main_mod = sys.modules["__main__"]  # should be rnn.py
  if isinstance(getattr(main_mod, "engine", None), Engine):
    # noinspection PyUnresolvedReferences
    return main_mod.engine
  # Maybe __main__ is not rnn.py, or config not yet loaded.
  # Anyway, try directly. (E.g. for SprintInterface.)
  import returnn.__main__ as rnn
  assert isinstance(rnn.engine, Engine)  # no other option anymore
  return rnn.engine
