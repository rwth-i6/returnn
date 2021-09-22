
"""
This provides lazy module wrappers such that old-style RETURNN-as-a-framework imports are still working.
E.g. like::

    import returnn.TFUtil
    import returnn.TFNativeOp

Some old RETURNN configs also might directly have imports like::

    import TFUtil
    import TFHorovod

This is supported as well.

"""

import sys
import os
import types
import typing
import importlib

old_to_new_mod_mapping = {
  "rnn": "__main__",
  "better_exchook": "util.better_exchook",
  "ActivationFunctions": "theano.activation_functions",
  "BestPathDecoder": "theano.ops.best_path_decoder",
  "BundleFile": "datasets.bundle_file",
  "CachedDataset": "datasets.cached",
  "CachedDataset2": "datasets.cached2",
  "Config": "config",
  "CTC": "theano.ops.ctc",
  "CustomLSTMFunctions": "theano.ops.custom_lstm_functions",
  "Dataset": "datasets.basic",
  "DebugHelpers": "util.debug_helpers",
  "Debug": "util.debug",
  "Device": "theano.device",
  "EngineBase": "engine.base",
  "EngineBatch": "engine.batch",
  "Engine": "theano.engine",
  "EngineTask": "theano.engine_task",
  "EngineUtil": "theano.engine_util",
  "Fsa": "util.fsa",
  "FunctionLoader": "theano.ops.function_loader",
  "GeneratingDataset": "datasets.generating",
  "HDFDataset": "datasets.hdf",
  "HyperParamTuning": "tf.hyper_param_tuning",
  "Inv": "theano.ops.inv",
  "LearningRateControl": "learning_rate_control",
  "LmDataset": "datasets.lm",
  "Log": "log",
  "MetaDataset": "datasets.meta",
  "MultiBatchBeam": "theano.ops.multi_batch_beam",
  "NativeOp": "native_op",
  "Network": "theano.network",
  "NetworkBaseLayer": "theano.layers.base",
  "NetworkCNNLayer": "theano.layers.cnn",
  "NetworkCopyUtils": "theano.network_copy_utils",
  "NetworkCtcLayer": "theano.layers.ctc",
  "NetworkDescription": "network_description",
  "NetworkHiddenLayer": "theano.layers.hidden",
  "NetworkLayer": "theano.layers.basic",
  "NetworkLstmLayer": "theano.layers.lstm",
  "NetworkOutputLayer": "theano.layers.output",
  "NetworkRecurrentLayer": "theano.layers.rec",
  "NetworkStream": "theano.network_stream",
  "NetworkTwoDLayer": "theano.layers.twod",
  "NormalizationData": "datasets.normalization_data",
  "NumpyDumpDataset": "datasets.numpy_dump",
  "OpBLSTM": "theano.ops.blstm",
  "OpInvAlign": "theano.ops.inv_align",
  "OpLSTM": "theano.ops.lstm",
  "OpLSTMCell": "theano.ops.lstm_cell",
  "OpLSTMCustom": "theano.ops.lstm_custom",
  "OpLSTMRec": "theano.ops.lstm_rec",
  "OpNumpyAlign": "theano.ops.numpy_align",
  "Pretrain": "pretrain",
  "RawWavDataset": "datasets.raw_wav",
  "RecurrentTransform": "theano.recurrent_transform",
  "Server": "theano.server",
  "SprintCache": "sprint.cache",
  "SprintControl": "sprint.control",
  "SprintDataset": "datasets.sprint",
  "SprintErrorSignals": "sprint.error_signals",
  "SprintExternInterface": "sprint.extern_interface",
  "SprintInterface": "sprint.interface",
  "StereoDataset": "datasets.stereo",
  "TaskSystem": "util.task_system",
  "TaskSystem_example": "util.task_system_example",
  "TFCompat": "tf.compat",
  "TFDataPipeline": "tf.data_pipeline",
  "TFDistributed": "tf.distributed",
  "TFEngine": "tf.engine",
  "TFHorovod": "tf.horovod",
  "TFKenLM": "tf.util.ken_lm",
  "TFNativeOp": "tf.native_op",
  "TFNetwork": "tf.network",
  "TFNetworkLayer": "tf.layers.basic",
  "TFNetworkRecLayer": "tf.layers.rec",
  "TFNetworkSegModLayer": "tf.layers.segmental_model",
  "TFNetworkSigProcLayer": "tf.layers.signal_processing",
  "TFSprint": "tf.sprint",
  "TFUpdater": "tf.updater",
  "TFUtil": "tf.util.basic",
  "TheanoUtil": "theano.util",
  "TorchWrapper": "theano.ops.torch_wrapper",
  "TwoStateBestPathDecoder": "theano.ops.two_state_best_path_decoder",
  "TwoStateHMMOp": "theano.ops.two_state_hmm",
  "Updater": "theano.updater",
  "Util": "util.basic",
}

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_mod_cache = {}  # new/old mod name -> mod


def setup(package_name=__package__, modules=None):
  """
  This does the setup, such that all the modules become available in the `returnn` package.
  It does not import all the modules now, but instead provides them lazily.

  :param str package_name: "returnn" by default
  :param dict[str,types.ModuleType]|None modules: if set, will do ``modules[old_mod_name] = mod``
  """
  for old_mod_name, new_mod_name in sorted(old_to_new_mod_mapping.items()):
    full_mod_name = "returnn.%s" % new_mod_name
    full_old_mod_name = "%s.%s" % (package_name, old_mod_name)
    if full_mod_name in _mod_cache:
      mod = _mod_cache[full_mod_name]
    elif full_mod_name in sys.modules:
      mod = sys.modules[full_mod_name]
      _mod_cache[full_mod_name] = mod
    else:
      mod = _LazyLoader(
        full_mod_name=full_mod_name,
        full_old_mod_name=full_old_mod_name, old_mod_name=old_mod_name, modules=modules)
    if old_mod_name not in sys.modules:
      sys.modules[old_mod_name] = mod
    if full_old_mod_name not in sys.modules:
      sys.modules[full_old_mod_name] = mod
    if modules is not None:
      modules[old_mod_name] = mod


class _LazyLoader(types.ModuleType):
  """
  Lazily import a module, mainly to avoid pulling in large dependencies.
  Code borrowed from TensorFlow, and simplified, and extended.
  """
  def __init__(self, full_mod_name, **kwargs):
    """
    :param str full_mod_name:
    """
    super(_LazyLoader, self).__init__(full_mod_name)
    fn = "%s/%s.py" % (_base_dir, full_mod_name.replace(".", "/"))
    if not os.path.exists(fn):
      fn = "%s/%s/__init__.py" % (_base_dir, full_mod_name.replace(".", "/"))
      assert os.path.exists(fn), "_LazyLoader: mod %r not found in %r" % (full_mod_name, _base_dir)
    self.__file__ = fn
    self._lazy_mod_config = dict(full_mod_name=full_mod_name, **kwargs)  # type: typing.Dict[str]

  def _load(self):
    full_mod_name = self.__name__
    lazy_mod_config = self._lazy_mod_config
    old_mod_name = lazy_mod_config.get("old_mod_name", None)
    full_old_mod_name = lazy_mod_config.get("full_old_mod_name", None)
    modules = lazy_mod_config.get("modules", None)
    if full_mod_name in _mod_cache:
      module = _mod_cache[full_mod_name]
    else:
      try:
        module = importlib.import_module(full_mod_name)
      except Exception:
        # Note: If we get any exception in the module itself (e.g. No module named 'theano' or so),
        # just pass it on. But this can happen.
        raise
      _mod_cache[full_mod_name] = module

    if old_mod_name:
      sys.modules[old_mod_name] = module
    if full_old_mod_name:
      sys.modules[full_old_mod_name] = module
    if modules is not None:
      assert old_mod_name
      modules[old_mod_name] = module

    # Do not set self.__dict__, because the module itself could later update itself.
    return module

  def __getattribute__(self, item):
    # Implement also __getattribute__ such that early access to just self.__dict__ (e.g. via vars(self)) also works.
    if item == "__dict__":
      # noinspection PyBroadException
      try:
        mod = self._load()
      except Exception:  # many things could happen
        print("WARNING: %s cannot be imported, __dict__ not available" % self.__name__)
        # In many cases, this is not so critical, because we likely just checked the dict content or so.
        # This should be safe, as we have this registered in sys.modules, and some code just iterates
        # through all sys.modules to check for something.
        # Any other attribute access will lead to the real exception.
        # We ignore this for __dict__, and just return a dummy empty dict.
        return {}
      return getattr(mod, "__dict__")
    return super(_LazyLoader, self).__getattribute__(item)

  def __getattr__(self, item):
    module = self._load()
    return getattr(module, item)

  def __dir__(self):
    module = self._load()
    return dir(module)

  def __setattr__(self, key, value):
    if key in ["__file__", "_lazy_mod_config"]:
      super(_LazyLoader, self).__setattr__(key, value)
      return
    module = self._load()
    setattr(module, key, value)
