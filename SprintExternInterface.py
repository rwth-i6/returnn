
"""
This is a Sprint interface implementation.
See SprintInterface.py for another Sprint interface.
This Sprint interface is to be used for ExternSprintDataset, which should automatically use it.
"""

import os
import TaskSystem
from TaskSystem import Pickler, Unpickler
from Util import to_bool

# Start Sprint PythonSegmentOrder interface. {
# We use the PythonSegmentOrder just to get an estimate (upper limit) about the number of sequences.

segmentOrderList = None; ":type: list[str] "

def getSegmentList(corpusName, segmentList, **kwargs):
  """
  Called by Sprint PythonSegmentOrder.
  Set python-segment-order = true in Sprint to use this.

  If this is used, this gets called really early.
  If it is used together with the Sprint PythonTrainer,
  it will get called way earlier before the init() below.
  It might also get called multiple times, e.g. if
  Sprint is in interactive mode to calc the seg count.
  This is optional. You can use the SprintInterface
  only for the PythonTrainer.

  :type corpusName: str
  :type segmentList: list[str]
  :rtype: list[str]
  :returns segment list. Can also be an iterator.
  """
  print("SprintExternInterface: getSegmentList(%r), num segments: %i" % (corpusName, len(segmentList)))
  global segmentOrderList
  segmentOrderList = segmentList
  # No shuffling here. We expect to do that via Sprint.
  return segmentList

# End Sprint PythonSegmentOrder interface. }

# Start Sprint PythonTrainer interface. {

isInitialized = False

def init(**kwargs):
  import better_exchook
  better_exchook.install()
  # This module can also be used for Sprint PythonControl, which will also call init().
  # We need to catch these cases.
  if "name" in kwargs and kwargs["name"] == "Sprint.PythonControl":
    return PythonControl.init(**kwargs)
  return init_PythonTrainer(**kwargs)

def _parse_config_str(config_str):
  assert isinstance(config_str, (str, unicode))
  config_list = config_str.split(",")
  config = {key: value for (key, value) in [s.split(":", 1) for s in config_list if s]}
  return config

def _common_init(config):
  if to_bool(config.get("EnableAutoNumpySharedMemPickling", False)) and not TaskSystem.SharedMemNumpyConfig["enabled"]:
    TaskSystem.SharedMemNumpyConfig["enabled"] = True
    print("SprintExternInterface[pid %i] EnableAutoNumpySharedMemPickling = True" % (os.getpid(),))

def init_PythonTrainer(inputDim, outputDim, config, targetMode, **kwargs):
  """
  Called by Sprint when it initializes the PythonTrainer.
  Set trainer = python-trainer in Sprint to enable.
  Note that Sprint will call this, i.e. the trainer init lazily quite late,
  only once it sees the first data.

  :type inputDim: int
  :type outputDim: int
  :param str config: config string, passed by Sprint. assumed to be ","-separated
  :param str targetMode: "target-alignment" or "criterion-by-sprint" or so
  """
  print("SprintExternInterface[pid %i]: PythonTrainer init_PythonTrainer()" % (os.getpid(),))
  print "inputDim:", inputDim
  print "outputDim:", outputDim
  print "config:", config
  print "targetMode:", targetMode
  print "other args:", kwargs

  global InputDim, OutputDim, isInitialized
  InputDim = inputDim
  OutputDim = outputDim
  isInitialized = True
  assert targetMode != "criterion-by-sprint"
  config = _parse_config_str(config)
  assert config["action"] == "ExternSprintDataset"
  _common_init(config)

  _init_global_sprintDataset(inputDim=inputDim, outputDim=outputDim, config=config)

sprintDataset = None; ":type: ExternSprintDatasetSource"

def _init_global_sprintDataset(inputDim, outputDim, config):
  global sprintDataset
  if sprintDataset: return
  numSegments = len(segmentOrderList) if segmentOrderList is not None else None
  sprintDataset = ExternSprintDatasetSource(c2p_fd=int(config["c2p_fd"]), p2c_fd=int(config["p2c_fd"]),
                                            inputDim=inputDim, outputDim=outputDim, numSegments=numSegments)


def exit():
  print "SprintExternInterface: PythonTrainer exit()"
  assert isInitialized
  sprintDataset.close()

def feedInput(features, weights=None, segmentName=None):  # unsupervised case
  feedInputAndTarget(features=features, weights=weights, segmentName=segmentName)

def feedInputAndTargetAlignment(features, targetAlignment, weights=None, segmentName=None):
  feedInputAndTarget(features=features, alignment=targetAlignment, weights=weights, segmentName=segmentName)

def feedInputAndTargetSegmentOrth(features, targetSegmentOrth, weights=None, segmentName=None):
  feedInputAndTarget(features=features, orthography=targetSegmentOrth, weights=weights, segmentName=segmentName)

feedInputUnsupervised = feedInput

def feedInputAndTarget(features, weights=None, segmentName=None,
                       orthography=None, alignment=None,
                       speaker_name=None, speaker_gender=None,
                       **kwargs):
  assert features.shape[0] == InputDim
  targets = {}
  if alignment is not None:
    targets["classes"] = alignment
  if orthography is not None:
    targets["orth"] = orthography
  sprintDataset.addNewData(segmentName=segmentName, features=features, targets=targets)

# End Sprint PythonTrainer interface. }

# Start Sprint PythonControl interface. {

class PythonControl:
  instance = None

  @classmethod
  def init(cls, **kwargs):  # called by global init().
    print("SprintExternInterface[pid %i]: PythonControl init %r" % (os.getpid(), kwargs))
    if cls.instance:
      return cls.instance
    cls.instance = cls(**kwargs)
    return cls.instance

  def __init__(self, config, **kwargs):
    self.config = _parse_config_str(config)
    _common_init(self.config)

  def init_processing(self, input_dim, output_dim, **kwargs):
    print("SprintExternInterface: PythonControl init_processing inputDim=%i, outputDim=%i, other:%r" % (input_dim, output_dim, kwargs))
    _init_global_sprintDataset(inputDim=input_dim, outputDim=output_dim, config=self.config)

  def process_segment(self, name, orthography, features, alignment, soft_alignment, **kwargs):
    assert sprintDataset
    targets = {}
    if orthography is not None:
      targets["orth"] = orthography
    if alignment is not None:
      targets["classes"] = alignment
    elif soft_alignment is not None:
      # We expect a sparse soft-alignment in coordinate format (time, class-idx, weight [0,1]).
      assert isinstance(soft_alignment, tuple)
      assert len(soft_alignment) == 3
      # We encode: sparse-coo-format, ndim == 2.
      targets["classes[sparse:coo:2:0]"] = soft_alignment[0]
      targets["classes[sparse:coo:2:1]"] = soft_alignment[1]
      targets["classes[sparse:coo:2:2]"] = soft_alignment[2]
    sprintDataset.addNewData(segmentName=name, features=features, targets=targets)

  def exit(self, **kwargs):
    print("SprintExternInterface: PythonControl exit %r" % kwargs)
    if sprintDataset:
      sprintDataset.close()

# End Sprint PythonControl interface. }


class ExternSprintDatasetSource:

  """
  This will send data to ExternSprintDataset over a pipe.
  We expect that we are child process and the parent process has spawned us via ExternSprintDataset
  and is waiting for our data.
  """

  def __init__(self, c2p_fd, p2c_fd, inputDim, outputDim, numSegments):
    """
    :param int c2p_fd: child-to-parent file descriptor
    :param int p2c_fd: parent-to-child file descriptor
    :type inputDim: int
    :type outputDim: int
    :type numSegments: int | None
    :param numSegments: can be None if not known in advance
    """
    self.pipe_c2p = os.fdopen(c2p_fd, "w")
    self.pipe_p2c = os.fdopen(p2c_fd, "r")
    self._send("init", (inputDim, outputDim, numSegments))

  def _send(self, dataType, args=None):
    Pickler(self.pipe_c2p).dump((dataType, args))
    self.pipe_c2p.flush()

  def addNewData(self, segmentName, features, targets):
    """
    :param numpy.ndarray features: 2D array, (feature,time)
    :param dict[str,numpy.ndarray] targets: each target is either 1D (time->idx) or 2D (time,class)
    """
    self._send("data", (segmentName, features, targets))

  def close(self):
    self._send("exit")
    self.pipe_c2p.close()
    self.pipe_p2c.close()

