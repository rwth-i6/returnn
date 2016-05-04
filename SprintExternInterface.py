
"""
This is a Sprint interface implementation.
See SprintInterface.py for another Sprint interface.
This Sprint interface is to be used for ExternSprintDataset, which should automatically use it.
"""

import os
from TaskSystem import Pickler, Unpickler

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

def init(inputDim, outputDim, config, targetMode, **kwargs):
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
  print "PythonTrainer SprintExternInterface init()"
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
  config = config.split(",")
  config = {key: value for (key, value) in [s.split(":", 1) for s in config if s]}
  assert config["action"] == "ExternSprintDataset"

  global sprintDataset
  numSegments = len(segmentOrderList) if segmentOrderList is not None else None
  sprintDataset = ExternSprintDatasetSource(c2p_fd=int(config["c2p_fd"]), p2c_fd=int(config["p2c_fd"]),
                                            inputDim=inputDim, outputDim=outputDim, numSegments=numSegments)

def exit():
  print "PythonTrainer SprintExternInterface exit()"
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

