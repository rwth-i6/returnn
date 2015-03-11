
# We expect that Theano works in the current Python env.

print "CRNN Python SprintInterface module load"

import os
import sys
import time
import re
from threading import Event, Thread
import thread

import numpy
import theano
import theano.tensor as T
import h5py

from SprintDataset import SprintDataset
from Log import log
from Device import get_gpu_names
import rnn
from Engine import Engine

DefaultSprintCrnnConfig = "config/crnn.config"

startTime = None
isInitialized = False
isStarted = False
InputDim = None
OutputDim = None
TargetMode = None
Task = "train"

config = None; """ :type: rnn.Config """
dataset = None; """ :type: SprintDataset """
engine = None; """ :type: Engine """

lastEpochBatchModel = None; """ :type: (int,int,str|None) """  # see getLastEpochBatch()


rnn.initBetterExchook()


# Start Sprint PythonSegmentOrder interface. {

def getSegmentList(corpusName, segmentList, segmentsInfo):
  """
  Called by Sprint PythonSegmentOrder.
  Set python-segment-order = true in Sprint to use this.

  If this is used, this gets called really early.
  If it is used together with the Sprint PythonTrainer,
  it will get called way earlier before the init() below.
  It might also get called multiple times, e.g. if
  Sprint is in interactive mode to calc the seg count.

  :type corpusName: str
  :type segmentList: list[str]
  :type segmentsInfo: dict[str,dict[str]]
  :rtype: list[str]
  :returns segment list. Can also be an iterator.
  """
  print("Sprint: getSegmentList(%r)" % corpusName)
  print("Corpus segments #: %i" % len(segmentList))
  print("(This can be further filtered in Sprint by a whitelist or so.)")

  # Init what we need. These can be called multiple times.
  initBase()
  initDataset()
  with dataset.lock:
    assert not isStarted
    dataset.initFromSegmentOrder(segmentList, segmentsInfo)
    dataset.finalized = False

  numEpochs = getNumEpochs()
  startEpoch, startSegmentIdx = getStartEpochBatch()
  print("Sprint: Starting with epoch %i, segment-idx %s." % (startEpoch, startSegmentIdx))
  print("Final epoch is: %i" % numEpochs)

  # Loop over multiple epochs. Epochs start at 1.
  for curEpoch in range(startEpoch, numEpochs + 1):
    if isStarted:
      waitUntilTrainerInEpoch(curEpoch)

    with dataset.lock:
      dataset.init_seq_order(epoch=curEpoch)
      segmentList = dataset.getSegmentList()

    print("Sprint epoch: %i" % curEpoch)
    startSegmentIdx = 0
    if curEpoch == startEpoch: startSegmentIdx = startSegmentIdx
    for curSegmentIdx in range(startSegmentIdx, len(segmentList)):
      yield segmentList[curSegmentIdx]

  dataset.finalize()

# End Sprint PythonSegmentOrder interface. }


# Start Sprint PythonTrainer interface. {

def init(inputDim, outputDim, config, targetMode, cudaEnabled, cudaActiveGpu):
  """
  Called by Sprint when it initializes the PythonTrainer.
  Set trainer = python-trainer in Sprint to enable.
  Note that Sprint will call this, i.e. the trainer init lazily quite late,
  only once it sees the first data.

  :type inputDim: int
  :type outputDim: int
  :param str config: config string, passed by Sprint. assumed to be ","-separated
  :param str targetMode: "target-alignment" or "criterion-by-sprint" or so
  :param bool cudaEnabled: whether Sprint has CUDA enabled
  :param int cudaActiveGpu: the GPU idx used by Sprint
  """
  print "Python train init()"
  print "inputDim:", inputDim
  print "outputDim:", outputDim
  print "config:", config
  print "targetMode:", targetMode
  print "Sprint cudaEnabled:", cudaEnabled
  print "Sprint cudaActiveGpu:", cudaActiveGpu
  global InputDim, OutputDim
  InputDim = inputDim
  OutputDim = outputDim

  config = config.split(",")
  config = {key: value for (key, value) in [s.split(":", 1) for s in config if s]}

  epoch = config.get("epoch", None)
  if epoch is not None:
    epoch = int(epoch)
    assert epoch >= 1

  global Task
  action = config["action"]
  if action == "train":
    pass
  elif action == "recog":
    epoch += 1  # We pass the last trained epoch.
    targetMode = "evaluate"
    Task = "evaluate"
  else:
    assert False, "unknown action: %r" % action

  initBase()
  # Note: Atm, we must know all the segment info in advance.
  # The CRNN Engine.train() depends on that.
  assert dataset, "need to be inited already via segment_order mod"
  assert dataset.num_seqs > 0, "need to have data seqs"
  dataset.setDimensions(inputDim, outputDim)
  dataset.initialize()

  setTargetMode(targetMode)
  start(epoch)


def exit():
  print "Python train exit()"
  assert isInitialized
  assert isStarted
  trainThread.join()
  rnn.finalize()
  print >> log.v3, ("elapsed total time: %f" % (time.time() - startTime))



def feedInput(features, weights=None, segmentName=None):
  #print "feedInput", segmentName
  assert features.shape[0] == InputDim
  if Task == "train":
    posteriors = train(segmentName, features)
  elif Task == "evaluate":
    posteriors = evaluate(features)
  else:
    assert False, "invalid task: %r" % Task
  assert posteriors.shape[0] == OutputDim
  return posteriors

def finishDiscard():
  print "finishDiscard()"
  raise NotImplementedError # TODO ...

def finishError(error, errorSignal, naturalPairingType=None):
  assert naturalPairingType == "softmax"
  assert Task == "train"
  # reformat. see train()
  error = numpy.array([error], dtype=theano.config.floatX)
  errorSignal = errorSignal.transpose()
  errorSignal = errorSignal[:, numpy.newaxis, :]
  errorSignal = numpy.array(errorSignal, dtype=theano.config.floatX)
  assert errorSignal.shape == Criterion.posteriors.shape

  Criterion.error = error
  Criterion.errorSignal = errorSignal
  Criterion.gotErrorSignal.set()


def feedInputAndTargetAlignment(features, targetAlignment, weights=None, segmentName=None):
  #print "feedInputAndTargetAlignment", segmentName
  assert features.shape[0] == InputDim
  train(segmentName, features, targetAlignment)


def feedInputAndTargetSegmentOrth(features, targetSegmentOrth, weights=None, segmentName=None):
  raise NotImplementedError


def feedInputUnsupervised(features, weights=None, segmentName=None):
  assert features.shape[0] == InputDim
  train(segmentName, features)

# End Sprint PythonTrainer interface. }



def dumpFlags():
  print "available GPUs:", get_gpu_names()

  import theano.sandbox.cuda as theano_cuda
  print "CUDA via", theano_cuda.__file__
  print "CUDA available:", theano_cuda.cuda_available

  print "THEANO_FLAGS:", rnn.TheanoFlags
  print "CUDA_LAUNCH_BLOCKING:", os.environ.get("CUDA_LAUNCH_BLOCKING")


def setTargetMode(mode):
  """
  :param str mode: target mode
  """
  global TargetMode
  assert config, "not initialized"
  TargetMode = mode
  task = "train"
  loss = config.value('loss', None)
  if TargetMode == "criterion-by-sprint":
    assert loss == "sprint"
  elif TargetMode == "target-alignment":
    # Crnn always expects an alignment, so this should be ok.
    pass
  elif TargetMode == "evaluate":
    # Will be handled below.
    task = "forward"
  else:
    assert False, "target-mode %s not supported yet..." % TargetMode
  config.set("task", task)



def initBase(configfile=None):
  """
  :type configfile: str | None
  :type inputDim: int
  :type outputDim: int
  """
  global isInitialized
  if isInitialized: return
  isInitialized = True

  if configfile is None:
    configfile = DefaultSprintCrnnConfig
  assert os.path.exists(configfile)

  rnn.initThreadJoinHack()
  rnn.initConfig(configfile, [])
  global config
  config = rnn.config
  rnn.initLog()
  rnn.initConfigJson()

  modelFileName = getLastEpochBatch()[2]
  devices = rnn.initDevices()
  network = rnn.initNeuralNetwork(modelFileName)

  rnn.printTaskProperties(devices, network)
  rnn.initEngine(devices, network)
  global engine
  engine = rnn.engine
  assert isinstance(engine, Engine)


def start(epoch=None):
  global config, engine, isInitialized, isStarted
  assert isInitialized, "need to call init() first"
  assert not isStarted
  assert dataset, "need to call initDataset() first"

  start_epoch, start_batch = getStartEpochBatch()
  # If some epoch is explicitly specified, it checks whether it matches.
  if epoch is not None:
    assert epoch == start_epoch

  def trainThreadFunc():
    try:
      assert TargetMode
      if TargetMode == "target-alignment":
        engine.train_config(config, train_data=dataset, dev_data=None, eval_data=None,
                            start_epoch=start_epoch, start_batch=start_batch)
      elif TargetMode == "criterion-by-sprint":
        # TODO ...
        raise NotImplementedError
      else:
        raise Exception("target-mode not supported: %s" % TargetMode)
    except Exception:
      try:
        print "crnn train failed"
        sys.excepthook(*sys.exc_info())
      finally:
        # Exceptions are fatal. Stop now.
        thread.interrupt_main()

  global trainThread
  trainThread = Thread(target=trainThreadFunc, name="Sprint CRNN train thread")
  trainThread.daemon = True
  trainThread.start()

  global startTime
  startTime = time.time()
  isStarted = True


def initDataset():
  global dataset
  if dataset: return
  dataset, _ = SprintDataset.load_data(config, rnn.getCacheSizes()[0])



def getNumEpochs():
  global config, engine
  assert engine
  assert config
  config_num_epochs = engine.config_get_num_epochs(config)
  with engine.lock:
    if engine.is_training:
      assert engine.num_epochs == config_num_epochs
  return config_num_epochs


def getLastEpochBatch():
  """
  :returns (epoch,batch,modelFilename)
  :rtype: (int,int|None,str|None)
  """
  global lastEpochBatchModel
  if lastEpochBatchModel: return lastEpochBatchModel

  global config
  assert config
  modelFileName = config.value('model', '')
  assert modelFileName, "need 'model' in config"

  from glob import glob
  files = glob(modelFileName + ".*")
  file_list = []; """ :type: list[(int,int,str)] """
  for fn in files:
    m = re.match(".*\\.([0-9]+)\\.([0-9]+)$", fn)
    if m:
      epoch, batch = map(int, m.groups())
    else:
      m = re.match(".*\\.([0-9]+)$", fn)
      if m:
        epoch = int(m.groups()[0])
        batch = None
      else:
        continue
    file_list += [(epoch, batch, fn)]
  if len(file_list) == 0:
    lastEpochBatchModel = (None, None, None)
  else:
    file_list.sort()
    lastEpochBatchModel = file_list[-1]
  return lastEpochBatchModel


def getStartEpochBatch():
  """
  We will always automatically determine the best start (epoch,batch) tuple
  based on existing model files.
  This ensures that the files are present and enforces that there are
  no old outdated files which should be ignored.
  Note that epochs start at idx 1 and batches at idx 0.
  :returns (epoch,batch)
  :rtype (int,int)
  """
  last_epoch, last_batch, _ = getLastEpochBatch()
  if last_epoch is None:
    start_epoch = 1
    start_batch = 0
  elif last_batch is None:
    # No batch -> start with next epoch.
    start_epoch = last_epoch + 1
    start_batch = 0
  else:
    # Stay in last epoch, start with next batch.
    start_epoch = last_epoch
    start_batch = last_batch + 1
  return start_epoch, start_batch


def waitUntilTrainerInEpoch(epoch):
  assert isStarted
  assert engine
  while True:
    with engine.lock:
      if engine.training_finished: return
      if engine.is_training:
        if engine.cur_epoch == epoch: return
        assert engine.cur_epoch < epoch  # would confuse the seq order otherwise...
      engine.cond.wait()


def train(segmentName, features, targets=None):
  """
  :param str|None segmentName: full name
  :param numpy.ndarray features: 2d array
  :param numpy.ndarray|None targets: 2d or 1d array
  :return:
  """
  assert engine is not None, "not initialized. call initBase()"
  assert dataset

  # is in format (feature,time)
  assert InputDim == features.shape[0]
  assert InputDim == engine.network.n_in
  T = features.shape[1]
  # must be in format: (time,feature)
  features = features.transpose()
  assert features.shape[0] == T
  assert features.shape[1] == InputDim
  assert len(features.shape) == 2

  if TargetMode.startswith("target-"):
    assert targets is not None
    assert targets.shape == (T,)  # is in format (time,)
  else:
    assert targets is None

  dataset.addNewData(segmentName, features, targets)

  if TargetMode == "criterion-by-sprint":

    # TODO...

    Criterion.gotPosteriors.clear()

    Criterion.gotPosteriors.wait()
    posteriors = Criterion.posteriors
    assert posteriors is not None

    # posteriors is in format (time,batch,emission)
    assert posteriors.shape[0] == T
    assert posteriors.shape[1] == 1
    OutputDim = posteriors.shape[2]
    assert OutputDim == engine.network.n_out
    assert len(posteriors.shape) == 3
    # reformat to Sprint expected format (emission,time)
    posteriors = posteriors[:,0,:]
    posteriors = posteriors.transpose()
    assert posteriors.shape[0] == OutputDim
    assert posteriors.shape[1] == T
    assert len(posteriors.shape) == 2

    return posteriors



def evaluate(features):
  assert engine is not None, "not initialized"

  # is in format (feature,time)
  assert InputDim == features.shape[0]
  assert InputDim == engine.network.n_in
  T = features.shape[1]
  # must be in format: (time,batch,feature)
  features = features.transpose()
  features = features[:, numpy.newaxis, :]
  assert features.shape[0] == T
  assert features.shape[1] == 1
  assert features.shape[2] == InputDim
  assert len(features.shape) == 3

  # TODO...
  posteriors = engine.evaluate(features)

  assert posteriors is not None

  # posteriors is in format (time,emission)
  assert posteriors.shape[0] == T
  OutputDim = posteriors.shape[1]
  assert OutputDim == engine.network.n_out
  assert len(posteriors.shape) == 2
  # reformat to Sprint expected format (emission,time)
  posteriors = posteriors.transpose()
  assert posteriors.shape[0] == OutputDim
  assert posteriors.shape[1] == T
  assert len(posteriors.shape) == 2

  return posteriors


class CrnnEngine:
  """
  modelled as a mixture of crnn.Engine and crnn.TrainProcess
  """

  def __init__(self, device, network, epoch, targetMode, task):
    self.device = device
    self.epoch = epoch
    self.targetMode = targetMode
    self.task = task

    # Copy over weights once.
    # (The original crnn.Engine does this for every mini-batch.)
    device.trainnet.set_params(network.get_params())
    # Ignore the original network, just use trainnet directly.
    self.network = device.trainnet

    # This is where we store the gradients:
    # self.network.gparams is a list of shared vars.
    self.gparams = dict(
      [(p, self.device.gradients[p])
       for p in self.network.gparams])
    self.rate = T.scalar('r')

  def initConfig(self):
    model = config.value('model', None)
    learning_rate = config.float('learning_rate', 0.01)
    momentum = config.float("momentum", 0.0)
    interval = config.int('save_interval', 1)
    adagrad = config.bool('adagrad', False)
    self.init(learning_rate=learning_rate, model=model, momentum=momentum, interval=interval, adagrad=adagrad)

  def init(self, learning_rate, model, momentum=0.0, interval=1, adagrad=False):
    """ derived from crnn.Engine.train """
    self.modelName = model
    self.interval = interval
    self.batch = 0
    self.learning_rate = learning_rate

    if self.task == "evaluate":
      self.evaluater = theano.function(
        inputs=[],
        outputs=[self.network.output.p_y_given_x],
        updates=[],
        name="Sprint CrnnWrapper evaluation",
        givens=self.device.make_input_givens(self.network),
        no_default_updates=True)

    elif self.task == "train":  # We do training.
      if momentum > 0:
        deltas = dict(
          [(p, theano.shared(value=numpy.zeros(p.get_value().shape, dtype=theano.config.floatX))) for p in
           self.network.gparams])
      if adagrad:
        sqrsum = dict(
          [(p, theano.shared(value=numpy.zeros(p.get_value().shape, dtype=theano.config.floatX))) for p in
           self.network.gparams])
      updates = []
      for param in self.network.gparams:
        upd = - self.rate * self.gparams[param]
        if momentum > 0:
          upd += momentum * deltas[param]
          updates.append((deltas[param], upd))
        if adagrad:
          updates.append((sqrsum[param], sqrsum[param] + self.gparams[param] ** 2))
          upd = upd * 0.1 / (0.1 + (sqrsum[param] + self.gparams[param] ** 2) ** 0.5)
        updates.append((param, param + upd))  # This will directly update the network weights.

      # This is now a combination of crnn.Engine.updater and crnn.Device.trainer.
      self.updater = theano.function(
        inputs=[self.rate],
        outputs=[self.network.cost],
        updates=updates,
        name="Sprint CrnnWrapper updater",
        givens=self.device.make_givens(self.network),
        no_default_updates=True)

    else:
      assert False, "unknown task: %r" % self.task

  def train(self, input_data, target_data):
    """ Derived from crnn.Engine.train() and crnn.Process.run(). """
    assert self.task == "train"
    assert self.updater is not None
    device = self.device
    self.batch += 1

    # Set data. See crnn.Process.allocate_devices().
    # Will be set in device.update_data() via device.run().
    device.data = input_data
    shape = input_data.shape[:-1]  # (T,#Batch)
    if target_data is not None:
      device.targets = target_data
      assert device.targets.shape == shape
      device.index = numpy.ones(shape=shape, dtype='int8')
    else:
      # Pass dummy data. Note that this is only valid if we use the Sprint criterion.
      device.targets = numpy.zeros(shape=(1,1), dtype='int32')
      device.index = numpy.zeros(shape=(1,1), dtype='int8')
    assert device.blocking
    device.update_data()

    # Update params. self.updater uses device.gradients.
    loss, = self.updater(self.learning_rate)
    if target_data is not None:
      # In case of the Sprint criterion, we will log there.
      print >> log.v5, "avg frame loss for segments:", loss.sum() / device.index.sum()

  def evaluate(self, input_data):
    assert self.task == "evaluate"
    assert self.evaluater is not None
    device = self.device
    self.batch += 1

    # Set data. See crnn.Process.allocate_devices().
    # Will be set in device.update_data() via device.run().
    device.data = input_data
    shape = input_data.shape[:-1]  # (T,#Batch)
    device.index = numpy.ones(shape=shape, dtype='int8')
    device.targets = numpy.zeros(shape=(1,1), dtype='int32')  # Dummy data
    assert device.blocking
    device.update_data()

    posteriors, = self.evaluater()
    return posteriors

  def _verbForTask(self):
    if self.task == "train": return "Trained"
    elif self.task == "evaluate": return "Evaluated"
    assert False, self.task

  def finalize(self):
    print >> log.v1, "%s epoch %s with %s mini-batches" % (self._verbForTask(), self.epoch, self.batch)

    if self.modelName and self.task == "train":
      self.save_model()
    else:
      print >> log.v3, "Not saving model (no model name)"

  def save_model(self, filename=None):
    if not filename:
      assert self.modelName
      filename = self.modelName + ".%03d" % self.epoch
    print >> log.v3, "Save model under %s" % filename
    model = h5py.File(filename, "w")
    self.network.save(model, self.epoch)
    model.close()



class Criterion(theano.Op):
  gotPosteriors = Event()
  gotErrorSignal = Event()
  posteriors = None
  error = None
  errorSignal = None

  def __eq__(self, other):
    return type(self) == type(other)

  def __hash__(self):
    return hash(type(self))

  def __str__(self):
    return self.__class__.__name__

  def make_node(self, posteriors, seq_lengths):
    # We get the posteriors here from the Network output function,
    # which should be softmax.
    posteriors = theano.tensor.as_tensor_variable(posteriors)
    seq_lengths = theano.tensor.as_tensor_variable(seq_lengths)
    assert seq_lengths.ndim == 1  # vector of seqs lengths
    return theano.Apply(op=self, inputs=[posteriors, seq_lengths], outputs=[T.fvector(), posteriors.type()])

  def perform(self, node, inputs, outputs):
    posteriors, seq_lengths = inputs
    nTimeFrames = posteriors.shape[0]
    seq_lengths = numpy.array([nTimeFrames])  # TODO: fix or so?

    self.__class__.posteriors = posteriors
    self.gotPosteriors.set()

    if numpy.isnan(posteriors).any():
      print >> log.v1, 'posteriors contain NaN!'
    if numpy.isinf(posteriors).any():
      print >> log.v1, 'posteriors contain Inf!'
      numpy.set_printoptions(threshold=numpy.nan)
      print >> log.v1, 'posteriors:', posteriors

    self.gotErrorSignal.wait()
    loss, errsig = self.error, self.errorSignal
    assert errsig.shape[0] == nTimeFrames

    outputs[0][0] = loss
    outputs[1][0] = errsig

    print >> log.v5, 'avg frame loss for segments:', loss.sum() / seq_lengths.sum(),
    print >> log.v5, 'time-frames:', seq_lengths.sum()


# HACK for now.
import crnn.SprintErrorSignals
import crnn.Network

crnn.SprintErrorSignals.SprintErrorSigOp = Criterion
crnn.Network.SprintErrorSigOp = Criterion


def demo():
  print "Note: Load this module via Sprint python-trainer to really use it."
  print "We are running a demo now."
  init(inputDim=493, outputDim=4501, config="",  # hardcoded, just a demo...
       targetMode="criterion-by-sprint", cudaEnabled=False, cudaActiveGpu=-1)
  assert os.path.exists("input-features.npy"), "run Sprint with python-trainer=dump first"
  features = numpy.load("input-features.npy")  # dumped via dump.py
  posteriors = feedInput(features)
  if not os.path.exists("posteriors.npy"):
    numpy.save("posteriors.npy", posteriors)
    print "Saved posteriors.npy. Now run Sprint with python-trainer=dump again."
    sys.exit()
  old_posteriors = numpy.load("posteriors.npy")
  assert numpy.array_equal(posteriors, old_posteriors)
  error = numpy.load("output-error.npy")  # dumped via dump.py
  error = float(error)
  errorSignal = numpy.load("output-error-signal.npy")  # dumped via dump.py
  finishError(error=error, errorSignal=errorSignal, naturalPairingType="softmax")
  exit()

if __name__ == "__main__":
  demo()
