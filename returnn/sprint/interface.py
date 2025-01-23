"""
This is a Sprint interface implementation, i.e. you would specify this module in your Sprint config.
(Sprint = the RWTH ASR toolkit.)
Note that there are multiple Sprint interface implementations provided.
This one would be used explicitly, e.g. for forwarding in recognition
or wherever else Sprint needs posteriors (a FeatureScorer).
Most of the other Sprint interfaces will be used automatically,
e.g. via ExternSprintDataset, when it spawns its Sprint subprocess.
"""

from __future__ import annotations

import os
import sys
import time
from threading import Event, Thread
import typing
import numpy

from returnn.datasets.basic import Dataset, init_dataset
from returnn.datasets.sprint import SprintDatasetBase
from returnn.engine.base import EngineBase
from returnn.log import log
import returnn.util.debug as debug
from returnn.util.basic import interrupt_main, to_bool, BackendEngine
import returnn.util.task_system as task_system
import returnn.__main__ as rnn

if typing.TYPE_CHECKING:
    import returnn.tf.engine

_rnn_file = rnn.__file__
_main_file = getattr(sys.modules["__main__"], "__file__", "")
if _rnn_file.endswith(".pyc"):
    _rnn_file = _rnn_file[:-1]
if _main_file.endswith(".pyc"):
    _main_file = _main_file[:-1]
if os.path.realpath(_rnn_file) == os.path.realpath(_main_file):
    rnn = sys.modules["__main__"]

DefaultSprintCrnnConfig = "config/crnn.config"

startTime = None  # type: typing.Optional[float]
isInitialized = False
isTrainThreadStarted = False
trainThread = None  # type: typing.Optional[Thread]
isExited = False
InputDim = None  # type: typing.Optional[int]
OutputDim = None  # type: typing.Optional[int]
MaxSegmentLength = 1
TargetMode = None  # type: typing.Optional[str]
Task = "train"

Engine = None  # type: typing.Optional[typing.Union[typing.Type[returnn.tf.engine.Engine]]]  # nopep8
config = None  # type: typing.Optional[rnn.Config]
sprintDataset = None  # type: typing.Optional[SprintDatasetBase]
customDataset = None  # type: typing.Optional[Dataset]
engine = None  # type: typing.Optional[typing.Union[returnn.tf.engine.Engine]]


# <editor-fold desc="generic init">
# Generic interface, should be compatible to any PythonControl-based, and PythonTrainer. {


def init(name=None, sprint_unit=None, **kwargs):
    """
    This will get called by various Sprint interfaces.
    Depending on `name` and `sprint_unit`, we can figure out which interface it is.
    For all PythonControl-based interfaces, we must return an object which will be used for further callbacks.

    :param str|None name:
    :param str|None sprint_unit:
    :return: some object or None
    :rtype: None|object
    """
    print(
        "RETURNN Python SprintInterface init: name %r, sprint_unit %r, pid %i, kwargs %r"
        % (name, sprint_unit, os.getpid(), kwargs)
    )
    if name is None:
        return init_python_trainer(**kwargs)
    elif name == "Sprint.PythonControl":
        # Any PythonControl interface.
        if sprint_unit == "PythonFeatureScorer":
            return init_python_feature_scorer(**kwargs)
        else:
            raise Exception(
                "SprintInterface: Did not expect init() PythonControl with sprint_unit=%r, kwargs=%r",
                (sprint_unit, kwargs),
            )
    else:
        raise Exception(
            "SprintInterface: Did not expect init() with name=%r, sprint_unit=%r, kwargs=%r",
            (name, sprint_unit, kwargs),
        )


# }
# </editor-fold>


# <editor-fold desc="PythonFeatureScorer">
# Start Sprint PythonFeatureScorer interface. {


# noinspection PyShadowingNames
def init_python_feature_scorer(config, **kwargs):
    """
    :param str config:
    :rtype: PythonFeatureScorer
    """
    sprint_opts = {key: value for (key, value) in [s.split(":", 1) for s in config.split(",") if s]}

    epoch = sprint_opts.get("epoch", None)
    if epoch is not None:
        epoch = int(epoch)
        assert epoch >= 1

    # see init_python_trainer()
    configfile = sprint_opts.get("configfile", None)
    assert sprint_opts.get("action", None) in (None, "forward"), "invalid action: %r" % sprint_opts["action"]

    _init_base(target_mode="forward", configfile=configfile, epoch=epoch, sprint_opts=sprint_opts)

    cls = PythonFeatureScorer
    if rnn.config.has("SprintInterfacePythonFeatureScorer"):
        cls = rnn.config.typed_value("SprintInterfacePythonFeatureScorer")
    return cls(sprint_opts=sprint_opts, **kwargs)


class PythonFeatureScorer:
    """
    Sprint API.
    """

    def __init__(self, callback, version_number, sprint_opts, **kwargs):
        """
        :param (str,)->object callback:
        :param int version_number:
        :param dict[str,str] sprint_opts:
        """
        print(
            "SprintInterface: PythonFeatureScorer(%s): version %i, sprint_opts %r, other %r"
            % (self.__class__.__name__, version_number, sprint_opts, kwargs)
        )
        self.input_dim = None
        self.output_dim = None
        self.callback = callback
        self.sprint_opts = sprint_opts
        self.priors = None  # type: typing.Optional[numpy.ndarray]
        self.segment_count = 0
        self.features = []  # type: typing.List[numpy.ndarray]
        self.scores = None  # type: typing.Optional[numpy.ndarray]

    def init(self, input_dim, output_dim):
        """
        Called by Sprint.

        :param int input_dim:
        :param int output_dim: number of emission classes
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        # see init_python_trainer()
        global InputDim, OutputDim
        InputDim = input_dim
        OutputDim = output_dim
        sprintDataset.set_dimensions(self.input_dim, self.output_dim)
        sprintDataset.initialize()

        _prepare_forwarding()
        self._load_priors()

        global startTime
        startTime = time.time()

    def _load_priors(self):
        """
        This will optionally initialize self.priors of shape (self.output_dim,), in -log space,
        already multiplied by any prior scale.

        :return: nothing
        """
        scale = float(self.sprint_opts["prior_scale"])
        if not scale:
            return
        filename = self.sprint_opts["prior_file"]
        # We expect a filename to the priors, stored as txt, in +log space.
        assert isinstance(filename, str)
        assert os.path.exists(filename)
        from returnn.util.basic import load_txt_vector

        prior = load_txt_vector(filename)  # +log space
        self.priors = -numpy.array(prior, dtype="float32") * numpy.float32(scale)  # -log space
        assert self.priors.shape == (self.output_dim,), "dim mismatch: %r != %i" % (self.priors.shape, self.output_dim)

    # noinspection PyMethodMayBeStatic
    def exit(self):
        """
        Called by Sprint at exit.
        """
        print("SprintInterface: PythonFeatureScorer: exit()")

    # noinspection PyMethodMayBeStatic
    def get_feature_buffer_size(self):
        """
        Called by Sprint.

        :return: -1 -> no limit
        """
        return -1

    # noinspection PyShadowingNames
    def add_feature(self, feature, time):
        """
        Called by Sprint.

        :param numpy.ndarray feature: shape (input_dim,)
        :param int time:
        """
        assert time == len(self.features)
        assert feature.shape == (self.input_dim,)
        self.features.append(feature)

    def reset(self, num_frames):
        """
        Called by Sprint.
        Called when we shall flush any buffers.

        :param int num_frames:
        """
        if num_frames > 0:
            self.segment_count += 1
        assert num_frames == len(self.features)
        del self.features[:]
        self.scores = None

    def get_segment_name(self):
        """
        :rtype: str
        """
        return "unknown-seq-name-%i" % self.segment_count

    def get_features(self, num_frames=None):
        """
        :param int|None num_frames:
        :return: shape (input_dim, num_frames)
        :rtype: numpy.ndarray
        """
        if num_frames is not None:
            assert 0 < num_frames == len(self.features)
        return numpy.stack(self.features, axis=1)

    def get_posteriors(self, num_frames=None):
        """
        :param int|None num_frames:
        :return: shape (output_dim, num_frames)
        :rtype: numpy.ndarray
        """
        if num_frames is None:
            num_frames = len(self.features)
        assert 0 < num_frames == len(self.features)
        posteriors = _forward(segment_name=self.get_segment_name(), features=self.get_features(num_frames=num_frames))
        assert posteriors.shape == (self.output_dim, num_frames)
        return posteriors

    def features_to_dataset(self, num_frames=None):
        """
        :param int|None num_frames:
        :return: (dataset, seq_idx)
        :rtype: (Dataset.Dataset, int)
        """
        segment_name = self.get_segment_name()
        features = self.get_features(num_frames=num_frames)
        return features_to_dataset(features=features, segment_name=segment_name)

    @property
    def engine(self):
        """
        :rtype: TFEngine.Engine|Engine.Engine
        """
        return rnn.engine

    @property
    def config(self):
        """
        :rtype: returnn.config.Config
        """
        return rnn.config

    def compute(self, num_frames):
        """
        Called by Sprint.
        All the features which we received so far should be evaluated.

        :param int num_frames:
        """
        assert 0 < num_frames == len(self.features)
        posteriors = self.get_posteriors(num_frames=num_frames)
        assert posteriors.shape == (self.output_dim, num_frames)
        scores = -numpy.log(posteriors)  # transfer to -log space
        if self.priors is not None:
            scores -= numpy.expand_dims(self.priors, axis=1)
        # We must return in -log space.
        self.scores = scores

    # noinspection PyShadowingNames
    def get_scores(self, time):
        """
        Called by Sprint.

        :param int time:
        :return: shape (output_dim,)
        :rtype: numpy.ndarray
        """
        # print("get scores, time", time, "max_frames", self.scores.shape[1])
        return self.scores[:, time]


# }
# </editor-fold>


# <editor-fold desc="PythonSegmentOrder">
# Start Sprint PythonSegmentOrder interface. {


# Names need to stay that way for compatibility.
# noinspection PyPep8Naming
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
    print("Sprint: getSegmentList(%r)" % corpusName)
    print("Corpus segments #: %i" % len(segmentList))
    print("(This can be further filtered in Sprint by a whitelist or so.)")

    # Init what we need. These can be called multiple times.
    # If we use both the PythonSegmentOrder and the PythonTrainer, this will be called first.
    # The PythonTrainer will be called lazily once it gets the first data.
    _init_base(configfile=kwargs.get("config", None))
    sprintDataset.use_multiple_epochs()

    finalEpoch = get_final_epoch()
    startEpoch = Engine.get_train_start_epoch(config)
    print("Sprint: Starting with epoch %i." % (startEpoch,))
    print("Final epoch is: %i" % finalEpoch)

    # Loop over multiple epochs. Epochs start at 1.
    for curEpoch in range(startEpoch, finalEpoch + 1):
        if isTrainThreadStarted:
            # So that the RETURNN train thread always has the SprintDatasetBase in a sane state before we reset it.
            sprintDataset.wait_for_returnn_epoch(curEpoch)
        sprintDataset.init_sprint_epoch(curEpoch)

        index_list = sprintDataset.get_seq_order_for_epoch(curEpoch, len(segmentList))
        orderedSegmentList = [segmentList[i] for i in index_list]
        assert len(orderedSegmentList) == len(segmentList)

        print("Sprint epoch: %i" % curEpoch)
        startSegmentIdx = 0
        if curEpoch == startEpoch:
            startSegmentIdx = startSegmentIdx
        for curSegmentIdx in range(startSegmentIdx, len(orderedSegmentList)):
            sprintDataset.set_complete_frac(
                float(curSegmentIdx - startSegmentIdx + 1) / (len(orderedSegmentList) - startSegmentIdx)
            )
            yield orderedSegmentList[curSegmentIdx]

        print("Sprint finished epoch %i" % curEpoch)
        sprintDataset.finish_sprint_epoch()

        if isTrainThreadStarted:
            assert sprintDataset.get_num_timesteps() > 0, (
                "We did not received any seqs. You are probably using a buffered feature extractor and the buffer is "
                + "bigger than the total number of time frames in the corpus."
            )

    sprintDataset.finalize_sprint()


# End Sprint PythonSegmentOrder interface. }
# </editor-fold>


# <editor-fold desc="PythonTrainer">
# Start Sprint PythonTrainer interface. {


# noinspection PyPep8Naming,PyShadowingNames
def init_python_trainer(inputDim, outputDim, config, targetMode, **kwargs):
    """
    Called by Sprint when it initializes the PythonTrainer.
    Set trainer = python-trainer in Sprint to enable.
    Note that Sprint will call this, i.e. the trainer init lazily quite late,
    only once it sees the first data.

    :type inputDim: int
    :type outputDim: int
    :param str config: config string, passed by Sprint. assumed to be ","-separated
    :param str targetMode: "target-alignment" or "criterion-by-sprint" or so
    :return: not expected to return anything
    :rtype: None
    """
    print("SprintInterface[pid %i] init()" % (os.getpid(),))
    print("inputDim:", inputDim)
    print("outputDim:", outputDim)
    print("config:", config)
    print("targetMode:", targetMode)
    print("other args:", kwargs)
    global InputDim, OutputDim, MaxSegmentLength
    InputDim = inputDim
    OutputDim = outputDim

    MaxSegmentLength = kwargs.get("maxSegmentLength", MaxSegmentLength)

    config = config.split(",")
    config = {key: value for (key, value) in [s.split(":", 1) for s in config if s]}

    if (
        to_bool(config.get("EnableAutoNumpySharedMemPickling", False))
        and not task_system.SharedMemNumpyConfig["enabled"]
    ):
        task_system.SharedMemNumpyConfig["enabled"] = True
        print("SprintInterface[pid %i] EnableAutoNumpySharedMemPickling = True" % (os.getpid(),))

    epoch = config.get("epoch", None)
    if epoch is not None:
        epoch = int(epoch)
        assert epoch >= 1

    configfile = config.get("configfile", None)

    global Task
    action = config["action"]
    Task = action
    if action == "train":
        pass
    elif action == "forward":
        assert targetMode in ["criterion-by-sprint", "forward-only"]
        targetMode = "forward"
    elif action == "nop":
        targetMode = None
    else:
        assert False, "unknown action: %r" % action

    _init_base(target_mode=targetMode, configfile=configfile, epoch=epoch, sprint_opts=config)
    sprintDataset.set_dimensions(inputDim, outputDim)
    sprintDataset.initialize()

    if Task == "train":
        _start_train_thread(epoch)
    elif Task == "forward":
        _prepare_forwarding()

    global startTime
    startTime = time.time()


# noinspection PyShadowingBuiltins
def exit():
    """
    Called by Sprint at exit.
    """
    print("SprintInterface[pid %i] exit()" % (os.getpid(),))
    assert isInitialized
    global isExited
    if isExited:
        print("SprintInterface[pid %i] exit called multiple times" % (os.getpid(),))
        return
    isExited = True
    if isTrainThreadStarted:
        engine.stop_train_after_epoch_request = True
        sprintDataset.finish_sprint_epoch()  # In case this was not called yet. (No PythonSegmentOrdering.)
        sprintDataset.finalize_sprint()  # In case this was not called yet. (No PythonSegmentOrdering.)
        trainThread.join()
    rnn.finalize()
    if startTime:
        print("SprintInterface[pid %i]: elapsed total time: %f" % (os.getpid(), time.time() - startTime), file=log.v3)
    else:
        print("SprintInterface[pid %i]: finished (unknown start time)" % os.getpid(), file=log.v3)


# Keep name for compatibility.
# noinspection PyPep8Naming,PyUnusedLocal
def feedInput(features, weights=None, segmentName=None):
    """
    :param numpy.ndarray features:
    :param numpy.array|None weights:
    :param str|None segmentName:
    :return: posteriors
    :rtype: numpy.ndarray
    """
    assert features.shape[0] == InputDim
    if Task == "train":
        posteriors = _train(segmentName, features)
    elif Task == "forward":
        posteriors = _forward(segmentName, features)
    else:
        assert False, "invalid task: %r" % Task
    assert posteriors.shape == (OutputDim * MaxSegmentLength, features.shape[1])
    return posteriors


# Keep name for compatibility.
# noinspection PyPep8Naming
def finishDiscard():
    """
    Discard some segment.
    """
    print("finishDiscard()")
    raise NotImplementedError  # TODO ...


# Keep name for compatibility.
# noinspection PyPep8Naming
def finishError(error, errorSignal, naturalPairingType=None):
    """
    :param float|numpy.ndarray error:
    :param numpy.ndarray errorSignal:
    :param str|None naturalPairingType: must be "softmax"
    :return: nothing
    """
    make_criterion_class()
    assert naturalPairingType == "softmax"
    assert Task == "train"
    # reformat. see train()
    error = numpy.array([error], dtype="float32")
    errorSignal = errorSignal.transpose()
    errorSignal = errorSignal[:, numpy.newaxis, :]
    errorSignal = numpy.array(errorSignal, dtype="float32")
    assert errorSignal.shape == Criterion.posteriors.shape

    Criterion.error = error
    Criterion.errorSignal = errorSignal
    Criterion.gotErrorSignal.set()


# Keep name for compatibility.
# noinspection PyPep8Naming,PyUnusedLocal
def feedInputAndTarget(
    features,
    weights=None,
    segmentName=None,
    orthography=None,
    alignment=None,
    speaker_name=None,
    speaker_gender=None,
    **kwargs,
):
    """
    :param numpy.ndarray features:
    :param numpy.ndarray|None weights:
    :param str|None segmentName:
    :param str|None orthography:
    :param numpy.ndarray|None alignment:
    :param str|None speaker_name:
    :param str|None speaker_gender:
    :return: nothing
    """
    assert features.shape[0] == InputDim
    targets = {}
    if alignment is not None:
        targets["classes"] = alignment
    if orthography is not None:
        targets["orth"] = orthography
    _train(segmentName, features, targets)


# Keep name for compatibility.
# noinspection PyPep8Naming,PyUnusedLocal
def feedInputAndTargetAlignment(features, targetAlignment, weights=None, segmentName=None):
    """
    :param numpy.ndarray features:
    :param numpy.ndarray targetAlignment:
    :param numpy.ndarray|None weights:
    :param str|None segmentName:
    :return: nothing
    """
    assert features.shape[0] == InputDim
    assert Task == "train"
    _train(segmentName, features, targetAlignment)


# Keep name for compatibility.
# noinspection PyPep8Naming,PyUnusedLocal
def feedInputAndTargetSegmentOrth(features, targetSegmentOrth, weights=None, segmentName=None):
    """
    :param numpy.ndarray features:
    :param str targetSegmentOrth:
    :param numpy.ndarray|None weights:
    :param segmentName:
    :return:
    """
    assert features.shape[0] == InputDim
    assert Task == "train"
    _train(segmentName, features, {"orth": targetSegmentOrth})


# Keep name for compatibility.
# noinspection PyPep8Naming,PyUnusedLocal
def feedInputUnsupervised(features, weights=None, segmentName=None):
    """
    :param numpy.ndarray features:
    :param numpy.ndarray|None weights:
    :param str|None segmentName:
    :return: nothing
    """
    assert features.shape[0] == InputDim
    _train(segmentName, features)


# Keep name for compatibility.
# noinspection PyPep8Naming
def feedInputForwarding(features, weights=None, segmentName=None):
    """
    :param numpy.ndarray features:
    :param numpy.ndarray|None weights:
    :param str|None segmentName:
    :rtype: numpy.ndarray
    """
    assert Task == "forward"
    return feedInput(features, weights=weights, segmentName=segmentName)


# End Sprint PythonTrainer interface. }
# </editor-fold>


def dump_flags():
    """
    Dump some relevant env flags.
    """
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("CUDA_LAUNCH_BLOCKING:", os.environ.get("CUDA_LAUNCH_BLOCKING"))


def set_target_mode(mode):
    """
    :param str mode: target mode
    """
    global TargetMode
    assert config, "not initialized"
    TargetMode = mode
    task = "train"
    loss = config.value("loss", None)
    if TargetMode == "criterion-by-sprint":
        assert loss == "sprint", "TargetMode is %s but loss is %s" % (TargetMode, loss)
    elif TargetMode == "target-alignment":
        # RETURNN always expects an alignment, so this is good just as-is.
        # This means that we will not calculate the criterion in Sprint.
        assert loss != "sprint", "invalid loss %s for target mode %s" % (loss, TargetMode)
    elif TargetMode == "forward":
        # Will be handled below.
        task = "forward"
        config.set("extract", ["posteriors"])
    else:
        assert False, "target-mode %s not supported yet..." % TargetMode

    if engine:
        # If we already initialized the engine, the value must not differ,
        # because e.g. Devices will init accordingly.
        orig_task = config.value("task", "train")
        assert orig_task == task

    config.set("task", task)


def _at_exit_handler():
    if not isExited:
        print("SprintInterface[pid %i] atexit handler, exit() was not called, calling it now" % (os.getpid(),))
        exit()
        print("All threads:")
        from returnn.util.debug import dump_all_thread_tracebacks

        dump_all_thread_tracebacks(exclude_self=True)


def _init_base(configfile=None, target_mode=None, epoch=None, sprint_opts=None):
    """
    :param str|None configfile: filename, via init(), this is set
    :param str|None target_mode: "forward" or so. via init(), this is set
    :param int epoch: via init(), this is set
    :param dict[str,str]|None sprint_opts: optional parameters to override values in configfile
    """
    global isInitialized
    isInitialized = True
    # Run through in any case. Maybe just to set targetMode.

    if not getattr(sys, "argv", None):
        # Set some dummy. Some code might want this (e.g. TensorFlow).
        sys.argv = [__file__]

    global Engine
    global config
    if not config:
        # Some subset of what we do in rnn.init().

        rnn.init_better_exchook()

        if configfile is None:
            configfile = DefaultSprintCrnnConfig
        assert os.path.exists(configfile)
        rnn.init_config(config_filename=configfile, extra_updates={"task": target_mode})
        assert rnn.config
        config = rnn.config
        if sprint_opts is not None:
            config.update(sprint_opts)

        rnn.init_log()
        rnn.returnn_greeting(config_filename=configfile)
        rnn.init_backend_engine()
        rnn.init_faulthandler(sigusr1_chain=True)

        if BackendEngine.is_tensorflow_selected():
            # Use TFEngine.Engine class instead of Engine.Engine.
            from returnn.tf.engine import Engine

        import atexit

        atexit.register(_at_exit_handler)

    if target_mode:
        set_target_mode(target_mode)

    _init_dataset()

    if target_mode and target_mode == "forward" and epoch:
        model_filename = config.value("model", "")
        fns = [
            EngineBase.epoch_model_filename(model_filename, epoch, is_pretrain=is_pretrain)
            for is_pretrain in [False, True]
        ]
        fn_postfix = ""
        if BackendEngine.is_tensorflow_selected():
            fn_postfix += ".meta"
        fns_existing = [fn for fn in fns if os.path.exists(fn + fn_postfix)]
        assert len(fns_existing) == 1, "%s not found" % fns
        model_epoch_filename = fns_existing[0]
        config.set("load", model_epoch_filename)
        assert EngineBase.get_epoch_model(config)[1] == model_epoch_filename, "%r != %r" % (
            EngineBase.get_epoch_model(config),
            model_epoch_filename,
        )

    global engine
    if not engine:
        rnn.print_task_properties()
        rnn.init_engine()
        engine = rnn.engine
        assert isinstance(engine, Engine)


def _start_train_thread(epoch=None):
    global config, engine, isInitialized, isTrainThreadStarted
    assert isInitialized, "need to call init() first"
    assert not isTrainThreadStarted
    assert sprintDataset, "need to call initDataset() first"
    assert Task == "train"

    def train_thread_func():
        """
        Train thread.
        """
        # noinspection PyBroadException
        try:
            assert TargetMode
            if TargetMode == "target-alignment":
                pass  # Ok.
            elif TargetMode == "criterion-by-sprint":
                # TODO ...
                raise NotImplementedError
            else:
                raise Exception("target-mode not supported: %s" % TargetMode)

            engine.init_train_from_config(config, train_data=sprintDataset)

            # If some epoch is explicitly specified, it checks whether it matches.
            if epoch is not None:
                assert epoch == engine.start_epoch

            # Do the actual training.
            engine.train()

        except KeyboardInterrupt:  # This happens at forced exit.
            pass

        except BaseException:  # Catch all, even SystemExit. We must stop the main thread then.
            try:
                print("RETURNN train failed")
                sys.excepthook(*sys.exc_info())
            finally:
                # Exceptions are fatal. Stop now.
                interrupt_main()

    global trainThread
    trainThread = Thread(target=train_thread_func, name="Sprint RETURNN train thread")
    trainThread.daemon = True  # However, at clean exit(), will will join this thread.
    trainThread.start()

    isTrainThreadStarted = True


def _prepare_forwarding():
    assert engine
    assert config
    # Should already be set via setTargetMode().
    assert config.list("extract") == [
        "posteriors"
    ], "You need to have extract = posteriors in your RETURNN config. You have: %s" % config.list("extract")

    # Load network.
    engine.init_network_from_config(config)


def _init_dataset():
    global sprintDataset, customDataset
    if sprintDataset:
        return
    assert config
    extra_opts = config.typed_value("sprint_interface_dataset_opts", {})
    assert isinstance(extra_opts, dict)
    sprintDataset = SprintDatasetBase.from_config(config, **extra_opts)
    if config.is_true("sprint_interface_custom_dataset"):
        custom_dataset_func = config.typed_value("sprint_interface_custom_dataset")
        assert callable(custom_dataset_func)
        custom_dataset_opts = custom_dataset_func(sprint_dataset=sprintDataset)
        customDataset = init_dataset(custom_dataset_opts)


def get_final_epoch():
    """
    :rtype: int
    """
    global config, engine
    assert engine
    assert config
    config_num_epochs = engine.config_get_final_epoch(config)
    return config_num_epochs


def _train(segment_name, features, targets=None):
    """
    :param str|None segment_name: full name
    :param numpy.ndarray features: 2d array
    :param numpy.ndarray|dict[str,numpy.ndarray|str]|None targets: 2d or 1d array
    """
    assert engine is not None, "not initialized. call initBase()"
    assert sprintDataset

    if sprintDataset.sprint_finalized:
        return
    sprintDataset.add_new_data(features, targets, segment_name=segment_name)

    # The CRNN train thread started via start() will do the actual training.

    if TargetMode == "criterion-by-sprint":

        # TODO...
        make_criterion_class()

        Criterion.gotPosteriors.clear()

        Criterion.gotPosteriors.wait()
        posteriors = Criterion.posteriors
        assert posteriors is not None

        # posteriors is in format (time,batch,emission)
        # assert posteriors.shape[0] == T
        assert posteriors.shape[1] == 1
        assert OutputDim * MaxSegmentLength == posteriors.shape[2]
        assert len(posteriors.shape) == 3
        # reformat to Sprint expected format (emission,time)
        posteriors = posteriors[:, 0, :]
        posteriors = posteriors.transpose()
        assert posteriors.shape[0] == OutputDim * MaxSegmentLength
        # assert posteriors.shape[1] == T
        assert len(posteriors.shape) == 2

        return posteriors


def features_to_dataset(features, segment_name):
    """
    :param numpy.ndarray features: format (input-feature,time) (via Sprint)
    :param str segment_name:
    :return: (dataset, seq-idx)
    :rtype: (Dataset.Dataset, int)
    """
    assert sprintDataset

    # Features are in Sprint format (feature,time).
    num_time = features.shape[1]
    assert features.shape == (InputDim, num_time)

    if customDataset:
        customDataset.finish_epoch()  # reset
        customDataset.init_seq_order(epoch=1, seq_list=[segment_name])

    # Fill the data for the current segment.
    sprintDataset.shuffle_frames_of_nseqs = 0  # We must not shuffle.
    sprintDataset.init_sprint_epoch(None)  # Reset cache. We don't need old seqs anymore.
    sprintDataset.init_seq_order(epoch=1, seq_list=[segment_name])
    seq = sprintDataset.add_new_data(features, segment_name=segment_name)

    if not customDataset:
        return sprintDataset, seq

    assert seq == 0
    return customDataset, 0


def _forward(segment_name, features):
    """
    :param numpy.ndarray features: format (input-feature,time) (via Sprint)
    :return: format (output-dim,time)
    :rtype: numpy.ndarray
    """
    print("Sprint forward", segment_name, features.shape)
    start_time = time.time()
    assert engine is not None, "not initialized"
    assert sprintDataset

    # Features are in Sprint format (feature,time).
    num_time = features.shape[1]
    assert features.shape == (InputDim, num_time)
    dataset, seq_idx = features_to_dataset(features=features, segment_name=segment_name)

    if BackendEngine.is_tensorflow_selected():
        posteriors = engine.forward_single(dataset=dataset, seq_idx=seq_idx)
    else:
        raise NotImplementedError("unknown backend engine")
    # If we have a sequence training criterion, posteriors might be in format (time,seq|batch,emission).
    if posteriors.ndim == 3:
        assert posteriors.shape == (num_time, 1, OutputDim * MaxSegmentLength)
        posteriors = posteriors[:, 0]
    # Posteriors are in format (time,emission).
    assert posteriors.shape == (num_time, OutputDim * MaxSegmentLength)
    # Reformat to Sprint expected format (emission,time).
    posteriors = posteriors.transpose()
    assert posteriors.shape == (OutputDim * MaxSegmentLength, num_time)
    stats = (numpy.min(posteriors), numpy.max(posteriors), numpy.mean(posteriors), numpy.std(posteriors))
    print("posteriors min/max/mean/std:", stats, "time:", time.time() - start_time)
    if numpy.isinf(posteriors).any() or numpy.isnan(posteriors).any():
        print("posteriors:", posteriors)
        debug_feat_fn = "/tmp/returnn.pid%i.sprintinterface.debug.features.txt" % os.getpid()
        debug_post_fn = "/tmp/returnn.pid%i.sprintinterface.debug.posteriors.txt" % os.getpid()
        print("Wrote to files %s, %s" % (debug_feat_fn, debug_post_fn))
        numpy.savetxt(debug_feat_fn, features)
        numpy.savetxt(debug_post_fn, posteriors)
        assert False, "Error, posteriors contain invalid numbers."

    return posteriors


Criterion = None  # type: typing.Optional[typing.Any]


def make_criterion_class():
    """
    Make Criterion Theano class.
    """
    global Criterion
    if Criterion:
        return

    # noinspection PyUnresolvedReferences,PyPackageRequirements
    import theano

    # noinspection PyUnresolvedReferences,PyPackageRequirements,PyPep8Naming
    import theano.tensor as T

    # noinspection PyAbstractClass
    class Criterion(theano.Op):
        """
        Criterion Theano Op.
        """

        gotPosteriors = Event()
        gotErrorSignal = Event()
        posteriors = None
        error = None
        errorSignal = None

        def __eq__(self, other):
            return type(self) == type(other)  # nopep8

        def __hash__(self):
            return hash(type(self))

        def __str__(self):
            return self.__class__.__name__

        def make_node(self, posteriors, seq_lengths):
            """
            :param numpy.ndarray posteriors:
            :param numpy.ndarray seq_lengths:
            :return:
            """
            # We get the posteriors here from the Network output function,
            # which should be softmax.
            posteriors = theano.tensor.as_tensor_variable(posteriors)
            seq_lengths = theano.tensor.as_tensor_variable(seq_lengths)
            assert seq_lengths.ndim == 1  # vector of seqs lengths
            return theano.Apply(op=self, inputs=[posteriors, seq_lengths], outputs=[T.fvector(), posteriors.type()])

        # noinspection PyUnusedLocal
        def perform(self, node, inputs, output_storage, params=None):
            """
            :param node:
            :param inputs:
            :param output_storage:
            :param params:
            :return:
            """
            posteriors, seq_lengths = inputs
            num_time_frames = posteriors.shape[0]
            seq_lengths = numpy.array([num_time_frames])  # TODO: fix or so?

            self.__class__.posteriors = posteriors
            self.gotPosteriors.set()

            if numpy.isnan(posteriors).any():
                print("posteriors contain NaN!", file=log.v1)
            if numpy.isinf(posteriors).any():
                print("posteriors contain Inf!", file=log.v1)
                numpy.set_printoptions(threshold=numpy.nan)
                print("posteriors:", posteriors, file=log.v1)

            self.gotErrorSignal.wait()
            loss, errsig = self.error, self.errorSignal
            assert errsig.shape[0] == num_time_frames

            output_storage[0][0] = loss
            output_storage[1][0] = errsig

            print("avg frame loss for segments:", loss.sum() / seq_lengths.sum(), end=" ", file=log.v5)
            print("time-frames:", seq_lengths.sum(), file=log.v5)


def demo():
    """
    Demo for SprintInterface.
    """
    print("Note: Load this module via Sprint python-trainer to really use it.")
    print("We are running a demo now.")
    init(
        inputDim=493,
        outputDim=4501,
        config="",  # hardcoded, just a demo...
        targetMode="criterion-by-sprint",
        cudaEnabled=False,
        cudaActiveGpu=-1,
    )
    assert os.path.exists("input-features.npy"), "run Sprint with python-trainer=dump first"
    features = numpy.load("input-features.npy")  # dumped via dump.py
    posteriors = feedInput(features)
    if not os.path.exists("posteriors.npy"):
        numpy.save("posteriors.npy", posteriors)
        print("Saved posteriors.npy. Now run Sprint with python-trainer=dump again.")
        sys.exit()
    old_posteriors = numpy.load("posteriors.npy")
    assert numpy.array_equal(posteriors, old_posteriors)
    error = numpy.load("output-error.npy")  # dumped via dump.py
    error = float(error)
    error_signal = numpy.load("output-error-signal.npy")  # dumped via dump.py
    finishError(error=error, errorSignal=error_signal, naturalPairingType="softmax")
    exit()


if __name__ == "__main__":
    debug.debug_shell(user_ns=locals(), user_global_ns=globals())
