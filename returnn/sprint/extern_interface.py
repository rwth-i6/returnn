"""
This is a Sprint interface implementation.
See SprintInterface.py for another Sprint interface.
This Sprint interface is to be used for ExternSprintDataset, which should automatically use it.
"""

from __future__ import annotations

import sys
import os
import typing
from returnn.util import better_exchook
import returnn.util.task_system as task_system
from returnn.util.task_system import Pickler
from returnn.util.basic import to_bool, unicode, BytesIO

# Start Sprint PythonSegmentOrder interface. {
# We use the PythonSegmentOrder just to get an estimate (upper limit) about the number of sequences.

segmentOrderList = None  # type: typing.Optional[typing.List[str]]


# Cannot change name, this need to stay like this for compatibility.
# noinspection PyPep8Naming,PyUnusedLocal
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
InputDim = None  # type: typing.Optional[int]
OutputDim = None  # type: typing.Optional[int]


def exchook(exc_type, exc_obj, exc_tb):
    """
    Replacement for sys.excepthook.
    """
    if exc_type is KeyboardInterrupt:
        print("SprintExternInterface[pid %i]: KeyboardInterrupt" % (os.getpid(),))
        sys.exit(1)
    better_exchook.better_exchook(exc_type, exc_obj, exc_tb)


def init(**kwargs):
    """
    Called by Sprint when it initializes the PythonTrainer,
    or also for PythonControl.
    Set trainer = python-trainer in Sprint to enable PythonTrainer for the Sprint nn-trainer tool.
    Note that Sprint will call this, i.e. the trainer init lazily quite late,
    only once it sees the first data.

    :param kwargs: all passed to :func:`_init_python_trainer` or :func:`PythonControl.init`
    :rtype: None|PythonControl
    """
    sys.excepthook = exchook
    # This module can also be used for Sprint PythonControl, which will also call init().
    # We need to catch these cases.
    if "name" in kwargs and kwargs["name"] == "Sprint.PythonControl":
        return PythonControl.init(**kwargs)
    return _init_python_trainer(**kwargs)


def _parse_config_str(config_str):
    assert isinstance(config_str, (str, unicode))
    config_list = config_str.split(",")
    config = {key: value for (key, value) in [s.split(":", 1) for s in config_list if s]}
    return config


def _common_init(config):
    if (
        to_bool(config.get("EnableAutoNumpySharedMemPickling", False))
        and not task_system.SharedMemNumpyConfig["enabled"]
    ):
        task_system.SharedMemNumpyConfig["enabled"] = True
        print("SprintExternInterface[pid %i] EnableAutoNumpySharedMemPickling = True" % (os.getpid(),))


# Cannot change argument names because of compatibility, as these are passed as-is from Sprint.
# noinspection PyPep8Naming
def _init_python_trainer(inputDim, outputDim, config, targetMode, **kwargs):
    """
    :type inputDim: int
    :type outputDim: int
    :param str config: config string, passed by Sprint. assumed to be ","-separated
    :param str targetMode: "target-alignment" or "criterion-by-sprint" or so
    """
    print("SprintExternInterface[pid %i]: PythonTrainer init_PythonTrainer()" % (os.getpid(),))
    print("inputDim:", inputDim)
    print("outputDim:", outputDim)
    print("config:", config)
    print("targetMode:", targetMode)
    print("other args:", kwargs)

    global InputDim, OutputDim, isInitialized
    InputDim = inputDim
    OutputDim = outputDim
    isInitialized = True
    assert targetMode != "criterion-by-sprint"
    config = _parse_config_str(config)
    assert config["action"] == "ExternSprintDataset"
    _common_init(config)

    _init_global_sprint_dataset(input_dim=inputDim, output_dim=outputDim, config=config)


sprintDataset = None  # type: typing.Optional[ExternSprintDatasetSource]


def _init_global_sprint_dataset(input_dim, output_dim, config):
    global sprintDataset
    if sprintDataset:
        return
    num_segments = len(segmentOrderList) if segmentOrderList is not None else None
    sprintDataset = ExternSprintDatasetSource(
        c2p_fd=int(config["c2p_fd"]),
        p2c_fd=int(config["p2c_fd"]),
        input_dim=input_dim,
        output_dim=output_dim,
        num_segments=num_segments,
    )


# Name need to stay like this, for compatibility.
# noinspection PyShadowingBuiltins
def exit():
    """
    Called by Sprint, to signal that it is exiting.
    """
    print("SprintExternInterface: PythonTrainer exit()")
    assert isInitialized
    sprintDataset.close()


# Name need to stay like this, for compatibility.
# noinspection PyPep8Naming
def feedInput(features, weights=None, segmentName=None):
    """
    Called by Sprint.
    Unsupervised case.

    :param numpy.ndarray features:
    :param numpy.ndarray|None weights:
    :param str|None segmentName:
    """
    feedInputAndTarget(features=features, weights=weights, segmentName=segmentName)


# Name need to stay like this, for compatibility.
# noinspection PyPep8Naming
def feedInputAndTargetAlignment(features, targetAlignment, weights=None, segmentName=None):
    """
    :param numpy.ndarray features:
    :param numpy.ndarray targetAlignment:
    :param numpy.ndarray|None weights:
    :param str|None segmentName:
    """
    feedInputAndTarget(features=features, alignment=targetAlignment, weights=weights, segmentName=segmentName)


# Name need to stay like this, for compatibility.
# noinspection PyPep8Naming
def feedInputAndTargetSegmentOrth(features, targetSegmentOrth, weights=None, segmentName=None):
    """
    :param numpy.ndarray features:
    :param str targetSegmentOrth:
    :param numpy.ndarray|None weights:
    :param str|None segmentName:
    """
    feedInputAndTarget(features=features, orthography=targetSegmentOrth, weights=weights, segmentName=segmentName)


feedInputUnsupervised = feedInput


# Name/params need to stay like this, for compatibility.
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
    """
    assert features.shape[0] == InputDim
    targets = {}
    if alignment is not None:
        targets["classes"] = alignment
    if orthography is not None:
        targets["orth"] = orthography
    sprintDataset.add_new_data(segment_name=segmentName, features=features, targets=targets)


# End Sprint PythonTrainer interface. }


# Start Sprint PythonControl interface. {


class PythonControl:
    """
    PythonControl, interface for Sprint.
    """

    instance = None

    @classmethod
    def init(cls, **kwargs):
        """
        Called by global init().

        :rtype: PythonControl
        """
        print("SprintExternInterface[pid %i]: PythonControl %s init %r" % (os.getpid(), __file__, kwargs))
        if cls.instance:
            return cls.instance
        cls.instance = cls(**kwargs)
        return cls.instance

    # Maybe other kwargs by Sprint.
    # noinspection PyUnusedLocal
    def __init__(self, config, **kwargs):
        self.config = _parse_config_str(config)
        _common_init(self.config)

    def init_processing(self, input_dim, output_dim, **kwargs):
        """
        Called by Sprint.

        :param int input_dim:
        :param int output_dim:
        :param kwargs: maybe others
        """
        print(
            "SprintExternInterface: PythonControl init_processing inputDim=%i, outputDim=%i, other:%r"
            % (input_dim, output_dim, kwargs)
        )
        _init_global_sprint_dataset(input_dim=input_dim, output_dim=output_dim, config=self.config)

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def process_segment(self, name, orthography, features, alignment, soft_alignment, speaker_name=None, **kwargs):
        """
        Called by Sprint.

        :param str name:
        :param str|None orthography:
        :param numpy.ndarray features:
        :param numpy.ndarray|None alignment:
        :param numpy.ndarray|None soft_alignment:
        :param str|None speaker_name:
        :param kwargs: maybe others
        """
        assert sprintDataset
        targets = {}
        if orthography is not None:
            targets["orth"] = orthography
        if speaker_name is not None:
            targets["speaker_name"] = speaker_name
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
        sprintDataset.add_new_data(segment_name=name, features=features, targets=targets)

    # noinspection PyMethodMayBeStatic
    def exit(self, **kwargs):
        """
        Called by Sprint.

        :param kwargs:
        """
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

    def __init__(self, c2p_fd, p2c_fd, input_dim, output_dim, num_segments):
        """
        :param int c2p_fd: child-to-parent file descriptor
        :param int p2c_fd: parent-to-child file descriptor
        :type input_dim: int
        :type output_dim: int
        :type num_segments: int | None
        :param num_segments: can be None if not known in advance
        """
        self.pipe_c2p = os.fdopen(c2p_fd, "wb")
        self.pipe_p2c = os.fdopen(p2c_fd, "rb")
        self._send("init", (input_dim, output_dim, num_segments))

    def _send(self, data_type, args=None):
        """
        :param str data_type:
        :param object args:
        """
        assert data_type is not None
        import struct

        stream = BytesIO()
        Pickler(stream).dump((data_type, args))
        raw_data = stream.getvalue()
        assert len(raw_data) > 0
        self.pipe_c2p.write(struct.pack("<i", len(raw_data)))
        self.pipe_c2p.write(raw_data)
        self.pipe_c2p.flush()

    def add_new_data(self, segment_name, features, targets):
        """
        :param str segment_name:
        :param numpy.ndarray features: 2D array, (feature,time)
        :param dict[str,numpy.ndarray] targets: each target is either 1D (time->idx) or 2D (time,class)
        """
        self._send("data", (segment_name, features, targets))

    def close(self):
        """
        Close pipe fds.
        """
        self._send("exit")
        self.pipe_c2p.close()
        self.pipe_p2c.close()


# End Sprint PythonControl interface. }
