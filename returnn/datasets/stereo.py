"""
This file contains dataset implementations to have an easy to use
interface for using RETURNN for regression.
Applications are for example speech enhancement or mask estimations
"""

from __future__ import annotations

__author__ = "menne"

import os
import numpy as np
from collections import deque
from .cached2 import CachedDataset2
from returnn.datasets.basic import DatasetSeq
from .bundle_file import BundleFile
from .normalization_data import NormalizationData
from returnn.log import log


class StereoDataset(CachedDataset2):
    """The purpose of this dataset is to be a base dataset for datasets which
    have an easy to use interface for using RETURNN as a regression tool
    """

    def __init__(self, partition_epoch=1, **kwargs):
        """constructor"""
        super(StereoDataset, self).__init__(**kwargs)
        self._seq_index_list = None
        self._partition_epoch = partition_epoch
        self._current_partition = 0
        self._seqs_per_epoch = None

    def initialize(self):
        self._seq_overhead = self._get_total_number_of_sequences() % self._partition_epoch
        super(StereoDataset, self).initialize()

    @property
    def num_seqs(self):
        """returns the number of sequences of the dataset

        :rtype: int
        """
        if self._num_seqs is not None:
            return self._num_seqs
        raise NotImplementedError

    def _get_total_number_of_sequences(self):
        raise NotImplementedError

    @property
    def seqs_per_epoch(self):
        if self._seqs_per_epoch is None:
            self._seqs_per_epoch = self._get_total_number_of_sequences() // self._partition_epoch
        return self._seqs_per_epoch

    def _collect_single_seq(self, seq_idx):
        """returns the sequence specified by the index seq_idx

        :type seq_idx: int
        :rtype: DatasetSeq | None
        :returns DatasetSeq or None if seq_idx >= num_seqs.
        """
        raise NotImplementedError

    def _get_partition_size(self, partition):
        partition_size = self.seqs_per_epoch
        if partition == self._partition_epoch - 1:
            partition_size += self._seq_overhead
        return partition_size

    def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
        """
        :type epoch: int|None
        :param epoch: epoch number
        :param list[str]|None seq_list:
        :param list[int]|None seq_order:
        :param seq_list: only None is currently supported
        Initialize lists:
          self.seq_index  # sorted seq idx
        """
        super(StereoDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
        if epoch is None:
            self._seq_index_list = range(self.num_seqs)
            return True

        self._current_partition = (epoch - 1) % self._partition_epoch
        partition_size = self._get_partition_size(self._current_partition)

        if seq_list is not None or seq_order is not None:
            raise NotImplementedError("init_seq_order of StereoDataset does not support a predefined seq_list yet.")
        else:
            seq_index = self.get_seq_order_for_epoch(
                epoch, partition_size, lambda s: self.get_seq_length(s).get("data", None)
            )

        self._seq_index_list = seq_index
        if epoch is not None:
            print("Reinitialize dataset seq order for epoch %i." % epoch, file=log.v4)

        return True


class StereoHdfDataset(StereoDataset):
    """A stereo dataset which needs an hdf file as input. The hdf file
    is supposed to always have group 'inputs' and for the training data it
    also needs to contain the group 'outputs'. Each group is supposed to
    contain one dataset per sequence. The names of the datasets are supposed
    to be consecutive numbers starting at 0.

    The datasets are 2D numpy arrays, where dimension 0 is the time axis and
    dimension 1 is the feature axis. Therefore dimension 0 of the 'input'
    dataset and the respective 'output' dataset need to be the same.
    """

    def __init__(
        self,
        hdfFile,
        num_outputs=None,
        normalizationFile=None,
        flag_normalizeInputs=True,
        flag_normalizeTargets=True,
        **kwargs,
    ):
        """Constructor

        :type hdfFile: str
        :param hdfFile: path to the hdf file. if a bundle file is given (*.bundle)
                        all hdf files listed in the bundle file will be used for
                        the dataset.
                        :see: BundleFile.BundleFile
        :type num_outputs: int
        :param num_outputs: this needs to be set if the stereo data hdf file
                            only contains 'inputs' data (e.g. for the extraction
                            process). Only if no 'outputs' data exists in the hdf
                            file num_outputs is used.
        :type normalizationFile: str | None
        :param normalizationFile: path to a HDF file with normalization data.
                                  The file is optional: if it is not provided then
                                  no normalization is performed.
                                  :see: NormalizationData.NormalizationData
        :type flag_normalizeInputs: bool
        :param flag_normalizeInputs: if True then inputs will be normalized
                                     provided that the normalization HDF file has
                                     necessary datasets (i.e. mean and variance)
        :type flag_normalizeTargets: bool
        :param flag_normalizeTargets: if True then targets will be normalized
                                     provided that the normalization HDF file has
                                     necessary datasets (i.e. mean and variance)
        """
        super(StereoHdfDataset, self).__init__(**kwargs)

        self._flag_normalizeInputs = flag_normalizeInputs
        self._flag_normalizeTargets = flag_normalizeTargets
        # properties of the object which will be set further
        self.num_inputs = None
        self.num_outputs = None
        self._filePaths = None
        self._fileHandlers = None
        self._seqMap = None
        self._normData = None

        if not os.path.isfile(hdfFile):
            raise IOError(hdfFile + " does not exits")
        self._initHdfFileHandlers(hdfFile)

        # set number of sequences in the dataset
        self._num_seqs = self._calculateNumberOfSequences()

        if normalizationFile is not None:
            self._setNormalization(normalizationFile)

        self._setInputAndOutputDimensions(num_outputs)

    def _initHdfFileHandlers(self, hdfFile):
        """Initialize HDF file handlers

        :type hdfFile: str
        :param hdfFile: path to an HDF file with sequences or to a bundle file
                        which should contain one path to an HDF file per line
                        :see: BundleFile.BundleFile
        """
        import h5py

        self._filePaths = []
        self._fileHandlers = []
        if hdfFile.endswith(".bundle"):  # a bundle file containing a list of hdf files is given
            bundle = BundleFile(hdfFile)
            for hdfFilePath in bundle.datasetFilePaths:
                self._filePaths.append(hdfFilePath)
                self._fileHandlers.append(h5py.File(hdfFilePath, "r"))
        else:  # only a single hdf file is given
            self._filePaths.append(hdfFile)
            self._fileHandlers.append(h5py.File(hdfFile, "r"))

    def _calculateNumberOfSequences(self):
        return self.seqs_per_epoch

    def _get_total_number_of_sequences(self):
        """Calculate and return the number of sequences in the dataset.
        This method also initializes a sequences map which maps sequence
        indices into HDF file handlers.

        :rtype: int
        :return: the number of sequences in the dataset
        """
        # initialize a sequence map to map the sequence index
        # from an hdf file into the corresponding
        # hdfFile and hdf-dataset name,
        # but it could e.g. be used for shuffling sequences as well
        self._seqMap = {}
        seqCounter = 0
        for fhIdx, fh in enumerate(self._fileHandlers):
            for k in fh["inputs"].keys():
                self._seqMap[seqCounter] = (fhIdx, k)
                seqCounter += 1
        return seqCounter

    def _setNormalization(self, normalizationFile):
        """Set optional normalization (mean and variance).
        Mean and variance are set only if they are provided.

        :type normalizationFile: string
        :param normalizationFile: path to an HDF normalization file which contains
                                  optional datasets "mean" and "variance".
                                  :see: NormalizationData.NormalizationData
        """
        if not os.path.isfile(normalizationFile):
            raise IOError(normalizationFile + " does not exist")
        self._normData = NormalizationData(normalizationFile)

    def _setInputAndOutputDimensions(self, num_outputs):
        """Set properties which correspond to input and output dimensions.

        :type num_outputs: int
        :param num_outputs: dimensionality of output features. used only if
                            the dataset does not have output features. Or if output
                            features are sparse
        """
        someSequence = self._collect_single_seq(0)
        self.num_inputs = someSequence.get_data("data").shape[1]
        if "outputs" in self._fileHandlers[0]:
            if len(someSequence.get_data("classes").shape) == 1:
                outputFeatDim = 1
            else:
                outputFeatDim = someSequence.get_data("classes").shape[1]
            if outputFeatDim == 1 and num_outputs is not None:
                self.num_outputs = {"classes": (num_outputs, outputFeatDim)}
            else:
                self.num_outputs = {"classes": (outputFeatDim, 2)}
        else:
            # in this case no output data is in the hdf file and
            # therfore the output dimension needs to be given
            # as an argument through the config file
            if num_outputs is None:
                raise ValueError(
                    "if no output data is contained in StereoDataset"
                    " the output dimension has to be specified by num_outputs"
                )
            self.num_outputs = {"classes": (num_outputs, 2)}

    def get_data_dim(self, key):
        """This is copied from CachedDataset2 but the assertion is
        removed (see CachedDataset2.py)

        :type key: str
        :rtype: int
        :return: number of classes, no matter if sparse or not
        """
        if key == "data":
            return self.num_inputs
        if key in self.num_outputs:
            d = self.num_outputs[key][0]
            return d
        self._load_something()
        if len(self.added_data[0].get_data(key).shape) == 1:
            return super(CachedDataset2, self).get_data_dim(key)  # unknown
        assert len(self.added_data[0].get_data(key).shape) == 2
        return self.added_data[0].get_data(key).shape[1]

    def __del__(self):
        """Closes HDF file handlers."""
        for fh in self._fileHandlers:
            try:
                fh.close()
            except Exception:
                pass

    @property
    def num_seqs(self):
        """Returns the number of sequences of the dataset

        :rtype: int
        :return: the number of sequences of the dataset.
        """
        # has been set during initialization of dataset ...
        if self._num_seqs is not None:
            return self._num_seqs
        # ... but for some reason _num_seqs is not set at specific points in the
        # execution of rnn.py therefore the following is a saveguard to fall back on
        self._num_seqs = self._calculateNumberOfSequences()
        return self._num_seqs

    def _collect_single_seq(self, seq_idx):
        """Returns the sequence specified by the index seq_idx.
        Normalization is applied to the input features if mean and variance
        have been specified during dataset creating (see the constructor).

        :type seq_idx: int
        :rtype: DatasetSeq | None
        :returns: None if seq_idx >= num_seqs or the corresponding sequence.
        """
        if self._seq_index_list is None:
            self.init_seq_order()

        if seq_idx >= len(self._seq_index_list):
            return None

        # map the seq_idx to the shuffled sequence indices
        shuf_seq_idx = self._seq_index_list[seq_idx]
        partition_offset = int(np.sum([self._get_partition_size(i1) for i1 in range(self._current_partition)]))
        shuf_seq_idx += partition_offset

        seqMapping = self._seqMap[shuf_seq_idx]
        fileIdx = seqMapping[0]
        datasetName = seqMapping[1]
        fileHandler = self._fileHandlers[fileIdx]
        inputFeatures = fileHandler["inputs"][datasetName][...]
        targets = None
        if "outputs" in fileHandler:
            targets = fileHandler["outputs"][datasetName][...]

        # optional normalization
        if self._normData is not None:
            assert isinstance(self._normData, NormalizationData)
            if self._flag_normalizeInputs:
                inputFeatures = StereoHdfDataset._normalizeVector(
                    inputFeatures, self._normData.inputMean, self._normData.inputVariance
                )
            if self._flag_normalizeTargets:
                targets = StereoHdfDataset._normalizeVector(
                    targets, self._normData.outputMean, self._normData.outputVariance
                )

        # enforce float32 to enable Theano optimizations
        inputFeatures = inputFeatures.astype(np.float32)
        if (targets is not None) and targets.shape[1] > 1:
            targets = targets.astype(np.float32)
        elif targets.shape[1] == 1:
            targets = np.reshape(targets.astype(np.int32), (targets.shape[0],))

        return DatasetSeq(seq_idx, inputFeatures, targets=targets)

    @staticmethod
    def _normalizeVector(v, mean, variance):
        """Helper method.
        Applies optional normalization to the given vector.

        :type v: numpy.ndarray | None
        :param v: vector if available or None otherwise
        :type mean: numpy.ndarray | None
        :param mean: mean
        :type variance: numpy.ndarray | None
        :param variance: variance
        :rtype: numpy.ndarray | None
        :return: normalized vector or None if it was None
        """
        if v is None:
            return v
        if mean is not None:
            v -= mean
        if variance is not None:
            v /= np.sqrt(variance)
        return v


class DatasetWithTimeContext(StereoHdfDataset):
    """This dataset composes a context feature by stacking together time frames."""

    def __init__(self, hdfFile, tau=1, **kwargs):
        """Constructor

        :type hdfFile: string
        :param hdfFile: see the StereoHdfDataset
        :type tau: int
        :param tau: how many time frames should be on the left and on the right.
                    E.g. if tau = 2 then the context feature will be created
                    by stacking two neighboring time frames from left and
                    two neighboring time frames from right:
                    newInputFeature = [ x_{t-2}, x_{t-1}, x_t, x_{t+1}, x_{t+2} ].
                    In general new feature will have shape
                    (2 * tau + 1) * originalFeatureDimensionality
                    Output features are not changed.
        :type kwargs: dictionary
        :param kwargs: the rest of the arguments passed to the StereoHdfDataset
        """
        if tau <= 0:
            raise ValueError("context parameter tau should be greater than zero")
        self._tau = tau
        super(DatasetWithTimeContext, self).__init__(hdfFile, **kwargs)

    def _collect_single_seq(self, seq_idx):
        """this method implements stacking the features

        :type seq_idx: int
        :param seq_idx: index of a sequence
        :rtype: DatasetSeq
        :return: DatasetSeq
        """
        if seq_idx >= self.num_seqs:
            return None
        originalSeq = super(DatasetWithTimeContext, self)._collect_single_seq(seq_idx)
        inputFeatures = originalSeq.get_data("data")
        frames, bins = inputFeatures.shape
        leftContext = deque()
        rightContext = deque()
        inFeatWithContext = []
        for i in range(self._tau):
            leftContext.append(np.zeros(bins))
            if i + 1 < frames:
                rightContext.append(inputFeatures[i + 1, ...])
            else:
                rightContext.append(np.zeros(bins))
        for t in range(frames):
            f = inputFeatures[t, ...]
            newFeature = np.concatenate(
                [np.concatenate(leftContext, axis=0), f, np.concatenate(rightContext, axis=0)], axis=0
            )
            inFeatWithContext.append(newFeature)
            leftContext.popleft()
            leftContext.append(f)
            rightContext.popleft()
            if t + 1 + self._tau < frames:
                rightContext.append(inputFeatures[t + 1 + self._tau, ...])
            else:
                rightContext.append(np.zeros(bins))
        inputFeatures = np.array(inFeatWithContext)
        targets = None
        if "classes" in originalSeq.get_data_keys():
            targets = originalSeq.get_data("classes")
        return DatasetSeq(seq_idx, inputFeatures, targets=targets)
