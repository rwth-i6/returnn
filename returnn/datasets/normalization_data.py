import os
import h5py
import numpy as np

from .bundle_file import BundleFile
from returnn.util.basic import long


class NormalizationData:
    """This class holds normalization data for inputs and outputs.
    It also contains methods to create the normalization HDF file.
    """

    GROUP_INPUTS = "inputs"
    GROUP_OUTPUTS = "outputs"

    DATASET_MEAN = "mean"
    DATASET_MEAN_OF_SQUARES = "meanOfSquares"
    DATASET_VARIANCE = "variance"
    DATASET_TOTAL_FRAMES = "totalNumberOfFrames"

    DATASET_TIME_DIMENSION_INDEX = 0
    DATASET_FEATURE_DIMENSION_INDEX = 1

    SUMMATION_PRECISION = 1e-5

    @staticmethod
    def createNormalizationFile(bundleFilePath, outputFilePath, dtype=np.float64, flag_includeOutputs=True):
        """Calculates means over inputs and outputs of datasets in the HDF files
        described by the given bundle file.

        :see: BundleFile.BundleFile

        Each HDF dataset file is expected to have the following groups:

          * NormalizationData.GROUP_INPUTS (the group for the input data)
          * NormalizationData.GROUP_OUTPUTS (the group for the output data)

        Each group may have datasets. Each dataset is expected to have
        shape (time frames, features).
        E.g. (267, 513) -- 267 time frames each containing a feature vector of
        dimensionality 513.

        The method writes results into the given output file.
        Availability of means and variances depends on whether the corresponding
        groups are available in the input dataset HDF files.

        !!! IMPORTANT !!!
        General rule of thumb: if one dataset file has both input and output
        groups then you should make sure that all the dataset files have them.
        Otherwise means and variance will not be correct.
        It is OK if *all* the datasets have only the input group.
        In this case means and variance only for inputs will be calculated.

        :type bundleFilePath: str
        :param bundleFilePath: path to the bundle file. :see: BundleFile.BundleFile
        :type outputFilePath: str
        :param outputFilePath: path to the output HDF normalization file.
        :type dtype: numpy.dtype
        :param dtype: type of data to use during calculations.
        :type flag_includeOutputs: bool
        :param flag_includeOutputs: if True then normalization data will be
                                    calculated for outputs (targets) as well.
        """
        NormalizationData._calculateNormalizationData(
            bundleFilePath, outputFilePath, NormalizationData.GROUP_INPUTS, dtype=dtype
        )
        if flag_includeOutputs:
            NormalizationData._calculateNormalizationData(
                bundleFilePath, outputFilePath, NormalizationData.GROUP_OUTPUTS, dtype=dtype
            )

    @staticmethod
    def _calculateNormalizationData(bundleFilePath, outputFilePath, groupName, dtype=np.float64):
        """Helper method.
        Calculates and writes into the output HDF file mean, mean of squares,
        variance and total number of frames for the datasets in the given HDF
        group.

        :type bundleFilePath: str
        :param bundleFilePath: path to the bundle file. :see: BundleFile.BundleFile
        :type outputFilePath: str
        :param outputFilePath: path to the output HDF normalization file. If file
                               already exists it will not be truncated.
        :type groupName: str
        :param groupName: name of the HDF group for which normalization data
                          should be calculated. Also, a group with this name will
                          be created in the output HDF file to store the calculated
                          normalization data.
        :type dtype: numpy.dtype
        :param dtype: type of data to use during calculations.
        """
        accumulatedSum = None
        accumulatedSumOfSqr = None
        totalFrames = long()
        bundle = BundleFile(bundleFilePath)
        for filePath in bundle.datasetFilePaths:
            with h5py.File(filePath, mode="r") as datasetFile:
                intermSum, intermSumOfSqr, intermTotalFrames = NormalizationData._accumulateSums(
                    datasetFile, groupName, dtype=dtype
                )
                accumulatedSum = NormalizationData._updateTotalSum(accumulatedSum, intermSum)
                accumulatedSumOfSqr = NormalizationData._updateTotalSum(accumulatedSumOfSqr, intermSumOfSqr)
                totalFrames += intermTotalFrames

        mean, meanOfSquares, variance = NormalizationData._calculateMeans(
            accumulatedSum, accumulatedSumOfSqr, totalFrames
        )

        with h5py.File(outputFilePath, mode="a") as out:
            NormalizationData._writeData(out, groupName, mean, meanOfSquares, variance, totalFrames, dtype=dtype)

    @staticmethod
    def _accumulateSums(f, groupName, dtype=np.float64):
        """Helper method.
        Accumulate sums and sums of squares over feature vectors for a given group.

        :type f: h5py.File
        :param f: handle to an opened HDF file with datasets
        :type groupName: str
        :param groupName: HDF group containing datasets
        :type dtype: numpy.dtype
        :param dtype: type of data to use during calculations.
        :rtype: tuple (numpy.ndarray | None, numpy.ndarray | None, long)
        :return: tuple (sum, sum of squares, total number of time frames)
                 if they are available
        """
        sum = None
        sumOfSqr = None
        totalFrames = np.int64(0)
        if groupName not in f:
            return sum, sumOfSqr, totalFrames
        group = f[groupName]
        datasetNames = group.keys()
        if len(datasetNames) == 0:
            return sum, sumOfSqr, totalFrames
        featDims = group[datasetNames[0]].shape[NormalizationData.DATASET_FEATURE_DIMENSION_INDEX]
        sum = np.zeros(featDims, dtype=dtype)
        sumOfSqr = np.zeros(featDims, dtype=dtype)
        for dsName in datasetNames:
            dataset = group[dsName][...]
            sum += np.sum(dataset, axis=NormalizationData.DATASET_TIME_DIMENSION_INDEX)
            sumOfSqr += np.sum(np.square(dataset), axis=NormalizationData.DATASET_TIME_DIMENSION_INDEX)
            totalFrames += dataset.shape[NormalizationData.DATASET_TIME_DIMENSION_INDEX]
        return sum, sumOfSqr, totalFrames

    @staticmethod
    def _updateTotalSum(totalSum, intermediateSum):
        """Helper method.
        Updates total sum with intermediate sum if the latter is available.

        :type totalSum: numpy.ndarray | None
        :param totalSum: total sum
        :type intermediateSum: numpy.ndarray | None
        :param intermediateSum: intermediate sum
        :rtype: numpy.ndarray | None
        :return: updated total sum if available
        """
        if totalSum is None and intermediateSum is None:
            return None
        if totalSum is None:
            return intermediateSum
        if intermediateSum is None:
            return totalSum
        # floating point summation check
        oldSum = totalSum
        newSum = np.add(totalSum, intermediateSum)
        sumErr = np.sum(np.abs(newSum - oldSum - intermediateSum))
        if sumErr > NormalizationData.SUMMATION_PRECISION:
            raise FloatingPointError(
                "sums have very different orders of magnitude." " summation error = {}".format(sumErr)
            )
        return newSum

    @staticmethod
    def _calculateMeans(totalSum, totalSumOfSqr, totalFrames):
        """Helper method.
        Calculate mean, mean of squares and variance if they are available.

        :type totalSum: numpy.ndarray | None
        :param totalSum: total sum of features
        :type totalSumOfSqr: numpy.ndarray | None
        :param totalSumOfSqr: total sum of squares of features
        :type totalFrames: long
        :param totalFrames: total number of timeframes
        :rtype: tuple (numpy.ndarray | None, numpy.ndarray | None, numpy.ndarray | None)
        :return: tuple (mean, mean of squares, variance) if they are available
        """
        mean = None
        meanOfSquares = None
        variance = None
        if totalSum is not None:
            assert totalFrames > 0
            mean = totalSum / totalFrames
        if mean is not None and totalSumOfSqr is not None:
            assert totalFrames > 0
            meanOfSquares = totalSumOfSqr / totalFrames
            # Var[X] = E[X ^ 2] - (E[X]) ^ 2
            variance = meanOfSquares - np.square(mean)
        return mean, meanOfSquares, variance

    @staticmethod
    def _writeData(f, groupName, mean, meanOfSqr, variance, totalFrames, dtype=np.float64):
        """Helper method.
        Writes means and variance for a given group.

        :type f: h5py.File
        :param f: handle to an opened HDF file to which data should be written.
        :type groupName: str
        :param groupName: HDF group name
        :type mean: numpy.ndarray | None
        :param mean: mean
        :type meanOfSqr: numpy.ndarray | None
        :param meanOfSqr: mean of squares
        :type variance: numpy.ndarray | None
        :param variance: variance
        :type totalFrames: long
        :param totalFrames: total number of time frames
        :type dtype: numpy.dtype
        :param dtype: type of data to use for writing the data
        """
        if groupName in f:
            del f[groupName]
        group = f.create_group(groupName)
        dsNames = [
            NormalizationData.DATASET_MEAN,
            NormalizationData.DATASET_MEAN_OF_SQUARES,
            NormalizationData.DATASET_VARIANCE,
        ]
        datasets = [mean, meanOfSqr, variance]
        for name, ds in zip(dsNames, datasets):
            NormalizationData._writeDataset(group, name, ds, dtype)
        if totalFrames > 0:
            group.create_dataset(NormalizationData.DATASET_TOTAL_FRAMES, data=totalFrames)

    @staticmethod
    def _writeDataset(group, datasetName, dataset, dtype=np.float64):
        """Helper Method.
        Writes dataset into an HDF group if the dataset is available.

        :type group: h5py.Group
        :param group: HDF group handle
        :type datasetName: str
        :param datasetName: name of the dataset
        :type dataset: numpy.ndarray | None
        :param dataset: actual data of the dataset
        :type dtype: numpy.dtype
        :param dtype: type of data to use for writing the data.
        """
        if dataset is None:
            return
        group.create_dataset(datasetName, data=dataset, dtype=dtype)

    def __init__(self, normalizationFilePath):
        """Reads normalization data from the given HDF file and saves it
        into the member variables.

        :type normalizationFilePath: str
        :param normalizationFilePath: path to the HDF file with normalization data.
        """
        self._normalizationFilePath = normalizationFilePath
        self._inputMean = None
        self._inputVariance = None
        self._outputMean = None
        self._outputVariance = None
        self._readNormalizationData()

    def _readNormalizationData(self):
        """Reads normalization data from the given HDF file.
        The file is expected to have the following structure.

        It may have two groups:
          * NormalizationData.GROUP_INPUTS (the group for the input data)
          * NormalizationData.GROUP_OUTPUTS (the group for the output data)

        Each group may have two datasets:
          * NormalizationData.DATASET_MEAN (the dataset for mean)
          * NormalizationData.DATASET_VARIANCE (the dataset for variance)

        Everything is optional e.g. when only the group for the input data
        is present and it contains only the dataset for mean then only this
        data will be read. No exception will be thrown.

        The groups may also contain additional optional information such as
        e.g. total number of time frames, mean of squares etc.
        However, this information is not read here.
        """
        if not os.path.isfile(self._normalizationFilePath):
            raise IOError(self._normalizationFilePath + " does not exist")
        with h5py.File(self._normalizationFilePath, mode="r") as f:
            self._inputMean, self._inputVariance = self._getMeanAndVarianceFromGroup(f, self.GROUP_INPUTS)
            self._outputMean, self._outputVariance = self._getMeanAndVarianceFromGroup(f, self.GROUP_OUTPUTS)

    @staticmethod
    def _getMeanAndVarianceFromGroup(f, groupName):
        """Reads mean and variance from the given group if they are available.
        Both mean and variance are optional i.e. they may be absent in the
        given HDF group.

        :type f: h5py.File
        :param f: handle to an opened HDF file with normalization data.
        :type groupName: str
        :param groupName: name of the HDF group from which mean and variance
                          should be read.
        :rtype: tuple (numpy.ndarray | None, numpy.ndarray | None)
        :return: a tuple (mean, variance) each of which may be None
                 if the data is not available.
        """
        mean = None
        variance = None
        if groupName not in f:
            return mean, variance
        group = f[groupName]
        if NormalizationData.DATASET_MEAN in group:
            mean = group[NormalizationData.DATASET_MEAN][...]
        if NormalizationData.DATASET_VARIANCE in group:
            variance = group[NormalizationData.DATASET_VARIANCE][...]
        return mean, variance

    @property
    def inputMean(self):
        """Mean of the input data.

        :rtype: numpy.ndarray | None
        :return: Mean of the input data if it is available or None otherwise.
        """
        return self._inputMean

    @property
    def inputVariance(self):
        """Variance of the input data.

        :rtype: numpy.ndarray | None
        :return: Variance of the input data if it is available or None otherwise.
        """
        return self._inputVariance

    @property
    def outputMean(self):
        """Mean of the output data.

        :rtype: numpy.ndarray | None
        :return: Mean of the output data if it is available or None otherwise.
        """
        return self._outputMean

    @property
    def outputVariance(self):
        """Variance of the output data.

        :rtype: numpy.ndarray | None
        :return: Variance of the output data if it is available or None otherwise.
        """
        return self._outputVariance
