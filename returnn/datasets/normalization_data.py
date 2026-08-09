"""
Normalization data (mean/variance) for inputs and outputs, and the HDF file holding it.
"""

import os
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
    def create_normalization_file(bundle_file_path, output_file_path, dtype=np.float64, flag_include_outputs=True):
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

        :type bundle_file_path: str
        :param bundle_file_path: path to the bundle file. :see: BundleFile.BundleFile
        :type output_file_path: str
        :param output_file_path: path to the output HDF normalization file.
        :type dtype: numpy.dtype
        :param dtype: type of data to use during calculations.
        :type flag_include_outputs: bool
        :param flag_include_outputs: if True then normalization data will be
                                    calculated for outputs (targets) as well.
        """
        NormalizationData._calculate_normalization_data(
            bundle_file_path, output_file_path, NormalizationData.GROUP_INPUTS, dtype=dtype
        )
        if flag_include_outputs:
            NormalizationData._calculate_normalization_data(
                bundle_file_path, output_file_path, NormalizationData.GROUP_OUTPUTS, dtype=dtype
            )

    @staticmethod
    def _calculate_normalization_data(bundle_file_path, output_file_path, group_name, dtype=np.float64):
        """Helper method.
        Calculates and writes into the output HDF file mean, mean of squares,
        variance and total number of frames for the datasets in the given HDF
        group.

        :type bundle_file_path: str
        :param bundle_file_path: path to the bundle file. :see: BundleFile.BundleFile
        :type output_file_path: str
        :param output_file_path: path to the output HDF normalization file. If file
                               already exists it will not be truncated.
        :type group_name: str
        :param group_name: name of the HDF group for which normalization data
                          should be calculated. Also, a group with this name will
                          be created in the output HDF file to store the calculated
                          normalization data.
        :type dtype: numpy.dtype
        :param dtype: type of data to use during calculations.
        """
        import h5py

        accumulated_sum = None
        accumulated_sum_of_sqr = None
        total_frames = long()
        bundle = BundleFile(bundle_file_path)
        for file_path in bundle.dataset_file_paths:
            with h5py.File(file_path, mode="r") as dataset_file:
                interm_sum, interm_sum_of_sqr, interm_total_frames = NormalizationData._accumulate_sums(
                    dataset_file, group_name, dtype=dtype
                )
                accumulated_sum = NormalizationData._update_total_sum(accumulated_sum, interm_sum)
                accumulated_sum_of_sqr = NormalizationData._update_total_sum(accumulated_sum_of_sqr, interm_sum_of_sqr)
                total_frames += interm_total_frames

        mean, mean_of_squares, variance = NormalizationData._calculate_means(
            accumulated_sum, accumulated_sum_of_sqr, total_frames
        )

        with h5py.File(output_file_path, mode="a") as out:
            NormalizationData._write_data(out, group_name, mean, mean_of_squares, variance, total_frames, dtype=dtype)

    @staticmethod
    def _accumulate_sums(f, group_name, dtype=np.float64):
        """Helper method.
        Accumulate sums and sums of squares over feature vectors for a given group.

        :type f: h5py.File
        :param f: handle to an opened HDF file with datasets
        :type group_name: str
        :param group_name: HDF group containing datasets
        :type dtype: numpy.dtype
        :param dtype: type of data to use during calculations.
        :rtype: tuple (numpy.ndarray | None, numpy.ndarray | None, long)
        :return: tuple (sum, sum of squares, total number of time frames)
                 if they are available
        """
        total_sum = None
        sum_of_sqr = None
        total_frames = np.int64(0)
        if group_name not in f:
            return total_sum, sum_of_sqr, total_frames
        group = f[group_name]
        # list(...): h5py's .keys() is a set-like view on py3, so it is NOT subscriptable --
        # the old `dataset_names[0]` raised TypeError.
        dataset_names = list(group.keys())
        if len(dataset_names) == 0:
            return total_sum, sum_of_sqr, total_frames
        feat_dims = group[dataset_names[0]].shape[NormalizationData.DATASET_FEATURE_DIMENSION_INDEX]
        total_sum = np.zeros(feat_dims, dtype=dtype)
        sum_of_sqr = np.zeros(feat_dims, dtype=dtype)
        for ds_name in dataset_names:
            dataset = group[ds_name][...]
            total_sum += np.sum(dataset, axis=NormalizationData.DATASET_TIME_DIMENSION_INDEX)
            sum_of_sqr += np.sum(np.square(dataset), axis=NormalizationData.DATASET_TIME_DIMENSION_INDEX)
            total_frames += dataset.shape[NormalizationData.DATASET_TIME_DIMENSION_INDEX]
        return total_sum, sum_of_sqr, total_frames

    @staticmethod
    def _update_total_sum(totalSum, intermediateSum):
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
                "sums have very different orders of magnitude. summation error = {}".format(sumErr)
            )
        return newSum

    @staticmethod
    def _calculate_means(totalSum, totalSumOfSqr, total_frames):
        """Helper method.
        Calculate mean, mean of squares and variance if they are available.

        :type totalSum: numpy.ndarray | None
        :param totalSum: total sum of features
        :type totalSumOfSqr: numpy.ndarray | None
        :param totalSumOfSqr: total sum of squares of features
        :type total_frames: long
        :param total_frames: total number of timeframes
        :rtype: tuple (numpy.ndarray | None, numpy.ndarray | None, numpy.ndarray | None)
        :return: tuple (mean, mean of squares, variance) if they are available
        """
        mean = None
        mean_of_squares = None
        variance = None
        if totalSum is not None:
            assert total_frames > 0
            mean = totalSum / total_frames
        if mean is not None and totalSumOfSqr is not None:
            assert total_frames > 0
            mean_of_squares = totalSumOfSqr / total_frames
            # Var[X] = E[X ^ 2] - (E[X]) ^ 2
            variance = mean_of_squares - np.square(mean)
        return mean, mean_of_squares, variance

    @staticmethod
    def _write_data(f, group_name, mean, meanOfSqr, variance, total_frames, dtype=np.float64):
        """Helper method.
        Writes means and variance for a given group.

        :type f: h5py.File
        :param f: handle to an opened HDF file to which data should be written.
        :type group_name: str
        :param group_name: HDF group name
        :type mean: numpy.ndarray | None
        :param mean: mean
        :type meanOfSqr: numpy.ndarray | None
        :param meanOfSqr: mean of squares
        :type variance: numpy.ndarray | None
        :param variance: variance
        :type total_frames: long
        :param total_frames: total number of time frames
        :type dtype: numpy.dtype
        :param dtype: type of data to use for writing the data
        """
        if group_name in f:
            del f[group_name]
        group = f.create_group(group_name)
        dsNames = [
            NormalizationData.DATASET_MEAN,
            NormalizationData.DATASET_MEAN_OF_SQUARES,
            NormalizationData.DATASET_VARIANCE,
        ]
        datasets = [mean, meanOfSqr, variance]
        for name, ds in zip(dsNames, datasets):
            NormalizationData._writeDataset(group, name, ds, dtype)
        if total_frames > 0:
            group.create_dataset(NormalizationData.DATASET_TOTAL_FRAMES, data=total_frames)

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
        import h5py

        if not os.path.isfile(self._normalizationFilePath):
            raise IOError(self._normalizationFilePath + " does not exist")
        with h5py.File(self._normalizationFilePath, mode="r") as f:
            self._inputMean, self._inputVariance = self._getMeanAndVarianceFromGroup(f, self.GROUP_INPUTS)
            self._outputMean, self._outputVariance = self._getMeanAndVarianceFromGroup(f, self.GROUP_OUTPUTS)

    @staticmethod
    def _getMeanAndVarianceFromGroup(f, group_name):
        """Reads mean and variance from the given group if they are available.
        Both mean and variance are optional i.e. they may be absent in the
        given HDF group.

        :type f: h5py.File
        :param f: handle to an opened HDF file with normalization data.
        :type group_name: str
        :param group_name: name of the HDF group from which mean and variance
                          should be read.
        :rtype: tuple (numpy.ndarray | None, numpy.ndarray | None)
        :return: a tuple (mean, variance) each of which may be None
                 if the data is not available.
        """
        mean = None
        variance = None
        if group_name not in f:
            return mean, variance
        group = f[group_name]
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
