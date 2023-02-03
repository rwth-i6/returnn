class BundleFile(object):
    """Holds paths to HDF dataset files."""

    def __init__(self, filePath):
        """Reads paths to HDF dataset files from a bundle file.
        Example of contents of a bundle file:

        /work/asr2/ryndin/crnnRegressionSpeechEnhancemenent/data/data_tr05_real_1_100.hdf
        /work/asr2/ryndin/crnnRegressionSpeechEnhancemenent/data/data_tr05_real_2_100.hdf
        /work/asr2/ryndin/crnnRegressionSpeechEnhancemenent/data/data_tr05_real_3_100.hdf
        /work/asr2/ryndin/crnnRegressionSpeechEnhancemenent/data/data_tr05_real_4_100.hdf
        /work/asr2/ryndin/crnnRegressionSpeechEnhancemenent/data/data_tr05_real_5_100.hdf
        /work/asr2/ryndin/crnnRegressionSpeechEnhancemenent/data/data_tr05_real_6_100.hdf
        /work/asr2/ryndin/crnnRegressionSpeechEnhancemenent/data/data_tr05_simu_1_100.hdf
        /work/asr2/ryndin/crnnRegressionSpeechEnhancemenent/data/data_tr05_simu_2_100.hdf
        /work/asr2/ryndin/crnnRegressionSpeechEnhancemenent/data/data_tr05_simu_3_100.hdf
        /work/asr2/ryndin/crnnRegressionSpeechEnhancemenent/data/data_tr05_simu_4_100.hdf
        /work/asr2/ryndin/crnnRegressionSpeechEnhancemenent/data/data_tr05_simu_5_100.hdf
        /work/asr2/ryndin/crnnRegressionSpeechEnhancemenent/data/data_tr05_simu_6_100.hdf

        :type filePath: str
        :param filePath: path to a bundle file which contains paths to HDF
                         dataset files. One path per line.
        """
        self._filePath = filePath
        self._datasetFilesPaths = []
        self._readDatasetFilesPaths()

    def _readDatasetFilesPaths(self):
        """Reads paths to HDF dataset files from a bundle file."""
        with open(self._filePath, "r") as bundleFile:
            self._datasetFilesPaths = filter(
                lambda f: bool(f),  # filter off empty lines
                map(lambda l: l.strip(), bundleFile.readlines()),  # strip spaces from left and right
            )

    @property
    def datasetFilePaths(self):
        """Paths to HDF dataset files.

        :rtype: list of str
        :return: Paths to HDF dataset files.
        """
        return self._datasetFilesPaths

    @property
    def numberOfDatasetFiles(self):
        """Number of HDF dataset files.

        :rtype: int
        :return: Number of HDF dataset files.
        """
        return len(self._datasetFilesPaths)
