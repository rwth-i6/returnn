"""
Bundle file: a text file listing the HDF dataset files that make up a corpus.
"""


class BundleFile:
    """Holds paths to HDF dataset files."""

    def __init__(self, file_path):
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

        :type file_path: str
        :param file_path: path to a bundle file which contains paths to HDF
                         dataset files. One path per line.
        """
        self._file_path = file_path
        self._dataset_files_paths = []
        self._read_dataset_files_paths()

    def _read_dataset_files_paths(self):
        """Reads paths to HDF dataset files from a bundle file."""
        with open(self._file_path, "r") as bundle_file:
            self._dataset_files_paths = filter(
                lambda f: bool(f),  # filter off empty lines
                map(lambda line: line.strip(), bundle_file.readlines()),  # strip spaces from left and right
            )

    @property
    def dataset_file_paths(self):
        """Paths to HDF dataset files.

        :rtype: list of str
        :return: Paths to HDF dataset files.
        """
        return self._dataset_files_paths

    @property
    def number_of_dataset_files(self):
        """Number of HDF dataset files.

        :rtype: int
        :return: Number of HDF dataset files.
        """
        return len(self._dataset_files_paths)
