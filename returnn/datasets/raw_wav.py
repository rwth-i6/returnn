"""
Provide :class:`RawWavDataset`.
"""

from __future__ import annotations

from .cached2 import CachedDataset2
from returnn.datasets.basic import DatasetSeq
from returnn.log import log
import tempfile
import numpy as np
import time


class RawWavDataset(CachedDataset2):
    """
    This dataset returns the raw waveform information of wav files as sequence input data
    It uses temporary hdf files to buffer the data, to avoid repeatedly reading the
    wav files.
    """

    # Need to keep names as-is for compatibility.
    # noinspection PyPep8Naming
    def __init__(self, listFile, frameLength, frameShift, num_outputs=None, **kwargs):
        """
        constructor

        :type listFile: string
        :param listFile: path to the file containing a list of wav file pathes (on path per line)
                         each line needs to contain exactly one wav file which is considered a sequence
        :type frameLength: int
        :param frameLength: length of one frame in samples
        :type frameShift: int
        :param frameShift: shift length of frame in samples
        :type num_outputs: int
        :param num_outputs: this needs to be set if the data set is used with
                            only input data (e.g. for the extraction
                            process).
        """
        self._flag_buffering = False
        super(RawWavDataset, self).__init__(**kwargs)
        self._listFile = listFile
        with open(self._listFile, "r") as f:
            self._wavFiles = f.readlines()
        self._wavFiles = [line.strip() for line in self._wavFiles]
        self._frameLength = frameLength
        self._frameShift = frameShift
        self._flag_pad = True  # specifies if signal is getting cut or zero padded for last frame

        self._num_seqs = len(self._wavFiles)
        self._seq_index_list = None

        self._hdfBufferHandler, self._hdfBufferPath = self._open_hdf_buffer()

        self.num_inputs = self._frameLength
        self.num_outputs = self._get_num_outputs(num_outputs)

    def _collect_single_seq(self, seq_idx):
        """
        returns the sequence specified by the index seq_idx

        :type seq_idx: int
        :rtype: DatasetSeq | None
        :returns DatasetSeq or None if seq_idx >= num_seqs.
        """
        wav_file_id = self._seq_index_list[seq_idx]
        if not self._isInBuffer(wav_file_id):
            self._load_wav_file_id_into_buffer(wav_file_id)

        return self._collect_single_seq_from_buffer(wav_file_id, seq_idx)

    def _collect_single_seq_from_buffer(self, wav_file_id, seq_idx):
        """
        returns the sequence specified by the index seq_idx

        :type wav_file_id: int
        :type seq_idx: int
        :rtype: DatasetSeq | None
        :returns DatasetSeq or None if seq_idx >= num_seqs.
        """
        input_features = self._get_input_features(wav_file_id)
        output_features = self._get_output_features(wav_file_id)
        input_features = input_features.astype(np.float32)
        if output_features is not None:
            output_features = output_features.astype(np.float32)
        return DatasetSeq(seq_idx, input_features, targets=output_features)

    def _get_num_outputs(self, num_outputs):
        """
        #TBD !!!
        """
        if num_outputs is None:
            raise ValueError("If no output data is used, num_outputs needs to be set for RawWavDataset")
        ret_num_outputs = {"classes": (num_outputs, 2)}
        return ret_num_outputs

    def _get_input_features(self, wav_file_id):
        """

        :type wav_file_id: int
        :param wav_file_id: list index of wav file for which to return the input features
        :rtype: 2D numpy.ndarray (frames, features)
        :return: the 2d array containing the time signal segment for each frame
        """
        if not self._isInBuffer(wav_file_id):
            self._load_wav_file_id_into_buffer(wav_file_id)

        # note: "timeSignal" here is the HDF group key, not a variable name
        time_signal = self._hdfBufferHandler["timeSignal"][str(wav_file_id)][...]
        frame_length = self._frameLength
        frame_shift = self._frameShift
        nr_of_frames = int(np.ceil((float(time_signal.shape[0] - frame_length) / frame_shift) + 1))
        if self._flag_pad:
            pad_length = (nr_of_frames - 1) * frame_shift + frame_length - time_signal.shape[0]
            time_signal_pad = np.zeros((time_signal.shape[0] + pad_length,))
            time_signal_pad[0 : time_signal.shape[0]] = time_signal
        else:
            nr_of_frames -= 1
            sig_length = (nr_of_frames - 1) * frame_shift + frame_length
            time_signal_pad = np.zeros((sig_length,))
            time_signal_pad[:] = time_signal[0:sig_length]

        input_features = np.zeros((nr_of_frames, frame_length), dtype=np.float32)
        for i1 in range(nr_of_frames):
            input_features[i1, :] = time_signal_pad[i1 * frame_shift : (i1 * frame_shift + frame_length)]
        return input_features

    def _get_output_features(self, wav_file_id):
        """

        :type wav_file_id: int
        :param wav_file_id: list index of wav file for which to return the output features
        :rtype: #TBD !!!
        :return: #TBD !!!
        """
        if not self._isInBuffer(wav_file_id):
            self._load_wav_file_id_into_buffer(wav_file_id)
        if str(wav_file_id) not in self._hdfBufferHandler["outputs"].keys():
            return None
        return None

    def _isInBuffer(self, wav_file_id):
        """
        returns true if the wav file has already been loaded into the hdf file buffer

        :type wav_file_id: int
        :rtype: bool
        """
        if str(wav_file_id) in self._hdfBufferHandler["timeSignal"].keys():
            return True
        else:
            return False

    def _load_wav_file_id_into_buffer(self, wav_file_id):
        """
        loads the specified wav file into the hdf file buffer

        :type wav_file_id: int
        :param wav_file_id: the list index specifying the wav file to be loaded to the buffer
        """
        if self._flag_buffering:
            time.sleep(3)
        self._flag_buffering = True
        if self._isInBuffer(wav_file_id):
            return False
        wav_file_path = self._wavFiles[wav_file_id]
        import scipy.io.wavfile

        (r, x) = scipy.io.wavfile.read(wav_file_path)
        self._hdfBufferHandler["timeSignal"].create_dataset(str(wav_file_id), data=x.astype(np.float32))
        self._flag_buffering = False
        return True

    def _open_hdf_buffer(self):
        """
        opens creates a local hdf file used as buffer to avoid reopening wav files

        :rtype: (h5py._hl.file.File, string)
        :return: (hdf buffer file handler, path to tmp file)
        """
        import h5py

        f_id, tmp_hdf_file_path = tempfile.mkstemp(suffix=".hdf")
        file_handler = h5py.File(tmp_hdf_file_path, "w")
        file_handler.create_group("timeSignal")
        file_handler.create_group("outputs")

        return file_handler, tmp_hdf_file_path

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
        super(RawWavDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)

        if epoch is None:
            self._seq_index_list = range(self.num_seqs)
            return True

        if seq_list is not None or seq_order is not None:
            raise NotImplementedError("init_seq_order of RawWavDataset does not support a predefined seq_list yet.")
        else:
            seq_index = self.get_seq_order_for_epoch(
                epoch, self.num_seqs, lambda s: self.get_seq_length(s).get("data", None)
            )

        self._seq_index_list = seq_index
        if epoch is not None:
            # Give some hint to the user in case he is wondering why the cache is reloading.
            print("Reinitialize dataset seq order for epoch %i." % epoch, file=log.v4)

        return True

    @property
    def num_seqs(self):
        """returns the number of sequences of the dataset

        :rtype: int
        """
        if self._num_seqs is None:
            self._num_seqs = len(self._wavFiles)
        return self._num_seqs
