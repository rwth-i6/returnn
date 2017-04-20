import h5py
from CachedDataset2 import CachedDataset2
from Dataset import DatasetSeq
from Log import log
import tempfile
import scipy.io.wavfile
import numpy as np
import time

class RawWavDataset(CachedDataset2):
  """
  This dataset returns the raw waveform information of wav files as sequence input data
  It uses temporary hdf files to buffer the data, to avoid repeatadly rading the
  wav files.
  """
  def __init__(self, listFile, frameLength, frameShift, num_outputs=None, **kwargs):
    """
    constructor

    :type listFile: string
    :param listFile: path to the file containing a list of wav file pathes (on path per line)
                     each line needs to contain exactly one wav file which is considered a sequence
    :type frameLength: int
    :param frameLenth: length of one frame in samples
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
    with open(self._listFile, 'r') as f:
      self._wavFiles = f.readlines()
    self._wavFiles = [l.strip() for l in self._wavFiles]
    self._frameLength = frameLength
    self._frameShift = frameShift
    self._flag_pad = True #spcifies if signal is getting cut or zero padded for last frame

    self._num_seqs = len(self._wavFiles) 
    self._seq_index_list = None

    self._hdfBufferHandler, self._hdfBufferPath = self._openHdfBuffer()

    self.num_inputs = self._frameLength 
    self.num_outputs = self._getNumOutputs(num_outputs)

  def _collect_single_seq(self, seq_idx):
    """
    returns the sequence specified by the index seq_idx

    :type seq_idx: int
    :rtype: DatasetSeq | None
    :returns DatasetSeq or None if seq_idx >= num_seqs.
    """
    wavFileId = self._seq_index_list[seq_idx]
    if not self._isInBuffer(wavFileId):
      self._loadWavFileIdIntoBuffer(wavFileId)

    return self._collect_single_seq_from_buffer(wavFileId, seq_idx)

  def _collect_single_seq_from_buffer(self, wavFileId, seq_idx):
    """
    returns the sequence specified by the index seq_idx

    :type wavFileId: int
    :type seq_idx: int
    :rtype: DatasetSeq | None
    :returns DatasetSeq or None if seq_idx >= num_seqs.
    """
    inputFeatures = self._getInputFeatures(wavFileId)
    outputFeatures = self._getOutputFeatures(wavFileId)
    inputFeatures = inputFeatures.astype(np.float32)
    if outputFeatures is not None:
      outputFeatures = targets.astype(np.float32)
    return DatasetSeq(seq_idx, inputFeatures, outputFeatures)

  def _getNumOutputs(self, num_outputs):
    """
    #TBD !!!
    """
    if num_outputs is None:
      raise ValueError('If no output data is used, num_outputs needs to be set for RawWavDataset')
    ret_num_outputs = {'classes': (num_outputs, 2)}
    return ret_num_outputs

  def _getInputFeatures(self, wavFileId):
    """

    :type wavFileId: int
    :param wavFileId: list index of wav file for which to return the input features
    :rtype: 2D numpy.ndarray (frames, features)
    :return: the 2d array containing the time signal segment for each frame
    """
    if not self._isInBuffer(wavFileId):
      self._loadWavFileIdIntoBuffer(wavFileId)

    timeSignal = self._hdfBufferHandler['timeSignal'][str(wavFileId)][...]
    frameLength = self._frameLength
    frameShift = self._frameShift
    nrOfFrames = int(np.ceil((float(timeSignal.shape[0]-frameLength)/frameShift) + 1))
    if self._flag_pad:
      padLength = (nrOfFrames -1) * frameShift + frameLength - timeSignal.shape[0]
      timeSignalPad = np.zeros((timeSignal.shape[0] + padLength, ))
      timeSignalPad[0:timeSignal.shape[0]] = timeSignal
    else:
      nrOfFrames -= 1  
      sigLength = (nrOfFrames -1) * frameShift + frameLength
      timeSignalPad = np.zeros((sigLength, ))
      timeSignalPad[:] = timeSignal[0:sigLength]
        
    inputFeatures = np.zeros((nrOfFrames, frameLength), dtype=np.float32)
    for i1 in range(nrOfFrames):
      inputFeatures[i1,:] = timeSignalPad[i1*frameShift:(i1*frameShift+frameLength)]
    return inputFeatures

  def _getOutputFeatures(self, wavFileId):
    """

    :type wavFileId: int
    :param wavFileId: list index of wav file for which to return the output features
    :rtype: #TBD !!!
    :return: #TBD !!!
    """
    if not self._isInBuffer(wavFileId):
      self._loadWavFileIdIntoBuffer(wavFileId)
    if not str(wavFileId) in self._hdfBufferHandler['outputs'].keys():
      return None
    else:
      #TBD !!!
      pass

  def _isInBuffer(self, wavFileId):
    """
    returns true if the wav file has already been loaded into the hdf file buffer

    :type wavFileId: int
    :rtype: bool
    """
    if str(wavFileId) in self._hdfBufferHandler['timeSignal'].keys():
      return True
    else:
      return False

  def _loadWavFileIdIntoBuffer(self, wavFileId):
    """
    loads the specified wav file into the hdf file buffer

    :type wavFileId: int
    :param wavFileId: the list index specifying the wav file to be loaded to the buffer
    """
    if self._flag_buffering:
        time.sleep(3)
    self._flag_buffering = True
    if self._isInBuffer(wavFileId):
        return False
    wavFilePath = self._wavFiles[wavFileId] 
    (r, x) = scipy.io.wavfile.read(wavFilePath)
    self._hdfBufferHandler['timeSignal'].create_dataset(str(wavFileId), data=x.astype(np.float32))
    self._flag_buffering = False
    return True

  def _openHdfBuffer(self):
    """
    opens creates a local hdf file used as buffer to avoid reopening wav files

    :rtype: (h5py._hl.file.File, string)
    :return: (hdf buffer file handler, path to tmp file)
    """
    fId, tmpHdfFilePath = tempfile.mkstemp(suffix=".hdf") 
    fileHandler = h5py.File(tmpHdfFilePath, 'w')
    fileHandler.create_group('timeSignal')
    fileHandler.create_group('outputs')

    return fileHandler, tmpHdfFilePath 

  def get_data_dim(self, key):
    """This is copied from CachedDataset2 but the assertion is
    removed (see CachedDataset2.py)

    :type key: str
    :rtype: int
    :return: number of classes, no matter if sparse or not
    """
    if key == 'data':
      return self.num_inputs
    if key in self.num_outputs:
      d = self.num_outputs[key][0]
      return d
    self._load_something()
    if len(self.added_data[0].get_data(key).shape) == 1:
      return super(CachedDataset2, self).get_data_dim(key)  # unknown
    assert len(self.added_data[0].get_data(key).shape) == 2
    return self.added_data[0].get_data(key).shape[1]

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    :type epoch: int|None
    :param epoch: epoch number
    :type seq_list: list[str] | None seq_list: In case we want to set a predefined order.
    :param seq_list: only None is currently supported
    Initialize lists:
      self.seq_index  # sorted seq idx
    """
    super(RawWavDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)

    if epoch is None:
        self._seq_index_list = range(self.num_seqs)
        return True

    if seq_list:
      raise NotImplementedError('init_seq_order of RawWavDataset does not support a predefined seq_list yet.')
    else:
      seq_index = self.get_seq_order_for_epoch(epoch, self.num_seqs, lambda s: self.get_seq_length(s).get('data', None))

    self._seq_index_list = seq_index
    if epoch is not None:
      # Give some hint to the user in case he is wondering why the cache is reloading.
      print >> log.v4, "Reinitialize dataset seq order for epoch %i." % epoch

    return True

  @property
  def num_seqs(self):
    """returns the number of sequences of the dataset

    :rtype: int
    """
    if self._num_seqs is None:
      self._num_seqs = len(self._wavFiles) 
    return self._num_seqs


