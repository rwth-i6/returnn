import h5py
from CachedDataset2 import CachedDataset2
from Dataset import DatasetSeq
from Log import log

class RawWavDataset(CachedDataset2):
    """
    This dataset returns the raw waveform information of wav files as sequence input data
    It uses temporary hdf files to buffer the data, to avoid repeatadly rading the
    wav files.
    """

  def __init__(self, listFile, frameLength, frameShift, **kwargs):
    """
    constructor

    :type listFile: string
    :param listFile: path to the file containing a list of wav file pathes (on path per line)
                     each line needs to contain exactly one wav file which is considered a sequence
    :type frameLength: int
    :param frameLenth: length of one frame in samples
    :type frameShift: int
    :param frameShift: shift length of frame in samples
    """
    super(RawWavDataset, self).__init__(**kwargs)
    self._listFile = listFile
    with open(self._listFile, 'r') as f:
      self._wavFiles = f.readlines()
    self._wavFiles = [l.strip() for l in self._wavFiles]
    self._frameLength = frameLength

    self._num_seqs = len(self._wavFiles) 
    self._seq_index_list = None
    self.num_inputs = self._frameLength 
    self.num_outputs = self._getNumOutputs()

  def _collect_single_seq(self, seq_idx):
    """returns the sequence specified by the index seq_idx

    :type seq_idx: int
    :rtype: DatasetSeq | None
    :returns DatasetSeq or None if seq_idx >= num_seqs.
    """
    raise NotImplementedError

  def _getNumberOfSequencesFromListFile(self, listFile):
    """
    returns the number of lines of the listFile

    :type listFile: string
    :param listFile: path to the file containing a list of wav file pathes (on path per line)
                     each line needs to contain exactly one wav file which is considered a sequence
    """
    #TBD !!!
    pass

  def _getNumOutputs(self):
    """
    #TBD !!!
    """
    #TBD !!!
    pass

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
    if self._num_seqs is not None:
      return self._num_seqs
    raise NotImplementedError


