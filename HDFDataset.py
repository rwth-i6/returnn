
import gc
import h5py
import numpy
import theano
from CachedDataset import CachedDataset
from Log import log


class HDFDataset(CachedDataset):

  @classmethod
  def load_data(cls, config, cache_byte_size, files_config_key, **kwargs):
    """
    :type config: Config.Config
    :type cache_byte_size: int
    :type chunking: str
    :type seq_ordering: str
    :rtype: (Dataset,int)
    :returns the dataset, and the cache byte size left over if we cache the whole dataset.
    """
    if not config.has(files_config_key):
      return None, 0
    data = cls.from_config(config, cache_byte_size=cache_byte_size, **kwargs)
    for f in config.list(files_config_key):
      data.add_file(f)
    data.initialize()
    return data, data.definite_cache_leftover

  def __init__(self, *args, **kwargs):
    super(HDFDataset, self).__init__(*args, **kwargs)
    self.files = []; """ :type: list[str] """
    self.file_start = [0]
    self.file_seq_start = []; """ :type: list[list[int]] """
    self.file_index = []; """ :type: list[int] """
    self.tags = []; """ :type: list[str] """
    self.targets = {}

  def add_file(self, filename):
    """
    Setups data:
      self.seq_lengths
      self.file_index
      self.file_start
      self.file_seq_start
    Use load_seqs() to load the actual data.
    :type filename: str
    """
    fin = h5py.File(filename, "r")
    if 'targets' in fin:
      self.labels = { k : [ item.split('\0')[0] for item in fin["targets/labels"][k][...].tolist() ] for k in fin['targets/labels'] }
    if not self.labels:
      labels = [ item.split('\0')[0] for item in fin["labels"][...].tolist() ]; """ :type: list[str] """
      self.labels = { 'classes' : labels }
      assert len(self.labels) == len(labels), "expected " + str(len(self.labels)) + " got " + str(len(labels))
    tags = [ item.split('\0')[0] for item in fin["seqTags"][...].tolist() ]; """ :type: list[str] """
    self.files.append(filename)
    seq_start = [0]
    if 'times' in fin:
      self.timestamps.extend(fin['times'][...].tolist())
    for l,t in zip(fin['seqLengths'][...].tolist(), tags):
      self._seq_lengths.append(l)
      seq_start.append(seq_start[-1] + l)
      self.tags.append(t)
    self.file_seq_start.append(seq_start)
    nseqs = len(seq_start) - 1
    self._num_seqs += nseqs
    self.file_index.extend([len(self.files) - 1] * nseqs)
    self.file_start.append(self.file_start[-1] + nseqs)
    self._num_timesteps = sum(self._seq_lengths)
    if 'maxCTCIndexTranscriptionLength' in fin.attrs:
      self.max_ctc_length = max(self.max_ctc_length, fin.attrs['maxCTCIndexTranscriptionLength'])
    if self.num_inputs == 0:
      self.num_inputs = fin.attrs['inputPattSize']
    assert self.num_inputs == fin.attrs['inputPattSize'], "wrong input dimension in file " + filename + " (expected " + str(self.num_inputs) + " got " + str(fin.attrs['inputPattSize']) + ")"
    if self.num_outputs == 0:
      if 'numLabels'in fin.attrs:
        self.num_outputs = fin.attrs['numLabels']
        assert self.num_outputs == fin.attrs['numLabels'], "wrong number of labels in file " + filename  + " (expected " + str(self.num_outputs) + " got " + str(fin.attrs['numLabels']) + ")"
      else:
        self.num_outputs = { k : fin['targets/size'].attrs[k] for k in fin['targets/size'].attrs }
    if 'ctcIndexTranscription' in fin:
      if self.ctc_targets is None:
        self.ctc_targets = fin['ctcIndexTranscription'][...]
      else:
        tmp = fin['ctcIndexTranscription'][...]
        pad_width = self.max_ctc_length - tmp.shape[1]
        tmp = numpy.pad(tmp, ((0,0),(0,pad_width)), 'constant', constant_values=-1)
        pad_width = self.max_ctc_length - self.ctc_targets.shape[1]
        self.ctc_targets = numpy.pad(self.ctc_targets, ((0,0),(0,pad_width)), 'constant', constant_values=-1)
        self.ctc_targets = numpy.concatenate((self.ctc_targets, tmp))
      self.num_running_chars = numpy.sum(self.ctc_targets != -1)
    if 'targets' in fin:
      for name in fin['targets/data']:
        self.targets[name] = numpy.zeros((self.get_num_timesteps(),), dtype=theano.config.floatX) - 1
    fin.close()

  def _load_seqs(self, start, end):
    """
    Load data sequences.
    As a side effect, will modify / fill-up:
      self.alloc_intervals
      self.targets
      self.chars

    :param int start: start sorted seq idx
    :param int end: end sorted seq idx
    """
    assert start < self.num_seqs
    assert end <= self.num_seqs
    selection = self.insert_alloc_interval(start, end)
    assert len(selection) <= end - start, "DEBUG: more sequences requested (" + str(len(selection)) + ") as required (" + str(end-start) + ")"
    file_info = [ [] for l in xrange(len(self.files)) ]; """ :type: list[(int,int)] """
    # file_info[i] is (sorted seq idx from selection, real seq idx)
    for idc in selection:
      ids = self._seq_index[idc]
      file_info[self.file_index[ids]].append((idc,ids))
    for i in xrange(len(self.files)):
      if len(file_info[i]) == 0:
        continue
      print >> log.v4, "loading file", self.files[i]
      fin = h5py.File(self.files[i], 'r')
      inputs = fin['inputs'][...]; """ :type: numpy.ndarray """
      for idc, ids in file_info[i]:
        s = ids - self.file_start[i]
        p = self.file_seq_start[i][s]
        l = self._seq_lengths[ids]
        if 'targets' in fin:
          for k in fin['targets/data']:
            y = fin['targets/data'][k][...]
            self.targets[k][self.get_seq_start(idc):self.get_seq_start(idc) + l] = y[p : p + l]
        x = inputs[p : p + l]
        self._set_alloc_intervals_data(idc, data=x)
      fin.close()
    gc.collect()
    assert self.is_cached(start, end)

  def get_tag(self, sorted_seq_idx):
    ids = self._seq_index[sorted_seq_idx]
    return self.tags[ids]
