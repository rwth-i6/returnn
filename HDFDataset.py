
import gc
import h5py
import numpy
import theano
from CachedDataset import CachedDataset
from Log import log

# Common attribute names for HDF dataset, which should be used in order to be proceed with HDFDataset class.
attr_seqLengths = 'seqLengths'
attr_inputPattSize = 'inputPattSize'
attr_numLabels = 'numLabels'
attr_times = 'times'
attr_ctcIndexTranscription = 'ctcIndexTranscription'

class HDFDataset(CachedDataset):

  def __init__(self, *args, **kwargs):
    super(HDFDataset, self).__init__(*args, **kwargs)
    self.files = []; """ :type: list[str] """
    self.file_start = [0]
    self.file_seq_start = []; """ :type: list[list[int]] """
    self.file_index = []; """ :type: list[int] """
    self.tags = []; """ :type: list[str] """

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
      assert len(self.labels['classes']) == len(labels), "expected " + str(len(self.labels['classes'])) + " got " + str(len(labels))
    tags = [ item.split('\0')[0] for item in fin["seqTags"][...].tolist() ]; """ :type: list[str] """
    self.files.append(filename)
    seq_start = [numpy.array([0,0])]
    if 'times' in fin:
      self.timestamps.extend(fin[attr_times][...].tolist())
    seq_lengths = fin[attr_seqLengths][...]
    if len(seq_lengths.shape) == 1:
      seq_lengths = numpy.array(zip(seq_lengths.tolist(), seq_lengths.tolist()))
    for l in seq_lengths:
      self._seq_lengths.append(numpy.array(l))
      seq_start.append(seq_start[-1] + l)
    self.tags += tags
    self.file_seq_start.append(seq_start)
    nseqs = len(seq_start) - 1
    self._num_seqs += nseqs
    self.file_index.extend([len(self.files) - 1] * nseqs)
    self.file_start.append(self.file_start[-1] + nseqs)
    self._num_timesteps = sum([s[0] for s in self._seq_lengths])
    self._num_codesteps = sum([s[1] for s in self._seq_lengths])
    if 'maxCTCIndexTranscriptionLength' in fin.attrs:
      self.max_ctc_length = max(self.max_ctc_length, fin.attrs['maxCTCIndexTranscriptionLength'])
    if self.num_inputs == 0:
      self.num_inputs = fin['inputs'][0].shape[0] #fin.attrs[attr_inputPattSize]
    assert self.num_inputs == fin['inputs'][0].shape[0], "wrong input dimension in file " + filename + " (expected " + str(self.num_inputs) + " got " + str(fin.attrs[attr_inputPattSize]) + ")"
    if not self.num_outputs:
      if 'targets/size' in  fin:
        self.num_outputs = { k : [fin['targets/size'].attrs[k], len(fin['targets/data'][k].shape)] for k in fin['targets/size'].attrs }
      else:
        self.num_outputs = { 'classes' : fin.attrs[attr_numLabels] }
        assert self.num_outputs['classes'] == fin.attrs[attr_numLabels], "wrong number of labels in file " + filename + " (expected " + str(self.num_outputs['classes']) + " got " + str(fin.attrs[attr_numLabels]) + ")"
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
        tdim = 1 if len(fin['targets/data'][name].shape) == 1 else fin['targets/data'][name].shape[1]
        if tdim == 1:
          self.targets[name] = numpy.zeros((self._num_codesteps,), dtype=theano.config.floatX) - 1
        else:
          self.targets[name] = numpy.zeros((self._num_codesteps,tdim), dtype=theano.config.floatX) - 1
    else:
      self.targets = { 'classes' : numpy.zeros((self._num_timesteps,), dtype=theano.config.floatX)  }
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
    file_info = [ [] for l in xrange(len(self.files)) ]; """ :type: list[list[int]] """
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
            self.targets[k][self.get_seq_start(idc)[1]:self.get_seq_start(idc)[1] + l[1]] = fin['targets/data/' + k][p[1] : p[1] + l[1]][...]
        x = inputs[p[0] : p[0] + l[0]]
        self._set_alloc_intervals_data(idc, data=x)
      fin.close()
    gc.collect()
    assert self.is_cached(start, end)

  def get_tag(self, sorted_seq_idx):
    ids = self._seq_index[sorted_seq_idx]
    return self.tags[ids]

  def len_info(self):
    return ", ".join(["HDF dataset",
                      "sequences: %i" % self.num_seqs,
                      "frames: %i" % self.get_num_timesteps()])
