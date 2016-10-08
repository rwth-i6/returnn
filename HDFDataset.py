
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
    self.data_dtype = {}; ":type: dict[str,str]"
    self.data_sparse = {}; ":type: dict[str,bool]"

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
    if 'times' in fin:
      self.timestamps.extend(fin[attr_times][...].tolist())
    seq_lengths = fin[attr_seqLengths][...]
    if 'targets' in fin:
      self.target_keys = sorted(fin['targets/labels'].keys())
    else:
      self.target_keys = ['classes']

    if len(seq_lengths.shape) == 1:
      seq_lengths = numpy.array(zip(*[seq_lengths.tolist() for i in range(len(self.target_keys)+1)]))

    seq_start = [numpy.zeros((seq_lengths.shape[1],),'int64')]
    if not self._seq_start:
      self._seq_start = [numpy.zeros((seq_lengths.shape[1],),'int64')]
    for l in seq_lengths:
      self._seq_lengths.append(numpy.array(l))
      seq_start.append(seq_start[-1] + l)
    self.tags += tags
    self.file_seq_start.append(seq_start)
    nseqs = len(seq_start) - 1
    for i in range(nseqs):
      self.tag_idx[tags[i]] = i + self._num_seqs
    self._num_seqs += nseqs
    self.file_index.extend([len(self.files) - 1] * nseqs)
    self.file_start.append(self.file_start[-1] + nseqs)
    self._num_timesteps = sum([s[0] for s in self._seq_lengths])
    self._num_codesteps = [ sum([s[i] for s in self._seq_lengths]) for i in range(1,len(self._seq_lengths[0])) ]
    if 'maxCTCIndexTranscriptionLength' in fin.attrs:
      self.max_ctc_length = max(self.max_ctc_length, fin.attrs['maxCTCIndexTranscriptionLength'])
    if len(fin['inputs'].shape) == 1:  # sparse
      num_inputs = [fin.attrs[attr_inputPattSize], 1]
    else:
      num_inputs = [fin['inputs'].shape[1], len(fin['inputs'].shape)] #fin.attrs[attr_inputPattSize]
    if self.num_inputs == 0:
      self.num_inputs = num_inputs[0]
    assert self.num_inputs == num_inputs[0], "wrong input dimension in file %s (expected %s got %s)" % (
                                             filename, self.num_inputs, num_inputs[0])
    if 'targets/size' in fin:
      num_outputs = { k : [fin['targets/size'].attrs[k], len(fin['targets/data'][k].shape)] for k in fin['targets/size'].attrs }
    else:
      num_outputs = { 'classes' : fin.attrs[attr_numLabels] }
    num_outputs["data"] = num_inputs
    if not self.num_outputs:
      self.num_outputs = num_outputs
    assert self.num_outputs == num_outputs, "wrong dimensions in file %s (expected %s got %s)" % (
                                            filename, self.num_outputs, num_outputs)
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
        self.data_dtype[name] = str(fin['targets/data'][name].dtype) if tdim > 1 else 'int32'
        #print name, self.data_dtype[name], fin['targets/data'][name][0:3][...]
        if self.data_dtype[name] == 'int32':
          self.targets[name] = numpy.zeros((self._num_codesteps[self.target_keys.index(name)],), dtype=theano.config.floatX) - 1
        else:
          self.targets[name] = numpy.zeros((self._num_codesteps[self.target_keys.index(name)],tdim), dtype=theano.config.floatX) - 1
    else:
      self.targets = { 'classes' : numpy.zeros((self._num_timesteps,), dtype=theano.config.floatX)  }
      self.data_dtype['classes'] = 'int32'
    self.data_dtype["data"] = fin['inputs'].dtype
    assert len(self.target_keys) == len(self._seq_lengths[0]) - 1
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
    file_info = [ [] for l in range(len(self.files)) ]; """ :type: list[list[int]] """
    # file_info[i] is (sorted seq idx from selection, real seq idx)
    for idc in selection:
      ids = self._seq_index[idc]
      file_info[self.file_index[ids]].append((idc,ids))
    for i in range(len(self.files)):
      if len(file_info[i]) == 0:
        continue
      print >> log.v4, "loading file", self.files[i]
      fin = h5py.File(self.files[i], 'r')
      for idc, ids in file_info[i]:
        s = ids - self.file_start[i]
        p = self.file_seq_start[i][s]
        l = self._seq_lengths[ids]
        if 'targets' in fin:
          for k in fin['targets/data']:
            ldx = self.target_keys.index(k) + 1
            self.targets[k][self.get_seq_start(idc)[ldx]:self.get_seq_start(idc)[ldx] + l[ldx]] = fin['targets/data/' + k][p[ldx] : p[ldx] + l[ldx]][...]
        self._set_alloc_intervals_data(idc, data=fin['inputs'][p[0] : p[0] + l[0]][...])
      fin.close()
    gc.collect()
    assert self.is_cached(start, end)

  def get_tag(self, sorted_seq_idx):
    ids = self._seq_index[self._index_map[sorted_seq_idx]]
    return self.tags[ids]

  def is_data_sparse(self, key):
    if key in self.num_outputs:
      return self.num_outputs[key][1] == 1
    if self.get_data_dtype(key).startswith("int"):
      return True
    return False

  def get_data_dtype(self, key):
    return self.data_dtype[key]

  def len_info(self):
    return ", ".join(["HDF dataset",
                      "sequences: %i" % self.num_seqs,
                      "frames: %i" % self.get_num_timesteps()])
