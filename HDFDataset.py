from __future__ import print_function
import collections
import gc
import h5py
import numpy
import random
import theano
from CachedDataset import CachedDataset
from CachedDataset2 import CachedDataset2
from Dataset import Dataset, DatasetSeq
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
    decode = lambda s: s if isinstance(s, str) else s.decode('utf-8')
    if 'targets' in fin:
      self.labels = { k : [ decode(item).split('\0')[0] for item in fin["targets/labels"][k][...].tolist() ] for k in fin['targets/labels'] }
    if not self.labels:
      labels = [ item.split('\0')[0] for item in fin["labels"][...].tolist() ]; """ :type: list[str] """
      self.labels = { 'classes' : labels }
      assert len(self.labels['classes']) == len(labels), "expected " + str(len(self.labels['classes'])) + " got " + str(len(labels))
    tags = [ decode(item).split('\0')[0] for item in fin["seqTags"][...].tolist() ]; """ :type: list[str] """
    self.files.append(filename)
    print("parsing file", filename, file=log.v5)
    if 'times' in fin:
      if self.timestamps is None:
        self.timestamps = fin[attr_times][...]
      else:
        self.timestamps = numpy.concatenate([self.timestamps, fin[attr_times][...]],axis=0) #.extend(fin[attr_times][...].tolist())
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
    self._num_timesteps += sum([s[0] for s in seq_lengths])
    if self._num_codesteps is None:
      self._num_codesteps = [ 0 for i in range(1,len(seq_lengths[0])) ]
    for i in range(1,len(seq_lengths[0])):
      self._num_codesteps[i-1] += sum([s[i] for s in seq_lengths])
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
        self.targets[name] = None
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
      print("loading file", self.files[i], file=log.v4)
      fin = h5py.File(self.files[i], 'r')
      for idc, ids in file_info[i]:
        s = ids - self.file_start[i]
        p = self.file_seq_start[i][s]
        l = self._seq_lengths[ids]
        if 'targets' in fin:
          for k in fin['targets/data']:
            if self.targets[k] is None:
              if self.data_dtype[k] == 'int32':
                self.targets[k] = numpy.zeros((self._num_codesteps[self.target_keys.index(k)],), dtype=theano.config.floatX) - 1
              else:
                self.targets[k] = numpy.zeros((self._num_codesteps[self.target_keys.index(k)],tdim), dtype=theano.config.floatX) - 1
            ldx = self.target_keys.index(k) + 1
            self.targets[k][self.get_seq_start(idc)[ldx]:self.get_seq_start(idc)[ldx] + l[ldx]] = fin['targets/data/' + k][p[ldx] : p[ldx] + l[ldx]]
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

# ------------------------------------------------------------------------------

class StreamParser(object):
  def __init__(self, seq_names, stream):
    self.seq_names = seq_names
    self.stream = stream

    self.num_features = None
    self.feature_type = None  # 1 for sparse, 2 for dense
    self.dtype        = None

  def get_data(self, seq_name):
    raise NotImplementedError()

  def get_seq_length(self, seq_name):
    raise NotImplementedError()

  def get_dtype(self):
    return self.dtype


class FeatureSequenceStreamParser(StreamParser):
  def __init__(self, *args, **kwargs):
    super(FeatureSequenceStreamParser, self).__init__(*args, **kwargs)

    for s in self.seq_names:
      seq_data = self.stream['data'][s]
      assert len(seq_data.shape) == 2

      if self.num_features is None:
        self.num_features = seq_data.shape[1]
      if self.dtype is None:
        self.dtype = seq_data.dtype

      assert seq_data.shape[1] == self.num_features
      assert seq_data.dtype    == self.dtype

    self.feature_type = 2

  def get_data(self, seq_name):
    return self.stream['data'][seq_name][...]

  def get_seq_length(self, seq_name):
    return self.stream['data'][seq_name].shape[0]


class SparseStreamParser(StreamParser):
  def __init__(self, *args, **kwargs):
    super(SparseStreamParser, self).__init__(*args, **kwargs)

    for s in self.seq_names:
      seq_data = self.stream['data'][s]
      assert len(seq_data.shape) == 1

      if self.dtype is None:
        self.dtype = seq_data.dtype
      assert seq_data.dtype    == self.dtype

    self.num_features = self.stream['feature_names'].shape[0]
    self.feature_type = 1

  def get_data(self, seq_name):
    return self.stream['data'][seq_name][:]

  def get_seq_length(self, seq_name):
    return self.stream['data'][seq_name].shape[0]


class NextGenHDFDataset(CachedDataset2):
  """
  """

  parsers = { 'feature_sequence' : FeatureSequenceStreamParser,
              'sparse'           : SparseStreamParser }

  def __init__(self, input_stream_name, *args, **kwargs):
    super(NextGenHDFDataset, self).__init__(*args, **kwargs)

    self.input_stream_name = input_stream_name

    self.files           = []
    self.h5_files        = []
    self.all_seq_names   = []
    self.seq_name_to_idx = {}
    self.file_indices    = []
    self.seq_order       = []
    self.all_parsers     = collections.defaultdict(list)


  def add_file(self, path):
    self.files.append(path)
    self.h5_files.append(h5py.File(path))

    cur_file = self.h5_files[-1]

    assert {'seq_names', 'streams'}.issubset(set(cur_file.keys())), "%s does not contain all required datasets/groups" % path

    seqs = list(cur_file['seq_names'])

    prev_no_seqs      = len(self.all_seq_names)
    seqs_in_this_file = len(seqs)
    self.seq_name_to_idx.update(zip(seqs, range(prev_no_seqs, prev_no_seqs + seqs_in_this_file + 1)))

    self.all_seq_names.extend(seqs)
    self.file_indices.extend([len(self.files) - 1] * len(seqs))

    all_streams = set(cur_file['streams'].keys())
    assert self.input_stream_name in all_streams, "%s does not contain the input stream %s" % (path, self.input_stream_name)

    parsers = { name : NextGenHDFDataset.parsers[stream.attrs['parser']](seqs, stream) for name, stream in cur_file['streams'].items()}
    for k, v in parsers.items():
      self.all_parsers[k].append(v)

    if len(self.files) == 1:
      self.num_outputs = { name : [parser.num_features, parser.feature_type] for name, parser in parsers.items() }
      self.num_inputs = self.num_outputs[self.input_stream_name][0]
    else:
      num_features = [(name, self.num_outputs[name][0], parser.num_features) for name, parser in parsers.items()]
      assert all(nf[1] == nf[2] for nf in num_features), '\n'.join("Number of features does not match for parser %s: %d (config) vs. %d (hdf-file)" % nf for nf in num_features if nf[1] != nf[2])


  def initialize(self):
    super(NextGenHDFDataset, self).initialize()

    self._num_seqs           = len(self.all_seq_names)
    self._estimated_num_seqs = self._num_seqs


  def init_seq_order(self, epoch=None, seq_list=None):
    """
    :type epoch: int|None
    :param list[str] | None seq_list: In case we want to set a predefined order.
    """
    super(NextGenHDFDataset, self).init_seq_order(epoch, seq_list)

    if seq_list is not None:
      self.seq_order = [self.seq_name_to_idx[s] for s in seq_list]
    else:
      self.seq_order = self.get_seq_order_for_epoch(epoch, len(self.all_seq_names), self._get_seq_length)

  def _get_seq_length(self, orig_seq_idx):
    """
    :type orig_seq_idx: int
    :rtype int
    """
    parser = self.all_parsers[self.input_stream_name][self.file_indices[orig_seq_idx]]
    return parser.get_seq_length(self.all_seq_names[orig_seq_idx])


  def _collect_single_seq(self, seq_idx):
    """
    :type seq_idx: int
    :rtype: DatasetSeq
    """
    if seq_idx >= len(self.seq_order):
      return None

    real_seq_index = self.seq_order[seq_idx]
    file_index     = self.file_indices[real_seq_index]
    seq_name       = self.all_seq_names[real_seq_index]
    targets        = { name : parsers[file_index].get_data(seq_name) for name, parsers in self.all_parsers.items() }
    features       = targets[self.input_stream_name]
    return DatasetSeq(seq_idx=seq_idx,
                      seq_tag=seq_name,
                      features=features,
                      targets=targets)

  def get_data_dtype(self, key):
    if key == 'data':
      return self.get_data_dtype(self.input_stream_name)
    return self.all_parsers[key][0].get_dtype()
