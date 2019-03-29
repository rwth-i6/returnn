
"""
Provides :class:`HDFDataset`.
"""

from __future__ import print_function
import collections
import gc
import h5py
import numpy
from CachedDataset import CachedDataset
from CachedDataset2 import CachedDataset2
from Dataset import Dataset, DatasetSeq
from Log import log
import Util


# Common attribute names for HDF dataset, which should be used in order to be proceed with HDFDataset class.
attr_seqLengths = 'seqLengths'
attr_inputPattSize = 'inputPattSize'
attr_numLabels = 'numLabels'
attr_times = 'times'
attr_ctcIndexTranscription = 'ctcIndexTranscription'


class HDFDataset(CachedDataset):

  def __init__(self, files=None, use_cache_manager=False, **kwargs):
    """
    :param None|list[str] files:
    :param bool use_cache_manager: uses :func:`Util.cf` for files
    """
    super(HDFDataset, self).__init__(**kwargs)
    self._use_cache_manager = use_cache_manager
    self.files = []; """ :type: list[str] """  # file names
    self.h5_files = []  # type: list[h5py.File]
    self.file_start = [0]
    self.file_seq_start = []; """ :type: list[numpy.ndarray] """
    self.file_index = []; """ :type: list[int] """
    self.data_dtype = {}; ":type: dict[str,str]"
    self.data_sparse = {}; ":type: dict[str,bool]"
    if files:
      for fn in files:
        self.add_file(fn)

  @staticmethod
  def _decode(s):
    """
    :param str|bytes s:
    :rtype: str
    """
    if not isinstance(s, str):
      s = s.decode("utf-8")
    s = s.split('\0')[0]
    return s

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
    if self._use_cache_manager:
      filename = Util.cf(filename)
    fin = h5py.File(filename, "r")
    if 'targets' in fin:
      self.labels = {
        k: [self._decode(item) for item in fin["targets/labels"][k][...].tolist()]
        for k in fin['targets/labels']}
    if not self.labels:
      labels = [item.split('\0')[0] for item in fin["labels"][...].tolist()]; """ :type: list[str] """
      self.labels = {'classes': labels}
      assert len(self.labels['classes']) == len(labels), "expected " + str(len(self.labels['classes'])) + " got " + str(len(labels))
    self.files.append(filename)
    if self.cache_byte_size_total_limit == 0:
      self.h5_files.append(fin)
    print("parsing file", filename, file=log.v5)
    if 'times' in fin:
      if self.timestamps is None:
        self.timestamps = fin[attr_times][...]
      else:
        self.timestamps = numpy.concatenate([self.timestamps, fin[attr_times][...]], axis=0)
    seq_lengths = fin[attr_seqLengths][...]
    if 'targets' in fin:
      self.target_keys = sorted(fin['targets/labels'].keys())
    else:
      self.target_keys = ['classes']

    if len(seq_lengths.shape) == 1:
      seq_lengths = numpy.array(zip(*[seq_lengths.tolist() for i in range(len(self.target_keys)+1)]))

    if len(self._seq_lengths) == 0:
      self._seq_lengths = numpy.array(seq_lengths)
    else:
      self._seq_lengths = numpy.concatenate((self._seq_lengths, seq_lengths), axis=0)
    if not self._seq_start:
      self._seq_start = [numpy.zeros((seq_lengths.shape[1],), 'int64')]
    seq_start = numpy.zeros((seq_lengths.shape[0] + 1, seq_lengths.shape[1]), dtype="int64")
    numpy.cumsum(seq_lengths, axis=0, dtype="int64", out=seq_start[1:])
    self._tags += fin["seqTags"][...].tolist()
    self.file_seq_start.append(seq_start)
    nseqs = len(seq_start) - 1
    self._num_seqs += nseqs
    self.file_index.extend([len(self.files) - 1] * nseqs)
    self.file_start.append(self.file_start[-1] + nseqs)
    self._num_timesteps += numpy.sum(seq_lengths[:, 0])
    if self._num_codesteps is None:
      self._num_codesteps = [0 for i in range(1, len(seq_lengths[0]))]
    for i in range(1, len(seq_lengths[0])):
      self._num_codesteps[i - 1] += numpy.sum(seq_lengths[:, i])
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
      num_outputs = {}
      for k in fin['targets/size'].attrs:
        if numpy.isscalar(fin['targets/size'].attrs[k]):
          num_outputs[k] = (int(fin['targets/size'].attrs[k]), len(fin['targets/data'][k].shape))
        else:  # hdf_dump will give directly as tuple
          assert fin['targets/size'].attrs[k].shape == (2,)
          num_outputs[k] = tuple([int(v) for v in fin['targets/size'].attrs[k]])
    else:
      num_outputs = {'classes': [int(fin.attrs[attr_numLabels]), 1]}
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
        self.data_dtype[str(name)] = str(fin['targets/data'][name].dtype)
        self.targets[str(name)] = None
        if str(name) not in self.num_outputs:
          ndim = len(fin['targets/data'][name].shape)
          dim = 1 if ndim == 1 else fin['targets/data'][name].shape[-1]
          self.num_outputs[str(name)] = (dim, ndim)
    self.data_dtype["data"] = str(fin['inputs'].dtype)
    assert len(self.target_keys) == len(self._seq_lengths[0]) - 1
    if self.cache_byte_size_total_limit > 0:
      fin.close()  # we always reopen them

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
    if self.cache_byte_size_total_limit == 0:
      # Just don't use the alloc intervals, or any of the other logic. Just load it on the fly when requested.
      return
    selection = self.insert_alloc_interval(start, end)
    assert len(selection) <= end - start, "DEBUG: more sequences requested (" + str(len(selection)) + ") as required (" + str(end-start) + ")"
    self.preload_set |= set(range(start,end)) - set(selection)
    file_info = [ [] for l in range(len(self.files)) ]; """ :type: list[list[int]] """
    # file_info[i] is (sorted seq idx from selection, real seq idx)
    for idc in selection:
      if self.sample(idc):
        ids = self._seq_index[idc]
        file_info[self.file_index[ids]].append((idc,ids))
      else:
        self.preload_set.add(idc)
    for i in range(len(self.files)):
      if len(file_info[i]) == 0:
        continue
      if start == 0 or self.cache_byte_size_total_limit > 0:  # suppress with disabled cache
        print("loading file %d/%d (seq range %i-%i)" % (i+1, len(self.files), start, end), self.files[i], file=log.v4)
      fin = h5py.File(self.files[i], 'r')
      inputs = fin['inputs']
      targets = None
      if 'targets' in fin:
        targets = {k: fin['targets/data/' + k] for k in fin['targets/data']}
      for idc, ids in file_info[i]:
        s = ids - self.file_start[i]
        p = self.file_seq_start[i][s]
        l = self._seq_lengths[ids]
        if 'targets' in fin:
          for k in fin['targets/data']:
            if self.targets[k] is None:
              self.targets[k] = numpy.zeros(
                (self._num_codesteps[self.target_keys.index(k)],) + targets[k].shape[1:], dtype=self.data_dtype[k]) - 1
            ldx = self.target_keys.index(k) + 1
            self.targets[k][self.get_seq_start(idc)[ldx]:self.get_seq_start(idc)[ldx] + l[ldx]] = targets[k][p[ldx] : p[ldx] + l[ldx]]
        self._set_alloc_intervals_data(idc, data=inputs[p[0] : p[0] + l[0]])
        self.preload_set.add(idc)
      fin.close()
    gc.collect()

  def get_data(self, seq_idx, key):
    if self.cache_byte_size_total_limit > 0:  # Use the cache?
      return super(HDFDataset, self).get_data(seq_idx, key)

    # Otherwise, directly read it from file now.
    real_seq_idx = self._seq_index[seq_idx]
    file_idx = self.file_index[real_seq_idx]
    fin = self.h5_files[file_idx]

    real_file_seq_idx = real_seq_idx - self.file_start[file_idx]
    pos = self.file_seq_start[file_idx][real_file_seq_idx]
    seq_len = self._seq_lengths[real_seq_idx]

    if key == "data":
      inputs = fin['inputs']
      data = inputs[pos[0]:pos[0] + seq_len[0]]
    else:
      assert 'targets' in fin
      targets = fin['targets/data/' + key]
      ldx = self.target_keys.index(key) + 1
      data = targets[pos[ldx]:pos[ldx] + seq_len[ldx]]
    return data

  def get_input_data(self, sorted_seq_idx):
    if self.cache_byte_size_total_limit > 0:  # Use the cache?
      return super(HDFDataset, self).get_input_data(sorted_seq_idx)
    return self.get_data(sorted_seq_idx, "data")

  def get_targets(self, target, sorted_seq_idx):
    if self.cache_byte_size_total_limit > 0:  # Use the cache?
      return super(HDFDataset, self).get_targets(target, sorted_seq_idx)
    return self.get_data(sorted_seq_idx, target)

  def _get_tag_by_real_idx(self, real_idx):
    s = self._tags[real_idx]
    s = self._decode(s)
    return s

  def get_tag(self, sorted_seq_idx):
    ids = self._seq_index[self._index_map[sorted_seq_idx]]
    return self._get_tag_by_real_idx(ids)

  def get_all_tags(self):
    return list(map(self._decode, self._tags))

  def get_total_num_seqs(self):
    return len(self._tags)

  def is_data_sparse(self, key):
    if self.get_data_dtype(key).startswith("int"):
      if key in self.num_outputs:
        return self.num_outputs[key][1] <= 1
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
      assert seq_data.dtype == self.dtype

    self.num_features = self.stream['feature_names'].shape[0]
    self.feature_type = 1

  def get_data(self, seq_name):
    return self.stream['data'][seq_name][:]

  def get_seq_length(self, seq_name):
    return self.stream['data'][seq_name].shape[0]


class SegmentAlignmentStreamParser(StreamParser):
  def __init__(self, *args, **kwargs):
    super(SegmentAlignmentStreamParser, self).__init__(*args, **kwargs)

    for s in self.seq_names:
      seq_data = self.stream['data'][s]

      if self.dtype is None:
        self.dtype = seq_data.dtype
      assert seq_data.dtype == self.dtype

      assert len(seq_data.shape) == 2
      assert seq_data.shape[1] == 2

    self.num_features = self.stream['feature_names'].shape[0]
    self.feature_type = 1

  def get_data(self, seq_name):
    # we return flatted two-dimensional data where the 2nd dimension is 2 [classs, segment end]
    length = self.get_seq_length(seq_name) // 2
    segments = self.stream['data'][seq_name][:]

    alignment = numpy.zeros((length,2,), dtype=self.dtype)
    num_segments = segments.shape[0]
    seg_end = 0
    for i in range(num_segments):
      next_seg_end = seg_end + segments[i,1]
      alignment[seg_end:next_seg_end,0] = segments[i,0]  # set class
      alignment[      next_seg_end-1,1] = 1              # mark segment end
      seg_end = next_seg_end

    alignment = alignment.reshape((-1,))
    return alignment

  def get_seq_length(self, seq_name):
    return 2 * sum(self.stream['data'][seq_name][:,1])


class NextGenHDFDataset(CachedDataset2):
  """
  """

  parsers = { 'feature_sequence'  : FeatureSequenceStreamParser,
              'sparse'            : SparseStreamParser,
              'segment_alignment' : SegmentAlignmentStreamParser }

  def __init__(self, input_stream_name, files=None, **kwargs):
    """
    :param str input_stream_name:
    :param None|list[str] files:
    """
    super(NextGenHDFDataset, self).__init__(**kwargs)

    self.input_stream_name = input_stream_name

    self.files           = []
    self.h5_files        = []
    self.all_seq_names   = []
    self.seq_name_to_idx = {}
    self.file_indices    = []
    self.seq_order       = []
    self.all_parsers     = collections.defaultdict(list)

    if files:
      for fn in files:
        self.add_file(fn)

  def add_file(self, path):
    self.files.append(path)
    self.h5_files.append(h5py.File(path))

    cur_file = self.h5_files[-1]

    assert {'seq_names', 'streams'}.issubset(set(cur_file.keys())), "%s does not contain all required datasets/groups" % path

    seqs = list(cur_file['seq_names'])
    norm_seqs = [self._normalize_seq_name(s) for s in seqs]

    prev_no_seqs      = len(self.all_seq_names)
    seqs_in_this_file = len(seqs)
    self.seq_name_to_idx.update(zip(seqs, range(prev_no_seqs, prev_no_seqs + seqs_in_this_file + 1)))

    self.all_seq_names.extend(seqs)
    self.file_indices.extend([len(self.files) - 1] * len(seqs))

    all_streams = set(cur_file['streams'].keys())
    assert self.input_stream_name in all_streams, "%s does not contain the input stream %s" % (path, self.input_stream_name)

    parsers = { name : NextGenHDFDataset.parsers[stream.attrs['parser']](norm_seqs, stream) for name, stream in cur_file['streams'].items()}
    for k, v in parsers.items():
      self.all_parsers[k].append(v)

    if len(self.files) == 1:
      self.num_outputs = { name : [parser.num_features, parser.feature_type] for name, parser in parsers.items() }
      self.num_inputs = self.num_outputs[self.input_stream_name][0]
    else:
      num_features = [(name, self.num_outputs[name][0], parser.num_features) for name, parser in parsers.items()]
      assert all(nf[1] == nf[2] for nf in num_features), '\n'.join("Number of features does not match for parser %s: %d (config) vs. %d (hdf-file)" % nf for nf in num_features if nf[1] != nf[2])

  def initialize(self):
    total_seqs               = len(self.all_seq_names)
    self._num_seqs           = total_seqs
    self._estimated_num_seqs = total_seqs

    super(NextGenHDFDataset, self).initialize()

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    :type epoch: int|None
    :param list[str] | None seq_list: In case we want to set a predefined order.
    """
    super(NextGenHDFDataset, self).init_seq_order(epoch, seq_list)

    if seq_list is not None:
      self.seq_order = [self.seq_name_to_idx[s] for s in seq_list]
    else:
      epoch = epoch or 1
      self.seq_order = self.get_seq_order_for_epoch(epoch, len(self.all_seq_names), self._get_seq_length)

  def _get_seq_length(self, orig_seq_idx):
    """
    :type orig_seq_idx: int
    :rtype int
    """
    parser = self.all_parsers[self.input_stream_name][self.file_indices[orig_seq_idx]]
    return parser.get_seq_length(self._normalize_seq_name(self.all_seq_names[orig_seq_idx]))

  def _collect_single_seq(self, seq_idx):
    """
    :type seq_idx: int
    :rtype: DatasetSeq
    """
    if seq_idx >= len(self.seq_order):
      return None

    real_seq_index   = self.seq_order[seq_idx]
    file_index       = self.file_indices[real_seq_index]
    seq_name         = self.all_seq_names[real_seq_index]
    norm_seq_name    = self._normalize_seq_name(seq_name)
    targets          = { name : parsers[file_index].get_data(norm_seq_name) for name, parsers in self.all_parsers.items() }
    features         = targets[self.input_stream_name]
    return DatasetSeq(seq_idx=seq_idx,
                      seq_tag=seq_name,
                      features=features,
                      targets=targets)

  def get_data_dtype(self, key):
    if key == 'data':
      return self.get_data_dtype(self.input_stream_name)
    return self.all_parsers[key][0].get_dtype()

  @staticmethod
  def _normalize_seq_name(name):
    """
    HDF Datasets cannot contain '/' in their name (this would create subgroups), we do not
    want this and thus replace it with '\' when asking for data from the parsers
    :type name: string
    :rtype: string
    """
    return name.replace('/', '\\')


class SiameseHDFDataset(CachedDataset2):
  """
  SiameseHDFDataset class allows to do sequence sampling for weakly-supervised training.
  It accepts data in the format of NextGenHDFDataset and performs sampling of sequence triplets before each epoch.
  Triplets are tuples of the format: (anchor seq, random seq with the same label, random seq with a different label)
  Here we assume that each dataset from the input .hdf has a single label.
  In the config we can access streams by e.g. ["data:features_0"], ["data:features_1"], ["data:features_2"].
  Split names depend on stream names in the input data, e.g. "features", "data", "classes", etc.
  SiameseHDFDataset method _collect_single_seq(self, seq_idx) returns a DatasetSeq with extended dictionary of targets.
  "data:features_0" key stands for features of anchor sequences from the input data.
  In NexGenHDFDataset it would correspond to "data:features" or "data".
  "data:features_1" is a key, which denote a pair of "data:features_0".
  For each anchor sequence SiameseHDFDataset randomly samples a sequence with the same label.
  "data:features_2" denotes the third element in a triplet tuple.
  For each anchor sequence SiameseHDFDataset randomly samples a sequence with a different label.
  Targets are splitted into different streams as well, e.g. "data:classes_0", "data:classes_1", "data:classes_2".

  SiameseHDFDataset also supports non-uniform sampling and accepts a path to .npz matrix.
  Rows of this matrix should have probabilities for each of the classes to be sampled.
  This probability distribution might reflect class similarities.

  This dataset might be useful for metric learning,
  where we want to learn such representations of input sequences, that those which belong to the same class are close together,
  while those with different labels should have representations far away from each other.
  """
  parsers = { 'feature_sequence'  : FeatureSequenceStreamParser,
              'sparse'            : SparseStreamParser,
              'segment_alignment' : SegmentAlignmentStreamParser }

  def __init__(self, input_stream_name, seq_label_stream='words', class_distribution=None, files=None, *args, **kwargs):
    """
    :param str input_stream_name: name of a feature stream
    :param str seq_label_stream: name of a stream with labels
    :param str class_distribution: path to .npz file of size n x n (n is a number of classes),
           where each line i contains probs of other classes to be picked in triplets
           when sampling a pair for element from class i
    :param files: list of paths to .hdf files
    :param args: dict[str]
    :param kwargs: dict[str]
    """
    super(SiameseHDFDataset, self).__init__(*args, **kwargs)
    self.input_stream_name = input_stream_name
    if class_distribution is not None:
      self.class_probs = numpy.load(class_distribution)['arr_0']
    else:
      self.class_probs = None
    self.files = []
    self.h5_files = []
    self.all_seq_names = []  # all_seq_names[(int)seq_index] = (string) sequence_name
    self.seq_name_to_idx = {}  # (string) sequence_name -> seq_index (int)
    self.file_indices = []  # file_indices[(int)seq_index] = file_index => indices of files to which sequences belongs to
    self.seq_order = []
    self.all_parsers = collections.defaultdict(list)
    self.seq_to_target = {}  # (string) sequence_name -> (int) class_index
    self.target_to_seqs = {}  # (int) class_index -> (string) sequence_names
    self.curr_epoch_triplets = []
    self.targets_stream = seq_label_stream
    if files:
      for fn in files:
        self.add_file(fn)

  def add_file(self, path):
    """
    register input files and sequences
    :param path: path to single .hdf file
    """
    self.files.append(path)
    self.h5_files.append(h5py.File(path, "r"))
    cur_file = self.h5_files[-1]
    assert {'seq_names', 'streams'}.issubset(set(cur_file.keys())), "%s does not contain all required datasets/groups" % path
    seqs = list(cur_file['seq_names'])
    norm_seqs = [self._normalize_seq_name(s) for s in seqs]

    prev_no_seqs = len(self.all_seq_names)
    seqs_in_this_file = len(seqs)
    self.seq_name_to_idx.update(zip(seqs, range(prev_no_seqs, prev_no_seqs + seqs_in_this_file + 1)))

    self.all_seq_names.extend(seqs)
    self.file_indices.extend([len(self.files) - 1] * len(seqs))

    all_streams = set(cur_file['streams'].keys())
    assert self.input_stream_name in all_streams, "%s does not contain the input stream %s" % (path, self.input_stream_name)
    if self.targets_stream is not None:
      assert self.targets_stream in all_streams, "%s does not contain the input stream %s" % (path, self.targets_stream)

    parsers = { name : SiameseHDFDataset.parsers[stream.attrs['parser']](norm_seqs, stream) for name, stream in cur_file['streams'].items()} # name - stream name (words, features, orth_features)
    for k, v in parsers.items():
      self.all_parsers[k].append(v)

    if len(self.files) == 1:
      self.num_outputs = { name : [parser.num_features, parser.feature_type] for name, parser in parsers.items() }
      self.num_inputs = self.num_outputs[self.input_stream_name][0]
    else:
      num_features = [(name, self.num_outputs[name][0], parser.num_features) for name, parser in parsers.items()]
      assert all(nf[1] == nf[2] for nf in num_features), '\n'.join("Number of features does not match for parser %s: %d (config) vs. %d (hdf-file)" % nf for nf in num_features if nf[1] != nf[2])

  def initialize(self):
    """
    initialize target_to_seqs and seq_to_target dicts
    """
    self.target_to_seqs = {}
    self.seq_to_target = {}
    for cur_file in self.h5_files:
      sequences = cur_file['streams'][self.targets_stream]['data'] # (string) seq_name -> (int) word_id
      for seq_name, value in sequences.items():
        seq_targ = int(value.value[0])
        if seq_targ in self.target_to_seqs.keys():
          self.target_to_seqs[seq_targ].append(seq_name)
        else:
          self.target_to_seqs[seq_targ] = [seq_name]
        self.seq_to_target[seq_name] = seq_targ

    super(SiameseHDFDataset, self).initialize()

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    :param epoch int|None : current epoch id
    :param list[str] | None seq_list: In case we want to set a predefined order.
    """
    super(SiameseHDFDataset, self).init_seq_order(epoch, seq_list)

    if seq_list is not None:
      self.seq_order = [self.seq_name_to_idx[s] for s in seq_list]
    else:
      epoch = epoch or 1
      self.seq_order = self.get_seq_order_for_epoch(epoch, len(self.all_seq_names), self._get_seq_length)

    # init random seed for siamese triplet sampling
    numpy.random.seed()
    self._init_triplets()

  def _init_triplets(self):
    """
    sample triplet for current epoch: (anchor_sample, sample_from_same_class, sample_from_diff_class)
    """
    self.curr_epoch_triplets = []
    # here we will intialize triplets before each epoch
    for seq_idx, real_seq_idx in enumerate(self.seq_order):
      seq_name = self.all_seq_names[real_seq_idx]
      seq_target = self.seq_to_target[seq_name]
      # randomly sample same pair
      same_words = self.target_to_seqs[seq_target]
      if len(same_words) > 1:
        pair_word_idx = numpy.random.randint(0, len(same_words))
        # sample again if pair sequence is the same sequence
        while same_words[pair_word_idx] == seq_name:
          pair_word_idx = numpy.random.randint(0, len(same_words))
        pair_seq_name = same_words[pair_word_idx]
        real_pair_idx = self.seq_name_to_idx[pair_seq_name]
      else:
        real_pair_idx = real_seq_idx
      # randomly sample third element from another class
      rand_target_val = self._sample_diff_class(seq_target)
      # sample again if random class is the same class as anchor
      while rand_target_val == seq_target:
        rand_target_val = self._sample_diff_class(seq_target)
      # sample an example from random_target
      rand_seq_id = numpy.random.randint(0, len(self.target_to_seqs[rand_target_val]))
      rand_seq_name = self.target_to_seqs[rand_target_val][rand_seq_id]
      real_third_idx = self.seq_name_to_idx[rand_seq_name]
      self.curr_epoch_triplets.append(tuple((real_seq_idx, real_pair_idx, real_third_idx)))

  def _sample_diff_class(self, anchor_seq_target):
    """
    draw a class from a space of all classes
    :param int anchor_seq_target: id of anchor class
    :return: int id of a drawn class
    """
    if self.class_probs is not None:
      distrib = self.class_probs[anchor_seq_target]
      classes = list(map(int, list(self.target_to_seqs.keys())))
      probs = numpy.array(distrib[classes])
      probs /= numpy.sum(probs)
      rand_target_val = numpy.random.choice(classes, size=1, p=probs)[0]
    else:
      random_target = numpy.random.randint(0, len(list(self.target_to_seqs.keys())))
      rand_target_val = list(self.target_to_seqs.keys())[random_target]

    return rand_target_val

  def _collect_single_seq(self, seq_idx):
    """
    :param int seq_idx: sequence id
    :rtype: DatasetSeq
    """
    if seq_idx >= len(self.seq_order):
      return None
    real_seq_index = self.seq_order[seq_idx]
    seq_name = self.all_seq_names[real_seq_index]

    curr_triplet = self.curr_epoch_triplets[seq_idx]
    targets = {}
    for id, sample in enumerate(curr_triplet):
      real_sample_seq_idx = sample
      sample_seq_name = self.all_seq_names[real_sample_seq_idx]
      sample_seq_file_index = self.file_indices[real_sample_seq_idx]
      norm_sample_seq_name = self._normalize_seq_name(sample_seq_name)
      for name, parsers in self.all_parsers.items():
        targets['%s_%d' % (name, id)] = parsers[sample_seq_file_index].get_data(norm_sample_seq_name)

    targets['%s_all' % self.targets_stream] = numpy.concatenate((targets['%s_0' % self.targets_stream], targets['%s_1'% self.targets_stream], targets['%s_2' % self.targets_stream]), axis=0)
    features = targets['%s_%d' % (self.input_stream_name, 0)]
    return DatasetSeq(seq_idx=seq_idx,
                      seq_tag=seq_name,
                      features=features,
                      targets=targets)

  def _get_seq_length(self, orig_seq_idx):
    """
    :type orig_seq_idx: int
    :rtype int
    """
    parser = self.all_parsers[self.input_stream_name][self.file_indices[orig_seq_idx]]
    return parser.get_seq_length(self._normalize_seq_name(self.all_seq_names[orig_seq_idx]))

  @staticmethod
  def _normalize_seq_name(name):
    """
    HDF Datasets cannot contain '/' in their name (this would create subgroups), we do not
    want this and thus replace it with '\' when asking for data from the parsers
    :type name: string
    :rtype: string
    """
    return name.replace('/', '\\')

  def is_data_sparse(self, key):
    """
    :param str key: e.g. "features_0" or "orth_features_0" or "words_0"
    :return: whether the data is sparse
    :rtype: bool
    """
    if "features" in key:
      return False
    return True

  def get_data_dim(self, key):
    """
    :param str key: e.g. "features_0", "features_1", "classes_0", etc.
    :return: number of classes, no matter if sparse or not
    :rtype: int
    """
    k = "_".join(key.split("_")[:-1]) if "_" in key else key
    if k in self.num_outputs:
      return self.num_outputs[k][0]
    return 1  # unknown


class SimpleHDFWriter:
  def __init__(self, filename, dim, labels=None, ndim=None):
    """
    :param str filename:
    :param int|None dim:
    :param int ndim: counted without batch
    :param list[str]|None labels:
    """
    if ndim is None:
      if dim is None:
        ndim = 1
      else:
        ndim = 2
    from Util import hdf5_strings
    self.dim = dim
    self.ndim = ndim
    self.labels = labels
    if labels:
      assert len(labels) == dim
    self._file = h5py.File(filename, "w")

    self._file.attrs['numTimesteps'] = 0  # we will increment this on-the-fly
    self._other_num_time_steps = 0
    self._file.attrs['inputPattSize'] = dim or 1
    self._file.attrs['numDims'] = 1  # ignored?
    self._file.attrs['numLabels'] = dim or 1
    self._file.attrs['numSeqs'] = 0  # we will increment this on-the-fly
    if labels:
      hdf5_strings(self._file, 'labels', labels)
    else:
      self._file.create_dataset('labels', (0,), dtype="S5")

    self._datasets = {}  # type: dict[str, h5py.Dataset]
    self._tags = []  # type: list[str]
    self._seq_lengths = self._file.create_dataset("seqLengths", (0, 2), dtype='i', maxshape=(None, 2))

  def _insert_h5_inputs(self, raw_data):
    """
    Inserts a record into the hdf5-file.
    Resizes if necessary.

    :param numpy.ndarray raw_data: shape=(time,data) or shape=(time,)
    """
    assert raw_data.ndim >= 1
    name = "inputs"
    if name not in self._datasets:
      self._datasets[name] = self._file.create_dataset(
        name, raw_data.shape, raw_data.dtype, maxshape=tuple(None for _ in raw_data.shape))
    else:
      old_shape = self._datasets[name].shape
      self._datasets[name].resize((old_shape[0] + raw_data.shape[0],) + old_shape[1:])
    # append raw data to dataset
    self._datasets[name][self._file.attrs['numTimesteps']:] = raw_data
    self._file.attrs['numTimesteps'] += raw_data.shape[0]
    self._file.attrs['numSeqs'] += 1

  def _insert_h5_other(self, data_key, raw_data, dtype=None, add_time_dim=False, dim=None):
    """
    :param str data_key:
    :param numpy.ndarray|int|float|list[int] raw_data: shape=(time,data) or shape=(time,) or shape=()...
    :param str dtype:
    :param bool add_time_dim:
    :param int|None dim:
    """
    if isinstance(raw_data, (int, float, list)):
      raw_data = numpy.array(raw_data)
    assert isinstance(raw_data, numpy.ndarray)
    if add_time_dim:
      raw_data = raw_data[None, :]
    assert raw_data.ndim > 0 and raw_data.shape[0] > 0
    if dtype:
      raw_data = raw_data.astype(dtype)
    if dim is None:
      if raw_data.ndim > 1:
        dim = raw_data.shape[-1]
      else:
        dim = 1  # dummy
    assert data_key != "inputs"
    name = data_key
    # Keep consistent with _insert_h5_inputs.
    if name not in self._datasets:
      if 'targets/data' not in self._file:
        self._file.create_group('targets/data')
      if 'targets/size' not in self._file:
        self._file.create_group('targets/size')
      if "targets/labels" not in self._file:
        self._file.create_group("targets/labels")
      Util.hdf5_strings(self._file, "targets/labels/%s" % data_key, ["dummy-label"])
      self._datasets[name] = self._file['targets/data'].create_dataset(
        data_key, raw_data.shape, raw_data.dtype, maxshape=tuple(None for _ in raw_data.shape))
      self._file['targets/size'].attrs[data_key] = [dim, raw_data.ndim]  # (dim, ndim)
    else:
      old_shape = self._datasets[name].shape
      self._datasets[name].resize((old_shape[0] + raw_data.shape[0],) + old_shape[1:])

    assert self._file.attrs['numSeqs'] > 0 and self._seq_lengths.shape[0] > 0  # assume _insert_h5_inputs called before

    if self._seq_lengths[self._file.attrs['numSeqs'] - 1, 1]:
      assert self._seq_lengths[self._file.attrs['numSeqs'] - 1, 1] == raw_data.shape[0]
    else:
      self._seq_lengths[self._file.attrs['numSeqs'] - 1, 1] = raw_data.shape[0]
      self._other_num_time_steps += raw_data.shape[0]

    offset = self._other_num_time_steps - raw_data.shape[0]
    hdf_data = self._datasets[name]
    hdf_data[offset:] = raw_data

  def insert_batch(self, inputs, seq_len, seq_tag, extra=None):
    """
    :param numpy.ndarray inputs: shape=(n_batch,time,data) (or (n_batch,time), or (n_batch,time1,time2), ...)
    :param list[int]|dict[int,list[int]|numpy.ndarray] seq_len: sequence lengths (per axis, excluding batch axis)
    :param list[str|bytes] seq_tag: sequence tags of length n_batch
    :param dict[str,numpy.ndarray]|None extra:
    """
    n_batch = len(seq_tag)
    assert n_batch == inputs.shape[0]
    assert inputs.ndim == self.ndim + 1  # one more for the batch-dim
    if not isinstance(seq_len, dict):
      seq_len = {0: seq_len}
    assert isinstance(seq_len, dict)
    assert all([isinstance(key, int) and isinstance(value, (list, numpy.ndarray)) for (key, value) in seq_len.items()])
    ndim_with_seq_len = self.ndim - (1 if self.dim else 0)
    assert all([0 <= key < ndim_with_seq_len for key in seq_len.keys()])
    assert len(seq_len) == ndim_with_seq_len
    assert all([n_batch == len(value) for (key, value) in seq_len.items()])
    assert all([max(value) == inputs.shape[key + 1] for (key, value) in seq_len.items()])
    if self.dim:
      assert self.dim == inputs.shape[-1]
    if extra:
      assert all([n_batch == value.shape[0] for value in extra.values()])

    seqlen_offset = self._seq_lengths.shape[0]
    self._seq_lengths.resize(seqlen_offset + n_batch, axis=0)

    for i in range(n_batch):
      self._tags.append(seq_tag[i])
      # Note: Currently, our HDFDataset does not support to have multiple axes with dynamic length.
      # Thus, we flatten all together, and calculate the flattened seq len.
      # (Ignore this if there is only a single time dimension.)
      flat_seq_len = numpy.prod([seq_len[axis][i] for axis in range(ndim_with_seq_len)])
      assert flat_seq_len > 0
      flat_shape = [flat_seq_len]
      if self.dim:
        flat_shape.append(self.dim)
      self._seq_lengths[seqlen_offset + i, 0] = flat_seq_len
      data = inputs[i]
      data = data[tuple([slice(None, seq_len[axis][i]) for axis in range(ndim_with_seq_len)])]
      data = numpy.reshape(data, flat_shape)
      self._insert_h5_inputs(data)
      if len(seq_len) > 1:
        # Note: Because we have flattened multiple axes with dynamic len into a single one,
        # we want to store the individual axes lengths. We store those in a separate data entry "sizes".
        # Note: We could add a dummy time-dim for this "sizes", and then have a feature-dim = number of axes.
        # However, we keep it consistent to how we handled it in our 2D MDLSTM experiments.
        self._insert_h5_other(
          "sizes", [seq_len[axis][i] for axis in range(ndim_with_seq_len)], add_time_dim=False, dtype="int32")
      if extra:
        assert len(seq_len) == 1  # otherwise you likely will get trouble with seq len mismatch
        for key, value in extra.items():
          self._insert_h5_other(key, value[i])

  def close(self):
    max_tag_len = max([len(d) for d in self._tags]) if self._tags else 0
    self._file.create_dataset('seqTags', shape=(len(self._tags),), dtype="S%i" % (max_tag_len + 1))
    for i, tag in enumerate(self._tags):
      self._file['seqTags'][i] = numpy.array(tag, dtype="S%i" % (max_tag_len + 1))
    self._file.close()


class HDFDatasetWriter:
  def __init__(self, filename):
    """
    :param str filename: for the HDF to write
    """
    print("Creating HDF dataset file %s" % filename, file=log.v3)
    self.filename = filename
    self.file = h5py.File(filename, "w")

  def close(self):
    self.file.close()

  def dump_from_dataset(self, dataset, epoch=1, start_seq=0, end_seq=float("inf"), use_progress_bar=True):
    """
    :param Dataset dataset: could be any dataset implemented as child of Dataset
    :param int epoch: for dataset
    :param int start_seq:
    :param int|float end_seq:
    :param bool use_progress_bar:
    """
    from Util import NumbersDict, human_size, progress_bar_with_time, try_run, PY3
    hdf_dataset = self.file

    print("Work on epoch: %i" % epoch, file=log.v3)
    dataset.init_seq_order(epoch)

    data_keys = sorted(dataset.get_data_keys())
    print("Data keys:", data_keys, file=log.v3)
    if "orth" in data_keys:  # special workaround for now, not handled
      data_keys.remove("orth")
    data_target_keys = [key for key in dataset.get_target_list() if key in data_keys]
    data_input_keys = [key for key in data_keys if key not in data_target_keys]
    assert len(data_input_keys) > 0 and len(data_target_keys) > 0
    if len(data_input_keys) > 1:
      if "data" in data_input_keys:
        default_data_input_key = "data"
      else:
        raise Exception("not sure which input data key to use from %r" % (data_input_keys,))
    else:
      default_data_input_key = data_input_keys[0]
    print("Using input data key:", default_data_input_key)
    if len(data_target_keys) > 1:
      if "classes" in data_target_keys:
        default_data_target_key = "classes"
      else:
        raise Exception("not sure which target data key to use from %r" % (data_target_keys,))
    else:
      default_data_target_key = data_target_keys[0]
    print("Using target data key:", default_data_target_key)

    hdf_data_key_map = {key: key for key in data_keys if key != default_data_input_key}
    if "data" in hdf_data_key_map:
      hdf_data_key_map["data"] = "classes"  # Replace "data" which is reserved for input key in HDFDataset.
      assert "classes" not in hdf_data_key_map

    # We need to do one run through the dataset to collect some stats like total len.
    print("Collect stats, iterate through all data...", file=log.v3)
    seq_idx = start_seq
    seq_idxs = []
    seq_tags = []
    seq_lens = []
    total_seq_len = NumbersDict(0)
    max_tag_len = 0
    dataset_num_seqs = try_run(lambda: dataset.num_seqs, default=None)  # can be unknown
    if end_seq != float("inf"):
      if dataset_num_seqs is not None:
        dataset_num_seqs = min(dataset_num_seqs, end_seq)
      else:
        dataset_num_seqs = end_seq
    if dataset_num_seqs is not None:
      dataset_num_seqs -= start_seq
      assert dataset_num_seqs > 0
    while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= end_seq:
      seq_idxs += [seq_idx]
      dataset.load_seqs(seq_idx, seq_idx + 1)
      seq_len = dataset.get_seq_length(seq_idx)
      seq_lens += [seq_len]
      tag = dataset.get_tag(seq_idx)
      seq_tags += [tag]
      max_tag_len = max(len(tag), max_tag_len)
      total_seq_len += seq_len
      if use_progress_bar and dataset_num_seqs is not None:
        progress_bar_with_time(float(seq_idx - start_seq) / dataset_num_seqs)
      seq_idx += 1
    num_seqs = len(seq_idxs)

    assert num_seqs > 0
    shapes = {}
    for data_key in data_keys:
      assert data_key in total_seq_len.dict
      shape = [total_seq_len[data_key]]
      shape += dataset.get_data_shape(data_key)
      print("Total len of %r is %s, shape %r, dtype %s" % (
        data_key, human_size(shape[0]), shape, dataset.get_data_dtype(data_key)), file=log.v3)
      shapes[data_key] = shape

    print("Set seq tags...", file=log.v3)
    hdf_dataset.create_dataset('seqTags', shape=(num_seqs,), dtype="S%i" % (max_tag_len + 1))
    for i, tag in enumerate(seq_tags):
      hdf_dataset['seqTags'][i] = numpy.array(tag, dtype="S%i" % (max_tag_len + 1))
      if use_progress_bar:
        progress_bar_with_time(float(i) / num_seqs)

    print("Set seq len info...", file=log.v3)
    hdf_dataset.create_dataset(attr_seqLengths, shape=(num_seqs, 2), dtype="int32")
    for i, seq_len in enumerate(seq_lens):
      data_len = seq_len[default_data_input_key]
      targets_len = seq_len[default_data_target_key]
      for data_key in data_target_keys:
        assert seq_len[data_key] == targets_len, "different lengths in multi-target not supported"
      if targets_len is None:
        targets_len = data_len
      hdf_dataset[attr_seqLengths][i] = [data_len, targets_len]
      if use_progress_bar:
        progress_bar_with_time(float(i) / num_seqs)

    print("Create arrays in HDF...", file=log.v3)
    hdf_dataset.create_group('targets/data')
    hdf_dataset.create_group('targets/size')
    hdf_dataset.create_group('targets/labels')
    for data_key in data_keys:
      if data_key == default_data_input_key:
        hdf_dataset.create_dataset(
          'inputs', shape=shapes[data_key], dtype=dataset.get_data_dtype(data_key))
      else:
        hdf_dataset['targets/data'].create_dataset(
          hdf_data_key_map[data_key], shape=shapes[data_key], dtype=dataset.get_data_dtype(data_key))
        hdf_dataset['targets/size'].attrs[hdf_data_key_map[data_key]] = dataset.num_outputs[data_key]
      if data_key in dataset.labels:
        labels = dataset.labels[data_key]
        if PY3:
          labels = [label.encode("utf8") for label in labels]
        assert len(labels) == dataset.num_outputs[data_key][0]
      else:
        labels = ["%s-class-%i" % (data_key, i) for i in range(dataset.get_data_dim(data_key))]
      print("Labels for %s:" % data_key, labels[:3], "...", file=log.v5)
      max_label_len = max(map(len, labels))
      if data_key != default_data_input_key:
        hdf_dataset['targets/labels'].create_dataset(hdf_data_key_map[data_key],
                                                     (len(labels),), dtype="S%i" % (max_label_len + 1))
        for i, label in enumerate(labels):
          hdf_dataset['targets/labels'][hdf_data_key_map[data_key]][i] = numpy.array(
            label, dtype="S%i" % (max_label_len + 1))

    # Again iterate through dataset, and set the data
    print("Write data...", file=log.v3)
    dataset.init_seq_order(epoch)
    offsets = NumbersDict(0)
    for seq_idx, tag in zip(seq_idxs, seq_tags):
      dataset.load_seqs(seq_idx, seq_idx + 1)
      tag_ = dataset.get_tag(seq_idx)
      assert tag == tag_  # Just a check for sanity. We expect the same order.
      seq_len = dataset.get_seq_length(seq_idx)
      for data_key in data_keys:
        if data_key == default_data_input_key:
          hdf_data = hdf_dataset['inputs']
        else:
          hdf_data = hdf_dataset['targets/data'][hdf_data_key_map[data_key]]
        data = dataset.get_data(seq_idx, data_key)
        hdf_data[offsets[data_key]:offsets[data_key] + seq_len[data_key]] = data

      if use_progress_bar:
        progress_bar_with_time(float(offsets[default_data_input_key]) / total_seq_len[default_data_input_key])

      offsets += seq_len

    assert offsets == total_seq_len  # Sanity check.

    # Set some old-format attribs. Not needed for newer CRNN versions.
    hdf_dataset.attrs[attr_inputPattSize] = dataset.num_inputs
    hdf_dataset.attrs[attr_numLabels] = dataset.num_outputs.get(default_data_target_key, (0, 0))[0]

    print("All done.", file=log.v3)
