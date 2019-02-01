
from __future__ import print_function
import collections
import functools as fun
import gc
import h5py
import numpy
import theano
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
          num_outputs[k] = (fin['targets/size'].attrs[k], len(fin['targets/data'][k].shape))
        else:  # hdf_dump will give directly as tuple
          assert fin['targets/size'].attrs[k].shape == (2,)
          num_outputs[k] = tuple(fin['targets/size'].attrs[k])
    else:
      num_outputs = {'classes': fin.attrs[attr_numLabels]}
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
    self.data_dtype["data"] = str(fin['inputs'].dtype)
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
      print("loading file %d/%d" % (i+1, len(self.files)), self.files[i], file=log.v4)
      fin = h5py.File(self.files[i], 'r')
      inputs = fin['inputs']
      if 'targets' in fin:
        targets = {k:fin['targets/data/' + k] for k in fin['targets/data']}
      if self.seq_ordering == 'default' or True:
        inputs = inputs[...]
        if 'targets' in fin:
          targets = {k:targets[k][...] for k in targets}
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
            self.targets[k][self.get_seq_start(idc)[ldx]:self.get_seq_start(idc)[ldx] + l[ldx]] = targets[k][p[ldx] : p[ldx] + l[ldx]]
        self._set_alloc_intervals_data(idc, data=inputs[p[0] : p[0] + l[0]])
        self.preload_set.add(idc)
      fin.close()
    gc.collect()

  def _get_tag_by_real_idx(self, real_idx):
    s = self._tags[real_idx]
    s = self._decode(s)
    return s

  def get_tag(self, sorted_seq_idx):
    ids = self._seq_index[self._index_map[sorted_seq_idx]]
    return self._get_tag_by_real_idx(ids)

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

  def __init__(self, input_stream_name, files=None, partition_epoch=1, **kwargs):
    """
    :param str input_stream_name:
    :param None|list[str] files:
    :param int partition_epoch:
    """
    super(NextGenHDFDataset, self).__init__(**kwargs)

    self.input_stream_name = input_stream_name
    self.partition_epoch   = partition_epoch

    self.files           = []
    self.h5_files        = []
    self.all_seq_names   = []
    self.seq_name_to_idx = {}
    self.file_indices    = []
    self.seq_order       = []
    self.all_parsers     = collections.defaultdict(list)

    self.partitions        = []
    self.current_partition = 1

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
    seqs_per_epoch           = total_seqs // self.partition_epoch
    self._num_seqs           = seqs_per_epoch
    self._estimated_num_seqs = seqs_per_epoch

    partition_sizes =   [seqs_per_epoch + 1] * (total_seqs % self.partition_epoch)\
                      + [seqs_per_epoch]     * (self.partition_epoch - total_seqs % self.partition_epoch)
    self.partitions = fun.reduce(lambda a, x: a + [a[-1] + x], partition_sizes, [0])  # cumulative sum

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
      self.current_partition = (epoch - 1) % self.partition_epoch
      partition_size         = self.partitions[self.current_partition + 1] - self.partitions[self.current_partition]
      self.seq_order         = self.get_seq_order_for_epoch(epoch, partition_size, self._get_seq_length)

  def _get_seq_length(self, orig_seq_idx):
    """
    :type orig_seq_idx: int
    :rtype int
    """
    partition_offset = self.partitions[self.current_partition]
    parser = self.all_parsers[self.input_stream_name][self.file_indices[partition_offset + orig_seq_idx]]
    return parser.get_seq_length(self._normalize_seq_name(self.all_seq_names[partition_offset + orig_seq_idx]))

  def _collect_single_seq(self, seq_idx):
    """
    :type seq_idx: int
    :rtype: DatasetSeq
    """
    if seq_idx >= len(self.seq_order):
      return None

    partition_offset = self.partitions[self.current_partition]
    real_seq_index   = partition_offset + self.seq_order[seq_idx]
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

