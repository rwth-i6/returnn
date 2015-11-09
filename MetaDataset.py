
from Dataset import Dataset, DatasetSeq, init_dataset, convert_data_dims
from CachedDataset2 import CachedDataset2
from Util import NumbersDict, load_json
from Log import log
from random import Random


class MetaDataset(CachedDataset2):
  """
  This wraps around one or multiple datasets and might provide extra information.
  Every dataset is expected to provide the the same sequences, where the sequence list
  is given by a file.
  """

  def __init__(self,
               seq_list_file, seq_lens_file,
               datasets,
               data_map, data_dims,
               data_dtypes=None,
               window=1, **kwargs):
    """
    :param str seq_list_file: filename. line-separated
    :param str seq_lens_file: filename. json. dict[str,dict[str,int]], seq-tag -> data-key -> len
    :param dict[str,dict[str]] datasets: dataset-key -> dataset-kwargs. including keyword 'class' and maybe 'files'
    :param dict[str,(str,str)] data_map: self-data-key -> (dataset-key, dataset-data-key).
      Should contain 'data' as key. Also defines the target-list, which is all except 'data'.
    :param dict[str,(int,int)] data_dims: self-data-key -> data-dimension, len(shape) (1 ==> sparse repr).
    :param dict[str,str] data_dtypes: self-data-key -> dtype. automatic if not specified
    """
    assert window == 1  # not implemented
    super(MetaDataset, self).__init__(**kwargs)
    assert self.shuffle_frames_of_nseqs == 0  # not implemented. anyway only for non-recurrent nets

    self.seq_list_original = open(seq_list_file).read().splitlines()
    self.tag_idx = {tag: idx for (idx, tag) in enumerate(self.seq_list_original)}
    self._num_seqs = len(self.seq_list_original)

    self.data_map = data_map
    self.dataset_keys = set([m[0] for m in self.data_map.values()]); ":type: set[str]"
    self.data_keys = set(self.data_map.keys()); ":type: set[str]"
    assert "data" in self.data_keys
    self.target_list = sorted(self.data_keys - ["data"])

    data_dims = convert_data_dims(data_dims)
    self.data_dims = data_dims
    assert "data" in data_dims
    for key in self.target_list:
      assert key in data_dims
    self.num_inputs = data_dims["data"][0]
    self.num_outputs = data_dims

    self.data_dtypes = {data_key: _select_dtype(data_key, data_dims, data_dtypes) for data_key in self.data_keys}

    if seq_lens_file:
      seq_lens = load_json(filename=seq_lens_file)
      assert isinstance(seq_lens, dict)
      # dict[str,NumbersDict], seq-tag -> data-key -> len
      self._seq_lens = {tag: NumbersDict(l) for (tag, l) in seq_lens.items()}
    else:
      self._seq_lens = None

    if self._seq_lens:
      self._num_timesteps = sum([self._seq_lens[s] for s in self.seq_list_original])
    else:
      self._num_timesteps = None

    # Will only init the needed datasets.
    self.datasets = {key: init_dataset(datasets[key]) for key in self.dataset_keys}

  def init_seq_order(self, epoch=None, seq_list=None):
    need_reinit = self.epoch is None or self.epoch != epoch
    super(MetaDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)
    if not need_reinit:
      return False

    if seq_list:
      seq_index = [self.tag_idx[tag] for tag in seq_list]
    else:
      if self._seq_lens:
        get_seq_len = lambda s: self._seq_lens[self.seq_list_original[s]]["data"]
      else:
        get_seq_len = None
      seq_index = self.get_seq_order_for_epoch(epoch, self.num_seqs, get_seq_len)
    self.seq_list_ordered = [self.seq_list_original[s] for s in seq_index]

    for dataset in self.datasets.values():
      dataset.init_seq_order(epoch=epoch, seq_list=self.seq_list_ordered)
    return True

  def _load_seqs(self, start, end):
    for dataset in self.datasets.values():
      dataset.load_seqs(start, end)
      for seq_idx in range(start, end):
        self._check_dataset_seq(dataset, seq_idx)
    super(MetaDataset, self)._load_seqs(start=start, end=end)

  def _check_dataset_seq(self, dataset, seq_idx):
    """
    :type dataset: Dataset
    :type seq_idx: int
    """
    dataset_seq_tag = dataset.get_tag(seq_idx)
    self_seq_tag = self.get_tag(seq_idx)
    assert dataset_seq_tag == self_seq_tag

  def _get_data(self, seq_idx, data_key):
    """
    :type seq_idx: int
    :type data_key: str
    :rtype: numpy.ndarray
    """
    dataset_key, dataset_data_key = self.data_map[data_key]
    dataset = self.datasets[dataset_key]; ":type: Dataset"
    return dataset.get_data(seq_idx, dataset_data_key)

  def _collect_single_seq(self, seq_idx):
    """
    :type seq_idx: int
    :rtype: DatasetSeq
    """
    seq_tag = self.seq_list_ordered[seq_idx]
    features = self._get_data(seq_idx, "data")
    targets = {target: self._get_data(seq_idx, target) for target in self.target_list}
    return DatasetSeq(seq_idx=seq_idx, seq_tag=seq_tag, features=features, targets=targets)

  def get_seq_length(self, sorted_seq_idx):
    if self._seq_lens:
      return self._seq_lens[self.seq_list_ordered[sorted_seq_idx]]
    return super(MetaDataset, self).get_seq_length(sorted_seq_idx)

  def get_tag(self, sorted_seq_idx):
    return self.seq_list_ordered[sorted_seq_idx]

  def get_target_list(self):
    return self.target_list

  def get_data_dtype(self, key):
    dtype = self.data_dtypes[key]
    if self.added_data:
      assert super(MetaDataset, self).get_data_dtype(key) == dtype
    return dtype


class ConcatDataset(CachedDataset2):
  """
  This concatenates multiple datasets. They are expected to provide the same data-keys and data-dimensions.
  It will go through the datasets always in order.
  """

  def __init__(self, datasets, **kwargs):
    """
    :param list[dict[str]] datasets: list of kwargs for init_dataset
    """
    super(ConcatDataset, self).__init__(**kwargs)
    self.datasets = [init_dataset(d_kwargs) for d_kwargs in datasets]
    assert self.datasets
    self.num_inputs = self.datasets[0].num_inputs
    self.num_outputs = self.datasets[0].num_outputs
    self.labels = self.datasets[0].labels
    for ds in self.datasets[1:]:
      assert ds.num_inputs == self.num_inputs
      assert ds.num_outputs == self.num_outputs

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    :type epoch: int|None
    :param list[str] | None seq_list: In case we want to set a predefined order.
    """
    need_reinit = self.epoch is None or self.epoch != epoch
    super(ConcatDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)
    self.dataset_seq_idx_offsets = [0]
    if not need_reinit:
      return False

    if seq_list:  # reference order
      seq_lists = []
      for dataset in self.datasets:
        # This depends on the num_seqs of our childs.
        seq_lists += seq_list[:dataset.num_seqs]
        seq_list = seq_list[dataset.num_seqs:]
      assert len(seq_list) == 0  # we have consumed all
    else:
      seq_lists = [None] * len(self.datasets)
      if self.seq_ordering != "default":
        # Not sure about these cases (random / sorted). Maybe a separate implementation makes more sense.
        raise NotImplementedError("seq_ordering %s" % self.seq_ordering)

    assert len(seq_lists) == len(self.datasets)
    for dataset, sub_list in zip(self.datasets, seq_lists):
      dataset.init_seq_order(epoch=epoch, seq_list=sub_list)
    return True

  def _get_dataset_for_seq_idx(self, seq_idx):
    i = 0
    while i < len(self.dataset_seq_idx_offsets):
      if seq_idx + self.dataset_seq_idx_offsets[i] < 0:
        return i - 1
      i += 1
    return i - 1

  def _load_seqs(self, start, end):
    sub_start = start
    # We maybe need to call load_seqs on several of our datasets, thus we need this loop.
    while True:
      dataset_idx = self._get_dataset_for_seq_idx(sub_start)
      dataset = self.datasets[dataset_idx]
      dataset_seq_idx_start = sub_start + self.dataset_seq_idx_offsets[dataset_idx]
      dataset_seq_idx_end = end + self.dataset_seq_idx_offsets[dataset_idx]
      dataset.load_seqs(dataset_seq_idx_start, dataset_seq_idx_end)
      if dataset.is_less_than_num_seqs(dataset_seq_idx_end):
        # We are still inside this dataset and have loaded everything.
        # Thus we can stop now.
        break
      # We have reached the end of the dataset.
      if dataset_idx + 1 == len(self.datasets):
        # We are at the last dataset.
        break
      # Continue with the next one.
      self.dataset_seq_idx_offsets[dataset_idx + 1:dataset_idx + 2] = [
        self.dataset_seq_idx_offsets[dataset_idx] - dataset.num_seqs]
      sub_start = -self.dataset_seq_idx_offsets[dataset_idx + 1]
    super(ConcatDataset, self)._load_seqs(start=start, end=end)

  def _collect_single_seq(self, seq_idx):
    dataset_idx = self._get_dataset_for_seq_idx(seq_idx)
    dataset = self.datasets[dataset_idx]
    dataset_seq_idx = seq_idx + self.dataset_seq_idx_offsets[dataset_idx]
    seq_tag = dataset.get_tag(dataset_seq_idx)
    features = dataset.get_input_data(dataset_seq_idx)
    targets = {k: dataset.get_targets(k, dataset_seq_idx) for k in dataset.get_target_list()}
    return DatasetSeq(seq_idx=seq_idx, seq_tag=seq_tag, features=features, targets=targets)

  @property
  def num_seqs(self):
    return sum([ds.num_seqs for ds in self.datasets])

  def get_target_list(self):
    return self.datasets[0].get_target_list()


class CombinedDataset(CachedDataset2):
  """
  This combines multiple different datasets, which provide different data-sources.
  E.g. one can provide am-dataset with data:acoustic-features -> classes:characters (acoustic model train data),
  and lm-dataset provides just data:characters (language model train data).
  You provide a mapping as in MetaDataset, e.g.
  am-data -> am-dataset:data, am-classes -> am-dataset:classes, lm-data -> lm-dataset:data.
  For each sequence idx, it will select one of the given datasets, fill in the data-keys of this dataset
  and will return empty sequences for the remaining datasets.
  The selection of the dataset will be random and equally distributed, over the sum of num-seqs.
  """

  def __init__(self,
               datasets,
               data_map, data_dims,
               data_dtypes=None,
               window=1, **kwargs):
    """
    :param dict[str,dict[str]] datasets: dataset-key -> dataset-kwargs. including keyword 'class' and maybe 'files'
    :param dict[str,(str,str)] data_map: self-data-key -> (dataset-key, dataset-data-key).
      Should contain 'data' as key. Also defines the target-list, which is all except 'data'.
    :param dict[str,(int,int)] data_dims: self-data-key -> data-dimension, len(shape) (1 ==> sparse repr).
    :param dict[str,str] data_dtypes: self-data-key -> dtype. automatic if not specified
    """
    assert window == 1  # not implemented
    super(CombinedDataset, self).__init__(**kwargs)
    assert self.shuffle_frames_of_nseqs == 0  # not implemented. anyway only for non-recurrent nets

    self.data_map = data_map
    self.dataset_keys = set([m[0] for m in self.data_map.values()]); ":type: set[str]"
    self.dataset_idxs = dict(enumerate(sorted(self.dataset_keys)))  # idx -> dataset-key
    self.data_keys = set(self.data_map.keys()); ":type: set[str]"
    assert "data" in self.data_keys
    self.target_list = sorted(self.data_keys - ["data"])

    data_dims = convert_data_dims(data_dims)
    self.data_dims = data_dims
    assert "data" in data_dims
    for key in self.target_list:
      assert key in data_dims
    self.num_inputs = data_dims["data"][0]
    self.num_outputs = data_dims

    self.data_dtypes = {data_key: _select_dtype(data_key, data_dims, data_dtypes) for data_key in self.data_keys}

    # Will only init the needed datasets.
    self.datasets = {key: init_dataset(datasets[key]) for key in self.dataset_keys}

    self._num_seqs = sum([ds.num_seqs for ds in self.datasets.values()])

  def _canonical_seqs_dataset_idxs(self):
    """
    :returns: list of dataset-idx, via self.dataset_idxs, so that we cover the sum of num-seqs
    :rtype: list[int]
    """
    l = []
    for i in range(len(self.datasets)):
      dataset = self.datasets[self.dataset_idxs[i]]
      l += [i] * dataset.num_seqs
    return l

  def _dataset_seq_idxs(self, seqs_dataset_idx):
    """
    :returns: list of (dataset-idx, dataset-seq-idx)
    :rtype: list[(int,int)]
    """
    l = []
    seq_idx_counter = [0] * len(self.datasets)  # dataset-idx -> dataset-seq-idx
    for dataset_idx in seqs_dataset_idx:
      seq_idx = seq_idx_counter[dataset_idx]
      seq_idx_counter[dataset_idx] += 1
      l += [(dataset_idx, seq_idx)]
    return l

  def init_seq_order(self, epoch=None, seq_list=None):
    assert seq_list is None, "seq_list not supported for %s" % self.__class__
    need_reinit = self.epoch is None or self.epoch != epoch
    super(CombinedDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)
    if not need_reinit:
      return False

    # We just select for which seq-idx we will use which dataset.
    # The ordering of the seqs in the datasets will not be set here
    # (do that in the config for the specific dataset).

    seqs_dataset_idx = self._canonical_seqs_dataset_idxs()
    if self.seq_ordering in ("default", "random"):  # default is random. this is different from base class!
      from random import Random
      rnd = Random(self.epoch)
      rnd.shuffle(seqs_dataset_idx)
    elif self.seq_ordering == "in-order":
      pass  # keep as-is
    elif self.seq_ordering == "reversed":
      seqs_dataset_idx = reversed(seqs_dataset_idx)
    else:
      raise Exception("seq_ordering %s not supported" % self.seq_ordering)

    self.dataset_seq_idxs = self._dataset_seq_idxs(seqs_dataset_idx)
    assert self.num_seqs == len(self.dataset_seq_idxs)

    for dataset in self.datasets.values():
      dataset.init_seq_order(epoch=epoch)
    return True

  def _load_seqs(self, start, end):
    for dataset in self.datasets.values():
      dataset.load_seqs(start, end)
      for seq_idx in range(start, end):
        self._check_dataset_seq(dataset, seq_idx)
    super(CombinedDataset, self)._load_seqs(start=start, end=end)

  def _check_dataset_seq(self, dataset, seq_idx):
    """
    :type dataset: Dataset
    :type seq_idx: int
    """
    dataset_seq_tag = dataset.get_tag(seq_idx)
    self_seq_tag = self.get_tag(seq_idx)
    assert dataset_seq_tag == self_seq_tag

  def _get_data(self, dataset_seq_idx, data_key):
    """
    :type dataset_seq_idx: int
    :type data_key: str
    :rtype: numpy.ndarray
    """
    dataset_key, dataset_data_key = self.data_map[data_key]
    dataset = self.datasets[dataset_key]; ":type: Dataset"
    return dataset.get_data(dataset_seq_idx, dataset_data_key)

  def _collect_single_seq(self, seq_idx):
    """
    :type seq_idx: int
    :rtype: DatasetSeq
    """
    dataset_idx, dataset_seq_idx = self.dataset_seq_idxs[seq_idx]
    dataset_key = self.dataset_idxs[dataset_idx]
    dataset = self.datasets[dataset_key]

    seq_tag = dataset_key + "-" + dataset.get_tag(dataset_seq_idx)
    features = self._get_data(dataset_seq_idx, "data")
    targets = {target: self._get_data(dataset_seq_idx, target) for target in self.target_list}
    return DatasetSeq(seq_idx=seq_idx, seq_tag=seq_tag, features=features, targets=targets)

  def get_target_list(self):
    return self.target_list

  def get_data_dtype(self, key):
    dtype = self.data_dtypes[key]
    if self.added_data:
      assert super(CombinedDataset, self).get_data_dtype(key) == dtype
    return dtype


class ChunkShuffleDataset(CachedDataset2):
  """
  This goes through a dataset, caches some recent chunks
  """

  def __init__(self, dataset,
               chunk_shuffle_cache=1000,
               batch_gen_batch_size=5000, batch_gen_max_seqs=1,
               batch_gen_recurrent_net=True,
               **kwargs):
    """
    :param dict[str] dataset: kwargs for init_dataset
    """
    super(ChunkShuffleDataset, self).__init__(**kwargs)
    self.dataset = init_dataset(dataset)
    assert self.dataset
    self.chunk_shuffle_cache = chunk_shuffle_cache
    self.batch_gen = None
    self.batch_gen_batch_size = batch_gen_batch_size
    self.batch_gen_max_seqs = batch_gen_max_seqs
    self.batch_gen_recurrent_net = batch_gen_recurrent_net
    self.num_inputs = self.dataset.num_inputs
    self.num_outputs = self.dataset.num_outputs
    self.labels = self.dataset.labels
    self.rng = Random(0)
    self.load_seqs_end = None

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    :type epoch: int|None
    :param list[str] | None seq_list: In case we want to set a predefined order.
    """
    need_reinit = self.epoch is None or self.epoch != epoch
    super(ChunkShuffleDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)
    self.load_seqs_end = 0
    self.rng.seed(epoch or 1)
    if not need_reinit:
      return False

    if seq_list:
      raise NotImplementedError("predefined order seq_list")
    if self.seq_ordering != "default":
      raise NotImplementedError("seq_ordering %s" % self.seq_ordering)

    self.dataset.init_seq_order(epoch=epoch)
    self.batch_gen = self.dataset.generate_batches(recurrent_net=self.batch_gen_recurrent_net,
                                                   batch_size=self.batch_gen_batch_size,
                                                   max_seqs=self.batch_gen_max_seqs)
    return True

  def _add_data(self, data, original_tag):
    """
    :type data: dict[str,numpy.ndarray]
    :type original_tag: str
    """
    features = data["data"]
    if not self.added_data:
      seq_idx = 0
      assert self.expected_load_seq_start == 0
    else:
      seq_idx = self.added_data[-1].seq_idx + 1
    tag = "%s.%i" % (original_tag, seq_idx)
    seq = DatasetSeq(seq_idx=seq_idx, features=features, targets=data, seq_tag=tag)
    self._num_timesteps_accumulated += seq.num_frames
    self.added_data += [seq]

  def _shuffle(self):
    start_seq_idx = self.added_data[0].seq_idx
    end_seq_idx = self.added_data[-1].seq_idx
    assert start_seq_idx < self.load_seqs_end < end_seq_idx
    start_idx = 0
    if start_seq_idx < self.load_seqs_end:
      start_idx = self.load_seqs_end - start_seq_idx
      assert self.added_data[start_idx].seq_idx == self.load_seqs_end
      start_seq_idx = self.load_seqs_end
    sublist = self.added_data[start_idx:]
    self.rng.shuffle(sublist)
    for i, seq in enumerate(sublist):
      seq.seq_idx = i + start_seq_idx
    assert sublist[-1].seq_idx == end_seq_idx
    self.added_data[start_idx:] = sublist

  def _add_more(self):
    """
    Adds each chunk/batch seq as a single DatasetSeq.
    :returns whether we added some more
    """
    if not self.batch_gen.has_more(): return False
    batch, = self.batch_gen.peek_next_n(1)
    assert batch.seqs
    self.dataset.load_seqs(batch.start_seq, batch.end_seq)

    used_data_keys = self.get_data_keys()
    for seq in batch.seqs:
      res_data = {}
      for k in used_data_keys:
        data = self.dataset.get_data(seq.seq_idx, k)
        if data is not None:
          res_data[k] = data[seq.seq_start_frame[k]:seq.seq_end_frame[k]]
      original_tag = self.dataset.get_tag(seq.seq_idx)
      self._add_data(data=res_data, original_tag=original_tag)

    self.batch_gen.advance(1)
    return True

  def _add_more_until(self, end, shuffle=False):
    if self.added_data and end <= self.added_data[-1].seq_idx: return True
    while self._add_more():
      assert self.added_data
      if end <= self.added_data[-1].seq_idx:
        if shuffle:
          self._shuffle()
        return True
    # We have reached the end.
    if not self.added_data:
      self._num_seqs = 0
      print >>log.v3, "warning: empty dataset"
    else:
      self._num_seqs = self.added_data[-1].seq_idx + 1
    self.reached_final_seq = True
    return False

  def is_less_than_num_seqs(self, seq_idx):
    """
    :type seq_idx: int
    :rtype: bool
    :returns whether seq_idx < num_seqs. In case num_seqs is not known in advance, it will wait
    until it knows that n is behind the end or that we have the seq.
    """
    if self._num_seqs is not None: return seq_idx < self._num_seqs
    if seq_idx < self.expected_load_seq_start: return True
    if self.added_data and seq_idx <= self.added_data[-1].seq_idx: return True
    return self._add_more_until(seq_idx)

  def _load_seqs(self, start, end):
    """
    :param int start: inclusive seq idx start
    :param int end: exclusive seq idx end
    """
    # We expect that start increase monotonic on each call
    # for not-yet-loaded data.
    # This will already be called with _load_seqs_superset indices.
    assert start >= self.expected_load_seq_start
    if start > self.expected_load_seq_start:
      # Cleanup old data.
      self._cleanup_old_seqs(start)
      self.expected_load_seq_start = start
    self.load_seqs_end = end
    self._add_more_until(end + self.chunk_shuffle_cache, shuffle=True)

  def _collect_single_seq(self, seq_idx):
    """
    :type seq_idx: int
    :rtype: DatasetSeq
    """
    assert False, "should not be called"

  def get_target_list(self):
    return self.dataset.get_target_list()


def _simple_to_bool(v):
  if v == 0: v = False
  if v == 1: v = True
  assert isinstance(v, bool)
  return v

def _select_dtype(key, data_dims, data_dtypes):
  if data_dtypes and key in data_dtypes:
    v = data_dtypes[key]
    assert isinstance(v, str)  # e.g. "int32" or "float32"
    return v
  assert key in data_dims
  if data_dims[key][1] == 1:  # sparse
    return "int32"  # standard for 1-of-k
  else:
    return "float32"  # standard otherwise


