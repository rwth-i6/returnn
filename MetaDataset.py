
from Dataset import Dataset, DatasetSeq, init_dataset
from Util import NumbersDict, load_json


class MetaDataset(Dataset):

  def __init__(self,
               seq_list_file, seq_lens_file,
               datasets,
               data_map, data_dims,
               data_1_of_k=None, data_dtypes=None,
               window=1, **kwargs):
    """
    :param str seq_list_file: filename. line-separated
    :param str seq_lens_file: filename. json. dict[str,dict[str,int]], seq-tag -> data-key -> len
    :param dict[str,dict[str]] datasets: dataset-key -> dataset-kwargs. including keyword 'class' and maybe 'files'
    :param dict[str,(str,str)] data_map: self-data-key -> (dataset-key, dataset-data-key).
      Should contain 'data' as key. Also defines the target-list, which is all except 'data'.
    :param dict[str,int] data_dims: self-data-key -> data-dimension.
    :param dict[str,bool] data_1_of_k: self-data-key -> whether it is 1-of-k or not. automatic if not specified
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

    self.data_dims = data_dims
    for v in data_dims.values():
      assert isinstance(v, int), "all values must be int in %r" % data_dims
    assert "data" in data_dims
    self.num_inputs = data_dims["data"]
    self.num_outputs = {data_key: data_dims[data_key] for data_key in self.target_list}

    self.data_1_of_k = {data_key: _select_1_of_k(data_key, data_1_of_k, data_dtypes) for data_key in self.data_keys}
    self.data_dtypes = {data_key: _select_dtype(data_key, self.data_1_of_k, data_dtypes) for data_key in self.data_keys}

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

    self.epoch = None
    self.init_seq_order()

  def init_seq_order(self, epoch=None, seq_list=None):
    if not epoch:
      epoch = 1
    self.expected_load_seq_start = 0
    self.reached_final_seq = False
    self.added_data = []; " :type: list[DatasetSeq] "
    self._num_timesteps_accumulated = 0

    if self.epoch == epoch:
      return
    self.epoch = epoch

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

  def _cleanup_old_seqs(self, seq_idx_end):
    i = 0
    while i < len(self.added_data):
      if self.added_data[i].seq_idx >= seq_idx_end:
        break
      i += 1
    del self.added_data[:i]

  def _get_seq(self, seq_idx):
    for data in self.added_data:
      if data.seq_idx == seq_idx:
        return data
    return None

  def is_cached(self, start, end):
    # Always False, to force that we call self._load_seqs().
    # This is important for our buffer management.
    return False

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
    if self.added_data:
      start = max(self.added_data[-1].seq_idx + 1, start)
    if end > self.num_seqs:
      end = self.num_seqs
    if end >= self.num_seqs:
      self.reached_final_seq = True
    for dataset in self.datasets.values():
      dataset.load_seqs(start, end)
      for seq_idx in range(start, end):
        self._check_dataset_seq(dataset, seq_idx)
    seqs = [self._collect_single_seq(seq_idx=seq_idx) for seq_idx in range(start, end)]
    self._num_timesteps_accumulated += sum([seq.num_frames for seq in seqs])
    self.added_data += seqs

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
    if dataset_data_key == "data":
      return dataset.get_input_data(seq_idx)
    return dataset.get_targets(dataset_data_key, seq_idx)

  def _collect_single_seq(self, seq_idx):
    """
    :type seq_idx: int
    :rtype: DatasetSeq
    """
    seq_tag = self.seq_list_ordered[seq_idx]
    features = self._get_data(seq_idx, "data")
    targets = {target: self._get_data(seq_idx, target) for target in self.target_list}
    return DatasetSeq(seq_idx=seq_idx, seq_tag=seq_tag, features=features, targets=targets)

  @property
  def num_seqs(self):
    return self._num_seqs

  def get_num_timesteps(self):
    if self._num_timesteps:
      return self._num_timesteps
    else:
      assert self.reached_final_seq
      return self._num_timesteps_accumulated

  def get_seq_length(self, sorted_seq_idx):
    if self._seq_lens:
      return self._seq_lens[self.seq_list_ordered[sorted_seq_idx]]
    # get_seq_length() can be called before the seq is loaded via load_seqs().
    # Thus, we just call load_seqs() ourselves here.
    assert sorted_seq_idx >= self.expected_load_seq_start
    self.load_seqs(self.expected_load_seq_start, sorted_seq_idx + 1)
    return self._get_seq(sorted_seq_idx).num_frames

  def get_target_list(self):
    return self.target_list

  def get_target_dim(self, target):
    """
    :type target: str
    :return: 1 for hard labels, num_outputs[target] for soft labels
    """
    if self.data_1_of_k[target]:
      return 1
    else:
      return self.data_dims[target]

  def get_target_type(self, target):
    return self.data_dtypes[target]

  def get_tag(self, sorted_seq_idx):
    return self.seq_list_ordered[sorted_seq_idx]

  def get_input_data(self, sorted_seq_idx):
    return self._get_seq(sorted_seq_idx).features

  def get_targets(self, target, sorted_seq_idx):
    return self._get_seq(sorted_seq_idx).targets[target]

  def get_ctc_targets(self, sorted_seq_idx):
    assert self._get_seq(sorted_seq_idx).ctc_targets



def _simple_to_bool(v):
  if v == 0: v = False
  if v == 1: v = True
  assert isinstance(v, bool)
  return v

def _select_1_of_k(key, data_1_of_k, data_dtypes):
  if data_1_of_k and key in data_1_of_k:
    v = data_1_of_k[key]
    return _simple_to_bool(v)
  if data_dtypes and key in data_dtypes:
    v = data_dtypes[key]
    if v.startswith("int"):
      return True  # int is likely a 1-of-k
    return False
  if key == "data":
    return False  # the data (input) is likely not 1-of-k
  return True  # all targets are likely 1-of-k encoded (for classification)

def _select_dtype(key, data_1_of_k, data_dtypes):
  if data_dtypes and key in data_dtypes:
    v = data_dtypes[key]
    assert isinstance(v, str)  # e.g. "int32" or "float32"
    return v
  if data_1_of_k and key in data_1_of_k:
    if data_1_of_k[key]:
      return "int32"  # standard for 1-of-k
    else:
      return "float32"  # standard otherwise
  if key == "data":
    return "float32"  # standard for input
  return "int32"  # all targets are likely 1-of-k encoded (for classification)


