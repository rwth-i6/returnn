"""
There are use cases in which we want to combine several datasets:

 * **Multimodality:** features from several datasets should be provided at the same time

   * Examples: multi-source translation, speech translation with source CTC loss
     for stability (needs both source audio and transcription)

 * **Multi-Task Learning:** several datasets should be used alternatingly,
   such that at each time the dataset of the corresponding task is selected

   * Examples: multi-task speech translation (either from audio or from text)

 * **Combination of Corpora:** the training data should be split into different datatsets.
   This allows creating a combined corpus dynamically
   and avoids manual concatenation/shuffling.

   * Examples: multi-lingual translation systems
     (datasets can be reused from corresponding bilingual systems)

The dataset classes MetaDataset and CombinedDataset which perform these tasks are implemented in MetaDataset.py.
"""

from __future__ import print_function

from Dataset import Dataset, DatasetSeq, init_dataset, convert_data_dims
from CachedDataset2 import CachedDataset2
from Util import NumbersDict, load_json
from Log import log
from random import Random
import numpy
import sys
import typing


class EpochWiseFilter:
  """
  Applies some filter to the sequences (e.g. by seq length) for some epoch.
  """

  def __init__(self, epochs_opts, debug_msg_prefix="EpochWiseFilter"):
    """
    :param dict[(int,int|None),dict[str]] epochs_opts: (ep_start, ep_end) -> epoch opts
    :param str debug_msg_prefix:
    """
    self.epochs_opts = epochs_opts
    self.debug_msg_prefix = debug_msg_prefix

  @classmethod
  def filter_epoch(cls, opts, seq_order, get_seq_len, debug_msg_prefix):
    """
    :param dict[str]|Util.CollectionReadCheckCovered opts:
    :param list[int] seq_order: list of seq idxs
    :param ((int)->int) get_seq_len: seq idx -> len
    :param str debug_msg_prefix:
    :return: new seq_order
    :rtype: list[int]
    """
    import Util
    if not isinstance(opts, Util.CollectionReadCheckCovered):
      opts = Util.CollectionReadCheckCovered(opts)
    if opts.get("max_mean_len"):
      max_mean_len = opts.get("max_mean_len")
      lens_and_seqs = numpy.array(sorted([(get_seq_len(idx), idx) for idx in seq_order]))
      best_num = Util.binary_search_any(
        cmp=lambda num: numpy.mean(lens_and_seqs[:num, 0]) - max_mean_len, low=1, high=len(lens_and_seqs) + 1)
      assert best_num is not None
      selected_seq_idxs = set(lens_and_seqs[:best_num, 1])
      # Select subset of seq_order. Keep order as-is.
      seq_order = [seq_idx for seq_idx in seq_order if seq_idx in selected_seq_idxs]
      print(
        ("%sOld mean seq len is %f, new is %f, requested max is %f."
         " Old num seqs is %i, new num seqs is %i.") %
        (debug_msg_prefix,
         float(numpy.mean(lens_and_seqs[:, 0])), float(numpy.mean(lens_and_seqs[:best_num, 0])),
         max_mean_len, len(lens_and_seqs), best_num),
        file=log.v4)
    opts.assert_all_read()
    return seq_order

  def filter(self, epoch, seq_order, get_seq_len):
    """
    :param int|None epoch:
    :param list[int] seq_order: list of seq idxs
    :param ((int)->int) get_seq_len: seq idx -> len
    :return: new seq_order
    """
    epoch = epoch or 1
    old_num_seqs = len(seq_order)
    any_filter = False
    for (ep_start, ep_end), value in sorted(self.epochs_opts.items()):
      if ep_start is None:
        ep_start = 1
      if ep_end is None or ep_end == -1:
        ep_end = sys.maxsize
      assert isinstance(ep_start, int) and isinstance(ep_end, int) and 1 <= ep_start <= ep_end
      assert isinstance(value, dict)
      if ep_start <= epoch <= ep_end:
        any_filter = True
        seq_order = self.filter_epoch(
          opts=value, debug_msg_prefix="%s, epoch %i. " % (self.debug_msg_prefix, epoch),
          seq_order=seq_order, get_seq_len=get_seq_len)
    if any_filter:
      print("%s, epoch %i. Old num seqs %i, new num seqs %i." % (
        self.debug_msg_prefix, epoch, old_num_seqs, len(seq_order)), file=log.v4)
    else:
      print("%s, epoch %i. No filter for this epoch." % (self.debug_msg_prefix, epoch), file=log.v4)
    return seq_order


class MetaDataset(CachedDataset2):
  """
  The MetaDataset is to be used in the case of **Multimodality**.
  Here, the datasets are expected to describe different features of the **same training sequences**.
  These features will all be available to the network at the same time.

  The datasets to be combined are given via the input parameter ``"datasets"``.
  To define which training examples from the different datasets belong together,
  a ``"seq_list_file"`` in pickle format has to be created.
  It contains a list of sequence tags for each dataset (see example below).
  Note, that in general each dataset type has its own tag format, e.g. for the TranslationDataset it is ``line-<n>``,
  for the SprintDataset it is ``<corpusname>/<recording>/<segment id>``.
  **Providing a sequence list can be omitted**, if the set of sequence tags is the same for all datasets.
  When using multiple ExternSprintDataset instances, the sprint segment file can be provided as sequence list.
  In this case the MetaDataset assumes that the sequences with equal tag correspond to each other.
  This e.g. works when combining TranslationDatasets if all the text files are sentence aligned.


  **Example of Sequence List:**

  .. code::

      { 'sprint': [
          'corpus/ted_1/1',
          'corpus/ted_1/2',
          'corpus/ted_1/3',
          'corpus/ted_1/4',
      'translation': [
          'line-0',
          'line-1',
          'line-2',
          'line-3']
      }

  Python dict stored in pickle file. E.g. the sequence tagged with 'corpus/ted_1/3' in the 'sprint' dataset
  corresponds to the sequence tagged 'line-2'
  in the 'translation' dataset.

  **Example of MetaDataset config:**

  .. code::

      train = {"class": "MetaDataset", "seq_list_file": "seq_list.pkl",
               "datasets": {"sprint": train_sprint, "translation": train_translation},
               "data_map": {"data": ("sprint", "data"),
               "target_text_sprint": ("sprint", "orth_classes"),
               "source_text": ("translation", "data"),
               "target_text": ("translation", "classes")},
               "seq_ordering": "random",
               "partition_epoch": 2,
      }

  This combines a SprintDataset and a TranslationDataset.
  These are defined as ``"train_sprint"`` and ``"train_translation"`` separately.
  *Note that the current implementation expects one input feature to be called "data".*
  """

  def __init__(self,
               datasets,
               data_map,
               seq_list_file=None,
               seq_order_control_dataset=None,
               seq_lens_file=None,
               data_dims=None,
               data_dtypes=None,
               window=1, **kwargs):
    """
    :param dict[str,dict[str]] datasets: dataset-key -> dataset-kwargs. including keyword 'class' and maybe 'files'
    :param dict[str,(str,str)] data_map: self-data-key -> (dataset-key, dataset-data-key).
      Should contain 'data' as key. Also defines the target-list, which is all except 'data'.
    :param str|None seq_list_file: filename. pickle. dict[str,list[str]], dataset-key -> list of sequence tags.
      Can be None if tag format is the same for all datasets.
        Then the sequence list will be default sequence order of default dataset (``data_map["data"][0]``),
        or seq_order_control_dataset.
    :param str|None seq_order_control_dataset: if set, this dataset will define the order for each epoch.
    :param str|None seq_lens_file: filename. json. dict[str,dict[str,int]], seq-tag -> data-key -> len.
      Use if getting sequence length from loading data is too costly.
    :param dict[str,(int,int)] data_dims: self-data-key -> data-dimension, len(shape) (1 ==> sparse repr).
       Deprecated/Only to double check. Read from data if not specified.
    :param dict[str,str] data_dtypes: self-data-key -> dtype. Read from data if not specified.
    """
    assert window == 1  # not implemented
    super(MetaDataset, self).__init__(**kwargs)
    assert self.shuffle_frames_of_nseqs == 0  # not implemented. anyway only for non-recurrent nets

    self.data_map = data_map
    self.dataset_keys = set([m[0] for m in self.data_map.values()]); ":type: set[str]"
    self.data_keys = set(self.data_map.keys()); ":type: set[str]"
    assert "data" in self.data_keys
    self.target_list = sorted(self.data_keys - {"data"})
    self.default_dataset_key = seq_order_control_dataset or self.data_map["data"][0]
    self.seq_order_control_dataset = seq_order_control_dataset

    # This will only initialize datasets needed for features occuring in data_map
    self.datasets = {
      key: init_dataset(datasets[key], extra_kwargs={"name": "%s_%s" % (self.name, key)})
      for key in self.dataset_keys}  # type: typing.Dict[str,Dataset]

    self.seq_list_original = self._load_seq_list(seq_list_file)
    self.num_total_seqs = len(self.seq_list_original[self.default_dataset_key])
    for key in self.dataset_keys:
      assert len(self.seq_list_original[key]) == self.num_total_seqs

    self.tag_idx = {tag: idx for (idx, tag) in enumerate(self.seq_list_original[self.default_dataset_key])}

    self._seq_lens = None  # type: typing.Optional[typing.Dict[str,NumbersDict]]
    self._num_timesteps = None  # type: typing.Optional[NumbersDict]
    if seq_lens_file:
      seq_lens = load_json(filename=seq_lens_file)
      assert isinstance(seq_lens, dict)
      # dict[str,NumbersDict], seq-tag -> data-key -> len
      self._seq_lens = {tag: NumbersDict(l) for (tag, l) in seq_lens.items()}
      self._num_timesteps = sum([self._seq_lens[s] for s in self.seq_list_original[self.default_dataset_key]])

    if data_dims:
      data_dims = convert_data_dims(data_dims)
      self.data_dims = data_dims
      assert "data" in data_dims
      for key in self.target_list:
        assert key in data_dims
    else:
      self.data_dims = {}

    for data_key in self.data_keys:
      dataset_key, dataset_data_key = self.data_map[data_key]
      dataset = self.datasets[dataset_key]
      if not data_dims:
        self.data_dims[data_key] = dataset.num_outputs[dataset_data_key]
      if dataset_data_key in dataset.labels:
        self.labels[data_key] = dataset.labels[dataset_data_key]

    self.num_inputs = self.data_dims["data"][0]
    self.num_outputs = self.data_dims

    self.data_dtypes = {data_key: _select_dtype(data_key, self.data_dims, data_dtypes) for data_key in self.data_keys}
    self.orig_seq_order_is_initialized = False
    self.seq_list_ordered = None  # type: typing.Optional[typing.Dict[str,typing.List[str]]]

  def _is_same_seq_name_for_each_dataset(self):
    """
    This should be fast.

    :rtype: bool
    """
    main_list = self.seq_list_original[self.default_dataset_key]
    for key, other_list in self.seq_list_original.items():
      if main_list is not other_list:
        return False
    return True

  def _load_seq_list(self, seq_list_file=None):
    """
    :param str seq_list_file:
    :return: dict: dataset key -> seq list
    :rtype: dict[str,list[str]]
    """
    if seq_list_file:
      if seq_list_file.endswith(".pkl"):
        import pickle
        seq_list = pickle.load(open(seq_list_file, 'rb'))
      elif seq_list_file.endswith(".gz"):
        import gzip
        seq_list = gzip.open(seq_list_file, "rt").read().splitlines()
      else:
        seq_list = open(seq_list_file).read().splitlines()
    else:
      # We create a sequence list from all the sequences of the default dataset and hope that it also applies to the
      # other datasets. This can only work if all datasets have the same tag format and the sequences in the other
      # datasets are a subset of those in the default dataset.
      default_dataset = self.datasets[self.default_dataset_key]
      assert isinstance(default_dataset, Dataset)
      print("Reading sequence list for MetaDataset %r from sub-dataset %r" % (self.name, default_dataset.name),
            file=log.v3)
      seq_list = default_dataset.get_all_tags()
      # Catch index out of bounds errors. Whether the tags are actually valid will be checked in _check_dataset_seq().
      for key in self.dataset_keys:
        if key != self.default_dataset_key and self.datasets[key].get_total_num_seqs() < len(seq_list):
          print("Dataset %r has less sequences (%i) than in sequence list (%i) read from %r, this cannot work out!" % (
            key, self.datasets[key].get_total_num_seqs(), len(seq_list), self.default_dataset_key), file=log.v1)
          other_tags = self.datasets[key].get_all_tags()
          for tag in seq_list:
            if tag not in other_tags:
              print(
                "Seq tag %r in dataset %r but not in dataset %r." % (tag, self.default_dataset_key, key), file=log.v1)
              break  # only print one
          for tag in other_tags:
            if tag not in seq_list:
              print(
                "Seq tag %r in dataset %r but not in dataset %r." % (tag, key, self.default_dataset_key), file=log.v1)
              break  # only print one
          raise Exception("Dataset %r is missing seqs." % key)

    assert isinstance(seq_list, (list, dict))
    if isinstance(seq_list, list):
      seq_list = {key: seq_list for key in self.dataset_keys}

    return seq_list

  def _get_dataset_seq_length(self, seq_idx):
    if not self.orig_seq_order_is_initialized:
      # To use get_seq_length() we first have to init the sequence order once in original order.
      # If sequence lengths are not needed by get_seq_order_for_epoch this is never executed.
      self.datasets[self.default_dataset_key].init_seq_order(
        epoch=self.epoch, seq_list=self.seq_list_original[self.default_dataset_key])
      self.orig_seq_order_is_initialized = True

    return self.datasets[self.default_dataset_key].get_seq_length(seq_idx)["data"]

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    :param int|None epoch:
    :param list[str]|None seq_list:
    :rtype: bool
    """
    need_reinit = self.epoch is None or self.epoch != epoch or seq_list
    super(MetaDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)

    if not need_reinit:
      self._num_seqs = len(self.seq_list_ordered[self.default_dataset_key])
      return False

    seq_order_dataset = None
    if seq_list:
      seq_index = [self.tag_idx[tag] for tag in seq_list]
    elif self.seq_order_control_dataset:
      seq_order_dataset = self.datasets[self.seq_order_control_dataset]
      assert isinstance(seq_order_dataset, Dataset)
      seq_order_dataset.init_seq_order(epoch=epoch)
      seq_index = seq_order_dataset.get_current_seq_order()
    else:
      if self._seq_lens:
        def get_seq_len(s):
          """
          :param int s:
          :rtype: int
          """
          return self._seq_lens[self.seq_list_original[self.default_dataset_key][s]]["data"]
      else:
        self.orig_seq_order_is_initialized = False
        get_seq_len = self._get_dataset_seq_length
      seq_index = self.get_seq_order_for_epoch(epoch, self.num_total_seqs, get_seq_len)
    self._num_seqs = len(seq_index)
    self.seq_list_ordered = {key: [ls[s] for s in seq_index] for (key, ls) in self.seq_list_original.items()}

    for dataset_key, dataset in self.datasets.items():
      assert isinstance(dataset, Dataset)
      if dataset is seq_order_dataset:
        continue
      dataset.init_seq_order(epoch=epoch, seq_list=self.seq_list_ordered[dataset_key])
    return True

  def _load_seqs(self, start, end):
    for dataset_key in self.dataset_keys:
      self.datasets[dataset_key].load_seqs(start, end)
      for seq_idx in range(start, end):
        self._check_dataset_seq(dataset_key, seq_idx)
    super(MetaDataset, self)._load_seqs(start=start, end=end)

  def _check_dataset_seq(self, dataset_key, seq_idx):
    """
    :param str dataset_key:
    :param int seq_idx:
    """
    dataset_seq_tag = self.datasets[dataset_key].get_tag(seq_idx)
    self_seq_tag = self.seq_list_ordered[dataset_key][seq_idx]
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
    seq_tag = self.seq_list_ordered[self.default_dataset_key][seq_idx]
    features = self._get_data(seq_idx, "data")
    targets = {target: self._get_data(seq_idx, target) for target in self.target_list}
    return DatasetSeq(seq_idx=seq_idx, seq_tag=seq_tag, features=features, targets=targets)

  def get_seq_length(self, sorted_seq_idx):
    """
    :param int sorted_seq_idx:
    :rtype: NumbersDict
    """
    if self._seq_lens:
      return self._seq_lens[self.seq_list_ordered[self.default_dataset_key][sorted_seq_idx]]
    return super(MetaDataset, self).get_seq_length(sorted_seq_idx)

  def get_tag(self, sorted_seq_idx):
    """
    :param int sorted_seq_idx:
    :rtype: str
    """
    return self.seq_list_ordered[self.default_dataset_key][sorted_seq_idx]

  def get_target_list(self):
    """
    :rtype: list[str]
    """
    return self.target_list

  def get_data_shape(self, data_key):
    """
    :param str data_key:
    :rtype: list[int]
    """
    dataset_key, dataset_data_key = self.data_map[data_key]
    return self.datasets[dataset_key].get_data_shape(dataset_data_key)

  def get_data_dtype(self, key):
    """
    :param str key:
    :rtype: str
    """
    dtype = self.data_dtypes[key]
    if self.added_data:
      assert super(MetaDataset, self).get_data_dtype(key) == dtype
    return dtype


class ClusteringDataset(CachedDataset2):
  """
  This is a special case of MetaDataset,
  with one main subdataset, and we add a cluster-idx for each seq.
  We will read the cluster-map (seq-name -> cluster-idx) here directly.
  """

  def __init__(self, dataset, cluster_map_file, n_clusters, single_cluster=False, **kwargs):
    """
    :param dict[str] dataset:
    :param cluster_map_file:
    :param int n_clusters:
    :param single_cluster:
    """
    super(CachedDataset2, self).__init__(**kwargs)
    self.dataset = init_dataset(dataset)
    self.n_clusters = n_clusters
    self.single_cluster = single_cluster
    self.cluster_map = self._load_cluster_map(cluster_map_file)
    self.cluster_idx_dtype = "int32"
    self.num_inputs = self.dataset.num_inputs
    self.num_outputs = self.dataset.num_outputs.copy()
    self.num_outputs["cluster_idx"] = (n_clusters, 1)  # will be a single int32
    self.expected_load_seq_start = 0

  def _load_cluster_map(self, filename):
    ls = open(filename).read().splitlines()
    assert "<coprus-key-map>" in ls[:3], "We expect the Sprint XML format."
    # It has lines like: <map-item key="CHiME3/dt05_bth/M03_22GC010M_BTH.CH5/1" value="0"/>
    import re
    pattern = re.compile('<map-item key="(.*)" value="(.*)"/>')
    cluster_map = {}  # type: typing.Dict[str,int]  # seq-name -> cluster-idx
    for l in ls:
      if not l.startswith("<map-item"):
        continue
      seq_name, cluster_idx_s = pattern.match(l).groups()
      cluster_idx = int(cluster_idx_s)
      assert 0 <= cluster_idx < self.n_clusters
      cluster_map[seq_name] = cluster_idx
    return cluster_map

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    :param int epoch:
    :param list[str]|int seq_list:
    :rtype: bool
    """
    self.dataset.init_seq_order(epoch=epoch, seq_list=seq_list)
    return super(ClusteringDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)

  def get_data_keys(self):
    """
    :rtype: list[str]
    """
    return self.dataset.get_data_keys() + ["cluster_idx"]

  def get_data_dtype(self, key):
    """
    :param str key:
    :rtype: str
    """
    if key == "cluster_idx":
      return self.cluster_idx_dtype
    return self.dataset.get_data_dtype(key)

  @property
  def num_seqs(self):
    """
    :rtype: int
    """
    return self.dataset.num_seqs

  def is_less_than_num_seqs(self, n):
    """
    :param int n:
    :rtype: bool
    """
    return self.dataset.is_less_than_num_seqs(n)

  def _load_seqs(self, start, end):
    """
    :param int start:
    :param int end:
    """
    self.dataset.load_seqs(start, end)
    super(ClusteringDataset, self)._load_seqs(start=start, end=end)

  def get_tag(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: str
    """
    return self.dataset.get_tag(seq_idx)

  def _collect_single_seq(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: DatasetSeq
    """
    seq_name = self.get_tag(seq_idx)
    data = {key: self.dataset.get_data(seq_idx=seq_idx, key=key) for key in self.dataset.get_data_keys()}
    data["cluster_idx"] = numpy.array([self.cluster_map[seq_name]], dtype=self.cluster_idx_dtype)
    return DatasetSeq(seq_idx=seq_idx, features=data["data"], targets=data)

  # noinspection PyMethodOverriding
  def _generate_batches(self, recurrent_net, batch_size, max_seqs=-1, seq_drop=0.0, max_seq_length=None,
                        used_data_keys=None):
    import sys
    if max_seq_length is None:
      max_seq_length = sys.maxsize
    if batch_size == 0:
      batch_size = sys.maxsize
    assert batch_size > 0
    if max_seqs == -1:
      max_seqs = float('inf')
    assert max_seqs > 0
    assert seq_drop <= 1.0
    chunk_size = self.chunk_size
    chunk_step = self.chunk_step
    from EngineBatch import Batch
    batch = Batch()
    last_seq_idx = None
    for seq_idx, t_start, t_end in self.iterate_seqs(
          chunk_size=chunk_size, chunk_step=chunk_step, used_data_keys=used_data_keys):
      if self.single_cluster:
        if last_seq_idx is not None and last_seq_idx != seq_idx:
          last_seq_name = self.get_tag(last_seq_idx)
          seq_name = self.get_tag(seq_idx)
          if self.cluster_map[last_seq_name] != self.cluster_map[seq_name]:
            print("ClusteringDataset::_generate_batches", last_seq_idx, "is not", seq_idx, file=log.v5)
            yield batch
            batch = Batch()
      length = t_end - t_start
      if max_seq_length < 0 and length['classes'] > -max_seq_length:
        continue
      elif 0 < max_seq_length < length.max_value():
        continue
      if length.max_value() > batch_size:
        print("warning: sequence length (%i) larger than limit (%i)" % (length.max_value(), batch_size), file=log.v4)
      if self.rnd_seq_drop.random() < seq_drop:
        continue
      dt, ds = batch.try_sequence_as_slice(length)
      if ds > 1 and ((dt * ds).max_value() > batch_size or ds > max_seqs):
        yield batch
        batch = Batch()
      print("batch add slice length", length, file=log.v5)
      batch.add_sequence_as_slice(seq_idx=seq_idx, seq_start_frame=t_start, length=length)
      last_seq_idx = seq_idx

    if batch.get_all_slices_num_frames().max_value() > 0:
      yield batch


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
    self.dataset_seq_idx_offsets = None  # type: typing.Optional[typing.List[int]]

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
    """
    :param int seq_idx:
    :rtype: int
    """
    i = 0
    while i < len(self.dataset_seq_idx_offsets):
      if seq_idx + self.dataset_seq_idx_offsets[i] < 0:
        return i - 1
      i += 1
    return i - 1

  def _load_seqs(self, start, end):
    """
    :param int start:
    :param int end:
    """
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
    """
    :param int seq_idx:
    :rtype: DatasetSeq
    """
    dataset_idx = self._get_dataset_for_seq_idx(seq_idx)
    dataset = self.datasets[dataset_idx]
    dataset_seq_idx = seq_idx + self.dataset_seq_idx_offsets[dataset_idx]
    seq_tag = dataset.get_tag(dataset_seq_idx)
    features = dataset.get_input_data(dataset_seq_idx)
    targets = {k: dataset.get_targets(k, dataset_seq_idx) for k in dataset.get_target_list()}
    return DatasetSeq(seq_idx=seq_idx, seq_tag=seq_tag, features=features, targets=targets)

  @property
  def num_seqs(self):
    """
    :rtype: int
    """
    return sum([ds.num_seqs for ds in self.datasets])

  def get_target_list(self):
    """
    :rtype: list[str]
    """
    return self.datasets[0].get_target_list()


class CombinedDataset(CachedDataset2):
  """
  The CombinedDataset is to be used in the cases of **Multi-Task Learning** and **Combination of Corpora**.
  Here, in general, the datasets describe **different training sequences**.
  For each sequence, only the features of the corresponding dataset will be available.
  Features of the other datasets are set to empty arrays.
  The input parameter ``"datasets"`` is the same as for the MetaDataset.
  The ``"data_map"`` is reversed to allow for several datasets mapping to the same feature.
  The ``"default"`` ``"seq_ordering"`` is to first go through all sequences of the first dataset,
  then the second and so on.
  All other sequence orderings (``"random"``, ``"sorted"``, ``"laplace"``, ...) are supported
  and based on this "default" ordering.
  There is a special sequence ordering ``"random_dataset"``, where we pick datasets at random,
  while keeping the sequence order within the datasets as is.
  To adjust the ratio of number of training examples from the different datasets in an epoch,
  one can use ``"repeat_epoch"`` in some of the datasets to
  increase their size relative to the others.
  Also, ``"partition_epoch"`` in some of the datasets can be used to shrink them relative to the others.

  **Example of CombinedDataset config:**

  .. code::

      train = {"class": "CombinedDataset",
               "datasets": {"sprint": train_sprint, "translation": train_translation},
               "data_map": {("sprint", "data"): "data",
                            ("sprint", "orth_classes"): "orth_classes",
                            ("translation", "data"): "source_text",
                            ("translation", "classes"): "orth_classes"},
               "seq_ordering": "default",
               "partition_epoch": 2,
       }

  This combines a SprintDataset and a TranslationDataset.
  These are defined as ``"train_sprint"`` and ``"train_translation"`` separately.
  *Note that the current implementation expects one input feature to be called "data".*

  Note: The mapping has been inverted. We now expect (dataset-key, dataset-data-key) -> self-data-key
  am-dataset:data -> am-data, am-dataset:classes -> am-classes, lm-dataset:data -> lm-data.
  For each sequence idx, it will select one of the given datasets, fill in the data-keys of this dataset
  and will return empty sequences for the remaining datasets.
  The default sequence ordering is to first go through all sequences of dataset 1, then dataset 2 and so on. If
  seq_ordering is set to 'random_dataset', we always pick one of the datasets at random (equally distributed over the
  sum of num-seqs), but still go through the sequences of a particular dataset in the order defined for it in the config
  (in order if not defined). For 'sorted' or 'laplace' the sequence length as provided by the datasets is used to sort
  all sequences jointly. Note, that this overrides the sequence order of the sub-datasets (also the case for 'random').
  'partition_epoch' of the CombinedDataset is applied to the joint sequence order for all sequences.
  'partition_epoch' of the sub-datasets is still applied. This can be used to adjust the relative size of
  the datasets. (However, do not combine 'partition_epoch' on both levels, as this leads to an unexpected selection
  of sequences.) To upscale a dataset, rather than downscaling the others via 'partition_epoch', use the
  'repeat_epoch' option.

  Also see :class:`MetaDataset`.
  """

  def __init__(self,
               datasets,
               data_map,
               data_dims=None,
               data_dtypes=None,
               window=1, **kwargs):
    """
    :param dict[str,dict[str]] datasets: dataset-key -> dataset-kwargs. including keyword 'class' and maybe 'files'
    :param dict[(str,str),str] data_map: (dataset-key, dataset-data-key) -> self-data-key.
      Should contain 'data' as key. Also defines the target-list, which is all except 'data'.
    :param dict[str,(int,int)] data_dims: self-data-key -> data-dimension, len(shape) (1 ==> sparse repr).
       Deprecated/Only to double check. Read from data if not specified.
    :param dict[str,str] data_dtypes: self-data-key -> dtype. Read from data if not specified.
    """
    assert window == 1  # not implemented
    super(CombinedDataset, self).__init__(**kwargs)
    assert self.shuffle_frames_of_nseqs == 0  # not implemented. anyway only for non-recurrent nets

    self.rnd = Random(self.epoch)
    self.dataset_keys = set([m[0] for m in data_map.keys()]); ":type: set[str]"
    self.dataset_idx2key_map = dict(enumerate(sorted(self.dataset_keys)))  # idx -> dataset-key
    self.data_keys = set(data_map.values()); ":type: set[str]"
    assert "data" in self.data_keys
    self.target_list = sorted(self.data_keys - {"data"})

    # Build target lookup table that maps from dataset_key and data_key (data key used by CombinedDataset)
    # to dataset_data_key (data_key used by the sub-dataset). This is needed in get_data() to access data
    # by data_key. Maps to None if data_key does not correspond to a feature in datasets[dataset_key].
    target_lookup_table = {}
    for dataset_key in self.dataset_keys:
      target_lookup_table[dataset_key] = {
        data_key: dataset_key_tuple[1]
        for dataset_key_tuple, data_key in data_map.items()
        if dataset_key_tuple[0] == dataset_key}
      for key in self.data_keys:
        target_lookup_table[dataset_key].setdefault(key, None)
    self.target_lookup_table = target_lookup_table

    # This will only initialize datasets needed for features occurring in data_map
    self.datasets = {key: init_dataset(datasets[key]) for key in self.dataset_keys}

    # noinspection PyBroadException
    try:
      self._num_seqs = sum([self.datasets[k].num_seqs for k in sorted(self.datasets.keys())])
      self.know_num_seqs_beforehand = True
    except Exception:
      self._estimated_num_seqs = sum([self.datasets[k].estimated_num_seqs for k in sorted(self.datasets.keys())])
      self.estimated_num_seq_per_subset = [self.datasets[k].estimated_num_seqs for k in sorted(self.datasets.keys())]
      self.know_num_seqs_beforehand = False

    if data_dims:
      data_dims = convert_data_dims(data_dims)
      self.data_dims = data_dims
      assert "data" in data_dims
      for key in self.target_list:
        assert key in data_dims
    else:
      self.data_dims = {}

    for dataset_key_tuple, data_key in data_map.items():
      dataset_key, dataset_data_key = dataset_key_tuple
      dataset = self.datasets[dataset_key]
      if not data_dims:
        self.data_dims[data_key] = dataset.num_outputs[dataset_data_key]
      if dataset_data_key in dataset.labels:
        self.labels[data_key] = dataset.labels[dataset_data_key]

    self.num_inputs = self.data_dims["data"][0]
    self.num_outputs = self.data_dims

    self.data_dtypes = {data_key: _select_dtype(data_key, self.data_dims, data_dtypes) for data_key in self.data_keys}

    self.dataset_seq_idx_list = None  # type: typing.Optional[typing.List[typing.Tuple[int,int]]]
    self.seq_order = None  # type: typing.Optional[typing.List[int]]
    self.dataset_sorted_seq_idx_list = None  # type: typing.Optional[typing.List[typing.Tuple[int,int]]]
    self.used_num_seqs_per_subset = None  # type: typing.Optional[typing.List[int]]

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    :param int epoch:
    :param list[str]|None seq_list:
    :rtype: bool
    """
    assert seq_list is None, "seq_list not supported for %s" % self.__class__
    need_reinit = self.epoch is None or self.epoch != epoch
    super(CombinedDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)

    if not need_reinit:
      if self.know_num_seqs_beforehand:
        self._num_seqs = len(self.seq_order)
      return False

    # First init sequence order for sub-datasets as usual to get a list of available sequences. This way sorting and
    # partition epoch of the individual sub-datasets is still supported. Later we will call init_seq_order again with a
    # sequence list to e.g. apply joint sorting or partition epoch of all sequences.
    for dataset in self.datasets.values():
      dataset.init_seq_order(epoch=epoch)

    if self.know_num_seqs_beforehand:
      self.dataset_seq_idx_list = self._create_dataset_seq_idx_list()

      if self.seq_ordering == "random_dataset":
        self.seq_order = self._get_random_dataset_seq_order()
      else:
        self.seq_order = self.get_seq_order_for_epoch(
            epoch=epoch, num_seqs=len(self.dataset_seq_idx_list), get_seq_len=self._get_seq_length)
      self._num_seqs = len(self.seq_order)

      # We only want to load those sequences in the sub-datasets that appear in self.seq_order. For this, we extract
      # sequence lists containing the subset of sequences for each dataset from self.seq_order.
      seq_lists = [[] for _ in self.datasets]
      for seq_idx in self.seq_order:
        dataset_idx, dataset_seq_idx = self.dataset_seq_idx_list[seq_idx]
        dataset = self.datasets[self.dataset_idx2key_map[dataset_idx]]
        seq_tag = dataset.get_tag(dataset_seq_idx)
        seq_lists[dataset_idx].append(seq_tag)

      # Re-initialize sequence orders of sub-datasets with created sequence list.
      for dataset_idx, dataset_key in self.dataset_idx2key_map.items():
        self.datasets[dataset_key].init_seq_order(epoch=epoch, seq_list=seq_lists[dataset_idx])

      # Apply seq_order to self.dataset_seq_idx.
      # We have to re-calculate the seq_idx's because we sorted the datasets in the previous step.
      self.dataset_sorted_seq_idx_list = []
      counters = [0] * len(self.datasets)
      for seq_idx in self.seq_order:
        dataset_idx, _ = self.dataset_seq_idx_list[seq_idx]
        dataset_seq_idx = counters[dataset_idx]
        counters[dataset_idx] += 1
        self.dataset_sorted_seq_idx_list.append((dataset_idx, dataset_seq_idx))

    else:
      self.dataset_sorted_seq_idx_list = []  # We will fill this as we go
      self.used_num_seqs_per_subset = [0] * len(self.datasets)

    return True

  def _create_dataset_seq_idx_list(self):
    """
    Creates a list of all available sequences, to which we can later apply the sequence ordering.
    It contains the index of the dataset and the sequence index within the dataset for every sequence.
    The sequences appear sorted by dataset first.

    :returns: list of (dataset-idx, dataset-seq-idx)
    :rtype: list[(int,int)]
    """
    dataset_seq_idxs = []
    for dataset_idx in range(len(self.datasets)):
      dataset = self.datasets[self.dataset_idx2key_map[dataset_idx]]
      for dataset_seq_idx in range(dataset.num_seqs):
        dataset_seq_idxs.append((dataset_idx, dataset_seq_idx))

    return dataset_seq_idxs

  def _get_random_dataset_seq_order(self):
    """
    Choose datasets randomly but preserve order within each dataset. This sorting method is unique to CombinedDataset.
    """
    # Create a list containing each dataset_idx dataset.num_seqs-times and shuffle it.
    dataset_ids = [idx_tuple[0] for idx_tuple in self.dataset_seq_idx_list]
    self.rnd.shuffle(dataset_ids)

    # Calculate the offset for all datasets (i.e. the index where sequences of a given dataset start
    # in self.dataset_seq_idx_list).
    dataset_offsets = [0]
    for dataset_idx in range(len(self.datasets)):
      dataset = self.datasets[self.dataset_idx2key_map[dataset_idx]]
      dataset_offsets.append(dataset_offsets[-1] + dataset.num_seqs)

    # Create the actual seq_order list.
    # We want to keep the order within the sub-datasets, thus we assign seq_ids by simply counting up for each dataset.
    # We however have to account for the different offsets needed when accessing self.dataset_seq_idx_list later.
    seq_order = []
    counters = [0] * len(self.datasets)
    for dataset_idx in dataset_ids:
      seq_order.append(counters[dataset_idx] + dataset_offsets[dataset_idx])
      counters[dataset_idx] += 1

    total_num_seqs = len(self.dataset_seq_idx_list)
    assert sum(counters) == total_num_seqs

    if self.partition_epoch:
      seq_order = self._apply_partition_epoch(seq_order, self.partition_epoch, self.epoch)
    if self.repeat_epoch:
      seq_order = seq_order * self.repeat_epoch

    return seq_order

  def _get_seq_length(self, seq_idx):
    dataset_idx, dataset_seq_idx = self.dataset_seq_idx_list[seq_idx]
    dataset = self.datasets[self.dataset_idx2key_map[dataset_idx]]

    return dataset.get_seq_length(dataset_seq_idx)["data"]

  def _expand_dataset_sec_idxs(self, num_values):
    """
    :param num_values: int Add num_values entries to the dataset-segment-idx mapping table
    :return:
    """
    for i in range(num_values):
      if self.seq_ordering == "default":  # i.e. in order
        dataset_idx = 0
        while dataset_idx < len(self.datasets):
          if self.datasets[self.dataset_idx2key_map[dataset_idx]].is_less_than_num_seqs(
                self.used_num_seqs_per_subset[dataset_idx]):
            break
          dataset_idx += 1
        else:
          return False  # No dataset has remaining data

      elif self.seq_ordering == "reversed":
        dataset_idx = len(self.datasets) - 1
        while dataset_idx >= 0:
          if self.datasets[self.dataset_idx2key_map[dataset_idx]].is_less_than_num_seqs(
                self.used_num_seqs_per_subset[dataset_idx]):
            break
          dataset_idx -= 1
        else:
          return False  # No dataset has remaining data

      elif self.seq_ordering == "random_dataset":
        while True:
          # Build probability table
          expected_remaining_seqs = [
            estimated - used
            for estimated, used in zip(self.estimated_num_seq_per_subset, self.used_num_seqs_per_subset)]
          total_remaining = float(sum(expected_remaining_seqs))

          if total_remaining < 0.1:  # We expect no more data, but try anyway
            nonempty_datasets = []
            for j, k in enumerate(sorted(self.datasets.keys())):
              if self.datasets[k].is_less_than_num_seqs(self.used_num_seqs_per_subset[j]):
                nonempty_datasets.append(j)
            if not nonempty_datasets:
              return False  # No more data to add
            dataset_idx = numpy.random.choice(nonempty_datasets)
            self.estimated_num_seq_per_subset[dataset_idx] += 1
            break

          else:  # We sample from all sets which should contain more data
            prob_table = [remaining / total_remaining for remaining in expected_remaining_seqs]
            dataset_idx = numpy.random.choice(len(self.datasets), p=prob_table)
            if self.datasets[self.dataset_idx2key_map[dataset_idx]].is_less_than_num_seqs(
                  self.used_num_seqs_per_subset[dataset_idx]):
              break  # Found good Data
            else:
              self.estimated_num_seq_per_subset[dataset_idx] = self.used_num_seqs_per_subset[dataset_idx]

      else:
        raise Exception("The sorting method '{}' is not implemented for the case that number of sequences"
                        "is not known in advance.".format(self.seq_ordering))

      # We now have a valid dataset index to take the next segment from
      self.dataset_sorted_seq_idx_list.append((dataset_idx, self.used_num_seqs_per_subset[dataset_idx]))
      self.used_num_seqs_per_subset[dataset_idx] += 1
    return True

  def _load_seqs(self, start, end):
    # If the segment order is not yet known, fix the next few segments
    if not self.know_num_seqs_beforehand and end > len(self.dataset_sorted_seq_idx_list):
      self._expand_dataset_sec_idxs(end - len(self.dataset_sorted_seq_idx_list))

    requested_seqs = self.dataset_sorted_seq_idx_list[start:end]

    for dataset_idx in range(len(self.datasets)):
      dataset = self.datasets[self.dataset_idx2key_map[dataset_idx]]
      sub_requested_seqs = [s[1] for s in requested_seqs if s[0] == dataset_idx]
      if not sub_requested_seqs:
        continue
      sub_start, sub_end = min(sub_requested_seqs), max(sub_requested_seqs)
      dataset.load_seqs(sub_start, sub_end + 1)
    super(CombinedDataset, self)._load_seqs(start=start, end=end)

  def _get_data(self, dataset_key, dataset_seq_idx, data_key):
    """
    :type dataset_seq_idx: int
    :type dataset_key: str
    :type data_key: str
    :rtype: numpy.ndarray
    """
    dataset_data_key = self.target_lookup_table[dataset_key][data_key]
    dataset = self.datasets[dataset_key]; ":type: Dataset"
    if dataset_data_key is not None:
      return dataset.get_data(dataset_seq_idx, dataset_data_key)
    else:
      return numpy.array([], self.data_dtypes[data_key])

  def _collect_single_seq(self, seq_idx):
    """
    :type seq_idx: int
    :rtype: DatasetSeq
    """
    if not self.is_less_than_num_seqs(seq_idx):
      return None
    dataset_idx, dataset_seq_idx = self.dataset_sorted_seq_idx_list[seq_idx]
    dataset_key = self.dataset_idx2key_map[dataset_idx]
    dataset = self.datasets[dataset_key]

    seq_tag = dataset.get_tag(dataset_seq_idx)
    features = self._get_data(dataset_key, dataset_seq_idx, "data")
    targets = {target: self._get_data(dataset_key, dataset_seq_idx, target) for target in self.target_list}
    return DatasetSeq(seq_idx=seq_idx, seq_tag=seq_tag, features=features, targets=targets)

  def is_less_than_num_seqs(self, n):
    """
    :param int n:
    :rtype: bool
    """
    if self.know_num_seqs_beforehand:
      return n < self._num_seqs
    else:
      if n < len(self.dataset_sorted_seq_idx_list):
        return True
      else:
        return self._expand_dataset_sec_idxs(n - len(self.dataset_sorted_seq_idx_list) + 1)

  def get_target_list(self):
    """
    :rtype: list[str]
    """
    return self.target_list

  def get_data_dtype(self, key):
    """
    :param str key:
    :rtype: str
    """
    dtype = self.data_dtypes[key]
    if self.added_data:
      assert super(CombinedDataset, self).get_data_dtype(key) == dtype
    return dtype

  def get_data_dim(self, key):
    """
    :param str key:
    :rtype: int
    """
    assert key in self.data_dims
    return self.data_dims[key][0]


class ConcatSeqsDataset(CachedDataset2):
  """
  This takes another dataset, and concatenates one or multiple seqs.
  """
  def __init__(self, dataset, seq_list_file, seq_len_file, seq_tag_delim=";", remove_in_between_postfix=None,
               use_cache_manager=False, epoch_wise_filter=None, **kwargs):
    """
    :param dict[str] dataset: kwargs for init_dataset
    :param str seq_list_file: filename. line-separated. seq_tag_delim.join(seq_tags) for concatenated seqs
    :param str seq_len_file: file with Python dict, (single) seg_name -> len, which is used for sorting
    :param str seq_tag_delim:
    :param bool use_cache_manager:
    :param dict[(int,int),dict] epoch_wise_filter: see :class:`EpochWiseFilter`
    :param dict[str,int]|None remove_in_between_postfix: data_key -> expected postfix label. e.g. {"targets": 0}
    """
    super(ConcatSeqsDataset, self).__init__(**kwargs)
    self.seq_tag_delim = seq_tag_delim
    self.remove_in_between_postfix = remove_in_between_postfix or {}
    self.epoch_wise_filter = EpochWiseFilter(epoch_wise_filter) if epoch_wise_filter else None
    dataset = dataset.copy()
    dataset.setdefault("name", "%s_subdataset" % self.name)
    self.sub_dataset = init_dataset(dataset)
    self.num_outputs = self.sub_dataset.num_outputs
    self.num_inputs = self.sub_dataset.num_inputs
    self.labels = self.sub_dataset.labels
    if use_cache_manager:
      import Util
      seq_list_file = Util.cf(seq_list_file)
      seq_len_file = Util.cf(seq_len_file)
    self.full_seq_list = open(seq_list_file).read().splitlines()
    self.seq_lens = eval(open(seq_len_file).read())
    assert isinstance(self.seq_lens, dict)
    self.full_seq_len_list = self._get_full_seq_lens_list()
    self.cur_seq_list = None  # type: typing.Optional[typing.List[str]]  # list of seq tags
    self.cur_sub_seq_idxs = None  # type: typing.Optional[typing.List[typing.List[int]]]  # list of list of sub seq idxs

  def _get_full_seq_lens_list(self):
    """
    :return: list where idx is same as in self.full_seq_list, maps to len (via self.seq_lens)
    :rtype: list[int]
    """
    ls = []
    for seq_tag in self.full_seq_list:
      sub_seq_tags = seq_tag.split(self.seq_tag_delim)
      ls.append(sum([self.seq_lens[sub_seq_tag] for sub_seq_tag in sub_seq_tags]))
    assert len(ls) == len(self.full_seq_list)
    return ls

  def init_seq_order(self, epoch=None, seq_list=None):
    """
    :param int epoch:
    :param list[str]|None seq_list:
    :rtype: bool
    """
    super(ConcatSeqsDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list)
    assert not seq_list
    if not seq_list:
      def get_seq_len(i):
        """
        :param int i:
        :rtype: int
        """
        return self.full_seq_len_list[i]
      seq_order = self.get_seq_order_for_epoch(
        epoch=epoch, num_seqs=len(self.full_seq_list), get_seq_len=get_seq_len)
      if self.epoch_wise_filter:
        self.epoch_wise_filter.debug_msg_prefix = str(self)
        seq_order = self.epoch_wise_filter.filter(epoch=epoch, seq_order=seq_order, get_seq_len=get_seq_len)
      seq_list = [self.full_seq_list[i] for i in seq_order]  # tag list
    self.cur_seq_list = seq_list
    self._num_seqs = len(seq_list)
    sub_seq_list = []
    sub_seq_idxs = []  # type: typing.List[typing.List[int]]  # list of list of seqs
    sub_seq_idx = 0
    for seq_tag in seq_list:
      sub_seq_tags = seq_tag.split(self.seq_tag_delim)
      sub_seq_idxs.append(list(range(sub_seq_idx, sub_seq_idx + len(sub_seq_tags))))
      sub_seq_idx = sub_seq_idxs[-1][-1] + 1
      sub_seq_list.extend(sub_seq_tags)
    assert sub_seq_idx == len(sub_seq_list) and len(seq_list) == len(sub_seq_idxs)
    self.cur_sub_seq_idxs = sub_seq_idxs
    return self.sub_dataset.init_seq_order(seq_list=sub_seq_list)

  def _collect_single_seq(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: DatasetSeq | None
    :returns DatasetSeq or None if seq_idx >= num_seqs.
    """
    assert self.cur_seq_list is not None, "call init_seq_order"
    if seq_idx >= len(self.cur_seq_list):
      return None
    seq_tag = self.cur_seq_list[seq_idx]
    sub_seq_tags = seq_tag.split(self.seq_tag_delim)
    sub_seq_idxs = self.cur_sub_seq_idxs[seq_idx]
    assert len(sub_seq_tags) == len(sub_seq_idxs)
    features = {key: [] for key in self.get_data_keys()}
    if seq_idx == 0:  # some extra check, but enough to do for first seq only
      sub_dataset_keys = self.sub_dataset.get_data_keys()
      for key in self.remove_in_between_postfix:
        assert key in sub_dataset_keys, "%s: remove_in_between_postfix key %r not in sub dataset data-keys %r" % (
          self, key, sub_dataset_keys)
    for sub_seq_idx, sub_seq_tag in zip(sub_seq_idxs, sub_seq_tags):
      self.sub_dataset.load_seqs(sub_seq_idx, sub_seq_idx + 1)
      sub_dataset_tag = self.sub_dataset.get_tag(sub_seq_idx)
      assert sub_dataset_tag == sub_seq_tag, "%s: expected tag %r for sub seq idx %i but got %r, part of seq %i %r" % (
        self, sub_seq_tag, sub_seq_idx, sub_dataset_tag, seq_idx, seq_tag)
      for key in self.get_data_keys():
        data = self.sub_dataset.get_data(sub_seq_idx, key)
        if key in self.remove_in_between_postfix and sub_seq_idx != sub_seq_idxs[-1]:
          assert data.ndim == 1 and data[-1] == self.remove_in_between_postfix[key]
          data = data[:-1]
        features[key].append(data)
    features = {key: numpy.concatenate(values, axis=0) for (key, values) in features.items()}
    return DatasetSeq(seq_idx=seq_idx, seq_tag=seq_tag, features=features)

  def get_data_keys(self):
    """
    :rtype: list[str]
    """
    return self.sub_dataset.get_data_keys()

  def get_target_list(self):
    """
    :rtype: list[str]
    """
    return self.sub_dataset.get_target_list()

  def get_data_dtype(self, key):
    """
    :param str key:
    :rtype: str
    """
    return self.sub_dataset.get_data_dtype(key)

  def get_data_dim(self, key):
    """
    :param str key:
    :rtype: int
    """
    return self.sub_dataset.get_data_dim(key)

  def is_data_sparse(self, key):
    """
    :param str key:
    :rtype: bool
    """
    return self.sub_dataset.is_data_sparse(key)

  def get_data_shape(self, key):
    """
    :param str key:
    :rtype: list[int]
    """
    return self.sub_dataset.get_data_shape(key)


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
    self.dataset_last_load_seq_end = None
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
    self.dataset_last_load_seq_end = 0
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
    See EngineUtil.assign_dev_data() for comparison.
    :returns whether we added some more
    """
    if not self.batch_gen.has_more():
      return False
    batches = self.batch_gen.peek_next_n(1)
    for batch in batches:
      assert batch.seqs
      if batch.end_seq > self.dataset_last_load_seq_end:
        self.dataset.load_seqs(batch.start_seq, batch.end_seq)
        self.dataset_last_load_seq_end = batch.end_seq

      used_data_keys = self.get_data_keys()
      for seq in batch.seqs:
        res_data = {}
        for k in used_data_keys:
          data = self.dataset.get_data(seq.seq_idx, k)
          if data is not None:
            res_data[k] = data[seq.seq_start_frame[k]:seq.seq_end_frame[k]]
        original_tag = self.dataset.get_tag(seq.seq_idx)
        self._add_data(data=res_data, original_tag=original_tag)

    self.batch_gen.advance(len(batches))
    return True

  def _add_more_until(self, end, shuffle=False):
    if self.added_data and end <= self.added_data[-1].seq_idx:
      return True
    while self._add_more():
      assert self.added_data
      if end <= self.added_data[-1].seq_idx:
        if shuffle:
          self._shuffle()
        return True
    # We have reached the end.
    if not self.added_data:
      self._num_seqs = 0
      print("warning: empty dataset", file=log.v3)
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
    if self._num_seqs is not None:
      return seq_idx < self._num_seqs
    if seq_idx < self.expected_load_seq_start:
      return True
    if self.added_data and seq_idx <= self.added_data[-1].seq_idx:
      return True
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
    """
    :rtype: list[str]
    """
    return self.dataset.get_target_list()


def _simple_to_bool(v):
  if v == 0:
    v = False
  if v == 1:
    v = True
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
