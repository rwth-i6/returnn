"""
Datasets dealing with audio
"""

import numpy
import typing

from .basic import DatasetSeq
from .cached2 import CachedDataset2
from .util.feature_extraction import ExtractAudioFeatures
from .util.vocabulary import Vocabulary
from returnn.util.basic import PY3


class OggZipDataset(CachedDataset2):
  """
  Generic dataset which reads a Zip file containing Ogg files for each sequence and a text document.
  The feature extraction settings are determined by the ``audio`` option,
  which is passed to :class:`ExtractAudioFeatures`.
  Does also support Wav files, and might even support other file formats readable by the 'soundfile'
  library (not tested). By setting ``audio`` or ``targets`` to ``None``, the dataset can be used in
  text only or audio only mode. The content of the zip file is:

    - a .txt file with the same name as the zipfile, containing a python list of dictionaries
    - a subfolder with the same name as the zipfile, containing the audio files

  The dictionaries in the .txt file must be a list of dicts, i.e. have the following structure:

  .. code::

    [{'text': 'some utterance text', 'duration': 2.3, 'file': 'sequence0.wav'},
     ...]

  The dict can optionally also have the entry ``'seq_name': 'arbitrary_sequence_name'``.
  If ``seq_name`` is not included, the seq_tag will be the name of the file.
  ``duration`` is mandatory, as this information is needed for the sequence sorting,
  however, it does not have to match the real duration in any way.
  """

  def __init__(self, path, audio, targets,
               targets_post_process=None,
               use_cache_manager=False, segment_file=None,
               zip_audio_files_have_name_as_prefix=True,
               fixed_random_seed=None, fixed_random_subset=None,
               epoch_wise_filter=None,
               **kwargs):
    """
    :param str|list[str] path: filename to zip
    :param dict[str]|None audio: options for :class:`ExtractAudioFeatures`. use {} for default. None means to disable.
    :param dict[str]|None targets: options for :func:`Vocabulary.create_vocab` (e.g. :class:`BytePairEncoding`)
    :param str|list[str]|((str)->str)|None targets_post_process: :func:`get_post_processor_function`, applied on orth
    :param bool use_cache_manager: uses :func:`Util.cf`
    :param str|None segment_file: .txt or .gz text file containing sequence tags that will be used as whitelist
    :param bool zip_audio_files_have_name_as_prefix:
    :param int|None fixed_random_seed: for the shuffling, e.g. for seq_ordering='random'. otherwise epoch will be used
    :param float|int|None fixed_random_subset:
      Value in [0,1] to specify the fraction, or integer >=1 which specifies number of seqs.
      If given, will use this random subset. This will be applied initially at loading time,
      i.e. not dependent on the epoch. It will use an internally hardcoded fixed random seed, i.e. it's deterministic.
    :param dict|None epoch_wise_filter: see init_seq_order
    """
    import os
    import zipfile
    import returnn.util.basic
    from .meta import EpochWiseFilter
    self._separate_txt_files = {}  # name -> filename
    if (
      isinstance(path, str)
      and os.path.splitext(path)[1] != ".zip"
      and os.path.isdir(path)
      and os.path.isfile(path + ".txt")):
      # Special case (mostly for debugging) to directly access the filesystem, not via zip-file.
      self.paths = [os.path.dirname(path)]
      self._names = [os.path.basename(path)]
      self._zip_files = None
      assert not use_cache_manager, "cache manager only for zip file"
    else:
      if not isinstance(path, (tuple, list)):
        path = [path]
      self.paths = []
      self._names = []
      for path_ in path:
        assert isinstance(path_, str)
        name, ext = os.path.splitext(os.path.basename(path_))
        if "." in name and ext == ".gz":
          name, ext = name[:name.rindex(".")], name[name.rindex("."):] + ext
        if use_cache_manager:
          path_ = returnn.util.basic.cf(path_)
        if ext == ".txt.gz":
          self._separate_txt_files[name] = path_
          continue
        assert ext == ".zip"
        self.paths.append(path_)
        self._names.append(name)
      self._zip_files = [zipfile.ZipFile(path) for path in self.paths]
    self.segments = None  # type: typing.Optional[typing.Set[str]]
    if segment_file:
      self._read_segment_list(segment_file)
    self.zip_audio_files_have_name_as_prefix = zip_audio_files_have_name_as_prefix
    kwargs.setdefault("name", self._names[0])
    super(OggZipDataset, self).__init__(**kwargs)
    self.targets = Vocabulary.create_vocab(**targets) if targets is not None else None
    if self.targets:
      self.labels["classes"] = self.targets.labels
    self.targets_post_process = None  # type: typing.Optional[typing.Callable[[str],str]]
    if targets_post_process:
      if callable(targets_post_process):
        self.targets_post_process = targets_post_process
      else:
        from .lm import get_post_processor_function
        self.targets_post_process = get_post_processor_function(targets_post_process)
    self._fixed_random_seed = fixed_random_seed
    self._audio_random = numpy.random.RandomState(1)
    self.feature_extractor = (
      ExtractAudioFeatures(random_state=self._audio_random, **audio) if audio is not None else None)
    self.num_inputs = self.feature_extractor.get_feature_dimension() if self.feature_extractor else 0
    self.num_outputs = {
      "raw": {"dtype": "string", "shape": ()},
      "orth": [256, 1]}
    # Note: "orth" is actually the raw bytes of the utf8 string,
    # so it does not make quite sense to associate a single str to each byte.
    # However, some other code might expect that the labels are all strings, not bytes,
    # and the API requires the labels to be strings.
    # The code in Dataset.serialize_data tries to decode this case as utf8 (if possible).
    self.labels["orth"] = [chr(i) for i in range(255)]
    if self.targets:
      self.num_outputs["classes"] = [self.targets.num_labels, 1]
    if self.feature_extractor:
      self.num_outputs["data"] = [self.num_inputs, 2]
    else:
      self.num_outputs["data"] = [0, 2]
    self._data = self._collect_data()
    if fixed_random_subset:
      self._filter_fixed_random_subset(fixed_random_subset)
    self.epoch_wise_filter = EpochWiseFilter(epoch_wise_filter) if epoch_wise_filter else None
    self._seq_order = None  # type: typing.Optional[typing.Sequence[int]]
    self.init_seq_order()

  def _read(self, filename, zip_index):
    """
    :param str filename: in zip-file
    :param int zip_index: index of the zip file to load, unused when loading without zip
    :rtype: bytes
    """
    import os
    if filename.endswith(".txt"):
      name, _ = os.path.splitext(filename)
      assert name == self._names[zip_index]
      if name in self._separate_txt_files:
        import gzip
        return gzip.open(self._separate_txt_files[name], "rb").read()
    if self._zip_files is not None:
      return self._zip_files[zip_index].read(filename)
    return open("%s/%s" % (self.paths[0], filename), "rb").read()

  def _collect_data_part(self, zip_index):
    """
    collect all the entries of a single zip-file or txt file
    :param int zip_index: index of the zip-file in self._zip_files, unused when loading without zip
    :return: data entries
    :rtype: list[dict[str]]
    """
    from returnn.util.literal_py_to_pickle import literal_eval
    data = literal_eval(self._read("%s.txt" % self._names[zip_index], zip_index))  # type: typing.List[typing.Dict[str]]
    assert data and isinstance(data, list)
    first_entry = data[0]
    assert isinstance(first_entry, dict)
    assert isinstance(first_entry["text"], str)
    assert isinstance(first_entry["duration"], float)
    # when 'audio' is None and sequence names are given, this dataset can be used in text-only mode
    if "file" in first_entry:
      assert isinstance(first_entry["file"], str)
    else:
      assert not self.feature_extractor, "%s: feature extraction is enabled, but no audio files are specified" % self
      assert isinstance(first_entry["seq_name"], str)
    # add index to data list
    for entry in data:
      entry['_zip_file_index'] = zip_index
    if self.segments:
      data[:] = [entry for entry in data if self._get_tag_from_info_dict(entry) in self.segments]
    return data

  def _collect_data(self):
    """
    :return: entries
    :rtype: list[dict[str]]
    """
    data = []
    if self._zip_files:
      for zip_index in range(len(self._zip_files)):
        zip_data = self._collect_data_part(zip_index)
        data += zip_data
    else:
      # collect data from a txt file
      data = self._collect_data_part(0)
    return data

  def _read_segment_list(self, segment_file):
    """
    read a list of segment names in either plain text or gzip

    :param str segment_file:
    """
    if segment_file.endswith(".gz"):
      import gzip
      segment_file_handle = gzip.open(segment_file)
      self.segments = set([s.decode() for s in segment_file_handle.read().splitlines()])
    else:
      segment_file_handle = open(segment_file)
      self.segments = set(segment_file_handle.read().splitlines())

  def _filter_fixed_random_subset(self, fixed_random_subset):
    """
    :param int fixed_random_subset:
    """
    if 0 < fixed_random_subset < 1:
      fixed_random_subset = int(len(self._data) * fixed_random_subset)
    assert isinstance(fixed_random_subset, int) and fixed_random_subset > 0
    rnd = numpy.random.RandomState(42)
    seqs = self._data
    rnd.shuffle(seqs)
    seqs = seqs[:fixed_random_subset]
    self._data = seqs

  def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
    """
    If random_shuffle_epoch1, for epoch 1 with "random" ordering, we leave the given order as is.
    Otherwise, this is mostly the default behavior.

    :param int|None epoch:
    :param list[str]|None seq_list: List of sequence tags, to set a predefined order.
    :param list[int]|None seq_order: List of corpus sequence indices, to set a predefined order.
    :rtype: bool
    :returns whether the order changed (True is always safe to return)
    """
    super(OggZipDataset, self).init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
    if not epoch:
      epoch = 1
    random_seed = self._fixed_random_seed or self._get_random_seed_for_epoch(epoch=epoch)
    self._audio_random.seed(random_seed)
    if self.targets:
      self.targets.set_random_seed(random_seed)

    def get_seq_len(i):
      """
      Returns the length based on the duration entry of the dataset,
      multiplied by 100 to avoid similar rounded durations.
      It is also used when using the dataset in text-only-mode (`audio` is None).
      :param int i:
      :rtype: int
      """
      return int(self._data[i]["duration"] * 100)

    if seq_order is not None:
      self._seq_order = seq_order
    elif seq_list is not None:
      seqs = {
        self._get_tag_from_info_dict(seq): i for i, seq in enumerate(self._data)
        if self._get_tag_from_info_dict(seq) in seq_list}
      for seq_tag in seq_list:
        assert seq_tag in seqs, ("did not found all requested seqs. we have eg: %s" % (
          self._get_tag_from_info_dict(self._data[0]),))
      self._seq_order = [seqs[seq_tag] for seq_tag in seq_list]
    else:
      num_seqs = len(self._data)
      self._seq_order = self.get_seq_order_for_epoch(
        epoch=epoch, num_seqs=num_seqs, get_seq_len=get_seq_len)
      if self.epoch_wise_filter:
        self.epoch_wise_filter.debug_msg_prefix = str(self)
        self._seq_order = self.epoch_wise_filter.filter(epoch=epoch, seq_order=self._seq_order, get_seq_len=get_seq_len)
    self._num_seqs = len(self._seq_order)

    return True

  def get_current_seq_order(self):
    """
    :rtype: list[int]
    """
    assert self._seq_order is not None
    return self._seq_order

  def _get_ref_seq_idx(self, seq_idx):
    """
    :param int seq_idx:
    :return: idx in self._reference_seq_order
    :rtype: int
    """
    return self._seq_order[seq_idx]

  def have_corpus_seq_idx(self):
    """
    :rtype: bool
    """
    return True

  def get_corpus_seq_idx(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: int
    """
    return self._get_ref_seq_idx(seq_idx)

  @staticmethod
  def _get_tag_from_info_dict(info):
    """
    :param dict[str] info:
    :rtype: str
    """
    return info.get("seq_name", info.get("file", ""))

  def get_tag(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: str
    """
    return self._get_tag_from_info_dict(self._data[self._get_ref_seq_idx(seq_idx)])

  def get_all_tags(self):
    """
    :rtype: list[str]
    """
    return [self._get_tag_from_info_dict(seq) for seq in self._data]

  def get_total_num_seqs(self):
    """
    :rtype: int
    """
    return len(self._data)

  def get_data_shape(self, key):
    """
    :returns get_data(*, key).shape[1:], i.e. num-frames excluded
    :rtype: list[int]
    """
    if key == "data" and self.feature_extractor is not None:
      if self.feature_extractor.num_channels is not None:
        return [self.feature_extractor.num_channels, self.feature_extractor.get_feature_dimension()]
    return super(OggZipDataset, self).get_data_shape(key)

  def _get_transcription(self, seq_idx):
    """
    :param int seq_idx:
    :return: (targets (e.g. bpe), txt)
    :rtype: (list[int], str)
    """
    seq = self._data[self._get_ref_seq_idx(seq_idx)]
    raw_targets_txt = seq["text"]
    targets_txt = raw_targets_txt
    if self.targets:
      if self.targets_post_process:
        targets_txt = self.targets_post_process(targets_txt)
      targets_seq = self.targets.get_seq(targets_txt)
    else:
      targets_seq = []
    return targets_seq, raw_targets_txt

  def _open_audio_file(self, seq_idx):
    """
    :param int seq_idx:
    :return: io.FileIO
    """
    import io
    seq = self._data[self._get_ref_seq_idx(seq_idx)]
    if self.zip_audio_files_have_name_as_prefix:
      audio_fn = "%s/%s" % (self._names[seq['_zip_file_index']], seq["file"])
    else:
      audio_fn = seq["file"]
    raw_bytes = self._read(audio_fn, seq['_zip_file_index'])
    return io.BytesIO(raw_bytes)

  def _collect_single_seq(self, seq_idx):
    """
    :param int seq_idx:
    :rtype: DatasetSeq
    """
    seq_tag = self.get_tag(seq_idx)
    if self.feature_extractor:
      with self._open_audio_file(seq_idx) as audio_file:
        features = self.feature_extractor.get_audio_features_from_raw_bytes(audio_file, seq_name=seq_tag)
    else:
      features = numpy.zeros(())  # currently the API requires some dummy values...
    targets, txt = self._get_transcription(seq_idx)
    targets = numpy.array(targets, dtype="int32")
    raw_txt = numpy.array(txt, dtype="object")
    orth = txt.encode("utf8")
    if PY3:
      assert isinstance(orth, bytes)
      orth = list(orth)
    else:
      orth = list(map(ord, orth))
    orth = numpy.array(orth, dtype="uint8")
    return DatasetSeq(
      features=features,
      targets={"classes": targets, "raw": raw_txt, "orth": orth},
      seq_idx=seq_idx,
      seq_tag=seq_tag)
