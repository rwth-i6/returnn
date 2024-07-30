"""
Datasets dealing with audio
"""

from __future__ import annotations
from typing import Optional, Any, List, Tuple, Dict
import numpy
import typing

from .basic import DatasetSeq
from .cached2 import CachedDataset2
from .util.feature_extraction import ExtractAudioFeatures
from .util.vocabulary import Vocabulary
from .util.strings import str_to_numpy_array
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

    def __init__(
        self,
        path,
        audio,
        targets,
        targets_post_process=None,
        use_cache_manager=False,
        segment_file=None,
        zip_audio_files_have_name_as_prefix=True,
        fixed_random_subset=None,
        fixed_random_subset_seed=42,
        epoch_wise_filter=None,
        **kwargs,
    ):
        """
        :param str|list[str] path: filename to zip
        :param dict[str]|None audio: options for :class:`ExtractAudioFeatures`.
            use {} for default. None means to disable.
        :param Vocabulary|dict[str]|None targets: options for :func:`Vocabulary.create_vocab`
            (e.g. :class:`BytePairEncoding`)
        :param str|list[str]|((str)->str)|None targets_post_process: :func:`get_post_processor_function`,
            applied on orth
        :param bool use_cache_manager: uses :func:`returnn.util.basic.cf`
        :param str|None segment_file: .txt or .gz text file containing sequence tags that will be used as whitelist
        :param bool zip_audio_files_have_name_as_prefix:
        :param float|int|None fixed_random_subset:
          Value in [0,1] to specify the fraction, or integer >=1 which specifies number of seqs.
          If given, will use this random subset. This will be applied initially at loading time,
          i.e. not dependent on the epoch.
          It uses the fixed fixed_random_subset_seed as seed, i.e. it's deterministic.
        :param int fixed_random_subset_seed: Seed for drawing the fixed random subset, default 42
        :param dict|None epoch_wise_filter: see init_seq_order
        """
        import os
        import zipfile
        import returnn.util.basic
        from .meta import EpochWiseFilter

        self._separate_txt_files = {}  # name -> filename
        self._path = path
        self._use_cache_manager = use_cache_manager
        self._zip_files: Optional[List[zipfile.ZipFile]] = None  # lazily loaded
        if (
            isinstance(path, str)
            and os.path.splitext(path)[1] != ".zip"
            and os.path.isdir(path)
            and os.path.isfile(path + ".txt")
        ):
            # Special case (mostly for debugging) to directly access the filesystem, not via zip-file.
            self.paths = [os.path.dirname(path)]
            self._names = [os.path.basename(path)]
            self._use_zip_files = False
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
                    name, ext = name[: name.rindex(".")], name[name.rindex(".") :] + ext
                if use_cache_manager:
                    path_ = returnn.util.basic.cf(path_)
                if ext == ".txt.gz":
                    self._separate_txt_files[name] = path_
                    continue
                assert ext == ".zip"
                self.paths.append(path_)
                self._names.append(name)
            self._use_zip_files = True
        self.segments: Optional[typing.Set[str]] = None  # lazily loaded
        self._segment_file = segment_file
        self.zip_audio_files_have_name_as_prefix = zip_audio_files_have_name_as_prefix
        kwargs.setdefault("name", self._names[0])
        super(OggZipDataset, self).__init__(**kwargs)
        if targets is None:
            self.targets = None  # type: typing.Optional[Vocabulary]
        elif isinstance(targets, dict):
            self.targets = Vocabulary.create_vocab(**targets)
        else:
            assert isinstance(targets, Vocabulary)
            self.targets = targets
        if self.targets:
            self.labels["classes"] = self.targets.labels
        self.targets_post_process = None  # type: typing.Optional[typing.Callable[[str],str]]
        if targets_post_process:
            if callable(targets_post_process):
                self.targets_post_process = targets_post_process
            else:
                from .lm import get_post_processor_function

                self.targets_post_process = get_post_processor_function(targets_post_process)
        self._audio_random = numpy.random.RandomState(1)
        self._audio = audio
        self.feature_extractor = (
            ExtractAudioFeatures(random_state=self._audio_random, **audio) if audio is not None else None
        )
        self.num_inputs = self.feature_extractor.get_feature_dimension() if self.feature_extractor else 0
        self.num_outputs = {"raw": {"dtype": "string", "shape": ()}, "orth": [256, 1]}
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
        self._data: Optional[List[Dict[str, Any]]] = None  # lazily loaded
        self._fixed_random_subset = fixed_random_subset
        self._fixed_random_subset_seed = fixed_random_subset_seed
        if epoch_wise_filter is None:
            self.epoch_wise_filter = None  # type: Optional[EpochWiseFilter]
        elif isinstance(epoch_wise_filter, dict):
            self.epoch_wise_filter = EpochWiseFilter(epoch_wise_filter)
        else:
            assert isinstance(epoch_wise_filter, EpochWiseFilter)
            self.epoch_wise_filter = epoch_wise_filter
        self._seq_order = None  # type: typing.Optional[typing.Sequence[int]]

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

    def _collect_data_part(self, zip_index) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        collect all the entries of a single zip-file or txt file
        :param int zip_index: index of the zip-file in self._zip_files, unused when loading without zip
        :return: data entries, example_entry
        """
        from returnn.util.literal_py_to_pickle import literal_eval

        data: List[Dict[str, Any]] = literal_eval(self._read("%s.txt" % self._names[zip_index], zip_index))
        assert data and isinstance(data, list)
        first_entry = data[0]
        assert isinstance(first_entry, dict)
        assert isinstance(first_entry["text"], str)
        assert isinstance(first_entry["duration"], float)
        # when 'audio' is None and sequence names are given, this dataset can be used in text-only mode
        if "file" in first_entry:
            assert isinstance(first_entry["file"], str)
        else:
            assert not self.feature_extractor, (
                "%s: feature extraction is enabled, but no audio files are specified" % self
            )
            assert isinstance(first_entry["seq_name"], str)
        # add index to data list
        for entry in data:
            entry["_zip_file_index"] = zip_index
        example_entry = None
        if data:
            example_entry = data[0]  # before filtering
        if self.segments:
            data[:] = [entry for entry in data if self._get_tag_from_info_dict(entry) in self.segments]
        return data, example_entry

    def _lazy_init(self):
        """
        :return: entries
        :rtype: list[dict[str]]
        """
        if self._data is not None:
            return

        if self._segment_file:
            self._read_segment_list(self._segment_file)

        if self._use_zip_files:
            import zipfile

            self._zip_files = [zipfile.ZipFile(path) for path in self.paths]

        data = []
        example_entry = None
        if self._use_zip_files:
            for zip_index in range(len(self._zip_files)):
                zip_data, example_entry = self._collect_data_part(zip_index)
                data += zip_data
        else:
            # collect data from a txt file
            data, example_entry = self._collect_data_part(0)

        assert len(data) > 0, (
            f"{self}: no data found? files {self._zip_files or self._path},"
            f" filter {self.segments}, example {example_entry}"
        )

        fixed_random_subset = self._fixed_random_subset
        if fixed_random_subset:
            if 0 < fixed_random_subset < 1:
                fixed_random_subset = int(len(data) * fixed_random_subset)
            assert isinstance(fixed_random_subset, int) and fixed_random_subset > 0
            rnd = numpy.random.RandomState(self._fixed_random_subset_seed)
            rnd.shuffle(data)
            data = data[:fixed_random_subset]

        self._data = data

    def finish_epoch(self, *, free_resources: bool = False):
        """finish epoch"""
        super().finish_epoch()
        self._seq_order = None
        self._num_seqs = 0
        if free_resources:
            # Basically undo the _lazy_init, such that _lazy_init would init again next time.
            self._data = None
            self.segments = None
            self._zip_files = None

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

        if epoch is None and seq_list is None and seq_order is None:
            # This is called via initialize() with epoch=None, just to init some other things.
            # We are not expected to have prepared any real epoch here.
            # We do an early exit here to defer the lazy init.
            self._num_seqs = 0
            return True

        self._lazy_init()
        random_seed = self._get_random_seed_for_epoch(epoch=epoch)
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
            seqs = {self._get_tag_from_info_dict(seq): i for i, seq in enumerate(self._data)}
            for seq_tag in seq_list:
                assert seq_tag in seqs, "Did not find all requested seqs. We have eg: %s" % (
                    self._get_tag_from_info_dict(self._data[0]),
                )
            self._seq_order = [seqs[seq_tag] for seq_tag in seq_list]
        else:
            num_seqs = len(self._data)
            self._seq_order = self.get_seq_order_for_epoch(epoch=epoch, num_seqs=num_seqs, get_seq_len=get_seq_len)
            if self.epoch_wise_filter:
                self.epoch_wise_filter.debug_msg_prefix = str(self)
                self._seq_order = self.epoch_wise_filter.filter(
                    epoch=epoch, seq_order=self._seq_order, get_seq_len=get_seq_len
                )
        self._num_seqs = len(self._seq_order)

        return True

    def supports_seq_order_sorting(self) -> bool:
        """supports sorting"""
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

    def get_corpus_seq_idx(self, seq_idx: int) -> int:
        """
        :param seq_idx:
        """
        return self._get_ref_seq_idx(seq_idx)

    @staticmethod
    def _get_tag_from_info_dict(info: Dict[str, Any]) -> str:
        """
        :param info:
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
        self._lazy_init()
        return [self._get_tag_from_info_dict(seq) for seq in self._data]

    def get_total_num_seqs(self, *, fast: bool = False) -> int:
        """
        :rtype: int
        """
        if fast and self._data is None:
            raise Exception(f"{self} not initialized")
        self._lazy_init()
        return len(self._data)

    def get_data_dtype(self, key: str) -> str:
        """:return: dtype of data entry with `key`"""
        if key == "data":
            return "float32"
        elif key == "classes":
            return "int32"
        elif key == "raw":
            return "string"
        elif key == "orth":
            return "uint8"
        else:
            raise ValueError(f"{self}: unknown data key: {key}")

    def get_data_keys(self) -> List[str]:
        """:return: available data keys"""
        keys = []
        if self.feature_extractor is not None:
            keys.append("data")
        if self.targets is not None:
            keys.append("classes")
        return [*keys, "orth", "raw"]

    def get_data_shape(self, key: str):
        """
        :returns get_data(*, key).shape[1:], i.e. num-frames excluded
        :rtype: list[int]
        """
        if key == "data":
            assert self.feature_extractor is not None
            if self.feature_extractor.num_channels is not None:
                return [self.feature_extractor.num_channels, self.feature_extractor.get_feature_dimension()]
            return [self.feature_extractor.get_feature_dimension()]
        elif key in ["classes", "orth", "raw"]:
            return []
        else:
            raise ValueError(f"{self}: unknown data key {key}")

    def is_data_sparse(self, key: str) -> bool:
        """:return: whether data entry with `key` is sparse"""
        return key == "classes"

    def _get_transcription(self, corpus_seq_idx: int):
        """
        :param corpus_seq_idx:
        :return: (targets (e.g. bpe), txt)
        :rtype: (list[int], str)
        """
        seq = self._data[corpus_seq_idx]
        raw_targets_txt = seq["text"]
        targets_txt = raw_targets_txt
        if self.targets:
            if self.targets_post_process:
                targets_txt = self.targets_post_process(targets_txt)
            targets_seq = self.targets.get_seq(targets_txt)
        else:
            targets_seq = []
        return targets_seq, raw_targets_txt

    def _open_audio_file(self, corpus_seq_idx: int):
        """
        :param corpus_seq_idx:
        :return: io.FileIO
        """
        import io

        seq = self._data[corpus_seq_idx]
        if self.zip_audio_files_have_name_as_prefix:
            audio_fn = "%s/%s" % (self._names[seq["_zip_file_index"]], seq["file"])
        else:
            audio_fn = seq["file"]
        raw_bytes = self._read(audio_fn, seq["_zip_file_index"])
        return io.BytesIO(raw_bytes)

    def _collect_single_seq(self, seq_idx):
        """
        :param int seq_idx:
        :rtype: DatasetSeq
        """
        corpus_seq_idx = self._get_ref_seq_idx(seq_idx)
        seq = self.get_corpus_seq(corpus_seq_idx)
        seq.seq_idx = seq_idx
        return seq

    def have_get_corpus_seq(self) -> bool:
        """
        :return: whether this dataset supports :func:`get_corpus_seq`
        """
        return True

    def get_corpus_seq(self, corpus_seq_idx: int) -> DatasetSeq:
        """
        :param corpus_seq_idx:
        :return: seq
        """
        self._lazy_init()
        seq_tag = self._get_tag_from_info_dict(self._data[corpus_seq_idx])
        features = {}
        if self.feature_extractor:
            with self._open_audio_file(corpus_seq_idx) as audio_file:
                data = self.feature_extractor.get_audio_features_from_raw_bytes(audio_file, seq_name=seq_tag)
            features["data"] = data
        targets, txt = self._get_transcription(corpus_seq_idx)
        if self.targets is not None:
            features["classes"] = numpy.array(targets, dtype="int32")
        raw_txt = str_to_numpy_array(txt)
        orth = txt.encode("utf8")
        if PY3:
            assert isinstance(orth, bytes)
            orth = list(orth)
        else:
            orth = list(map(ord, orth))
        orth = numpy.array(orth, dtype="uint8")
        return DatasetSeq(
            features={**features, "raw": raw_txt, "orth": orth},
            seq_idx=corpus_seq_idx,
            seq_tag=seq_tag,
        )
