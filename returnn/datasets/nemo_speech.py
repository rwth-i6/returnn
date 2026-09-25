"""
:class:`NemoSpeechDataset`: train on NeMo Speech (Lhotse) datasets with RETURNN epoch semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Any, Callable, Sequence, Dict, List, Tuple
from bisect import bisect_right
from collections import deque
import functools
import os
import sys
import time
import types
import numpy

from returnn.log import log
from returnn.util import better_exchook
from returnn.util.basic import try_run, NumbersDict
from returnn.util.multi_proc_non_daemonic_spawn import NonDaemonicSpawnContext
from returnn.config import SubProcCopyGlobalConfigPreInitFunc
from .basic import DatasetSeq
from .cached2 import CachedDataset2
from .util.vocabulary import Vocabulary
from .util.strings import str_to_numpy_array

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from lhotse.cut import Cut

    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from omegaconf import DictConfig

    # noinspection PyProtectedMember
    from multiprocessing.connection import Connection as mpConnection

__all__ = ["NemoSpeechDataset"]

_mp = NonDaemonicSpawnContext(process_pre_init_func=SubProcCopyGlobalConfigPreInitFunc())


class NemoSpeechDataset(CachedDataset2):
    """
    Speech dataset from a NeMo Speech (Lhotse) data config,
    e.g. a data mixture defined via nested ``input_cfg``.
    Requires ``nemo`` (NeMo Speech) and ``lhotse`` with indexed data access.

    NeMo Speech does the reading, filtering and audio loading:
    :func:`read_cutset_from_config` builds the sources (manifest readers, tarred audio, index packs),
    with the same maps and filters as :func:`get_lhotse_sampler_from_config`.
    Only the selection of the records differs from NeMo's infinite stream, see epochs below.
    RETURNN does the batching, i.e. NeMo batching and bucketing options are ignored.

    Sharding as in NeMo:
    one shard per (rank, NeMo dataloader worker), ``shard_id = rank * num_workers + worker_id``,
    with ``num_workers`` from the NeMo config,
    reading exactly the records which NeMo assigns to it.
    Every shard runs in its own worker process, like a NeMo dataloader worker.
    Seeds as in NeMo: ``shard_seed`` int or ``"randomized"`` (``"trng"`` is not reproducible, not supported).
    RETURNN dataset sharding must not be used on top.

    Epochs:
    One epoch is ``draws_per_epoch`` draws (sum over all shards).
    A draw is one record from one source (leaf of the ``input_cfg`` tree), like one step of NeMo's multiplexer:
    records which NeMo's source skips (e.g. ``_skipme``) are not drawn,
    records rejected by the filters are.
    Draws are distributed over the sources by their weights,
    within one draw of the exact weight at every epoch boundary.
    Every source continues where the previous epoch stopped, in complete passes over its records.
    So epoch N is determined by the epoch number alone (arithmetic over the quotas of earlier epochs),
    and starting directly at epoch N gives the same data as running epochs 1 to N
    (assuming the same world size and ``num_workers``).
    Within an epoch, the order is random (``shard_seed``, epoch), shuffled in windows of ``shuffle_window`` draws.
    ``num_seqs`` is only known at the end of an epoch, ``complete_frac`` is the fraction of the draws.
    Random transformations (``audio`` features, sampling ``targets``) are seeded per seq occurrence,
    except SentencePiece sampling, which only has a global RNG.
    Workers prefetch up to ``buffer_size`` seqs, also into the next epoch, tagged by epoch.

    Metadata access (tags, raw audio lengths) does not load the audio,
    the bounds check (``is_less_than_num_seqs(n)``) only the audio of seq n.
    Seqs whose audio fails to load (``fault_tolerant_audio_loading``) are dropped,
    so tags obtained before loading can shift then.
    The valid records (without ``_skipme``) are determined once per shard at startup,
    from the index pack routes, or otherwise by decoding the manifest records.

    Not supported (raises):
    NeMo map-style dataset mode (non-tarred data, ``force_map_dataset``),
    non-indexed sources (``indexed: false``),
    skipped records in sources which reshuffle per pass (``shuffle_shards`` of non-tarred sources),
    ``max_open_streams``, multi-config, multimodal sampling, and NeMo data augmentations.

    Example::

        train = {
            "class": "NemoSpeechDataset",
            "nemo_config": {
                "input_cfg": "/data/granary/input_cfg.yaml",
                "indexed": True,
                "index_pack_root": "/data/granary/idxpacks",
                "shard_seed": 1234,
                "num_workers": 4,
                "sample_rate": 16000,
                "max_duration": 40.0,
            },
            "draws_per_epoch": 2_000_000,
            "targets": {"class": "SentencePieces", "model_file": "/data/spm.model"},
        }
    """

    def __init__(
        self,
        nemo_config: Union[Dict[str, Any], str, os.PathLike, Callable[[], Union[Dict[str, Any], str, os.PathLike]]],
        *,
        draws_per_epoch: int,
        targets: Union[Vocabulary, Dict[str, Any], None] = None,
        tokenizer: Optional[Any] = None,
        audio: Optional[Dict[str, Any]] = None,
        shuffle_window: Optional[int] = None,
        buffer_size: int = 32,
        preload_next_epoch: bool = True,
        use_worker_procs: bool = True,
        _rank_and_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        """
        :param nemo_config: NeMo Lhotse dataloader config (e.g. ``train_ds``),
            as dict, YAML file, or callable returning one of those.
            Must define the data via ``input_cfg`` or ``manifest_filepath`` (+ ``tarred_audio_filepaths``).
        :param draws_per_epoch: epoch size, in draws (see class docstring), summed over all shards.
        :param targets: options for :func:`Vocabulary.create_vocab` for the "classes" key.
            Exclusive with ``tokenizer``.
        :param tokenizer: NeMo tokenizer (e.g. ``SentencePieceTokenizer``),
            or a function (or ``functools.partial``) without args returning one,
            for the "classes" key.
            As in NeMo, it also enables the token-per-second filters (with ``pretokenize``).
        :param audio: options for :class:`ExtractAudioFeatures`.
            None (default): the raw samples as loaded by NeMo, shape (time, 1).
        :param shuffle_window: number of draws which are shuffled together.
            None: NeMo ``shuffle_buffer_size`` if NeMo ``shuffle`` is set, else 1.
        :param buffer_size: number of seqs each worker prefetches
        :param preload_next_epoch: whether workers continue into the next epoch when the current is done
        :param use_worker_procs: run every shard in its own process.
            If False, all shards of this rank run in the current process (same data, e.g. for debugging).
        :param _rank_and_size: internal, set when pickled
        """
        super().__init__(**kwargs)
        if self.partition_epoch != 1 or self.repeat_epoch != 1:
            raise ValueError(f"{self}: partition_epoch/repeat_epoch not supported, use draws_per_epoch")
        if self.seq_ordering != "default":
            raise ValueError(f"{self}: seq_ordering {self.seq_ordering!r} not supported, use shuffle_window")
        if self._num_shards != 1:
            raise ValueError(
                f"{self}: RETURNN dataset sharding not supported,"
                f" this dataset follows the NeMo sharding (distributed rank x NeMo num_workers)"
            )
        if targets is not None and tokenizer is not None:
            raise ValueError(f"{self}: specify either targets or tokenizer, not both")
        self.nemo_config = nemo_config
        self.draws_per_epoch = draws_per_epoch
        self.shuffle_window = shuffle_window
        self.buffer_size = buffer_size
        self.preload_next_epoch = preload_next_epoch
        self.use_worker_procs = use_worker_procs
        if _rank_and_size is None:
            # noinspection PyProtectedMember
            from .distrib_files import _get_rank_and_size

            _rank_and_size = _get_rank_and_size()
        self._rank_and_size = tuple(_rank_and_size)

        if isinstance(targets, dict):
            targets = Vocabulary.create_vocab(**targets)
        self.targets: Optional[Vocabulary] = targets
        self.tokenizer = tokenizer
        self.audio = audio
        self._audio_feature_dim = 1
        if audio is not None:
            from .util.feature_extraction import ExtractAudioFeatures

            self._audio_feature_dim = ExtractAudioFeatures(**audio).get_feature_dimension()

        self.num_inputs = self._audio_feature_dim
        self.num_outputs = {
            "data": [self._audio_feature_dim, 2],
            "raw": {"dtype": "string", "shape": ()},
            "orth": [256, 1],
        }
        self.labels["orth"] = [chr(i) for i in range(256)]
        if self.targets is not None:
            self.num_outputs["classes"] = [self.targets.num_labels, 1]
            self.labels["classes"] = self.targets.labels
        elif self.tokenizer is not None:
            self.num_outputs["classes"] = [_create_nemo_tokenizer(self.tokenizer).vocab_size, 1]

        self._num_nemo_workers: Optional[int] = None  # lazy, _get_num_nemo_workers
        self._shard_draws: Optional[List[int]] = None  # per shard of this rank, set in init_seq_order
        self._shards: Optional[List[Union[_ShardStream, _ShardWorkerProc]]] = None  # lazy, _lazy_init_shards
        self._shards_epoch: Optional[int] = None  # the epoch the shards are initialized for
        # Merge state for the current epoch, see _merge_next.
        self._merge_rr = 0
        self._merge_seq_idx = 0
        self._merge_local_next: List[int] = []
        self._merge_exhausted: List[bool] = []
        self._merge_draws_done: List[int] = []
        self._pending: Dict[int, _ShardItem] = {}  # seq idx -> merged, not yet collected item

    def __del__(self):
        if getattr(self, "_shards", None):
            for shard in self._shards:
                try_run(shard.exit, kwargs={"join": False})

    def _uses_custom_distributed_sharding(self) -> bool:
        return True

    def _get_num_nemo_workers(self) -> int:
        if self._num_nemo_workers is None:
            config = _load_nemo_config(self.nemo_config, structured=False)
            self._num_nemo_workers = max(int(config.get("num_workers", 0) or 0), 1)
        return self._num_nemo_workers

    def _get_shard_draws(self) -> List[int]:
        rank, world_size = self._rank_and_size
        num_workers = self._get_num_nemo_workers()
        num_shards = world_size * num_workers
        if self.draws_per_epoch < num_shards:
            raise ValueError(f"{self}: draws_per_epoch {self.draws_per_epoch} < num shards {num_shards}")
        return [
            self.draws_per_epoch // num_shards + int(rank * num_workers + worker_id < self.draws_per_epoch % num_shards)
            for worker_id in range(num_workers)
        ]

    def _lazy_init_shards(self):
        if self._shards is not None:
            return
        rank, world_size = self._rank_and_size
        num_workers = self._get_num_nemo_workers()
        num_shards = world_size * num_workers
        opts = dict(
            nemo_config=self.nemo_config,
            targets=self.targets,
            tokenizer=self.tokenizer,
            audio=self.audio,
            shuffle_window=self.shuffle_window,
            buffer_size=self.buffer_size,
            preload_next_epoch=self.preload_next_epoch,
        )
        self._shards = []
        for worker_id, draws in enumerate(self._get_shard_draws()):
            shard_id = rank * num_workers + worker_id
            shard_opts = dict(
                opts, rank=rank, worker_id=worker_id, num_workers=num_workers, world_size=world_size, draws=draws
            )
            if self.use_worker_procs:
                shard = _ShardWorkerProc(name=f"{self.__class__.__name__} {self.name} shard {shard_id}", **shard_opts)
            else:
                shard = _ShardStream(**shard_opts)
            self._shards.append(shard)
        print(
            f"{self}: rank {rank}/{world_size}, {num_workers} NeMo workers,"
            f" shards {rank * num_workers}..{(rank + 1) * num_workers - 1} of {num_shards},"
            f" draws per epoch {self._shard_draws}",
            file=log.v4,
        )

    def init_seq_order(self, epoch: Optional[int] = None, seq_list=None, seq_order=None) -> bool:
        """
        :param epoch:
        :param seq_list:
        :param seq_order:
        :return: whether the order changed (True is always safe to return)
        """
        super().init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
        if seq_list is not None or seq_order is not None:
            raise NotImplementedError(f"{self}: seq_list/seq_order not supported")
        if epoch is None:
            self._num_seqs = 0
            return True
        # The shards (maybe worker procs) are only started on the first data access,
        # as the main proc of the PyTorch engine does not need the data.
        self._shards_epoch = None
        self._shard_draws = self._get_shard_draws()
        num_shards = len(self._shard_draws)
        self._merge_rr = 0
        self._merge_seq_idx = 0
        self._merge_local_next = [0] * num_shards
        self._merge_exhausted = [False] * num_shards
        self._merge_draws_done = [0] * num_shards
        self._pending.clear()
        self._estimated_num_seqs = sum(self._shard_draws)  # upper bound
        return True

    def finish_epoch(self, *, free_resources: bool = False):
        """finish epoch"""
        super().finish_epoch(free_resources=free_resources)
        if free_resources and self._shards is not None:
            for shard in self._shards:
                shard.exit()
            self._shards = None
            self._shards_epoch = None

    def _merge_next(self, *, load: bool) -> Optional[_ShardItem]:
        """
        Round-robin over the shards of this rank, like the PyTorch DataLoader over its workers.
        Shards which reached the end of the epoch are left out.
        """
        num_shards = len(self._shards)
        while not all(self._merge_exhausted):
            shard_idx = self._merge_rr % num_shards
            self._merge_rr += 1
            if self._merge_exhausted[shard_idx]:
                continue
            item = self._shards[shard_idx].get(self.epoch, self._merge_local_next[shard_idx], load=load)
            self._merge_local_next[shard_idx] += 1
            if item.seq_tag is None:  # end of epoch
                self._merge_exhausted[shard_idx] = True
                self._merge_draws_done[shard_idx] = self._shard_draws[shard_idx]
                continue
            self._merge_draws_done[shard_idx] = item.draw_idx + 1
            item.shard_idx = shard_idx
            item.complete_frac = sum(self._merge_draws_done) / sum(self._shard_draws)
            return item
        return None

    def _get_merged_item(self, seq_idx: int, *, load: bool) -> Optional[_ShardItem]:
        """
        :return: merged item for seq_idx, maybe without audio (then ``"data" not in item.features``),
            or None at the end of the epoch
        """
        if self._shards_epoch != self.epoch:
            assert self._merge_seq_idx == 0
            self._lazy_init_shards()
            for shard in self._shards:
                shard.init_epoch(self.epoch)
            self._shards_epoch = self.epoch
        while self._merge_seq_idx <= seq_idx:
            # Seqs before seq_idx (if skipped) only get decoded and filtered, their audio is not loaded.
            item = self._merge_next(load=load and self._merge_seq_idx == seq_idx)
            if item is None:
                return None
            if item.audio_failed:
                continue  # fault tolerant audio loading: dropped, does not take a seq idx
            if self._merge_seq_idx >= self.expected_load_seq_start:
                self._pending[self._merge_seq_idx] = item
            self._merge_seq_idx += 1
        return self._pending.get(seq_idx)

    def _get_loaded_item(self, seq_idx: int) -> Optional[_ShardItem]:
        """
        :return: merged item for seq_idx with audio, or None at the end of the epoch.
            Seqs whose audio fails to load (fault tolerant audio loading) are dropped here,
            and the later seqs shift down.
        """
        while True:
            item = self._get_merged_item(seq_idx, load=True)
            if item is None or "data" in item.features:
                return item
            # Was merged for metadata access only.
            data = self._shards[item.shard_idx].load(item.epoch, item.draw_idx)
            if data is not None:
                item.features["data"] = data
                return item
            del self._pending[seq_idx]
            self._pending = {(idx - 1 if idx > seq_idx else idx): v for idx, v in self._pending.items()}
            self._merge_seq_idx -= 1
            if self._num_seqs is not None:
                self._num_seqs -= 1

    def _collect_single_seq(self, seq_idx: int) -> Optional[DatasetSeq]:
        if self.epoch is None:
            return None
        for idx in [idx for idx in self._pending if idx < seq_idx]:
            del self._pending[idx]
        assert seq_idx >= self._merge_seq_idx or seq_idx in self._pending, f"{self}: cannot go back to {seq_idx}"
        item = self._get_loaded_item(seq_idx)
        if item is None:
            return None
        del self._pending[seq_idx]
        return DatasetSeq(
            seq_idx=seq_idx, seq_tag=item.seq_tag, features=item.features, complete_frac=item.complete_frac
        )

    def is_less_than_num_seqs(self, n: int) -> bool:
        """
        :return: whether n < num_seqs.
            Only the audio of seq n itself is loaded (not of the seqs before),
            as only loading tells whether the seq is there (fault tolerant audio loading).
            A known num_seqs might come from metadata only, so it is only an upper bound here.
        """
        if self.epoch is None:
            return False
        if self._num_seqs is not None and n >= self._num_seqs:
            return False
        if n < self.expected_load_seq_start or self._get_seq(n) is not None:
            return True
        if self._get_loaded_item(n) is not None:
            return True
        self._num_seqs = self._merge_seq_idx
        self.reached_final_seq = True
        return False

    def get_tag(self, sorted_seq_idx: int) -> str:
        """:return: seq tag, without loading the audio"""
        if self.epoch is None or self._get_seq(sorted_seq_idx) or sorted_seq_idx < self.expected_load_seq_start:
            return super().get_tag(sorted_seq_idx)
        item = self._get_merged_item(sorted_seq_idx, load=False)
        assert item is not None, f"{self}: seq {sorted_seq_idx} out of range"
        return item.seq_tag

    def get_seq_length(self, sorted_seq_idx: int) -> NumbersDict:
        """:return: seq lengths, without loading the audio if not needed"""
        if self.epoch is None or self._get_seq(sorted_seq_idx) or sorted_seq_idx < self.expected_load_seq_start:
            return super().get_seq_length(sorted_seq_idx)
        item = self._get_merged_item(sorted_seq_idx, load=False)
        assert item is not None, f"{self}: seq {sorted_seq_idx} out of range"
        if item.data_len is None:  # audio features, length only known after the feature extraction
            return super().get_seq_length(sorted_seq_idx)
        lens = {key: (v.shape[0] if v.ndim >= 1 else 1) for key, v in item.features.items()}
        return NumbersDict({**lens, "data": item.data_len})

    def get_data_keys(self) -> List[str]:
        """:return: available data keys"""
        keys = ["data"]
        if "classes" in self.num_outputs:
            keys.append("classes")
        return [*keys, "orth", "raw"]

    def get_target_list(self) -> List[str]:
        """:return: target keys"""
        return [key for key in self.get_data_keys() if key != "data"]

    def get_data_dim(self, key: str) -> int:
        """:return: dim of data entry with `key`"""
        if key == "raw":
            return 0
        return self.num_outputs[key][0]

    def get_data_dtype(self, key: str) -> str:
        """:return: dtype of data entry with `key`"""
        return {"data": "float32", "classes": "int32", "orth": "uint8", "raw": "string"}[key]

    def get_data_shape(self, key: str) -> List[int]:
        """:return: shape of data entry with `key`, without time axis"""
        if key == "data":
            return [self._audio_feature_dim]
        return []

    def is_data_sparse(self, key: str) -> bool:
        """:return: whether data entry with `key` is sparse"""
        return key in ("classes", "orth")


class _ShardItem:
    """
    One accepted seq of a shard, or the end of an epoch (``seq_tag is None``).
    """

    __slots__ = (
        "epoch",
        "local_idx",
        "draw_idx",
        "seq_tag",
        "features",
        "data_len",
        "audio_failed",
        "shard_idx",
        "complete_frac",
    )

    def __init__(
        self,
        *,
        epoch: int,
        local_idx: int,
        draw_idx: int,
        seq_tag: Optional[str],
        features: Optional[Dict[str, numpy.ndarray]] = None,
        data_len: Optional[int] = None,
        audio_failed: bool = False,
    ):
        """
        :param features: the text features, and "data" if the audio was loaded
        :param data_len: length of "data", if known without loading the audio
        :param audio_failed: loading the audio failed (fault tolerant audio loading)
        """
        self.epoch = epoch
        self.local_idx = local_idx
        self.draw_idx = draw_idx
        self.seq_tag = seq_tag
        self.features = features
        self.data_len = data_len
        self.audio_failed = audio_failed
        self.shard_idx: Optional[int] = None  # set by the merge in the dataset
        self.complete_frac: Optional[float] = None  # set by the merge in the dataset

    def __getstate__(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)


class _ShardStream:
    """
    One NeMo data shard (rank x NeMo worker):
    plans the epochs, decodes and filters the drawn records, loads the audio, and buffers the seqs.
    """

    def __init__(
        self,
        *,
        nemo_config,
        targets: Optional[Vocabulary],
        tokenizer: Optional[Any],
        audio: Optional[Dict[str, Any]],
        shuffle_window: Optional[int],
        buffer_size: int,
        preload_next_epoch: bool,
        rank: int,
        worker_id: int,
        num_workers: int,
        world_size: int,
        draws: int,
    ):
        config = _load_nemo_config(nemo_config)
        _check_nemo_config(config)
        self.shard_id = rank * num_workers + worker_id
        self.num_shards = world_size * num_workers
        if not isinstance(config.seed, int):
            raise ValueError(f"NemoSpeechDataset: NeMo seed must be an int, got {config.seed!r}")
        # Same as lhotse worker_init_fn, which NeMo uses for its dataloader workers.
        self.process_seed = config.seed + 100 * worker_id + 100000 * rank
        self.mux_seed = _resolve_nemo_seed(config.shard_seed, process_seed=self.process_seed)
        self.fault_tolerant_audio_loading = bool(config.fault_tolerant_audio_loading)
        self.sample_rate = int(config.sample_rate)
        if shuffle_window is None:
            shuffle_window = int(config.shuffle_buffer_size or 1) if config.shuffle else 1
        self.targets = targets
        self.nemo_tokenizer = None
        if tokenizer is not None:
            # noinspection PyUnresolvedReferences,PyPackageRequirements
            from nemo.collections.common.tokenizers.aggregate_tokenizer import TokenizerWrapper

            self.nemo_tokenizer = TokenizerWrapper(_create_nemo_tokenizer(tokenizer))
        self.feature_extractor = None
        if audio is not None:
            from .util.feature_extraction import ExtractAudioFeatures

            self.feature_extractor = ExtractAudioFeatures(random_state=numpy.random.RandomState(1), **audio)

        self.sources = _NemoSources(config, tokenizer=self.nemo_tokenizer)
        self.partitions = [
            _get_partition(
                source.node, shard_id=self.shard_id, num_shards=self.num_shards, process_seed=self.process_seed
            )
            for source in self.sources.sources
        ]
        # Draws are over the valid records only, see _get_valid_partition_indices.
        self.valid_indices = [
            _get_valid_partition_indices(source.node, partition)
            for source, partition in zip(self.sources.sources, self.partitions)
        ]
        self.planner = _DrawPlanner(
            weights=[source.weight for source in self.sources.sources],
            partition_lens=[
                len(partition) if valid is None else len(valid)
                for partition, valid in zip(self.partitions, self.valid_indices)
            ],
            draws_per_epoch=draws,
            seed=self.mux_seed,
            shuffle_window=shuffle_window,
        )
        self.draws = draws
        self.buffer_size = buffer_size
        self.preload_next_epoch = preload_next_epoch

        self._requested_epoch: Optional[int] = None
        self._buffer: deque[_ShardItem] = deque()
        self._plans: Dict[int, _EpochPlan] = {}  # epoch -> plan, for the requested and production epoch
        self._load_plan: Optional[_EpochPlan] = None  # see load
        # Production state.
        self._prod_epoch: Optional[int] = None
        self._prod_plan: Optional[_EpochPlan] = None
        self._prod_draw = 0
        self._prod_local = 0
        self._prod_done = False

    def exit(self, *, join: bool = True):
        """no resources to free"""

    def _get_plan(self, epoch: int) -> _EpochPlan:
        if epoch not in self._plans:
            for epoch_ in list(self._plans):
                if epoch_ not in (self._requested_epoch, self._prod_epoch):
                    del self._plans[epoch_]
            self._plans[epoch] = self.planner.get_epoch_plan(epoch - 1)
        return self._plans[epoch]

    def _reset_production(self, epoch: int):
        self._prod_epoch = epoch
        self._prod_plan = self._get_plan(epoch)
        self._prod_draw = 0
        self._prod_local = 0
        self._prod_done = False

    def _get_seed(self, epoch: int, draw_idx: int) -> int:
        """
        Seed for the random transformations (audio features, target sampling) of one seq occurrence,
        so the result does not depend on what was loaded or skipped before.
        """
        entropy = [self.mux_seed % (2**63), self.shard_id, epoch, draw_idx]
        return int(numpy.random.SeedSequence(entropy).generate_state(1)[0])

    def _decode_draw(self, plan: _EpochPlan, draw_idx: int) -> Optional[Cut]:
        """:return: cut, or None if rejected by a filter"""
        source_idx, pos = plan.get_draw(draw_idx)
        partition, valid = self.partitions[source_idx], self.valid_indices[source_idx]
        pass_idx, i = divmod(pos, len(partition) if valid is None else len(valid))
        if valid is not None:
            i = int(valid[i])
        token = partition.get_token(pass_idx, i)
        try:
            return self.sources.decode(source_idx, token)
        except IndexError as exc:  # record skipped by the NeMo source
            if not partition.pass_invariant:
                raise NotImplementedError(
                    f"NemoSpeechDataset: skipped record {token!r} in {self.sources.sources[source_idx].node},"
                    " skipped records are not supported for sources which reshuffle per pass"
                ) from exc
            return None  # e.g. skip_missing_manifest_entries, not known before decoding

    def init_epoch(self, epoch: int):
        """
        Start the given epoch (1-based).
        Keeps the seqs of this epoch which were already prefetched.
        """
        self._requested_epoch = epoch
        while self._buffer and self._buffer[0].epoch != epoch:
            self._buffer.popleft()
        preloaded = self._buffer[0].local_idx == 0 if self._buffer else self._prod_local == 0
        if self._prod_epoch != epoch or not preloaded:
            self._buffer.clear()
            self._reset_production(epoch)

    def _produce(self, *, load: bool) -> _ShardItem:
        """produce the next item (next accepted seq, or end of epoch) of the production epoch"""
        assert not self._prod_done
        plan = self._prod_plan
        while self._prod_draw < self.draws:
            draw_idx = self._prod_draw
            self._prod_draw += 1
            cut = self._decode_draw(plan, draw_idx)
            if cut is None:
                continue  # skipped or rejected by a filter
            seed = self._get_seed(self._prod_epoch, draw_idx)
            item = _ShardItem(
                epoch=self._prod_epoch,
                local_idx=self._prod_local,
                draw_idx=draw_idx,
                seq_tag=cut.id,
                features=self._get_text_features(cut, seed=seed),
                data_len=cut.num_samples if self.feature_extractor is None else None,
            )
            self._prod_local += 1
            if load:
                data = self._load_audio_data(cut, seed=seed)
                if data is None:
                    item.audio_failed = True
                else:
                    item.features["data"] = data
            return item
        self._prod_done = True
        return _ShardItem(epoch=self._prod_epoch, local_idx=self._prod_local, draw_idx=self.draws, seq_tag=None)

    def load(self, epoch: int, draw_idx: int) -> Optional[numpy.ndarray]:
        """
        :return: "data" of an accepted draw, by random access, or None if loading the audio failed
        """
        # Own plan instance, to not disturb the chunk state of the production.
        if self._load_plan is None or self._load_plan.epoch0 != epoch - 1:
            self._load_plan = self.planner.get_epoch_plan(epoch - 1)
        cut = self._decode_draw(self._load_plan, draw_idx)
        assert cut is not None, f"shard {self.shard_id}: draw {draw_idx} of epoch {epoch} is not an accepted seq"
        return self._load_audio_data(cut, seed=self._get_seed(epoch, draw_idx))

    def _load_audio_data(self, cut: Cut, *, seed: int) -> Optional[numpy.ndarray]:
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        from lhotse.audio.utils import suppress_audio_loading_errors

        audio = None
        with suppress_audio_loading_errors(enabled=self.fault_tolerant_audio_loading):
            audio = cut.load_audio()
        if audio is None:
            return None
        assert audio.ndim == 2 and audio.shape[0] == 1, f"expected mono audio, got shape {audio.shape}"
        audio = audio[0]
        if self.feature_extractor is not None:
            self.feature_extractor.random_state.seed(seed)
            return self.feature_extractor.get_audio_features(audio=audio, sample_rate=self.sample_rate, seq_name=cut.id)
        return audio.astype("float32")[:, None]

    def _get_text_features(self, cut: Cut, *, seed: int) -> Dict[str, numpy.ndarray]:
        texts = [sup.text for sup in cut.supervisions if sup.text]
        text = " ".join(texts)
        orth = numpy.frombuffer(text.encode("utf8"), dtype="uint8").copy()
        features = {"raw": str_to_numpy_array(text), "orth": orth}
        if self.targets is not None:
            # Note: SentencePiece sampling uses a global RNG and ignores this.
            self.targets.set_random_seed(seed)
            features["classes"] = numpy.array(self.targets.get_seq(text), dtype="int32")
        elif self.nemo_tokenizer is not None:
            # Same as NeMo LhotseSpeechToTextBpeDataset.
            features["classes"] = numpy.concatenate(
                [
                    numpy.asarray(
                        sup.tokens if hasattr(sup, "tokens") else self.nemo_tokenizer(sup.text or "", sup.language),
                        dtype="int32",
                    )
                    for sup in cut.supervisions
                ]
            )
        return features

    def get(self, epoch: int, local_idx: int, *, load: bool = True) -> _ShardItem:
        """
        :param epoch: must be the current epoch
        :param local_idx: accepted seq index within this shard, must not decrease
        :param load: whether the audio is needed. Only relevant if the seq is not buffered already.
        :return: the seq item, or the end of the epoch (``seq_tag is None``)
        """
        assert epoch == self._requested_epoch
        while (
            self._buffer
            and self._buffer[0].seq_tag is not None
            and (self._buffer[0].epoch, self._buffer[0].local_idx) < (epoch, local_idx)
        ):
            self._buffer.popleft()
        if self._buffer and self._buffer[0].epoch == epoch:
            item = self._buffer[0]
            if item.local_idx == local_idx:
                self._buffer.popleft()
                return item
            if item.seq_tag is None:
                return item  # end of epoch before local_idx
            raise Exception(f"shard {self.shard_id}: seq {local_idx} of epoch {epoch} was already dropped")
        assert self._prod_epoch == epoch and not self._buffer
        while True:
            if self._prod_done:
                return _ShardItem(epoch=epoch, local_idx=self._prod_local, draw_idx=self.draws, seq_tag=None)
            item = self._produce(load=load and self._prod_local == local_idx)
            if item.seq_tag is None or item.local_idx == local_idx:
                return item

    def prefetch_step(self) -> bool:
        """
        Prefetch one seq into the buffer.

        :return: whether something was done
        """
        if self._requested_epoch is None or len(self._buffer) >= self.buffer_size:
            return False
        if self._prod_done:
            if not self.preload_next_epoch or self._prod_epoch > self._requested_epoch:
                return False
            self._reset_production(self._prod_epoch + 1)
        self._buffer.append(self._produce(load=True))
        return True


class _ShardWorkerProc:
    """
    Runs a :class:`_ShardStream` in a subprocess, like a NeMo dataloader worker.
    Same interface as :class:`_ShardStream`.
    """

    def __init__(self, *, name: str, **shard_opts):
        parent_conn, child_conn = _mp.Pipe()
        self.conn: mpConnection = parent_conn
        self.proc = _mp.Process(name=name, target=_shard_worker_proc_loop, args=(shard_opts, child_conn), daemon=True)
        self.proc.start()
        # Closed here, stays open in the child. Reads fail when the child dies, instead of hanging.
        child_conn.close()

    def init_epoch(self, epoch: int):
        """start epoch"""
        self.conn.send(("init_epoch", {"epoch": epoch}))

    def get(self, epoch: int, local_idx: int, *, load: bool = True) -> _ShardItem:
        """get seq"""
        self.conn.send(("get", {"epoch": epoch, "local_idx": local_idx, "load": load}))
        msg, item = self.conn.recv()
        assert msg == "item"
        return item

    def load(self, epoch: int, draw_idx: int) -> Optional[numpy.ndarray]:
        """load data by draw"""
        self.conn.send(("load", {"epoch": epoch, "draw_idx": draw_idx}))
        msg, data = self.conn.recv()
        assert msg == "data"
        return data

    def exit(self, *, join: bool = True):
        """exit"""
        if self.proc is None:
            return
        try:
            self.conn.send(("exit", {}))
        except (BrokenPipeError, EOFError, ConnectionResetError):
            pass  # already exited
        if join:
            self.proc.join()
        self.proc = None

    def __del__(self):
        try_run(self.exit, kwargs={"join": False})


def _shard_worker_proc_loop(shard_opts: Dict[str, Any], conn: mpConnection):
    if sys.platform == "linux":
        with open("/proc/self/comm", "w") as f:
            f.write(f"NeMo shard {shard_opts['worker_id']}")
    better_exchook.setup_all()
    stream = _ShardStream(**shard_opts)
    # Like the lhotse worker_init_fn in NeMo dataloader workers. Not done in-process, to keep the global RNG.
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from lhotse.utils import fix_random_seed

    fix_random_seed(stream.process_seed)
    try:
        while True:
            while not conn.poll():
                if not stream.prefetch_step():
                    break
            msg, kwargs = conn.recv()
            if msg == "exit":
                break
            elif msg == "init_epoch":
                stream.init_epoch(**kwargs)
            elif msg == "get":
                conn.send(("item", stream.get(**kwargs)))
            elif msg == "load":
                conn.send(("data", stream.load(**kwargs)))
            else:
                raise Exception(f"unknown msg {msg!r}")
    except (KeyboardInterrupt, EOFError):  # when parent dies
        pass


class _NemoSource:
    """
    One leaf of the NeMo source graph, i.e. one data source of the ``input_cfg`` tree.
    """

    def __init__(self, *, node: Any, weight: float, ops: List[Tuple[str, Callable, Optional[Callable]]]):
        """
        :param node: leaf iterator node with random access, e.g. :class:`LazyNeMoTarredIterator`
        :param weight: effective sampling weight (product over the multiplexers on the path)
        :param ops: maps and filters on the path from the leaf to the top, innermost first
        """
        self.node = node
        self.weight = weight
        self.ops = ops


class _NemoSources:
    """
    The NeMo data pipeline (sources, maps, filters) as a list of sources with random access.
    """

    def __init__(self, config: DictConfig, *, tokenizer: Optional[Any]):
        cuts = _make_nemo_cutset(config, tokenizer=tokenizer)
        self.sources: List[_NemoSource] = []
        _collect_sources(cuts, weight=1.0, ops=[], out=self.sources)

    def decode(self, source_idx: int, token: Any) -> Optional[Cut]:
        """
        :return: the cut for the given record, with all NeMo maps applied,
            or None if the record is rejected by a NeMo filter.
            No audio is loaded.
            Raises IndexError for records which the NeMo source skips (e.g. ``_skipme``).
        """
        source = self.sources[source_idx]
        item = source.node[token]
        for kind, func, apply_func in source.ops:
            if kind == "map":
                if apply_func is None or apply_func(item):
                    item = func(item)
            elif kind == "filter":
                if not func(item):
                    return None
            else:
                raise ValueError(f"invalid op kind {kind!r}")
        return item


def _collect_sources(node: Any, *, weight: float, ops: list, out: List[_NemoSource]):
    """
    Walk the lhotse lazy iterator graph (as built by NeMo) and collect the leaves as sources.
    """
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from lhotse.lazy import (
        resolve_iterator_source,
        LazyMapper,
        LazyFilter,
        LazyRepeater,
        LazyIteratorMultiplexer,
        LazyInfiniteApproximateMultiplexer,
    )

    node = resolve_iterator_source(node)
    if isinstance(node, LazyMapper):
        _collect_sources(node.source, weight=weight, ops=[("map", node.fn, node.apply_fn)] + ops, out=out)
    elif isinstance(node, LazyFilter):
        _collect_sources(node.source, weight=weight, ops=[("filter", node.predicate, None)] + ops, out=out)
    elif isinstance(node, LazyRepeater):
        assert node.preserve_id, f"unexpected {node}"
        _collect_sources(node.source, weight=weight, ops=ops, out=out)  # repetitions are the passes here
    elif isinstance(node, LazyIteratorMultiplexer):
        assert not node.stop_early, f"unexpected {node}"
        weights = numpy.array(node.weights, dtype="float64")
        assert weights.min() >= 0 and weights.sum() > 0, f"invalid weights {node.weights}"
        weights /= weights.sum()
        for sub_node, sub_weight in zip(node.sources, weights):
            _collect_sources(sub_node, weight=weight * float(sub_weight), ops=ops, out=out)
    elif isinstance(node, LazyInfiniteApproximateMultiplexer):
        raise NotImplementedError("NeMo option max_open_streams not supported")
    else:
        out.append(_NemoSource(node=node, weight=weight, ops=ops))


def _load_nemo_config(nemo_config, *, structured: bool = True) -> DictConfig:
    """
    :param nemo_config: as given by the user
    :param structured: fill in the NeMo defaults. Otherwise, NeMo is not imported.
    """
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from omegaconf import OmegaConf

    if callable(nemo_config):
        # noinspection PyCallingNonCallable
        nemo_config = nemo_config()
    if isinstance(nemo_config, (str, os.PathLike)):
        config = OmegaConf.load(os.fspath(nemo_config))
    else:
        config = OmegaConf.create(nemo_config)
    if not structured:
        return config

    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from nemo.collections.common.data.lhotse.dataloader import make_structured_with_schema_warnings

    return make_structured_with_schema_warnings(config)


def _check_nemo_config(config: DictConfig):
    """
    Raise on NeMo options which change the example stream in ways not covered here.
    """
    unsupported = {
        "multi_config": bool(config.multi_config),
        "use_multimodal_sampling": bool(config.use_multimodal_sampling),
        "prompt_format": config.prompt_format is not None,
        "cut_text_into_windows_tokens": config.cut_text_into_windows_tokens is not None,
        "channel_selector": config.channel_selector is not None,
        "max_open_streams": config.max_open_streams is not None,
        "metadata_only": bool(config.metadata_only),
        "noise_path": config.noise_path is not None,
        "perturb_speed": bool(config.perturb_speed),
        "truncate_duration": config.truncate_duration is not None,
        "cut_into_windows_duration": config.cut_into_windows_duration is not None,
        "pad_min_duration": config.pad_min_duration is not None,
        "concatenate_samples": bool(config.concatenate_samples),
        "lowpass_enabled": bool(config.lowpass_enabled),
        "clipping_enabled": bool(config.clipping_enabled),
        "rir_enabled": bool(config.rir_enabled),
        "compression_enabled": bool(config.compression_enabled),
    }
    unsupported = [k for k, v in unsupported.items() if v]
    if unsupported:
        raise NotImplementedError(f"NemoSpeechDataset: NeMo options not supported: {unsupported}")


def _make_nemo_cutset(config: DictConfig, *, tokenizer: Optional[Any]):
    """
    The NeMo CutSet with all maps and filters,
    i.e. :func:`get_lhotse_sampler_from_config` up to (excluding) the sampler.
    """
    from functools import partial

    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config

    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from nemo.collections.common.data.lhotse import dataloader as nemo_dl

    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from nemo.collections.common.data.lhotse import sampling as nemo_sampling

    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from nemo.collections.common.data.lhotse.audio_token_estimator import AudioTokenEstimator

    # Finite: the passes over the sources are done here, not via lhotse repeat.
    config = config.copy()
    config.force_finite = True
    cuts, is_tarred = read_cutset_from_config(config)
    if not nemo_dl.determine_use_iterable_dataset(is_tarred, config):
        # NeMo shards map-style datasets in its sampler (by batch), not by records.
        raise NotImplementedError(
            "NemoSpeechDataset: NeMo map-style dataset mode (non-tarred data, or force_map_dataset) not supported,"
            " only iterable-style (e.g. tarred data, or force_iterable_dataset)"
        )
    audio_token_estimator = AudioTokenEstimator.from_config(
        config.audio_token_estimator, sample_rate=config.sample_rate
    )
    cuts = cuts.map(partial(nemo_dl.resample, sampling_rate=config.sample_rate), apply_fn=None)
    pretokenize = tokenizer is not None and config.pretokenize
    if pretokenize:
        cuts = cuts.map(partial(nemo_dl.tokenize, tokenizer=tokenizer), apply_fn=None)
    cuts = cuts.filter(nemo_sampling.DurationFilter(config.min_duration, config.max_duration))
    cuts = cuts.filter(
        nemo_sampling.TokenCountFilter(
            config.min_tokens,
            config.max_tokens,
            measure_total_length=config.measure_total_length,
            audio_token_estimator=audio_token_estimator,
        )
    )
    cuts = cuts.filter(nemo_sampling.ValidationStatusFilter(config.keep))
    cuts = cuts.filter(
        nemo_sampling.SpeakerFilter(
            nemo_dl.resolve_excluded_speaker_ids(config.excluded_speaker_ids),
            speaker_fields=config.speaker_filter_fields,
        )
    )
    cuts = cuts.filter(nemo_sampling.CERFilter(config.max_cer))
    cuts = cuts.filter(nemo_sampling.ContextSpeakerSimilarityFilter(config.min_context_speaker_similarity))
    if pretokenize:
        cuts = cuts.filter(nemo_sampling.TokenPerSecondFilter(config.min_tps, config.max_tps))
        cuts = cuts.filter(nemo_sampling.TokenPerTokenFilter(config.min_tpt, config.max_tpt))
    # Adds the bucketing filter for 2D bucketing, as in NeMo.
    cuts, _ = nemo_dl.determine_sampling_constraint(
        cuts, nemo_dl.determine_bucket_duration_bins(config), config, audio_token_estimator=audio_token_estimator
    )
    return cuts


def _create_nemo_tokenizer(tokenizer: Any):
    """:return: NeMo tokenizer, from the instance or function given by the user"""
    if isinstance(tokenizer, (types.FunctionType, functools.partial)):
        tokenizer = tokenizer()
    return tokenizer


def _resolve_nemo_seed(seed: Union[int, str], *, process_seed: int) -> int:
    """
    Like :func:`lhotse.dataset.dataloading.resolve_seed` in a NeMo dataloader worker.
    """
    if isinstance(seed, int):
        return seed
    if seed == "randomized":
        return process_seed
    raise ValueError(f"NemoSpeechDataset: seed {seed!r} not supported, use an int or 'randomized' (not reproducible)")


class _Partition:
    """
    The records of one source which belong to one shard, for one pass over the source.
    Same as in the NeMo/lhotse iterators with worker partition.
    """

    # Whether every pass has the same tokens in the same order.
    pass_invariant = True

    def __len__(self) -> int:
        raise NotImplementedError

    def get_token(self, pass_idx: int, i: int) -> Any:
        """
        :param pass_idx: pass over the source (the lhotse repeat epoch)
        :param i: index within the partition, 0 <= i < len(self)
        :return: token for random access into the source node
        """
        raise NotImplementedError


class _StridedPartition(_Partition):
    """``shard_id + i * num_shards``, as :class:`PartitionedIndexedIterator` without shuffling"""

    def __init__(self, n: int, *, shard_id: int, num_shards: int):
        self.n = n
        self.shard_id = shard_id
        self.num_shards = num_shards

    def __len__(self) -> int:
        return (self.n - self.shard_id + self.num_shards - 1) // self.num_shards if self.n > self.shard_id else 0

    def get_token(self, pass_idx: int, i: int) -> int:
        """token"""
        return self.shard_id + i * self.num_shards


class _ShuffledPartition(_Partition):
    """:class:`LazyShuffledRange`, with a seed per pass"""

    def __init__(
        self,
        n: int,
        *,
        seed_for_pass: Callable[[int], int],
        pass_invariant: bool,
        shard_id: int,
        num_shards: int,
    ):
        self.n = n
        self.seed_for_pass = seed_for_pass
        self.pass_invariant = pass_invariant
        self.shard_id = shard_id
        self.num_shards = num_shards
        self._range_pass_idx: Optional[int] = None
        self._range = None

    def __len__(self) -> int:
        return (self.n - self.shard_id + self.num_shards - 1) // self.num_shards if self.n > self.shard_id else 0

    def get_token(self, pass_idx: int, i: int) -> int:
        """token"""
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        from lhotse.indexing import LazyShuffledRange

        if self._range_pass_idx != pass_idx:
            self._range = LazyShuffledRange(
                self.n, seed=self.seed_for_pass(pass_idx), shard_id=self.shard_id, num_shards=self.num_shards
            )
            self._range_pass_idx = pass_idx
        return self._range[i]


class _ConcatPartition(_Partition):
    """sub partitions one after another, as in a sequential :class:`LazyIteratorChain`"""

    def __init__(self, parts: Sequence[Tuple[_Partition, Callable[[Any], Any]]]):
        self.parts = parts
        self.pass_invariant = all(part.pass_invariant for part, _ in parts)
        self._offsets = numpy.cumsum([0] + [len(part) for part, _ in parts]).tolist()

    def __len__(self) -> int:
        return self._offsets[-1]

    def get_token(self, pass_idx: int, i: int) -> Any:
        """token"""
        part_idx = bisect_right(self._offsets, i) - 1
        part, wrap = self.parts[part_idx]
        return wrap(part.get_token(pass_idx, i - self._offsets[part_idx]))


def _get_partition(node: Any, *, shard_id: int, num_shards: int, process_seed: int) -> _Partition:
    """
    The partition of the source ``node`` for the given shard.

    This mirrors the iteration logic of the indexed lhotse/NeMo iterators with worker partition,
    as there is no public API to get the partition tokens without iterating.
    """
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from lhotse.lazy import LazyIteratorChain, LazyIndexedManifestIterator

    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from lhotse.packed_lazy import LazyPackedManifestIterator

    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from nemo.collections.common.data.lhotse.nemo_adapters import (
        LazyNeMoIterator,
        LazyNeMoTarredIterator,
        LazyParquetIterator,
    )

    kwargs = dict(shard_id=shard_id, num_shards=num_shards)

    def _resolve_seed(value: Union[int, str]) -> int:
        return _resolve_nemo_seed(value, process_seed=process_seed)

    if isinstance(node, (LazyNeMoIterator, LazyNeMoTarredIterator, LazyParquetIterator)) and not node.indexed:
        raise NotImplementedError(f"NemoSpeechDataset: needs indexed sources (NeMo option indexed=true), got {node}")
    if isinstance(node, LazyNeMoIterator):
        return _get_partition(node.source, **kwargs, process_seed=process_seed)  # passes tokens to its source
    if isinstance(node, (LazyNeMoTarredIterator, LazyParquetIterator, LazyIndexedManifestIterator)):
        # noinspection PyProtectedMember
        iter_state = node._iter_state  # PartitionedIndexedIterator
        # noinspection PyProtectedMember
        if iter_state._shuffle:
            # noinspection PyProtectedMember
            seed = iter_state._seed
            return _ShuffledPartition(len(node), seed_for_pass=lambda _: seed, pass_invariant=True, **kwargs)
        return _StridedPartition(len(node), **kwargs)
    if isinstance(node, LazyPackedManifestIterator):
        if node.shuffle_shards:
            seed = _resolve_seed(node.seed)
            return _ShuffledPartition(
                len(node), seed_for_pass=lambda pass_idx: seed + pass_idx, pass_invariant=False, **kwargs
            )
        collection = node.collection
        return _ConcatPartition(
            [
                (_StridedPartition(collection.shard_length(k), **kwargs), lambda token, k_=k: (k_, token))
                for k in range(collection.sequence_count)
            ]
        )
    if isinstance(node, LazyIteratorChain):
        if node.shuffle_iters:
            if not node.is_indexed:
                raise NotImplementedError(f"NemoSpeechDataset: needs indexed sources, got {node}")
            seed = _resolve_seed(node.seed)
            return _ShuffledPartition(
                len(node), seed_for_pass=lambda pass_idx: seed + pass_idx, pass_invariant=False, **kwargs
            )
        return _ConcatPartition(
            [
                (_get_partition(sub_node, **kwargs, process_seed=process_seed), lambda token, i_=i: (i_, token))
                for i, sub_node in enumerate(node.sources)
            ]
        )
    raise NotImplementedError(f"NemoSpeechDataset: source type {type(node).__name__} not supported: {node}")


def _get_valid_partition_indices(node: Any, partition: _Partition) -> Optional[numpy.ndarray]:
    """
    The records of the partition which the NeMo source iterator yields,
    i.e. without explicitly skipped records (``_skipme``),
    as NeMo skips them before the multiplexer, so they do not count as draw.

    :return: partition indices of the valid records, in partition order, or None if all are valid.
        Also None for partitions which differ per pass (skipped records then raise at decoding).
    """
    if not partition.pass_invariant:
        return None
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoTarredIterator

    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from nemo.collections.common.data.lhotse.nemo_tar_routing import NEMO_TAR_SKIP_ORDINAL

    # noinspection PyProtectedMember
    if (
        isinstance(node, LazyNeMoTarredIterator)
        and getattr(node, "_packed_indexed", False)
        and node._packed_tar_ordinal_map is not None
    ):
        # Fast path: the index pack routes mark the skipped records, see LazyNeMoTarredIterator._decode_packed_cut_at.
        # noinspection PyProtectedMember
        manifests, ordinal_map, shard_map = (
            node._packed_manifest_collection,
            node._packed_tar_ordinal_map,
            node._packed_tar_shard_map,
        )

        def _is_valid(token: Any) -> bool:
            loc = manifests.locate(int(token))
            if (
                shard_map is not None
                and shard_map.value_in_shard(loc.shard_index, loc.local_index) == NEMO_TAR_SKIP_ORDINAL
            ):
                return False
            return ordinal_map.value_in_shard(loc.shard_index, loc.local_index) != NEMO_TAR_SKIP_ORDINAL

    else:
        # The source itself raises IndexError for records its iterator skips. Decodes the manifest record.

        def _is_valid(token: Any) -> bool:
            try:
                node[token]
            except IndexError:
                return False
            return True

    start_time = time.monotonic()
    num = len(partition)
    mask = numpy.fromiter((_is_valid(partition.get_token(0, i)) for i in range(num)), dtype=bool, count=num)
    print(
        f"NemoSpeechDataset: {type(node).__name__}: {int(mask.sum())} of {num} records valid,"
        f" checked in {time.monotonic() - start_time:.1f}s",
        file=log.v4,
    )
    return None if mask.all() else numpy.nonzero(mask)[0]


class _EpochPlan:
    """
    The draws of one epoch of one shard: which source, and which position in the source stream.
    The draws come in chunks, each an independent random arrangement,
    so that any draw can be computed with memory O(chunk size + num sources).
    """

    def __init__(
        self, *, seed: int, epoch0: int, start: numpy.ndarray, quota: numpy.ndarray, chunk_size: int, window: int
    ):
        self.seed = seed
        self.epoch0 = epoch0
        self.start = start  # per source, first position in the source stream
        self.quota = quota  # per source, number of draws in this epoch
        self.num_draws = int(quota.sum())
        self.chunk_size = chunk_size
        self.window = window
        # Per source, draws in the chunks before _counts_before_chunk_idx.
        self._counts_before_chunk_idx = 0
        self._counts_before = numpy.zeros_like(quota)
        self._cur_chunk_idx: Optional[int] = None
        self._cur_chunk: Optional[Tuple[numpy.ndarray, numpy.ndarray]] = None

    def _rng(self, chunk_idx: int) -> numpy.random.Generator:
        return numpy.random.default_rng(numpy.random.SeedSequence([self.seed % (2**63), self.epoch0, chunk_idx]))

    def _chunk_counts(self, chunk_idx: int) -> Tuple[numpy.random.Generator, numpy.ndarray, numpy.ndarray]:
        # Only one checkpoint is kept: sequential access continues from it, going back replays from the start.
        if chunk_idx < self._counts_before_chunk_idx:
            self._counts_before_chunk_idx, self._counts_before = 0, numpy.zeros_like(self.quota)
        while self._counts_before_chunk_idx < chunk_idx:
            _, counts = self._draw_chunk_counts(self._counts_before_chunk_idx, self._counts_before)
            self._counts_before = self._counts_before + counts
            self._counts_before_chunk_idx += 1
        rng, counts = self._draw_chunk_counts(chunk_idx, self._counts_before)
        return rng, self._counts_before, counts

    def _draw_chunk_counts(self, chunk_idx: int, before: numpy.ndarray) -> Tuple[numpy.random.Generator, numpy.ndarray]:
        rng = self._rng(chunk_idx)
        remaining = self.quota - before
        size = min(self.chunk_size, self.num_draws - chunk_idx * self.chunk_size)
        counts = rng.multivariate_hypergeometric(remaining, size)
        return rng, counts

    def _get_chunk(self, chunk_idx: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        if self._cur_chunk_idx == chunk_idx:
            return self._cur_chunk
        rng, before, counts = self._chunk_counts(chunk_idx)
        sources = numpy.repeat(numpy.arange(len(counts), dtype="int32"), counts)
        rng.shuffle(sources)
        # Position of each draw: the sources are consumed in order.
        order = numpy.argsort(sources, kind="stable")
        ordinal = numpy.empty(len(sources), dtype="int64")
        ordinal[order] = numpy.arange(len(sources)) - numpy.repeat(numpy.cumsum(counts) - counts, counts)
        positions = self.start[sources] + before[sources] + ordinal
        if self.window > 1:  # shuffle the draws within each window, like a shuffle buffer
            perm = numpy.lexsort((rng.random(len(sources)), numpy.arange(len(sources)) // self.window))
            sources, positions = sources[perm], positions[perm]
        self._cur_chunk_idx = chunk_idx
        self._cur_chunk = (sources, positions)
        return self._cur_chunk

    def get_draw(self, draw_idx: int) -> Tuple[int, int]:
        """
        :return: (source index, position in the source stream) of the draw.
            The source stream is the concatenation of the passes over the shard partition of the source.
        """
        assert 0 <= draw_idx < self.num_draws
        chunk_idx, i = divmod(draw_idx, self.chunk_size)
        sources, positions = self._get_chunk(chunk_idx)
        return int(sources[i]), int(positions[i])


class _DrawPlanner:
    """
    Distributes the draws of every epoch over the sources, for one shard.
    """

    def __init__(
        self,
        *,
        weights: Sequence[float],
        partition_lens: Sequence[int],
        draws_per_epoch: int,
        seed: int,
        shuffle_window: int,
        chunk_size: int = 2**16,
    ):
        # Limit of numpy multivariate_hypergeometric, see _EpochPlan.
        assert draws_per_epoch < 10**9, f"NemoSpeechDataset: draws per epoch and shard {draws_per_epoch} too large"
        weights = numpy.array(weights, dtype="float64")
        # Sources without records in this shard are never drawn (NeMo: the source iterator is empty).
        weights[numpy.array(partition_lens) == 0] = 0.0
        if weights.sum() <= 0:
            raise ValueError("NemoSpeechDataset: no source has records in this shard")
        self.probs = weights / weights.sum()
        self.draws_per_epoch = draws_per_epoch
        self.seed = seed
        self.shuffle_window = max(shuffle_window, 1)
        self.chunk_size = max(chunk_size, self.shuffle_window)
        self._cum_epoch0 = 0
        self._cum = numpy.zeros(len(weights), dtype="int64")

    def get_cumulative_draws(self, epoch0: int) -> numpy.ndarray:
        """
        :param epoch0: 0-based epoch
        :return: per source, number of draws in all epochs before
        """
        if epoch0 < self._cum_epoch0:
            self._cum_epoch0 = 0
            self._cum = numpy.zeros_like(self._cum)
        while self._cum_epoch0 < epoch0:
            self._cum = _apportion_next(self._cum, self.probs, self.draws_per_epoch)
            self._cum_epoch0 += 1
        return self._cum

    def get_epoch_plan(self, epoch0: int) -> _EpochPlan:
        """
        :param epoch0: 0-based epoch
        """
        start = self.get_cumulative_draws(epoch0).copy()
        end = _apportion_next(start, self.probs, self.draws_per_epoch)
        return _EpochPlan(
            seed=self.seed,
            epoch0=epoch0,
            start=start,
            quota=end - start,
            chunk_size=self.chunk_size,
            window=self.shuffle_window,
        )


def _apportion_next(cum: numpy.ndarray, probs: numpy.ndarray, num_draws: int) -> numpy.ndarray:
    """
    Distribute ``num_draws`` further draws over the sources,
    such that the cumulative counts follow ``probs`` as close as possible.
    Largest remainder method on the deficits.
    The result never decreases, and stays within about one draw of ``probs * total``.

    :param cum: per source, cumulative draws so far
    :param probs: per source, probability, sum 1
    :param num_draws: new draws
    :return: new cumulative draws, sum is ``cum.sum() + num_draws``
    """
    total = int(cum.sum()) + num_draws
    target = probs * total - cum  # sum is num_draws (up to float error)
    base = numpy.maximum(numpy.floor(target), 0).astype("int64")
    frac = target - base
    remaining = num_draws - int(base.sum())
    if remaining > 0:
        order = numpy.lexsort((numpy.arange(len(frac)), -frac))  # largest frac first, ties by index
        base[order[:remaining]] += 1
    elif remaining < 0:
        candidates = numpy.nonzero(base > 0)[0]
        order = candidates[numpy.lexsort((candidates, frac[candidates]))]  # smallest frac first
        base[order[:-remaining]] -= 1
    assert int(base.sum()) == num_draws and base.min() >= 0
    return cum + base
