"""
Provides :class:`PostprocessingDataset`.
"""

from __future__ import annotations

from collections import deque
from itertools import islice
import numpy
from numpy.random import RandomState
import select
import sys
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar

from returnn.config import SubProcCopyGlobalConfigPreInitFunc
from returnn.datasets.basic import DatasetSeq
from returnn.datasets.util.strings import str_to_numpy_array
from returnn.datasets.util.vocabulary import Vocabulary
from returnn.tensor import Tensor, TensorDict
from returnn.tensor.dim import Dim
from returnn.util import basic as util, better_exchook
from returnn.util.multi_proc_non_daemonic_spawn import NonDaemonicSpawnContext
from .basic import Dataset, init_dataset
from .cached2 import CachedDataset2

# noinspection PyProtectedMember
from multiprocessing.connection import Connection as mpConnection

_mp = NonDaemonicSpawnContext(process_pre_init_func=SubProcCopyGlobalConfigPreInitFunc())


__all__ = ["PostprocessingDataset", "LaplaceOrdering", "Sequential"]


class PostprocessingDataset(CachedDataset2):
    """
    A dataset that allows for generic online post-processing of data from another
    dataset using a function on the segment level and on the level of multiple
    segments via an iterator.

    This allows integrating various data augmentation techniques like e.g. Mixup,
    SpecAugment or speed perturbation into the data loading pipeline.

    The integration into the data loading pipeline makes it easy to distribute the
    data processing work across multiple CPU cores and in turn frees the GPU from
    data preprocessing tasks.

    Multiprocessing can either be done using :class:``MultiProcDataset`` or by setting
    `num_workers > 0` on this class.

    The latter only applies parallelism to the post-processing functions themselves,
    and does not duplicate the underlying dataset once per worker.
    This is often fast enough and has the advantage of lower memory consumption.

    Example usage::

        from returnn.tensor.dim import Dim, DimTypes

        time_dim = Dim(None, kind=DimTypes.Spatial)
        new_data_dim = Dim(128)

        train = {
            "class": "PostprocessingDataset",
            "dataset": {
                "class": "HDFDataset",
                "files": ["/path/to/data.hdf"],
            },
            # one of them, but not both:
            # (data: TensorDict, *, rng: numpy.random.RandomState, **kwargs) -> TensorDict
            "map_seq": map_seq,
            # (iter: Iterator[TensorDict], *, rng: numpy.random.RandomState, **kwargs) -> Iterator[TensorDict]
            "map_seq_stream": map_seqs,
            # only required when data shapes change wrt. the wrapped dataset:
            "map_outputs": {
                "data": {"dims": [time_dim, new_data_dim]},
            },
        }

    The postprocessor functions operate on ``TensorDict``s, which have entries for
    all data keys in the underlying dataset.

    There may also be additional "meta" entries in the tensor dicts, like ``complete_frac``,
    ``seq_idx`` and ``seq_tag``.
    These should be copied over in a manner that is reasonable for the use case at hand and
    ensures forwards compatibility as well as reasonably possible.

    The dataset itself does not support its own seq ordering and relies on the wrapped
    dataset for seq ordering instead. Specifying a ``seq_ordering`` other than ``default``
    results in an error.

    However, we provide an iterator that implements the common `laplace:.NUM_SEQS_PER_BIN`-variant
    of seq ordering that any custom ``map_seq_stream``-style postprocessing iterator can be composed
    with to implement the ordering via :class:`LaplaceOrdering`.

    Like this::

        from returnn.datasets.postprocessing import LaplaceOrdering, Sequential

        def my_map_seq_stream(iterator):
            ...

        train = {
            "class": "PostprocessingDataset",
            # ...
            "map_seq_stream": Sequential(
                my_map_seq_stream,
                LaplaceOrdering(num_seqs_per_bin=1000),
            ),
        }
    """

    def __init__(
        self,
        *,
        dataset: Dict[str, Any],
        map_seq: Optional[Callable] = None,
        map_seq_stream: Optional[Callable] = None,
        map_outputs: Optional[Dict[str, Any]] = None,
        map_seq_stream_preserves_num_seqs: Optional[bool] = None,
        buf_size: int = 1,
        num_workers: int = 0,
        **kwargs,
    ):
        """
        :param dataset: inner dataset to be post-processed
        :param map_seq: post processor function operating on the single-segment level.
            Signature: `(data: TensorDict, *, rng: numpy.random.RandomState, **kwargs) -> TensorDict`
            To avoid confusion on the order of how the processing functions are applied to the data, only one of
            ``map_seq`` and ``map_seq_stream`` can be specified at a time.
            To ensure forwards compatibility, the function must accept ``**kwargs`` as its last argument.
            This is enforced by passing randomly named parameters at runtime.
        :param map_seq_stream: post processor function operating on the multiple segment level via an iterator.
            Allows merging multiple segments into one, or generating multiple output segments from one input segment.
            Signature:
                ``(iter: Iterator[TensorDict], *, rng: numpy.random.RandomState, **kwargs) -> Iterator[TensorDict]``
            To avoid confusion on the order of how the processing functions are applied to the data, only one of
            ``map_seq`` and ``map_seq_stream`` can be specified at a time.
            To ensure forwards compatibility, the function must accept ``**kwargs`` as its last argument.
            This is enforced by passing randomly named parameters at runtime.
        :param map_outputs: Type and axis specification of the outputs of the mapping functions,
            like extern_data and model_outputs.
            To simplify the common case when no shapes change, this value can be left unspecified. The dataset then
            assumes the same data layout as returned by the wrapped dataset.
            Example: `map_outputs={"data": {"dim": 42}}`
        :param map_seq_stream_preserves_num_seqs: whether the function in map_seq_stream preserves the number of
            sequences, i.e. for every input sequence there is exactly one output sequence.
        :param buf_size: Buffer size for each worker, number of seqs to prefetch. Must be > 0.
        :param num_workers: If > 0, configures the number of worker processes to use for data postprocessing.
            Only the postprocessing is distributed across subprocesses,
            the underlying dataset is only instantiated once.
            This usually has lower memory consumption than using :class:``MultiProcDataset``.
        :param kwargs: see :class:`CachedDataset2`, :class:`Dataset`
        """
        super().__init__(**kwargs)

        if self.seq_ordering != "default":
            raise ValueError(f"{self}: specify seq_ordering in wrapped dataset, not in {self.__class__.__name__}")
        if map_seq is None and map_seq_stream is None:
            raise ValueError(f"{self}: need to either set map_seq or map_seq_stream")
        if map_seq and map_seq_stream:
            raise ValueError(f"{self}: cannot set both map_seq and map_seq_stream")
        if map_seq and map_seq_stream_preserves_num_seqs is not None:
            raise ValueError(f"{self}: map_seq_stream_preserves_num_seqs is only allowed with map_seq_stream")

        if buf_size < 1:
            raise ValueError(f"{self}: buf_size must be > 0, but got {buf_size}")
        if num_workers < 0:
            raise ValueError(f"{self}: num_workers must be >= 0, but got {num_workers}")

        self._dataset_def = dataset
        self._map_seq = map_seq
        self._map_seq_stream = map_seq_stream
        if map_seq_stream_preserves_num_seqs is None and map_seq_stream is not None:
            map_seq_stream_preserves_num_seqs = getattr(map_seq_stream, "preserves_num_seqs", None)
        assert map_seq_stream_preserves_num_seqs is None or isinstance(map_seq_stream_preserves_num_seqs, bool)
        self._map_seq_stream_preserves_num_seqs = map_seq_stream_preserves_num_seqs
        self._map_outputs = map_outputs
        self._seq_list_for_validation: Optional[List[str]] = None

        self._dataset = init_dataset(self._dataset_def, parent_dataset=self)
        if self._map_seq_stream is None or self._map_seq_stream_preserves_num_seqs is True:
            # if the stream mapper is set, the num_seqs may change and the estimation is less accurate
            self._estimated_num_seqs = self._dataset.estimated_num_seqs
        self._data_iter: Optional[Iterator[Tuple[int, TensorDict]]] = None
        self._data_iter_produced_num_seqs = 0

        self._buf_size = buf_size
        # Ensure only one feeder thread at a time accesses the wrapped dataset to
        # prevent race conditions while moving from one epoch to the next.
        self._dataset_lock = threading.Lock()
        self._multi_proc_data_iter: Optional[_MultiProcDataIter] = None  # store for cleanup
        self._num_workers = num_workers
        self._worker_procs: Optional[List[_WorkerProcParent]] = None

        self._in_tensor_dict_template = TensorDict(
            {name: self._make_tensor_template_from_input(name) for name in self._dataset.get_data_keys()}
        )
        self.labels = {}
        if self._map_outputs is not None:
            self._out_tensor_dict_template = TensorDict()
            self._out_tensor_dict_template.update(self._map_outputs, auto_convert=True)
        else:
            self._out_tensor_dict_template = self._in_tensor_dict_template.copy_template()
            self.labels = self._dataset.labels.copy()
        # update only after _out_tensor_dict_template has been created from _in_tensor_dict_template
        self._in_tensor_dict_template.update(
            {
                "complete_frac": {"dims": (), "dtype": "float32"},
                "seq_idx": {"dims": (), "dtype": "int32"},
                "seq_tag": {"dims": (), "dtype": "string"},
            },
            auto_convert=True,
        )
        self.num_outputs = {
            k: (t.sparse_dim.size if t.sparse_dim else t.shape[-1] if len(t.shape) > 0 else 1, t.ndim)
            for k, t in self._out_tensor_dict_template.data.items()
        }
        self._default_input = "data" if "data" in self.num_outputs else next(iter(self.num_outputs.keys()))
        self.num_inputs = self.num_outputs[self._default_input][0]

        for k, t in self._out_tensor_dict_template.data.items():
            if self.labels.get(k):
                continue
            if t.vocab:
                self.labels[k] = t.vocab.labels
            elif t.sparse_dim:  # sparse_dim but not vocab
                self.labels[k] = list(map(str, range(t.sparse_dim.dimension)))  # dummy labels

    def init_seq_order(
        self, epoch: Optional[int] = None, seq_list: Optional[List[str]] = None, seq_order: Optional[List[int]] = None
    ):
        """
        :param epoch:
        :param seq_list:
        :param seq_order:
        :return: whether the order changed (True is always safe to return)
        """
        super().init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)

        if self._map_seq_stream is not None:
            if seq_list is not None:
                raise ValueError("map_seq_stream is set, cannot specify custom seq_list")
            if seq_order is not None:
                raise ValueError("map_seq_stream is set, cannot specify custom seq_order")

        if self._multi_proc_data_iter is not None:
            self._multi_proc_data_iter.stop()
            self._multi_proc_data_iter = None

        if epoch is None and seq_list is None and seq_order is None:
            self._num_seqs = 0
            return True

        if self._num_workers > 0:
            self._lazy_init_worker_procs()
            assert self._worker_procs is not None and len(self._worker_procs) == self._num_workers
            parent_conns, child_conns = zip(*[_mp.Pipe() for _ in range(self._num_workers)])
            base_rng_seed = self._get_random_seed_for_epoch(epoch=epoch) * 683859 * self._num_workers
            for i, (worker, child_conn) in enumerate(zip(self._worker_procs, child_conns)):
                worker.init_seq_order(
                    epoch=epoch,
                    rng_seed=(base_rng_seed + 30411 * i) % (2**32 - 1),
                    seq_list=seq_list,
                    seq_pipe=child_conn,
                )
            data_iter = self._multi_proc_data_iter = self._init_multi_proc_data_iter(
                epoch=epoch, feeder_to_worker_conns=parent_conns, seq_list=seq_list, seq_order=seq_order
            )
        else:
            self._dataset.init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
            data_iter = _build_mapping_iter(
                _iterate_dataset(self._dataset, in_tensor_dict_template=self._in_tensor_dict_template),
                map_seq=self._map_seq,
                map_seq_stream=self._map_seq_stream,
                epoch=epoch,
                out_tensor_dict_template=self._out_tensor_dict_template,
                rng=RandomState(self._get_random_seed_for_epoch(epoch=epoch)),
                seq_list_for_validation=seq_list,
            )
        self._data_iter = enumerate(data_iter)
        self._data_iter_produced_num_seqs = 0
        self._seq_list_for_validation = seq_list
        if self._map_seq_stream is None or self._map_seq_stream_preserves_num_seqs is True:
            # If we don't have an iterable mapper (or the user explicitly specifies this),
            # we know the number of segments exactly equals the number of segments in the wrapped dataset
            try:
                self._num_seqs = self._dataset.num_seqs
            except NotImplementedError:
                pass  # some datasets don't know their num_seqs
        return True

    def __del__(self):
        if self._multi_proc_data_iter is not None:
            self._multi_proc_data_iter.stop(join=True)
            self._multi_proc_data_iter = None
        if not self._worker_procs:
            return
        got_exception = False
        for parent in self._worker_procs:
            # noinspection PyBroadException
            try:
                parent.exit(join=False)
            except Exception:
                got_exception = True
        if got_exception:
            return
        for parent in self._worker_procs:
            util.try_run(parent.worker_proc.join)

    def get_current_seq_order(self):
        """:return: current seq order of wrapped dataset, if map_seq_stream is not used"""
        if self._map_seq_stream is not None:
            raise Exception(f"{self}: get_current_seq_order is not allowed when map_seq_stream is set.")
        assert self._dataset is not None
        return self._dataset.get_current_seq_order()

    def get_data_keys(self):
        """:return: available data keys"""
        return list(self._out_tensor_dict_template.data.keys())

    def get_data_dtype(self, key):
        """:return: dtype of data entry `key`"""
        return self._out_tensor_dict_template.data[key].dtype

    def get_total_num_seqs(self, *, fast=False):
        """:return: total num seqs excluding partition_epoch"""
        if self._map_seq_stream is not None:
            raise util.OptionalNotImplementedError(
                f"{self}: get_total_num_seqs not allowed when map_seq_stream is set."
            )
        assert self._dataset is not None
        return self._dataset.get_total_num_seqs(fast=fast)

    def get_all_tags(self) -> List[str]:
        """:return: all tags"""
        if self._map_seq_stream is not None:
            raise util.OptionalNotImplementedError(f"{self}: get_all_tags not allowed when map_seq_stream is set.")
        assert self._dataset is not None
        return self._dataset.get_all_tags()

    def supports_sharding(self) -> bool:
        """:return: whether this dataset supports sharding"""
        assert self._dataset is not None
        return self._dataset.supports_sharding()

    def finish_epoch(self, *, free_resources=False):
        """finish_epoch"""
        super().finish_epoch(free_resources=free_resources)
        if not free_resources:
            return
        if self._multi_proc_data_iter is not None:
            self._multi_proc_data_iter.stop(join=True)
            self._multi_proc_data_iter = None
        if self._worker_procs is not None:
            for wp in self._worker_procs:
                wp.exit(join=True)
            self._worker_procs = None

    def _collect_single_seq(self, seq_idx: int) -> Optional[DatasetSeq]:
        while True:
            try:
                loaded_seq_idx, tensor_dict = next(self._data_iter)
                self._data_iter_produced_num_seqs += 1
                if self._num_seqs is not None:
                    assert self._data_iter_produced_num_seqs <= self._num_seqs, (
                        f"{self}: map_seq_stream yielded more seqs ({self._data_iter_produced_num_seqs}) "
                        f"than expected ({self._num_seqs}). map_seq_stream_preserves_num_seqs is set to "
                        f"{self._map_seq_stream_preserves_num_seqs}"
                    )
            except StopIteration:
                if self._num_seqs is not None:
                    assert self._data_iter_produced_num_seqs == self._num_seqs, (
                        f"{self}: map_seq_stream yielded {self._data_iter_produced_num_seqs} seqs, "
                        f"while {self._num_seqs} were expected. map_seq_stream_preserves_num_seqs is set to "
                        f"{self._map_seq_stream_preserves_num_seqs}"
                    )
                return None
            assert loaded_seq_idx <= seq_idx, "_collect_single_seq must be done monotonically"
            if loaded_seq_idx != seq_idx:
                continue
            complete_frac = (
                float(tensor_dict.data["complete_frac"].raw_tensor) if "complete_frac" in tensor_dict.data else None
            )
            seq_tag = str(tensor_dict.data["seq_tag"].raw_tensor) if "seq_tag" in tensor_dict.data else f"seq-{seq_idx}"
            features = {k: t.raw_tensor for k, t in tensor_dict.data.items() if k not in ["complete_frac", "seq_tag"]}
            seq = DatasetSeq(complete_frac=complete_frac, features=features, seq_idx=seq_idx, seq_tag=seq_tag)
            return seq

    def _make_tensor_template_from_input(self, data_key: str) -> Tensor:
        dtype = self._dataset.get_data_dtype(data_key)
        if dtype == "string":
            dims = []
        else:
            feature_dims = [
                Dim(dimension=dim, name=f"{data_key}_dim{i + 1}")
                for i, dim in enumerate(self._dataset.get_data_shape(data_key))
            ]
            dims = [Dim(dimension=None, name=f"{data_key}_frame"), *feature_dims]
        sparse_dim = None
        if self._dataset.is_data_sparse(data_key):
            sparse_dim = Dim(dimension=self._dataset.get_data_dim(data_key), name=f"{data_key}_sparse")
            labels = self._dataset.labels.get(data_key)
            if labels and len(labels) == sparse_dim.dimension:
                sparse_dim.vocab = Vocabulary.create_vocab_from_labels(self._dataset.labels[data_key])
        return Tensor(data_key, dims=dims, dtype=dtype, sparse_dim=sparse_dim)

    def _lazy_init_worker_procs(self):
        if self._worker_procs is not None:
            return
        self._worker_procs = [
            _WorkerProcParent(
                name=f"{self.__class__.__name__} {self.name} worker",
                buffer_size=self._buf_size,
                index=i,
                map_seq=self._map_seq,
                map_seq_stream=self._map_seq_stream,
                out_tensor_dict_template=self._out_tensor_dict_template,
            )
            for i in range(self._num_workers)
        ]

    def _init_multi_proc_data_iter(
        self,
        *,
        epoch: int,
        feeder_to_worker_conns: Sequence[mpConnection],
        seq_list: Optional[List[str]] = None,
        seq_order: Optional[List[int]] = None,
    ) -> _MultiProcDataIter:
        assert len(feeder_to_worker_conns) == self._num_workers

        quit_event = threading.Event()
        dataset_thread = threading.Thread(
            target=self._init_seq_order_and_distribute_seqs_to_children,
            kwargs={
                "epoch": epoch,
                "quit_event": quit_event,
                "seq_list": seq_list,
                "seq_order": seq_order,
                "worker_conns": feeder_to_worker_conns,
            },
            name=f"{self.__class__.__name__} feeder ep {epoch}",
        )
        # parent_conns are not closed here, because they move to a different thread, not process,
        # and so they must remain open.
        dataset_thread.start()
        data_iter = _MultiProcDataIter(
            dataset_thread=dataset_thread, quit_event=quit_event, worker_procs=self._worker_procs
        )
        return data_iter

    def _init_seq_order_and_distribute_seqs_to_children(
        self,
        *,
        epoch: int,
        quit_event: threading.Event,
        seq_list: Optional[List[str]] = None,
        seq_order: Optional[List[int]] = None,
        worker_conns: Sequence[mpConnection],
    ):
        """
        Initialize the wrapped dataset and distribute the contained sequences to the child worker processes.
        """

        assert self._buf_size > 0
        assert len(worker_conns) > 0
        assert self._num_workers > 0

        caches: List[deque[TensorDict]] = [deque() for _ in range(len(worker_conns))]

        def _any_conn_ready() -> bool:
            ready, _, _ = select.select(worker_conns, [], [], 0)
            return len(ready) > 0

        def _maybe_distrib_seq(*, timeout=0.1):
            assert timeout >= 0.0
            # do not block indefinitely to periodically check the quit_event
            ready_conns, _, _ = select.select(worker_conns, [], [], timeout)
            assert len(worker_conns) == len(caches)
            for child_queue, cache in zip(worker_conns, caches):
                if child_queue not in ready_conns:
                    continue
                msg, _ = child_queue.recv()
                assert msg == "get_seq"
                tensor_dict = cache.popleft() if len(cache) > 0 else None
                child_queue.send(("seq", tensor_dict))

        # Lock ensures that only one thread at a time accesses the wrapped dataset.
        # This protects against issues while moving from one epoch to the next.
        with self._dataset_lock:
            self._dataset.init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
            data_iter = _iterate_dataset(self._dataset, in_tensor_dict_template=self._in_tensor_dict_template)
            data_iter = enumerate(data_iter)

            def _add_to_cache() -> bool:
                try:
                    idx, tensor_dict = next(data_iter)
                    caches[idx % len(caches)].append(tensor_dict)
                    return True
                except StopIteration:
                    return False

            while not quit_event.is_set():
                # fetch seqs until all caches have at least one seq,
                # if no child is waiting for seqs also fill until buf_size
                while any(len(cache) == 0 for cache in caches) or (
                    sum(len(cache) for cache in caches) < self._buf_size and not _any_conn_ready()
                ):
                    if not _add_to_cache():
                        break
                if all(len(c) == 0 for c in caches):
                    break
                try:
                    _maybe_distrib_seq()
                except (BrokenPipeError, EOFError):
                    # queue is closed, i.e. the worker process crashed for some reason -> stop
                    break

        for queue in worker_conns:
            try:
                queue.send(("seq", None))
            except (BrokenPipeError, EOFError):
                # queue is already closed, i.e. the worker process died
                pass
            finally:
                queue.close()


def _iterate_dataset(dataset: Dataset, *, in_tensor_dict_template: TensorDict) -> Iterator[TensorDict]:
    """
    :return: generator providing data samples in the form of a TensorDict
    """
    data_keys = dataset.get_data_keys()

    seq_index = 0
    while dataset.is_less_than_num_seqs(seq_index):
        dataset.load_seqs(seq_index, seq_index + 1)

        tensor_dict = in_tensor_dict_template.copy_template()
        for data_key in data_keys:
            tensor_dict.data[data_key].raw_tensor = dataset.get_data(seq_index, data_key)

        complete_frac = dataset.get_complete_frac(seq_index, allow_only_lr_suitable=True)
        if complete_frac is not None:
            comp_frac_raw_tensor = numpy.array(complete_frac, dtype=numpy.float32)
            tensor_dict.data["complete_frac"].raw_tensor = comp_frac_raw_tensor
        seq_idx_raw_tensor = numpy.array(seq_index, dtype=numpy.int32)
        tensor_dict.data["seq_idx"].raw_tensor = seq_idx_raw_tensor
        seq_tag_raw_tensor = str_to_numpy_array(dataset.get_tag(seq_index))
        tensor_dict.data["seq_tag"].raw_tensor = seq_tag_raw_tensor

        yield tensor_dict
        seq_index += 1


def _build_mapping_iter(
    data_iter: Iterator[TensorDict],
    *,
    map_seq: Optional[Callable] = None,
    map_seq_stream: Optional[Callable] = None,
    epoch: int,
    out_tensor_dict_template: TensorDict,
    rng: RandomState,
    seq_list_for_validation: Optional[List[str]] = None,
) -> Iterator[TensorDict]:
    """
    Build an iterator applying the mapping functions on the given dataset iterator.

    :param data_iter: iterator providing data samples in the form of a TensorDict
    :param map_seq: see :class:`PostprocessingDataset`
    :param map_seq_stream: see :class:`PostprocessingDataset`
    :param epoch: current epoch number
    :param out_tensor_dict_template: template for the output TensorDicts, used for validation
    :param rng: random number generator to use
    :param seq_list_for_validation: optional list of seq tags to validate against when processing the data
    :return: an iterator applying both the segment level and across-segment transformations on the given dataset
    """

    def _validate_tensor_dict_iter(inner: Iterator[TensorDict]) -> Iterator[TensorDict]:
        last_complete_frac = 0.0
        for t_dict in inner:
            assert isinstance(t_dict, TensorDict), (
                f"postprocessing mapper function must produce a {TensorDict.__name__}, "
                f"but got a {type(t_dict).__name__}"
            )
            if "complete_frac" in t_dict.data:  # sanity check complete_frac
                complete_frac = float(t_dict.data["complete_frac"].raw_tensor)
                assert 0.0 <= complete_frac <= 1.0, f"complete_frac must be in [0, 1], but got {complete_frac}"
                assert complete_frac >= last_complete_frac, (
                    "complete_frac must be monotonically increasing, "
                    f"but got {complete_frac} after {last_complete_frac}"
                )
                last_complete_frac = complete_frac
            for data_key, out_t in out_tensor_dict_template.data.items():
                in_t = t_dict.data[data_key]
                assert in_t.ndim == out_t.batch_ndim, (
                    f"Dim number mismatch for {data_key}: {in_t.ndim} != {out_t.batch_ndim}. "
                    "Postprocessing data tensors must not have a batch dimension."
                )
                assert in_t.dtype == out_t.dtype, f"dtype mismatch for {data_key}: '{in_t.dtype}' != '{out_t.dtype}'"
                for i, (in_dim, out_shape) in enumerate(zip(in_t.dims, out_t.shape)):
                    assert in_dim.dimension is None or in_dim.dimension == out_shape, (
                        f"Dim {i} mismatch on {data_key}: {in_dim.dimension} must either be `None` or equal {out_shape}"
                    )
            yield t_dict

    def _apply_map_seq(tensor_dict: TensorDict) -> TensorDict:
        comp_frac_raw_tensor = (
            tensor_dict.data["complete_frac"].raw_tensor if "complete_frac" in tensor_dict.data else None
        )
        seq_index_raw = tensor_dict.data["seq_idx"].raw_tensor
        seq_index = int(seq_index_raw.item())
        seq_tag_raw_tensor = tensor_dict.data["seq_tag"].raw_tensor

        tensor_dict = map_seq(tensor_dict, epoch=epoch, seq_idx=seq_index, rng=rng, **util.get_fwd_compat_kwargs())
        assert isinstance(tensor_dict, TensorDict), (
            f"map_seq must produce a {TensorDict.__name__}, but produced {type(tensor_dict).__name__}"
        )

        # Re-adding the complete_frac/seq_idx/seq_tag here causes no harm in case they are dropped
        # since we don't add/drop any segments w/ the non-iterator postprocessing function.
        if "complete_frac" not in tensor_dict.data and comp_frac_raw_tensor is not None:
            tensor_dict.data["complete_frac"] = Tensor(
                "complete_frac", dims=(), dtype="float32", raw_tensor=comp_frac_raw_tensor
            )
        if "seq_idx" not in tensor_dict.data:
            tensor_dict.data["seq_idx"] = Tensor("seq_idx", dims=(), dtype="int32", raw_tensor=seq_index_raw)
        if "seq_tag" not in tensor_dict.data:
            tensor_dict.data["seq_tag"] = Tensor("seq_tag", dims=(), dtype="string", raw_tensor=seq_tag_raw_tensor)

        if seq_list_for_validation is not None:
            seq_tag = seq_list_for_validation[seq_index]
            tag_of_seq = tensor_dict.data["seq_tag"].raw_tensor.item()
            assert tag_of_seq == seq_tag, (
                f"seq tag mismath: {tag_of_seq} != {seq_tag} for seq index {seq_index} when seq list is given"
            )

        return tensor_dict

    assert map_seq or map_seq_stream, "need to specify either map_seq or map_seq_stream"
    assert not (map_seq and map_seq_stream), "cannot set both map_seq and map_seq_stream"
    if map_seq is not None:
        data_iter = (_apply_map_seq(t_dict) for t_dict in data_iter)
    if map_seq_stream is not None:
        data_iter = map_seq_stream(data_iter, epoch=epoch, rng=rng, **util.get_fwd_compat_kwargs())
        assert isinstance(data_iter, Iterator), (
            f"map_seq_stream must produce an {Iterator.__name__}, but produced {type(data_iter).__name__}"
        )
    return _validate_tensor_dict_iter(data_iter)


class _MultiProcDataIter:
    """
    Data iter that pulls from the worker processes in a well-defined order and
    manages the lifetime of the feeder thread.

    Also ensures monotonicity of complete_frac, which would otherwise be no longer
    guaranteed if there is more than one worker.
    """

    def __init__(
        self, *, dataset_thread: threading.Thread, quit_event: threading.Event, worker_procs: List[_WorkerProcParent]
    ):
        self.dataset_thread = dataset_thread
        self.quit_event = quit_event
        assert len(worker_procs) > 0
        self.worker_procs = worker_procs

        self._complete_frac = 0.0  # need to force monotonicity
        self._workers_exhausted = [False for _ in range(len(worker_procs))]
        self._worker_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> Optional[TensorDict]:
        if self.quit_event.is_set():
            raise StopIteration

        while True:
            if all(self._workers_exhausted):
                break
            worker_idx = self._worker_idx
            self._worker_idx = (self._worker_idx + 1) % len(self.worker_procs)
            if self._workers_exhausted[worker_idx]:
                continue
            seq = self.worker_procs[worker_idx].get_seq()
            if seq is not None:
                return self._ensure_complete_frac_monotonic(seq)
            self._workers_exhausted[worker_idx] = True

        # when we reach this point, all workers are exhausted and we stop
        self.stop()
        raise StopIteration

    def stop(self, *, join=True):
        """
        Stop the iterator and the dataset thread.

        Once this is called, the iterator cannot be used anymore.
        """
        if self.quit_event.is_set():
            return
        self.quit_event.set()
        if join:
            util.try_run(self.dataset_thread.join)

    def _ensure_complete_frac_monotonic(self, seq: TensorDict) -> TensorDict:
        """
        Enforce monotonicity of `complete_frac` in the given `TensorDict`.
        """
        if "complete_frac" not in seq.data:
            return seq
        complete_frac = float(seq.data["complete_frac"].raw_tensor)
        assert 0.0 <= complete_frac <= 1.0, f"complete_frac must be in [0, 1], but got {complete_frac}"
        self._complete_frac = max(complete_frac, self._complete_frac)
        seq.data["complete_frac"].raw_tensor = numpy.array(self._complete_frac, dtype=numpy.float32)
        return seq

    def __del__(self):
        # noinspection PyBroadException
        try:
            self.stop(join=False)
        except Exception:
            pass


class _WorkerProcParent:
    def __init__(
        self,
        *,
        buffer_size: int,
        index: int,
        name: str,
        map_seq: Optional[Callable],
        map_seq_stream: Optional[Callable],
        out_tensor_dict_template: TensorDict,
    ):
        parent_conn, child_conn = _mp.Pipe()
        self.parent_conn = parent_conn

        self.worker_proc = _mp.Process(
            name=f"{name} worker {index}",
            target=_worker_proc_loop,
            args=(index, child_conn, buffer_size, map_seq, map_seq_stream, out_tensor_dict_template),
            daemon=True,
        )
        self.worker_proc.start()

        # Make sure the child connection is closed here.
        # It stays open in the child, until the child dies.
        # When that happens, now any consecutive read on the pipe
        # should yield an exception -- which is what we want,
        # otherwise it would just hang.
        child_conn.close()

    def init_seq_order(
        self,
        *,
        epoch: int,
        rng_seed: int,
        seq_list: Optional[List[str]],
        seq_pipe: mpConnection,
    ):
        """init_seq_order"""
        args = {"epoch": epoch, "rng_seed": rng_seed, "seq_list": seq_list, "seq_pipe": seq_pipe}
        self.parent_conn.send(("init_seq_order", args))
        msg, _ = self.parent_conn.recv()
        assert msg == "init_seq_order"
        # seq_pipe is owned by the child process,
        # and so must be closed in the parent to avoid hangs
        seq_pipe.close()

    def get_seq(self) -> Optional[TensorDict]:
        """get_seq"""
        self.parent_conn.send(("get_seq", {}))
        msg, seq = self.parent_conn.recv()
        assert msg == "seq"
        return seq

    def exit(self, *, join: bool = True):
        """exit"""
        self.parent_conn.send(("exit", {}))
        if join:
            self.worker_proc.join()

    def __del__(self):
        # noinspection PyBroadException
        try:
            self.exit(join=False)
        except Exception:
            pass
        else:
            util.try_run(self.worker_proc.join)


def _worker_proc_loop(
    index: int,
    parent_conn: mpConnection,
    buffer_size: int,
    map_seq: Optional[Callable],
    map_seq_stream: Optional[Callable],
    out_tensor_dict_template: TensorDict,
):
    if sys.platform == "linux":
        with open("/proc/self/comm", "w") as f:
            f.write(f"PP worker {index}")
    better_exchook.setup_all()

    assert isinstance(buffer_size, int) and buffer_size > 0
    assert isinstance(index, int)
    assert isinstance(parent_conn, mpConnection)

    cache: deque[TensorDict] = deque()

    data_iter: Optional[Iterator[TensorDict]] = None
    feeder_conn: Optional[mpConnection] = None

    def _add_to_cache():
        nonlocal data_iter
        if data_iter is None:
            return False
        try:
            seq = next(data_iter)
        except StopIteration:
            data_iter = None
            return False
        cache.append(seq)
        return True

    def _iter_pipe(q: mpConnection) -> Iterator[TensorDict]:
        assert isinstance(q, mpConnection)

        while True:
            try:
                q.send(("get_seq", None))
                seq_msg, item = q.recv()
            except (BrokenPipeError, EOFError):
                # queue is closed
                break
            assert seq_msg == "seq"
            if item is None:
                break
            assert isinstance(item, TensorDict)
            yield item

    try:
        while True:
            while len(cache) < buffer_size and not parent_conn.poll():
                if not _add_to_cache():
                    break
            msg, kwargs = parent_conn.recv()
            if msg == "exit":
                break
            elif msg == "get_seq":
                if not cache:
                    _add_to_cache()
                parent_conn.send(("seq", cache.popleft() if cache else None))
            elif msg == "init_seq_order":
                epoch = kwargs["epoch"]
                if sys.platform == "linux":
                    with open("/proc/self/comm", "w") as f:
                        f.write(f"PP worker {index} ep {epoch}")
                if feeder_conn is not None:
                    feeder_conn.close()
                feeder_conn = kwargs["seq_pipe"]
                data_iter = _build_mapping_iter(
                    _iter_pipe(feeder_conn),
                    epoch=epoch,
                    map_seq=map_seq,
                    map_seq_stream=map_seq_stream,
                    out_tensor_dict_template=out_tensor_dict_template,
                    rng=RandomState(kwargs["rng_seed"]),
                    seq_list_for_validation=kwargs["seq_list"],
                )
                assert isinstance(data_iter, Iterator)
                cache.clear()
                parent_conn.send(("init_seq_order", None))
            else:
                raise Exception(f"unknown msg {msg!r}")
    except KeyboardInterrupt:  # when parent dies
        pass
    except EOFError:  # when parent dies
        pass
    finally:
        if feeder_conn is not None:
            feeder_conn.close()
        parent_conn.close()


class LaplaceOrdering(Callable[[Iterator[TensorDict]], Iterator[TensorDict]]):
    """
    Iterator compatible with :class:`PostprocessingDataset`'s ``map_seq_stream`` applying
    laplace sequence ordering based on the number of segments per bin.

    To be composed with any custom data postprocessing logic via :class:`Sequential`.
    """

    preserves_num_seqs = True

    def __init__(self, num_seqs_per_bin: int, length_key: str = "data"):
        """
        :param num_seqs_per_bin: number of segments in a single laplace bin.
        :param length_key: data key to determine the segment length from for ordering.
        """
        self.length_key = length_key
        assert num_seqs_per_bin > 0
        self.num_seqs_per_bin = num_seqs_per_bin

    def __call__(self, iterator: Iterator[TensorDict], **kwargs) -> Iterator[TensorDict]:
        """:return: generator applying laplace sequence ordering on the data"""
        iterator = iter(iterator)
        is_down_phase = False

        seq_buffer = list(islice(iterator, self.num_seqs_per_bin))
        has_ended = False
        while True:
            # Make sure to not reorder the monotonically increasing values for complete_frac
            # so that the trainer can calculate the appropriate learning rates.
            complete_frac_values = [tdict.data["complete_frac"].raw_tensor for tdict in seq_buffer]
            seq_buffer.sort(key=self._get_seq_len, reverse=is_down_phase)
            for sorted_item, comp_frac in zip(seq_buffer, complete_frac_values):
                sorted_item.data["complete_frac"].raw_tensor = comp_frac

            next_seq_buffer = []

            # Yield items to trainer while gradually pulling more data from PP function.
            # This optimizes CPU load when multiple workers are used.
            for item in seq_buffer:
                yield item

                try:
                    if not has_ended:
                        next_seq_buffer.append(next(iterator))
                except StopIteration:
                    has_ended = True

            if len(seq_buffer) < self.num_seqs_per_bin:
                assert has_ended and not next_seq_buffer
                break

            is_down_phase = not is_down_phase
            seq_buffer = next_seq_buffer

    def _get_seq_len(self, tdict: TensorDict) -> int:
        """
        :return: segment length of the segment in `tdict` as measured by `self.length_key` for comparison.
        """
        return tdict.data[self.length_key].raw_tensor.shape[0]


T = TypeVar("T", TensorDict, Iterator[TensorDict])


class Sequential:
    """
    Callable that composes multiple postprocessing functions into one by sequential application,
    i.e. Sequential(f, g)(x) = (g ∘ f)(x) = g(f(x)).

    Can either compose ``map_seq``-style single-segment processor functions or ``map_seq_stream``-style
    iterators operating on multiple segments. Just make sure not to mix both styles.
    """

    def __init__(self, *postprocessing_funcs: Callable):
        """
        :param postprocessing_funcs: Postprocessing functions to compose.
        """
        self.funcs = postprocessing_funcs

    def __call__(self, arg: T, **kwargs) -> T:
        """:return: result of sequential application of the postprocessing functions"""

        for func in self.funcs:
            arg = func(arg, **kwargs)
        return arg

    @property
    def preserves_num_seqs(self):
        """:return: whether the composed functions all preserve the number of sequences"""
        return all(getattr(f, "preserves_num_seqs", False) for f in self.funcs)
