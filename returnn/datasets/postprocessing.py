"""
Provides :class:`PostprocessingDataset`.
"""

from __future__ import annotations

from collections import deque
from itertools import islice
import numpy
from numpy.random import RandomState
import sys
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar

from returnn.datasets.basic import DatasetSeq
from returnn.datasets.util.strings import str_to_numpy_array
from returnn.datasets.util.vocabulary import Vocabulary
from returnn.tensor import Tensor, TensorDict
from returnn.tensor.dim import Dim
from returnn.util import basic as util, better_exchook
from returnn.util.multi_proc_non_daemonic_spawn import NonDaemonicSpawnContext
from returnn.config import SubProcCopyGlobalConfigPreInitFunc
from .basic import init_dataset
from .cached2 import CachedDataset2

# noinspection PyProtectedMember
from multiprocessing.connection import Connection as mpConnection

# noinspection PyProtectedMember
from multiprocessing.queues import Queue as mpQueue

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

    Multiprocessing can either be done using ``MultiProcDataset`` or by specifying
    the ``num_workers`` parameter of this class.
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
        :param buf_size: buffer size for each worker, number of seqs to prefetch.
            Must be > 0. Only relevant when num_workers > 0.
        :param num_workers: number of worker processes to use for data postprocessing.

            This does not apply parallelism to the wrapped dataset, but only to the postprocessing step.

            Conceptually, this achieves similar results to using MultiProcDataset, but with potentially lower
            memory consumption, since only the postprocessing step is parallelized and not the wrapped/underlying
            source dataset that is postprocessed.

            When set to 0, postprocessing happens inline.
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
            raise ValueError(f"{self}: buf_size must be >= 1, but got {buf_size}")
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
        # Ensures only one feeder thread at a time accesses the wrapped dataset. Only used if num_workers > 0.
        self._dataset_lock: Optional[threading.Lock] = None
        self._multi_proc_data_iter: Optional[_MultiProcDataIter] = None
        self._num_workers = num_workers

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

        if epoch is None and seq_list is None and seq_order is None:
            self._num_seqs = 0
            return True

        if self._num_workers > 0:
            assert self._buf_size > 0
            if self._dataset_lock is None:
                self._dataset_lock = threading.Lock()
            seq_queues = [_mp.Queue(maxsize=self._buf_size) for _ in range(self._num_workers)]
            worker_procs = [
                _WorkerProcParent(
                    name=f"{self.__class__.__name__} {self.name} ep {epoch}",
                    epoch=epoch,
                    buffer_size=self._buf_size,
                    map_seq=self._map_seq,
                    map_seq_stream=self._map_seq_stream,
                    out_tensor_dict_template=self._out_tensor_dict_template,
                    rng_seed=self._get_random_seed_for_epoch(epoch=epoch * self._num_workers + i),
                    seq_list=seq_list,
                    seq_queue=seq_queue,
                )
                for i, seq_queue in enumerate(seq_queues)
            ]
            quit_event = threading.Event()
            # dataset thread takes care of init_seq_order of the wrapped dataset
            dataset_thread = threading.Thread(
                target=self._init_and_distribute_seqs_to_children,
                kwargs={
                    "child_queues": seq_queues,
                    "dataset_lock": self._dataset_lock,
                    "epoch": epoch,
                    "quit_event": quit_event,
                    "seq_list": seq_list,
                    "seq_order": seq_order,
                },
                name=f"{self.__class__.__name__} {self.name} dataset ep {epoch}",
            )
            dataset_thread.start()
            if self._multi_proc_data_iter is not None:
                self._multi_proc_data_iter.stop()
            data_iter = self._multi_proc_data_iter = _MultiProcDataIter(
                dataset_thread=dataset_thread, quit_event=quit_event, worker_procs=worker_procs
            )
        else:
            assert self._dataset is not None
            self._dataset.init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
            data_iter = _build_mapping_iter(
                self._iterate_dataset(),
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

    def _iterate_dataset(self) -> Iterator[TensorDict]:
        """
        :return: generator providing data samples in the form of a TensorDict
        """
        data_keys = self._dataset.get_data_keys()

        seq_index = 0
        while self._dataset.is_less_than_num_seqs(seq_index):
            self._dataset.load_seqs(seq_index, seq_index + 1)

            tensor_dict = self._in_tensor_dict_template.copy_template()
            for data_key in data_keys:
                tensor_dict.data[data_key].raw_tensor = self._dataset.get_data(seq_index, data_key)

            complete_frac = self._dataset.get_complete_frac(seq_index, allow_only_lr_suitable=True)
            if complete_frac is not None:
                comp_frac_raw_tensor = numpy.array(complete_frac, dtype=numpy.float32)
                tensor_dict.data["complete_frac"].raw_tensor = comp_frac_raw_tensor
            seq_idx_raw_tensor = numpy.array(seq_index, dtype=numpy.int32)
            tensor_dict.data["seq_idx"].raw_tensor = seq_idx_raw_tensor
            seq_tag_raw_tensor = str_to_numpy_array(self._dataset.get_tag(seq_index))
            tensor_dict.data["seq_tag"].raw_tensor = seq_tag_raw_tensor

            yield tensor_dict
            seq_index += 1

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

    def _init_and_distribute_seqs_to_children(
        self,
        *,
        child_queues: Sequence[mpQueue],
        dataset_lock: threading.Lock,
        epoch: int,
        quit_event: threading.Event,
        seq_list: Optional[List[str]] = None,
        seq_order: Optional[List[int]] = None,
    ):
        assert len(child_queues) > 0

        # Lock ensures that only one thread at a time accesses the wrapped dataset.
        #
        # This protects against issues while moving from one epoch to the next.
        with dataset_lock:
            assert self._dataset is not None
            self._dataset.init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)

            for seq_idx, tensor_dict in enumerate(self._iterate_dataset()):
                if quit_event.is_set():
                    break
                try:
                    child_queues[seq_idx % len(child_queues)].put(tensor_dict, block=True)  # block for backpressure
                except ValueError:
                    # queue is closed, i.e. the worker process crashed for some reason -> stop
                    break
            for q in child_queues:
                try:
                    q.put(None, block=True)  # signal end of data
                except ValueError:
                    # queue is already closed, i.e. the worker process died
                    pass


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
    Data iter that pulls from the worker processes and manages their lifetime.
    """

    def __init__(
        self,
        *,
        dataset_thread: threading.Thread,
        quit_event: threading.Event,
        worker_procs: List[_WorkerProcParent],
    ):
        self.dataset_thread = dataset_thread
        self.quit_event = quit_event
        assert len(worker_procs) > 0
        self.worker_procs = worker_procs

        self._workers_exhausted = [False for _ in range(len(worker_procs))]
        self._worker_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> Optional[TensorDict]:
        if self.quit_event.is_set():
            raise StopIteration

        while True:
            seq = self.worker_procs[self._worker_idx].get_seq()
            self._worker_idx = (self._worker_idx + 1) % len(self.worker_procs)
            if seq is None:
                self._workers_exhausted[self._worker_idx] = True
                if all(self._workers_exhausted):
                    break
                else:
                    continue
            return seq

        # when we reach this point, all workers are exhausted and we stop
        self.stop()
        raise StopIteration

    def stop(self, *, join=True):
        """Stop the worker processes and the dataset thread."""
        if self.quit_event.is_set():
            return
        self.quit_event.set()
        for wp in self.worker_procs:
            wp.exit(join=join)
        if join:
            util.try_run(self.dataset_thread.join)

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
        name: str,
        epoch: int,
        buffer_size: int,
        map_seq: Optional[Callable],
        map_seq_stream: Optional[Callable],
        out_tensor_dict_template: TensorDict,
        rng_seed: int,
        seq_list: Optional[List[str]],
        seq_queue: mpQueue,
    ):
        parent_conn, child_conn = _mp.Pipe()
        self.parent_conn = parent_conn

        self.worker_proc = _mp.Process(
            name=f"{name} worker ep {epoch}",
            target=_worker_proc_loop,
            args=(
                epoch,
                buffer_size,
                map_seq,
                map_seq_stream,
                out_tensor_dict_template,
                rng_seed,
                seq_list,
                child_conn,
                seq_queue,
            ),
            daemon=True,
        )
        self.worker_proc.start()

        # Make sure the child connection is closed here.
        # It stays open in the child, until the child dies.
        # When that happens, now any consecutive read on the pipe
        # should yield an exception -- which is what we want,
        # otherwise it would just hang.
        child_conn.close()

    def get_seq(self) -> Optional[DatasetSeq]:
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
    epoch: int,
    buffer_size: int,
    map_seq: Optional[Callable],
    map_seq_stream: Optional[Callable],
    out_tensor_dict_template: TensorDict,
    rng_seed: int,
    seq_list: Optional[List[str]],
    parent_conn: mpConnection,
    seq_queue: mpQueue,
):
    if sys.platform == "linux":
        with open("/proc/self/comm", "w") as f:
            f.write(f"PP worker {epoch}")
    better_exchook.setup_all()

    assert isinstance(epoch, int)
    assert isinstance(buffer_size, int)
    assert buffer_size > 0
    assert map_seq or map_seq_stream, "need to specify either map_seq or map_seq_stream"
    assert not (map_seq and map_seq_stream), "cannot set both map_seq and map_seq_stream"
    assert map_seq is None or isinstance(map_seq, Callable)
    assert map_seq_stream is None or isinstance(map_seq_stream, Callable)
    assert isinstance(out_tensor_dict_template, TensorDict)
    assert isinstance(rng_seed, int)
    assert isinstance(parent_conn, mpConnection)
    assert isinstance(seq_queue, mpQueue)

    cache: deque[TensorDict] = deque()

    def _iter_queue(q: mpQueue) -> Iterator[TensorDict]:
        while True:
            try:
                item = q.get(block=True)
            except ValueError:
                # queue is closed
                break
            if item is None:
                break
            yield item

    data_iter = _build_mapping_iter(
        _iter_queue(seq_queue),
        map_seq=map_seq,
        map_seq_stream=map_seq_stream,
        epoch=epoch,
        out_tensor_dict_template=out_tensor_dict_template,
        rng=RandomState(rng_seed),
        seq_list_for_validation=seq_list,
    )
    assert isinstance(data_iter, Iterator)

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

    try:
        while True:
            while not parent_conn.poll():
                while len(cache) < buffer_size:
                    if not _add_to_cache():
                        break
            msg, kwargs = parent_conn.recv()
            if msg == "exit":
                break
            elif msg == "get_seq":
                if not cache:
                    _add_to_cache()
                parent_conn.send(("seq", cache.popleft() if cache else None))
            else:
                raise Exception(f"unknown msg {msg!r}")
    except KeyboardInterrupt:  # when parent dies
        pass
    except EOFError:  # when parent dies
        pass


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
