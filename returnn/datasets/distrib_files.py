"""
:class:`DistributeFilesDataset`

https://github.com/rwth-i6/returnn/issues/1519
"""

from __future__ import annotations

from typing import Union, Optional, Any, Callable, Sequence, Tuple, List, Dict
import os
import sys
import numpy
from returnn.log import log
from returnn.util.basic import override_env_var, try_run
from returnn.util.multi_proc_non_daemonic_spawn import NonDaemonicSpawnContext
from returnn.config import SubProcCopyGlobalConfigPreInitFunc
from .basic import init_dataset, extend_dataset_dict_from_parent_dataset, DatasetSeq, RANDOM_SEED_OFFSET_ENV_VAR
from .cached2 import CachedDataset2

# noinspection PyProtectedMember
from multiprocessing.connection import Connection as mpConnection

_mp = NonDaemonicSpawnContext(process_pre_init_func=SubProcCopyGlobalConfigPreInitFunc())


__all__ = ["DistributeFilesDataset"]

Filename = str
FileTree = Union[Filename, Tuple["FileTree", ...], Dict[Any, "FileTree"], List["FileTree"]]


class DistributeFilesDataset(CachedDataset2):
    """
    Dataset that distributes files over subepochs and then creates a
    sub dataset for every sub epoch for a given (random) subset of the files.
    The sub dataset is user-defined via a function ``get_sub_epoch_dataset``.
    Thus, this dataset wraps the sub datasets.

    It is conceptually very similar to :class:`ConcatDataset` in the sense
    that it concatenates all the sub datasets together to form one larger dataset.

    This scheme allows to shuffle over the files,
    which makes shuffling much more efficient over a large dataset
    at the cost of no longer shuffling over the full dataset in every subepoch.
    Instead, the quality of the shuffle depends on the number of files the dataset is
    split into -- the more files per subepoch, the better.

    Additionally, this scheme allows to prefetch and cache the upcoming needed files,
    e.g. copying them from a NFS to local disk,
    when the local disk can not store the whole dataset,
    and/or when the local disk would allow for faster access.
    For that, the user can use :class:`returnn.util.file_cache.CachedFile`
    inside the returned dict from ``get_sub_epoch_dataset``.

    It was also designed with multi-GPU training in mind,
    where each GPU should get a set of files to work on,
    and the sub epochs across the GPUs would not be exactly of the same length.
    Whenever one GPU finishes its sub epoch, all remaining data on the other GPUs is dropped.
    First, our scheme tries to make the length of the sub epochs as equally distributed as possible
    after random shuffling by using the size of the files as proxy for the length of the contents.
    Further, due to the random shuffling, there should not be any bias in the data distribution.
    Specifically, we don't want that some data might be visited more often than others
    (at least its expected value should be the same).

    In case the dataset grows so large it is unreasonable to expect one worker to
    ever see all the data, this dataset can also shard the file list on a per-worker
    basis before distributing across subepochs.
    This behavior can be configured by setting the property ``"distrib_shard_files": True``.
    The dataset attempts to split the files as evenly as possible based on the file size.

    Example usage::

        def get_sub_epoch_dataset(files_subepoch: List[str]) -> Dict[str, Any]:
          from returnn.util.file_cache import CachedFile
          return {
            "class": "HDFDataset",
            "files": [CachedFile(fn) for fn in files_subepoch],
          }

        train = {
          "class": "DistributeFilesDataset",
          "files": [
            "/nfs/big_data_1.hdf",
            ...
          ],  # M files
          "get_sub_epoch_dataset": get_sub_epoch_dataset,
          "partition_epoch": P,  # P << M
        }

    Instead of a plain list of strings, you can also provide a list of any nested structures to keep
    multimodal data together::

        def get_sub_epoch_dataset(files_subepoch: List[Tuple[str, str]]) -> Dict[str, Any]:
          from returnn.util.file_cache import CachedFile

          alignments, features = tuple(zip(*files_subepoch)) # transpose

          return {
            "class": "MetaDataset",
            "data_map": {"classes": ("alignments", "data"), "data": ("features", "data")},
            "datasets": {
              "alignments": {
                "class": "HDFDataset",
                "files": alignments,
                "seq_ordering": "random",
              },
              "features": {
                "class": "HDFDataset",
                "files": features,
              },
            },
            "seq_order_control_dataset": "alignments",
          }

        train = {
          "class": "DistributeFilesDataset",
          "files": [
            ("/nfs/alignment_1.hdf", "/nfs/features_1.hdf"),
            ...
          ],  # M entries
          "get_sub_epoch_dataset": get_sub_epoch_dataset,
          "partition_epoch": P,  # P << M
        }

    In this case the file sizes for sub epoch distribution are summed up per list entry
    by iterating over the structure leaves.

    For some discussion, see https://github.com/rwth-i6/returnn/issues/1519 and
    https://github.com/rwth-i6/returnn/issues/1524.
    """

    def __init__(
        self,
        *,
        files: List[FileTree],
        get_sub_epoch_dataset: Callable[[List[FileTree]], Dict[str, Any]],
        preload_next_n_sub_epochs: int = 1,
        buffer_size: int = 1,
        distrib_shard_files: bool = False,
        _meta_info_cache: Optional[Dict[str, Any]] = None,
        _distrib_info: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        """
        :param files: the files to shuffle over, can also be a list of arbitrarily nested python objects
            to keep associated heterogeneous data together
        :param get_sub_epoch_dataset: callable which returns a dataset dict for a given subset of files
        :param preload_next_n_sub_epochs: how many sub epoch datasets to preload
        :param buffer_size: buffer size for each worker, amount of seqs to prefetch
        :param distrib_shard_files: set to true to shard the data across worker processes in
            distributed training scenaria
        :param _meta_info_cache: for internal use
        :param _distrib_info: for internal use
        """
        super().__init__(**kwargs)
        self.files = files
        self.get_sub_epoch_dataset = get_sub_epoch_dataset
        assert preload_next_n_sub_epochs >= 0
        self.preload_next_n_sub_epochs = preload_next_n_sub_epochs
        self.buffer_size = buffer_size
        self._file_sizes: Optional[Dict[str, int]] = None  # key -> size. for equal distribution across sub epochs
        self._data_keys: Optional[List[str]] = None
        self._num_seqs: Optional[int] = None

        self._workers: Dict[int, _WorkerProcParent] = {}  # epoch -> worker
        self._files_order_cache: Dict[int, List[List[FileTree]]] = {}  # full epoch (0-indexed) -> files order

        self.distrib_shard_files = distrib_shard_files
        if distrib_shard_files:
            if _distrib_info:
                # If we're in a child process `_get_rank_and_size()` no longer works,
                # so we pass the info about the shards via a pickled property.
                # See also Dataset.__reduce__.
                self._shard_index = _distrib_info["shard_index"]
                self._num_shards = _distrib_info["num_shards"]
            else:
                self._shard_index, self._num_shards = _get_rank_and_size()
        else:
            self._shard_index = 0
            self._num_shards = 1

        if _meta_info_cache:
            # This allows to skip the lazy init in self.initialize().
            # This can be used when pickling/deepcopying an instance of this dataset,
            # see Dataset.__reduce__.
            self.num_inputs = _meta_info_cache["num_inputs"]
            self.num_outputs = _meta_info_cache["num_outputs"]
            self.labels = _meta_info_cache["labels"]
            self._data_keys = _meta_info_cache["data_keys"]
            self._file_sizes = _meta_info_cache["file_sizes"]

        if len(files) < self.partition_epoch:
            raise ValueError(f"{self}: len(files) {len(files)} < partition_epoch {self.partition_epoch}")

    def initialize(self):
        """init"""
        self._lazy_init_num_outputs()
        super().initialize()

    @property
    def _distrib_info(self):
        return {"num_shards": self._num_shards, "shard_index": self._shard_index}

    @property
    def _meta_info_cache(self):
        if not self.num_outputs:
            return None
        return {
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "labels": self.labels,
            "data_keys": self._data_keys,
            "file_sizes": self._file_sizes,
        }

    def _uses_custom_distributed_sharding(self) -> bool:
        return self._num_shards > 1

    def _lazy_init_num_outputs(self):
        if self.num_outputs:
            return
        # First, we need to know the num_inputs, num_outputs, total_num_seqs, labels.
        # Init the dataset with the first file.
        dataset_dict = self._get_sub_dataset_dict(files=[self.files[0]])
        dataset = init_dataset(dataset_dict, extra_kwargs={"seq_ordering": "default"}, parent_dataset=self)
        self.num_inputs = dataset.num_inputs
        self.num_outputs = dataset.num_outputs
        self.labels = dataset.labels
        self._data_keys = dataset.get_data_keys()

    def _lazy_init_file_sizes(self):
        import tree

        if self._file_sizes:
            return
        self._file_sizes = {
            _get_key_for_file_tree(t): sum((os.path.getsize(fn) for fn in tree.flatten(t)), 0) for t in self.files
        }

    def __del__(self):
        if hasattr(self, "_workers"):  # might not be set after an early exception in __init__
            for k, worker in self._workers.items():
                try_run(worker.exit, kwargs={"join": False})
            self._workers.clear()

    def init_seq_order(self, epoch: Optional[int] = None, seq_list=None, seq_order=None) -> bool:
        """
        :param epoch:
        :param seq_list:
        :param seq_order:
        :return: whether the order changed (True is always safe to return)
        """
        super().init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
        if seq_list is not None:
            raise Exception(f"{self}: seq_list not supported")
        if seq_order is not None:
            raise Exception(f"{self}: seq_order not supported")
        if epoch is None:
            self._num_seqs = 0
            return True

        self._lazy_init_file_sizes()

        full_epoch_0idx = (epoch - 1) // self.partition_epoch

        # Cleanup and fill _files_order_cache, also shard files across GPU workers
        for k in list(self._files_order_cache.keys()):
            if k < full_epoch_0idx:
                del self._files_order_cache[k]
        for ep_ in range(epoch, epoch + self.preload_next_n_sub_epochs + 1):
            full_epoch_0idx_ = (ep_ - 1) // self.partition_epoch
            if full_epoch_0idx_ in self._files_order_cache:
                continue
            if self.seq_ordering == "default":
                files_order_flat = self.files
            elif self.seq_ordering == "random":
                # when sharding, _get_random_seed_for_epoch makes sure to use a fixed
                # random_seed_offset
                rnd_seed = self._get_random_seed_for_epoch(full_epoch_0idx_ * self.partition_epoch + 1)
                random_generator = numpy.random.RandomState(rnd_seed)
                files_order_flat = list(self.files)
                random_generator.shuffle(files_order_flat)
            else:
                raise ValueError(f"{self}: seq_ordering {self.seq_ordering!r} not supported")
            file_bins = self._distribute_evenly_by_size(
                num_bins=self._num_shards * self.partition_epoch,
                file_sizes=self._file_sizes,
                files_order=files_order_flat,
            )
            self_index_base = self.partition_epoch * self._shard_index
            self_index_end = self_index_base + self.partition_epoch
            self._files_order_cache[full_epoch_0idx_] = file_bins[self_index_base:self_index_end]

        # Cleanup and fill _workers.
        for k, worker in list(self._workers.items()):
            if k < epoch:
                worker.exit()
                del self._workers[k]
        for ep_ in range(epoch, epoch + self.preload_next_n_sub_epochs + 1):
            if ep_ in self._workers:
                continue
            full_epoch_0idx_ = (ep_ - 1) // self.partition_epoch
            files_order: List[List[FileTree]] = self._files_order_cache[full_epoch_0idx_]
            files_for_subep = files_order[(ep_ - 1) % self.partition_epoch]
            print(f"{self}: using files for epoch {ep_}: {files_for_subep}", file=log.v4)
            dataset_dict = self._get_sub_dataset_dict(files=files_for_subep)
            worker = _WorkerProcParent(
                name=f"{self.__class__.__name__} {self.name} ep {epoch}",
                epoch=ep_,
                full_epoch_0idx=full_epoch_0idx_,
                dataset_dict=dataset_dict,
                buffer_size=self.buffer_size,
            )
            self._workers[ep_] = worker

        self._num_seqs = self._workers[epoch].get_num_seqs()
        return True

    def _get_sub_dataset_dict(self, files: List[FileTree]) -> Dict[str, Any]:
        import tree

        dataset_dict = self.get_sub_epoch_dataset(files)
        dataset_dict = extend_dataset_dict_from_parent_dataset(dataset_dict, parent_dataset=self)

        flat_sub_dset = tree.flatten_with_path(dataset_dict)

        part_epoch_cfg = next(
            ((path, v) for path, v in flat_sub_dset if path[-1] == "partition_epoch" and v != 1), None
        )
        if part_epoch_cfg is not None:
            path, subeps = part_epoch_cfg
            raise ValueError(
                f"{self}: sub dataset should not have partition_epoch, "
                f'but got "partition_epoch": {subeps} at {".".join(path)} in {dataset_dict}.'
            )

        # Heuristic check for well-definedness of seq ordering. Might need to be extended in the
        # future if there are other ways of defining a seq order than the ones below.
        if (
            not any(path[-1] == "seq_ordering" for path, _ in flat_sub_dset)
            and not any(path[-1] == "seq_order_control_dataset" for path, _ in flat_sub_dset)
            and not any(path[-1] == "map_seq_stream" for path, _ in flat_sub_dset)
        ):
            raise ValueError(
                f"{self}: there should be an explicit seq_ordering somewhere in the sub dataset "
                f"(or seq_order_control_dataset for MetaDataset or map_seq_stream for PostprocessingDataset), "
                f"but found none in {dataset_dict}."
            )

        return dataset_dict

    @staticmethod
    def _distribute_evenly_by_size(
        *, num_bins: int, file_sizes: Dict[str, int], files_order: Sequence[FileTree]
    ) -> List[List[FileTree]]:
        """
        Distributes the files from files_order into ``num_bins`` while attempting
        to make every bin as evenly sized (based on ``file_sizes``) as possible.
        """

        total_size = sum(file_sizes.values())
        avg_size_per_sub_epoch = total_size / num_bins
        # Now evenly distribute the files over the bins.
        # Note that many one-pass variants of algorithms to evenly distribute
        # can end up with some empty bins,
        # so we need to make sure that this is not the case.
        # E.g. consider the seqs of size [1,1,78,120] and num_bins=4.
        # That has avg size per sub epoch 50.
        # Some simple algorithms could end up with the sub epochs
        # [[1,1], [78], [120], []] or [[1,1,78], [120], [], []].
        # Or consider [5,5]+[10]*7, num_bins=5, which has avg size 16.
        # A simple algorithm could end up with [[5,5,10], [10,10], [10,10], [10,10], []].
        # See test_DistributeFilesDataset_distribute_evenly_by_size for some test cases.
        assert len(files_order) >= num_bins
        files_per_bin = [[] for _ in range(num_bins)]
        assert len(files_per_bin) == num_bins
        bin_idx = 0
        size_taken = 0
        total_size_taken = 0
        for i, f_tree in enumerate(files_order):
            size = file_sizes[_get_key_for_file_tree(f_tree)]
            num_remaining = len(files_order) - i
            total_size_taken += size

            if num_remaining <= num_bins - bin_idx - 1:
                # All remaining sub epochs must be filled.
                assert files_per_bin[bin_idx]
                bin_idx += 1
                avg_size_per_sub_epoch = (total_size - total_size_taken) / (num_bins - bin_idx)
                files_per_bin[bin_idx].append(f_tree)
                size_taken = size
                continue
            if bin_idx == num_bins - 1:
                # We are done. Just add the rest to the last sub epoch.
                files_per_bin[bin_idx].append(f_tree)
                size_taken += size
                continue
            if size_taken + size <= avg_size_per_sub_epoch:
                files_per_bin[bin_idx].append(f_tree)
                size_taken += size
                continue
            # We should increase the sub epoch index.
            # We need to decide where to add this file, to the current or the next sub epoch.
            if not files_per_bin[bin_idx] or (
                # Better to add this file to the current sub epoch?
                abs((size_taken + size) - avg_size_per_sub_epoch)
                <= abs(size_taken - avg_size_per_sub_epoch)
            ):
                files_per_bin[bin_idx].append(f_tree)
                size_taken = 0
            else:
                files_per_bin[bin_idx + 1].append(f_tree)
                size_taken = size
            bin_idx += 1
            avg_size_per_sub_epoch = (total_size - total_size_taken) / (num_bins - bin_idx)
        assert all(files_per_bin)
        return files_per_bin

    def _collect_single_seq(self, seq_idx: int) -> Optional[DatasetSeq]:
        seq = self._workers[self.epoch].get_data_seq(seq_idx)
        if not seq:
            return None
        seq.seq_idx = seq_idx
        return seq

    def have_seqs(self) -> bool:
        """have seqs"""
        return bool(self.files)

    def finish_epoch(self, *, free_resources: bool = False):
        """finish epoch"""
        super().finish_epoch(free_resources=free_resources)
        if free_resources:
            for worker in self._workers.values():
                worker.exit()
            self._workers.clear()
            self._files_order_cache.clear()
        else:
            if self.epoch in self._workers:
                worker = self._workers.pop(self.epoch)
                worker.exit()

    def get_data_keys(self) -> List[str]:
        """data keys"""
        if self._data_keys is None:
            self._lazy_init_num_outputs()
        return self._data_keys


def _get_key_for_file_tree(t: FileTree) -> str:
    """generates a deterministic key given a file tree"""
    import tree

    return ":".join(tree.flatten(t))


def _get_rank_and_size() -> Tuple[int, int]:
    """
    :return: tuple (rank, size): the global rank and size for distributed trainings
    """

    from returnn.config import get_global_config

    config = get_global_config(raise_exception=False)
    if not config:
        return 0, 1
    if config.typed_value("torch_distributed") is not None:
        import returnn.torch.distributed

        ctx = returnn.torch.distributed.get_ctx(config=config)
        return ctx.rank(), ctx.size()
    elif config.is_true("use_horovod"):
        assert config.bool("use_tensorflow", False) or config.value("backend", "").startswith("tensorflow")

        import returnn.tf.horovod

        ctx = returnn.tf.horovod.get_ctx(config=config)
        return ctx.rank(), ctx.size()
    else:
        return 0, 1


class _WorkerProcParent:
    def __init__(
        self,
        *,
        name: str,
        epoch: int,
        full_epoch_0idx: int,
        dataset_dict: Dict[str, Any],
        buffer_size: int,
    ):
        # the dataset makes sure this is set
        assert "random_seed_offset" in dataset_dict

        self.epoch = epoch
        self.full_epoch_0idx = full_epoch_0idx
        self.dataset_dict = dataset_dict

        parent_conn, child_conn = _mp.Pipe()
        self.parent_conn: mpConnection = parent_conn

        # the env will be forwarded to the child process
        with override_env_var(RANDOM_SEED_OFFSET_ENV_VAR, str(dataset_dict["random_seed_offset"])):
            self.worker_proc = _mp.Process(
                name=f"{name} worker ep {epoch}",
                target=_worker_proc_loop,
                args=(epoch, buffer_size, dataset_dict, child_conn),
                daemon=True,
            )
            self.worker_proc.start()

        # Make sure the child connection is closed here.
        # It stays open in the child, until the child dies.
        # When that happens, now any consecutive read on the pipe
        # should yield an exception -- which is what we want,
        # otherwise it would just hang.
        child_conn.close()

        # This worker operates on one sub epoch.
        # The data used here is only for this sub epoch.
        # Without shuffling, the next time we would use this data is in the next full epoch.
        # Thus, use the full epoch as the epoch here.
        # Make sure this init is async - we should not wait for this in the main process when it is not used yet.
        self.parent_conn.send(("init_seq_order", {"epoch": self.full_epoch_0idx + 1}))
        self._has_num_seqs = False
        self._num_seqs: Optional[int] = None  # remains None if unknown

    def _lazy_wait_for_init_seq_order(self):
        if self._has_num_seqs:
            return
        msg, num_seqs = self.parent_conn.recv()
        assert msg == "num_seqs" and (num_seqs is None or isinstance(num_seqs, int))
        self._num_seqs = num_seqs
        self._has_num_seqs = True

    def get_num_seqs(self) -> Optional[int]:
        """num seqs for this sub epoch"""
        self._lazy_wait_for_init_seq_order()
        return self._num_seqs

    def get_data_seq(self, seq_idx: int) -> Optional[DatasetSeq]:
        """get data seq"""
        self._lazy_wait_for_init_seq_order()
        self.parent_conn.send(("get_data_seq", {"seq_idx": seq_idx}))
        msg, data = self.parent_conn.recv()
        assert msg == "data_seq"
        return data

    def exit(self, *, join: bool = True):
        """exit"""
        self._lazy_wait_for_init_seq_order()
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
            try_run(self.worker_proc.join)


def _worker_proc_loop(
    epoch: int,
    buffer_size: int,
    dataset_dict: Dict[str, Any],
    parent_conn: mpConnection,
):
    if sys.platform == "linux":
        with open("/proc/self/comm", "w") as f:
            f.write(f"CFD worker {epoch}")

    assert isinstance(epoch, int) and isinstance(buffer_size, int)
    assert isinstance(dataset_dict, dict)
    assert isinstance(parent_conn, mpConnection)

    dataset = init_dataset(dataset_dict)

    got_init_seq_order = False
    cache: List[DatasetSeq] = []
    next_seq_idx = 0

    # noinspection PyShadowingNames
    def _add_to_cache():
        nonlocal next_seq_idx
        if len(cache) >= buffer_size:
            return False
        if not dataset.is_less_than_num_seqs(next_seq_idx):
            return False
        dataset.load_seqs(next_seq_idx, next_seq_idx + 1)
        seq_tag = dataset.get_tag(next_seq_idx)
        features = {data_key: dataset.get_data(next_seq_idx, data_key) for data_key in dataset.get_data_keys()}
        res = DatasetSeq(seq_idx=next_seq_idx, seq_tag=seq_tag, features=features)
        cache.append(res)
        next_seq_idx += 1
        return True

    # noinspection PyShadowingNames
    def _get_from_cache(seq_idx: int) -> Optional[DatasetSeq]:
        if not cache:
            return None
        if seq_idx > cache[-1].seq_idx:
            return None
        for seq in cache:
            assert seq.seq_idx <= seq_idx
            if seq.seq_idx == seq_idx:
                return seq
        assert False

    # noinspection PyShadowingNames
    def _get(seq_idx: int) -> Optional[DatasetSeq]:
        if cache and seq_idx < cache[0].seq_idx:
            raise Exception(
                f"requested seq idx {seq_idx} is smaller than cache start {cache[0].seq_idx}, cannot go backwards"
            )
        res = _get_from_cache(seq_idx)
        if res:
            return res
        if not dataset.is_less_than_num_seqs(seq_idx):
            return None
        assert next_seq_idx <= seq_idx
        while True:
            if not _add_to_cache():
                raise Exception(
                    f"buffer too small, requested seq idx {seq_idx},"
                    f" cache starts at {cache[0].seq_idx if cache else None}"
                )
            assert cache[-1].seq_idx <= seq_idx
            if cache[-1].seq_idx == seq_idx:
                return cache[-1]

    try:
        while True:
            if got_init_seq_order:
                while not parent_conn.poll():
                    if not _add_to_cache():
                        break
            msg, kwargs = parent_conn.recv()
            if msg == "exit":
                break
            elif msg == "get_data_seq":
                seq_idx = kwargs["seq_idx"]
                while cache and cache[0].seq_idx < seq_idx:
                    cache.pop(0)
                res = _get(seq_idx)
                parent_conn.send(("data_seq", res))
            elif msg == "init_seq_order":
                # We are responsible to get the seq order and distrib it to all the other workers.
                dataset.init_seq_order(**kwargs)
                try:
                    num_seqs = dataset.num_seqs
                except NotImplementedError:
                    num_seqs = None
                parent_conn.send(("num_seqs", num_seqs))
                got_init_seq_order = True
                next_seq_idx = 0
                cache[:] = []
            else:
                raise Exception(f"unknown msg {msg!r}")
    except KeyboardInterrupt:  # when parent dies
        pass
