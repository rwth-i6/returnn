"""
:class:`MegatronDataset`: token IDs in the Megatron Core indexed dataset format (``.bin`` and ``.idx`` file).
"""

from __future__ import annotations

from typing import Optional, Union, Any, Sequence, Dict, List
import os
import struct
import numpy

from .basic import DatasetSeq
from .cached2 import CachedDataset2
from .util.vocabulary import Vocabulary

__all__ = ["MegatronDataset"]


class MegatronDataset(CachedDataset2):
    """
    Token IDs in the Megatron Core indexed dataset format (``MMIDIDX`` version 1),
    as written by Megatron's ``IndexedDatasetBuilder``,
    see https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py.
    The data file (``.bin``) has the token IDs of all seqs,
    the index file (``.idx``) the seq lengths, their byte offsets into the data file and the document boundaries.
    Both files are memory-mapped on the first access, so the dataset is cheap to create and to pickle.

    Every entry of the index file is one seq, in the order of the index file.
    Data keys:

    - ``data``: the token IDs of the seq as stored (including EOD or other special IDs), sparse with ``vocab_size``
    - ``document_idx``: scalar, the index of the document the seq belongs to

    Seq tags are ``seq-<i>``, with ``i`` the index of the entry in the index file.

    The index file is given on its own, so a different segmentation (another index file)
    can refer to an unchanged data file.
    Multiple shards (data and index file each) can be combined via :class:`DistributeFilesDataset`.
    Megatron's ``GPTDataset`` packing, sampling and blending are not done here.
    Only integer token IDs are supported: no float data, no multimodal index.

    Example::

        train = {
            "class": "MegatronDataset",
            "data_file": "/data/tokens.bin",
            "index_file": "/data/tokens.idx",
            "vocab_size": 131072,
            "seq_ordering": "random",
            "partition_epoch": 10,
        }
    """

    def __init__(
        self,
        *,
        data_file: Union[str, os.PathLike],
        index_file: Union[str, os.PathLike],
        vocab_size: int,
        vocab: Optional[Union[Vocabulary, Dict[str, Any]]] = None,
        **kwargs,
    ):
        """
        :param data_file: the ``.bin`` file with the token IDs
        :param index_file: the ``.idx`` file, referring to ``data_file``
        :param vocab_size: dim of ``data``. All token IDs must be below it.
            The index file does not store it, and it is not inferred from the data.
        :param vocab: labels of ``data``, see :func:`Vocabulary.create_vocab`. Must have ``vocab_size`` labels.
        """
        super().__init__(**kwargs)
        if not 0 < vocab_size <= 2**31:
            raise ValueError(f"{self}: vocab_size {vocab_size} must be in [1, 2**31]")
        if isinstance(vocab, dict):
            vocab = Vocabulary.create_vocab(**vocab)
        if vocab is not None and vocab.num_labels != vocab_size:
            raise ValueError(f"{self}: vocab has {vocab.num_labels} labels, but vocab_size is {vocab_size}")
        self.data_file = data_file
        self.index_file = index_file
        self.vocab_size = vocab_size
        self.vocab = vocab
        self.num_inputs = vocab_size
        self.num_outputs = {"data": (vocab_size, 1), "document_idx": (1, 0)}
        if vocab is not None:
            self.labels["data"] = vocab.labels

        # All lazy, see _lazy_init.
        self._dtype: Optional[numpy.dtype] = None
        self._seq_lens: Optional[numpy.ndarray] = None
        self._seq_pointers: Optional[numpy.ndarray] = None
        self._document_indices: Optional[numpy.ndarray] = None
        self._data: Optional[numpy.ndarray] = None
        self._seq_order: Optional[Sequence[int]] = None  # init_seq_order

    def _lazy_init(self):
        if self._seq_lens is not None:
            return

        index_file = os.fspath(self.index_file)
        index_size = os.path.getsize(index_file)
        if index_size < _IndexHeaderSize:
            raise ValueError(f"{self}: index file {index_file!r} truncated, size {index_size}")
        index = numpy.memmap(index_file, dtype="uint8", mode="r")
        magic, version, dtype_code, num_seqs, num_document_indices = struct.unpack(
            _IndexHeaderFormat, index[:_IndexHeaderSize].tobytes()
        )
        if magic != _IndexMagic or version != 1:
            raise ValueError(
                f"{self}: index file {index_file!r} is not in the Megatron MMIDIDX version 1 format,"
                f" header {magic!r}, version {version}"
            )
        if dtype_code not in _DTypes:
            raise ValueError(
                f"{self}: index file {index_file!r}: dtype code {dtype_code} not supported,"
                f" only integer token IDs, codes {_DTypes}"
            )
        dtype = numpy.dtype(_DTypes[dtype_code]).newbyteorder("<")
        expected_index_size = _IndexHeaderSize + num_seqs * (4 + 8) + num_document_indices * 8
        if index_size != expected_index_size:
            raise ValueError(
                f"{self}: index file {index_file!r} has size {index_size},"
                f" expected {expected_index_size} for {num_seqs} seqs and {num_document_indices} document indices"
                f" (multimodal index files are not supported)"
            )
        offset = _IndexHeaderSize
        seq_lens = index[offset : offset + num_seqs * 4].view("<i4")
        offset += num_seqs * 4
        seq_pointers = index[offset : offset + num_seqs * 8].view("<i8")
        offset += num_seqs * 8
        document_indices = index[offset : offset + num_document_indices * 8].view("<i8")

        data_file = os.fspath(self.data_file)
        data_size = os.path.getsize(data_file)
        if num_document_indices == 0 or document_indices[0] != 0 or document_indices[-1] != num_seqs:
            raise ValueError(
                f"{self}: index file {index_file!r}: document indices must go from 0 to num seqs {num_seqs},"
                f" got {document_indices[:1].tolist()} ... {document_indices[-1:].tolist()}"
            )
        # Chunked, to not create temporary arrays of the size of the whole index.
        for start in range(0, max(num_seqs, num_document_indices), _CheckChunkSize):
            end = start + _CheckChunkSize
            if numpy.any(numpy.diff(document_indices[start : end + 1]) < 0):
                raise ValueError(f"{self}: index file {index_file!r}: document indices not monotonic")
            lens = seq_lens[start:end].astype("int64")
            pointers = seq_pointers[start:end]
            invalid = (lens < 0) | (pointers < 0) | (pointers + lens * dtype.itemsize > data_size)
            if numpy.any(invalid):
                i = start + int(numpy.argmax(invalid))
                raise ValueError(
                    f"{self}: index file {index_file!r}: seq {i} with length {seq_lens[i]} at byte offset"
                    f" {seq_pointers[i]} is not within the data file {data_file!r} of size {data_size}"
                )

        self._dtype = dtype
        self._seq_lens = seq_lens
        self._seq_pointers = seq_pointers
        self._document_indices = document_indices
        # An empty file cannot be memory-mapped.
        self._data = numpy.memmap(data_file, dtype="uint8", mode="r") if data_size > 0 else numpy.zeros(0, "uint8")

    def init_seq_order(
        self,
        epoch: Optional[int] = None,
        seq_list: Optional[Sequence[str]] = None,
        seq_order: Optional[Sequence[int]] = None,
    ) -> bool:
        """
        :param epoch:
        :param seq_list: List of sequence tags, to set a predefined order.
        :param seq_order: List of corpus sequence indices, to set a predefined order.
        :returns whether the order changed (True is always safe to return)
        """
        super().init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
        if seq_order is not None:
            self._seq_order = seq_order
        elif seq_list is not None:
            self._seq_order = [self._get_corpus_seq_idx_from_tag(tag) for tag in seq_list]
        elif epoch is None:
            self._seq_order = ()
        else:
            self._lazy_init()
            self._seq_order = self.get_seq_order_for_epoch(
                epoch=epoch, num_seqs=len(self._seq_lens), get_seq_len=self._seq_lens.__getitem__
            )
        self._num_seqs = len(self._seq_order)
        return True

    def _get_corpus_seq_idx_from_tag(self, tag: str) -> int:
        self._lazy_init()
        idx = tag[len("seq-") :]
        if not idx.isdigit() or tag != f"seq-{int(idx)}" or int(idx) >= len(self._seq_lens):
            raise ValueError(f"{self}: seq_list tag {tag!r} not in dataset")
        return int(idx)

    def _collect_single_seq(self, seq_idx: int) -> DatasetSeq:
        corpus_seq_idx = self.get_corpus_seq_idx(seq_idx)
        self._lazy_init()
        pointer = int(self._seq_pointers[corpus_seq_idx])
        seq_len = int(self._seq_lens[corpus_seq_idx])
        data = self._data[pointer : pointer + seq_len * self._dtype.itemsize].view(self._dtype)
        if seq_len > 0 and (data.min() < 0 or data.max() >= self.vocab_size):
            raise ValueError(
                f"{self}: seq {corpus_seq_idx} has token IDs in [{data.min()}, {data.max()}],"
                f" not within vocab_size {self.vocab_size}"
            )
        document_idx = int(numpy.searchsorted(self._document_indices, corpus_seq_idx, side="right")) - 1
        features = {"data": data.astype("int32"), "document_idx": numpy.array(document_idx, dtype="int64")}
        return DatasetSeq(seq_idx, features=features, seq_tag=f"seq-{corpus_seq_idx}")

    def supports_sharding(self) -> bool:
        """:return: whether this dataset supports sharding"""
        return True

    def supports_seq_order_sorting(self) -> bool:
        """:return: whether "sorted" or "sorted_reverse" is supported for seq_ordering"""
        return True

    def get_current_seq_order(self) -> Sequence[int]:
        """:return: list of corpus seq idx"""
        assert self._seq_order is not None, f"{self}: init_seq_order not called"
        return self._seq_order

    def have_corpus_seq_idx(self) -> bool:
        """:return: whether :func:`get_corpus_seq_idx` can be used"""
        return True

    def get_corpus_seq_idx(self, seq_idx: int) -> int:
        """:return: corpus seq idx, i.e. the index of the entry in the index file"""
        assert self._seq_order is not None, f"{self}: init_seq_order not called"
        return int(self._seq_order[seq_idx])

    def get_tag(self, sorted_seq_idx: int) -> str:
        """:return: tag of the sequence"""
        return f"seq-{self.get_corpus_seq_idx(sorted_seq_idx)}"

    def get_all_tags(self) -> List[str]:
        """:return: all tags"""
        return [f"seq-{i}" for i in range(self.get_total_num_seqs())]

    def get_total_num_seqs(self, *, fast: bool = False) -> int:
        """:return: total number of sequences in the dataset, i.e. entries in the index file"""
        self._lazy_init()
        return len(self._seq_lens)

    def get_data_keys(self) -> List[str]:
        """:return: available data keys"""
        return ["data", "document_idx"]

    def get_target_list(self) -> List[str]:
        """:return: target keys"""
        return []

    def get_data_dim(self, key: str) -> int:
        """:return: dim of data entry with `key`"""
        return self.num_outputs[key][0]

    def get_data_dtype(self, key: str) -> str:
        """:return: dtype of data entry with `key`"""
        return {"data": "int32", "document_idx": "int64"}[key]

    def get_data_shape(self, key: str) -> List[int]:
        """:return: shape of data entry with `key`, without time axis"""
        return []

    def is_data_sparse(self, key: str) -> bool:
        """:return: whether data entry with `key` is sparse"""
        return key == "data"


# Header of the index file: magic, version, dtype code, num seqs, num document indices.
_IndexHeaderFormat = "<9sQBQQ"
_IndexHeaderSize = struct.calcsize(_IndexHeaderFormat)
_IndexMagic = b"MMIDIDX\x00\x00"

# Megatron DType codes of integer types. 6 and 7 are float64 and float32.
_DTypes = {1: "uint8", 2: "int8", 3: "int16", 4: "int32", 5: "int64", 8: "uint16"}

_CheckChunkSize = 2**24
