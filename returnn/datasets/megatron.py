"""
:class:`MegatronDataset`: token IDs in the Megatron Core indexed dataset format (``.bin`` and ``.idx`` files).
"""

from __future__ import annotations

from typing import Optional, Union, Any, Sequence, Dict, List, Tuple
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
    A data file (``.bin``) has the token IDs of all seqs,
    its index file (``.idx``) the seq lengths, their byte offsets into the data file and the document boundaries.
    The files are memory-mapped on the first access, so the dataset is cheap to create and to pickle.

    Every entry of an index file is one seq, in the order of the index files.
    The data key ``data`` has the token IDs of the seq as stored (including EOD or other special IDs),
    sparse with ``vocab_size``.
    Seq tags are ``<name>/doc-<d>/seq-<j>``, the ``j``-th entry of document ``d`` of the index file,
    with ``<name>`` the base name of the index file without ``.idx``.
    So the tag of a seq does not depend on the other files or their order,
    e.g. with :class:`DistributeFilesDataset`.

    The index files are given on their own, so a different segmentation (other index files)
    can refer to unchanged data files.
    For shuffling over many shards, combine this with :class:`DistributeFilesDataset`.
    Megatron's ``GPTDataset`` packing, sampling and blending are not done here.
    Only integer token IDs are supported: no float data, no multimodal index.

    Example::

        train = {
            "class": "MegatronDataset",
            "data_file": ["/data/tokens-0.bin", "/data/tokens-1.bin"],
            "index_file": ["/data/tokens-0.idx", "/data/tokens-1.idx"],
            "vocab_size": 131072,
            "seq_ordering": "random",
            "partition_epoch": 10,
        }
    """

    def __init__(
        self,
        *,
        data_file: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
        index_file: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
        vocab_size: int,
        vocab: Optional[Union[Vocabulary, Dict[str, Any]]] = None,
        **kwargs,
    ):
        """
        :param data_file: the ``.bin`` file with the token IDs, or a list of them
        :param index_file: the ``.idx`` file for ``data_file``, or a list of them, one per data file.
            Their base names must be unique, as they are part of the seq tags.
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
        data_files = [data_file] if isinstance(data_file, (str, os.PathLike)) else list(data_file)
        index_files = [index_file] if isinstance(index_file, (str, os.PathLike)) else list(index_file)
        if len(data_files) != len(index_files):
            raise ValueError(f"{self}: got {len(data_files)} data files but {len(index_files)} index files")
        self.data_file = data_file
        self.index_file = index_file
        self.vocab_size = vocab_size
        self.vocab = vocab
        self.num_inputs = vocab_size
        self.num_outputs = {"data": (vocab_size, 1)}
        if vocab is not None:
            self.labels["data"] = vocab.labels

        self._file_names = [(os.fspath(d), os.fspath(i)) for d, i in zip(data_files, index_files)]
        tag_names = [os.path.basename(i) for _, i in self._file_names]
        self._tag_names = [name[: -len(".idx")] if name.endswith(".idx") else name for name in tag_names]
        self._file_idx_by_tag_name = {name: file_idx for file_idx, name in enumerate(self._tag_names)}
        if len(self._file_idx_by_tag_name) != len(self._tag_names):
            raise ValueError(f"{self}: index file base names {self._tag_names} not unique")
        # All lazy, see _lazy_init.
        self._files: Optional[List[_MegatronFile]] = None
        self._seq_offsets: Optional[numpy.ndarray] = None  # [num files + 1], first seq of every file
        self._seq_order: Optional[Sequence[int]] = None  # init_seq_order

    def _lazy_init(self):
        if self._files is not None:
            return
        files = [_MegatronFile(d, i, error_prefix=f"{self}: ") for d, i in self._file_names]
        self._seq_offsets = numpy.cumsum([0] + [len(f.seq_lens) for f in files])
        self._files = files

    def _get_file_and_local_seq_idx(self, corpus_seq_idx: int) -> Tuple[int, int]:
        file_idx = int(numpy.searchsorted(self._seq_offsets, corpus_seq_idx, side="right")) - 1
        return file_idx, corpus_seq_idx - int(self._seq_offsets[file_idx])

    def _get_seq_len(self, corpus_seq_idx: int) -> int:
        file_idx, local_seq_idx = self._get_file_and_local_seq_idx(corpus_seq_idx)
        return int(self._files[file_idx].seq_lens[local_seq_idx])

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
                epoch=epoch, num_seqs=int(self._seq_offsets[-1]), get_seq_len=self._get_seq_len
            )
        self._num_seqs = len(self._seq_order)
        return True

    def _get_corpus_seq_idx_from_tag(self, tag: str) -> int:
        self._lazy_init()
        name, _, doc_and_seq = tag.rpartition("/doc-")
        doc_str, _, seq_str = doc_and_seq.partition("/seq-")
        if (
            name in self._file_idx_by_tag_name
            and doc_str.isdigit()
            and seq_str.isdigit()
            and tag == f"{name}/doc-{int(doc_str)}/seq-{int(seq_str)}"
        ):
            file_idx = self._file_idx_by_tag_name[name]
            document_idx, seq_in_doc = int(doc_str), int(seq_str)
            document_indices = self._files[file_idx].document_indices
            if document_idx < len(document_indices) - 1:
                start, end = document_indices[document_idx : document_idx + 2].tolist()
                if seq_in_doc < end - start:
                    return int(self._seq_offsets[file_idx]) + start + seq_in_doc
        raise ValueError(f"{self}: seq_list tag {tag!r} not in dataset")

    def _collect_single_seq(self, seq_idx: int) -> DatasetSeq:
        corpus_seq_idx = self.get_corpus_seq_idx(seq_idx)
        self._lazy_init()
        file_idx, local_seq_idx = self._get_file_and_local_seq_idx(corpus_seq_idx)
        data = self._files[file_idx].get_seq(local_seq_idx)
        if len(data) > 0 and (data.min() < 0 or data.max() >= self.vocab_size):
            raise ValueError(
                f"{self}: seq {self.get_tag(seq_idx)} has token IDs in [{data.min()}, {data.max()}],"
                f" not within vocab_size {self.vocab_size}"
            )
        return DatasetSeq(seq_idx, features={"data": data.astype("int32")}, seq_tag=self.get_tag(seq_idx))

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
        """:return: corpus seq idx, i.e. the index of the entry over all index files"""
        assert self._seq_order is not None, f"{self}: init_seq_order not called"
        return int(self._seq_order[seq_idx])

    def get_tag(self, sorted_seq_idx: int) -> str:
        """:return: tag of the sequence, ``<name>/doc-<d>/seq-<j>``"""
        self._lazy_init()
        file_idx, local_seq_idx = self._get_file_and_local_seq_idx(self.get_corpus_seq_idx(sorted_seq_idx))
        document_indices = self._files[file_idx].document_indices
        document_idx = int(numpy.searchsorted(document_indices, local_seq_idx, side="right")) - 1
        seq_in_doc = local_seq_idx - int(document_indices[document_idx])
        return f"{self._tag_names[file_idx]}/doc-{document_idx}/seq-{seq_in_doc}"

    def get_all_tags(self) -> List[str]:
        """:return: all tags"""
        self._lazy_init()
        tags = []
        for file, name in zip(self._files, self._tag_names):
            doc_lens = numpy.diff(file.document_indices)
            document_idx = numpy.repeat(numpy.arange(len(doc_lens)), doc_lens)
            seq_in_doc = numpy.arange(len(file.seq_lens)) - file.document_indices[document_idx]
            tags += [f"{name}/doc-{d}/seq-{j}" for d, j in zip(document_idx.tolist(), seq_in_doc.tolist())]
        return tags

    def get_total_num_seqs(self, *, fast: bool = False) -> int:
        """:return: total number of sequences in the dataset, i.e. entries in all index files"""
        self._lazy_init()
        return int(self._seq_offsets[-1])

    def get_data_keys(self) -> List[str]:
        """:return: available data keys"""
        return ["data"]

    def get_data_dtype(self, key: str) -> str:
        """:return: dtype of data entry with `key`"""
        assert key == "data", f"{self}: unknown data key {key!r}"
        return "int32"


class _MegatronFile:
    """
    One data file with its index file, memory-mapped, with all index entries checked.
    """

    def __init__(self, data_file: str, index_file: str, *, error_prefix: str):
        index_size = os.path.getsize(index_file)
        if index_size < _IndexHeaderSize:
            raise ValueError(f"{error_prefix}index file {index_file!r} truncated, size {index_size}")
        index = numpy.memmap(index_file, dtype="uint8", mode="r")
        magic, version, dtype_code, num_seqs, num_document_indices = struct.unpack(
            _IndexHeaderFormat, index[:_IndexHeaderSize].tobytes()
        )
        if magic != _IndexMagic or version != 1:
            raise ValueError(
                f"{error_prefix}index file {index_file!r} is not in the Megatron MMIDIDX version 1 format,"
                f" header {magic!r}, version {version}"
            )
        if dtype_code not in _DTypes:
            raise ValueError(
                f"{error_prefix}index file {index_file!r}: dtype code {dtype_code} not supported,"
                f" only integer token IDs, codes {_DTypes}"
            )
        dtype = numpy.dtype(_DTypes[dtype_code]).newbyteorder("<")
        expected_index_size = _IndexHeaderSize + num_seqs * (4 + 8) + num_document_indices * 8
        if index_size != expected_index_size:
            raise ValueError(
                f"{error_prefix}index file {index_file!r} has size {index_size},"
                f" expected {expected_index_size} for {num_seqs} seqs and {num_document_indices} document indices"
                f" (multimodal index files are not supported)"
            )
        offset = _IndexHeaderSize
        seq_lens = index[offset : offset + num_seqs * 4].view("<i4")
        offset += num_seqs * 4
        seq_pointers = index[offset : offset + num_seqs * 8].view("<i8")
        offset += num_seqs * 8
        document_indices = index[offset : offset + num_document_indices * 8].view("<i8")

        data_size = os.path.getsize(data_file)
        if num_document_indices == 0 or document_indices[0] != 0 or document_indices[-1] != num_seqs:
            raise ValueError(
                f"{error_prefix}index file {index_file!r}: document indices must go from 0 to num seqs {num_seqs},"
                f" got {document_indices[:1].tolist()} ... {document_indices[-1:].tolist()}"
            )
        # Chunked, to not create temporary arrays of the size of the whole index.
        for start in range(0, max(num_seqs, num_document_indices), _CheckChunkSize):
            end = start + _CheckChunkSize
            if numpy.any(numpy.diff(document_indices[start : end + 1]) < 0):
                raise ValueError(f"{error_prefix}index file {index_file!r}: document indices not monotonic")
            lens = seq_lens[start:end].astype("int64")
            pointers = seq_pointers[start:end]
            invalid = (lens < 0) | (pointers < 0) | (pointers + lens * dtype.itemsize > data_size)
            if numpy.any(invalid):
                i = start + int(numpy.argmax(invalid))
                raise ValueError(
                    f"{error_prefix}index file {index_file!r}: seq {i} with length {seq_lens[i]} at byte offset"
                    f" {seq_pointers[i]} is not within the data file {data_file!r} of size {data_size}"
                )

        self.dtype = dtype
        self.seq_lens = seq_lens
        self.seq_pointers = seq_pointers
        self.document_indices = document_indices
        # An empty file cannot be memory-mapped.
        self.data = numpy.memmap(data_file, dtype="uint8", mode="r") if data_size > 0 else numpy.zeros(0, "uint8")

    def get_seq(self, seq_idx: int) -> numpy.ndarray:
        """:return: token IDs of the seq, as stored, i.e. in the dtype of the file"""
        pointer = int(self.seq_pointers[seq_idx])
        return self.data[pointer : pointer + int(self.seq_lens[seq_idx]) * self.dtype.itemsize].view(self.dtype)


# Header of the index file: magic, version, dtype code, num seqs, num document indices.
_IndexHeaderFormat = "<9sQBQQ"
_IndexHeaderSize = struct.calcsize(_IndexHeaderFormat)
_IndexMagic = b"MMIDIDX\x00\x00"

# Megatron DType codes of integer types. 6 and 7 are float64 and float32.
_DTypes = {1: "uint8", 2: "int8", 3: "int16", 4: "int32", 5: "int64", 8: "uint16"}

_CheckChunkSize = 2**24
