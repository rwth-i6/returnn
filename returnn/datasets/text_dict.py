"""
:class:`TextDictDataset`
"""

from __future__ import annotations
from typing import Optional, Union, Any, Sequence, Tuple, List, Dict
import numpy as np

from returnn.log import log
from .basic import DatasetSeq
from .cached2 import CachedDataset2
from .util.vocabulary import Vocabulary


class TextDictDataset(CachedDataset2):
    """
    This dataset can read files in the format as usually generated from RETURNN search,
    i.e. with beam like (item_format = "list_with_scores")::

        {
            seq_tag: [(score1, text1), (score2, text2), ...],
            ...
        }

    Or without beam like (item_format = "single")::

        {
            seq_tag: text,
            ...
        }

    The data keys:

        data: The single (or best) sequence (encoded via vocab).
        data_flat: for list_with_scores, all sequences concatenated (encoded via vocab), in the given order
        data_seq_lens: for list_with_scores, the sequence lengths of each seq in data_flat
        scores: for list_with_scores, the scores of each seq in data_flat
    """

    def __init__(
        self,
        *,
        filename: str,
        item_format: str = "list_with_scores",
        vocab: Union[Vocabulary, Dict[str, Any]],
        **kwargs,
    ):
        """
        :param filename: text dict file. can be gzipped.
        :param item_format: "list_with_scores" or "single"
        :param vocab: to encode the text as a label sequence. See :class:`Vocabulary.create_vocab`.
        """
        super().__init__(**kwargs)
        self.filename = filename
        self.item_format = item_format
        self.vocab = vocab if isinstance(vocab, Vocabulary) else Vocabulary.create_vocab(**vocab)
        self.num_inputs = self.vocab.num_labels
        self.num_outputs = {}
        self.labels = {}
        if item_format == "list_with_scores":
            self.num_outputs.update(
                {
                    "data": (self.vocab.num_labels, 1),
                    "data_flat": (self.vocab.num_labels, 1),
                    "data_seq_lens": (1, 1),
                    "scores": (1, 1),
                }
            )
            self.labels.update({"data_flat": self.vocab.labels})
        elif item_format == "single":
            self.num_outputs.update({"data": (self.vocab.num_labels, 1)})
            self.labels.update({"data": self.vocab.labels})
        else:
            raise ValueError(f"invalid item_format {item_format!r}")
        self._data_values: Optional[List[Union[List[Tuple[float, str]], str]]] = None  # lazily loaded
        self._seq_tags: Optional[List[str]] = None  # lazily loaded
        self._seq_order: Optional[Sequence[int]] = None  # via init_seq_order

    def _load(self):
        if self._data_values is not None:
            return

        if self.filename.endswith(".gz"):
            import gzip

            txt = gzip.GzipFile(self.filename, "rb").read()
        else:
            txt = open(self.filename, "rb").read()

        from returnn.util.literal_py_to_pickle import literal_eval

        # Note: literal_py_to_pickle.literal_eval is quite efficient.
        # However, currently, it does not support inf/nan literals,
        # so it might break for some input.
        # We might want to put a simple fallback to eval here if needed.
        # Or maybe extend literal_py_to_pickle.literal_eval to support inf/nan literals.
        try:
            data: Dict[str, Any] = literal_eval(txt)
        except Exception as exc:
            print(f"{self}: Warning: literal_py_to_pickle.literal_eval failed:", file=log.v3)
            print(f"  {type(exc).__name__}: {exc}", file=log.v3)
            print("  Fallback to eval...", file=log.v3)
            data: Dict[str, Any] = eval(txt)
        assert data is not None
        assert isinstance(data, dict)
        assert len(data) > 0
        # Check some data.
        key, value = next(iter(data.items()))
        assert isinstance(key, str), f"{self}: expected seq tag as keys, got {key!r} ({type(key)})"  # seq tag
        if self.item_format == "single":
            assert isinstance(value, str), f"{self}: expected str ({self.item_format}), got {value!r} ({type(value)})"
        elif self.item_format == "list_with_scores":
            assert isinstance(value, list), f"{self}: expected list ({self.item_format}), got {value!r} ({type(value)})"
            assert len(value) > 0, f"{self}: expected non-empty list ({self.item_format}), got {value!r} for seq {key}"
            value0 = value[0]
            assert (
                isinstance(value0, tuple)
                and len(value0) == 2
                and isinstance(value0[0], float)
                and isinstance(value0[1], str)
            ), f"{self}: expected (score,text) tuples ({self.item_format}), got {value0!r} ({type(value0)})"
        else:
            raise ValueError(f"invalid item_format {self.item_format!r}")
        self._data_values = list(data.values())
        self._seq_tags = list(data.keys())

    def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
        """init seq order"""
        super().init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
        if epoch is None and seq_list is None and seq_order is None:
            self._num_seqs = 0
            return True

        random_seed = self._get_random_seed_for_epoch(epoch=epoch)
        self.vocab.set_random_seed(random_seed)

        if self.item_format == "single":

            def _get_seq_len(i: int) -> int:
                return len(self._data_values[i])

        elif self.item_format == "list_with_scores":

            def _get_seq_len(i: int) -> int:
                values = self._data_values[i]
                return sum(len(text) for _, text in values)

        else:
            raise ValueError(f"invalid item_format {self.item_format!r}")

        if seq_order is not None:
            self._seq_order = seq_order
        elif seq_list is not None:
            raise NotImplementedError(f"{self}: seq_list not supported yet")
        else:
            self._load()
            num_seqs = len(self._data_values)
            self._seq_order = self.get_seq_order_for_epoch(epoch=epoch, num_seqs=num_seqs, get_seq_len=_get_seq_len)
        self._num_seqs = len(self._seq_order)

    def _collect_single_seq(self, seq_idx: int) -> Optional[DatasetSeq]:
        orig_seq_idx = self._seq_order[seq_idx]
        seq_tag = self._seq_tags[orig_seq_idx]
        data = self._data_values[orig_seq_idx]

        res = {}
        labels_dtype = self.get_data_dtype("data")
        if self.item_format == "single":
            res["data"] = np.array(self.vocab.get_seq(data), dtype=labels_dtype)
        elif self.item_format == "list_with_scores":
            _, best = max(data, key=lambda x: x[0])
            res["data"] = np.array(self.vocab.get_seq(best), dtype=labels_dtype)
            labels = [self.vocab.get_seq(txt) for _, txt in data]
            res["data_flat"] = np.array(sum(labels, []), dtype=labels_dtype)
            res["data_seq_lens"] = np.array([len(seq) for seq in labels], dtype=self.get_data_dtype("data_seq_lens"))
            res["scores"] = np.array([score for score, _ in data], dtype=self.get_data_dtype("scores"))
        else:
            raise ValueError(f"invalid item_format {self.item_format!r}")
        return DatasetSeq(seq_idx=seq_idx, features=res, seq_tag=seq_tag)

    def supports_sharding(self) -> bool:
        """:return: whether this dataset supports sharding"""
        return True

    def supports_seq_order_sorting(self) -> bool:
        """supports sorting"""
        return True

    def get_current_seq_order(self) -> Sequence[int]:
        """:return: seq order"""
        assert self._seq_order is not None, "init_seq_order not called"
        return self._seq_order

    def have_corpus_seq_idx(self) -> bool:
        """
        :return: whether we can use :func:`get_corpus_seq_idx`
        """
        return True

    def get_corpus_seq_idx(self, seq_idx: int) -> int:
        """
        :param seq_idx:
        """
        assert self._seq_order is not None, "init_seq_order not called"
        return self._seq_order[seq_idx]

    def get_tag(self, seq_idx: int) -> str:
        """
        :param seq_idx:
        :return: seq tag
        """
        self._load()
        return self._seq_tags[self._seq_order[seq_idx]]

    def get_all_tags(self) -> List[str]:
        """:return: all tags"""
        self._load()
        return self._seq_tags

    def get_total_num_seqs(self, *, fast: bool = False) -> int:
        """:return: total num seqs in dataset (not for (sub)epoch)"""
        self._load()
        return len(self._data_values)

    def get_data_dim(self, key: str) -> int:
        """:return: dim of data entry with `key`"""
        if key == "data" or key == "data_flat":
            return self.vocab.num_labels
        elif key == "data_seq_lens":
            return 1
        elif key == "scores":
            return 1
        else:
            raise ValueError(f"{self}: unknown data key: {key}")

    def get_data_dtype(self, key: str) -> str:
        """:return: dtype of data entry with `key`"""
        if key == "data" or key == "data_flat":
            return "int32"
        elif key == "data_seq_lens":
            return "int32"
        elif key == "scores":
            return "float32"
        else:
            raise ValueError(f"{self}: unknown data key: {key}")

    def get_data_keys(self) -> List[str]:
        """:return: available data keys"""
        return list(self.num_outputs.keys())

    def get_data_shape(self, key: str) -> List[str]:
        """
        :returns get_data(*, key).shape[1:], i.e. num-frames excluded
        """
        return []  # all are scalar or sparse

    def is_data_sparse(self, key: str) -> bool:
        """:return: whether data entry with `key` is sparse"""
        return key == "data" or key == "data_flat"
