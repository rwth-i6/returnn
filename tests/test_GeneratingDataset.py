# -*- coding: utf8 -*-

from __future__ import annotations

import _setup_test_env  # noqa
import unittest
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false
from returnn.datasets.generating import *
from returnn.datasets.basic import DatasetSeq
from returnn.util.basic import PY3, unicode
import os
import unittest

from returnn.util import better_exchook
from returnn.log import log


my_dir = os.path.dirname(os.path.realpath(__file__))


def test_init():
    dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=4)
    assert_equal(dataset.num_inputs, 2)
    assert_equal(dataset.num_outputs, {"classes": (3, 1), "data": (2, 2)})
    assert_equal(dataset.num_seqs, 4)


def test_load_seqs():
    dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=4)
    dataset.init_seq_order(epoch=1)
    dataset.load_seqs(0, 1)
    dataset.load_seqs(1, 3)


@unittest.skipIf(not os.path.exists("/tmp/enwik8.zip"), "we will not trigger the download")
def test_Enwik8Corpus_batch_num_seqs():
    dataset = Enwik8Corpus(path="/tmp", subset="validation", seq_len=13)
    dataset.init_seq_order(epoch=17)
    data = b""
    n = 0
    while dataset.is_less_than_num_seqs(n) and n < 100:
        dataset.load_seqs(n, n + 1)
        data += bytes(dataset.get_data(n, "data"))
        n += 1

    batch_size = 23
    batch_data = [b"" for i in range(batch_size)]
    dataset = Enwik8Corpus(path="/tmp", subset="validation", seq_len=9, batch_num_seqs=batch_size)
    dataset.init_seq_order(epoch=31)
    n = 0
    while dataset.is_less_than_num_seqs(n) and n < 100:
        dataset.load_seqs(n, n + 1)
        new_data = bytes(dataset.get_data(n, "data"))
        batch_data[n % batch_size] += new_data
        n += 1
    assert data.startswith(batch_data[0])


def test_StaticDataset_custom_keys():
    dataset = StaticDataset([{"source": numpy.array([1, 2, 3]), "target": numpy.array([3, 4, 5, 6, 7])}])
    dataset.init_seq_order(epoch=1)
    assert dataset.num_seqs == 1
    assert_equal(dataset.get_data_keys(), ["source", "target"])
    assert_equal(dataset.num_outputs["source"][1], 1)
    assert_equal(dataset.num_outputs["target"][1], 1)
    dataset.load_seqs(0, 1)
    assert_equal(list(dataset.get_data(0, "source")), [1, 2, 3])
    assert_equal(list(dataset.get_data(0, "target")), [3, 4, 5, 6, 7])


def test_StaticDataset_custom_keys_with_dims():
    dataset = StaticDataset(
        data=[{"source": numpy.array([1, 2, 3]), "target": numpy.array([3, 4, 5, 6, 7])}],
        output_dim={"source": [5, 1], "target": [10, 1]},
    )
    dataset.init_seq_order(epoch=1)
    assert dataset.num_seqs == 1
    assert_equal(dataset.get_data_keys(), ["source", "target"])
    assert_equal(dataset.num_outputs["source"][1], 1)
    assert_equal(dataset.num_outputs["target"][1], 1)
    dataset.load_seqs(0, 1)
    assert_equal(list(dataset.get_data(0, "source")), [1, 2, 3])
    assert_equal(list(dataset.get_data(0, "target")), [3, 4, 5, 6, 7])


def test_StaticDataset_utf8():
    s = "wër"
    print("some unicode str:", s, "repr:", repr(s), "type:", type(s), "len:", len(s))
    assert len(s) == 3
    if PY3:
        assert isinstance(s, str)
        s_byte_list = list(s.encode("utf8"))
    else:
        assert isinstance(s, unicode)
        s_byte_list = list(map(ord, s.encode("utf8")))
    print("utf8 byte list:", s_byte_list)
    assert len(s_byte_list) == 4 > 3
    raw = numpy.array(s_byte_list, dtype="uint8")
    assert_equal(raw.tolist(), [119, 195, 171, 114])
    data = StaticDataset([{"data": raw}], output_dim={"data": (255, 1)})
    if "data" not in data.labels:
        data.labels["data"] = [chr(i) for i in range(255)]  # like in SprintDataset
    data.init_seq_order(epoch=1)
    data.load_seqs(0, 1)
    raw_ = data.get_data(seq_idx=0, key="data")
    assert_equal(raw.tolist(), raw_.tolist())
    assert data.can_serialize_data(key="data")
    s_serialized = data.serialize_data(key="data", data=raw)
    print("serialized:", s_serialized, "repr:", repr(s_serialized), "type:", type(s_serialized))
    assert_equal(s, s_serialized)


# might be moved to a separate test_MetaDataset ...
def test_ConcatSeqsDataset():
    num_seqs = 2
    seq_len = 3
    sub_dataset = StaticDataset([{"data": numpy.array(range(1, seq_len + 1))}] * num_seqs)
    from returnn.datasets.meta import ConcatSeqsDataset
    import tempfile

    seq_list_f = tempfile.NamedTemporaryFile(mode="w", prefix="seq-list", suffix=".txt")
    seq_len_f = tempfile.NamedTemporaryFile(mode="w", prefix="seq-len", suffix=".txt")
    with seq_list_f, seq_len_f:
        seq_len_f.write("{\n")
        for i in range(num_seqs):
            seq_list_f.write("seq-%i\n" % i)
            seq_len_f.write("'seq-%i': %i,\n" % (i, seq_len))
        seq_len_f.write("}\n")
        for i in range(0, num_seqs, 2):
            seq_list_f.write("seq-%i;seq-%i\n" % (i, i + 1))
        seq_list_f.flush()
        seq_len_f.flush()
        dataset = ConcatSeqsDataset(dataset=sub_dataset, seq_list_file=seq_list_f.name, seq_len_file=seq_len_f.name)
        dataset.init_seq_order(epoch=1)
        concat_num_seqs = num_seqs + num_seqs // 2
        dataset.load_seqs(0, concat_num_seqs)
        assert dataset.num_seqs == concat_num_seqs == 3
        assert_equal(dataset.get_data(0, "data").tolist(), [1, 2, 3])
        assert_equal(dataset.get_data(1, "data").tolist(), [1, 2, 3])
        assert_equal(dataset.get_data(2, "data").tolist(), [1, 2, 3, 1, 2, 3])


# might be moved to a separate test_MetaDataset ...
def test_ConcatSeqsDataset_repeat_in_between_last_frame_up_to_multiple_of():
    num_seqs = 2
    sub_dataset = StaticDataset([{"data": numpy.array([1, 2])}, {"data": numpy.array([1, 2, 3])}])
    from returnn.datasets.meta import ConcatSeqsDataset
    import tempfile

    seq_list_f = tempfile.NamedTemporaryFile(mode="w", prefix="seq-list", suffix=".txt")
    seq_len_f = tempfile.NamedTemporaryFile(mode="w", prefix="seq-len", suffix=".txt")
    with seq_list_f, seq_len_f:
        seq_len_f.write("%r\n" % ({"seq-0": 2, "seq-1": 3},))
        seq_len_f.flush()
        seq_list_f.write("seq-0\n")
        seq_list_f.write("seq-1\n")
        seq_list_f.write("seq-0;seq-1;seq-1\n")
        seq_list_f.flush()
        concat_num_seqs = 3
        dataset = ConcatSeqsDataset(
            dataset=sub_dataset,
            repeat_in_between_last_frame_up_to_multiple_of={"data": 5},
            seq_list_file=seq_list_f.name,
            seq_len_file=seq_len_f.name,
        )
        dataset.init_seq_order(epoch=1)
        dataset.load_seqs(0, concat_num_seqs)
        assert dataset.num_seqs == concat_num_seqs == 3
        assert_equal(dataset.get_data(0, "data").tolist(), [1, 2])
        assert_equal(dataset.get_data(1, "data").tolist(), [1, 2, 3])
        assert_equal(dataset.get_data(2, "data").tolist(), [1, 2, 2, 2, 2, 1, 2, 3, 3, 3, 1, 2, 3])


def test_BytePairEncoding_unicode():
    bpe = BytePairEncoding(
        bpe_file="%s/bpe-unicode-demo.codes" % my_dir,
        vocab_file="%s/bpe-unicode-demo.vocab" % my_dir,
        unknown_label="<unk>",
    )
    assert_equal(bpe.num_labels, 189)
    assert_equal(bpe.id_to_label(5), "z")
    assert_equal(bpe.label_to_id("z"), 5)
    assert_equal(bpe.bpe._bpe_codes[("n", "d</w>")], 1)
    assert_equal(bpe.id_to_label(6), "å")
    assert_equal(bpe.label_to_id("å"), 6)
    assert_equal(bpe.bpe._bpe_codes[("à", "nd</w>")], 2)

    def get_bpe_seq(text):
        """
        :param str text:
        :rtype: str
        """
        bpe_label_seq = bpe.get_seq(text)
        res = " ".join(bpe.id_to_label(i) for i in bpe_label_seq)
        print("%r -> %r" % (text, res))
        return res

    assert_equal(get_bpe_seq("kod"), "k@@ o@@ d")  # str
    assert_equal(get_bpe_seq("kod"), "k@@ o@@ d")  # unicode
    assert_equal(get_bpe_seq("råt"), "råt")
    assert_equal(
        get_bpe_seq("råt råt iz ďër iz ďër ám àn iz ďër ë låk ë kod áv dres wër yù wêk dù ďë àsk"),
        "råt råt iz ďër iz ďër ám à@@ n iz ďër ë låk ë k@@ o@@ d áv d@@ r@@ e@@ s w@@ ër yù w@@ ê@@ k dù ďë à@@ s@@ k",
    )


if __name__ == "__main__":
    better_exchook.install()
    if len(sys.argv) <= 1:
        for k, v in sorted(globals().items()):
            if k.startswith("test_"):
                print("-" * 40)
                print("Executing: %s" % k)
                try:
                    v()
                except unittest.SkipTest as exc:
                    print("SkipTest:", exc)
                print("-" * 40)
        print("Finished all tests.")
    else:
        assert len(sys.argv) >= 2
        for arg in sys.argv[1:]:
            print("Executing: %s" % arg)
            if arg in globals():
                globals()[arg]()  # assume function and execute
            else:
                eval(arg)  # assume Python code and execute
