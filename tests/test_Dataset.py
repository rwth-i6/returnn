# Also see test_SprintDataset.py.

from __future__ import annotations

from typing import Any, List, Dict
import os
import sys
import _setup_test_env  # noqa
import unittest
import numpy
import tempfile
import contextlib
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false
from returnn.datasets.generating import Task12AXDataset, DummyDataset, DummyDatasetMultipleSequenceLength
from returnn.engine.batch import Batch
from returnn.datasets.basic import Dataset, DatasetSeq, init_dataset
from returnn.util.basic import NumbersDict
from returnn.util import better_exchook


def dummy_iter_dataset(dataset: Dataset) -> List[DatasetSeq]:
    """
    :param Dataset dataset:
    :return: seqs
    """
    dataset.init_seq_order(epoch=1)
    data_keys = dataset.get_data_keys()
    seq_idx = 0
    seqs = []
    while dataset.is_less_than_num_seqs(seq_idx):
        dataset.load_seqs(seq_idx, seq_idx + 1)
        data = {}
        for key in data_keys:
            data[key] = dataset.get_data(seq_idx=seq_idx, key=key)
        tag = dataset.get_tag(seq_idx)
        seq = DatasetSeq(seq_idx=seq_idx, seq_tag=tag, features=data)
        seqs.append(seq)
        seq_idx += 1
    print("Iterated through %r, num seqs %i" % (dataset, seq_idx))
    return seqs


def compare_dataset_seqs(seqs1: List[DatasetSeq], seqs2: List[DatasetSeq]):
    """
    :param list[DatasetSeq] seqs1:
    :param list[DatasetSeq] seqs2:
    """
    assert len(seqs1) == len(seqs2)
    for i, (seq1, seq2) in enumerate(zip(seqs1, seqs2)):
        assert seq1.seq_idx == seq2.seq_idx == i
        assert seq1.seq_tag == seq2.seq_tag, f"seq1 tag {seq1.seq_tag!r} != seq2 tag {seq2.seq_tag!r} for seq idx {i}"
        assert set(seq1.features.keys()) == set(seq2.features.keys())
        for key in seq1.features.keys():
            assert seq1.features[key].shape == seq2.features[key].shape
            assert seq1.features[key].dtype == seq2.features[key].dtype
            assert (seq1.features[key] == seq2.features[key]).all()


def test_Task12AXDataset_deepcopy():
    from copy import deepcopy

    dataset = Task12AXDataset(num_seqs=10)
    dataset = deepcopy(dataset)
    dataset.init_seq_order(1)
    n = dataset.num_seqs
    for i in range(n):
        dataset.load_seqs(i, i + 1)
        targets = dataset.get_data(i, "classes")
        print(targets)
    assert not dataset.is_less_than_num_seqs(n)


def test_Task12AXDataset_inf():
    dataset = Task12AXDataset(num_seqs=float("inf"))
    dataset.init_seq_order(1)
    n = 10
    for i in range(n):
        dataset.load_seqs(i, i + 1)
        targets = dataset.get_data(i, "classes")
        print(targets)
    assert dataset.is_less_than_num_seqs(n)


def test_Task12AXDataset_random():
    dataset = Task12AXDataset(num_seqs=10, seq_ordering="random")
    dataset.init_seq_order(1)
    n = dataset.num_seqs
    for i in range(n):
        dataset.load_seqs(i, i + 1)
        targets = dataset.get_data(i, "classes")
        print(targets)
    assert not dataset.is_less_than_num_seqs(n)


def test_generate_batches():
    dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=20)
    dataset.init_seq_order(1)
    batch_gen = dataset.generate_batches(recurrent_net=False, max_seqs=2, batch_size=5)
    while batch_gen.has_more():
        batch_gen.peek_next_n(1)
        batch_gen.advance(1)


def test_generate_batches_recurrent():
    dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=20)
    dataset.init_seq_order(1)
    batch_gen = dataset.generate_batches(recurrent_net=True, max_seqs=2, batch_size=5)
    while batch_gen.has_more():
        batch_gen.peek_next_n(1)
        batch_gen.advance(1)


def test_iterate_seqs_no_chunking_1():
    dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=2, seq_len=11)
    dataset.chunk_step = 0
    dataset.chunk_size = 0
    dataset.init_seq_order(1)
    seqs = list(dataset.iterate_seqs())
    assert_equal(len(seqs), 2)
    assert_equal(seqs[0], (0, 0, 11))  # seq-idx, start-frame, end-frame
    assert_equal(seqs[1], (1, 0, 11))


def test_iterate_seqs_chunking_1():
    dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=2, seq_len=11)
    dataset.chunk_step = 5
    dataset.chunk_size = 10
    dataset.init_seq_order(1)
    seqs = list(dataset.iterate_seqs())
    for s in seqs:
        print(s)
    assert_equal(len(seqs), 6)
    assert_equal(seqs[0], (0, 0, 10))  # seq-idx, start-frame, end-frame
    assert_equal(seqs[1], (0, 5, 11))
    assert_equal(seqs[2], (0, 10, 11))
    assert_equal(seqs[3], (1, 0, 10))
    assert_equal(seqs[4], (1, 5, 11))
    assert_equal(seqs[5], (1, 10, 11))


def test_iterate_seqs_chunking_varying_sequence_length():
    dataset = DummyDatasetMultipleSequenceLength(
        input_dim=2, output_dim=3, num_seqs=2, seq_len={"data": 24, "classes": 12}
    )
    dataset.chunk_size = {"data": 12, "classes": 6}
    dataset.chunk_step = {"data": 6, "classes": 3}
    dataset.init_seq_order(1)
    seqs = list(dataset.iterate_seqs())
    for s in seqs:
        print(s)
    assert_equal(len(seqs), 8)
    assert_equal(seqs[0], (0, NumbersDict({"data": 0, "classes": 0}), NumbersDict({"data": 12, "classes": 6})))
    assert_equal(seqs[1], (0, NumbersDict({"data": 6, "classes": 3}), NumbersDict({"data": 18, "classes": 9})))
    assert_equal(seqs[2], (0, NumbersDict({"data": 12, "classes": 6}), NumbersDict({"data": 24, "classes": 12})))
    assert_equal(seqs[3], (0, NumbersDict({"data": 18, "classes": 9}), NumbersDict({"data": 24, "classes": 12})))
    assert_equal(seqs[4], (1, NumbersDict({"data": 0, "classes": 0}), NumbersDict({"data": 12, "classes": 6})))
    assert_equal(seqs[5], (1, NumbersDict({"data": 6, "classes": 3}), NumbersDict({"data": 18, "classes": 9})))
    assert_equal(seqs[6], (1, NumbersDict({"data": 12, "classes": 6}), NumbersDict({"data": 24, "classes": 12})))
    assert_equal(seqs[7], (1, NumbersDict({"data": 18, "classes": 9}), NumbersDict({"data": 24, "classes": 12})))


def test_iterate_seqs_custom_chunking():
    default_key = "data"
    chunk_step = 5
    chunk_size = 10

    def _custom_chunking_func(dataset, seq_idx_start, **_kwargs):
        assert isinstance(dataset, Dataset)
        seq_idx = seq_idx_start
        while dataset.is_less_than_num_seqs(seq_idx):
            length = dataset.get_seq_length(seq_idx)
            t = NumbersDict.constant_like(0, numbers_dict=length)
            while length[default_key] > t[default_key]:
                chunk_start = NumbersDict(t)
                chunk_end = NumbersDict.min([t + chunk_size, length])
                yield seq_idx, chunk_start, chunk_end
                t += chunk_step
            seq_idx += 1

    dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=2, seq_len=11)
    dataset.custom_chunking_func = _custom_chunking_func
    dataset.init_seq_order(1)
    seqs = list(dataset.iterate_seqs())
    for s in seqs:
        print(s)
    assert_equal(len(seqs), 6)
    assert_equal(seqs[0], (0, 0, 10))  # seq-idx, start-frame, end-frame
    assert_equal(seqs[1], (0, 5, 11))
    assert_equal(seqs[2], (0, 10, 11))
    assert_equal(seqs[3], (1, 0, 10))
    assert_equal(seqs[4], (1, 5, 11))
    assert_equal(seqs[5], (1, 10, 11))


def test_batches_recurrent_1():
    dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=2, seq_len=11)
    dataset.init_seq_order(1)
    dataset.chunk_size = 10
    dataset.chunk_step = 5
    batch_gen = dataset.generate_batches(recurrent_net=True, max_seqs=1, batch_size=20)
    all_batches = []
    " :type: list[Batch] "
    while batch_gen.has_more():
        (batch,) = batch_gen.peek_next_n(1)
        assert_is_instance(batch, Batch)
        print("batch:", batch)
        print("batch seqs:", batch.seqs)
        all_batches.append(batch)
        batch_gen.advance(1)

    # Each batch will have 1 batch-slice (max_seqs) and up to 10 frames (chunk_size).
    # For each seq, we get 3 chunks (chunk_step 5 for 11 frames).
    # Thus, 6 batches.
    assert_equal(len(all_batches), 6)

    assert_equal(all_batches[0].start_seq, 0)
    assert_equal(all_batches[0].end_seq, 1)  # exclusive
    assert_equal(len(all_batches[0].seqs), 1)  # 1 BatchSeqCopyPart
    assert_equal(all_batches[0].seqs[0].seq_idx, 0)
    assert_equal(all_batches[0].seqs[0].seq_start_frame, 0)
    assert_equal(all_batches[0].seqs[0].seq_end_frame, 10)
    assert_equal(all_batches[0].seqs[0].frame_length, 10)
    assert_equal(all_batches[0].seqs[0].batch_slice, 0)
    assert_equal(all_batches[0].seqs[0].batch_frame_offset, 0)

    assert_equal(all_batches[1].start_seq, 0)
    assert_equal(all_batches[1].end_seq, 1)  # exclusive
    assert_equal(len(all_batches[1].seqs), 1)  # 1 BatchSeqCopyPart
    assert_equal(all_batches[1].seqs[0].seq_idx, 0)
    assert_equal(all_batches[1].seqs[0].seq_start_frame, 5)
    assert_equal(all_batches[1].seqs[0].seq_end_frame, 11)
    assert_equal(all_batches[1].seqs[0].frame_length, 6)
    assert_equal(all_batches[1].seqs[0].batch_slice, 0)
    assert_equal(all_batches[1].seqs[0].batch_frame_offset, 0)

    assert_equal(all_batches[2].start_seq, 0)
    assert_equal(all_batches[2].end_seq, 1)  # exclusive
    assert_equal(len(all_batches[2].seqs), 1)  # 1 BatchSeqCopyPart
    assert_equal(all_batches[2].seqs[0].seq_idx, 0)
    assert_equal(all_batches[2].seqs[0].seq_start_frame, 10)
    assert_equal(all_batches[2].seqs[0].seq_end_frame, 11)
    assert_equal(all_batches[2].seqs[0].frame_length, 1)
    assert_equal(all_batches[2].seqs[0].batch_slice, 0)
    assert_equal(all_batches[2].seqs[0].batch_frame_offset, 0)

    assert_equal(all_batches[3].start_seq, 1)
    assert_equal(all_batches[3].end_seq, 2)  # exclusive
    assert_equal(len(all_batches[3].seqs), 1)  # 1 BatchSeqCopyPart
    assert_equal(all_batches[3].seqs[0].seq_idx, 1)
    assert_equal(all_batches[3].seqs[0].seq_start_frame, 0)
    assert_equal(all_batches[3].seqs[0].seq_end_frame, 10)
    assert_equal(all_batches[3].seqs[0].frame_length, 10)
    assert_equal(all_batches[3].seqs[0].batch_slice, 0)
    assert_equal(all_batches[3].seqs[0].batch_frame_offset, 0)

    # ...


def test_batches_non_recurrent_1():
    dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=2, seq_len=11)
    dataset.init_seq_order(1)
    batch_gen = dataset.generate_batches(recurrent_net=False, max_seqs=2, batch_size=5)
    all_batches = []  # type: list[Batch]
    while batch_gen.has_more():
        (batch,) = batch_gen.peek_next_n(1)
        assert_is_instance(batch, Batch)
        print("batch:", batch)
        print("batch seqs:", batch.seqs)
        all_batches.append(batch)
        batch_gen.advance(1)

    # Each batch will have 5 frames (batch_size), not more, i.e. a single seq.
    # There are 2 * 11 frames in total, so 5 batches, because we concat the 2 seqs, in the non-recurrent case.
    assert_equal(len(all_batches), 5)

    assert_equal(all_batches[0].start_seq, 0)
    assert_equal(all_batches[0].end_seq, 1)  # exclusive
    assert_equal(len(all_batches[0].seqs), 1)  # 1 BatchSeqCopyPart
    assert_equal(all_batches[0].seqs[0].seq_idx, 0)
    assert_equal(all_batches[0].seqs[0].seq_start_frame, 0)
    assert_equal(all_batches[0].seqs[0].seq_end_frame, 5)
    assert_equal(all_batches[0].seqs[0].frame_length, 5)
    assert_equal(all_batches[0].seqs[0].batch_slice, 0)
    assert_equal(all_batches[0].seqs[0].batch_frame_offset, 0)

    assert_equal(all_batches[1].start_seq, 0)
    assert_equal(all_batches[1].end_seq, 1)  # exclusive
    assert_equal(len(all_batches[1].seqs), 1)  # 1 BatchSeqCopyPart
    assert_equal(all_batches[1].seqs[0].seq_idx, 0)
    assert_equal(all_batches[1].seqs[0].seq_start_frame, 5)
    assert_equal(all_batches[1].seqs[0].seq_end_frame, 10)
    assert_equal(all_batches[1].seqs[0].frame_length, 5)
    assert_equal(all_batches[1].seqs[0].batch_slice, 0)
    assert_equal(all_batches[1].seqs[0].batch_frame_offset, 0)

    assert_equal(all_batches[2].start_seq, 0)
    assert_equal(all_batches[2].end_seq, 2)  # exclusive. now both seq 0 and 1
    assert_equal(len(all_batches[2].seqs), 2)  # two copies, BatchSeqCopyPart
    assert_equal(all_batches[2].seqs[0].seq_idx, 0)
    assert_equal(all_batches[2].seqs[0].seq_start_frame, 10)
    assert_equal(all_batches[2].seqs[0].seq_end_frame, 11)
    assert_equal(all_batches[2].seqs[0].frame_length, 1)
    assert_equal(all_batches[2].seqs[0].batch_slice, 0)
    assert_equal(all_batches[2].seqs[0].batch_frame_offset, 0)
    assert_equal(all_batches[2].seqs[1].seq_idx, 1)
    assert_equal(all_batches[2].seqs[1].seq_start_frame, 0)
    assert_equal(all_batches[2].seqs[1].seq_end_frame, 4)
    assert_equal(all_batches[2].seqs[1].frame_length, 4)
    assert_equal(all_batches[2].seqs[1].batch_slice, 0)
    assert_equal(all_batches[2].seqs[1].batch_frame_offset, 1)

    assert_equal(all_batches[3].start_seq, 1)
    assert_equal(all_batches[3].end_seq, 2)  # exclusive
    assert_equal(len(all_batches[3].seqs), 1)  # 1 BatchSeqCopyPart
    assert_equal(all_batches[3].seqs[0].seq_idx, 1)
    assert_equal(all_batches[3].seqs[0].seq_start_frame, 4)
    assert_equal(all_batches[3].seqs[0].seq_end_frame, 9)
    assert_equal(all_batches[3].seqs[0].frame_length, 5)
    assert_equal(all_batches[3].seqs[0].batch_slice, 0)
    assert_equal(all_batches[3].seqs[0].batch_frame_offset, 0)

    assert_equal(all_batches[4].start_seq, 1)
    assert_equal(all_batches[4].end_seq, 2)  # exclusive
    assert_equal(len(all_batches[4].seqs), 1)  # 1 BatchSeqCopyPart
    assert_equal(all_batches[4].seqs[0].seq_idx, 1)
    assert_equal(all_batches[4].seqs[0].seq_start_frame, 9)
    assert_equal(all_batches[4].seqs[0].seq_end_frame, 11)
    assert_equal(all_batches[4].seqs[0].frame_length, 2)
    assert_equal(all_batches[4].seqs[0].batch_slice, 0)
    assert_equal(all_batches[4].seqs[0].batch_frame_offset, 0)


def test_batches_context_window():
    context_window = 2
    ctx_lr = context_window - 1
    ctx_left = ctx_lr // 2
    ctx_right = ctx_lr - ctx_left

    dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=1, seq_len=11, context_window=context_window)
    dataset.init_seq_order(1)
    dataset.chunk_size = 5
    dataset.chunk_step = 5
    batch_gen = dataset.generate_batches(recurrent_net=True, max_seqs=1, batch_size=20)
    all_batches = []  # type: list[Batch]
    while batch_gen.has_more():
        (batch,) = batch_gen.peek_next_n(1)
        assert_is_instance(batch, Batch)
        print("batch:", batch)
        print("batch seqs:", batch.seqs)
        all_batches.append(batch)
        batch_gen.advance(1)

    # Each batch will have 1 batch-slice (max_seqs) and up to 10 frames (chunk_size).
    # For each seq, we get 3 chunks (chunk_step 5 for 11 frames).
    # Thus, 3 batches.
    assert_equal(len(all_batches), 3)
    b0, b1, b2 = all_batches
    assert isinstance(b0, Batch)
    assert isinstance(b1, Batch)
    assert isinstance(b2, Batch)

    assert_equal(b0.start_seq, 0)
    assert_equal(b0.end_seq, 1)  # exclusive
    assert_equal(len(b0.seqs), 1)  # 1 BatchSeqCopyPart
    assert_equal(b0.seqs[0].seq_idx, 0)
    assert_equal(b0.seqs[0].seq_start_frame["classes"], 0)
    assert_equal(b0.seqs[0].seq_end_frame["classes"], 5)
    assert_equal(b0.seqs[0].frame_length["classes"], 5)
    assert_equal(b0.seqs[0].seq_start_frame["data"], 0 - ctx_left)
    assert_equal(b0.seqs[0].seq_end_frame["data"], 5 + ctx_right)
    assert_equal(b0.seqs[0].frame_length["data"], 5 + ctx_lr)
    assert_equal(b0.seqs[0].batch_slice, 0)
    assert_equal(b0.seqs[0].batch_frame_offset, 0)

    assert_equal(b1.start_seq, 0)
    assert_equal(b1.end_seq, 1)  # exclusive
    assert_equal(len(b1.seqs), 1)  # 1 BatchSeqCopyPart
    assert_equal(b1.seqs[0].seq_idx, 0)
    assert_equal(b1.seqs[0].seq_start_frame["classes"], 5)
    assert_equal(b1.seqs[0].seq_end_frame["classes"], 10)
    assert_equal(b1.seqs[0].frame_length["classes"], 5)
    assert_equal(b1.seqs[0].seq_start_frame["data"], 5 - ctx_left)
    assert_equal(b1.seqs[0].seq_end_frame["data"], 10 + ctx_right)
    assert_equal(b1.seqs[0].frame_length["data"], 5 + ctx_lr)
    assert_equal(b1.seqs[0].batch_slice, 0)
    assert_equal(b1.seqs[0].batch_frame_offset, 0)

    assert_equal(b2.start_seq, 0)
    assert_equal(b2.end_seq, 1)  # exclusive
    assert_equal(len(b2.seqs), 1)  # 1 BatchSeqCopyPart
    assert_equal(b2.seqs[0].seq_idx, 0)
    assert_equal(b2.seqs[0].seq_start_frame["classes"], 10)
    assert_equal(b2.seqs[0].seq_end_frame["classes"], 11)
    assert_equal(b2.seqs[0].frame_length["classes"], 1)
    assert_equal(b2.seqs[0].seq_start_frame["data"], 10 - ctx_left)
    assert_equal(b2.seqs[0].seq_end_frame["data"], 11 + ctx_right)
    assert_equal(b2.seqs[0].frame_length["data"], 1 + ctx_lr)
    assert_equal(b2.seqs[0].batch_slice, 0)
    assert_equal(b2.seqs[0].batch_frame_offset, 0)


def test_task12ax_window():
    from returnn.datasets.generating import Task12AXDataset

    window = 3
    dataset_kwargs = dict(num_seqs=10)
    dataset1 = Task12AXDataset(**dataset_kwargs)
    dataset2 = Task12AXDataset(window=window, **dataset_kwargs)
    input_dim = dataset1.num_inputs
    dataset1.initialize()
    dataset2.initialize()
    dataset1.init_seq_order(epoch=1)
    dataset2.init_seq_order(epoch=1)
    dataset1.load_seqs(0, 1)
    dataset2.load_seqs(0, 1)
    assert_equal(dataset1.get_data_dim("data"), input_dim)
    assert_equal(dataset2.get_data_dim("data"), input_dim * window)
    data1 = dataset1.get_data(0, "data")
    data2 = dataset2.get_data(0, "data")
    seq_len = data1.shape[0]
    assert_equal(data1.shape, (seq_len, input_dim))
    assert_equal(data2.shape, (seq_len, window * input_dim))
    data2a = data2.reshape(seq_len, window, input_dim)
    print("data1:")
    print(data1)
    print("data2:")
    print(data2)
    print("data1[0]:")
    print(data1[0])
    print("data2[0]:")
    print(data2[0])
    print("data2a[0,0]:")
    print(data2a[0, 0])
    assert_equal(list(data2a[0, 0]), [0] * input_dim)  # zero-padded left
    assert_equal(list(data2a[0, 1]), list(data1[0]))
    assert_equal(list(data2a[0, 2]), list(data1[1]))
    assert_equal(list(data2a[1, 0]), list(data1[0]))
    assert_equal(list(data2a[1, 1]), list(data1[1]))
    assert_equal(list(data2a[1, 2]), list(data1[2]))
    assert_equal(list(data2a[-1, 2]), [0] * input_dim)  # zero-padded right


def test_get_seq_order():
    dataset = Dataset()
    num_seqs = 30

    def get_seq_len(i):
        return i**2 % 17  # some dummy lengths

    for seq_ordering in [
        "default",
        "default_every_n:5",
        "sorted",
        "sorted_reverse",
        "random:3",
        "laplace:3",
        "laplace:.10",
        "sort_bin_shuffle:3",
        "sort_bin_shuffle_x2:.10",
    ]:

        dataset.seq_ordering = seq_ordering

        # test full epoch
        dataset.partition_epoch = 1
        epoch = 3
        seq_index = dataset.get_seq_order_for_epoch(epoch, num_seqs, get_seq_len)

        assert isinstance(seq_index, (list, range, numpy.ndarray))
        assert len(set(seq_index)) == num_seqs  # right number of sequences, no duplicates

        # test partitioned epoch
        partition_epoch = 4
        dataset.partition_epoch = partition_epoch
        all_partitions_seq_index = []
        for epoch in range(1, partition_epoch + 1):
            partition_seq_index = dataset.get_seq_order_for_epoch(epoch, num_seqs, get_seq_len)
            all_partitions_seq_index += list(partition_seq_index)

        # Make sure partitions combined result in full epoch. This tests the random seed of Dataset which should be
        # fixed across partitions.
        assert set(all_partitions_seq_index) == set(seq_index)


@contextlib.contextmanager
def create_ogg_zip_txt_only_dataset_opts(*, text: str = "hello world", seq_tag: str = "sequence0.wav"):
    """create OggZipDataset dict using temp data, consisting of a single sequence with text only"""
    import zipfile

    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp_zip_file, tempfile.NamedTemporaryFile(
        suffix=".txt"
    ) as tmp_vocab_file:
        with zipfile.ZipFile(tmp_zip_file.name, "w") as zip_file:
            zip_file.writestr(
                os.path.basename(tmp_zip_file.name)[:-4] + ".txt",
                repr([{"text": text, "duration": 2.3, "file": seq_tag}]),
            )
        vocab = {"@": 2, " ": 1, ".": 0}
        vocab.update({chr(i): i - ord("a") + 3 for i in range(ord("a"), ord("z") + 1)})
        tmp_vocab_file.write(repr(vocab).encode("utf8"))
        tmp_vocab_file.flush()

        yield {
            "class": "OggZipDataset",
            "path": tmp_zip_file.name,
            "audio": None,
            "targets": {"class": "CharacterTargets", "vocab_file": tmp_vocab_file.name, "seq_postfix": [0]},
        }


@contextlib.contextmanager
def create_ogg_zip_txt_only_dataset(*, text: str = "hello world", seq_tag: str = "sequence0.wav"):
    """create OggZipDataset via :func:`create_ogg_zip_txt_only_dataset_opts`"""
    from returnn.datasets.audio import OggZipDataset

    with create_ogg_zip_txt_only_dataset_opts(text=text, seq_tag=seq_tag) as opts:
        dataset = init_dataset(opts)
        assert isinstance(dataset, OggZipDataset)
        yield dataset


@contextlib.contextmanager
def create_ogg_zip_txt_only_dataset_mult_seqs(*, seed: int = 1, num_seqs: int = 100, max_seq_len: int = 100):
    """create OggZipDataset"""
    import zipfile
    from returnn.datasets.audio import OggZipDataset

    rnd = numpy.random.RandomState(seed)

    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp_zip_file, tempfile.NamedTemporaryFile(
        suffix=".txt"
    ) as tmp_vocab_file:
        vocab = {"@": 2, " ": 1, ".": 0}
        vocab.update({chr(i): i - ord("a") + 3 for i in range(ord("a"), ord("z") + 1)})
        tmp_vocab_file.write(repr(vocab).encode("utf8"))
        tmp_vocab_file.flush()

        seqs = []
        for i in range(num_seqs):
            text = "".join(rnd.choice(list(vocab.keys())) for _ in range(rnd.randint(1, max_seq_len + 1)))
            seqs.append({"text": text, "duration": rnd.uniform(1.0, 5.0), "file": f"seq{i}.wav"})

        with zipfile.ZipFile(tmp_zip_file.name, "w") as zip_file:
            zip_file.writestr(
                os.path.basename(tmp_zip_file.name)[:-4] + ".txt",
                repr(seqs),
            )

        opts = {
            "class": "OggZipDataset",
            "path": tmp_zip_file.name,
            "audio": None,
            "targets": {"class": "CharacterTargets", "vocab_file": tmp_vocab_file.name, "seq_postfix": []},
        }
        dataset = init_dataset(opts)
        assert isinstance(dataset, OggZipDataset)
        yield dataset


def test_OggZipDataset():
    from returnn.datasets.audio import OggZipDataset

    _demo_txt = "some utterance text"

    with create_ogg_zip_txt_only_dataset(text=_demo_txt) as dataset:
        assert isinstance(dataset, OggZipDataset)
        assert dataset.have_seqs()
        dataset.init_seq_order(epoch=1)
        dataset.load_seqs(0, 1)
        raw = dataset.get_data(0, "raw")
        orth = dataset.get_data(0, "orth")
        classes = dataset.get_data(0, "classes")
        print("raw:", raw)
        print("orth:", orth)
        print("classes:", classes)
        assert isinstance(raw, numpy.ndarray) and raw.dtype.name.startswith("str") and raw.shape == ()
        raw_ = raw.item()
        assert isinstance(raw_, str) and raw_ == _demo_txt
        assert isinstance(orth, numpy.ndarray) and orth.dtype == numpy.uint8 and orth.ndim == 1
        orth_ = orth.tostring()
        assert orth_.decode("utf8") == _demo_txt
        assert isinstance(classes, numpy.ndarray) and classes.dtype == numpy.int32 and classes.ndim == 1
        classes_ = "".join([dataset.targets.id_to_label(c) for c in classes])
        assert classes_ == _demo_txt + "."


def test_MetaDataset():
    _demo_txt = "some utterance text"

    with create_ogg_zip_txt_only_dataset_opts(text=_demo_txt) as sub_ds_opts:
        meta_ds_opts = {
            "class": "MetaDataset",
            "datasets": {"sub": sub_ds_opts},
            "data_map": {"classes": ("sub", "classes")},
            "seq_order_control_dataset": "sub",
        }
        dataset = init_dataset(meta_ds_opts)
        assert dataset.have_seqs()
        dataset.init_seq_order(epoch=1)
        dataset.load_seqs(0, 1)
        classes = dataset.get_data(0, "classes")
        print("classes:", classes)
        assert isinstance(classes, numpy.ndarray) and classes.dtype == numpy.int32 and classes.ndim == 1
        assert len(classes) == len(_demo_txt) + 1


def test_MapDatasetWrapper():
    from returnn.datasets.map import MapDatasetBase

    class _MyCustomMapDataset(MapDatasetBase):
        def __init__(self):
            super().__init__(data_types={"data": {"shape": (None, 3)}})

        def __len__(self):
            return 1

        def __getitem__(self, item):
            return {"data": numpy.zeros((5, 3))}

    ds = init_dataset({"class": "MapDatasetWrapper", "map_dataset": _MyCustomMapDataset})
    (res,) = dummy_iter_dataset(ds)
    assert isinstance(res, DatasetSeq)
    assert res.features["data"].shape == (5, 3)


def test_ConcatFilesDataset_get_files_per_sub_epochs():
    from returnn.datasets.concat_files import ConcatFilesDataset

    def _test(sizes: List[int], partition_epoch: int, expected: List[List[int]]):
        files = [f"file-{i}" for i in range(len(sizes))]
        file_sizes = {f: s for f, s in zip(files, sizes)}
        res = ConcatFilesDataset._get_files_per_sub_epochs(
            partition_epoch=partition_epoch, file_sizes=file_sizes, files_order=files
        )
        assert all(res) and len(res) == partition_epoch
        assert set(sum(res, [])) == set(files)
        res_ = [[file_sizes[fn] for fn in sub_epoch] for sub_epoch in res]
        assert res_ == expected

    _test([1, 1, 78, 120], 4, [[1], [1], [78], [120]])
    _test([1, 1, 1, 56, 141], 5, [[1], [1], [1], [56], [141]])
    _test([1, 1, 1, 56, 141], 4, [[1, 1], [1], [56], [141]])
    _test([5, 5] + [10] * 7, 5, [[5, 5, 10], [10, 10], [10, 10], [10], [10]])


def test_ConcatFilesDataset():
    from returnn.datasets.concat_files import ConcatFilesDataset
    from test_HDFDataset import generate_hdf_from_other

    # Create a few HDF files such that we can easily verify the data later.
    hdf_files = []
    num_hdf_files = 20
    seq_len = 5
    num_seqs = 5
    max_idx = num_hdf_files * num_seqs * seq_len
    n_tgt_dim = max_idx
    for hdf_idx in range(num_hdf_files):
        hdf_files.append(
            generate_hdf_from_other(
                {
                    "class": "StaticDataset",
                    "data": [
                        {
                            "classes": numpy.array(
                                [hdf_idx * num_seqs * seq_len + seq_idx * seq_len + t for t in range(seq_len)],
                                dtype="int32",
                            ),
                        }
                        for seq_idx in range(num_seqs)
                    ],
                    "output_dim": {"classes": [n_tgt_dim, 1]},
                },
                suffix=f"-{hdf_idx}.hdf",
                use_cache=False,
            )
        )

    # Test to load all together.
    single_hdf_dataset = init_dataset({"class": "HDFDataset", "files": hdf_files})
    total_num_seqs = 0
    for seq in dummy_iter_dataset(single_hdf_dataset):
        assert seq.seq_idx == total_num_seqs
        data = seq.features["classes"]
        assert data.shape == (seq_len,)
        assert data.tolist() == list(range(seq.seq_idx * seq_len, (seq.seq_idx + 1) * seq_len))
        total_num_seqs += 1
    assert total_num_seqs == num_hdf_files * num_seqs

    # Test to load via ConcatFilesDataset.

    def _get_sub_epoch_dataset(files_subepoch: List[str]) -> Dict[str, Any]:
        return {"class": "HDFDataset", "files": files_subepoch, "seq_ordering": "default"}

    partition_epoch = 5
    assert num_hdf_files % partition_epoch == 0  # just for easier testing here
    concat_dataset = init_dataset(
        {
            "class": "ConcatFilesDataset",
            "files": hdf_files,
            "get_sub_epoch_dataset": _get_sub_epoch_dataset,
            "partition_epoch": partition_epoch,
        }
    )
    assert isinstance(concat_dataset, ConcatFilesDataset)
    assert concat_dataset.get_data_keys() == ["classes"]
    num_hdfs_per_part = num_hdf_files // partition_epoch
    global_seq_idx = 0
    for sub_epoch in range(1, partition_epoch + 1):
        print(f"Sub-epoch {sub_epoch}...")
        concat_dataset.init_seq_order(sub_epoch)
        if sub_epoch == 1:
            assert concat_dataset._files_order_cache == {
                0: [hdf_files[ep * num_hdfs_per_part : (ep + 1) * num_hdfs_per_part] for ep in range(partition_epoch)]
            }
        assert (
            concat_dataset._workers[sub_epoch].dataset_dict["files"]
            == hdf_files[(sub_epoch - 1) * num_hdfs_per_part : sub_epoch * num_hdfs_per_part]
        )
        # We preload one sub-epoch.
        assert set(concat_dataset._workers.keys()) == {sub_epoch, sub_epoch + 1}  # cur sub-epoch + next sub-epoch
        for ep, worker in concat_dataset._workers.items():
            assert worker.worker_proc.is_alive()
        next_part_idx = sub_epoch % partition_epoch  # wrap around at the end
        assert (
            concat_dataset._workers[sub_epoch + 1].dataset_dict["files"]
            == hdf_files[next_part_idx * num_hdfs_per_part : (next_part_idx + 1) * num_hdfs_per_part]
        )
        local_seq_idx = 0
        while concat_dataset.is_less_than_num_seqs(local_seq_idx):
            print(f"Sub-epoch {sub_epoch}, seq {local_seq_idx} (global seq {global_seq_idx})...")
            concat_dataset.load_seqs(local_seq_idx, local_seq_idx + 1)
            data = concat_dataset.get_data(local_seq_idx, "classes")
            assert data.shape == (seq_len,)
            assert data.tolist() == list(range(global_seq_idx * seq_len, (global_seq_idx + 1) * seq_len))
            local_seq_idx += 1
            global_seq_idx += 1
    assert global_seq_idx == num_hdf_files * num_seqs


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
