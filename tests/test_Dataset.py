# Also see test_SprintDataset.py.

from __future__ import annotations

from typing import Any, Iterator, List, Dict, Optional
import os
import sys
import _setup_test_env  # noqa
import unittest
import numpy
import tempfile
import contextlib
from returnn.datasets.generating import Task12AXDataset, DummyDataset, DummyDatasetMultipleSequenceLength
from returnn.engine.batch import Batch
from returnn.datasets.basic import Dataset, DatasetSeq, init_dataset
from returnn.util.basic import NumbersDict
from returnn.util import better_exchook


def dummy_iter_dataset(dataset: Dataset, *, epoch: int = 1) -> List[DatasetSeq]:
    """
    :param Dataset dataset:
    :return: seqs
    """
    dataset.init_seq_order(epoch=epoch)
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
    assert len(seqs) == 2
    assert seqs[0] == (0, 0, 11)  # seq-idx, start-frame, end-frame
    assert seqs[1] == (1, 0, 11)


def test_iterate_seqs_chunking_1():
    dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=2, seq_len=11)
    dataset.chunk_step = 5
    dataset.chunk_size = 10
    dataset.init_seq_order(1)
    seqs = list(dataset.iterate_seqs())
    for s in seqs:
        print(s)
    assert len(seqs) == 6
    assert seqs[0] == (0, 0, 10)  # seq-idx, start-frame, end-frame
    assert seqs[1] == (0, 5, 11)
    assert seqs[2] == (0, 10, 11)
    assert seqs[3] == (1, 0, 10)
    assert seqs[4] == (1, 5, 11)
    assert seqs[5] == (1, 10, 11)


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
    assert len(seqs) == 8
    assert seqs[0] == (0, NumbersDict({"data": 0, "classes": 0}), NumbersDict({"data": 12, "classes": 6}))
    assert seqs[1] == (0, NumbersDict({"data": 6, "classes": 3}), NumbersDict({"data": 18, "classes": 9}))
    assert seqs[2] == (0, NumbersDict({"data": 12, "classes": 6}), NumbersDict({"data": 24, "classes": 12}))
    assert seqs[3] == (0, NumbersDict({"data": 18, "classes": 9}), NumbersDict({"data": 24, "classes": 12}))
    assert seqs[4] == (1, NumbersDict({"data": 0, "classes": 0}), NumbersDict({"data": 12, "classes": 6}))
    assert seqs[5] == (1, NumbersDict({"data": 6, "classes": 3}), NumbersDict({"data": 18, "classes": 9}))
    assert seqs[6] == (1, NumbersDict({"data": 12, "classes": 6}), NumbersDict({"data": 24, "classes": 12}))
    assert seqs[7] == (1, NumbersDict({"data": 18, "classes": 9}), NumbersDict({"data": 24, "classes": 12}))


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
    assert len(seqs) == 6
    assert seqs[0] == (0, 0, 10)  # seq-idx, start-frame, end-frame
    assert seqs[1] == (0, 5, 11)
    assert seqs[2] == (0, 10, 11)
    assert seqs[3] == (1, 0, 10)
    assert seqs[4] == (1, 5, 11)
    assert seqs[5] == (1, 10, 11)


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
        assert isinstance(batch, Batch)
        print("batch:", batch)
        print("batch seqs:", batch.seqs)
        all_batches.append(batch)
        batch_gen.advance(1)

    # Each batch will have 1 batch-slice (max_seqs) and up to 10 frames (chunk_size).
    # For each seq, we get 3 chunks (chunk_step 5 for 11 frames).
    # Thus, 6 batches.
    assert len(all_batches) == 6

    assert all_batches[0].start_seq == 0
    assert all_batches[0].end_seq == 1  # exclusive
    assert len(all_batches[0].seqs) == 1  # 1 BatchSeqCopyPart
    assert all_batches[0].seqs[0].seq_idx == 0
    assert all_batches[0].seqs[0].seq_start_frame == 0
    assert all_batches[0].seqs[0].seq_end_frame == 10
    assert all_batches[0].seqs[0].frame_length == 10
    assert all_batches[0].seqs[0].batch_slice == 0
    assert all_batches[0].seqs[0].batch_frame_offset == 0

    assert all_batches[1].start_seq == 0
    assert all_batches[1].end_seq == 1  # exclusive
    assert len(all_batches[1].seqs) == 1  # 1 BatchSeqCopyPart
    assert all_batches[1].seqs[0].seq_idx == 0
    assert all_batches[1].seqs[0].seq_start_frame == 5
    assert all_batches[1].seqs[0].seq_end_frame == 11
    assert all_batches[1].seqs[0].frame_length == 6
    assert all_batches[1].seqs[0].batch_slice == 0
    assert all_batches[1].seqs[0].batch_frame_offset == 0

    assert all_batches[2].start_seq == 0
    assert all_batches[2].end_seq == 1  # exclusive
    assert len(all_batches[2].seqs) == 1  # 1 BatchSeqCopyPart
    assert all_batches[2].seqs[0].seq_idx == 0
    assert all_batches[2].seqs[0].seq_start_frame == 10
    assert all_batches[2].seqs[0].seq_end_frame == 11
    assert all_batches[2].seqs[0].frame_length == 1
    assert all_batches[2].seqs[0].batch_slice == 0
    assert all_batches[2].seqs[0].batch_frame_offset == 0

    assert all_batches[3].start_seq == 1
    assert all_batches[3].end_seq == 2  # exclusive
    assert len(all_batches[3].seqs) == 1  # 1 BatchSeqCopyPart
    assert all_batches[3].seqs[0].seq_idx == 1
    assert all_batches[3].seqs[0].seq_start_frame == 0
    assert all_batches[3].seqs[0].seq_end_frame == 10
    assert all_batches[3].seqs[0].frame_length == 10
    assert all_batches[3].seqs[0].batch_slice == 0
    assert all_batches[3].seqs[0].batch_frame_offset == 0

    # ...


def test_batches_non_recurrent_1():
    dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=2, seq_len=11)
    dataset.init_seq_order(1)
    batch_gen = dataset.generate_batches(recurrent_net=False, max_seqs=2, batch_size=5)
    all_batches = []  # type: list[Batch]
    while batch_gen.has_more():
        (batch,) = batch_gen.peek_next_n(1)
        assert isinstance(batch, Batch)
        print("batch:", batch)
        print("batch seqs:", batch.seqs)
        all_batches.append(batch)
        batch_gen.advance(1)

    # Each batch will have 5 frames (batch_size), not more, i.e. a single seq.
    # There are 2 * 11 frames in total, so 5 batches, because we concat the 2 seqs, in the non-recurrent case.
    assert len(all_batches) == 5

    assert all_batches[0].start_seq == 0
    assert all_batches[0].end_seq == 1  # exclusive
    assert len(all_batches[0].seqs) == 1  # 1 BatchSeqCopyPart
    assert all_batches[0].seqs[0].seq_idx == 0
    assert all_batches[0].seqs[0].seq_start_frame == 0
    assert all_batches[0].seqs[0].seq_end_frame == 5
    assert all_batches[0].seqs[0].frame_length == 5
    assert all_batches[0].seqs[0].batch_slice == 0
    assert all_batches[0].seqs[0].batch_frame_offset == 0

    assert all_batches[1].start_seq == 0
    assert all_batches[1].end_seq == 1  # exclusive
    assert len(all_batches[1].seqs) == 1  # 1 BatchSeqCopyPart
    assert all_batches[1].seqs[0].seq_idx == 0
    assert all_batches[1].seqs[0].seq_start_frame == 5
    assert all_batches[1].seqs[0].seq_end_frame == 10
    assert all_batches[1].seqs[0].frame_length == 5
    assert all_batches[1].seqs[0].batch_slice == 0
    assert all_batches[1].seqs[0].batch_frame_offset == 0

    assert all_batches[2].start_seq == 0
    assert all_batches[2].end_seq == 2  # exclusive. now both seq 0 and 1
    assert len(all_batches[2].seqs) == 2  # two copies, BatchSeqCopyPart
    assert all_batches[2].seqs[0].seq_idx == 0
    assert all_batches[2].seqs[0].seq_start_frame == 10
    assert all_batches[2].seqs[0].seq_end_frame == 11
    assert all_batches[2].seqs[0].frame_length == 1
    assert all_batches[2].seqs[0].batch_slice == 0
    assert all_batches[2].seqs[0].batch_frame_offset == 0
    assert all_batches[2].seqs[1].seq_idx == 1
    assert all_batches[2].seqs[1].seq_start_frame == 0
    assert all_batches[2].seqs[1].seq_end_frame == 4
    assert all_batches[2].seqs[1].frame_length == 4
    assert all_batches[2].seqs[1].batch_slice == 0
    assert all_batches[2].seqs[1].batch_frame_offset == 1

    assert all_batches[3].start_seq == 1
    assert all_batches[3].end_seq == 2  # exclusive
    assert len(all_batches[3].seqs) == 1  # 1 BatchSeqCopyPart
    assert all_batches[3].seqs[0].seq_idx == 1
    assert all_batches[3].seqs[0].seq_start_frame == 4
    assert all_batches[3].seqs[0].seq_end_frame == 9
    assert all_batches[3].seqs[0].frame_length == 5
    assert all_batches[3].seqs[0].batch_slice == 0
    assert all_batches[3].seqs[0].batch_frame_offset == 0

    assert all_batches[4].start_seq == 1
    assert all_batches[4].end_seq == 2  # exclusive
    assert len(all_batches[4].seqs) == 1  # 1 BatchSeqCopyPart
    assert all_batches[4].seqs[0].seq_idx == 1
    assert all_batches[4].seqs[0].seq_start_frame == 9
    assert all_batches[4].seqs[0].seq_end_frame == 11
    assert all_batches[4].seqs[0].frame_length == 2
    assert all_batches[4].seqs[0].batch_slice == 0
    assert all_batches[4].seqs[0].batch_frame_offset == 0


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
        assert isinstance(batch, Batch)
        print("batch:", batch)
        print("batch seqs:", batch.seqs)
        all_batches.append(batch)
        batch_gen.advance(1)

    # Each batch will have 1 batch-slice (max_seqs) and up to 10 frames (chunk_size).
    # For each seq, we get 3 chunks (chunk_step 5 for 11 frames).
    # Thus, 3 batches.
    assert len(all_batches) == 3
    b0, b1, b2 = all_batches
    assert isinstance(b0, Batch)
    assert isinstance(b1, Batch)
    assert isinstance(b2, Batch)

    assert b0.start_seq == 0
    assert b0.end_seq == 1  # exclusive
    assert len(b0.seqs) == 1  # 1 BatchSeqCopyPart
    assert b0.seqs[0].seq_idx == 0
    assert b0.seqs[0].seq_start_frame["classes"] == 0
    assert b0.seqs[0].seq_end_frame["classes"] == 5
    assert b0.seqs[0].frame_length["classes"] == 5
    assert b0.seqs[0].seq_start_frame["data"] == 0 - ctx_left
    assert b0.seqs[0].seq_end_frame["data"] == 5 + ctx_right
    assert b0.seqs[0].frame_length["data"] == 5 + ctx_lr
    assert b0.seqs[0].batch_slice == 0
    assert b0.seqs[0].batch_frame_offset == 0

    assert b1.start_seq == 0
    assert b1.end_seq == 1  # exclusive
    assert len(b1.seqs) == 1  # 1 BatchSeqCopyPart
    assert b1.seqs[0].seq_idx == 0
    assert b1.seqs[0].seq_start_frame["classes"] == 5
    assert b1.seqs[0].seq_end_frame["classes"] == 10
    assert b1.seqs[0].frame_length["classes"] == 5
    assert b1.seqs[0].seq_start_frame["data"] == 5 - ctx_left
    assert b1.seqs[0].seq_end_frame["data"] == 10 + ctx_right
    assert b1.seqs[0].frame_length["data"] == 5 + ctx_lr
    assert b1.seqs[0].batch_slice == 0
    assert b1.seqs[0].batch_frame_offset == 0

    assert b2.start_seq == 0
    assert b2.end_seq == 1  # exclusive
    assert len(b2.seqs) == 1  # 1 BatchSeqCopyPart
    assert b2.seqs[0].seq_idx == 0
    assert b2.seqs[0].seq_start_frame["classes"] == 10
    assert b2.seqs[0].seq_end_frame["classes"] == 11
    assert b2.seqs[0].frame_length["classes"] == 1
    assert b2.seqs[0].seq_start_frame["data"] == 10 - ctx_left
    assert b2.seqs[0].seq_end_frame["data"] == 11 + ctx_right
    assert b2.seqs[0].frame_length["data"] == 1 + ctx_lr
    assert b2.seqs[0].batch_slice == 0
    assert b2.seqs[0].batch_frame_offset == 0


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
    assert dataset1.get_data_dim("data") == input_dim
    assert dataset2.get_data_dim("data") == input_dim * window
    data1 = dataset1.get_data(0, "data")
    data2 = dataset2.get_data(0, "data")
    seq_len = data1.shape[0]
    assert data1.shape == (seq_len, input_dim)
    assert data2.shape == (seq_len, window * input_dim)
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
    assert list(data2a[0, 0]) == [0] * input_dim  # zero-padded left
    assert list(data2a[0, 1]) == list(data1[0])
    assert list(data2a[0, 2]) == list(data1[1])
    assert list(data2a[1, 0]) == list(data1[0])
    assert list(data2a[1, 1]) == list(data1[1])
    assert list(data2a[1, 2]) == list(data1[2])
    assert list(data2a[-1, 2]) == [0] * input_dim  # zero-padded right


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


def test_get_seq_order_laplace_reference():
    num_seqs = 3023
    rnd = numpy.random.RandomState(42)
    seq_lens = rnd.randint(1, 23, size=[num_seqs])
    get_seq_len = seq_lens.__getitem__

    dataset = Dataset()
    dataset.epoch = 1
    dataset.seq_ordering = "laplace:.100"

    seq_index_ = dataset.get_seq_order_for_epoch(epoch=1, num_seqs=num_seqs, get_seq_len=get_seq_len)
    assert isinstance(seq_index_, (list, range, numpy.ndarray))
    assert len(set(seq_index_)) == num_seqs  # right number of sequences, no duplicates
    print("current implementation returns seq_lens[seq_index]:", list(seq_lens[seq_index_]))

    tmp = dataset.seq_ordering.split(":")[1:]
    if len(tmp) == 0:
        bins = 2
    else:
        if tmp[0].startswith("."):  # starting with "." -> approx chunk size (num of seqs in one bin)
            bins = max(num_seqs // int(tmp[0][1:]), 2)
        else:  # the number of bins
            bins = int(tmp[0])
    assert len(tmp) <= 1
    rnd_seed = dataset.epoch
    random_generator = numpy.random.RandomState(rnd_seed)
    seq_index = random_generator.permutation(num_seqs)
    out_index = []
    for i in range(bins):
        if i == bins - 1:
            part = seq_index[i * len(seq_index) // bins :].tolist()
        else:
            part = seq_index[i * len(seq_index) // bins : (i + 1) * len(seq_index) // bins].tolist()
        part.sort(key=get_seq_len, reverse=(i % 2 == 1))
        out_index += part
    seq_index = out_index
    print("reference seq_lens[seq_index]:", list(seq_lens[seq_index]))

    assert len(seq_index) == num_seqs == len(seq_index_)
    assert seq_index == list(seq_index_)


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


def test_LmDataset_char_based():
    from returnn.datasets.lm import LmDataset

    with tempfile.NamedTemporaryFile("wt", suffix=".txt") as txt_file, tempfile.NamedTemporaryFile(
        "wt", suffix=".syms"
    ) as orth_syms_file:
        txt_file.write("Hello world\n")
        txt_file.write("Next line\n")
        txt_file.flush()
        orth_syms_file.write("[END]\n")
        covered = set()
        for c in open(txt_file.name, "r").read():
            if c not in covered and c != "\n":
                orth_syms_file.write(c + "\n")
                covered.add(c)
        orth_syms_file.flush()

        dataset = init_dataset(
            {
                "class": "LmDataset",
                "corpus_file": txt_file.name,
                "orth_symbols_file": orth_syms_file.name,
            }
        )
        assert isinstance(dataset, LmDataset)
        dataset.init_seq_order(epoch=1)
        dataset.load_seqs(0, 2)
        orth = dataset.get_data(0, "data")
        assert orth.tolist() == [1, 2, 3, 3, 4, 5, 6, 4, 7, 3, 8, 0]
        orth = dataset.get_data(1, "data")
        assert orth.tolist() == [9, 2, 10, 11, 5, 3, 12, 13, 2, 0]
        assert not dataset.is_less_than_num_seqs(2)


def test_LmDataset_word_based():
    from returnn.datasets.lm import LmDataset

    with tempfile.NamedTemporaryFile("wt", suffix=".txt") as txt_file, tempfile.NamedTemporaryFile(
        "wt", suffix=".syms"
    ) as orth_syms_file:
        txt_file.write("Hello world\n")
        txt_file.write("Next line\n")
        txt_file.flush()
        orth_syms_file.write("[END]\n")
        orth_syms_file.write("Hello\n")
        orth_syms_file.write("world\n")
        orth_syms_file.write("Next\n")
        orth_syms_file.write("line\n")
        orth_syms_file.flush()

        dataset = init_dataset(
            {
                "class": "LmDataset",
                "corpus_file": txt_file.name,
                "word_based": True,
                "orth_symbols_file": orth_syms_file.name,
            }
        )
        assert isinstance(dataset, LmDataset)
        dataset.init_seq_order(epoch=1)
        dataset.load_seqs(0, 2)
        orth = dataset.get_data(0, "data")
        assert orth.tolist() == [1, 2, 0]
        orth = dataset.get_data(1, "data")
        assert orth.tolist() == [3, 4, 0]
        assert not dataset.is_less_than_num_seqs(2)


def test_LmDataset_vocab_based():
    from returnn.datasets.lm import LmDataset

    with tempfile.NamedTemporaryFile("wt", suffix=".txt") as txt_file, tempfile.NamedTemporaryFile(
        "wt", suffix=".syms"
    ) as orth_syms_file:
        txt_file.write("Hello world\n")
        txt_file.write("Next line\n")
        txt_file.flush()
        orth_syms_file.write(
            repr(
                {
                    "[END]": 0,
                    "Hello": 1,
                    "world": 2,
                    "Next": 3,
                    "line": 4,
                }
            )
        )
        orth_syms_file.write("\n")
        orth_syms_file.flush()

        dataset = init_dataset(
            {
                "class": "LmDataset",
                "corpus_file": txt_file.name,
                "orth_vocab": {
                    "class": "SamplingBytePairEncoding",  # just as an example
                    "vocab_file": orth_syms_file.name,
                    "breadth_prob": 0.0,
                    "unknown_label": None,
                },
            }
        )
        assert isinstance(dataset, LmDataset)
        dataset.init_seq_order(epoch=1)
        dataset.load_seqs(0, 2)
        orth = dataset.get_data(0, "data")
        assert orth.tolist() == [1, 2]
        orth = dataset.get_data(1, "data")
        assert orth.tolist() == [3, 4]
        assert not dataset.is_less_than_num_seqs(2)


def test_LmDataset_pickle():
    import pickle
    from returnn.datasets.lm import LmDataset

    with tempfile.NamedTemporaryFile("wt", suffix=".txt") as txt_file, tempfile.NamedTemporaryFile(
        "wt", suffix=".syms"
    ) as orth_syms_file:
        txt_file.write("Hello world\n")
        txt_file.write("Next line\n")
        txt_file.flush()
        orth_syms_file.write(repr({"[END]": 0, "Hello": 1, "world": 2, "Next": 3, "line": 4}))
        orth_syms_file.write("\n")
        orth_syms_file.flush()

        dataset = init_dataset(
            {
                "class": "LmDataset",
                "corpus_file": txt_file.name,
                "orth_vocab": {
                    "vocab_file": orth_syms_file.name,
                    "unknown_label": None,
                },
            }
        )
        assert isinstance(dataset, LmDataset)
        assert dataset._orths_offsets_and_lens is None  # not yet loaded, will be lazily loaded

        s = pickle.dumps(dataset)
        dataset = pickle.loads(s)
        assert isinstance(dataset, LmDataset)
        assert dataset._orths_offsets_and_lens is None  # not yet loaded, will be lazily loaded

        dataset.init_seq_order(epoch=1)
        assert dataset._orths_offsets_and_lens is not None  # loaded now
        dataset.load_seqs(0, 2)
        orth = dataset.get_data(0, "data")
        assert orth.tolist() == [1, 2]
        orth = dataset.get_data(1, "data")
        assert orth.tolist() == [3, 4]
        assert not dataset.is_less_than_num_seqs(2)


def test_LmDataset_multiple_and_gzipped():
    import pickle
    import gzip
    from returnn.datasets.lm import LmDataset

    with tempfile.NamedTemporaryFile("wb", suffix=".txt.gz") as gz_txt_file, tempfile.NamedTemporaryFile(
        "wb", suffix=".txt.gz"
    ) as gz_txt_file2, tempfile.NamedTemporaryFile("wt", suffix=".txt") as txt_file3, tempfile.NamedTemporaryFile(
        "wt", suffix=".syms"
    ) as orth_syms_file:
        with gzip.GzipFile(gz_txt_file.name, "wb") as txt_file:
            txt_file.write(b"Hello world\n")
            txt_file.write(b"Next line\n")
        with gzip.GzipFile(gz_txt_file2.name, "wb") as txt_file2:
            txt_file2.write(b"Next world\n")
        txt_file3.write("Hello line\n")
        txt_file3.flush()
        orth_syms_file.write("[END]\n")
        covered = set()
        for c in gzip.open(gz_txt_file.name, "rt").read():
            if c not in covered and c != "\n":
                orth_syms_file.write(c + "\n")
                covered.add(c)
        orth_syms_file.flush()

        dataset = init_dataset(
            {
                "class": "LmDataset",
                "corpus_file": [gz_txt_file.name, gz_txt_file2.name, txt_file3.name],
                "orth_symbols_file": orth_syms_file.name,
            }
        )
        assert isinstance(dataset, LmDataset)
        s = pickle.dumps(dataset)
        dataset = pickle.loads(s)
        assert isinstance(dataset, LmDataset)

        dataset.init_seq_order(epoch=1)
        dataset.load_seqs(0, 4)
        orth = dataset.get_data(0, "data")
        assert orth.tolist() == [1, 2, 3, 3, 4, 5, 6, 4, 7, 3, 8, 0]
        orth = dataset.get_data(1, "data")
        assert orth.tolist() == [9, 2, 10, 11, 5, 3, 12, 13, 2, 0]
        orth = dataset.get_data(2, "data")
        assert orth.tolist() == [9, 2, 10, 11, 5, 6, 4, 7, 3, 8, 0]
        orth = dataset.get_data(3, "data")
        assert orth.tolist() == [1, 2, 3, 3, 4, 5, 3, 12, 13, 2, 0]
        assert not dataset.is_less_than_num_seqs(4)


def test_LmDataset_sorted():
    # https://github.com/rwth-i6/returnn/commit/660b07e8b766de3f2148169e15818a8ea001b4bb
    from returnn.datasets.lm import LmDataset

    with tempfile.NamedTemporaryFile("wt", suffix=".txt") as txt_file:
        txt_file.write("Hello world\n")
        txt_file.write("Next line\n")
        txt_file.write("Shorty\n")
        txt_file.write("Longer than all the others\n")
        txt_file.flush()

        dataset = init_dataset(
            {
                "class": "LmDataset",
                "corpus_file": txt_file.name,
                "orth_vocab": {"class": "Utf8ByteTargets"},
                "seq_ordering": "sorted",
            }
        )
        assert isinstance(dataset, LmDataset)
        dataset.init_seq_order(epoch=1)
        dataset.load_seqs(0, 4)
        orth = dataset.get_data(0, "data")
        assert (orth == numpy.array(list("Shorty".encode("utf8")))).all()
        orth = dataset.get_data(1, "data")
        assert (orth == numpy.array(list("Next line".encode("utf8")))).all()
        orth = dataset.get_data(2, "data")
        assert (orth == numpy.array(list("Hello world".encode("utf8")))).all()
        orth = dataset.get_data(3, "data")
        assert (orth == numpy.array(list("Longer than all the others".encode("utf8")))).all()
        assert not dataset.is_less_than_num_seqs(4)


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


def test_DistributeFilesDataset_distribute_evenly_by_size():
    from returnn.datasets.distrib_files import DistributeFilesDataset

    def _test(
        sizes: List[int], partition_epoch: int, expected: List[List[int]], files_order: Optional[List[str]] = None
    ):
        files = files_order
        if files is None:
            files = [f"file-{i}" for i in range(len(sizes))]
        file_sizes = {f: s for f, s in zip(files, sizes)}
        res = DistributeFilesDataset._distribute_evenly_by_size(
            num_bins=partition_epoch, file_sizes=file_sizes, files_order=files
        )
        assert all(res) and len(res) == partition_epoch
        assert set(sum(res, [])) == set(files)
        res_ = [[file_sizes[fn] for fn in sub_epoch] for sub_epoch in res]
        assert res_ == expected

    _test([1, 10, 1, 1], 3, [[1], [10], [1, 1]])
    _test([1, 1, 78, 120], 4, [[1], [1], [78], [120]])
    _test([1, 1, 1, 56, 141], 5, [[1], [1], [1], [56], [141]])
    _test([1, 1, 1, 56, 141], 4, [[1, 1], [1], [56], [141]])
    _test([5, 5] + [10] * 7, 5, [[5, 5, 10], [10, 10], [10], [10], [10, 10]])
    _test(
        [1] * 29,
        8,
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    )
    _test([1], 5, [[1, 1]] * 5, files_order=["file"] * 10)  # test duplicate files

    def _test_stddev(sizes: List[int], partition_epoch: int, max_stddev_percent: float):
        avg_per_bin = sum(sizes) / partition_epoch
        files = [f"file-{i}" for i in range(len(sizes))]
        file_sizes = {f: s for f, s in zip(files, sizes)}
        res = DistributeFilesDataset._distribute_evenly_by_size(
            num_bins=partition_epoch, file_sizes=file_sizes, files_order=files
        )
        sizes = [sum([file_sizes[s] for s in v]) for v in res]
        assert numpy.std(sizes) <= max_stddev_percent * avg_per_bin

    # This test verifies the algorithm distributes evenly in easy cases
    rng = numpy.random.RandomState(42)
    _test_stddev(rng.uniform(low=97, high=103, size=(100,)), 20, 0.03)


def iter_identity(x, **kwargs):
    yield from x


def test_DistributeFilesDataset():
    from returnn.datasets.distrib_files import DistributeFilesDataset
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

    # Test to load via DistributeFilesDataset.

    def _get_sub_epoch_dataset(files_subepoch: List[str]) -> Dict[str, Any]:
        return {"class": "HDFDataset", "files": files_subepoch, "seq_ordering": "default"}

    partition_epoch = 5
    assert num_hdf_files % partition_epoch == 0  # just for easier testing here
    concat_dataset = init_dataset(
        {
            "class": "DistributeFilesDataset",
            "files": hdf_files,
            "get_sub_epoch_dataset": _get_sub_epoch_dataset,
            "partition_epoch": partition_epoch,
        }
    )
    assert isinstance(concat_dataset, DistributeFilesDataset)
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

    # Test DistributeFilesDataset in conjunction w/ a dataset, that does not have a
    # defined num_seqs:

    def _get_subepoch_dset_w_unknown_num_seqs(files_subepoch: List[str]) -> Dict[str, Any]:
        hdf_dset = {"class": "HDFDataset", "files": files_subepoch, "seq_ordering": "default"}
        return {"class": "PostprocessingDataset", "dataset": hdf_dset, "map_seq_stream": iter_identity}

    concat_dataset = init_dataset(
        {
            "class": "DistributeFilesDataset",
            "files": hdf_files,
            "get_sub_epoch_dataset": _get_subepoch_dset_w_unknown_num_seqs,
            "partition_epoch": partition_epoch,
        }
    )
    assert isinstance(concat_dataset, DistributeFilesDataset)
    for sub_epoch in range(1, partition_epoch + 1):
        print(f"Sub-epoch {sub_epoch}...")
        concat_dataset.init_seq_order(epoch=sub_epoch)
        concat_dataset.load_seqs(0, 1)


def test_PostprocessingDataset():
    from returnn.tensor.tensor_dict import TensorDict

    _demo_txt = "some utterance text that has a few words"

    def _add_1337_to_classes(tdict: TensorDict, **kwargs) -> TensorDict:
        tdict.data["classes"] += 1337
        return tdict

    with create_ogg_zip_txt_only_dataset_opts(text=_demo_txt) as sub_ds_opts:
        ds_opts = {
            "class": "PostprocessingDataset",
            "dataset": sub_ds_opts,
            "map_seq": _add_1337_to_classes,
        }
        dataset = init_dataset(ds_opts)
        dataset.get_data_keys()
        dataset.init_seq_order(epoch=1)
        assert dataset.have_seqs()
        dataset.load_seqs(0, 1)

        classes = dataset.get_data(0, "classes")
        assert all(c - 1337 >= 0 for c in classes)

        # assert that default seq tags have been replaced w/ ones from oggzip dset
        assert not dataset.get_tag(0).startswith("seq-")

    count = 0

    def _repeat2(input_iter: Iterator[TensorDict], **kwargs) -> Iterator[TensorDict]:
        nonlocal count

        for tdict in input_iter:
            count += 1

            yield tdict
            yield tdict

    with create_ogg_zip_txt_only_dataset_opts(text=_demo_txt) as sub_ds_opts:
        ds_opts = {
            "class": "PostprocessingDataset",
            "dataset": sub_ds_opts,
            "map_seq_stream": _repeat2,
        }
        dataset = init_dataset(ds_opts)
        dataset.init_seq_order(epoch=1)
        assert dataset.have_seqs()

        dataset.load_seqs(0, 2)
        for i in range(2):
            classes = dataset.get_data(i, "classes")
            assert len(classes) > 0
        assert count == 1

    # test laplace ordering
    with create_ogg_zip_txt_only_dataset_mult_seqs(num_seqs=6) as sub_ds_opts:
        from returnn.datasets.postprocessing import LaplaceOrdering

        ds_opts = {
            "class": "PostprocessingDataset",
            "dataset": sub_ds_opts,
            "map_seq_stream": LaplaceOrdering(3, "classes"),
        }
        dataset = init_dataset(ds_opts)
        dataset.init_seq_order(epoch=1)
        assert dataset.have_seqs()
        dataset.load_seqs(0, 6)

        prev_len = None
        prev_complete_frac = None
        for i in range(3):
            classes = dataset.get_data(i, "classes")
            complete_frac = dataset.get_complete_frac(i, allow_only_lr_suitable=True)
            assert prev_len is None or classes.shape[0] >= prev_len
            assert prev_complete_frac is None or complete_frac >= prev_complete_frac
            prev_len = classes.shape[0]
            prev_complete_frac = complete_frac
        for i in range(3, 6):
            classes = dataset.get_data(i, "classes")
            complete_frac = dataset.get_complete_frac(i, allow_only_lr_suitable=True)
            assert classes.shape[0] <= prev_len or i == 3
            assert complete_frac >= prev_complete_frac
            prev_len = classes.shape[0]
            prev_complete_frac = complete_frac
        assert prev_complete_frac == 1

    # test composition
    from returnn.datasets.postprocessing import Sequential

    func = Sequential(lambda x: x * 10, lambda y: y + 1)
    assert func(2) == 21


def test_MultiEpochDataset():
    from returnn.datasets.meta import MultiEpochDataset
    from returnn.datasets.cached2 import CachedDataset2

    in_dim, out_dim = 11, 7
    seq_len = 5
    inner_num_seqs = 10

    class _MyDataset(CachedDataset2):
        def __init__(self):
            super().__init__()
            self.num_inputs = in_dim
            self.num_outputs = {"classes": out_dim}

        # noinspection PyShadowingNames
        def init_seq_order(self, epoch=None, seq_list=None, seq_order=None):
            """init seq order"""
            super().init_seq_order(epoch=epoch, seq_list=seq_list, seq_order=seq_order)
            self._num_seqs = inner_num_seqs

        def _collect_single_seq(self, seq_idx: int) -> Optional[DatasetSeq]:
            if seq_idx >= self._num_seqs:
                return None
            return DatasetSeq(
                seq_idx=seq_idx,
                seq_tag=repr({"epoch": self.epoch, "seq_idx": seq_idx}),
                features=numpy.zeros((seq_len, in_dim)),
                targets={"classes": numpy.zeros((seq_len,), dtype=numpy.int32)},
            )

    inner_dataset = _MyDataset()
    inner_dataset.initialize()

    multi_epoch = 3
    dataset = MultiEpochDataset(dataset=inner_dataset, multi_epoch=multi_epoch)
    for outer_epoch in [1, 7]:
        seqs = dummy_iter_dataset(dataset, epoch=outer_epoch)
        assert len(seqs) == inner_num_seqs * multi_epoch
        outer_seq_idx = 0
        sub_ep = (outer_epoch - 1) * multi_epoch + 1  # 1-based
        sub_seq_idx = 0
        for seq in seqs:
            assert outer_seq_idx == seq.seq_idx
            assert seq.features["data"].shape == (seq_len, in_dim)
            assert seq.features["classes"].shape == (seq_len,)
            print("seq:", seq.seq_tag)
            d = eval(seq.seq_tag)  # seq tag is dict repr
            assert isinstance(d, dict)
            assert d["epoch"] == sub_ep
            assert d["seq_idx"] == sub_seq_idx
            # Calc next expected values.
            if sub_seq_idx >= inner_num_seqs - 1:
                sub_seq_idx = 0
                sub_ep += 1
            else:
                sub_seq_idx += 1
            outer_seq_idx += 1
        assert outer_seq_idx == len(seqs)
        assert sub_ep == outer_epoch * multi_epoch + 1 and sub_seq_idx == 0


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
