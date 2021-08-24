
# Also see test_SprintDataset.py.

from __future__ import print_function

import sys
import _setup_test_env  # noqa
import unittest
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false
from returnn.datasets.generating import GeneratingDataset, DummyDataset, DummyDatasetMultipleSequenceLength
from returnn.engine.batch import Batch
from returnn.datasets.basic import DatasetSeq
from returnn.util.basic import NumbersDict

from returnn.util import better_exchook


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
  dataset.init_seq_order(1)
  seqs = list(dataset.iterate_seqs(chunk_size=0, chunk_step=0, used_data_keys=None))
  assert_equal(len(seqs), 2)
  assert_equal(seqs[0], (0, 0, 11))  # seq-idx, start-frame, end-frame
  assert_equal(seqs[1], (1, 0, 11))


def test_iterate_seqs_chunking_1():
  dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=2, seq_len=11)
  dataset.init_seq_order(1)
  seqs = list(dataset.iterate_seqs(chunk_size=10, chunk_step=5, used_data_keys=None))
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
  dataset = DummyDatasetMultipleSequenceLength(input_dim=2, output_dim=3, num_seqs=2, seq_len={'data': 24, 'classes': 12})
  dataset.init_seq_order(1)
  seqs = list(dataset.iterate_seqs(chunk_size={'data': 12, 'classes': 6}, chunk_step={'data': 6, 'classes': 3}, used_data_keys=None))
  for s in seqs:
    print(s)
  assert_equal(len(seqs), 8)
  assert_equal(seqs[0], (0, NumbersDict({'data':0, 'classes': 0}), NumbersDict({'data':12, 'classes': 6})))
  assert_equal(seqs[1], (0, NumbersDict({'data':6, 'classes': 3}), NumbersDict({'data':18, 'classes': 9})))
  assert_equal(seqs[2], (0, NumbersDict({'data':12, 'classes': 6}), NumbersDict({'data':24, 'classes': 12})))
  assert_equal(seqs[3], (0, NumbersDict({'data':18, 'classes': 9}), NumbersDict({'data':24, 'classes': 12})))
  assert_equal(seqs[4], (1, NumbersDict({'data':0, 'classes': 0}), NumbersDict({'data':12, 'classes': 6})))
  assert_equal(seqs[5], (1, NumbersDict({'data':6, 'classes': 3}), NumbersDict({'data':18, 'classes': 9})))
  assert_equal(seqs[6], (1, NumbersDict({'data':12, 'classes': 6}), NumbersDict({'data':24, 'classes': 12})))
  assert_equal(seqs[7], (1, NumbersDict({'data':18, 'classes': 9}), NumbersDict({'data':24, 'classes': 12})))


def test_batches_recurrent_1():
  dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=2, seq_len=11)
  dataset.init_seq_order(1)
  dataset.chunk_size = 10
  dataset.chunk_step = 5
  batch_gen = dataset.generate_batches(recurrent_net=True, max_seqs=1, batch_size=20)
  all_batches = []; " :type: list[Batch] "
  while batch_gen.has_more():
    batch, = batch_gen.peek_next_n(1)
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
    batch, = batch_gen.peek_next_n(1)
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
    batch, = batch_gen.peek_next_n(1)
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
