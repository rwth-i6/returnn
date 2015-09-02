
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false
from GeneratingDataset import GeneratingDataset, DummyDataset
from EngineBatch import Batch
from Dataset import DatasetSeq
from Log import log
import numpy as np

log.initialize()


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
  seqs = list(dataset._iterate_seqs(chunk_size=0, chunk_step=0))
  assert_equal(len(seqs), 2)
  assert_equal(seqs[0], (0, 0, 11))  # seq-idx, start-frame, end-frame
  assert_equal(seqs[1], (1, 0, 11))

def test_iterate_seqs_chunking_1():
  dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=2, seq_len=11)
  dataset.init_seq_order(1)
  seqs = list(dataset._iterate_seqs(chunk_size=10, chunk_step=5))
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
  all_batches = []; " :type: list[Batch] "
  while batch_gen.has_more():
    batch, = batch_gen.peek_next_n(1)
    assert_is_instance(batch, Batch)
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
  assert_equal(all_batches[3].seqs[0].seq_end_frame, 5)
  assert_equal(all_batches[3].seqs[0].frame_length, 5)
  assert_equal(all_batches[3].seqs[0].batch_slice, 0)
  assert_equal(all_batches[3].seqs[0].batch_frame_offset, 0)

def test_batches_non_recurrent_1():
  dataset = DummyDataset(input_dim=2, output_dim=3, num_seqs=2, seq_len=11)
  dataset.init_seq_order(1)
  batch_gen = dataset.generate_batches(recurrent_net=False, max_seqs=2, batch_size=5)
  all_batches = []; " :type: list[Batch] "
  while batch_gen.has_more():
    batch, = batch_gen.peek_next_n(1)
    assert_is_instance(batch, Batch)
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

  assert_equal(all_batches[3].start_seq, 1)
  assert_equal(all_batches[3].end_seq, 2)  # exclusive
  assert_equal(len(all_batches[3].seqs), 1)  # 1 BatchSeqCopyPart
  assert_equal(all_batches[3].seqs[0].seq_idx, 1)
  assert_equal(all_batches[3].seqs[0].seq_start_frame, 0)
  assert_equal(all_batches[3].seqs[0].seq_end_frame, 5)
  assert_equal(all_batches[3].seqs[0].frame_length, 5)
  assert_equal(all_batches[3].seqs[0].batch_slice, 0)
  assert_equal(all_batches[3].seqs[0].batch_frame_offset, 0)
