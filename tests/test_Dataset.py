
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
import numpy as np

from returnn.util import better_exchook

from returnn.log import log
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


def iterate_seqs_rnnt(dataset, chunk_size=None, chunk_step=None, used_data_keys=None,
                      alignment_key="alignment", classes_key="classes", data_key="data",
                      blank_index=0):
  """
  Takes chunking into consideration, special case for RNN-T.
  Because RNN-T does not operate time-synchronous
  (but rather alignment-synchronously), we have to take this into
  consideration when doing chunking.
  This implementation currently chunks along the data-dimension,
  that means that the chunks are constant for "data" (except the last chunk).
  Also because our alignment is computed on the encoder-axis,
  the chunking is performed on this level also. When you have time-reduction
  from your input to the alignment, you have to consider this when outputting
  chunks for "data" (e.g. just multiply with time-reduction factor).

  Note: chunking_variance is not supported.

  :param Dataset dataset:
  :param int|dict|NumbersDict chunk_size:
  :param int|dict|NumbersDict chunk_step:
  :param set(str)|None used_data_keys:
  :param str alignment_key:
  :param str classes_key:
  :param str data_key:
  :param int blank_index:
  :return: generator which yields tuples (seq index, seq start, seq end)
  :rtype: list[(int,NumbersDict,NumbersDict)]
  """
  if chunk_size is None:
    chunk_size = dataset.chunk_size
  if chunk_step is None:
    chunk_step = dataset.chunk_step
  chunk_size = NumbersDict(chunk_size)
  chunk_step = NumbersDict(chunk_step)
  print("Dataset %r: min_chunk_size: %i" % (dataset, dataset.min_chunk_size))
  print("chunk_step:", chunk_step)
  print("chunk_size:", chunk_size)

  s = 0
  data_keys = dataset.data_keys  # for other than `StaticDataset` this may be dataset.get_data_keys()
  assert alignment_key in data_keys, "key '%s' in data-keys %r" % (alignment_key, data_keys)
  assert classes_key in data_keys, "No key '%s' in data-keys %r" % (classes_key, data_keys)
  assert data_key in data_keys, "No key '%s' in data-keys %r" % (classes_key, data_keys)
  assert data_key in chunk_size.keys(), "specify the chunking only in one key"
  assert classes_key not in chunk_size.keys(), "specify the chunking only one key"
  assert alignment_key not in chunk_size.keys(), "specify the chunking only one key"

  def compute_dynamic_chunk_sizes(chunk):
    """
    Computes the dynamic chunk size from a given chunk.
    We currently support just data, classes and alignment itself.

    :param np.ndarray chunk: from alignment
    """
    nr_of_symbols = len(np.where(chunk != blank_index)[0])
    nr_of_blanks = len(chunk) - nr_of_symbols
    num_dict = NumbersDict({
      data_key: nr_of_blanks,
      alignment_key: len(chunk),
      classes_key: nr_of_symbols,
    })
    return num_dict

  while dataset.is_less_than_num_seqs(s):
    length = dataset.get_seq_length(s)
    if chunk_size == 0:
      yield s, NumbersDict.constant_like(0, numbers_dict=length), length
    else:
      alignment = dataset.get_data(s, alignment_key)

      # in the given chunk of alignment data, we check for:
      # * blank indices (advance input)
      # * non-blank indices (advance target/classes)
      # We can either advance input or target by a constant amount each time.
      # Currently only advanced input-len a constant amount is supported.

      blank_indices = np.where(alignment == blank_index)[0]  # [T]
      blank_length = len(blank_indices)
      nr_of_chunks = (blank_length - 1) // chunk_step[data_key] + 1

      # index into the different data-keys
      t = NumbersDict({data_key: 0, classes_key: 0, alignment_key: 0})
      for chunk_idx in range(nr_of_chunks):
        chunk_start = t
        data_chunk_indices = blank_indices[t[data_key]:t[data_key]+chunk_size[data_key]]
        # we might miss some data (in the alignment) after the last blank.
        end = len(alignment) if len(data_chunk_indices) < chunk_size[data_key] else data_chunk_indices[-1]+1
        alignment_size_chunk = alignment[chunk_start[alignment_key]:end]
        dynamic_chunk_size = compute_dynamic_chunk_sizes(alignment_size_chunk)
        chunk_start = t.copy()
        chunk_end = NumbersDict.min([chunk_start + dynamic_chunk_size, length])

        # compute dynamic chunk step:
        chunk_indices_step = blank_indices[t[data_key]:t[data_key] + chunk_step[data_key]]
        end_step = len(alignment) if len(chunk_indices_step) < chunk_step[data_key] else chunk_indices_step[-1] + 1
        alignment_step_chunk = alignment[chunk_start[alignment_key]:end_step]
        dynamic_chunk_step = compute_dynamic_chunk_sizes(alignment_step_chunk)
        t += dynamic_chunk_step
        yield s, chunk_start, chunk_end
    s += 1
  return


def test_iterate_seqs_chunking_rnnt():
  # Simulate transducer data: classes, alignment, data
  np.random.seed(42)
  blank_index = 0
  n_batch = 3
  alignments = [
    np.array([0, 0, 1, 0, 2, 0, 0, 1, 0, 0, 3]),  # T=7, U=4
    np.array([0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 4]),  # T=8, U=3
    np.array([1, 2, 1, 0, 5, 0, 1, 1, 2, 4, 3]),  # T=2, U=9
  ]
  data = [np.arange(7), np.arange(8), np.arange(2)]
  classes = [
    np.array([1, 2, 1, 3]),
    np.array([2, 2, 4]),
    np.array([1, 2, 1, 0, 5, 0, 1, 1, 2, 4, 3]),
  ]
  from returnn.datasets.generating import StaticDataset
  dataset = StaticDataset([{"data": data[i],
                            "classes": classes[i],
                            "alignment": alignments[i]} for i in range(n_batch)],
                          output_dim={"data": (5, 1),
                                      "classes": (5, 1),
                                      "alignment": (5, 1)})
  dataset.init_seq_order(1)

  def wrapper_iterate_seqs(self, chunk_size=None, chunk_step=None, used_data_keys=None):
    return iterate_seqs_rnnt(self, chunk_size, chunk_step, used_data_keys,
                             alignment_key="alignment", classes_key="classes",
                             data_key="data", blank_index=blank_index)
  StaticDataset.iterate_seqs = wrapper_iterate_seqs  # we simply monkey-patch it

  # first test, size==step
  seqs = list(dataset.iterate_seqs(chunk_size={'data': 6}, chunk_step={'data': 6}, used_data_keys=None))
  print("We got %i seqs." % len(seqs))
  for s in seqs:
    print(s)

  # seq[0]: (seq index, seq start, seq end)
  # np.array([0, 0, 1, 0, 2, 0, 0, 1, 0, 0, 3]),  # T=7, U=4
  # -> [0, 0, 1, 0, 2, 0, 0, 1, 0] (T=[0,6), U=[0,3), len(align)=[0,3))
  # -> [0, 3]  # (T=[6,7), U=[3,4), A=[9,11))
  assert_equal(seqs[0], (0,
                         NumbersDict({'data': 0, 'classes': 0, 'alignment': 0}),
                         NumbersDict({'data': 6, 'classes': 3, 'alignment': 9})))
  assert_equal(seqs[1], (0,
                         NumbersDict({'data': 6, 'classes': 3, 'alignment': 9}),
                         NumbersDict({'data': 7, 'classes': 4, 'alignment': 11})))

  # np.array([0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 4]),  # T=8, U=3
  # -> [0, 0, 0, 0, 0, 0]  # (T=[0,6), U=[0,0), A=[0,6))
  # -> [0, 2, 2, 0, 4]     # (T=[6,8), U=[0,3), A=[6,11))
  assert_equal(seqs[2], (1,
                         NumbersDict({'data': 0, 'classes': 0, 'alignment': 0}),
                         NumbersDict({'data': 6, 'classes': 0, 'alignment': 6})))
  assert_equal(seqs[3], (1,
                         NumbersDict({'data': 6, 'classes': 0, 'alignment': 6}),
                         NumbersDict({'data': 8, 'classes': 3, 'alignment': 11})))

  # np.array([1, 2, 1, 0, 5, 0, 1, 1, 2, 4, 3]),  # T=2, U=9
  # -> [0, 0]  # (T=[0,2), U=[0,9), A=[0,11))
  assert_equal(seqs[4], (2,
                         NumbersDict({'data': 0, 'classes': 0, 'alignment': 0}),
                         NumbersDict({'data': 2, 'classes': 9, 'alignment': 11})))

  #
  # Testing for overlapping chunks!
  #
  seqs = list(dataset.iterate_seqs(chunk_size={'data': 6}, chunk_step={'data': 3}, used_data_keys=None))
  print("We got %i seqs." % len(seqs))
  for s in seqs:
    print(s)
  # np.array([0, 0, 1, 0, > 2, 0, 0, 1, 0, > 0, 3]),  # T=7, U=4
  # -> [0, 0, 1, 0, 2, 0, 0, 1, 0]  # (T=[0,6), U=[0,3), A=[0,9))
  # -> [2, 0, 0, 1, 0, 0, 3]        # (T=[3,7), U=[1,4), A=[4,11))
  # -> [0, 3]                       # (T=[6,7), U=[3,4), A=[9, 11))
  assert_equal(seqs[0], (0,
                         NumbersDict({'data': 0, 'classes': 0, 'alignment': 0}),
                         NumbersDict({'data': 6, 'classes': 3, 'alignment': 9})))
  assert_equal(seqs[1], (0,
                         NumbersDict({'data': 3, 'classes': 1, 'alignment': 4}),
                         NumbersDict({'data': 7, 'classes': 4, 'alignment': 11})))
  assert_equal(seqs[2], (0,
                         NumbersDict({'data': 6, 'classes': 3, 'alignment': 9}),
                         NumbersDict({'data': 7, 'classes': 4, 'alignment': 11})))

  # np.array([0, 0, 0, > 0, 0, 0, > 0, 2, 2, 0, 4]),  # T=8, U=3, A=11
  # -> [0, 0, 0, 0, 0, 0]        # T=[0,6),  U=[0,0), A=[0,6)
  # -> [0, 0, 0, 0, 2, 2, 0, 4]  # T=[3,8), U=[0,3), A=[3,11)
  # -> [0, 2, 2, 0, 4]           # T=[6,8), U=[0,3), A=[6,11)
  assert_equal(seqs[3], (1,
                         NumbersDict({'data': 0, 'classes': 0, 'alignment': 0}),
                         NumbersDict({'data': 6, 'classes': 0, 'alignment': 6})))
  assert_equal(seqs[4], (1,
                         NumbersDict({'data': 3, 'classes': 0, 'alignment': 3}),
                         NumbersDict({'data': 8, 'classes': 3, 'alignment': 11})))
  assert_equal(seqs[5], (1,
                         NumbersDict({'data': 6, 'classes': 0, 'alignment': 6}),
                         NumbersDict({'data': 8, 'classes': 3, 'alignment': 11})))

  # np.array([1, 2, 1, 0, 5, 0, 1, 1, 2, 4, 3]),  # T=2, U=9
  # -> [1, 2, 1, 0, 5, 0, 1, 1, 2, 4, 3]  # (T=2, U=9, A=11)
  assert_equal(seqs[6], (2,
                         NumbersDict({'data': 0, 'classes': 0, 'alignment': 0}),
                         NumbersDict({'data': 2, 'classes': 9, 'alignment': 11})))


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
