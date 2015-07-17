

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE"

from NetworkCtcLayer import *
from nose.tools import assert_equal


def test1_seq():
  pcx = numpy.array([[0.1, 0.3, 0.2, 0.1, 0.3],
                     [0.3, 0.1, 0.2, 0.1, 0.3],
                     [0.2, 0.1, 0.4, 0.2, 0.1]])
  assert numpy.allclose(pcx.sum(axis=1), 1)
  log_pcx = numpy.log(pcx)
  targets = numpy.array([1,2])
  return log_pcx, targets


def manual_ctc(log_pcx, targets):
  # pcx is (time,label)
  # targets is (seqlen,)
  pcx = numpy.exp(log_pcx)
  T, L = pcx.shape
  S, = targets.shape
  pcx_y = pcx[:, targets]  # (time,seqlen)
  fw = numpy.zeros_like(pcx_y)  # (time,seqlen)
  for t in range(T):
    if t > 0:
      fw_prev = fw[t - 1]
    else:
      fw_prev = numpy.array([1.] + [0.] * (S - 1))  # initial
    fw[t] = fw_prev  # loops
    fw[t, 1:] += fw_prev[:-1]  # forward step
    fw[t] *= pcx_y[t]
  bw = numpy.zeros_like(pcx_y)  # (time,seqlen)
  for t in reversed(range(T)):
    if t < T - 1:
      bw_prev = bw[t + 1]
    else:
      bw_prev = numpy.array([0.] * (S - 1) + [1.])  # initial
    bw[t] = bw_prev  # loops
    bw[t, :-1] += bw_prev[1:]  # backward step
    bw[t] *= pcx_y[t]
  probs = (fw * bw) / pcx_y
  return probs

def make_batch(seqs):
  assert seqs
  max_time = 0
  max_seq_len = 0
  label_dim = None
  for log_pcx, targets in seqs:
    assert len(log_pcx.shape) == 2
    assert len(targets.shape) == 1
    if label_dim is None: label_dim = log_pcx.shape[1]
    assert label_dim == log_pcx.shape[1]
    max_time = max(max_time, log_pcx.shape[0])
    max_seq_len = max(max_seq_len, targets.shape[0])
  batch_log_pcx = numpy.zeros((max_time, len(seqs), label_dim), dtype='float32')
  batch_targets = numpy.zeros((max_seq_len, len(seqs)), dtype='int32')
  time_mask = numpy.zeros((max_time, len(seqs)), dtype='int8')
  for batch_idx, (log_pcx, targets) in enumerate(seqs):
    batch_log_pcx[:,batch_idx,:] = log_pcx
    batch_targets[:,batch_idx] = targets
    time_mask[0:len(log_pcx), batch_idx] = 1
  seq_lens = numpy.array([len(targets) for _, targets in seqs], dtype='int32')
  return batch_log_pcx, time_mask, batch_targets, seq_lens



class TestNetworkCtcLayer(object):

  def test_uniq_with_lengths(self):
    s1 = [[5, 3, 3, 1, 0, 0],
          [4, 4, 1, 1, 2, 2],
          [7, 7, 2, 0, 0, 0]]
    s1 = theano.shared(numpy.array(s1).T)
    s2 = T.neq(s1, 0)
    out_seqs, seq_lens = uniq_with_lengths(s1, s2)
    out_seqs = out_seqs.eval()
    seq_lens = seq_lens.eval()
    assert_equal(list(seq_lens), [3, 3, 2])
    assert_equal(out_seqs.shape, (3, 3))
    assert_equal(list(out_seqs[:, 0]), [5, 3, 1])
    assert_equal(list(out_seqs[:, 1]), [4, 1, 2])
    assert_equal(list(out_seqs[:, 2])[:2], [7, 2])

  def test_ctc(self):
    log_pcx, time_mask, targets, seq_lens = map(theano.shared, make_batch([test1_seq()]))
    theano_log_ctc_probs = ctc(log_pcx, time_mask, targets, seq_lens)
    manual_ctc_probs = manual_ctc(*test1_seq())
    assert numpy.allclose(numpy.exp(theano_log_ctc_probs.eval())[:,0,:], manual_ctc_probs)
    theano_ctc_total = log_sum(theano_log_ctc_probs, axis=2)
    manual_ctc_total = manual_ctc_probs.sum(axis=1)
    assert numpy.allclose(numpy.exp(theano_ctc_total.eval()[:,0]), manual_ctc_total)
