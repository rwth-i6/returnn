
import theano
import theano.tensor as T
import theano.printing
from math import log
import numpy


# This follows somewhat the code from here:
# https://blog.wtf.sg/2014/10/06/connectionist-temporal-classification-ctc-with-theano/
# https://github.com/mohammadpz/CTC-Connectionist-Temporal-Classification/blob/master/ctc_cost.py
# Credits: Shawn Tan, Rakesh Var, Mohammad Pezeshki


def log_add(a, b):
  m = T.maximum(a, b)
  return m + T.log1p(T.exp(a + b - 2 * m))

def log_sum(a, axis=None, keepdims=False):
  if axis is None:
    assert keepdims is False  # not implemented atm
    return log_sum(a.flatten(), axis=0)
  assert isinstance(axis, int)  # current implementation only for exactly one axis
  m, argm = T.max_and_argmax(a, axis=axis, keepdims=True)
  exp_a = T.exp(a - m)
  idx = T.arange(a.shape[axis]).dimshuffle(['x'] * axis + [0] + ['x'] * (a.ndim - axis - 1))
  exp_a = T.switch(T.eq(idx, argm), 0, exp_a)
  sum = T.sum(exp_a, axis=axis, keepdims=True)
  res = m + T.log1p(sum)
  if not keepdims:
    if axis is not None:
      res = res.dimshuffle([i for i in range(res.ndim) if i != axis])
    else:
      res = res.dimshuffle()  # expect a scalar
  return res

def log_mul(a, b):
  return a + b

def log_div(a, b):
  return a - b


almost_zero_pos_float = 1e-38

def log_path_probs(log_pcx_y, time_mask, seq_lens, forward=True):
  """
  No blanks. Calculates the forward/backward probabilities.

  :param log_pcx_y: softmax output for labels, log-space. shape (time,batch,seqlen) -> log prob
  :param time_mask: (time,batch) -> 0 or 1
  :param seq_lens: (batch,) -> seqlen
  :return: log probabilities. shape (time,batch,seqlen)
  """

  def step(log_pcx_yt, t_mask, fw_prev):
    # log_pcx_yt is (batch,seqlen) for the current time frame
    # t_mask is (batch,) -> 0 or 1
    # fw/fw_prev is also (batch,seqlen)
    fw = fw_prev  # loops
    if forward:
      fw = T.set_subtensor(fw[:, 1:], log_add(fw[:, 1:], fw_prev[:, :-1]))  # forward one symbol
    else:
      fw = T.set_subtensor(fw[:, :-1], log_add(fw[:, :-1], fw_prev[:, 1:]))  # backward one symbol
    fw = log_mul(log_pcx_yt, fw)
    return T.switch(t_mask.dimshuffle(0, 'x'), fw, fw_prev)

  # The initial alpha for recursion is not exactly correct because we will allow the first *two* states at t=0.
  # But it's much simpler.
  fw_initial = T.zeros_like(log_pcx_y[0])  # (batch,seqlen)
  fw_initial += log(almost_zero_pos_float)  # 0
  if forward:
    fw_initial = T.set_subtensor(fw_initial[:, 0], log(1))
  else:
    fw_initial = T.set_subtensor(fw_initial[:, seq_lens - 1], log(1))

  probs, _ = theano.scan(
    step,
    go_backwards=not forward,
    sequences=[log_pcx_y, time_mask],
    outputs_info=[fw_initial]
  )
  # probs is (time,batch,seqlen)
  return probs[::1 if forward else -1]

def ctc(log_pcx, time_mask, targets, seq_lens):
  """
  No blanks. Calculates the CTC cost.

  :param log_pcx: softmax output, log-space. shape (time,batch,label) -> log prob
  :param time_mask: (time,batch) -> 0 or 1
  :param targets: target seq, shape (seqlen,batch) -> label. seqlen <= time.
  :param seq_lens: (batch,) -> seqlen
  :return: probs: (time,batch,seqlen) -> log prob
  """

  num_batches = log_pcx.shape[1]
  all_batches = T.arange(num_batches).dimshuffle(0, 'x')
  log_pcx_y = log_pcx[:, all_batches, targets.T]  # (time,batch,seqlen)

  forward_probs = log_path_probs(log_pcx_y, time_mask, seq_lens, forward=True)  # (time,batch,seqlen)
  backward_probs = log_path_probs(log_pcx_y, time_mask, seq_lens, forward=False)
  probs = log_div(log_mul(forward_probs, backward_probs), log_pcx_y)
  return probs

def ctc_cost(*args, **kwargs):
  """
  :returns total negative log probability (scalar)
  """
  log_probs = ctc(*args, **kwargs)
  total_prob = log_sum(log_probs, axis=2)
  total_prob = log_sum(total_prob)
  return -total_prob  # neg log probs


def uniq_with_lengths(seq, time_mask):
  """
  :param seq: (time,batch) -> label
  :param time_mask: (time,batch) -> 0 or 1
  :return: out_seqs, seq_lens.
  out_seqs is (max_seq_len,batch) -> label, where max_seq_len <= time.
  seq_lens is (batch,) -> len.
  """
  num_batches = seq.shape[1]
  diffs = T.ones_like(seq)
  diffs = T.set_subtensor(diffs[1:], seq[1:] - seq[:-1])
  time_range = T.arange(seq.shape[0]).dimshuffle([0] + ['x'] * (seq.ndim - 1))
  idx = T.switch(T.neq(diffs, 0) * time_mask, time_range, -1)  # (time,batch) -> idx or -1
  seq_lens = T.sum(T.ge(idx, 0), axis=0)  # (batch,) -> len
  max_seq_len = T.max(seq_lens)

  # I don't know any better way without scan.
  # http://stackoverflow.com/questions/31379971/uniq-for-2d-theano-tensor
  def step(batch_idx, out_seq_b1):
    #out_seq = seq[T.ge(idx[:, batch_idx], 0).nonzero(), batch_idx][0]
    out_seq = seq[:, batch_idx][T.ge(idx[:, batch_idx], 0).nonzero()]
    return T.concatenate((out_seq, T.zeros((max_seq_len - out_seq.shape[0],), dtype=seq.dtype)))

  out_seqs, _ = theano.scan(
    step,
    sequences=[T.arange(num_batches)],
    outputs_info=[T.zeros((max_seq_len,), dtype=seq.dtype)]
  )
  # out_seqs is (batch,max_seq_len)
  return out_seqs.T, seq_lens


def test_uniq_with_lengths():
  s1 = [[5, 3, 3, 1, 0, 0],
        [4, 4, 1, 1, 2, 2],
        [7, 7, 2, 0, 0, 0]]
  s1 = theano.shared(numpy.array(s1).T)
  s2 = T.neq(s1, 0)
  out_seqs, seq_lens = uniq_with_lengths(s1, s2)
  out_seqs = out_seqs.eval()
  seq_lens = seq_lens.eval()
  assert list(seq_lens) == [3, 3, 2]
  assert out_seqs.shape == (3, 3)
  assert list(out_seqs[:, 0]) == [5, 3, 1]
  assert list(out_seqs[:, 1]) == [4, 1, 2]
  assert list(out_seqs[:, 2])[:2] == [7, 2]


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

def test_make_batch(seqs):
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

def test():
  log_pcx, time_mask, targets, seq_lens = map(theano.shared, test_make_batch([test1_seq()]))
  theano_log_ctc_probs = ctc(log_pcx, time_mask, targets, seq_lens)
  manual_ctc_probs = manual_ctc(*test1_seq())
  assert numpy.allclose(numpy.exp(theano_log_ctc_probs.eval())[:,0,:], manual_ctc_probs)
  theano_ctc_total = log_sum(theano_log_ctc_probs, axis=2)
  manual_ctc_total = manual_ctc_probs.sum(axis=1)
  assert numpy.allclose(numpy.exp(theano_ctc_total.eval()[:,0]), manual_ctc_total)
