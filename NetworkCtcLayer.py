
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


