import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.cuda
from torch.utils.cpp_extension import load_inline


def load_lstm_ops():
  import os
  root = os.path.dirname(os.path.abspath(__file__))
  ops = open("{}/PTNativeOps.cpp".format(root)).read()
  kns = open("{}/PTNativeKernels.cpp".format(root)).read()
  lstm = load_inline(name='t', cpp_sources=[ops], cuda_sources=[kns])
  return lstm

lstm = load_lstm_ops()


class LstmOp(autograd.Function):

  @staticmethod
  def forward(ctx, *inputs):
    """
    void lstm_fwd_op_base(
    at::Tensor Z, at::Tensor Wr, at::Tensor c, at::Tensor i,
    at::Tensor Y, at::Tensor d
    );
    inputs = X(Z), Wr(V_h), y0, c0, i, direction, device
    """
    ctx.device = inputs[-2]
    ctx.direction = inputs[-1]
    T, B, D = inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2] // 4
    Y = torch.zeros((T, B, D), device=ctx.device)
    d = torch.zeros((B, D), device=ctx.device)
    op = lstm.lstm_forward_op_cpu if ctx.device.type == 'cpu' else lstm.lstm_forward_op_gpu
    op(inputs[0], inputs[1], inputs[3], inputs[4], Y, d)
    ctx.save_for_backward(*inputs[:5], Y, d)
    return Y, inputs[0], d

  @staticmethod
  def backward(ctx, *grad_outputs):
    """
    void lstm_bwd_op_base(
    at::Tensor Wr, at::Tensor c, at::Tensor i, at::Tensor Y,
    at::Tensor Dd, at::Tensor DZ, at::Tensor DWr, at::Tensor Dc, at::Tensor tmpDc
    );
    """
    DWr = torch.zeros_like(ctx.saved_tensors[1], device=ctx.device)
    Dc = torch.zeros_like(ctx.saved_tensors[3], device=ctx.device)
    op = lstm.lstm_backward_op_cpu if ctx.device.type == 'cpu' else lstm.lstm_backward_op_gpu
    op(
      ctx.saved_tensors[1], ctx.saved_tensors[3], ctx.saved_tensors[4],
      ctx.saved_tensors[5],
      grad_outputs[2].contiguous(), ctx.saved_tensors[0], DWr, Dc, grad_outputs[0].contiguous()
    )
    return ctx.saved_tensors[0], DWr, None, Dc, None, None, None


class SingleLayerLstm(nn.Module):

  def __init__(self, n_input, n_hidden, direction=1):
    super(SingleLayerLstm, self).__init__()
    self.input_size = n_input
    self.hidden_size = n_hidden
    self.gate_size = self.hidden_size * 4
    self.direction = direction

    shape_wf = (self.input_size, self.gate_size)
    shape_wr = (self.hidden_size, self.gate_size)

    using_pytorch_init = True
    if using_pytorch_init:
      stdv = 1.0 / math.sqrt(self.hidden_size)  # the same initializer as PyTorch's RNN implementation
      self.W_in = nn.Parameter(torch.empty(shape_wf).uniform_(-stdv, stdv))
      self.W_re = nn.Parameter(torch.empty(shape_wr).uniform_(-stdv, stdv))
      self.b = nn.Parameter(torch.zeros(self.gate_size).uniform_(-2 * stdv, 2 * stdv))
    else:
      stdv_wf = math.sqrt(6.0 / (self.input_size + self.gate_size))
      stdv_wr = math.sqrt(6.0 / (self.hidden_size + self.gate_size))
      self.W_in = nn.Parameter(torch.empty(shape_wf).uniform_(-stdv_wf, stdv_wf))
      self.W_re = nn.Parameter(torch.empty(shape_wr).uniform_(-stdv_wr, stdv_wr))
      self.b = nn.Parameter(torch.zeros(self.gate_size))

  def forward(self, X, i=None, h0=None, c0=None):
    device = self.W_in.device
    assert device == X.device
    T, B = X.shape[0], X.shape[1]
    if h0 is None:
      h0 = torch.zeros((B, self.hidden_size), requires_grad=False, device=device)
      c0 = h0
    if i is None:
      i = torch.ones((T, B), requires_grad=False, device=device)
    i = i.float()

    if self.direction == -1:
      idx = torch.arange(T - 1, -1, -1).to(X.device)
      X = X.index_select(0, idx)
      i = i.index_select(0, idx)

    intern = torch.einsum("ijk,kl->ijl", X, self.W_in) + self.b

    Y, C, d = LstmOp.apply(intern, self.W_re, h0, c0, i, device, self.direction)

    if self.direction == -1:
      idx = torch.arange(T - 1, -1, -1).to(X.device)
      Y = Y.index_select(0, idx)
      
    return Y, (C, d)
