import math
import time
import random
import os
import numpy

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.cuda
from torch.utils.cpp_extension import load_inline
import torch.backends.cudnn

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
os.environ['PYTHONHASHSEED'] = str(1234)
numpy.random.seed(1234)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_lstm_ops():
  import os
  root = os.path.dirname(os.path.abspath(__file__))
  ops = open("{}/PTNativeOps.cpp".format(root)).read()
  kns = open("{}/PTNativeKernels.cpp".format(root)).read()
  lstm = load_inline(name='t', cpp_sources=[ops], cuda_sources=[kns])
  return lstm

lstm = load_lstm_ops()
# device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

class LstmOp(autograd.Function):
  
  @staticmethod
  def forward(ctx, *inputs):
    ctx.device = inputs[-1]
    T, B, D = inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2] // 4
    Y = torch.zeros((T, B, D), device=ctx.device)
    C = torch.zeros((T, B, D), device=ctx.device)
    H = torch.zeros((T, B, 4 * D), device=ctx.device)
    d = torch.zeros((B, D), device=ctx.device)
    lstm.lstm_forward_op(*inputs[:5], 0, 1, Y, C, H, d)
    ctx.save_for_backward(*inputs[:5], Y, C, H)
    return Y, C, d
  
  @staticmethod
  def backward(ctx, *grad_outputs):
    DX = torch.zeros_like(ctx.saved_tensors[0], device=ctx.device)
    DW = torch.zeros_like(ctx.saved_tensors[1], device=ctx.device)
    Dy0 = torch.zeros_like(ctx.saved_tensors[2], device=ctx.device)
    Dc0 = torch.zeros_like(ctx.saved_tensors[3], device=ctx.device)
    lstm.lstm_backward_op(
      *ctx.saved_tensors[:5],
      0, 1,
      *ctx.saved_tensors[5:],
      grad_outputs[0].contiguous(), grad_outputs[2].contiguous(),
      DX, DW, Dy0, Dc0
    )
    return DX, DW, Dy0, Dc0, None, None


class SingleLayerLstm(nn.Module):
  
  def __init__(self, n_input, n_hidden):
    super(SingleLayerLstm, self).__init__()
    self.input_size = n_input
    self.hidden_size = n_hidden
    self.gate_size = self.hidden_size * 4
    stdv = 1.0 / math.sqrt(self.hidden_size)  # the same initializer as PyTorch's RNN implementation
    shape_wf = (self.input_size, self.gate_size)
    shape_wr = (self.hidden_size, self.gate_size)
    self.Wf = nn.Parameter(torch.empty(shape_wf).uniform_(-stdv, stdv))
    self.Wr = nn.Parameter(torch.empty(shape_wr).uniform_(-stdv, stdv))
    self.bf = nn.Parameter(torch.empty(self.gate_size).uniform_(-2 * stdv, 2 * stdv))
  
  def forward(self, X, i = None, device='cpu', h0=None, c0=None):
    T, B = X.shape[0], X.shape[1]
    if h0 is None:
      h0 = torch.zeros((B, self.hidden_size), requires_grad=False, device=device)
      c0 = h0
    if i is None:
      i = torch.ones((T, B), requires_grad=False, device=device)
    i = i.float()
    intern = torch.einsum("ijk,kl->ijl", (X, self.Wf)) + self.bf
    Y, C, d = LstmOp.apply(intern, self.Wr, h0, c0, i, device)
    return Y, (C, d)




