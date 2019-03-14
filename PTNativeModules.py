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
  start = time.time()
  import os
  root = os.path.dirname(os.path.abspath(__file__))
  ops = open("{}/PTNativeOps.cpp".format(root)).read()
  kns = open("{}/PTNativeKernels.cpp".format(root)).read()
  lstm = load_inline(name='t', cpp_sources=[ops], cuda_sources=[kns])
  print()
  print("Local LstmOperator compiled and loaded. Took {:.4f} seconds!".format(time.time() - start))
  print()
  return lstm

lstm = load_lstm_ops()
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


class LstmOpLowMem(autograd.Function):
  device = 'cpu'
  
  @staticmethod
  def forward(ctx, *inputs):
    T, B, D1 = inputs[0].shape
    D2 = inputs[1].shape[0] - D1
    Y = torch.zeros((T, B, D2), device=LstmOpLowMem.device)
    C = torch.zeros((T, B, D2), device=LstmOpLowMem.device)
    d = torch.zeros((B, D2), device=LstmOpLowMem.device)
    lstm.lstm_forward_op_low_mem(*inputs, 0, 1, Y, C, d)
    ctx.save_for_backward(*inputs, Y, C)
    return Y, C, d
  
  @staticmethod
  def backward(ctx, *grad_outputs):
    DX = torch.zeros_like(ctx.saved_tensors[0], device=LstmOpLowMem.device)
    DW = torch.zeros_like(ctx.saved_tensors[1], device=LstmOpLowMem.device)
    Db = torch.zeros_like(ctx.saved_tensors[2], device=LstmOpLowMem.device)
    Dh = torch.zeros_like(ctx.saved_tensors[3], device=LstmOpLowMem.device)
    Dc = torch.zeros_like(ctx.saved_tensors[4], device=LstmOpLowMem.device)
    lstm.lstm_backward_op_low_mem(
      *ctx.saved_tensors[:6], 0, 1,
      *ctx.saved_tensors[6:],
      grad_outputs[0], grad_outputs[2],
      DX, DW, Db, Dh, Dc
    )
    return DX, DW, Db, Dh, Dc, None


class LstmOpLowmemVar(autograd.Function):
  
  @staticmethod
  def forward(ctx, *inputs):
    T, B, D1 = inputs[0].shape
    D2 = inputs[1].shape[0] - D1
    Y = torch.zeros((T, B, D2), device=device)
    C = torch.zeros((T, B, D2), device=device)
    d = torch.zeros((B, D2), device=device)
    x_h = torch.zeros((T, B, D1 + D2), device=device)
    intern = torch.zeros((T, B, 4 * D2), device=device)
    lstm.lstm_forward_op_low_mem_var(*inputs, 0, 1, Y, C, d, x_h, intern)
    ctx.save_for_backward(*inputs, Y, C, x_h, intern)
    return Y, C, d
  
  @staticmethod
  def backward(ctx, *grad_outputs):
    DX = torch.zeros_like(ctx.saved_tensors[0], device=device)
    DW = torch.zeros_like(ctx.saved_tensors[1], device=device)
    Db = torch.zeros_like(ctx.saved_tensors[2], device=device)
    Dh = torch.zeros_like(ctx.saved_tensors[3], device=device)
    Dc = torch.zeros_like(ctx.saved_tensors[4], device=device)
    lstm.lstm_backward_op_low_mem_var(
      *ctx.saved_tensors[:6], 0, 1,
      *ctx.saved_tensors[6:8],
      grad_outputs[0], grad_outputs[2],
      DX, DW, Db, Dh, Dc,
      *ctx.saved_tensors[8:]
    )
    return DX, DW, Db, Dh, Dc, None


class LstmLowMem(nn.Module):
  
  def __init__(self, n_in, n_out):
    super(LstmLowMem, self).__init__()
    self.n_in = n_in
    self.n_out = n_out
    self.W = nn.Parameter(
      torch.empty(self.n_in + self.n_out, 4 * self.n_out, device=device).uniform_(-math.sqrt(n_out), math.sqrt(n_out)))
    self.b = nn.Parameter(torch.empty(self.n_out * 4, device=device).uniform_(-math.sqrt(n_out), math.sqrt(n_out)))
  
  def forward(self, X, i, y0, c0):
    return LstmOpLowMem.apply(X, self.W, self.b, y0, c0, i)


class LstmOp(autograd.Function):
  device = 'cpu'
  
  @staticmethod
  def forward(ctx, *inputs):
    T, B, D = inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2] // 4
    Y = torch.zeros((T, B, D), device=LstmOp.device)
    C = torch.zeros((T, B, D), device=LstmOp.device)
    H = torch.zeros((T, B, 4 * D), device=LstmOp.device)
    d = torch.zeros((B, D), device=LstmOp.device)
    lstm.lstm_forward_op(*inputs, 0, 1, Y, C, H, d)
    ctx.save_for_backward(*inputs, Y, C, H)
    return Y, C, d
  
  @staticmethod
  def backward(ctx, *grad_outputs):
    DX = torch.zeros_like(ctx.saved_tensors[0], device=LstmOp.device)
    DW = torch.zeros_like(ctx.saved_tensors[1], device=LstmOp.device)
    Dy0 = torch.zeros_like(ctx.saved_tensors[2], device=LstmOp.device)
    Dc0 = torch.zeros_like(ctx.saved_tensors[3], device=LstmOp.device)
    lstm.lstm_backward_op(
      *ctx.saved_tensors[:5], 0, 1,
      *ctx.saved_tensors[5:8],
      grad_outputs[0], grad_outputs[2],
      DX, DW, Dy0, Dc0
    )
    return DX, DW, Dy0, Dc0, None


class SingleLayerLstm(nn.Module):
  
  device = 'cpu'
  
  def __init__(self, n_input, n_hidden):
    super(SingleLayerLstm, self).__init__()
    self.input_size = n_input
    self.hidden_size = n_hidden
    self.gate_size = self.hidden_size * 4
    stdv = 1.0 / math.sqrt(self.hidden_size)  # the same initializer as PyTorch's RNN implementation
    shape_wf = (self.input_size, self.gate_size)
    shape_wr = (self.hidden_size, self.gate_size)
    self.Wf = nn.Parameter(torch.empty(shape_wf, device=device).uniform_(-stdv, stdv))
    self.Wr = nn.Parameter(torch.empty(shape_wr, device=device).uniform_(-stdv, stdv))
    self.bf = nn.Parameter(torch.empty(self.gate_size, device=device).uniform_(-2 * stdv, 2 * stdv))
  
  def forward(self, X, h0=None, c0=None, i=None):
    T, B = X.shape[0], X.shape[1]
    if h0 is None:
      h0 = torch.zeros((B, self.hidden_size), requires_grad=False, device=self.device)
      c0 = h0
    if i is None:
      i = torch.ones((T, B), requires_grad=False, device=self.device)
    intern = torch.einsum("ijk,kl->ijl", (X, self.Wf)) + self.bf
    LstmOp.device = self.device
    return LstmOp.apply(intern, self.Wr, h0, c0, i)


class LstmNative(nn.Module):
  
  device = 'cpu'
  
  def __init__(self, n_input, n_hidden, n_layers=1):
    super(LstmNative, self).__init__()
    SingleLayerLstm.device = self.device
    self.layers = nn.ModuleList([
      SingleLayerLstm(n_input, n_hidden) if i == 0 else SingleLayerLstm(n_hidden, n_hidden)
      for i in range(n_layers)
    ])
    
  def forward(self, X, h0=None, c0=None, i=None):
    X = (X, None, None)
    for lstm in self.layers:
      X = lstm(X[0], h0, c0, i)
    return X


def test_LstmNative():
  n_in, n_out, n_batch, T, ites = 2000, 512, 32, 500, 100
  m = LstmNative(n_in, n_out)
  mm = nn.LSTM(n_in, n_out)
  m.cuda()
  mm.cuda()
  m_forward, m_backward = 0, 0
  mm_forward, mm_backward = 0, 0
  
  for _ in range(ites):
    
    X = torch.randn(T, n_batch, n_in, device=device)
    i = torch.ones(T, n_batch, device=device)
    h0 = torch.zeros(n_batch, n_out, device=device)
    c0 = torch.zeros(n_batch, n_out, device=device)
    
    torch.cuda.synchronize();  start = time.time()
    y, _, d = m(X, i, h0, c0)
    torch.cuda.synchronize();  m_forward += time.time() - start
    
    loss = torch.sum((y - d) ** 2)
    
    torch.cuda.synchronize();  start = time.time()
    loss.backward()
    torch.cuda.synchronize();  m_backward += time.time() - start
    
    h0 = torch.stack((h0,))
    c0 = torch.stack((c0,))
    
    torch.cuda.synchronize();  start = time.time()
    y, (_, d) = mm(X, (h0, c0))
    torch.cuda.synchronize();  mm_forward += time.time() - start
    
    los = torch.sum((y - d) ** 2)
    
    torch.cuda.synchronize();  start = time.time()
    los.backward()
    torch.cuda.synchronize();  mm_backward += time.time() - start
  
  res = " " * 20 + "{:30s}".format("LSTM(Our Kernels)") + "{:30s}".format("LSTM(PyTorch Module)") + "\n"
  res += "{:20s}".format("Forward") + "{:30s}".format("{:.6f}".format(m_forward)) + "{:30s}".format(
    "{:.6f}".format(mm_forward)) + "\n"
  res += "{:20s}".format("Backward") + "{:30s}".format("{:.6f}".format(m_backward)) + "{:30s}".format(
    "{:.6f}".format(mm_backward))
  print(res)


# test_LstmNative()

