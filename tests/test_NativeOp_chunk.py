
import NativeOp
import numpy
from numpy.testing.utils import assert_almost_equal
import theano.tensor as T
import TheanoUtil
import sys
f32 = "float32"


import better_exchook
from Log import log

better_exchook.replace_traceback_format_tb()
log.initialize()  # some code might need it


chunk = NativeOp.chunk
unchunk = NativeOp.unchunk
naive_chunk_start_frames = NativeOp.Chunking.naive_chunk_start_frames


def get_num_chunks(n_time, chunk_size, chunk_step):
  return len(naive_chunk_start_frames(n_time, chunk_size, chunk_step))


def naive_chunk(x, chunk_size, chunk_step):
  if x.ndim == 3:  # (time,batch,dim)
    if x.shape[1] == 1:
      return naive_chunk(x[:, 0], chunk_size, chunk_step)
    raise NotImplemented
  assert x.ndim == 2  # (time,dim)
  chunk_start_frames = naive_chunk_start_frames(x.shape[0], chunk_size, chunk_step)
  n_chunks = len(chunk_start_frames)
  out = numpy.zeros((chunk_size, n_chunks, x.shape[1]), dtype=x.dtype)
  oindex = numpy.zeros((chunk_size, n_chunks), dtype="int8")
  for b, t in enumerate(chunk_start_frames):
    t2 = min(t + chunk_size, x.shape[0])
    out[:t2 - t, b] = x[t:t2]
    oindex[:t2 - t, b] = 1
  return out, oindex


def test_chunk():
  n_time = 101
  n_batch = 1
  n_dim = 5
  chunk_size = 11
  chunk_step = 7
  numpy.random.seed(1234)
  _x = numpy.random.randn(n_time, n_batch, n_dim).astype(f32)
  _out, _oindex = naive_chunk(_x, chunk_size, chunk_step)
  _index = numpy.ones((n_time, n_batch), dtype="int8")
  x = T.as_tensor(_x)
  index = T.as_tensor(_index)
  out, oindex = chunk(x, index=index, chunk_size=chunk_size, chunk_step=chunk_step)
  _out2 = out.eval()
  _oindex2 = oindex.eval()
  assert _out.shape == _out2.shape
  assert _oindex.shape == _oindex2.shape
  assert_almost_equal(_oindex, _oindex2)
  assert_almost_equal(_out, _out2)


def test_chunk_unchunk():
  n_time = 101
  n_batch = 3
  n_dim = 5
  _x = numpy.random.randn(n_time, n_batch, n_dim).astype(f32)
  _index = numpy.ones((n_time, n_batch), dtype="int8")
  x = T.as_tensor(_x)
  index = T.as_tensor(_index)
  chunk_size = 11
  chunk_step = 7
  n_chunks = get_num_chunks(n_time, chunk_size, chunk_step)
  #print >>sys.stderr, "Should have %i n_chunks." % n_chunks
  out, oindex = chunk(x, index=index, chunk_size=chunk_size, chunk_step=chunk_step)
  x2, index2, factors = unchunk(out, index=oindex, chunk_size=chunk_size, chunk_step=chunk_step, n_time=x.shape[0], n_batch=x.shape[1])
  _x2 = x2.eval()
  _index2 = index2.eval()
  assert_almost_equal(_x, _x2)
  assert numpy.all(_index2)


def test_chunk_unchunk_grad():
  n_time = 101
  n_batch = 3
  n_dim = 5
  numpy.random.seed(1234)
  _x = numpy.random.randn(n_time, n_batch, n_dim).astype(f32)
  _index = numpy.ones((n_time, n_batch), dtype="int8")
  x = T.as_tensor(_x)
  index = T.as_tensor(_index)
  chunk_size = 11
  chunk_step = 7
  out, oindex = chunk(x, index=index, chunk_size=chunk_size, chunk_step=chunk_step)
  x2, index2, factors = unchunk(out, index=oindex, chunk_size=chunk_size, chunk_step=chunk_step, n_time=x.shape[0], n_batch=x.shape[1])
  grad = T.grad(T.sum((x2 - x) ** 2), wrt=x)
  _grad = grad.eval()
  assert_almost_equal(_grad, 0)


def test_chunk_unchunk_grad2():
  n_time = 101
  n_batch = 3
  n_dim = 5
  numpy.random.seed(1234)
  _x = numpy.random.randn(n_time, n_batch, n_dim).astype(f32)
  _Dx2 = numpy.random.randn(n_time, n_batch, n_dim).astype(f32)
  _index = numpy.ones((n_time, n_batch), dtype="int8")
  x = T.as_tensor(_x)
  Dx2 = T.as_tensor(_Dx2)
  index = T.as_tensor(_index)
  chunk_size = 11
  chunk_step = 7

  out, oindex = chunk(x, index=index, chunk_size=chunk_size, chunk_step=chunk_step)
  chunk_op = NativeOp.Chunking.make_op()
  assert type(out.owner.op) is type(chunk_op)

  x2, index2, factors = unchunk(out, index=oindex, chunk_size=chunk_size, chunk_step=chunk_step, n_time=x.shape[0], n_batch=x.shape[1])
  unchunk_op = NativeOp.UnChunking.make_op()
  assert type(x2.owner.op) is type(unchunk_op)

  Dout, _, _, _, _, _ = unchunk_op.grad(x2.owner.inputs, (Dx2, None, None))
  Dx, _, _, _, _ = chunk_op.grad(out.owner.inputs, (Dout, None))
  _Dx = Dx.eval()
  assert_almost_equal(_Dx, _Dx2)
