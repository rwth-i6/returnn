
import sys
import os

import _setup_test_env  # noqa
import returnn.native_op as native_op
import numpy
from numpy.testing.utils import assert_almost_equal
import theano.tensor as T
import returnn.theano.util as theano_util
import sys
f32 = "float32"


from returnn.util import better_exchook
from returnn.log import log

better_exchook.replace_traceback_format_tb()
log.initialize()  # some code might need it
theano_util.monkey_patches()


from returnn.theano.native_op import subtensor_batched_index


def test_1():
  n_time = 11
  n_batch = 3
  n_dim = 5
  numpy.random.seed(1234)
  _x = numpy.random.randn(n_time, n_batch, n_dim).astype(f32)
  _idx = numpy.random.randint(0, n_dim, (n_time, n_batch))
  assert _idx.shape == (n_time, n_batch)
  x = T.as_tensor(_x)
  idx = T.as_tensor(_idx)
  y = subtensor_batched_index(x, idx)
  ts = T.arange(x.shape[0] * x.shape[1])
  y2 = x.reshape((ts.shape[0], x.shape[2]))[ts, idx.flatten()[ts]].reshape(idx.shape)
  _y = y.eval()
  _y2 = y2.eval()
  assert_almost_equal(_y, _y2)


def test_2():
  n_time = 11
  n_dim = 5
  numpy.random.seed(1234)
  _x = numpy.random.randn(n_time, n_dim).astype(f32)
  _idx = numpy.random.randint(0, n_dim, (n_time,))
  assert _idx.shape == (n_time,)
  x = T.as_tensor(_x)
  idx = T.as_tensor(_idx)
  y = subtensor_batched_index(x, idx)
  ts = T.arange(x.shape[0])
  y2 = x[ts, idx[ts]]
  _y = y.eval()
  _y2 = y2.eval()
  assert_almost_equal(_y, _y2)


def test_grad_1():
  n_time = 11
  n_dim = 5
  numpy.random.seed(1234)
  _x = numpy.random.randn(n_time, n_dim).astype(f32)  # should not be needed
  _Dy = numpy.random.randn(n_time).astype(f32)
  _idx = numpy.random.randint(0, n_dim, (n_time,))
  assert _Dy.shape == _idx.shape
  assert _idx.shape == (n_time,)
  x = T.as_tensor(_x)
  Dy = T.as_tensor(_Dy)
  idx = T.as_tensor(_idx)
  y = subtensor_batched_index(x, idx)
  Dx = T.grad(None, x, known_grads={y: Dy})
  _Dx = Dx.eval()
  ts = T.arange(x.shape[0])
  Dx2 = T.zeros((n_time, n_dim), dtype=f32)
  Dx2 = T.set_subtensor(Dx2[ts, idx[ts]], Dy)
  _Dx2 = Dx2.eval()
  assert_almost_equal(_Dx, _Dx2)
