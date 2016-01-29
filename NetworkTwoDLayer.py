from NetworkHiddenLayer import _NoOpLayer
from cuda_implementation.MultiDirectionalTwoDLSTMOp import MultiDirectionalTwoDLSTMOpInstance
from cuda_implementation.OneDToTwoDOp import OneDToTwoDOp
import theano
import theano.tensor as T
import numpy
from math import sqrt


class OneDToTwoDLayer(_NoOpLayer):
  layer_class = "1Dto2D"
  recurrent = False

  def __init__(self, **kwargs):
    super(OneDToTwoDLayer, self).__init__(**kwargs)
    assert len(self.sources) == 2
    n_in = self.sources[0].attrs['n_out']
    n_out = n_in
    sizes = T.cast(self.sources[1].output, "float32")
    assert sizes.ndim == 2
    sizes = sizes.reshape((2, sizes.size / 2)).dimshuffle(1, 0)
    self.output_sizes = sizes
    X = self.sources[0].output
    assert X.ndim == 3
    assert X.dtype == "float32"
    Y = OneDToTwoDOp()(X, sizes)
    self.output = Y
    self.set_attr('n_out', n_out)


forget_gate_initial_bias = 1.0
lambda_gate_initial_bias = 0.0


class TwoDLSTMLayer(_NoOpLayer):
  layer_class = "mdlstm"
  recurrent = True

  def __init__(self, n_out, **kwargs):
    super(TwoDLSTMLayer, self).__init__(**kwargs)
    assert len(self.sources) == 1
    source = self.sources[0]
    n_in = source.attrs['n_out']
    X = source.output
    sizes = source.output_sizes
    self.output_sizes = sizes

    b1 = self.create_and_add_bias(n_out, "1")
    b2 = self.create_and_add_bias(n_out, "2")
    b3 = self.create_and_add_bias(n_out, "3")
    b4 = self.create_and_add_bias(n_out, "4")

    W1, V_h1, V_v1 = self.create_and_add_2d_lstm_weights(n_in, n_out, "1")
    W2, V_h2, V_v2 = self.create_and_add_2d_lstm_weights(n_in, n_out, "2")
    W3, V_h3, V_v3 = self.create_and_add_2d_lstm_weights(n_in, n_out, "3")
    W4, V_h4, V_v4 = self.create_and_add_2d_lstm_weights(n_in, n_out, "4")

    Y1, Y2, Y3, Y4 = MultiDirectionalTwoDLSTMOpInstance(X, W1, W2, W3, W4, V_h1, V_h2, V_h3, V_h4,
                                                        V_v1, V_v2, V_v3, V_v4, b1, b2, b3, b4, sizes)[:4]
    Y = 0.25 * (Y1 + Y2 + Y3 + Y4)

    self.output = Y
    self.set_attr('n_out', n_out)

  def create_and_add_2d_lstm_weights(self, n, m, name_suffix):
    W, U, V = self.create_xavier_weights((n, 5 * m), "W" + name_suffix), \
              self.create_xavier_weights((m, 5 * m), "U" + name_suffix), \
              self.create_xavier_weights((m, 5 * m), "V" + name_suffix)
    W = self.add_param(W)
    U = self.add_param(U)
    V = self.add_param(V)
    return W, U, V

  def create_xavier_weights(self, shape, name):
    p = shape[0] + numpy.prod(shape[1:])
    W = numpy.asarray(self.rng.uniform(low=-sqrt(6) / sqrt(p), high = sqrt(6) / sqrt(p), size=shape),
                           dtype=theano.config.floatX)
    return theano.shared(value=W, borrow=True, name=name + "_" + self.name)

  def create_and_add_bias(self, n_cells, name_suffix):
    b_val = numpy.zeros((5 * n_cells,), dtype=theano.config.floatX)
    b_val[1 * n_cells:2 * n_cells] = forget_gate_initial_bias
    b_val[2 * n_cells:3 * n_cells] = lambda_gate_initial_bias
    b = theano.shared(b_val, borrow=True, name="b" + name_suffix + "_" + self.name)
    b = self.add_param(b)
    return b

