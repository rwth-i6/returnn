from NetworkHiddenLayer import Layer
from Log import log
from cuda_implementation.OneDToTwoDOp import OneDToTwoDOp
from cuda_implementation.CropToBatchImageSizeOp import CropToBatchImageSizeInstance
from cuda_implementation.MultiDirectionalTwoDLSTMOp import MultiDirectionalTwoDLSTMOpInstance
from cuda_implementation.CuDNNConvHWBCOp import CuDNNConvHWBCOpValidInstance
from cuda_implementation.PoolHWBCOp import PoolHWBCOp
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import numpy
from math import sqrt
from ActivationFunctions import strtoact


class TwoDBaseLayer(Layer):
  def __init__(self, n_out, **kwargs):
    kwargs['n_out'] = n_out
    super(TwoDBaseLayer, self).__init__(**kwargs)
    #like in _NoOpLayer
    self.params = {}  # Reset all params.
    self.set_attr('from', ",".join([s.name for s in self.sources]))

  def create_xavier_weights(self, shape, name):
    p = shape[0] + numpy.prod(shape[1:])
    W = numpy.asarray(self.rng.uniform(low=-sqrt(6) / sqrt(p), high = sqrt(6) / sqrt(p), size=shape),
                      dtype=theano.config.floatX)
    return theano.shared(value=W, borrow=True, name=name + "_" + self.name)


class OneDToTwoDLayer(TwoDBaseLayer):
  layer_class = "1Dto2D"
  recurrent = False

  def __init__(self, **kwargs):
    super(OneDToTwoDLayer, self).__init__(1, **kwargs)
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


class OneDToTwoDFixedSizeLayer(TwoDBaseLayer):
  layer_class = "1Dto2D_fixed_size"
  recurrent = True

  def __init__(self, **kwargs):
    super(OneDToTwoDFixedSizeLayer, self).__init__(1, **kwargs)
    assert len(self.sources) == 1
    X = self.sources[0].output
    assert X.ndim == 3
    assert X.dtype == "float32"

    height = X.shape[2]
    width = T.maximum(T.sum(self.index, axis=0), T.ones_like(self.index[0]))
    batch = X.shape[1]
    sizes = T.zeros((batch, 2), dtype="float32")
    sizes = T.set_subtensor(sizes[:, 0], height)
    sizes = T.set_subtensor(sizes[:, 1], width)
    Y = T.unbroadcast(X.dimshuffle(2, 0, 1, 'x'), 3)
    self.output = Y
    self.output_sizes = sizes
    n_out = 1
    self.set_attr('n_out', n_out)


forget_gate_initial_bias = 1.0
lambda_gate_initial_bias = 0.0


class TwoDLSTMLayer(TwoDBaseLayer):
  layer_class = "mdlstm"
  recurrent = True

  def __init__(self, n_out, collapse_output=False, **kwargs):
    super(TwoDLSTMLayer, self).__init__(n_out, **kwargs)
    assert len(self.sources) == 1
    source = self.sources[0]
    n_in = source.attrs['n_out']
    X = source.output
    assert X.ndim == 4
    sizes = source.output_sizes
    self.output_sizes = sizes

    #dropout
    assert len(self.masks) == 1
    mask = self.masks[0]
    if mask is not None:
      X = self.mass * mask * X

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

    self.set_attr('n_out', n_out)
    self.set_attr('collapse_output', collapse_output)
    if collapse_output:
      Y = Y.sum(axis=0)
      self.index = T.ones((Y.shape[0],Y.shape[1]),dtype='int8')
    self.output = Y

  def create_and_add_2d_lstm_weights(self, n, m, name_suffix):
    W, U, V = self.create_xavier_weights((n, 5 * m), "W" + name_suffix), \
              self.create_xavier_weights((m, 5 * m), "U" + name_suffix), \
              self.create_xavier_weights((m, 5 * m), "V" + name_suffix)
    W = self.add_param(W)
    U = self.add_param(U)
    V = self.add_param(V)
    return W, U, V

  def create_and_add_bias(self, n_cells, name_suffix):
    b_val = numpy.zeros((5 * n_cells,), dtype=theano.config.floatX)
    b_val[1 * n_cells:2 * n_cells] = forget_gate_initial_bias
    b_val[2 * n_cells:3 * n_cells] = lambda_gate_initial_bias
    b = theano.shared(b_val, borrow=True, name="b" + name_suffix + "_" + self.name)
    b = self.add_param(b)
    return b


printed_cudnn_warning = False


def conv_crop_pool_op(X, sizes, W, b, n_in, n_maps, filter_height, filter_width, poolsize):
  global printed_cudnn_warning
  import theano.sandbox.cuda as theano_cuda
  have_cudnn = theano_cuda.cuda_enabled and theano.sandbox.cuda.dnn.dnn_available()
  if theano_cuda.cuda_enabled and not have_cudnn and not printed_cudnn_warning:
    print >> log.v1, "warning, cudnn not available, using theano conv implementation"
    printed_cudnn_warning = True

  if have_cudnn:
    conv_op = CuDNNConvHWBCOpValidInstance
    pool_op = PoolHWBCOp(poolsize)
    conv_out = conv_op(X, W, b)
    crop_out = CropToBatchImageSizeInstance(conv_out, sizes)
    Y = pool_op(crop_out)
    return Y
  else:
    #note: this solution uses alot of dimshuffles and so also alot of memory
    #I only have this so that I can still run on my laptop for testing
    #it's not really useful for productive use and also not much tested
    filter_shape = (n_maps, n_in, filter_height, filter_width)
    X_shuffled = X.dimshuffle(2, 3, 0, 1)
    conv_out = conv.conv2d(input=X_shuffled, border_mode="valid", filters=W, filter_shape=filter_shape,
                           image_shape=(None, n_in, None, None))
    crop_out = CropToBatchImageSizeInstance(conv_out.dimshuffle(2, 3, 0, 1), sizes).dimshuffle(2, 3, 0, 1)
    if poolsize == (1, 1):
      Y = crop_out
    else:
      #pooling cannot handle width > 512 (only with cuDNN), so we swap the axes and swap them back afterwards
      crop_out = crop_out.dimshuffle(0, 1, 3, 2)
      pooled_out = downsample.max_pool_2d(
        input=crop_out,
        #max_pool_2d wants the sizes in the other order
        ds=poolsize[::-1],
        ignore_border=True
      )
      #unshuffle it
      Y = pooled_out.dimshuffle(0, 1, 3, 2)
    Y = Y.dimshuffle(2, 3, 0, 1)
    Y += b
    return Y


class ConvPoolLayer2(TwoDBaseLayer):
  layer_class = "conv2"
  recurrent = False

  def __init__(self, n_features, filter, pool_size, activation="tanh", **kwargs):
    kwargs['n_out'] = n_features
    super(ConvPoolLayer2, self).__init__(**kwargs)
    assert len(self.sources) == 1
    source = self.sources[0]
    n_in = source.attrs['n_out']
    X = source.output
    assert X.ndim == 4
    sizes = source.output_sizes

    self.set_attr('n_features', n_features)
    self.set_attr('filter', filter)
    self.set_attr('pool_size', pool_size)
    self.set_attr('activation', activation)

    #TODO: maybe this ordering is not consistent with Dewis implementation
    self.filter_height = filter[0]
    self.filter_width = filter[1]
    self.pool_size = pool_size

    W = self.create_conv_weights(n_features, n_in, self.filter_height, self.filter_width)
    b = self.create_and_add_bias(n_features)

    Z = conv_crop_pool_op(X, sizes, W, b, n_in, n_features, self.filter_height, self.filter_width, pool_size)
    Y = strtoact(activation)(Z)
    self.output = Y
    self.output_sizes = self.output_size_from_input_size(sizes)
    self.set_attr('n_out', n_features)

  def create_conv_weights(self, n_features, n_in, filter_height, filter_width, name_suffix = ""):
    filter_shape = (n_features, n_in, filter_height, filter_width)
    W = self.create_xavier_weights(filter_shape, "W" + name_suffix)
    W = self.add_param(W)
    return W

  def create_and_add_bias(self, n_out, name_suffix=""):
    b_val = numpy.zeros((n_out,), dtype=theano.config.floatX)
    b = theano.shared(b_val, borrow=True, name="b" + name_suffix + "_" + self.name)
    b = self.add_param(b)
    return b

  def output_size_from_input_size(self, sizes):
    heights = sizes[:, 0]
    widths = sizes[:, 1]
    heights = heights - self.filter_height + 1
    widths = widths - self.filter_width + 1
    p1, p2 = self.pool_size
    heights //= p1
    widths //= p2
    return T.concatenate((heights[:, None], widths[:, None]), axis=1)
