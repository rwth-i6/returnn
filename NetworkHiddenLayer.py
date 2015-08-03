
from theano import tensor as T
from NetworkBaseLayer import Layer
from ActivationFunctions import strtoact


class HiddenLayer(Layer):
  recurrent = False

  def __init__(self, activation="tanh", **kwargs):
    """
    :type activation: str | list[str]
    """
    kwargs.setdefault("layer_class", "hidden")
    super(HiddenLayer, self).__init__(**kwargs)
    self.set_attr('activation', activation.encode("utf8"))
    self.activation = strtoact(activation)
    self.W_in = [self.add_param(self.create_forward_weights(s.attrs['n_out'],
                                                            self.attrs['n_out'],
                                                            name="W_in_%s_%s" % (s.name, self.name)))
                 for s in self.sources]
    self.set_attr('from', ",".join([s.name for s in self.sources]))


class ForwardLayer(HiddenLayer):
  def __init__(self, **kwargs):
    kwargs.setdefault("layer_class", "hidden")
    super(ForwardLayer, self).__init__(**kwargs)
    self.z = self.b
    assert len(self.sources) == len(self.masks) == len(self.W_in)
    for s, m, W_in in zip(self.sources, self.masks, self.W_in):
      if s.attrs['sparse']:
        self.z += W_in[T.cast(s.output, 'int32')].reshape((s.output.shape[0],s.output.shape[1],s.output.shape[2] * W_in.shape[1]))
      elif m is None:
        self.z += self.dot(s.output, W_in)
      else:
        self.z += self.dot(self.mass * m * s.output, W_in)
    self.make_output(self.z if self.activation is None else self.activation(self.z))


class StateToAct(ForwardLayer):
  def __init__(self, **kwargs):
    kwargs['n_out'] = 1
    kwargs.setdefault("layer_class", "state_to_act")
    super(StateToAct, self).__init__(**kwargs)
    self.make_output(T.concatenate([s.act[-1][-1] for s in self.sources], axis=-1).dimshuffle('x',0,1).repeat(self.sources[0].output.shape[0], axis = 0))
    self.attrs['n_out'] = sum([s.attrs['n_out'] for s in self.sources])

import theano
from theano.tensor.nnet import conv
import numpy

class ConvPoolLayer(ForwardLayer):
  def __init__(self, dx, dy, fx, fy, **kwargs):
    kwargs.setdefault("layer_class", "convpool")
    kwargs['n_out'] = fx * fy
    super(ConvPoolLayer, self).__init__(**kwargs)
    self.set_attr('dx', dx) # receptive fields
    self.set_attr('dy', dy)
    self.set_attr('fx', fx) # receptive fields
    self.set_attr('fy', fy)

    # instantiate 4D tensor for input
    n_in = numpy.sum([s.output for s in self.sources])
    assert n_in == dx * dy
    x_in  = T.concatenate([s.output for s in self.sources], axis = -1).dimshuffle(0,1,2,'x').reshape(self.sources[0].shape[0], self.sources[0].shape[1],dx, dy)
    range = 1.0 / numpy.sqrt(dx*dy)
    self.W = self.add_param(theano.shared( numpy.asarray(self.rng.uniform(low=-range,high=range,size=(2,1,fx,fy)), dtype = theano.config.floatX), name = "W_%s" % self.name), name = "W_%s" % self.name)
    conv_out = conv.conv2d(input, W)

    # initialize shared variable for weights.
    w_shp = (2, 3, 9, 9)
    w_bound = numpy.sqrt(3 * 9 * 9)
    W = theano.shared( numpy.asarray(
                rng.uniform(
                    low=-1.0 / w_bound,
                    high=1.0 / w_bound,
                    size=w_shp),
                dtype=input.dtype), name ='W')

    # initialize shared variable for bias (1D tensor) with random values
    # IMPORTANT: biases are usually initialized to zero. However in this
    # particular application, we simply apply the convolutional layer to
    # an image without learning the parameters. We therefore initialize
    # them to random values to "simulate" learning.
    b_shp = (2,)
    b = theano.shared(numpy.asarray(
                rng.uniform(low=-.5, high=.5, size=b_shp),
                dtype=input.dtype), name ='b')

    # build symbolic expression that computes the convolution of input with filters in w
    conv_out = conv.conv2d(input, W)

    # build symbolic expression to add bias and apply activation function, i.e. produce neural net layer output
    # A few words on ``dimshuffle`` :
    #   ``dimshuffle`` is a powerful tool in reshaping a tensor;
    #   what it allows you to do is to shuffle dimension around
    #   but also to insert new ones along which the tensor will be
    #   broadcastable;
    #   dimshuffle('x', 2, 'x', 0, 1)
    #   This will work on 3d tensors with no broadcastable
    #   dimensions. The first dimension will be broadcastable,
    #   then we will have the third dimension of the input tensor as
    #   the second of the resulting tensor, etc. If the tensor has
    #   shape (20, 30, 40), the resulting tensor will have dimensions
    #   (1, 40, 1, 20, 30). (AxBxC tensor is mapped to 1xCx1xAxB tensor)
    #   More examples:
    #    dimshuffle('x') -> make a 0d (scalar) into a 1d vector
    #    dimshuffle(0, 1) -> identity
    #    dimshuffle(1, 0) -> inverts the first and second dimensions
    #    dimshuffle('x', 0) -> make a row out of a 1d vector (N to 1xN)
    #    dimshuffle(0, 'x') -> make a column out of a 1d vector (N to Nx1)
    #    dimshuffle(2, 0, 1) -> AxBxC to CxAxB
    #    dimshuffle(0, 'x', 1) -> AxB to Ax1xB
    #    dimshuffle(1, 'x', 0) -> AxB to Bx1xA
    output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

    # create theano function to compute filtered images
    f = theano.function([input], output)
