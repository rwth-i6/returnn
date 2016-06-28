
import theano
import numpy
import TheanoUtil
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from NetworkHiddenLayer import _NoOpLayer
from cuda_implementation.FractionalMaxPoolingOp import fmp
from math import ceil, sqrt

import sys
def my_print(op, x):
  print '===>', x.shape

  with open('/u/suryani/x.txt', 'a') as f:
    f.write('==>' + str(x) + '\n')


class CNN(_NoOpLayer):

  def __init__(self, n_features=1, filter=1, d_row=-1, pool_size=(2, 2), mode="max",
               border_mode="valid", ignore_border=True, dropout=0.0, seeds=23455, **kwargs):

    """

        :param n_features: integer
            the number of feature map(s) / filter(S) that will be used for the filter shape
            default: 1

        :param filter: integer or tuple of length 2
            the number of row(s) or/and columns(s) from the filter shape
            this filter supports either square or unsymmetric shape.
            for the square shape, we only need one integer parameter that represents row and column
            otherwise, we need to specify the tupple for (rows, cols)
            default: 1

        :param d_row: integer
            the number of row(s)/height from the input
            this has to be filled only for the first convolutional neural network layer
            the remaining layer will used the number of row from the previous layer
            default: -1

        :param pool_size: tuple of length 2
            factor by which to downscale (vertical, horizontal)
            default: (2, 2)
            Note:
            for Fractional Max Pooling, the pool_size must be symmetry (e.g: (2, 2))
            if value is (1, 1), there is no pooling layer

        :param mode: string
            mode of the pooling layer.
            this mode supports:
            "max" -- to apply the max pooling layer
            "sum" -- to apply the sum pooling layer
            "avg" -- to apply the average pooling layer without the padding
            "fmp" -- to apply the fractional max pooling
            default: "max"

        :param border_mode: string
            "valid"-- only apply filter to complete patches of the image. Generates
                      output of shape: (image_shape - filter_shape + 1)
            "full" -- zero-pads image to multiple of filter shape to generate output
                      of shape: (image_shape + filter_shape - 1)
            "same" -- the size of image will remain the same with the previous layer
            default: "valid"

        :param ignore_border: boolean
            True  -- (5, 5) input with pool_size = (2, 2), will generate a (2, 2) output.
            False -- (5, 5) input with pool_size = (2, 2), will generate a (3, 3) output.
            default: True

        :param dropout: float or double
            dropout value for regularization
            default: 0.0

        :param seeds: integer
            seeds value for random stream
            default: 23455

    """

    super(CNN, self).__init__(**kwargs)
    self.status = self.get_status(self.sources)  # [is_conv_layer, n_sources]
    self.is_1d = self.layer_class == "conv_1d"
    dimension = self.sources[0].attrs["n_out"]  # input dimension

    if self.status[0]:  # if the previous layer is convolution layer
      self.stack_size = self.sources[0].attrs["n_features"]  # set stack size from the feature maps of previous layer
      self.d_row = self.sources[0].attrs["d_row"]  # set number of input row(height) from the previous convolution layer
      dimension /= self.stack_size  # calculate the input dimension

      # check whether number of inputs are more than 1 for concatenating the inputs
      if self.status[1] != 1:
        # check the spatial dimension of all inputs
        assert all((s.attrs["n_out"] / s.attrs["n_features"]) == dimension for s in
                   self.sources), "The spatial dimension of all inputs have to be the same!"
        self.stack_size = sum([s.attrs["n_features"] for s in self.sources])  # set the stack_size by concatenating maps
    else:  # not convolution layer
      self.stack_size = 1  # set stack_size of first convolution layer as channel of the image (gray scale image)

      if self.is_1d:
        self.d_row = self.sources[0].attrs["n_out"]
      elif d_row == -1:  # assume quadratic patch
        self.d_row = int(sqrt(dimension))
      else:  # stated in JSON file
        self.d_row = d_row

      assert self.status[1] == 1, "Except CNN, the input is only one!"

    # calculate the width of input
    self.d_col = dimension / self.d_row

    # set kernel size to tuple
    if type(filter) == int:
      self.filters = [filter, filter]
    else:
      self.filters = filter

    # when kernel of convolution layer is 1x1, it gave the same size even full or valid border mode
    if self.filters == [1, 1]:
      border_mode = "valid"

    self.pool_size = pool_size

    # number of output dimension validation based on the border_mode
    if border_mode == "valid":
      new_d_row = (self.d_row - self.filters[0] + 1) / self.pool_size[0]
      new_d_col = (self.d_col - self.filters[1] + 1) / self.pool_size[1]
    elif border_mode == "full":
      new_d_row = (self.d_row + self.filters[0] - 1) / self.pool_size[0]
      new_d_col = (self.d_col + self.filters[1] - 1) / self.pool_size[1]
    elif border_mode == "same":
      new_d_row = self.d_row / self.pool_size[0]
      new_d_col = self.d_col / self.pool_size[1]
    else:
      assert False, "invalid border_mode %r" % border_mode

    self.border_mode = border_mode

    assert (mode == "max" or mode == "sum" or mode == "avg" or mode == "fmp"), "invalid pooling mode!"
    if mode == "fmp":
      new_d_row = int(ceil(new_d_row))
      new_d_col = int(ceil(new_d_col))

    assert (new_d_row > 0), "invalid spatial dimensions!"
    self.n_out = new_d_row * n_features

    if not self.is_1d:
      assert (new_d_col > 0), "invalid spatial dimensions!"
      self.n_out *= new_d_col

    self.n_features = n_features
    self.mode = mode
    self.ignore_border = ignore_border
    self.dropout = dropout
    self.seeds = seeds

    # filter shape is tuple/list of length 4 which is (nb filters, stack size, filter row, filter col)
    self.filter_shape = (self.n_features, self.stack_size, self.filters[0], self.filters[1])

    # set attributes
    self.set_attr("n_features", self.n_features)
    self.set_attr("filters", self.filters)
    self.set_attr("d_row", new_d_row)  # number of output row
    self.set_attr("pool_size", self.pool_size)
    self.set_attr("mode", self.mode)
    self.set_attr("border_mode", self.border_mode)
    self.set_attr("ignore_border", self.ignore_border)
    self.set_attr("dropout", self.dropout)
    self.set_attr("seeds", self.seeds)
    self.set_attr("n_out", self.n_out)  # number of output dimension

  def get_status(self, sources):
    n_sources = len(sources)
    is_conv_layer = all(s.layer_class in ("conv", "frac_conv", "conv_1d") for s in sources)
    return [is_conv_layer, n_sources]

  # function for calculating the weight parameter of convolution layer
  def create_weights(self, filter_shape, pool_size, seeds):
    rng = numpy.random.RandomState(seeds)
    fan_in = numpy.prod(filter_shape[1:])  # stack_size * filter_row * filter_col
    fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(pool_size))
    #         (n_features * (filter_row * filter_col)) / (pool_size[0] * pool_size[1])

    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    return self.shared(
      numpy.asarray(
        rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
        dtype=theano.config.floatX
      ),
      borrow=True,
      name="W_conv"
    )

  # function for calculating the bias parameter of convolution layer
  def create_bias(self, n_features):
    return self.shared(
      numpy.zeros(
        (n_features,),
        dtype=theano.config.floatX
      ),
      borrow=True,
      name="b_conv"
    )

  def calculate_index(self, inputs, index):
    return T.set_subtensor(inputs[((numpy.int8(1) - self.index.flatten()) > 0).nonzero()], T.zeros_like(inputs[index]))

  def calculate_dropout(self, dropout, inputs):
    assert dropout < 1.0, "Dropout have to be less than 1.0"
    mass = T.constant(1.0 / (1.0 - dropout), dtype="float32")
    random = RandomStreams(self.rng.randint(1234) + 1)

    if self.train_flag:
      inputs = inputs * T.cast(random.binomial(n=1, p=1 - dropout, size=inputs.shape), theano.config.floatX)
    else:
      inputs = inputs * mass
    return inputs

  def convolution(self, border_mode, w, inputs):
    # convolution function
    # when border mode = same, remove width and height from beginning and last based on the filter size
    if border_mode == "same":
      new_filter_size_row = (w.shape[2] - 1) / 2
      new_filter_size_col = (w.shape[3] - 1) / 2
      conv_out = conv.conv2d(
        input=inputs,
        filters=w,
        border_mode="full"
      )[:, :, new_filter_size_row:-new_filter_size_row, new_filter_size_col:-new_filter_size_col]
    else:
      conv_out = conv.conv2d(
        input=inputs,
        filters=w,
        border_mode=border_mode
      )
    conv_out.name = "conv_layer_conv_out"
    if not self.is_1d:
      conv_out = self.calculate_index(conv_out, 0)
    return conv_out

  def pooling(self, inputs, pool_size, ignore_border, modes):
    if pool_size == [1, 1]:
      return inputs

    if modes == "avg":
      modes = "average_exc_pad"

    if modes == "fmp":
      height = inputs.shape[2]
      width = inputs.shape[3]
      batch2 = inputs.shape[0]
      X = inputs.dimshuffle(2, 3, 0, 1)  # (row, col, batches, filters)
      sizes = T.zeros((batch2, 2))
      sizes = T.set_subtensor(sizes[:, 0], height)
      sizes = T.set_subtensor(sizes[:, 1], width)
      pooled_out, _ = fmp(X, sizes, pool_size[0])
      return pooled_out.dimshuffle(2, 3, 0, 1)

    return downsample.pool.pool_2d(
      input=inputs,
      ds=pool_size,
      ignore_border=ignore_border,
      mode=modes
    )

  def run_cnn(self, filter_shape, pool_size, seeds, n_features, inputs, dropout, border_mode, ignore_border, mode):
    # weight parameter
    w = self.add_param(self.create_weights(filter_shape, pool_size, seeds))

    # bias parameter
    b = self.add_param(self.create_bias(n_features))

    # dropout
    if dropout > 0.0:
      inputs = self.calculate_dropout(dropout, inputs)

    if border_mode == 'valid':
      self.index = self.index[filter_shape[1] / 2:-filter_shape[1] / 2 + 1]

    # convolutions function
    conv_out = self.convolution(border_mode, w, inputs)  # (batch, nb filters, nb row, nb col)

    # max pooling function

    pool_out = self.pooling(conv_out, pool_size, ignore_border, mode)
    self.index = downsample.pool.pool_2d(
      input=self.index.dimshuffle(1,'x',0),
      ds=[1,pool_size[1]],
      ignore_border=ignore_border,
      mode=mode
    ).dimshuffle(2,0,1).flatten(ndim=2)
    #self.index = theano.printing.Print("i", attrs=['shape'])(self.index)
    #pool_out = theano.printing.Print("p", attrs=['shape'])(pool_out)

    # calculate the output with bias parameters
    output = T.tanh(pool_out + b.dimshuffle("x", 0, "x", "x"))  # (time*batch, filter, out-row, out-col)
    output.name = "conv_layer_output_plus_bias"

    if not self.is_1d:
      output = self.calculate_index(output, 0)
    return output


class NewConv(CNN):
  layer_class = "conv"

  """
      This is class for Convolution Neural Networks
      Get the reference from deeplearning.net/tutorial/lenet.html
  """

  def __init__(self, **kwargs):
    super(NewConv, self).__init__(**kwargs)

    # our CRNN input is 3D tensor that consists of (time, batch, dim)
    # however, the convolution function only accept 4D tensor which is (batch size, stack size, nb row, nb col)
    # therefore, we should convert our input into 4D tensor
    inputs = self.sources[0].output  # (time, batch, input-dim = row * col * stack_size)
    time = inputs.shape[0]
    batch = inputs.shape[1]

    if self.status[0]:
      self.input = T.concatenate([s.Output for s in self.sources], axis=1)  # (batch, stack size, row, col)
    else:
      inputs2 = inputs.reshape((time * batch, self.d_row, self.d_col, self.stack_size))  # (time*batch, row, col, stack)
      self.input = inputs2.dimshuffle(0, 3, 1, 2)  # (batch, stack_size, row, col)
    self.input.name = "conv_layer_input_final"

    self.Output = self.run_cnn(self.filter_shape, self.pool_size, self.seeds, self.n_features, self.input, self.dropout,
                               self.border_mode, self.ignore_border, self.mode)  # (time*batch, maps, out-row, out-col)

    # our CRNN only accept 3D tensor (time, batch, dim)
    # so, we have to convert back the output to 3D tensor
    output2 = self.Output.dimshuffle(0, 2, 3, 1)  # (time*batch, out-row, out-col, filter)
    self.Output2 = output2.reshape((time, batch, output2.shape[1] * output2.shape[2] * output2.shape[3]))  # (time, batch, out-dim)
    self.make_output(self.Output2)


class ConcatConv(CNN):
  layer_class = "conv_1d"

  """
      This is class for Convolution Neural Networks that concatenated by time axis
      Get the reference from deeplearning.net/tutorial/lenet.html
  """

  def __init__(self, **kwargs):
    super(ConcatConv, self).__init__(**kwargs)

    inputs = T.concatenate([s.output for s in self.sources], axis=2)  # (time, batch, input-dim = row * features)
    time = inputs.shape[0]
    batch = inputs.shape[1]
    #self.index = T.concatenate([self.index, T.zeros((time-self.index.shape[0], self.index.shape[1]), 'int8')], axis=0)
    #self.index = theano.printing.Print(global_fn=my_print)(self.index)

    if self.status[0]:
      self.input = T.concatenate([s.tmp_Output for s in self.sources], axis=3)  # (batch, stack_size, row, time)
    else:
      inputs2 = inputs.reshape((time, batch, inputs.shape[2], self.stack_size))  # (time, batch, row, stack)
      self.input = inputs2.dimshuffle(1, 3, 2, 0)  # (batch, stack_size, row, time)

    self.input.name = "conv_layer_input_final"

    self.tmp_Output = self.run_cnn(self.filter_shape, self.pool_size, self.seeds,
                                   self.n_features, self.input, self.dropout,
                                   self.border_mode, self.ignore_border, self.mode)  # (batch, features, out-row, out-col)

    #self.index = TheanoUtil.downsample(self.index, axis=0, factor=self.pool_size[1], method="min")[:self.tmp_Output.shape[3]]
    # our CRNN only accept 3D tensor (time, batch, dim)
    # so, we have to convert back the output to 3D tensor
    output2 = self.tmp_Output.dimshuffle(3, 0, 1, 2)  # (time, batch, features, out-row)
    self.Output2 = output2.reshape((output2.shape[0], output2.shape[1], output2.shape[2] * output2.shape[3]))  # (time, batch, out-dim)
    #self.Output2 = self.calculate_index(self.Output2, 0)
    #self.Output2 = T.concatenate([self.Output2, T.zeros((self.input.shape[3] - self.index.shape[0], self.Output2.shape[1], self.Output2.shape[2]), 'int8')], axis=0)
    #self.Output2 = T.set_subtensor(self.Output2[((numpy.int8(1) - self.index.flatten()) > 0).nonzero()], T.zeros_like(self.Output2[0]))
    #self.Output2 = theano.printing.Print(global_fn=my_print)(self.Output2)
    self.make_output(self.Output2)
