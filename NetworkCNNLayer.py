import theano
import numpy

from math import ceil, sqrt

from theano import tensor as T
from theano.tensor.nnet import conv2d
try:
  from theano.tensor.signal import pool
except ImportError:
  pool = None

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from NetworkHiddenLayer import _NoOpLayer
from ActivationFunctions import strtoact
from cuda_implementation.FractionalMaxPoolingOp import fmp

class CNN(_NoOpLayer):
  recurrent = True

  def __init__(self, n_features=1, filter=1, d_row=-1, border_mode="valid",
               conv_stride=(1,1), pool_size=(2,2), ignore_border=True,
               pool_stride=(2,2), pool_padding=(0,0), mode="max",
               activation="tanh", dropout=0.0, factor=0.5,
               force_sample=False, **kwargs):
    super(CNN, self).__init__(**kwargs)

    src = self.sources
    self.status = self.get_status(src)  # [is_conv_layer, n_sources]
    self.is_1d = self.layer_class == "conv_1d"
    is_resnet = self.layer_class == "resnet"

    dimension = src[0].attrs["n_out"]  # input dimension

    if self.status[0]:  # if the previous layer is convolution layer
      stack_size = src[0].attrs["n_features"]  # set stack size from the feature maps of previous layer
      d_row = src[0].attrs["d_row"]
      dimension /= stack_size

      # check whether number of inputs are more than 1 for concatenating the inputs
      if self.status[1] != 1 and (not is_resnet):
        # check the spatial dimension of all inputs
        assert all((s.attrs["n_out"] / s.attrs["n_features"]) == dimension
                   for s in src), \
          "The spatial dimension of all inputs have to be the same!"
        stack_size = sum([s.attrs["n_features"] for s in src])  # set the stack_size by concatenating maps
    else:  # not convolution layer
      stack_size = 1  # set stack_size of first convolution layer as channel of the image (gray scale image)

      if self.is_1d:
        d_row = dimension
      elif d_row == -1:
        d_row = int(sqrt(dimension))

      assert self.status[1] == 1, "Except CNN, the input is only one!"

    # calculate the width of input
    d_col = dimension/d_row

    # set kernel size to tuple
    if type(filter) == int:
      filter = [filter, filter]

    if filter == [1, 1]:
      border_mode = "valid"


    if is_resnet:
      n_features = src[1].attrs['n_features']
      border_mode = "same"
      pool_size = [1, 1]

    _, new_d_row = self.get_dim(d_row, filter[0], pool_size[0],
                                border_mode, conv_stride[0])
    border_mode, new_d_col = self.get_dim(d_col, filter[1], pool_size[1],
                                          border_mode, conv_stride[1])

    assert (mode == "max" or mode == "sum" or
            mode == "avg" or mode == "fmp"), "invalid pooling mode!"

    if mode == "fmp":
      new_d_row = int(ceil(new_d_row))
      new_d_col = int(ceil(new_d_col))

    assert (new_d_row > 0), "invalid spatial rows dimensions!"
    n_out = new_d_row * n_features

    if not self.is_1d:
      assert (new_d_col > 0), "invalid spatial columns dimensions!"
      n_out *= new_d_col

    # filter shape is tuple/list of length 4 which is (nb filters, stack size, filter row, filter col)
    self.filter_shape = (n_features, stack_size, filter[0], filter[1])
    self.input_shape = [d_row, d_col]
    self.modes = [border_mode, ignore_border, mode, activation]
    self.pool_params = [pool_size, pool_stride, pool_padding, conv_stride]
    self.other_params = [dropout, factor]

    self.force_sample = bool(force_sample)

    # set attributes
    self.set_attr("n_features", n_features)
    self.set_attr("d_row", new_d_row)  # number of output row
    self.set_attr("n_out", n_out)  # number of output dimension

  def get_status(self, sources):
    n_sources = len(sources)
    is_conv_layer = all(s.layer_class in ("conv", "frac_conv", "conv_1d", "resnet")
                        for s in sources)
    return [is_conv_layer, n_sources]

  def get_dim(self, input, filters, pools, border_mode, stride):
    if border_mode == "valid":
      result = (input - filters + 1)
    elif border_mode == "full":
      result = (input + filters - 1)
    elif border_mode == "same":
      border_mode = "half"
      result = input
    else:
      assert False, "Invalid border_mode!!!"

    if stride != 1:
      result = int(ceil(result/float(stride)))

    result /= pools

    return border_mode, result

  def calculate_index(self, inputs):
    if inputs.ndim == 3:  # TBD
      return T.set_subtensor(
        inputs[((numpy.int8(1) - self.index.flatten()) > 0).nonzero()],
        T.zeros_like(inputs[0])
      )
    else:  # assume BFHW
      B = inputs.shape[0]
      inputs = inputs.dimshuffle(3, 0, 1, 2)  # WBFH
      inputs = self.calculate_index(
        inputs.reshape(
          (inputs.shape[0] * inputs.shape[1],
           inputs.shape[2],
           inputs.shape[3])
        )
      )
      return inputs.reshape((inputs.shape[0] / B, B, inputs.shape[1],
                             inputs.shape[2])).dimshuffle(1, 2, 3, 0)

  def calculate_dropout(self, dropout, inputs):
    assert dropout < 1.0, "Dropout have to be less than 1.0"
    mass = T.constant(1.0 / (1.0 - dropout), dtype="float32")
    random = RandomStreams(self.rng.randint(1234) + 1)

    if self.train_flag:
      inputs = inputs * T.cast(
        random.binomial(n=1, p=1 - dropout, size=inputs.shape),
        theano.config.floatX
      )
    else:
      inputs = inputs * mass
    return inputs

  def convolution(self, inputs, filter_shape, stride, border_mode, factor, pool_size):
    #W_bound = numpy.sqrt(2./filter_shape[0]*numpy.prod(filter_shape[2:]))*factor
    fan_in = numpy.prod(filter_shape[1:])  # stack_size * filter_row * filter_col
    fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(pool_size))
    #         (n_features * (filter_row * filter_col)) / (pool_size[0] * pool_size[1])

    W_bound = numpy.sqrt(6. / (fan_in + fan_out)) * factor
    W = self.add_param(
      self.shared(
        value=numpy.asarray(
          self.rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
          dtype='float32'
        ),
        borrow=True,
        name="W_conv_" + self.name
      )
    )

    conv_out = conv2d(
      input=inputs,
      filters=W,
      filter_shape=filter_shape,
      subsample=stride,
      border_mode=border_mode
    )

    conv_out.name = "conv_out_" + self.name
    conv_out = self.calculate_index(conv_out)
    return conv_out

  def pooling(self, inputs, pool_size, ignore_border, stride, pad, mode):
    if pool_size == [1, 1]:
      return inputs

    if mode == "avg":
      mode = "average_exc_pad"

    if mode == "fmp":
      height = inputs.shape[2]
      width = inputs.shape[3]
      batch = inputs.shape[0]
      X = inputs.dimshuffle(2, 3, 0, 1)  # (row, col, batches, filters)
      sizes = T.zeros((batch, 2))
      sizes = T.set_subtensor(sizes[:, 0], height)
      sizes = T.set_subtensor(sizes[:, 1], width)
      pooled_out, _ = fmp(X, sizes, pool_size[0])
      return pooled_out.dimshuffle(2, 3, 0, 1)

    pool_out = pool.pool_2d(
      input=inputs,
      ds=pool_size,
      ignore_border=ignore_border,
      st=stride,
      padding=pad,
      mode=mode
    )
    pool_out.name = "pool_out_"+self.name
    return pool_out

  def bias_term(self, inputs, n_features, activation):
    b = self.add_param(
      self.shared(
        value=numpy.zeros((n_features,), dtype='float32'),
        borrow=True,
        name="b_conv_" + self.name
      )
    )

    act = strtoact('identity') if activation == 'maxout' else strtoact(activation)
    output = act(inputs + b.dimshuffle("x", 0, "x", "x"))  # (time*batch, filter, out-row, out-col)
    output.name = "output_bias_term_"+self.name
    output = self.calculate_index(output)
    return output

  def run_cnn(self, inputs, filter_shape, params, modes, others):
    # dropout
    if others[0] > 0.0:
      inputs = self.calculate_dropout(others[0], inputs)

    conv_out = self.convolution(inputs, filter_shape, params[3], modes[0], others[1], params[0])
    pool_out = self.pooling(conv_out, params[0], modes[1], params[1], params[2], modes[2])

    if self.is_1d:
      self.index = self.pooling(self.index.dimshuffle(1, 'x', 0),
                                [1, params[0][1]],
                                modes[1],
                                params[1],
                                params[2],
                                modes[2]).dimshuffle(2, 0, 1).flatten(2)

    output = self.bias_term(pool_out, filter_shape[0], modes[3])

    return output


class NewConv(CNN):
  layer_class = "conv"

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
      inputs2 = inputs.reshape((time * batch, self.input_shape[0],
                                self.input_shape[1], self.filter_shape[1]))  # (time*batch, row, col, stack)
      self.input = inputs2.dimshuffle(0, 3, 1, 2)  # (batch, stack_size, row, col)
    self.input.name = "conv_layer_input_final"

    if self.modes[3] != "tanh":
      act = strtoact(self.modes[3])
      self.modes[3] = "identity"

    self.Output = self.run_cnn(
      inputs=self.input,
      filter_shape=self.filter_shape,
      params=self.pool_params,
      modes=self.modes,
      others=self.other_params
    )

    # our CRNN only accept 3D tensor (time, batch, dim)
    # so, we have to convert back the output to 3D tensor
    # self.make_output(self.Output2)
    if self.attrs['batch_norm']:
      self.Output = self.batch_norm(
        h=self.Output.reshape(
          (self.Output.shape[0],
           self.Output.shape[1] * self.Output.shape[2] * self.Output.shape[3])
        ),
        dim=self.attrs['n_out'],
        force_sample=self.force_sample
      ).reshape(self.Output.shape)
      if self.modes[3] != "tanh":
        self.Output = act(self.Output)

    if self.modes[3] == 'maxout':
      self.Output = T.max(self.Output, axis=1).dimshuffle(0, 'x', 1, 2)
      self.attrs['n_out'] /= self.attrs['n_features']
      self.attrs['n_features'] = 1

    output2 = self.Output.dimshuffle(0, 2, 3, 1)  # (time*batch, out-row, out-col, filter)
    self.output = output2.reshape((time, batch, output2.shape[1] * output2.shape[2] * output2.shape[3]))  # (time, batch, out-dim)


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

    if self.status[0]:
      self.input = T.concatenate([s.Output for s in self.sources], axis=3)  # (batch, stack_size, row, time)
    else:
      inputs2 = inputs.reshape((time, batch, inputs.shape[2], self.filter_shape[1]))  # (time, batch, row, stack)
      self.input = inputs2.dimshuffle(1, 3, 2, 0)  # (batch, stack_size, row, time)
    self.input.name = "conv_layer_input_final"

    if self.pool_params[0][1] > 1:
      xp = T.constant(self.pool_params[0][1], 'int32')
      self.input = T.concatenate([self.input, T.zeros((batch, self.filter_shape[1], self.input.shape[2],
                                                       xp - T.mod(self.input.shape[3], xp)), 'float32')], axis=3)
      self.index = T.concatenate([self.index, T.zeros((xp - T.mod(self.index.shape[0], xp), batch), 'int8')], axis=0)

    if self.modes[0] == "valid":
      if self.filter_shape[3] > 1:
        idx = int(self.filter_shape[3] / 2)
        self.index = self.index[idx:-idx]

    self.Output = self.run_cnn(
      inputs=self.input,
      filter_shape=self.filter_shape,
      params=self.pool_params,
      modes=self.modes,
      others=self.other_params
    )

    if self.attrs['batch_norm']:
      self.Output = self.batch_norm(
        h=self.Output.dimshuffle(0, 2, 3, 1).reshape(
          (self.Output.shape[0] * self.Output.shape[2] * self.Output.shape[3],
           self.Output.shape[1])
        ),
        dim=self.attrs['n_features'],
        force_sample=self.force_sample
      ).reshape((self.Output.shape[0],
                 self.Output.shape[2],
                 self.Output.shape[3],
                 self.Output.shape[1])).dimshuffle(0, 3, 1, 2)

    # our CRNN only accept 3D tensor (time, batch, dim)
    # so, we have to convert back the output to 3D tensor
    output2 = self.Output.dimshuffle(3, 0, 1, 2)  # (time, batch, features, out-row)
    self.output = output2.reshape((output2.shape[0], output2.shape[1],
                                   output2.shape[2] * output2.shape[3]))  # (time, batch, out-dim)


class ResNet(CNN):
  layer_class = "resnet"

  def __init__(self, **kwargs):
    super(ResNet, self).__init__(**kwargs)

    assert self.status[1] == 2, "Only accept 2 sources!"
    assert self.status[0], "Only accept cnn layers!"

    x = self.sources[0]
    f_x = self.sources[1]

    time = x.output.shape[0]
    batch = x.output.shape[1]

    self.input = T.add(x.Output, f_x.Output)
    self.Output = T.nnet.relu(self.input)

    if self.attrs['batch_norm']:
      self.Output = self.batch_norm(
        h=self.Output.reshape(
          (self.Output.shape[0],
           self.Output.shape[1] * self.Output.shape[2] * self.Output.shape[3])
        ),
        dim=self.attrs['n_out'],
        force_sample=self.force_sample
      ).reshape(self.Output.shape)

    output2 = self.Output.dimshuffle(0, 2, 3, 1)  # (time*batch, out-row, out-col, filter)
    self.output = output2.reshape((time, batch, output2.shape[1] * output2.shape[2] * output2.shape[3]))  # (time, batch, out-dim)

