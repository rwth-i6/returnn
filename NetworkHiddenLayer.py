
from theano import tensor as T
from NetworkLayer import Layer


class HiddenLayer(Layer):
  recurrent = False

  def __init__(self, sources, n_out, L1=0.0, L2=0.0, activation=T.tanh, dropout=0.0, mask="unity", connection="full", layer_class="hidden", name=""):
    """
    :param list[NetworkLayer.SourceLayer] sources: list of source layers
    :type n_out: int
    :type L1: float
    :type L2: float
    :type activation: theano.Op
    :type dropout: float
    :param str mask: mask
    :param str connection: unused
    :param str layer_class: layer class name
    :param str name: name
    """
    super(HiddenLayer, self).__init__(sources, n_out, L1, L2, layer_class, mask, dropout, name=name)
    self.activation = activation
    self.W_in = [self.add_param(self.create_forward_weights(s.attrs['n_out'],
                                                            self.attrs['n_out'],
                                                            name=self.name + "_" + s.name),
                                "W_in_%s_%s" % (s.name, self.name))
                 for s in sources]
    self.set_attr('from', ",".join([s.name for s in sources]))


class ForwardLayer(HiddenLayer):
  def __init__(self, sources, n_out, L1 = 0.0, L2 = 0.0, activation = T.tanh, dropout = 0, mask = "unity", layer_class = "hidden", name = ""):
    super(ForwardLayer, self).__init__(sources, n_out, L1, L2, activation, dropout, mask, layer_class = layer_class, name = name)
    z = self.b
    assert len(sources) == len(self.masks) == len(self.W_in)
    for s, m, W_in in zip(sources, self.masks, self.W_in):
      W_in.set_value(self.create_uniform_weights(s.attrs['n_out'], n_out, s.attrs['n_out'] + n_out, "W_in_%s_%s"%(s.name, self.name)).get_value())
      if mask == "unity":
        z += T.dot(s.output, W_in)
      else:
        z += T.dot(self.mass * m * s.output, W_in)
    self.output = z if self.activation is None else self.activation(z)


class ConvPoolLayer(ForwardLayer):
  def __init__(self, sources, n_out, L1 = 0.0, L2 = 0.0, activation = T.tanh, dropout = 0, mask = "unity", layer_class = "convpool", name = ""):
    super(ConvPoolLayer, self).__init__(sources, n_out, L1, L2, activation, dropout, mask, layer_class = layer_class, name = name)
