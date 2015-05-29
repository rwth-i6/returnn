
import numpy
from theano import tensor as T
import theano
from NetworkHiddenLayer import HiddenLayer


class RecurrentLayer(HiddenLayer):
  recurrent = True

  def __init__(self, index, reverse=False, truncation=-1, compile=True, projection=None, **kwargs):
    kwargs.setdefault("layer_class", "recurrent")
    kwargs.setdefault("activation", "tanh")
    super(RecurrentLayer, self).__init__(**kwargs)
    self.set_attr('reverse', reverse)
    self.set_attr('truncation', truncation)
    if projection: self.set_attr('projection', projection)
    n_in = sum([s.attrs['n_out'] for s in self.sources])
    n_out = self.attrs['n_out']
    self.act = self.create_bias(n_out)
    if projection:
      n_re_in = projection
    else:
      n_re_in = n_out
    self.W_re = self.add_param(self.create_random_normal_weights(n=n_re_in, m=n_out, scale=n_in,
                                                                 name="W_re_%s" % self.name))
    if projection:
      self.W_proj = self.add_param(self.create_forward_weights(n_out, projection, name='W_proj_%s' % self.name))
    else:
      self.W_proj = None
    for s, W in zip(self.sources, self.W_in):
      W.set_value(self.create_random_normal_weights(n=s.attrs['n_out'], m=n_out, scale=n_in,
                                                    name=W.name).get_value())
    self.index = index
    self.o = theano.shared(value = numpy.ones((n_out,), dtype='int8'), borrow=True)
    if compile: self.compile()

  def compile(self):
    def step(x_t, i_t, h_p):
      h_pp = T.dot(h_p, self.W_re) if self.W_proj else h_p
      i = T.outer(i_t, self.o)
      z = T.dot(h_pp, self.W_re) + self.b
      for i in range(len(self.sources)):
        z += T.dot(self.mass * self.masks[i] * x_t[i], self.W_in[i])
      #z = (T.dot(x_t, self.mass * self.mask * self.W_in) + self.b) * T.nnet.sigmoid(T.dot(h_p, self.W_re))
      h_t = (z if self.activation is None else self.activation(z))
      return h_t * i
    self.output, _ = theano.scan(step,
                                 name="scan_%s" % self.name,
                                 go_backwards=self.attrs['reverse'],
                                 truncate_gradient=self.attrs['truncation'],
                                 sequences = [T.stack(self.sources), self.index],
                                 outputs_info = [T.alloc(self.act, self.sources[0].output.shape[1], self.attrs['n_out'])])
    self.output = self.output[::-(2 * self.attrs['reverse'] - 1)]

  def create_recurrent_weights(self, n, m):
    nin = n + m + m + m
    return self.create_random_normal_weights(n, m, nin), self.create_random_normal_weights(m, m, nin)
