from __future__ import print_function

import theano
import numpy
try:
  import scipy
  import scipy.signal
except ImportError:
  scipy = None
import json
import h5py
import sys
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.ifelse import ifelse
try:
  from theano.tensor.signal import pool
except ImportError:  # old Theano or so...
  pool = None
from returnn.util.basic import unicode, long
from returnn.theano.layers.base import Layer
from returnn.theano.activation_functions import strtoact, strtoact_single_joined, elu
import returnn.theano.util as theano_util
from returnn.theano.util import class_idx_seq_to_1_of_k
from returnn.log import log
from math import ceil
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from returnn.theano.util import print_to_file, DumpOp
from returnn.theano.ops.inv_align import InvAlignOp


class HiddenLayer(Layer):
  def __init__(self, activation="sigmoid", **kwargs):
    """
    :type activation: str | list[str]
    """
    super(HiddenLayer, self).__init__(**kwargs)
    self.set_attr('activation', activation.encode("utf8"))
    self.activation = strtoact(activation)
    self.W_in = [self.add_param(self.create_forward_weights(s.attrs['n_out'],
                                                            self.attrs['n_out'],
                                                            name="W_in_%s_%s" % (s.name, self.name)))
                 for s in self.sources]
    self.set_attr('from', ",".join([s.name for s in self.sources]))

  def get_linear_forward_output(self, with_bias=True, sources=None):
    if with_bias:
      z = self.b
    else:
      z = 0
    if sources is None:
      sources = self.sources
    assert len(sources) == len(self.masks) == len(self.W_in)
    for s, m, W_in in zip(sources, self.masks, self.W_in):
      if s.attrs['sparse']:
        if s.output.ndim == 3: out_dim = s.output.shape[2]
        elif s.output.ndim == 2: out_dim = 1
        else: assert False, s.output.ndim
        z += W_in[T.cast(s.output, 'int32')].reshape((s.output.shape[0],s.output.shape[1],out_dim * W_in.shape[1]))
      elif m is None:
        z += self.dot(s.output, W_in)
      else:
        z += self.dot(self.mass * m * s.output, W_in)
    if self.attrs.get('input_scale', 1.0) != 1.0:
      z *= numpy.float32(self.attrs['input_scale'])
    return z


class ForwardLayer(HiddenLayer):
  layer_class = "hidden"

  def __init__(self, sparse_window = 1, **kwargs):
    super(ForwardLayer, self).__init__(**kwargs)
    self.set_attr('sparse_window', sparse_window) # TODO this is ugly
    self.attrs['n_out'] = sparse_window * kwargs['n_out']
    self.z = self.get_linear_forward_output()
    self.make_output(self.z if self.activation is None else self.activation(self.z))


class SharedForwardLayer(HiddenLayer):
  layer_class = "hidden_shared"

  def __init__(self, base = None, sparse_window = 1, **kwargs):
    kwargs['n_out'] = base[0].b.get_value().shape[0]
    super(SharedForwardLayer, self).__init__(**kwargs)
    self.params = {}
    self.W_in = base[0].W_in
    self.b = base[0].b
    self.set_attr('sparse_window', sparse_window) # TODO this is ugly
    self.attrs['n_out'] = sparse_window * kwargs['n_out']
    self.z = self.get_linear_forward_output()
    self.make_output(self.z if self.activation is None else self.activation(self.z))

class ClippingLayer(HiddenLayer):
  layer_class = "clip"

  def __init__(self, sparse_window = 1, **kwargs):
    super(ClippingLayer, self).__init__(**kwargs)
    z = self.get_linear_forward_output()
    target = 'classes' if not 'target' in self.attrs else self.attrs['target']
    i = (self.y_in[target].flatten() > 0).nonzero()
    znew = z.reshape((z.shape[0]*z.shape[1],z.shape[2]))
    #self.make_output(z)
    self.make_output(znew[i].reshape((T.sum(self.y_in[target]), z.shape[1], z.shape[2])))
    self.index = T.ones((self.output.shape[0], self.output.shape[1]), 'int8')

class EmbeddingLayer(ForwardLayer):
  layer_class = "embedding"

  def __init__(self, **kwargs):
    super(EmbeddingLayer, self).__init__(**kwargs)
    self.z -= self.b
    self.make_output(self.z if self.activation is None else self.activation(self.z))


class _NoOpLayer(Layer):
  """
  Use this as a base class if you want to remove all params by the Layer base class.
  Note that this overwrites n_out, so take care of that yourself.
  """
  def __init__(self, **kwargs):
    # The base class will already have a bias.
    # We will reset all this.
    # This is easier for now than to refactor the ForwardLayer.
    kwargs['n_out'] = 1  # This is a hack so that the super init is fast. Will be reset later.
    super(_NoOpLayer, self).__init__(**kwargs)
    self.params = {}  # Reset all params.
    self.set_attr('from', ",".join([s.name for s in self.sources]))


def concat_sources(sources, masks=None, mass=None, unsparse=False, expect_source=True):
  """
  :type sources: list[Layer]
  :type masks: None | list[theano.Variable]
  :type mass: None | theano.Variable
  :param bool unsparse: whether to make sparse sources into 1-of-k
  :param bool expect_source: whether to throw an exception if there is no source
  :returns (concatenated sources, out dim)
  :rtype: (theano.Variable, int)
  """
  if masks is None: masks = [None] * len(sources)
  else: assert mass
  assert len(sources) == len(masks)
  zs = []
  n_out = 0
  have_sparse = False
  have_non_sparse = False
  for s, m in zip(sources, masks):
    if s.attrs['sparse']:
      if s.output.ndim == 3: out = s.output.reshape((s.output.shape[0], s.output.shape[1]))
      elif s.output.ndim == 2: out = s.output
      else: assert False, s.output.ndim
      if unsparse:
        n_out += s.attrs['n_out']
        have_non_sparse = True
        out_1_of_k = class_idx_seq_to_1_of_k(out, num_classes=s.attrs['n_out'])
        zs += [out_1_of_k]
      else:
        zs += [out.reshape((out.shape[0], out.shape[1], 1))]
        assert not have_non_sparse, "mixing sparse and non-sparse sources"
        if not have_sparse:
          have_sparse = True
          n_out = s.attrs['n_out']
        else:
          assert n_out == s.attrs['n_out'], "expect same num labels but got %i != %i" % (n_out, s.attrs['n_out'])
    else:  # non-sparse source
      n_out += s.attrs['n_out']
      have_non_sparse = True
      assert not have_sparse, "mixing sparse and non-sparse sources"
      if m is None:
        zs += [s.output]
      else:
        zs += [mass * m * s.output]
  if len(zs) > 1:
    # We get (time,batch,dim) input shape.
    # Concat over dimension, axis=2.
    return T.concatenate(zs, axis=2), n_out
  elif len(zs) == 1:
    return zs[0], n_out
  else:
    if expect_source:
      raise Exception("We expected at least one source but did not get any.")
    return None, 0
_concat_sources = concat_sources


class CopyLayer(_NoOpLayer):
  """
  It's mostly the Identity function. But it will make sparse to non-sparse.
  """
  layer_class = "copy"

  def __init__(self, activation=None, **kwargs):
    super(CopyLayer, self).__init__(**kwargs)
    if activation:
      self.set_attr('activation', activation.encode("utf8"))
    act_f = strtoact_single_joined(activation)
    self.z, n_out = concat_sources(self.sources, masks=self.masks, mass=self.mass, unsparse=True)
    self.set_attr('n_out', n_out)
    self.make_output(act_f(self.z))


class WindowLayer(_NoOpLayer):
  layer_class = "window"

  def __init__(self, window, delta=0, delta_delta=0, **kwargs):
    super(WindowLayer, self).__init__(**kwargs)
    source, n_out = concat_sources(self.sources, unsparse=False)
    self.set_attr('n_out', n_out * window)
    self.set_attr('window', window)
    self.set_attr('delta', delta)
    self.set_attr('delta_delta', delta_delta)
    from returnn.theano.util import windowed_batch, delta_batch
    out = windowed_batch(source, window=window)
    #d = delta_batch()  # TODO...
    self.make_output(out)


class WindowContextLayer(_NoOpLayer):
  layer_class = "window_context"

  def __init__(self, window, average='concat', direction = -1, scan=False, n_out=None, **kwargs):
    super(WindowContextLayer, self).__init__(**kwargs)
    source, n_in = concat_sources(self.sources, unsparse=False)
    if n_out is not None:
      b = self.create_bias(n_out)
      W = self.create_random_normal_weights(n_in, n_out)
      source = T.tanh(b + T.dot(source, W))
    else:
      n_out = n_in

    self.set_attr('n_out', n_out)
    self.set_attr('window', window)
    self.set_attr('average', average)
    self.set_attr('direction', direction)

    if average == 'exponential':
      weights = numpy.float32(1) / T.arange(1, window + 1,dtype='float32')[::-1]
    elif average == 'uniform':
      weights = numpy.float32(1) / (T.cast(window,'float32') * T.ones((window,),'float32'))
    elif average == 'concat':
      weights = None
      self.set_attr('n_out', n_out * window)
    else:
      assert False, "invalid averaging method: " + str(average)

    if scan:
      source = source[::-direction]
      inp = T.concatenate([T.zeros((window - 1, source.shape[1], source.shape[2]), 'float32'), source], axis=0)
      def wnd(x, i, inp, weights):
        return T.dot(inp[i:i + window].dimshuffle(1, 2, 0), weights), i
      mapped_out, _ = theano.map(wnd, sequences=[source, T.arange(source.shape[0])], non_sequences=[inp, weights])
      self.make_output(mapped_out[0][::-direction])
    else:
      from returnn.theano.util import context_batched
      out = context_batched(source[::-direction], window=window)[::-direction]
      self.make_output(out)


class DownsampleLayer(_NoOpLayer):
  """
  E.g. method == "average", axis == 0, factor == 2 -> each 2 time-frames are averaged.
  See TheanoUtil.downsample. You can also use method == "max".
  """
  layer_class = "downsample"

  def __init__(self, factor, axis, method="average", padding=False, sample_target=False, fit_target=False, base=None, **kwargs):
    super(DownsampleLayer, self).__init__(**kwargs)
    self.set_attr("method", method)
    if isinstance(axis, (str)):
      axis = json.loads(axis)
    if isinstance(axis, set): axis = tuple(axis)
    assert isinstance(axis, int) or isinstance(axis, (tuple, list)), "int or list[int] expected for axis"
    if isinstance(axis, int): axis = [axis]
    axis = list(sorted(axis))
    self.set_attr("axis", axis)
    if isinstance(factor, (str)):
      factor = json.loads(factor)
    assert isinstance(factor, (int, float)) or isinstance(axis, (tuple, list)), "int|float or list[int|float] expected for factor"
    if isinstance(factor, (int, float)): factor = [factor] * len(axis)
    assert len(factor) == len(axis)
    self.set_attr("factor", factor)
    z, z_dim = concat_sources(self.sources, unsparse=False)
    target = self.attrs.get('target','classes')
    self.y_out = self.network.y[target] if base is None else base[0].y_out
    self.index_out =  self.network.j[target] if base is None else base[0].index_out
    n_out = z_dim
    import theano.ifelse
    for f, a in zip(factor, axis):
      if f == 1:
        continue
      if a == 0:
        if padding:
          z = T.concatenate([z,T.zeros((f-T.mod(z.shape[a], f), z.shape[1], z.shape[2]), 'float32')],axis=0)
        z = theano_util.downsample(z, axis=a, factor=f, method=method)
        if sample_target or fit_target:
          if self.y_out.dtype == 'float32':
            if padding:
              self.y_out = T.concatenate(
                [self.y_out, T.zeros((f - T.mod(self.y_out.shape[0], f), self.y_out.shape[1], self.y_out.shape[2]),
                                     'float32')], axis=0)
            if sample_target:
              self.y_out = theano_util.downsample(self.y_out, axis=0, factor=f, method=method)
          else:
            if padding:
              self.y_out = T.concatenate(
                [self.y_out, T.zeros((f - T.mod(self.y_out.shape[0], f), self.y_out.shape[1]), 'int32')], axis=0)
            if sample_target:
              self.y_out = theano_util.downsample(self.y_out, axis=0, factor=f, method='max')
      else:
        z = theano_util.downsample(z, axis=a, factor=f, method=method)
        if a < self.y_out.ndim:
          self.y_out = theano_util.downsample(self.y_out, axis=a, factor=f, method='max')
      if a == 0:
        self.index = self.sources[0].index
        if padding:
          self.index = T.concatenate([self.index, T.zeros((f-T.mod(self.index.shape[0], f), self.index.shape[1]), 'int8')], axis=0)
          if fit_target:
            self.index_out = self.index
        self.index = theano_util.downsample(self.index, axis=0, factor=f, method="min")
        if sample_target:
          self.index_out = theano_util.downsample(self.index_out, axis=0, factor=f, method="min")
        elif not fit_target:
          self.index_out = self.index if base is None else base[0].index_out
      elif a == 2:
        n_out = int(n_out / f)
    output = z
    if method == 'concat':
      n_out *= numpy.prod(factor)
    elif method == 'mlp':
      self.DP = self.add_param(self.create_forward_weights(n_out * numpy.prod(factor),z_dim,self.name + "_DP"))
      self.b = self.add_param(self.create_bias(z_dim))
      output = T.nnet.relu(T.dot(output,self.DP) + self.b)
    elif method == 'lstm':
      num_batches = z.shape[2]
      #z = theano.printing.Print("a", attrs=['shape'])(z)
      z = z.dimshuffle(1,0,2,3).reshape((z.shape[1],z.shape[0]*z.shape[2],z.shape[3]))
      #z = theano.printing.Print("b", attrs=['shape'])(z)
      from math import sqrt
      from returnn.theano.activation_functions import elu
      l = sqrt(6.) / sqrt(6 * n_out)
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n_out, n_out)), dtype=theano.config.floatX)
      self.A_in = self.add_param(self.shared(value=values, borrow=True, name = "A_in_" + self.name))
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n_out, n_out)), dtype=theano.config.floatX)
      self.A_re = self.add_param(self.shared(value=values, borrow=True, name = "A_re_" + self.name))
      def lstmk(z_t, y_p, c_p):
        z_t += T.dot(y_p, self.A_re)
        partition = z_t.shape[1] / 4
        ingate = T.nnet.sigmoid(z_t[:,:partition])
        forgetgate = T.nnet.sigmoid(z_t[:,partition:2*partition])
        outgate = T.nnet.sigmoid(z_t[:,2*partition:3*partition])
        input = T.tanh(z_t[:,3*partition:4*partition])
        c_t = forgetgate * c_p + ingate * input
        y_t = outgate * T.tanh(c_t)
        return (y_t, c_t)
      def attent(xt, yp, W_re):
        return T.tanh(xt + elu(T.dot(yp, W_re)))
        #return T.tanh(T.dot(xt, W_in) + T.dot(yp, W_re))
      ####z, _ = theano.scan(attent, sequences = T.dot(z,self.A_in), outputs_info = [T.zeros_like(z[0])], non_sequences=[self.A_re])
      result, _ = theano.scan(lstmk, sequences = T.dot(z,self.A_in), outputs_info = [T.zeros_like(z[0]),T.zeros_like(z[0])])
      z = result[0]
      #from OpLSTM import LSTMOpInstance
      #inp = T.alloc(numpy.cast[theano.config.floatX](0), z.shape[0], z.shape[1], z.shape[2] * 4) + T.dot(z,self.A_in)
      #sta = T.alloc(numpy.cast[theano.config.floatX](0), z.shape[1], z.shape[2])
      #idx = T.alloc(numpy.cast[theano.config.floatX](1), z.shape[0], z.shape[1])
      #result = LSTMOpInstance(inp, self.A_re, sta, idx)
      #result = LSTMOpInstance(T.dot(z,self.A_in), self.A_re, T.zeros_like(z[0]), T.ones_like(z[:,:,0]))
      output = z[-1].reshape((z.shape[1] / num_batches, num_batches, z.shape[2]))
      #output = result[0][0].reshape((z.shape[1] / num_batches, num_batches, z.shape[2]))
    elif method == 'batch':
      self.index = theano_util.downsample(self.sources[0].index, axis=0, factor=factor[0], method="batch")
      #z = theano.printing.Print("d", attrs=['shape'])(z)
    self.set_attr('n_out', n_out)
    self.make_output(output)

    if fit_target:
      self.output = print_to_file('o.out', self.output, shape=True)
      self.index_out = print_to_file('o.idx', self.index_out, shape=True)
      self.y_out = print_to_file('o.y', self.y_out, shape=True)

class UpsampleLayer(_NoOpLayer):
  layer_class = "upsample"

  def __init__(self, factor, axis, time_like_last_source=False, method="nearest-neighbor", **kwargs):
    super(UpsampleLayer, self).__init__(**kwargs)
    self.set_attr("method", method)
    self.set_attr("time_like_last_source", time_like_last_source)
    if isinstance(axis, (str, unicode)):
      axis = json.loads(axis)
    if isinstance(axis, set): axis = tuple(axis)
    assert isinstance(axis, int) or isinstance(axis, (tuple, list)), "int or list[int] expected for axis"
    if isinstance(axis, int): axis = [axis]
    axis = list(sorted(axis))
    self.set_attr("axis", axis)
    if isinstance(factor, (str, unicode)):
      factor = json.loads(factor)
    assert isinstance(factor, (int, float)) or isinstance(axis, (tuple, list)), "int|float or list[int|float] expected for factor"
    if isinstance(factor, (int, float)): factor = [factor] * len(axis)
    assert len(factor) == len(axis)
    self.set_attr("factor", factor)
    sources = self.sources
    assert len(sources) > 0
    if time_like_last_source:
      assert len(sources) >= 2
      source_for_time = sources[-1]
      sources = sources[:-1]
    else:
      source_for_time = None
    z, z_dim = concat_sources(sources, unsparse=False)
    n_out = z_dim
    for f, a in zip(factor, axis):
      target_axis_len = None
      if a == 0:
        assert source_for_time, "not implemented yet otherwise. but this makes most sense anyway."
        self.index = source_for_time.index
        target_axis_len = self.index.shape[0]
      elif a == 2:
        n_out = int(n_out * f)
      z = theano_util.upsample(z, axis=a, factor=f, method=method, target_axis_len=target_axis_len)
    self.set_attr('n_out', n_out)
    self.make_output(z)


class RepetitionLayer(_NoOpLayer):
  layer_class = "rep"

  def __init__(self, factor, **kwargs):
    super(RepetitionLayer, self).__init__(**kwargs)
    factor = numpy.int32(factor)
    self.set_attr("factor", factor)
    inp, n_out = _concat_sources(self.sources, masks=self.masks, mass=self.mass)
    self.set_attr('n_out', n_out)
    time, batch, dim = inp.shape[0], inp.shape[1], inp.shape[2]

    self.index = self.index.dimshuffle(0,'x',1).repeat(factor,axis=1).reshape((time * factor, batch))
    self.output = inp.dimshuffle(0,'x',1,2).repeat(factor,axis=1).reshape((time * factor,batch,dim))


class FrameConcatZeroLayer(_NoOpLayer): # TODO: This is not correct for max_seqs > 1
  """
  Concats zero at the start (left=True) or end in the time-dimension.
  I.e. you can e.g. delay the input by N frames.
  See also FrameConcatZeroLayer (frame_cutoff).
  """
  layer_class = "frame_concat_zero"

  def __init__(self, num_frames, left=True, **kwargs):
    super(FrameConcatZeroLayer, self).__init__(**kwargs)
    self.set_attr("num_frames", num_frames)
    self.set_attr("left", left)
    assert len(self.sources) == 1
    s = self.sources[0]
    for attr in ["n_out", "sparse"]:
      self.set_attr(attr, s.attrs[attr])
    inp = s.output
    # We get (time,batch,dim) input shape.
    time_shape = [inp.shape[i] for i in range(1, inp.ndim)]
    zeros_shape = [num_frames] + time_shape
    zeros = T.zeros(zeros_shape, dtype=inp.dtype)
    if left:
      self.output = T.concatenate([zeros, inp], axis=0)
      self.index = T.concatenate([T.repeat(s.index[:1], num_frames, axis=0), s.index], axis=0)
    else:
      self.output = T.concatenate([inp, zeros], axis=0)
      self.index = T.concatenate([s.index, T.repeat(s.index[-1:], num_frames, axis=0)], axis=0)


class FrameCutoffLayer(_NoOpLayer): # TODO: This is not correct for max_seqs > 1
  """
  Cutoffs frames at the start (left=True) or end in the time-dimension.
  You should use this when you used FrameConcatZeroLayer(frame_concat_zero).
  """
  layer_class = "frame_cutoff"

  def __init__(self, num_frames, left=True, **kwargs):
    super(FrameCutoffLayer, self).__init__(**kwargs)
    self.set_attr("num_frames", num_frames)
    self.set_attr("left", left)
    x_in, n_in = _concat_sources(self.sources, masks=self.masks, mass=self.mass)
    i_in = self.sources[0].index
    self.set_attr("n_out", n_in)
    if left:
      self.output = x_in[num_frames:]
      self.index = i_in[num_frames:]
    else:
      self.output = x_in[:-num_frames]
      self.index = i_in[:-num_frames]


class ReverseLayer(_NoOpLayer):
  """
  Reverses the time-dimension.
  """
  layer_class = "reverse"

  def __init__(self, **kwargs):
    super(ReverseLayer, self).__init__(**kwargs)
    assert len(self.sources) == 1
    s = self.sources[0]
    for attr in ["n_out", "sparse"]:
      self.set_attr(attr, s.attrs[attr])
    # We get (time,batch,dim) input shape.
    self.index = s.index[::-1]
    self.output = s.output[::-1]


class CalcStepLayer(_NoOpLayer):
  layer_class = "calc_step"

  def __init__(self, n_out=None, from_prev="", apply=False, step=None, initial="zero", **kwargs):
    super(CalcStepLayer, self).__init__(**kwargs)
    if n_out is not None:
      self.set_attr("n_out", n_out)
    if from_prev:
      self.set_attr("from_prev", from_prev.encode("utf8"))
    self.set_attr("apply", apply)
    if step is not None:
      self.set_attr("step", step)
    self.set_attr("initial", initial.encode("utf8"))
    if not apply:
      assert n_out is not None
      assert self.network
      if self.network.calc_step_base:
        prev_layer = self.network.calc_step_base.get_layer(from_prev)
        if not prev_layer:
          self.network.calc_step_base.print_network_info("Prev-Calc-Step network")
          raise Exception("%s not found in prev calc step network" % from_prev)
        assert n_out == prev_layer.attrs["n_out"]
        self.output = prev_layer.output
      else:
        # First calc step. Just use zero.
        shape = [self.index.shape[0], self.index.shape[1], n_out]
        if initial == "zero":
          self.output = T.zeros(shape, dtype="float32")
        elif initial == "param":
          values = numpy.asarray(self.rng.normal(loc=0.0, scale=numpy.sqrt(12. / n_out), size=(n_out,)), dtype="float32")
          initial_param = self.add_param(self.shared(value=values, borrow=True, name="output_initial"))
          self.output = initial_param.dimshuffle('x', 'x', 0)
        else:
          raise Exception("CalcStepLayer: initial %s invalid" % initial)
    else:
      assert step is not None
      assert len(self.sources) == 1
      assert not from_prev
      # We will refer to the previous calc-step layer this way
      # so that we ensure that we have already traversed it.
      # This is important so that share_params correctly works.
      from_prev = self.sources[0].name
      assert self.network
      subnetwork = self.network.get_calc_step(step)
      prev_layer = subnetwork.get_layer(from_prev)
      assert prev_layer, "%s not found in subnetwork" % from_prev
      if n_out is not None:
        assert n_out == prev_layer.attrs["n_out"]
      self.set_attr("n_out", prev_layer.attrs["n_out"])
      self.output = prev_layer.output


class SubnetworkLayer(_NoOpLayer):
  layer_class = "subnetwork"
  recurrent = True  # we don't know. depends on the subnetwork.

  def __init__(self, n_out, subnetwork, load="<random>", data_map=None, trainable=True,
               concat_sources=True,
               **kwargs):
    """
    :param int n_out: output dimension of output layer
    :param dict[str,dict] network: subnetwork as dict (JSON content)
    :param list[str] data_map: maps the sources (from) of the layer to data input.
      the list should be as long as the sources.
      default is ["data"], i.e. it expects one source and maps it as data in the subnetwork.
    :param bool concat_sources: if we concatenate all sources into one, like it is standard for most other layers
    :param str load: load string. filename but can have placeholders via str.format. Or "<random>" for no load.
    :param bool trainable: if we take over all params from the subnetwork
    """
    super(SubnetworkLayer, self).__init__(**kwargs)
    self.set_attr("n_out", n_out)
    if isinstance(subnetwork, str):
      subnetwork = json.loads(subnetwork)
    self.set_attr("subnetwork", subnetwork)
    self.set_attr("load", load)
    if isinstance(data_map, str):
      data_map = json.loads(data_map)
    if data_map:
      self.set_attr("data_map", data_map)
    self.set_attr('concat_sources', concat_sources)
    self.set_attr("trainable", trainable)
    self.trainable = trainable
    if concat_sources:
      assert not data_map, "We expect the implicit canonical data_map with concat_sources."
      assert self.sources
      data, n_in = _concat_sources(self.sources, masks=self.masks, mass=self.mass)
      s0 = self.sources[0]
      sub_n_out = {"data": [n_in, 1 if s0.attrs['sparse'] else 2],
                   "classes": [n_out, 1 if self.attrs['sparse'] else 2]}
      data_map_d = {"data": data}
      data_map_di = {"data": s0.index, "classes": self.index}
      data_map = []
    else:  # not concat_sources
      if not data_map:
        data_map = ["data"]
      assert isinstance(data_map, list)
      assert len(data_map) == len(self.sources)
      sub_n_out = {"classes": [n_out, 1 if self.attrs['sparse'] else 2]}
      data_map_d = {}
      data_map_di = {"classes": self.index}
      for k, s in zip(data_map, self.sources):
        sub_n_out[k] = [s.attrs["n_out"], s.output.ndim - 1]
        data_map_d[k] = s.output
        data_map_di[k] = s.index
    print("New subnetwork", self.name, "with data", {k: s.name for (k, s) in zip(data_map, self.sources)}, sub_n_out, file=log.v2)
    self.subnetwork = self.network.new_subnetwork(
      json_content=subnetwork, n_out=sub_n_out, data_map=data_map_d, data_map_i=data_map_di)
    self.subnetwork.print_network_info(name="layer %r subnetwork" % self.name)
    assert self.subnetwork.output["output"].attrs['n_out'] == n_out
    if trainable:
      self.params.update(self.subnetwork.get_params_shared_flat_dict())
    if load == "<random>":
      print("subnetwork with random initialization", file=log.v2)
    else:
      from returnn.config import get_global_config
      config = get_global_config()  # this is a bit hacky but works fine in all my cases...
      model_filename = load % {"self": self,
                               "global_config_load": config.value("load", None),
                               "global_config_epoch": config.value("epoch", 0)}
      print("loading subnetwork weights from", model_filename, file=log.v2)
      import h5py
      model_hdf = h5py.File(model_filename, "r")
      self.subnetwork.load_hdf(model_hdf)
      print("done loading subnetwork weights for", self.name, file=log.v2)
    self.output = self.subnetwork.output["output"].output

  def cost(self):
    if not self.trainable:
      return super(SubnetworkLayer, self).cost()
    try:
      const_cost = T.get_scalar_constant_value(self.subnetwork.total_cost)
      if const_cost == 0:
        return None, None
    except T.NotScalarConstantError:
      pass
    return self.subnetwork.total_cost, self.subnetwork.known_grads

  def make_constraints(self):
    if not self.trainable:
      return super(SubnetworkLayer, self).make_constraints()
    return self.subnetwork.total_constraints


class ClusterDependentSubnetworkLayer(_NoOpLayer):
  layer_class = "clustersubnet"
  recurrent = True  # we don't know. depends on the subnetwork.

  def __init__(self, n_out, subnetwork, n_clusters, load="<random>", data_map=None, trainable=True,
               concat_sources=True,
               **kwargs):
    """
    :param int n_out: output dimension of output layer
    :param dict[str,dict] network: subnetwork as dict (JSON content)
    :param list[str] data_map: maps the sources (from) of the layer to data input.
      the list should be as long as the sources.
      default is ["data"], i.e. it expects one source and maps it as data in the subnetwork.
    :param str load: load string. filename but can have placeholders via str.format. Or "<random>" for no load.
    :param bool trainable: if we take over all params from the subnetwork
    """
    super(ClusterDependentSubnetworkLayer, self).__init__(**kwargs)
    self.set_attr("n_out", n_out)
    if isinstance(subnetwork, str):
      subnetwork = json.loads(subnetwork)
    self.set_attr("subnetwork", subnetwork)
    self.set_attr("load", load)
    if isinstance(data_map, str):
      data_map = json.loads(data_map)
    if data_map:
      self.set_attr("data_map", data_map)
    self.set_attr('concat_sources', concat_sources)
    self.set_attr("trainable", trainable)
    self.trainable = trainable
    self.set_attr("n_clusters", n_clusters)
    self.n_clusters = n_clusters
    print("ClusterDependentSubnetworkLayer: have %s clusters" % self.n_clusters, file=log.v2)
    assert len(self.sources) >= 2, "need input, ..., cluster_map"
    sources, cluster_map_source = self.sources[:-1], self.sources[-1]
    if concat_sources:
      assert not data_map, "We expect the implicit canonical data_map with concat_sources."
      assert self.sources
      data, n_in = _concat_sources(sources, masks=self.masks[:-1], mass=self.mass)
      s0 = sources[0]
      sub_n_out = {"data": [n_in, 1 if s0.attrs['sparse'] else 2],
                   "classes": [n_out, 1 if self.attrs['sparse'] else 2]}
      data_map_d = {"data": data}
      data_map_di = {"data": s0.index, "classes": self.index}
      data_map = []
    else:  # not concat_sources
      if not data_map:
        data_map = ["data"]
      assert isinstance(data_map, list)
      assert len(data_map) == len(sources)
      sub_n_out = {"classes": [n_out, 1 if self.attrs['sparse'] else 2]}
      data_map_d = {}
      data_map_di = {"classes": self.index}
      for k, s in zip(data_map, sources):
        sub_n_out[k] = [s.attrs["n_out"], s.output.ndim - 1]
        data_map_d[k] = s.output
        data_map_di[k] = s.index
    self.subnetworks = []
    for idx in range(0, self.n_clusters):
      print("New subnetwork", self.name, "with data", {k: s.name for (k, s) in zip(data_map, sources)}, sub_n_out, file=log.v2)
      self.subnetworks.append(self.network.new_subnetwork(
        json_content=subnetwork, n_out=sub_n_out, data_map=data_map_d, data_map_i=data_map_di))
      assert self.subnetworks[idx].output["output"].attrs['n_out'] == n_out
      if trainable:
        self.params.update(self.subnetworks[idx].get_params_shared_flat_dict())
      if load == "<random>":
        print("subnetwork with random initialization", file=log.v2)
      else:
        from returnn.config import get_global_config
        config = get_global_config()  # this is a bit hacky but works fine in all my cases...
        model_filename = load % {"self": self,
                                 "global_config_load": config.value("load", None),
                                 "global_config_epoch": config.int("epoch", 0)}
        print("loading subnetwork weights from", model_filename, file=log.v2)
        import h5py
        model_hdf = h5py.File(model_filename, "r")
        self.subnetworks[idx].load_hdf(model_hdf)
        print("done loading subnetwork weights for", self.name, file=log.v2)
    self.ref = cluster_map_source.output[0]

    ## generate output lists and sums with ifelse to only compute specified paths

    # output
    self.zero_output = T.zeros_like(self.subnetworks[0].output["output"].output)
    self.y = [ifelse(T.prod(T.neq(idx, self.ref)), self.zero_output, self.subnetworks[idx].output["output"].output) for idx in range(0, self.n_clusters)]
    self.z = self.y[0]
    for idx in range(1, self.n_clusters):
      self.z += self.y[idx]
    self.output = self.z

    # costs
    self.costs = [ifelse(T.prod(T.neq(idx, self.ref)), T.constant(0), self.subnetworks[idx].total_cost) for idx in
                  range(0, self.n_clusters)]
    self.total_cost = T.sum([self.costs[idx] for idx in range(0, self.n_clusters)])

    # grads
    # TODO for each TheanoVar in dict do the ifelse thing
    self.output_grads = {}
    if not self.subnetworks[0].known_grads:
      print("known grads is empty", file=log.v5)
    else:
      raise NotImplementedError

    # constraints
    self.constraints = [ifelse(T.prod(T.neq(idx, self.ref)), T.constant(0), self.subnetworks[idx].total_constraints) for idx in
                        range(0, self.n_clusters)]
    self.total_constraints = T.sum([self.costs[idx] for idx in range(0, self.n_clusters)])

  def cost(self):
    if not self.trainable:
      return super(SubnetworkLayer, self).cost()
    try:
      const_cost = T.get_scalar_constant_value(self.total_cost)
      if const_cost == 0:
        return None, None
    except T.NotScalarConstantError:
      pass
    return self.total_cost, self.output_grads

  def make_constraints(self):
    if not self.trainable:
      return super(SubnetworkLayer, self).make_constraints()
    return self.total_constraints

  def update_cluster_target(self, seq_tag):
    self.ref.set_value(self.cluster_dict(seq_tag))


class IndexToVecLayer(_NoOpLayer):
  # IndexToVec convert a running index to a vektor like onehot
  # source: [time][batch][1]
  # out: [time][batch][n_out]
  layer_class = "idx_to_vec"

  def __init__(self, n_out, **kwargs):
    super(IndexToVecLayer, self).__init__(**kwargs)
    self.set_attr('n_out', n_out)

    z = T.cast(theano_util.class_idx_seq_to_1_of_k(self.sources[0].output, n_out), dtype="float32")
    self.output = z  # (time, batch, n_out)


class InterpolationLayer(_NoOpLayer):
  # InterpolationLayer interpolates between several layers given an interpolation vector
  # source: (n-1) sources[n_out] 1 source[n-1]
  # out: [time][batch][n_out]
  layer_class = "interp"

  def __init__(self, n_out, **kwargs):
    super(InterpolationLayer, self).__init__(**kwargs)
    self.set_attr('n_out', n_out)

    dict_s = []
    for s, m in zip(self.sources[:-1], self.masks[:-1]):
      assert s.attrs['n_out'] == n_out
      if m is None:
        s_data = s.output
      else:
        s_data = self.mass * m * s.output
      s_shuffled = s_data.dimshuffle(0, 1, 2, 'x')
      dict_s += [s_shuffled]
      Y = T.concatenate(dict_s, axis=3)  # [time][batch][n_out][n-1]

    interp_vec = self.sources[-1].output

    # if only one interpolation vector for the whole time is given, extens vector along time axis
    import theano.ifelse
    x = theano.ifelse.ifelse(T.eq(interp_vec.shape[0],1), T.extra_ops.repeat(interp_vec, Y.shape[0], axis=0), interp_vec)

    i, j, m, k = Y.shape  # time, batch, n_out, interp
    x_ = x.reshape((i * j, k))
    Y_ = Y.reshape((i * j, m, k))
    z_ = T.batched_tensordot(x_, Y_, (1, 2))
    z = z_.reshape((i, j, m))

    self.output = z

class ChunkingSublayer(_NoOpLayer):
  layer_class = "chunking_sublayer"
  recurrent = True  # we don't know

  def __init__(self, n_out, sublayer,
               chunk_size, chunk_step,
               chunk_distribution="uniform",
               add_left_context=0,
               add_right_context=0,
               normalize_output=True,
               trainable=False,
               **kwargs):
    super(ChunkingSublayer, self).__init__(**kwargs)
    self.set_attr('n_out', n_out)
    self.set_attr('chunk_size', chunk_size)
    self.set_attr('chunk_step', chunk_step)
    if isinstance(sublayer, str):
      sublayer = json.loads(sublayer)
    self.set_attr('sublayer', sublayer.copy())
    self.set_attr('chunk_distribution', chunk_distribution)
    self.set_attr('add_left_context', add_left_context)
    self.set_attr('add_right_context', add_right_context)
    self.set_attr('normalize_output', normalize_output)
    self.set_attr('trainable', trainable)
    self.trainable = trainable

    sub_n_out = sublayer.pop("n_out", None)
    if sub_n_out: assert sub_n_out == n_out
    if trainable:
      sublayer["train_flag"] = self.train_flag
      sublayer["mask"] = self.attrs.get("mask", "none")
      sublayer["dropout"] = self.attrs.get("dropout", 0.0)

    assert len(self.sources) == 1
    source = self.sources[0].output
    n_in = self.sources[0].attrs["n_out"]
    index = self.sources[0].index
    assert source.ndim == 3  # not complicated to support others, just not implemented
    t_last_start = T.maximum(source.shape[0] - chunk_size, 1)
    t_range = T.arange(t_last_start, step=chunk_step)

    from returnn.theano.layers.base import SourceLayer
    from returnn.theano.layers.basic import get_layer_class
    def make_sublayer(source, index, name):
      layer_opts = sublayer.copy()
      cl = layer_opts.pop("class")
      layer_class = get_layer_class(cl)
      source_layer = SourceLayer(name="%s_source" % name, n_out=n_in, x_out=source, index=index)
      layer = layer_class(sources=[source_layer], index=index, name=name, n_out=n_out, network=self.network, **layer_opts)
      self.sublayer = layer
      return layer
    self.sublayer = None

    output = T.zeros((source.shape[0], source.shape[1], n_out), dtype=source.dtype)
    output_index_sum = T.zeros([source.shape[0], source.shape[1]], dtype="float32")
    def step(t_start, output, output_index_sum, source, index):
      t_end = T.minimum(t_start + chunk_size, source.shape[0])
      if add_left_context > 0:
        t_start_c = T.maximum(t_start - add_left_context, 0)
      else:
        t_start_c = t_start
      if add_right_context > 0:
        t_end_c = T.minimum(t_end + add_right_context, source.shape[0])
      else:
        t_end_c = t_end
      chunk = source[t_start_c:t_end_c]
      chunk_index = index[t_start_c:t_end_c]
      layer = make_sublayer(source=chunk, index=chunk_index, name="%s_sublayer" % self.name)
      l_output = layer.output
      l_index_f32 = T.cast(layer.index, dtype="float32")
      if add_left_context > 0:
        l_output = l_output[t_start - t_start_c:]
        l_index_f32 = l_index_f32[t_start - t_start_c:]
      if add_right_context > 0:
        l_output = l_output[:l_output.shape[0] + t_end - t_end_c]
        l_index_f32 = l_index_f32[:l_index_f32.shape[0] + t_end - t_end_c]
      if chunk_distribution == "uniform": pass  # just leave it as it is
      elif chunk_distribution == "triangle":
        ts = T.arange(1, t_end - t_start + 1)
        ts_rev = ts[::-1]
        tri = T.cast(T.minimum(ts, ts_rev), dtype="float32").dimshuffle(0, 'x')  # time,batch
        l_index_f32 = l_index_f32 * tri
      elif chunk_distribution == "hamming":  # https://en.wikipedia.org/wiki/Window_function#Hamming_window
        ts = T.arange(0, t_end - t_start)
        alpha = 0.53836
        w = alpha - (1.0 - alpha) * T.cos(2.0 * numpy.pi * ts / (ts.shape[0] - 1))  # always >0
        w_bc = T.cast(w, dtype="float32").dimshuffle(0, 'x')  # time,batch
        l_index_f32 = l_index_f32 * w_bc
      elif chunk_distribution.startswith("gauss("):  # https://en.wikipedia.org/wiki/Window_function#Gaussian_window
        modeend = chunk_distribution.find(")")
        assert modeend >= 0
        sigma = float(chunk_distribution[len("gauss("):modeend])
        ts = T.arange(0, t_end - t_start)
        N = ts.shape[0] - 1
        w = T.exp(-0.5 * ((ts - N / 2.0) / (sigma * N / 2.0)) ** 2)  # always >0
        w_bc = T.cast(w, dtype="float32").dimshuffle(0, 'x')  # time,batch
        l_index_f32 = l_index_f32 * w_bc
      else:
        assert False, "unknown chunk distribution %r" % chunk_distribution
      assert l_index_f32.ndim == 2
      output = T.inc_subtensor(output[t_start:t_end], l_output * l_index_f32.dimshuffle(0, 1, 'x'))
      output_index_sum = T.inc_subtensor(output_index_sum[t_start:t_end], l_index_f32)
      return [output, output_index_sum]

    (output, output_index_sum), _ = theano.reduce(
      step, sequences=[t_range],
      non_sequences=[source, index],
      outputs_info=[output, output_index_sum])
    self.scan_output = output
    self.scan_output_index_sum = output_index_sum
    self.index = T.gt(output_index_sum, 0)
    assert output.ndim == 3
    if normalize_output:
      output_index_sum = T.maximum(output_index_sum, numpy.float32(1.0))
      assert output_index_sum.ndim == 2
      output = output / output_index_sum.dimshuffle(0, 1, 'x')  # renormalize
    self.make_output(output)
    assert self.sublayer
    if trainable:
      self.params.update({"sublayer." + name: param for (name, param) in self.sublayer.params.items()})

  def cost(self):
    if not self.trainable:
      return super(ChunkingSublayer, self).cost()
    cost, known_grads = self.sublayer.cost()
    if cost is None:
      return None, None
    return cost * self.sublayer.cost_scale(), known_grads

  def make_constraints(self):
    if not self.trainable:
      return super(ChunkingSublayer, self).make_constraints()
    return self.sublayer.make_constraints()


class TimeChunkingLayer(_NoOpLayer):
  layer_class = "time_chunking"

  def __init__(self, n_out, chunk_size, chunk_step, **kwargs):
    super(TimeChunkingLayer, self).__init__(**kwargs)
    self.set_attr("n_out", n_out)
    self.set_attr("chunk_size", chunk_size)
    self.set_attr("chunk_step", chunk_step)
    x, n_in = concat_sources(self.sources, masks=self.masks, mass=self.mass, unsparse=True)
    self.source_index = self.index
    from returnn.theano.native_op import chunk
    self.output, self.index = chunk(x, index=self.source_index, chunk_size=chunk_size, chunk_step=chunk_step)


class TimeUnChunkingLayer(_NoOpLayer):
  layer_class = "time_unchunking"

  def __init__(self, n_out, chunking_layer, **kwargs):
    super(TimeUnChunkingLayer, self).__init__(**kwargs)
    self.set_attr("n_out", n_out)
    self.set_attr("chunking_layer", chunking_layer)
    x, n_in = concat_sources(self.sources, masks=self.masks, mass=self.mass, unsparse=True)
    self.source_index = self.index
    chunking_layer_o = self.network.get_layer(chunking_layer)
    assert isinstance(chunking_layer_o, TimeChunkingLayer)
    chunk_size = chunking_layer_o.attrs["chunk_size"]
    chunk_step = chunking_layer_o.attrs["chunk_step"]
    n_time = chunking_layer_o.source_index.shape[0]
    n_batch = chunking_layer_o.source_index.shape[1]
    from returnn.theano.native_op import unchunk
    self.output, self.index, _ = unchunk(
      x, index=chunking_layer_o.index, chunk_size=chunk_size, chunk_step=chunk_step, n_time=n_time, n_batch=n_batch)


class TimeFlatLayer(_NoOpLayer):
  layer_class = "time_flat"

  def __init__(self, chunk_size, chunk_step, **kwargs):
    super(TimeFlatLayer, self).__init__(**kwargs)
    self.set_attr("chunk_size", chunk_size)
    self.set_attr("chunk_step", chunk_step)
    x, n_in = concat_sources(self.sources, masks=self.masks, mass=self.mass, unsparse=True)
    self.set_attr("n_out", n_in)
    self.source_index = self.index
    n_time = self.index.shape[0] * chunk_size
    n_batch = self.index.shape[1]
    from returnn.theano.native_op import unchunk
    self.output, self.index, _ = unchunk(
      x, index=self.index, chunk_size=chunk_size, chunk_step=chunk_step, n_time=n_time, n_batch=n_batch)


class RBFLayer(_NoOpLayer):
  """
  Use radial basis function.
  """
  layer_class = "rbf"

  def __init__(self, n_out, **kwargs):
    super(RBFLayer, self).__init__(**kwargs)
    self.set_attr("n_out", n_out)
    x, n_in = concat_sources(self.sources, masks=self.masks, mass=self.mass, unsparse=True)
    self.W_in = self.add_param(self.create_forward_weights(n_in, n_out, name="W_in_%s" % self.name))
    self.b = self.add_param(self.create_bias(n_out, name="w_%s" % self.name))

    # Note: The naive squared distance (x - W)**2 would take too much memory (time*batch*n_in*n_out).
    # d = self.W_in.dimshuffle('x', 'x', 0, 1) - x.dimshuffle(0, 1, 2, 'x')  # time,batch,n_in,n_out
    # ds = T.sum(d, axis=2)
    # Thus, we need to avoid that.
    # Another form of the same is sum(x**2 + W**2 - 2*W*x, axis=<n_in>).
    x_sqr = T.sum(T.sqr(x), axis=2)  # time,batch
    W_sqr = T.sum(T.sqr(self.W_in), axis=0)  # n_out
    xW = T.dot(x, self.W_in)  # time,batch,n_out
    ds = x_sqr.dimshuffle(0, 1, 'x') + W_sqr.dimshuffle('x', 'x', 0) - numpy.float32(2) * xW  # time,batch,n_out
    w = T.nnet.sigmoid(self.b).dimshuffle('x', 'x', 0)  # time,batch,n_out
    self.output = T.exp(- w * ds)


class LinearCombLayer(_NoOpLayer):
  """
  Linear combination of each `n_comb` elements with bias.
  """
  layer_class = "linear_comb"

  def __init__(self, n_out, n_comb, activation=None, **kwargs):
    super(LinearCombLayer, self).__init__(**kwargs)
    self.set_attr("n_out", n_out)
    self.set_attr("n_comb", n_comb)
    if activation:
      self.set_attr("activation", activation)
    x, n_in = concat_sources(self.sources, masks=self.masks, mass=self.mass, unsparse=True)
    assert n_in % n_comb == 0
    assert n_out == n_in / n_comb
    self.W = self.add_param(self.create_forward_weights(n_out, n_comb, name="W_in_%s" % self.name))
    self.b = self.add_param(self.create_bias(n_out, name="b_%s" % self.name))
    assert x.ndim == 3
    x_c = x.reshape((x.shape[0], x.shape[1], n_out, n_comb))
    z = T.sum(self.W.dimshuffle('x', 'x', 0, 1) * x_c, axis=3) + self.b.dimshuffle('x', 'x', 0)
    if activation:
      act_f = strtoact_single_joined(activation)
      self.output = act_f(z)
    else:
      self.output = z


class PolynomialExpansionLayer(_NoOpLayer):
  layer_class = "polynomial_expansion"

  def __init__(self, n_degree, n_out=None, **kwargs):
    super(PolynomialExpansionLayer, self).__init__(**kwargs)
    x, n_in = concat_sources(self.sources, masks=self.masks, mass=self.mass, unsparse=True)
    if not n_out: n_out = n_degree * n_in
    self.set_attr("n_out", n_out)
    self.set_attr("n_degree", n_degree)
    assert n_out == n_in * n_degree
    static_rng = numpy.random.RandomState(1234)
    def make_permut():
      return T.constant(static_rng.permutation(n_in))
    xs = [x]
    for i in range(2, n_degree + 1):
      xl = xs[-1][:, :, make_permut()]
      xs += [xl * x]
    assert len(xs) == n_degree
    z = T.concatenate(xs, axis=2)
    self.output = z


class RandomSelectionLayer(_NoOpLayer):
  layer_class = "random_selection"

  def __init__(self, n_out, **kwargs):
    super(RandomSelectionLayer, self).__init__(**kwargs)
    self.set_attr("n_out", n_out)
    x, n_in = concat_sources(self.sources, masks=self.masks, mass=self.mass, unsparse=True)
    assert n_in >= n_out
    static_rng = numpy.random.RandomState(1234)
    P = T.constant(static_rng.permutation(n_in)[:n_out])
    self.output = x[:, :, P]


class TimeBlurLayer(_NoOpLayer):
  layer_class = "time_blur"
  recurrent = True  # Force no frame shuffling or so.

  def __init__(self, t_start, t_end, t_step, distribution, **kwargs):
    super(TimeBlurLayer, self).__init__(**kwargs)
    z, n_in = concat_sources(self.sources)
    n_out = n_in
    self.set_attr('n_out', n_out)
    self.set_attr('t_start', t_start)
    self.set_attr('t_end', t_end)
    self.set_attr('t_step', t_step)
    self.set_attr('distribution', distribution)

    t_offsets = numpy.arange(start=t_start, stop=t_end, step=t_step)
    nums = numpy.arange(t_offsets.shape[0])
    if distribution == "uniform":
      t_weights = numpy.ones_like(nums, dtype="float32")
    elif distribution == "triangle":
      nums_rev = nums[::-1]
      t_weights = 1 + numpy.minimum(nums, nums_rev)
    elif distribution == "hamming":  # https://en.wikipedia.org/wiki/Window_function#Hamming_window
      alpha = 0.53836
      t_weights = alpha - (1.0 - alpha) * numpy.cos(2.0 * numpy.pi * nums / (nums.shape[0] - 1))  # always >0
    elif distribution.startswith("gauss("):  # https://en.wikipedia.org/wiki/Window_function#Gaussian_window
      modeend = distribution.find(")")
      assert modeend >= 0
      sigma = float(distribution[len("gauss("):modeend])
      N = nums.shape[0] - 1
      t_weights = numpy.exp(-0.5 * ((nums - N / 2.0) / (sigma * N / 2.0)) ** 2)  # always >0
    else:
      assert False, "unknown distribution %r" % distribution

    findex = T.cast(self.index, dtype="float32")  # to be able to use on GPU
    self.output = T.zeros_like(z)
    weight_sum = T.zeros_like(findex)  # time,batch
    assert len(t_weights) == len(t_offsets)
    for t_offset, w in zip(t_offsets, t_weights):
      if t_offset < 0:
        z_slice = slice(-t_offset, None)
        o_slice = slice(0, t_offset)
      elif t_offset == 0:
        z_slice = slice(None, None)
        o_slice = slice(None, None)
      else:  # t_offset > 0
        z_slice = slice(0, -t_offset)
        o_slice = slice(t_offset, None)
      w = findex[z_slice] * numpy.float32(w)  # time,batch
      self.output = T.inc_subtensor(self.output[o_slice], z[z_slice] * w.dimshuffle(0, 1, 'x'))
      weight_sum = T.inc_subtensor(weight_sum[o_slice], w)
    self.output = self.output / T.maximum(weight_sum.dimshuffle(0, 1, 'x'), numpy.float32(0.0001))

class MfccLayer(_NoOpLayer):
  """
  The source layer of this layer should be the DftLayer
  """
  layer_class = "mfcc_layer"
  recurrent = True #Event though the layer is not recurrent the implementation does not work with "False" -> reason unclear

  def __init__(self, dftSize=512, samplingFrequency=16*1e3, fl=0, fh=None, nrOfFilters=40, nrOfMfccCoefficients=None, **kwargs):
    super(MfccLayer, self).__init__(**kwargs)
    self.set_attr('target', 'classes')
    if nrOfMfccCoefficients==None:
      nrOfMfccCoefficients = nrOfFilters
    if fh==None:
      fh = samplingFrequency/2.0

    #create MFCC filter matrix
    filterMatrix = self.getMfccFilterMatrix(samplingFrequency, fl, fh, dftSize, nrOfFilters, 0)
    #apply MFCC filter
    filtered = T.log(T.dot(self.sources[0].output ** 2, filterMatrix))
    #convert to cepstral domain
    numpy.reshape(numpy.asarray(range(10)), (10, 1)) * numpy.reshape(numpy.asarray(range(10))+0.5, (1, 10))
    dctIndMatrix = numpy.reshape(numpy.asarray(range(nrOfFilters)), (nrOfFilters, 1)) * numpy.reshape((numpy.asarray(range(nrOfFilters)) + 1), (1, nrOfFilters))
    dctMatrix = T.cos(numpy.pi/nrOfFilters * dctIndMatrix)
    mfccs = T.dot(filtered, dctMatrix)
    self.set_attr('n_out', nrOfMfccCoefficients)
    self.make_output(mfccs[:,:,0:nrOfMfccCoefficients])

  def batch_norm(self, h, dim, use_shift=False, use_std=False, use_sample=0.0, force_sample=True, index=None, **kwargs):
    """
    overwrite function from Layer to  change default parameters of batch_norm: use_shift, use_std and foce_sample
    """
    return super(MfccLayer, self).batch_norm(h=h, dim=dim, use_shift=use_shift, use_std=use_std, use_sample=use_sample, force_sample=force_sample, index=index)

  def melScale(self, freq):
    """
    returns the respective value on the mel scale

    :type freq: float
    :param freq: frequency value to transform onto mel scale
    :rtype: float
    """
    return 1125.0 * numpy.log(1 + float(freq)/700)

  def invMelScale(self, melVal):
    """
    returns the respective value in the frequency domain

    :type melVal: float
    :param melVal: value in mel domain
    :rtype: float
    """
    return 700.0 * (numpy.exp(float(melVal) / 1125) - 1)

  def getMfccFilterMatrix(self, samplingFrequency, fl, fh, dftSize, nrOfFilters, flag_areaNormalized=0):
    """
    returns the filter bank matrix used for the MFCCs
    For mathematical details see the book "speech language processing" by Huang et. al. pp. 314

    #TBD !!!
    :type dftSize: int
    :param dftSize: size of dft
    :type nrOfFilters: int
    :param nrOfFilters: the number of filters used for the filterbank
    :type flag_areaNormalized: int
    :param flag_areaNormalized: flag that specifies which filter bank will be returned
                0 - not normalized filter bank
                1 - normalized filter bank where each filter covers an area of 1
    """
    boundaryPoints=[numpy.round(dftSize/float(samplingFrequency) * self.invMelScale(self.melScale(fl) + m * (self.melScale(fh) - self.melScale(fl))/(float(nrOfFilters) + 1))) for m in range(nrOfFilters+2)]
    filterMatrixNumerator = numpy.zeros((int(numpy.floor(dftSize/2.0)+1), nrOfFilters))
    filterMatrixDenominator = numpy.ones((int(numpy.floor(dftSize/2.0)+1), nrOfFilters))
    if flag_areaNormalized==0:
      for i1 in range(nrOfFilters):
        m = i1 + 1
        #- rising flank of filter
        filterMatrixNumerator[int(numpy.ceil(boundaryPoints[m-1])):int(numpy.floor(boundaryPoints[m])),i1] = 2 * (numpy.asarray(range(int(numpy.ceil(boundaryPoints[m-1])), int(numpy.floor(boundaryPoints[m])))) - boundaryPoints[m-1])
        filterMatrixDenominator[int(numpy.ceil(boundaryPoints[m-1])):int(numpy.floor(boundaryPoints[m])),i1] = (boundaryPoints[m+1] - boundaryPoints[m-1]) * (boundaryPoints[m] - boundaryPoints[m-1])
        #- falling flank of filter
        filterMatrixNumerator[int(numpy.floor(boundaryPoints[m])):int(numpy.floor(boundaryPoints[m+1])),i1] = 2 * (boundaryPoints[m+1] - numpy.asarray(range(int(numpy.floor(boundaryPoints[m])), int(numpy.floor(boundaryPoints[m+1])))))
        filterMatrixDenominator[int(numpy.floor(boundaryPoints[m])):int(numpy.floor(boundaryPoints[m+1])),i1] = (boundaryPoints[m+1] - boundaryPoints[m-1]) * (boundaryPoints[m+1] - boundaryPoints[m])

      filterMatrix = numpy.divide(filterMatrixNumerator, filterMatrixDenominator)
      return filterMatrix
    else:
      for i1 in range(nrOfFilters):
        m = i1 + 1
        #- rising flank of filter
        filterMatrixNumerator[int(numpy.ceil(boundaryPoints[m-1])):int(numpy.floor(boundaryPoints[m])),i1] = (numpy.asarray(range(int(numpy.ceil(boundaryPoints[m-1])), int(numpy.floor(boundaryPoints[m])))) - boundaryPoints[m-1])
        filterMatrixDenominator[int(numpy.ceil(boundaryPoints[m-1])):int(numpy.floor(boundaryPoints[m])),i1] = (boundaryPoints[m] - boundaryPoints[m-1])
        #- falling flank of filter
        filterMatrixNumerator[int(numpy.floor(boundaryPoints[m])):int(numpy.floor(boundaryPoints[m+1])),i1] =(boundaryPoints[m+1] - numpy.asarray(range(int(numpy.floor(boundaryPoints[m])), int(numpy.floor(boundaryPoints[m+1])))))
        filterMatrixDenominator[int(numpy.floor(boundaryPoints[m])):int(numpy.floor(boundaryPoints[m+1])),i1] =(boundaryPoints[m+1] - boundaryPoints[m])

      filterMatrix = numpy.divide(filterMatrixNumerator, filterMatrixDenominator)
      return filterMatrix

class Preemphasis(_NoOpLayer):
  """
  This layer is expecting a time signal as input and applying the preemphasis to the segment.
  (This is not completely correct application of preemphasis, since the first element of the segment does not
  know its predecessor in the time signal, therefore the effect is different than applying preemphasis on the
  complete signal beforehand)
  """
  layer_class = "preemphasis_layer"
  recurrent = True #Event though the layer is not recurrent the implementation does not work with "False" -> reason unclear

  def __init__(self, alpha=1.0, **kwargs):
    """
    """
    super(Preemphasis, self).__init__(**kwargs)
    self.set_attr('target', 'classes')
    inputVec = self.sources[0].output
    n_in = self.sources[0].attrs["n_out"]
    self.set_attr('n_out', n_in)
    preemphMatrix = numpy.zeros((n_in, n_in))
    numpy.fill_diagonal(preemphMatrix, 1)
    preemphMatrix[numpy.arange(n_in-1)+1, numpy.arange(n_in-1)] = -1 * alpha
    outputVec = T.dot(inputVec, preemphMatrix.transpose())
    self.make_output(outputVec)

class EnergyNormalization(_NoOpLayer):
  """
  This layer expects a (chunkted) time signal at the input. It normalizes the signal energy of the input chunk.
  """
  layer_class = "energy_normalization_layer"
  recurrent = True #Event though the layer is not recurrent the implementation does not work with "False" -> reason unclear

  def __init__(self, **kwargs):
    """
    """
    super(EnergyNormalization, self).__init__(**kwargs)
    self.set_attr('target', 'classes')
    # normalization matrix
    inputVec = self.sources[0].output
    self.set_attr('n_out', self.sources[0].attrs["n_out"])
    normFactor = 1.0 / T.sqrt(T.dot(inputVec.T, inputVec))
    outputVec = normFactor * inputVec
    self.make_output(outputVec)

class DftLayer(_NoOpLayer):
  """
  This layer is applying the DFT of the input vector. The input is expected to be a segment of the time signal
  cut out with the rectangular function (so no windowing has been done)
  The output of the layer is the absolute values of the complex DFT coefficients. Only non negative coefficients
  are returned because of symmetric spectrum
  """
  layer_class = "dft_layer_abs"
  recurrent = True #Even though the layer is not recurrent the implementation does not work with "False" -> reason unclear
  # (reason: sequences are concatenated otherwise, breaking windowing borders)

  def __init__(self, dftLength=512, windowName='hamming', flag_useSqrtWindow=False, **kwargs):
    super(DftLayer, self).__init__(**kwargs)
    self.set_attr('target', 'classes')
    # DFT properties
    nrOfFreqBins=int(numpy.floor(dftLength/2.0) + 1)
    # windowing
    win = scipy.signal.get_window(windowName, dftLength)
    if flag_useSqrtWindow:
        win = numpy.sqrt(win)
    windowedInput = self.sources[0].output * win
    # create DFT matrix
    nVec = numpy.asarray(range(dftLength))
    kVec = numpy.asarray(range(nrOfFreqBins))
    indexMatrix = numpy.dot(numpy.reshape(kVec, (kVec.shape[0], 1)), numpy.transpose(numpy.reshape(nVec, (nVec.shape[0], 1))))
    dftRealMatrix = numpy.sin(2*numpy.pi*indexMatrix/(float(dftLength)))
    dftImagMatrix = numpy.cos(2*numpy.pi*indexMatrix/(float(dftLength)))
    # apply DFT matrix
    dftAbsCoeff = T.sqrt(T.dot(windowedInput, numpy.transpose(dftRealMatrix))**2 + T.dot(windowedInput, numpy.transpose(dftImagMatrix))**2)
    self.set_attr('n_out', kVec.shape[0])
    self.make_output(dftAbsCoeff)

class GaussianFilter1DLayer(_NoOpLayer):
  layer_class = "gaussian_filter_1d"
  recurrent = True  # Force no frame shuffling or so.

  def __init__(self, sigma, axis, window_radius=40, **kwargs):
    super(GaussianFilter1DLayer, self).__init__(**kwargs)
    z, n_in = concat_sources(self.sources)
    n_out = n_in
    self.set_attr('n_out', n_out)
    self.set_attr('sigma', sigma)
    self.set_attr('axis', axis)
    self.set_attr('window_radius', window_radius)
    from returnn.theano.util import gaussian_filter_1d
    self.output = gaussian_filter_1d(z, sigma=sigma, axis=axis, window_radius=window_radius)


class TimeWarpLayer(_NoOpLayer):
  """
  Like https://en.wikipedia.org/wiki/Image_warping, controlled by NN.
  A bit like simple local feed-forward attention,
  where the attention is controlled by the input (encoder) and not output (decoder).
  Maybe similar: A Hybrid Dynamic Time Warping-Deep Neural Network Architecture for Unsupervised Acoustic Modeling, http://ewan.website/interspeech_2015_dnn_dtw.pdf
  Implementation is very similar to TimeBlurLayer except
  that the weight distribution is different every time frame
  and controlled by a NN.
  Note that this warp is applied locally. See also :class:`TimeWarpGlobalLayer`.
  """
  layer_class = "time_warp"
  recurrent = True  # Force no frame shuffling or so.

  def __init__(self, t_start, t_end, t_step, sigma, input_window, input_proj=None, **kwargs):
    super(TimeWarpLayer, self).__init__(**kwargs)
    z, n_in = concat_sources(self.sources)
    n_out = n_in
    self.set_attr('n_out', n_out)
    self.set_attr('t_start', t_start)
    self.set_attr('t_end', t_end)
    self.set_attr('t_step', t_step)
    self.set_attr('sigma', sigma)
    self.set_attr('input_window', input_window)
    if input_proj:
      self.set_attr('input_proj', input_proj)

    n_batch = z.shape[1]
    n_warp_in = n_in
    warp_in = z
    if input_proj:
      self.W_warp_in_proj = self.add_param(self.create_forward_weights(n=n_in, m=input_proj, name="W_warp_in_proj"))
      warp_in = T.dot(z, self.W_warp_in_proj)
      n_warp_in = input_proj
    self.W_warp = self.add_param(self.create_random_normal_weights(n=input_window, m=n_warp_in, name="W_warp"))

    conv_input = warp_in.dimshuffle(1, 'x', 0, 2)  # batch,stack,row(time),col(feature)
    in_win_right = input_window // 2
    in_win_left = input_window - in_win_right - 1
    conv_input = T.concatenate([T.zeros((n_batch, 1, in_win_left, n_warp_in), dtype="float32"),
                                conv_input,
                                T.zeros((n_batch, 1, in_win_right, n_warp_in), dtype="float32")],
                               axis=2)
    filter_W = self.W_warp.dimshuffle('x', 'x', 0, 1)  # filter,stack,row(time_window),col(feature)
    conv_out = T.nnet.conv2d(conv_input, filter_W, border_mode='valid',
                             filter_shape=[1, 1, input_window, n_warp_in],
                             image_shape=[None, 1, None, n_warp_in])
    # conv_out is 4D (batch size, nb filters=1, output row=time, output col=1).
    warp = conv_out[:, 0, :, 0].dimshuffle(1, 0)  # time,batch
    warp_bc = warp.dimshuffle('x', 0, 1)  # offset,time,batch

    t_offsets = numpy.arange(start=t_start, stop=t_end, step=t_step)  # offset
    t_offsets_bc = T.constant(t_offsets, dtype="float32").dimshuffle(0, 'x', 'x')  # offset,time,batch
    # https://en.wikipedia.org/wiki/Window_function#Gaussian_window
    # If warp would be all 0, the weighting would always be highest at t_offset == 0.
    N = t_end - t_start - 1
    assert N > 0
    warp_bc = T.nnet.sigmoid(warp_bc) * numpy.float32(N) + numpy.float32(t_start)  # force in range [t_start,t_end)
    t_weights = T.exp(numpy.float32(-0.5) *
                      T.sqr((t_offsets_bc - warp_bc) /
                            T.cast(sigma * N / 2.0, dtype="float32")))  # offset,time,batch

    findex = T.cast(self.index, dtype="float32")  # to be able to use on GPU
    self.output = T.zeros_like(z)
    weight_sum = T.zeros_like(findex)  # time,batch
    for idx, t_offset in enumerate(t_offsets):
      if t_offset < 0:
        z_slice = slice(-t_offset, None)
        o_slice = slice(0, t_offset)
      elif t_offset == 0:
        z_slice = slice(None, None)
        o_slice = slice(None, None)
      else:  # t_offset > 0
        z_slice = slice(0, -t_offset)
        o_slice = slice(t_offset, None)
      w = findex[z_slice] * t_weights[idx, z_slice]  # time,batch
      self.output = T.inc_subtensor(self.output[o_slice], z[z_slice] * w.dimshuffle(0, 1, 'x'))
      weight_sum = T.inc_subtensor(weight_sum[o_slice], w)
    self.output = self.output / T.maximum(weight_sum.dimshuffle(0, 1, 'x'), numpy.float32(0.0001))


class TimeWarpGlobalLayer(_NoOpLayer):
  """
  Similar to :class:`TimeWarpLayer` but different.
  This warp is cumulative and applied globally.
  """
  layer_class = "time_warp_global"
  recurrent = True  # Force no frame shuffling or so.

  def __init__(self, n_out=None, renorm_time=True, window_size=30, sigma2=0.5, **kwargs):
    super(TimeWarpGlobalLayer, self).__init__(**kwargs)
    x, n_in = concat_sources(self.sources)
    if n_out: assert n_out == n_in
    else: n_out = n_in
    self.set_attr('n_out', n_out)
    self.set_attr('renorm_time', renorm_time)
    self.set_attr('window_size', window_size)
    self.set_attr('sigma2', sigma2)
    n_time = x.shape[0]
    n_batch = x.shape[1]
    n_time_f32 = T.cast(n_time, dtype="float32")
    i = T.cast(self.sources[0].index, dtype="float32")  # so that it can run on gpu. time,batch
    f32 = numpy.float32
    m = 1  # warp values: compression at idx
    self.W_warp = self.add_var_random_mat(n_in, m, name="W_warp")
    self.b_warp = self.add_param(self.create_bias(m, name="b_warp")).dimshuffle('x', 0)  # batch,m
    from returnn.theano.activation_functions import relu
    warp = T.dot(x, self.W_warp) + self.b_warp  # time,batch,m
    # Right now, we can only shrink the time.
    warp_compr = relu(warp[:, :, 0])  # time,batch
    warp_compr = T.minimum(warp_compr, f32(10))  # time,batch
    idxs = T.cumsum(i / (f32(1) + warp_compr), axis=0) - f32(1)  # time,batch
    if renorm_time:
      # Normalize so that the last idx is always n_time - 1.
      norm_fac = (n_time_f32 - f32(1)) / T.maximum(T.max(idxs[-1]), f32(1))
      idxs *= norm_fac
      new_time = n_time  # old time
    else:
      new_time = T.cast(T.max(idxs[-1]), dtype="int64")

    tgt_idxs = T.cast(T.arange(new_time), dtype="float32")  # new_time
    # Windows, so that the first aligns left and the last aligns right in [0,n_time-1],
    # for all 0..new_time-1.
    w_start_idx_fac = (n_time - window_size + f32(1)) / (new_time - f32(1))
    src_w_start_idxs = T.cast(tgt_idxs * w_start_idx_fac, dtype="int64")  # new_time
    src_w_end_idxs = T.minimum(src_w_start_idxs + window_size, n_time)
    def step(tgt_idx, src_w_start, src_w_end, y_p, x, i, tgt_idxs):
      tgt_idxs_w = tgt_idxs[src_w_start:src_w_end]  # window,batch
      x_w = x[src_w_start:src_w_end]  # window,batch,feature
      i_w = i[src_w_start:src_w_end]  # window,batch
      tgt_idx_bc = tgt_idx.dimshuffle('x', 'x')  # window,batch
      # gauss window
      f_e = T.exp(-T.sqr(T.cast(tgt_idx_bc - tgt_idxs_w, dtype="float32"))
                  / f32(2.0 * sigma2))  # window,batch
      f_e = f_e * i_w  # window,batch
      norm = T.sum(f_e, axis=0, keepdims=True)  # window,batch
      norm = T.maximum(norm, f32(0.00001))  # window,batch
      norm = T.minimum(T.inv(norm), f32(window_size))  # window,batch
      f_e = (f_e * norm).dimshuffle(0, 1, 'x')  # window,batch,feature
      y_t = T.sum(x_w * f_e, axis=0)  # batch,feature
      return y_t
    y_init = T.zeros((n_batch, n_out), dtype="float32")
    y, _ = theano.scan(step,
                       sequences=[tgt_idxs, src_w_start_idxs, src_w_end_idxs],
                       outputs_info=[y_init],
                       non_sequences=[x, i, idxs])
    # self.index = y_i # TODO index if we have new_time != n_time
    self.output = y

  def add_var_random_mat(self, n, m, name):
    l = numpy.sqrt(1.0 / n)
    values = self.rng.uniform(size=(n, m), low=-l, high=l)
    values = numpy.asarray(values, dtype=theano.config.floatX)
    var = self.shared(value=values, borrow=True, name=name)
    return self.add_param(var)


class ConstantLayer(_NoOpLayer):
  layer_class = "constant"

  def __init__(self, value, n_out, dtype="float32", **kwargs):
    super(ConstantLayer, self).__init__(**kwargs)
    self.set_attr("value", value)
    self.set_attr("dtype", dtype)
    self.set_attr("n_out", n_out)
    value = T.constant(numpy.array(value), dtype=dtype)
    if value.ndim == 0:
      value = value.dimshuffle('x', 'x', 'x')
    elif value.ndim == 1:
      value = value.dimshuffle('x', 'x', 0)
    else:
      raise Exception("ndim %i not supported" % value.ndim)
    assert value.ndim == 3
    source = self.sources[0]
    shape = [source.output.shape[0], source.output.shape[1], n_out]
    value += T.zeros(shape, dtype=dtype)  # so we have the same shape as the source output
    self.make_output(value)


class AddZeroRowsLayer(_NoOpLayer):
  layer_class = "add_zero_rows"

  def __init__(self, row_index, number=1, **kwargs):
    super(AddZeroRowsLayer, self).__init__(**kwargs)
    z, n_out = concat_sources(self.sources, unsparse=True)
    assert 0 <= row_index <= n_out
    n_out += number
    self.set_attr("n_out", n_out)
    self.set_attr("row_index", row_index)
    self.set_attr("number", number)
    self.make_output(T.concatenate(
      [z[:, :, :row_index],
       T.zeros((z.shape[0], z.shape[1], number), dtype=z.dtype),
       z[:, :, row_index:]],
      axis=2))


class RemoveRowsLayer(_NoOpLayer):
  layer_class = "remove_rows"

  def __init__(self, row_index, number=1, **kwargs):
    super(RemoveRowsLayer, self).__init__(**kwargs)
    z, n_out = concat_sources(self.sources, unsparse=True)
    assert 0 <= row_index + number <= n_out
    n_out -= number
    self.set_attr("n_out", n_out)
    self.set_attr("row_index", row_index)
    self.set_attr("number", number)
    self.make_output(T.concatenate(
      [z[:, :, :row_index],
       z[:, :, row_index + number:]],
      axis=2))


class BinOpLayer(_NoOpLayer):
  layer_class = "bin_op"

  def __init__(self, op=None, n_out=None, **kwargs):
    """
    :type op: str
    """
    super(BinOpLayer, self).__init__(**kwargs)
    assert len(self.sources) == 2
    s1, s2 = self.sources
    assert s1.attrs["n_out"] == s2.attrs["n_out"]
    if n_out is not None:
      assert n_out == s1.attrs["n_out"]
    assert op
    self.set_attr('op', op.encode("utf8"))
    self.set_attr('n_out', s1.attrs["n_out"])
    if ":" in op:
      op, act = op.split(":", 1)
    else:
      act = None
    op_f = self.get_bin_op(op)
    act_f = strtoact_single_joined(act)
    self.make_output(act_f(op_f(s1.output, s2.output)))

  @staticmethod
  def get_bin_op(op):
    """
    :type op: str
    :rtype: theano.Op
    """
    m = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
    if op in m:
      op = m[op]
    # Assume it's in theano.tensor.
    return getattr(T, op)


class GenericCodeLayer(_NoOpLayer):
  layer_class = "generic_code"

  def __init__(self, code, n_out, **kwargs):
    """
    :param str code: generic Python code used for eval(). must return some output
    """
    super(GenericCodeLayer, self).__init__(**kwargs)
    self.set_attr('n_out', n_out)
    code = code.encode("utf8")
    self.set_attr('code', code)
    output = eval(code, {"self": self, "s": self.sources,
                         "T": T, "theano": theano, "numpy": numpy, "TU": theano_util,
                         "f32": numpy.float32})
    self.make_output(output)


class DualStateLayer(ForwardLayer):
  layer_class = "dual"

  def __init__(self, acts = "relu", acth = "tanh", **kwargs):
    super(DualStateLayer, self).__init__(**kwargs)
    self.set_attr('acts', acts)
    self.set_attr('acth', acth)
    self.activations = [strtoact(acth), strtoact(acts)]
    self.params = {}
    self.W_in = []
    self.act = [self.b,self.b]  # TODO b is not in params anymore?
    for s,m in zip(self.sources,self.masks):
      assert len(s.act) == 2
      for i,a in enumerate(s.act):
        self.W_in.append(self.add_param(self.create_forward_weights(s.attrs['n_out'],
                                                                    self.attrs['n_out'],
                                                                    name="W_in_%s_%s_%d" % (s.name, self.name, i))))
        if s.attrs['sparse']:
          self.act[i] += self.W_in[-1][T.cast(s.act[i], 'int32')].reshape((s.act[i].shape[0],s.act[i].shape[1],s.act[i].shape[2] * self.W_in[-1].shape[1]))
        elif m is None:
          self.act[i] += self.dot(s.act[i], self.W_in[-1])
        else:
          self.act[i] += self.dot(self.mass * m * s.act[i], self.W_in[-1])
    for i in range(2):
      self.act[i] = self.activations[i](self.act[i])
    self.make_output(self.act[0])


class StateToAct(ForwardLayer):
  layer_class = "state_to_act"

  def __init__(self, dual=False, **kwargs):
    kwargs['n_out'] = 1
    super(StateToAct, self).__init__(**kwargs)
    self.set_attr("dual", dual)
    self.params = {}
    self.act = [ T.concatenate([s.act[i][-1] for s in self.sources], axis=1).dimshuffle('x',0,1) for i in range(len(self.sources[0].act)) ] # 1BD
    self.attrs['n_out'] = sum([s.attrs['n_out'] for s in self.sources])
    if dual and len(self.act) > 1:
      self.make_output(T.tanh(self.act[1]))
      self.act[0] = T.tanh(self.act[1])
    else:
      self.make_output(self.act[0])
    #if 'target' in self.attrs:
    #  self.output = self.output.repeat(self.index.shape[0],axis=0)
    #else:
    self.index = T.ones((1, self.index.shape[1]), dtype = 'int8')


class StateVector(ForwardLayer):
  layer_class = "state_vector"

  def __init__(self, output_activation='identity', idx=-1, **kwargs):
    kwargs['n_out'] = 1
    super(StateVector, self).__init__(**kwargs)
    self.params = {}
    f = strtoact_single_joined(output_activation)
    xin = []
    for s in self.sources:
      if hasattr(s,'act'):
        xin.append(f(s.act[1][idx]))
      else:
        xin.append(f(s.output[idx]))
    self.act = [T.concatenate(xin, axis=1).dimshuffle('x', 0, 1)]
    self.act.append(T.zeros_like(self.act[0]))
    self.attrs['n_out'] = sum([s.attrs['n_out'] for s in self.sources])
    self.make_output(self.act[0])
    self.index = T.ones((1, self.index.shape[1]), dtype='int8')


class TimeShift(_NoOpLayer):
  layer_class = "time_shift"

  def __init__(self, base=None, n_shift=1, **kwargs):
    super(TimeShift, self).__init__(**kwargs)
    self.set_attr('n_shift', n_shift)
    self.attrs['n_out'] = self.sources[0].attrs['n_out']
    if n_shift > 1:
      self.output = T.concatenate([T.zeros_like(self.sources[0].output[:n_shift]),self.sources[0].output[:-n_shift]],axis=0)
    else:
      self.output = T.concatenate([T.zeros_like(self.sources[0].output[0]).dimshuffle('x',0,1), self.sources[0].output[:-1]], axis=0)


class TimeConcatLayer(HiddenLayer):
  layer_class = "time_concat"

  def __init__(self, **kwargs):
    kwargs['n_out'] = kwargs['sources'][0].attrs['n_out']
    super(TimeConcatLayer, self).__init__(**kwargs)
    self.make_output(T.concatenate([x.output for x in self.sources],axis=0))
    self.index = T.concatenate([x.index for x in self.sources],axis=0)


class KernelLayer(ForwardLayer):
  layer_class = "kernel"

  def __init__(self, kernel='gauss', base=None, sigma=4.0, **kwargs):
    super(KernelLayer, self).__init__(**kwargs)
    self.params = {}
    sigma = T.constant(sigma,'float32')
    m = self.sources[0].output.dimshuffle('x', 1, 0, 2).repeat(base[0].output.shape[0],axis=0)  # TBVD
    self.pm = numpy.float32(1) / T.sqrt(T.constant(numpy.float32(2)*numpy.pi,'float32')*sigma)
    x = base[0].output.dimshuffle(0, 1, 'x', 2).repeat(m.shape[2], axis=2)  # TBVD
    self.punk = T.exp(-T.sum((x - m.mean(axis=2,keepdims=True))**2/sigma,axis=3)) / T.sqrt(T.constant(numpy.float32(2)*numpy.pi,'float32')*sigma)  # TBVD
    q = (m.sum(axis=2,keepdims=True) - m) / T.cast(m.shape[2]-1,'float32')
    if kernel == 'gauss':
      self.negative = T.exp(-T.sum((x - q)**2/sigma,axis=3)) / T.sqrt(T.constant(numpy.float32(2)*numpy.pi,'float32')*sigma)
      self.output = T.exp(-T.sum((x - m)**2/sigma,axis=3)) / T.sqrt(T.constant(numpy.float32(2)*numpy.pi,'float32')*sigma)
    else:
      raise NotImplementedError()


class CollapseLayer(HiddenLayer):
  layer_class = "collapse"

  def __init__(self, axis=0, **kwargs):
    super(CollapseLayer, self).__init__(**kwargs)
    self.set_attr('axis', axis)
    self.params = {}
    xin = []
    sin = []
    for s in self.sources:
      if hasattr(s,'act'):
        xin.append(s.act[0])
        sin.append(s.act[1])
      else:
        xin.append(s.output)
        sin.append(T.zeros_like(xin[-1]))
    xin = T.concatenate(xin,axis=2).dimshuffle('x',1,0,2)
    sin = T.concatenate(sin,axis=2).dimshuffle('x',1,0,2)
    xin = xin.reshape((1,xin.shape[1],xin.shape[2] * xin.shape[3]))
    sin = sin.reshape((1,sin.shape[1],sin.shape[2] * sin.shape[3]))
    self.make_output(xin)
    self.index = T.ones((1,xin.shape[1]),'int8')
    self.act = [xin,sin]


class HDF5DataLayer(Layer):
  recurrent=True
  layer_class = "hdf5"

  def __init__(self, filename, dset, **kwargs):
    kwargs['n_out'] = 1
    super(HDF5DataLayer, self).__init__(**kwargs)
    self.set_attr('filename', filename)
    self.set_attr('dset', dset)
    import h5py
    h5 = h5py.File(filename, "r")
    data = h5[dset][...]
    self.output = self.shared(value=data.astype('float32'), borrow=True, name=self.name)
    self.index = T.ones((1, self.index.shape[1]), dtype = 'int8')
    h5.close()

class BaseInterpolationLayer(ForwardLayer): # takes a base defined over T and input defined over T' and outputs a T' vector built over an input dependent linear combination of the base elements
  layer_class = "base"

  def __init__(self, base=None, method="softmax", output_weights = False, **kwargs):
    assert base, "missing base in " + kwargs['name']
    kwargs['n_out'] = 1
    super(BaseInterpolationLayer, self).__init__(**kwargs)
    self.set_attr('base', ",".join([b.name for b in base]))
    self.set_attr('method', method)
    self.W_base = [ self.add_param(self.create_forward_weights(bs.attrs['n_out'], 1, name='W_base_%s_%s' % (bs.attrs['n_out'], self.name)), name='W_base_%s_%s' % (bs.attrs['n_out'], self.name)) for bs in base ]
    self.base = T.concatenate([b.output for b in base], axis=2) # TBD
    # self.z : T'
    bz = 0 # : T
    for x,W in zip(base, self.W_base):
      bz += T.dot(x.output,W) # TB1
    z = bz.reshape((bz.shape[0],bz.shape[1])).dimshuffle('x',1,0) + self.z.reshape((self.z.shape[0],self.z.shape[1])).dimshuffle(0,1,'x') # T'BT
    h = z.reshape((z.shape[0] * z.shape[1], z.shape[2])) # (T'xB)T
    if method == 'softmax':
      h_e = T.exp(h).dimshuffle(1,0)
      w = (h_e / T.sum(h_e, axis=0)).dimshuffle(1,0).reshape(z.shape).dimshuffle(2,1,0,'x').repeat(self.base.shape[2], axis=3) # TBT'D
      #w = T.nnet.softmax(h).reshape(z.shape).dimshuffle(2,1,0,'x').repeat(self.base.shape[2], axis=3) # TBT'D
    else:
      assert False, "invalid method %s in %s" % (method, self.name)

    self.set_attr('n_out', sum([b.attrs['n_out'] for b in base]))
    if output_weights:
      self.make_output((h_e / T.sum(h_e, axis=0, keepdims=True)).dimshuffle(1,0).reshape((self.base.shape[0],z.shape[1],z.shape[0])).dimshuffle(2,1,0))
    else:
      self.make_output(T.sum(self.base.dimshuffle(0,1,'x',2).repeat(z.shape[0], axis=2) * w, axis=0, keepdims=False).dimshuffle(1,0,2)) # T'BD


class ChunkingLayer(ForwardLayer): # Time axis reduction like in pLSTM described in http://arxiv.org/pdf/1508.01211v1.pdf
  layer_class = "chunking"

  def __init__(self, chunk_size=1, method = 'concat', **kwargs):
    assert chunk_size >= 1
    kwargs['n_out'] = sum([s.attrs['n_out'] for s in kwargs['sources']]) * chunk_size
    super(ChunkingLayer, self).__init__(**kwargs)
    self.set_attr('chunk_size', chunk_size)
    z = T.concatenate([s.output for s in self.sources], axis=2) # TBD
    residual = z.shape[0] % chunk_size
    padding = T.neq(residual,0) * (chunk_size - residual)

    calloc = T.alloc(numpy.cast[theano.config.floatX](0), z.shape[0] + padding, z.shape[1], z.shape[2])
    container = T.set_subtensor(
      calloc[:z.shape[0]],
      z).dimshuffle('x',0,1,2).reshape((chunk_size,calloc.shape[0] / chunk_size,calloc.shape[1],calloc.shape[2])) # CTBD
    z = T.concatenate([z,T.zeros((padding,z.shape[1],z.shape[2]), 'float32')], axis=0).dimshuffle('x',0,1,2).reshape((chunk_size,(z.shape[0] + padding) / chunk_size,z.shape[1],z.shape[2]))
    #ialloc = T.alloc(numpy.cast['int32'](1), z.shape[1], self.index.shape[1])
    self.index = T.set_subtensor(T.ones((z.shape[1]*z.shape[0],z.shape[2]),'int8')[:self.index.shape[0]],self.index)[::chunk_size]

    if method == 'concat':
      output = z.dimshuffle(1,2,3,0).reshape((z.shape[1], z.shape[2], z.shape[3] * chunk_size))
    elif method == 'average':
      output = z.mean(axis=0)
    self.make_output(output)


class DimToTimeLayer(ForwardLayer):
  layer_class = "dim_to_time"

  def __init__(self, n_time, **kwargs):
    kwargs['n_out'] = sum([s.attrs['n_out'] for s in kwargs['sources']]) // n_time
    super(DimToTimeLayer, self).__init__(**kwargs)
    self.params = {}
    x = T.concatenate([s.output for s in self.sources], axis=2) # TBD
    z = x.reshape((x.shape[0], x.shape[1], n_time, x.shape[2]//n_time))
    self.output = z.dimshuffle(0,2,1,3).reshape((x.shape[0] * n_time, x.shape[1], x.shape[2]//n_time))
    self.index = self.index.dimshuffle(0,'x',1).repeat(n_time,axis=1).reshape((self.index.shape[0] * n_time, self.index.shape[1]))

class TimeToBatchLayer(ForwardLayer):
  layer_class = "time_to_batch"

  def __init__(self, **kwargs):
    kwargs['n_out'] = sum([s.attrs['n_out'] for s in kwargs['sources']])
    super(TimeToBatchLayer, self).__init__(**kwargs)
    self.params = {}
    z = T.concatenate([s.output for s in self.sources], axis=2) # TBD
    self.n_batch = self.index.shape[1]
    self.output = z.reshape((1,z.shape[0] * z.shape[1],z.shape[2]))
    self.index = self.index.reshape((1, self.index.shape[0] * self.index.shape[1]))


class BatchToTimeLayer(ForwardLayer):
  layer_class = "batch_to_time"

  def __init__(self, base, **kwargs):
    kwargs['n_out'] = sum([s.attrs['n_out'] for s in kwargs['sources']])
    n_batch = base[0].n_batch
    super(BatchToTimeLayer, self).__init__(**kwargs)
    self.params = {}
    z = T.concatenate([s.output for s in self.sources], axis=2) # TBD
    z = z.reshape((z.shape[0] * z.shape[1],z.shape[2]))
    self.output = z.reshape((z.shape[0]//n_batch,n_batch,z.shape[1]))
    self.index = self.index.reshape((self.index.shape[0] * self.index.shape[1],))
    self.index = self.index.reshape((self.index.shape[0]//n_batch,n_batch))


class ConcatBatchLayer(_NoOpLayer):
  layer_class = "concat_batch"

  def __init__(self, **kwargs):
    super(ConcatBatchLayer, self).__init__(**kwargs)
    self.attrs['n_out'] = self.sources[0].attrs['n_out']
    self.output = T.concatenate([s.output for s in self.sources], axis=1)
    self.index = T.concatenate([s.index for s in self.sources], axis=1)

class SplitBatchLayer(_NoOpLayer):
  layer_class = "split_batch"

  def __init__(self, n_parts=1, part=0, **kwargs):
    super(SplitBatchLayer, self).__init__(**kwargs)
    self.attrs['n_out'] = sum([s.attrs['n_out'] for s in self.sources])
    chunk = self.index.shape[1] / n_parts
    self.output = T.concatenate([s.output for s in self.sources],axis=2)
    self.output = self.output[:,part*chunk:(part+1)*chunk]
    self.index = self.sources[0].index[:,part*chunk:(part+1)*chunk]

class DuplicateIndexBatchLayer(_NoOpLayer): # TODO: this is a hack
  layer_class = "duplicate_index_batch"

  def __init__(self, **kwargs):
    super(DuplicateIndexBatchLayer, self).__init__(**kwargs)
    self.attrs['n_out'] = 1
    self.index = T.concatenate([self.index] * 2, axis=1)
    self.output = T.zeros(self.index.shape,'float32').dimshuffle(0,1,'x')


class LengthLayer(HiddenLayer):
  layer_class = "length"
  def __init__(self, min_len=0.0, max_len=1.0, use_real=0.0, err='ce', oracle=False, pad=0, **kwargs):
    kwargs['n_out'] = 2
    super(LengthLayer, self).__init__(**kwargs)
    self.set_attr('min_len',min_len)
    self.set_attr('max_len',max_len)
    z = self.get_linear_forward_output()
    z = T.nnet.softmax(z.reshape((z.shape[0]*z.shape[1],z.shape[2]))).reshape(z.shape)
    p = T.clip(T.cast(self.index,'float32') * z[:,:,1], T.constant(1e-20, 'float32'), T.constant(1., 'float32'))
    p = p / T.sum(p,axis=0) #T.nnet.softmax(z[:,:,1].dimshuffle(1,0)).dimshuffle(1,0)
    real = T.sum(T.cast(self.sources[0].target_index,'float32'), axis=0)
    hyp = T.sum(T.arange(z.shape[0],dtype='float32').dimshuffle(0,'x').repeat(z.shape[1],axis=1) * p, axis=0) + numpy.float32(1)
    idx = (self.index.flatten() > 0).nonzero()
    if err == 'ce':
      targets = T.set_subtensor(T.zeros((z.shape[0],z.shape[1]),'int32')[T.sum(self.sources[0].target_index,axis=0) - numpy.int32(1)], numpy.int32(1)).flatten()
      z = z.reshape((z.shape[0]*z.shape[1],z.shape[2]))
      nll, _ = T.nnet.crossentropy_softmax_1hot(x=z[idx], y_idx=targets[idx])
      self.cost_len = T.sum(nll)
    elif err == 'l2':
      self.cost_len = T.sum(self.sources[0].target_index) * T.mean((real - hyp)**2)
    elif err == 'exp':
      self.cost_len = T.sum(self.sources[0].target_index) * T.mean(T.switch(T.lt(real + numpy.float32(1.),hyp),
                                                                   T.sqrt(hyp-real), (real - hyp)**2))
    else:
      assert False, "invalid error: %s" % err

    if (oracle or self.train_flag) and T.and_(T.eq(self.index.shape[1], 1), T.le(self.sources[0].target_index[0],3)):
      self.length = T.cast(numpy.float32(use_real) * real + numpy.float32(1. - use_real) * hyp,'int32')
    else:
      self.length = T.cast(hyp,'int32')
    #self.length += numpy.int32(pad)
    idx, _ = theano.map(lambda l_t,m_t:T.concatenate([T.ones((l_t, ), 'int8'), T.zeros((m_t - l_t, ), 'int8')]),
                        sequences = [self.length], non_sequences=[T.max(self.length) + 1])
    self.index = idx.dimshuffle(1,0)[:-1]
    self.output = self.b

  def cost(self):
    return self.cost_len, None

  def cost_scale(self):
    return T.constant(self.attrs.get("cost_scale", 1.0), dtype="float32")

class LengthProjectionLayer(HiddenLayer):
  layer_class = "length_projection"
  def __init__(self, use_real=1.0, oracle=True, eval_oracle=False, pad=0, smo=0.0, avg=10.0, method="mapq", **kwargs):
    kwargs['n_out'] = 1
    real = T.sum(T.cast(kwargs['index'],'float32'),axis=0)
    kwargs['index'] = T.ones((1,kwargs['index'].shape[1]), 'int8')
    super(LengthProjectionLayer, self).__init__(**kwargs)
    self.params = {}
    self.set_attr('method',method)
    self.set_attr('pad', pad)
    self.set_attr('smo', smo)
    z = T.concatenate([s.output[::s.attrs['direction']][-1] for s in self.sources], axis=1)
    q = T.concatenate([s.act[1][::s.attrs['direction']][-1] for s in self.sources], axis=1)
    zf = T.concatenate([s.output for s in self.sources], axis=2)
    #z = T.sum(zf,axis=0)
    dim = sum([s.attrs['n_out'] for s in self.sources])

    self.b = self.add_param(self.create_bias(1, "b_%s" % self.name))
    #self.cost_scl = T.cast(T.sum(self.sources[0].index),'float32')
    if method == 'scale':
      self.W = self.add_param(self.create_random_uniform_weights(1, dim, l=0.0001, name='W_%s' % self.name))
      self.bs = self.add_param(self.create_bias(1, "bs_%s" % self.name))
      hyp = T.nnet.sigmoid(
        T.sum(self.W.repeat(z.shape[0], axis=0) * z, axis=1) + self.b.repeat(z.shape[0], axis=0)) * (self.bs[0] + T.constant(avg,'float32'))
      self.cost_val = T.sqrt(T.sum(((hyp - real) ** 2) * T.cast(real, 'float32')))
    elif method == 'scaleq':
      self.W_a = self.add_param(self.create_forward_weights(dim,dim,'A'))
      self.ba = self.add_param(self.create_bias(dim, "ba_%s" % self.name))
      z = T.tanh(T.dot(z,self.W_a) + self.ba)
      self.W = self.add_param(self.create_random_uniform_weights(1, dim, l=0.001, name='W_%s' % self.name))
      self.bs = self.add_param(self.create_bias(1, "bs_%s" % self.name))
      hyp = T.nnet.sigmoid(
        T.sum(self.W.repeat(z.shape[0], axis=0) * z, axis=1) + self.b.repeat(z.shape[0], axis=0)) * (
            self.bs[0] + T.constant(avg, 'float32'))
      self.cost_val = T.sqrt(T.sum(((hyp - real) ** 2) * T.cast(real, 'float32')))
    elif method == 'exp':
      self.Q = self.add_param(self.create_random_uniform_weights(dim, 1, l=0.001, name='Q_%s' % self.name))
      expect = T.exp(T.dot(zf,self.Q)[:,:,0] + self.b[0])
      expect = expect / expect.sum(axis=0,keepdims=True)
      hyp = T.cast(T.argmax(expect,axis=0),'float32') + T.constant(1.,'float32')
      self.cost_val = -T.sum(T.log(expect[T.cast(real,'int32')-1,T.arange(expect.shape[1])]) * T.cast(real, 'float32'))
    elif method == 'map':
      self.W = self.add_param(self.create_random_uniform_weights(1, dim, l=0.001, name='W_%s' % self.name))
      hyp = T.sum(self.W.repeat(q.shape[0],axis=0) * z,axis=1) + self.b.repeat(z.shape[0], axis=0) + T.constant(avg,'float32') #T.sum(self.sources[0].index, axis=0) #+ T.constant(1,'float32')
      self.cost_val = T.sqrt(T.sum(((hyp - real) ** 2) * T.cast(real, 'float32')))
    elif method == 'maps':
      #self.W = self.add_param(self.create_random_uniform_weights(1, dim, l=0.001, name='W_%s' % self.name))
      #self.bs = self.add_param(self.create_bias(1, "bs_%s" % self.name))
      hyp = T.sum(q, axis=1) + self.b.repeat(z.shape[0], axis=0) + T.constant(avg, 'float32')
      self.cost_val = T.sqrt(T.sum(((hyp - real) ** 2) * T.cast(real, 'float32')))
    elif method == 'mapq':
      self.W_a = self.add_param(self.create_forward_weights(dim, dim*4, 'A'))
      self.ba = self.add_param(self.create_bias(dim*4, "ba_%s" % self.name))
      z = T.nnet.relu(T.dot(q, self.W_a) + self.ba)
      self.W = self.add_param(self.create_random_uniform_weights(1, dim*4, l=0.001, name='W_%s' % self.name))
      hyp = T.sum(self.W.repeat(z.shape[0], axis=0) * z, axis=1) + self.b.repeat(z.shape[0], axis=0) + T.constant(avg, 'float32')
      self.cost_val = T.sqrt(T.sum(((hyp - real) ** 2) * T.cast(real, 'float32')))
    #if smo != 0:
    #  self.scl = T.sum(T.exp(T.constant(smo,'float32') * (real-hyp))) + T.constant(1,'float32')
    #else:
    #  self.scl = T.constant(1.,'float32')
    #  #self.error_val = T.sum(T.abs_(hyp - T.cast(real, 'float32')) * T.cast(real, 'float32'))
    self.error_val = T.sum(T.cast(T.neq(T.cast(T.round(hyp),'int32'), real), 'float32') * T.cast(real, 'float32'))
    if self.train_flag or (oracle and not self.eval_flag) or eval_oracle:
      self.length = (1. - use_real) * T.round(hyp) + use_real * real
    else:
      self.length = T.round(hyp)
    self.length = T.maximum(self.length,T.ones_like(self.length))
    self.length = T.cast(self.length + T.constant(pad,'float32'),'int32')
    idx, _ = theano.map(lambda l_t,m_t:T.concatenate([T.ones((l_t, ), 'int8'), T.zeros((m_t - l_t, ), 'int8')]),
                        sequences = [self.length], non_sequences=[T.max(self.length) + 1])
    self.index = idx.dimshuffle(1,0)[:-1]
    self.output = self.length.dimshuffle('x',0,'x')

  def cost(self):
    return self.cost_val, None

  def errors(self):
    return self.error_val

  def cost_scale(self):
    return T.constant(self.attrs.get("cost_scale", 1.0), dtype="float32")


class LengthUnitLayer(HiddenLayer):
  layer_class = "length_unit"
  def __init__(self, min_len = 1, max_len = 32, **kwargs):
    kwargs['n_out'] = 1
    super(LengthUnitLayer, self).__init__(**kwargs)
    q = T.nnet.sigmoid(self.get_linear_forward_output()[-1:])
    self.length = numpy.float32(min_len) + q * numpy.float32(max_len - min_len + 1)
    idx, _ = theano.map(lambda l_t, m_t: T.concatenate([T.ones((l_t,), 'int8'), T.zeros((m_t - l_t,), 'int8')]),
                        sequences=[self.length], non_sequences=[T.max(self.length) + 1])
    self.index = idx.dimshuffle(1, 0)[:-1]
    self.output = self.length.dimshuffle('x', 0, 'x')

class SegmentLayer(_NoOpLayer):
  layer_class = 'segment'

  def __init__(self, **kwargs):
    super(SegmentLayer, self).__init__(**kwargs)
    assert len(kwargs['sources']) == 2
    self.attrs['n_out'] = kwargs['sources'][0].attrs['n_out']
    cutoff = self.index.shape[0] / 2
    concat = T.concatenate([kwargs['sources'][0].output[:cutoff],kwargs['sources'][1].output[cutoff:]],axis=0)
    self.make_output(concat)


class AttentionLayer(_NoOpLayer):
  layer_class = 'attention'

  def __init__(self, base, conv_x=None, conv_y=None, **kwargs):
    super(AttentionLayer, self).__init__(**kwargs)
    if conv_x:
      self.set_attr('conv_x',conv_x)
    if conv_y:
      self.set_attr('conv_y',conv_y)
    self.set_attr('base', ",".join([b.name for b in base]))
    self.attrs['n_out'] = sum([s.attrs['n_out'] for s in base])
    self.W_out = self.add_param(self.create_forward_weights(base[0].attrs['n_out'], self.attrs['n_out']), 'W_out')
    self.b_out = self.add_param(self.create_bias(self.attrs['n_out']), 'b_out')
    base = base[0].output if len(base) == 1 else T.concatenate([b.output for b in base], axis=2)
    base = T.tanh(T.dot(base,self.W_out) + self.b_out)
    attention = T.zeros((self.index.shape[0],self.index.shape[1],base.shape[0]), 'float32')
    for src in kwargs['sources']:
      for att in src.attention:
        attention += att
    attention = attention / attention.sum(axis=2, keepdims=True) # NBT
    att, _ = theano.map(lambda att,base: T.sum(base*att.dimshuffle(1,0,'x').repeat(base.shape[2],axis=2),axis=0),
                        sequences=[attention], non_sequences=[base])
    self.make_output(att)
    self.act = [ att, T.zeros_like(att) ]
    #attention = attention.dimshuffle(0,1,'x',2).repeat(base.shape[2],axis=2) # NBDT
    #self.make_output(T.sum(base.dimshuffle('x',1,2,0).repeat(self.index.shape[0],axis=0) * attention,axis=3))


class SourceAttentionLayer(_NoOpLayer):
  layer_class = 'source_attention'

  def __init__(self, base, n_tmp=64, **kwargs):
    super(SourceAttentionLayer, self).__init__(**kwargs)
    self.set_attr('base', ",".join([b.name for b in base]))
    self.attrs['n_out'] = sum([s.attrs['n_out'] for s in base])
    n_base = sum([s.attrs['n_out'] for s in base])
    n_in = sum([s.attrs['n_out'] for s in self.sources])
    x_in = self.sources[0].output if len(self.sources) == 1 else T.concatenate([s.output for s in self.sources],axis=2)
    B = base[0].output if len(base) == 1 else T.concatenate([b.output for b in base], axis=2)
    self.W_base = self.add_param(self.create_forward_weights(n_base, n_tmp), 'W_base')
    self.b_base = self.add_param(self.create_bias(n_tmp), 'b_base')
    self.W_in = self.add_param(self.create_forward_weights(n_in, n_tmp), 'W_in')
    self.b_in = self.add_param(self.create_bias(n_tmp), 'b_in')
    C = T.tanh(T.dot(B,self.W_base) + self.b_base)
    X = T.tanh(T.dot(x_in, self.W_in) + self.b_in)
    def attmap(x,C,B):
      D = x.dimshuffle('x',0,1).repeat(C.shape[0],axis=0)
      e = T.exp(T.sqrt(T.sum((D-C)**2,axis=2))) # TB
      e = e / e.sum(axis=0,keepdims=True)
      return T.sum(B * e.dimshuffle(0,1,'x').repeat(B.shape[2],axis=2),axis=0)

    x_out, _ = theano.map(attmap, sequences=[X], non_sequences=[C, B])
    self.make_output(x_out)


class ReverseAttentionLayer(_NoOpLayer):
  layer_class = 'reverse_attention'

  def __init__(self, base = None, **kwargs):
    super(ReverseAttentionLayer, self).__init__(**kwargs)
    self.attention = []
    T = base[0].attention[0].shape[2]
    B = self.index.shape[1]
    N = self.index.shape[0]
    D = self.sources[0].output.shape[2]
    att = base[0].attention[0] # NBT
    att = att / att.sum(axis=0,keepdims=True)
    res = att.dimshuffle(0,1,2,'x').repeat(D,axis=3) * self.sources[0].output.dimshuffle(0,1,'x',2).repeat(T,axis=2) # NBTD
    self.output = res.sum(axis=0).dimshuffle(1,0,2) # TBD
    self.index = base[0].index # TB
    self.attrs['n_out'] = self.sources[0].attrs['n_out']



class AttentionVectorLayer(_NoOpLayer):
  layer_class = 'attention_vector'

  def __init__(self, base, template, **kwargs):
    super(AttentionVectorLayer, self).__init__(**kwargs)
    target = None if not 'target' in kwargs else kwargs['target']
    self.set_attr('base', ",".join([b.name for b in base]))
    self.set_attr('template',template)
    self.attrs['n_out'] = sum([s.attrs['n_out'] for s in base])
    self.params = {}
    memory = base[0].output if len(base) == 1 else T.concatenate([b.output for b in base], axis=2)
    W_base = self.add_param(self.create_random_uniform_weights(self.attrs['n_out'], template, l=0.01, name='W_base'), 'W_base')
    b_base = self.add_param(self.create_bias(template,name='b_base'), 'b_base')
    base = T.tanh(T.dot(memory, W_base) + b_base) #/ T.constant(template,'float32')

    state = T.concatenate([s.output for s in self.sources], axis=2).repeat(memory.shape[0],axis=0)
    n_in = sum([s.attrs['n_out'] for s in self.sources])
    W_state = self.add_param(self.create_random_uniform_weights(n_in, template, l=0.1, name='W_state'), 'W_state')
    b_state = self.add_param(self.create_bias(template,name='b_state'), 'b_state')
    state = T.tanh(T.dot(state, W_state) + b_state) #/ T.constant(template,'float32')

    #base = self.batch_norm(base,template,use_shift=False)
    #state = self.batch_norm(state,template,use_shift=False)
    #state = state / T.sqrt(T.sum(state**2))
    #base = base / T.sqrt(T.sum(base ** 2))
    #sim = T.exp(-T.sqrt(T.sqr(base - state).sum(axis=2)))
    sim = T.exp(-T.sqr(base - state).sum(axis=2) / T.constant(template,'float32'))
    #sim = T.exp(T.sum(state * base,axis=2)) # / T.constant(template,'float32'))
    alpha = (sim / T.sum(sim,axis=0,keepdims=True))
    self.make_output(T.sum(alpha.dimshuffle(0,1,'x').repeat(self.attrs['n_out'],axis=2) * memory,axis=0,keepdims=True))
    self.act = [self.output, T.zeros_like(self.output)]

    self.cost_val = 0
    if target:
      trg = self.y_in[target].flatten()
      #idx = T.sum(T.arange(0,base.shape[0],dtype='float32').dimshuffle(0,'x').repeat(base.shape[1],axis=1) * alpha,axis=0)
      self.cost_val = -T.sum(T.log(alpha[trg,T.arange(base.shape[1],dtype='int32')]))
      self.error_val = T.cast(T.argmax(alpha,axis=0) - trg,'float32')**2

  def cost(self):
    return self.cost_val, None


class StateAlignmentLayer(HiddenLayer):
  layer_class = 'state_alignment'

  def __init__(self, target, prior_scale = 0.0, **kwargs):
    kwargs['n_out'] = kwargs['y'][target].n_out
    super(StateAlignmentLayer,self).__init__(**kwargs)
    z = self.z.reshape((self.z.shape[0] * self.z.shape[1], self.z.shape[2]))
    p = T.nnet.softmax(z).reshape(self.z.shape)
    custom_init = numpy.ones((self.attrs['n_out'],), 'float32') / numpy.float32(self.attrs['n_out'])
    custom = T.mean(p[(self.index.flatten() > 0).nonzero()], axis=0)
    priors = self.add_param(theano.shared(custom_init, 'priors'), 'priors',
                                 custom_update=custom,
                                 custom_update_normalized=True)

    nlog_scores = T.log(p) - numpy.float32(prior_scale) * T.log(priors)
    states = InvAlignOp([1e10, 0., 1.9, 3., 2.5, 2., 1.4])(self.sources[0].index, self.index, -nlog_scores, self.y)
    index_flat = T.set_subtensor(self.index.flatten()[(T.eq(states.flatten(), -1) > 0).nonzero()], numpy.int8(0))
    k = (states.flatten() + numpy.int8(1) > 0).nonzero()

    inp, _ = theano.scan(lambda x, i, h, p: (x + h if i == p else x, i),
                         sequences=[self.z, states], outputs_info=[T.zeros_like(self.z[0]), -T.ones_like(states[0])])
    self.output = inp[0]



class SignalSplittingLayer(HiddenLayer):
  layer_class = "signal_splitter"
  def __init__(self, base, p=0.5, oracle=False, **kwargs):
    kwargs['n_out'] = 1
    real = T.sum(T.cast(kwargs['index'],'float32'),axis=0)
    #kwargs['index'] = T.ones((1,kwargs['index'].shape[1]), 'int8')
    super(SignalSplittingLayer, self).__init__(**kwargs)
    self.params = {}
    z = T.concatenate([s.output[-1] for s in base], axis=1)
    dim = sum([s.attrs['n_out'] for s in base])
    self.W = self.add_param(self.create_random_uniform_weights(1, dim, l = 0.1, name='W_%s' % self.name))
    self.b = self.add_param(self.create_bias(1, "b_%s" % self.name))
    hyp = (T.nnet.sigmoid(T.sum(self.W.repeat(z.shape[0],axis=0) * z,axis=1)) + self.b.repeat(z.shape[0],axis=0)) * T.sum(base[0].index, axis=0)
    self.cost_val = 0.0 #T.sum((hyp - real)**2)
    self.p = p
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(self.rng.randint(1234) + 1)
    if self.train_flag:
      self.xflag = T.cast(srng.binomial(n=1, p=1.0 - p, size=(1,)), 'float32')[0]
    else:
      self.xflag = T.constant(1., 'float32')
    if oracle:
      self.length = real
    else:
      self.length = self.xflag * T.ceil(hyp) + (1 - self.xflag) * real
    self.length = T.cast(self.length, 'int32')

    idx, _ = theano.map(lambda l_t,m_t:T.concatenate([T.ones((l_t, ), 'int8'), T.zeros((m_t - l_t, ), 'int8')]),
                        sequences = [self.length], non_sequences=[T.max(self.length) + 1])
    self.index = idx.dimshuffle(1,0)[:-1]
    self.output = z

  def cost(self):
    return self.cost_val, None

  def cost_scale(self):
    return T.constant(self.attrs.get("cost_scale", 1.0), dtype="float32")


class RoutingLayer(HiddenLayer):
  layer_class = "signal_router"
  def __init__(self, base, p=0.5, oracle=False, **kwargs):
    kwargs['n_out'] = 1
    real = T.sum(T.cast(kwargs['index'],'float32'),axis=0)
    #kwargs['index'] = T.ones((1,kwargs['index'].shape[1]), 'int8')
    super(RoutingLayer, self).__init__(**kwargs)
    self.params = {}
    z = T.concatenate([s.output[-1] for s in base], axis=1)
    self.cost_val = 0.0 #T.sum((hyp - real)**2)
    self.p = p
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(self.rng.randint(1234) + 1)
    if self.train_flag:
      self.xflag = T.cast(srng.binomial(n=1, p=1.0 - p, size=(1,)), 'float32')[0]
    else:
      self.xflag = T.constant(1., 'float32')
    import theano.ifelse
    self.output = z #theano.ifelse.ifelse(self.xflag, base[0].output, base[1].output)
    self.act = [ theano.ifelse.ifelse(self.xflag, base[0].act[i][-1:], base[1].act[i][-1:]) for i in range(len(base[0].act))]
    self.length = real
    #idx, _ = theano.map(lambda l_t,m_t:T.concatenate([T.ones((l_t, ), 'int8'), T.zeros((m_t - l_t, ), 'int8')]),
    #                    sequences = [self.length], non_sequences=[T.max(self.length) + 1])
    #self.index = idx.dimshuffle(1,0)[:-1]

  def cost(self):
    return self.cost_val, None

  def cost_scale(self):
    return T.constant(self.attrs.get("cost_scale", 1.0), dtype="float32")


class RandomRouteLayer(_NoOpLayer):
  layer_class = "random_route"

  def __init__(self, p=None, test_route=-1, n_out=None, **kwargs):
    super(RandomRouteLayer, self).__init__(**kwargs)
    assert len(kwargs['sources']) > 1, "There is no route to select."
    if p is None:
      p = [1. / len(kwargs['sources'])] * len(kwargs['sources'])
    if isinstance(p, (int, long, float)):
      p = [p]
    assert isinstance(p, (list, tuple))
    if len(p) == len(self.sources) - 1:
      p.append(1.-sum(p))
    assert sum(p) == 1. and all([x>=0. for x in p])
    assert len(p) == len(self.sources)
    if not n_out:
      n_out = self.sources[0].attrs['n_out']
    assert all([n_out == s.attrs['n_out'] for s in self.sources])
    self.set_attr('n_out', n_out)
    self.set_attr('p', p)
    self.set_attr('test_route', test_route)
    import theano.ifelse
    if not self.train_flag:
      if test_route >= 0:
        self.output = self.sources[test_route].output
      else:
        output = numpy.float32(p[0]) * self.sources[0].output
        for s, pc in zip(self.sources[1:], p[1:]):
          output += numpy.float32(pc) * s.output
        self.output = output
    else:
      from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
      rng = RandomStreams(self.rng.randint(1234) + 1)
      rv = rng.uniform((1,), low=0.0, high=1.0, dtype="float32")[0]
      output = self.sources[0].output
      p0 = p[0]
      for s, pc in zip(self.sources[1:], p[1:]):
        output = theano.ifelse.ifelse(T.gt(rv, numpy.float32(p0)), s.output, output)
        p0 += pc
      self.output = output


class AdaptiveDepthLayer(HiddenLayer):
  layer_class = "adaptive_depth"

  def __init__(self, eps=0.01, tau=0.01, bias=-1.0, damping='graves', **kwargs):
    kwargs['n_out'] = 1
    super(AdaptiveDepthLayer, self).__init__(**kwargs)
    self.attrs['n_out'] = kwargs['sources'][0].attrs['n_out']
    self.attrs['tau'] = tau
    self.attrs['eps'] = eps
    self.attrs['bias'] = bias
    self.attrs['damping'] = damping
    assert all([l.attrs['n_out'] == self.attrs['n_out'] for l in kwargs['sources']])
    shape = self.index.shape
    P = T.zeros((shape[0],shape[1]),'float32')
    M = T.zeros((shape[0],shape[1]),'int8')
    H = T.zeros((shape[0],shape[1],self.attrs['n_out']),'float32')
    threshold = T.constant(1. - eps,'float32')
    self.cost_val = T.constant(0,'float32')
    for W,layer,i in zip(self.W_in,kwargs['sources'],range(len(self.W_in))):
      h = layer.act[1][::layer.attrs['direction']] if layer.layer_class=='rec' else layer.output
      if damping == 'random':
        p = T.cast(self.rng.uniform(size=shape),'float32')
        del self.params[W.name]
      else:
        p = T.nnet.sigmoid(T.dot(h,W) + self.b + T.constant(bias,'float32'))[:,:,0]
      N = T.ge(P + p, threshold) * T.lt(P, threshold)
      M += N
      Q = N.dimshuffle(0,1,'x').repeat(layer.attrs['n_out'],axis=2)
      H = Q * layer.output + (numpy.float32(1.) - Q) * H
      if damping == 'graves':
        self.cost_val += T.sum((numpy.float32(i+2) - P) * T.cast(N,'float32')) # ((1-P) + (i+1))*N
      elif damping == 'expected':
        self.cost_val += T.sum(T.cast(N,'float32') * p * numpy.float32(i+1))
      P += p
    # target probability not reached
    M = numpy.float32(1.) - T.cast(M,'float32')
    H += M.dimshuffle(0,1,'x').repeat(layer.attrs['n_out'],axis=2) * layer.output
    if damping == 'graves':
      self.cost_val += T.sum((numpy.float32(len(self.W_in)+1) - P) * M)
    elif damping == 'expected':
      self.cost_val += T.sum(M * P * numpy.float32(len(self.W_in)))
    self.make_output(H)

  def cost(self):
    return self.cost_val,None
  def cost_scale(self):
    return T.constant(self.attrs['tau'],'float32')


class AttentionReshapeLayer(_NoOpLayer):
  layer_class = 'attention_reshape'

  def __init__(self, conf=0.3, pad=1, cap=1, **kwargs):
    super(AttentionReshapeLayer, self).__init__(**kwargs)
    assert cap >= pad
    target = 'classes' if not 'target' in self.attrs else self.attrs['target']
    x_in, n_in = concat_sources(self.sources)
    x_in = x_in.reshape((self.index.shape[0] * self.index.shape[1], n_in))
    self.set_attr('n_out', n_in)
    self.set_attr('conf', conf)
    self.set_attr('pad', pad)
    self.set_attr('cap',cap)
    conf = T.constant(conf,'float32')
    pad = T.constant(pad,'int32')
    cap = T.constant(cap, 'int32')
    attention = T.constant(0,'float32')
    for src in kwargs['sources']:
      for att in src.attention:
        attention += att
    attention = attention / attention.sum(axis=2,keepdims=True)
    B = (T.argmax(attention, axis=2) + T.arange(attention.shape[1],dtype='float32') * T.cast(attention.shape[2],'float32')).flatten()
    H = T.cast(T.max(T.ge(attention,conf),axis=2),'float32') * T.cast(self.index,'float32') # NB
    Q = T.switch(T.le(H.shape[0], cap), H, H[:-self.attrs['cap']])
    def smooth(h, h_p, c_p):
      c_t = (c_p + numpy.float32(1.))
      h_t = T.cast(T.and_(T.cast(h,'int32'),T.ge(c_t,cap)),'float32')
      return h_t, c_t * T.cast(1-h_t,'float32')
    outputs, _ = theano.scan(smooth, sequences=[Q], outputs_info=[T.zeros_like(Q[0]),T.zeros((self.index.shape[1],),'float32')])
    marker = T.switch(T.le(H.shape[0],cap), outputs[0], T.concatenate([outputs[0],H[-self.attrs['cap']:]],axis=0))
    marker = T.inc_subtensor(marker[-1],numpy.float32(1.)).flatten()
    idx = (marker > 0).nonzero()
    length_y = T.arange(marker.shape[0],dtype='int32')[idx[1:]] - T.arange(marker.shape[0],dtype='int32')[idx[:-1]]
    offset_y = T.extra_ops.cumsum(length_y)
    max_len_y = T.max(length_y)+numpy.int32(1)
    length_x = T.cast(B[idx[1:]] - B[idx[:-1]],'int32')
    offset_x = T.extra_ops.cumsum(length_x)
    max_len_x = T.max(length_x)+numpy.int32(1)
    def cut(l_x, o_x, l_y, o_y, X, Y, maxx, maxy):
      x = T.concatenate([X[o_x:o_x + l_x], T.zeros((maxx-l_x,n_in),'float32')], axis=0)
      i = T.concatenate([T.zeros((l_x,),'int8'), T.zeros((maxx - l_x,),'int8')], axis=0)
      y = T.concatenate([Y[o_y:o_y + l_y], T.zeros((maxy-l_y,),'int32')], axis=0)
      return x, y, i
    outputs, _ = theano.map(cut, sequences=[length_x,offset_x,length_y,offset_y],
                            non_sequences=[x_in,self.y_in[target],max_len_x,max_len_y])
    self.y_out = outputs[1].dimshuffle(1,0)[:-1]
    self.index = outputs[2].dimshuffle(1,0)[:-1]
    self.make_output(outputs[0].dimshuffle(1,0,2)[:-1])


class DetectionLayer(HiddenLayer):
  layer_class = "detection"
  def __init__(self, label_idx, **kwargs):
    kwargs['n_out'] = 2
    super(DetectionLayer, self).__init__(**kwargs)
    z = self.get_linear_forward_output()
    z = T.nnet.softmax(z.reshape((z.shape[0]*z.shape[1],z.shape[2]))).reshape(z.shape)
    idx = (self.index.flatten() > 0).nonzero()
    targets = T.eq(self.y_in[self.attrs['target']], label_idx)
    z = z.reshape((z.shape[0]*z.shape[1],z.shape[2]))
    nll, _ = T.nnet.crossentropy_softmax_1hot(x=z[idx], y_idx=targets[idx])
    self.cost_val = T.sum(nll)
    self.output = z

  def cost(self):
    return self.cost_val, None


class TruncationLayer(Layer):
  layer_class = "trunc"

  def __init__(self, n_trunc, **kwargs):
    kwargs['n_out'] = sum([s.attrs['n_out'] for s in kwargs['sources']])
    super(TruncationLayer, self).__init__(**kwargs)
    self.set_attr('from', ",".join([s.name for s in self.sources]))
    self.set_attr('n_trunc', n_trunc)
    n_trunc = T.switch(T.gt(n_trunc, self.index.shape[0]), self.index.shape[0], n_trunc)
    z = T.concatenate([s.output for s in self.sources], axis=2)
    self.index = self.index[:n_trunc]
    self.make_output(z[:n_trunc])
    #self.make_output(z)


class CorruptionLayer(_NoOpLayer): # x = x + noise
  layer_class = "corruption"
  from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
  rng = RandomStreams(hash(layer_class) % 2147462579)

  def __init__(self, noise='gaussian', p=0.0, clip=False, **kwargs):
    super(CorruptionLayer, self).__init__(**kwargs)
    self.set_attr('noise', noise)
    self.set_attr('p', p)
    self.set_attr('n_out', sum([s.attrs['n_out'] for s in self.sources]))

    z = T.concatenate([s.output for s in self.sources], axis=2)
    if noise == 'gaussian':
      z += self.rng.normal(size=z.shape,avg=0,std=p,dtype='float32') #+ (z - T.mean(z, axis=(0,1), keepdims=True)) / T.std(z, axis=(0,1), keepdims=True)
    elif noise == 'binomial':
      z += self.rng.binomial(size=z.shape, p=p, dtype='float32')
    if clip:
      z = T.clip(z,numpy.float32(0),numpy.float32(1))
    self.make_output(z)


class InputBase(Layer):
  layer_class = "input_base"

  def __init__(self, **kwargs):
    kwargs['n_out'] = 1
    super(InputBase, self).__init__(**kwargs)
    assert len(self.sources) == 1
    self.set_attr('from', ",".join([s.name for s in self.sources]))
    self.make_output(self.sources[0].W_in[0].dimshuffle(0,'x',1).repeat(self.index.shape[1],axis=1))
    self.set_attr('n_out', self.sources[0].W_in[0].get_value().shape[1])


class ConvPoolLayer(ForwardLayer):
  layer_class = "convpool"

  def __init__(self, dx, dy, fx, fy, **kwargs):
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
    self.W = self.add_param(self.shared( numpy.asarray(self.rng.uniform(low=-range,high=range,size=(2,1,fx,fy)), dtype = theano.config.floatX), name = "W_%s" % self.name), name = "W_%s" % self.name)
    conv_out = conv.conv2d(x_in, self.W)

    # initialize shared variable for weights.
    w_shp = (2, 3, 9, 9)
    w_bound = numpy.sqrt(3 * 9 * 9)
    W = self.shared( numpy.asarray(
                self.rng.uniform(
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
    b = self.shared(numpy.asarray(
                self.rng.uniform(low=-.5, high=.5, size=b_shp),
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


class LossLayer(Layer):
  layer_class = "loss"

  def __init__(self, loss, copy_input=None, **kwargs):
    """
    :param theano.Variable index: index for batches
    :param str loss: e.g. 'ce'
    """
    super(LossLayer, self).__init__(**kwargs)
    y = self.y_in
    if copy_input:
      self.set_attr("copy_input", copy_input.name)
    if not copy_input:
      self.z = self.b
      self.W_in = [self.add_param(self.create_forward_weights(source.attrs['n_out'], self.attrs['n_out'],
                                                              name="W_in_%s_%s" % (source.name, self.name)))
                   for source in self.sources]

      assert len(self.sources) == len(self.masks) == len(self.W_in)
      assert len(self.sources) > 0
      for source, m, W in zip(self.sources, self.masks, self.W_in):
        if source.attrs['sparse']:
          self.z += W[T.cast(source.output[:,:,0], 'int32')]
        elif m is None:
          self.z += self.dot(source.output, W)
        else:
          self.z += self.dot(self.mass * m * source.output, W)
    else:
      self.z = copy_input.output
    self.set_attr('from', ",".join([s.name for s in self.sources]))
    if self.y.dtype.startswith('int'):
      i = (self.index.flatten() > 0).nonzero()
    elif self.y.dtype.startswith('float'):
      i = (self.index.flatten() > 0).nonzero()
    self.j = ((T.constant(1.0) - self.index.flatten()) > 0).nonzero()
    loss = loss.encode("utf8")
    self.attrs['loss'] = self.loss
    n_reps = T.switch(T.eq(self.z.shape[0], 1), self.index.shape[0], 1)
    output = self.output.repeat(n_reps,axis=0)
    y_m = output.reshape((output.shape[0]*output.shape[1],output.shape[2]))
    self.known_grads = None
    if loss == 'ce':
      if self.y.type == T.ivector().type:
        nll, pcx = T.nnet.crossentropy_softmax_1hot(x=y_m[i], y_idx=y[i])
      else:
        pcx = T.nnet.softmax(y_m[i])
        nll = -T.dot(T.log(T.clip(pcx, 1.e-38, 1.e20)), y[i].T)
      self.constraints += T.sum(nll)
      self.make_output(pcx.reshape(output.shape))
    elif loss == 'entropy':
      h_e = T.exp(self.y_m) #(TB)
      pcx = T.clip((h_e / T.sum(h_e, axis=1, keepdims=True)).reshape((self.index.shape[0],self.index.shape[1],self.attrs['n_out'])), 1.e-6, 1.e6) # TBD
      ee = self.index * -T.sum(pcx * T.log(pcx)) # TB
      nll, pcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m, y_idx=self.y) # TB
      ce = nll.reshape(self.index.shape) * self.index # TB
      y = self.y.reshape(self.index.shape) * self.index # TB
      f = T.any(T.gt(y,0), axis=0) # B
      self.constraints += T.sum(f * T.sum(ce, axis=0) + (1-f) * T.sum(ee, axis=0))
      self.make_output(pcx.reshape(output.shape))
    elif loss == 'priori':
      pcx = T.nnet.softmax(y_m)[i, y[i]]
      pcx = T.clip(pcx, 1.e-38, 1.e20)  # For pcx near zero, the gradient will likely explode.
      self.constraints += -T.sum(T.log(pcx))
      self.make_output(pcx.reshape(output.shape))
    elif loss == 'sse':
      if self.y.dtype.startswith('int'):
        y_f = T.cast(T.reshape(y, (y.shape[0] * y.shape[1]), ndim=1), 'int32')
        y_oh = T.eq(T.shape_padleft(T.arange(self.attrs['n_out']), y_f.ndim), T.shape_padright(y_f, 1))
        self.constraints += T.mean(T.sqr(y_m[i] - y_oh[i]))
      else:
        self.constraints += T.sum(T.sqr(y_m[i] - y.reshape(y_m.shape)[i]))
      self.make_output(y_m[i].reshape(output.shape))
    else:
      raise NotImplementedError()

    if y.dtype.startswith('int'):
      if y.type == T.ivector().type:
        self.error = T.sum(T.neq(T.argmax(y_m[i], axis=-1), y[i]))
      else:
        self.error = T.sum(T.neq(T.argmax(y_m[i], axis=-1), T.argmax(y[i], axis = -1)))
    elif y.dtype.startswith('float'):
      self.error = T.sum(T.sqr(y_m[i] - y.reshape(y_m.shape)[i]))
    else:
      raise NotImplementedError()


class ErrorsLayer(_NoOpLayer):
  layer_class = "errors"

  def __init__(self, target, **kwargs):
    super(ErrorsLayer, self).__init__(**kwargs)
    self.set_attr("target", target)
    assert target in self.network.y
    self.y = self.network.y[target]
    assert self.y.ndim == 2
    n_out = self.network.n_out[target][0]
    self.set_attr("n_out", n_out)
    self.set_attr("sparse", True)
    self.z, z_dim = concat_sources(self.sources, unsparse=True)
    assert z_dim == n_out
    self.output = T.neq(T.argmax(self.z, axis=2), self.y) * self.index

  def errors(self):
    """
    :rtype: theano.Variable
    """
    return T.sum(self.output)


class TorchLayer(_NoOpLayer):
  recurrent = True  # who knows
  layer_class = "torch"

  def __init__(self, n_out, lua_fw_func, lua_bw_func, params, lua_file=None, **kwargs):
    super(TorchLayer, self).__init__(**kwargs)
    self.set_attr("n_out", n_out)
    if isinstance(params, (str, unicode)):
      params = json.loads(params)
    self.set_attr("params", params)
    assert isinstance(params, (tuple, list))  # list[param-init-dict]

    args = []
    args_info = []  # dict with ndim, shape, n_in

    for s, m in zip(self.sources, self.masks):
      arg_info = {"ndim": s.output.ndim, "n_out": s.attrs['n_out'], "type": "input_source"}
      args_info += [arg_info]
      arg_info["shape"] = [None] * s.output.ndim
      if not s.attrs['sparse']:
        arg_info["shape"][-1] = s.attrs['n_out']
      if s.attrs['sparse'] or m is None:
        args += [s.output]
      else:
        args += [self.mass * m * s.output]

    for param_init_dict in params:
      assert isinstance(param_init_dict, dict)
      assert "name" in param_init_dict
      param_init_dict = param_init_dict.copy()
      param_init_dict["name"] += "_%s" % self.name
      func_name = param_init_dict.pop("class", "create_random_uniform_weights")
      func = getattr(self, func_name)
      p = func(**param_init_dict)
      assert isinstance(p, theano.compile.SharedVariable)
      p = self.add_param(p)
      p_shape = p.get_value(borrow=True, return_internal_type=True).shape
      p_ndim = len(p_shape)
      args += [p]
      args_info += [{"ndim": p_ndim, "shape": tuple(p_shape), "type": "input_param"}]

    args += [self.index]
    args_info += [{"ndim": 2, "shape": (None, None), "gradient": "disconnected", "type": "input_index"}]

    from returnn.theano.ops.torch_wrapper import TorchWrapperOp
    op = TorchWrapperOp(
      name=self.name,
      in_info=args_info,
      # Hardcoded output shape for now, only feature dim can be configured.
      out_info=[{"n_out": n_out, "ndim": 3, "shape": ((0, 0), (0, 1), n_out),
                 "dtype": "float32", "type": "output"}],
      lua_file=lua_file, lua_fw_func=lua_fw_func, lua_bw_func=lua_bw_func)
    self.output = op(*args)


class NativeLayer(_NoOpLayer):
  recurrent = True  # who knows
  layer_class = "native"

  def __init__(self, n_out, native_class, params, **kwargs):
    super(NativeLayer, self).__init__(**kwargs)
    self.set_attr("n_out", n_out)
    if isinstance(params, (str, unicode)):
      params = json.loads(params)
    self.set_attr("params", params)
    assert isinstance(params, (tuple, list))  # list[param-init-dict]
    self.set_attr('native_class', native_class)

    import returnn.native_op
    native_class_cls = getattr(returnn.native_op, native_class)
    assert issubclass(native_class_cls, returnn.native_op.NativeOpGenBase)
    op = native_class_cls().make_theano_op()

    args = []
    args_info = []  # dict with ndim, shape, n_in

    x, n_in = concat_sources(self.sources, masks=self.masks, mass=self.mass)
    args += [x]
    args_info += [{"ndim": 3, "shape": (None, None, n_in),
                   "type": "input_source", "name": "x"}]

    for param_init_dict in params:
      assert isinstance(param_init_dict, dict)
      assert "name" in param_init_dict
      assert "shape" in param_init_dict
      param_init_dict = param_init_dict.copy()
      param_init_dict["name"] += "_%s" % self.name
      p = self._create_eval_params(**param_init_dict)
      assert isinstance(p, theano.compile.SharedVariable)
      p = self.add_param(p)
      p_shape = p.get_value(borrow=True, return_internal_type=True).shape
      p_ndim = len(p_shape)
      args += [p]
      args_info += [{"ndim": p_ndim, "shape": tuple(p_shape),
                     "type": "input_param", "name": param_init_dict["name"]}]

    args += [self.index]
    args_info += [{"ndim": 2, "shape": (None, None), "gradient": "disconnected", "type": "input_index"}]

    from returnn.theano.util import make_var_tuple
    args = make_var_tuple(native_class_cls.map_layer_inputs_to_op(*args))
    outputs = make_var_tuple(op(*args))
    self.output = native_class_cls.map_layer_output_from_op(*outputs)

    def print_fn(op, x):
      import numpy
      first = x[(0,) * x.ndim]
      stats = (first, x.shape, numpy.min(x), numpy.max(x), numpy.mean(x), numpy.std(x),
               numpy.isinf(x).any(), numpy.isnan(x).any())
      print(op.message, "first/shape/min/max/mean/std/any-inf/any-nan:", stats)
    #self.output = theano.printing.Print("native_out", global_fn=print_fn)(self.output)


class DumpLayer(_NoOpLayer):
  layer_class = "dump"
  # You can set this var to a dict to get the content in there
  # instead of being written to disc.
  global_debug_container = None

  def __init__(self, filename, with_grad=True, n_out=None, **kwargs):
    super(DumpLayer, self).__init__(**kwargs)
    self.output, n_in = concat_sources(self.sources, masks=self.masks, mass=self.mass)
    if n_out: assert n_out == n_in
    n_out = n_in
    self.set_attr("n_out", n_out)
    self.set_attr("filename", filename)
    self.set_attr("with_grad", with_grad)

    if self.train_flag:
      self.output = DumpOp(filename, container=self.global_debug_container, with_grad=with_grad)(self.output)
      self.index = DumpOp(filename + ".index", container=self.global_debug_container, with_grad=False)(self.index)


class AlignmentLayer(ForwardLayer):
  layer_class = "align"

  def __init__(self, direction='inv', tdps=None, nstates=1, nstep=1, min_skip=0, max_skip=30, search='align', train_skips=False,
               base=None, output_attention=False, output_z=False, reduce_output=True, blank=False, **kwargs):
    assert direction == 'inv'
    target = kwargs['target']
    if tdps is None:
      tdps = [0.]
    if len(tdps) - 2 < max_skip:
      tdps += [tdps[-1]] * (max_skip - len(tdps) + 2)
    else:
      max_skip = len(tdps) - 2
    if base is None:
      base = []
    kwargs['n_out'] = kwargs['y_in'][target].n_out + blank
    n_cls = kwargs['y_in'][target].n_out
    super(AlignmentLayer, self).__init__(**kwargs)
    if base:
      self.params = base[0].params
      self.W_in = base[0].W_in
      self.b = base[0].b
      self.z = self.get_linear_forward_output()
    self.set_attr('search', search)
    n_out = sum([s.attrs['n_out'] for s in self.sources])
    x_in = T.concatenate([s.output for s in self.sources],axis=2)
    self.set_attr('n_out', n_out)
    if tdps is None:
      tdps = [0.]
    if len(tdps) - 2 < max_skip:
      tdps += [tdps[-1]] * (max_skip - len(tdps) + 2)
    for i in range(len(tdps)):
      if i % nstep != 0:
        tdps[i] = 1e30
    if min_skip > 0:
      tdps[:min_skip] = [1e30] * min_skip
    self.cost_val = T.constant(0)
    self.error_val = T.constant(0)
    if self.eval_flag:
      if search == 'time':
        self.index = self.sources[0].index
        self.output = x_in
        self.y_out = self.y_in[target].reshape(self.index.shape)
        return
    else:
      if search == 'time':
        search = 'align'
    z_in = self.z.reshape((self.z.shape[0] * self.z.shape[1], self.z.shape[2]))
    p_in = T.nnet.softmax(z_in).reshape(self.z.shape)
    self.p_y_given_x = p_in
    y_in = self.y_in[target].reshape(self.index.shape)
    if train_skips:
      W_skip = self.add_param(self.create_forward_weights(n_out, len(tdps), name="W_skip_%s" % self.name))
      b_skip = self.add_param(self.create_bias(len(tdps), name='b_skip_%s' % self.name))
      t_in = T.dot(x_in,W_skip) + b_skip
      q_in = T.nnet.softmax(t_in.reshape((t_in.shape[0]*t_in.shape[1],t_in.shape[2]))).reshape(t_in.shape)
    if search == 'linear':
      max_length_y = self.z.shape[0] / y_in.shape[0] #+ T.mod(self.z.shape[0], y_in.shape[0])
      y_out = y_in.flatten() #reshape((y_in.shape[0]*y_in.shape[1]))
      y_out = y_out.repeat(max_length_y,axis=0).reshape((y_in.shape[0] * max_length_y,y_in.shape[1]))
      y_out = T.concatenate([y_out,y_out[-1:].repeat(T.mod(self.z.shape[0], y_in.shape[0]) + 1,axis=0)])[:-1]
      self.y_out = y_out
      rindex = self.index.flatten() #reshape(self.index.shape[0]*self.index.shape[1])
      rindex = rindex.repeat(max_length_y,axis=0).reshape((self.index.shape[0] * max_length_y,y_in.shape[1]))
      rindex = T.concatenate([rindex,rindex[-1:].repeat(T.mod(self.z.shape[0], y_in.shape[0]) + 1,axis=0)])[:-1]
      norm = T.sum(self.index,dtype='float32')/T.sum(rindex,dtype='float32')
      self.index = rindex
      self.output = x_in
      idx = (self.index.flatten() > 0).nonzero()
      nll, _ = T.nnet.crossentropy_softmax_1hot(x=z_in[idx], y_idx=y_out.flatten()[idx])
      self.cost_val = norm * T.sum(nll)
      self.error_val = norm * T.sum(T.neq(T.argmax(z_in[idx], axis=1), y_out.flatten()[idx]))
      return

    if self.train_flag or search == 'align':
      y_out, att, rindex = InvAlignOp(tdps, nstates)(self.sources[0].index, self.index, -T.log(p_in), y_in)
      max_length_y = y_out.shape[0]
      norm = numpy.float32(1./nstates)
      ratt = att
      index = theano.gradient.disconnected_grad(rindex)
      self.y_out = y_out
    elif search == 'search':
      from returnn.theano.ops.inv_align import InvBacktrackOp
      y, att, idx = InvBacktrackOp(tdps, nstates, 0)(self.sources[0].index, -T.log(p_in), -T.log(q_in))
      if not self.eval_flag:
        y_out, ratt, rindex = InvAlignOp(tdps, nstates)(self.sources[0].index, self.index, -T.log(p_in), y_in)
      norm = numpy.float32(1./nstates)
      max_length_y = T.maximum(T.max(idx.sum(axis=0, acc_dtype='int32')), y_in.shape[0])
      index = idx[:max_length_y]
      att = att[:max_length_y]
      y_pad = T.zeros((max_length_y - y_in.shape[0] + 1, y_in.shape[1]), 'int32')
      self.y_out = T.concatenate([y_in, y_pad], axis=0)[:-1]
    elif search == 'decode':
      from returnn.theano.ops.inv_align import InvDecodeOp
      y, att, idx = InvDecodeOp(tdps, nstates, 0)(self.sources[0].index, -T.log(p_in))
      norm = T.sum(self.index, dtype='float32') / T.sum(idx, dtype='float32')
      max_length_y = T.max(idx.sum(axis=0, acc_dtype='int32'))
      index = idx[:max_length_y]
      att = att[:max_length_y]
      y_pad = T.zeros((max_length_y - y_in.shape[0] + 1, y_in.shape[1]), 'int32')
      self.y_out = T.concatenate([y_in, y_pad], axis=0)[:-1]
    else:
      assert search == 'time'

    self.att = att
    if output_attention:
      self.output = T.cast(att, 'float32').dimshuffle(0,1,'x')
      self.output = T.concatenate([self.output,T.zeros_like(self.output[-1:])],axis=0)
      self.output = T.set_subtensor(self.output[T.sum(index,axis=0,dtype='int32'),T.arange(self.output.shape[1])], numpy.int32(-1))
      self.attrs['n_out'] = 1
      return
    else:
      if reduce_output:
        if output_z:
          z_out = self.z.dimshuffle(1, 0, 2).reshape((self.z.shape[0] * self.z.shape[1], self.z.shape[2]))[att.flatten()]
          self.output = z_out.reshape((max_length_y, self.z.shape[1], z_out.shape[1]))
          self.attrs['n_out'] = n_cls
        else:
          x_out = x_in.dimshuffle(1, 0, 2).reshape((x_in.shape[0] * x_in.shape[1], x_in.shape[2]))[att.flatten()]
          self.output = x_out.reshape((max_length_y, self.z.shape[1], x_out.shape[1]))
        self.p_y_given_x = self.output
        self.index = index
      else:
        self.output = self.z if output_z else x_in
        self.p_y_given_x = p_in
        if output_z:
          self.attrs['n_out'] = n_cls
        self.index = self.sources[0].index

    #if self.attrs['search'] == 'time' or self.eval_flag:
    #  return

    if search in ['align', 'decode', 'search']:
      idx = (rindex.flatten() > 0).nonzero()
      if train_skips:
        t_out = t_in.dimshuffle(1, 0, 2).reshape((self.z.shape[0] * self.z.shape[1], t_in.shape[2]))[ratt.flatten()]
        q_out = T.concatenate([T.zeros_like(ratt[:1]),ratt[1:] - ratt[:-1]],axis=0).flatten()
        nll, _ = T.nnet.crossentropy_softmax_1hot(x=t_out[idx], y_idx=q_out[idx])
        self.cost_val = norm * T.sum(nll)
        self.error_val = norm * T.sum(T.neq(T.argmax(t_out[idx], axis=1), q_out[idx]))
      else:
        z_out = self.z.dimshuffle(1, 0, 2).reshape((self.z.shape[0] * self.z.shape[1], self.z.shape[2]))[att.flatten()]
        y_out = self.y_out.flatten()
        nll, _ = T.nnet.crossentropy_softmax_1hot(x=z_out[idx], y_idx=y_out[idx])
        self.cost_val = T.sum(nll)
        self.error_val = norm * T.sum(T.neq(T.argmax(z_out[idx], axis=1), y_out[idx]))
        if blank:
          jdx = self.sources[0].index.flatten()
          norm = self.index.sum(dtype='float32') / self.sources[0].index.sum(dtype='float32')
          z_tot = self.z.reshape((self.z.shape[0]*self.z.shape[1],self.z.shape[2]))[jdx]
          bnll, _ = T.nnet.crossentropy_softmax_1hot(x=z_tot,
                                                     y_idx=T.zeros(z_tot.shape[:1],'int32') + numpy.int32(n_cls))
          rnll, _ = T.nnet.crossentropy_softmax_1hot(x=z_out,
                                                     y_idx=T.zeros(z_out.shape[:1], 'int32') + numpy.int32(n_cls))
          self.cost_val += T.sum(bnll) - T.sum(rnll)
        self.cost_val *= norm
    elif search == 'search':
      z_out = self.z.dimshuffle(1, 0, 2).reshape((self.z.shape[0] * self.z.shape[1], self.z.shape[2]))[ratt.flatten()]
      if train_skips:
        y_out = self.y_out * len(tdps)
        y_out = T.inc_subtensor(y_out[1:], att[1:] - att[:-1]).flatten()
      else:
        y_out = self.y_out
      idx = (rindex.flatten() > 0).nonzero()
      nll, _ = T.nnet.crossentropy_softmax_1hot(x=z_out[idx], y_idx=y_out[idx])
      self.cost_val = norm * T.sum(nll)
      self.error_val = norm * T.sum(T.neq(T.argmax(z_out[idx], axis=1), y_out[idx]))
    elif search == 'ctc':
      from returnn.theano.ops.best_path_decoder import BestPathDecodeOp
      from theano.tensor.extra_ops import cpu_contiguous
      return T.sum(BestPathDecodeOp()(p_in, cpu_contiguous(self.y.dimshuffle(1, 0)), self.index_for_ctc()))

  def cost(self):
    return self.cost_val, None

  def errors(self):
    return self.error_val


class CAlignmentLayer(ForwardLayer):
  layer_class = "calign"

  def __init__(self, direction='inv', tdps=None, nstates=1, nstep=1, min_skip=1, max_skip=30, search='align', train_skips=False, train_emission=False, clip_emission=1.0, train_attention=False,
               compute_priors=False,
               base=None, coverage=0, output_z=False, reduce_output=True, blank=None, nil = None, focus='last', mode='viterbi', **kwargs):
    assert direction == 'inv'
    target = kwargs['target'] if 'target' in kwargs else 'classes'
    if base is None:
      base = []
    kwargs['n_out'] = kwargs['y_in'][target].n_out #+ blank
    self.n_cls = kwargs['y_in'][target].n_out
    super(CAlignmentLayer, self).__init__(**kwargs)
    self.index = self.network.j[target]
    self.cost_scale_val = numpy.float32(1)
    if base:
      if base[0].layer_class == 'calign':
        self.params = {}
        self.W_in = base[0].W_in
        self.b = base[0].b
        self.z = self.get_linear_forward_output()
      elif base[0].layer_class == 'disc':
        self.cost_scale_val = (base[0].gen_error_val / T.sum(base[0].index,dtype='float32')) * (base[0].real_error_val / T.sum(base[0].index,dtype='float32'))
    self.set_attr('search', search)
    n_out = sum([s.attrs['n_out'] for s in self.sources])
    x_in = T.concatenate([s.output for s in self.sources],axis=2)
    self.x_in = x_in
    self.set_attr('n_out', n_out)
    self.set_attr('max_skip', max_skip)
    self.set_attr('nstates', nstates)
    if tdps is None:
      tdps = [0.]
    if len(tdps) - 2 < max_skip:
      tdps += [tdps[-1]] * (max_skip - len(tdps) + 2)
    for i in range(len(tdps)):
      if i % nstep != 0:
        tdps[i] = 1e30
    if min_skip > 0:
      tdps[:min_skip] = [1e30] * min_skip
    if nil is None:
      nil = -1
    elif nil < 0:
      nil = self.n_cls + nil
    self.cost_val = T.constant(0)
    self.error_val = T.constant(0)
    if self.eval_flag:
      if search == 'time':
        self.index = self.sources[0].index
        self.output = x_in
        self.y_out = self.y_in[target].reshape(self.index.shape)
        return
    else:
      if search == 'time':
        search = 'align'
    z_in = self.z.reshape((self.z.shape[0] * self.z.shape[1], self.z.shape[2]))
    p_in = T.nnet.softmax(z_in).reshape(self.z.shape)
    y_in = self.y_in[target].reshape(self.index.shape)
    from theano.tensor.extra_ops import cpu_contiguous
    from returnn.theano.ops.inv import InvOp
    self.attention = InvOp(min_skip, max_skip, nstates, focus, nil, coverage, mode)(-T.log(p_in), cpu_contiguous(y_in),
                                                                                    T.sum(self.sources[0].index, axis=0,
                                                                                          dtype='int32'),
                                                                                    T.sum(self.index, axis=0,
                                                                                          dtype='int32'))
    self.attention = theano.gradient.disconnected_grad(self.attention)  # NBT
    self.y_out = y_in.dimshuffle(0, 'x', 1).repeat(nstates, axis=1).reshape(
      (self.index.shape[0] * nstates, self.index.shape[1]))
    rindex = self.index.dimshuffle(0, 'x', 1).repeat(nstates, axis=1).reshape(
      (self.index.shape[0] * nstates, self.index.shape[1]))
    index = theano.gradient.disconnected_grad(rindex)
    norm = numpy.float32(1. / nstates)

    x_out = T.batched_dot(x_in.dimshuffle(1, 2, 0), self.attention.dimshuffle(1, 2, 0)).dimshuffle(2, 0, 1)  # NBD
    #z_out = T.batched_dot(self.z.dimshuffle(1, 2, 0), self.attention.dimshuffle(1, 2, 0)).dimshuffle(2, 0, 1) # NBC
    z_out = self.b + T.dot(x_out,T.concatenate(self.W_in,axis=0))

    if train_emission:
      W_skip = self.add_param(self.create_forward_weights(n_out, 2, name="W_skip_%s" % self.name))
      b_skip = self.add_param(self.create_bias(2, name='b_skip_%s' % self.name))
      q_in = T.dot(x_in, W_skip) + b_skip
      q_in = T.nnet.softmax(q_in.reshape((q_in.shape[0] * q_in.shape[1], q_in.shape[2]))).reshape(q_in.shape)

    if train_attention:
      W_att = self.add_param(self.create_forward_weights(n_out, 1, name="W_att_%s" % self.name))
      b_att = self.add_param(self.create_bias(1, name='b_att_%s' % self.name))
      q_in = T.dot(x_in, W_att) + b_att
      q_in = T.nnet.sigmoid(q_in)

    if reduce_output:
      self.output = z_out if output_z else x_out
      self.index = index
      self.p_y_given_x = T.nnet.softmax(z_out.reshape((z_out.shape[0]*z_out.shape[1],z_out.shape[2]))).reshape(z_out.shape)
      if train_emission: #  and not self.train_flag:
        def encode(x_t,q_t,x_p,q_p,i_p):
          q_c = q_t + q_p
          write_flag = T.ge(q_c, numpy.float32(clip_emission))
          q = T.switch(write_flag, q_c - numpy.float32(clip_emission), q_c)
          x = x_t * q_t.dimshuffle(0,'x').repeat(x_t.shape[1],axis=1)
          x += T.switch(write_flag.dimshuffle(0,'x').repeat(x_t.shape[1],axis=1), T.zeros_like(x_p), x_p)
          #x = x_t #* T.cast(write_flag.dimshuffle(0, 'x').repeat(x_t.shape[1], axis=1),'float32')
          return x, q, T.cast(write_flag,'float32')

        out, _ = theano.scan(encode, sequences=[x_in,q_in[:,:,1]],
                             outputs_info=[T.zeros_like(x_in[0]), T.zeros((q_in.shape[1],),'float32'), T.zeros((q_in.shape[1],),'float32')])
        x, q, i = out[:3]
        def select(x_b,i_b,L):
          idx = (i_b > 0).nonzero()
          len = T.cast(T.sum(i_b),'int32')
          buf = T.zeros((L,x_b.shape[1]),'float32')
          buf = T.set_subtensor(buf[:len],x_b[idx])
          ind = T.zeros((L, ), 'float32')
          ind = T.set_subtensor(ind[:len], numpy.float32(1))
          return buf, ind

        out, _ = theano.map(select, sequences=[x.dimshuffle(1,0,2), i.dimshuffle(1,0)],
                            non_sequences=[T.max(T.sum(i,axis=0,dtype='int32'))+numpy.int32(1)])
        self.output = out[0].dimshuffle(1,0,2)[:-1]
        self.index = T.cast(out[1].dimshuffle(1,0),'int8')[:-1]
    else:
      self.output = self.z if output_z else x_in
      self.index = self.sources[0].index
      self.p_y_given_x = p_in

    self.reduced_index = index

    if output_z:
      self.attrs['n_out'] = self.n_cls

    idx = (rindex.flatten() > 0).nonzero()
    if train_skips:
      y_out = T.dot(self.attention, T.arange(x_in.shape[0],dtype='float32')) # NB
      y_out = T.concatenate([T.zeros_like(y_out[:1]), y_out],axis=0) # (N+1)B
      y_out = T.cast(T.round(y_out[1:] - y_out[:-1]) * T.cast(rindex,'float32'),'int32') # NB
      #y_out = print_to_file('out',y_out)

      W_skip = self.add_param(self.create_forward_weights(n_out, max_skip, name="W_skip_%s" % self.name))
      b_skip = self.add_param(self.create_bias(max_skip, name='b_skip_%s' % self.name))
      z_out = T.dot(x_out, W_skip) + b_skip
      self.q_in = T.nnet.softmax(self.z.reshape((self.z.shape[0] * self.z.shape[1], self.z.shape[2]))).reshape(self.z.shape)

    elif train_emission:
      idx = (self.sources[0].index.flatten() > 0).nonzero()
      norm = T.sum(self.network.j[target],dtype='float32') / T.sum(self.sources[0].index,dtype='float32')
      #W_skip = self.add_param(self.create_forward_weights(n_out, 2, name="W_skip_%s" % self.name))
      #b_skip = self.add_param(self.create_bias(2, name='b_skip_%s' % self.name))
      y_out = T.sum(self.attention,axis=0).dimshuffle(1,0)

      #y_out = y_out / y_out.sum(axis=0,keepdims=True)
      y_out = y_out.flatten().dimshuffle(0, 'x') # (TB)
      y_out = T.concatenate([numpy.float32(1) - y_out, y_out], axis=1) # (TB)2
      #z_out = T.dot(x_in, W_skip) + b_skip # TB2
      #z_out = T.nnet.softmax(z_out.reshape((z_out.shape[0] * z_out.shape[1], z_out.shape[2]))).reshape(z_out.shape)
      #self.output *= z_out[:, :, 1].dimshuffle(0, 1, 'x').repeat(self.output.shape[2], axis=2)
      z_out = q_in.reshape((q_in.shape[0] * q_in.shape[1], q_in.shape[2])) # (TB)2
      self.cost_val = norm * -T.sum(y_out[idx] * T.log(z_out[idx]))
      self.error_val = norm * T.sum(T.ge(T.sqr(z_out[idx,1]-y_out[idx,1]),numpy.float32(1./self.n_cls)))
      self.p_y_given_x = q_in[:,:,1:]
      self.attrs['n_cls'] = 1
      return
    elif train_attention:
      idx = (self.sources[0].index.flatten() > 0).nonzero()
      y_out = T.round(T.sum(self.attention,axis=0).dimshuffle(1,0)).cast('int32') # TB
      y_out = (y_out.flatten() > 0).nonzero()
      self.cost_val = -T.log(q_in.flatten()[y_out[idx]]).sum()
      self.error_val = T.sum(T.neq(T.round(q_in).flatten().cast('int32')[idx], y_out[idx]))
      return
    else:
      y_out = self.y_out

    y_out = y_out.flatten()
    z_out = z_out.reshape((z_out.shape[0] * z_out.shape[1], z_out.shape[2]))
    nll, _ = T.nnet.crossentropy_softmax_1hot(x=z_out[idx], y_idx=y_out[idx])
    self.cost_val = norm * T.sum(nll)
    self.error_val = norm * T.sum(T.neq(T.argmax(z_out[idx], axis=1), y_out[idx]))

    if compute_priors:
      self.set_attr('compute_priors', compute_priors)
      custom = T.mean(theano.tensor.extra_ops.to_one_hot(y_out[idx], self.n_cls, 'float32'), axis=0)
      custom_init = numpy.ones((self.n_cls,), 'float32') / numpy.float32(self.n_cls)
      self.priors = self.add_param(theano.shared(custom_init, 'priors'), 'priors',
                                   custom_update=custom,
                                   custom_update_normalized=True,
                                   custom_update_exp_average=False)

  def cost(self):
    return self.cost_val * self.cost_scale_val, None

  def errors(self):
    return self.error_val


class InvBacktrackLayer(ForwardLayer):
  layer_class = "ibt"

  def __init__(self, direction='inv', tdps=None, nstates=1, nstep=1, min_skip=1, max_skip=30, search='align', train_skips=False, train_emission=False, clip_emission=1.0,
               base=None, coverage=0, output_z=False, reduce_output=True, blank=None, nil = None, focus='last', mode='viterbi', **kwargs):
    assert direction == 'inv'
    target = kwargs['target'] if 'target' in kwargs else 'classes'
    if base is None:
      base = []
    kwargs['n_out'] = kwargs['y_in'][target].n_out #+ blank
    self.n_cls = kwargs['y_in'][target].n_out
    super(InvBacktrackLayer, self).__init__(**kwargs)
    self.index = self.network.j[target]
    self.cost_scale_val = numpy.float32(1)
    if base:
      if base[0].layer_class == 'calign':
        self.params = {}
        self.W_in = base[0].W_in
        self.b = base[0].b
        self.z = self.get_linear_forward_output()
      elif base[0].layer_class == 'disc':
        self.cost_scale_val = (base[0].gen_error_val / T.sum(base[0].index,dtype='float32')) * (base[0].real_error_val / T.sum(base[0].index,dtype='float32'))
    self.set_attr('search', search)
    n_out = sum([s.attrs['n_out'] for s in self.sources])
    x_in = T.concatenate([s.output for s in self.sources],axis=2)
    self.x_in = x_in
    self.set_attr('n_out', n_out)
    self.set_attr('max_skip', max_skip)
    if tdps is None:
      tdps = [0.]
    if len(tdps) - 2 < max_skip:
      tdps += [tdps[-1]] * (max_skip - len(tdps) + 2)
    for i in range(len(tdps)):
      if i % nstep != 0:
        tdps[i] = 1e30
    if min_skip > 0:
      tdps[:min_skip] = [1e30] * min_skip
    if nil is None:
      nil = -1
    elif nil < 0:
      nil = self.n_cls + nil
    self.cost_val = T.constant(0)
    self.error_val = T.constant(0)
    if self.eval_flag:
      if search == 'time':
        self.index = self.sources[0].index
        self.output = x_in
        self.y_out = self.y_in[target].reshape(self.index.shape)
        return
    else:
      if search == 'time':
        search = 'align'
    z_in = self.z.reshape((self.z.shape[0] * self.z.shape[1], self.z.shape[2]))
    p_in = T.nnet.softmax(z_in).reshape(self.z.shape)
    y_in = self.y_in[target].reshape(self.index.shape)
    from theano.tensor.extra_ops import cpu_contiguous
    from returnn.theano.ops.inv import InvOpBackTrace
    self.attention, self.backtrace = InvOpBackTrace(min_skip, max_skip, nstates,
                                                    focus, nil, coverage, mode)(-T.log(p_in), cpu_contiguous(y_in),
                                                                                 T.sum(self.sources[0].index, axis=0,
                                                                                       dtype='int32'),
                                                                                 T.sum(self.index, axis=0,
                                                                                       dtype='int32'))
    self.attention = theano.gradient.disconnected_grad(self.attention)  # NBT
    self.backtrace = theano.gradient.disconnected_grad(self.backtrace)  # NBT
    self.y_out = y_in.dimshuffle(0, 'x', 1).repeat(nstates, axis=1).reshape(
      (self.index.shape[0] * nstates, self.index.shape[1]))
    rindex = self.index.dimshuffle(0, 'x', 1).repeat(nstates, axis=1).reshape(
      (self.index.shape[0] * nstates, self.index.shape[1]))
    index = theano.gradient.disconnected_grad(rindex)
    norm = numpy.float32(1. / nstates)

    x_out = T.batched_dot(x_in.dimshuffle(1, 2, 0), self.attention.dimshuffle(1, 2, 0)).dimshuffle(2, 0, 1)  # NBD
    #z_out = T.batched_dot(self.z.dimshuffle(1, 2, 0), self.attention.dimshuffle(1, 2, 0)).dimshuffle(2, 0, 1) # NBC
    z_out = self.b + T.dot(x_out,T.concatenate(self.W_in,axis=0))

    if train_emission:
      W_skip = self.add_param(self.create_forward_weights(n_out, 2, name="W_skip_%s" % self.name))
      b_skip = self.add_param(self.create_bias(2, name='b_skip_%s' % self.name))
      q_in = T.dot(x_in, W_skip) + b_skip
      q_in = T.nnet.softmax(q_in.reshape((q_in.shape[0] * q_in.shape[1], q_in.shape[2]))).reshape(q_in.shape)


    if reduce_output:
      self.output = z_out if output_z else x_out
      self.index = index
      self.p_y_given_x = T.nnet.softmax(z_out.reshape((z_out.shape[0]*z_out.shape[1],z_out.shape[2]))).reshape(z_out.shape)
      if train_emission: #  and not self.train_flag:
        def encode(x_t,q_t,x_p,q_p,i_p):
          q_c = q_t + q_p
          write_flag = T.ge(q_c, numpy.float32(clip_emission))
          q = T.switch(write_flag, q_c - numpy.float32(clip_emission), q_c)
          x = x_t * q_t.dimshuffle(0,'x').repeat(x_t.shape[1],axis=1)
          x += T.switch(write_flag.dimshuffle(0,'x').repeat(x_t.shape[1],axis=1), T.zeros_like(x_p), x_p)
          #x = x_t #* T.cast(write_flag.dimshuffle(0, 'x').repeat(x_t.shape[1], axis=1),'float32')
          return x, q, T.cast(write_flag,'float32')

        out, _ = theano.scan(encode, sequences=[x_in,q_in[:,:,1]],
                             outputs_info=[T.zeros_like(x_in[0]), T.zeros((q_in.shape[1],),'float32'), T.zeros((q_in.shape[1],),'float32')])
        x, q, i = out[:3]
        def select(x_b,i_b,L):
          idx = (i_b > 0).nonzero()
          len = T.cast(T.sum(i_b),'int32')
          buf = T.zeros((L,x_b.shape[1]),'float32')
          buf = T.set_subtensor(buf[:len],x_b[idx])
          ind = T.zeros((L, ), 'float32')
          ind = T.set_subtensor(ind[:len], numpy.float32(1))
          return buf, ind

        out, _ = theano.map(select, sequences=[x.dimshuffle(1,0,2), i.dimshuffle(1,0)],
                            non_sequences=[T.max(T.sum(i,axis=0,dtype='int32'))+numpy.int32(1)])
        self.output = out[0].dimshuffle(1,0,2)[:-1]
        self.index = T.cast(out[1].dimshuffle(1,0),'int8')[:-1]
    else:
      self.output = self.z if output_z else x_in
      self.index = self.sources[0].index
      self.p_y_given_x = p_in

    self.reduced_index = index

    if output_z:
      self.attrs['n_out'] = self.n_cls

    idx = (rindex.flatten() > 0).nonzero()
    if train_skips:
      y_out = T.dot(self.attention, T.arange(x_in.shape[0],dtype='float32')) # NB
      y_out = T.concatenate([T.zeros_like(y_out[:1]), y_out],axis=0) # (N+1)B
      y_out = T.cast(T.round(y_out[1:] - y_out[:-1]) * T.cast(self.index,'float32'),'int32') # NB

      W_skip = self.add_param(self.create_forward_weights(n_out, max_skip, name="W_skip_%s" % self.name))
      b_skip = self.add_param(self.create_bias(max_skip, name='b_skip_%s' % self.name))
      z_out = T.dot(x_out, W_skip) + b_skip
      self.q_in = T.nnet.softmax(self.z.reshape((self.z.shape[0] * self.z.shape[1], self.z.shape[2]))).reshape(self.z.shape)
    elif train_emission:
      idx = (self.sources[0].index.flatten() > 0).nonzero()
      norm = T.sum(self.network.j[target],dtype='float32') / T.sum(self.sources[0].index,dtype='float32')
      #W_skip = self.add_param(self.create_forward_weights(n_out, 2, name="W_skip_%s" % self.name))
      #b_skip = self.add_param(self.create_bias(2, name='b_skip_%s' % self.name))
      y_out = T.sum(self.attention,axis=0).dimshuffle(1,0)

      #y_out = y_out / y_out.sum(axis=0,keepdims=True)
      y_out = y_out.flatten().dimshuffle(0, 'x') # (TB)
      y_out = T.concatenate([numpy.float32(1) - y_out, y_out], axis=1) # (TB)2
      #z_out = T.dot(x_in, W_skip) + b_skip # TB2
      #z_out = T.nnet.softmax(z_out.reshape((z_out.shape[0] * z_out.shape[1], z_out.shape[2]))).reshape(z_out.shape)
      #self.output *= z_out[:, :, 1].dimshuffle(0, 1, 'x').repeat(self.output.shape[2], axis=2)
      z_out = q_in.reshape((q_in.shape[0] * q_in.shape[1], q_in.shape[2])) # (TB)2
      self.cost_val = norm * -T.sum(y_out[idx] * T.log(z_out[idx]))
      self.error_val = norm * T.sum(T.ge(T.sqr(z_out[idx,1]-y_out[idx,1]),numpy.float32(1./self.n_cls)))
      return
    else:
      y_out = self.y_out

    y_out = y_out.flatten()
    z_out = z_out.reshape((z_out.shape[0] * z_out.shape[1], z_out.shape[2]))
    nll, _ = T.nnet.crossentropy_softmax_1hot(x=z_out[idx], y_idx=y_out[idx])
    self.cost_val = norm * T.sum(nll)
    self.error_val = norm * T.sum(T.neq(T.argmax(z_out[idx], axis=1), y_out[idx]))

  def cost(self):
    return self.cost_val * self.cost_scale_val, None

  def errors(self):
    return self.error_val

class FAlignmentLayer(ForwardLayer):
  layer_class = "falign"

  def make_tdps(self, tdps, max_skip):
    if tdps is None:
      tdps = [1e10, 0., 3.]
    if len(tdps) - 2 < max_skip:
      tdps += [tdps[-1]] * (max_skip - len(tdps) + 2)
    return tdps


  def __init__(self, direction='inv', tdps=None, nstates=1, nstep=1, min_skip=1, max_skip=10, search='align', train_skips=False,
               base=None, output_attention=False, output_z=False, reduce_output=True, blank=False, focus='last', mode='viterbi', **kwargs):
    assert direction == 'inv'
    target = kwargs['target']
    tdps = self.make_tdps(tdps, max_skip)
    max_skip = len(tdps) - 2
    kwargs['n_out'] = kwargs['y_in'][target].n_out + blank
    n_cls = kwargs['y_in'][target].n_out
    super(FAlignmentLayer, self).__init__(**kwargs)
    if base is not None:
      self.params = base[0].params
      self.W_in = base[0].W_in
      self.b = base[0].b
      self.z = self.get_linear_forward_output()
    self.set_attr('search', search)
    n_out = sum([s.attrs['n_out'] for s in self.sources])
    x_in = T.concatenate([s.output for s in self.sources],axis=2)

    self.cost_val = T.constant(0)
    self.error_val = T.constant(0)

    z_in = self.z.reshape((self.z.shape[0] * self.z.shape[1], self.z.shape[2]))
    p_in = T.nnet.softmax(z_in).reshape(self.z.shape)
    self.p_y_given_x = p_in
    y_in = self.y_in[target].reshape(self.index.shape)
    from theano.tensor.extra_ops import cpu_contiguous
    from returnn.theano.ops.inv import InvAlignOp as InvAlign
    alpha = InvAlign(min_skip, max_skip, nstates, focus)(-T.log(self.p_y_given_x), cpu_contiguous(y_in),
                                                          T.sum(self.sources[0].index, axis=0, dtype='int32'),
                                                          T.sum(self.index, axis=0, dtype='int32'))
    alpha = theano.gradient.disconnected_grad(alpha) # (NS)BT
    self.y_out = y_in.dimshuffle(0, 'x', 1).repeat(nstates, axis=1).reshape(
      (self.index.shape[0] * nstates, self.index.shape[1]))
    self.index = self.index.dimshuffle(0, 'x', 1).repeat(nstates, axis=1).reshape(
      (self.index.shape[0] * nstates, self.index.shape[1]))
    norm = numpy.float32(1. / nstates)

    idx = (self.index.flatten() > 0).nonzero()
    x_out, _ = theano.map(
      lambda x, a: T.sum(
        x.dimshuffle(0, 1, 'x').repeat(a.shape[1], axis=2) * a.dimshuffle(0, 'x', 1).repeat(x.shape[1], axis=1),
        axis=0),
      sequences=[x_in.dimshuffle(1, 0, 2), alpha.dimshuffle(1, 2, 0)])  # BDN
    x_out = x_out.dimshuffle(2, 0, 1)  # NBD
    #self.output = alpha
    self.set_attr('n_out', n_out)
    self.output = alpha
    self.output = x_out


class FStdAlignmentLayer(ForwardLayer):
  layer_class = "fstdalign"

  def __init__(self, direction='inv', base=None, nstates=3, skip_tdp=0, **kwargs):
    assert direction == 'inv'
    target = kwargs['target']
    if base is None:
      base = []
    kwargs['n_out'] = kwargs['y_in'][target].n_out
    n_cls = kwargs['y_in'][target].n_out
    super(FStdAlignmentLayer, self).__init__(**kwargs)
    if base:
      self.params = base[0].params
      self.W_in = base[0].W_in
      self.b = base[0].b
      self.z = self.get_linear_forward_output()
    n_out = sum([s.attrs['n_out'] for s in self.sources])
    x_in = T.concatenate([s.output for s in self.sources],axis=2)
    self.set_attr('n_out', n_out)
    self.cost_val = T.constant(0)
    self.error_val = T.constant(0)
    z_in = self.z.reshape((self.z.shape[0] * self.z.shape[1], self.z.shape[2]))
    p_in = T.nnet.softmax(z_in).reshape(self.z.shape)
    self.p_y_given_x = p_in
    y_in = self.y_in[target].reshape(self.index.shape)

    from theano.tensor.extra_ops import cpu_contiguous
    from returnn.theano.ops.inv import StdOpFull
    att = StdOpFull(skip_tdp, nstates)(-T.log(self.p_y_given_x), cpu_contiguous(y_in),
                                        T.sum(self.sources[0].index, axis=0, dtype='int32'),
                                        T.sum(self.index, axis=0, dtype='int32'))
    y_out = y_in.dimshuffle(0, 'x', 1).repeat(nstates, axis=1).reshape(
      (self.index.shape[0] * nstates, self.index.shape[1]))
    rindex = self.index.dimshuffle(0, 'x', 1).repeat(nstates, axis=1).reshape(
      (self.index.shape[0] * nstates, self.index.shape[1]))
    norm = numpy.float32(1. / nstates)
    self.y_out = y_out
    idx = (rindex.flatten() > 0).nonzero()
    att = theano.gradient.disconnected_grad(att)  # TB(NS)
    x_out, _ = theano.map(lambda x, a: T.sum(
      x.dimshuffle(0, 1, 'x').repeat(a.shape[1], axis=2) * a.dimshuffle(0, 'x', 1).repeat(x.shape[1], axis=1), axis=0),
                          sequences=[x_in.dimshuffle(1, 0, 2), att.dimshuffle(1, 2, 0)])  # BDN
    x_out = x_out.dimshuffle(2, 0, 1)  # NBD

    z_out = T.dot(x_out, self.W_in[0]) + self.b
    self.output = x_out
    z_out = z_out.reshape((z_out.shape[0] * z_out.shape[1], z_out.shape[2]))
    x_out = x_out.reshape((x_out.shape[0] * x_out.shape[1], x_out.shape[2]))
    nll, _ = T.nnet.crossentropy_softmax_1hot(x=z_out[idx], y_idx=y_out.flatten()[idx])
    self.cost_val = norm * T.sum(nll)
    self.error_val = norm * T.sum(T.neq(T.argmax(z_out[idx], axis=1), y_out.flatten()[idx]))

  def cost(self):
    return self.cost_val, None

  def errors(self):
    return self.error_val



class InvAlignSegmentationLayer(_NoOpLayer):
  layer_class = "invalignsegment"

  def __init__(self, window=0, win=20,base=None, **kwargs):

    super(InvAlignSegmentationLayer, self).__init__(**kwargs)
    if base:
        kwargs['n_out'] = base[0].attrs['n_out']
        self.set_attr('n_out', base[0].attrs['n_out'])
    else:
        kwargs['n_out'] = self.sources[0].y_in[self.sources[0].attrs['target']].n_out
        self.set_attr('n_out', self.sources[0].y_in[self.sources[0].attrs['target']].n_out)
    self.set_attr('window', window)
    self.set_attr('win', win)
    assert len(self.sources) == 1
    source_index = self.sources[0].reduced_index.T.flatten().nonzero()
    y_out = self.sources[0].y_out.T.flatten()[source_index]
    if not self.eval_flag:
      assert self.sources[0].attention is not None
      b = self.sources[0].attention.shape[1]
      t = self.sources[0].attention.shape[2]
      att = self.sources[0].attention.argmax(axis=2)
      att = T.switch(self.sources[0].reduced_index > 0, att + (T.arange(b) * t), att)
      att = att.T
      if window:
          maxlen = T.cast(window/2,'int32')
          att = att.nonzero_values()
          cond = T.arange(-maxlen,maxlen).repeat(att.shape[0]).reshape((2*maxlen,att.shape[0]))
          att_rep = att.repeat(T.cast(2*maxlen,'int32')).reshape((att.shape[0],T.cast(2*maxlen,'int32'))).T #repeat att maxlen times
          finalcond = T.maximum(att_rep + cond,-1)
          finalcond = T.switch(T.lt(finalcond,T.max(att)),finalcond,-1)
      else:
          maxlen = T.concatenate([T.stack(att[0,0]),T.extra_ops.diff(att).flatten().sort()])[-1]
          # concatenate first index in each row, i.e., if att is [[3,5,8],[15,17,20]], make it  [[0,3,5,8],[12,15,17,20]]
          att_with_firstindex = T.concatenate([T.maximum(0,att[:,0].dimshuffle(0,'x')-maxlen),att],axis=1)
          att_sorted = att_with_firstindex.sort() #sort the rows so that [0,3,5,8,0,0,0] becomes [0,0,0,0,3,5,8]
          att_wo_lastcol = T.concatenate([[0],att_sorted[:,:att_sorted.shape[1]-1].flatten().nonzero_values()])
          ind = att_wo_lastcol.shape[0] - T.sum(T.extra_ops.diff(att_with_firstindex).flatten()>0)
          att_wo_lastcol = att_wo_lastcol[ind:]
          att_rep = att_wo_lastcol.repeat(T.cast(maxlen,'int32')).reshape((att_wo_lastcol.shape[0],T.cast(maxlen,'int32'))).T #repeat att maxlen times
          incr = T.arange(1,maxlen+1).repeat(att_wo_lastcol.shape[0]).reshape((T.cast(maxlen,'int32'),att_wo_lastcol.shape[0])) #range of maxlen repeated att(shape) times
          maskarr = T.extra_ops.diff(att_with_firstindex).flatten() #diff array
          maskarr = T.clip(maskarr,0,T.max(maskarr)).nonzero_values() #clip negative values to 0 and remove zeroes
          # repeat maxlen times (this now contains the length of each segment)
          maskarr = maskarr.repeat(T.cast(maxlen,'int32')).reshape((att_wo_lastcol.shape[0],T.cast(maxlen,'int32'))).dimshuffle(1,0)
          #comparing incr and maskarr, you get the value to be added to att_rep at each row and column.
          # If incr > maskarr, then cond has -att_rep-1 so that when it is subtracted from att_rep,
          #we get -1. Later z is concatenated with a row of 0s at the end so that this is retreived when z[-1] is encountered (to simulate [3,4,0,0] for example)
          cond     = T.switch(T.lt(incr, maskarr+1), incr, -att_rep - 1)
          finalcond = att_rep + cond
#      finalcond = theano.printing.Print('%s finalcond'%self.name,attrs=['shape'])(finalcond)
      finalcond = finalcond.sort(axis=0)
      if base:
          z = base[0].output.dimshuffle(1,0,2).reshape((base[0].output.shape[0]*base[0].output.shape[1],base[0].output.shape[2]))
      else:
          z = self.sources[0].z.dimshuffle(1, 0, 2).reshape((self.sources[0].z.shape[0] * self.sources[0].z.shape[1], self.sources[0].z.shape[2]))
      z = T.concatenate([z,T.zeros((1,z.shape[1]))],axis=0)
      result = z[T.cast(finalcond,'int32')]
    else:
      timesteps = self.sources[0].sources[0].output.shape[0]
      batches = self.sources[0].sources[0].output.shape[1]
      self.timesteps = timesteps
      self.batches = batches
      z = self.sources[0].sources[0].output.dimshuffle(1, 0, 2).reshape(
        (self.sources[0].sources[0].output.shape[0] * self.sources[0].sources[0].output.shape[1], self.sources[0].sources[0].output.shape[2]))
      att = T.arange(timesteps).repeat(win).reshape((timesteps, win)) + T.arange(win)
      att = att.T
      att = T.where(att >= timesteps, -timesteps * batches, att)
      fullind = T.tile(att, (1, batches))
      fullind = fullind + (T.arange(batches) * timesteps).repeat(timesteps)
      fullind = T.where(fullind < 0, -1, fullind)
      z = T.concatenate([z, T.zeros((1, z.shape[1]))], axis=0)
      result = z[fullind]
      self.fullind = fullind
    self.z = result
    self.make_output(result)
    y_out = y_out[:result.shape[1]]
    self.y_out = y_out.repeat(result.shape[0]).reshape((result.shape[1],result.shape[0])).T
    self.index = T.ones((self.output.shape[0], self.output.shape[1]), 'int8')

class InvAlignSegmentationLayer2(_NoOpLayer):
  layer_class = "invalignsegment2"

  def __init__(self, window=0, win=20,base=None, join_states=False, **kwargs):

    super(InvAlignSegmentationLayer2, self).__init__(**kwargs)
    if base:
        kwargs['n_out'] = base[0].attrs['n_out']
        self.set_attr('n_out', base[0].attrs['n_out'])
    else:
        kwargs['n_out'] = self.sources[0].attrs['n_out']
        self.set_attr('n_out', self.sources[0].attrs['n_out'])
    self.set_attr('window', window)
    self.set_attr('win', win)
    self.attention = self.sources[0].attention
    self.nstates = self.sources[0].attrs['nstates']
    assert len(self.sources) == 1
    self.inv_att = self.sources[0].attention
    source_index = self.sources[0].reduced_index.T.flatten().nonzero()
    if not self.eval_flag:
#    if self.eval_flag:
      result = T.concatenate([s.output for s in self.sources],axis=-1)
      self.index = self.sources[0].index
    else:
      timesteps = self.sources[0].output.shape[0]
      batches = self.sources[0].output.shape[1]
      self.timesteps = timesteps
      self.batches = batches
      z = self.sources[0].output.dimshuffle(1, 0, 2).reshape(
        (self.sources[0].output.shape[0] * self.sources[0].output.shape[1], self.sources[0].output.shape[2]))
      att = T.arange(timesteps).repeat(win).reshape((timesteps, win)) + T.arange(win)
      att = att.T
      att = T.where(att >= timesteps, -timesteps * batches, att)
      fullind = T.tile(att, (1, batches))
      fullind = fullind + (T.arange(batches) * timesteps).repeat(timesteps)
      fullind = T.where(fullind < 0, -1, fullind)
      z = T.concatenate([z, T.zeros((1, z.shape[1]))], axis=0)
      result = z[fullind]
      self.fullind = fullind
      self.index = T.ones((result.shape[0], result.shape[1]), 'int8')
    self.z = result
    self.make_output(result)

    # code to create y_out for frame-wise classification within the segments
    y_out = self.sources[0].y_out.T
    y_out = T.concatenate([y_out,T.zeros((y_out.shape[0],1))-numpy.int32(1)],axis=1) #adding -1 at the end to account for unused timesteps at the end of the sequence
    diffarr,maxlen = self.find_diff_array(self.sources[0].attention.argmax(axis=2))
    y_outrep = self.set_yout(y_out,diffarr.flatten(),T.cast(maxlen,'int32'))
    y_outrep = y_outrep[:self.output.shape[0]*self.output.shape[1]]
    self.y_out = T.cast(y_outrep.reshape((self.output.shape[1],self.output.shape[0])).T,'int32')

  def find_diff_array(self, att):
    att = att.T
    att = T.concatenate([T.zeros((att.shape[0], numpy.int32(1))), att], axis=1)
    maxlen = T.concatenate([T.stack(att[0, 0]), T.extra_ops.diff(att).flatten().sort()])[-1] + \
             numpy.int32(1)  # maxlen for the segments
    att = att[:, 1:]
    att = T.switch(self.sources[0].reduced_index > 0,
                   att.T + (T.arange(self.sources[0].attention.shape[1]) * self.sources[0].attention.shape[2]),
                   att.T)  # scale the indices of batches according to the batch number
    att = att.T
    last_index = (T.arange(att.shape[0]) + numpy.int32(1)) * self.output.shape[0] - \
                 numpy.int32(1)  # last index for at that denotes the last timestep
    att = T.switch(att > 0, att, last_index.dimshuffle(0, 'x'))
    att_with_firstindex = T.concatenate([T.maximum((T.arange(att.shape[0]) * self.output.shape[0]).dimshuffle(0, 'x'),
                                                   att[:, 0].dimshuffle(0, 'x') - maxlen), att], axis=1)
    att_with_first_and_last_index = T.concatenate([att_with_firstindex, last_index.dimshuffle(0, 'x')], axis=1)
    maxlen = T.extra_ops.diff(att_with_first_and_last_index).flatten().sort()[-1] + \
             numpy.int32(1)  # maxlen for the segments including unused timesteps
    diffarr = T.extra_ops.diff(att_with_first_and_last_index)
    diffarr = T.inc_subtensor(diffarr[:, 0], numpy.int32(1))  # add 1 to differences in the first row to include the first timestep as well
    return diffarr,maxlen

  def set_yout(self,y_out,diff,maxdiff):
    newdiff = T.cast((diff.repeat(maxdiff).reshape((diff.shape[0],maxdiff))+(T.arange(diff.shape[0])*maxdiff).dimshuffle(0,'x')).flatten(),'int32')
    res = T.cast(newdiff-T.arange(newdiff.shape[0])-1,'int32')
    res_1hot = (res>=0).flatten().nonzero()
    y_out_rep = y_out.repeat(maxdiff)
    y_out_rep = y_out_rep[res_1hot]
    return y_out_rep

class ReshapeLayer(StateVector):
  layer_class = "reshape"

  def __init__(self, base=None, **kwargs):
    super(ReshapeLayer, self).__init__(**kwargs)
    assert base is not None
    self.base = base
    if self.eval_flag:
      #get the original timesteps, batches and window parameter
      t = base[0].timesteps
      b = base[0].batches
      w = base[0].attrs['win']
      d = self.attrs['n_out']
      z = T.concatenate([s.output for s in self.sources],axis=-1)
      ze = z.reshape((z.shape[0]*z.shape[1],z.shape[2])) #T*B,D
      fullind = base[0].fullind#full index from invalignsegment layer
      ze = T.concatenate([ze,T.zeros((1,ze.shape[1]))],axis=0)
      for i in range(w):
        fullind = T.set_subtensor(fullind[i],T.roll(fullind[i],i))
        if i>0:
            fullind = T.inc_subtensor(fullind[i],T.where(fullind[i]>0,i*t*b-i,0))
      self.fullind = fullind
      zfinal = ze[fullind.T.flatten()].dimshuffle('x',0,1)
      self.make_output(zfinal)
      self.act = [self.output, T.zeros_like(self.output)]
      self.index = T.ones((1,self.output.shape[1]),'int8')


class SegmentFinalStateLayer(_NoOpLayer):
  layer_class = "segfinal"

  def __init__(self, base=None, use_full_label=False, **kwargs):
    super(SegmentFinalStateLayer, self).__init__(**kwargs)
    kwargs['n_out'] = sum([s.attrs['n_out'] for s in kwargs['sources']])
    self.set_attr('n_out',kwargs['n_out'])
    if not self.eval_flag:
#    if self.eval_flag:
      if hasattr(self.sources[0],'inv_att'):
        inv_att = self.sources[0].inv_att.dimshuffle(2,1,0) #TBN
      else:
        assert base
        if isinstance(base[0],CAlignmentLayer):
          inv_att = base[0].attention.dimshuffle(2,1,0) #TBN
        else:
          inv_att = base[0].inv_att.dimshuffle(2,1,0) #TBN
      z = self.sources[0].output.dimshuffle(1,0,2).reshape((self.sources[0].output.shape[0]*self.sources[0].output.shape[1],self.sources[0].output.shape[2]))
      if not use_full_label:
        max_att = T.max(inv_att,axis=-1).T.flatten().nonzero()
      else:
        max_att = T.max(base[0].sources[0].attention.dimshuffle(2,1,0),axis=-1).T.flatten().nonzero()
      z_aln = z[max_att]
      z_aln = z_aln.dimshuffle('x',0,1)
      self.make_output(z_aln)
      self.index = T.ones((self.output.shape[0],self.output.shape[1]))
    else:
      assert base is not None
      self.base = base
      #get the original timesteps, batches and window parameter
      if isinstance(base[0],CAlignmentLayer):
        self.make_output(self.sources[0].output)
        self.index = self.sources[0].index
      else:
        t = base[0].timesteps
        b = base[0].batches
        w = base[0].attrs['win']
        d = self.attrs['n_out']
        z = T.concatenate([s.output for s in self.sources],axis=-1)
        ze = z.reshape((z.shape[0]*z.shape[1],z.shape[2])) #T*B,D
        fullind = base[0].fullind#full index from invalignsegment layer
        ze = T.concatenate([ze,T.zeros((1,ze.shape[1]))],axis=0)
        for i in range(w):
          fullind = T.set_subtensor(fullind[i],T.roll(fullind[i],i))
          if i>0:
              fullind = T.inc_subtensor(fullind[i],T.where(fullind[i]>0,i*t*b-i,0))
        self.fullind = fullind
        zfinal = ze[fullind.T.flatten()].dimshuffle('x',0,1)
        self.make_output(zfinal)
        self.act = [self.output, T.zeros_like(self.output)]
        self.index = T.ones((1,self.output.shape[1]),'int8')

class ScaleGradientOp(theano.gof.Op):
  view_map = {0: [0]}

  __props__ = ('scale',)

  def __init__(self, scale):
    super(ScaleGradientOp, self).__init__()
    self.scale = numpy.float32(scale)

  def make_node(self, x):
    return theano.gof.graph.Apply(self, [x], [x.type.make_variable()])

  def perform(self, node, inputs, output_storage):
    xin, = inputs
    xout, = output_storage
    xout[0] = xin

  def grad(self, input, output_gradients):
    return [self.scale * output_gradients[0]]


class ScaleGradLayer(_NoOpLayer):
  layer_class = 'scale_grad'

  def __init__(self, scale = 1.0, disconnect = False, **kwargs):
    super(ScaleGradLayer, self).__init__(**kwargs)
    self.attrs['n_out'] = self.sources[0].attrs['n_out']
    if scale == 0 and disconnect:
      self.output = theano.gradient.zero_grad(self.sources[0].output)
    else:
      self.output = ScaleGradientOp(scale)(self.sources[0].output)


class DiscriminatorLayer(ForwardLayer):
  layer_class = 'disc'

  def __init__(self, base = None, pgen=0.5, alpha=1, forge=False, ncritic=0, n_tmp=1, dynamic_scaling=False, error_scaling=False, loss='ce', **kwargs):
    kwargs['n_out'] = 2
    super(DiscriminatorLayer, self).__init__(**kwargs)
    if not base:
      base = []
    self.params = {}
    W = self.add_param(self.create_random_normal_weights(self.sources[0].attrs['n_out'], n_tmp, scale=10000., name="W_%s" % self.name))
    b = self.add_param(self.create_bias(n_tmp))
    self.W = W
    self.b = b
    self.cost_val = numpy.float32(0)
    self.error_val = numpy.float32(0)
    self.known_grads = {}
    lng = T.sum(self.index, dtype='float32')
    batch_idx = self.add_param(theano.shared(numpy.zeros((1,),'float32'), 'batch_idx'), 'batch_idx',
                               custom_update=numpy.ones((1,),'float32'))
    if ncritic < 0:
      iscritic = T.eq(T.mod(T.cast(batch_idx,'int32')[0], numpy.int32(-ncritic + 1)), 0)
    else:
      iscritic = T.neq(T.mod(T.cast(batch_idx, 'int32')[0], numpy.int32(ncritic + 1)), 0)

    if forge:
      self.params = {}
      W = base[0].W
      b = base[0].b
      basecost = base[0].cost_val
      base = []
      preal = numpy.float32(1.0)

    def make_cost(src, real):
      ratio = lng / T.sum(src.index, dtype='float32')
      idx = (src.index.flatten() > 0).nonzero()
      eps = T.constant(1e-3)
      z = T.dot(src.output, W) + b
      if loss == 'exp':
        pcx = T.nnet.softmax(z.reshape((z.shape[0] * z.shape[1], z.shape[2])))[idx, 0]
        if not real: pcx = numpy.float32(1.) - pcx
        lss = -T.sum(T.log(pcx + eps))
        err = T.sum(T.lt(pcx, numpy.float32(0.5)))
      elif loss == 'ce':
        pcx = T.nnet.sigmoid(T.sum(z, axis=2)).flatten()[idx]
        if not real: pcx = numpy.float32(1.) - pcx
        pcx = T.clip(pcx,eps,numpy.float32(1)-eps)
        lss = -T.sum(T.log(pcx))
        err = T.sum(T.lt(pcx, numpy.float32(0.5)))
      elif loss == 'sse':
        pcx = T.nnet.sigmoid(T.sum(z, axis=2)).flatten()[idx]
        if not real: pcx = numpy.float32(1.) - pcx
        lss = T.sum((pcx - numpy.float32(1))**2)
        err = T.sum(T.lt(pcx, numpy.float32(0.5)))
      elif loss == 'emd':
        z = T.nnet.relu(T.sum(z,axis=2), alpha).flatten()[idx]
        if not real: z = -z
        lss = T.sum(z)
        err = T.sum(T.lt(z, numpy.float32(0.0)))
      self.cost_val += ratio * lss
      self.error_val += ratio * err

    for src in self.sources: # real
      make_cost(src, True)
    self.real_error_val = self.error_val / numpy.float32(len(self.sources + base))
    if base:
      for src in base: # gen
        make_cost(src, False)
    self.gen_error_val = self.error_val / numpy.float32(len(self.sources + base)) - self.real_error_val


    self.error_val /= numpy.float32(len(self.sources + base))
    self.cost_val /= numpy.float32(len(self.sources + base))

    self.cost_val += numpy.float32(10.) * (T.sum(W**2) + T.sum(b**2)) / numpy.float32(self.sources[0].attrs['n_out'] * n_tmp)

    if forge:
      if dynamic_scaling:
        self.cost_scale_val = T.clip(self.cost_val / basecost, numpy.float32(0.25), numpy.float32(4.0))
      else:
        self.cost_scale_val = numpy.float32(1.0)
      if ncritic:
        self.cost_val = ifelse(iscritic, ScaleGradientOp(0)(self.cost_val), self.cost_val)
    else:
      if error_scaling:
        self.cost_scale_val = numpy.float32(1.0)
        self.cost_val *= self.error_val / lng
      else:
        self.cost_scale_val = numpy.float32(1.0) # numpy.float32(len(self.sources + base))
      if ncritic:
        self.cost_val = ifelse(iscritic, self.cost_val, ScaleGradientOp(0)(self.cost_val))


  def cost(self):
    return self.cost_val, self.known_grads

  def cost_scale(self):
    return self.cost_scale_val * T.constant(self.attrs.get("cost_scale", 1.0), dtype="float32")

  def errors(self):
    return self.error_val


class SumLayer(_NoOpLayer):
  layer_class = 'sum'

  def __init__(self, **kwargs):
    super(SumLayer, self).__init__(**kwargs)
    self.attrs['n_out'] = self.sources[0].attrs['n_out']
    self.output = sum([s.output for s in self.sources])
    self.index = self.sources[0].index

class BlurLayer(_NoOpLayer):
  layer_class = "blur"
  from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
  rng = RandomStreams(hash(layer_class) % 2147462579)

  def __init__(self, ctx=5, p=1.0, **kwargs):
    super(BlurLayer, self).__init__(**kwargs)
    x_in, self.attrs['n_out'] = concat_sources(self.sources)
    kernel = self.rng.binomial(size=(1, 1, ctx, ctx), p=p, dtype='float32')
    kernel = kernel / T.maximum(kernel.sum(),numpy.float32(1))
    from theano.tensor.nnet import conv2d
    self.output = conv2d(input=x_in.dimshuffle(1,'x',2,0), border_mode=(int(ctx/2),int(ctx/2)), filters=kernel, filter_shape=(1, 1, ctx, ctx), input_shape=(None, 1, self.attrs['n_out'], None)).dimshuffle(3,0,2,1)[:,:,:,0]
    self.index = self.sources[0].index

class TanhToSigmoidLayer(_NoOpLayer):
  layer_class = 'tanh_to_sigmoid'

  def __init__(self, **kwargs):
    super(TanhToSigmoidLayer, self).__init__(**kwargs)
    x_in, self.attrs['n_out'] = concat_sources(self.sources)
    self.output = (x_in + numpy.float32(1)) / numpy.float32(2)
    self.index = self.sources[0].index

class SigmoidToTanhLayer(_NoOpLayer):
  layer_class = 'sigmoid_to_tanh'

  def __init__(self, **kwargs):
    super(SigmoidToTanhLayer, self).__init__(**kwargs)
    x_in, self.attrs['n_out'] = concat_sources(self.sources)
    self.output = x_in * numpy.float32(2) - numpy.float32(1)
    self.index = self.sources[0].index

class RNNBlockLayer(ForwardLayer):
  recurrent = True
  layer_class = 'rnnblock'

  def __init__(self, num_layers=1, direction=0, **kwargs):
    # this has to be provided in THEANO_FLAGS as e.g. contexts=gpu0->cuda0
    context_name = kwargs.get('device', str(theano.config.device))
    #if context_name == 'cpu':
    #  context_name = 'gpu0'
    kwargs['device'] = context_name
    #kwargs['n_out'] *= 2
    super(RNNBlockLayer, self).__init__(**kwargs)
    self.params = {}
    #self.attrs['n_out'] /= 2
    #self.set_attr('nout', self.attrs['n_out'] / 4)
    from theano.gpuarray import dnn
    from theano.gpuarray.type import gpuarray_shared_constructor
    from theano.tensor.extra_ops import cpu_contiguous
    #from theano.sandbox.cuda.basic_ops import gpu_contiguous

    rnnb = dnn.RNNBlock(
      dtype=theano.config.floatX,
      hidden_size=self.attrs['n_out'],
      num_layers=num_layers,
      rnn_mode='lstm',
      input_mode='linear',
      direction_mode='unidirectional' if direction != 0 else 'bidirectional',
      context_name=context_name if context_name != 'cpu' else 'gpu0'
      )

    buffer_size = 1 # self.attrs['n_out'] * num_layers
    #X = self.get_linear_forward_output()
    #X = T.concatenate([s.output for s in self.sources],axis=2)[::direction or 1]
    X = cpu_contiguous(T.concatenate([s.output for s in self.sources], axis=2)[::direction or 1])
    #X = cpu_contiguous(self.sources[0].output[::direction or 1])
    #X = T.concatenate([X,T.zeros((X.shape[0],batch_size - X.shape[1] + 1,X.shape[2]),X.dtype)],axis=1)[:,:-1]
    n_in = sum([s.attrs['n_out'] for s in self.sources])
    psize = rnnb.get_param_size([buffer_size, n_in])
    l = numpy.sqrt(6.) / numpy.sqrt(4*self.attrs['n_out'])
    pvalue = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(psize,)), dtype=theano.config.floatX)
    if context_name == 'cpu':
      params_cudnn = self.add_param(self.create_bias(psize,name='cudnn_%s' % self.name))
    else:
      params_cudnn = self.add_param(gpuarray_shared_constructor(pvalue, target=context_name,name='cudnn_%s' % self.name))
    c_init = cpu_contiguous(T.alloc(numpy.cast[theano.config.floatX](0), num_layers, X.shape[1], self.attrs['n_out']))
    h_init = cpu_contiguous(T.alloc(numpy.cast[theano.config.floatX](0), num_layers, X.shape[1], self.attrs['n_out']))

    W_out = self.add_param(self.create_random_uniform_weights(self.attrs['n_out'], self.y_in[self.attrs['target']].n_out))
    b_out = self.add_param(self.create_bias(self.y_in[self.attrs['target']].n_out))

    if context_name == 'cpu':
      self.cost_val = T.constant(0)
      self.error_val = T.constant(0)
      self.known_grads = {}
      return

    out = rnnb.apply(params_cudnn, X, h_init, c_init)[0]
    out = out[::-1]
    out = T.dot(out,W_out) + b_out
    self.y_m = out.reshape((out.shape[0] * out.shape[1],out.shape[2]))

    self.i = (self.index.flatten()>0).nonzero()
    self.y_data_flat = self.y_in[self.attrs['target']].flatten()
    nll, _ = T.nnet.crossentropy_softmax_1hot(x=self.y_m[self.i], y_idx=self.y_data_flat[self.i])
    self.cost_val = T.sum(nll)

    #self.cost_val = -T.sum(T.log(out[:,self.y_in[self.attrs['target']].flatten()][(self.index.flatten()>0).nonzero()]))
    self.known_grads = { params_cudnn : T.grad(self.cost_val, params_cudnn) }
    self.output = out
    self.index = self.sources[0].index

    self.error_val = T.sum(T.neq(T.argmax(self.y_m[self.i], axis=-1), self.y_data_flat[self.i]))

  def cost(self):
    return self.cost_val, self.known_grads

  def errors(self):
    return self.error_val


class SignalValue(ForwardLayer):
  layer_class = 'sigval'

  def __init__(self, begin=0, sidx=0, risk=0.1, margin=0.0, copy_output=None, **kwargs):
    kwargs['n_out'] = 2
    super(SignalValue, self).__init__(**kwargs)
    self.params = {}
    self.error_val = T.constant(0)
    self.known_grads = {}
    self.set_attr('begin', begin)
    self.set_attr('sidx', sidx)
    if not 'target' in self.attrs:
      self.attrs['target'] = 'classes'
    norm = T.sum(self.index[begin:], axis=0, dtype='float32') / T.sum(self.index, axis=0, dtype='float32')
    z = self.get_linear_forward_output()

    q = T.nnet.sigmoid(z)
    margin = 0.35
    margin = numpy.float32(margin) #q[:,:,1] * numpy.float32(margin)
    p = q[:,:,0]
    #kwargs['n_out'] = 2

    #n_in = sum([s.attrs['n_out'] for s in self.sources])
    #x_in = self.sources[0].output if len(self.sources) == 1 else T.concatenate([s.output for s in self.sources], axis=2)
    #W_margin = self.add_param(self.create_forward_weights(n_in, 1, name="W_margin_%s" % self.name))
    #b_margin = self.add_param(self.create_bias(1, name='b_margin_%s' % self.name))
    #margin = T.nnet.sigmoid(T.dot(x_in, W_margin) + b_margin)[:,:,0] * numpy.float32(margin)

    #p = T.nnet.sigmoid(z)
    r = copy_output.y_out if copy_output is not None else self.network.y[self.attrs['target']]
    r = r.reshape((p.shape[0],p.shape[1],4))
    p = p[begin:,:]
    rb = r[begin:,:,sidx]
    rs = r[begin:,:,sidx+1]
    #rn = rb.max(axis=0, keepdims=True)
    #rb /= rn
    #rs /= rn
    self.index = self.index[begin:]
    #margin = numpy.float32(margin)
    step = numpy.float32(0) * T.ones((self.index.shape[1],),'float32') #/ T.sum(self.index,axis=0,dtype='float32')#numpy.float32(1) / T.sum(self.index,axis=0,dtype='float32')
    stash = numpy.float32(1) # T.cast(p.shape[0], 'float32')
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(self.rng.randint(1234) + 1)
    risk = numpy.float32(risk)
    mom = numpy.float32(1.0)

    def accumulate(p, rb, rs, bp, ep):
      rnd = numpy.float32(1) #T.cast(srng.uniform(size=(1,), low=0, high=risk)[0],'float32')
      wb = rnd * T.maximum(p - margin, numpy.float32(0)) / (numpy.float32(1.00001) - margin)
      ws = rnd * T.maximum(numpy.float32(1.0) - p - margin, numpy.float32(0)) / (numpy.float32(1.00001) - margin)
      bp = mom * bp + step
      ep = mom * ep + step / rs
      bd, ed = wb * bp, ws * ep
      ba, ea = ed * rs, bd / rb
      return bp - bd + ba, ep - ed + ea
      #return T.maximum(bp - bd + ba, T.zeros_like(bp)), T.maximum(ep - ed + ea, T.zeros_like(ep))
      #return T.clip(bp - bd + ba,numpy.float32(0),numpy.float32(10)), T.clip(ep - ed + ea,numpy.float32(0),numpy.float32(10)/rs)

    binit = T.ones((p.shape[1],), dtype='float32') * stash
    einit = (T.ones((p.shape[1],), dtype='float32') * stash) / rs[0]

    c, _ = theano.scan(accumulate,sequences=[p,rb,rs],outputs_info=[binit,einit])

    bcost = T.extra_ops.cumsum(step.dimshuffle('x',0).repeat(c[0].shape[0],axis=0), axis=0)
    ecost = rs * T.extra_ops.cumsum(step / rs, axis=0)
    tcost = bcost + ecost + stash + stash #* rs / rs[0]
    total = (c[0] + c[1] * rs) / tcost - numpy.float32(1)
    #steps = T.arange(total.shape[0], dtype='float32').dimshuffle(0,'x').repeat(total.shape[1],axis=1)
    #total = total * steps / T.sum(steps,axis=0,keepdims=True)
    cost = T.sum(norm * T.sum(total,axis=0))
    self.error_val = T.sum((numpy.float32(1.) - total[-1]) * T.cast(total.shape[0],'float32') / norm) #T.sum(T.lt(total,numpy.float32(0)),dtype='float32',axis=0))
    self.cost_val = T.sum(T.sum(self.index,dtype='float32',axis=0) / norm) - cost

    self.cost_scale_val = numpy.float32(1)
    #self.cost_val -= T.constant(0.01) * T.sum((c[0] * c[1] * rs)/(tcost*tcost)) #(T.sum(p * T.log(p + T.constant(1e-5))) + T.sum((T.constant(1)-p) * T.log((T.constant(1)-p) + T.constant(1e-5))))
    #self.cost_scale_val = T.mean(T.cast(T.argmax(total[::-1],axis=0),'float32') + numpy.float32(1)) / T.cast(total.shape[0],'float32') #numpy.float32(1)
    #out = T.concatenate([p.dimshuffle(0,1,'x'), margin.dimshuffle(0,1,'x')],axis=2)
    self.p_y_given_y = p.dimshuffle(0,1,'x')
    self.output = p.dimshuffle(0,1,'x')
    self.margin = margin

  def cost(self):
    return self.cost_val, self.known_grads

  def cost_scale(self):
    return self.cost_scale_val * T.constant(self.attrs.get("cost_scale", 1.0), dtype="float32")

  def errors(self):
    return self.error_val


class SegmentInputLayer(_NoOpLayer):
  layer_class = 'segment_input'

  class ReinterpretCastOp(theano.Op):
    itypes = (T.imatrix,)
    otypes = (T.fmatrix,)

    def perform(self, node, inputs, output_storage):
      output_storage[0][0] = inputs[0].view(dtype='float32')

  def __init__(self, window=15, input_is_sparse=False, num_classes=None, **kwargs):
    super(SegmentInputLayer, self).__init__(**kwargs)

    assert len(self.sources) == 1
    self.set_attr('n_out', self.sources[0].attrs['n_out'])
    self.set_attr('window', window)

    src_out   = self.sources[0].output
    src_index = self.sources[0].index

    f = src_out.shape[0]  # number of frames
    b = src_out.shape[1]  # number of batches
    d = src_out.shape[2]  # feature dimension

    if input_is_sparse:
      rs = src_out.dimshuffle(1, 0).reshape((f * b,))
    else:
      rs = src_out.dimshuffle(1, 0, 2).reshape((f * b, d))
    rs_idx = src_index.dimshuffle(1, 0).flatten()

    frames_idx = T.arange(f * b)[(rs_idx>0).nonzero()]\
                  .repeat(self.attrs['window'])\
                  .reshape((rs_idx.sum(), self.attrs['window']))\
                 + T.arange(self.attrs['window'])

    # this filter has entries <= -1 for all elements that do not belong to the same sequence as the first frame
    frame_filter_1 = (f
                      - (frames_idx[:,0] % f)\
                         .repeat(self.attrs['window'])\
                         .reshape((rs_idx.sum(), self.attrs['window']))\
                      - T.arange(self.attrs['window']))

    # this filter has entries 0 for all elements that are discarded by self.index, 1 otherwise
    frame_filter_2 = T.concatenate([rs_idx, T.zeros((self.attrs['window'] * b,), dtype='int8')])[frames_idx.flatten()].reshape((rs_idx.sum(), self.attrs['window']))

    frames_idx = T.switch(frame_filter_1 * frame_filter_2 > 0, frames_idx, -1).dimshuffle(1, 0)

    # we add an additional vector with zeros s.t. the invalid entries from the filters above result in a feature vector of zeros
    zero = T.zeros((1,), dtype='int8') if input_is_sparse else T.zeros((1, src_out.shape[2]))
    self.z = T.concatenate([rs, zero], axis=0)[frames_idx]
    if input_is_sparse:
      self.z = T.extra_ops.to_one_hot(self.z.flatten(), num_classes).reshape((self.z.shape[0], self.z.shape[1], num_classes))
    self.make_output(self.z)

    self.index = T.cast((frame_filter_1 * frame_filter_2).clip(0, 1), 'int8').T

    inv_batch_idx   = frames_idx[0,:]
    batch_idx       = -T.ones((f * b,), dtype='int32')
    batch_idx       = T.set_subtensor(batch_idx[inv_batch_idx], T.arange(inv_batch_idx.size, dtype='int32')).reshape((b, f)).T
    self.batch_idxs = self.ReinterpretCastOp()(batch_idx)

class UnsegmentInputLayer(_NoOpLayer):
  layer_class = 'unsegment_input'

  class UnsegmentInputOp(theano.Op):
    itypes = (T.ftensor3, T.bmatrix)
    otypes = (T.ftensor3,)

    def perform(self, node, inputs, output_storage):
      post  = inputs[0]
      index = inputs[1]

      num_frames  = index.shape[0]
      num_batches = index.shape[1]
      window_size = post.shape[0]
      dim         = post.shape[2]

      out = numpy.zeros((num_frames, num_batches, window_size, dim), dtype='float32')

      cur = 0
      for b in range(num_batches):
        for f in range(num_frames):
          if index[f, b] == 0:
            continue

          cur_seq_num_frames = min(window_size, num_frames - f)
          for w in range(cur_seq_num_frames):
            out[f + w, b, w, :] = post[w, cur, :]

          cur += 1

      out = out.reshape((out.shape[0], out.shape[1], out.shape[2] * out.shape[3]))
      output_storage[0][0] = out

  def __init__(self, original_output, **kwargs):
    super(UnsegmentInputLayer, self).__init__(**kwargs)

    assert len(self.sources) == 1

    self.set_attr('original_output', original_output)

    self.index = self.network.get_layer(original_output).index
    out = self.UnsegmentInputOp()(self.sources[0].p_y_given_x, self.index)
    self.make_output(out)

class SegmentClassTargets(_NoOpLayer):
  layer_class = 'segment_class_targets'

  class BuildClassesOp(theano.Op):
    itypes = (T.iscalar, T.iscalar, T.imatrix, T.bmatrix)
    otypes = (T.ftensor3, T.bmatrix)

    def perform(self, node, inputs, output_storage):
      num_classes = inputs[0]
      window      = inputs[1]
      classes     = inputs[2]
      index       = inputs[3]

      assert classes.shape == index.shape

      num_frames = classes.shape[0]
      num_batches = classes.shape[1]
      num_start_frames = index.sum()

      out = numpy.zeros((window, num_start_frames, num_classes), dtype='float32')
      out_index = numpy.zeros((window, num_start_frames), dtype='int8')
      cur = 0
      for b in range(num_batches):
        for f in range(num_frames):
          if index[f, b] != 1:
            continue

          cur_seq_num_frames = min(window, num_frames - f)
          out_index[0:cur_seq_num_frames, cur] = 1
          for w in range(cur_seq_num_frames):
            c = classes[f + w, b]
            out[w:,cur,c] += 1.0

          cur += 1

      out /= numpy.arange(1, window + 1).reshape((window, 1, 1))

      output_storage[0][0] = out
      output_storage[1][0] = out_index

  def __init__(self, num_classes, window=15, **kwargs):
    super(SegmentClassTargets, self).__init__(**kwargs)

    assert len(self.sources) == 1
    self.set_attr('n_out', self.sources[0].attrs['n_out'])
    self.set_attr('num_classes', num_classes)
    self.set_attr('window', window)

    self.y_out, self.index = SegmentClassTargets.BuildClassesOp()(T.TensorConstant(theano.tensor.iscalar, self.attrs['num_classes']),
                                                                  T.TensorConstant(theano.tensor.iscalar, self.attrs['window']),
                                                                  self.sources[0].output, self.sources[0].index)
    self.output = self.y_out
