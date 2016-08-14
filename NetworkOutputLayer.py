
import numpy
from theano import tensor as T
import theano
from BestPathDecoder import BestPathDecodeOp
from CTC import CTCOp
from OpNumpyAlign import NumpyAlignOp
from NativeOp import FastBaumWelchOp
from NetworkBaseLayer import Layer
from SprintErrorSignals import sprint_loss_and_error_signal, SprintAlignmentAutomataOp
from TheanoUtil import time_batch_make_flat, grad_discard_out_of_bound
from Util import as_str
from Log import log


#from Accumulator import AccumulatorOpInstance

#def step(*args): # requires same amount of memory
#  xs = args[:(len(args)-1)/2]
#  ws = args[(len(args)-1)/2:-1]
#  b = args[-1]
#  out = b
#  for w,x in zip(ws,xs):
#    out += T.dot(x,w)
#  return out

class OutputLayer(Layer):
  layer_class = "softmax"

  def __init__(self, loss, y, dtype=None, copy_input=None, copy_output=None, time_limit=0, compute_priors=False,
               softmax_smoothing=1.0, grad_clip_z=None, grad_discard_out_of_bound_z=None, normalize_length=False,
               apply_softmax=True,
               **kwargs):
    """
    :param theano.Variable index: index for batches
    :param str loss: e.g. 'ce'
    """
    super(OutputLayer, self).__init__(**kwargs)
    self.set_attr("normalize_length", normalize_length)
    if dtype:
      self.set_attr('dtype', dtype)
    if copy_input:
      self.set_attr("copy_input", copy_input.name)
    if grad_clip_z is not None:
      self.set_attr("grad_clip_z", grad_clip_z)
    if grad_discard_out_of_bound_z is not None:
      self.set_attr("grad_discard_out_of_bound_z", grad_discard_out_of_bound_z)
    if not apply_softmax:
      self.set_attr("apply_softmax", apply_softmax)
    if not copy_input:
      self.z = self.b
      self.W_in = [self.add_param(self.create_forward_weights(source.attrs['n_out'], self.attrs['n_out'],
                                                              name="W_in_%s_%s" % (source.name, self.name)))
                   for source in self.sources]

      assert len(self.sources) == len(self.masks) == len(self.W_in)
      assert len(self.sources) > 0
      for source, m, W in zip(self.sources, self.masks, self.W_in):
        source_output = source.output
        #4D input from TwoD Layers -> collapse height dimension
        if source_output.ndim == 4:
          source_output = source_output.sum(axis=0)
        if source.attrs['sparse']:
          if source.output.ndim == 3:
            input = source_output[:,:,0]  # old sparse format
          else:
            assert source_output.ndim == 2
            input = source.output
          self.z += W[T.cast(input, 'int32')]
        elif m is None:
          self.z += self.dot(source_output, W)
        else:
          self.z += self.dot(self.mass * m * source_output, W)
    else:
      self.z = copy_input.output
    assert self.z.ndim == 3
    if grad_clip_z is not None:
      grad_clip_z = numpy.float32(grad_clip_z)
      self.z = theano.gradient.grad_clip(self.z, -grad_clip_z, grad_clip_z)
    if grad_discard_out_of_bound_z is not None:
      grad_discard_out_of_bound_z = numpy.float32(grad_discard_out_of_bound_z)
      self.z = grad_discard_out_of_bound(self.z, -grad_discard_out_of_bound_z, grad_discard_out_of_bound_z)
    if not copy_output:
      self.y = y
    else:
      self.index = copy_output.index
      self.y = copy_output.y_out
    if isinstance(y, T.Variable):
      self.y_data_flat = time_batch_make_flat(y)
    else:
      assert self.attrs.get("target", "").endswith("[sparse:coo]")
      assert isinstance(self.y, tuple)
      assert len(self.y) == 3
      s0, s1, weight = self.y
      from NativeOp import max_and_argmax_sparse
      n_time = self.z.shape[0]
      n_batch = self.z.shape[1]
      mask = self.network.j[self.attrs.get("target", "").replace("[sparse:coo]", "[sparse:coo:2:0]")]
      out_arg = T.zeros((n_time, n_batch), dtype="float32")
      out_max = T.zeros((n_time, n_batch), dtype="float32") - numpy.float32(1e16)
      out_arg, out_max = max_and_argmax_sparse(s0, s1, weight, mask, out_arg, out_max)
      assert out_arg.ndim == 2
      self.y_data_flat = out_arg.astype("int32")

    self.norm = numpy.float32(1)
    self.target_index = self.index
    if time_limit == 'inf':
      #target_length = self.index.shape[0]
      #mass = T.cast(T.sum(self.index),'float32')
      #self.index = theano.ifelse.ifelse(T.gt(self.z.shape[0],target_length),self.sources[0].index,self.index)
      #self.norm = mass / T.cast(T.sum(self.index),'float32')
      num = T.cast(T.sum(self.index), 'float32')
      if self.eval_flag:
        self.index = self.sources[0].index
      else:
        import theano.ifelse
        padx = T.zeros((T.abs_(self.index.shape[0] - self.z.shape[0]), self.index.shape[1], self.z.shape[2]), 'float32') + self.z[-1]
        pady = T.zeros((T.abs_(self.index.shape[0] - self.z.shape[0]), self.index.shape[1]), 'int32') #+ y[-1]
        padi = T.ones((T.abs_(self.index.shape[0] - self.z.shape[0]), self.index.shape[1]), 'int8')
        self.z = theano.ifelse.ifelse(T.lt(self.z.shape[0], self.index.shape[0]),
                                      T.concatenate([self.z,padx],axis=0), self.z)
        #self.z = theano.ifelse.ifelse(T.gt(self.z.shape[0], self.index.shape[0]),self.z[:self.index.shape[0]], self.z)
        self.y_data_flat = time_batch_make_flat(theano.ifelse.ifelse(T.gt(self.z.shape[0],self.index.shape[0]),
                                                                     T.concatenate([y,pady], axis=0), y))
        #self.index = theano.ifelse.ifelse(T.gt(self.z.shape[0], self.index.shape[0]), T.concatenate([T.ones((self.z.shape[0] - self.index.shape[0],self.z.shape[1]),'int8'), self.index], axis=0), self.index)
        self.index = theano.ifelse.ifelse(T.gt(self.z.shape[0], self.index.shape[0]),
                                          T.concatenate([padi,self.index],axis=0),self.index)
      self.norm *= num / T.cast(T.sum(self.index),'float32')
    elif time_limit > 0:
      end = T.min([self.z.shape[0], T.constant(time_limit, 'int32')])
      nom = T.cast(T.sum(self.index),'float32')
      self.index = T.set_subtensor(self.index[end:], T.zeros_like(self.index[end:]))
      self.norm = nom / T.cast(T.sum(self.index),'float32')
      self.z = T.set_subtensor(self.z[end:], T.zeros_like(self.z[end:]))

    #xs = [s.output for s in self.sources]
    #self.z = AccumulatorOpInstance(*[self.b] + xs + self.W_in)
    #outputs_info = None #[ T.alloc(numpy.cast[theano.config.floatX](0), index.shape[1], self.attrs['n_out']) ]

    #self.z, _ = theano.scan(step,
    #                        sequences = [s.output for s in self.sources],
    #                        non_sequences = self.W_in + [self.b])

    self.set_attr('from', ",".join([s.name for s in self.sources]))
    self.i = (self.index.flatten() > 0).nonzero()
    self.j = ((1 - self.index.flatten()) > 0).nonzero()
    self.loss = as_str(loss.encode("utf8"))
    self.attrs['loss'] = self.loss
    self.attrs['compute_priors'] = compute_priors
    if softmax_smoothing != 1.0:
      self.attrs['softmax_smoothing'] = softmax_smoothing
      print >> log.v3, "Logits before the softmax scaled with factor ", softmax_smoothing
      self.z *= numpy.float32(softmax_smoothing)
    if self.loss == 'priori':
      self.priori = self.shared(value=numpy.ones((self.attrs['n_out'],), dtype=theano.config.floatX), borrow=True)

    #self.make_output(self.z, collapse = False)
    # Note that self.output is going to be overwritten in our derived classes.
    self.output = self.make_consensus(self.z) if self.depth > 1 else self.z

  def create_bias(self, n, prefix='b', name=""):
    if not name:
      name = "%s_%s" % (prefix, self.name)
    assert n > 0
    bias = numpy.log(1.0 / n)  # More numerical stable.
    value = numpy.zeros((n,), dtype=theano.config.floatX) + bias
    return self.shared(value=value, borrow=True, name=name)

  def entropy(self):
    """
    :rtype: theano.Variable
    """
    return -T.sum(self.p_y_given_x[self.i] * T.log(self.p_y_given_x[self.i]))

  def errors(self):
    """
    :rtype: theano.Variable
    """
    if self.attrs.get("target", "") == "null":
      return None
    if self.y_data_flat.dtype.startswith('int'):
      if self.y_data_flat.type == T.ivector().type:
        if self.attrs['normalize_length']:
          return self.norm * T.sum(T.max(T.neq(T.argmax(self.output[:self.index.shape[0]], axis=2), self.y) * T.cast(self.index,'float32'),axis=0))
        return self.norm * T.sum(T.neq(T.argmax(self.y_m[self.i], axis=-1), self.y_data_flat[self.i]))
      else:
        return self.norm * T.sum(T.neq(T.argmax(self.y_m[self.i], axis=-1), T.argmax(self.y_data_flat[self.i], axis = -1)))
    elif self.y_data_flat.dtype.startswith('float'):
      return T.mean(T.sqr(self.y_m[self.i] - self.y_data_flat.reshape(self.y_m.shape)[self.i]))
      #return T.sum(T.sqr(self.y_m[self.i] - self.y.flatten()[self.i]))
      #return T.sum(T.sum(T.sqr(self.y_m - self.y.reshape(self.y_m.shape)), axis=1)[self.i])
      #return T.sum(T.sqr(self.y_m[self.i] - self.y.reshape(self.y_m.shape)[self.i]))
      #return T.sum(T.sum(T.sqr(self.z - (self.y.reshape((self.index.shape[0], self.index.shape[1], self.attrs['n_out']))[:self.z.shape[0]])), axis=2).flatten()[self.i])
      #return T.sum(T.sqr(self.y_m[self.i] - (self.y.reshape((self.index.shape[0], self.index.shape[1], self.attrs['n_out']))[:self.z.shape[0]]).reshape(self.y_m.shape)[self.i]))
      #return T.sum(T.sqr(self.y_m[self.i] - self.y.reshape(self.y_m.shape)[self.i]))
    else:
      raise NotImplementedError()


class FramewiseOutputLayer(OutputLayer):
  def __init__(self, **kwargs):
    super(FramewiseOutputLayer, self).__init__(**kwargs)
    self.initialize()

  def initialize(self):
    #self.y_m = self.output.dimshuffle(2,0,1).flatten(ndim = 2).dimshuffle(1,0)
    output = self.output
    self.y_m = output.reshape((output.shape[0]*output.shape[1],output.shape[2]))
    self.y_pred = T.argmax(self.y_m[self.i], axis=1, keepdims=True)
    if not self.attrs.get("apply_softmax", True):
      self.p_y_given_x = self.y_m
      self.z = T.log(self.z)
      self.y_m = T.log(self.y_m)
    elif self.loss in ['ce', 'entropy', 'none']: self.p_y_given_x = T.nnet.softmax(self.y_m)
    elif self.loss == 'sse': self.p_y_given_x = self.y_m
    elif self.loss == 'priori': self.p_y_given_x = T.nnet.softmax(self.y_m) / self.priori
    else: assert False, "invalid loss: " + self.loss
    self.output = self.p_y_given_x.reshape(self.output.shape)
    if self.attrs['compute_priors']:
      custom = T.mean(self.p_y_given_x[self.i], axis=0) if self.attrs.get('trainable',True) else T.constant(0,'float32')
      self.priors = self.add_param(theano.shared(numpy.zeros((self.attrs['n_out'],), 'float32'), 'priors'), 'priors',
                                   custom_gradient=custom,
                                   custom_gradient_normalized=True and self.attrs.get('trainable',True))

  def cost(self):
    """
    :rtype: (theano.Variable | None, dict[theano.Variable,theano.Variable] | None)
    :returns: cost, known_grads
    """
    known_grads = None
    if not self.attrs.get("apply_softmax", True):
      if self.loss != "ce": raise NotImplementedError
      assert self.p_y_given_x.ndim == 2  # flattened
      index = T.cast(self.index, "float32").flatten()
      index_bc = index.dimshuffle(0, 'x')
      y_idx = self.y_data_flat
      assert y_idx.ndim == 1
      p = T.clip(self.p_y_given_x, numpy.float32(1.e-38), numpy.float32(1.e20))
      from NativeOp import subtensor_batched_index
      logp = T.log(subtensor_batched_index(p, y_idx))
      assert logp.ndim == 1
      nll = -T.sum(logp * index)
      # the grad for p is: -y_ref/p
      known_grads = {self.p_y_given_x: -T.inv(p) * T.extra_ops.to_one_hot(self.y_data_flat, self.attrs["n_out"]) * index_bc}
      return self.norm * nll, known_grads
    elif self.loss == 'ce' or self.loss == 'priori':
      if self.attrs.get("target", "").endswith("[sparse:coo]"):
        assert isinstance(self.y, tuple)
        assert len(self.y) == 3
        from NativeOp import crossentropy_softmax_and_gradient_z_sparse
        y_mask = self.network.j[self.attrs.get("target", "").replace("[sparse:coo]", "[sparse:coo:2:0]")]
        ce, grad_z = crossentropy_softmax_and_gradient_z_sparse(
          self.z, self.index, self.y[0], self.y[1], self.y[2], y_mask)
        return self.norm * T.sum(ce), {self.z: grad_z}
      if self.y_data_flat.type == T.ivector().type:
        # Use crossentropy_softmax_1hot to have a more stable and more optimized gradient calculation.
        # Theano fails to use it automatically; I guess our self.i indexing is too confusing.
        #idx = self.index.flatten().dimshuffle(0,'x').repeat(self.y_m.shape[1],axis=1) # faster than line below
        #nll, pcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m * idx, y_idx=self.y_data_flat * self.index.flatten())
        nll, pcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m[self.i], y_idx=self.y_data_flat[self.i])
        #nll, pcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m, y_idx=self.y_data_flat)
        #nll = -T.log(T.nnet.softmax(self.y_m)[self.i,self.y_data_flat[self.i]])
        #z_c = T.exp(self.z[:,self.y])
        #nll = -T.log(z_c / T.sum(z_c,axis=2,keepdims=True))
        #nll, pcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m, y_idx=self.y_data_flat)
        #nll = T.set_subtensor(nll[self.j], T.constant(0.0))
      else:
        nll = -T.dot(T.log(T.clip(self.p_y_given_x[self.i], 1.e-38, 1.e20)), self.y_data_flat[self.i].T)
      return self.norm * T.sum(nll), known_grads
    elif self.loss == 'entropy':
      h_e = T.exp(self.y_m) #(TB)
      pcx = T.clip((h_e / T.sum(h_e, axis=1, keepdims=True)).reshape((self.index.shape[0],self.index.shape[1],self.attrs['n_out'])), 1.e-6, 1.e6) # TBD
      ee = -T.sum(pcx[self.i] * T.log(pcx[self.i])) # TB
      #nll, pcxs = T.nnet.crossentropy_softmax_1hot(x=self.y_m[self.i], y_idx=self.y[self.i])
      nll, _ = T.nnet.crossentropy_softmax_1hot(x=self.y_m, y_idx=self.y_data_flat) # TB
      ce = nll.reshape(self.index.shape) * self.index # TB
      y = self.y_data_flat.reshape(self.index.shape) * self.index # TB
      f = T.any(T.gt(y,0), axis=0) # B
      return T.sum(f * T.sum(ce, axis=0) + (1-f) * T.sum(ee, axis=0)), known_grads
      #return T.sum(T.switch(T.gt(T.sum(y,axis=0),0), T.sum(ce, axis=0), -T.sum(ee, axis=0))), known_grads
      #return T.switch(T.gt(T.sum(self.y_m[self.i]),0), T.sum(nll), -T.sum(pcx * T.log(pcx))), known_grads
    elif self.loss == 'priori':
      pcx = self.p_y_given_x[self.i, self.y_data_flat[self.i]]
      pcx = T.clip(pcx, 1.e-38, 1.e20)  # For pcx near zero, the gradient will likely explode.
      return -T.sum(T.log(pcx)), known_grads
    elif self.loss == 'sse':
      if self.y_data_flat.dtype.startswith('int'):
        y_f = T.cast(T.reshape(self.y_data_flat, (self.y_data_flat.shape[0] * self.y_data_flat.shape[1]), ndim=1), 'int32')
        y_oh = T.eq(T.shape_padleft(T.arange(self.attrs['n_out']), y_f.ndim), T.shape_padright(y_f, 1))
        return T.mean(T.sqr(self.p_y_given_x[self.i] - y_oh[self.i])), known_grads
      else:
        #return T.sum(T.sum(T.sqr(self.y_m - self.y.reshape(self.y_m.shape)), axis=1)[self.i]), known_grads
        return T.mean(T.sqr(self.y_m[self.i] - self.y_data_flat.reshape(self.y_m.shape)[self.i])), known_grads
        #return T.sum(T.sum(T.sqr(self.z - (self.y.reshape((self.index.shape[0], self.index.shape[1], self.attrs['n_out']))[:self.z.shape[0]])), axis=2).flatten()[self.i]), known_grads
        #y_z = T.set_subtensor(T.zeros((self.index.shape[0],self.index.shape[1],self.attrs['n_out']), dtype='float32')[:self.z.shape[0]], self.z).flatten()
        #return T.sum(T.sqr(y_z[self.i] - self.y[self.i])), known_grads
        #return T.sum(T.sqr(self.y_m - self.y[:self.z.shape[0]*self.index.shape[1]]).flatten()[self.i]), known_grads
    elif self.loss == "none":
      return None, None
    else:
      assert False, "unknown loss: %s. maybe fix LayerNetwork.make_classifier" % self.loss


class DecoderOutputLayer(FramewiseOutputLayer): # must be connected to a layer with self.W_lm_in
#  layer_class = "decoder"

  def __init__(self, **kwargs):
    kwargs['loss'] = 'ce'
    super(DecoderOutputLayer, self).__init__(**kwargs)
    self.set_attr('loss', 'decode')

  def cost(self):
    res = 0.0
    for s in self.y_s:
      nll, pcx = T.nnet.crossentropy_softmax_1hot(x=s.reshape((s.shape[0]*s.shape[1],s.shape[2]))[self.i], y_idx=self.y_data_flat[self.i])
      res += T.sum(nll) #T.sum(T.log(s.reshape((s.shape[0]*s.shape[1],s.shape[2]))[self.i,self.y_data_flat[self.i]]))
    return res / float(len(self.y_s)), None

  def initialize(self):
    output = 0
    self.y_s = []
    #i = T.cast(self.index.dimshuffle(0,1,'x').repeat(self.attrs['n_out'],axis=2),'float32')
    for s in self.sources:
      self.y_s.append(T.dot(s.output,s.W_lm_in))
      output += self.y_s[-1]
      #output += T.concatenate([T.dot(s.output[:-1],s.W_lm_in), T.eye(self.attrs['n_out'], 1).flatten().dimshuffle('x','x',0).repeat(self.index.shape[1], axis=1)], axis=0)
    self.params = {}
    self.y_m = output.reshape((output.shape[0]*output.shape[1],output.shape[2]))
    h = T.exp(self.y_m)
    self.p_y_given_x = T.nnet.softmax(self.y_m) #h / h.sum(axis=1,keepdims=True) #T.nnet.softmax(self.y_m)
    self.y_pred = T.argmax(self.y_m[self.i], axis=1, keepdims=True)
    self.output = self.p_y_given_x.reshape(self.output.shape)


class SequenceOutputLayer(OutputLayer):
  def __init__(self, prior_scale=0.0, log_prior=None, ce_smoothing=0.0, exp_normalize=True, loss_like_ce=False, sprint_opts=None, **kwargs):
    super(SequenceOutputLayer, self).__init__(**kwargs)
    self.prior_scale = prior_scale
    if prior_scale:
      self.set_attr("prior_scale", prior_scale)
    self.log_prior = log_prior
    self.ce_smoothing = ce_smoothing
    if ce_smoothing:
      self.set_attr("ce_smoothing", ce_smoothing)
    self.exp_normalize = exp_normalize
    if not exp_normalize:
      self.set_attr("exp_normalize", exp_normalize)
    self.loss_like_ce = loss_like_ce
    if loss_like_ce:
      self.set_attr("loss_like_ce", loss_like_ce)
    self.sprint_opts = sprint_opts
    if sprint_opts:
      self.set_attr("sprint_opts", sprint_opts)
    self.initialize()

  def initialize(self):
    assert self.loss in ('ctc', 'ce_ctc', 'ctc2', 'sprint', 'viterbi', 'fast_bw'), 'invalid loss: ' + self.loss
    self.y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim = 2)
    if not self.attrs.get("apply_softmax", True):
      p_y_given_x = self.y_m
      self.p_y_given_x = self.z
      self.z = T.log(self.z)
      self.y_m = T.log(self.y_m)
    else:
      p_y_given_x = T.nnet.softmax(self.y_m)
      self.p_y_given_x = T.reshape(T.nnet.softmax(self.y_m), self.z.shape)
    self.y_pred = T.argmax(p_y_given_x, axis=-1)
    self.output = self.p_y_given_x.reshape(self.output.shape)
    if self.attrs['compute_priors']:
      self.priors = self.add_param(theano.shared(numpy.ones((self.attrs['n_out'],), 'float32') / self.attrs['n_out'], 'priors'), 'priors',
                                   custom_gradient=T.mean(p_y_given_x[self.i], axis=0),
                                   custom_gradient_normalized=True)
      self.log_prior = T.log(self.priors)

  def index_for_ctc(self):
    for source in self.sources:
      if hasattr(source, "output_sizes"):
        return T.cast(source.output_sizes[:, 1], "int32")
    return T.cast(T.sum(self.sources[0].index, axis=0), 'int32')

  def output_index(self):
    for source in self.sources:
      if hasattr(source, "output_sizes"):
        return source.index
    if self.loss == 'viterbi':
      return self.sources[0].index
    return super(SequenceOutputLayer, self).output_index()

  def cost(self):
    """
    :param y: shape (time*batch,) -> label
    :return: error scalar, known_grads dict
    """
    y_f = T.cast(T.reshape(self.y_data_flat, (self.y_data_flat.shape[0] * self.y_data_flat.shape[1]), ndim = 1), 'int32')
    known_grads = None
    if self.loss == 'sprint':
      if not isinstance(self.sprint_opts, dict):
        import json
        self.sprint_opts = json.loads(self.sprint_opts)
      assert isinstance(self.sprint_opts, dict), "you need to specify sprint_opts in the output layer"
      if self.exp_normalize:
        log_probs = T.log(self.p_y_given_x)
      else:
        log_probs = self.z
      if self.attrs['compute_priors']: # use own priors, assume prior scale in sprint config to be 0.0
        log_probs -= T.constant(self.prior_scale, 'float32') * self.log_prior
      err, grad = sprint_loss_and_error_signal(
        output_layer=self,
        target=self.attrs.get("target", "classes"),
        sprint_opts=self.sprint_opts,
        log_posteriors=log_probs,
        seq_lengths=T.sum(self.index, axis=0)
      )
      err = err.sum()
      if self.loss_like_ce:
        y_ref = T.clip(self.p_y_given_x - grad, numpy.float32(0), numpy.float32(1))
        err = -T.sum(T.switch(T.cast(self.index, "float32").dimshuffle(0, 1, 'x'),
                              y_ref * T.log(self.p_y_given_x),
                              numpy.float32(0)))
      if self.ce_smoothing:
        err *= numpy.float32(1.0 - self.ce_smoothing)
        grad *= numpy.float32(1.0 - self.ce_smoothing)
        if not self.prior_scale:  # we kept the softmax bias as it was
          nll, pcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m[self.i], y_idx=self.y_data_flat[self.i])
        else:  # assume that we have subtracted the bias by the log priors beforehand
          assert self.log_prior is not None
          # In this case, for the CE calculation, we need to add the log priors again.
          y_m_prior = T.reshape(self.z + numpy.float32(self.prior_scale) * self.log_prior,
                                (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim=2)
          nll, pcx = T.nnet.crossentropy_softmax_1hot(x=y_m_prior[self.i], y_idx=self.y_data_flat[self.i])
        ce = numpy.float32(self.ce_smoothing) * T.sum(nll)
        err += ce
        grad += T.grad(ce, self.z)
      known_grads = {self.z: grad}
      return err, known_grads
    elif self.loss == 'fast_bw':
      if not isinstance(self.sprint_opts, dict):
        import json
        self.sprint_opts = json.loads(self.sprint_opts)
      assert isinstance(self.sprint_opts, dict), "you need to specify sprint_opts in the output layer"
      scores = -T.log(self.p_y_given_x)
      edges, weights, start_end_states, state_buffer = SprintAlignmentAutomataOp(self.sprint_opts)(self.network.tags)
      float_idx = T.cast(self.index, "float32")
      fwdbwd = FastBaumWelchOp.make_op()(scores, edges, weights, start_end_states, float_idx, state_buffer)
      err = (T.exp(-fwdbwd) * scores * float_idx.dimshuffle(0, 1, 'x')).sum()
      return err, known_grads
    elif self.loss == 'ctc':
      from theano.tensor.extra_ops import cpu_contiguous
      err, grad, priors = CTCOp()(self.p_y_given_x, cpu_contiguous(self.y.dimshuffle(1, 0)), self.index_for_ctc())
      known_grads = {self.z: grad}
      return err.sum(), known_grads, priors.sum(axis=0)
    elif self.loss == 'ce_ctc':
      y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim=2)
      p_y_given_x = T.nnet.softmax(y_m)
      #pcx = p_y_given_x[(self.i > 0).nonzero(), y_f[(self.i > 0).nonzero()]]
      pcx = p_y_given_x[self.i, self.y_data_flat[self.i]]
      ce = -T.sum(T.log(pcx))
      return ce, known_grads
    elif self.loss == 'ctc2':
      from NetworkCtcLayer import ctc_cost, uniq_with_lengths, log_sum
      max_time = self.z.shape[0]
      num_batches = self.z.shape[1]
      time_mask = self.index.reshape((max_time, num_batches))
      y_batches = self.y_data_flat.reshape((max_time, num_batches))
      targets, seq_lens = uniq_with_lengths(y_batches, time_mask)
      log_pcx = self.z - log_sum(self.z, axis=0, keepdims=True)
      err = ctc_cost(log_pcx, time_mask, targets, seq_lens)
      return err, known_grads
    elif self.loss == 'viterbi':
      y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim=2)
      scores = T.log(self.p_y_given_x) - self.prior_scale * T.log(self.priors)
      y = NumpyAlignOp(False)(self.sources[0].index,self.index,-scores,self.y)
      self.y_data_flat = y.flatten()
      nll, pcx = T.nnet.crossentropy_softmax_1hot(x=y_m[self.i], y_idx=self.y_data_flat[self.i])
      return T.sum(nll), known_grads

  def errors(self):
    if self.loss in ('ctc', 'ce_ctc'):
      from theano.tensor.extra_ops import cpu_contiguous
      return T.sum(BestPathDecodeOp()(self.p_y_given_x, cpu_contiguous(self.y.dimshuffle(1, 0)), self.index_for_ctc()))
    elif self.loss == 'viterbi':
      scores = T.log(self.p_y_given_x) - self.prior_scale * T.log(self.priors)
      y = NumpyAlignOp(False)(self.sources[0].index, self.index, -scores, self.y)
      self.y_data_flat = y.flatten()
      return super(SequenceOutputLayer, self).errors()
    else:
      return super(SequenceOutputLayer, self).errors()


class UnsupervisedOutputLayer(OutputLayer):
  def __init__(self, base, prior_scale=0.0, prior_confidence=0.0, posterior_confidence=0.0, **kwargs):
    kwargs['loss'] = 'ce'
    super(UnsupervisedOutputLayer, self).__init__(**kwargs)
    if base:
      self.set_attr('base', base[0].name)
    self.set_attr('prior_scale', prior_scale)
    self.set_attr('prior_confidence', prior_confidence)
    self.set_attr('posterior_confidence', posterior_confidence)
    self.lm_score = T.constant(0.0,'float32')
    z_f = self.z.reshape((self.z.shape[0] * self.z.shape[1], self.z.shape[2]))
    self.y_m = z_f
    i_f = T.cast(self.index.flatten(), 'float32').dimshuffle(0, 'x').repeat(z_f.shape[1], axis=1)
    self.p = T.nnet.softmax(z_f)
    self.N = T.maximum(T.sum(i_f), numpy.float32(1))
    if base is None:
      custom = T.mean(self.p[self.i], axis=0) if self.attrs.get('trainable',True) else T.constant(0,'float32')
      self.priors = self.add_param(theano.shared(numpy.zeros((self.attrs['n_out'],), 'float32'), 'priors'), 'priors',
                                   custom_gradient=custom,
                                   custom_gradient_normalized=True and self.attrs.get('trainable',True))
    self.running_priors = self.add_param(
      theano.shared(numpy.ones((self.attrs['n_out'],), 'float32') / numpy.float32(self.attrs['n_out']),
                    'running_priors'), 'running_priors',
      custom_gradient=T.mean(self.p[self.i],axis=0),
      custom_gradient_normalized=True)

    if base is not None:
      if self.attrs['prior_confidence'] == 1.0:
        self.lm_score = -T.sum(T.log(T.max(base[0].output, axis=2)))
      elif self.attrs['prior_confidence'] == 0.0:
        self.lm_score = -T.sum(T.log(base[0].output)) / T.constant(self.attrs['n_out'],'float32')
      else:
        p = base[0].output.reshape(self.p.shape)**T.constant(1.-prior_confidence,'float32')
        p = p / p.sum(axis=1, keepdims=True)
        self.lm_score = -T.sum(p * i_f * T.log(base[0].output.reshape(p.shape)))
    else:
      self.lm_score = 0.0
      assert prior_scale == 0.0

  def cost(self):
    known_grads = None
    if self.train_flag or True:
      i_f = T.cast(self.index.flatten(), 'float32').dimshuffle(0, 'x').repeat(self.p.shape[1], axis=1)
      prior_scale = T.constant(self.attrs['prior_scale'],'float32')
      entropy_scale = T.constant(1. - self.attrs['prior_scale'],'float32')
      confidence = T.constant(1. - self.attrs['posterior_confidence'],'float32')
      if self.attrs['posterior_confidence'] == 1.0:
        H = -T.sum(T.cast(self.index.flatten(), 'float32') * T.log(T.max(self.p,axis=1)))
      else:
        p = (self.p**confidence)
        p = p / p.sum(axis=1,keepdims=True)
        H = -T.sum(T.sum(p * i_f * T.log(self.p)))
      L = self.lm_score
      #L = -T.sum(self.priors * T.log(batch_prior)) * self.N
      #Q = -T.sum(self.priors.dimshuffle('x',0).repeat(self.p.shape[0],axis=0) * i_f * T.log(self.p))
      #H = theano.printing.Print("H")(H)
      #L = theano.printing.Print("L")(L)
      #return entropy_scale * H + lm_scale * self.lm_score, known_grads
      #return entropy_scale * H + prior_scale * Q + lm_scale * self.lm_score, known_grads
      #return H, known_grads
      #return H * L, known_grads
      return entropy_scale * H + prior_scale * L, known_grads
    else:
      nll, _ = T.nnet.crossentropy_softmax_1hot(x=self.y_m[self.i], y_idx=self.y_data_flat[self.i])
      return T.sum(nll), known_grads

  def errors(self):
    """
    :rtype: theano.Variable
    """
    if self.y_data_flat.type == T.ivector().type:
      return self.norm * T.sum(T.neq(T.argmax(self.p[self.i], axis=-1), self.y_data_flat[self.i]))
    else:
      return self.norm * T.sum(T.neq(T.argmax(self.p[self.i], axis=-1), T.argmax(self.y_data_flat[self.i], axis=-1)))
