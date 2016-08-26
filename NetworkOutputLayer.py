
import numpy
import os
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

  def __init__(self, loss, y, dtype=None, copy_input=None, copy_output=None, time_limit=0,
               compute_priors=False, compute_priors_exp_average=0,
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
    if compute_priors:
      self.set_attr('compute_priors', compute_priors)
      if compute_priors_exp_average:
        self.set_attr('compute_priors_exp_average', compute_priors_exp_average)
    if softmax_smoothing != 1.0:
      self.attrs['softmax_smoothing'] = softmax_smoothing
      print >> log.v4, "Logits before the softmax scaled with factor ", softmax_smoothing
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
    self.p_y_given_x_flat = self.p_y_given_x  # a bit inconsistent here... it's always flat at the moment
    self.output = self.p_y_given_x.reshape(self.output.shape)
    if self.attrs.get('compute_priors', False):
      custom = T.mean(self.p_y_given_x[self.i], axis=0) if self.attrs.get('trainable',True) else T.constant(0,'float32')
      exp_average = self.attrs.get("compute_priors_exp_average", 0)
      self.priors = self.add_param(theano.shared(numpy.zeros((self.attrs['n_out'],), 'float32'), 'priors'), 'priors',
                                   custom_update=custom,
                                   custom_update_normalized=(not exp_average) and self.attrs.get('trainable',True),
                                   custom_update_exp_average=exp_average)

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
  def __init__(self, prior_scale=0.0, log_prior=None, ce_smoothing=0.0, exp_normalize=True, am_scale=1, gamma=1, bw_norm_class_avg=False, loss_like_ce=False, trained_softmax_prior=False, sprint_opts=None, **kwargs):
    super(SequenceOutputLayer, self).__init__(**kwargs)
    self.prior_scale = prior_scale
    if prior_scale:
      self.set_attr("prior_scale", prior_scale)
    if log_prior is not None:
      # We expect a filename to the priors, stored as txt, in +log space.
      assert isinstance(log_prior, str)
      self.set_attr("log_prior", log_prior)
      from Util import load_txt_vector
      assert os.path.exists(log_prior)
      log_prior = load_txt_vector(log_prior)
      assert len(log_prior) == self.attrs['n_out'], "dim missmatch: %i != %i" % (len(log_prior), self.attrs['n_out'])
      log_prior = numpy.array(log_prior, dtype="float32")
    self.log_prior = log_prior
    self.ce_smoothing = ce_smoothing
    if ce_smoothing:
      self.set_attr("ce_smoothing", ce_smoothing)
    self.exp_normalize = exp_normalize
    if not exp_normalize:
      self.set_attr("exp_normalize", exp_normalize)
    if am_scale != 1:
      self.set_attr("am_scale", am_scale)
    if gamma != 1:
      self.set_attr("gamma", gamma)
    if bw_norm_class_avg:
      self.set_attr("bw_norm_class_avg", bw_norm_class_avg)
    self.loss_like_ce = loss_like_ce
    if loss_like_ce:
      self.set_attr("loss_like_ce", loss_like_ce)
    if trained_softmax_prior:
      self.set_attr('trained_softmax_prior', trained_softmax_prior)
      assert not self.attrs.get('compute_priors', False)
      initialization = numpy.zeros((self.attrs['n_out'],), 'float32')
      if self.log_prior is not None:
        # Will use that as initialization.
        assert self.log_prior.shape == initialization.shape
        initialization = self.log_prior
      self.trained_softmax_prior_p = self.add_param(theano.shared(initialization, 'trained_softmax_prior_p'))
      self.priors = T.nnet.softmax(self.trained_softmax_prior_p).reshape((self.attrs['n_out'],))
      self.log_prior = T.log(self.priors)
    self.sprint_opts = sprint_opts
    if sprint_opts:
      self.set_attr("sprint_opts", sprint_opts)
    self.initialize()

  def initialize(self):
    assert self.loss in ('ctc', 'ce_ctc', 'ctc2', 'sprint', 'viterbi', 'fast_bw'), 'invalid loss: ' + self.loss
    self.y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim = 2)
    if not self.attrs.get("apply_softmax", True):
      self.p_y_given_x_flat = self.y_m
      self.p_y_given_x = self.z
      self.z = T.log(self.z)
      self.y_m = T.log(self.y_m)
    else:
      self.p_y_given_x_flat = T.nnet.softmax(self.y_m)
      self.p_y_given_x = T.reshape(T.nnet.softmax(self.y_m), self.z.shape)
    self.y_pred = T.argmax(self.p_y_given_x_flat, axis=-1)
    self.output = self.p_y_given_x.reshape(self.output.shape)
    if self.attrs.get('compute_priors', False):
      exp_average = self.attrs.get("compute_priors_exp_average", 0)
      self.priors = self.add_param(theano.shared(numpy.ones((self.attrs['n_out'],), 'float32') / self.attrs['n_out'], 'priors'), 'priors',
                                   custom_update=T.mean(self.p_y_given_x_flat[self.i], axis=0),
                                   custom_update_normalized=not exp_average,
                                   custom_update_exp_average=exp_average)
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
    # In case that our target has another index, self.index will be that index.
    # However, the right index for self.p_y_given_x and many others is the index from the source layers.
    src_index = self.sources[0].index
    if self.loss == 'sprint':
      if not isinstance(self.sprint_opts, dict):
        import json
        self.sprint_opts = json.loads(self.sprint_opts)
      assert isinstance(self.sprint_opts, dict), "you need to specify sprint_opts in the output layer"
      if self.exp_normalize:
        log_probs = T.log(self.p_y_given_x)
      else:
        log_probs = self.z
      if self.prior_scale: # use own priors, assume prior scale in sprint config to be 0.0
        assert self.log_prior is not None
        log_probs -= numpy.float32(self.prior_scale) * self.log_prior
      err, grad = sprint_loss_and_error_signal(
        output_layer=self,
        target=self.attrs.get("target", "classes"),
        sprint_opts=self.sprint_opts,
        log_posteriors=log_probs,
        seq_lengths=T.sum(src_index, axis=0)
      )
      err = err.sum()
      if self.loss_like_ce:
        y_ref = T.clip(self.p_y_given_x - grad, numpy.float32(0), numpy.float32(1))
        err = -T.sum(T.switch(T.cast(src_index, "float32").dimshuffle(0, 1, 'x'),
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
      y = self.p_y_given_x
      assert y.ndim == 3
      nlog_scores = -T.log(y)  # in -log space
      am_scores = nlog_scores
      am_scale = self.attrs.get("am_scale", 1)
      if am_scale != 1:
        am_scale = numpy.float32(am_scale)
        am_scores *= am_scale
      if self.prior_scale:
        assert self.log_prior is not None
        # Scores are in -log space, self.log_prior is in +log space.
        # We want to subtract the prior, thus `-=`.
        am_scores -= -self.log_prior * numpy.float32(self.prior_scale)
      edges, weights, start_end_states, state_buffer = SprintAlignmentAutomataOp(self.sprint_opts)(self.network.tags)
      float_idx = T.cast(src_index, "float32")
      float_idx_bc = float_idx.dimshuffle(0, 1, 'x')
      idx_sum = T.sum(float_idx)
      fwdbwd = FastBaumWelchOp.make_op()(am_scores, edges, weights, start_end_states, float_idx, state_buffer)
      gamma = self.attrs.get("gamma", 1)
      need_renorm = False
      if gamma != 1:
        fwdbwd *= numpy.float32(gamma)
        need_renorm = True
      bw = T.exp(-fwdbwd)
      if self.attrs.get("bw_norm_class_avg", False):
        cavg = T.sum(bw * float_idx_bc, axis=(0, 1), keepdims=True) / idx_sum
        bw /= T.clip(cavg, numpy.float32(1.e-20), numpy.float(1.e20))
        need_renorm = True
      if need_renorm:
        bw /= T.clip(T.sum(bw, axis=2, keepdims=True), numpy.float32(1.e-20), numpy.float32(1.e20))
      err = (bw * nlog_scores * float_idx_bc).sum()
      known_grads = {self.z: (y - bw) * float_idx_bc}
      if self.prior_scale and self.attrs.get('trained_softmax_prior', False):
        bw_sum0 = T.sum(bw * float_idx_bc, axis=(0, 1))
        assert bw_sum0.ndim == self.priors.ndim == 1
        # Note that this is the other way around as usually (`bw - y` instead of `y - bw`).
        # That is because the prior is in the denominator.
        known_grads[self.trained_softmax_prior_p] = numpy.float32(self.prior_scale) * (bw_sum0 - self.priors * idx_sum)
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
      nlog_scores = T.log(self.p_y_given_x) - self.prior_scale * T.log(self.priors)
      y = NumpyAlignOp(False)(src_index, self.index, -nlog_scores, self.y)
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
  def __init__(self, base, prior_scale=0.0, prior_confidence=0.0, posterior_confidence=0.0, entropy_target=0.0, variance_scale=0.0, decay=0.55, **kwargs):
    kwargs['loss'] = 'ce'
    super(UnsupervisedOutputLayer, self).__init__(**kwargs)
    if base:
      self.set_attr('base', base[0].name)
    self.set_attr('prior_scale', prior_scale)
    self.set_attr('variance_scale', variance_scale)
    self.set_attr('prior_confidence', prior_confidence)
    self.set_attr('posterior_confidence', posterior_confidence)
    self.set_attr('entropy_target', entropy_target)
    self.lm_score = T.constant(0.0,'float32')
    z_f = self.z.reshape((self.z.shape[0] * self.z.shape[1], self.z.shape[2]))
    self.y_m = z_f
    i_f = T.cast(self.index.flatten(), 'float32').dimshuffle(0, 'x').repeat(z_f.shape[1], axis=1)
    self.p = self.output.reshape(z_f.shape) #T.nnet.softmax(z_f)
    self.N = T.maximum(T.sum(i_f), numpy.float32(1))
    if base is None:
      custom = T.mean(self.p[self.i], axis=0) if self.attrs.get('trainable',True) else T.constant(0,'float32')
      self.priors = self.add_param(theano.shared(numpy.zeros((self.attrs['n_out'],), 'float32'), 'priors'), 'priors',
                                   custom_update=custom,
                                   custom_update_normalized=True and self.attrs.get('trainable',True))
    self.running_priors = self.add_param(
      theano.shared(numpy.ones((self.attrs['n_out'],), 'float32') / numpy.float32(self.attrs['n_out']),
                    'running_priors'), 'running_priors',
      custom_update=T.mean(self.p[self.i],axis=0),
      custom_update_normalized=True)

    from TheanoUtil import print_to_file
    eps = T.constant(1e-30, 'float32')
    if decay > 0:
      from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
      srng = RandomStreams(self.rng.randint(1234) + 1)
      std = decay #numpy.float32(1) / (T.cast(self.network.epoch,'float32') ** decay)
      #noise = srng.normal(size=self.p.shape, avg=0, std=std, dtype='float32')**2
      ##noise = print_to_file('noise', noise)
      #self.p += noise
      #self.p = self.p / self.p.sum(axis=1, keepdims=True)
      #std = print_to_file('noise', std)
      self.p *= srng.binomial(self.p.shape,p=(1. - decay * self.train_flag)) + eps
      self.p = self.p / self.p.sum(axis=1, keepdims=True)

    self.p = print_to_file('probs', self.p, True)

    if base is not None:
      if self.attrs['prior_confidence'] == 'inf':
        self.lm_score = -T.sum(T.log(T.max(base[0].output, axis=2, keepdims=True)))
      else:
        p = self.p
        #p = base[0].output.reshape(self.p.shape)
        if self.attrs['prior_confidence'] != 1:
          p = p ** T.constant(prior_confidence, 'float32')
          p = p / p.sum(axis=1, keepdims=True)
        #self.lm_score = -T.sum(base[0].output.reshape(p.shape) * i_f * T.log(p + eps))
        #self.lm_score += -T.sum(p * i_f * T.log(base[0].output.reshape(p.shape) + eps))
        #self.lm_score = T.sum(p * i_f * T.log(p/base[0].output.reshape(p.shape) + eps))
        #self.lm_score = T.sum(base[0].output.reshape(p.shape) * i_f * (T.log(base[0].output.reshape(p.shape) + eps) - T.log(p + eps)))
        self.clm = base[0].output
        self.plm = base[1].output
        self.lmp = base[0].output
        self.lm_score = T.sum(
          p * i_f * (T.log(p + eps) - T.log(base[0].output.reshape(p.shape) + eps)))
        self.ppl = -T.sum(base[0].output.reshape(p.shape) * i_f * T.log(base[0].output.reshape(p.shape) + eps))
    else:
      self.lm_score = 0.0
      assert prior_scale == 0.0

  def cost(self):
    known_grads = None
    eps = T.constant(1e-30, 'float32')
    if self.train_flag or True:
      i_f = T.cast(self.index.flatten(), 'float32').dimshuffle(0, 'x').repeat(self.p.shape[1], axis=1)
      prior_scale = T.constant(self.attrs['prior_scale'],'float32')
      entropy_scale = T.constant(1. - self.attrs['prior_scale'],'float32')
      variance_scale = T.constant(self.attrs['variance_scale'], 'float32')
      entropy_target = T.constant( self.attrs['entropy_target'],'float32') * self.ppl
      p = self.p
      if self.attrs['posterior_confidence'] == 'inf':
        H = -T.sum(T.cast(self.index.flatten(), 'float32') * T.log(T.max(self.p,axis=1)))
      else:
        if self.attrs['posterior_confidence'] != 1:
          p = self.p**T.constant(self.attrs['posterior_confidence'],'float32')
          p = p / p.sum(axis=1,keepdims=True)
        H = -T.sum(p * i_f * T.log(self.p + T.constant(1e-30,'float32')))
      L = self.lm_score
      from TheanoUtil import print_to_file
      #H = print_to_file('H', H)
      #entropy_target = print_to_file('entropy_target', entropy_target)
      #batch_p = T.mean(self.p, axis=0, keepdims=True)
      #U = T.sum(numpy.float32(1) - (self.p - batch_p)**2)
      #U = -T.sum(T.log(self.p * batch_p + eps))
      #U = T.sum(batch_p * i_f * (T.log(batch_p + eps) - T.log(self.lmp + eps)))
      q = self.p.reshape(self.z.shape) + eps
      q = q / q.sum(axis=1,keepdims=True)
      #q = print_to_file('q', q)
      r = (q / q.sum(axis=2, keepdims=True)).reshape(self.p.shape)
      #U = T.log(T.constant(self.attrs['n_out'])) -T.sum(q * T.log(q+eps)) * T.log(T.constant(self.attrs['n_out'])) / T.log(T.cast(q.shape[1], 'float32'))
      #U = -T.sum(q * T.log(q)) #/ T.log(T.cast(q.shape[1], 'float32') + 1) #* T.log(T.constant(self.attrs['n_out'])) / T.log(T.cast(q.shape[1], 'float32') + eps)
      U = -T.sum(T.log(T.max(q,axis=2,keepdims=True)))
      #U = T.sum(T.var(self.p.reshape(self.z.shape),axis=1))
      #U += T.sum(r * i_f * (T.log(r) - T.log(self.lmp.reshape(p.shape) + eps)))
      #batch_entropy = -T.sum(batch_p * i_f * T.log(batch_p + T.constant(1e-30, 'float32')))
      #U = T.sum(batch_p * i_f * T.log(self.p + T.constant(1e-30, 'float32'))) + batch_entropy
      #L = -T.sum(self.priors * T.log(batch_prior)) * self.N
      #Q = -T.sum(self.priors.dimshuffle('x',0).repeat(self.p.shape[0],axis=0) * i_f * T.log(self.p))
      #H = theano.printing.Print("H")(H)
      #L = theano.printing.Print("L")(L)
      #return entropy_scale * H + lm_scale * self.lm_score, known_grads
      #return entropy_scale * H + prior_scale * Q + lm_scale * self.lm_score, known_grads
      #return H, known_grads
      #return H * L, known_grads
      #U = T.sum(T.constant(1.,'float32') - T.var(self.p,axis=0,keepdims=True))
      U = print_to_file('U', U)
      #L = -T.sum(T.log(T.sum(self.lmp * self.output,axis=2)))
      hyp = self.y_in['zpi'].flatten()
      #plm = self.plm.reshape(self.p.shape)[hyp].dimshuffle(0,'x').repeat(p.shape[1],axis=1)
      #clm = self.clm.reshape(self.p.shape)[hyp].dimshuffle(0,'x').repeat(p.shape[1],axis=1)

      clm = self.clm.reshape(p.shape)[T.arange(p.shape[0],dtype='int32'),hyp].reshape(self.index.shape)
      plm = self.plm.reshape(p.shape)[T.arange(p.shape[0],dtype='int32'),hyp].reshape(self.index.shape)
      pcx = p[T.arange(p.shape[0],dtype='int32'),hyp].reshape(self.index.shape)
      #return -T.sum(T.log(clm)) - T.sum(T.log((self.p[T.arange(self.p.shape[0],dtype='int32'),hyp] - T.log(plm)))), known_grads
      #return -T.sum(T.prod(clm,axis=0) * T.prod(pcx / plm,axis=0)), known_grads
      return -T.sum(T.sum(T.log(clm), axis=0) + T.sum(T.log(pcx / plm), axis=0)), known_grads
      return entropy_scale * T.maximum(H - entropy_target,numpy.float32(0)) + prior_scale * L + variance_scale * U, known_grads
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
