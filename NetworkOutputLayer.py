import numpy
import os
from theano import tensor as T
import theano
from BestPathDecoder import BestPathDecodeOp
from CTC import CTCOp
from TwoStateHMMOp import TwoStateHMMOp
from OpNumpyAlign import NumpyAlignOp
from NativeOp import FastBaumWelchOp
from NetworkBaseLayer import Layer
from SprintErrorSignals import sprint_loss_and_error_signal, SprintAlignmentAutomataOp
from TheanoUtil import time_batch_make_flat, grad_discard_out_of_bound
from Util import as_str
from Log import log


# from Accumulator import AccumulatorOpInstance

# def step(*args): # requires same amount of memory
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
               use_source_index=False,
               compute_priors=False, compute_priors_exp_average=0,
               softmax_smoothing=1.0, grad_clip_z=None, grad_discard_out_of_bound_z=None, normalize_length=False,
               exclude_labels=[],
               apply_softmax=True,
               substract_prior_from_output=False,
               input_output_similarity=None,
               input_output_similarity_scale=1,
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
    if substract_prior_from_output:
      self.set_attr("substract_prior_from_output", substract_prior_from_output)
    if input_output_similarity:
      self.set_attr("input_output_similarity", input_output_similarity)
      self.set_attr("input_output_similarity_scale", input_output_similarity_scale)
    if use_source_index:
      self.set_attr("use_source_index", use_source_index)
      src_index = self.sources[0].index
      self.index = src_index
    if not copy_input:
      self.z = self.b
      self.W_in = [self.add_param(self.create_forward_weights(source.attrs['n_out'], self.attrs['n_out'],
                                                              name="W_in_%s_%s" % (source.name, self.name)))
                   for source in self.sources]

      assert len(self.sources) == len(self.masks) == len(self.W_in)
      assert len(self.sources) > 0
      for source, m, W in zip(self.sources, self.masks, self.W_in):
        source_output = source.output
        # 4D input from TwoD Layers -> collapse height dimension
        if source_output.ndim == 4:
          source_output = source_output.sum(axis=0)
        if source.attrs['sparse']:
          if source.output.ndim == 3:
            input = source_output[:, :, 0]  # old sparse format
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
    if y is None:
      self.y_data_flat = None
    elif isinstance(y, T.Variable):
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
      # target_length = self.index.shape[0]
      # mass = T.cast(T.sum(self.index),'float32')
      # self.index = theano.ifelse.ifelse(T.gt(self.z.shape[0],target_length),self.sources[0].index,self.index)
      # self.norm = mass / T.cast(T.sum(self.index),'float32')
      num = T.cast(T.sum(self.index), 'float32')
      if self.eval_flag:
        self.index = self.sources[0].index
      else:
        import theano.ifelse
        padx = T.zeros((T.abs_(self.index.shape[0] - self.z.shape[0]), self.index.shape[1], self.z.shape[2]),
                       'float32') + self.z[-1]
        pady = T.zeros((T.abs_(self.index.shape[0] - self.z.shape[0]), self.index.shape[1]), 'int32')  # + y[-1]
        padi = T.ones((T.abs_(self.index.shape[0] - self.z.shape[0]), self.index.shape[1]), 'int8')
        self.z = theano.ifelse.ifelse(T.lt(self.z.shape[0], self.index.shape[0]),
                                      T.concatenate([self.z, padx], axis=0), self.z)
        # self.z = theano.ifelse.ifelse(T.gt(self.z.shape[0], self.index.shape[0]),self.z[:self.index.shape[0]], self.z)
        self.y_data_flat = time_batch_make_flat(theano.ifelse.ifelse(T.gt(self.z.shape[0], self.index.shape[0]),
                                                                     T.concatenate([y, pady], axis=0), y))
        # self.index = theano.ifelse.ifelse(T.gt(self.z.shape[0], self.index.shape[0]), T.concatenate([T.ones((self.z.shape[0] - self.index.shape[0],self.z.shape[1]),'int8'), self.index], axis=0), self.index)
        self.index = theano.ifelse.ifelse(T.gt(self.z.shape[0], self.index.shape[0]),
                                          T.concatenate([padi, self.index], axis=0), self.index)
      self.norm *= num / T.cast(T.sum(self.index), 'float32')
    elif time_limit > 0:
      end = T.min([self.z.shape[0], T.constant(time_limit, 'int32')])
      nom = T.cast(T.sum(self.index), 'float32')
      self.index = T.set_subtensor(self.index[end:], T.zeros_like(self.index[end:]))
      self.norm = nom / T.cast(T.sum(self.index), 'float32')
      self.z = T.set_subtensor(self.z[end:], T.zeros_like(self.z[end:]))

    # xs = [s.output for s in self.sources]
    # self.z = AccumulatorOpInstance(*[self.b] + xs + self.W_in)
    # outputs_info = None #[ T.alloc(numpy.cast[theano.config.floatX](0), index.shape[1], self.attrs['n_out']) ]

    # self.z, _ = theano.scan(step,
    #                        sequences = [s.output for s in self.sources],
    #                        non_sequences = self.W_in + [self.b])

    self.set_attr('from', ",".join([s.name for s in self.sources]))
    index_flat = self.index.flatten()
    for label in exclude_labels:
      index_flat = T.set_subtensor(index_flat[(T.eq(self.y_data_flat, label) > 0).nonzero()], numpy.int8(0))
    self.i = (index_flat > 0).nonzero()
    self.j = ((numpy.int32(1) - index_flat) > 0).nonzero()
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

    if input_output_similarity:
      # First a self-similarity of input and output,
      # and then add -similarity or distance between those to the constraints,
      # so that the input and output correlate on a frame-by-frame basis.
      # Here some other similarities/distances we could try:
      # http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
      # https://brenocon.com/blog/2012/03/cosine-similarity-pearson-correlation-and-ols-coefficients/
      from TheanoUtil import self_similarity_cosine
      self_similarity = self_similarity_cosine  # maybe other
      data_layer = self.find_data_layer()
      assert data_layer
      assert data_layer.output.ndim == 3
      n_time = data_layer.output.shape[0]
      n_batch = data_layer.output.shape[1]
      findex = T.cast(self.output_index(), "float32")
      findex_bc = findex.reshape((n_time * n_batch,)).dimshuffle(0, 'x')
      findex_sum = T.sum(findex)
      data = data_layer.output.reshape((n_time * n_batch, data_layer.output.shape[2])) * findex_bc
      assert self.z.ndim == 3
      z = self.z.reshape((n_time * n_batch, self.z.shape[2])) * findex_bc
      data_self_sim = T.flatten(self_similarity(data))
      z_self_sim = T.flatten(self_similarity(z))
      assert data_self_sim.ndim == z_self_sim.ndim == 1
      sim = T.dot(data_self_sim, z_self_sim)  # maybe others make sense
      assert sim.ndim == 0
      # sim is ~ proportional to T * T, so divide by T.
      sim *= numpy.float32(input_output_similarity_scale) / findex_sum
      self.constraints -= sim

    # self.make_output(self.z, collapse = False)
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
          return self.norm * T.sum(
            T.max(T.neq(T.argmax(self.output[:self.index.shape[0]], axis=2), self.y) * T.cast(self.index, 'float32'),
                  axis=0))
        return self.norm * T.sum(T.neq(T.argmax(self.y_m[self.i], axis=-1), self.y_data_flat[self.i]))
      else:
        return self.norm * T.sum(
          T.neq(T.argmax(self.y_m[self.i], axis=-1), T.argmax(self.y_data_flat[self.i], axis=-1)))
    elif self.y_data_flat.dtype.startswith('float'):
      return T.mean(T.sqr(self.y_m[self.i] - self.y_data_flat.reshape(self.y_m.shape)[self.i]))
    else:
      raise NotImplementedError()

  def _maybe_substract_prior_from_output(self):
    if not self.attrs.get("substract_prior_from_output", False): return
    log_out = T.log(T.clip(self.output, numpy.float32(1.e-20), numpy.float(1.e20)))
    prior_scale = numpy.float32(self.attrs.get("prior_scale", 1))
    self.output = T.exp(log_out - self.log_prior * prior_scale)
    self.p_y_given_x = self.output


class FramewiseOutputLayer(OutputLayer):
  def __init__(self, **kwargs):
    super(FramewiseOutputLayer, self).__init__(**kwargs)
    self.initialize()

  def initialize(self):
    # self.y_m = self.output.dimshuffle(2,0,1).flatten(ndim = 2).dimshuffle(1,0)
    output = self.output
    self.y_m = output.reshape((output.shape[0] * output.shape[1], output.shape[2]))
    self.y_pred = T.argmax(self.y_m[self.i], axis=1, keepdims=True)
    if not self.attrs.get("apply_softmax", True):
      self.p_y_given_x = self.y_m
      self.z = T.log(self.z)
      self.y_m = T.log(self.y_m)
    elif self.loss in ['ce', 'entropy', 'none']:
      self.p_y_given_x = T.nnet.softmax(self.y_m)
    elif self.loss == 'sse':
      self.p_y_given_x = self.y_m
    elif self.loss == 'priori':
      self.p_y_given_x = T.nnet.softmax(self.y_m) / self.priori
    else:
      assert False, "invalid loss: " + self.loss
    self.p_y_given_x_flat = self.p_y_given_x  # a bit inconsistent here... it's always flat at the moment
    self.output = self.p_y_given_x.reshape(self.output.shape)
    if self.attrs.get('compute_priors', False):
      custom = T.mean(self.p_y_given_x[self.i], axis=0) if self.attrs.get('trainable', True) else T.constant(0,
                                                                                                             'float32')
      exp_average = self.attrs.get("compute_priors_exp_average", 0)
      self.priors = self.add_param(theano.shared(numpy.zeros((self.attrs['n_out'],), 'float32'), 'priors'), 'priors',
                                   custom_update=custom,
                                   custom_update_normalized=(not exp_average) and self.attrs.get('trainable', True),
                                   custom_update_exp_average=exp_average)
      self.log_prior = T.log(self.priors)
    self._maybe_substract_prior_from_output()

  def cost(self):
    """
    :rtype: (theano.Variable | None, dict[theano.Variable,theano.Variable] | None)
    :returns: cost, known_grads
    """
    if self.loss == "none":
      return None, None
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
      known_grads = {
        self.p_y_given_x: -T.inv(p) * T.extra_ops.to_one_hot(self.y_data_flat, self.attrs["n_out"]) * index_bc}
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
        # idx = self.index.flatten().dimshuffle(0,'x').repeat(self.y_m.shape[1],axis=1) # faster than line below
        # nll, pcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m * idx, y_idx=self.y_data_flat * self.index.flatten())
        nll, pcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m[self.i], y_idx=self.y_data_flat[self.i])
        # nll, pcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m, y_idx=self.y_data_flat)
        # nll = -T.log(T.nnet.softmax(self.y_m)[self.i,self.y_data_flat[self.i]])
        # z_c = T.exp(self.z[:,self.y])
        # nll = -T.log(z_c / T.sum(z_c,axis=2,keepdims=True))
        # nll, pcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m, y_idx=self.y_data_flat)
        # nll = T.set_subtensor(nll[self.j], T.constant(0.0))
      else:
        nll = -T.dot(T.log(T.clip(self.p_y_given_x[self.i], 1.e-38, 1.e20)), self.y_data_flat[self.i].T)
      return self.norm * T.sum(nll), known_grads
    elif self.loss == 'entropy':
      h_e = T.exp(self.y_m)  # (TB)
      pcx = T.clip((h_e / T.sum(h_e, axis=1, keepdims=True)).reshape(
        (self.index.shape[0], self.index.shape[1], self.attrs['n_out'])), 1.e-6, 1.e6)  # TBD
      ee = -T.sum(pcx[self.i] * T.log(pcx[self.i]))  # TB
      # nll, pcxs = T.nnet.crossentropy_softmax_1hot(x=self.y_m[self.i], y_idx=self.y[self.i])
      nll, _ = T.nnet.crossentropy_softmax_1hot(x=self.y_m, y_idx=self.y_data_flat)  # TB
      ce = nll.reshape(self.index.shape) * self.index  # TB
      y = self.y_data_flat.reshape(self.index.shape) * self.index  # TB
      f = T.any(T.gt(y, 0), axis=0)  # B
      return T.sum(f * T.sum(ce, axis=0) + (1 - f) * T.sum(ee, axis=0)), known_grads
      # return T.sum(T.switch(T.gt(T.sum(y,axis=0),0), T.sum(ce, axis=0), -T.sum(ee, axis=0))), known_grads
      # return T.switch(T.gt(T.sum(self.y_m[self.i]),0), T.sum(nll), -T.sum(pcx * T.log(pcx))), known_grads
    elif self.loss == 'priori':
      pcx = self.p_y_given_x[self.i, self.y_data_flat[self.i]]
      pcx = T.clip(pcx, 1.e-38, 1.e20)  # For pcx near zero, the gradient will likely explode.
      return -T.sum(T.log(pcx)), known_grads
    elif self.loss == 'sse':
      if self.y_data_flat.dtype.startswith('int'):
        y_f = T.cast(T.reshape(self.y_data_flat, (self.y_data_flat.shape[0] * self.y_data_flat.shape[1]), ndim=1),
                     'int32')
        y_oh = T.eq(T.shape_padleft(T.arange(self.attrs['n_out']), y_f.ndim), T.shape_padright(y_f, 1))
        return T.mean(T.sqr(self.p_y_given_x[self.i] - y_oh[self.i])), known_grads
      else:
        # return T.sum(T.sum(T.sqr(self.y_m - self.y.reshape(self.y_m.shape)), axis=1)[self.i]), known_grads
        return T.sum(
          T.mean(T.sqr(self.y_m[self.i] - self.y_data_flat.reshape(self.y_m.shape)[self.i]), axis=1)), known_grads
        # return T.sum(T.sum(T.sqr(self.z - (self.y.reshape((self.index.shape[0], self.index.shape[1], self.attrs['n_out']))[:self.z.shape[0]])), axis=2).flatten()[self.i]), known_grads
        # y_z = T.set_subtensor(T.zeros((self.index.shape[0],self.index.shape[1],self.attrs['n_out']), dtype='float32')[:self.z.shape[0]], self.z).flatten()
        # return T.sum(T.sqr(y_z[self.i] - self.y[self.i])), known_grads
        # return T.sum(T.sqr(self.y_m - self.y[:self.z.shape[0]*self.index.shape[1]]).flatten()[self.i]), known_grads
    else:
      assert False, "unknown loss: %s. maybe fix LayerNetwork.make_classifier" % self.loss


class DecoderOutputLayer(FramewiseOutputLayer):  # must be connected to a layer with self.W_lm_in
  #  layer_class = "decoder"

  def __init__(self, **kwargs):
    kwargs['loss'] = 'ce'
    super(DecoderOutputLayer, self).__init__(**kwargs)
    self.set_attr('loss', 'decode')

  def cost(self):
    res = 0.0
    for s in self.y_s:
      nll, pcx = T.nnet.crossentropy_softmax_1hot(x=s.reshape((s.shape[0] * s.shape[1], s.shape[2]))[self.i],
                                                  y_idx=self.y_data_flat[self.i])
      res += T.sum(nll)
    return res / float(len(self.y_s)), None

  def initialize(self):
    output = 0
    self.y_s = []
    # i = T.cast(self.index.dimshuffle(0,1,'x').repeat(self.attrs['n_out'],axis=2),'float32')
    for s in self.sources:
      self.y_s.append(T.dot(s.output, s.W_lm_in) + s.b_lm_in)
      output += self.y_s[-1]
    self.params = {}
    self.y_m = output.reshape((output.shape[0] * output.shape[1], output.shape[2]))
    h = T.exp(self.y_m)
    self.p_y_given_x = T.nnet.softmax(self.y_m)  # h / h.sum(axis=1,keepdims=True) #T.nnet.softmax(self.y_m)
    self.y_pred = T.argmax(self.y_m[self.i], axis=1, keepdims=True)
    self.output = self.p_y_given_x.reshape(self.output.shape)


class SequenceOutputLayer(OutputLayer):
  def __init__(self, prior_scale=0.0, log_prior=None, use_label_priors=0,
               compute_priors_via_baum_welch=False,
               ce_smoothing=0.0, ce_target_layer_align=None,
               exp_normalize=True,
               am_scale=1, gamma=1, bw_norm_class_avg=False,
               sigmoid_outputs=False, exp_outputs=False, loss_with_softmax_prob=False,
               loss_like_ce=False, trained_softmax_prior=False,
               sprint_opts=None, warp_ctc_lib=None,
               **kwargs):
    super(SequenceOutputLayer, self).__init__(**kwargs)
    self.prior_scale = prior_scale
    if use_label_priors:
      self.set_attr("use_label_priors", use_label_priors)
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
    if compute_priors_via_baum_welch:
      self.set_attr("compute_priors_via_baum_welch", compute_priors_via_baum_welch)
      assert self.attrs.get("compute_priors", False)
    self.log_prior = log_prior
    self.ce_smoothing = ce_smoothing
    if ce_smoothing:
      self.set_attr("ce_smoothing", ce_smoothing)
    if ce_target_layer_align:
      self.set_attr("ce_target_layer_align", ce_target_layer_align)
    self.exp_normalize = exp_normalize
    if not exp_normalize:
      self.set_attr("exp_normalize", exp_normalize)
    if sigmoid_outputs:
      self.set_attr("sigmoid_outputs", sigmoid_outputs)
    if exp_outputs:
      self.set_attr("exp_outputs", exp_outputs)
    if loss_with_softmax_prob:
      self.set_attr("loss_with_softmax_prob", loss_with_softmax_prob)
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
    if warp_ctc_lib:
      self.set_attr("warp_ctc_lib", warp_ctc_lib)
    self.initialize()

  def initialize(self):
    assert self.loss in (
      'ctc', 'ce_ctc', 'hmm', 'ctc2', 'sprint', 'viterbi', 'fast_bw', 'warp_ctc'), 'invalid loss: ' + self.loss
    self.y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim=2)
    if not self.attrs.get("apply_softmax", True):
      self.p_y_given_x_flat = self.y_m
      self.p_y_given_x = self.z
      self.z = T.log(self.z)
      self.y_m = T.log(self.y_m)
    else:
      self.p_y_given_x_flat = T.nnet.softmax(self.y_m)
      self.p_y_given_x = T.reshape(T.nnet.softmax(self.y_m), self.z.shape)
    self.y_pred = T.argmax(self.p_y_given_x_flat, axis=-1)
    self.output = self.p_y_given_x
    if self.attrs.get('compute_priors', False):
      exp_average = self.attrs.get("compute_priors_exp_average", 0)
      custom = T.mean(self.p_y_given_x_flat[self.i], axis=0)
      custom_init = numpy.ones((self.attrs['n_out'],), 'float32') / numpy.float32(self.attrs['n_out'])
      if self.attrs.get('use_label_priors', 0) > 0:  # use labels to compute priors in first epoch
        custom_0 = T.mean(theano.tensor.extra_ops.to_one_hot(self.y_data_flat[self.i], self.attrs['n_out'], 'float32'),
                          axis=0)
        custom = T.switch(T.le(self.network.epoch, self.attrs.get('use_label_priors', 0)), custom_0, custom)
        custom_init = numpy.zeros((self.attrs['n_out'],), 'float32')
      self.priors = self.add_param(theano.shared(custom_init, 'priors'), 'priors',
                                   custom_update=custom,
                                   custom_update_normalized=not exp_average,
                                   custom_update_exp_average=exp_average)
      self.log_prior = T.log(T.maximum(self.priors, numpy.float32(1e-20)))
    self._maybe_substract_prior_from_output()

  def index_for_ctc(self):
    for source in self.sources:
      if hasattr(source, "output_sizes"):
        return T.cast(source.output_sizes[:, 1], "int32")
    return T.cast(T.sum(T.cast(self.sources[0].index, 'int32'), axis=0), 'int32')

  def output_index(self):
    for source in self.sources:
      if hasattr(source, "output_sizes"):
        return source.index
    if self.loss in ['viterbi', 'ctc', 'hmm', 'warp_ctc']:
      return self.sources[0].index
    return super(SequenceOutputLayer, self).output_index()

  def cost(self):
    """
    :param y: shape (time*batch,) -> label
    :return: error scalar, known_grads dict
    """
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
      if self.prior_scale:  # use own priors, assume prior scale in sprint config to be 0.0
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
      if self.attrs.get("sigmoid_outputs", False):
        y = T.nnet.sigmoid(self.z)
      assert y.ndim == 3
      y = T.clip(y, numpy.float32(1.e-20), numpy.float(1.e20))
      nlog_scores = -T.log(y)  # in -log space
      if self.attrs.get("exp_outputs", False):
        y = T.exp(self.z)
        nlog_scores = -self.z  # in -log space
      am_scores = nlog_scores
      am_scale = self.attrs.get("am_scale", 1)
      if am_scale != 1:
        am_scale = numpy.float32(am_scale)
        am_scores *= am_scale
      if self.prior_scale and not self.attrs.get("substract_prior_from_output", False):
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
      if self.attrs.get("compute_priors_via_baum_welch", False):
        assert self.priors.custom_update is not None
        self.priors.custom_update = T.sum(bw * float_idx_bc, axis=(0, 1)) / idx_sum
      if self.attrs.get("bw_norm_class_avg", False):
        cavg = T.sum(bw * float_idx_bc, axis=(0, 1), keepdims=True) / idx_sum
        bw /= T.clip(cavg, numpy.float32(1.e-20), numpy.float(1.e20))
        need_renorm = True
      if need_renorm:
        bw /= T.clip(T.sum(bw, axis=2, keepdims=True), numpy.float32(1.e-20), numpy.float32(1.e20))
      self.baumwelch_alignment = bw
      if self.ce_smoothing > 0:
        target_layer = self.attrs.get("ce_target_layer_align", None)
        assert target_layer  # we could also use self.y but so far we only want this
        bw2 = self.network.output[target_layer].baumwelch_alignment
        bw = numpy.float32(self.ce_smoothing) * bw2 + numpy.float32(1 - self.ce_smoothing) * bw
      if self.attrs.get("loss_with_softmax_prob", False):
        y = self.p_y_given_x
        nlog_scores = -T.log(T.clip(y, numpy.float32(1.e-20), numpy.float(1.e20)))
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
    elif self.loss == 'hmm':
      from theano.tensor.extra_ops import cpu_contiguous
      err, grad, priors = TwoStateHMMOp()(self.p_y_given_x, cpu_contiguous(self.y.dimshuffle(1, 0)),
                                          self.index_for_ctc())
      known_grads = {self.z: grad}
      return err.sum(), known_grads, priors.sum(axis=0)
    elif self.loss == 'warp_ctc':
      import os
      os.environ['CTC_LIB'] = self.attrs.get('warp_ctc_lib', "/usr/lib")
      try:
        from theano_ctc import ctc_cost
        # from theano_ctc.cpu_ctc import CpuCtc
      except Exception:
        assert False, "install this: https://github.com/mcf06/theano_ctc"
      from TheanoUtil import print_to_file
      yr = T.set_subtensor(self.y.flatten()[self.j], numpy.int32(-1)).reshape(self.y.shape).dimshuffle(1, 0)
      yr = print_to_file('yr', yr)
      cost = T.mean(ctc_cost(self.p_y_given_x, yr, self.index_for_ctc()))
      # cost = T.mean(CpuCtc()(self.p_y_given_x, yr, self.index_for_ctc()))
      cost = print_to_file('cost', cost)
      return cost, known_grads
    elif self.loss == 'ce_ctc':
      y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim=2)
      p_y_given_x = T.nnet.softmax(y_m)
      # pcx = p_y_given_x[(self.i > 0).nonzero(), y_f[(self.i > 0).nonzero()]]
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
    if self.loss in ('ctc', 'ce_ctc', 'ctc_warp'):
      from theano.tensor.extra_ops import cpu_contiguous
      return T.sum(BestPathDecodeOp()(self.p_y_given_x, cpu_contiguous(self.y.dimshuffle(1, 0)), self.index_for_ctc()))
    elif self.loss == 'hmm':
      #TODO
      return T.as_tensor_variable(numpy.cast["float32"](0))
    elif self.loss == 'viterbi':
      scores = T.log(self.p_y_given_x) - self.prior_scale * T.log(self.priors)
      y = NumpyAlignOp(False)(self.sources[0].index, self.index, -scores, self.y)
      self.y_data_flat = y.flatten()
      return super(SequenceOutputLayer, self).errors()
    else:
      return super(SequenceOutputLayer, self).errors()


from TheanoUtil import print_to_file


class UnsupervisedOutputLayer(OutputLayer):
  def __init__(self, base, momentum=0.1, oracle=False, msteps=100, esteps=200, **kwargs):
    kwargs['loss'] = 'ce'
    super(UnsupervisedOutputLayer, self).__init__(**kwargs)
    if base:
      self.set_attr('base', base[0].name)
    self.set_attr('momentum', momentum)
    self.set_attr('oracle', oracle)
    self.set_attr('msteps', msteps)
    self.set_attr('esteps', esteps)
    eps = T.constant(1e-30, 'float32')
    pc = theano.gradient.disconnected_grad(base[1].output)  # TBV
    pc = print_to_file('pc', pc)
    pcx = base[0].output  # TBV

    self.cnt = self.add_param(theano.shared(numpy.zeros((1,), 'float32'), 'cnt'),
                              custom_update=T.constant(1, 'float32'))
    domax = T.ge(T.mod(T.cast(self.cnt[0], 'int32'), numpy.int32(msteps + esteps)), esteps)

    hyp = T.mean(pcx, axis=1, keepdims=True)
    hyp = hyp / hyp.sum(axis=2, keepdims=True)

    self.hyp = self.add_param(
      theano.shared(numpy.ones((self.attrs['n_out'],), 'float32') / numpy.float32(self.attrs['n_out']), 'hyp'), 'hyp',
      custom_update=T.mean(hyp[:, 0, :], axis=0),
      custom_update_condition=domax,
      custom_update_normalized=True,
      custom_update_exp_average=1. / (1. - momentum))
    hyp = numpy.float32(1. - momentum) * hyp + numpy.float32(momentum) * self.hyp.dimshuffle('x', 'x', 0).repeat(
      hyp.shape[1], axis=1).repeat(hyp.shape[0], axis=0)

    order = T.argsort(self.hyp)[::-1]
    # order = print_to_file('order', order)

    shyp = hyp[:, :, order]
    spcx = pcx[:, :, order]

    # spcx = print_to_file('pcx', spcx)
    # shyp = print_to_file('shyp', shyp)

    K = numpy.float32(1. / (1. - momentum)) * T.sum(T.sum(pc * T.log(pc / shyp), axis=2), axis=0)
    Q = -T.sum(T.sum(pcx * T.log(pcx), axis=2), axis=0)

    # K = print_to_file('K', K)
    # Q = print_to_file('Q', Q)

    self.L = T.sum(T.switch(domax, Q, K))
    self.y_m = spcx.reshape((spcx.shape[0] * spcx.shape[1], spcx.shape[2]))

  def cost(self):
    known_grads = None
    if self.train_flag and not self.attrs['oracle']:
      return self.L, known_grads
    else:
      p = self.y_m
      # p = self.x_m / (self.x_m.sum(axis=1, keepdims=True) + self.punk)
      # if self.attrs['oracle'] and self.train_flag:
      #  p = self.x_m / (self.x_m.sum(axis=1,keepdims=True) + self.punk)
      nll, _ = T.nnet.crossentropy_softmax_1hot(x=p[self.i], y_idx=self.y_data_flat[self.i])
      return T.sum(nll), known_grads

  def errors(self):
    """
    :rtype: theano.Variable
    """
    if self.y_data_flat.type == T.ivector().type:
      return self.norm * T.sum(T.neq(T.argmax(self.y_m[self.i], axis=-1), self.y_data_flat[self.i]))
    else:
      return self.norm * T.sum(T.neq(T.argmax(self.y_m[self.i], axis=-1), T.argmax(self.y_data_flat[self.i], axis=-1)))
