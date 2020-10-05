
from __future__ import print_function

import numpy
import os
from theano import tensor as T
import theano
import theano.ifelse
from returnn.theano.ops.best_path_decoder import BestPathDecodeOp
from returnn.theano.ops.two_state_best_path_decoder import TwoStateBestPathDecodeOp
from returnn.theano.ops.ctc import CTCOp
from returnn.theano.ops.two_state_hmm import TwoStateHMMOp
from returnn.theano.ops.numpy_align import NumpyAlignOp
from returnn.native_op import FastBaumWelchOp, SegmentFastBaumWelchOp, MultiEndFastBaumWelchOp
from returnn.theano.layers.base import Layer
from .hidden import CAlignmentLayer
from returnn.sprint.error_signals import sprint_loss_and_error_signal, SprintAlignmentAutomataOp
from returnn.theano.util import time_batch_make_flat, grad_discard_out_of_bound, DumpOp
from returnn.util.basic import as_str
from returnn.log import log


class OutputLayer(Layer):
  layer_class = "softmax"

  def __init__(self, loss, y, dtype=None, reshape_target=False, copy_input=None, copy_output=None, time_limit=0,
               use_source_index=False,
               auto_fix_target_length=False,
               sigmoid_outputs=False, exp_outputs=False, gauss_outputs=False, activation=None,
               prior_scale=0.0, log_prior=None, use_label_priors=0,
               compute_priors_via_baum_welch=False,
               compute_priors=False, compute_priors_exp_average=0, compute_priors_accumulate_batches=None,
               compute_distortions=False,
               softmax_smoothing=1.0, grad_clip_z=None, grad_discard_out_of_bound_z=None, normalize_length=False,
               exclude_labels=[], include_labels=[],
               apply_softmax=True, batchwise_softmax=False,
               substract_prior_from_output=False,
               input_output_similarity=None,
               input_output_similarity_scale=1,
               scale_by_error=False,
               copy_weights=False,
               target_delay=0,
               compute_sequence_weights=False,
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
    if reshape_target:
      self.set_attr("reshape_target",reshape_target)
    if grad_clip_z is not None:
      self.set_attr("grad_clip_z", grad_clip_z)
    if compute_distortions:
      self.set_attr("compute_distortions", compute_distortions)
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
    if compute_sequence_weights:
      self.set_attr('compute_sequence_weights', compute_sequence_weights)
    if not copy_input or copy_weights:
      if copy_weights:
        self.params = {}
        self.b = self.add_param(copy_input.b)
        self.W_in = [ self.add_param(W) for W in copy_input.W_in ]
        self.masks = copy_input.masks
        self.mass = copy_input.mass
      else:
        self.W_in = [self.add_param(self.create_forward_weights(source.attrs['n_out'], self.attrs['n_out'],
                                                                name="W_in_%s_%s" % (source.name, self.name)))
                     for source in self.sources]
      self.z = self.b
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
      self.params = {}
      self.z = copy_input.output
    assert self.z.ndim == 3
    if grad_clip_z is not None:
      grad_clip_z = numpy.float32(grad_clip_z)
      self.z = theano.gradient.grad_clip(self.z, -grad_clip_z, grad_clip_z)
    if grad_discard_out_of_bound_z is not None:
      grad_discard_out_of_bound_z = numpy.float32(grad_discard_out_of_bound_z)
      self.z = grad_discard_out_of_bound(self.z, -grad_discard_out_of_bound_z, grad_discard_out_of_bound_z)
    if auto_fix_target_length:
      self.set_attr("auto_fix_target_length", auto_fix_target_length)
      source_index = self.sources[0].index
      from returnn.theano.util import pad
      self.index = pad(source=self.index, axis=0, target_axis_len=source_index.shape[0])
      if y is not None:
        y = pad(source=y, axis=0, target_axis_len=source_index.shape[0])
    if not copy_output:
      self.y = y
      self.norm = numpy.float32(1)
    else:
      if hasattr(copy_output, 'index_out'):
        self.norm = T.sum(self.index, dtype='float32') / T.sum(copy_output.index_out, dtype='float32')
        self.index = copy_output.index_out
      else:
        self.norm = T.sum(self.index, dtype='float32') / T.sum(copy_output.index, dtype='float32')
        self.index = copy_output.index
      self.y = y = copy_output.y_out
      self.copy_output = copy_output
    if y is None:
      self.y_data_flat = None
    elif isinstance(y, T.Variable):
      if reshape_target:
          if copy_output:
            if isinstance(copy_output,CAlignmentLayer):
              ind = copy_output.reduced_index.T.flatten()
              self.y_data_flat = y.T.flatten()
              self.y_data_flat = self.y_data_flat[(ind > 0).nonzero()]
              self.index = T.ones((self.z.shape[0], self.z.shape[1]), 'int8')
            else:
              self.y_data_flat = time_batch_make_flat(y)
              #self.y_data_flat = theano.printing.Print('ydataflat',attrs=['shape'])(self.y_data_flat)
          else:
            src_index = self.sources[0].index
            self.index = src_index
            self.y_data_flat = y.T.flatten()
            self.y_data_flat = self.y_data_flat[(self.y_data_flat >= 0).nonzero()]
      else:
        self.y_data_flat = time_batch_make_flat(y)
    else:
      assert self.attrs.get("target", "").endswith("[sparse:coo]")
      assert isinstance(self.y, tuple)
      assert len(self.y) == 3
      s0, s1, weight = self.y
      from returnn.theano.native_op import max_and_argmax_sparse
      n_time = self.z.shape[0]
      n_batch = self.z.shape[1]
      mask = self.network.j[self.attrs.get("target", "").replace("[sparse:coo]", "[sparse:coo:2:0]")]
      out_arg = T.zeros((n_time, n_batch), dtype="float32")
      out_max = T.zeros((n_time, n_batch), dtype="float32") - numpy.float32(1e16)
      out_arg, out_max = max_and_argmax_sparse(s0, s1, weight, mask, out_arg, out_max)
      assert out_arg.ndim == 2
      self.y_data_flat = out_arg.astype("int32")
    self.target_index = self.index
    if time_limit == 'inf':
      num = T.cast(T.sum(self.index), 'float32')
      if self.eval_flag:
        self.index = self.sources[0].index
      else:
        padx = T.zeros((T.abs_(self.index.shape[0] - self.z.shape[0]), self.index.shape[1], self.z.shape[2]),
                       'float32') + self.z[-1]
        pady = T.zeros((T.abs_(self.index.shape[0] - self.z.shape[0]), self.index.shape[1]), 'int32')  # + y[-1]
        padi = T.ones((T.abs_(self.index.shape[0] - self.z.shape[0]), self.index.shape[1]), 'int8')
        self.z = theano.ifelse.ifelse(T.lt(self.z.shape[0], self.index.shape[0]),
                                      T.concatenate([self.z, padx], axis=0), self.z)
        self.y_data_flat = time_batch_make_flat(theano.ifelse.ifelse(T.gt(self.z.shape[0], self.index.shape[0]),
                                                                     T.concatenate([y, pady], axis=0), y))
        self.index = theano.ifelse.ifelse(T.gt(self.z.shape[0], self.index.shape[0]),
                                          T.concatenate([padi, self.index], axis=0), self.index)
      self.norm *= num / T.cast(T.sum(self.index), 'float32')
    elif time_limit > 0:
      end = T.min([self.z.shape[0], T.constant(time_limit, 'int32')])
      num = T.cast(T.sum(self.index), 'float32')
      self.index = T.set_subtensor(self.index[end:], T.zeros_like(self.index[end:]))
      self.norm = num / T.cast(T.sum(self.index), 'float32')
      self.z = T.set_subtensor(self.z[end:], T.zeros_like(self.z[end:]))

    if target_delay > 0:
      self.z = T.concatenate([self.z[target_delay:],self.z[-1].dimshuffle('x',0,1).repeat(target_delay,axis=0)],axis=0)

    self.set_attr('from', ",".join([s.name for s in self.sources]))
    index_flat = self.index.flatten()
    assert not (exclude_labels and include_labels)
    if include_labels:
      exclude_labels = [ i for i in range(self.attrs['n_out']) if not i in include_labels ]
    assert len(exclude_labels) < self.attrs['n_out']
    for label in exclude_labels:
      index_flat = T.set_subtensor(index_flat[(T.eq(self.y_data_flat, label) > 0).nonzero()], numpy.int8(0))
    self.i = (index_flat > 0).nonzero()
    self.j = ((numpy.int32(1) - index_flat) > 0).nonzero()
    self.loss = as_str(loss.encode("utf8"))
    self.attrs['loss'] = self.loss
    if softmax_smoothing != 1.0:
      self.attrs['softmax_smoothing'] = softmax_smoothing
      print("Logits before the softmax scaled with factor ", softmax_smoothing, file=log.v4)
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
      from returnn.theano.util import self_similarity_cosine
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

    if sigmoid_outputs:
      self.set_attr("sigmoid_outputs", sigmoid_outputs)
    if exp_outputs:
      self.set_attr("exp_outputs", exp_outputs)
    if gauss_outputs:
      self.set_attr("gauss_outputs", gauss_outputs)
    if activation:
      self.set_attr("activation", activation)

    self.y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim=2)
    if self.loss == 'sse' or not self.attrs.get("apply_softmax", True):
      self.p_y_given_x = self.z
    elif self.loss == 'sse_sigmoid':
      self.p_y_given_x = T.nnet.sigmoid(self.z)
    elif exp_outputs:  # or not exp_normalize:
      self.p_y_given_x = T.exp(self.z)
    elif sigmoid_outputs:
      self.p_y_given_x = T.nnet.sigmoid(self.z)
    elif gauss_outputs:
      self.p_y_given_x = T.exp(-T.sqr(self.z))
    elif activation:
      from returnn.theano.activation_functions import strtoact_single_joined
      act_f = strtoact_single_joined(activation)
      self.p_y_given_x = act_f(self.z)
    elif batchwise_softmax:
      n_frames   = self.z.shape[0]
      n_batches  = self.z.shape[1]
      n_features = self.z.shape[2]
      y_m = T.switch(T.eq(self.index.reshape((n_frames, n_batches, 1)), 0), float('-inf'), self.z)
      time_major = y_m.dimshuffle(1, 0, 2).reshape((n_batches, n_frames * n_features))
      softmax = T.nnet.softmax(time_major)
      self.p_y_given_x = softmax.reshape((n_batches, n_frames, n_features)).dimshuffle(1, 0, 2)
    else:  # standard case
      self.p_y_given_x = T.reshape(T.nnet.softmax(self.y_m), self.z.shape)
    if self.loss == "priori":
      self.p_y_given_x /= self.priori
    self.p_y_given_x_flat = T.reshape(self.p_y_given_x, self.y_m.shape)
    self.y_pred = T.argmax(self.p_y_given_x_flat, axis=-1)
    self.output = self.p_y_given_x

    self.prior_scale = prior_scale
    if prior_scale:
      self.set_attr("prior_scale", prior_scale)
    if log_prior is not None:
      # We expect a filename to the priors, stored as txt, in +log space.
      assert isinstance(log_prior, str)
      self.set_attr("log_prior", log_prior)
      from returnn.util.basic import load_txt_vector
      assert os.path.exists(log_prior)
      log_prior = load_txt_vector(log_prior)
      assert len(log_prior) == self.attrs['n_out'], "dim missmatch: %i != %i" % (len(log_prior), self.attrs['n_out'])
      log_prior = numpy.array(log_prior, dtype="float32")
    self.log_prior = log_prior
    if compute_priors_via_baum_welch:
      self.set_attr("compute_priors_via_baum_welch", compute_priors_via_baum_welch)
      assert compute_priors
    if compute_priors:
      self.set_attr('compute_priors', compute_priors)
      if compute_priors_exp_average:
        self.set_attr('compute_priors_exp_average', compute_priors_exp_average)
      if compute_priors_accumulate_batches:
        self.set_attr("compute_priors_accumulate_batches", compute_priors_accumulate_batches)
      custom = T.mean(self.p_y_given_x_flat[(self.sources[0].index.flatten()>0).nonzero()], axis=0)

      custom_init = numpy.ones((self.attrs['n_out'],), 'float32') / numpy.float32(self.attrs['n_out'])
      if use_label_priors > 0:  # use labels to compute priors in first epoch
        self.set_attr("use_label_priors", use_label_priors)
        custom_0 = T.mean(theano.tensor.extra_ops.to_one_hot(self.y_data_flat[self.i], self.attrs['n_out'], 'float32'),
                          axis=0)
        custom = T.switch(T.le(self.network.epoch, use_label_priors), custom_0, custom)
      self.priors = self.add_param(theano.shared(custom_init, 'priors'), 'priors',
                                   custom_update=custom,
                                   custom_update_normalized=not compute_priors_exp_average,
                                   custom_update_exp_average=compute_priors_exp_average,
                                   custom_update_accumulate_batches=compute_priors_accumulate_batches)
      self.log_prior = T.log(T.maximum(self.priors, numpy.float32(1e-20)))

    if self.attrs.get("substract_prior_from_output", False):
      log_out = T.log(T.clip(self.output, numpy.float32(1.e-20), numpy.float(1.e20)))
      prior_scale = numpy.float32(self.attrs.get("prior_scale", 1))
      self.output = T.exp(log_out - self.log_prior * prior_scale)
      self.p_y_given_x = self.output
      self.p_y_given_x_flat = T.reshape(self.p_y_given_x, self.y_m.shape)

    if self.attrs.get('compute_distortions', False):
      p = self.p_y_given_x_flat[self.i]
      momentum = p[:-1] * p[1:]
      momentum = T.sum(momentum, axis=-1)
      loop = T.mean(momentum)
      forward = numpy.float32(1) - loop
      self.distortions = {
        'loop': self.add_param(theano.shared(numpy.ones((1,), 'float32') * numpy.float32(0.5), 'loop'), 'loop',
                               custom_update=loop,
                               custom_update_normalized=True),
        'forward': self.add_param(theano.shared(numpy.ones((1,), 'float32') * numpy.float32(0.5), 'forward'), 'forward',
                                  custom_update=forward,
                                  custom_update_normalized=True)
      }

    self.cost_scale_val = T.constant(1)
    if scale_by_error and self.train_flag:
      rpcx = self.p_y_given_x_flat[T.arange(self.p_y_given_x_flat.shape[0]),self.y_data_flat]
      #rpcx -= rpcx.min()
      rpcx /= rpcx.max()
      #weight = T.constant(1) - rpcx
      #weight = (T.constant(1) - self.p_y_given_x_flat[T.arange(self.p_y_given_x_flat.shape[0]),self.y_data_flat])
      #weight = weight.dimshuffle(0,'x').repeat(self.z.shape[2],axis=1).reshape(self.z.shape)
      #weight = T.cast(T.neq(T.argmax(self.p_y_given_x_flat, axis=1), self.y_data_flat), 'float32').dimshuffle(0,'x').repeat(self.z.shape[2],axis=1).reshape(self.z.shape)
      weight = T.cast(T.eq(T.argmax(self.p_y_given_x_flat, axis=1), self.y_data_flat), 'float32').dimshuffle(0,'x').repeat(self.z.shape[2], axis=1).reshape(self.z.shape)
      self.p_y_given_x = T.exp(weight * T.log(self.p_y_given_x))
      #self.z = self.p_y_given_x
      self.p_y_given_x_flat = self.p_y_given_x.reshape((self.p_y_given_x.shape[0]*self.p_y_given_x.shape[1],self.p_y_given_x.shape[2]))
      self.y_m = T.reshape(self.p_y_given_x, (self.p_y_given_x.shape[0] * self.p_y_given_x.shape[1], self.p_y_given_x.shape[2]), ndim=2)

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
    return -T.sum(self.p_y_given_x_flat[self.i] * T.log(self.p_y_given_x_flat[self.i]))

  def errors(self):
    """
    :rtype: theano.Variable
    """
    if self.attrs.get("target", "") == "null":
      return None
    if self.loss in [ "sse", "entropy" ]:
      return None
    if self.y_data_flat.dtype.startswith('int'):
      if self.y_data_flat.type == T.ivector().type:
        if self.attrs['normalize_length']:
          return self.norm * T.sum(
            T.max(T.neq(T.argmax(self.output[:self.index.shape[0]], axis=2), self.y) * T.cast(self.index, 'float32'),
                  axis=0))
        return self.norm * T.sum(T.neq(T.argmax(self.p_y_given_x_flat[self.i], axis=-1), self.y_data_flat[self.i]))
      else:
        return self.norm * T.sum(
          T.neq(T.argmax(self.p_y_given_x_flat[self.i], axis=-1), T.argmax(self.y_data_flat[self.i], axis=-1)))
    elif self.y_data_flat.dtype.startswith('float'):
      return T.mean(T.sqr(self.p_y_given_x_flat[self.i] - self.y_data_flat.reshape(self.y_m.shape)[self.i]))
    else:
      raise NotImplementedError()


class FramewiseOutputLayer(OutputLayer):

  def cost(self):
    """
    :rtype: (theano.Variable | None, dict[theano.Variable,theano.Variable] | None)
    :returns: cost, known_grads
    """
    if self.loss == "none":
      return None, None
    known_grads = None
    if not self.attrs.get("apply_softmax", True):
      assert self.p_y_given_x_flat.ndim == 2 \
             and self.y_data_flat.ndim == 2  # flattened
      if self.loss == "ce":
        index = T.cast(self.index, "float32").flatten()
        index_bc = index.dimshuffle(0, 'x')
        y_idx = self.y_data_flat
        assert y_idx.ndim == 1
        p = T.clip(self.p_y_given_x_flat, numpy.float32(1.e-38), numpy.float32(1.e20))
        from returnn.theano.native_op import subtensor_batched_index
        logp = T.log(subtensor_batched_index(p, y_idx))
        assert logp.ndim == 1
        nll = -T.sum(logp * index)
        # the grad for p is: -y_ref/p
        known_grads = {
          self.p_y_given_x_flat: -T.inv(p) * T.extra_ops.to_one_hot(self.y_data_flat, self.attrs["n_out"]) * index_bc}
        return self.norm * nll, known_grads
      elif self.loss == "sse":
        netOutput = self.p_y_given_x_flat
        groundTruth = self.y_data_flat
        sseLoss = T.mean(
          T.sum(
            T.sqr(netOutput - groundTruth),
            axis=(0,1)
          )
        )
        return sseLoss, known_grads
      else:
        raise NotImplementedError
    elif self.loss == 'ce' or self.loss == 'priori':
      if self.attrs.get("target", "").endswith("[sparse:coo]"):
        assert isinstance(self.y, tuple)
        assert len(self.y) == 3
        from returnn.theano.native_op import crossentropy_softmax_and_gradient_z_sparse
        y_mask = self.network.j[self.attrs.get("target", "").replace("[sparse:coo]", "[sparse:coo:2:0]")]
        ce, grad_z = crossentropy_softmax_and_gradient_z_sparse(
          self.z, self.index, self.y[0], self.y[1], self.y[2], y_mask)
        return self.norm * T.sum(ce), {self.z: grad_z}
      if self.y_data_flat.type == T.ivector().type:
        # Use crossentropy_softmax_1hot to have a more stable and more optimized gradient calculation.
        # Theano fails to use it automatically; I guess our self.i indexing is too confusing.
        if self.attrs.get("auto_fix_target_length"):
          from returnn.theano.util import pad
          xx = theano.ifelse.ifelse(T.lt(self.y_m[self.i].shape[0], 1), pad(self.y_m[self.i],0,1), self.y_m[self.i])
          yy = theano.ifelse.ifelse(T.lt(self.y_m[self.i].shape[0], 1), pad(self.y_data_flat[self.i],0,1), self.y_data_flat[self.i])
          nll, pcx = T.nnet.crossentropy_softmax_1hot(x=xx, y_idx=yy)
        elif self.attrs.get('compute_sequence_weights',False):
          self.y_data_flat = T.set_subtensor(self.y_data_flat[self.j],numpy.int8(0))
          nll_raw, pcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m, y_idx=self.y_data_flat)
          self.seq_weight = T.sum(nll_raw.reshape((self.z.shape[0],self.z.shape[1])),axis=0) / T.sum(self.index,axis=0,dtype='float32')
          nll = nll_raw[self.i]
        else:
          nll, pcx = T.nnet.crossentropy_softmax_1hot(x=self.y_m[self.i], y_idx=self.y_data_flat[self.i])
      else:
        target  = self.y_data_flat[self.i]
        output = T.clip(self.p_y_given_x_flat[self.i], 1.e-38, 1.e20)
        nll = -T.log(output) * target
        self.norm *= self.p_y_given_x.shape[1] * T.inv(T.sum(self.index))

      if self.attrs.get("auto_fix_target_length"):
        return self.norm * theano.ifelse.ifelse(T.eq(self.index.sum(),0), 0.0, T.sum(nll)), known_grads
      else:
        return self.norm * T.sum(nll), known_grads
    elif self.loss == 'entropy':
      he = T.nnet.softmax(self.y_m[self.i])  # (TB)
      ee = -T.sum(he * T.log(T.clip(he,numpy.float32(1e-6),numpy.float32(1.-1e-6))))
      #q = numpy.float32(0.1)
      #ee = (-T.sum(T.log(T.max(he,axis=1))) - q)**2
      return ee, known_grads
      pcx = T.clip((h_e / T.sum(h_e, axis=1, keepdims=True)).reshape(
        (self.index.shape[0], self.index.shape[1], self.attrs['n_out'])), 1.e-6, 1.e6)  # TBD
      ee = -T.sum(pcx[self.i] * T.log(pcx[self.i]))  # TB
      return ee, known_grads
      nll, _ = T.nnet.crossentropy_softmax_1hot(x=self.y_m, y_idx=self.y_data_flat)  # TB
      ce = nll.reshape(self.index.shape) * self.index  # TB
      y = self.y_data_flat.reshape(self.index.shape) * self.index  # TB
      f = T.any(T.gt(y, 0), axis=0)  # B
      return T.sum(f * T.sum(ce, axis=0) + (1 - f) * T.sum(ee, axis=0)), known_grads
    elif self.loss == 'priori':
      pcx = self.p_y_given_x_flat[self.i, self.y_data_flat[self.i]]
      pcx = T.clip(pcx, 1.e-38, 1.e20)  # For pcx near zero, the gradient will likely explode.
      return -T.sum(T.log(pcx)), known_grads
    elif self.loss == 'sse':
      if self.y_data_flat.dtype.startswith('int'):
        y_f = T.cast(T.reshape(self.y_data_flat, (self.y_data_flat.shape[0] * self.y_data_flat.shape[1]), ndim=1),
                     'int32')
        y_oh = T.eq(T.shape_padleft(T.arange(self.attrs['n_out']), y_f.ndim), T.shape_padright(y_f, 1))
        return T.mean(T.sqr(self.p_y_given_x_flat[self.i] - y_oh[self.i])), known_grads
      else:
        return T.sum(
          T.mean(T.sqr(self.y_m[self.i] - self.y_data_flat.reshape(self.y_m.shape)[self.i]), axis=1)), known_grads
    elif self.loss == 'sse_sigmoid':
      return 1.0 / 2.0 * T.nnet.binary_crossentropy(T.clip(self.p_y_given_x_flat[self.i], 1.e-38, 1.0 - 1.e-5), self.y_data_flat[self.i]).mean(), known_grads
    elif self.loss == 'sigmoid_binary_crossentropy':
      from theano.tensor.extra_ops import to_one_hot
      z_s = T.nnet.sigmoid(self.y_m)
      self.y_s = z_s.reshape(self.z.shape)
      return T.nnet.binary_crossentropy(T.clip(z_s, 1.e-5, 1 - 1.e-5)[self.i], to_one_hot(self.y_data_flat[self.i],self.attrs['n_out'])).sum(), known_grads
    elif self.loss == "generic_ce":
      # Should be generic for any activation function.
      # (Except when the labels are not independent, such as for softmax.)
      y = self.p_y_given_x  # Can be anything, e.g. exp or sigmoid, but not softmax.
      y /= T.sum(y, axis=2, keepdims=True)
      nlog_scores = -T.log(T.clip(y, numpy.float32(1.e-20), numpy.float(1.e20)))
      from returnn.theano.util import class_idx_seq_to_1_of_k
      y_idx = self.y
      assert y_idx.ndim == 2
      bw = class_idx_seq_to_1_of_k(y_idx, num_classes=self.attrs["n_out"])
      assert bw.ndim == 3
      err_inner = bw * nlog_scores
      src_index = self.sources[0].index
      float_idx = T.cast(src_index, "float32")
      float_idx_bc = float_idx.dimshuffle(0, 1, 'x')
      err = (err_inner * float_idx_bc).sum()
      grad_f = T.grad(None, self.z, known_grads={T.log(self.p_y_given_x): T.ones(y.shape, y.dtype)})
      known_grads = {self.z: grad_f * (y - bw) * float_idx_bc}
      return err, known_grads
    else:
      assert False, "unknown loss: %s. maybe fix LayerNetwork.make_classifier" % self.loss

  def cost_scale(self):
    return self.cost_scale_val * T.constant(self.attrs.get("cost_scale", 1.0), dtype="float32")


class DecoderOutputLayer(FramewiseOutputLayer):  # must be connected to a layer with self.W_lm_in
  #  layer_class = "decoder"

  def __init__(self, **kwargs):
    kwargs['loss'] = 'ce'
    super(DecoderOutputLayer, self).__init__(**kwargs)
    self.set_attr('loss', 'decode')

    output = 0
    self.y_s = []
    for s in self.sources:
      self.y_s.append(T.dot(s.output, s.W_lm_in) + s.b_lm_in)
      output += self.y_s[-1]
    self.params = {}
    self.y_m = output.reshape((output.shape[0] * output.shape[1], output.shape[2]))
    h = T.exp(self.y_m)
    self.p_y_given_x = T.nnet.softmax(self.y_m)
    self.y_pred = T.argmax(self.y_m[self.i], axis=1, keepdims=True)
    self.output = self.p_y_given_x.reshape(self.output.shape)

  def cost(self):
    res = 0.0
    for s in self.y_s:
      nll, pcx = T.nnet.crossentropy_softmax_1hot(x=s.reshape((s.shape[0] * s.shape[1], s.shape[2]))[self.i],
                                                  y_idx=self.y_data_flat[self.i])
      res += T.sum(nll)
    return res / float(len(self.y_s)), None


class SequenceOutputLayer(OutputLayer):
  def __init__(self,
               ce_smoothing=0.0, ce_target_layer_align=None,
               am_scale=1, gamma=1, bw_norm_class_avg=False,
               fast_bw_opts=None, seg_fast_bw_opts=None,
               loss_like_ce=False, trained_softmax_prior=False,
               sprint_opts=None, warp_ctc_lib=None,
               **kwargs):
    if fast_bw_opts is None: fast_bw_opts = {}
    if seg_fast_bw_opts is None: seg_fast_bw_opts = {}
    self._handle_old_kwargs(kwargs, fast_bw_opts=fast_bw_opts)
    super(SequenceOutputLayer, self).__init__(**kwargs)

    self.ce_smoothing = ce_smoothing
    if ce_smoothing:
      self.set_attr("ce_smoothing", ce_smoothing)
    if ce_target_layer_align:
      self.set_attr("ce_target_layer_align", ce_target_layer_align)

    if fast_bw_opts:
      if not isinstance(fast_bw_opts, dict):
        import json
        fast_bw_opts = json.loads(fast_bw_opts)
      self.set_attr("fast_bw_opts", fast_bw_opts)
    from returnn.util.basic import CollectionReadCheckCovered
    self.fast_bw_opts = CollectionReadCheckCovered(fast_bw_opts or {})

    if not isinstance(seg_fast_bw_opts, dict):
      import json
      seg_fast_bw_opts = json.loads(seg_fast_bw_opts)
    self.set_attr("seg_fast_bw_opts", seg_fast_bw_opts)
    self.seg_fast_bw_opts = seg_fast_bw_opts

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

    if sprint_opts is not None:
      if not isinstance(sprint_opts, dict):
        import json
        sprint_opts = json.loads(sprint_opts)
      self.set_attr("sprint_opts", sprint_opts)
    self.sprint_opts = sprint_opts

    if warp_ctc_lib:
      self.set_attr("warp_ctc_lib", warp_ctc_lib)
    assert self.loss in (
      'ctc', 'ce_ctc', 'hmm', 'ctc2', 'sprint', 'viterbi', 'fast_bw', 'seg_fast_bw', 'lf_mmi', 'ctc_warp', 'ctc_rasr', 'inv'), 'invalid loss: ' + self.loss

  def _handle_old_kwargs(self, kwargs, fast_bw_opts):
    if "loss_with_softmax_prob" in kwargs:
      fast_bw_opts["loss_with_softmax_prob"] = kwargs.pop("loss_with_softmax_prob")

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
    float_idx = T.cast(src_index, "float32")
    float_idx_bc = float_idx.dimshuffle(0, 1, 'x')
    idx_sum = T.sum(float_idx)
    if self.loss == 'sprint':
      assert isinstance(self.sprint_opts, dict), "you need to specify sprint_opts in the output layer"
      log_probs = T.log(self.p_y_given_x)
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
    elif self.loss == 'inv':
      S = 5
      N = self.index.shape[0]
      B = self.index.shape[1]
      ldx = self.y.dimshuffle('x', 0, 1).repeat(S, axis=0).reshape((N * S, B))
      scores = -T.log(self.p_y_given_x) # TBC
      #scores = theano.printing.Print("before", attrs=['shape'])(scores)
      scores, _ = theano.scan(lambda y,x: x[:,T.arange(B),y],[ldx],non_sequences=[scores])
      scores = scores.dimshuffle(0,2,1)
      #scores = theano.printing.Print("after", attrs=['shape'])(scores)
      index = self.index.dimshuffle('x', 0, 1).repeat(S, axis=0).reshape((N * S, B))
      edges, weights, start_end_states, state_buffer = SprintAlignmentAutomataOp(self.sprint_opts)(self.network.tags)
      #from returnn.theano.util import print_to_file
      #edges = theano.printing.Print("edges", attrs=['shape'])(edges)
      #weights = theano.printing.Print("weights", attrs=['shape'])(weights)
      fwdbwd, _ = FastBaumWelchOp().make_theano_op()(scores, edges, weights, start_end_states, T.cast(index, 'float32'), state_buffer)
      def viterbi(op,x):
        print(x.argmin(axis=-1))
      #fwdbwd = theano.printing.Print(global_fn=viterbi)(fwdbwd)
      #fwdbwd.argmin(axis=-1).flatten()
      idx = (index.flatten() > 0).nonzero()
      err = T.exp(-fwdbwd) * scores
      return T.constant(1./S,dtype='float32') * T.sum(err.reshape((err.shape[0] * err.shape[1], err.shape[2]))[idx]), None
    elif self.loss == 'ctc_rasr':
      idx = (src_index.flatten() > 0).nonzero()
      emissions = self.p_y_given_x
      #if self.attrs.get('compute_priors', False):
      #  emissions = T.exp(T.log(emissions) - self.prior_scale * T.log(T.maximum(self.priors, 1e-10)))
      scores = -T.log(emissions.reshape(self.z.shape))
      edges, weights, start_end_states, state_buffer = SprintAlignmentAutomataOp(self.sprint_opts)(self.network.tags)
      fwdbwd, _ = FastBaumWelchOp().make_theano_op()(scores, edges, weights, start_end_states, float_idx, state_buffer)
      err = T.exp(-fwdbwd) * scores
      return T.sum(err.reshape((err.shape[0]*err.shape[1],err.shape[2]))[idx]), None
    elif self.loss == 'fast_bw':
      if self.fast_bw_opts.get("bw_from"):
        out2 = self.fast_bw_opts.get("bw_from")
        bw = self.network.output[out2].baumwelch_alignment
        obs_scores = self.network.output[out2].obs_scores
      else:
        def get_am_scores(layer):
          y = layer.p_y_given_x
          assert y.ndim == 3
          if layer.fast_bw_opts.get("merge_y_from"):
            factor = layer.fast_bw_opts.get("merge_y_from_factor", 0.5)
            out2 = layer.fast_bw_opts.get("merge_y_from")
            y2 = layer.network.output[out2].p_y_given_x
            y = numpy.float32(factor) * y2 + numpy.float32(1.0 - factor) * y
          if layer.fast_bw_opts.get("y_gauss_blur_sigma"):
            from returnn.theano.util import gaussian_filter_1d
            y = gaussian_filter_1d(y, axis=0,
              sigma=numpy.float32(layer.fast_bw_opts["y_gauss_blur_sigma"]),
              window_radius=int(layer.fast_bw_opts.get("y_gauss_blur_window", layer.fast_bw_opts["y_gauss_blur_sigma"])))
          if layer.fast_bw_opts.get("y_lower_clip"):
            y = T.maximum(y, numpy.float32(layer.fast_bw_opts.get("y_lower_clip")))
          y = T.clip(y, numpy.float32(1.e-20), numpy.float(1.e20))
          nlog_scores = -T.log(y)  # in -log space
          am_scores = nlog_scores
          am_scale = layer.attrs.get("am_scale", 1)
          if am_scale != 1:
            am_scale = numpy.float32(am_scale)
            am_scores *= am_scale
          if layer.prior_scale and not layer.attrs.get("substract_prior_from_output", False):
            assert layer.log_prior is not None
            # Scores are in -log space, self.log_prior is in +log space.
            # We want to subtract the prior, thus `-=`.
            am_scores -= -layer.log_prior * numpy.float32(layer.prior_scale)
          return am_scores
        am_scores = get_am_scores(self)
        if self.fast_bw_opts.get("merge_am_from"):
          factor = self.fast_bw_opts.get("merge_am_from_factor", 0.5)
          out2 = self.fast_bw_opts.get("merge_am_from")
          am2 = get_am_scores(self.network.output[out2])
          am_scores = numpy.float32(factor) * am2 + numpy.float32(1.0 - factor) * am_scores
        if self.fast_bw_opts.get("fsa_source", "sprint") == "sprint":
          assert isinstance(self.sprint_opts, dict), "you need to specify sprint_opts in the output layer"
          edges, weights, start_end_states, state_buffer = SprintAlignmentAutomataOp(self.sprint_opts)(self.network.tags)
        elif self.fast_bw_opts.get("fsa_source") == "ctc_from_uniq_y":
          from returnn.util.fsa import ctc_fsa_for_label_seq
          num_lables = self.network.n_out[self.attrs["target"]][0]
          assert self.attrs["n_out"] == num_lables + 1  # one added for blank
          from returnn.util.basic import uniq
          from theano.compile.ops import as_op  # http://deeplearning.net/software/theano/extending/extending_theano.html#as-op
          @as_op(itypes=[theano.tensor.fmatrix, theano.tensor.fmatrix],
                 otypes=[theano.tensor.fmatrix])  # TODO...
          def fsa_op(labels, index_mask):
            """
            :param numpy.ndarray labels: (time,batch) -> label index
            :param numpy.ndarray index_mask: shape (time,batch) -> 0 or 1
            :return: (edges, weights, start_end_states)  # TODO of shape...?
            :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
            """
            assert index_mask.ndim == labels.ndim == 2
            assert index_mask.shape == labels.shape
            for batch in range(index_mask.shape[1]):
              sub_labels = labels[:, batch][index_mask[:, batch].nonzero()]
              sub_labels = uniq(sub_labels)
              num_states, edges = ctc_fsa_for_label_seq(num_labels=num_lables, label_seq=sub_labels)
            # TODO...
          edges, weights, start_end_states = fsa_op(self.y, self.target_index)
          state_buffer = T.zeros()  # TODO...
        elif self.fast_bw_opts.get("fsa_source") == "ctc_from_chars":
          from returnn.util.fsa import ctc_fsa_for_label_seq
          num_lables = self.network.n_out[self.attrs["target"]][0]
          assert self.attrs["n_out"] == num_lables + 1  # one added for blank
          from returnn.util.basic import uniq
          def get_seq_labels(seq_name):
            pass  # TODO... maybe from file? or corpus? or sprint?
          from theano.compile.ops import \
            as_op  # http://deeplearning.net/software/theano/extending/extending_theano.html#as-op
          @as_op(itypes=[theano.tensor.fmatrix, theano.tensor.fmatrix],
                 otypes=[theano.tensor.fmatrix])  # TODO...
          def fsa_op(tags):
            """
            :param numpy.ndarray tags: seq names (frame,batch) ... TODO...
            :return: (edges, weights, start_end_states)  # TODO of shape...?
            :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray)
            """
            assert tags.ndim == 2
            for batch in range(tags.shape[1]):
              labels = get_seq_labels(seq_name=tags[:, batch])  # TODO...?
              num_states, edges = ctc_fsa_for_label_seq(num_labels=num_lables, label_seq=labels)
              # TODO...
          edges, weights, start_end_states = fsa_op(self.y, self.target_index)
          state_buffer = T.zeros()  # TODO...
        else:
          raise Exception("invalid fsa_source %r" % self.fast_bw_opts.get("fsa_source"))
        fwdbwd, obs_scores = FastBaumWelchOp().make_theano_op()(am_scores, edges, weights, start_end_states, float_idx, state_buffer)
        gamma = self.attrs.get("gamma", 1)
        need_renorm = False
        if gamma != 1:
          fwdbwd *= numpy.float32(gamma)
          need_renorm = True
        bw = T.exp(-fwdbwd)
        if self.attrs.get("compute_priors_via_baum_welch", False):
          assert self.priors.custom_update is not None
          self.priors.custom_update = T.sum(bw * float_idx_bc, axis=(0, 1)) / idx_sum
        if self.fast_bw_opts.get("bw_norm_class_avg"):
          cavg = T.sum(bw * float_idx_bc, axis=(0, 1), keepdims=True) / idx_sum
          bw /= T.clip(cavg, numpy.float32(1.e-20), numpy.float(1.e20))
          need_renorm = True
        if need_renorm:
          bw /= T.clip(T.sum(bw, axis=2, keepdims=True), numpy.float32(1.e-20), numpy.float32(1.e20))
      self.baumwelch_alignment = bw
      self.obs_scores = obs_scores
      if self.ce_smoothing > 0:
        target_layer = self.attrs.get("ce_target_layer_align", None)
        assert target_layer  # we could also use self.y but so far we only want this
        bw2 = self.network.output[target_layer].baumwelch_alignment
        bw = numpy.float32(self.ce_smoothing) * bw2 + numpy.float32(1 - self.ce_smoothing) * bw
      y = self.p_y_given_x
      if self.fast_bw_opts.get("loss_with_softmax_prob"):
        y = T.reshape(T.nnet.softmax(self.y_m), self.z.shape)
      if self.fast_bw_opts.get("loss_with_sigmoid_prob"):
        y = T.nnet.sigmoid(self.z)
      if self.fast_bw_opts.get("loss_with_out_norm"):
        y /= T.sum(y, axis=2, keepdims=True)
      nlog_scores = -T.log(T.clip(y, numpy.float32(1.e-20), numpy.float(1.e20)))
      err_inner = bw * nlog_scores
      if self.fast_bw_opts.get("log_score_penalty"):
        err_inner -= numpy.float32(self.fast_bw_opts["log_score_penalty"]) * nlog_scores
      #idx = (src_index.flatten() > 0).nonzero()
      #err = T.sum(err_inner.reshape((err_inner.shape[0]*err_inner.shape[1],err_inner.shape[2]))[idx])
      # use the log-likelihood of the sequence as the error output
      if self.fast_bw_opts.get("use_obs_score_as_error"):
        err = (obs_scores * T.cast(self.index,'float32') / T.sum(self.index, axis=0, dtype='float32', keepdims=True)).sum()
      else:
        err = (err_inner * float_idx_bc).sum()
      known_grads = {self.z: (y - bw) * float_idx_bc}
      if self.fast_bw_opts.get("gauss_grad"):
        known_grads[self.z] *= -2 * self.z
      if self.fast_bw_opts.get("generic_act_grad"):  # maybe use together with loss_with_out_norm
        known_grads[self.z] *= T.grad(None, self.z, known_grads={T.log(self.p_y_given_x): T.ones(y.shape, y.dtype)})
      if self.fast_bw_opts.get("no_explicit_z_grad"):
        del known_grads[self.z]
      if self.prior_scale and self.attrs.get('trained_softmax_prior', False):
        bw_sum0 = T.sum(bw * float_idx_bc, axis=(0, 1))
        assert bw_sum0.ndim == self.priors.ndim == 1
        # Note that this is the other way around as usually (`bw - y` instead of `y - bw`).
        # That is because the prior is in the denominator.
        known_grads[self.trained_softmax_prior_p] = numpy.float32(self.prior_scale) * (bw_sum0 - self.priors * idx_sum)
      self.fast_bw_opts.assert_all_read()
      return err, known_grads

    elif self.loss == 'seg_fast_bw':
      am_score_scales      = self.seg_fast_bw_opts.get('am_score_scales', [1.0])
      const_gradient_scale = self.seg_fast_bw_opts.get('const_gradient_scale', 1.0)
      length_models        = self.seg_fast_bw_opts.get('length_models', [])
      scale_gradient       = self.seg_fast_bw_opts.get('scale_gradient', False)
      state_models         = self.seg_fast_bw_opts.get('state_models', None)

      # support for legacy parameters
      if 'loop_emission_idxs' in self.seg_fast_bw_opts:
        loop_emission_idxs   = self.seg_fast_bw_opts.get('loop_emission_idxs', [])
        loop_scores          = self.seg_fast_bw_opts.get('loop_scores', (0.0, 0.0))
        state_model = { leidx : ('loop', 1, loop_scores[0], loop_scores[1]) for leidx in loop_emission_idxs }

      segment_layer = self.network.hidden[self.seg_fast_bw_opts['segment_layer']]
      batch_idxs = segment_layer.batch_idxs
      bw_args = { 'segmentwise_normalization' : self.seg_fast_bw_opts.get('segmentwise_normalization', False),
                  'dump_targets_interval'     : self.seg_fast_bw_opts.get('dump_targets_interval', None) }

      assert len(am_score_scales) > 0

      class BuildSimpleFsaOp(theano.Op):
        itypes = (T.imatrix,)
        # the first and last output are actually uint32
        otypes = (T.fmatrix, T.fvector, T.fmatrix)

        def __init__(self, state_models=None):
          if state_models is None:
            state_models = {}

          self.state_models = state_models

        def perform(self, node, inputs, output_storage, params=None):
          labels = inputs[0]

          from_states = []
          to_states = []
          emission_idxs = []
          seq_idxs = []
          weights = []
          start_end_states = []

          cur_state = 0
          edges = []
          weights = []
          start_end_states = []
          for b in range(labels.shape[1]):
            seq_start_state = cur_state
            for l in range(labels.shape[0]):
              label = labels[l, b]
              if label < 0:
                continue
              state_model = self.state_models.get(labels[l, b], ('default', 0, 0.0))
              params = state_model[1:]
              state_model = state_model[0]
              if state_model == 'default':
                # default state model where we transition to the next label
                length_model, edge_weight = params
                edges.append((cur_state, cur_state + 1, label, length_model, b))
                weights.append(edge_weight)
                cur_state += 1
              elif state_model == 'loop':
                # allow looping in the current state before proceeding to the next one
                length_model, fwd_score, loop_score = params
                edges.append((cur_state, cur_state, label, length_model, b))
                weights.append(loop_score)
                edges.append((cur_state, cur_state + 1, label, length_model, b))
                weights.append(fwd_score)
                cur_state += 1
              elif state_model == 'double':
                # choose between emitting the label once or twice
                lm_once, lm_twice_1, lm_twice_2, once_score, twice_score = params
                edges.append((cur_state, cur_state + 2, label, lm_once, b))
                weights.append(once_score)
                edges.append((cur_state, cur_state + 1, label, lm_twice_1, b))
                weights.append(0.5 * twice_score)
                edges.append((cur_state + 1, cur_state + 2, label, lm_twice_2, b))
                weights.append(0.5 * twice_score)
                cur_state += 2

            start_end_states.append([seq_start_state, cur_state])

            cur_state += 1

          edges = sorted(edges, key=lambda e: e[1] - e[0])

          output_storage[0][0] = numpy.asarray(edges, dtype='uint32').T.copy().view(dtype='float32')
          output_storage[1][0] = numpy.array(weights, dtype='float32')
          output_storage[2][0] = numpy.asarray(start_end_states, dtype='uint32').T.copy().view(dtype='float32')

      edges, weights, start_end_states = BuildSimpleFsaOp(state_models)(self.y)
      fwdbwd, _, pw = SegmentFastBaumWelchOp(**bw_args).make_theano_op()(self.p_y_given_x, batch_idxs, edges, weights, start_end_states,
                                                                         length_models, T.cast(segment_layer.index, 'float32'),
                                                                         am_score_scales, self.network.epoch)
      bw = T.exp(-fwdbwd)
      self.y_data_flat = bw
      nlog_scores = -T.log(T.clip(self.p_y_given_x, numpy.float32(1.e-20), numpy.float(1.e20)))

      idx = segment_layer.index.reshape((bw.shape[0], bw.shape[1], 1))
      err = bw * nlog_scores * idx
      grad = (self.p_y_given_x - bw) * idx

      if scale_gradient:
        pw  = T.clip(pw.reshape((pw.shape[0], pw.shape[1], 1)) * const_gradient_scale, 1.e-20, 1.0)
        grad *= pw
        err  *= pw

      err = err.sum()
      known_grads = { self.z: grad }

      return err, known_grads
    elif self.loss == 'lf_mmi':
      # Get AM scores for current utterances
      am_scores = -T.log(self.p_y_given_x)
      am_scale = self.attrs.get("am_scale", 1)
      if am_scale != 1:
        am_scale = numpy.float32(am_scale)
        am_scores *= am_scale

      # Get alignment FST for numerator
      if self.fast_bw_opts.get("num_fsa_source", "sprint") == "sprint":
        assert isinstance(self.sprint_opts, dict), "you need to specify sprint_opts in the output layer"
        edges, weights, start_end_states, state_buffer = SprintAlignmentAutomataOp(self.sprint_opts)(self.network.tags)
      else:
        raise Exception("invalid fsa_source %r" % self.fast_bw_opts.get("fsa_source"))

      # Calculate numerator part
      fwdbwd, obs_scores = FastBaumWelchOp().make_theano_op()(am_scores, edges, weights, start_end_states, float_idx, state_buffer)
      self.baumwelch_alignment = T.exp(-fwdbwd)
      self.num_scores = obs_scores

      def loop_fkt(some_variable, seq_index, prev_fwdbwd, prev_scores):
        # Get search FST for denominator
        if self.fast_bw_opts.get("den_fsa_source", "file") == "file":
          import os
          assert isinstance(self.fast_bw_opts.get("den_fsa_file"), str) and os.path.exists(self.fast_bw_opts.get("den_fsa_file")),\
            "you need to specify the path to the search FSA in den_fsa_file"

          class LoadWfstOp(theano.Op):
            """
            Op: maps segment names (tags) to fsa automata (load from disk) that can be used to compute a BW-alignment
            """

            __props__ = ("filename",)

            def __init__(self, filename):
              super(LoadWfstOp, self).__init__()
              from returnn.util.basic import make_hashable
              self.filename = make_hashable(filename)
              self.single_wfst = None  # type: dict

            def make_node(self, tags):
              # the edges/start_end_state output has to be a float matrix because that is the only dtype supported
              # by CudaNdarray. We need unsigned ints. Thus we return a view on the unsigned int matrix
              return theano.Apply(self, [tags],
                                  [T.fmatrix(), T.fvector(), T.fvector(), T.fmatrix(), T.fvector(), T.fmatrix()])

            def perform(self, node, inputs, output_storage, params=None):
              tags = inputs[0]
              try:
                _ = iter(tags)
              except TypeError:
                tags = [tags]

              if self.single_wfst is None:
                print("LoadWfstOp: Loading WFST from %r" % self.filename, file=log.v3)
                import xml.etree.ElementTree as ET

                tree = ET.parse(self.filename)
                root = tree.getroot()
                single_wfst = dict()
                single_wfst['edges'] = []
                single_wfst['weights'] = []
                single_wfst['start_states'] = numpy.array([root.attrib['initial']], dtype=numpy.uint32)
                single_wfst['end_states'] = []
                single_wfst['end_state_weigths'] = []
                self.single_wfst = dict()
                self.single_wfst['num_states'] = len(root)

                for state in root:
                  if state.tag != 'state':
                    continue  # not interested in input-alphabet
                  state_id = numpy.uint32(state.attrib['id'])
                  if state[0].tag == 'final':
                    single_wfst['end_states'].append([numpy.uint32(0), state_id])
                    if state[1].tag == 'weight':
                      single_wfst['end_state_weigths'].append(numpy.float32(state[1].text))
                    else:
                      single_wfst['end_state_weigths'].append(numpy.float32(0.))
                  for arc in state:
                    if arc.tag != 'arc':
                      continue  # alredy handeled 'final' and 'weight'
                    target = numpy.uint32(arc.attrib['target'])
                    emission_id = numpy.uint32(arc[0].text)
                    if len(arc) > 1:
                      weight = numpy.float32(arc[1].text)
                    else:
                      weight = numpy.float32(0.)
                    single_wfst['edges'].append([state_id, target, emission_id, numpy.uint32(0)])
                    single_wfst['weights'].append(weight)
                for key, val in single_wfst.items():
                  self.single_wfst[key] = numpy.array(val)

              assert isinstance(self.single_wfst, dict)  # PyCharm confused otherwise

              offset = 0
              all_edges = []
              all_weights = []
              all_start_states = []
              all_end_states = []
              all_end_state_weigths = []
              for tag in tags:
                edges = numpy.transpose(numpy.copy(self.single_wfst['edges']))
                edges[0:2, :] += offset
                edges[3, :] = tag
                all_edges.append(edges)
                all_weights.append(self.single_wfst['weights'])
                all_start_states.append(self.single_wfst['start_states'] + offset)
                end_states = numpy.copy(self.single_wfst['end_states'])
                end_states[:, 1] += offset
                end_states[:, 0] = tag
                all_end_states.append(end_states)
                all_end_state_weigths.append(self.single_wfst['end_state_weigths'])
                offset += self.single_wfst['num_states']

              output_storage[0][0] = numpy.hstack(all_edges).view(dtype='float32')
              output_storage[1][0] = numpy.hstack(all_weights)
              output_storage[2][0] = numpy.hstack(all_start_states).view(dtype='float32')
              output_storage[3][0] = numpy.hstack(all_end_states).view(dtype='float32')
              output_storage[4][0] = numpy.hstack(all_end_state_weigths)
              output_storage[5][0] = numpy.empty((2, self.single_wfst['num_states'] * len(tags)), dtype='float32')

          edges, weights, start_states, end_states, end_state_weigths, state_buffer = LoadWfstOp(self.fast_bw_opts.get("den_fsa_file"))(seq_index)
        else:
          raise Exception("invalid fsa_source %r" % self.fast_bw_opts.get("fsa_source"))

        # Calculate denominator part
        fwdbwd, obs_scores = MultiEndFastBaumWelchOp().make_theano_op()(am_scores, edges, weights, start_states, end_states, end_state_weigths, float_idx, state_buffer)
        return T.set_subtensor(prev_fwdbwd[:,seq_index,:], fwdbwd[:,seq_index,:]) , T.set_subtensor(prev_scores[:,seq_index], obs_scores[:,seq_index])

      [foo,bar], scan_updates = theano.scan(fn=loop_fkt,
                                            outputs_info=[T.zeros_like(fwdbwd),T.zeros_like(obs_scores)],
                                            sequences=[am_scores[0],T.arange(1000)])

      [fwdbwd, obs_scores] = [foo[-1],bar[-1]]

      self.baumwelch_denominator =T.exp(-fwdbwd)
      self.den_scores = obs_scores

      # TODO: check weather loss is correct
      err = ((self.num_scores - self.den_scores) *  T.cast(self.index,'float32') / T.sum( T.cast(self.index,'float32'), axis=0, dtype='float32', keepdims=True)).sum()

      if self.fast_bw_opts.get('numerator_smoothing') :
        num = (1 - self.fast_bw_opts.get('numerator_smoothing')) * self.baumwelch_alignment + self.fast_bw_opts.get('numerator_smoothing') * T.extra_ops.to_one_hot(self.y_data_flat, self.baumwelch_alignment.shape[-1]).reshape(self.baumwelch_alignment.shape)
      else:
        num = self.baumwelch_alignment

      grad = (self.baumwelch_denominator - num) * float_idx_bc

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


      return err, {self.z: grad }

    elif self.loss == 'ctc':
      from theano.tensor.extra_ops import cpu_contiguous
      err, grad, priors = CTCOp()(self.p_y_given_x, cpu_contiguous(self.y.dimshuffle(1, 0)), self.index_for_ctc())
      known_grads = {self.z: grad * numpy.float32(self.attrs.get('cost_scale', 1))}
      self.seq_weight = err / T.sum(self.index_for_ctc(),axis=0,dtype='float32')
      return err.sum(), known_grads, priors.sum(axis=0)
    elif self.loss == 'hmm':
      from theano.tensor.extra_ops import cpu_contiguous
      emissions = self.p_y_given_x
      tdp_loop = T.as_tensor_variable(numpy.cast["float32"](0))
      tdp_fwd = T.as_tensor_variable(numpy.cast["float32"](0))
      if self.attrs.get('compute_priors', False):
        emissions = T.exp(T.log(emissions) - self.prior_scale *  T.log(T.maximum(self.priors,1e-10)))
      if self.attrs.get('compute_distortions', False):
        tdp_loop = T.as_tensor_variable(T.log(self.distortions['loop'][0]))
        tdp_fwd = T.as_tensor_variable(T.log(self.distortions['forward'][0]))
      err, grad, priors = TwoStateHMMOp()(emissions, cpu_contiguous(self.y.dimshuffle(1, 0)),
                                          self.index_for_ctc(),tdp_loop,tdp_fwd)
      known_grads = {self.z: grad * numpy.float32(self.attrs.get('cost_scale', 1))}
      return err.sum(), known_grads, priors.sum(axis=0)
    elif self.loss == 'warp_ctc':
      import os
      os.environ['CTC_LIB'] = self.attrs.get('warp_ctc_lib', "/usr/lib")
      try:
        # noinspection PyUnresolvedReferences
        from theano_ctc import ctc_cost
        # from theano_ctc.cpu_ctc import CpuCtc
      except Exception:
        assert False, "install this: https://github.com/mcf06/theano_ctc"
      from returnn.theano.util import print_to_file
      yr = T.set_subtensor(self.y.flatten()[self.j], numpy.int32(-1)).reshape(self.y.shape).dimshuffle(1, 0)
      yr = print_to_file('yr', yr)
      cost = T.mean(ctc_cost(self.p_y_given_x, yr, self.index_for_ctc()))
      cost = print_to_file('cost', cost)
      return cost, known_grads
    elif self.loss == 'ce_ctc':
      y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]), ndim=2)
      p_y_given_x = T.nnet.softmax(y_m)
      pcx = p_y_given_x[self.i, self.y_data_flat[self.i]]
      ce = -T.sum(T.log(pcx))
      return ce, known_grads
    elif self.loss == 'ctc2':
      from .ctc import ctc_cost, uniq_with_lengths, log_sum
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
    if self.loss in ('ctc', 'ce_ctc', 'ctc_warp', 'ctc_rasr', 'inv') or (self.loss == 'fast_bw' and self.fast_bw_opts.get('ctc',False)):
      from theano.tensor.extra_ops import cpu_contiguous
      emissions = self.p_y_given_x
      if self.attrs.get('compute_priors', False):
        emissions = T.exp(T.log(emissions) - self.prior_scale * T.log(T.maximum(self.priors, 1e-10)))
      return T.sum(BestPathDecodeOp()(self.p_y_given_x, cpu_contiguous(self.y.dimshuffle(1, 0)), self.index_for_ctc()))
      #return T.sum(TwoStateBestPathDecodeOp()(emissions, cpu_contiguous(self.y.dimshuffle(1, 0)), self.index_for_ctc()))
    elif self.loss == 'hmm' or (self.loss == 'fast_bw' and self.fast_bw_opts.get('decode',False)):
      emissions = self.p_y_given_x
      if self.attrs.get('compute_priors', False):
        emissions = T.exp(T.log(emissions) - self.prior_scale * T.log(T.maximum(self.priors, 1e-10)))
      from theano.tensor.extra_ops import cpu_contiguous
      return T.sum(TwoStateBestPathDecodeOp()(emissions, cpu_contiguous(self.y.dimshuffle(1, 0)), self.index_for_ctc()))
    elif self.loss == 'inv':
      return 0
    elif self.loss == 'viterbi':
      scores = T.log(self.p_y_given_x) - self.prior_scale * T.log(self.priors)
      y = NumpyAlignOp(False)(self.sources[0].index, self.index, -scores, self.y)
      self.y_data_flat = y.flatten()
      return super(SequenceOutputLayer, self).errors()
    elif self.loss == "fast_bw":
      if self.fast_bw_opts.get("fsa_source") == "ctc_from_y":
        # TODO ... use Util.uniq / TheanoUtil.uniq ....
        from theano.tensor.extra_ops import cpu_contiguous
        return T.sum(
          BestPathDecodeOp()(self.p_y_given_x, cpu_contiguous(self.y.dimshuffle(1, 0)), self.index_for_ctc()))
      elif self.fast_bw_opts.get("fsa_source") == "ctc_from_chars":
        # TODO... maybe share code with cost(). we need the same target label seq anyway.
        pass
      else:
        return super(SequenceOutputLayer, self).errors()
    elif self.loss == "seg_fast_bw":
        return None
    else:
      return super(SequenceOutputLayer, self).errors()


from returnn.theano.util import print_to_file


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

    shyp = hyp[:, :, order]
    spcx = pcx[:, :, order]

    K = numpy.float32(1. / (1. - momentum)) * T.sum(T.sum(pc * T.log(pc / shyp), axis=2), axis=0)
    Q = -T.sum(T.sum(pcx * T.log(pcx), axis=2), axis=0)

    self.L = T.sum(T.switch(domax, Q, K))
    self.y_m = spcx.reshape((spcx.shape[0] * spcx.shape[1], spcx.shape[2]))

  def cost(self):
    known_grads = None
    if self.train_flag and not self.attrs['oracle']:
      return self.L, known_grads
    else:
      p = self.y_m
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
