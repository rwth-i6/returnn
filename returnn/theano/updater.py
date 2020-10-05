
from __future__ import print_function

import theano
import numpy
import os
from returnn.log import log
from math import sqrt
import theano.tensor.shared_randomstreams
import theano.tensor as T
import theano.ifelse
import theano.compile
from returnn.theano.util import opt_contiguous_on_gpu


class Updater:

  """
  This defines how to update the model parameters per mini-batch.
  All kind of gradient-based optimization methods are implemented here, such as Adam etc.
  """

  @classmethod
  def initFromConfig(cls, config):
    """
    Will construct a :class:`Updater` instance where all params are automatically determined by the given config.

    :param Config.Config config:
    :rtype: Updater
    """
    kwargs = {}
    for k, v in cls._get_kwarg_defaults().items():
      if type(v) == bool: g = config.bool
      elif type(v) == float: g = config.float
      elif type(v) == int: g = config.int
      else: assert False, "invalid default type: %s = (%s) %s" % (k, type(v), v)
      kwargs[k] = g(k, v)
    return cls(**kwargs)

  @classmethod
  def initRule(cls, rule, **kwargs):
    if rule != "default":
      kwargs[rule] = True
    return cls(**kwargs)

  @classmethod
  def _get_kwarg_defaults(cls):
    import inspect
    arg_spec = inspect.getargspec(cls.__init__)
    N_defs = len(arg_spec.defaults)
    N_args = len(arg_spec.args)
    defaults = {arg_spec.args[N_args - N_defs + i]: d for i, d in enumerate(arg_spec.defaults)}
    return defaults

  # Note that the default value type is important for initFromConfig to determine
  # whether to call config.bool/config.int/etc.
  def __init__(self,
               momentum=0.0, nesterov_momentum=0.0, momentum2=0.0,
               gradient_clip=-1.0,
               update_clip=-1.0,
               weight_clip=0.0,
               adagrad=False,
               adadelta=False, adadelta_decay=0.90, adadelta_offset=1e-6,
               max_norm=0.0,
               adasecant=False,
               adam=False,
               adamdelta=False,
               adam_fit_learning_rate=1000,
               adamax=False,
               nadam=False,
               ena=False,
               nadam_decay=0.004,  # Magical 250.0 denominator in nesterov scaling of i_t
               eve=False,
               gradient_l2_norm=0,
               mean_normalized_sgd=False,
               mean_normalized_sgd_average_interpolation=0.5,
               rmsprop=0.0,
               smorms3=False,
               update_multiple_models=0, update_multiple_models_average_step=0,
               update_multiple_models_average_step_i=0, update_multiple_models_averaging=True,
               update_multiple_models_param_is_cur_model=False,
               multi_batch_update=0,
               variance_reduction=False,
               enforce_triangular_matrix_zero=False,
               gradient_noise=0.0,
               gradient_noise_decay=0.55,
               grad_noise_rel_grad_norm=0.0,
               reset_update_params=False
               ):
    """
    Initializes the Updater class.
    All the params determine the specific optimization variants and their hyper params.
    Normally this is constructed by :func:`Updater.initFromConfig`.

    :param momentum:
    :param nesterov_momentum:
    :param momentum2:
    :param gradient_clip:
    :param update_clip:
    :param adagrad:
    :param adadelta:
    :param adadelta_decay:
    :param adadelta_offset:
    :param max_norm:
    :param adasecant:
    :param adam:
    :param adamdelta:
    :param adam_fit_learning_rate:
    :param adamax:
    :param eve: Eve optimizer - Adam with a feedback from loss
    :param adamvr: Adam with adasecant variance reduction
    :param nadam: Adam with nag part momentum
    :param nadam_decay:
    :param mean_normalized_sgd:
    :param mean_normalized_sgd_average_interpolation:
    :param rmsprop:
    :param smorms3:
    :param update_multiple_models:
    :param update_multiple_models_average_step:
    :param update_multiple_models_average_step_i:
    :param update_multiple_models_averaging:
    :param update_multiple_models_param_is_cur_model:
    :param multi_batch_update:
    :param enforce_triangular_matrix_zero:
    :param gradient_noise:
    :param gradient_noise_decay:
    :param grad_noise_rel_grad_norm:
    :param reset_update_params:
    """
    self.rng = numpy.random.RandomState(0o101)
    self.momentum = numpy.float32(momentum)
    self.nesterov_momentum = numpy.float32(nesterov_momentum)
    self.momentum2 = numpy.float32(momentum2)
    self.gradient_clip = numpy.float32(gradient_clip)
    self.update_clip = numpy.float32(update_clip)
    self.weight_clip = numpy.float32(weight_clip)
    self.max_norm = max_norm
    self.adagrad = adagrad
    self.adadelta = adadelta
    self.adadelta_decay = numpy.float32(adadelta_decay)
    self.adadelta_offset = numpy.float32(adadelta_offset)
    self.adasecant = adasecant
    self.ena = ena
    self.nadam = nadam
    self.nadam_decay = nadam_decay
    self.adam = adam
    self.eve = eve
    self.adamdelta = adamdelta
    self.adam_fit_learning_rate = adam_fit_learning_rate
    self.variance_reduction = variance_reduction
    self.adamax = adamax
    self.mean_normalized_sgd = mean_normalized_sgd
    self.mean_normalized_sgd_average_interpolation = numpy.float32(mean_normalized_sgd_average_interpolation)
    self.rmsprop = rmsprop
    self.smorms3 = smorms3
    self.update_multiple_models = update_multiple_models
    self.update_multiple_models_averaging = update_multiple_models_averaging
    self.update_multiple_models_average_step = update_multiple_models_average_step
    self.update_multiple_models_average_step_i = update_multiple_models_average_step_i
    self.update_multiple_models_param_is_cur_model = update_multiple_models_param_is_cur_model
    self.multi_batch_update = multi_batch_update
    self.enforce_triangular_matrix_zero = enforce_triangular_matrix_zero
    self.gradient_noise = gradient_noise
    self.gradient_noise_decay = gradient_noise_decay
    self.grad_noise_rel_grad_norm = grad_noise_rel_grad_norm
    self.reset_update_params = reset_update_params
    self.gradient_l2_norm = gradient_l2_norm
    self.device = str(theano.config.device)
    self.params = {}
    self.pid = -1
    if self.adadelta:
      self.momentum = 0.0
      self.nesterov_momentum = 0.0
      self.momentum2 = 0.0
      print("using adadelta with decay", self.adadelta_decay, ", offset", self.adadelta_offset, file=log.v4)
    if self.adagrad:
      print("using adagrad", file=log.v4)
    if self.momentum:
      print("using momentum %f" % self.momentum, file=log.v4)
    if self.nesterov_momentum:
      print("using simplified nesterov momentum %f" % self.nesterov_momentum, file=log.v4)
    if self.momentum2:
      print("using reverted momentum %f" % self.momentum2, file=log.v4)
    if self.gradient_clip > 0:
      print("using gradient clipping %f" % self.gradient_clip, file=log.v4)
    if self.update_clip > 0:
      print("using update clipping %f" % self.update_clip, file=log.v4)
    if self.rmsprop:
      print("using RMSProp with rho = %f" % self.rmsprop, file=log.v4)
    if self.smorms3:
      print("using SMORMS3", file=log.v4)
    if self.adamax:
      print("using AdaMax with b1 = 0.9 and b2 = 0.999", file=log.v4)
    if self.adam:
      print("using adam", file=log.v4)
    if self.nadam:
      print("using adam with nag and momentum schedule", file=log.v4)
    if self.eve:
      print("using eve optimizer (Adam with feedback)", file=log.v4)

  def initVars(self, network, net_param_deltas):
    """
    Initializes the Theano shared variables.
    This should be called in the process where you want to do the updating.
    All further calls must be from the same process.
    The network.gparams must be created in the same process.

    :type network: Network.LayerNetwork
    :type net_param_deltas: dict[theano.compile.sharedvalue.SharedVariable,theano.Variable] | None
    """
    assert not self.isInitialized
    self.pid = os.getpid()
    self.network = network
    if net_param_deltas is not None:
      self.update_on_device = True
      self.net_train_param_deltas = net_param_deltas
    else:
      self.update_on_device = False
      self.net_train_param_deltas = {p : theano.shared(numpy.zeros(p.get_value(borrow=True,
                                                                              return_internal_type=True).shape,
                                                                  dtype=theano.config.floatX))
                                     for p in network.train_params_vars}
      " :type: dict[theano.compile.sharedvalue.SharedVariable,theano.compile.sharedvalue.SharedVariable] "
    self.learning_rate_var = theano.shared(value=numpy.cast[theano.config.floatX](0), name="learning_rate")
    " :type: theano.compile.sharedvalue.SharedVariable "
    self.i = self.var(numpy.float32(0 if self.reset_update_params else network.update_step), name="updater_i")
    self.e = self.var(numpy.float32(0 if self.reset_update_params else network.update_step), name="updater_epoch")

    if self.momentum > 0 and not self.ena:
      self.deltas = {p: self.var(p, zero=True, name="momentum_deltas_%s" % p.name)
                     for p in network.train_params_vars}

    if self.adagrad:
      self.accu = {p: self.var(p, zero=True, name="adagrad_accu_%s" % p.name)
                   for p in network.train_params_vars}

    if self.adadelta or self.adamdelta:
      # http://arxiv.org/pdf/1212.5701v1.pdf
      self.eg2 = {p: self.var(p, zero=True, name="adadelta_eg2_%s" % p.name)
                  for p in self.network.train_params_vars} #E[g^2]
      self.edx2 = {p: self.var(p, zero=True, name="adadelta_edx2_%s" % p.name)
                  for p in self.network.train_params_vars} #E[\delta x^2]
      self.dx = {p: self.var(p, zero=True, name="adadelta_dx_%s" % p.name)
                 for p in self.network.train_params_vars} #\delta x

  @property
  def isInitialized(self):
    return self.pid >= 0

  def setNetParamDeltas(self, net_param_deltas):
    assert self.pid == os.getpid()
    assert self.update_on_device == False
    for p in net_param_deltas:
      self.net_train_param_deltas[p].set_value(net_param_deltas[p], borrow=True)

  def norm_constraint(self, tensor_var, max_norm, norm_axes=None, epsilon=1e-12):
    ndim = tensor_var.ndim

    if norm_axes is not None:
        sum_over = tuple(norm_axes)
    elif ndim == 2:  # DenseLayer
        sum_over = (0,)
    elif ndim == 3:  # Depth
        sum_over = (0,2)
    else:
        sum_over = (0,)

    dtype = numpy.dtype(theano.config.floatX).type
    norms = T.sqrt(T.sum(T.sqr(tensor_var), axis=sum_over, keepdims=True))
    target_norms = T.clip(norms, 0, dtype(max_norm))
    constrained_output = \
        (tensor_var * (target_norms / (dtype(epsilon) + norms)))

    return constrained_output

  def _var_get_value(self, value, zero=False, dtype="float32"):
    if zero:
      if isinstance(value, theano.compile.SharedVariable):
        value = value.get_value(borrow=True, return_internal_type=True)
      shape = value.shape
      value = numpy.zeros(shape, dtype=dtype)
    else:
      if isinstance(value, theano.compile.SharedVariable):
        value = value.get_value()
      value = numpy.asarray(value).astype(dtype)
    return value

  def var(self, value, name="", broadcastable=None, dtype="float32", zero=False):
    orig_value = value
    if broadcastable is None and isinstance(value, theano.compile.SharedVariable):
      broadcastable = value.broadcastable
    value = self._var_get_value(value, zero=zero, dtype=dtype)
    kwargs = {"value": value}
    if name: kwargs["name"] = name
    if broadcastable: kwargs["broadcastable"] = broadcastable
    param = theano.shared(**kwargs)
    self.params[param] = {
      "value": orig_value, "zero": zero,
      "broadcastable": broadcastable, "name": name, "dtype": dtype}
    return param

  def reset(self):
    for param, info in self.params.items():
      if info["zero"]: continue
      value = info["value"]
      if isinstance(value, theano.compile.SharedVariable):
        # We copied from this shared var. This should be done here in all cases.
        # The first copy might even be invalid because networks params
        # are not loaded in the beginning, so this is important.
        value = value.get_value()
        value = numpy.asarray(value).astype(info["dtype"])
        param.set_value(value)
    if self.reset_update_params:
      # Also reset all remaining params.
      for param, info in self.params.items():
        value = info["value"]
        if info["zero"]:
          value = self._var_get_value(value, zero=True, dtype=info["dtype"])
        if isinstance(value, theano.compile.SharedVariable):
          continue  # this is handled above
        value = numpy.asarray(value).astype(info["dtype"])
        param.set_value(value)

  def getUpdateList(self):
    assert self.pid == os.getpid()
    updates = []
    " :type: list[(theano.SharedVariable, theano.Variable)] "
    upd = { p: 0 for p in self.net_train_param_deltas.keys() }
    grads = {p: T.switch(T.or_(T.isinf(g), T.isnan(g)), numpy.float32(0), g) for (p, g) in self.net_train_param_deltas.items()}
    #grads = {p: g for (p, g) in self.net_train_param_deltas.items()}

    if self.mean_normalized_sgd:
      # https://www-i6.informatik.rwth-aachen.de/publications/download/903/WieslerSimonRichardAlexerSchl%7Bu%7DterRalfNeyHermann--Mean-normalizedstochasticgradientforlarge-scaledeeplearning--2014.pdf
      assert self.update_on_device, "not implemented otherwise. we need the activation running average"
      for layer_name, layer in sorted(self.network.hidden.items()) + sorted(self.network.output.items()):
        if not hasattr(layer, "W_in"): continue
        assert len(layer.sources) == len(layer.W_in)
        all_in_train = layer.b in self.network.train_params_vars
        sparse_input = False
        for s, W in zip(layer.sources, layer.W_in):
          if W not in self.network.train_params_vars: all_in_train = False
          if s.attrs['sparse']: sparse_input = True
        if not all_in_train:
          print("Mean-normalized SGD: layer", layer_name, "not trained", file=log.v4)
          continue
        if sparse_input:
          print("Mean-normalized SGD: layer", layer_name, "has sparse input, not supported yet", file=log.v4)
          continue
        print("Mean-normalized SGD: used for W_in of layer", layer_name, file=log.v4)
        avg_f = numpy.float32(self.mean_normalized_sgd_average_interpolation)
        delta_b = grads[layer.b]
        for s, W_in in zip(layer.sources, layer.W_in):
          avg_v = self.var(numpy.zeros((s.attrs["n_out"],), dtype="float32"),
                           name="avg_%s_%s" % (s.name, layer.name))
          # Without the opt_contiguous_on_gpu, I get a crash (together with LSTMP)...
          cur_avg = T.mean(opt_contiguous_on_gpu(s.output), axis=(0, 1))
          avg = avg_f * avg_v + (numpy.float32(1.0) - avg_f) * cur_avg
          updates.append((avg_v, avg))
          grads[W_in] -= T.outer(avg, delta_b)
          grads[layer.b] -= T.dot(grads[W_in].T, avg)

    self.counter = self.var(0, name="counter", dtype="int64")
    updates.append((self.counter, self.counter + 1))
    dt = T.cast(1.,'float32') #T.cast(T.max(T.sum(self.network.output.values()[0].index,axis=0)), 'float32')
    i_t = self.i + dt #1.
    prev_epoch = self.var(numpy.zeros((), dtype="int32"),'prev_epoch',dtype='int32')
    updates.append((prev_epoch, self.network.epoch))
    e_t = T.switch(T.eq(self.network.epoch,prev_epoch), self.e + dt,
                   T.switch(T.eq(i_t, self.e), numpy.float32(1), self.e/numpy.float32(2) + dt))
    updates.append((self.e, e_t))
    beta1=numpy.float32(0.9)
    beta2=numpy.float32(0.999)
    beta3=numpy.float32(0.999)

    default_output_layer = None
    batch_num_output_frames = None
    if self.network.output:
      if "output" in self.network.output:
        default_output_layer = self.network.output["output"]
      else:
        default_output_layer = sorted(self.network.output.items())[0][1]
      batch_num_output_frames = T.sum(default_output_layer.index)

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    srng = RandomStreams(self.rng.randint(1234) + 1)
    total_grad_norm = numpy.float32(0)
    for grad in grads.values(): total_grad_norm += T.sum(grad * grad)
    n_total_params = 0
    for grad in grads.values(): n_total_params += T.prod(grad.shape)
    avg_grad_norm = total_grad_norm / T.cast(n_total_params, dtype="float32")

    for param in grads.keys():
      # This loops sets upd[param], where param_new = param + upd[param].

      if hasattr(param, 'custom_update'):
        # For this case, Device will set grads[param] to custom_update (thus like negative gradient).
        # We also don't apply the learning rate here.
        if param.custom_update_normalized:  # cumulative moving average
          upd[param] = (grads[param] - param) / e_t
        elif param.custom_update_exp_average:  # exponential moving average
          alpha = numpy.float32(param.custom_update_exp_average)
          upd[param] = alpha * (grads[param] - param)  # ((alpha - 1) * old + alpha * new)
        else:
          upd[param] = grads[param]
        if param.custom_update_condition is not None:
          upd[param] = T.switch(param.custom_update_condition, upd[param], 0)
        if param.custom_update_accumulate_batches is not None:
          assert param.custom_update_accumulate_batches >= 1
          do_update_now = T.eq(self.counter % param.custom_update_accumulate_batches, param.custom_update_accumulate_batches - 1)
          accumulated_param = self.var(param, name="%s_accumulated" % param.name, zero=True)
          accumulated_param_new = accumulated_param + upd[param]
          updates.append((
            accumulated_param,
            theano.ifelse.ifelse(
              do_update_now,
              T.zeros_like(param),
              accumulated_param_new
            )
          ))
          upd[param] = theano.ifelse.ifelse(
            do_update_now,
            accumulated_param_new / numpy.float32(param.custom_update_accumulate_batches),
            T.zeros_like(param)
          )
        continue

      if param.live_update is not None:
        continue

      if param.layer.device != self.device and param.layer.device is not None:
        grads[param] = grads[param].transfer(self.device)
      gradient_scale = param.layer.gradient_scale
      if isinstance(param.layer.gradient_scale, list):
        gradient_scale = T.switch(T.eq(T.mod(i_t - 1,gradient_scale[1]),0), gradient_scale[0], numpy.float32(0))
      deltas = grads[param] * T.cast(gradient_scale,'float32')
      if self.max_norm > 0:
        deltas = self.norm_constraint(deltas, self.max_norm)
      if self.gradient_l2_norm:
        deltas = deltas / (deltas.norm(2) + numpy.float32(1e-10))
        #deltas = T.switch(T.le(T.constant(self.gradient_l2_norm),self.network.epoch),
        #                  deltas / (deltas.norm(2) + numpy.float32(1e-10)),
        #                  deltas)

      if self.gradient_noise > 0.0: # http://arxiv.org/pdf/1511.06807v1.pdf
        nu = self.gradient_noise # try 0.01 0.3 1.0
        gamma = self.gradient_noise_decay
        sigma = nu / (1 + i_t)**gamma
        deltas += srng.normal(size=deltas.shape, ndim=deltas.ndim, avg=0.0, std=sigma, dtype="float32")
      if self.grad_noise_rel_grad_norm > 0.0:
        # Idea extended from here: RandomOut, http://arxiv.org/pdf/1602.05931v2.pdf
        # The total gradient norm is a measure how much error there is.
        # If the relative gradient norm is low, it means that this element
        # has low impact on the loss function. -> Change that, add noise.
        elemwise_grad_norm = grads[param] * grads[param]
        eps = numpy.float32(1e-7)
        rel_elemwise_grad_norm = elemwise_grad_norm - avg_grad_norm
        sigma = self.grad_noise_rel_grad_norm
        noise = srng.normal(size=deltas.shape, ndim=deltas.ndim, avg=0.0, std=sigma, dtype="float32")
        noise *= T.maximum(-rel_elemwise_grad_norm, numpy.float32(1))
        deltas += noise
      #print param, param.get_value().shape, numpy.prod(param.get_value().shape)
      if self.gradient_clip > 0:
        # Note that there is also theano.gradient.grad_clip, which would clip it already
        # at the backprop step and which would affect also other dependent gradients.
        # However, this is simpler for now.
        # Also note that this is yet without the learning rate factor -
        # this might be different to other gradient clipping implementations.
        deltas = T.clip(deltas, -self.gradient_clip, self.gradient_clip)
      #if self.momentum > 0:
      #  upd[p] += self.momentum * self.deltas[param]
      if self.variance_reduction:
        self.start_var_reduction = 0
        self.use_corrected_grad = True
        self.decay = 0.9 #75
        self.delta_clip = 0 #50.0
        self.gamma_clip = 2.5  # 1.8
        eps = numpy.float32(1e-7)
        deltas = deltas / (deltas.norm(2) + eps)

        taus_x_t = self.var((numpy.ones_like(param.get_value()) + eps) * 2.1, name="taus_x_t_" + param.name)

        # Variance reduction parameters
        # Numerator of the gamma:
        gamma_nume_sqr = self.var(numpy.zeros_like(param.get_value()) + eps, name="gamma_nume_sqr_" + param.name)
        # Denominator of the gamma:
        gamma_deno_sqr = self.var(numpy.zeros_like(param.get_value()) + eps, name="gamma_deno_sqr_" + param.name)
        # For the covariance parameter := E[\gamma \alpha]_{t-1}
        cov_num_t = self.var(numpy.zeros_like(param.get_value()) + eps, name="cov_num_t_" + param.name)
        # mean_grad := E[g]_{t-1}
        mean_grad = self.var(numpy.zeros_like(param.get_value()) + eps, name="mean_grad_%s" % param.name)
        # mean_squared_grad := E[g^2]_{t-1}
        mean_square_grad = self.var(numpy.zeros_like(param.get_value()) + eps, name="msg_" + param.name)
        # mean_square_dx := E[(\Delta x)^2]_{t-1}
        mean_square_dx = self.var(value=numpy.zeros_like(param.get_value()), name="msd_" + param.name)
        old_grad = self.var(value=numpy.zeros_like(param.get_value()) + eps, name="old_grad_" + param.name)

        # The uncorrected gradient of previous of the previous update:
        old_plain_grad = self.var(numpy.zeros_like(param.get_value()) + eps, name="old_plain_grad_" + param.name)
        mean_curvature = self.var(numpy.zeros_like(param.get_value()) + eps, name="mean_curvature_" + param.name)
        mean_curvature_sqr = self.var(numpy.zeros_like(param.get_value()) + eps,
                                      name="mean_curvature_sqr_" + param.name)

        # Initialize the E[\Delta]_{t-1}
        mean_dx = self.var(numpy.zeros_like(param.get_value()), name="mean_dx_" + param.name)

        # Block-wise normalize the gradient:
        # For the first time-step, assume that delta_x_t := deltas
        cond = T.eq(self.i, 0)
        msdx = cond * deltas ** 2 + (1 - cond) * mean_square_dx
        mdx = cond * deltas + (1 - cond) * mean_dx

        """
        Compute the new updated values.
        """
        # E[g_i^2]_t
        new_mean_squared_grad = mean_square_grad * self.decay + T.sqr(deltas) * (1 - self.decay)
        new_mean_squared_grad.name = "msg_" + param.name
        # E[g_i]_t
        new_mean_grad = mean_grad * self.decay + deltas * (1 - self.decay)
        new_mean_grad.name = "nmg_" + param.name
        # Keep the rms for numerator and denominator of gamma.
        new_gamma_nume_sqr = gamma_nume_sqr * (1 - 1 / taus_x_t) + T.sqr(
          (deltas - old_grad) * (old_grad - new_mean_grad)) / taus_x_t
        new_gamma_nume_sqr.name = "ngammasqr_num_" + param.name
        new_gamma_deno_sqr = gamma_deno_sqr * (1 - 1 / taus_x_t) + T.sqr(
          (new_mean_grad - deltas) * (old_grad - new_mean_grad)) / taus_x_t
        new_gamma_deno_sqr.name = "ngammasqr_den_" + param.name

        gamma = T.sqrt(gamma_nume_sqr) / T.sqrt(gamma_deno_sqr + eps)
        gamma.name = "gamma_" + param.name

        if self.gamma_clip:
          gamma = T.minimum(gamma, self.gamma_clip)

        momentum_step = gamma * new_mean_grad
        corrected_grad_cand = (deltas + momentum_step) / (1 + gamma)

        # For starting the variance reduction.
        if self.start_var_reduction > -1:
          cond = T.le(self.start_var_reduction, self.i)
          corrected_grad = cond * corrected_grad_cand + (1 - cond) * deltas
        else:
          corrected_grad = deltas

        # Use the gradients from the previous update
        # to compute the \nabla f(x_t) - \nabla f(x_{t-1})
        cur_curvature = deltas - old_plain_grad
        new_curvature_ave = mean_curvature * (1 - 1 / taus_x_t) + cur_curvature / taus_x_t
        new_curvature_ave.name = "ncurve_ave_" + param.name

        # Average average curvature
        nc_ave = new_curvature_ave
        new_curvature_sqr_ave = mean_curvature_sqr * (1 - 1 / taus_x_t) + T.sqr(cur_curvature) / taus_x_t
        new_curvature_sqr_ave.name = "ncurve_sqr_ave_" + param.name

        # Unbiased average squared curvature
        nc_sq_ave = new_curvature_sqr_ave

        slope = self.learning_rate_var
        rms_dx_tm1 = T.sqrt(msdx + slope)
        rms_curve_t = T.sqrt(new_curvature_sqr_ave + slope)

        # This is where the update step is being defined
        # delta_x_t = -scaled_lr * (rms_dx_tm1 / rms_curve_t - cov_num_t / (new_curvature_sqr_ave + epsilon))
        delta_x_t = -(rms_dx_tm1 / rms_curve_t - cov_num_t / (new_curvature_sqr_ave + self.learning_rate_var))
        delta_x_t.name = "delta_x_t_" + param.name
        delta_x_t = delta_x_t * corrected_grad

        new_taus_t = (1 - T.sqr(mdx) / (msdx + eps)) * taus_x_t + self.var(1 + eps, name="stabilized")
        # To compute the E[\Delta^2]_t
        new_mean_square_dx = msdx * (1 - 1 / taus_x_t) + T.sqr(delta_x_t) / taus_x_t
        # To compute the E[\Delta]_t
        new_mean_dx = mean_dx * (1 - 1 / taus_x_t) + delta_x_t / taus_x_t

        # Perform the outlier detection:
        # This outlier detection is slightly different:
        self.upper_bound_tau = 1e8
        self.lower_bound_tau = 1.5
        new_taus_t = T.switch(
          T.or_(abs(deltas - new_mean_grad) > (2 * T.sqrt(new_mean_squared_grad - new_mean_grad ** 2)),
                abs(cur_curvature - nc_ave) > (2 * T.sqrt(nc_sq_ave - nc_ave ** 2))),
          self.var(2.2), new_taus_t)

        # Apply the bound constraints on tau:
        new_taus_t = T.maximum(self.lower_bound_tau, new_taus_t)
        new_taus_t = T.minimum(self.upper_bound_tau, new_taus_t)

        new_cov_num_t = cov_num_t * (1 - 1 / taus_x_t) + (delta_x_t * cur_curvature) * (1 / taus_x_t)
        deltas = -delta_x_t

        # Apply updates
        updates.append((mean_square_grad, new_mean_squared_grad))
        updates.append((mean_square_dx, new_mean_square_dx))
        updates.append((mean_dx, new_mean_dx))
        updates.append((gamma_nume_sqr, new_gamma_nume_sqr))
        updates.append((gamma_deno_sqr, new_gamma_deno_sqr))
        updates.append((taus_x_t, new_taus_t))
        updates.append((cov_num_t, new_cov_num_t))
        updates.append((mean_grad, new_mean_grad))
        updates.append((old_plain_grad, deltas))
        updates.append((mean_curvature, new_curvature_ave))
        updates.append((mean_curvature_sqr, new_curvature_sqr_ave))
        updates.append((old_grad, corrected_grad))

      if self.adasecant:
        # https://github.com/caglar/adasecant_wshp_paper/blob/master/adasecant/codes/learning_rule.py
        self.use_adam = False
        self.use_adagrad = False
        self.use_adadelta = False
        self.skip_nan_inf = True
        self.start_var_reduction = 1
        self.use_corrected_grad = True
        self.decay = 0.75
        self.delta_clip = 50.0
        self.gamma_clip = 1.8
        eps = numpy.float32(1e-7)
        if not self.gradient_l2_norm:
          deltas = deltas / (deltas.norm(2) + eps)
        if self.use_adagrad:
          sum_square_grad = self.var(param.get_value(borrow=True) * 0., name="sum_square_grad_%s" % param.name, broadcastable=param.broadcastable)
        if self.use_adadelta:
          eg2 = self.var(param.get_value(borrow=True) * 0., name= "eg2_%s" % param.name, broadcastable=param.broadcastable)
          edx2 = self.var(param.get_value(borrow=True) * 0., name= "edx2_%s" % param.name, broadcastable=param.broadcastable)
        if self.use_adam:
          m_prev = self.var(param, zero=True, name="adam_m_%s" % param.name)
          v_prev = self.var(param, zero=True, name="adam_v_%s" % param.name)

        taus_x_t = self.var((numpy.ones_like(param.get_value()) + eps) * 2.1, name="taus_x_t_" + param.name)

        #Variance reduction parameters
        #Numerator of the gamma:
        gamma_nume_sqr = self.var(numpy.zeros_like(param.get_value()) + eps, name="gamma_nume_sqr_" + param.name)
        #Denominator of the gamma:
        gamma_deno_sqr = self.var(numpy.zeros_like(param.get_value()) + eps, name="gamma_deno_sqr_" + param.name)
        #For the covariance parameter := E[\gamma \alpha]_{t-1}
        cov_num_t = self.var(numpy.zeros_like(param.get_value()) + eps, name="cov_num_t_" + param.name)
        # mean_grad := E[g]_{t-1}
        mean_grad = self.var(numpy.zeros_like(param.get_value()) + eps, name="mean_grad_%s" % param.name)
        # mean_squared_grad := E[g^2]_{t-1}
        mean_square_grad = self.var(numpy.zeros_like(param.get_value()) + eps, name="msg_" + param.name)
        # mean_square_dx := E[(\Delta x)^2]_{t-1}
        mean_square_dx = self.var(value = numpy.zeros_like(param.get_value()), name="msd_" + param.name)
        old_grad = self.var(value = numpy.zeros_like(param.get_value()) + eps, name="old_grad_" + param.name)

        #The uncorrected gradient of previous of the previous update:
        old_plain_grad = self.var(numpy.zeros_like(param.get_value()) + eps, name="old_plain_grad_" + param.name)
        mean_curvature = self.var(numpy.zeros_like(param.get_value()) + eps, name="mean_curvature_" + param.name)
        mean_curvature_sqr = self.var(numpy.zeros_like(param.get_value()) + eps, name="mean_curvature_sqr_" + param.name)

        # Initialize the E[\Delta]_{t-1}
        mean_dx = self.var(numpy.zeros_like(param.get_value()), name="mean_dx_" + param.name)

        # Block-wise normalize the gradient:
        #For the first time-step, assume that delta_x_t := deltas
        cond = T.eq(self.i, 0)
        msdx = cond * deltas**2 + (1 - cond) * mean_square_dx
        mdx = cond * deltas + (1 - cond) * mean_dx

        """
        Compute the new updated values.
        """
        # E[g_i^2]_t
        new_mean_squared_grad = mean_square_grad * self.decay + T.sqr(deltas) * (1 - self.decay)
        new_mean_squared_grad.name = "msg_" + param.name
        # E[g_i]_t
        new_mean_grad = mean_grad * self.decay + deltas * (1 - self.decay)
        new_mean_grad.name = "nmg_" + param.name
        # Keep the rms for numerator and denominator of gamma.
        new_gamma_nume_sqr = gamma_nume_sqr * (1 - 1 / taus_x_t) + T.sqr((deltas - old_grad) * (old_grad - new_mean_grad)) / taus_x_t
        new_gamma_nume_sqr.name = "ngammasqr_num_" + param.name
        new_gamma_deno_sqr = gamma_deno_sqr * (1 - 1 / taus_x_t) + T.sqr((new_mean_grad - deltas) * (old_grad - new_mean_grad)) / taus_x_t
        new_gamma_deno_sqr.name = "ngammasqr_den_" + param.name

        gamma = T.sqrt(gamma_nume_sqr) / T.sqrt(gamma_deno_sqr + eps)
        gamma.name = "gamma_" + param.name

        if self.gamma_clip:
          gamma = T.minimum(gamma, self.gamma_clip)

        momentum_step = gamma * new_mean_grad
        corrected_grad_cand = (deltas + momentum_step) / (1 + gamma)

        #For starting the variance reduction.
        if self.start_var_reduction > -1:
            cond = T.le(self.start_var_reduction, self.i)
            corrected_grad = cond * corrected_grad_cand + (1 - cond) * deltas
        else:
            corrected_grad = deltas
        if self.use_adagrad:
          g = corrected_grad
          # Accumulate gradient (windowed version)
          new_sum_squared_grad = (
              sum_square_grad + T.sqr(g)
          )

          rms_g_t = T.sqrt(new_sum_squared_grad)
          rms_g_t = T.maximum(rms_g_t, 1.0)
        if self.use_adadelta:
          decay = self.decay #self.adadelta_decay
          offset = eps #self.adadelta_offset
          g2 = T.sqr(corrected_grad)
          eg2_new = decay * eg2 + (1 - decay) * g2
          rms_g_t = T.sqrt(eg2_new + offset) / T.sqrt(edx2 + offset) #- 1.0 / dx_new
          #rms_g_t = T.maximum(rms_g_t, 1.0)

        # Use the gradients from the previous update
        # to compute the \nabla f(x_t) - \nabla f(x_{t-1})
        cur_curvature = deltas - old_plain_grad
        new_curvature_ave = mean_curvature * (1 - 1 / taus_x_t) + cur_curvature / taus_x_t
        new_curvature_ave.name = "ncurve_ave_" + param.name

        #Average average curvature
        nc_ave = new_curvature_ave
        new_curvature_sqr_ave = mean_curvature_sqr * (1 - 1 / taus_x_t) + T.sqr(cur_curvature) / taus_x_t
        new_curvature_sqr_ave.name = "ncurve_sqr_ave_" + param.name

        #Unbiased average squared curvature
        nc_sq_ave = new_curvature_sqr_ave

        slope = self.learning_rate_var
        rms_dx_tm1 = T.sqrt(msdx + slope)
        rms_curve_t = T.sqrt(new_curvature_sqr_ave + slope)

        #This is where the update step is being defined
        #delta_x_t = -scaled_lr * (rms_dx_tm1 / rms_curve_t - cov_num_t / (new_curvature_sqr_ave + epsilon))
        delta_x_t = -(rms_dx_tm1 / rms_curve_t - cov_num_t / (new_curvature_sqr_ave + self.learning_rate_var))
        delta_x_t.name = "delta_x_t_" + param.name

        # This part seems to be necessary for only RNNs
        # For feedforward networks this does not seem to be important.
        if self.delta_clip:
          delta_x_t = delta_x_t.clip(-self.delta_clip, self.delta_clip)
        if self.use_adagrad or self.use_adadelta:
          delta_x_t = delta_x_t * corrected_grad / rms_g_t
        elif self.use_adam:
          m_t = beta1 * m_prev + (numpy.float32(1) - beta1) * deltas
          v_t = beta2 * v_prev + (numpy.float32(1) - beta2) * deltas ** 2
          a_t = T.cast(T.sqrt(1 - beta2 ** i_t) / (1 - beta1 ** i_t), dtype="float32")
          delta_x_t = delta_x_t * corrected_grad * a_t
        else:
          #logger.info("Clipped adagrad is disabled.")
          delta_x_t = delta_x_t * corrected_grad

        new_taus_t = (1 - T.sqr(mdx) / (msdx + eps)) * taus_x_t + self.var(1 + eps, name="stabilized")
        #To compute the E[\Delta^2]_t
        new_mean_square_dx = msdx * (1 - 1 / taus_x_t) + T.sqr(delta_x_t) / taus_x_t
        #To compute the E[\Delta]_t
        new_mean_dx = mean_dx * (1 - 1 / taus_x_t) + delta_x_t / taus_x_t

        #Perform the outlier detection:
        #This outlier detection is slightly different:
        self.upper_bound_tau = 1e8
        self.lower_bound_tau = 1.5
        new_taus_t = T.switch(T.or_(abs(deltas - new_mean_grad) > (2 * T.sqrt(new_mean_squared_grad  - new_mean_grad**2)),
                                    abs(cur_curvature - nc_ave) > (2 * T.sqrt(nc_sq_ave - nc_ave**2))),
                                    self.var(2.2), new_taus_t)

        #Apply the bound constraints on tau:
        new_taus_t = T.maximum(self.lower_bound_tau, new_taus_t)
        new_taus_t = T.minimum(self.upper_bound_tau, new_taus_t)

        new_cov_num_t = cov_num_t * (1 - 1 / taus_x_t) + (delta_x_t * cur_curvature) * (1 / taus_x_t)
        upd[param] = delta_x_t

        # Apply updates
        updates.append((mean_square_grad, new_mean_squared_grad))
        updates.append((mean_square_dx, new_mean_square_dx))
        updates.append((mean_dx, new_mean_dx))
        updates.append((gamma_nume_sqr, new_gamma_nume_sqr))
        updates.append((gamma_deno_sqr, new_gamma_deno_sqr))
        updates.append((taus_x_t, new_taus_t))
        updates.append((cov_num_t, new_cov_num_t))
        updates.append((mean_grad, new_mean_grad))
        updates.append((old_plain_grad, deltas))
        updates.append((mean_curvature, new_curvature_ave))
        updates.append((mean_curvature_sqr, new_curvature_sqr_ave))
        #updates.append((param, param + update_step))

        if self.use_adagrad:
          updates.append((sum_square_grad, new_sum_squared_grad))
        if self.use_adadelta:
          edx2_new = self.decay * edx2 + (1 - self.decay) * delta_x_t ** 2
          updates.append((eg2, eg2_new))
          updates.append((edx2, edx2_new))
          #updates.append((dx, dx_new))
        if self.use_adam:
          updates.append((m_prev, m_t))
          updates.append((v_prev, v_t))

        if self.use_corrected_grad:
          updates.append((old_grad, corrected_grad))

      elif self.nadam: # http://cs229.stanford.edu/proj2015/054_report.pdf
        from functools import reduce

        m_cache = self.var(reduce(lambda x, y: x * y,[beta1 * ( 1 - 0.5 * 0.96**( (i + 1) * float(self.nadam_decay) ) ) for i in range(int(self.network.update_step - 1))] + [1]), name="momemtum_cache")
        m_prev = self.var(param, zero=True, name="nadam_m_%s" % param.name)
        v_prev = self.var(param, zero=True, name="nadam_v_%s" % param.name)
        self.adam_offset = numpy.float32(1e-8)
        init_flag = T.le(i_t,numpy.float32(self.network.update_step + 1))

        mt =  beta1 * ( 1 - 0.5 * 0.96**( i_t * float(self.nadam_decay) ) ) # momentum schedule, http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
        mtnext = beta1 * ( 1 - 0.5 * 0.96**( (i_t + 1) * float(self.nadam_decay) ) ) # for simplified NAG

        m_cache_new = m_cache * mt
        bias_corr = m_cache_new * mtnext

        _deltas = deltas / T.cast(1 - m_cache_new, dtype="float32")

        m = beta1 * T.switch(init_flag, deltas, m_prev) + (numpy.float32(1) - beta1) * deltas
        _m = m / T.cast(1 - bias_corr, dtype="float32") # bias correction (with momentum schedule (include the next t+1))

        v = beta2 * T.switch(init_flag, deltas**2, v_prev) + (numpy.float32(1) - beta2) * (deltas**2)
        _v = v / T.cast(1 - beta2 ** i_t, dtype="float32")

        __m = T.cast(1 - mt, dtype="float32") * _deltas + T.cast(mtnext, dtype="float32") * _m

        step = -self.learning_rate_var * gradient_scale * __m / ( T.sqrt(_v) + self.adam_offset )

        upd[param] += T.switch(init_flag, T.zeros_like(step), step)

        updates.append((m_cache, m_cache_new))
        updates.append((m_prev, m))
        updates.append((v_prev, v))

      elif self.eve:  # https://arxiv.org/pdf/1611.01505v1.pdf https://github.com/jayanthkoushik/sgd-feedback/blob/master/src/eve.py
        loss = self.network.get_objective()  # current objective value
        loss_prev = self.var(0, name="loss at t-1")
        d_prev = self.var(1, name="d_(t-1)")

        m_prev = self.var(param, zero=True, name="eve_m_%s" % param.name)
        v_prev = self.var(param, zero=True, name="eve_v_%s" % param.name)
        thl = numpy.float32(0.1)
        thu = numpy.float32(10.)

        loss_ch_fact = T.cast(loss / loss_prev, dtype="float32")

        ch_fact_lbound = T.switch(T.gt(loss, loss_prev), 1 + thl, 1 / (1 + thu))
        ch_fact_ubound = T.switch(T.gt(loss, loss_prev), 1 + thu, 1 / (1 + thl))

        loss_ch_fact = T.switch(T.lt(loss_ch_fact, ch_fact_lbound), ch_fact_lbound, loss_ch_fact)
        loss_ch_fact = T.switch(T.gt(loss_ch_fact, ch_fact_ubound), ch_fact_ubound, loss_ch_fact)
        loss_hat = T.switch(T.gt(i_t, 1), loss_prev * loss_ch_fact, loss)

        d_den = T.switch(T.gt(loss_hat, loss_prev), loss_prev, loss_hat)
        d_t = T.cast( (beta3 * d_prev) + (1. - beta3) * T.abs_((loss_hat - loss_prev)) / d_den, dtype="float32" )
        d_t = T.cast( T.switch(T.gt(i_t, 1), d_t, 1.), dtype="float32" )
        updates.append((d_prev, d_t))

        m_t = beta1 * m_prev + (numpy.float32(1) - beta1) * deltas
        mhat_t = m_t / (1. - beta1**i_t)
        updates.append((m_prev, m_t))

        v_t = beta2 * v_prev + (numpy.float32(1) - beta2) * deltas ** 2
        vhat_t = v_t / (1. - beta2**i_t)
        updates.append((v_prev, v_t))

        self.adam_offset = numpy.float32(1e-16)
        step = self.learning_rate_var * mhat_t / ((T.sqrt(vhat_t) * d_t) + self.adam_offset)
        upd[param] += -step
        updates.append((loss_prev, loss))

      elif self.adam:
        #epsilon = numpy.float32(1e-8)
        #epsilon = numpy.float32(1.0)
        self.adam_offset = numpy.float32(1e-16)
        m_prev = self.var(param, zero=True, name="adam_m_%s" % param.name)
        v_prev = self.var(param, zero=True, name="adam_v_%s" % param.name)

        m_t = beta1 * m_prev + (numpy.float32(1) - beta1) * deltas
        v_t = beta2 * v_prev + (numpy.float32(1) - beta2) * deltas ** 2
        a_t = self.learning_rate_var
        if self.adam_fit_learning_rate:
          a_t *= T.cast(T.sqrt(1 - beta2 ** i_t) / (1 - beta1 ** i_t), dtype="float32")
        step = a_t * m_t / (T.sqrt(v_t) + self.adam_offset)

        updates.append((m_prev, m_t))
        updates.append((v_prev, v_t))
        #updates.append((param, param - step))
        upd[param] += -step

      elif self.adamax:
        epsilon = numpy.float32(1e-8)
        m_prev = self.var(param, zero=True, name="adamax_m_%s" % param.name)
        v_prev = self.var(param, zero=True, name="adamax_v_%s" % param.name)
        m_t = beta1 * m_prev + (numpy.float32(1) - beta1) * deltas
        v_t = T.maximum(beta2 * v_prev, abs(deltas) + epsilon)
        step = (self.learning_rate_var / (numpy.float32(1) - beta1 ** i_t)) * (m_t / v_t)
        updates.append((m_prev, m_t))
        updates.append((v_prev, v_t))
        upd[param] += -step

      elif self.adagrad:
        epsilon = numpy.float32(1e-6)
        accu_new = self.accu[param] + deltas ** 2
        updates.append((self.accu[param], accu_new))
        upd[param] += -self.learning_rate_var * deltas / T.sqrt(accu_new + epsilon)
        #updates.append((self.sqrsum[param], self.sqrsum[param] + deltas ** 2))
        #upd = upd * 0.1 / (0.1 + (self.sqrsum[param] + deltas ** 2) ** 0.5)

      elif self.adamdelta: # adam moment normalization + adadelta learning rate scaling
        self.adam_offset = numpy.float32(1e-16)
        m_prev = self.var(param, zero=True, name="adam_m_%s" % param.name)
        v_prev = self.var(param, zero=True, name="adam_v_%s" % param.name)

        m_t = beta1 * m_prev + (numpy.float32(1) - beta1) * deltas
        v_t = beta2 * v_prev + (numpy.float32(1) - beta2) * deltas ** 2
        g = m_t / (T.sqrt(v_t) + self.adam_offset)

        updates.append((m_prev, m_t))
        updates.append((v_prev, v_t))

        decay = self.adadelta_decay
        offset = self.adadelta_offset
        g = deltas
        g2 = g ** 2
        eg2_new = decay * self.eg2[param] + (numpy.float32(1) - decay) * g2
        dx_new = - g * T.sqrt(self.edx2[param] + offset) / T.sqrt(eg2_new + offset)
        edx2_new = decay * self.edx2[param] + (numpy.float32(1) - decay) * dx_new ** 2

        upd[param] += self.learning_rate_var * dx_new

        updates.append((self.eg2[param], eg2_new))
        updates.append((self.edx2[param], edx2_new))
        updates.append((self.dx[param], dx_new))

      elif self.adadelta:
        decay = self.adadelta_decay
        offset = self.adadelta_offset
        g = deltas
        g2 = g ** 2
        eg2_new = decay * self.eg2[param] + (numpy.float32(1) - decay) * g2
        dx_new = - g * T.sqrt(self.edx2[param] + offset) / T.sqrt(eg2_new + offset)
        edx2_new = decay * self.edx2[param] + (numpy.float32(1) - decay) * dx_new ** 2
        updates.append((self.eg2[param], eg2_new))
        updates.append((self.edx2[param], edx2_new))
        updates.append((self.dx[param], dx_new))
        upd[param] += self.learning_rate_var * dx_new

      elif self.rmsprop:
        # https://github.com/fchollet/keras/blob/master/keras/optimizers.py#L156
        # https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py#L398-L453
        accumulator = self.var(param, zero=True, name="accumulator_%s" % param.name)
        epsilon = numpy.float32(1e-8)
        accumulator_new = numpy.float32(self.rmsprop) * accumulator + (numpy.float32(1) - numpy.float32(self.rmsprop)) * deltas ** numpy.float32(2)
        updates.append((accumulator, accumulator_new))
        upd[param] += - (self.learning_rate_var * deltas) / (T.sqrt(accumulator_new) + epsilon)

      elif self.smorms3:
        # http://sifter.org/~simon/journal/20150420.html
        # https://www.reddit.com/r/MachineLearning/comments/3edb42/rmsprop_loses_to_smorms3_beware_the_epsilon/?
        epsilon = numpy.float32(1e-16)
        g = self.var(param, zero=True, name="g")
        g2 = self.var(param, zero=True, name="g2")
        mem = self.var(param.get_value() * numpy.float32(0) + numpy.float32(1), name="mem")
        r = numpy.float32(1) / (mem + numpy.float32(1))
        g_ = (numpy.float32(1) - r) * g + r * deltas
        g2_ = (numpy.float32(1) - r) * g2 + r * T.sqr(deltas)
        gg_ = T.sqr(g_)
        denoise = gg_ / (g2_ + epsilon)
        mem_ = numpy.float32(1) + mem * (numpy.float32(1) - denoise)
        updates.extend([(g, g_), (g2, g2_), (mem, mem_)])
        upd[param] += - (T.minimum(self.learning_rate_var, denoise) * deltas) / (T.sqrt(g2_) + epsilon)

      else:  # SGD
        upd[param] += - self.learning_rate_var * deltas

      if self.momentum > 0 and not self.ena:
        updates.append((self.deltas[param], upd[param]))
        upd[param] += self.deltas[param] * self.momentum
      if self.nesterov_momentum > 0:
        #The following code inspired by https://github.com/fidlej/optim/raw/master/dok/nesterov_simple.pdf
        velocity = self.var(param, zero=True, name="nesterov_velocity_%s" % param.name)
        tmp = self.nesterov_momentum * velocity + upd[param]
        updates.append((velocity, tmp))
        upd[param] += tmp*self.nesterov_momentum
      if self.momentum2 > 0:
        velocity = self.var(param, zero=True, name="momentum2_velocity_%s" % param.name)
        upd[param] += velocity * self.momentum2
        updates.append((velocity, upd[param]))

    if self.update_clip > 0:
      for p, u in list(upd.items()):
        if not u: continue
        upd[p] = T.clip(u, -self.update_clip, self.update_clip)

    if self.multi_batch_update > 1:
      do_update_now = T.eq(self.counter % self.multi_batch_update, self.multi_batch_update - 1)
      self.multi_batch_num_output_frames = self.var(0, name="multi_batch_num_output_frames", dtype="int64")
      multi_batch_num_output_frames = self.multi_batch_num_output_frames + batch_num_output_frames
      updates.append((self.multi_batch_num_output_frames, T.switch(do_update_now, 0, multi_batch_num_output_frames)))

      for param in grads.keys():
        multi_batch_update_param = self.var(param, name="%s_multi_batch_update" % param.name, zero=True)
        new_multi_batch_update_param = multi_batch_update_param + upd[param]
        updates.append((
          multi_batch_update_param,
          theano.ifelse.ifelse(
            do_update_now,
            T.zeros_like(param),
            new_multi_batch_update_param
          )
        ))
        upd[param] = theano.ifelse.ifelse(
          do_update_now,
          new_multi_batch_update_param / numpy.float32(self.multi_batch_update),
          T.zeros_like(param)
        )

    # Simulate multi GPU training. This might help for regularization.
    if self.update_multiple_models:
      if not self.update_multiple_models_average_step:
        self.update_multiple_models_average_step = self.update_multiple_models
      cur_model = self.counter % self.update_multiple_models
      average_step_i = self.update_multiple_models_average_step_i % self.update_multiple_models_average_step

      for param in grads.keys():
        models = [param]
        for i in range(self.update_multiple_models - 1):
          # Note that when we call this function, where they are just randomly initialized,
          # so it's important that reset() updates the var properly later.
          models += [self.var(param, name="%s_model_%i" % (param.name, i))]

        models_new = []
        if self.update_multiple_models_param_is_cur_model:
          # Current model is always the first one.
          models_new += [models[0] + upd[param]]
          models_new += models[1:]
        else:
          for i, model_param in enumerate(models):
            is_not_cur_model = T.neq(cur_model, i)
            models_new += [theano.ifelse.ifelse(is_not_cur_model, model_param, model_param + upd[param])]

        if self.update_multiple_models_averaging:
          is_not_cur_average_step = T.neq(self.counter % self.update_multiple_models_average_step, average_step_i)
          average_new_model = reduce(T.add, models_new[1:], models_new[0]) / numpy.float32(self.update_multiple_models)
          for i in range(len(models)):
            models_new[i] = theano.ifelse.ifelse(is_not_cur_average_step, models_new[i], average_new_model)

        if self.update_multiple_models_param_is_cur_model:
          # Rotate, so that the next model becomes the first one.
          models_new = models_new[1:] + models_new[:-1]

        updates.extend(zip(models, models_new))

      upd.clear()

    #if upd:
      #updates.append((param, self.norm_constraint(param + upd, 1.0)))
      #updates.append((param, param + upd))
    for p in upd:
      if upd[p]:
        if p.layer.attrs.get('weight_clip', 0.0) > 0.0:
          clip = numpy.float32(p.layer.attrs['weight_clip'])
          updates.append((p, T.clip(p + upd[p], -clip, clip)))
        elif self.weight_clip > 0.0:
          updates.append((p, T.clip(p + upd[p], -self.weight_clip, self.weight_clip)))
        else:
          updates.append((p, p + upd[p]))
    updates.append((self.i, i_t))
    if self.adasecant:
      dt = 1 #T.cast(T.max(T.sum(self.network.output.values()[0].index,axis=0)), 'float32')
      #updates.append((step, step + dt))

    if self.enforce_triangular_matrix_zero:
      assert self.update_on_device, "not implemented otherwise. we need to know if a param belongs to an output layer"
      ps = []
      for i, (p, upd) in enumerate(list(updates)):
        if p not in self.net_train_param_deltas: continue
        if p.ndim != 2: continue
        if p.layer in self.network.output.values(): continue
        ps += [p]
        upd = upd * T.tri(p.shape[0], p.shape[1], dtype="float32")
        updates[i] = (p, upd)
      print("enforce_triangular_matrix_zero for:", ps, file=log.v4)

    #for u in updates:
    #  print ">>>>", u
    return updates

  def setLearningRate(self, learning_rate):
    """
    :type learning_rate: float
    """
    assert self.pid == os.getpid()
    self.learning_rate_var.set_value(learning_rate)

  def update(self):
    assert self.pid == os.getpid()
    updates = self.getUpdateList()
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpuarray').excluding('gpu')
    updater = theano.function(inputs=[], updates=updates, name="updater", mode=mode_with_gpu)
    return updater()
