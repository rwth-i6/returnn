
import theano
import numpy
import os
from Log import log
from math import sqrt
from theano.compat.python2x import OrderedDict
import theano.tensor as T

class Updater:

  @classmethod
  def initFromConfig(cls, config):
    import rnn
    kwargs = {
      "gradient_clip": config.float('gradient_clip', -1),
      "adagrad": config.bool('adagrad', False),
      "adadelta": config.bool('adadelta', False),
      "adasecant": config.bool('adasecant', False),
      "adam": config.bool('adam', False),
      "max_norm" : config.float('max_norm', 0.0),
      "adadelta_decay": config.float('adadelta_decay', 0.90),
      "adadelta_offset": config.float('adadelta_offset', 1e-6),
      "momentum": config.float("momentum", 0)}
    return cls(**kwargs)

  @classmethod
  def initRule(cls, rule, **kwargs):
    kwargs.setdefault('momentum', 0)
    kwargs.setdefault('gradient_clip', -1)
    kwargs.setdefault('adadelta_decay', 0.90)
    kwargs.setdefault('adadelta_offset', 1e-6)
    kwargs.setdefault('adagrad', False)
    kwargs.setdefault('adadelta', False)
    kwargs.setdefault('adasecant', False)
    kwargs.setdefault('adam', False)
    kwargs.setdefault('max_norm', 0.0)
    if rule != "default":
      kwargs[rule] = True
    return cls(**kwargs)

  def __init__(self, momentum, gradient_clip, adagrad, adadelta, adadelta_decay, adadelta_offset, max_norm, adasecant, adam):
    """
    :type momentum: float
    :type gradient_clip: float
    :type adagrad: bool
    :type adadelta: bool
    """
    self.rng = numpy.random.RandomState(0101)
    self.momentum = momentum
    self.gradient_clip = gradient_clip
    self.max_norm = max_norm
    self.adagrad = adagrad
    self.adadelta = adadelta
    self.adasecant = adasecant
    self.adam = adam
    self.adadelta_decay = adadelta_decay
    self.adadelta_offset = adadelta_offset
    self.params = {}
    self.pid = -1
    assert not (self.adagrad and self.adadelta and self.adasecant and self.adam)
    if self.adadelta:
      self.momentum = 0.0
      print >> log.v4, "using adadelta with decay", self.adadelta_decay, ", offset", self.adadelta_offset
    if self.adagrad:
      print >> log.v4, "using adagrad"
    if self.momentum:
      print >> log.v4, "using momentum %f" % self.momentum
    if self.gradient_clip > 0:
      print >> log.v4, "using gradient clipping %f" % self.gradient_clip

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
      self.net_train_param_deltas = net_param_deltas
    else:
      self.net_train_param_deltas = {p : theano.shared(numpy.zeros(p.get_value(borrow=True,
                                                                              return_internal_type=True).shape,
                                                                  dtype=theano.config.floatX))
                                     for p in network.train_params_vars}
      " :type: dict[theano.compile.sharedvalue.SharedVariable,theano.compile.sharedvalue.SharedVariable] "
    self.learning_rate_var = theano.shared(value=numpy.cast[theano.config.floatX](0))
    " :type: theano.compile.sharedvalue.SharedVariable "

    if self.momentum > 0:
      self.deltas = {p: theano.shared(
                     value=numpy.zeros(p.get_value(borrow=True, return_internal_type=True).shape,
                                       dtype=theano.config.floatX), borrow=True,
                     name="deltas_%s" % p)
                     for p in network.train_params_vars}
    if self.adagrad:
      self.accu = {}
      for p in self.network.train_params_vars:
        shape = p.get_value(borrow=True, return_internal_type=True).shape
        #scale = numpy.sqrt(12. / numpy.sum(shape))
        #values = numpy.asarray(self.rng.normal(loc=0.0, scale=scale, size=shape), dtype=theano.config.floatX)
        #values = p.get_value()
        values = numpy.zeros(shape, dtype=theano.config.floatX)
        self.accu[p] = theano.shared(value=values)

      #self.accu = {p: theano.shared(
      #               value=numpy.zeros(p.get_value(borrow=True, return_internal_type=True).shape,
      #                                 dtype=theano.config.floatX), borrow=True, name="accu_%s " % p)
      #            for p in self.network.train_params_vars}
      #self.sqrsum = {p: theano.shared(
      #               value=numpy.zeros(p.get_value(borrow=True, return_internal_type=True).shape,
      #                                 dtype=theano.config.floatX), borrow=True,
      #               name="sqrsum_%s " % p)
      #               for p in self.network.train_params_vars}
    if self.adadelta:
      # http://arxiv.org/pdf/1212.5701v1.pdf
      self.eg2 = {p: theano.shared(value=numpy.zeros(p.get_value(borrow=True, return_internal_type=True).shape,
                                                     dtype=theano.config.floatX))
                  for p in self.network.train_params_vars} #E[g^2]
      self.edx2 = {p: theano.shared(value=numpy.zeros(p.get_value(borrow=True, return_internal_type=True).shape,
                                                      dtype=theano.config.floatX))
                  for p in self.network.train_params_vars} #E[\delta x^2]
      self.dx = {p: theano.shared(value=numpy.zeros(p.get_value(borrow=True, return_internal_type=True).shape,
                                                    dtype=theano.config.floatX))
                  for p in self.network.train_params_vars} #\delta x

  @property
  def isInitialized(self):
    return self.pid >= 0

  def setNetParamDeltas(self, net_param_deltas):
    assert self.pid == os.getpid()
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

  def var(self, value, name = "", broadcastable = None, reset = True):
    if broadcastable:
      param = theano.shared(value = numpy.asarray(value).astype('float32'), name = name, broadcastable=broadcastable)
    else:
      param = theano.shared(value = numpy.asarray(value).astype('float32'), name = name)
    if reset:
      self.params[param] = value
    return param

  def reset(self):
    return # this needs to be done smarter
    for param in self.params:
      param.set_value(self.params[param])

  def getUpdateList(self):
    assert self.pid == os.getpid()
    updates = []
    " :type: list[(theano.SharedVariable, theano.Variable)] "
    upd = { p: 0 for p in self.net_train_param_deltas.keys() }
    eps = 1e-7
    if self.adasecant:
      grads = OrderedDict({p: self.net_train_param_deltas[p] / (self.net_train_param_deltas[p].norm(2) + eps) for p in self.net_train_param_deltas.keys()})
      #grads = OrderedDict({p: self.net_train_param_deltas[target][p] for p in self.net_train_param_deltas[target].keys()})
      step = self.var(0, "adasecant_step")
    else:
      grads = self.net_train_param_deltas
    i = theano.shared(numpy.float32(1))
    for param in grads.keys():
      deltas = T.switch(T.or_(T.isinf(grads[param]), T.isnan(grads[param])), 0, grads[param])
      if self.max_norm > 0:
        deltas = self.norm_constraint(deltas, self.max_norm)
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
      if self.adasecant:
        # https://github.com/caglar/adasecant_wshp_paper/blob/master/adasecant/codes/learning_rule.py
        self.use_adagrad = False
        self.use_adadelta = False #True #True
        self.skip_nan_inf = True
        self.start_var_reduction = -1
        self.use_corrected_grad = True  #True
        self.decay = 0.9 #0.95
        ### default
        self.delta_clip = 50.0
        self.outlier_detection = False # True #True
        self.gamma_clip = None #1.8
        ### aggressive
        #self.delta_clip = None
        #self.outlier_detection = False
        #self.gamma_clip = None
        ### conservative
        #self.delta_clip = 50.0 #None
        #self.outlier_detection = True #False
        #self.gamma_clip = 1.8 #None #1.8

        if self.skip_nan_inf:
          #If norm of the gradients of a parameter is inf or nan don't update that parameter
          #That might be useful for RNNs.
          grads[param] = T.switch(T.or_(T.isinf(grads[param]), T.isnan(grads[param])), 0, grads[param])

        grads[param] = T.unbroadcast(grads[param], -1)

        #print param, param.get_value().shape, numpy.prod(param.get_value().shape)

        grads[param].name = "grad_%s" % param.name
        mean_grad = self.var(param.get_value() * 0. + eps, name="mean_grad_%s" % param.name, broadcastable=param.broadcastable)
        slow_constant = 2.1
        #mean_corrected_grad = self.var(param.get_value() * 0 + eps, name="mean_corrected_grad_%s" % param.name)
        if self.use_adagrad:
          # sum_square_grad := \sum_i g_i^2
          sum_square_grad = self.var(param.get_value(borrow=True) * 0., name="sum_square_grad_%s" % param.name, broadcastable=param.broadcastable)
        if self.use_adadelta:
          eg2 = self.var(param.get_value(borrow=True) * 0., name= "eg2_%s" % param.name, broadcastable=param.broadcastable)
          edx2 = self.var(param.get_value(borrow=True) * 0., name= "edx2_%s" % param.name, broadcastable=param.broadcastable)
          #dx = self.var(param.get_value(borrow=True) * 0., name= "dx_%s" % param.name)
        taus_x_t = self.var((numpy.ones_like(param.get_value()) + eps) * slow_constant,
                           name="taus_x_t_" + param.name, broadcastable=param.broadcastable)
        self.taus_x_t = taus_x_t

        #Variance reduction parameters
        #Numerator of the gamma:
        gamma_nume_sqr = self.var(numpy.zeros_like(param.get_value()) + eps,
                                  name="gamma_nume_sqr_" + param.name, broadcastable=param.broadcastable)

        #Denominator of the gamma:
        gamma_deno_sqr = self.var(numpy.zeros_like(param.get_value()) + eps,
                                  name="gamma_deno_sqr_" + param.name, broadcastable=param.broadcastable)

        #For the covariance parameter := E[\gamma \alpha]_{t-1}
        cov_num_t = self.var(numpy.zeros_like(param.get_value()) + eps,
                             name="cov_num_t_" + param.name, broadcastable=param.broadcastable)

        # mean_squared_grad := E[g^2]_{t-1}
        mean_square_grad = self.var(numpy.zeros_like(param.get_value()) + eps,
                                    name="msg_" + param.name, broadcastable=param.broadcastable)

        # mean_square_dx := E[(\Delta x)^2]_{t-1}
        mean_square_dx = self.var(value = param.get_value() * 0., name="msd_" + param.name, broadcastable=param.broadcastable)
        if self.use_corrected_grad:
            old_grad = self.var(value = param.get_value() * 0. + eps, broadcastable=param.broadcastable)

        #The uncorrected gradient of previous of the previous update:
        old_plain_grad = self.var(param.get_value() * 0 + eps, broadcastable=param.broadcastable)
        mean_curvature = self.var(param.get_value() * 0 + eps, broadcastable=param.broadcastable)
        mean_curvature_sqr = self.var(param.get_value() * 0 + eps, broadcastable=param.broadcastable)

        # Initialize the E[\Delta]_{t-1}
        mean_dx = self.var(param.get_value() * 0., broadcastable=param.broadcastable)

        # Block-wise normalize the gradient:
        norm_grad = grads[param]

        #For the first time-step, assume that delta_x_t := norm_grad
        cond = T.eq(step, 0)
        msdx = cond * norm_grad**2 + (1 - cond) * mean_square_dx
        mdx = cond * norm_grad + (1 - cond) * mean_dx

        """
        Compute the new updated values.
        """
        # E[g_i^2]_t
        new_mean_squared_grad = (
            mean_square_grad * (self.decay)  +
            T.sqr(norm_grad) * (1 - self.decay)
        )
        new_mean_squared_grad.name = "msg_" + param.name
        # E[g_i]_t
        new_mean_grad = (
            mean_grad * (self.decay) +
            norm_grad * (1 - self.decay)
        )
        new_mean_grad.name = "nmg_" + param.name

        mg = new_mean_grad
        mgsq = new_mean_squared_grad

        # Keep the rms for numerator and denominator of gamma.
        new_gamma_nume_sqr = (
            gamma_nume_sqr * (1 - 1 / taus_x_t) +
            T.sqr((norm_grad - old_grad) * (old_grad - mg)) / taus_x_t
        )
        new_gamma_nume_sqr.name = "ngammasqr_num_" + param.name

        new_gamma_deno_sqr = (
            gamma_deno_sqr * (1 - 1 / taus_x_t) +
            T.sqr((mg - norm_grad) * (old_grad - mg)) / taus_x_t
        )
        new_gamma_deno_sqr.name = "ngammasqr_den_" + param.name

        gamma = T.sqrt(gamma_nume_sqr) / T.sqrt(gamma_deno_sqr + eps)
        gamma.name = "gamma_" + param.name

        if self.gamma_clip:
            gamma = T.minimum(gamma, self.gamma_clip)

        momentum_step = gamma * mg
        corrected_grad_cand = (norm_grad + momentum_step) / (1 + gamma)

        #For starting the variance reduction.
        if self.start_var_reduction > -1:
            cond = T.le(self.start_var_reduction, step)
            corrected_grad = cond * corrected_grad_cand + (1 - cond) * norm_grad
        else:
            corrected_grad = norm_grad

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
        cur_curvature = norm_grad - old_plain_grad
        cur_curvature_sqr = T.sqr(cur_curvature)

        new_curvature_ave = (
            mean_curvature * (1 - 1 / taus_x_t) +
            (cur_curvature / taus_x_t)
        )
        new_curvature_ave.name = "ncurve_ave_" + param.name

        #Average average curvature
        nc_ave = new_curvature_ave

        new_curvature_sqr_ave = (
            mean_curvature_sqr * (1 - 1 / taus_x_t) +
            (cur_curvature_sqr / taus_x_t)
        )
        new_curvature_sqr_ave.name = "ncurve_sqr_ave_" + param.name

        #Unbiased average squared curvature
        nc_sq_ave = new_curvature_sqr_ave

        epsilon = self.learning_rate_var #lr_scalers.get(param, 1.) * self.learning_rate_var
        scaled_lr = self.var(1) #lr_scalers.get(param, 1.) * theano.shared(1.0, dtype = theano.config.floatX)
        rms_dx_tm1 = T.sqrt(msdx + epsilon)

        rms_curve_t = T.sqrt(new_curvature_sqr_ave + epsilon)

        #This is where the update step is being defined
        delta_x_t = - scaled_lr * (rms_dx_tm1 / rms_curve_t - cov_num_t / (new_curvature_sqr_ave + epsilon))
        delta_x_t.name = "delta_x_t_" + param.name

        # This part seems to be necessary for only RNNs
        # For feedforward networks this does not seem to be important.
        if self.delta_clip:
            #logger.info("Clipping will be applied on the adaptive step size.")
            delta_x_t = delta_x_t.clip(-self.delta_clip, self.delta_clip)
            if self.use_adagrad or self.use_adadelta:
                delta_x_t = delta_x_t * corrected_grad / rms_g_t
            else:
                #logger.info("Clipped adagrad is disabled.")
                delta_x_t = delta_x_t * corrected_grad
        else:
            #logger.info("Clipping will not be applied on the adaptive step size.")
            if self.use_adagrad or self.use_adadelta:
                delta_x_t = delta_x_t * corrected_grad / rms_g_t
            else:
                #logger.info("Clipped adagrad will not be used.")
                delta_x_t = delta_x_t * corrected_grad

        new_taus_t = (1 - T.sqr(mdx) / (msdx + eps)) * taus_x_t + self.var(1 + eps, name="stabilized")

        #To compute the E[\Delta^2]_t
        new_mean_square_dx = (
             msdx * (1 - 1 / taus_x_t) +
             (T.sqr(delta_x_t) / taus_x_t)
         )

        #To compute the E[\Delta]_t
        new_mean_dx = (
            mean_dx * (1 - 1 / taus_x_t) +
            (delta_x_t / (taus_x_t))
        )

        if self.outlier_detection:
          #Perform the outlier detection:
          #This outlier detection is slightly different:
          self.upper_bound_tau = 1e8
          self.lower_bound_tau = 1.5
          new_taus_t = T.switch(T.or_(abs(norm_grad - mg) > (2 * T.sqrt(mgsq  - mg**2)),
                                      abs(cur_curvature - nc_ave) > (2 * T.sqrt(nc_sq_ave - nc_ave**2))),
                                      self.var(2.2), new_taus_t)

          #Apply the bound constraints on tau:
          new_taus_t = T.maximum(self.lower_bound_tau, new_taus_t)
          new_taus_t = T.minimum(self.upper_bound_tau, new_taus_t)
        else:
          new_taus_t = new_taus_t

        new_cov_num_t = (
            cov_num_t * (1 - 1 / taus_x_t) +
            (delta_x_t * cur_curvature) * (1 / taus_x_t)
        )

        upd[param] += delta_x_t

        # Apply updates
        updates.append((mean_square_grad, new_mean_squared_grad))
        updates.append((mean_square_dx, new_mean_square_dx))
        updates.append((mean_dx, new_mean_dx))
        updates.append((gamma_nume_sqr, new_gamma_nume_sqr))
        updates.append((gamma_deno_sqr, new_gamma_deno_sqr))
        updates.append((taus_x_t, new_taus_t))
        updates.append((cov_num_t, new_cov_num_t))
        updates.append((mean_grad, new_mean_grad))
        updates.append((old_plain_grad, norm_grad))
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

        if self.use_corrected_grad:
          updates.append((old_grad, corrected_grad))

      elif self.adam:
        #def adam(loss, all_params, learning_rate=0.0002, beta1=0.1, beta2=0.001,
        epsilon=1e-8
        gamma=1-1e-8
        beta1=0.1
        beta2=0.001

        i_t = i + 1.
        fix1 = 1. - (1. - beta1)**i_t
        fix2 = 1. - (1. - beta2)**i_t
        beta1_t = 1-(1-beta1)*gamma**(i_t-1)   # ADDED
        learning_rate_t = self.learning_rate_var * (T.sqrt(fix2) / fix1)

        m = theano.shared(
            numpy.zeros(param.get_value().shape, dtype=theano.config.floatX))
        v = theano.shared(
            numpy.zeros(param.get_value().shape, dtype=theano.config.floatX))

        m_t = (beta1_t * deltas) + ((1. - beta1_t) * m) # CHANGED from b_t to use beta1_t
        v_t = (beta2 * deltas**2) + ((1. - beta2) * v)
        g_t = m_t / (T.sqrt(v_t) + epsilon)
        param_i_t = param - (learning_rate_t * g_t)

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((param, param_i_t) )
        return updates

      elif self.adagrad:
        epsilon = 1e-6
        accu_new = self.accu[param] + deltas ** 2
        updates.append((self.accu[param], accu_new))
        upd[param] += -self.learning_rate_var * deltas / T.sqrt(accu_new + epsilon)
        #updates.append((self.sqrsum[param], self.sqrsum[param] + deltas ** 2))
        #upd = upd * 0.1 / (0.1 + (self.sqrsum[param] + deltas ** 2) ** 0.5)
      elif self.adadelta:
        # http://arxiv.org/pdf/1212.5701v1.pdf
        decay = self.adadelta_decay
        offset = self.adadelta_offset
        g = deltas
        g2 = g ** 2
        eg2_new = decay * self.eg2[param] + (1 - decay) * g2
        dx_new = - g * T.sqrt(self.edx2[param] + offset) / T.sqrt(eg2_new + offset)
        edx2_new = decay * self.edx2[param] + (1 - decay) * dx_new ** 2
        updates.append((self.eg2[param], eg2_new))
        updates.append((self.edx2[param], edx2_new))
        updates.append((self.dx[param], dx_new))
        upd[param] += self.learning_rate_var * dx_new
      else:
        upd[param] += - self.learning_rate_var * deltas
      if self.momentum > 0:
        updates.append((self.deltas[param], upd[param]))
        upd[param] += self.deltas[param] * self.momentum
    #if upd:
      #updates.append((param, self.norm_constraint(param + upd, 1.0)))
      #updates.append((param, param + upd))
    updates.extend([(p, p + upd[p]) for p in upd if upd[p]])
    updates.append((i, i+1))
    if self.adasecant:
      updates.append((step, step + 1))
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
    updater = theano.function(inputs=[], updates=updates, name="updater")
    return updater()
