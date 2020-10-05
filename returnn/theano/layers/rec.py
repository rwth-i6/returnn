
import numpy
from theano import tensor as T
import theano
from .hidden import HiddenLayer, CAlignmentLayer
from returnn.theano.layers.base import Container, Layer
from returnn.theano.activation_functions import strtoact
from math import sqrt
from returnn.theano.ops.lstm import LSTMOpInstance
from returnn.theano.ops.lstm import LSTMSOpInstance
from returnn.theano.ops.blstm import BLSTMOpInstance
import returnn.theano.recurrent_transform as recurrent_transform_mod
import json
from returnn.theano.util import print_to_file
from theano.ifelse import ifelse


class Unit(Container):
  """
  Abstract descriptor class for all kinds of recurrent units.
  """
  def __init__(self, n_units, n_in, n_out, n_re, n_act):
    """
    :param n_units: number of cells
    :param n_in: cell fan in
    :param n_out: cell fan out
    :param n_re: recurrent fan in
    :param n_act: number of outputs
    """
    self.n_units, self.n_in, self.n_out, self.n_re, self.n_act = n_units, n_in, n_out, n_re, n_act
    self.slice = T.constant(self.n_units, dtype='int32')
    self.params = {}

  def set_parent(self, parent):
    """
    :type parent: RecurrentUnitLayer
    """
    self.parent = parent

  def scan(self, x, z, non_sequences, i, outputs_info, W_re, W_in, b, go_backwards=False, truncate_gradient=-1):
    """
    Executes the iteration over the time axis (usually with theano.scan)
    :param step: python function to be executed
    :param x: unmapped input tensor in (time,batch,dim) shape
    :param z: same as x but already transformed to self.n_in
    :param non_sequences: see theano.scan
    :param i: index vector in (time, batch) shape
    :param outputs_info: see theano.scan
    :param W_re: recurrent weight matrix
    :param W_in: input weight matrix
    :param b: input bias
    :param go_backwards: whether to scan the sequence from 0 to T or from T to 0
    :param truncate_gradient: see theano.scan
    :return:
    """
    self.outputs_info = outputs_info
    self.non_sequences = non_sequences
    self.W_re = W_re
    self.W_in = W_in
    self.b = b
    self.go_backwards = go_backwards
    self.truncate_gradient = truncate_gradient
    try:
      self.xc = z if not x else T.concatenate([s.output for s in x], axis = -1)
    except Exception:
      self.xc = z if not x else T.concatenate(x, axis = -1)

    outputs, _ = theano.scan(self.step,
                             #strict = True,
                             truncate_gradient = truncate_gradient,
                             go_backwards = go_backwards,
                             sequences = [i,self.xc,z],
                             non_sequences = non_sequences,
                             outputs_info = outputs_info)
    return outputs

  def scan_seg(self, x, z, att, non_sequences, i, outputs_info, W_re, W_in, b, go_backwards=False, truncate_gradient=-1):
    """
    Executes the iteration over the time axis (usually with theano.scan)
    :param step: python function to be executed
    :param x: unmapped input tensor in (time,batch,dim) shape
    :param z: same as x but already transformed to self.n_in
    :param non_sequences: see theano.scan
    :param i: index vector in (time, batch) shape
    :param outputs_info: see theano.scan
    :param W_re: recurrent weight matrix
    :param W_in: input weight matrix
    :param b: input bias
    :param go_backwards: whether to scan the sequence from 0 to T or from T to 0
    :param truncate_gradient: see theano.scan
    :return:
    """
    self.outputs_info = outputs_info
    self.non_sequences = non_sequences
    self.W_re = W_re
    self.W_in = W_in
    self.b = b
    self.go_backwards = go_backwards
    self.truncate_gradient = truncate_gradient
    try:
      self.xc = z if not x else T.concatenate([s.output for s in x], axis = -1)
    except Exception:
      self.xc = z if not x else T.concatenate(x, axis = -1)

    outputs, _ = theano.scan(self.step,
                             #strict = True,
                             truncate_gradient = truncate_gradient,
                             go_backwards = go_backwards,
                             sequences = [i,self.xc,z,att],
                             non_sequences = non_sequences,
                             outputs_info = outputs_info)
    return outputs


class VANILLA(Unit):
  """
  A simple tanh unit
  """
  def __init__(self, n_units,  **kwargs):
    super(VANILLA, self).__init__(n_units, n_units, n_units, n_units, 1)

  def step(self, i_t, x_t, z_t, z_p, h_p):
    """
    performs one iteration of the recursion
    :param i_t: index at time step t
    :param x_t: raw input at time step t
    :param z_t: mapped input at time step t
    :param z_p: previous input from time step t-1
    :param h_p: previous hidden activation from time step t-1
    :return:
    """
    return [ T.tanh(z_t + z_p) ]


class LSTME(Unit):
  """
  A theano based LSTM implementation
  """
  def __init__(self, n_units, **kwargs):
    super(LSTME, self).__init__(
      n_units=n_units,
      n_in=n_units * 4,  # input gate, forget gate, output gate, net input
      n_out=n_units,
      n_re=n_units * 4,
      n_act=2  # output, cell state
    )
    self.o_output = T.as_tensor(numpy.ones((n_units,), dtype='float32'))
    self.o_h = T.as_tensor(numpy.ones((n_units,), dtype='float32'))

  def step(self, i_t, x_t, z_t, y_p, c_p, *other_args):
    # See Unit.scan() for seqs.
    # args: seqs (x_t = unit.xc, z_t, i_t), outputs (# unit.n_act, y_p, c_p, ...), non_seqs (none)
    other_outputs = []
    if self.recurrent_transform:
      state_vars = other_args[:len(self.recurrent_transform.state_vars)]
      self.recurrent_transform.set_sorted_state_vars(state_vars)
      z_r, r_updates = self.recurrent_transform.step(y_p)
      z_t += z_r
      for v in self.recurrent_transform.get_sorted_state_vars():
        other_outputs += [r_updates[v]]
    z_t += T.dot(y_p, self.W_re)
    partition = z_t.shape[1] // 4
    ingate = T.nnet.sigmoid(z_t[:,:partition])
    forgetgate = T.nnet.sigmoid(z_t[:,partition:2*partition])
    outgate = T.nnet.sigmoid(z_t[:,2*partition:3*partition])
    input = T.tanh(z_t[:,3*partition:4*partition])
    c_t = forgetgate * c_p + ingate * input
    y_t = outgate * T.tanh(c_t)
    i_output = T.outer(i_t, self.o_output)
    i_h = T.outer(i_t, self.o_h)
    # return: next outputs (# unit.n_act, y_t, c_t, ...)
    return (y_t * i_output, c_t * i_h + c_p * (1 - i_h)) + tuple(other_outputs)

class LSTMS(Unit):
  """
  A theano based LSTM implementation
  """
  def __init__(self, n_units, **kwargs):
    super(LSTMS, self).__init__(
      n_units=n_units,
      n_in=n_units * 4,  # input gate, forget gate, output gate, net input
      n_out=n_units,
      n_re=n_units * 4,
      n_act=2  # output, cell state
    )
    self.o_output = T.as_tensor(numpy.ones((n_units,), dtype='float32'))
    self.o_h = T.as_tensor(numpy.ones((n_units,), dtype='float32'))

  def step(self, i_t, x_t, z_t, att_p, y_p, c_p, *other_args):
    # See Unit.scan() for seqs.
    # args: seqs (x_t = unit.xc, z_t, i_t), outputs (# unit.n_act, y_p, c_p, ...), non_seqs (none)
    other_outputs = []
    #att_p = theano.printing.Print('att in lstms', attrs=['__str__'])(att_p)
    if self.recurrent_transform:
      state_vars = other_args[:len(self.recurrent_transform.state_vars)]
      self.recurrent_transform.set_sorted_state_vars(state_vars)
      z_r, r_updates = self.recurrent_transform.step(y_p)
      z_t += z_r
      for v in self.recurrent_transform.get_sorted_state_vars():
        other_outputs += [r_updates[v]]
    maxatt = att_p.repeat(z_t.shape[1]).reshape((z_t.shape[0],z_t.shape[1]))#.dimshuffle(1,0)
    #maxatt = theano.printing.Print('maxatt',attrs=['__str__','shape'])(maxatt)
    z_t = T.switch(maxatt>0,z_t,z_t + T.dot(y_p, self.W_re))
    #z_t += T.dot(y_p, self.W_re)
    #z_t = theano.printing.Print('z_t lstms',attrs=['shape'])(z_t)

    partition = z_t.shape[1] // 4
    ingate = T.nnet.sigmoid(z_t[:,:partition])
    forgetgate = ((T.nnet.sigmoid(z_t[:,partition:2*partition])).T * (1.-att_p)).T
    outgate = T.nnet.sigmoid(z_t[:,2*partition:3*partition])
    input = T.tanh(z_t[:,3*partition:4*partition])
    #c_t = ((forgetgate * c_p + ingate * input).T * (1.-T.max(att_p,axis=-1))).T
    c_t = forgetgate * c_p + ingate * input
    y_t = outgate * T.tanh(c_t)
    i_output = T.outer(i_t, self.o_output)
    i_h = T.outer(i_t, self.o_h)
    # return: next outputs (# unit.n_act, y_t, c_t, ...)
    return (y_t * i_output, c_t * i_h + c_p * (1 - i_h)) + tuple(other_outputs)

class LEAKYLSTM(Unit):
  """
  A 1D cell proposed in http://jmlr.org/papers/volume17/14-203/14-203.pdf
  The simplified equations can be seen in Table 7, page 36.
  Type A with gamma_3==0.
  This cell has 3 units instead of 4 like LSTM
  """
  def __init__(self, n_units, **kwargs):
    super(LEAKYLSTM, self).__init__(
      n_units=n_units,
      n_in=n_units * 3,  # forget gate (FG), output gate (OG), net input (IN)
      n_out=n_units,
      n_re=n_units * 3,
      n_act=2  # output, cell state
    )
    self.o_output = T.as_tensor(numpy.ones((n_units,), dtype='float32'))
    self.o_h = T.as_tensor(numpy.ones((n_units,), dtype='float32'))

  def step(self, i_t, x_t, z_t, y_p, c_p, *other_args):
    # See Unit.scan() for seqs.
    # args: seqs (x_t = unit.xc, z_t, i_t), outputs (# unit.n_act, y_p, c_p, ...), non_seqs (none)
    other_outputs = []
    if self.recurrent_transform:
      state_vars = other_args[:len(self.recurrent_transform.state_vars)]
      self.recurrent_transform.set_sorted_state_vars(state_vars)
      z_r, r_updates = self.recurrent_transform.step(y_p)
      z_t += z_r
      for v in self.recurrent_transform.get_sorted_state_vars():
        other_outputs += [r_updates[v]]
    z_t += T.dot(y_p, self.W_re)
    partition = z_t.shape[1] // 3 #number of units
    forgetgate = T.nnet.sigmoid(z_t[:,:partition])
    outgate = T.nnet.sigmoid(z_t[:,partition:2*partition])
    input = T.tanh(z_t[:,2*partition:3*partition])
    # c(t) = (1 - FG(t)) * IN(t) + FG(t) * c(t-1)
    c_t = (1-forgetgate) * input + forgetgate * c_p
    # y(t) = OG(t) * c(t) HINT: There can be added an additional nonlinearity (substitute c_t:=T.tanh(x_t))
    y_t = outgate * c_t
    i_output = T.outer(i_t, self.o_output)
    i_h = T.outer(i_t, self.o_h)
    # return: next outputs (# unit.n_act, y_t, c_t, ...)
    return (y_t * i_output, c_t * i_h + c_p * (1 - i_h)) + tuple(other_outputs)


class LEAKYLPLSTM(Unit):
  """
  A 1D cell proposed in http://jmlr.org/papers/volume17/14-203/14-203.pdf
  The simplified equations can be seen in Table 7, page 36.
  Type A.
  This cell has 4 units like the LSTM
  """
  def __init__(self, n_units, **kwargs):
    super(LEAKYLPLSTM, self).__init__(
      n_units=n_units,
      n_in=n_units * 4,  # forget gate (FG), output gate 1 (OG1), output gate 2 (OG2), net input (IN)
      n_out=n_units,
      n_re=n_units * 4,
      n_act=2  # output, cell state
    )
    self.o_output = T.as_tensor(numpy.ones((n_units,), dtype='float32'))
    self.o_h = T.as_tensor(numpy.ones((n_units,), dtype='float32'))

  def step(self, i_t, x_t, z_t, y_p, c_p, *other_args):
    # See Unit.scan() for seqs.
    # args: seqs (x_t = unit.xc, z_t, i_t), outputs (# unit.n_act, y_p, c_p, ...), non_seqs (none)
    other_outputs = []
    if self.recurrent_transform:
      state_vars = other_args[:len(self.recurrent_transform.state_vars)]
      self.recurrent_transform.set_sorted_state_vars(state_vars)
      z_r, r_updates = self.recurrent_transform.step(y_p)
      z_t += z_r
      for v in self.recurrent_transform.get_sorted_state_vars():
        other_outputs += [r_updates[v]]
    z_t += T.dot(y_p, self.W_re)
    partition = z_t.shape[1] // 4 #number of units
    forgetgate = T.nnet.sigmoid(z_t[:,:partition])
    outgate1 = T.nnet.sigmoid(z_t[:,partition:2*partition])
    outgate2 = T.nnet.sigmoid(z_t[:,2*partition:3*partition])
    input = T.tanh(z_t[:,3*partition:4*partition])
    # c(t) = (1 - FG(t)) * IN(t) + FG(t) * c(t-1)
    c_t = (1-forgetgate) * input + forgetgate * c_p
    # y(t) = tanh( OG1(t) * c(t) + OG2(t) * c(t-1) ) HINT: The additional nonlinearity maybe has not a significant effect
    y_t = T.tanh(outgate1 * c_t + outgate2 * c_p)
    i_output = T.outer(i_t, self.o_output)
    i_h = T.outer(i_t, self.o_h)
    # return: next outputs (# unit.n_act, y_t, c_t, ...)
    return (y_t * i_output, c_t * i_h + c_p * (1 - i_h)) + tuple(other_outputs)


class PIDLSTM(Unit):
  """
  A 1D cell proposed in http://jmlr.org/papers/volume17/14-203/14-203.pdf
  The simplified equations can be seen in Table 7, page 36.
  Type E. This cell works as a dynamic PID filter of the input. The forget gate
  determines if it has PD od PI characteristic, the Proportional gate gates the P/I part,
  the Difference gate the D/P part. It can have advantages if there is no subsampling in
  the layer.
  This cell has 4 units like the LSTM
  """
  def __init__(self, n_units, **kwargs):
    super(PIDLSTM, self).__init__(
      n_units=n_units,
      n_in=n_units * 4,  # forget gate (FG), Proportinal gate (PG), Difference gate (DG), net input (IN)
      n_out=n_units,
      n_re=n_units * 4,
      n_act=2  # output, cell state
    )
    self.o_output = T.as_tensor(numpy.ones((n_units,), dtype='float32'))
    self.o_h = T.as_tensor(numpy.ones((n_units,), dtype='float32'))

  def step(self, i_t, x_t, z_t, y_p, c_p, *other_args):
    # See Unit.scan() for seqs.
    # args: seqs (x_t = unit.xc, z_t, i_t), outputs (# unit.n_act, y_p, c_p, ...), non_seqs (none)
    other_outputs = []
    if self.recurrent_transform:
      state_vars = other_args[:len(self.recurrent_transform.state_vars)]
      self.recurrent_transform.set_sorted_state_vars(state_vars)
      z_r, r_updates = self.recurrent_transform.step(y_p)
      z_t += z_r
      for v in self.recurrent_transform.get_sorted_state_vars():
        other_outputs += [r_updates[v]]
    z_t += T.dot(y_p, self.W_re)
    partition = z_t.shape[1] // 4 #number of units
    forgetgate = T.nnet.sigmoid(z_t[:,:partition])
    propgate = T.nnet.sigmoid(z_t[:,partition:2*partition])
    diffgate = T.nnet.sigmoid(z_t[:,2*partition:3*partition])
    input = T.tanh(z_t[:,3*partition:4*partition])
    # c(t) = (1 - FG(t)) * IN(t) + FG(t) * c(t-1)
    c_t = (1-forgetgate) * input + forgetgate * c_p
    # y(t) = tanh( PG(t) * c(t) + DG(t) * ( c(t) - c(t-1)) ) HINT: The additional nonlinearity maybe has not a significant effect
    y_t = T.tanh(propgate * c_t + diffgate * ( c_t - c_p))
    i_output = T.outer(i_t, self.o_output)
    i_h = T.outer(i_t, self.o_h)
    # return: next outputs (# unit.n_act, y_t, c_t, ...)
    return (y_t * i_output, c_t * i_h + c_p * (1 - i_h)) + tuple(other_outputs)


class LSTMP(Unit):
  """
  Very fast custom LSTM implementation
  """
  def __init__(self, n_units, **kwargs):
    super(LSTMP, self).__init__(n_units, n_units * 4, n_units, n_units * 4, 2)

  def scan(self, x, z, non_sequences, i, outputs_info, W_re, W_in, b, go_backwards = False, truncate_gradient = -1):
    z = T.inc_subtensor(z[-1 if go_backwards else 0], T.dot(outputs_info[0],W_re))
    result = LSTMOpInstance(z[::-(2 * go_backwards - 1)], W_re, outputs_info[1], i[::-(2 * go_backwards - 1)])
    return [ result[0], result[2].dimshuffle('x',0,1) ]

class LSTMPS(Unit):
  """
  Very fast custom LSTM implementation for segment encoding
  """
  def __init__(self, n_units, **kwargs):
    super(LSTMPS, self).__init__(n_units, n_units * 4, n_units, n_units * 4, 2)

  def scan_seg(self, x, z, non_sequences, i, att, outputs_info, W_re, W_in, b, go_backwards = False, truncate_gradient = -1):
    z = T.inc_subtensor(z[-1 if go_backwards else 0], T.dot(outputs_info[0],W_re))
    result = LSTMSOpInstance(z[::-(2 * go_backwards - 1)], W_re, outputs_info[1], i[::-(2 * go_backwards - 1)], att)
    return [ result[0], result[2].dimshuffle('x',0,1) ]

class LSTMB(Unit):
  """
  Very fast custom BLSTM implementation
  """
  def __init__(self, n_units, **kwargs):
    super(LSTMB, self).__init__(n_units, n_units * 8, n_units * 2, n_units * 4, 2)

  def scan(self, x, z, non_sequences, i, outputs_info, W_re, W_in, b, go_backwards = False, truncate_gradient = -1):
    W_re_b = self.parent.add_param(
      self.parent.create_recurrent_weights(self.n_units, self.n_re, name="W_re_b_%s" % self.parent.name))
    z_f = z[:,:,:z.shape[2]/2]
    z_b = z[::-1,:,z.shape[2]/2:]
    z_f = T.inc_subtensor(z_f[0], T.dot(outputs_info[0], W_re))
    z_b = T.inc_subtensor(z_b[0], T.dot(outputs_info[0], W_re_b))
    result = BLSTMOpInstance(z_f,z_b, W_re, W_re_b, outputs_info[1], T.zeros_like(outputs_info[1]), i, i[::-1])
    return [ T.concatenate([result[0],result[1][::-1]],axis=2), T.concatenate([result[4],result[5][::-1]],axis=1).dimshuffle('x',0,1) ]
BLSTM = LSTMB # alternative name


class LSTMC(Unit):
  """
  The same implementation as above, but it executes a theano function (recurrent transform)
  in each iteration. This allows for additional dependencies in the recursion of the LSTM.
  """
  def __init__(self, n_units, **kwargs):
    super(LSTMC, self).__init__(n_units, n_units * 4, n_units, n_units * 4, 2)

  def scan(self, x, z, non_sequences, i, outputs_info, W_re, W_in, b, go_backwards = False, truncate_gradient = -1):
    assert self.parent.recurrent_transform
    import returnn.theano.ops.lstm_custom as op_lstm_custom
    op = op_lstm_custom.register_func(self.parent.recurrent_transform)
    custom_vars = self.parent.recurrent_transform.get_sorted_custom_vars()
    initial_state_vars = self.parent.recurrent_transform.get_sorted_state_vars_initial()
    # See op_lstm_custom.LSTMCustomOp.
    # Inputs args are: Z, c, y0, i, W_re, custom input vars, initial state vars
    # Results: (output) Y, (gates and cell state) H, (final cell state) d, state vars sequences
    op_res = op(z[::-(2 * go_backwards - 1)],
                outputs_info[1], outputs_info[0], i[::-(2 * go_backwards - 1)], T.ones((i.shape[1],),'float32'), W_re, *(custom_vars + initial_state_vars))
    result = [ op_res[0], op_res[2].dimshuffle('x',0,1) ] + op_res[3:]
    assert len(result) == len(outputs_info)
    return result


class LSTMR(Unit):
  """
  Same as LSTMC but without recurrent matrix multiplication
  """
  def __init__(self, n_units, **kwargs):
    super(LSTMR, self).__init__(n_units, n_units * 4, n_units, n_units * 4, 2)
    self.n_re = 0

  def scan(self, x, z, non_sequences, i, outputs_info, W_re, W_in, b, go_backwards = False, truncate_gradient = -1):
    assert self.parent.recurrent_transform
    import returnn.theano.ops.lstm_rec as op_lstm_rec
    op = op_lstm_rec.register_func(self.parent.recurrent_transform)
    custom_vars = self.parent.recurrent_transform.get_sorted_custom_vars()
    initial_state_vars = self.parent.recurrent_transform.get_sorted_state_vars_initial()
    # See op_lstm_rec.LSTMRecOp.
    # Inputs args are: Z, c, y0, i, custom input vars, initial state vars
    # Results: (output) Y, (gates and cell state) H, (final cell state) d, state vars sequences
    op_res = op(z[::-(2 * go_backwards - 1)],
                outputs_info[1], outputs_info[0], i[::-(2 * go_backwards - 1)], *(custom_vars + initial_state_vars))
    result = [ op_res[0], op_res[2].dimshuffle('x',0,1) ] + op_res[3:]
    assert len(result) == len(outputs_info)
    return result


class GRU(Unit):
  """
  Gated recurrent unit as described in http://arxiv.org/abs/1502.02367
  """
  def __init__(self, n_units, **kwargs):
    super(GRU, self).__init__(n_units, n_units * 3, n_units, n_units * 2, 2)
    l = sqrt(6.) / sqrt(n_units * 3)
    rng = numpy.random.RandomState(1234)
    values = numpy.asarray(rng.uniform(low=-l, high=l, size=(n_units, n_units)), dtype=theano.config.floatX)
    self.W_reset = theano.shared(value=values, borrow=True, name = "W_reset")
    self.params['W_reset'] = self.W_reset

  def step(self, i_t, x_t, z_t, z_p, h_p):
    CI, GR, GU = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid]
    u_t = GU(z_t[:,:self.slice] + z_p[:,:self.slice])
    r_t = GR(z_t[:,self.slice:2*self.slice] + z_p[:,self.slice:2*self.slice])
    h_c = CI(z_t[:,2*self.slice:] + T.dot(r_t * h_p, self.W_reset))
    return z_t, u_t * h_p + (1 - u_t) * h_c


class SRU(Unit):
  """
  Same as GRU but without reset weights, which allows for a faster computation on GPUs
  """
  def __init__(self, n_units, **kwargs):
    super(SRU, self).__init__(n_units, n_units * 3, n_units, n_units * 3, 1)

  def step(self, i_t, x_t, z_t, z_p, h_p):
    CI, GR, GU = [T.tanh, T.nnet.sigmoid, T.nnet.sigmoid]
    u_t = GU(z_t[:,:self.slice] + z_p[:,:self.slice])
    r_t = GR(z_t[:,self.slice:2*self.slice] + z_p[:,self.slice:2*self.slice])
    h_c = CI(z_t[:,2*self.slice:3*self.slice] + r_t * z_p[:,2*self.slice:3*self.slice])
    return  u_t * h_p + (1 - u_t) * h_c

class RecurrentUnitLayer(Layer):
  """
  Layer class to execute recurrent units
  """
  recurrent = True
  layer_class = "rec"
  last_segment_flag = 0

  def __init__(self,
               n_out = None,
               n_units = None,
               direction = 1,
               truncation = -1,
               sampling = 1,
               encoder = None,
               unit = 'lstm',
               n_dec = 0,
               attention = "none",
               recurrent_transform = "none",
               recurrent_transform_attribs = "{}",
               attention_template = 128,
               attention_distance = 'l2',
               attention_step = "linear",
               attention_beam = 0,
               attention_norm = "exp",
               attention_momentum = "none",
               attention_sharpening = 1.0,
               attention_nbest = 0,
               attention_store = False,
               attention_smooth = False,
               attention_glimpse = 1,
               attention_filters = 1,
               attention_accumulator = 'sum',
               attention_loss = 0,
               attention_bn = 0,
               attention_lm = 'none',
               attention_ndec = 1,
               attention_memory = 0,
               attention_alnpts = 0,
               attention_epoch  = 1,
               attention_segstep=0.01,
               attention_offset=0.95,
               attention_method="epoch",
               attention_scale=10,
               context=-1,
               context_span=1,
               base = None,
               aligner = None,
               lm = False,
               force_lm = False,
               droplm = 1.0,
               forward_weights_init=None,
               bias_random_init_forget_shift=0.0,
               copy_weights_from_base=False,
               segment_input=False,
               join_states=False,
               state_memory=False,
               state_memory_pos=-1,
               sample_segment=None,
               **kwargs):
    """
    :param n_out: number of cells
    :param n_units: used when initialized via Network.from_hdf_model_topology
    :param direction: process sequence in forward (1) or backward (-1) direction
    :param truncation: gradient truncation
    :param sampling: scan every nth frame only
    :param encoder: list of encoder layers used as initalization for the hidden state
    :param unit: cell type (one of 'lstm', 'vanilla', 'gru', 'sru')
    :param n_dec: absolute number of steps to unfold the network if integer, else relative number of steps from encoder
    :param recurrent_transform: name of recurrent transform
    :param recurrent_transform_attribs: dictionary containing parameters for a recurrent transform
    :param attention_template:
    :param attention_distance:
    :param attention_step:
    :param attention_beam:
    :param attention_norm:
    :param attention_sharpening:
    :param attention_nbest:
    :param attention_store:
    :param attention_align:
    :param attention_glimpse:
    :param attention_lm:
    :param base: list of layers which outputs are considered as based during attention mechanisms
    :param lm: activate RNNLM
    :param force_lm: expect previous labels to be given during testing
    :param droplm: probability to take the expected output as predecessor instead of the real one when LM=true
    :param bias_random_init_forget_shift: initialize forget gate bias of lstm networks with this value
    """
    source_index = None
    if len(kwargs['sources']) == 1 and (kwargs['sources'][0].layer_class.endswith('length') or kwargs['sources'][0].layer_class.startswith('length')):
      kwargs['sources'] = []
      source_index = kwargs['index']
    unit_given = unit
    from returnn.theano.device import is_using_gpu
    if unit == 'lstm':  # auto selection
      if not is_using_gpu():
        unit = 'lstme'
      elif recurrent_transform == 'none' and (not lm or droplm == 0.0):
        unit = 'lstmp'
      else:
        unit = 'lstmc'
    elif unit in ("lstmc", "lstmp") and not is_using_gpu():
      unit = "lstme"
    if segment_input:
      if is_using_gpu():
        unit = "lstmps"
      else:
        unit = "lstms"
    if n_out is None:
      assert encoder
      n_out = sum([enc.attrs['n_out'] for enc in encoder])
    kwargs.setdefault("n_out", n_out)
    if n_units is not None:
      assert n_units == n_out
    self.attention_weight = T.constant(1.,'float32')
    if len(kwargs['sources']) == 1 and kwargs['sources'][0].layer_class.startswith('length'):
      kwargs['sources'] = []
    elif len(kwargs['sources']) == 1 and kwargs['sources'][0].layer_class.startswith('signal'):
      kwargs['sources'] = []
    super(RecurrentUnitLayer, self).__init__(**kwargs)
    self.set_attr('from', ",".join([s.name for s in self.sources]) if self.sources else "null")
    self.set_attr('n_out', n_out)
    self.set_attr('unit', unit_given.encode("utf8"))
    self.set_attr('truncation', truncation)
    self.set_attr('sampling', sampling)
    self.set_attr('direction', direction)
    self.set_attr('lm', lm)
    self.set_attr('force_lm', force_lm)
    self.set_attr('droplm', droplm)
    if bias_random_init_forget_shift:
      self.set_attr("bias_random_init_forget_shift", bias_random_init_forget_shift)
    self.set_attr('attention_beam', attention_beam)
    self.set_attr('recurrent_transform', recurrent_transform.encode("utf8"))
    if isinstance(recurrent_transform_attribs, str):
      recurrent_transform_attribs = json.loads(recurrent_transform_attribs)
    if attention_template is not None:
      self.set_attr('attention_template', attention_template)
    self.set_attr('recurrent_transform_attribs', recurrent_transform_attribs)
    self.set_attr('attention_distance', attention_distance.encode("utf8"))
    self.set_attr('attention_step', attention_step.encode("utf8"))
    self.set_attr('attention_norm', attention_norm.encode("utf8"))
    self.set_attr('attention_sharpening', attention_sharpening)
    self.set_attr('attention_nbest', attention_nbest)
    attention_store = attention_store or attention_smooth or attention_momentum != 'none'
    self.set_attr('attention_store', attention_store)
    self.set_attr('attention_smooth', attention_smooth)
    self.set_attr('attention_momentum', attention_momentum.encode('utf8'))
    self.set_attr('attention_glimpse', attention_glimpse)
    self.set_attr('attention_filters', attention_filters)
    self.set_attr('attention_lm', attention_lm)
    self.set_attr('attention_bn', attention_bn)
    self.set_attr('attention_accumulator', attention_accumulator)
    self.set_attr('attention_ndec', attention_ndec)
    self.set_attr('attention_memory', attention_memory)
    self.set_attr('attention_loss', attention_loss)
    self.set_attr('n_dec', n_dec)
    self.set_attr('segment_input', segment_input)
    self.set_attr('attention_alnpts', attention_alnpts)
    self.set_attr('attention_epoch', attention_epoch)
    self.set_attr('attention_segstep', attention_segstep)
    self.set_attr('attention_offset', attention_offset)
    self.set_attr('attention_method', attention_method)
    self.set_attr('attention_scale', attention_scale)
    if segment_input:
      if not self.eval_flag:
      #if self.eval_flag:
        if isinstance(self.sources[0],RecurrentUnitLayer):
          self.inv_att = self.sources[0].inv_att #NBT
        else:
          if not join_states:
            self.inv_att = self.sources[0].attention #NBT
          else:
            assert hasattr(self.sources[0], "nstates"), "source does not have number of states!"
            ns = self.sources[0].nstates
            self.inv_att = self.sources[0].attention[(ns-1)::ns]
        inv_att = T.roll(self.inv_att.dimshuffle(2, 1, 0),1,axis=0)#TBN
        inv_att = T.set_subtensor(inv_att[0],T.zeros((inv_att.shape[1],inv_att.shape[2])))
        inv_att = T.max(inv_att,axis=-1)
      else:
        inv_att = T.zeros((self.sources[0].output.shape[0],self.sources[0].output.shape[1]))
    if encoder and hasattr(encoder[0],'act'):
      self.set_attr('encoder', ",".join([e.name for e in encoder]))
    if base:
      self.set_attr('base', ",".join([b.name for b in base]))
    else:
      base = encoder
    self.base = base
    self.encoder = encoder
    if aligner:
      self.aligner = aligner
    self.set_attr('n_units', n_out)
    unit = eval(unit.upper())(**self.attrs)
    assert isinstance(unit, Unit)
    self.unit = unit
    kwargs.setdefault("n_out", unit.n_out)
    n_out = unit.n_out
    self.set_attr('n_out', unit.n_out)
    if n_dec < 0:
      source_index = self.index
      n_dec *= -1
    if n_dec != 0:
      self.target_index = self.index
      if isinstance(n_dec,float):
        if not source_index:
          source_index = encoder[0].index if encoder else base[0].index
        lengths = T.cast(T.ceil(T.sum(T.cast(source_index,'float32'),axis=0) * n_dec), 'int32')
        idx, _ = theano.map(lambda l_i, l_m:T.concatenate([T.ones((l_i,),'int8'),T.zeros((l_m-l_i,),'int8')]),
                            [lengths], [T.max(lengths)+1])
        self.index = idx.dimshuffle(1,0)[:-1]
        n_dec = T.cast(T.ceil(T.cast(source_index.shape[0],'float32') * numpy.float32(n_dec)),'int32')
      else:
        if encoder:
          self.index = encoder[0].index
        self.index = T.ones((n_dec,self.index.shape[1]),'int8')
    else:
      n_dec = self.index.shape[0]
    # initialize recurrent weights
    self.W_re = None
    if unit.n_re > 0:
      self.W_re = self.add_param(self.create_recurrent_weights(unit.n_units, unit.n_re, name="W_re_%s" % self.name))
    # initialize forward weights
    bias_init_value = self.create_bias(unit.n_in).get_value()
    if bias_random_init_forget_shift:
      assert unit.n_units * 4 == unit.n_in  # (input gate, forget gate, output gate, net input)
      bias_init_value[unit.n_units:2 * unit.n_units] += bias_random_init_forget_shift
    self.b.set_value(bias_init_value)
    if not forward_weights_init:
      forward_weights_init = "random_uniform(p_add=%i)" % unit.n_re
    else:
      self.set_attr('forward_weights_init', forward_weights_init)
    self.forward_weights_init = forward_weights_init
    self.W_in = []
    sample_mean, gamma = None, None
    if copy_weights_from_base:
      self.params = {}
      #self.W_re = self.add_param(base[0].W_re)
      #self.W_in = [ self.add_param(W) for W in base[0].W_in ]
      #self.b = self.add_param(base[0].b)
      self.W_re = base[0].W_re
      self.W_in = base[0].W_in
      self.b = base[0].b
      if self.attrs.get('batch_norm', False):
        sample_mean = base[0].sample_mean
        gamma = base[0].gamma
      #self.masks = base[0].masks
      #self.mass = base[0].mass
    else:
      for s in self.sources:
        W = self.create_forward_weights(s.attrs['n_out'], unit.n_in, name="W_in_%s_%s" % (s.name, self.name))
        self.W_in.append(self.add_param(W))
    # make input
    z = self.b
    for x_t, m, W in zip(self.sources, self.masks, self.W_in):
      if x_t.layer_class == 'source':
        RecurrentUnitLayer.last_segment_flag = T.all(T.eq(x_t.output[-1],T.ones_like(x_t.output[0])))
        self.index = T.switch(RecurrentUnitLayer.last_segment_flag, T.set_subtensor(self.index[-1],T.zeros_like(self.index[-1])), self.index)
      if x_t.attrs['sparse']:
        if x_t.output.ndim == 3: out_dim = x_t.output.shape[2]
        elif x_t.output.ndim == 2: out_dim = 1
        else: assert False, x_t.output.ndim
        if x_t.output.ndim == 3:
          z += W[T.cast(x_t.output[:,:,0], 'int32')]
        elif x_t.output.ndim == 2:
          z += W[T.cast(x_t.output, 'int32')]
        else:
          assert False, x_t.output.ndim
      elif m is None:
        z += T.dot(x_t.output, W)
      else:
        z += self.dot(self.mass * m * x_t.output, W)
    num_batches = self.index.shape[1]
    self.num_batches = num_batches
    non_sequences = []
    if self.attrs['lm'] or attention_lm != 'none':
      if not 'target' in self.attrs:
        self.attrs['target'] = 'classes'
      if self.attrs['droplm'] > 0.0 or not (self.train_flag or force_lm):
        if copy_weights_from_base:
          self.W_lm_in = base[0].W_lm_in
          self.b_lm_in = base[0].b_lm_in
        else:
          l = sqrt(6.) / sqrt(unit.n_out + self.y_in[self.attrs['target']].n_out)
          values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(unit.n_out, self.y_in[self.attrs['target']].n_out)), dtype=theano.config.floatX)
          self.W_lm_in = self.add_param(self.shared(value=values, borrow=True, name = "W_lm_in_"+self.name))
          self.b_lm_in = self.create_bias(self.y_in[self.attrs['target']].n_out, 'b_lm_in')
      l = sqrt(6.) / sqrt(unit.n_in + self.y_in[self.attrs['target']].n_out)
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(self.y_in[self.attrs['target']].n_out, unit.n_in)), dtype=theano.config.floatX)
      if copy_weights_from_base:
        self.W_lm_out = base[0].W_lm_out
      else:
        self.W_lm_out = self.add_param(self.shared(value=values, borrow=True, name = "W_lm_out_"+self.name))
      if self.attrs['droplm'] == 0.0 and (self.train_flag or force_lm):
        self.lmmask = 1
        #if recurrent_transform != 'none':
        #  recurrent_transform = recurrent_transform[:-3]
      elif self.attrs['droplm'] < 1.0 and (self.train_flag or force_lm):
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        srng = RandomStreams(self.rng.randint(1234) + 1)
        self.lmmask = T.cast(srng.binomial(n=1, p=1.0 - self.attrs['droplm'], size=self.index.shape), theano.config.floatX).dimshuffle(0,1,'x').repeat(unit.n_in,axis=2)
      else:
        self.lmmask = T.zeros_like(self.index, dtype='float32').dimshuffle(0,1,'x').repeat(unit.n_in,axis=2)

    if recurrent_transform == 'input': # attention is just a sequence dependent bias (lstmp compatible)
      src = []
      src_names = []
      n_in = 0
      for e in base:
        #src_base = [ s for s in e.sources if s.name not in src_names ]
        #src_names += [ s.name for s in e.sources ]
        src_base = [ e ]
        src_names += [e.name]
        src += [s.output for s in src_base]
        n_in += sum([s.attrs['n_out'] for s in src_base])
      self.xc = T.concatenate(src, axis=2)
      l = sqrt(6.) / sqrt(self.attrs['n_out'] + n_in)
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n_in, 1)), dtype=theano.config.floatX)
      self.W_att_xc = self.add_param(self.shared(value=values, borrow=True, name = "W_att_xc"))
      values = numpy.asarray(self.rng.uniform(low=-l, high=l, size=(n_in, self.attrs['n_out'] * 4)), dtype=theano.config.floatX)
      self.W_att_in = self.add_param(self.shared(value=values, borrow=True, name = "W_att_in"))
      zz = T.exp(T.tanh(T.dot(self.xc, self.W_att_xc))) # TB1
      self.zc = T.dot(T.sum(self.xc * (zz / T.sum(zz, axis=0, keepdims=True)).repeat(self.xc.shape[2],axis=2), axis=0, keepdims=True), self.W_att_in)
      recurrent_transform = 'none'
    elif recurrent_transform == 'attention_align':
      max_skip = base[0].attrs['max_skip']
      values = numpy.zeros((max_skip,), dtype=theano.config.floatX)
      self.T_b = self.add_param(self.shared(value=values, borrow=True, name="T_b"), name="T_b")
      l = sqrt(6.) / sqrt(self.attrs['n_out'] + max_skip)
      values = numpy.asarray(self.rng.uniform(
        low=-l, high=l, size=(self.attrs['n_out'], max_skip)), dtype=theano.config.floatX)
      self.T_W = self.add_param(self.shared(value=values, borrow=True, name="T_W"), name="T_W")
      y_t = T.dot(self.base[0].attention, T.arange(self.base[0].output.shape[0], dtype='float32'))  # NB
      y_t = T.concatenate([T.zeros_like(y_t[:1]), y_t], axis=0)  # (N+1)B
      y_t = y_t[1:] - y_t[:-1]  # NB
      self.y_t = y_t # T.clip(y_t,numpy.float32(0),numpy.float32(max_skip - 1))

      self.y_t = T.cast(self.base[0].backtrace,'float32')
    elif recurrent_transform == 'attention_segment':
      assert aligner.attention, "Segment-wise attention requires attention points!"

    recurrent_transform_inst = recurrent_transform_mod.transform_classes[recurrent_transform](layer=self)
    assert isinstance(recurrent_transform_inst, recurrent_transform_mod.RecurrentTransformBase)
    unit.recurrent_transform = recurrent_transform_inst
    self.recurrent_transform = recurrent_transform_inst
    # scan over sequence
    for s in range(self.attrs['sampling']):
      index = self.index[s::self.attrs['sampling']]

      if context > 0:
        time, batch, dim = z.shape[0], z.shape[1], z.shape[2]

        def context_window(idx, x_in, i_in):
          x_out = x_in[idx:idx + context]
          x_out = x_out.dimshuffle(1,0,2).reshape((batch, dim * context))
          i_out = i_in[idx:idx + 1].repeat(context, axis=0).dimshuffle(1,0)
          return x_out, i_out

        assert context % context_span == 0
        assert context_span <= context
        if context_span > 1:
          pad = context_span - (time % context_span)
          z = ifelse(T.eq(time % context_span, 0), z,
                     T.concatenate([z, T.zeros((pad, batch, dim), dtype='float32')], axis=0))
          index = ifelse(T.eq(time % context_span, 0), index,
                         T.concatenate([index, T.ones((pad, batch), dtype='int8')], axis=0))
          time = index.shape[0]
        z = z[::direction or 1]
        index = index[::direction or 1]
        if context_span < context:
          z = T.concatenate([T.zeros((context - context_span, z.shape[1], z.shape[2]), dtype='float32'), z])
        if context_span > 1:
          z = T.concatenate([z, T.zeros((context_span - 1, z.shape[1], z.shape[2]), dtype='float32')])
        out, _ = theano.map(context_window, sequences=[T.arange(0,time,context_span)], non_sequences=[z, index])
        z = out[0][::direction or 1]  # (T/S)B(DC)
        index = out[1][::direction or 1]  # (T/S)BC
        original_direction = direction
        direction = 1

        z = z.reshape((time * batch // context_span, context * dim))  # ((T/S)B)(CD)
        z = z.reshape((time * batch // context_span, context, dim)).dimshuffle(1,0,2)  # C((T/S)B)D
        index = index.reshape((time * batch // context_span, context)).dimshuffle(1,0)  # C((T/S)B)

        num_batches = time * batch // context_span

      sequences = z
      sources = self.sources
      if state_memory:
        outputs_info = [
          self.add_param(self.shared(numpy.zeros((state_memory or 1, unit.n_units), dtype='float32'), name='init_%d_%s' % (a, self.name)))
          for a in range(unit.n_act)]  # has to be initialized for train and test
      elif encoder:
        if recurrent_transform == "attention_segment":
          if hasattr(encoder[0],'act'):
            outputs_info = [T.concatenate([e.act[i][-1] for e in encoder], axis=1) for i in range(unit.n_act)]
          else:
            outputs_info[0] = self.aligner.output[-1]
        elif hasattr(encoder[0],'act'):
          outputs_info = [T.concatenate([e.act[i][-1] for e in encoder], axis=1) for i in range(unit.n_act)]
        else:
          outputs_info = [T.concatenate([e[i] for e in encoder], axis=1) for i in range(unit.n_act)]
        sequences += T.alloc(numpy.cast[theano.config.floatX](0), n_dec, num_batches, unit.n_in) + (self.zc if self.attrs['recurrent_transform'] == 'input' else numpy.float32(0))
      else:
        outputs_info = [T.alloc(numpy.cast[theano.config.floatX](0), num_batches, unit.n_units) for a in range(unit.n_act)]
      if self.attrs['lm'] and self.attrs['droplm'] == 0.0 and (self.train_flag or force_lm):
        if self.network.y[self.attrs['target']].ndim == 3:
          sequences += T.dot(self.network.y[self.attrs['target']],self.W_lm_out)
        else:
          y = self.y_in[self.attrs['target']].flatten()
          sequences += self.W_lm_out[y].reshape((index.shape[0],index.shape[1],unit.n_in))

      if sequences == self.b:
        sequences += T.alloc(numpy.cast[theano.config.floatX](0), n_dec, num_batches, unit.n_in) + (self.zc if self.attrs['recurrent_transform'] == 'input' else numpy.float32(0))

      if unit.recurrent_transform:
        outputs_info += unit.recurrent_transform.get_sorted_state_vars_initial()

      index_f = T.cast(index, theano.config.floatX)
      unit.set_parent(self)

      if segment_input:
        outputs = unit.scan_seg(x=sources,
                                z=sequences[s::self.attrs['sampling']],
                                att=inv_att,
                                non_sequences=non_sequences,
                                i=index_f,
                                outputs_info=outputs_info,
                                W_re=self.W_re,
                                W_in=self.W_in,
                                b=self.b,
                                go_backwards=direction == -1,
                                truncate_gradient=self.attrs['truncation'])
      else:
        outputs = unit.scan(x=sources,
                            z=sequences[s::self.attrs['sampling']],
                            non_sequences=non_sequences,
                            i=index_f,
                            outputs_info=outputs_info,
                            W_re=self.W_re,
                            W_in=self.W_in,
                            b=self.b,
                            go_backwards=direction == -1,
                            truncate_gradient=self.attrs['truncation'])

      if not isinstance(outputs, list):
        outputs = [outputs]
      if outputs:
        outputs[0].name = "%s.act[0]" % self.name
        if context > 0:
          for i in range(len(outputs)):
            # outputs[i] is C((T/S)B)D
            outputs[i] = outputs[i][-context_span:][::original_direction or 1]  # S((T/S)B)D
            outputs[i] = outputs[i].dimshuffle(1, 0, 2).reshape(
              (outputs[i].shape[0] * outputs[i].shape[1] // batch, batch, outputs[i].shape[2]))[:self.index.shape[0]]
          index = index[-context_span:][::original_direction or 1]
          self.index = index.dimshuffle(1, 0).reshape(
            (index.shape[0] * index.shape[1] // batch, batch))[:self.index.shape[0]]

      if unit.recurrent_transform:
        unit.recurrent_transform_state_var_seqs = outputs[-len(unit.recurrent_transform.state_vars):]

      if self.attrs['sampling'] > 1:
        if s == 0:
          self.act = [T.alloc(numpy.cast['float32'](0), self.index.shape[0], self.index.shape[1], n_out) for act in outputs]
        self.act = [T.set_subtensor(tot[s::self.attrs['sampling']], act) for tot,act in zip(self.act, outputs)]
      else:
        self.act = outputs[:unit.n_act]
        if len(outputs) > unit.n_act:
          self.aux = outputs[unit.n_act:]
        if state_memory:
          for i in range(len(self.act)):
            self.params["init_%d_%s" % (i, self.name)].live_update = ifelse(
              RecurrentUnitLayer.last_segment_flag,
              T.zeros_like(self.act[i][0]), self.act[i][state_memory_pos])
    if self.attrs['attention_store']:
      self.attention = [self.aux[i].dimshuffle(0,2,1) for i,v in enumerate(sorted(unit.recurrent_transform.state_vars.keys())) if v.startswith('att_')] # NBT
      for i in range(len(self.attention)):
        vec = T.eye(self.attention[i].shape[2], 1, -direction * (self.attention[i].shape[2] - 1))
        last = vec.dimshuffle(1, 'x', 0).repeat(self.index.shape[1], axis=1)
        self.attention[i] = T.concatenate([self.attention[i][1:],last],axis=0)[::direction]

    self.cost_val = numpy.float32(0)
    if recurrent_transform == 'attention_align':
      back = T.ceil(self.aux[sorted(unit.recurrent_transform.state_vars.keys()).index('t')])
      def make_output(base, yout, trace, length):
        length = T.cast(length, 'int32')
        idx = T.cast(trace[:length][::-1],'int32')
        x_out = T.concatenate([base[idx],T.zeros((self.index.shape[0] + 1 - length, base.shape[1]), 'float32')],axis=0)
        y_out = T.concatenate([yout[idx,T.arange(length)],T.zeros((self.index.shape[0] + 1 - length, ), 'float32')],axis=0)
        return x_out, y_out

      output, _ = theano.map(make_output,
                             sequences = [base[0].output.dimshuffle(1,0,2),
                                          self.y_t.dimshuffle(1,2,0),
                                          back.dimshuffle(1,0),
                                          T.sum(self.index,axis=0,dtype='float32')])
      self.attrs['n_out'] = base[0].attrs['n_out']
      self.params.update(unit.params)
      self.output = output[0].dimshuffle(1,0,2)[:-1]

      z = T.dot(self.act[0], self.T_W)[:-1] + self.T_b
      z = z.reshape((z.shape[0] * z.shape[1], z.shape[2]))
      idx = (self.index[1:].flatten() > 0).nonzero()
      idy = (self.index[1:][::-1].flatten() > 0).nonzero()
      y_out = T.cast(output[1],'int32').dimshuffle(1, 0)[:-1].flatten()
      nll, _ = T.nnet.crossentropy_softmax_1hot(x=z[idx], y_idx=y_out[idy])
      self.cost_val = T.sum(nll)
      recog = T.argmax(z[idx], axis=1)
      real = y_out[idy]
      self.errors = lambda: T.sum(T.neq(recog, real))

    if recurrent_transform == 'batch_norm':
      self.params['sample_mean_batch_norm'].custom_update = T.dot(T.mean(self.act[0],axis=[0,1]),self.W_re)
      self.params['sample_mean_batch_norm'].custom_update_normalized = True

    self.make_output(self.act[0][::direction or 1], sample_mean=sample_mean, gamma=gamma)
    self.params.update(unit.params)

  def cost(self):
    """
    :rtype: (theano.Variable | None, dict[theano.Variable,theano.Variable] | None)
    :returns: cost, known_grads
    """
    cost_val = self.cost_val
    if self.unit.recurrent_transform:
      transform_cost = self.unit.recurrent_transform.cost()
      if transform_cost is not None:
        cost_val += transform_cost
    return cost_val, {}

  def create_seg_wise_encoder_output(self, att, aligner=None):
    assert aligner,"please provide an inverted aligner!"
    t = self.base[0].output.shape[0]
    b = self.base[0].output.shape[1]
    att_with_first_index = T.concatenate([T.zeros((1,att.shape[1]))-numpy.float32(1),att],axis=0) #(N+1)B
    max_diff = T.cast(T.extra_ops.diff(att_with_first_index,axis=0).flatten().sort()[-1],'int32')
    reduced_index = aligner.reduced_index.repeat(max_diff).reshape((aligner.reduced_index.shape[0], aligner.reduced_index.shape[1],max_diff)) #NB(max_diff)
    att_wo_last_ind = att_with_first_index[:-1] #NB
    att_wo_last_ind +=numpy.int32(1)
    att_rep = att_wo_last_ind.repeat(max_diff).reshape((att_wo_last_ind.shape[0],att_wo_last_ind.shape[1],max_diff))#NB(max_diff)
    att_rep = T.switch(reduced_index>0, att_rep + T.arange(max_diff),T.zeros((1,),'float32')-numpy.float32(1))
    att_rep = att_rep.dimshuffle(0,2,1) #N(max_diff)B
    reduced_index = reduced_index.dimshuffle(0,2,1) #N(max_diff)B
    att_rep = T.switch(reduced_index > 0,att_rep + (T.arange(b) * t),T.zeros((1,),'float32')-numpy.float32(1))
    att_rep = att_rep.clip(0,(t*b-1))
    diff_arr = att_with_first_index[1:]-att_with_first_index[:-1]
    diff_arr = diff_arr.clip(0,max_diff) - numpy.float32(1)#NB
    mask = diff_arr.dimshuffle(0,'x',1).repeat(max_diff,axis=1) - T.arange(max_diff).dimshuffle('x',0,'x')
    ind = T.cast(T.where(T.lt(mask,numpy.float32(0)),T.zeros((1,),'float32'),numpy.float32(1)),'int8')
    self.rec_transform_enc = att_rep
    self.rec_transform_index = ind



class RecurrentUpsampleLayer(RecurrentUnitLayer):
  layer_class = 'recurrent_upsample'

  def __init__(self, factor, **kwargs):
    h = T.concatenate([s.act[0] for s in kwargs['sources']],axis=2)
    time = h.shape[0]
    batch = h.shape[1]
    src = Layer([],sum([s.attrs['n_out'] for s in kwargs['sources']]),kwargs['index'])
    src.output = h.reshape((1,h.shape[0] * h.shape[1], h.shape[2])).repeat(factor,axis=0)
    src.index = kwargs['sources'][0].index
    src.layer_class = ''
    kwargs['sources'] = [ src ]
    kwargs['index'] = kwargs['index'].flatten().dimshuffle('x',0).repeat(factor,axis=0)
    kwargs['n_dec'] = factor
    super(RecurrentUpsampleLayer, self).__init__(**kwargs)
    self.index = self.index.reshape((self.index.shape[0]*time,batch))
    self.output = self.output.reshape((self.output.shape[0]*time,batch,self.output.shape[2]))


class LinearRecurrentLayer(HiddenLayer):
  """
  Inspired from: http://arxiv.org/abs/1510.02693
  Basically a very simple LSTM.
  """
  recurrent = True
  layer_class = "linear_recurrent"

  def __init__(self, n_out, direction=1, **kwargs):
    super(LinearRecurrentLayer, self).__init__(n_out=n_out, **kwargs)
    self.set_attr('direction', direction)
    a = T.nnet.sigmoid(self.add_param(self.create_bias(n_out, "a")))
    x = self.get_linear_forward_output()  # time,batch,n_out
    i = T.cast(self.index, dtype="float32")  # so that it can run on gpu. (time,batch)
    def step(x_t, i_t, h_p):
      i_t_bc = i_t.dimshuffle(0, 'x')  # batch,n_out
      return (a * h_p + x_t) * i_t_bc + h_p * (numpy.float32(1) - i_t_bc)
    n_batch = x.shape[1]
    h_initial = T.zeros((n_batch, n_out), dtype="float32")
    go_backwards = {1:False, -1:True}[direction]
    h, _ = theano.scan(step, sequences=[x, i], outputs_info=[h_initial], go_backwards=go_backwards)
    h = h[::direction]
    self.output = self.activation(h)
