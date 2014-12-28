#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from numpy import sqrt
import cPickle
import theano
import theano.tensor as T
from Log import log

class Layer(object):
  def __init__(self, name=''):
    self.name = name
    self.rng = numpy.random.RandomState(1234)
    
  def _create_bias_weights(self, n):
    return theano.shared(numpy.zeros((n,), dtype=theano.config.floatX), borrow=True, name='b_' + self.name)

  def _create_forward_weights(self, n, m):
    n_in = n + m
    scale = numpy.sqrt(12. / (n_in))
    return self._create_random_weights(n, m, scale)
    
  def _create_recurrent_weights(self, n, m):
    #nin = n + m + m + m
    #return self._create_random_weights(n, m, nin), self._create_random_weights(m, m, nin)
    #TODO für zu große hidden sizes klappt das nicht gut..
    return self._create_uniform_weights(n, m, n + m), self._create_uniform_weights(m, m, n + m)

  def _create_random_weights(self, n, m, s):
    values = numpy.asarray(self.rng.normal(loc = 0.0, scale = s, size=(n, m)), dtype=theano.config.floatX)
    return theano.shared(value = values, borrow = True, name = 'W_' + self.name)
    
  def _create_uniform_weights(self, n, m, p = 0):
    if p == 0: p = n + m
    values = numpy.asarray(self.rng.uniform(low = - sqrt(6) / sqrt(p), high = sqrt(6) / sqrt(p), size=(n, m)), dtype=theano.config.floatX)
    return theano.shared(value = values, borrow = True, name = 'W_' + self.name)
    
  def _create_lstm_weights(self, n, m):
    return self._create_uniform_weights(n, m * 4, n + m + m * 4), self._create_uniform_weights(m, m * 4, n + m + m * 4)
    
  def _create_2d_lstm_weights(self, n, m):
    return self._create_uniform_weights(n, m * 5, n + m + m * 5), self._create_uniform_weights(m, m * 5, n + m + m * 5), self._create_uniform_weights(m, m * 5, n + m + m * 5)

  def _create_2d_forward_weights(self, n, m):
    return self._create_uniform_weights(n, m, n + m)

class ForwardLayer(Layer):  
  def __init__(self, x, n_in, n_out, name=''):
    super(ForwardLayer, self).__init__(name)
    self.name = name
    self.W = self._create_forward_weights(n_in, n_out)
    self.b = self._create_bias_weights(n_out)
    self.params = [self.W, self.b]
    self.z = T.dot(x, self.W) + self.b
    self.y = self.z

class ForwardAndActivationLayer(ForwardLayer):
  def __init__(self, x, n_in, n_out, f):
    super(ForwardAndActivationLayer, self).__init__(x, n_in, n_out)
    self.y = f(self.z)

class SigmoidLayer(ForwardAndActivationLayer):
  def __init__(self, x, n_in, n_out):
    super(SigmoidLayer, self).__init__(x, n_in, n_out, T.nnet.sigmoid)

class SoftmaxLayer(ForwardLayer):
  def __init__(self, x, n_in, n_out):
    super(SoftmaxLayer, self).__init__(x, n_in, n_out)
    y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1], self.z.shape[2]))
    self.y = T.nnet.softmax(y_m)
    self.y = T.reshape(self.y, self.z.shape)
    
class TwoDSoftmaxLayer(Layer):
  def __init__(self, x, n_in, n_out):
    super(TwoDSoftmaxLayer, self).__init__()
    self.W = self._create_2d_forward_weights(n_in, n_out)
    self.b = self._create_bias_weights(n_out)
    self.params = [self.W, self.b]
    self.z = T.dot(x, self.W) + self.b    
    y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1] * self.z.shape[2], self.z.shape[3]))
    self.y = T.nnet.softmax(y_m)
    self.y = T.reshape(self.y, self.z.shape)    
    self.y = T.mean(T.mean(self.y, axis=0), axis=0)
    
class MultiSourceTwoDSoftmaxLayer(Layer):
  def __init__(self, sources, n_ins, n_out):
    super(MultiSourceTwoDSoftmaxLayer, self).__init__()
    self.Ws = [self._create_2d_forward_weights(n_in, n_out) for n_in in n_ins]
    self.b = self._create_bias_weights(n_out)
    self.params = self.Ws + [self.b]
    self.z = self.b
    for x, W in zip(sources, self.Ws):
      self.z += T.dot(x, W)
    y_m = T.sum(self.z, axis=[0,1])
    self.y = T.nnet.softmax(y_m)
    
class NoCollapseMultiSourceTwoDSoftmaxLayer(Layer):
  def __init__(self, sources, n_ins, n_out):
    super(NoCollapseMultiSourceTwoDSoftmaxLayer, self).__init__()
    self.Ws = [self._create_2d_forward_weights(n_in, n_out) for n_in in n_ins]
    self.b = self._create_bias_weights(n_out)
    self.params = self.Ws + [self.b]
    self.z = self.b
    for x, W in zip(sources, self.Ws):
      self.z += T.dot(x, W)
    y_m = T.reshape(self.z, (self.z.shape[0] * self.z.shape[1] * self.z.shape[2], self.z.shape[3]))
    self.y = T.nnet.softmax(y_m)
    self.y = T.reshape(self.y, self.z.shape)    
    self.y = T.mean(T.mean(self.y, axis=0), axis=0)

class RecurrentLayer(Layer):
  def __init__(self, x, n_in, n_out, name=''):
    super(RecurrentLayer, self).__init__(name)
    self.W, self.V = self._create_recurrent_weights(n_in, n_out)
    self.b = self._create_bias_weights(n_out)    
    self.initial_act = self._create_bias_weights(n_out)
    self.params = [self.W, self.V, self.b]

    def _step(x_t, y_tm1):
      z_t = T.dot(x_t, self.W) + T.dot(y_tm1, self.V) + self.b
      y_t = T.nnet.sigmoid(z_t)
      return y_t
      
    self.y, _ = theano.scan(_step, sequences=[x],
                            outputs_info=[T.alloc(self.initial_act, x.shape[1], n_out)])
                            
class LSTMLayer(Layer):
  def __init__(self, x, n_in, n_cells, name=''):
    super(LSTMLayer, self).__init__(name)
    self.W, self.V = self._create_lstm_weights(n_in, n_cells)
    self.b = self._create_bias_weights(4 * n_cells)  
    self.initial_act = self._create_bias_weights(n_cells)
    self.initial_state = self._create_bias_weights(n_cells)
    self.params = [self.W, self.V, self.b]

    def _step(x_t, c_tm1, y_tm1):
      z_t = T.dot(x_t, self.W) + T.dot(y_tm1, self.V) + self.b
      partition = z_t.shape[1] / 4
      ingate = T.nnet.sigmoid(z_t[:,:partition])
      forgetgate = T.nnet.sigmoid(z_t[:,partition:2*partition])
      outgate = T.nnet.sigmoid(z_t[:,2*partition:3*partition])
      input = T.tanh(z_t[:,3*partition:4*partition])
      c_t = forgetgate * c_tm1 + ingate * input
      y_t = outgate * T.tanh(c_t)
      return c_tm1, y_t
      
    [state, self.y], _ = theano.scan(_step, sequences=[x],
                            outputs_info=[T.alloc(self.initial_state, x.shape[1], n_cells),
                                          T.alloc(self.initial_act, x.shape[1], n_cells)])
                                          
#naive implementation with nested scan
#mainly used to check the correctness of the optimized version
class SlowTwoDLSTMLayer(Layer):    
  def __init__(self, x, n_in, n_cells, name=''):
    super(SlowTwoDLSTMLayer, self).__init__(name)
    self.W, self.V_u, self.V_v = self._create_2d_lstm_weights(n_in, n_cells)
    self.b = self._create_bias_weights(5 * n_cells)  
    self.params = [self.W, self.V_u, self.V_v, self.b]
    
    #naming conventions:
    #height x width (or other way round) = u x v
    # y_u1v2 means y_{u-1,v-2}  (no number means -0)
    def _inner_step(x_uv, c_u1v, y_u1v, c_uv1, y_uv1):
        z_t = T.dot(x_uv, self.W) + T.dot(y_u1v, self.V_u) + T.dot(y_uv1, self.V_v) + self.b
        #z_t : (batch x feature) * (feature x n_cells) -> (batch x n_cells)
        partition = z_t.shape[1] / 5
        ingate = T.nnet.sigmoid(z_t[:,0*partition:1*partition])
        forgetgate_u = T.nnet.sigmoid(z_t[:,1*partition:2*partition])
        forgetgate_v = T.nnet.sigmoid(z_t[:,2*partition:3*partition])
        outgate = T.nnet.sigmoid(z_t[:,3*partition:4*partition])
        input = T.tanh(z_t[:,4*partition:5*partition])
        c_uv = forgetgate_u * c_u1v + forgetgate_v * c_uv1 + ingate * input
        y_uv = outgate * T.tanh(c_uv)
        return c_uv, y_uv
    
    def _step(x_u, c_u1, y_u1):
        out_info = [T.alloc(numpy.cast[theano.config.floatX](0), x.shape[2],n_cells),
                    T.alloc(numpy.cast[theano.config.floatX](0), x.shape[2],n_cells)]
        [c_u,y_u], _ = theano.scan(_inner_step, outputs_info=out_info, sequences=[x_u, c_u1, y_u1])
        return c_u, y_u
    
    out_info = [T.alloc(numpy.cast[theano.config.floatX](0), x.shape[1],x.shape[2],n_cells),
                T.alloc(numpy.cast[theano.config.floatX](0), x.shape[1],x.shape[2],n_cells)]
    [_, self.y], _ = theano.scan(_step, outputs_info=out_info, sequences=[x])

#faster implementation of SlowTwoDLSTMLayer
class TwoDLSTMLayer(Layer):
  def __init__(self, x, height, width, n_in, n_cells, name=''):
    super(TwoDLSTMLayer, self).__init__(name)
    self.W, self.V_u, self.V_v = self._create_2d_lstm_weights(n_in, n_cells)
    self.b = self._create_bias_weights(5 * n_cells)  
    self.params = [self.W, self.V_u, self.V_v, self.b]
    
    #strange bug: height, width and n_cells must be atleast 2 each...
    assert height > 1 and width > 1 and n_cells > 1

    batch_size = x.shape[2]
    numel = height * width
    n_diags = width + height - 1
    diag_size = min(width, height)
    
    idx_y = numpy.zeros(numel, dtype='int16')
    idx_x = numpy.zeros(numel, dtype='int16')
    #used to unshuffle the scan result (note: different shuffling than the input)
    unshuffle_idx_y = numpy.zeros(numel, dtype='int16')
    unshuffle_idx_x = numpy.zeros(numel, dtype='int16')
    
    diag_start_idx = 0
    diag_end_idx = 1
    vert_dst_start_idx = 2
    vert_dst_end_idx = 3
    hor_dst_start_idx = 4
    hor_dst_end_idx = 5
    x_dst_start_idx = 6
    
    diag_infos = numpy.zeros((n_diags,7), dtype='int16')
        
    idx = 0
    diag_idx = 0
    if height >= width:
      for y_ in xrange(width):
        diag_start = idx
        diag_infos[diag_idx, diag_start_idx] = diag_start
        diag_infos[diag_idx, vert_dst_start_idx] = 0
        diag_infos[diag_idx, vert_dst_end_idx] = y_
        diag_infos[diag_idx, hor_dst_start_idx] = 1
        diag_infos[diag_idx, hor_dst_end_idx] = y_+1
        for x_ in xrange(y_+1):
            idx_y_ = y_-x_
            idx_x_ = x_
            idx_y[idx] = idx_y_
            idx_x[idx] = idx_x_
            unshuffle_idx_y[idx_y_*width+idx_x_] = diag_idx
            unshuffle_idx_x[idx_y_*width+idx_x_] = idx - diag_start
            idx += 1
        diag_idx += 1
      for y_ in xrange(width, height):
        diag_start = idx
        diag_infos[diag_idx, diag_start_idx] = diag_start
        diag_infos[diag_idx, vert_dst_start_idx] = 0
        diag_infos[diag_idx, vert_dst_end_idx] = width
        diag_infos[diag_idx, hor_dst_start_idx] = 1
        diag_infos[diag_idx, hor_dst_end_idx] = width
        for x_ in xrange(width):
            idx_y_ = y_-x_
            idx_x_ = x_
            idx_y[idx] = idx_y_
            idx_x[idx] = idx_x_
            unshuffle_idx_y[idx_y_*width+idx_x_] = diag_idx
            unshuffle_idx_x[idx_y_*width+idx_x_] = idx - diag_start
            idx += 1
        diag_idx += 1
      for x_ in xrange(1,width):
        diag_start = idx
        diag_infos[diag_idx, diag_start_idx] = diag_start
        diag_infos[diag_idx, vert_dst_start_idx] = x_
        diag_infos[diag_idx, vert_dst_end_idx] = width
        diag_infos[diag_idx, hor_dst_start_idx] = x_
        diag_infos[diag_idx, hor_dst_end_idx] = width
        for y_off in xrange(width-x_):
            idx_y_ = height-1-y_off
            idx_x_ = x_+y_off
            idx_y[idx] = idx_y_
            idx_x[idx] = idx_x_
            unshuffle_idx_y[idx_y_*width+idx_x_] = diag_idx
            unshuffle_idx_x[idx_y_*width+idx_x_] = idx - diag_start + x_
            idx += 1
        diag_idx += 1            
    else: #width>height     
      for x_ in xrange(height):
        diag_start = idx
        diag_infos[diag_idx, diag_start_idx] = diag_start
        diag_infos[diag_idx, vert_dst_start_idx] = height - 1 - x_
        diag_infos[diag_idx, vert_dst_end_idx] = height - 1
        diag_infos[diag_idx, hor_dst_start_idx] = height - x_
        diag_infos[diag_idx, hor_dst_end_idx] = height
        for y_ in xrange(x_+1):
            idx_y_ = x_-y_
            idx_x_ = y_
            idx_y[idx] = idx_y_
            idx_x[idx] = idx_x_
            unshuffle_idx_y[idx_y_*width+idx_x_] = diag_idx
            unshuffle_idx_x[idx_y_*width+idx_x_] = idx - diag_start + height - 1 - diag_idx
            idx += 1
        diag_idx += 1
      for x_ in xrange(height, width):
        diag_start = idx
        diag_infos[diag_idx, diag_start_idx] = diag_start
        diag_infos[diag_idx, vert_dst_start_idx] = 0
        diag_infos[diag_idx, vert_dst_end_idx] = height - 1
        diag_infos[diag_idx, hor_dst_start_idx] = 0
        diag_infos[diag_idx, hor_dst_end_idx] = height
        for y_ in xrange(height):
            idx_y_ = height-1-y_
            idx_x_ = x_-(height-1-y_)
            idx_y[idx] = idx_y_
            idx_x[idx] = idx_x_
            unshuffle_idx_y[idx_y_*width+idx_x_] = diag_idx
            unshuffle_idx_x[idx_y_*width+idx_x_] = idx - diag_start
            idx += 1
        diag_idx += 1
      for y_ in xrange(1,height):
        diag_start = idx
        diag_infos[diag_idx, diag_start_idx] = diag_start
        diag_infos[diag_idx, vert_dst_start_idx] = 0
        diag_infos[diag_idx, vert_dst_end_idx] = height - y_
        diag_infos[diag_idx, hor_dst_start_idx] = 0
        diag_infos[diag_idx, hor_dst_end_idx] = height - y_
        for x_off in xrange(height-y_):
            idx_y_ = height-1-x_off
            idx_x_ = width-(height-y_-x_off)
            idx_y[idx] = idx_y_
            idx_x[idx] = idx_x_
            unshuffle_idx_y[idx_y_*width+idx_x_] = diag_idx
            unshuffle_idx_x[idx_y_*width+idx_x_] = idx - diag_start
            idx += 1
        diag_idx += 1

    diag_infos[:-1,diag_end_idx] = diag_infos[1:,diag_start_idx]
    diag_infos[-1,diag_end_idx] = numel
    diag_infos[:,x_dst_start_idx] = diag_infos[:,vert_dst_start_idx]
    
    #hack to avoid theano bug with empty ranges on gpu
    if height >= width:
      diag_infos[0, vert_dst_end_idx] = 1
      diag_infos[0, hor_dst_end_idx] = 2
    else:
      diag_infos[0, vert_dst_start_idx] = 0
      diag_infos[0, vert_dst_end_idx] = 1
      diag_infos[0, hor_dst_start_idx] = 0
      diag_infos[0, hor_dst_end_idx] = 1
    
    x_shuffled = x[idx_y, idx_x]
    
    #index d for diagonal index
    def _step(diag_info, c_d1, y_d1, x_shuffled):
      diag_start = diag_info[diag_start_idx]
      diag_end = diag_info[diag_end_idx]
      vert_dst_start = diag_info[vert_dst_start_idx]
      vert_dst_end = diag_info[vert_dst_end_idx]
      if height >= width: #TODO parameter dafür übergeben
        vert_src_start = vert_dst_start
        vert_src_end = vert_dst_end
      else:
        vert_src_start = vert_dst_start + 1
        vert_src_end = vert_dst_end + 1
      hor_dst_start = diag_info[hor_dst_start_idx]
      hor_dst_end = diag_info[hor_dst_end_idx]
      if height >= width: #TODO parameter dafür übergeben
        hor_src_start = hor_dst_start - 1
        hor_src_end = hor_dst_end - 1
      else:
        hor_src_start = hor_dst_start
        hor_src_end = hor_dst_end
      #x_dst_start = vert_dst_start
      x_dst_start = diag_info[x_dst_start_idx]
      x_dst_end = x_dst_start + diag_end - diag_start

      x_src = x_shuffled[diag_start:diag_end]
      y_u1 = y_d1[vert_src_start:vert_src_end]
      c_u1 = c_d1[vert_src_start:vert_src_end]
      y_v1 = y_d1[hor_src_start:hor_src_end]
      c_v1 = c_d1[hor_src_start:hor_src_end]
        
      #z_t = T.dot(x_uv, self.W) + T.dot(y_u1v, self.V_u) + T.dot(y_uv1, self.V_v) + self.b -> subtensor
      z_t = T.alloc(self.b, diag_size, batch_size, 5*n_cells)
      z_t = T.inc_subtensor(z_t[x_dst_start:x_dst_end], T.dot(x_src, self.W))
      #these 2 dont work on gpu without the hack (see above), gpu seems not to like empty index ranges...
      z_t = T.inc_subtensor(z_t[vert_dst_start:vert_dst_end], T.dot(y_u1, self.V_u))      
      z_t = T.inc_subtensor(z_t[hor_dst_start:hor_dst_end], T.dot(y_v1, self.V_v))
      
      #z_t : (diag_size x batch x feature) * (feature x n_cells) -> (diag_size x batch x n_cells)
      partition = n_cells
      ingate = T.nnet.sigmoid(z_t[:,:,0*partition:1*partition])
      forgetgate_u = T.nnet.sigmoid(z_t[:,:,1*partition:2*partition])
      forgetgate_v = T.nnet.sigmoid(z_t[:,:,2*partition:3*partition])
      outgate = T.nnet.sigmoid(z_t[:,:,3*partition:4*partition])
      input = T.tanh(z_t[:,:,4*partition:5*partition])
      #c_d = forgetgate_u * c_u1 + forgetgate_v * c_v1 + ingate * input -> subtensor
      c_d = T.alloc(numpy.cast[theano.config.floatX](0), diag_size, batch_size, n_cells)
      c_d = T.set_subtensor(c_d[vert_dst_start:vert_dst_end], forgetgate_u[vert_dst_start:vert_dst_end] * c_u1)
      c_d = T.inc_subtensor(c_d[hor_dst_start:hor_dst_end], forgetgate_v[hor_dst_start:hor_dst_end] * c_v1)
      c_d += ingate * input #TODO maybe subtensor
      y_d = outgate * T.tanh(c_d) #TODO maybe subtensor
      return c_d, y_d

    [_, self.y], _ = theano.scan(_step, sequences=[diag_infos],
                                     non_sequences=[x_shuffled],
                        outputs_info = [T.alloc(numpy.cast[theano.config.floatX](0), diag_size, batch_size, n_cells),
             	                       T.alloc(numpy.cast[theano.config.floatX](0), diag_size, batch_size, n_cells)])
    
    #print 'idx_y\n', idx_y
    #print 'idx_x\n', idx_x
    #print 'unshuffle_idx_y\n', unshuffle_idx_y
    #print 'unshuffle_idx_x\n', unshuffle_idx_x
    #print 'diag_infos\n', diag_infos
    
    #unshuffle y
    self.y =  self.y[unshuffle_idx_y, unshuffle_idx_x].reshape((height, width, batch_size, n_cells))

class Network:
  def __init__(self, config, input_dim):
    self.x = T.ftensor4('x')
    self.x.tag.test_value = numpy.zeros((28,28,128,1), theano.config.floatX)

    #input_dim = config.int('input_dim', default=0, force_key=True)
    output_dim = config.int('output_dim', default=0, force_key=True)
    hidden_sizes = config.int_list('hidden_sizes', force_key=True)

    print >> log.v4, 'hidden_sizes', hidden_sizes

    self.params = []
    self.layers = []
    
    #TODO
    height = 28
    width = 28

    inp = self.x
    inp_dim = input_dim
    #for h in hidden_sizes:
    #  print >> log.v4, 'TwoDLSTMLayer'
    #  self.layers.append(TwoDLSTMLayer(inp, height, width, inp_dim, h))
    #  self.params.extend(self.layers[-1].params)
    #  inp = self.layers[-1].y
    #  inp_dim = h
    #  self.layers.append(TwoDSoftmaxLayer(inp, inp_dim, output_dim))
    
    print >> log.v4, '4-directional TwoDLSTMLayer'
    assert len(hidden_sizes) == 1
    h = hidden_sizes[-1]
    top_down_left_right = TwoDLSTMLayer(inp, height, width, inp_dim, h)
    bot_up_left_right = TwoDLSTMLayer(inp[::-1], height, width, inp_dim, h)
    top_down_right_left = TwoDLSTMLayer(inp[:,::-1], height, width, inp_dim, h)
    bot_up_right_left = TwoDLSTMLayer(inp[::-1,::-1], height, width, inp_dim, h)
    direction_layers = [top_down_left_right, bot_up_left_right, top_down_right_left, bot_up_right_left]
    direction_outputs = [top_down_left_right.y, bot_up_left_right.y[::-1], top_down_right_left.y[:,::-1], bot_up_right_left.y[::-1,::-1]]
    self.layers.extend(direction_layers)
    for layer in direction_layers:
      self.params.extend(layer.params)
    #self.layers.append(MultiSourceTwoDSoftmaxLayer(direction_outputs, [h] * 4, output_dim))
    self.layers.append(NoCollapseMultiSourceTwoDSoftmaxLayer(direction_outputs, [h] * 4, output_dim))
    self.params.extend(self.layers[-1].params)
    
    self.p_y_given_x = self.layers[-1].y
    self.y_pred = T.argmax(self.p_y_given_x, axis=1)

  def _cost(self, y):
    return -T.sum(T.log(self.p_y_given_x[T.arange(y.shape[0]),y]), axis=0)
    
  def _errors(self, y):
    return T.sum(T.neq(self.y_pred, y))

  def compile_train_fn(self, y, x_shared, y_shared, learning_rate, momentum):
    cost = self._cost(y)    
    errors = self._errors(y)
    deltas = dict([(p, theano.shared(value = numpy.zeros(p.get_value().shape, dtype = theano.config.floatX))) for p in self.params])
    updates = []
    for param in self.params:
      upd = momentum * deltas[param] - learning_rate * T.grad(cost, param)
      updates.append((deltas[param], upd))
      updates.append((param, param + upd))

    return theano.function(inputs=[], outputs=[cost,errors], givens={self.x : x_shared, y : y_shared}, updates=updates), deltas

  def compile_train_fn_adadelta(self, y, x_shared, y_shared, decay, offset):
    cost = self._cost(y)    
    errors = self._errors(y)
    eg2 = dict([(p, theano.shared(value = numpy.zeros(p.get_value().shape, dtype = theano.config.floatX))) for p in self.params]) #E[g^2]
    edx2 = dict([(p, theano.shared(value = numpy.zeros(p.get_value().shape, dtype = theano.config.floatX))) for p in self.params]) #E[\delta x^2]
    dx = dict([(p, theano.shared(value = numpy.zeros(p.get_value().shape, dtype = theano.config.floatX))) for p in self.params]) #\delta x
    updates = []
    for param in self.params:
      g = T.grad(cost, param)      
      g2 = g ** 2
      eg2_new = decay * eg2[param] + (1-decay) * g2
      dx_new = - T.sqrt(edx2[param] + offset) / T.sqrt(eg2_new + offset) * g
      edx2_new = decay * edx2[param] + (1-decay) * dx_new ** 2
      updates.append((eg2[param], eg2_new))
      updates.append((edx2[param], edx2_new))
      updates.append((dx[param], dx_new))
      updates.append((param, param + dx_new))

    return theano.function(inputs=[], outputs=[cost,errors], givens={self.x : x_shared, y : y_shared}, updates=updates)
  
  def compile_eval_fn(self, y, x_shared, y_shared):
    cost = self._cost(y)
    errors = self._errors(y)
    return theano.function(inputs=[], outputs=[cost,errors], givens={self.x : x_shared, y : y_shared})
    #return theano.function(inputs=[], outputs=[cost,errors], givens={self.x : x_shared, y : y_shared}, mode='FAST_COMPILE')

  def save(self, filename):
    with open(filename, 'wb') as f:
      for param in self.params:
        cPickle.dump(param.get_value(), f)
        