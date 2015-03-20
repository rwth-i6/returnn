#!/usr/bin/env python

import time
import numpy
import theano
import theano.tensor as T

from Log import log

class Engine(object):
  def __init__(self, config, network):
    self.batch_size = config.int('batch_size', 128)
    print >> log.v4, 'batch size:', self.batch_size
    self.learning_rate = config.float('learning_rate', 0.01)
    #print >> log.v4, 'learning rate:', self.learning_rate
    self.momentum = config.float('momentum', 0.9)
    #print >> log.v4, 'momentum:', self.momentum
    self.num_epochs = config.int('num_epochs', 100)
    self.nesterov_momentum = config.bool('nesterov_momentum', True)
    #print >> log.v4, 'nesterov_momentum:', self.nesterov_momentum

    self.adadelta_decay = 0.95
    print >> log.v4, 'adadelta_decay:', self.adadelta_decay
    self.adadelta_offset = 1e-6
    print >> log.v4, 'adadelta_offset:', self.adadelta_offset

    self.model = config.value('model', None)
    if self.model is not None:
      print >> log.v4, 'model', self.model

    self.network = network

    self.y = T.ivector('y')
    self.y.tag.test_value = numpy.zeros(128, 'int32')
    self.x_shared = theano.shared(numpy.zeros((1,1,1,1), dtype=theano.config.floatX), borrow=True)
    self.t_shared = theano.shared(numpy.zeros((1,), dtype=theano.config.floatX), borrow=True)
    y_shared = T.cast(self.t_shared, 'int32')
    self.lr_shared = theano.shared(numpy.cast[theano.config.floatX](self.learning_rate))
    self.momentum_shared = theano.shared(numpy.cast[theano.config.floatX](self.momentum))

    #self.train_model, self.deltas = network.compile_train_fn(self.y, self.x_shared, y_shared, self.lr_shared, self.momentum_shared)
    self.train_model = network.compile_train_fn_adadelta(self.y, self.x_shared, y_shared, self.adadelta_decay, self.adadelta_offset)
    self.eval_model = network.compile_eval_fn(self.y, self.x_shared, y_shared)

  def _train_epoch(self, train_x, train_y):
    n_images = train_y.size
    n_images_processed = 0
    total_score, total_errs = 0,0
    while n_images_processed < n_images:
      this_batch_size = min(n_images - n_images_processed, self.batch_size)
      batch_start = n_images_processed
      batch_end = n_images_processed + this_batch_size

      self.x_shared.set_value(train_x[:,:,batch_start:batch_end,:])
      self.t_shared.set_value(train_y[batch_start:batch_end])

      #if self.nesterov_momentum:
      #  for param in self.network.params:
      #     param.set_value(param.get_value() + self.momentum * self.deltas[param].get_value())

      score, errs = self.train_model()

      #if self.nesterov_momentum:
      #  for param in self.network.params:
      #  param.set_value(param.get_value() - self.momentum * self.deltas[param].get_value())

      total_score += score
      total_errs += errs
      n_images_processed += this_batch_size
    return total_score / n_images, float(total_errs) / n_images

  def _eval(self, data_x, data_y):
    n_images = data_y.size
    n_images_processed = 0
    total_score, total_errs = 0,0
    while n_images_processed < n_images:
      this_batch_size = min(n_images - n_images_processed, 3 * self.batch_size) #use 3 times larger batch size for evaluation
      batch_start = n_images_processed
      batch_end = n_images_processed + this_batch_size

      self.x_shared.set_value(data_x[:,:,batch_start:batch_end,:])
      self.t_shared.set_value(data_y[batch_start:batch_end])
      score, errs = self.eval_model()
      total_score += score
      total_errs += errs
      n_images_processed += this_batch_size
    return total_score / n_images, float(total_errs) / n_images

  def train(self, data_train, data_valid):
    train_x, train_y = data_train
    valid_x, valid_y = data_valid

    print >> log.v5, 'starting training...'
    for epoch in xrange(1, self.num_epochs+1):
      epoch_start = time.clock()
      #train_score, train_err = 0, 0
      #valid_score, valid_err = 0, 0
      train_score, train_err = self._train_epoch(train_x, train_y)
      valid_score, valid_err = self._eval(valid_x, valid_y)
      epoch_end = time.clock()
      elapsed = epoch_end - epoch_start
      print >> log.v1, 'epoch', epoch, 'elapsed:', elapsed, 'train: score:', train_score, 'err:', train_err, 'valid: score:', valid_score, 'err:', valid_err
      if self.model is not None:
        self.network.save_hdf(self.model + '.' + "{0:04d}".format(epoch))
