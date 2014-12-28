#!/usr/bin/env python

import numpy
import gzip
import cPickle
import theano

def convert_to_seq(data, n_imgs, n_slices):
  data = data.reshape(n_imgs, 28, 28)
  assert 28 % n_slices == 0
  slice_size = 28 / n_slices
  data_seq = numpy.zeros((n_slices,n_imgs,28 * slice_size), dtype=theano.config.floatX)
  #time x batch x feature
  for i in xrange(n_imgs):
    for t in xrange(n_slices):
      data_seq[t,i,:] = data[i,t*slice_size:(t+1)*slice_size,:].reshape(28 * slice_size)

  return data_seq
  
def convert_to_2d(data, n_imgs):
  #height x width x batch x feature #TODO width/height richtigrum?
  data = data.reshape(n_imgs, 28, 28)
  data_2d = numpy.zeros((28,28,n_imgs,1), dtype=theano.config.floatX)
  for i in xrange(n_imgs):
    data_2d[:,:,i,0] = data[i]
  return data_2d
  

def load_data(path):
  filename = path + 'mnist.pkl.gz'
  #n_slices = 28
  
  f = gzip.open(filename, 'rb')
  train, valid, test = cPickle.load(f)
  f.close()
  
  train_x, train_y = train
  #train_x = train_x.reshape(train_y.size, 28*28)
  #train_x_seq = convert_to_seq(train_x, train_y.size, n_slices)
  train_x_2d = convert_to_2d(train_x, train_y.size)
  train_y = train_y.astype('float32')
  train = (train_x_2d, train_y)
    
  valid_x, valid_y = valid
  #valid_x = valid_x.reshape(valid_y.size, 28*28)
  #valid_x_seq = convert_to_seq(valid_x, valid_y.size, n_slices)
  valid_x_2d = convert_to_2d(valid_x, valid_y.size)
  valid_y = valid_y.astype('float32')
  valid = (valid_x_2d, valid_y)
  
  test_x, test_y = test
  #test_x = test_x.reshape(test_y.size, 28*28)
  #test_x_seq = convert_to_seq(test_x, test_y.size, n_slices)
  test_x_2d = convert_to_2d(test_x, test_y.size)
  test_y = test_y.astype('float32')
  test = (test_x_2d, test_y)
  
  return train, valid, test
  