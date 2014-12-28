#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

from Log import log
from Config import Config

from Network import Network
from Engine import Engine
from Dataset import load_data

def load_config():
  # initialize config file
  assert os.path.isfile(sys.argv[1]), "config file not found"  
  config = Config()
  config.load_file(sys.argv[1])
  return config

def check_usage():
  if len(sys.argv) != 2:
    print "usage:", sys.argv[0], "config"
    sys.argv = [sys.argv[0], 'lastconfig']
 
    #sys.argv = [sys.argv[0], dir + 'lastconfig']
    #os.chdir(dir)

def init_log(config):
  logs = config.list('log')
  log_verbosity = config.int_list('log_verbosity')
  log_format = config.list('log_format')
  log.initialize(logs = logs, verbosity = log_verbosity, formatter = log_format)
  
def main():    
  check_usage()
  config = load_config()
  init_log(config)
  data_train, data_valid, data_test = load_data('data/')
  input_dim = data_train[0].shape[3]
  network = Network(config, input_dim)
  engine = Engine(config, network)
  engine.train(data_train, data_valid)

#--------------------------------------------------------

#test if TwoDLSTMLayer works correctly by comparing to SlowTwoDLSTMLayer
def test():
  check_usage()
  config = load_config()
  init_log(config) 
    
  import numpy
  import theano
  import theano.tensor as T  
  from Network import SlowTwoDLSTMLayer
  from Network import TwoDLSTMLayer
  
  theano.config.mode = 'FAST_COMPILE'
  theano.config.optimizer = 'None'
  theano.config.exception_verbosity = 'high'
  
  x = T.ftensor4('x')
  batch = 1
  n_in = 1
  n_cells = 2 #strange bug: n_cells must be atleast 2!

  #for height,width in [[10,5],[5,10]]:
  #for height,width in [[2,2],[10,5]]:
  for height,width in [[2,2],[10,5],[2,3],[5,10]]:
    #height x width x batch x feature
    #x_test = numpy.ones((height,width,batch,n_in), dtype=theano.config.floatX)
    x_test = numpy.cast[theano.config.floatX](numpy.random.random((height,width,batch,n_in)))

    name = 'SlowTwoDLSTMLayer'
    layer = SlowTwoDLSTMLayer(x, n_in, n_cells, name)
    f = theano.function(inputs=[x], outputs=layer.y)
    y1 = f(x_test)
    #print name, '\n', y1
  
    name = 'TwoDLSTMLayer'
    layer = TwoDLSTMLayer(x, height, width, n_in, n_cells, name)
    f = theano.function(inputs=[x], outputs=layer.y)
    y2 = f(x_test)
    print name, '\n', y2.shape, '\n', y2
  
    assert numpy.allclose(y1, y2)
  
  print 'done'
  
#(almost) minimal example to reproduce the bug with empty ranges on gpu
def test2():
  import theano
  import theano.tensor as T
  import numpy  
  theano.config.mode = 'FAST_COMPILE'
  theano.config.optimizer = 'None'
  theano.config.exception_verbosity = 'high'
  
  height = 2
  width = 2
  batch_size = 2
  n_cells = 2
  n_in = 2
  diag_size = min(width, height)
  
  #diag_infos = numpy.array([[0,0,],[0,1],[1,2]], dtype='int16')
  diag_infos = numpy.array([[0,0],[0,1]], dtype='int16')
  
  V_u = theano.shared(numpy.zeros((n_cells, 5*n_cells), dtype=theano.config.floatX))  
  x = T.ftensor3('x')
    
  #index d for diagonal index
  def _step(diag_info, y_d1, x):
    start = diag_info[0]
    end = diag_info[1]  
    y_u1 = y_d1[start:end]
    z_t = T.alloc(numpy.cast[theano.config.floatX](0), diag_size, batch_size, 5*n_cells)
    z_t = T.inc_subtensor(z_t[start:end], T.dot(y_u1, V_u))      
    #z_t = T.inc_subtensor(z_t[start:end], T.dot(y_u1, V_u) * (end > start))      
    #z_t = T.switch(end > start, T.inc_subtensor(z_t[start:end], T.dot(y_u1, V_u)), z_t)      
    return z_t[:,:,:n_cells]
  
  y, _ = theano.scan(_step, sequences=[diag_infos],
                        non_sequences=[x],
                        outputs_info = [T.alloc(numpy.cast[theano.config.floatX](0), diag_size, batch_size, n_cells)])
             	                       
  f = theano.function(inputs=[x], outputs=y)
  x_test = numpy.zeros((height * width, batch_size, n_in), dtype=theano.config.floatX)
  print f(x_test)

if __name__ == '__main__':
  #test()
  #test2()
  main()
