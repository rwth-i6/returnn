

from __future__ import print_function

import numpy
import sys
import os
import h5py
from Network import LayerNetwork
from EngineTask import TrainTaskThread, EvalTaskThread, HDFForwardTaskThread, ClassificationTaskThread, PriorEstimationTaskThread
import tornado.web
from tornado.ioloop import IOLoop
from tornado import gen
import Device


#TODO: implement classification handler
class ClassifyHandler(tornado.web.RequestHandler):
  pass
  #@gen.coroutine
  #def get(self):
  

#TODO: implement new config handler


#TODO: implement training handler


class Server:
  
  #TODO: implement multi engine management
  
  enigines = []
  
  def __init__(self, devices):
    """
        :type devices: list[Device.Device]
    """
    application = tornado.web.Application([
      (r"/classify", ClassifyHandler),
    ])
    
    application.listen(3033)
    IOLoop.instance().start()



  
  
