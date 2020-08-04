
from __future__ import print_function
from returnn.util.task_system import AsyncTask, ProcConnectionDied
from returnn.log import log
import numpy
import json
try:
  import SimpleHTTPServer
  import SocketServer
  import BaseHTTPServer
except ImportError:  # Python3
  import http.server as SimpleHTTPServer
  import socketserver as SocketServer
  BaseHTTPServer = SimpleHTTPServer

try:
  from thread import start_new_thread
except ImportError:
  # noinspection PyUnresolvedReferences
  from _thread import start_new_thread


class NetworkStream:
  class ThreadingServer(SocketServer.ThreadingMixIn, BaseHTTPServer.HTTPServer):
    pass

  def count(self):
    return {'count': self.counter}

  def data(self, start_index, count = 1):
    idx = len(self.cache) - (self.counter - start_index)
    if idx + count > self.cache_size:
      count = self.cache_size - idx
    if idx < 0:
      return {'error': 'requested batch too old'}
    if idx >= self.cache_size:
      return {'error': 'requested batch too new'}
    else:
      return self.cache[idx:idx+count]

  def __init__(self, name, port, cache_size = 100):
    self.name = name
    self.cache = []
    self.cache_size = cache_size
    self.counter = 0
    from jsonrpclib.SimpleJSONRPCServer import SimpleJSONRPCServer  # https://pypi.python.org/pypi/jsonrpclib/0.1.6
    server = SimpleJSONRPCServer(('0.0.0.0', port))
    server.register_function(self.count, 'count')
    server.register_function(self.data, 'data')
    print("json-rpc streaming on port", port, file=log.v3)
    start_new_thread(server.serve_forever,())

  def update(self, task, data, tags = []):
    data = numpy.asarray(data)
    package = {'task' : task, 'data' : data.tostring(), 'tags' : tags, 'dtype' : str(data.dtype), 'shape' : data.shape}
    self.cache.append(package)
    if len(self.cache) > self.cache_size:
      self.cache.pop(0)
    self.counter += 1
