from TaskSystem import AsyncTask, ProcConnectionDied
from Log import log
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

  def data(self, idx):
    idx = len(self.cache) - (self.counter - idx)
    if idx < 0:
      return {'error': 'requested batch too old'}
    else:
      return self.cache[idx]

  def __init__(self, name, port, cache_size = 100):
    from jsonrpclib.SimpleJSONRPCServer import SimpleJSONRPCServer  # https://pypi.python.org/pypi/jsonrpclib/0.1.6
    server = SimpleJSONRPCServer(('0.0.0.0', port))
    server.register_function(self.count, 'count')
    server.register_function(self.data, 'data')
    print >> log.v3, "json-rpc streaming on port", port
    server.serve_forever()

  def update(self, task, data, tags = []):
    package = {'task' : task, 'data' : numpy.asrray(data).tostring(), 'tags' : tags}
    self.cache.append(package)
    if len(self.cache) > self.cache_size:
      self.cache = self.cache.pop(0)
    self.counter += 1
