from TaskSystem import AsyncTask, ProcConnectionDied
from Log import log
import numpy
import socket

try:
  from thread import start_new_thread
except ImportError:
  # noinspection PyUnresolvedReferences
  from _thread import start_new_thread

class NetworkStream:
  def __init__(self, name, port):
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.bind((socket.gethostname(), port))
    self.socket.listen(5)
    self.clients = {}
    self.name = name
    self.port = port
    start_new_thread(self.server_loop,())

  def server_loop(self):
    while True:
      (client, address) = self.socket.accept()
      self.clients[address] = client

  def update(self, batch):
    batch = numpy.asarray(batch).tostring()
    for address,client in self.clients.values():
      totalsent = 0
      while totalsent < len(batch):
        sent = self.sock.send(batch[totalsent:])
      if sent == 0:
        del self.clients[client]
        print >> log.v1, "client disconnected:", address
