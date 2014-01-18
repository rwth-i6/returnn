#! /usr/bin/python2.7

import SprintCache
import numpy
import theano
import time
import sys
import theano.tensor as T
from Log import log
from collections import OrderedDict
import operator
import gc

class Batch:
  def __init__(self, start = [0, 0]):
    self.shape = [0, 0]
    self.start = start
    self.nseqs = 1
  def try_sequence(self, length): return [max(self.shape[0], length), self.shape[1] + 1]
  def add_sequence(self, length): self.shape = self.try_sequence(length)
  def add_frames(self, length): self.shape = [self.shape[0] + length, max(self.shape[1], 1)]
  def size(self): return self.shape[0] * self.shape[1]

class Engine:
  def __init__(self, devices, network):
    self.network = network
    self.devices = devices
    self.gparams = dict([(p, theano.shared(value = numpy.zeros(p.get_value().shape, dtype = theano.config.floatX))) for p in self.network.gparams])
    self.network_complexity = 0
    n_in = self.network.n_in
    z = 1
    for h in network.hidden_info:
      self.network_complexity += n_in * h[1]
      z += 1
      if h[0] == 'recurrent':
        self.network_complexity += h[1] * h[1]
        z += 1
      elif h[0] == 'lstm':
        self.network_complexity += 4 * h[1] * h[1]
        z += 1
      n_in = h[1]
    self.network_complexity += n_in * self.network.n_out
    self.network_complexity /= float(z)
    self.network_weight = network.num_params() + network.n_out
    self.rate = T.scalar('r')
  
  def set_batch_size(self, data, batch_size, batch_step, max_seqs = -1):
    batches = []
    batch = Batch([0,0])
    if max_seqs == -1: max_seqs = data.num_seqs
    if batch_step == -1: batch_step = batch_size 
    s = 0
    while s < data.num_seqs:
      length = data.seq_lengths[data.seq_index[s]]
      if self.network.recurrent:
        if length > batch_size:
          print >> log.v4, "warning: sequence length (" + str(length) + ") larger than limit (" + str(batch_size) + ")"
        dt, ds = batch.try_sequence(length)
        if ds == 1:
          batch.add_sequence(length)
        else:
          if dt * ds > batch_size or ds > max_seqs:
            batches.append(batch)
            s = s - ds + min(batch_step, ds)
            batch = Batch([s, 0])
            length = data.seq_lengths[data.seq_index[s]]
          batch.add_sequence(length)
      else:
        while length > 0:
          nframes = min(length, batch_size - batch.shape[0])
          if nframes == 0:
            batches.append(batch)
            batch = Batch([s, data.seq_lengths[data.seq_index[s]] - length])
            nframes = min(length, batch_size)
          batch.add_frames(nframes)
          length -= min(nframes, batch_step)
        if s != data.num_seqs - 1: batch.nseqs += 1
      s += 1
    batches.append(batch)
    return batches
        
  def allocate_devices(self, data, batches, start_batch):
    devices = []
    for batch, device in zip(batches[start_batch:], self.devices):
      device.data = numpy.zeros(batch.shape + [data.num_inputs * data.window], dtype=theano.config.floatX)
      device.targets = numpy.zeros(batch.shape, dtype = theano.config.floatX)
      device.ctc_targets = numpy.zeros((batch.shape[1], data.max_ctc_length), dtype = theano.config.floatX)
      device.index = numpy.zeros(batch.shape, dtype = 'int8')
      if self.network.recurrent:
        data.load_seqs(batch.start[0], batch.start[0] + batch.shape[1])
        idi = data.alloc_interval_index(batch.start[0])
        for s in xrange(batch.start[0], batch.start[0] + batch.shape[1]):
          ids = data.seq_index[s]
          l = data.seq_lengths[ids]
          o = data.seq_start[s] + batch.start[1] - data.seq_start[data.alloc_intervals[idi][0]]
          q = batch.start[0] - s
          device.data[:l, q] = data.alloc_intervals[idi][2][o:o + l]
          device.targets[:l, q] = data.targets[data.seq_start[s] + batch.start[1]:data.seq_start[s] + batch.start[1] + l]
          device.ctc_targets[q] = data.ctc_targets[ids]
          device.index[:l, q] = numpy.ones((l,), dtype = 'int8')
      else:
        data.load_seqs(batch.start[0], batch.start[0] + batch.nseqs)
        idi = data.alloc_interval_index(batch.start[0])
        o = data.seq_start[batch.start[0]] + batch.start[1] - data.seq_start[data.alloc_intervals[idi][0]]
        l = batch.shape[0]
        device.data[:l, 0] = data.alloc_intervals[idi][2][o:o + l]
        device.targets[:l, 0] = data.targets[data.seq_start[batch.start[0]] + batch.start[1]:data.seq_start[batch.start[0]] + batch.start[1] + l] #data.targets[o:o + l]
        device.index[:l, 0] = numpy.ones((l,), dtype = 'int8')
      devices.append(device)
    return devices
    
  def process(self, task, updater, data, batches, start_batch = 0):
    score, error, nseqs, pseqs = 0, 0, 0, 0
    num_batches = start_batch
    num_data_batches = len(batches)
    while num_batches < num_data_batches:
      #alloc_batches = self.allocate_batches(data, batches, num_batches)
      alloc_devices = self.allocate_devices(data, batches, num_batches)
      for batch, device in enumerate(alloc_devices):
        if self.network.recurrent:
          print >> log.v5, "running", device.data.shape[1], "sequences (" + str(device.data.shape[0] * device.data.shape[1]) + " nts)", 
        else:
          print >> log.v5, "running", device.data.shape[0], "frames",
        print >> log.v5, "of batch", batch + num_batches, "/", num_data_batches, "on device", device.name
        device.run(task, self.network)
      for batch, device in enumerate(alloc_devices):
        try: result = device.result()
        except RuntimeError: result = None
        if result == None:
          print >> log.v2, "device", device.name, "crashed on batch", batch + num_batches
          return -1, batch + num_batches
        score += result[0]
        if task == 'train':
          assert len(result) == len(self.network.gparams) + 1
          for p,q in zip(self.network.gparams, result[1:]):
            self.gparams[p].set_value(self.gparams[p].get_value() + numpy.array(q))
        elif task == 'eval':
          error += result[1]
      if task == 'train':
        updater(self.learning_rate)
        for p in self.network.gparams:
          self.gparams[p].set_value(numpy.zeros(p.get_value().shape, dtype = theano.config.floatX))
      num_batches += len(alloc_devices)
    return score / data.num_timesteps, float(error) / data.num_timesteps
  
  def train_config(self, config, train, dev = None, eval = None, start_epoch = 0):
    batch_size, batch_step = config.int_pair('batch_size', (1,1))
    model = config.value('model', None)
    interval = config.int('save_interval', 1)
    learning_rate = config.float('learning_rate', 0.01)
    momentum = config.float("momentum", 0.9)
    num_epochs = config.int('num_epochs', 5)
    max_seqs = config.int('max_seqs', -1)
    start_batch = config.int('start_batch', 0)
    self.train(num_epochs, learning_rate, batch_size, batch_step, train, dev, eval, momentum, model, interval, start_epoch, start_batch, max_seqs)

  def train(self, num_epochs, learning_rate, batch_size, batch_step, train, dev = None, eval = None, momentum = 0.9, model = None, interval = 1, start_epoch = 0, start_batch = 0, max_seqs = -1):
    self.data = {}    
    if dev: self.data["dev"] = dev
    if eval: self.data["eval"] = eval
    for name in self.data.keys():
      self.data[name] = (self.data[name], self.set_batch_size(self.data[name], batch_size, batch_size)) # max(max(self.data[name].seq_lengths), batch_size)))
    deltas = dict([(p, theano.shared(value = numpy.zeros(p.get_value().shape, dtype = theano.config.floatX))) for p in self.network.gparams])
    self.learning_rate = learning_rate
    updates = []
    for param in self.network.gparams:
        #upd = momentum * deltas[param] - learning_rate * self.gparams[param]
        upd = momentum * deltas[param] - self.rate * self.gparams[param]
        updates.append((deltas[param], upd))
        updates.append((param, param + upd))
    updater = theano.function(inputs = [self.rate],
                              updates = updates)
    train_batches = self.set_batch_size(train, batch_size, batch_step, max_seqs)
    for epoch in xrange(start_epoch + 1, start_epoch + num_epochs + 1):
      epoch_start_time = time.time()
      score, error = self.process('train', updater, train, train_batches, start_batch)
      start_batch = 0
      if score == -1:
        self.network.save(model + ".crash_" + str(error), epoch - 1)
        sys.exit(1)
      if model and (epoch % interval == 0):
        self.network.save(model + ".%03d" % epoch, epoch)
      if log.verbose[1]:
        print >> log.v1, "epoch", epoch, "elapsed:", str(time.time() - epoch_start_time), "score:", score,
        for name in self.data.keys():
          data, num_batches = self.data[name]
          score, error = self.process('eval', updater, data, num_batches)
          print >> log.v1, name + ":", "score", score, "error", error,
        print >> log.v1, ''
    if model:
      self.network.save(model + ".%03d" % (start_epoch + num_epochs), start_epoch + num_epochs)
      
  def forward(self, device, data, cache_file):
    cache = SprintCache.FileArchive(cache_file)
    batches = self.set_batch_size(data, data.num_timesteps, data.num_timesteps, 1)
    num_data_batches = len(batches)
    num_batches = 0
    toffset = 0
    while num_batches < num_data_batches:
      alloc_devices = self.allocate_devices(data, batches, num_batches)
      for batch, device in enumerate(alloc_devices):
        print >> log.v4, "processing batch", num_batches + batch, "of", num_data_batches
        device.run('extract', self.network)
        features = numpy.concatenate(device.result(), axis = 1) #reduce(operator.add, device.result())
        print >> log.v5, "extracting", len(features[0]), "features over", len(features), "time steps for sequence", data.tags[num_batches + batch]
        times = zip(range(0, len(features)), range(1, len(features) + 1)) if not data.timestamps else data.timestamps[toffset : toffset + len(features)]
        #times = zip(range(0, len(features)), range(1, len(features) + 1))
        toffset += len(features)
        cache.addFeatureCache(data.tags[num_batches + batch], numpy.asarray(features), numpy.asarray(times))
      num_batches += len(alloc_devices)
    cache.finalize()

  def analyze(self, device, data, statistics):
    num_labels = len(data.labels)
    if "mle" in statistics:
      mle_labels = list(OrderedDict.fromkeys([ label.split('_')[0] for label in data.labels ]))
      mle_map = [mle_labels.index(label.split('_')[0]) for label in data.labels]
      num_mle_labels = len(mle_labels)
      confusion_matrix = numpy.zeros((num_mle_labels, num_mle_labels), dtype = 'int32')
    else:
      confusion_matrix = numpy.zeros((num_labels, num_labels), dtype = 'int32')
    batches = self.set_batch_size(data, data.num_timesteps, 1)
    num_data_batches = len(batches)
    num_batches = 0
    while num_batches < num_data_batches:
      alloc_batches = self.allocate_batches(data, batches, num_batches)
      for batch, device in enumerate(alloc_devices):
        device.run('analyze', batch, self.network)
        result = device.result()
        max_c = numpy.argmax(result[0], axis=1)
        if self.network.recurrent:
          real_c = device.targets[:,device.batch_start[batch] : device.batch_start[batch + 1]].flatten()
        else:
          real_c = device.targets[device.batch_start[batch] : device.batch_start[batch + 1]].flatten()
        for i in xrange(max_c.shape[0]):
          #print real_c[i], max_c[i], len(confusion_matrix[0])
          if "mle" in statistics:
            confusion_matrix[mle_map[int(real_c[i])], mle_map[int(max_c[i])]] += 1
          else:
            confusion_matrix[real_c[i], max_c[i]] += 1
      num_batches += alloc_batches
    
    if "confusion_matrix" in statistics:
      print >> log.v1, "confusion matrix:"
      for i in xrange(confusion_matrix.shape[0]):
        for j in xrange(confusion_matrix.shape[1]):
          print >> log.v1, str(confusion_matrix[i,j]).rjust(3),
        print >> log.v1, ''
    
    if "confusion_list" in statistics:
      n = 30
      print >> log.v1, "confusion top" + str(n) + ":"
      top = []
      for i in xrange(confusion_matrix.shape[0]):
        for j in xrange(confusion_matrix.shape[1]):
          if i != j:
            if "mle" in statistics:
              top.append([mle_labels[i] + " -> " + mle_labels[j], confusion_matrix[i,j]]) 
            else:
              top.append([data.labels[i] + " -> " + data.labels[j], confusion_matrix[i,j]])
      top.sort(key = lambda x:x[1], reverse = True)
      for i in xrange(n):
        print >> log.v1, top[i][0], top[i][1], str(100 * top[i][1] / float(data.num_timesteps)) + "%"
        
    if "error" in statistics:
      print >> log.v1, "error:", 1.0 - sum([confusion_matrix[i,i] for i in xrange(confusion_matrix.shape[0])]) / float(data.num_timesteps) 
    
