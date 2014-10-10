#! /usr/bin/python2.7

import SprintCache
import numpy
import theano
import h5py
import time
import sys
import theano.tensor as T
from Log import log
from Util import hdf5_strings
from collections import OrderedDict
import threading

class Batch:
  def __init__(self, start = [0, 0]):
    self.shape = [0, 0]
    self.start = start
    self.nseqs = 1
  def try_sequence(self, length): return [max(self.shape[0], length), self.shape[1] + 1]
  def add_sequence(self, length): self.shape = self.try_sequence(length)
  def add_frames(self, length): self.shape = [self.shape[0] + length, max(self.shape[1], 1)]
  def size(self): return self.shape[0] * self.shape[1]
  
class Process(threading.Thread):
    def __init__(self, task, network, devices, data, batches, start_batch = 0):
      threading.Thread.__init__(self)
      self.start_batch = start_batch 
      self.devices = devices
      self.network = network
      self.batches = batches
      self.task = task
      self.data = data
      self.daemon = True
      self.start()
      
    def allocate_devices(self, start_batch):
      devices = []
      num_batches = start_batch
      for device in self.devices:
        shape = [0, 0]
        device_batches = min(num_batches + device.num_batches, len(self.batches))
        for batch in self.batches[num_batches : device_batches]:
          shape = [max(shape[0], batch.shape[0]), shape[1] + batch.shape[1]]
        if shape[1] == 0: break
        device.data = numpy.zeros(shape + [self.data.num_inputs * self.data.window], dtype=theano.config.floatX)
        device.targets = numpy.zeros(shape, dtype = theano.config.floatX)
        device.ctc_targets = numpy.zeros((shape[1], self.data.max_ctc_length), dtype = theano.config.floatX)
        device.index = numpy.zeros(shape, dtype = 'int8')
        offset = 0
        for batch in self.batches[num_batches : device_batches]:
          if self.network.recurrent:
            self.data.load_seqs(batch.start[0], batch.start[0] + batch.shape[1])
            idi = self.data.alloc_interval_index(batch.start[0])
            for s in xrange(batch.start[0], batch.start[0] + batch.shape[1]):
              ids = self.data.seq_index[s]
              l = self.data.seq_lengths[ids]
              o = self.data.seq_start[s] + batch.start[1] - self.data.seq_start[self.data.alloc_intervals[idi][0]]
              q = s - batch.start[0] + offset
              device.data[:l, q] = self.data.alloc_intervals[idi][2][o:o + l]
              device.targets[:l, q] = self.data.targets[self.data.seq_start[s] + batch.start[1]:self.data.seq_start[s] + batch.start[1] + l]
              if self.data.ctc_targets is not None:
                device.ctc_targets[q] = self.data.ctc_targets[ids]
              device.index[:l, q] = numpy.ones((l,), dtype = 'int8')
            offset += batch.shape[1]
          else:
            self.data.load_seqs(batch.start[0], batch.start[0] + batch.nseqs)
            idi = self.data.alloc_interval_index(batch.start[0])
            o = self.data.seq_start[batch.start[0]] + batch.start[1] - self.data.seq_start[self.data.alloc_intervals[idi][0]]
            l = batch.shape[0]
            device.data[offset:offset + l, 0] = self.data.alloc_intervals[idi][2][o:o + l]
            device.targets[offset:offset + l, 0] = self.data.targets[self.data.seq_start[batch.start[0]] + batch.start[1]:self.data.seq_start[batch.start[0]] + batch.start[1] + l] #data.targets[o:o + l]
            device.index[offset:offset + l, 0] = numpy.ones((l,), dtype = 'int8')
            offset += l
        num_batches = device_batches
        devices.append(device)
      return devices, num_batches - start_batch
      
    def evaluate(self, batch, result):
      self.result = result
    def initialize(self): pass
    def finalize(self): pass
    def run(self):
      start_time = time.time()
      num_data_batches = len(self.batches)
      num_batches = self.start_batch
      self.initialize()
      print >> log.v5, "starting process", self.task
      while num_batches < num_data_batches:
        alloc_devices, num_alloc_batches = self.allocate_devices(num_batches)
        batch = num_batches
        for device in alloc_devices:
          if self.network.recurrent: print >> log.v5, "running", device.data.shape[1], "sequences (" + str(device.data.shape[0] * device.data.shape[1]) + " nts)", 
          else: print >> log.v5, "running", device.data.shape[0], "frames",
          if device.num_batches == 1: print >> log.v5, "of batch", batch,
          else: print >> log.v5, "of batches", str(batch) + "-" + str(batch + device.num_batches - 1),
          print >> log.v5, "/", num_data_batches, "on device", device.name
          device.run(self.task, self.network)
          batch += device.num_batches
        batch = num_batches
        device_results = []
        for device in alloc_devices:
          try: result = device.result()
          except RuntimeError: result = None
          if result == None:
            print >> log.v2, "device", device.name, "crashed on batch", batch
            self.last_batch = batch
            self.score = -1
            return -1
          device_results.append(result)
        self.evaluate(num_batches, device_results)
        num_batches += num_alloc_batches
      self.finalize()
      self.elapsed = (time.time() - start_time)
        
class TrainProcess(Process):
  def __init__(self, network, devices, data, batches, learning_rate, gparams, updater, start_batch = 0):
    super(TrainProcess, self).__init__('train', network, devices, data, batches, start_batch)
    self.updater = updater
    self.learning_rate = learning_rate
    self.gparams = gparams
  def initialize(self):
    self.score = 0
  def evaluate(self, batch, result):
    if result == None:
      self.score = None
    else:
      for res in result:
        self.score += res[0]
        for p,q in zip(self.network.gparams, res[1:]):
          self.gparams[p].set_value(self.gparams[p].get_value() + numpy.array(q))
      self.updater(self.learning_rate)
      for p in self.network.gparams:
        self.gparams[p].set_value(numpy.zeros(p.get_value().shape, dtype = theano.config.floatX))
  def finalize(self):
    self.score /= float(self.data.num_timesteps)

class EvalProcess(Process):
    def __init__(self, network, devices, data, batches, start_batch = 0):
      super(EvalProcess, self).__init__('eval', network, devices, data, batches, start_batch)
    def initialize(self):
      self.score = 0
      self.error = 0
    def evaluate(self, batch, result):
      self.score += sum([res[0] for res in result]) 
      self.error += sum([res[1] for res in result])
    def finalize(self):
      self.score /= float(self.data.num_timesteps)
      self.error /= float(self.data.num_timesteps)
      
class SprintCacheForwardProcess(Process):
    def __init__(self, network, devices, data, batches, cache, merge = {}, start_batch = 0):
      super(SprintCacheForwardProcess, self).__init__('extract', network, devices, data, batches, start_batch)
      self.cache = cache
      self.merge = merge
    def initialize(self):
      self.toffset = 0
    def evaluate(self, batch, result):
      features = numpy.concatenate(result, axis = 1) #reduce(operator.add, device.result())
      if self.merge.keys():
        merged = numpy.zeros((len(features), len(self.merge.keys())), dtype = theano.config.floatX)
        for i in xrange(len(features)): 
          for j, label in enumerate(self.merge.keys()):
            for k in self.merge[label]:
              merged[i, j] += numpy.exp(features[i, k])
          z = max(numpy.sum(merged[i]), 0.000001)
          merged[i] = numpy.log(merged[i] / z)
        features = merged
      print >> log.v5, "extracting", len(features[0]), "features over", len(features), "time steps for sequence", self.data.tags[self.data.seq_index[batch]]
      times = zip(range(0, len(features)), range(1, len(features) + 1)) if not self.data.timestamps else self.data.timestamps[self.toffset : self.toffset + len(features)]
      #times = zip(range(0, len(features)), range(1, len(features) + 1))
      self.toffset += len(features)
      self.cache.addFeatureCache(self.data.tags[self.data.seq_index[batch]], numpy.asarray(features), numpy.asarray(times))

class HDFForwardProcess(Process):
    def __init__(self, network, devices, data, batches, cache, merge = {}, start_batch = 0):
      super(HDFForwardProcess, self).__init__('extract', network, devices, data, batches, start_batch)
      self.tags = []
      self.merge = merge
      self.cache = cache
      cache.attrs['numSeqs'] = data.num_seqs
      cache.attrs['numTimesteps'] = data.num_timesteps
      cache.attrs['inputPattSize'] = data.num_inputs
      cache.attrs['numDims'] = 1
      cache.attrs['numLabels'] = data.num_outputs
      hdf5_strings(cache, 'labels', data.labels)
      self.targets = cache.create_dataset("targetClasses", (data.num_timesteps,), dtype='i')
      self.seq_lengths = cache.create_dataset("seqLengths", (data.num_seqs,), dtype='i')
      self.seq_dims = cache.create_dataset("seqDims", (data.num_seqs, 1), dtype='i')
      if data.timestamps:
        times = cache.create_dataset("times", data.timestamps.shape, dtype='i')
        times[...] = data.timestamps

    def initialize(self):
      self.toffset = 0
    def finalize(self):
      hdf5_strings(self.cache, 'seqTags', self.tags)

    def evaluate(self, batch, result):
      features = numpy.concatenate(result, axis = 1)
      if not "inputs" in self.cache:
        self.inputs = self.cache.create_dataset("inputs", (self.cache.attrs['numTimesteps'], features.shape[2]), dtype='f')
      if self.merge.keys():
        merged = numpy.zeros((len(features), len(self.merge.keys())), dtype = theano.config.floatX)
        for i in xrange(len(features)): 
          for j, label in enumerate(self.merge.keys()):
            for k in self.merge[label]:
              merged[i, j] += numpy.exp(features[i, k])
          z = max(numpy.sum(merged[i]), 0.000001)
          merged[i] = numpy.log(merged[i] / z)
        features = merged
      print >> log.v5, "extracting", features.shape[2], "features over", features.shape[1], "time steps for sequence", self.data.tags[self.data.seq_index[batch]]
      self.seq_dims[batch] = [features.shape[1]]
      self.seq_lengths[batch] = features.shape[1]
      self.inputs[self.toffset:self.toffset + features.shape[1]] = numpy.asarray(features)
      self.toffset += features.shape[1]
      self.tags.append(self.data.tags[self.data.seq_index[batch]])
      
class Engine:      
  def __init__(self, devices, network):
    self.network = network
    self.devices = devices
    self.gparams = dict([(p, theano.shared(value = numpy.zeros(p.get_value().shape, dtype = theano.config.floatX))) for p in self.network.gparams])
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
          if nframes == 0 or batch.nseqs > max_seqs:
            batches.append(batch)
            batch = Batch([s, data.seq_lengths[data.seq_index[s]] - length])
            nframes = min(length, batch_size)
          batch.add_frames(nframes)
          length -= min(nframes, batch_step)
        if s != data.num_seqs - 1: batch.nseqs += 1
      s += 1
    batches.append(batch)
    return batches
  
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
    if self.network.loss == 'priori':
      prior = train.calculate_priori()
      self.network.output.priori.set_value(prior)
      self.network.output.initialize()
    for param in self.network.gparams:
        #upd = momentum * deltas[param] - learning_rate * self.gparams[param]
        upd = momentum * deltas[param] - self.rate * self.gparams[param]
        updates.append((deltas[param], upd))
        updates.append((param, param + upd))
    updater = theano.function(inputs = [self.rate], updates = updates)
    train_batches = self.set_batch_size(train, batch_size, batch_step, max_seqs)
    tester = None
    #training_devices = self.devices[:-1] if len(self.devices) > 1 else self.devices
    #testing_device = self.devices[-1]
    training_devices = self.devices
    testing_device = self.devices[0]
    for epoch in xrange(start_epoch + 1, start_epoch + num_epochs + 1):
      trainer = TrainProcess(self.network, training_devices, train, train_batches, learning_rate, self.gparams, updater, start_batch)
      if tester:
        if False and len(self.devices) > 1:
          if tester.isAlive(): 
            #print >> log.v3, "warning: waiting for test score of previous epoch"
            tester.join(9044006400)
        print >> log.v1, name + ":", "score", tester.score, "error", tester.error
      trainer.join(9044006400)
      start_batch = 0
      if trainer.score == None:
        self.save_model(model + ".crash_" + str(trainer.error), epoch - 1)
        sys.exit(1)
      if model and (epoch % interval == 0):
        self.save_model(model + ".%03d" % epoch, epoch)
      if log.verbose[1]:
        for name in self.data.keys():
          data, num_batches = self.data[name]
          tester = EvalProcess(self.network, [testing_device], data, num_batches)
          if True or len(self.devices) == 1:
            tester.join(9044006400)
            trainer.elapsed += tester.elapsed
        print >> log.v1, "epoch", epoch, "elapsed:", trainer.elapsed, "score:", trainer.score,
    if model:
      self.save_model(model + ".%03d" % (start_epoch + num_epochs), start_epoch + num_epochs)
    if tester:
      if len(self.devices) > 1: tester.join(9044006400)
      print >> log.v1, name + ":", "score", tester.score, "error", tester.error

  def save_model(self, filename, epoch):
    model = h5py.File(filename, "w")
    self.network.save(model, epoch)
    model.close()
      
  def forward_to_sprint(self, device, data, cache_file, combine_labels = ''):
    cache = SprintCache.FileArchive(cache_file)
    batches = self.set_batch_size(data, data.num_timesteps, data.num_timesteps, 1)
    merge = {}
    if combine_labels != '':
      for index, label in enumerate(data.labels):
        merged = combine_labels.join(label.split(combine_labels)[:-1])
        if merged == '': merged = label
        if not merged in merge.keys():
          merge[merged] = []
        merge[merged].append(index)
      import codecs
      label_file = codecs.open(cache_file + ".labels", encoding = 'utf-8', mode = 'w')
      for key in merge.keys():
        label_file.write(key + "\n")
      label_file.close()
    forwarder = SprintCacheForwardProcess(self.network, self.devices, data, batches, cache, merge)
    forwarder.join(9044006400)
    cache.finalize()

  def forward(self, device, data, output_file, combine_labels = ''):
    cache =  h5py.File(output_file, "w")
    batches = self.set_batch_size(data, data.num_timesteps, data.num_timesteps, 1)
    merge = {}
    if combine_labels != '':
      for index, label in enumerate(data.labels):
        merged = combine_labels.join(label.split(combine_labels)[:-1])
        if merged == '': merged = label
        if not merged in merge.keys():
          merge[merged] = []
        merge[merged].append(index)
      import codecs
      label_file = codecs.open(cache_file + ".labels", encoding = 'utf-8', mode = 'w')
      for key in merge.keys():
        label_file.write(key + "\n")
      label_file.close()
    forwarder = HDFForwardProcess(self.network, self.devices, data, batches, cache, merge)
    forwarder.join(9044006400)
    cache.close()
  
  def classify(self, device, data, label_file):
    batches = self.set_batch_size(data, data.num_timesteps, data.num_timesteps, 1)
    num_data_batches = len(batches)
    num_batches = 0
    out = open(label_file, 'w')
    while num_batches < num_data_batches:
      alloc_devices = self.allocate_devices(data, batches, num_batches)
      for batch, device in enumerate(alloc_devices):
        device.run('classify', self.network)
        labels = numpy.concatenate(device.result(), axis = 1)
        print >> log.v5, "labeling", len(labels), "time steps for sequence", data.tags[num_batches + batch]
        print >> out, data.tags[num_batches + batch],
        for label in labels: print >> out, data.labels[label],
        print >> out, ''
      num_batches += len(alloc_devices)
    out.close()

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
      alloc_devices = self.allocate_devices(data, batches, num_batches)
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
      num_batches += len(alloc_devices)
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
