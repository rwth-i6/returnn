#!/usr/bin/env python

# Sprint
layerCount = 0  # automatically determined
archiverExec = "./sprint-executables/archiver"

# Crnn
configFile = "config/crnn.config"
inputDim = 0  # via config num_inputs
outputDim = 0  # via config num_outputs


import os
import numpy as np
import argparse
import itertools

import better_exchook
better_exchook.install()


def parseSprintLayer(lines, float_type):
  """
  :type lines: list[str]
  :type float_type: str
  :rtype: (numpy.ndarray, numpy.ndarray)
  """
  nColumns = None
  rows = []
  for l in lines:
    if l.startswith("<"): continue
    elems = l.split()
    elems = map(float, elems)
    if len(rows) == 0: nColumns = len(elems)
    else: assert nColumns == len(elems)
    rows += [elems]
  dtype = "float64" if float_type == "f64" else "float32"
  matrix = np.array(rows, dtype=dtype)
  assert matrix.shape == (len(rows), nColumns)
  matrix = matrix.transpose()  # Crnn format
  bias = matrix[0,:]
  weights = matrix[1:,:]
  print "Sprint layer bias:", bias.shape, "weights:", weights.shape
  return bias, weights


def loadSprintNetwork(params_prefix_path, first_layer, float_type):
  """
  :type params_prefix_path: str
  :type first_layer: int
  :type float_type: str
  :rtype: list[(numpy.ndarray, numpy.ndarray)]
  """
  assert float_type in ["f32", "f64"]
  from subprocess import Popen, PIPE

  global layerCount
  layers = []
  for l in itertools.count():
    fn = "%s-%s-layer-%s.bin" % (params_prefix_path, float_type, l + first_layer)
    if not os.path.exists(fn):
      if l == 0:
        raise Exception("Did not found any Sprint NN layer. First: %r" % fn)
      break
    print("Loading Sprint NN layer %i" % (l + 1))

    p = Popen(
      [archiverExec, "--mode", "show", "--type", "bin-matrix", "--full-precision", "true", fn],
      stdout=PIPE, stderr=PIPE)
    out,err = p.communicate()
    assert p.returncode == 0, "Return %i, Error: %s" % (p.returncode, err)

    out = out.splitlines()
    layers += [parseSprintLayer(out, float_type)]

  layerCount = len(layers)
  assert layerCount > 0
  print("Sprint NN layer count: %i" % layerCount)
  return layers


def saveCrnnLayer(layer, bias, weights):
  assert len(layer.params) == 2  # weight matrix + bias vector

  biasParamName = "b_%s" % layer.name
  weightParamName = [key for key in layer.params.keys() if key.startswith("W_in_")][0]
  biasParams = layer.params[biasParamName]
  biasOld = biasParams.get_value()
  weightParams = layer.params[weightParamName]
  weightsOld = weightParams.get_value()

  assert biasOld.shape == bias.shape
  assert weightsOld.shape == weights.shape

  biasParams.set_value(bias)
  weightParams.set_value(weights)

  print("Saved Crnn layer %s" % layer.name)


def saveCrnnNetwork(epoch, layers):
  """
  :type epoch: int
  :type layers: list[(numpy.ndarray, numpy.ndarray)]
  """
  print("Loading Crnn")

  from Network import LayerNetwork
  from NetworkHiddenLayer import ForwardLayer
  from NetworkOutputLayer import OutputLayer
  from Pretrain import pretrainFromConfig
  from Engine import Engine

  pretrain = pretrainFromConfig(config)
  is_pretrain_epoch = pretrain and epoch <= pretrain.get_train_num_epochs()
  modelFilename = config.value("model", None)
  assert modelFilename, "need 'model' in config"
  filename = Engine.epoch_model_filename(modelFilename, epoch, is_pretrain_epoch)
  assert not os.path.exists(filename), "already exists"
  if is_pretrain_epoch:
    network = pretrain.get_network_for_epoch(epoch)
  else:
    network = LayerNetwork.from_config_topology(config)
  nHiddenLayers = len(network.hidden)

  # print network topology
  print "Crnn Network layer topology:"
  print "input dim:", network.n_in
  print "hidden layer count:", nHiddenLayers
  print "output dim:", network.n_out["classes"]
  print "net weights #:", network.num_params()
  print "net params:", network.train_params_vars
  print "net output:", network.output["output"]

  assert network.n_in == inputDim
  #assert network.n_out == outputDim
  assert nHiddenLayers + 1 == layerCount  # hidden + output layer
  assert len(layers) == layerCount
  for i, (layerName, hidden) in enumerate(sorted(network.hidden.items())):
    # Some checks whether this is a forward-layer.
    assert isinstance(hidden, ForwardLayer)

    saveCrnnLayer(hidden, *layers[i])

  assert isinstance(network.output["output"], OutputLayer)
  saveCrnnLayer(network.output["output"], *layers[len(layers) - 1])

  import h5py
  print("Save Crnn model under %s" % filename)
  model = h5py.File(filename, "w")
  network.save_hdf(model, epoch)
  model.close()


def main():
  global configFile, archiverExec, inputDim, outputDim
  parser = argparse.ArgumentParser()
  parser.add_argument('--sprintLoadParams', required=True,
                      help='Sprint NN params path prefix')
  parser.add_argument('--sprintFirstLayer', default=1, type=int,
                      help='Sprint NN params first layer (default 1)')
  parser.add_argument('--crnnSaveEpoch', type=int, required=True,
                      help='save this train epoch number in Crnn model')
  parser.add_argument('--crnnConfigFile', required=True,
                      help='CRNN config file')
  parser.add_argument('--sprintArchiverExec', default=archiverExec,
                      help='path to Sprint/RASR archiver executable')
  parser.add_argument('--floatType', default="f32",
                      help='float type (f32/f64)')
  args = parser.parse_args()

  configFile = args.crnnConfigFile
  assert os.path.exists(configFile), "CRNN config file not found"
  archiverExec = args.sprintArchiverExec
  assert os.path.exists(archiverExec), "Sprint archiver not found"
  assert args.crnnSaveEpoch >= 1

  from Config import Config
  global config
  config = Config()
  config.load_file(configFile)

  inputDim = config.int('num_inputs', None)
  outputDim = config.int('num_outputs', None)
  assert inputDim and outputDim

  layers = loadSprintNetwork(params_prefix_path=args.sprintLoadParams,
                             first_layer=args.sprintFirstLayer,
                             float_type=args.floatType)
  saveCrnnNetwork(epoch=args.crnnSaveEpoch, layers=layers)

  print("Done.")


if __name__ == "__main__":
  main()
