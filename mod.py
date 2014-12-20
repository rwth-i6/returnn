#! /usr/bin/python2.7

import os
import sys
import h5py
import json

from optparse import OptionParser
from Network import LayerNetwork

if __name__ == '__main__':
  # initialize config file
  parser = OptionParser()
  parser.add_option("-j", "--json", dest = "json",
                    help = "[STRING] Path to combination file.")
  parser.add_option("-o", "--output", dest = "output",
                    help = "[STRING] Output model.")
  parser.add_option("-d", "--dump", dest = "dump",
                    help = "[STRING] Dump json file of output model.")
  
  (options, args) = parser.parse_args()

  assert options.json, "no combination file specified"
  assert options.output, "no output file specified"

  models = []
  for m in args:
    models.append(h5py.File(m, "r"))

  com = json.loads(open(options.json, 'r').read())
  network_layers = {}
  network_json = {}
  for name in com:
    #print "layer: ", name,
    sources = []
    layer = com[name]
    assert "units" in layer, "no unit source specified for layer %s"  % layer
    for source in layer["units"]:
      n, h = source
      sources.append(LayerNetwork.from_model(models[n], 'unity').hidden[h])
    meta = 0 if not "meta" in layer else layer ['meta']
    axis = 1 if not "axis" in layer else layer ['axis']
    if "modify_attrs" in layer:
      for p in layer['modify_attrs'].keys():
        sources[meta].attrs[p] = layer['modify_attrs'][p]
    if "overwrite_attrs" in layer:
      for p in layer['overwrite_attrs'].keys():
        s, l = layer['overwrite_attrs'][p]
        sources[meta].attrs[p] = sources[s].attrs[p]
    if "overwrite_names" in layer:
      for p in layer['overwrite_names']:
        s, m = p
        sources[s].params.update(m)
    for s in sources[1:]:
      sources[meta].concat_units(s, axis)  
    network_layers[name] = sources[meta]
    network_json[name] = sources[meta].to_json()
    #print ""

json_content = json.dumps(network_json)
if options.dump:
  out = open(options.dump, "w")
  print >> out, json_content
  out.close()

network = LayerNetwork.from_json(json.dumps(network_json))
for layer in network_layers:
  if layer.name == network.output.name:
    network.output.set_params(layer.get_params())
  else:
    network.hidden[layer.name].set_params(layer.get_params())

model = h5py.File(filename, "w")
network.save(model, 0)
model.close()