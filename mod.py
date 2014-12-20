#! /usr/bin/python2.7

import os
import sys
import h5py
import json

from optparse import OptionParser
from Network import LayerNetwork, OutputLayer

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
  n_in, n_out = 0, 0
  for name in com:
    #print "layer: ", name,
    sources = []
    layer = com[name]
    assert "units" in layer, "no unit source specified for layer %s"  % layer
    for source in layer["units"]:
      n, h = source
      net = LayerNetwork.from_model(models[n], 'unity')
      if h == net.output.name:
        sources.append(net.output)
      else:
        try:
          sources.append(net.hidden[h])
        except KeyError:
          print >> sys.stderr, "unable to find layer \"%s\" in model number %d" % (h, n)
          sys.exit(1)
    meta = 0 if not "meta" in layer else layer['meta']
    axis = 1 if not "axis" in layer else layer['axis']
    if "modify_attrs" in layer:
      sources[meta].attrs.update(layer['modify_attrs'])
    if "overwrite_attrs" in layer:
      sources[meta].attrs.update(layer['overwrite_attrs'])
    if "overwrite_names" in layer:
      for p in layer['overwrite_names']:
        s, m = p
        for q in m.keys():
          sources[s].params[m[q]] = sources[s].params.pop(q)
    for i, s in enumerate(sources):
      if i != meta:
        sources[meta].concat_units(s, axis)
    if sources[meta].sources[0].name == "data":
      if axis == 0:
        n_in = sum([ sum([t.attrs['n_out'] for t in s.sources]) for s in sources ])
      else:
        n_in = sum([t.attrs['n_out'] for t in sources[meta].sources])
    if isinstance(sources[meta], OutputLayer):
      n_out = sources[meta].attrs['n_out']
    network_layers[name] = sources[meta]
    network_json[name] = sources[meta].to_json()
    #print ""

json_content = json.dumps(network_json, indent=4)
if options.dump:
  out = open(options.dump, "w")
  print >> out, json_content
  out.close()

if n_in == 0: print >> sys.stderr, "missing network input"
if n_out == 0: print >> sys.stderr, "missing network output"
if n_in * n_out == 0: sys.exit(1)

network = LayerNetwork.from_json(json_content, n_in, n_out)
for k in network_layers.keys():
  layer = network_layers[k]
  if layer.name == network.output.name:
    network.output.set_params(layer.get_params())
  else:
    network.hidden[layer.name].set_params(layer.get_params())

model = h5py.File(options.output, "w")
network.save(model, 1)
model.close()