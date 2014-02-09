#! /usr/bin/python2.7

import os
import sys

from Log import log
from Config import Config
from optparse import OptionParser
from Network import LayerNetwork

if __name__ == '__main__':  
  if len(sys.argv) != 2:
    print "usage:", sys.argv[0], "[params] network_1 network_2 ..."
    sys.exit(1)
  
  # initialize config file
  assert os.path.isfile(sys.argv[1]), "config file not found"  
  parser = OptionParser()
  parser.add_option("-a", "--add", dest = "add",
                    help = "[INTEGER/LIST] Add specified number of neurons.")
  parser.add_option("-b", "--bias", action = 'store_true', dest = "bias",
                    help = "[BOOL] Insert / Remove bias.")
  parser.add_option("-c", "--config", dest = "config",
                    help = "[STRING] Config file.")
  parser.add_option("-d", "--delete", dest = "delete",
                    help = "[INTEGER/LIST] Delete specified number of neurons.")
  parser.add_option("-l", "--layer", dest = "layer",
                    help = "[INTEGER/LIST] Select layers.")
  parser.add_option("-n", "--name", dest = "name",
                    help = "[STRING] Change name of specified layers")
  
  (options, args) = parser.parse_args()
  config = Config()
  if options.config: config.load_file(options.config)
  #for arg in args:
  assert options.has_key('layer'), "no layers specified"
  
  
  # initialize log file
  logs = config.list('log', [])
  log_verbosity = config.int_list('log_verbosity', [])
  log_format = config.list('log_format', [])
  log.initialize(logs = ['stdout', verbosity = [], formatter = [])