
from __future__ import print_function

import sys
import os

import _setup_test_env  # noqa
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false
import returnn.sprint.interface as SprintAPI
from tempfile import mkdtemp
from returnn.theano.engine import Engine
from returnn.config import Config
from returnn.theano.network import LayerNetwork
import returnn.theano.util as theano_util
import shutil
import numpy


theano_util.monkey_patches()


def install_sigint_handler():
  import signal

  def signal_handler(signal, frame):
    print("\nSIGINT at:")
    better_exchook.print_tb(tb=frame, file=sys.stdout)
    print("")

    # It's likely that SIGINT was caused by Util.interrupt_main().
    # We might have a stacktrace from there.
    if getattr(sys, "exited_frame", None) is not None:
      print("interrupt_main via:")
      better_exchook.print_tb(tb=sys.exited_frame, file=sys.stdout)
      print("")
      sys.exited_frame = None
      # Normal exception instead so that Nose will catch it.
      raise Exception("Got SIGINT!")
    else:
      print("\nno sys.exited_frame\n")
      # Normal SIGINT. Normal Nose exit.
      if old_action:
        old_action()
      else:
        raise KeyboardInterrupt

  old_action = signal.signal(signal.SIGINT, signal_handler)


install_sigint_handler()


def create_first_epoch(config_filename):
  config = Config()
  config.load_file(config_filename)
  engine = Engine([])
  engine.init_train_from_config(config=config, train_data=None)
  engine.epoch = 1
  engine.save_model(engine.get_epoch_model_filename(), epoch=engine.epoch)
  Engine._epoch_model = None


def test_forward():
  tmpdir = mkdtemp("returnn-test-sprint")
  olddir = os.getcwd()
  os.chdir(tmpdir)

  open("config", "w").write(
    """
    num_inputs 2
    num_outputs 3
    hidden_size 1
    hidden_type forward
    activation relu
    bidirectional false
    model model
    log_verbosity 5
    """)

  create_first_epoch("config")

  inputDim = 2
  outputDim = 3
  SprintAPI.init(inputDim=inputDim, outputDim=outputDim,
                 config="action:forward,configfile:config,epoch:1",
                 targetMode="forward-only")
  assert isinstance(SprintAPI.engine, Engine)
  assert isinstance(SprintAPI.engine.network, LayerNetwork)
  print("used data keys via net:", SprintAPI.engine.network.get_used_data_keys())
  print("used data keys via dev:", SprintAPI.engine.devices[0].used_data_keys)

  features = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
  seq_len = features.shape[0]
  posteriors = SprintAPI._forward("segment1", features.T).T
  assert_equal(posteriors.shape, (seq_len, outputDim))

  SprintAPI.exit()

  os.chdir(olddir)
  shutil.rmtree(tmpdir)

