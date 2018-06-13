
# start test like this:  nosetests-2.7  tests/test_TFEngine.py
# or directly:  python3 test_TFEngine.py test_engine_rec_subnet_count


import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import sys
import os
sys.path += ["."]  # Python 3 hack
sys.path += [os.path.dirname(os.path.abspath(__file__)) + "/.."]
from TFEngine import *
import Util
import TFUtil
TFUtil.debugRegisterBetterRepr()
from Config import Config
from nose.tools import assert_equal, assert_is_instance
import unittest
import numpy
import numpy.testing
from pprint import pprint
import better_exchook
better_exchook.replace_traceback_format_tb()
from Log import log
log.initialize(verbosity=[5])

session = tf.InteractiveSession()


def test_DataProvider():
  """
  :param Dataset.Dataset dataset:
  :param int seq_idx:
  :param str|None output_layer_name: e.g. "output". if not set, will read from config "forward_output_layer"
  :return: numpy array, output in time major format (time,batch,dim)
  :rtype: numpy.ndarray
  """
  from GeneratingDataset import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)

  extern_data = ExternData()
  extern_data.init_from_dataset(dataset)

  # No Runner instance here but a very simplified version of Runner.run().
  # First we need a custom DataProvider with a custom BatchSetGenerator
  # which will yield only one single batch for the provided sequence idx.
  seq_idx = 0
  n_batch = 1
  batch = Batch()
  batch.add_frames(seq_idx=seq_idx, seq_start_frame=0, length=dataset.get_seq_length(seq_idx))
  batch_generator = iter([batch])
  batches = BatchSetGenerator(dataset, generator=batch_generator)
  from TFDataPipeline import FeedDictDataProvider
  data_provider = FeedDictDataProvider(
    tf_session=session, extern_data=extern_data,
    data_keys=["data", "classes"],
    dataset=dataset, batches=batches)

  feed_dict, meta = data_provider.get_feed_dict(single_threaded=True)
  print(feed_dict, meta)
  assert_is_instance(feed_dict, dict)
  assert extern_data.data["data"].placeholder in feed_dict
  assert extern_data.data["data"].size_placeholder[0] in feed_dict
  assert extern_data.data["classes"].placeholder in feed_dict
  assert extern_data.data["classes"].size_placeholder[0] in feed_dict
  data = feed_dict[extern_data.data["data"].placeholder]
  data_size = feed_dict[extern_data.data["data"].size_placeholder[0]]
  classes = feed_dict[extern_data.data["classes"].placeholder]
  classes_size = feed_dict[extern_data.data["classes"].size_placeholder[0]]
  assert_is_instance(data, numpy.ndarray)
  assert_is_instance(data_size, numpy.ndarray)
  assert_is_instance(classes, numpy.ndarray)
  assert_is_instance(classes_size, numpy.ndarray)
  assert_equal(data.shape, (n_batch, seq_len, n_data_dim))
  assert_equal(data_size.shape, (n_batch,))
  assert_equal(classes.shape, (n_batch, seq_len))
  assert_equal(classes_size.shape, (n_batch,))
  assert_equal(list(data_size), [seq_len])
  assert_equal(list(classes_size), [seq_len])
  numpy.testing.assert_almost_equal(list(data[0, 0]), [-0.5, -0.4])
  numpy.testing.assert_almost_equal(list(data[0, -1]), [0.3, 0.4])
  assert_equal(classes.tolist(), [[1, 2, 0, 1, 2]])


def test_engine_train():
  from GeneratingDataset import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "/tmp/model",
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {"output": {"class": "softmax", "loss": "ce"}},
    "start_epoch": 1,
    "num_epochs": 2
  })
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()

  engine.finalize()


def test_engine_train_accum_grad_multiple_step():
  from GeneratingDataset import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=7, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "/tmp/model",
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {"output": {"class": "softmax", "loss": "ce"}},
    "start_epoch": 1,
    "num_epochs": 2,
    "accum_grad_multiple_step": 3,
  })
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()
  engine.finalize()


def test_engine_train_accum_grad_multiple_step_sparse():
  from GeneratingDataset import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=7, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "/tmp/model",
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {"output": {"class": "softmax", "loss": "ce", "from": ["data:classes"]}},
    "start_epoch": 1,
    "num_epochs": 2,
    "accum_grad_multiple_step": 3,
  })
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()
  engine.finalize()


def test_engine_train_grad_noise_sparse():
  # Not sure how to test for it in a simple way...
  # You might see "Converting sparse IndexedSlices to a dense Tensor of unknown shape."
  # but that is normally related to sth else.

  def wrapped_slice_to_tensor(value, dtype=None, name=None, as_ref=False):
    print("wrapped_slice_to_tensor:", value)
    #assert "flatten_with_seq_len_mask" in value.name
    from tensorflow.python.ops import gradients_impl
    return gradients_impl._IndexedSlicesToTensor(value, dtype=dtype, name=name, as_ref=as_ref)

  from tensorflow.python.framework import ops
  ops.register_tensor_conversion_function(tf.IndexedSlices, wrapped_slice_to_tensor, priority=1)

  # Anyway, for now, just try to trigger relevant code,
  # ie. in add_scaled_noise_to_gradients(),
  # and don't really check whether it works.

  from GeneratingDataset import Task12AXDataset
  train_data = Task12AXDataset(num_seqs=5)
  cv_data = Task12AXDataset(num_seqs=2)
  n_data_dim = train_data.num_outputs["data"][0]
  n_classes_dim = train_data.num_outputs["classes"][0]

  config = Config()
  config.update({
    "model": "/tmp/model",
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {
      "hidden": {
        "class": "linear", "activation": "tanh", "n_out": 10,
        "from": ["data:classes"],  # sparse input
      },
      "hidden2": {
        "class": "linear", "activation": "tanh", "n_out": 10,
        "from": ["data:classes"],  # sparse input
      },
      "hidden3": {"class": "linear", "activation": "tanh", "n_out": 10, "from": ["hidden", "hidden2"]},
      "output": {
        "class": "linear", "activation": "softmax", "loss": "ce",
        "from": ["hidden", "hidden2", "hidden3"],
        "target": "classes"  # sparse output
        }},
    "start_epoch": 1,
    "num_epochs": 2,
    "learning_rate": 0.01,
    "nadam": True,
    "gradient_noise": 0.3,
    "batch_size": 100
  })
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()
  engine.finalize()


def test_engine_analyze():
  from GeneratingDataset import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "/tmp/model",
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {"output": {"class": "softmax", "loss": "ce"}},
    "sil_label_idx": 0,
  })
  engine = Engine(config=config)
  # Normally init_network_from_config but that requires an existing network model.
  # engine.init_network_from_config(config=config)
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=None, eval_data=None)

  engine.analyze(data=dataset, statistics=None)

  engine.finalize()


def test_engine_forward_single():
  from GeneratingDataset import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "/tmp/model",
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {"output": {"class": "softmax", "loss": "ce"}}
  })
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=None, eval_data=None)

  engine.forward_single(dataset=dataset, seq_idx=0)

  engine.finalize()


def test_engine_forward_to_hdf():
  from GeneratingDataset import DummyDataset
  import tempfile
  output_file = tempfile.mktemp(suffix=".hdf", prefix="nose-tf-forward")
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  num_seqs = 20
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim,
                         num_seqs=num_seqs, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "/tmp/model",
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {"output": {"class": "softmax", "loss": "ce"}},
    "output_file": output_file,
  })

  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=None, eval_data=None,)

  engine.forward_to_hdf(data=dataset, output_file=output_file, batch_size=5)

  engine.finalize()

  assert os.path.exists(output_file)
  import h5py
  with h5py.File(output_file, 'r') as f:
    assert f['inputs'].shape == (seq_len*num_seqs, n_classes_dim)
    assert f['seqLengths'].shape == (num_seqs,2)
    assert f['seqTags'].shape == (num_seqs,)
    assert f.attrs['inputPattSize'] == n_classes_dim
    assert f.attrs['numSeqs'] == num_seqs
    assert f.attrs['numTimesteps'] == seq_len * num_seqs

  from HDFDataset import HDFDataset
  ds = HDFDataset()
  ds.add_file(output_file)

  assert_equal(ds.num_inputs, n_classes_dim) # forwarded input is network output
  assert_equal(ds.get_num_timesteps(), seq_len*num_seqs)
  assert_equal(ds.num_seqs, num_seqs)

  os.remove(output_file)


def test_engine_rec_subnet_count():
  from GeneratingDataset import DummyDataset
  seq_len = 5
  # The dataset is actually not used.
  n_data_dim = 2
  n_classes_dim = 3
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "/tmp/model",
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {
      "output": {
        "class": "rec",
        "from": ["data"],  # actually not used, except that it defines the length
        "unit": {
        "output": {
          "class": "activation", "activation": "identity + 1",
          "from": ["prev:output"], "initial_output": 0,  # note: initial output is for t == -1
          "out_type": {"dim": 1, "dtype": "int32"}}
      }}}
  })
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=None, eval_data=None)

  out = engine.forward_single(dataset=dataset, seq_idx=0)
  assert_equal(out.shape, (seq_len, 1))
  assert_equal(out.dtype, numpy.int32)
  assert_equal(list(out[:,0]), list(range(1, seq_len + 1)))

  engine.finalize()


def check_engine_search(extra_rec_kwargs=None):
  """
  :param dict[str] extra_rec_kwargs:
  """
  from Util import dict_joined
  from GeneratingDataset import DummyDataset
  from TFNetworkRecLayer import RecLayer, _SubnetworkRecCell
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 7
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "/tmp/model",
    "batch_size": 5000,
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {
      "output": dict_joined({
        "class": "rec", "from": [], "max_seq_len": 10, "target": "classes",
        "unit": {
          "prob": {"class": "softmax", "from": ["prev:output"], "loss": "ce", "target": "classes"},
          "output": {"class": "choice", "beam_size": 4, "from": ["prob"], "target": "classes", "initial_output": 0},
          "end": {"class": "compare", "from": ["output"], "value": 0}
        }
      }, extra_rec_kwargs or {}),
      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance"}
    }
  })
  engine = Engine(config=config)
  # Normally init_network can be used. We only do init_train here to randomly initialize the network.
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=None, eval_data=None)
  print("network:")
  pprint(engine.network.layers)
  assert "output" in engine.network.layers
  assert "decision" in engine.network.layers

  rec_layer = engine.network.layers["output"]
  assert isinstance(rec_layer, RecLayer)
  assert isinstance(rec_layer.cell, _SubnetworkRecCell)
  if rec_layer._optimize_move_layers_out:
    assert_equal(set(rec_layer.cell.input_layers_moved_out), {"output"})
    assert_equal(set(rec_layer.cell.output_layers_moved_out), {"prob"})
    assert_equal(set(rec_layer.cell.layers_in_loop), set())
  else:
    assert_equal(set(rec_layer.cell.layers_in_loop), {"prob", "output", "end"})

  # Now reinit for search.
  assert not engine.use_search_flag
  engine.use_search_flag = True
  engine.use_dynamic_train_flag = False
  print("Reinit network with search flag.")
  engine.init_network_from_config(config=config)

  engine.search(dataset=dataset)
  print("error keys:")
  pprint(engine.network.error_by_layer)
  assert engine.network.total_objective is not None
  assert "decision" in engine.network.error_by_layer

  engine.finalize()


def test_engine_search_no_optim():
  check_engine_search({"optimize_move_layers_out": False})


def test_engine_search():
  check_engine_search()


def check_engine_search_attention(extra_rec_kwargs=None):
  """
  :param dict[str] extra_rec_kwargs:
  """
  from Util import dict_joined
  from GeneratingDataset import DummyDataset
  from TFNetworkRecLayer import RecLayer, _SubnetworkRecCell
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 7
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)
  print("Hello search!")

  config = Config()
  config.update({
    "model": "/tmp/model",
    "batch_size": 5000,
    "max_seqs": 2,
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {
      "encoder": {"class": "linear", "activation": "tanh", "n_out": 5},
      "output": dict_joined({
        "class": "rec",
        "from": [],
        "target": "classes", "max_seq_len": 10,
        "unit": {
          'output': {'class': 'choice', 'target': 'classes', 'beam_size': 4, 'from': ["output_prob"]},
          "end": {"class": "compare", "from": ["output"], "value": 0},
          'orth_embed': {'class': 'linear', 'activation': None, 'from': ['output'], "n_out": 7},
          "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:c", "prev:orth_embed"], "n_out": 7},
          "c_in": {"class": "linear", "activation": "tanh", "from": ["s", "prev:orth_embed"], "n_out": 5},
          "c": {"class": "dot_attention", "from": ["c_in"], "base": "base:encoder", "base_ctx": "base:encoder"},
          "output_prob": {"class": "softmax", "from": ["prev:s", "c"], "target": "classes", "loss": "ce"}
        },
      }, extra_rec_kwargs or {}),
      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance"}
    },
    "debug_print_layer_output_template": True,
    "debug_print_layer_output_shape": True
  })
  engine = Engine(config=config)
  print("Init network...")
  engine.start_epoch = 1
  engine.use_dynamic_train_flag = False
  engine.use_search_flag = True
  engine.init_network_from_config(config)
  print("network:")
  pprint(engine.network.layers)
  assert "output" in engine.network.layers
  assert "decision" in engine.network.layers

  rec_layer = engine.network.layers["output"]
  assert isinstance(rec_layer, RecLayer)
  assert isinstance(rec_layer.cell, _SubnetworkRecCell)
  if rec_layer._optimize_move_layers_out:
    assert_equal(set(rec_layer.cell.input_layers_moved_out), set())
    assert_equal(set(rec_layer.cell.output_layers_moved_out), set())
    assert_equal(set(rec_layer.cell.layers_in_loop), {"end", "output", "output_prob", "c", "c_in", "orth_embed", "s"})
  else:
    assert not rec_layer.cell.input_layers_moved_out
    assert not rec_layer.cell.output_layers_moved_out
    assert_equal(set(rec_layer.cell.layers_in_loop), {"end", "output", "output_prob", "c", "c_in", "orth_embed", "s"})

  print("Search...")
  engine.search(dataset=dataset)
  print("error keys:")
  pprint(engine.network.error_by_layer)
  assert engine.network.total_objective is not None
  assert "decision" in engine.network.error_by_layer

  engine.finalize()


def test_engine_search_attention_no_optim():
  check_engine_search_attention({"optimize_move_layers_out": False})


def test_engine_search_attention():
  check_engine_search_attention()


def check_engine_train_simple_attention(lstm_unit):
  net_dict = {
    "lstm0_fw": {"class": "rec", "unit": lstm_unit, "n_out": 20, "dropout": 0.0, "L2": 0.01, "direction": 1},
    "lstm0_bw": {"class": "rec", "unit": lstm_unit, "n_out": 20, "dropout": 0.0, "L2": 0.01, "direction": -1},

    "lstm1_fw": {"class": "rec", "unit": lstm_unit, "n_out": 20, "dropout": 0.0, "L2": 0.01, "direction": 1,
                 "from": ["lstm0_fw", "lstm0_bw"]},
    "lstm1_bw": {"class": "rec", "unit": lstm_unit, "n_out": 20, "dropout": 0.0, "L2": 0.01, "direction": -1,
                 "from": ["lstm0_fw", "lstm0_bw"]},

    "encoder": {"class": "linear", "activation": "tanh", "from": ["lstm1_fw", "lstm1_bw"], "n_out": 20},
    "enc_ctx": {"class": "linear", "activation": "tanh", "from": ["encoder"], "n_out": 20},

    "output": {"class": "rec", "from": [], "unit": {
      'orth_embed': {'class': 'linear', 'activation': None, 'from': ['data:classes'], "n_out": 10},
      "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:c", "prev:orth_embed"], "n_out": 20},
      "c_in": {"class": "linear", "activation": "tanh", "from": ["s", "prev:orth_embed"], "n_out": 20},
      "c": {"class": "dot_attention", "from": ["c_in"], "base": "base:encoder", "base_ctx": "base:enc_ctx",
            "n_out": 20},
      "output": {"class": "softmax", "from": ["prev:s", "c"], "target": "classes"}
    }, "target": "classes", "loss": "ce"}

  }

  from GeneratingDataset import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "/tmp/model",
    "batch_size": 100,
    "max_seqs": 2,
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": net_dict,
    "start_epoch": 1,
    "num_epochs": 2,
    "learning_rate": 0.01,
    "nadam": True,
    "gradient_noise": 0.3,
    "debug_add_check_numerics_ops": True,
    "debug_print_layer_output_template": True,
    "debug_print_layer_output_shape": True,
    "debug_add_check_numerics_on_output": True,
  })
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=dataset, eval_data=None)
  print("Extern data:")
  pprint(engine.network.extern_data.data)
  print("Used data keys:")
  pprint(engine.network.used_data_keys)
  engine.train()
  engine.finalize()


# @unittest.skip("crash on OSX? https://github.com/tensorflow/tensorflow/issues/14285")
def test_engine_train_simple_attention_lstmp():
  check_engine_train_simple_attention(lstm_unit="lstmp")


def test_engine_train_simple_attention_nativelstm2():
  check_engine_train_simple_attention(lstm_unit="nativelstm2")


def test_engine_train_simple_attention_basiclstm():
  check_engine_train_simple_attention(lstm_unit="basiclstm")


def test_rec_optim_all_out():
  from GeneratingDataset import DummyDataset
  from TFNetworkRecLayer import RecLayer, _SubnetworkRecCell
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 7
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "/tmp/model",
    "batch_size": 5000,
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {
      "output": {
        "class": "rec", "optimize_move_layers_out": True, "from": [], "max_seq_len": 10, "target": "classes",
        "unit": {
          "prob": {"class": "softmax", "from": ["prev:output"], "loss": "ce", "target": "classes"},
          "output": {"class": "choice", "beam_size": 4, "from": ["prob"], "target": "classes", "initial_output": 0},
          "end": {"class": "compare", "from": ["output"], "value": 0}
        }
      },
      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance"}
    }
  })
  engine = Engine(config=config)
  # Normally init_network can be used. We only do init_train here to randomly initialize the network.
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=None, eval_data=None)
  print("network:")
  pprint(engine.network.layers)
  assert "output" in engine.network.layers
  assert "decision" in engine.network.layers

  rec_layer = engine.network.layers["output"]
  assert isinstance(rec_layer, RecLayer)
  assert isinstance(rec_layer.cell, _SubnetworkRecCell)
  assert rec_layer._optimize_move_layers_out
  # Now it was initialized and optimized for training.
  assert_equal(set(rec_layer.cell.input_layers_moved_out), {"output"})
  assert_equal(set(rec_layer.cell.output_layers_moved_out), {"prob"})
  assert_equal(set(rec_layer.cell.layers_in_loop), set())

  # Now reinit for search.
  assert not engine.use_search_flag
  engine.use_search_flag = True
  engine.use_dynamic_train_flag = False
  print("Reinit network with search flag.")
  engine.init_network_from_config(config=config)

  rec_layer = engine.network.layers["output"]
  assert isinstance(rec_layer, RecLayer)
  assert isinstance(rec_layer.cell, _SubnetworkRecCell)
  assert rec_layer._optimize_move_layers_out
  # Now it was initialized and optimized for search.
  assert_equal(set(rec_layer.cell.input_layers_moved_out), set())
  assert_equal(set(rec_layer.cell.output_layers_moved_out), set())
  assert_equal(set(rec_layer.cell.layers_in_loop), {"prob", "output", "end"})

  engine.search(dataset=dataset)
  print("error keys:")
  pprint(engine.network.error_by_layer)
  assert engine.network.total_objective is not None
  assert "decision" in engine.network.error_by_layer

  engine.finalize()


def test_rec_subnet_train_t3b():
  beam_size = 2
  network = {
    "data_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6},
    "lstm0_fw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 6, "dropout": 0.1, "L2": 0.01, "direction": 1, "from": ["data_embed"] },
    "lstm0_bw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 6, "dropout": 0.1, "L2": 0.01, "direction": -1, "from": ["data_embed"] },
    "lstm1_fw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 6, "dropout": 0.1, "L2": 0.01, "direction": 1, "from": ["data_embed"] },
    "lstm1_bw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 6, "dropout": 0.1, "L2": 0.01, "direction": -1, "from": ["data_embed"] },
    "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
    "enc_ctx": {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"], "n_out": 5},
    "enc_emb": {"class": "copy", "from": ["enc_ctx"]},

    "output": {"class": "rec", "from": [], "unit": {
      'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["output_prob"]},
      "end": {"class": "compare", "from": ["output"], "value": 0},
      'orth_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6},
      "s_in": {"class": "linear", "activation": "tanh", "from": ["prev:c", "prev:orth_embed"], "n_out": 5},
      "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["s_in"], "n_out": 5},  # h_t
      "c_in": {"class": "copy", "from": ["s"]},
      "c": {"class": "dot_attention", "from": ["c_in"], "base": "base:enc_emb", "base_ctx": "base:enc_ctx"},
      "t1": {"class": "linear", "activation": "tanh", "from": ["c", "s"], "n_out": 6},
      "t2": {"class": "linear", "activation": "tanh", "from": ["t1"], "n_out": 6},
      "t3": {"class": "linear", "activation": "tanh", "from": ["t2"], "n_out": 6},
      "output_prob": {"class": "softmax", "from": ["t3"], "target": "classes", "loss": "ce"}
    }, "target": "classes", "max_seq_len": 75},

    "decision": {
      "class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes",
      "loss_opts": {
        "debug_print": True
      }
    }
  }

  from GeneratingDataset import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "/tmp/model",
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": network,
    "start_epoch": 1,
    "num_epochs": 2,
    "batch_size": 10,
    "nadam": True,
    "learning_rate": 0.01,
    "debug_add_check_numerics_ops": True
  })
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()

  engine.finalize()


def test_rec_subnet_train_t3d():
  beam_size = 2
  network = {
    "data_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6},
    "lstm0_fw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 5, "dropout": 0.1, "L2": 0.01, "direction": 1, "from": ["data_embed"] },
    "lstm0_bw" : { "class": "rec", "unit": "nativelstm2", "n_out" : 5, "dropout": 0.1, "L2": 0.01, "direction": -1, "from": ["data_embed"] },
    "encoder_state": {"class": "get_last_hidden_state", "from": ["lstm0_fw", "lstm0_bw"], 'key': 'c', "n_out": 2*5},
    "enc_state_embed": {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder_state"], "n_out": 5},
    "encoder": {"class": "copy", "from": ["lstm0_fw", "lstm0_bw"]},
    "enc_ctx": {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"], "n_out": 5},
    "enc_emb": {"class": "copy", "from": ["enc_ctx"]},

    "output": {"class": "rec", "from": [], "unit": {
      'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["output_prob"]},
      "end": {"class": "compare", "from": ["output"], "value": 0},
      'orth_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6},
      "s_in": {"class": "linear", "activation": "tanh", "from": ["prev:c", "prev:orth_embed"], "n_out": 5},
      "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["s_in"], "initial_state": {"c": "base:enc_state_embed", "h": 0}, "n_out": 5},  # h_t
      "c_in": {"class": "copy", "from": ["s"]},
      "c": {"class": "dot_attention", "from": ["c_in"], "base": "base:enc_emb", "base_ctx": "base:enc_ctx",
      "energy_factor": 1.0/numpy.sqrt(5)},
      "att": {"class": "linear", "activation": "tanh", "from": ["c", "s"], "n_out": 6},  # \tilde h
      "output_prob": {"class": "softmax", "from": ["att"], "target": "classes", "loss": "ce"}
    }, "target": "classes", "max_seq_len": 75},
  }

  from GeneratingDataset import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "/tmp/model",
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": network,
    "start_epoch": 1,
    "num_epochs": 2,
    "batch_size": 10,
    "nadam": True,
    "learning_rate": 0.01,
    "debug_add_check_numerics_ops": True
  })
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()


def test_rec_subnet_train_t3d_simple():
  beam_size = 2
  network = {
    "encoder": {"class": "linear", "activation": "tanh", "n_out": 5},
    "output": {"class": "rec", "from": [], "unit": {
      'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["output_prob"]},
      "end": {"class": "compare", "from": ["output"], "value": 0},
      'orth_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6},
      "s_in": {"class": "linear", "activation": "tanh", "from": ["prev:c", "prev:orth_embed"], "n_out": 5},
      "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["s_in"], "n_out": 5},
      "c_in": {"class": "copy", "from": ["s"]},
      "c": {"class": "dot_attention", "from": ["c_in"], "base": "base:encoder", "base_ctx": "base:encoder"},
      "att": {"class": "linear", "activation": "tanh", "from": ["c", "s"], "n_out": 6},
      "output_prob": {"class": "softmax", "from": ["att"], "target": "classes", "loss": "ce"}
    }, "target": "classes", "max_seq_len": 75},
  }

  from GeneratingDataset import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "/tmp/model",
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": network,
    "start_epoch": 1,
    "num_epochs": 2,
    "batch_size": 10,
    "nadam": True,
    "learning_rate": 0.01,
    "debug_add_check_numerics_ops": True
  })
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()


def deterministic_train_check(layer_opts):
  """
  Training should be deterministic, i.e. running it twice should result in exactly the same result.
  """
  network = {
    "hidden": layer_opts,
    "output": {"class": "softmax", "from": ["hidden"], "target": "classes", "loss": "ce"},
  }
  n_data_dim = 2
  n_classes_dim = 3
  config = Config()
  config.update({
    "model": "/tmp/model",
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": network,
    "start_epoch": 1,
    "num_epochs": 2,
    "batch_size": 10,
    "nadam": True,
    "learning_rate": 0.01,
    "debug_add_check_numerics_ops": True
  })

  from GeneratingDataset import DummyDataset
  seq_len = 5
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)

  score_results = {}  # run_idx -> epoch (1, 2) -> error_key ('dev_score', ...) -> score
  fwd_results = {}  # run_idx -> numpy array

  for run_idx in range(3):
    print("Run %i:" % run_idx)
    # Will always reinit the TF session and all random generators,
    # thus it should be deterministic.
    engine = Engine(config=config)
    engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
    engine.train()

    print("Run %i: Train results:" % run_idx)
    pprint(engine.learning_rate_control.epochData)
    score_results[run_idx] = {ep: d.error for (ep, d) in engine.learning_rate_control.epochData.items()}

    print("Run %i: Forward cv seq 0:" % run_idx)
    cv_data.init_seq_order(epoch=1)
    out = engine.forward_single(cv_data, 0)
    print(out)
    assert isinstance(out, numpy.ndarray)
    assert out.shape == (seq_len, n_classes_dim)
    fwd_results[run_idx] = out

    if run_idx > 0:
      for ep, error_dict in sorted(score_results[run_idx].items()):
        for error_key, error_value in sorted(error_dict.items()):
          prev_error_value = score_results[run_idx - 1][ep][error_key]
          print("Epoch %i, error key %r, current value %f vs prev value %f, equal?" % (
            ep, error_key, error_value, prev_error_value))
          numpy.testing.assert_almost_equal(error_value, prev_error_value)
      print("Output equal to previous?")
      prev_out = fwd_results[run_idx - 1]
      numpy.testing.assert_almost_equal(out, prev_out)

    engine.finalize()


def test_deterministic_train_linear():
  deterministic_train_check({"class": "linear", "activation": "tanh", "n_out": 5})


def test_deterministic_train_rec_nativelstm2():
  deterministic_train_check({"class": "rec", "unit": "nativelstm2", "n_out": 5})


def _create_deterministic_layer_checks():
  from TFNetworkLayer import get_layer_class_name_list, get_layer_class
  from Util import collect_mandatory_class_init_kwargs
  for cls_name in get_layer_class_name_list():
    cls = get_layer_class(cls_name)
    if cls.__name__.startswith("_"):
      continue
    mandatory_kwargs = collect_mandatory_class_init_kwargs(cls)
    mandatory_kwargs.remove("name")
    mandatory_kwargs.remove("network")
    print("Class %s (%s), mandatory: %r" % (cls.__name__, cls_name, mandatory_kwargs))
    # We could automatically add checks for layers, via deterministic_train_check(),
    # and then add "test_xxx" to globals().
    # For many kwargs, we can guess some arg (e.g. activation="tanh").
    # Some layers need specific args.
    # We also need a blacklist because some layers cannot work via deterministic_train_check(),
    # because of shape.
    # So far, we don't do this here.
  pass


def test_rec_subnet_auto_optimize():
  """
  rec subnet can automatically move out layers from the loop.
  It should result in an equivalent model.
  Thus, training should be equivalent.
  Also, training the one model, and then importing it in the original model, should work.
  """
  from TFNetworkRecLayer import RecLayer, _SubnetworkRecCell
  n_data_dim = 2
  n_classes_dim = 3
  from GeneratingDataset import DummyDataset
  seq_len = 5
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=10, seq_len=seq_len)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)

  def create_config(optimize_move_layers_out):
    """
    :param bool optimize_move_layers_out:
    :rtype: Config
    """
    beam_size = 2
    # Depending on optimize_move_layers_out, the order of the initialization of the variables might be different.
    # To make sure it's the same, we init with zero.
    # Actually, in the whole network, there should not be any randomness because of that for this check.
    weights_init = 0.01
    network = {
      "encoder": {"class": "linear", "activation": "tanh", "n_out": 5, "forward_weights_init": weights_init},
      "output": {
        "class": "rec", "from": [],
        "target": "classes", "max_seq_len": 75,
        "optimize_move_layers_out": optimize_move_layers_out,
        "unit": {
          'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["output_prob"]},
          "end": {"class": "compare", "from": ["output"], "value": 0},
          'orth_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6, "forward_weights_init": weights_init},
          "s_in": {"class": "linear", "activation": "tanh", "from": ["prev:c", "prev:orth_embed"], "n_out": 5, "forward_weights_init": weights_init},
          "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["s_in"], "n_out": 5, "weights_init": weights_init},
          "c_in": {"class": "copy", "from": ["s"]},
          "c": {"class": "dot_attention", "from": ["c_in"], "base": "base:encoder", "base_ctx": "base:encoder"},
          "att": {"class": "linear", "activation": "tanh", "from": ["c", "s"], "n_out": 6, "forward_weights_init": weights_init},
          "output_prob": {"class": "softmax", "from": ["att"], "target": "classes", "loss": "ce", "forward_weights_init": weights_init}
        },
      },
    }
    config = Config()
    config.update({
      "model": "/tmp/model",
      "num_outputs": n_classes_dim,
      "num_inputs": n_data_dim,
      "network": network,
      "start_epoch": 1,
      "num_epochs": 2,
      "batch_size": 10,
      "nadam": True,
      "learning_rate": 0.01
    })
    return config

  score_results = {}  # run_idx -> epoch (1, 2) -> error_key ('dev_score', ...) -> score
  fwd_results = {}  # run_idx -> numpy array

  def run(run_idx, optimize_move_layers_out):
    """
    :param int run_idx:
    :param bool optimize_move_layers_out:
    """
    print("Run %i:" % run_idx)
    # Will always reinit the TF session and all random generators,
    # thus it should be deterministic.
    config = create_config(optimize_move_layers_out=optimize_move_layers_out)
    engine = Engine(config=config)
    engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)

    rec_layer = engine.network.layers["output"]
    assert isinstance(rec_layer, RecLayer)
    assert isinstance(rec_layer.cell, _SubnetworkRecCell)
    if optimize_move_layers_out:
      assert_equal(set(rec_layer.cell.input_layers_moved_out), {"output", "orth_embed"})
      assert_equal(set(rec_layer.cell.output_layers_moved_out), {"output_prob", "att"})
    else:
      assert not rec_layer.cell.input_layers_moved_out
      assert not rec_layer.cell.output_layers_moved_out

    print("Run %i: Train now..." % run_idx)
    engine.train()

    print("Run %i: Train results:" % run_idx)
    pprint(engine.learning_rate_control.epochData)
    score_results[run_idx] = {ep: d.error for (ep, d) in engine.learning_rate_control.epochData.items()}

    print("Run %i: Forward cv seq 0:" % run_idx)
    cv_data.init_seq_order(epoch=1)
    out = engine.forward_single(cv_data, 0)
    print(out)
    assert isinstance(out, numpy.ndarray)
    assert out.shape == (seq_len,)  # label sequence
    fwd_results[run_idx] = out

    if len(score_results) > 1:
      for ep, error_dict in sorted(score_results[run_idx].items()):
        for error_key, error_value in sorted(error_dict.items()):
          prev_error_value = score_results[run_idx - 1][ep][error_key]
          print("Epoch %i, error key %r, current value %f vs prev value %f, equal?" % (
            ep, error_key, error_value, prev_error_value))
          numpy.testing.assert_almost_equal(error_value, prev_error_value, decimal=3)
      print("Output equal to previous?")
      prev_out = fwd_results[run_idx - 1]
      numpy.testing.assert_almost_equal(out, prev_out, decimal=3)

    engine.finalize()

  run(run_idx=1, optimize_move_layers_out=False)
  run(run_idx=2, optimize_move_layers_out=False)
  run(run_idx=3, optimize_move_layers_out=True)


def test_rec_subnet_construct_1():
  """
  Test for a bug in SearchChoices.translate_to_common_search_beam with the prev-layer template.
  """
  n_data_dim = 2
  n_classes_dim = 3
  from GeneratingDataset import DummyDataset
  seq_len = 5
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=10, seq_len=seq_len)

  beam_size = 2
  net_dict = {
    "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6},

    "lstm0_fw": {"class": "rec", "unit": "nativelstm2", "n_out": 10, "direction": 1, "from": ["source_embed"]},
    "lstm0_bw": {"class": "rec", "unit": "nativelstm2", "n_out": 10, "direction": -1, "from": ["source_embed"]},

    "lstm1_fw": {"class": "rec", "unit": "nativelstm2", "n_out": 10, "direction": 1, "from": ["lstm0_fw", "lstm0_bw"]},
    "lstm1_bw": {"class": "rec", "unit": "nativelstm2", "n_out": 10, "direction": -1, "from": ["lstm0_fw", "lstm0_bw"]},

    "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
    "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
    "inv_fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": 1},

    "output": {"class": "rec", "from": [], "unit": {
      'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["output_prob"], "initial_output": 0},
      "end": {"class": "compare", "from": ["output"], "value": 0},
      'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6, "initial_output": 0},
      "weight_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"], "n_out": 10},
      "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s2"], "n_out": 20},
      "prev_s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev_s_state"], "n_out": 10},
      "energy_in": {"class": "combine", "kind": "add", "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"], "n_out": 10},
      "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
      "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 1},
      "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},
      "accum_att_weights": {"class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
                            "eval": "source(0) + source(1) * source(2) * 0.5",
                            "out_type": {"dim": 1, "shape": (None, 1)}},
      "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder"},
      "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["target_embed", "att"], "n_out": 10},
      "s2": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["s"], "n_out": 10},
      "readout_in": {"class": "linear", "from": ["prev:s2", "prev:target_embed", "att"], "activation": None, "n_out": 10},
      "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
      "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce", "dropout": 0.3}
    }, "target": "classes", "max_seq_len": 7},

    "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"}
  }

  with tf.Graph().as_default():
    extern_data = ExternData()
    extern_data.init_from_dataset(train_data)
    print("Construct train net")
    train_net = TFNetwork(extern_data=extern_data, train_flag=True)
    train_net.construct_from_dict(net_dict)
    print("Construct search net")
    search_net = TFNetwork(extern_data=extern_data, train_flag=False, eval_flag=True, search_flag=True)
    search_net.construct_from_dict(net_dict)


def test_rec_subnet_construct_2():
  n_data_dim = 2
  n_classes_dim = 3
  from GeneratingDataset import DummyDataset
  seq_len = 5
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=10, seq_len=seq_len)

  beam_size = 2
  net_dict = {
    "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6},

    "lstm0_fw": {"class": "rec", "unit": "nativelstm2", "n_out": 10, "direction": 1, "from": ["source_embed"]},
    "lstm0_bw": {"class": "rec", "unit": "nativelstm2", "n_out": 10, "direction": -1, "from": ["source_embed"]},

    "lstm1_fw": {"class": "rec", "unit": "nativelstm2", "n_out": 10, "direction": 1, "from": ["lstm0_fw", "lstm0_bw"]},
    "lstm1_bw": {"class": "rec", "unit": "nativelstm2", "n_out": 10, "direction": -1, "from": ["lstm0_fw", "lstm0_bw"]},

    "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
    "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
    "inv_fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": 1},

    "output": {"class": "rec", "from": [], "unit": {
      'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["output_prob"], "initial_output": 0},
      "end": {"class": "compare", "from": ["output"], "value": 0},
      'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6, "initial_output": 0},
      "weight_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"], "n_out": 10},
      "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
      "prev_s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev_s_state"], "n_out": 10},
      "energy_in": {"class": "combine", "kind": "add", "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"], "n_out": 10},
      "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
      "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 1},
      "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},
      "accum_att_weights": {"class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
                            "eval": "source(0) + source(1) * source(2) * 0.5",
                            "out_type": {"dim": 1, "shape": (None, 1)}},
      "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder"},
      "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["target_embed", "att"], "n_out": 10},
      "s2": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["s"], "n_out": 10},
      "readout_in": {"class": "linear", "from": ["prev:s2", "prev:target_embed", "att"], "activation": None, "n_out": 10},
      "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
      "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce", "dropout": 0.3}
    }, "target": "classes", "max_seq_len": 75},

    "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"}
  }

  with tf.Graph().as_default():
    extern_data = ExternData()
    extern_data.init_from_dataset(train_data)
    print("Construct train net")
    train_net = TFNetwork(extern_data=extern_data, train_flag=True)
    train_net.construct_from_dict(net_dict)
    rec_layer = train_net.layers["output"]
    from TFNetworkRecLayer import RecLayer, _SubnetworkRecCell
    assert isinstance(rec_layer, RecLayer)
    assert isinstance(rec_layer.cell, _SubnetworkRecCell)
    assert_equal(set(rec_layer.cell.input_layers_moved_out), {"output", "target_embed"})
    assert_equal(set(rec_layer.cell.output_layers_moved_out), {"output_prob", "readout", "readout_in", "s2"})
    print("Construct search net")
    search_net = TFNetwork(extern_data=extern_data, train_flag=False, eval_flag=True, search_flag=True)
    search_net.construct_from_dict(net_dict)


def test_rec_subnet_construct_3():
  n_data_dim = 2
  n_classes_dim = 3
  from GeneratingDataset import DummyDataset
  seq_len = 5
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=10, seq_len=seq_len)

  beam_size = 2
  net_dict = {
    "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6},

    "lstm0_fw": {"class": "rec", "unit": "nativelstm2", "n_out": 10, "direction": 1, "from": ["source_embed"]},
    "lstm0_bw": {"class": "rec", "unit": "nativelstm2", "n_out": 10, "direction": -1, "from": ["source_embed"]},

    "lstm1_fw": {"class": "rec", "unit": "nativelstm2", "n_out": 10, "direction": 1, "from": ["lstm0_fw", "lstm0_bw"]},
    "lstm1_bw": {"class": "rec", "unit": "nativelstm2", "n_out": 10, "direction": -1, "from": ["lstm0_fw", "lstm0_bw"]},

    "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
    "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},
    "inv_fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": 1},

    "output": {"class": "rec", "from": [], "unit": {
      'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["output_prob"], "initial_output": 0},
      "end": {"class": "compare", "from": ["output"], "value": 0},
      'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6, "initial_output": 0},
      "weight_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"], "n_out": 10},
      "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 20},
      "prev_s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev_s_state"], "n_out": 10},
      "energy_in": {"class": "combine", "kind": "add", "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"], "n_out": 10},
      "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
      "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 1},
      "att_weights": {"class": "softmax_over_spatial", "from": ["energy"]},
      "accum_att_weights": {"class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
                            "eval": "source(0) + source(1) * source(2) * 0.5",
                            "out_type": {"dim": 1, "shape": (None, 1)}},
      "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder"},
      "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["target_embed", "att"], "n_out": 10},
      "s2": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:s", "prev:target_embed", "att"], "n_out": 10},
      "readout_in": {"class": "linear", "from": ["s2"], "activation": None, "n_out": 10},
      "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
      "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce", "dropout": 0.3}
    }, "target": "classes", "max_seq_len": 75},

    "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", "target": "classes"}
  }

  with tf.Graph().as_default():
    extern_data = ExternData()
    extern_data.init_from_dataset(train_data)
    print("Construct train net")
    train_net = TFNetwork(extern_data=extern_data, train_flag=True)
    train_net.construct_from_dict(net_dict)
    rec_layer = train_net.layers["output"]
    from TFNetworkRecLayer import RecLayer, _SubnetworkRecCell
    assert isinstance(rec_layer, RecLayer)
    assert isinstance(rec_layer.cell, _SubnetworkRecCell)
    assert_equal(set(rec_layer.cell.input_layers_moved_out), {"output", "target_embed"})
    assert_equal(set(rec_layer.cell.output_layers_moved_out), {"output_prob", "readout", "readout_in", "s2"})
    print("Construct search net")
    search_net = TFNetwork(extern_data=extern_data, train_flag=False, eval_flag=True, search_flag=True)
    search_net.construct_from_dict(net_dict)


def test_rec_subnet_eval_init_out_apply0():
  # network
  # (also defined by num_inputs & num_outputs)
  beam_size = 3
  AttNumHeads = 2
  EncKeyTotalDim = AttNumHeads * 2
  EncKeyPerHeadDim = EncKeyTotalDim // AttNumHeads
  EncValueTotalDim = AttNumHeads * 2
  EncValuePerHeadDim = EncValueTotalDim // AttNumHeads
  network = {
    "lstm0_fw": {"class": "rec", "unit": "nativelstm2", "n_out": 2, "direction": 1},
    "lstm0_bw": {"class": "rec", "unit": "nativelstm2", "n_out": 2, "direction": -1},
    "lstm0_pool": {"class": "pool", "mode": "max", "padding": "same", "pool_size": (2,),
                   "from": ["lstm0_fw", "lstm0_bw"], "trainable": False},
    "encoder": {"class": "copy", "from": ["lstm0_pool"]},
    "enc_ctx0": {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"],
                 "n_out": EncKeyTotalDim},  # (B, enc-T, D)
    "enc_ctx": {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncKeyPerHeadDim), "from": ["enc_ctx0"]},
    "enc_value0": {"class": "linear", "activation": None, "with_bias": False, "from": ["encoder"],
                   "n_out": EncValueTotalDim},
    "enc_value": {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncValuePerHeadDim),
                  "from": ["enc_value0"]},  # (B, enc-T, H, D'/H)
    "inv_fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": ["encoder"], "n_out": 1},

    "output": {"class": "rec", "from": [], "unit": {
      'output': {'class': 'choice', 'target': 'classes', 'beam_size': beam_size, 'from': ["output_prob"],
                 "initial_output": 0},
      "end": {"class": "compare", "from": ["output"], "value": 0},
      'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 6,
                       "initial_output": 0},  # feedback_input
      "weight_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev:accum_att_weights"],
                          "n_out": 2},  # (B, enc-T, 1000)
      "prev_s_state": {"class": "get_last_hidden_state", "from": ["prev:s"], "n_out": 4},
      "prev_s_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": ["prev_s_state"],
                          "n_out": 2},  # (B, D)  -- Q (query). D should be same as enc_ctx
      "att_query0": {"class": "combine", "kind": "add", "from": ["weight_feedback", "prev_s_feedback"], "n_out": 2},
      "att_query1": {"class": "activation", "activation": "tanh", "from": ["att_query0"]},
      "att_query2": {"class": "linear", "activation": None, "with_bias": False, "from": ["att_query1"],
                     "n_out": EncKeyTotalDim},  # (B, enc-T, D)
      "att_query": {"class": "split_dims", "axis": "F", "dims": (AttNumHeads, EncKeyPerHeadDim),
                    "from": ["att_query2"]},  # (B, enc-T, H, D/H)
      "energy": {"class": "dot", "red1": -1, "red2": -1, "var1": None, "var2": None,
                 "from": ["base:enc_ctx", "att_query"], "debug": True},  # (B, enc-T, H, 1)

      "att_weights": {"class": "softmax_over_spatial", "from": ["energy"], "energy_factor": EncKeyPerHeadDim ** -0.5},
      "att_weights_avg": {"class": "reduce", "axes": -2, "mode": "avg", "from": ["att_weights"]},  # (B, enc-T, 1)
      "accum_att_weights": {"class": "eval",
                            "from": ["prev:accum_att_weights", "att_weights_avg", "base:inv_fertility"],
                            "eval": "source(0) + source(1) * source(2) * 0.5",
                            "out_type": {"dim": 1, "shape": (None, 1)}, "initial_output": "apply(0)"},  # (B, enc-T, 1)
      "att0": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, H, V)
      "att": {"class": "merge_dims", "axes": "except_batch", "from": ["att0"]},  # (B, H*V)

      "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["target_embed", "att"], "n_out": 2},  # transform
      "readout_in": {"class": "linear", "from": ["prev:s", "prev:target_embed", "att"], "activation": None,
                     "n_out": 2},  # merge + post_merge bias
      "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
      "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"}
    }, "target": "classes", "max_seq_len": 7},
  }

  from GeneratingDataset import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "/tmp/model",
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": network,
    "start_epoch": 1,
    "num_epochs": 2,
    "batch_size": 10,
    "nadam": True,
    "learning_rate": 0.01
  })

  print("Create engine.")
  engine = Engine(config=config)
  print("Init for train.")
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  print("Train.")
  engine.train()
  print("Search.")
  engine.search(cv_data)


def test_net_safe_log_to_log_softmax():
  n_out = 5
  net_dict = {
    "ff_in_window": {"class": "window", "window_size": 3, "trainable": False},
    "ff_in": {"class": "merge_dims", "axes": "except_time", "from": ["ff_in_window"], "trainable": False},
    "ff0": {"class": "hidden", "activation": "relu", "n_out": 8, "L2": 0.01, "from": ["ff_in"]},
    "ff_out": {"class": "softmax", "n_out": n_out, "from": ["ff0"]},
    "ff_out_prior": {
      "class": "accumulate_mean", "exp_average": 0.001,
      "is_prob_distribution": True, "from": ["ff_out"]},
    "output": {
      "class": "combine", "kind": "eval", "from": ["ff_out", "ff_out_prior"],
      "eval": "safe_log(source(0)) - safe_log(source(1))",
      "eval_locals": {"am_scale": 0.1, "prior_scale": 0.5 * 0.1}
    },
  }
  net = TFNetwork(extern_data=ExternData(data={"data": {"dim": 3}, "classes": {"dim": n_out, "sparse": True}}))
  net.construct_from_dict(net_dict)
  output_layer = net.get_default_output_layer(must_exist=True)
  out = output_layer.output.placeholder
  print(out)
  from TFUtil import print_graph_output
  print_graph_output(out)
  assert out.op.type == "Sub"
  assert len(out.op.inputs) == 2
  sub_in0, sub_in1 = out.op.inputs
  print(sub_in0, sub_in1)
  assert isinstance(sub_in0, tf.Tensor)
  assert isinstance(sub_in1, tf.Tensor)
  assert "/safe_log" in sub_in0.name
  assert "/safe_log" in sub_in1.name
  assert sub_in1.op.type == "Log"
  # This is what we want to test now:
  # :func:`safe_log` should figure out that a softmax was used before and then use log_softmax for a stable calculation.
  # See also :func:`test_check_base_op_type_and_replace_softmax`.
  if sub_in0.op.type != "LogSoftmax" and sub_in0.op.inputs[0].op.type != "LogSoftmax":
    # It failed. Print some helpful information.
    print("not LogSoftmax:", sub_in0)
    print("inputs:", list(sub_in0.op.inputs))
    print("inputs':", list(sub_in0.op.inputs[0].op.inputs))
    print("inputs'':", list(sub_in0.op.inputs[0].op.inputs[0].op.inputs))
  assert sub_in0.op.type == "LogSoftmax" or sub_in0.op.inputs[0].op.type == "LogSoftmax"


if __name__ == "__main__":
  try:
    better_exchook.install()
    if len(sys.argv) <= 1:
      for k, v in sorted(globals().items()):
        if k.startswith("test_"):
          print("-" * 40)
          print("Executing: %s" % k)
          try:
            v()
          except unittest.SkipTest as exc:
            print("SkipTest:", exc)
          print("-" * 40)
    else:
      assert len(sys.argv) >= 2
      for arg in sys.argv[1:]:
        print("Executing: %s" % arg)
        if arg in globals():
          globals()[arg]()  # assume function and execute
        else:
          eval(arg)  # assume Python code and execute
  finally:
    session.close()
    del session
    tf.reset_default_graph()
    import threading
    if len(list(threading.enumerate())) > 1:
      print("Warning, more than one thread at exit:")
      better_exchook.dump_all_thread_tracebacks()
