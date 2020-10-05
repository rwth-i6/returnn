
from __future__ import print_function

import sys
import os

import _setup_test_env  # noqa
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false, assert_greater, assert_almost_equal, assert_is
from returnn.config import Config
from returnn.theano.engine import Engine
from returnn.theano.device import Device
from returnn.log import log
from returnn.datasets.generating import StaticDataset, DummyDataset
from returnn.theano.engine_util import assign_dev_data_single_seq
import returnn.theano.util as theano_util
import numpy
from pprint import pprint
import theano
from theano import tensor as T
from returnn.util import better_exchook


theano_util.monkey_patches()


def get_num_params(p_list):
  return sum([numpy.prod(v.get_value(borrow=True, return_internal_type=True).shape[0:]) for v in p_list])


def test_single_target_init():
  config_single = Config()
  config_single.update({
    "multiprocessing": False,
    "blocking": True,
    "device": "cpu",
    "num_epochs": 1,
    "num_inputs": 3,
    "num_outputs": 2,
    "target": "t1",
  })
  config_single.network_topology_json = """
  {
  "out": {"class": "softmax", "loss": "ce", "target": "t1"}
  }
  """

  dev = Device("cpu", config=config_single, blocking=True)
  num_params = get_num_params(dev.trainnet.get_all_params_vars())
  assert_equal(num_params, 3 * 2 + 2, "W, b")


def test_single_default_target_init():
  config_single_default = Config()
  config_single_default.update({
    "multiprocessing": False,
    "blocking": True,
    "device": "cpu",
    "num_epochs": 1,
    "num_inputs": 3,
    "num_outputs": 2,
  })
  config_single_default.network_topology_json = """
  {
  "output": {"class": "softmax", "loss": "ce"}
  }
  """

  dev = Device("cpu", config=config_single_default, blocking=True)
  num_params = get_num_params(dev.trainnet.get_all_params_vars())
  assert_equal(num_params, 3 * 2 + 2, "W, b")


def test_multi_target_init():
  config = Config()
  config.update({
    "multiprocessing": False,
    "blocking": True,
    "device": "cpu",
    "num_epochs": 1,
    "num_inputs": 3,
    "num_outputs": {"t1": 4, "t2": 5},
    "learning_rate": 1.0,
  })
  config.network_topology_json = """
  {
  "fw0": {"class": "hidden", "activation": "identity", "n_out": 3},
  "out1": {"class": "softmax", "loss": "ce", "target": "t1", "from": ["fw0"]},
  "out2": {"class": "softmax", "loss": "ce", "target": "t2", "from": ["fw0"]}
  }
  """

  device = Device("cpu", config=config, blocking=True)
  assert_true(device.trainnet, "train network initialized")
  assert_true(device.testnet, "test network initialized")
  param_vars = device.trainnet.get_all_params_vars()
  print("params:", param_vars)
  assert_equal(len(param_vars), 6, "W, b vars for each out, and fw")
  num_params = get_num_params(param_vars)
  assert_equal(num_params, (3 * 3 + 3) + (3 * 4 + 4) + (3 * 5 + 5), "W, b for each out, and fw")
  assert_in("fw0", device.testnet.hidden)
  assert_in("out1", device.testnet.output)
  assert_in("out2", device.testnet.output)
  assert_is(device.testnet.j["t1"], device.testnet.output["out1"].index)
  assert_true(device.updater)
  update_list = device.updater.getUpdateList()
  print("update list:")
  pprint(update_list)
  update_dict = dict(update_list)
  assert_equal(len(update_dict), len(update_list), "all params in update list only once")
  assert_in("fw0", device.trainnet.hidden)
  assert_equal(len(device.trainnet.hidden), 1)
  assert_in("W_in_data_fw0", device.trainnet.hidden["fw0"].params)
  assert_in("b_fw0", device.trainnet.hidden["fw0"].params)
  assert_equal(len(device.trainnet.hidden["fw0"].params), 2)
  assert_in("out1", device.trainnet.output)
  assert_equal(len(device.trainnet.output), 2)
  assert_in("W_in_fw0_out1", device.trainnet.output["out1"].params)
  assert_in("b_out1", device.trainnet.output["out1"].params)
  assert_equal(len(device.trainnet.output["out1"].params), 2)
  assert_in(device.trainnet.hidden["fw0"].params["W_in_data_fw0"], update_dict)
  assert_in(device.trainnet.hidden["fw0"].params["b_fw0"], update_dict)
  assert_in(device.trainnet.output["out1"].params["W_in_fw0_out1"], update_dict)
  assert_in(device.trainnet.output["out1"].params["b_out1"], update_dict)
  assert_in(device.trainnet.output["out2"].params["W_in_fw0_out2"], update_dict)
  assert_in(device.trainnet.output["out2"].params["b_out2"], update_dict)
  # assert_equal(len(update_dict), 6)  # updater adds other stuff...

  # Set net params.
  net_params = {
    "fw0": {"W_in_data_fw0": numpy.identity(3, dtype="float32"),
            "b_fw0": numpy.zeros((3,), dtype="float32")},
    "out1": {"W_in_fw0_out1": numpy.arange(0.0, 1.2, 0.1, dtype="float32").reshape((3, 4)),
             "b_out1": numpy.arange(0.0, 4, dtype="float32")},
    "out2": {"W_in_fw0_out2": numpy.arange(0.0, 1.5, 0.1, dtype="float32").reshape((3, 5)),
             "b_out2": numpy.arange(0.0, 5, dtype="float32")}
  }
  device.trainnet.set_params_by_dict(net_params)
  device.testnet.set_params_by_dict(net_params)

  # Show params.
  for p in param_vars:
    print("init %s:" % p)
    pprint(p.get_value())

  # Init dataset.
  dataset = StaticDataset(data=[{
    "data": numpy.array([[0.1, 0.2, -0.3]], dtype="float32"),
    "t1": numpy.array([2]),
    "t2": numpy.array([4])
  }], output_dim=config.typed_value("num_outputs"))
  dataset.init_seq_order()
  assert_equal(dataset.is_data_sparse("data"), False)
  assert_equal(dataset.is_data_sparse("t1"), True)
  assert_equal(dataset.is_data_sparse("t2"), True)

  # Copy to device allocation.
  success = assign_dev_data_single_seq(device, dataset, 0)
  assert_true(success, "failed to allocate & assign data")

  # Check allocated data.
  assert_equal(device.targets["data"].shape, (1, 1, 3))  # input shape. (time,batch,dim)
  assert_in("t1", device.targets)
  assert_in("t2", device.targets)
  assert_equal(device.targets["t1"].shape, (1, 1))
  assert_equal(device.targets["t2"].shape, (1, 1))
  assert_equal(device.output_index["data"].shape, (1, 1))
  numpy.testing.assert_equal(device.output_index["data"], numpy.array([[1]]))
  assert_equal(device.output_index["t1"].shape, (1, 1))
  numpy.testing.assert_equal(device.output_index["t1"], numpy.array([[1]]))

  # Forward test.
  device.update_data()
  device.testnet.costs["out1"].name = "out1_cost"  # nice in the func graph
  out_i1 = device.testnet.output["out1"].index
  out_i1_nonzero = device.testnet.output["out1"].i
  nll1, pcx1 = T.nnet.crossentropy_softmax_1hot(x=device.testnet.output["out1"].y_m[out_i1_nonzero],
                                                y_idx=device.testnet.output["out1"].y_data_flat[out_i1_nonzero])
  forward_func = theano.function(
    inputs=[device.block_start, device.block_end],
    outputs=[
      device.testnet.j["t1"], out_i1, out_i1_nonzero[0], nll1, pcx1,
      device.testnet.costs["out1"],
      device.testnet.output["out1"].p_y_given_x,
      device.testnet.costs["out2"],
      device.testnet.output["out2"].p_y_given_x],
    givens=device.make_givens(device.testnet),
    no_default_updates=True,
    on_unused_input='warn',
    name="forward")
  #print "forward func:"
  #theano.printing.debugprint(forward_func)
  net_j1, out_i1_val, out_i1_nz_val, nll1_val, pcx1_val, t1_cost, t1_y, t2_cost, t2_y = forward_func(0, 1)
  print("forward results:")
  pprint(net_j1)
  pprint(out_i1_val)
  pprint(out_i1_nz_val)
  pprint(nll1_val)
  pprint(pcx1_val)
  pprint(t1_cost)
  pprint(t1_y)
  pprint(t2_cost)
  pprint(t2_y)
  assert_equal(net_j1, numpy.array([[1]]))
  assert_equal(out_i1_val, numpy.array([[1]]))
  assert_equal(out_i1_nz_val, numpy.array([0]))
  assert_almost_equal(nll1_val, numpy.array([t1_cost]))
  numpy.testing.assert_almost_equal(t1_y, pcx1_val[None,...])
  assert_almost_equal(t1_cost, 1.440189698561195, places=6)
  assert_almost_equal(t2_cost, 0.45191439593759336, places=6)
  numpy.testing.assert_almost_equal(t1_y, numpy.array([[[ 0.0320586 ,  0.08714432,  0.23688282,  0.64391426]]]), decimal=6)
  numpy.testing.assert_almost_equal(t2_y, numpy.array([[[ 0.01165623,  0.03168492,  0.08612854,  0.23412166,  0.63640865]]]), decimal=6)

  # One train step.
  device.set_learning_rate(config.typed_value("learning_rate"))
  device.run("train")
  output_list, outputs_format = device.result()
  assert_is_instance(output_list, list)
  assert_true(outputs_format, "for train, we should always get the format")
  outputs = Device.make_result_dict(output_list, outputs_format)
  pprint(outputs)
  assert_in("cost:out1", outputs)
  assert_greater(outputs["cost:out1"], 0)
  assert_almost_equal(outputs["cost:out1"], t1_cost)

  # Get net params.
  params = device.get_net_train_params(device.trainnet)
  references_params = {
    "W_in_data_fw0":
      numpy.array([[  1.00055406e+00,   5.54056978e-04,   5.54056978e-04],
                   [  1.10811396e-03,   1.00110811e+00,   1.10811396e-03],
                   [ -1.66217093e-03,  -1.66217093e-03,   9.98337829e-01]]),
    "b_fw0":
      numpy.array([ 0.00554057,  0.00554057,  0.00554057]),
    "W_in_fw0_out1":
      numpy.array([[-0.00320586,  0.09128557,  0.27631172,  0.23560857],
                   [ 0.39358828,  0.48257114,  0.75262344,  0.57121715],
                   [ 0.80961758,  0.9261433 ,  0.77106485,  1.29317428]]),
    "b_out1":
      numpy.array([-0.0320586 ,  0.91285568,  2.76311718,  2.35608574]),
    "W_in_fw0_out2":
      numpy.array([[ -1.16562310e-03,   9.68315079e-02,   1.91387146e-01,
                      2.76587834e-01,   4.36359135e-01],
                   [  4.97668754e-01,   5.93663016e-01,   6.82774291e-01,
                      7.53175669e-01,   9.72718271e-01],
                   [  1.00349687e+00,   1.10950548e+00,   1.22583856e+00,
                      1.37023650e+00,   1.29092259e+00]]),
    "b_out2":
      numpy.array([-0.01165623,  0.96831508,  1.91387146,  2.76587834,  4.36359135])
  }
  assert_equal(len(param_vars), len(params))
  for p, v in zip(param_vars, params):
    print("%s:" % p)
    pprint(v)
    assert_true(p.name)
    numpy.testing.assert_almost_equal(references_params[p.name], v, decimal=6)


def test_combi_auto_enc():
  config = Config()
  config.update({
    "multiprocessing": False,
    "blocking": True,
    "device": "cpu",
    "num_epochs": 1,
    "num_inputs": 3,
    "num_outputs": {"classes": 2},
    "learning_rate": 1.0,
    "network": {
      "output": {"class": "softmax", "loss": "ce", "target": "classes"},
      "auto-enc": {"class": "softmax", "loss": "sse", "dtype": "float32", "target": "data"}
    }
  })

  device = Device("cpu", config=config, blocking=True)

  # Set net params.
  def get_net_params(with_auto_enc=True):
    d = {
      "output": {"W_in_data_output": numpy.arange(0.1, 0.7, 0.1, dtype="float32").reshape((3, 2)),
                 "b_output": numpy.arange(0.0, 2, dtype="float32")}
    }
    if with_auto_enc:
      d["auto-enc"] = {"W_in_data_auto-enc": numpy.arange(0.1, 1.0, 0.1, dtype="float32").reshape((3, 3)),
                       "b_auto-enc": numpy.arange(0.0, 3, dtype="float32")}
    return d
  device.trainnet.set_params_by_dict(get_net_params())
  device.testnet.set_params_by_dict(get_net_params())

  # Show params.
  for p in device.trainnet.get_all_params_vars():
    print("init %s:" % p)
    pprint(p.get_value())

  # Init dataset.
  dataset = StaticDataset(data=[{
    "data": numpy.array([[0.1, 0.2, -0.3]], dtype="float32"),
    "classes": numpy.array([1]),
  }], output_dim=config.typed_value("num_outputs"))
  dataset.init_seq_order()

  # Copy to device allocation.
  success = assign_dev_data_single_seq(device, dataset, 0)
  assert_true(success, "failed to allocate & assign data")

  # One train step.
  device.set_learning_rate(config.typed_value("learning_rate"))
  device.run("train")
  output_list, outputs_format = device.result()
  assert_is_instance(output_list, list)
  assert_true(outputs_format, "for train, we should always get the format")
  outputs = Device.make_result_dict(output_list, outputs_format)
  pprint(outputs)
  assert_in("cost:output", outputs)
  assert_in("cost:auto-enc", outputs)
  expected_cost_output = 0.3132616877555847
  assert_almost_equal(outputs["cost:output"], expected_cost_output, places=6)
  exact_cost_output = outputs["cost:output"]
  assert_almost_equal(outputs["cost:auto-enc"], 1.7544001340866089, places=6)

  # Now, drop the auto-enc from the network, and redo the same thing.
  del config.typed_value("network")["auto-enc"]
  device = Device("cpu", config=config, blocking=True)
  device.trainnet.set_params_by_dict(get_net_params(with_auto_enc=False))
  device.testnet.set_params_by_dict(get_net_params(with_auto_enc=False))
  for p in device.trainnet.get_all_params_vars():
    print("second run, init %s:" % p)
    pprint(p.get_value())
  dataset.init_seq_order()  # reset. probably not needed
  success = assign_dev_data_single_seq(device, dataset, 0)
  assert_true(success, "failed to allocate & assign data")
  device.set_learning_rate(config.typed_value("learning_rate"))
  device.run("train")
  output_list, outputs_format = device.result()
  assert_is_instance(output_list, list)
  assert_true(outputs_format, "for train, we should always get the format")
  outputs = Device.make_result_dict(output_list, outputs_format)
  pprint(outputs)
  assert_in("cost:output", outputs)
  assert_not_in("cost:auto-enc", outputs)
  assert_almost_equal(outputs["cost:output"], expected_cost_output, places=6)
  assert_equal(outputs["cost:output"], exact_cost_output)


def test_combi_auto_enc_longer():
  config = Config()
  config.update({
    "multiprocessing": False,
    "blocking": True,
    "device": "cpu",
    "num_epochs": 1,
    "num_inputs": 3,
    "num_outputs": {"classes": 2},
    "learning_rate": 1.0,
    "adadelta": True,
    "network": {
      "output": {"class": "softmax", "loss": "ce", "target": "classes"},
      "auto-enc": {"class": "softmax", "loss": "sse", "dtype": "float32", "target": "data"}
    }
  })

  device = Device("cpu", config=config, blocking=True)

  # Set net params.
  def get_net_params(with_auto_enc=True):
    d = {
      "output": {"W_in_data_output": numpy.arange(0.1, 0.7, 0.1, dtype="float32").reshape((3, 2)),
                 "b_output": numpy.arange(0.0, 2, dtype="float32")}
    }
    if with_auto_enc:
      d["auto-enc"] = {"W_in_data_auto-enc": numpy.arange(0.1, 1.0, 0.1, dtype="float32").reshape((3, 3)),
                       "b_auto-enc": numpy.arange(0.0, 3, dtype="float32")}
    return d
  device.trainnet.set_params_by_dict(get_net_params())
  device.testnet.set_params_by_dict(get_net_params())

  # Show params.
  for p in device.trainnet.get_all_params_vars():
    print("init %s:" % p)
    pprint(p.get_value())

  # Init dataset.
  dataset = DummyDataset(input_dim=config.typed_value("num_inputs"),
                         output_dim=config.typed_value("num_outputs"),
                         num_seqs=10)
  dataset.init_seq_order()

  cost_output_sum = 0.0
  for seq_idx in range(dataset.num_seqs):
    # Copy to device allocation.
    success = assign_dev_data_single_seq(device, dataset, seq_idx)
    assert_true(success, "failed to allocate & assign data")

    # One train step.
    device.set_learning_rate(config.typed_value("learning_rate"))
    device.run("train")
    output_list, outputs_format = device.result()
    assert_is_instance(output_list, list)
    assert_true(outputs_format, "for train, we should always get the format")
    outputs = Device.make_result_dict(output_list, outputs_format)
    print(("seq %i" % seq_idx))
    pprint(outputs)
    assert_in("cost:output", outputs)
    assert_in("cost:auto-enc", outputs)
    cost_output_sum += outputs["cost:output"]

  # Now, drop the auto-enc from the network, and redo the same thing.
  del config.typed_value("network")["auto-enc"]
  device = Device("cpu", config=config, blocking=True)
  device.trainnet.set_params_by_dict(get_net_params(with_auto_enc=False))
  device.testnet.set_params_by_dict(get_net_params(with_auto_enc=False))
  for p in device.trainnet.get_all_params_vars():
    print("second run, init %s:" % p)
    pprint(p.get_value())
  dataset.init_seq_order()  # reset

  cost2_output_sum = 0.0
  for seq_idx in range(dataset.num_seqs):
    # Copy to device allocation.
    success = assign_dev_data_single_seq(device, dataset, seq_idx)
    assert_true(success, "failed to allocate & assign data")

    # One train step.
    device.set_learning_rate(config.typed_value("learning_rate"))
    device.run("train")
    output_list, outputs_format = device.result()
    assert_is_instance(output_list, list)
    assert_true(outputs_format, "for train, we should always get the format")
    outputs = Device.make_result_dict(output_list, outputs_format)
    print(("seq %i" % seq_idx))
    pprint(outputs)
    assert_in("cost:output", outputs)
    assert_not_in("cost:auto-enc", outputs)
    cost2_output_sum += outputs["cost:output"]

  assert_equal(cost_output_sum, cost2_output_sum)
  assert_almost_equal(cost_output_sum, 16.028842568397522, places=6)
