
# start test like this:  nosetests-2.7  tests/test_TFEngine.py
# or directly:  python3 test_TFEngine.py test_engine_rec_subnet_count

from __future__ import print_function

import _setup_test_env  # noqa
import tensorflow as tf
from returnn.tf.engine import *
from returnn.tf.util.data import Dim, SpatialDim, FeatureDim
from returnn.tf.network import ExternData
from returnn.config import Config
from nose.tools import assert_equal, assert_is_instance, assert_raises
import unittest
import numpy
import numpy.testing
from pprint import pprint
import contextlib
from returnn.util import better_exchook


print("TF version:", tf.__version__)


@contextlib.contextmanager
def make_scope():
  """
  :rtype: tf.compat.v1.Session
  """
  with tf.Graph().as_default() as graph:
    with tf_compat.v1.Session(graph=graph) as session:
      yield session


def _get_tmp_file(suffix):
  """
  :param str suffix:
  :return: filename
  :rtype: str
  """
  import tempfile
  f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
  f.close()
  fn = f.name
  import atexit
  atexit.register(lambda: os.remove(fn))
  return fn


def _get_tmp_dir():
  """
  :return: dirname
  :rtype: str
  """
  import tempfile
  import shutil
  import atexit
  name = tempfile.mkdtemp()
  assert name and os.path.isdir(name) and not os.listdir(name)
  atexit.register(lambda: shutil.rmtree(name))
  return name


def _cleanup_old_models(config):
  """
  :param Config config:
  """
  model_prefix = config.value("model", None)
  from glob import glob
  files = glob("%s.*" % model_prefix)
  if files:
    print("Delete old models:", files)
    for fn in files:
      os.remove(fn)


session = tf_compat.v1.InteractiveSession()


def test_FeedDictDataProvider():
  from returnn.datasets.generating import DummyDataset
  num_seqs = 2
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=num_seqs, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)

  extern_data = ExternData()
  extern_data.init_from_dataset(dataset)

  # No Runner instance here but a very simplified version of Runner.run().
  # First we need a custom DataProvider with a custom BatchSetGenerator
  # which will yield only one single batch for the provided sequence idx.
  n_batch = 1
  input_batches = []
  for seq_idx in range(num_seqs):
    batch = Batch()
    batch.add_frames(seq_idx=seq_idx, seq_start_frame=0, length=dataset.get_seq_length(seq_idx))
    input_batches.append(batch)
  batch_generator = iter(input_batches)
  batches = BatchSetGenerator(dataset, generator=batch_generator)
  from returnn.tf.data_pipeline import FeedDictDataProvider
  data_provider = FeedDictDataProvider(
    tf_session=session, extern_data=extern_data,
    data_keys=["data", "classes"],
    dataset=dataset, batches=batches)

  # The values that happen to be produced by DummyDataset for the two sequences.
  expected_first_data = [[-0.5, -0.4], [-0.4, -0.3]]
  expected_last_data = [[0.3, 0.4], [ 0.4, -0.5]]
  expected_classes = [[1, 2, 0, 1, 2], [2, 0, 1, 2, 0]]

  for seq_idx in range(num_seqs):
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
    numpy.testing.assert_almost_equal(list(data[0, 0]), expected_first_data[seq_idx])
    numpy.testing.assert_almost_equal(list(data[0, -1]), expected_last_data[seq_idx])
    assert_equal(classes.tolist(), [expected_classes[seq_idx]])

  with assert_raises(AssertionError):  # assert that there are batches left should fail
    feed_dict, meta = data_provider.get_feed_dict(single_threaded=True)


def test_DatasetDataProvider():
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  num_seqs = 5
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=num_seqs, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)

  n_batch = 2
  config = Config({
    "max_seqs": n_batch,
  })

  with make_scope() as session:
    extern_data = ExternData()
    extern_data.init_from_dataset(dataset, auto_create_placeholders=False)

    from returnn.tf.data_pipeline import DatasetDataProvider
    data_provider = DatasetDataProvider(
      extern_data=extern_data, config=config, datasets={"train": dataset})

    input_context = data_provider.contexts["train"]
    assert isinstance(input_context.final_dataset, tf.data.Dataset)
    assert input_context.final_dataset_init_iterator_op.graph is session.graph

    data_provider.set_current_dataset(dataset_name="train")
    data_provider.start_threads(session=session)

    data, data_size, classes, classes_size = session.run([
      extern_data.data["data"].placeholder,
      extern_data.data["data"].get_sequence_lengths(),
      extern_data.data["classes"].placeholder,
      extern_data.data["classes"].get_sequence_lengths()])

    assert_is_instance(data, numpy.ndarray)
    assert_is_instance(data_size, numpy.ndarray)
    assert_is_instance(classes, numpy.ndarray)
    assert_is_instance(classes_size, numpy.ndarray)
    assert_equal(data.shape, (n_batch, seq_len, n_data_dim))
    assert_equal(data_size.shape, (n_batch,))
    assert_equal(classes.shape, (n_batch, seq_len))
    assert_equal(classes_size.shape, (n_batch,))
    assert_equal(list(data_size), [seq_len] * n_batch)
    assert_equal(list(classes_size), [seq_len] * n_batch)
    numpy.testing.assert_almost_equal(list(data[0, 0]), [-0.5, -0.4])
    numpy.testing.assert_almost_equal(list(data[0, -1]), [0.3, 0.4])
    assert_equal(classes[0].tolist(), [1, 2, 0, 1, 2])

    step = 1  # step 0 was above
    while True:
      try:
        res = session.run(data_provider.iterator_next_element)
      except tf.errors.OutOfRangeError as exc:
        print("Got out-of-range (as expected):", exc.message)
        break
      print("step %i, res %r" % (step, res))
      step += 1
      if step > 10 * num_seqs:
        break  # should not get here...

    print("Finished after %i steps." % step)
    assert step == (num_seqs - 1) // n_batch + 1

    data_provider.stop_threads()


def test_engine_train(additional_config=None):
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {"output": {"class": "softmax", "loss": "ce", "from": "data:data"}},
    "start_epoch": 1,
    "num_epochs": 2
  })
  if additional_config:
    config.update(additional_config)
  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()

  engine.finalize()


def test_engine_train_newbob():
  additional_config = {
    "num_epochs": 5,
    "learning_rate_control": "newbob",
  }
  test_engine_train(additional_config)


def test_engine_train_optimizer_class():
  from returnn.tf.updater import NormalizedSGD
  additional_config = {
    "optimizer": {"class": NormalizedSGD},
  }
  test_engine_train(additional_config)


def test_engine_train_nadam_optimizer():
  additional_config = {
    "optimizer": {"class": "nadam"},
  }
  test_engine_train(additional_config)


@unittest.skipIf(tf.__version__.startswith("1."), "TF 1")
def test_engine_train_keras_optimizer():
  additional_config = {
    "optimizer": {"class": "NadamKeras"},
  }
  test_engine_train(additional_config)


def test_engine_train_new_dataset_pipeline():
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=5, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=3, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {"output": {"class": "softmax", "loss": "ce", "from": "data:data"}},
    "start_epoch": 1,
    "num_epochs": 2,
    "max_seqs": 2,
    "dataset_pipeline": True
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  assert engine.dataset_provider
  engine.train()
  engine.finalize()


def test_engine_train_uneven_batches():
  rnd = numpy.random.RandomState(42)
  from returnn.datasets.generating import StaticDataset
  n_data_dim = 2
  n_classes_dim = 3

  def get_data(num_seqs):
    return [
      {
        "data": rnd.uniform(-1., 1., (seq_len, n_data_dim)).astype("float32"),
        "classes": rnd.choice(range(n_classes_dim), (seq_len,)).astype("int32")
      }
      for seq_len in [rnd.choice(list(range(1, 50)) + list(range(1, 20))) for _ in range(num_seqs)]]

  train_data = StaticDataset(
    input_dim=n_data_dim, output_dim=n_classes_dim,
    data=get_data(20))
  print("train data seq lens:", [len(d["data"]) for d in train_data.data])
  train_data.init_seq_order(epoch=1)
  cv_data = StaticDataset(input_dim=n_data_dim, output_dim=n_classes_dim, data=get_data(3))
  print("cv data seq lens:", [len(d["data"]) for d in cv_data.data])
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {
      "rnn": {"class": "rec", "unit": "lstm", "n_out": 3, "from": "data:data"},  # make it recurrent
      "output": {"class": "softmax", "loss": "ce", "from": "rnn"}},
    "start_epoch": 1,
    "num_epochs": 2,
    "batch_size": 50,  # set it such that sometimes we have num-seqs 1, 2 or 3 in a single batch
    "optimizer": {"class": "adam"},
    "learning_rate": 0.001,
    "tf_log_memory_usage": True,
    "log_batch_size": True
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()

  engine.finalize()


def test_engine_train_dummy_distributed():
  import returnn.tf.distributed
  rnd = numpy.random.RandomState(42)
  from returnn.datasets.generating import StaticDataset
  n_data_dim = 2
  n_classes_dim = 3

  def get_data(num_seqs):
    return [
      {
        "data": rnd.uniform(-1., 1., (seq_len, n_data_dim)).astype("float32"),
        "classes": rnd.choice(range(n_classes_dim), (seq_len,)).astype("int32")
      }
      for seq_len in [rnd.choice(list(range(1, 50)) + list(range(1, 20))) for _ in range(num_seqs)]]

  train_data = StaticDataset(
    input_dim=n_data_dim, output_dim=n_classes_dim,
    data=get_data(20))
  print("train data seq lens:", [len(d["data"]) for d in train_data.data])
  train_data.init_seq_order(epoch=1)
  cv_data = StaticDataset(input_dim=n_data_dim, output_dim=n_classes_dim, data=get_data(3))
  print("cv data seq lens:", [len(d["data"]) for d in cv_data.data])
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {
      "rnn": {"class": "rec", "unit": "lstm", "n_out": 3, "from": "data:data"},  # make it recurrent
      "output": {"class": "softmax", "loss": "ce", "from": "rnn"}},
    "start_epoch": 1,
    "num_epochs": 2,
    "batch_size": 50,  # set it such that sometimes we have num-seqs 1, 2 or 3 in a single batch
    "optimizer": {"class": "adam"},
    "learning_rate": 0.001,
    "tf_log_memory_usage": True,
    "log_batch_size": True,
    "distributed_tf": {"local_only": True},
  })
  _cleanup_old_models(config)

  with returnn.tf.distributed._temporary_init_distributed_tf(config=config):
    assert returnn.tf.distributed.is_enabled()
    engine = Engine(config=config)
    engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
    tf_session = engine.tf_session
    assert isinstance(tf_session, tf.compat.v1.Session)
    print("Session uses target:", tf_session.sess_str)
    assert tf_session.sess_str == returnn.tf.distributed.get_session_target().encode("utf8")
    engine.train()
    engine.finalize()


def test_engine_train_subnet_loss():
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {
      "output": {
        "class": "subnetwork",
        "from": "data:data",
        "subnetwork": {
          "output": {"class": "softmax", "loss": "ce", "from": "data"}
        }}},
    "start_epoch": 1,
    "num_epochs": 1,
    "batch_size": 50
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()
  engine.finalize()


def test_engine_train_rec_subnet_loss_optimized():
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {
      "output": {
        "class": "rec",
        "target": "classes",
        "from": "data:data",
        "unit": {
          "output": {"class": "softmax", "loss": "ce", "from": "data:source"}
        }}},
    "start_epoch": 1,
    "num_epochs": 1,
    "batch_size": 50
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()
  engine.finalize()


def test_engine_train_rec_subnet_loss_non_optimized():
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {
      "output": {
        "class": "rec",
        "optimize_move_layers_out": False,
        "target": "classes",
        "from": "data:data",
        "unit": {
          "output": {"class": "softmax", "loss": "ce", "from": "data:source"}
        }}},
    "start_epoch": 1,
    "num_epochs": 1,
    "batch_size": 50
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()
  engine.finalize()


def test_engine_train_accum_grad_multiple_step():
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=7, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {"output": {"class": "softmax", "loss": "ce", "from": "data:data"}},
    "start_epoch": 1,
    "num_epochs": 2,
    "accum_grad_multiple_step": 3,
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()
  engine.finalize()


def test_engine_train_accum_grad_multiple_step_sparse():
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=7, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {"output": {"class": "softmax", "loss": "ce", "from": ["data:classes"]}},
    "start_epoch": 1,
    "num_epochs": 2,
    "accum_grad_multiple_step": 3,
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()
  engine.finalize()


def test_engine_train_grad_noise_sparse():
  # Not sure how to test for it in a simple way...
  # You might see "Converting sparse IndexedSlices to a dense Tensor of unknown shape."
  # but that is normally related to sth else.

  # Anyway, for now, just try to trigger relevant code,
  # ie. in add_scaled_noise_to_gradients(),
  # and don't really check whether it works.

  from returnn.datasets.generating import Task12AXDataset
  train_data = Task12AXDataset(num_seqs=5)
  cv_data = Task12AXDataset(num_seqs=2)
  n_data_dim = train_data.num_outputs["data"][0]
  n_classes_dim = train_data.num_outputs["classes"][0]

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
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
    "optimizer": {"class": "nadam"},
    "gradient_noise": 0.3,
    "batch_size": 100
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()
  engine.finalize()


def test_engine_analyze():
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {"output": {"class": "softmax", "loss": "ce", "from": "data:data"}},
    "sil_label_idx": 0,
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  # Normally init_network_from_config but that requires an existing network model.
  # engine.init_network_from_config(config=config)
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=None, eval_data=None)

  engine.analyze(data=dataset, statistics=None)

  engine.finalize()


def test_engine_forward_single():
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {"output": {"class": "softmax", "loss": "ce", "from": "data:data"}}
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=None, eval_data=None)

  engine.forward_single(dataset=dataset, seq_idx=0)

  engine.finalize()


def test_engine_forward_to_hdf():
  from returnn.datasets.generating import DummyDataset
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
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {"output": {"class": "softmax", "loss": "ce", "from": "data:data"}},
    "output_file": output_file,
  })
  _cleanup_old_models(config)

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

  from returnn.datasets.hdf import HDFDataset
  ds = HDFDataset()
  ds.add_file(output_file)

  assert_equal(ds.num_inputs, n_classes_dim) # forwarded input is network output
  assert_equal(ds.get_num_timesteps(), seq_len*num_seqs)
  assert_equal(ds.num_seqs, num_seqs)

  os.remove(output_file)


def test_engine_rec_subnet_count():
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  # The dataset is actually not used.
  n_data_dim = 2
  n_classes_dim = 3
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
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
  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=None, eval_data=None)

  out = engine.forward_single(dataset=dataset, seq_idx=0)
  assert_equal(out.shape, (seq_len,))
  assert_equal(out.dtype, numpy.int32)
  assert_equal(list(out[:]), list(range(1, seq_len + 1)))

  engine.finalize()


def test_engine_end_layer(extra_rec_kwargs=None):
  """
  :param dict[str] extra_rec_kwargs:
  """
  from returnn.util.basic import dict_joined
  from returnn.datasets.generating import DummyDataset
  from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell
  seq_len = 5
  n_data_dim = 1
  n_classes_dim = 5
  dataset = DummyDataset(input_dim=n_data_dim,
                         output_dim=n_classes_dim,
                         num_seqs=2,
                         seq_len=seq_len)

  dataset.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "batch_size": 5000,
    "extern_data": {
      "data": {"dim": n_data_dim, "available_for_inference": False},
      "classes": {"dim": n_classes_dim, "sparse": True, "available_for_inference": True}},
    "target": "classes",
    "network": {
      "enc0": {"class": "linear", "activation": "sigmoid", "n_out": 3, "from": "data:classes"},
      "enc1": {"class": "reduce", "mode": "max", "axis": "t", "from": "enc0"},
      "output": dict_joined({
        "class": "rec", "from": [], "max_seq_len": 10, "target": "classes",
        "unit": {
          "output": {"class": "linear", "activation": "tanh", "n_out": n_classes_dim,
                     "from": ["prev:output", "base:enc1"]},
          'stop_token': {'class': 'linear', 'activation': None, 'n_out': 1, 'loss': 'bin_ce', 'loss_scale': 1.0,
                         'target': 'data', 'from': 'output'},
          'stop_token_sigmoid': {'class': 'activation', 'activation': 'sigmoid', 'from': 'stop_token'},
          'end_compare': {'class': 'compare', 'kind': 'greater', 'from': 'stop_token_sigmoid', 'value': 0.5},
          'end': {'class': 'squeeze', 'from': 'end_compare', 'axis': 'F'},
        }
      }, extra_rec_kwargs or {}),
    }
  })
  engine = Engine(config=config)
  # Normally init_network can be used. We only do init_train here to randomly initialize the network.
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=None, eval_data=None)
  print("network:")
  pprint(engine.network.layers)
  assert "output" in engine.network.layers

  rec_layer = engine.network.layers["output"]
  assert isinstance(rec_layer, RecLayer)
  assert isinstance(rec_layer.cell, _SubnetworkRecCell)
  assert_equal(set(rec_layer.cell.input_layers_moved_out), set())
  assert_equal(set(rec_layer.cell.output_layers_moved_out), {"stop_token"})
  assert_equal(set(rec_layer.cell.layers_in_loop), {"output"})

  # Now reinit for search.
  assert not engine.use_search_flag
  engine.use_search_flag = True
  engine.use_dynamic_train_flag = False
  print("Reinit network with search flag.")
  engine.init_network_from_config(config=config)

  engine.search(dataset=dataset)
  print("error keys:")
  pprint(engine.network.losses_dict)
  assert engine.network.total_objective is not None

  engine.use_search_flag = False
  print("Reinit network without search flag.")
  engine.init_network_from_config(config=config)
  hdf_fn = _get_tmp_file(suffix=".hdf")
  os.remove(hdf_fn)  # forward_to_hdf expects that the file does not exist
  engine.forward_to_hdf(data=dataset, output_file=hdf_fn)

  engine.finalize()


def check_engine_search(extra_rec_kwargs=None):
  """
  :param dict[str] extra_rec_kwargs:
  """
  from returnn.util.basic import dict_joined
  from returnn.datasets.generating import DummyDataset
  from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 7
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "batch_size": 5000,
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {
      "enc0": {"class": "linear", "activation": "sigmoid", "n_out": 3, "from": "data:data"},
      "enc1": {"class": "reduce", "mode": "max", "axis": "t", "from": "enc0"},
      "output": dict_joined({
        "class": "rec", "from": [], "max_seq_len": 10, "target": "classes",
        "unit": {
          "embed": {"class": "linear", "from": "prev:output", "activation": "sigmoid", "n_out": 3},
          "prob": {"class": "softmax", "from": ["embed", "base:enc1"], "loss": "ce", "target": "classes"},
          "output": {"class": "choice", "beam_size": 4, "from": "prob", "target": "classes", "initial_output": 0},
          "end": {"class": "compare", "from": "output", "value": 0}
        }
      }, extra_rec_kwargs or {}),
      "decision": {"class": "decide", "from": "output", "loss": "edit_distance"}
    }
  })
  _cleanup_old_models(config)
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
    assert_equal(set(rec_layer.cell.input_layers_moved_out), set())
    assert_equal(set(rec_layer.cell.output_layers_moved_out), {"output", "embed", "prob"})
    assert_equal(set(rec_layer.cell.layers_in_loop), set())
  else:
    assert_equal(
      set(rec_layer.cell.layers_in_loop).difference({"data:classes"}),
      {"embed", "prob", "output", "end"})

  # Now reinit for search.
  assert not engine.use_search_flag
  engine.use_search_flag = True
  engine.use_dynamic_train_flag = False
  print("Reinit network with search flag.")
  engine.init_network_from_config(config=config)

  engine.search(dataset=dataset)
  print("error keys:")
  pprint(engine.network.losses_dict)
  assert engine.network.total_objective is not None
  assert "decision" in engine.network.losses_dict

  engine.finalize()


def test_engine_search_no_optim():
  check_engine_search({"optimize_move_layers_out": False})


def test_engine_search():
  check_engine_search()


def check_engine_search_attention(extra_rec_kwargs=None):
  """
  :param dict[str] extra_rec_kwargs:
  """
  from returnn.util.basic import dict_joined
  from returnn.datasets.generating import DummyDataset
  from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 7
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)
  print("Hello search!")

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "batch_size": 5000,
    "max_seqs": 2,
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {
      "encoder": {"class": "linear", "activation": "tanh", "n_out": 5, "from": "data:data"},
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
  _cleanup_old_models(config)
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
  pprint(engine.network.losses_dict)
  assert engine.network.total_objective is not None
  assert "decision" in engine.network.losses_dict

  engine.finalize()


def test_engine_search_attention_no_optim():
  check_engine_search_attention({"optimize_move_layers_out": False})


def test_engine_search_attention():
  check_engine_search_attention()


def run_dummy_training(net_dict):
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "batch_size": 100,
    "max_seqs": 2,
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": net_dict,
    "start_epoch": 1,
    "num_epochs": 2,
    "learning_rate": 0.01,
    "optimizer": {"class": "nadam"},
    "gradient_noise": 0.3,
    "debug_add_check_numerics_ops": True,
    "debug_print_layer_output_template": True,
    "debug_print_layer_output_shape": True,
    "debug_add_check_numerics_on_output": True,
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=dataset, eval_data=None)
  print("Extern data:")
  pprint(engine.network.extern_data.data)
  print("Used data keys:")
  pprint(engine.network.used_data_keys)
  engine.train()
  engine.finalize()


def check_engine_train_simple_attention(lstm_unit):
  net_dict = {
    "lstm0_fw": {"class": "rec", "unit": lstm_unit, "n_out": 20, "dropout": 0.0, "L2": 0.01, "direction": 1,
                 "from": "data:data"},
    "lstm0_bw": {"class": "rec", "unit": lstm_unit, "n_out": 20, "dropout": 0.0, "L2": 0.01, "direction": -1,
                 "from": "data:data"},

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
  run_dummy_training(net_dict)


# @unittest.skip("crash on OSX? https://github.com/tensorflow/tensorflow/issues/14285")
def test_engine_train_simple_attention_lstmp():
  check_engine_train_simple_attention(lstm_unit="lstmp")


def test_engine_train_simple_attention_nativelstm2():
  check_engine_train_simple_attention(lstm_unit="nativelstm2")


def test_engine_train_simple_attention_basiclstm():
  check_engine_train_simple_attention(lstm_unit="basiclstm")


def test_attention_train_then_search():
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 7
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  dev_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dev_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "batch_size": 5000,
    "max_seqs": 2,
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "num_epochs": 1,
    "network": {
      "encoder": {"class": "linear", "activation": "tanh", "n_out": 5, "from": "data:data"},
      "output": {
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
      },
      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance"}
    },
    "debug_print_layer_output_template": True,
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  print("Train...")
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=dev_data)
  engine.train()

  print("Search...")
  engine.use_search_flag = True
  engine.use_dynamic_train_flag = False
  engine.init_network_from_config(config)
  engine.search(dataset=dev_data)
  print("error keys:")
  pprint(engine.network.losses_dict)
  assert engine.network.total_objective is not None
  assert "decision" in engine.network.losses_dict

  engine.finalize()


def test_attention_subnetwork_base_dependency():
  net_dict = {
    "encoder": {"class": "linear", "activation": "tanh", "n_out": 14, "from": "data:data"},
    "decoder": {
      "class": "rec",
      "from": [],
      "target": "classes", "max_seq_len": 10,
      "unit": {
        'output': {'class': 'choice', 'target': 'classes', 'beam_size': 4, 'from': "output_prob"},
        "end": {"class": "compare", "from": "output", "value": 0},
        'orth_embed': {'class': 'linear', 'activation': None, 'from': 'output', "n_out": 7},
        "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:att", "prev:orth_embed"], "n_out": 7},
        "att": {
          "class": "subnetwork",
          "from": ["s", "prev:orth_embed"],
          "concat_sources": True,
          "subnetwork": {
            "c_in": {"class": "linear", "activation": None, "from": "data", "n_out": 14},
            "output": {
              "class": "dot_attention", "from": "c_in",
              "base": "base:base:encoder", "base_ctx": "base:base:encoder"},
          }
        },
        "output_prob": {"class": "softmax", "from": ["prev:s", "att"], "target": "classes", "loss": "ce"}
      },
    },
    "output": {'class': "copy", 'from': 'decoder'}
  }
  run_dummy_training(net_dict)


def test_attention_subnetwork_from_dependency():
  net_dict = {
    "encoder": {"class": "linear", "activation": "tanh", "n_out": 14, "from": "data:data"},
    "decoder": {
      "class": "rec",
      "from": [],
      "target": "classes", "max_seq_len": 10,
      "unit": {
        'output': {'class': 'choice', 'target': 'classes', 'beam_size': 4, 'from': "output_prob"},
        "end": {"class": "compare", "from": "output", "value": 0},
        'orth_embed': {'class': 'linear', 'activation': None, 'from': 'output', "n_out": 7},
        "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:att", "prev:orth_embed"], "n_out": 7},
        "att": {
          "class": "subnetwork",
          "from": ["s", "prev:orth_embed", "base:encoder"],
          "concat_sources": False,
          "subnetwork": {
            "c_in": {"class": "linear", "activation": None, "from": ["data:0", "data:1"], "n_out": 14},
            "output": {"class": "dot_attention", "from": ["c_in"], "base": "data:2", "base_ctx": "data:2"},
          }
        },
        "output_prob": {"class": "softmax", "from": ["prev:s", "att"], "target": "classes", "loss": "ce"}
      },
    },
    "output": {'class': "copy", 'from': 'decoder'}
  }
  run_dummy_training(net_dict)


def test_attention_no_encoder_dependency():
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 7
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  dev_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dev_data.init_seq_order(epoch=1)

  from returnn.tf.util.basic import Dim
  enc_time = SpatialDim("enc time")

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "batch_size": 5000,
    "max_seqs": 2,
    "extern_data": {'data': {'dim': n_data_dim, 'same_dim_tags_as': {'T': enc_time}},
                    'classes': {'dim': n_classes_dim, 'sparse': True}},
    "num_epochs": 1,
    "network": {
      "encoder": {"class": "linear", "activation": "tanh", "n_out": 5, 'out_type': {}, "from": "data:data"},
      "enc_transformed": {'class': 'linear', 'from': ['encoder'], 'n_out': 1, 'activation': None},
      "zero_enc_transformed": {'class': 'eval', 'from': ['enc_transformed'], 'eval': 'tf.zeros_like(source(0))'},
      "output": {
        "class": "rec",
        "from": [],
        "target": "classes", "max_seq_len": 5,
        "unit": {
          'output': {'class': 'choice', 'target': 'classes', 'beam_size': 4, 'from': ["output_prob"]},
          "end": {"class": "compare", "from": ["output"], "value": 0},
          'orth_embed': {'class': 'linear', 'activation': None, 'from': ['output'], "n_out": 7},
          "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:c", "prev:orth_embed"], "n_out": 7},
          "output_prob": {"class": "softmax", "from": ["prev:s", "c"], "target": "classes", "loss": "ce"},
          # basic attention
          "s_transformed": {'class': 'linear', 'from': ['s'], 'n_out': 6, 'activation': None},
          "att_energy_tanh": {'class': 'activation', 'from': ['att_energy_in'], 'activation': 'tanh'},
          "att_energy": {'class': 'linear', 'from': ['att_energy_tanh'], 'n_out': 1, 'activation': None,
                         'out_type': {
                           'same_dim_tags_as': {'T': enc_time}},
                         },
          "att_weights": {'class': 'softmax_over_spatial', 'from': ['att_energy']},
          # feedback
          "accum_att_weights": {'class': 'combine', 'kind': 'add',
                                'from': ['att_weights', 'prev:accum_att_weights'],
                                'initial_output': "base:zero_enc_transformed",
                                },
          "convolved_att": {'class': 'conv', 'from': ['prev:accum_att_weights'], 'filter_size': (3,),
                            'n_out': 4, 'padding': 'same'},
          "location_feedback": {'class': 'linear', 'from': ['convolved_att'], 'n_out': 6, 'activation': None},
          "att_energy_in": {'class': 'combine', 'kind': 'add', 'from': ['location_feedback', 's_transformed']},
          "c": {"class": "generic_attention", "base": "base:encoder", "weights": "att_weights", "auto_squeeze": True},
        },
      },
      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance"}
    },
    "debug_print_layer_output_template": True,
    "debug_runtime_sanity_checks": True,
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  print("Train...")
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=dev_data)
  engine.train()

  print("Search...")
  engine.use_search_flag = True
  engine.use_dynamic_train_flag = False
  engine.init_network_from_config(config)
  engine.search(dataset=dev_data)
  print("error keys:")
  pprint(engine.network.losses_dict)
  assert engine.network.total_objective is not None
  assert "decision" in engine.network.losses_dict

  engine.finalize()


def check_attention_variant(recurrent_unit_dict):
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 7
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  dev_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dev_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "batch_size": 5000,
    "max_seqs": 2,
    "extern_data": {'data': {'dim': n_data_dim},
                    'classes': {'dim': n_classes_dim, 'sparse': True}},
    "num_epochs": 1,
    "network": {
      "encoder": {"class": "linear", "activation": "tanh", "n_out": 5, "from": "data:data"},
      "enc_transformed": {'class': 'linear', 'from': ['encoder'], 'n_out': 6, 'activation': None},
      "output": {
        "class": "rec",
        "from": [],
        "target": "classes", "max_seq_len": 5,
        "unit": recurrent_unit_dict,
      },
      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance"}
    },
    "debug_print_layer_output_template": True,
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  print("Train...")
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=dev_data)
  engine.train()

  print("Search...")
  engine.use_search_flag = True
  engine.use_dynamic_train_flag = False
  engine.init_network_from_config(config)
  engine.search(dataset=dev_data)
  print("error keys:")
  pprint(engine.network.losses_dict)
  assert engine.network.total_objective is not None
  assert "decision" in engine.network.losses_dict

  engine.finalize()


def test_attention_convolutional_feedback_variant1():
  recurrent_unit_dict = {
    'output': {'class': 'choice', 'target': 'classes', 'beam_size': 4, 'from': ["output_prob"]},
    "end": {"class": "compare", "from": ["output"], "value": 0},
    'orth_embed': {'class': 'linear', 'activation': None, 'from': ['output'], "n_out": 7},
    "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:c", "prev:orth_embed"], "n_out": 7},
    "output_prob": {"class": "softmax", "from": ["prev:s", "c"], "target": "classes", "loss": "ce"},
    # basic attention
    "s_transformed": {'class': 'linear', 'from': ['s'], 'n_out': 6, 'activation': None},
    "att_energy_tanh": {'class': 'activation', 'from': ['att_energy_in'], 'activation': 'tanh'},
    "att_energy": {'class': 'linear', 'from': ['att_energy_tanh'], 'n_out': 1, 'activation': None},
    "att_weights": {'class': 'softmax_over_spatial', 'from': ['att_energy']},
    'accum_att_weights': {'class': 'eval',
                          'eval': 'source(0) + source(1)',
                          'from': ['prev:accum_att_weights', 'att_weights'],
                          'is_output_layer': True,
                          'out_type': {'dim': 1, 'shape': (None, 1)}},
    'feedback_pad_left': {'axes': 'stag:extern_data:data',
                          'class': 'pad',
                          'from': ['prev:accum_att_weights'],
                          'mode': 'constant',
                          'padding': ((2, 0),),
                          'value': 1},
    'feedback_pad_right': {'axes': 'stag:extern_data:data',
                           'class': 'pad',
                           'from': ['feedback_pad_left'],
                           'mode': 'constant',
                           'padding': ((0, 2),),
                           'value': 0},
    "convolved_att": {'class': 'conv', 'from': ['feedback_pad_right'], 'filter_size': (5,),
                      'n_out': 4, 'padding': 'valid'},
    "location_feedback": {'class': 'linear', 'from': ['convolved_att'], 'n_out': 6, 'activation': None},
    "att_energy_in": {'class': 'combine', 'kind': 'add', 'from': [
      'base:enc_transformed', 'location_feedback', 's_transformed']},
    "c": {"class": "generic_attention", "base": "base:encoder", "weights": "att_weights", "auto_squeeze": True},
  }

  check_attention_variant(recurrent_unit_dict)


def test_attention_convolutional_feedback_variant2():
  recurrent_unit_dict = {
    'output': {'class': 'choice', 'target': 'classes', 'beam_size': 4, 'from': ["output_prob"]},
    "end": {"class": "compare", "from": ["output"], "value": 0},
    'orth_embed': {'class': 'linear', 'activation': None, 'from': ['output'], "n_out": 7},
    "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:c", "prev:orth_embed"], "n_out": 7},
    "output_prob": {"class": "softmax", "from": ["prev:s", "c"], "target": "classes", "loss": "ce"},
    # basic attention
    "s_transformed": {'class': 'linear', 'from': ['s'], 'n_out': 6, 'activation': None},
    "att_energy_tanh": {'class': 'activation', 'from': ['att_energy_in'], 'activation': 'tanh'},
    "att_energy": {'class': 'linear', 'from': ['att_energy_tanh'], 'n_out': 1, 'activation': None,
                   },
    "att_weights": {'class': 'softmax_over_spatial', 'from': ['att_energy']},
    # feedback
    "accum_att_weights": {'class': 'combine', 'kind': 'add',
                          'from': ['att_weights', 'prev:accum_att_weights'],
                          },
    "convolved_att": {'class': 'conv', 'from': ['prev:accum_att_weights'], 'filter_size': (5,),
                      'n_out': 4, 'padding': 'same'},
    "location_feedback": {'class': 'linear', 'from': ['convolved_att'], 'n_out': 6, 'activation': None},
    "att_energy_in": {'class': 'combine', 'kind': 'add', 'from': [
      'base:enc_transformed', 'location_feedback', 's_transformed']},
    "c": {"class": "generic_attention", "base": "base:encoder", "weights": "att_weights", "auto_squeeze": True},
  }

  check_attention_variant(recurrent_unit_dict)


def test_attention_convolutional_feedback_variant3():
  recurrent_unit_dict = {
    'output': {'class': 'choice', 'target': 'classes', 'beam_size': 4, 'from': ["output_prob"]},
    "end": {"class": "compare", "from": ["output"], "value": 0},
    'orth_embed': {'class': 'linear', 'activation': None, 'from': ['output'], "n_out": 7},
    "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:c", "prev:orth_embed"], "n_out": 7},
    "output_prob": {"class": "softmax", "from": ["prev:s", "c"], "target": "classes", "loss": "ce"},
    # basic attention
    "s_transformed": {'class': 'linear', 'from': ['s'], 'n_out': 6, 'activation': None},
    "att_energy_tanh": {'class': 'activation', 'from': ['att_energy_in'], 'activation': 'tanh'},
    "att_energy": {'class': 'linear', 'from': ['att_energy_tanh'], 'n_out': 1, 'activation': None,
                   },
    "att_weights": {'class': 'softmax_over_spatial', 'from': ['att_energy']},
    "accum_att_weights": {'class': 'combine', 'kind': 'add',
                          'from': ['att_weights', 'prev:accum_att_weights'],
                          },
    'feedback_pad_left': {'axes': 'stag:extern_data:data',
                          'class': 'pad',
                          'from': ['prev:accum_att_weights'],
                          'mode': 'constant',
                          'padding': ((2, 0),),
                          'value': 1},
    'feedback_pad_right': {'axes': 'stag:extern_data:data',
                           'class': 'pad',
                           'from': ['feedback_pad_left'],
                           'mode': 'constant',
                           'padding': ((0, 2),),
                           'value': 0},
    "convolved_att": {'class': 'conv', 'from': ['feedback_pad_right'], 'filter_size': (5,),
                      'n_out': 4, 'padding': 'valid'},
    "location_feedback": {'class': 'linear', 'from': ['convolved_att'], 'n_out': 6, 'activation': None},
    "att_energy_in": {'class': 'combine', 'kind': 'add', 'from': [
      'base:enc_transformed', 'location_feedback', 's_transformed']},
    "c": {"class": "generic_attention", "base": "base:encoder", "weights": "att_weights", "auto_squeeze": True},
  }

  check_attention_variant(recurrent_unit_dict)


def test_attention_search_in_train_then_search():
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 7
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  dev_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dev_data.init_seq_order(epoch=1)

  def make_net_dict(task):
    """
    :param str task:
    :rtype: dict[str,dict[str]]
    """
    return {
      "encoder": {"class": "linear", "activation": "tanh", "n_out": 5, "from": "data:data"},
      "output": {
        "class": "rec",
        "from": [],
        'only_on_search': True,
        "target": "classes",
        "max_seq_len": "max_len_from('base:encoder')",
        "unit": {
          'output': {'class': 'choice', 'target': 'classes', 'beam_size': 4, 'from': ["output_prob"]},
          "end": {"class": "compare", "from": ["output"], "value": 0},
          'orth_embed': {'class': 'linear', 'activation': None, 'from': ['output'], "n_out": 7},
          "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:c", "prev:orth_embed"], "n_out": 7},
          "c_in": {"class": "linear", "activation": "tanh", "from": ["s", "prev:orth_embed"], "n_out": 5},
          "c": {"class": "dot_attention", "from": ["c_in"], "base": "base:encoder", "base_ctx": "base:encoder"},
          "output_prob": {
            "class": "softmax", "from": ["prev:s", "c"], "dropout": 0.3,
            "target": "layer:opt_completion_soft_targets" if task == "train" else "classes", "loss": "ce"},

          "edit_dist_table": {"class": "edit_distance_table", "from": "output", "target": "layer:base:data:classes"},
          "opt_completions": {"class": "optimal_completions", "from": "prev:edit_dist_table",
                              "target": "layer:base:data:classes"},
          "opt_completion_soft_targets": {
            "class": "eval", "eval": "tf.nn.softmax(-20. * tf.cast(source(0), tf.float32))",
            "from": "opt_completions", "out_type": {"dtype": "float32"}}
        }},

      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance", 'only_on_search': True}
    }

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "batch_size": 5000,
    "max_seqs": 2,
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "num_epochs": 1,
    "network": make_net_dict(task="train"),
    "search_train_network_layers": ["output", "decision"],
    "search_output_layer": "decision",
    "debug_print_layer_output_template": True
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  print("Train...")
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=dev_data)
  engine.train()

  print("Search...")
  config.set("network", make_net_dict(task="search"))
  engine.use_search_flag = True
  engine.use_dynamic_train_flag = False
  engine.init_network_from_config(config)
  engine.search(dataset=dev_data)
  print("error keys:")
  pprint(engine.network.losses_dict)
  assert engine.network.total_objective is not None
  assert "decision" in engine.network.losses_dict

  engine.finalize()


def check_train_and_search_two_targets(net_dict):
  """
  Tests training and search for network architectures having two targets ("classes_0", "classes_1")
  and two corresponding output layers ("decision_0", "decision_1").
  """
  from returnn.datasets.meta import MetaDataset
  from returnn.tf.util.basic import Dim
  from test_HDFDataset import generate_hdf_from_other

  n_data_dim = 2
  n_classes_dim_0 = 7
  n_classes_dim_1 = 8

  data_0 = {"class": "DummyDataset", "input_dim": n_data_dim, "output_dim": n_classes_dim_0,
    "num_seqs": 2, "seq_len": 5}
  data_0 = generate_hdf_from_other(data_0)
  data_1 = {"class": "DummyDataset", "input_dim": n_data_dim, "output_dim": n_classes_dim_1,
    "num_seqs": 2, "seq_len": 5}
  data_1 = generate_hdf_from_other(data_1)

  data = MetaDataset(datasets={"data_0": data_0, "data_1": data_1},
    data_map={
      "data": ("data_1", "data"),
      "classes_0": ("data_0", "classes"),
      "data_1": ("data_1", "data"),
      "classes_1": ("data_1", "classes")},
  )
  data.init_seq_order()

  dec_time = SpatialDim("dec time")

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "batch_size": 5000,
    "max_seqs": 2,
    "extern_data": {"data": {"dim": n_data_dim, "sparse": False},
      "classes_0": {"dim": n_classes_dim_0, "sparse": True, "same_dim_tags_as": {"t": dec_time}},
      "classes_1": {"dim": n_classes_dim_1, "sparse": True, "same_dim_tags_as": {"t": dec_time}},
    },
    "num_epochs": 1,
    "network": net_dict,
    "debug_print_layer_output_template": True,
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  print("Train...")
  engine.init_train_from_config(config=config, train_data=data, dev_data=None)
  engine.train()

  print("Search...")
  engine.use_search_flag = True
  engine.use_dynamic_train_flag = False
  engine.init_network_from_config(config)
  engine.search(dataset=data, output_layer_names=["decision_0", "decision_1"])
  assert engine.network.total_objective is not None
  assert "decision_0" in engine.network.losses_dict
  assert "decision_1" in engine.network.losses_dict

  engine.finalize()


def test_attention_two_targets():
  """
  Tests training and search when using a ChoiceLayer with two targets.
  """
  net_dict = {
    "encoder": {"class": "linear", "activation": "tanh", "n_out": 5, "from": "data:data"},
    "output": {
      "class": "rec",
      "from": [],
      "target": "classes_1", "max_seq_len": 10,
      "unit": {
        "end": {"class": "compare", "from": ["output_0"], "value": 0},
        "orth_embed_0": {'class': 'linear', 'activation': None, 'from': ['output_0'], "n_out": 7},
        "orth_embed_1": {'class': 'linear', 'activation': None, 'from': ['output_1'], "n_out": 7},
        "orth_embed": {"class": "copy", "from": ["orth_embed_0", "orth_embed_1"]},
        "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:c", "prev:orth_embed"], "n_out": 7},
        "c_in": {"class": "linear", "activation": "tanh", "from": ["s", "prev:orth_embed"], "n_out": 5},
        "c": {"class": "dot_attention", "from": ["c_in"], "base": "base:encoder", "base_ctx": "base:encoder"},
        "output_prob_0": {"class": "softmax", "from": ["prev:s", "c"], "target": "classes_0", "loss": "ce"},
        "output_prob_1": {"class": "softmax", "from": ["prev:s", "c"], "target": "classes_1", "loss": "ce"},
        "output": {'class': 'choice', 'target': ['classes_0', "classes_1"], 'beam_size': 4,
          'from': ["output_prob_0", "output_prob_1"], "source_beam_sizes": [2, 6]},

        "output_0": {"class": "copy", "from": ["output/out_0"], "is_output_layer": True},
        "output_1": {"class": "copy", "from": ["output/out_1"], "is_output_layer": True},
      },
    },
    "output_0": {"class": "copy", "from": ["output/output_0"], "target": "classes_0"},
    "output_1": {"class": "copy", "from": ["output/output_1"], "target": "classes_1"},

    "decision_0": {"class": "decide", "from": ["output_0"], "loss": "edit_distance", "target": "classes_0"},
    "decision_1": {"class": "decide", "from": ["output_1"], "loss": "edit_distance", "target": "classes_1"},
  }

  check_train_and_search_two_targets(net_dict=net_dict)

  # Also test with label feedback from only one of the outputs. This is a special case, where during search the
  # sub-layer "output_1" == "output/out_1" could be optimized out of the loop, but the root layer is in the loop.
  net_dict["output"]["unit"]["orth_embed"]["from"] = ["orth_embed_0"]

  check_train_and_search_two_targets(net_dict=net_dict)


def test_attention_two_dependent_targets():
  """
  Tests training and search when having two ChoiceLayers in the loop that depend on each other.
  Note, there will be different beam sizes in different parts of the recurrent unit.
  """
  beam_size_0 = 5
  beam_size_1 = 3

  net_dict = {
    "encoder": {"class": "linear", "activation": "tanh", "n_out": 5, "from": "data:data"},
    "output": {
      "class": "rec",
      "from": [],
      "target": "classes_1", "max_seq_len": 10,
      "unit": {
        "end": {"class": "compare", "from": ["output_0"], "value": 0},
        "orth_embed_0": {'class': 'linear', 'activation': None, 'from': ['output_0'], "n_out": 7},
        "orth_embed_1": {'class': 'linear', 'activation': None, 'from': ['output_1'], "n_out": 7},
        "orth_embed": {"class": "copy", "from": ["orth_embed_0", "orth_embed_1"]},
        "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:c", "prev:orth_embed"], "n_out": 7},
        "c_in": {"class": "linear", "activation": "tanh", "from": ["s", "prev:orth_embed"], "n_out": 5},
        "c": {"class": "dot_attention", "from": ["c_in"], "base": "base:encoder", "base_ctx": "base:encoder"},
        "output_prob_0": {"class": "softmax", "from": ["prev:s", "c"], "target": "classes_0", "loss": "ce"},
        "output_prob_1": {"class": "softmax", "from": ["prev:s", "c", "orth_embed_0"],
          "target": "classes_1", "loss": "ce"},
        # Important for real experiments: apply length normalization only once (in last choice layer).
        "output_0": {'class': 'choice', 'target': 'classes_0', 'beam_size': beam_size_0, 'from': "output_prob_0",
          "is_output_layer": True, "length_normalization": False},
        "output_1": {'class': 'choice', 'target': 'classes_1', 'beam_size': beam_size_1, 'from': "output_prob_1",
          "is_output_layer": True},

        "output": {"class": "copy", "from": "output_1"},
      },
    },
    "output_0": {"class": "copy", "from": ["output/output_0"], "target": "classes_0"},
    "output_1": {"class": "copy", "from": ["output/output_1"], "target": "classes_1"},

    "decision_0": {"class": "decide", "from": ["output_0"], "loss": "edit_distance", "target": "classes_0"},
    "decision_1": {"class": "decide", "from": ["output_1"], "loss": "edit_distance", "target": "classes_1"},
  }

  check_train_and_search_two_targets(net_dict=net_dict)


def test_rec_optim_all_out():
  from returnn.datasets.generating import DummyDataset
  from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 7
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "batch_size": 5000,
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": {
      "enc0": {"class": "linear", "activation": "sigmoid", "n_out": 3, "from": "data:data"},
      "enc1": {"class": "reduce", "mode": "max", "axis": "t", "from": "enc0"},
      "output": {
        "class": "rec", "optimize_move_layers_out": True, "from": [], "max_seq_len": 10, "target": "classes",
        "unit": {
          "embed": {"class": "linear", "from": "prev:output", "activation": "sigmoid", "n_out": 3},
          "prob": {"class": "softmax", "from": ["embed", "base:enc1"], "loss": "ce", "target": "classes"},
          "output": {"class": "choice", "beam_size": 4, "from": ["prob"], "target": "classes", "initial_output": 0},
          "end": {"class": "compare", "from": ["output"], "value": 0}
        }
      },
      "decision": {"class": "decide", "from": ["output"], "loss": "edit_distance"}
    }
  })
  _cleanup_old_models(config)
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
  assert_equal(set(rec_layer.cell.input_layers_moved_out), set())
  assert_equal(set(rec_layer.cell.output_layers_moved_out), {"output", "prob", "embed"})
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
  assert_equal(set(rec_layer.cell.layers_in_loop), {"prob", "output", "end", "embed"})

  engine.search(dataset=dataset)
  print("error keys:")
  pprint(engine.network.losses_dict)
  assert engine.network.total_objective is not None
  assert "decision" in engine.network.losses_dict

  engine.finalize()


def test_rec_subnet_train_t3b():
  beam_size = 2
  network = {
    "data_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6, "from": "data:data"},
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

  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": network,
    "start_epoch": 1,
    "num_epochs": 2,
    "batch_size": 10,
    "optimizer": {"class": "nadam"},
    "learning_rate": 0.01,
    "debug_add_check_numerics_ops": True
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()

  engine.finalize()


def test_rec_subnet_train_t3d():
  beam_size = 2
  network = {
    "data_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6, "from": "data:data"},
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

  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": network,
    "start_epoch": 1,
    "num_epochs": 2,
    "batch_size": 10,
    "optimizer": {"class": "nadam"},
    "learning_rate": 0.01,
    "debug_add_check_numerics_ops": True
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  engine.train()


def test_rec_subnet_train_t3d_simple():
  beam_size = 2
  network = {
    "encoder": {"class": "linear", "activation": "tanh", "n_out": 5, "from": "data:data"},
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

  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": network,
    "start_epoch": 1,
    "num_epochs": 2,
    "batch_size": 10,
    "optimizer": {"class": "nadam"},
    "learning_rate": 0.01,
    "debug_add_check_numerics_ops": True
  })
  _cleanup_old_models(config)
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
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": network,
    "start_epoch": 1,
    "num_epochs": 2,
    "batch_size": 10,
    "optimizer": {"class": "adam"},
    "learning_rate": 0.01,
    "debug_add_check_numerics_ops": True
  })
  _cleanup_old_models(config)

  from returnn.datasets.generating import DummyDataset
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
    pprint(engine.learning_rate_control.epoch_data)
    score_results[run_idx] = {ep: d.error for (ep, d) in engine.learning_rate_control.epoch_data.items()}

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
  deterministic_train_check({"class": "linear", "activation": "tanh", "n_out": 5, "from": "data:data"})


def test_deterministic_train_rec_nativelstm2():
  deterministic_train_check({"class": "rec", "unit": "nativelstm2", "n_out": 5, "from": "data:data"})


def _create_deterministic_layer_checks():
  from returnn.tf.layers.basic import get_layer_class_name_list, get_layer_class
  from returnn.util.basic import collect_mandatory_class_init_kwargs
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
  from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell
  n_data_dim = 2
  n_classes_dim = 3
  from returnn.datasets.generating import DummyDataset
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
      "encoder": {
        "class": "linear", "activation": "tanh", "n_out": 5, "forward_weights_init": weights_init, "from": "data:data"},
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
      "model": "%s/model" % _get_tmp_dir(),
      "num_outputs": n_classes_dim,
      "num_inputs": n_data_dim,
      "network": network,
      "start_epoch": 1,
      "num_epochs": 2,
      "batch_size": 10,
      "optimizer": {"class": "nadam"},
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
    _cleanup_old_models(config)
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
    print("Losses:")
    pprint(engine.network.losses_dict)

    print("Run %i: Train now..." % run_idx)
    engine.train()

    print("Run %i: Train results:" % run_idx)
    pprint(engine.learning_rate_control.epoch_data)
    score_results[run_idx] = {ep: d.error for (ep, d) in engine.learning_rate_control.epoch_data.items()}

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
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=10, seq_len=seq_len)

  beam_size = 2
  net_dict = {
    "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6, "from": "data:data"},

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
      "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder", "auto_squeeze": True},
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
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=10, seq_len=seq_len)

  beam_size = 2
  net_dict = {
    "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6, "from": "data:data"},

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
      "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder", "auto_squeeze": True},
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
    from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell
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
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=10, seq_len=seq_len)

  beam_size = 2
  net_dict = {
    "source_embed": {"class": "linear", "activation": None, "with_bias": False, "n_out": 6, "from": "data:data"},

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
      "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder", "auto_squeeze": True},
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
    from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell
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
  EncKeyTotalDim = AttNumHeads * 5
  EncKeyPerHeadDim = EncKeyTotalDim // AttNumHeads
  EncValueTotalDim = AttNumHeads * 5
  EncValuePerHeadDim = EncValueTotalDim // AttNumHeads
  network = {
    "lstm0_fw": {"class": "rec", "unit": "nativelstm2", "n_out": 2, "direction": 1, "from": "data:data"},
    "lstm0_bw": {"class": "rec", "unit": "nativelstm2", "n_out": 2, "direction": -1, "from": "data:data"},
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
      "energy": {"class": "dot", "red1": "F", "red2": "F", "var1": None, "var2": None,
                 "from": ["base:enc_ctx", "att_query"], "debug": True},  # (B, enc-T, H, 1)

      "att_weights": {"class": "softmax_over_spatial", "from": ["energy"], "energy_factor": EncKeyPerHeadDim ** -0.5},
      "att_weights_avg": {"class": "reduce", "axes": "dim:%i" % AttNumHeads, "mode": "avg", "from": ["att_weights"]},  # (B, enc-T, 1)
      "accum_att_weights": {"class": "eval",
                            "from": ["prev:accum_att_weights", "att_weights_avg", "base:inv_fertility"],
                            "eval": "source(0) + source(1) * source(2) * 0.5",
                            "out_type": {"dim": 1, "shape": (None, 1)}, "initial_output": "apply(0)"},  # (B, enc-T, 1)
      "att0": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"},  # (B, H, V)
      "att": {"class": "merge_dims", "axes": ["dim:%i" % AttNumHeads, "dim:%i" % EncValuePerHeadDim], "from": "att0"},  # (B, H*V)

      "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["target_embed", "att"], "n_out": 2},  # transform
      "readout_in": {"class": "linear", "from": ["prev:s", "prev:target_embed", "att"], "activation": None,
                     "n_out": 2},  # merge + post_merge bias
      "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
      "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"}
    }, "target": "classes", "max_seq_len": 7},
  }

  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": network,
    "start_epoch": 1,
    "num_epochs": 2,
    "batch_size": 10,
    "optimizer": {"class": "adam"},
    "learning_rate": 0.01,
    "debug_print_layer_output_template": True,
    "debug_runtime_sanity_checks": True,
  })
  _cleanup_old_models(config)

  print("Create engine.")
  engine = Engine(config=config)
  print("Init for train.")
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
  print("Train.")
  engine.train()
  print("Search.")
  engine.search(cv_data)


def test_search_multi_choice_hdf_dump():
  """
  Checking multiple things here:

  * train and search, with various configurations. The net includes multiple choices.
  * HDFDumpLayer, and loading these files.
  """
  EpochSplit = 3

  # Make HDF dataset such that multi-epoch works nicely with predefined seq tags.
  n_in, n_out = 2, 8
  from test_HDFDataset import generate_hdf_from_other
  hdf_dataset_fns = {
    key: generate_hdf_from_other({
      "class": "TaskNumberBaseConvertDataset",
      "input_base": n_in, "output_base": n_out,  # make input longer than output
      "num_seqs": {"train": EpochSplit * 10, "dev": 13}[key],
    }, suffix="-%s.hdf" % key)
    for key in ["train", "dev"]}

  tmp_model_dir = _get_tmp_dir()
  num_epochs = 5
  learning_rate = 0.001
  StoreAlignmentUpToEpoch = num_epochs
  AlignmentFilenamePattern = tmp_model_dir + "/alignments.%i.hdf"
  EncKeyTotalDim = 20
  target = "classes0"
  beam_size = 3
  AttNumHeads = 1

  def get_most_recent_align_hdf_files(epoch0):
    """
    :param int epoch0: 0-based (sub) epoch
    :return: filenames or None if there is nothing completed yet
    :rtype: list[str]|None
    """
    if epoch0 < EpochSplit:
      return None
    if epoch0 > StoreAlignmentUpToEpoch:
      epoch0 = StoreAlignmentUpToEpoch  # first epoch after
    i = ((epoch0 - EpochSplit) // EpochSplit) * EpochSplit
    return [AlignmentFilenamePattern % j for j in range(i, i + EpochSplit)]

  def get_dataset_dict(key, hdf_files=None):
    """
    :param str key: "train" or "dev"
    :param list[str]|None hdf_files:
    :rtype: dict[str]
    """
    d = {
      "class": "HDFDataset",
      "files": [hdf_dataset_fns[key]]
    }
    if key == "train":
      d["partition_epoch"] = EpochSplit
    if hdf_files:
      align_opts = {
        "class": "HDFDataset", "files": hdf_files,
        "unique_seq_tags": True  # dev set can exist multiple times
        }
      d = {
        "class": "MetaDataset",
        "datasets": {"main": d, "align": align_opts},
        "data_map": {
          "data": ("main", "data"),
          "classes": ("main", "classes"),
          "alignment": ("align", "data"),
          "align_score": ("align", "scores")},
        "seq_order_control_dataset": "main",  # it must support get_all_tags
      }
    return d

  def t_linear(source, **kwargs):
    import tensorflow as tf
    from returnn.tf.util.basic import where_bc
    enc = source(1, as_data=True, auto_convert=False)
    dec = source(0, as_data=True, auto_convert=False)
    enc_lens = enc.get_sequence_lengths()
    dec_lens = dec.get_sequence_lengths()
    dec_shape = tf.shape(dec.placeholder)
    dec_time_dim = dec_shape[dec.time_dim_axis]
    dec_times = tf.expand_dims(tf.range(dec_time_dim), axis=0)  # (1,dec-T)
    x = tf.cast(dec_times + 1, tf.float32)  # (1,dec-T)
    # We want: x[dec_len - 1] == enc_time - 1.
    factors = tf.maximum(tf.cast(enc_lens - 1, tf.float32), 0.0) / tf.maximum(tf.cast(dec_lens, tf.float32), 1.0)  # [B]
    # The definition does not allow loops, thus this is the minimum factor.
    factors = tf.maximum(factors, 1.0)
    factors = tf.expand_dims(factors, axis=1)  # (B,1)
    x = x * factors  # (B,dec-T)
    x = tf.cast(tf.round(x), tf.int32)
    # Note: If this causes loops in the very last frame, this is ok currently.
    x = tf.minimum(x, tf.expand_dims(enc_lens - 1, axis=1))
    # fix cheating gold targets with end flag filter. must be 0
    x = where_bc(tf.less(dec_times, tf.expand_dims(dec_lens, axis=1)), x, 0)
    return x

  def get_net_dict(task, pretrain_idx):
    """
    :param str task: "train" or "search"
    :param int|None pretrain_idx: starts at 0. note that this has a default repetition factor of 6
    :return: net_dict or None if pretrain should stop
    :rtype: dict[str,dict[str]|int]|None
    """
    # Note: epoch0 is 0-based here! I.e. in contrast to elsewhere, where it is 1-based.
    # Also, we never use #repetition here, such that this is correct.
    # This is important because of sub-epochs and storing the HDF files,
    # to know exactly which HDF files cover the dataset completely.
    epoch0 = pretrain_idx
    net_dict = {}  # type: typing.Dict[str,typing.Union[typing.Dict[str],int]]

    have_existing_align = False  # only in training, and only in pretrain, and only after the first epoch
    if pretrain_idx is not None:
      net_dict["#config"] = {}

      if task == "train":
        most_recent_align_hdf_files = get_most_recent_align_hdf_files(epoch0)
        have_existing_align = bool(most_recent_align_hdf_files)

        net_dict["#config"].update({
          "train": get_dataset_dict("train", hdf_files=most_recent_align_hdf_files),
          "dev": get_dataset_dict("dev", hdf_files=most_recent_align_hdf_files),
        })

      # Do this in the very beginning.
      lr_warmup = list(numpy.linspace(0.0001, learning_rate, num=4))
      lr_warmup += [learning_rate] * 10
      if pretrain_idx < len(lr_warmup):
        net_dict["#config"]["learning_rate"] = lr_warmup[pretrain_idx]
      pretrain_idx -= len(lr_warmup)

    use_t_search_as_target = not have_existing_align or epoch0 < StoreAlignmentUpToEpoch

    net_dict["#info"] = {
      "epoch0": epoch0,  # Set this here such that a new construction for every pretrain idx is enforced in all cases.
      "have_existing_align": have_existing_align,
      "use_t_search_as_target": use_t_search_as_target,
    }

    # We use this pretrain construction during the whole training time (epoch0 > num_epochs).
    if pretrain_idx is not None and epoch0 % EpochSplit == 0 and epoch0 > num_epochs:
      # Stop pretraining now.
      return None

    net_dict.update({
      "encoder": {"class": "linear", "n_out": 10, "activation": "relu", "from": "data"},
      "enc_ctx": {"class": "linear", "activation": None, "from": ["encoder"], "n_out": EncKeyTotalDim},
      "enc_value": {"class": "copy", "from": "encoder"},  # (B, enc-T, D)
      "enc_seq_len": {"class": "length", "from": "encoder", "sparse": True},

      # for task "search" / search_output_layer
      "decision": {
        "class": "decide", "from": "output", "loss": "edit_distance", "target": target,
        'only_on_search': True},

      "t_linear": {
        "class": "eval", "from": ["data:%s" % target, "encoder"], "eval": t_linear,
        "out_type": {
          "batch_dim_axis": 0, "time_dim_axis": 1, "shape": (None,), "sparse": True, "dtype": "int32", "dim": None},
        "size_target": target},

      "0_t_target": {
        "class": "postfix_in_time", "from": "data:classes", "postfix": 0,
        "register_as_extern_data": target},

      # Target for decoder ('output') with search ("extra.search") in training.
      # The layer name must be smaller than "t_target" such that this is created first.
      "1_t_base": {
        "class": "copy",
        "from": "existing_alignment" if have_existing_align else "t_linear",
        "register_as_extern_data": "t_base"},

      "2_t_target": {
        "class": "copy",
        "from": "extra.search:t_search_or_fallback" if use_t_search_as_target else "data:t_base",
        "register_as_extern_data": "t_target" if task == "train" else None},
    })

    if have_existing_align:
      net_dict.update({
        # This should be compatible to t_linear or t_search.
        "existing_alignment": {
          "class": "reinterpret_data", "from": "data:alignment",
          "set_sparse": True,  # not sure what the HDF gives us
          "set_sparse_dim": None,
          "size_base": "data:%s" % target,
        },
        # This should be compatible to search_score.
        "existing_align_score": {
          "class": "squeeze", "from": "data:align_score", "axis": "f",
          "loss": "as_is", "loss_scale": 0
        }
      })

    def get_output_dict(train, search, t_target, beam_size=beam_size):
      """
      :param bool train:
      :param bool search:
      :param str|None t_target:
      :param int beam_size:
      :rtype: dict[str]
      """
      return {
        "class": "rec", "from": [], "back_prop": (task == "train") and train,
        "unit": {
          "s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["s"],
                            "n_out": EncKeyTotalDim},
          "energy_in": {"class": "combine", "kind": "add", "from": ["base:enc_ctx", "s_transformed"],
                        "n_out": EncKeyTotalDim},
          "energy_tanh": {"class": "activation", "activation": "tanh", "from": "energy_in"},

          "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"],
                     "n_out": AttNumHeads},  # (B, enc-T, H)
          "energy1": {"class": "squeeze", "axis": "f", "from": "energy"},  # (B, enc-T)
          "energy2": {"class": "reinterpret_data", "from": "energy1", "set_axes": {"t": "stag:extern_data:data"}},

          # Segment boundaries:
          # - t0/t1/t is the right side (inclusive)
          # - prev:t is the left side (exclusive)
          # - t_start/prev_t_plus1 is the left side (inclusive)

          "prev_t_plus1": {"class": "eval", "from": "prev:t", "eval": "source(0) + 1"},
          "t_start": {
            "class": "eval", "from": ["prev_t_plus1", "base:enc_seq_len"],
            "eval": "tf.minimum(source(0), source(1) - 1)"},  # to avoid nans

          "t_weights": {
            "class": "softmax_over_spatial", "from": "energy2", "axis": "stag:extern_data:data",
            "start": "t_start"},
          "t_weights1": {
            # ChoiceLayer works on the feature axis.
            "class": "reinterpret_data", "from": "t_weights", "set_axes": {"f": "stag:extern_data:data"},
            # Loss for weights.
            "target": t_target if train else None,
            "loss": "ce" if (train and t_target) else None,
            "loss_scale": 0.1 if (train and t_target) else None,
          },
          "t0": {
            "class": "choice", "from": "t_weights1",
            "target": t_target, "cheating": bool(t_target),  # add this in training
            "beam_size": beam_size * 4 if task == "search" else beam_size,
            "keep_beams": task == "search",
            "length_normalization": False, "initial_output": -1},  # (B,)
          # Note: If beam-size > enc_seq_len, we end up with invalid t in the beam. Fix that.
          "t1": {
            "class": "eval", "from": ["t0", "t_start", "base:enc_seq_len"],
            "eval": "tf.clip_by_value(source(0), source(1), source(2) - 1)"},
          "t": {
            "class": "copy", "from": "t1", "initial_output": -1, "is_output_layer": bool(search)},
          "window_start": {"class": "eval", "from": "t", "eval": "source(0) - 5"},

          "att_weights": {
            "class": "softmax_over_spatial", "from": "energy2", "axis": "stag:extern_data:data",
            "window_start": "window_start",
            "window_size": 10},  # (B, enc-T)
          "att_soft": {"class": "generic_attention", "weights": "att_weights", "base": "base:enc_value"}, # (B, V)
          "att": {"class": "copy", "from": "att_soft"},

          "s": {"class": "rnn_cell", "unit": "standardlstm", "from": ["prev:target_embed", "prev:att"], "n_out": 10},
          "readout_in": {"class": "linear", "from": ["s", "prev:target_embed", "att"], "activation": None,
                         "n_out": 10},
          "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
          "output_prob": {"class": "softmax", "from": ["readout"], "dropout": 0.3, "target": target,
                          "loss": "ce" if train else None},

          'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'],
                           "n_out": 10, "initial_output": "var"},
          'output': {
            'class': 'choice', 'target': target, 'beam_size': beam_size, 'from': ["output_prob"],
            "initial_output": 0,
            'search': task != 'train', "length_normalization": task != "train"},

          "end": {"class": "compare", "from": "output", "value": 0},

        },
        "target": [target, t_target] if t_target else [target],
        "size_target": t_target,
        "include_eos": True,  # make sure no empty seqs
        "max_seq_len": "max_len_from('base:encoder')"}

    if task == "train":
      if use_t_search_as_target:
        net_dict.update({
          "extra.search:output":
            get_output_dict(
              train=False, search=True, t_target="t_base",
              beam_size=beam_size),
          "extra.search:t_search": {"class": "decide", "from": "extra.search:output/t"},
          "extra.search:search_loss": {
            "class": "decide", "from": "extra.search:output", "loss": "search_score", "loss_scale": 0},
          "extra.search:search_score": {
            "class": "eval", "from": "extra.search:search_loss",
            "out_type": {
              "dtype": "float32", "sparse": False,
              "shape": (), "dim": None, "batch_dim_axis": 0, "time_dim_axis": None},
            "eval": "(source(0, auto_convert=False),"
                    "tf.squeeze(self.sources[0].search_choices.beam_scores, axis=1)"
                    "/ tf.cast(source(0, auto_convert=False, as_data=True).get_sequence_lengths(), tf.float32))"
                    "[-1]",
            "loss": "as_is", "loss_scale": 0},
          "use_t_search":
              {"class": "compare", "kind": "less", "from": ["existing_align_score", "extra.search:search_score"]}
              if have_existing_align else
              {"class": "constant", "value": True},
          "t_search_or_fallback": {
              "class": "switch", "condition": "use_t_search",
              "true_from": "extra.search:t_search", "false_from": "data:t_base"}
              if have_existing_align else
              {"class": "copy", "from": "data:t_base"},
          "t_search_or_fallback_score":
              {"class": "switch", "condition": "use_t_search",
               "true_from": "extra.search:search_score", "false_from": "existing_align_score"}
              if have_existing_align else
              {"class": "copy", "from": "extra.search:search_score"},
        })
        if epoch0 is not None and epoch0 < StoreAlignmentUpToEpoch:
            net_dict.update({
                "extra.search:t_search_dump": {
                    "class": "hdf_dump", "from": "t_search_or_fallback",
                    "extra": {"scores": "t_search_or_fallback_score"},
                    "filename": AlignmentFilenamePattern % epoch0,
                    "is_output_layer": True},
                })

      net_dict["output"] = get_output_dict(train=True, search=False, t_target="t_target")
    else:
      net_dict["output"] = get_output_dict(train=True, search=True, t_target=None)

    return net_dict

  from returnn.datasets import init_dataset

  def run(task):
    """
    :param str task: "train" or "search"
    """
    print("-" * 80)
    print("Task:", task)

    def custom_construction_algo(idx, net_dict):
      return get_net_dict(task=task, pretrain_idx=idx)

    config = Config({
      "task": task,
      "train": get_dataset_dict("train"),
      "dev": get_dataset_dict("dev"),
      "extern_data": {
        "data": {"dim": n_in, "sparse": True},
        "classes": {"dim": n_out, "sparse": True},
        "alignment": {"dim": None, "shape": (None,), "dtype": "int32", "sparse": True},
        "align_score": {"shape": (1,), "dtype": "float32"},
      },
      "debug_print_layer_output_template": True,
      "network": get_net_dict(task=task, pretrain_idx=None),
      "pretrain": {"copy_param_mode": "subset", "construction_algo": custom_construction_algo},
      "batch_size": 1000,
      "max_seqs": 2,
      "optimizer": {"class": "adam"},
      "learning_rate": learning_rate,
      "use_learning_rate_control_always": True,
      "learning_rate_control": "newbob_multi_epoch",
      "learning_rate_control_error_measure": "dev_error_output/output_prob",
      "model": "%s/model" % tmp_model_dir,
      "cleanup_old_models": True,
      "num_epochs": num_epochs,
    })
    train_data = init_dataset(config.typed_value("train"))
    dev_data = init_dataset(config.typed_value("dev"))
    engine = Engine(config=config)
    if task == "train":
      engine.init_train_from_config(config, train_data, dev_data)
      engine.train()
    elif task == "search":
      engine.use_search_flag = True
      config.set("load_epoch", num_epochs)
      engine.init_network_from_config(config)
      engine.search(
        dev_data,
        do_eval=config.bool("search_do_eval", True),
        output_layer_names=config.typed_value("search_output_layer", "output"))
    else:
      raise NotImplementedError("task %r" % task)

  run("train")
  run("search")


def test_net_safe_log_to_log_softmax():
  n_out = 5
  net_dict = {
    "ff_in_window": {"class": "window", "window_size": 4, "from": "data:data"},  # (B,T,4,3)
    "ff_in": {"class": "merge_dims", "axes": ["dim:3", "dim:4"], "from": "ff_in_window"},  # (B,T,9)
    "ff0": {"class": "hidden", "activation": "relu", "n_out": 8, "L2": 0.01, "from": "ff_in"},  # (B,T,8)
    "ff_out": {"class": "softmax", "n_out": n_out, "from": "ff0"},  # (B,T,5)
    "ff_out_prior": {
      "class": "accumulate_mean", "exp_average": 0.001,
      "is_prob_distribution": True, "from": "ff_out"},  # (5,)
    "output": {
      "class": "combine", "kind": "eval", "from": ["ff_out", "ff_out_prior"],
      "eval": "safe_log(source(0)) - safe_log(source(1))",
      "eval_locals": {"am_scale": 0.1, "prior_scale": 0.5 * 0.1}
    },
  }
  net = TFNetwork(
    extern_data=ExternData(data={"data": {"dim": 3}, "classes": {"dim": n_out, "sparse": True}}),
    config=Config({"debug_print_layer_output_template": True}))
  net.construct_from_dict(net_dict)
  output_layer = net.get_default_output_layer(must_exist=True)
  out = output_layer.output.placeholder
  print(out)
  from returnn.tf.util.basic import print_graph_output
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


def test_preload_from_files():
  import tempfile
  model_tmp_dir = tempfile.mkdtemp("tmp-checkpoint")
  model_filename = model_tmp_dir + "/model"
  with make_scope() as session:
    config = Config()
    n_in, n_hidden, n_out = 2, 5, 3
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        "l1": {
          "class": "linear", "activation": None, "n_out": n_hidden, "from": "data:data",
          'bias_init': 1.0, 'forward_weights_init': 'orthogonal'},
        "output": {
          "class": "linear", "activation": None, "n_out": n_out, "from": "l1",
          'bias_init': 2.0, 'forward_weights_init': 'orthogonal'}
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    network.initialize_params(session)
    params_orig_dump = network.get_params_serialized(session)
    print("l1:")
    print(params_orig_dump.values_dict["l1"]["W"])
    print(params_orig_dump.values_dict["l1"]["b"])
    print("output:")
    print(params_orig_dump.values_dict["output"]["W"])
    print(params_orig_dump.values_dict["output"]["b"])
    assert(params_orig_dump.values_dict["l1"]["W"].any())
    assert(params_orig_dump.values_dict["output"]["W"].any())
    network.save_params_to_file(filename=model_filename, session=session)

  config = Config()
  config.update({
    "num_outputs": n_out,
    "num_inputs": n_in,
    "network": {
      "l0": {"class": "linear", "activation": None, "n_out": n_in, "from": "data:data"},
      "main_l1": {"class": "linear", "activation": None, "n_out": n_hidden, "from": ["l0"]},
      "main_output": {"is_output_layer": True, "class": "linear", "activation": None, "n_out": n_out, "from": ["main_l1"]},
    },
    "preload_from_files": {
      'train_base': {
        'filename': model_filename,
        'prefix': 'main_',
        'init_for_train': True,
      }
    },
    "device": "cpu",
    "start_epoch": 1,
    "num_epochs": 1,
    "batch_size": 50,
    "model": model_tmp_dir + "/clone_model",
  })

  from returnn.datasets.generating import DummyDataset
  from returnn.tf.engine import Engine
  seq_len = 5
  n_data_dim = n_in
  n_classes_dim = n_out
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)

  network = engine.network
  params_dump = network.get_params_serialized(engine.tf_session)
  for layer_name in ["l1", "output"]:
    layer_orig = params_orig_dump.values_dict[layer_name]
    layer_clone_main = params_dump.values_dict["main_" + layer_name]
    for param_name in ["W", "b"]:
      param_orig = layer_orig[param_name]
      param_clone_main = layer_clone_main[param_name]
      numpy.testing.assert_array_equal(param_orig, param_clone_main)

    main = engine.network.layers["main_" + layer_name]
    assert_equal(set(main.params.keys()), {"W", "b"})

  engine.finalize()


def test_preload_from_files_with_reuse():
  import tempfile
  model_tmp_dir = tempfile.mkdtemp("tmp-checkpoint")
  model_filename = model_tmp_dir + "/model"
  with make_scope() as session:
    config = Config()
    n_in, n_hidden, n_out = 2, 5, 3
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        "l1": {
          "class": "linear", "activation": None, "n_out": n_hidden, "from": "data:data",
          'bias_init': 1.0, 'forward_weights_init': 'orthogonal'},
        "output": {
          "class": "linear", "activation": None, "n_out": n_out, "from": ["l1"],
          'bias_init': 2.0, 'forward_weights_init': 'orthogonal'}
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    network.initialize_params(session)
    params_orig_dump = network.get_params_serialized(session)
    print("l1:")
    print(params_orig_dump.values_dict["l1"]["W"])
    print(params_orig_dump.values_dict["l1"]["b"])
    print("output:")
    print(params_orig_dump.values_dict["output"]["W"])
    print(params_orig_dump.values_dict["output"]["b"])
    assert(params_orig_dump.values_dict["l1"]["W"].any())
    assert(params_orig_dump.values_dict["output"]["W"].any())
    network.save_params_to_file(filename=model_filename, session=session)

  config = Config()
  config.update({
    "num_outputs": n_out,
    "num_inputs": n_in,
    "network": {
      "l0": {"class": "linear", "activation": None, "n_out": n_in, "from": "data:data"},
      "main_l1": {"class": "linear", "activation": None, "n_out": n_hidden, "from": ["l0"]},
      "main_output": {"is_output_layer": True, "class": "linear", "activation": None, "n_out": n_out, "from": ["main_l1"]},
      "clone_l0": {"class": "linear", "activation": None, "n_out": n_in, "from": "main_output"},
      "clone_l1": {"class": "linear", "activation": None, "n_out": n_hidden, "from": ["clone_l0"], "reuse_params": "main_l1"},
      "clone_output": {"is_output_layer": True, "class": "linear", "activation": None, "n_out": n_out, "from": ["clone_l1"], "reuse_params": "main_output"},
    },
    "preload_from_files": {
      'train_base': {
        'filename': model_filename,
        'prefix': 'main_',
        'init_for_train': True,
      }
    },
    "device": "cpu",
    "start_epoch": 1,
    "num_epochs": 1,
    "batch_size": 50,
    "model": model_tmp_dir + "/clone_model",
  })

  from returnn.datasets.generating import DummyDataset
  from returnn.tf.engine import Engine
  seq_len = 5
  n_data_dim = n_in
  n_classes_dim = n_out
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)

  network = engine.network
  params_dump = network.get_params_serialized(engine.tf_session)
  for layer_name in ["l1", "output"]:
    layer_orig = params_orig_dump.values_dict[layer_name]
    layer_clone_main = params_dump.values_dict["main_" + layer_name]
    layer_clone_clone = params_dump.values_dict["clone_" + layer_name]
    for param_name in ["W", "b"]:
      param_orig = layer_orig[param_name]
      param_clone_main = layer_clone_main[param_name]
      numpy.testing.assert_array_equal(param_orig, param_clone_main)

    main = engine.network.layers["main_" + layer_name]
    clone = engine.network.layers["clone_" + layer_name]
    assert_equal(set(main.params.keys()), {"W", "b"})
    assert_equal(set(clone.params.keys()), set())

  engine.finalize()


def test_preload_from_files_ignore_missing():
  import tempfile
  model_tmp_dir = tempfile.mkdtemp("tmp-checkpoint")
  model_filename = model_tmp_dir + "/model"
  with make_scope() as session:
    config = Config()
    n_in, n_hidden, n_out = 2, 5, 3
    config.update({
      "device": "cpu",
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        "l1": {"class": "linear", "activation": None, "n_out": n_hidden, "from": "data:data"},
        "output": {"class": "linear", "activation": None, "n_out": n_out, "from": ["l1"]}
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    network.initialize_params(session)
    params_orig_dump = network.get_params_serialized(session)
    print("l1:")
    print(params_orig_dump.values_dict["l1"]["W"])
    print(params_orig_dump.values_dict["l1"]["b"])
    print("output:")
    print(params_orig_dump.values_dict["output"]["W"])
    print(params_orig_dump.values_dict["output"]["b"])
    assert(params_orig_dump.values_dict["l1"]["W"].any())
    assert(params_orig_dump.values_dict["output"]["W"].any())
    network.save_params_to_file(filename=model_filename, session=session)

  config = Config()
  config.update({
    "num_outputs": n_out,
    "num_inputs": n_in,
    "network": {
      "l0": {"class": "linear", "activation": None, "n_out": n_in, "from": "data:data"},
      "l1": {"class": "linear", "activation": None, "n_out": n_hidden, "from": ["l0"]},
      "output": {"is_output_layer": True, "class": "linear", "activation": None, "n_out": n_out, "from": ["l1"]},
    },
    "preload_from_files": {
      'train_base': {
        'filename': model_filename,
        'prefix': '',
        'init_for_train': True,
        'ignore_missing': True
      }
    },
    "device": "cpu",
    "start_epoch": 1,
    "num_epochs": 1,
    "batch_size": 50,
    "model": model_tmp_dir + "/clone_model",
  })

  from returnn.datasets.generating import DummyDataset
  from returnn.tf.engine import Engine
  seq_len = 5
  n_data_dim = n_in
  n_classes_dim = n_out
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)

  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)

  network = engine.network
  params_dump = network.get_params_serialized(engine.tf_session)
  for layer_name in ["l1", "output"]:
    layer_orig = params_orig_dump.values_dict[layer_name]
    layer_clone_main = params_dump.values_dict[layer_name]
    for param_name in ["W", "b"]:
      param_orig = layer_orig[param_name]
      param_clone_main = layer_clone_main[param_name]
      numpy.testing.assert_array_equal(param_orig, param_clone_main)

    main = engine.network.layers[layer_name]
    assert_equal(set(main.params.keys()), {"W", "b"})

  engine.finalize()


# Test `init_network_from_config` for eval when both `model_epoch_filename` and `preload_from_files` are not None.
def test_init_network_from_config_preload_from_files_eval():
  import tempfile
  model_tmp_dir = tempfile.mkdtemp("-tmp-checkpoint")
  # Name ending with ".042" for `save_params_to_file` to generate a model checkpoint for epoch 42.
  # The same files are also used for pre-loading.
  preload_model_filename = model_tmp_dir + "/model.042"
  with make_scope() as session:
    config = Config()
    n_in, n_hidden, n_out = 2, 5, 3
    config.update({
      "num_outputs": n_out,
      "num_inputs": n_in,
      "network": {
        "l1": {"class": "linear", "activation": None, "n_out": n_hidden, "from": "data:data"},
        "output": {"class": "linear", "activation": None, "n_out": n_out, "from": ["l1"]}
      }
    })
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(config.typed_dict["network"])
    network.initialize_params(session)
    network.save_params_to_file(filename=preload_model_filename, session=session)

  config = Config()
  config.update({
    "num_outputs": n_out,
    "num_inputs": n_in,
    "network": {
      "l1": {"class": "linear", "activation": None, "n_out": n_hidden, "from": "data:data"},
      "main_l1": {"class": "linear", "activation": None, "n_out": n_hidden, "from": "data:data"},
      "add": {"class": "eval", "eval": "source(0) + source(1)", "n_out": n_hidden, "from": ["l1", "main_l1"]},
      "output": {"is_output_layer": True, "class": "linear", "activation": None, "n_out": n_out, "from": ["add"]},
    },
    "preload_from_files": {
      'train_base': {
        'filename': preload_model_filename,  # Pre-load from an arbitrary file.
        'prefix': 'main_',
      }
    },
    "task": "eval",
    "load_epoch": 42,  # Load from a checkpoint.
    "device": "cpu",
    "batch_size": 50,
    "model": model_tmp_dir + "/model",
  })

  from returnn.datasets.generating import DummyDataset
  from returnn.tf.engine import Engine
  seq_len = 5
  n_data_dim = n_in
  n_classes_dim = n_out
  cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  cv_data.init_seq_order(epoch=1)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=None, dev_data=cv_data, eval_data=None)
  engine.finalize()


def test_TikhonovRegularizationLayer():
  """
  Tests :class:`TikhonovRegularizationLayer`.
  """
  net_dict = {}
  layer_n_out = 10
  layer_common_args = {"class": "linear", "activation": "relu", "n_out": layer_n_out, "L2": 0.01}

  def layer(sources, **kwargs):
    args = kwargs.copy()
    for k, v in layer_common_args.items():
      args.setdefault(k, v)
    args.setdefault("from", sources)
    return args

  def make_network(num_layers):
    net_dict["input"] = {"class": "tikhonov_regularization", "meta_loss_scale": 0.1, "from": "data"}
    sources = ["input"]
    for i in range(num_layers):
      net_dict["layer%i" % i] = layer(sources=sources)
      sources = ["layer%i" % i]
    net_dict["output"] = {"class": "softmax", "loss": "ce", "loss_opts": {"use_fused": False}, "from": sources}

  make_network(num_layers=3)

  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=10, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  dev_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dev_data.init_seq_order(epoch=1)

  config = Config({
    "model": "%s/model" % _get_tmp_dir(),
    "batch_size": 100,
    "max_seqs": 2,
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "network": net_dict,
    "start_epoch": 1,
    "num_epochs": 2,
    "learning_rate": 0.01,
    "optimizer": {"class": "adam"},
    "debug_print_layer_output_template": True,
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=dev_data, eval_data=None)
  print("Extern data:")
  pprint(engine.network.extern_data.data)
  print("Used data keys:")
  pprint(engine.network.used_data_keys)
  engine.train()
  engine.finalize()


def test_grad_summaries():
  from returnn.datasets.generating import DummyDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  engine = Engine(config=Config({
    "network": {
      "output": {"class": "linear", "activation": "tanh", "from": "data", "n_out": 3, "loss": "mse"}
    },
    "model": "%s/model" % _get_tmp_dir(),
    "batch_size": 100,
    "max_seqs": 2,
    "num_outputs": n_classes_dim,
    "num_inputs": n_data_dim,
    "num_epochs": 1,
    "learning_rate": 0.01,
    "optimizer": {"class": "adam"},
    "debug_print_layer_output_template": True,
    "debug_grad_summaries": True,
  }))
  print("extern data:", engine.config.typed_value("extern_data"))

  engine.init_train_from_config()

  def extra_fetches_cb(summary_proto):
    """
    :param bytes summary_proto: protobuf for summaries
    """
    from tensorflow.core.framework import summary_pb2
    summaries = summary_pb2.Summary.FromString(summary_proto)
    summary_list = [val.tag for val in summaries.value]
    assert any([v.startswith("grads/") for v in summary_list])
    assert any(["global_grad_norm" in v for v in summary_list])
    assert any([v.startswith("vars/") for v in summary_list])
    for val in summaries.value:
      print("%s: %r" % (val.tag, val.simple_value))

  batches = train_data.generate_batches(
    recurrent_net=engine.network.recurrent,
    batch_size=200,
    max_seqs=100,
    used_data_keys=engine.network.used_data_keys)
  forwarder = Runner(
    engine=engine, dataset=train_data, batches=batches,
    train=True, eval=False,
    extra_fetches={
      "summary_proto": lambda: engine.network._get_all_merged_summaries(),
    },
    extra_fetches_callback=extra_fetches_cb)
  forwarder.run(report_prefix="test_grad_summaries")
  if not forwarder.finalized:
    raise Exception("Error happened. Exit now.")


def test_unflatten_2d():
  # See also test_SimpleHDFWriter_ndim1_var_len.
  # And unflatten_nd, and UnflattenNdLayer.
  from returnn.datasets.hdf import HDFDataset, SimpleHDFWriter
  from returnn.datasets.basic import set_config_num_inputs_outputs_from_dataset
  # E.g. attention weights, shape (dec-time,enc-time) per seq.
  fn = _get_tmp_file(suffix=".hdf")
  os.remove(fn)  # SimpleHDFWriter expects that the file does not exist
  writer = SimpleHDFWriter(filename=fn, dim=None, ndim=2, labels=None)
  dec_seq_lens = [11, 7, 5]
  enc_seq_lens = [13, 6, 8]
  batch1_data = numpy.random.normal(
    size=(len(dec_seq_lens), max(dec_seq_lens), max(enc_seq_lens))).astype("float32")
  writer.insert_batch(
    inputs=batch1_data,
    seq_len={0: dec_seq_lens, 1: enc_seq_lens},
    seq_tag=["seq-%i" % i for i in range(len(dec_seq_lens))])
  writer.close()

  dataset = HDFDataset(files=[fn])
  dataset.initialize()

  # Check first entry. (test_SimpleHDFWriter_ndim1_var_len does a full test.)
  dataset.init_seq_order(epoch=1)
  dataset.load_seqs(0, 2)  # does not matter
  data1 = dataset.get_data(0, "data")
  assert data1.shape == (dec_seq_lens[0] * enc_seq_lens[0],)
  data1_len = dataset.get_seq_length(0)
  assert data1_len["data"] == dec_seq_lens[0] * enc_seq_lens[0]
  assert data1_len["sizes"] == 2

  dataset.init_seq_order(epoch=2)
  engine = Engine(config=Config({
    "network": {
      "output": {"class": "unflatten_nd", "from": "data", "sizes": "data:sizes", "num_axes": 2}
    },
    "debug_print_layer_output_template": True
  }))
  set_config_num_inputs_outputs_from_dataset(config=engine.config, dataset=dataset)
  print("extern data:", engine.config.typed_value("extern_data"))

  engine.init_train_from_config()
  output_layer = engine.network.get_default_output_layer()
  assert output_layer.output.is_batch_major
  assert set(output_layer.output.size_placeholder.keys()) == {0, 1}

  def extra_fetches_cb(output, seq_len_1, seq_len_2, seq_tag, seq_idx):
    """
    :param numpy.ndarray output: shape=(n_batch,t1,t2)
    :param list[int] seq_len_1:
    :param list[int] seq_len_2:
    :param list[str] seq_tag: sequence tags of length n_batch
    :param list[int] seq_idx: of length n_batch
    """
    n_batch = len(seq_tag)
    assert n_batch == len(seq_idx) == len(seq_len_1) == len(seq_len_2) == output.shape[0]
    print("Got batch (N: %i), seq len 1: %r, seq len 2: %r, tags: %r, seq idx %r, out shape %r." % (
      n_batch, seq_len_1, seq_len_2, seq_tag, seq_idx, output.shape))
    for b in range(n_batch):
      assert dec_seq_lens[seq_idx[b]] == seq_len_1[b]
      assert enc_seq_lens[seq_idx[b]] == seq_len_2[b]
      numpy.testing.assert_almost_equal(
        batch1_data[seq_idx[b], :seq_len_1[b], :seq_len_2[b]], output[b, :seq_len_1[b], :seq_len_2[b]])

  batches = dataset.generate_batches(
    recurrent_net=engine.network.recurrent,
    batch_size=200,
    max_seqs=100,
    used_data_keys=engine.network.used_data_keys)
  forwarder = Runner(
    engine=engine, dataset=dataset, batches=batches,
    train=False, eval=False,
    extra_fetches={
      'output': output_layer.output.placeholder,
      "seq_len_1": output_layer.output.size_placeholder[0],
      "seq_len_2": output_layer.output.size_placeholder[1],
      "seq_tag": engine.network.get_seq_tags(),
      "seq_idx": engine.network.get_extern_data("seq_idx", mark_data_key_as_used=True)
    },
    extra_fetches_callback=extra_fetches_cb)
  forwarder.run(report_prefix="test_unflatten_2d")
  if not forwarder.finalized:
    raise Exception("Error happened. Exit now.")


def test_attention_forward_hdf_then_unflatten_2d():
  # See also test_SimpleHDFWriter_ndim1_var_len.
  # And unflatten_nd, and UnflattenNdLayer.
  from returnn.datasets.hdf import HDFDataset
  from returnn.datasets.basic import set_config_num_inputs_outputs_from_dataset
  from returnn.datasets.generating import TaskNumberBaseConvertDataset
  from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell

  # Simple version of e.g.:
  # https://github.com/rwth-i6/returnn-experiments/blob/master/2018-attention/wmt2017ende/blocks-flstm.enc6l.decb.pretrain2.adam.lr1e_3.mseqs100.bs4000.ls01.tembi0.invfert.oeps1e_8.gradnoise0.seqsort1000.config
  att_net_dict = {
    "input_embed": {"class": "linear", "activation": None, "n_out": 10, "from": "data:data"},

    "lstm0_fw": {"class": "rec", "unit": "LSTMBlock", "n_out": 10, "direction": 1, "from": ["input_embed"]},
    "lstm0_bw": {"class": "rec", "unit": "LSTMBlock", "n_out": 10, "direction": -1, "from": ["input_embed"]},

    "lstm1_fw": {"class": "rec", "unit": "LSTMBlock", "n_out": 10, "direction": 1, "from": ["lstm0_fw", "lstm0_bw"]},
    "lstm1_bw": {"class": "rec", "unit": "LSTMBlock", "n_out": 10, "direction": -1, "from": ["lstm0_fw", "lstm0_bw"]},

    "encoder": {"class": "copy", "from": ["lstm1_fw", "lstm1_bw"]},
    "enc_ctx": {"class": "linear", "activation": None, "with_bias": True, "from": ["encoder"], "n_out": 10},

    "output": {"class": "rec", "from": [], "unit": {
      'output': {'class': 'choice', 'target': 'classes', 'beam_size': 5, 'from': ["output_prob"],
                 "initial_output": 0},
      "end": {"class": "compare", "from": ["output"], "value": 0},
      'target_embed': {'class': 'linear', 'activation': None, "with_bias": False, 'from': ['output'], "n_out": 10,
                       "initial_output": 0},
      "s_state": {"class": "get_last_hidden_state", "from": ["s"], "n_out": 20},
      "s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": ["s_state"], "n_out": 10},
      "energy_in": {"class": "combine", "kind": "add", "from": ["base:enc_ctx", "s_transformed"], "n_out": 10},
      "energy_tanh": {"class": "activation", "activation": "tanh", "from": ["energy_in"]},
      # (B, enc-T, 1)
      "energy": {"class": "linear", "activation": None, "with_bias": False, "from": ["energy_tanh"], "n_out": 1},
      "att_weights": {"class": "softmax_over_spatial", "from": ["energy"], "is_output_layer": True},  # (B, enc-T, 1)
      "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder", "auto_squeeze": True},
      "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:target_embed", "prev:att"], "n_out": 10},
      "readout_in": {"class": "linear", "from": ["s", "prev:target_embed", "att"], "activation": None, "n_out": 10},
      "readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["readout_in"]},
      "output_prob": {"class": "softmax", "from": ["readout"], "target": "classes", "loss": "ce"}
    }, "target": "classes"}
  }
  config = Config({
    "allow_random_model_init": True,
    "network": att_net_dict,
    "debug_print_layer_output_template": True,
    "batch_size": 50,
    "max_seqs": 4
  })
  task_dataset = TaskNumberBaseConvertDataset(num_seqs=17)
  task_dataset.initialize()
  task_dataset.init_seq_order(epoch=1)
  set_config_num_inputs_outputs_from_dataset(config=config, dataset=task_dataset)
  print("task extern data:", config.typed_value("extern_data"))

  att_engine = Engine(config=config)
  att_engine.use_dynamic_train_flag = True  # we need targets for the seq len
  att_engine.init_network_from_config(config)
  att_output_layer = att_engine.network.get_layer("output/att_weights")
  print("att weights layer:", att_output_layer, att_output_layer.output.size_placeholder)
  att_rec_layer = att_engine.network.layers["output"]
  assert isinstance(att_rec_layer, RecLayer)
  assert isinstance(att_rec_layer.cell, _SubnetworkRecCell)
  inner_att_output_layer = att_rec_layer.cell.net.layers["att_weights"]
  print("inner att weights layer:", inner_att_output_layer, inner_att_output_layer.output.size_placeholder)

  # dec-time, enc-time. the 1 is just an artifact of the construct
  assert att_output_layer.output.copy_as_batch_spatial_major().shape == (None, None, 1)
  assert len(att_output_layer.output.size_placeholder) == 2  # encoder and decoder time
  hdf_fn = _get_tmp_file(suffix=".hdf")
  os.remove(hdf_fn)  # forward_to_hdf expects that the file does not exist
  att_engine.forward_to_hdf(output_file=hdf_fn, data=task_dataset, output_layer=att_output_layer)

  hdf_dataset = HDFDataset(files=[hdf_fn])
  hdf_dataset.initialize()

  # Check first entry. (test_SimpleHDFWriter_ndim1_var_len does a full test.)
  task_dataset.init_seq_order(epoch=1)
  hdf_dataset.init_seq_order(epoch=1)
  task_dataset.load_seqs(0, 2)  # does not matter
  hdf_dataset.load_seqs(0, 2)  # does not matter
  data1 = hdf_dataset.get_data(0, "data")
  task_seq_lens = task_dataset.get_seq_length(0)
  assert data1.shape == (task_seq_lens["classes"] * task_seq_lens["data"], 1)  # see att_output_layer above
  data1_len = hdf_dataset.get_seq_length(0)
  assert data1_len["data"] == task_seq_lens["classes"] * task_seq_lens["data"]
  assert data1_len["sizes"] == 2
  sizes1 = hdf_dataset.get_data(0, "sizes")
  assert sizes1.shape == (2,)
  assert sizes1.tolist() == [task_seq_lens["classes"], task_seq_lens["data"]]

  task_dataset.init_seq_order(epoch=1)
  hdf_dataset.init_seq_order(epoch=2)
  unflatten_engine = Engine(config=Config({
    "network": {
      "output": {"class": "unflatten_nd", "from": "data", "sizes": "data:sizes", "num_axes": 2}
    },
    "debug_print_layer_output_template": True
  }))
  set_config_num_inputs_outputs_from_dataset(config=unflatten_engine.config, dataset=hdf_dataset)
  print("hdf extern data:", unflatten_engine.config.typed_value("extern_data"))

  unflatten_engine.init_train_from_config()
  unflatten_output_layer = unflatten_engine.network.get_default_output_layer()
  assert unflatten_output_layer.output.is_batch_major
  assert set(unflatten_output_layer.output.size_placeholder.keys()) == {0, 1}

  def extra_fetches_cb(output, seq_len_1, seq_len_2, seq_tag, seq_idx):
    """
    :param numpy.ndarray output: shape=(n_batch,t1,t2)
    :param list[int] seq_len_1:
    :param list[int] seq_len_2:
    :param list[str] seq_tag: sequence tags of length n_batch
    :param list[int] seq_idx: of length n_batch
    """
    n_batch = len(seq_tag)
    assert n_batch == len(seq_idx) == len(seq_len_1) == len(seq_len_2) == output.shape[0]
    print("Got batch (N: %i), seq len 1: %r, seq len 2: %r, tags: %r, seq idx %r, out shape %r." % (
      n_batch, seq_len_1, seq_len_2, seq_tag, seq_idx, output.shape))
    task_dataset.load_seqs(min(seq_idx), max(seq_idx) + 1)
    for b in range(n_batch):
      task_seq_lens = task_dataset.get_seq_length(seq_idx[b])
      assert task_seq_lens["classes"] == seq_len_1[b]
      assert task_seq_lens["data"] == seq_len_2[b]

  batches = hdf_dataset.generate_batches(
    recurrent_net=unflatten_engine.network.recurrent,
    batch_size=200,
    max_seqs=100,
    used_data_keys=unflatten_engine.network.used_data_keys)
  forwarder = Runner(
    engine=unflatten_engine, dataset=hdf_dataset, batches=batches,
    train=False, eval=False,
    extra_fetches={
      'output': unflatten_output_layer.output.placeholder,
      "seq_len_1": unflatten_output_layer.output.size_placeholder[0],
      "seq_len_2": unflatten_output_layer.output.size_placeholder[1],
      "seq_tag": unflatten_engine.network.get_seq_tags(),
      "seq_idx": unflatten_engine.network.get_extern_data("seq_idx", mark_data_key_as_used=True)
    },
    extra_fetches_callback=extra_fetches_cb)
  forwarder.run(report_prefix="test_attention_forward_hdf_then_unflatten_2d")
  if not forwarder.finalized:
    raise Exception("Error happened. Exit now.")


def test_preinit_reset_train_dataset():
  """
  This is a complex test.
  We have some default dataset.
  Then we run through it, and use HDFDumpLayer to dump some info.
  Then later we overwrite the dataset (via pretrain `#config`) to load that dumped data.
  Also, make sure that all not-used-anymore data gets unloaded.
  """
  # For the default dataset, we want something which is not frame-synced (i.e. input has different length than output).
  # TaskNumberBaseConvertDataset has this property.
  # Also, we need a dataset which supports init_seq_order with custom seq_list,
  # which is needed for the MetaDataset.
  # TaskNumberBaseConvertDataset does not support this, so we convert it to HDF.
  print("Preparing data...")
  from test_HDFDataset import generate_hdf_from_other, get_test_tmp_file
  from returnn.datasets import init_dataset
  n_in, n_out = 2, 8
  default_train_hdf_fn = generate_hdf_from_other({
    "class": "TaskNumberBaseConvertDataset", "num_seqs": 11,
    "input_base": n_in, "output_base": n_out})
  default_dev_hdf_fn = generate_hdf_from_other({
    "class": "TaskNumberBaseConvertDataset", "num_seqs": 5, "fixed_random_seed": 42,
    "input_base": n_in, "output_base": n_out})
  default_train_dataset_opts = {"class": "HDFDataset", "files": [default_train_hdf_fn]}
  default_dev_dataset_opts = {"class": "HDFDataset", "files": [default_dev_hdf_fn]}
  default_train_dataset = init_dataset(default_train_dataset_opts)
  default_dev_dataset = init_dataset(default_dev_dataset_opts)

  def get_meta_dataset_opts(base_opts, hdf_dump_fn):
    """
    :param dict[str] base_opts:
    :param str hdf_dump_fn:
    :rtype: dict[str]
    """
    return {
      "class": "MetaDataset",
      "datasets": {"base": base_opts, "dump": {"class": "HDFDataset", "files": [hdf_dump_fn]}},
      "data_map": {
        "data": ("base", "data"),
        "classes": ("base", "classes"),
        "dump": ("dump", "data")
      },
      "seq_order_control_dataset": "base"
    }

  dump_hdf_filenames = [None, get_test_tmp_file(".dump1.hdf"), get_test_tmp_file(".dump2.hdf")]
  num_epochs = len(dump_hdf_filenames)
  for fn in dump_hdf_filenames:
    if fn:
      os.remove(fn)  # HDFDumpLayer expects that they don't exist

  def get_net_dict(idx=None, net_dict=None):
    """
    :param int|None idx:
    :param net_dict:
    :return: dict
    """
    if idx is not None and idx >= num_epochs:
      return None
    net_dict = {
      "#idx": idx,  # informal, and trigger reinit in all cases
      "embed": {"class": "linear", "from": "data", "with_bias": False, "activation": None, "n_out": 10},
      "lstm1": {"class": "rec", "unit": "BasicLSTM", "from": "embed", "n_out": 10},
      "lstm2": {"class": "rec", "unit": "BasicLSTM", "from": "lstm1", "n_out": 10},
      "output": {"class": "softmax", "from": "lstm2", "loss": "ctc"}
    }
    if idx is not None and dump_hdf_filenames[idx]:
      net_dict["hdf_dump"] = {
        "class": "hdf_dump",
        "from": "data:classes",
        "filename": dump_hdf_filenames[idx],
        "is_output_layer": True  # trigger usage of this layer
      }
    if idx is not None and idx >= 1 and dump_hdf_filenames[idx - 1]:
      net_dict["#config"] = {
        "train": get_meta_dataset_opts(default_train_dataset_opts, dump_hdf_filenames[idx - 1]),
        "dev": get_meta_dataset_opts(default_dev_dataset_opts, dump_hdf_filenames[idx - 1])}
      # Some dummy usage of the extra data.
      net_dict["print"] = {
        "class": "print", "from": "data:dump",
        "is_output_layer": True  # trigger usage of this layer
      }
    return net_dict

  config = Config({
    "train": default_train_dataset_opts,
    "dev": default_dev_dataset_opts,
    "extern_data": {
      "data": {"dim": n_in, "sparse": True},
      "classes": {"dim": n_out, "sparse": True},
      "dump": {"dim": n_out, "sparse": True}  # we dump the classes
    },
    "network": get_net_dict(),
    "pretrain": {"construction_algo": get_net_dict},
    "num_epochs": num_epochs,
    "debug_print_layer_output_template": True,
    "batch_size": 50, "max_seqs": 3,
    "tf_log_dir": None
  })

  print("Create engine.")
  engine = Engine(config=config)
  print("Init training.")
  engine.init_train_from_config(train_data=default_train_dataset, dev_data=default_dev_dataset)
  print("Train.")
  engine.train()
  engine.finalize()
  print("Finished training.")

  # Now some tests.
  print("Testing.")
  for fn in dump_hdf_filenames:
    if fn:
      assert os.path.exists(fn)  # should have been created now


def test_non_available_data_construction():
  from returnn.datasets.generating import DummyDataset, StaticDataset
  import tempfile
  output_file = tempfile.mktemp(suffix=".hdf", prefix="nose-tf-forward")
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 7
  train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  train_data.init_seq_order(epoch=1)
  dev_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dev_data.init_seq_order(epoch=1)
  search_data = StaticDataset(
    data=[{'data': numpy.ones((seq_len, n_data_dim), dtype="float32")}], output_dim={'data': (n_data_dim, 2)})
  search_data.init_seq_order(epoch=1)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "batch_size": 5000,
    "max_seqs": 2,
    "extern_data": {
      'data': {'dim': n_data_dim, 'shape': (None, n_data_dim), 'available_for_inference': True},
      'classes': {'dim': n_classes_dim, 'shape': (None,), 'sparse': True, 'available_for_inference': False}},
    "num_epochs": 1,
    "network": {
      "data_target": {
        "class": "linear", "activation": "tanh", "from": "data:classes", "n_out": 4,
        "register_as_extern_data": "extra_target"},
      "input": {"class": "linear", "from": "data", "activation": "tanh", "n_out": 3},
      "extra_output": {
        "class": "linear", "from": "input", "activation": "tanh", "n_out": 4,
        "target": "extra_target", "loss": "mse"},
      "output": {"class": "softmax", "from": "input", "target": "classes", "loss": "ce"},
    }
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  print("Train...")
  engine.init_train_from_config(config=config, train_data=train_data, dev_data=dev_data)
  engine.train()

  print("Forward...")
  engine.use_search_flag = False
  engine.use_eval_flag = False
  engine.use_dynamic_train_flag = False
  engine.init_network_from_config(config)
  engine.forward_to_hdf(data=search_data, output_file=output_file, batch_size=1)

  print("Search...")
  engine.use_search_flag = True
  engine.init_network_from_config(config)
  engine.search(dataset=search_data, do_eval=False)
  print("error keys:")

  engine.finalize()


def test_regression_choice():
  net_dict = {
    "encoder": {"class": "linear", "activation": "tanh", "from": ["data:classes"], "n_out": 20},
    "enc_ctx": {"class": "linear", "activation": "tanh", "from": ["encoder"], "n_out": 20},

    "output": {"class": "rec", "from": [], "max_seq_len": 10, "unit": {
      "s": {"class": "rnn_cell", "unit": "LSTMBlock", "from": ["prev:c", "feedback"], "n_out": 20},
      "c_in": {"class": "linear", "activation": "tanh", "from": ["s"], "n_out": 20},
      "c": {"class": "dot_attention", "from": ["c_in"], "base": "base:encoder", "base_ctx": "base:enc_ctx",
            "n_out": 20},
      "output": {"class": "linear", "from": ["s", "c"], "target": "data", 'n_out': 2, 'activation': None,
                 "loss": "mse"},
      "choice": {'class': 'choice', 'beam_size': 1, 'input_type': 'regression', 'from': 'output',
                 'target': 'data', 'n_out': 2},
      "feedback": {'class': 'linear', 'from': 'prev:choice', 'n_out': 5, 'activation': 'tanh'},
      'end_compare': {'class': 'compare', 'from': ['stop_token_sigmoid'], 'kind': 'greater', 'value': 0.5},
      'end': {'class': 'squeeze', 'from': ['end_compare'], 'axis': 'F'},
      'stop_token_sigmoid': {'activation': 'sigmoid', 'class': 'linear', 'from': ['s'], 'n_out': 1}
    }, "target": "data"}

  }

  from returnn.datasets.generating import DummyDataset, StaticDataset
  seq_len = 5
  n_data_dim = 2
  n_classes_dim = 3
  dataset = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
  dataset.init_seq_order(epoch=1)
  search_data = StaticDataset(
    data=[{'classes': numpy.ones((seq_len,), dtype="int32")}], output_dim={'classes': (n_classes_dim, 1)})
  search_data.init_seq_order(epoch=1)

  import tempfile
  output_file = tempfile.mktemp(suffix=".hdf", prefix="nose-tf-forward")

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "batch_size": 100,
    "max_seqs": 2,
    "extern_data": {
      'data': {'dim': n_data_dim, 'shape': (None, n_data_dim), 'available_for_inference': False},
      'classes': {'dim': n_classes_dim, 'shape': (None,), 'sparse': True, 'available_for_inference': True}},
    "network": net_dict,
    "start_epoch": 1,
    "num_epochs": 2,
    "learning_rate": 0.01,
    "optimizer": {"class": "adam"},
    "debug_print_layer_output_template": True,
    "debug_print_layer_output_shape": True,
  })

  _cleanup_old_models(config)
  engine = Engine(config=config)
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=dataset, eval_data=None)
  print("Extern data:")
  pprint(engine.network.extern_data.data)
  print("Used data keys:")
  pprint(engine.network.used_data_keys)
  engine.train()

  print("Search...")
  engine.use_search_flag = True
  engine.use_eval_flag = False
  engine.use_dynamic_train_flag = False
  engine.init_network_from_config(config)
  engine.search(dataset=search_data, do_eval=False)

  print("Forward...")
  engine.use_search_flag = False
  engine.use_eval_flag = False
  engine.use_dynamic_train_flag = False
  engine.init_network_from_config(config)
  engine.forward_to_hdf(data=dataset, output_file=output_file, batch_size=1)

  engine.finalize()


def test_engine_create_network_varying_flags():
  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "num_outputs": 3,
    "num_inputs": 2,
    "network":  {
      "enc0": {"class": "linear", "from": "data", "activation": "sigmoid", "n_out": 3},
      "enc": {"class": "reduce", "axis": "T", "mode": "mean", "from": "enc0"},
      "output": {
        "class": "rec", "from": [], "target": "classes", "max_seq_len": 10,
        "unit": {
          "embed": {"class": "linear", "from": "prev:output", "activation": "sigmoid", "n_out": 3},
          "prob": {"class": "softmax", "from": ["embed", "base:enc"], "loss": "ce", "target": "classes"},
          "output": {"class": "choice", "beam_size": 4, "from": "prob", "target": "classes", "initial_output": 0},
          "end": {"class": "compare", "from": "output", "value": 0}
        }
      },
    },
  })
  engine = Engine(config=config)
  engine.create_network(
    config=config, rnd_seed=0, train_flag=False, eval_flag=False, search_flag=False,
    net_dict=config.typed_dict['network'])
  engine.create_network(
    config=config, rnd_seed=0, train_flag=True, eval_flag=False, search_flag=False,
    net_dict=config.typed_dict['network'])
  engine.create_network(
    config=config, rnd_seed=0, train_flag=False, eval_flag=True, search_flag=False,
    net_dict=config.typed_dict['network'])
  engine.create_network(
    config=config, rnd_seed=0, train_flag=False, eval_flag=False, search_flag=True,
    net_dict=config.typed_dict['network'])

  engine.finalize()


def test_engine_search_output_file():
  import tempfile
  import pickle
  from returnn.datasets.generating import StaticDataset
  from returnn.datasets.util.vocabulary import CharacterTargets
  seq_len = 5
  n_data_dim = 10
  n_classes_dim = 2
  num_seqs = 4
  rnd = numpy.random.RandomState(42)

  dataset = StaticDataset([{
    "data": rnd.randint(0, n_data_dim, size=(seq_len,), dtype=numpy.int32),
    "classes": rnd.randint(0, n_classes_dim, size=(seq_len,), dtype=numpy.int32),
    "classes2": rnd.randint(0, n_classes_dim, size=(seq_len,), dtype=numpy.int32),
    "classes3": rnd.randint(0, n_classes_dim, size=(seq_len,), dtype=numpy.int32),
  } for _ in range(num_seqs)], output_dim={
    "data": (n_data_dim, 1),
    "classes": (n_classes_dim, 1),
    "classes2": (n_classes_dim, 1),
    "classes3": (n_classes_dim, 1),
  })
  dataset.labels = {
    "classes": [str(i) for i in range(n_classes_dim)]
  }
  vocab_dict = {str(i): i for i in range(n_classes_dim)}
  vocab_file = tempfile.mktemp(suffix=".pkl", prefix="vocab")
  with open(vocab_file, 'wb') as fp:
    pickle.dump(vocab_dict, fp)
  vocab = CharacterTargets(vocab_file, unknown_label=None)
  dataset.init_seq_order(epoch=1)

  from returnn.tf.util.data import FeatureDim
  classes_dim = FeatureDim("classes", dimension=n_classes_dim, vocab=vocab)

  config = Config()
  config.update({
    "model": "%s/model" % _get_tmp_dir(),
    "batch_size": 5000,
    "extern_data": {
      'data': {'dim': n_data_dim, 'sparse': True},
      'classes': {'dim': n_classes_dim, 'sparse': True},
      'classes2': {'dim': n_classes_dim, 'sparse': True, 'vocab': vocab},
    },
    "network": {
      "enc0": {"class": "linear", "activation": "sigmoid", "n_out": 3, "from": "data:data"},
      "enc1": {"class": "reduce", "mode": "max", "axis": "t", "from": "enc0"},
      "output": {
        "class": "rec", "from": [], "max_seq_len": 10, 'target': 'classes',
        "unit": {
          "embed": {"class": "linear", "from": "prev:output", "activation": "sigmoid", "n_out": 3},
          "prob": {"class": "softmax", "from": ["embed", "base:enc1"], "loss": "ce", "target": "classes", 'is_output_layer': True},
          "output": {"class": "choice", "beam_size": 4, "from": "prob", "target": "classes", "initial_output": 0},
          "end": {"class": "compare", "from": "output", "value": 0}
        }
      },
      "decision": {"class": "decide", "from": "output", 'is_output_layer': True},
      "decision2": {"class": "decide", "from": "output", 'is_output_layer': True, "target": "classes2"},
      "_decision3": {"class": "decide", "from": "output"},
      "decision3": {"class": "reinterpret_data", "from": "_decision3", 'is_output_layer': True, "target": "classes",
                    "set_sparse_dim": classes_dim},
      "best_score": {"class": "decide", "from": "all_scores", 'only_on_search': True, 'is_output_layer': True},
      "all_scores": {"class": "choice_get_beam_scores", "from": "output", 'is_output_layer': True, 'only_on_search': True},
      "probs": {"class": "copy", "from": "output/prob", 'is_output_layer': True, 'target': None},
    },
    'search_output_layer': ['decision', 'decision2', 'decision3', 'best_score', 'all_scores', 'probs'],
  })
  _cleanup_old_models(config)
  engine = Engine(config=config)
  # Normally init_network can be used. We only do init_train here to randomly initialize the network.
  engine.init_train_from_config(config=config, train_data=dataset, dev_data=None, eval_data=None)
  print("network:")
  pprint(engine.network.layers)

  # Now reinit for search.
  assert not engine.use_search_flag
  engine.use_search_flag = True
  engine.use_dynamic_train_flag = False
  print("Reinit network with search flag.")
  engine.init_network_from_config(config=config)

  output_file = tempfile.mktemp(suffix=".py", prefix="search_output_file")
  engine.search(dataset=dataset, output_layer_names=config.typed_value("search_output_layer", "output"), output_file=output_file, output_file_format='py')
  print("error keys:")
  pprint(engine.network.losses_dict)
  assert engine.network.total_objective is not None

  # Validate generated output file
  with open(output_file) as fp:
    output_text = fp.read()
    print(output_text)
    from numpy import array, float32
    output = eval(output_text, {'array': array, 'float32': float32})
    assert set(config.typed_value('search_output_layer')) == set(output['seq-0'].keys())
    assert all([seq_output['decision'] == seq_output['decision2'] for seq_output in output.values()])
    assert all([seq_output['decision'] == seq_output['decision3'] for seq_output in output.values()])

  engine.finalize()


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
    try:
      session.close()
      tf_compat.v1.reset_default_graph()
    except Exception as exc:
      print("test finally handler, exception:", type(exc).__name__, ":", exc)
    import threading
    if len(list(threading.enumerate())) > 1:
      print("Warning, more than one thread at exit:")
      better_exchook.dump_all_thread_tracebacks()
    del session
