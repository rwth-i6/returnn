
import sys
sys.path += ["."]  # Python 3 hack

from Config import Config
from LearningRateControl import *
from nose.tools import assert_equal
import numpy
import unittest

import better_exchook
better_exchook.replace_traceback_format_tb()

from Log import log
log.initialize()


def test_save_load():
  import tempfile
  with tempfile.NamedTemporaryFile(mode="w") as f:
    filename = f.name
  assert not os.path.exists(filename)
  control = LearningRateControl(default_learning_rate=1.0, filename=filename)
  assert 2 not in control.epoch_data
  control.epoch_data[2] = LearningRateControl.EpochData(learningRate=0.0008, error={
    'dev_error_ctc': 0.22486433815293946,
    'dev_error_decision': 0.0,
    'dev_error_output/output_prob': 0.16270349413262444,
    'dev_score_ctc': 1.0732941136466485,
    'dev_score_output/output_prob': 0.7378438060027533,
    'train_error_ctc': 0.13954045252681482,
    'train_error_decision': 0.0,
    'train_error_output/output_prob': 0.106904268810835,
    'train_score_ctc': 0.5132869609859635,
    'train_score_output/output_prob': 0.5098970897590558,
  })
  control.save()
  assert os.path.exists(filename)
  control = LearningRateControl(default_learning_rate=1.0, filename=filename)
  assert 2 in control.epoch_data
  data = control.epoch_data[2]
  numpy.testing.assert_allclose(data.learning_rate, 0.0008)
  assert "dev_error_output/output_prob" in data.error
  numpy.testing.assert_allclose(data.error["dev_error_output/output_prob"], 0.16270349413262444)


def test_load():
  import tempfile
  with tempfile.NamedTemporaryFile(mode="w") as f:
    f.write("""{
      1: EpochData(learningRate=0.0008, error={
      'dev_error_ctc': 0.21992561489090365,
      'dev_error_decision': 0.0,
      'dev_error_output/output_prob': 0.1597158714185534,
      'dev_score_ctc': 1.0742086989480388,
      'dev_score_output/output_prob': 0.7316125233255415,
      'train_error_ctc': 0.11740542462939381,
      'train_error_decision': 0.0,
      'train_error_output/output_prob': 0.10000902651529825,
      'train_score_ctc': 0.42154947919396146,
      'train_score_output/output_prob': 0.4958179737218142,
      }),
      2: EpochData(learningRate=0.0008, error={
      'dev_error_ctc': 0.22486433815293946,
      'dev_error_decision': 0.0,
      'dev_error_output/output_prob': 0.16270349413262444,
      'dev_score_ctc': 1.0732941136466485,
      'dev_score_output/output_prob': 0.7378438060027533,
      'train_error_ctc': 0.13954045252681482,
      'train_error_decision': 0.0,
      'train_error_output/output_prob': 0.106904268810835,
      'train_score_ctc': 0.5132869609859635, 
      'train_score_output/output_prob': 0.5098970897590558,
      }),
    }""")
    f.flush()
    control = LearningRateControl(default_learning_rate=1.0, filename=f.name)
    assert set(control.epoch_data.keys()) == {1, 2}
    data = control.epoch_data[2]
    numpy.testing.assert_allclose(data.learning_rate, 0.0008)
    assert "dev_error_output/output_prob" in data.error
    numpy.testing.assert_allclose(data.error["dev_error_output/output_prob"], 0.16270349413262444)


def test_init_error_old():
  config = Config()
  config.update({"learning_rate_control": "newbob", "learning_rate_control_error_measure": "dev_score"})
  lrc = load_learning_rate_control_from_config(config)
  assert isinstance(lrc, NewbobRelative)
  lrc.get_learning_rate_for_epoch(1)
  lrc.set_epoch_error(1, {"train_score": 1.9344199658230012})
  lrc.set_epoch_error(1, {"dev_score": 1.99, "dev_error": 0.6})
  error = lrc.get_epoch_error_dict(1)
  assert "train_score" in error
  assert "dev_score" in error
  assert "dev_error" in error
  assert_equal(lrc.get_error_key(1), "dev_score")
  lrc.get_learning_rate_for_epoch(2)
  lrc.set_epoch_error(2, {"train_score": 1.8})
  lrc.set_epoch_error(2, {"dev_score": 1.9, "dev_error": 0.5})
  lrc.get_learning_rate_for_epoch(3)


def test_init_error_new():
  config = Config()
  config.update({"learning_rate_control": "newbob", "learning_rate_control_error_measure": "dev_score"})
  lrc = load_learning_rate_control_from_config(config)
  assert isinstance(lrc, NewbobRelative)
  lrc.get_learning_rate_for_epoch(1)
  lrc.set_epoch_error(1, {"train_score": {'cost:output': 1.9344199658230012}})
  lrc.set_epoch_error(1, {"dev_score": {'cost:output': 1.99}, "dev_error": {'error:output': 0.6}})
  error = lrc.get_epoch_error_dict(1)
  assert "train_score" in error
  assert "dev_score" in error
  assert "dev_error" in error
  assert_equal(lrc.get_error_key(1), "dev_score")
  lrc.get_learning_rate_for_epoch(2)
  lrc.set_epoch_error(2, {"train_score": {'cost:output': 1.8}})
  lrc.set_epoch_error(2, {"dev_score": {'cost:output': 1.9}, "dev_error": {'error:output': 0.5}})
  lrc.get_learning_rate_for_epoch(3)


def test_init_error_muliple_out():
  config = Config()
  config.update({"learning_rate_control": "newbob", "learning_rate_control_error_measure": "dev_score"})
  lrc = load_learning_rate_control_from_config(config)
  assert isinstance(lrc, NewbobRelative)
  lrc.get_learning_rate_for_epoch(1)
  lrc.set_epoch_error(1, {"train_score": {'cost:output': 1.95, "cost:out2": 2.95}})
  lrc.set_epoch_error(1, {"dev_score": {'cost:output': 1.99, "cost:out2": 2.99},
                        "dev_error": {'error:output': 0.6, "error:out2": 0.7}})
  error = lrc.get_epoch_error_dict(1)
  assert "train_score_output" in error
  assert "train_score_out2" in error
  assert "dev_score_output" in error
  assert "dev_score_out2" in error
  assert "dev_error_output" in error
  assert "dev_error_out2" in error
  assert_equal(lrc.get_error_key(1), "dev_score_output")
  lrc.get_learning_rate_for_epoch(2)
  lrc.set_epoch_error(2, {"train_score": {'cost:output': 1.8, "cost:out2": 2.8}})
  lrc.set_epoch_error(2, {"dev_score": {'cost:output': 1.9, "cost:out2": 2.9},
                        "dev_error": {'error:output': 0.5, "error:out2": 0.6}})
  lrc.get_learning_rate_for_epoch(3)


def test_newbob():
  lr = 0.01
  config = Config()
  config.update({"learning_rate_control": "newbob", "learning_rate": lr})
  lrc = load_learning_rate_control_from_config(config)
  assert isinstance(lrc, NewbobRelative)
  assert_equal(lrc.get_learning_rate_for_epoch(1), lr)
  lrc.set_epoch_error(1, {"train_score": {'cost:output': 1.9344199658230012}})
  lrc.set_epoch_error(1, {"dev_score": {'cost:output': 1.99}, "dev_error": {'error:output': 0.6}})
  error = lrc.get_epoch_error_dict(1)
  assert "train_score" in error
  assert "dev_score" in error
  assert "dev_error" in error
  assert_equal(lrc.get_error_key(1), "dev_score")
  assert_equal(lrc.get_learning_rate_for_epoch(2), lr)  # epoch 2 cannot be a different lr yet
  lrc.set_epoch_error(2, {"train_score": {'cost:output': 1.8}})
  lrc.set_epoch_error(2, {"dev_score": {'cost:output': 1.9}, "dev_error": {'error:output': 0.5}})
  lrc.get_learning_rate_for_epoch(3)


def test_newbob_multi_epoch():
  lr = 0.0005
  config = Config()
  config.update({
    "learning_rate_control": "newbob_multi_epoch",
    "learning_rate_control_relative_error_relative_lr": True,
    "newbob_multi_num_epochs": 6,
    "newbob_multi_update_interval": 1,
    "learning_rate": lr})
  lrc = load_learning_rate_control_from_config(config)
  assert isinstance(lrc, NewbobMultiEpoch)
  assert_equal(lrc.get_learning_rate_for_epoch(1), lr)
  lrc.set_epoch_error(1, {
    'dev_error': 0.50283176046904721,
    'dev_score': 2.3209858321263455,
    'train_score': 3.095824052426714,
  })
  assert_equal(lrc.get_learning_rate_for_epoch(2), lr)  # epoch 2 cannot be a different lr yet


if __name__ == "__main__":
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
    print("Finished all tests.")
  else:
    assert len(sys.argv) >= 2
    for arg in sys.argv[1:]:
      print("Executing: %s" % arg)
      if arg in globals():
        globals()[arg]()  # assume function and execute
      else:
        eval(arg)  # assume Python code and execute
