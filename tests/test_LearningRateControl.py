
import sys
sys.path += ["."]  # Python 3 hack

from Config import Config
from LearningRateControl import *
from nose.tools import assert_equal

import better_exchook
better_exchook.replace_traceback_format_tb()

from Log import log
log.initialize()


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
