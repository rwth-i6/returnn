
from Config import Config
from LearningRateControl import NewbobRelative, loadLearningRateControlFromConfig
from nose.tools import assert_equal


def test_init_error_old():
  config = Config()
  config.update({"learning_rate_control": "newbob", "learning_rate_control_error_measure": "dev_score"})
  lrc = loadLearningRateControlFromConfig(config)
  assert isinstance(lrc, NewbobRelative)
  lrc.getLearningRateForEpoch(1)
  lrc.setEpochError(1, {"train_score": 1.9344199658230012})
  lrc.setEpochError(1, {"dev_score": 1.99, "dev_error": 0.6})
  error = lrc.getEpochErrorDict(1)
  assert "train_score" in error
  assert "dev_score" in error
  assert "dev_error" in error
  assert_equal(lrc.getErrorKey(1), "dev_score")
  lrc.getLearningRateForEpoch(2)
  lrc.setEpochError(2, {"train_score": 1.8})
  lrc.setEpochError(2, {"dev_score": 1.9, "dev_error": 0.5})
  lrc.getLearningRateForEpoch(3)


def test_init_error_new():
  config = Config()
  config.update({"learning_rate_control": "newbob", "learning_rate_control_error_measure": "dev_score"})
  lrc = loadLearningRateControlFromConfig(config)
  assert isinstance(lrc, NewbobRelative)
  lrc.getLearningRateForEpoch(1)
  lrc.setEpochError(1, {"train_score": {'cost:output': 1.9344199658230012}})
  lrc.setEpochError(1, {"dev_score": {'cost:output': 1.99}, "dev_error": {'error:output': 0.6}})
  error = lrc.getEpochErrorDict(1)
  assert "train_score" in error
  assert "dev_score" in error
  assert "dev_error" in error
  assert_equal(lrc.getErrorKey(1), "dev_score")
  lrc.getLearningRateForEpoch(2)
  lrc.setEpochError(2, {"train_score": {'cost:output': 1.8}})
  lrc.setEpochError(2, {"dev_score": {'cost:output': 1.9}, "dev_error": {'error:output': 0.5}})
  lrc.getLearningRateForEpoch(3)


def test_init_error_muliple_out():
  config = Config()
  config.update({"learning_rate_control": "newbob", "learning_rate_control_error_measure": "dev_score"})
  lrc = loadLearningRateControlFromConfig(config)
  assert isinstance(lrc, NewbobRelative)
  lrc.getLearningRateForEpoch(1)
  lrc.setEpochError(1, {"train_score": {'cost:output': 1.95, "cost:out2": 2.95}})
  lrc.setEpochError(1, {"dev_score": {'cost:output': 1.99, "cost:out2": 2.99},
                        "dev_error": {'error:output': 0.6, "error:out2": 0.7}})
  error = lrc.getEpochErrorDict(1)
  assert "train_score_output" in error
  assert "train_score_out2" in error
  assert "dev_score_output" in error
  assert "dev_score_out2" in error
  assert "dev_error_output" in error
  assert "dev_error_out2" in error
  assert_equal(lrc.getErrorKey(1), "dev_score_output")
  lrc.getLearningRateForEpoch(2)
  lrc.setEpochError(2, {"train_score": {'cost:output': 1.8, "cost:out2": 2.8}})
  lrc.setEpochError(2, {"dev_score": {'cost:output': 1.9, "cost:out2": 2.9},
                        "dev_error": {'error:output': 0.5, "error:out2": 0.6}})
  lrc.getLearningRateForEpoch(3)
