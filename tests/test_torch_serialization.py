"""
Checkpoint serialization tests.
"""

import _setup_test_env  # noqa
import os
from unittest.mock import patch
import torch
import pytest


@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
def test_save_checkpoint_exhausts_retries(error_type, tmp_path):
    from returnn.torch.util.serialization import save_checkpoint

    filename = str(tmp_path / "model.pt")
    torch.save({"old": True}, filename)
    error = error_type("persistent write failure")
    with patch("torch.save", side_effect=error) as save, patch("time.sleep") as sleep:
        with pytest.raises(error_type) as caught:
            save_checkpoint({"new": True}, filename)
    assert caught.value is error
    assert save.call_count == 3
    assert sleep.call_count == 2
    assert torch.load(filename) == {"old": True}


def test_save_checkpoint_does_not_retry_value_error(tmp_path):
    from returnn.torch.util.serialization import save_checkpoint

    with patch("torch.save", side_effect=ValueError("invalid object")) as save, patch("time.sleep") as sleep:
        with pytest.raises(ValueError):
            save_checkpoint({}, str(tmp_path / "model.pt"))
    assert save.call_count == 1
    sleep.assert_not_called()


def test_save_checkpoint_replaces_stale_temp_file(tmp_path):
    from returnn.torch.util.serialization import save_checkpoint

    filename = str(tmp_path / "model.pt")
    (tmp_path / "model.pt.tmp_write").write_bytes(b"stale partial checkpoint")
    with patch("time.sleep") as sleep:
        save_checkpoint({"new": True}, filename)
    assert torch.load(filename) == {"new": True}
    assert not os.path.exists(filename + ".tmp_write")
    sleep.assert_not_called()


@pytest.mark.parametrize("kind", ["model", "optimizer"])
def test_checkpoint_callers_retry(kind, tmp_path):
    from returnn.config import Config
    from returnn.torch.engine import Engine
    from returnn.torch.updater import Updater

    model = torch.nn.Linear(2, 3)
    config = Config({"device": "cpu", "optimizer": {"class": "adam"}})
    filename = str(tmp_path / "checkpoint.pt")
    if kind == "model":
        engine = Engine(config=config)
        engine._pt_model = model
        engine.epoch = 1
        engine.get_epoch_model_filename = lambda: filename[:-3]
        engine._do_save = lambda: True
        save = engine._save_model
    else:
        updater = Updater(config=config, network=model, device=torch.device("cpu"))
        updater.create_optimizer()

        def save():
            updater.save_optimizer(filename)

    original_save = torch.save
    attempts = []

    def fail_once(obj, path):
        attempts.append(path)
        if len(attempts) == 1:
            raise OSError("transient write failure")
        original_save(obj, path)

    with patch("torch.save", fail_once), patch("time.sleep"):
        save()
    assert len(attempts) == 2
    assert kind in torch.load(filename)


def test_save_checkpoint_retries_partial_write(tmp_path):
    from returnn.torch.util.serialization import save_checkpoint

    filename = str(tmp_path / "model.pt")
    torch.save({"old": True}, filename)
    original_save = torch.save
    attempts = []

    def save_with_failure(obj, path):
        assert not os.path.exists(path)
        assert torch.load(filename) == {"old": True}
        attempts.append(path)
        if len(attempts) < 3:
            with open(path, "wb") as f:
                f.write(b"partial checkpoint")
            if len(attempts) == 1:
                raise OSError("transient filesystem failure")
            raise RuntimeError("PytorchStreamWriter failed writing file data/0: file write failed")
        original_save(obj, path)

    with patch("torch.save", save_with_failure), patch("time.sleep") as sleep:
        save_checkpoint({"new": torch.tensor([3])}, filename)
    assert len(attempts) == 3
    assert sleep.call_count == 2
    assert torch.load(filename)["new"].tolist() == [3]
    assert not os.path.exists(filename + ".tmp_write")
