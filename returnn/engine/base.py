"""
Provides :class:`EngineBase`.
"""

from __future__ import annotations

from typing import Optional, Set
import os
import sys

from returnn.config import Config, get_global_config
from returnn.learning_rate_control import load_learning_rate_control_from_config, LearningRateControl
from returnn.log import log
from returnn.pretrain import Pretrain
from returnn.util import basic as util
from returnn.forward_iface import ForwardCallbackIface
from returnn.datasets import Dataset


class EngineBase:
    """
    Base class for a backend engine, such as :class:`TFEngine.Engine`.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        :param config:
        """
        if config is None:
            config = get_global_config(auto_create=True)
        self.config = config
        self.epoch = 0
        self.global_train_step = None  # type: Optional[int]
        self.pretrain = None  # type: Optional[Pretrain]
        self.model_filename = None  # type: Optional[str]
        self.learning_rate = 0.0  # set in init_train_epoch
        self.learning_rate_control = None  # type: Optional[LearningRateControl]

    def init_network_from_config(self, config: Optional[Config] = None):
        """
        Initialize network/model

        :param config:
        """

    def init_train_from_config(self, config: Optional[Config] = None):
        """
        Initialize all engine parts needed for training

        :param config:
        """
        if not config:
            config = self.config
        self.learning_rate_control = load_learning_rate_control_from_config(config)
        self.learning_rate = self.learning_rate_control.default_learning_rate

    @classmethod
    def config_get_final_epoch(cls, config):
        """
        :param returnn.config.Config config:
        :rtype: int
        """
        num_epochs = config.int("num_epochs", 5)
        if config.has("load_epoch"):
            num_epochs = max(num_epochs, config.int("load_epoch", 0))
        return num_epochs

    @classmethod
    def get_existing_models(cls, config: Config, *, for_training: Optional[bool] = None):
        """
        :param config:
        :param for_training: if True, will only return models which are suitable for resuming training.
            E.g. in case of PyTorch, it means that the optimizer state should be present.
            By default, will be True if the task is "train".
        :return: dict epoch -> model filename (without extension)
        :rtype: dict[int,str]
        """
        model_filename = config.value("model", "")
        if not model_filename:
            return {}
        # Automatically search the filesystem for existing models.
        file_list = {}
        if for_training is None:
            for_training = config.value("task", "train") == "train"
        for epoch in range(1, cls.config_get_final_epoch(config) + 1):
            for is_pretrain in [False, True] if util.BackendEngine.is_tensorflow_selected() else [False]:
                fn = cls.epoch_model_filename(model_filename, epoch, is_pretrain=is_pretrain)
                if os.path.exists(fn):
                    file_list[epoch] = fn
                    break
                if util.BackendEngine.is_tensorflow_selected():
                    if os.path.exists(fn + ".index"):
                        file_list[epoch] = fn
                        break
                elif util.BackendEngine.is_torch_selected():
                    if os.path.exists(fn + ".pt"):
                        if for_training:
                            # In case of training, we only want to consider the model if the optimizer state exists.
                            if not os.path.exists(fn + ".opt.pt"):
                                continue
                        file_list[epoch] = fn
                        break
        return file_list

    @classmethod
    def get_start_epoch_no_existing_model(cls, config: Config) -> int:
        """
        :return: start epoch if no model exists
        """
        start_epoch_mode = config.value("start_epoch", "auto")
        if start_epoch_mode == "auto":
            return 1
        else:
            start_epoch = int(start_epoch_mode)
            assert start_epoch >= 1
            return start_epoch

    @classmethod
    def get_epoch_model(cls, config: Config):
        """
        :return: (epoch, model_filename). epoch is the epoch of the model filename.
        :rtype: (int|None, str|None)
        """
        start_epoch_mode = config.value("start_epoch", "auto")
        if start_epoch_mode == "auto":
            start_epoch = None
        else:
            start_epoch = int(start_epoch_mode)
            assert start_epoch >= 1

        load_model_epoch_filename = util.get_checkpoint_filepattern(config.value("load", ""))
        if load_model_epoch_filename:
            assert os.path.exists(load_model_epoch_filename + util.get_model_filename_postfix()) or os.path.exists(
                load_model_epoch_filename
            ), "config option load=%r, file %r does not exist" % (
                config.value("load", ""),
                load_model_epoch_filename + util.get_model_filename_postfix(),
            )

        import_model_train_epoch1 = util.get_checkpoint_filepattern(config.value("import_model_train_epoch1", ""))
        if import_model_train_epoch1:
            assert os.path.exists(import_model_train_epoch1 + util.get_model_filename_postfix())

        existing_models = cls.get_existing_models(config)
        load_epoch = config.int("load_epoch", -1)
        if load_model_epoch_filename:
            if load_epoch <= 0:
                load_epoch = util.model_epoch_from_filename(load_model_epoch_filename)
        else:
            if load_epoch > 0:  # ignore if load_epoch == 0
                assert load_epoch in existing_models
                load_model_epoch_filename = existing_models[load_epoch]
                assert util.model_epoch_from_filename(load_model_epoch_filename) == load_epoch

        # Only use this when we don't train.
        # For training, we first consider existing models
        # before we take the 'load' into account when in auto epoch mode.
        # In all other cases, we use the model specified by 'load'.
        if load_model_epoch_filename and (config.value("task", "train") != "train" or start_epoch is not None):
            if config.value("task", "train") == "train" and start_epoch is not None:
                # Ignore the epoch. To keep it consistent with the case below.
                epoch = None
            else:
                epoch = load_epoch
            epoch_model = (epoch, load_model_epoch_filename)

        # In case of training, always first consider existing models.
        # This is because we reran RETURNN training, we usually don't want to train from scratch
        # but resume where we stopped last time.
        elif existing_models:
            epoch_model = sorted(existing_models.items())[-1]
            if load_model_epoch_filename:
                print("note: there is a 'load' which we ignore because of existing model", file=log.v4)

        elif config.value("task", "train") == "train" and import_model_train_epoch1 and start_epoch in [None, 1]:
            epoch_model = (0, import_model_train_epoch1)

        # Now, consider this also in the case when we train, as an initial model import.
        elif load_model_epoch_filename:
            # Don't use the model epoch as the start epoch in training.
            # We use this as an import for training.
            epoch_model = (load_epoch, load_model_epoch_filename)

        else:
            epoch_model = (None, None)

        if start_epoch == 1:
            if epoch_model[0]:  # existing model
                print("warning: there is an existing model: %s" % (epoch_model,), file=log.v4)
                epoch_model = (None, None)
        elif (start_epoch or 0) > 1:
            if epoch_model[0]:
                if epoch_model[0] != start_epoch - 1:
                    print("warning: start_epoch %i but there is %s" % (start_epoch, epoch_model), file=log.v4)
                epoch_model = start_epoch - 1, existing_models[start_epoch - 1]

        return epoch_model

    @classmethod
    def get_train_start_epoch(cls, config: Config) -> int:
        """
        We will always automatically determine the best start (epoch,batch) tuple
        based on existing model files.
        This ensures that the files are present and enforces that there are
        no old outdated files which should be ignored.
        Note that epochs start at idx 1 and batches at idx 0.

        :param config:
        :return: epoch
        """
        start_batch_mode = config.value("start_batch", "auto")
        if start_batch_mode != "auto":
            raise Exception(f"custom start_batch {start_batch_mode!r} not supported")
        last_epoch, _ = cls.get_epoch_model(config)
        if last_epoch is None:
            start_epoch = 1
        else:
            # Start with next epoch.
            start_epoch = last_epoch + 1
        return start_epoch

    @classmethod
    def epoch_model_filename(cls, model_filename: str, epoch: int, *, is_pretrain: bool = False) -> str:
        """
        :param model_filename:
        :param epoch:
        :param is_pretrain:
        """
        if sys.platform == "win32" and model_filename.startswith("/tmp/"):
            import tempfile

            model_filename = tempfile.gettempdir() + model_filename[len("/tmp") :]
        return model_filename + (".pretrain" if is_pretrain else "") + ".%03d" % epoch

    def get_epoch_model_filename(self, epoch=None):
        """
        :param int|None epoch:
        :return: filename, excluding TF specific postfix
        :rtype: str
        """
        if not epoch:
            epoch = self.epoch
        return self.epoch_model_filename(self.model_filename, epoch, is_pretrain=self.is_pretrain_epoch(epoch=epoch))

    def get_epoch_str(self):
        """
        :return: e.g. "epoch 3", or "pretrain epoch 5"
        :rtype: str
        """
        return ("pretrain " if self.is_pretrain_epoch() else "") + "epoch %s" % self.epoch

    def is_pretrain_epoch(self, epoch=None):
        """
        :param int|None epoch:
        :return: whether this epoch is covered by the pretrain logic
        :rtype: bool
        """
        if not epoch:
            epoch = self.epoch
        return self.pretrain and epoch <= self.pretrain.get_train_num_epochs()

    def is_first_epoch_after_pretrain(self):
        """
        :return: whether the current epoch is the first epoch right after pretraining
        :rtype: bool
        """
        return self.pretrain and self.epoch == self.pretrain.get_train_num_epochs() + 1

    def forward_with_callback(self, *, dataset: Dataset, callback: ForwardCallbackIface):
        """
        Iterate through the dataset, calling `forward_step` from user config,
        collecting outputs in `rf.get_run_ctx()` via `mark_as_output` calls,
        and then calling `callback` for each entry.
        """
        raise NotImplementedError

    def _do_save(self):
        """
        :return: whether to perform save on disk in this process. e.g. for Horovod rank != 0, do not save.
        :rtype: bool
        """
        import returnn.util.basic

        return returnn.util.basic.should_write_to_disk(config=self.config)

    @staticmethod
    def delete_model(filename):
        """
        :param str filename:
        :return: accumulated file-size in bytes of deleted files
        :rtype: int
        """
        raise NotImplementedError

    def cleanup_old_models(self, ask_for_confirmation=False):
        """
        :param bool ask_for_confirmation: if True, will ask the user interactively to confirm
        """
        if not self._do_save():
            return
        from returnn.util.basic import CollectionReadCheckCovered, human_bytes_size, confirm
        from returnn.util.math import next_power_of_two
        from itertools import count

        opts = CollectionReadCheckCovered(self.config.get_of_type("cleanup_old_models", dict, {}))
        existing_models = self.get_existing_models(config=self.config, for_training=False)
        if self.learning_rate_control is not None:
            lr_control = self.learning_rate_control
        else:
            lr_control = load_learning_rate_control_from_config(self.config)
        epochs = sorted(existing_models.keys())
        if not epochs:
            print("Cannot cleanup models, no models found.", file=log.v2)
            return
        keep_last_n = opts.get("keep_last_n", 2)
        keep_best_n = opts.get("keep_best_n", 4)
        assert keep_last_n >= 1 and keep_best_n >= 0
        if max(keep_last_n, keep_best_n) >= len(epochs):
            print(
                (
                    "Only %i epochs stored so far and keeping last %i epochs and best %i epochs,"
                    " thus not cleaning up any epochs yet."
                )
                % (len(epochs), keep_last_n, keep_best_n),
                file=log.v2,
            )
            return
        keep_epochs = set()  # type: Set[int]
        default_keep_pattern = set()
        if epochs[-1] <= 10:
            keep_every = 4
            keep_doubles_of = 5
        elif epochs[-1] <= 50:
            keep_every = 20
            keep_doubles_of = 5
        elif epochs[-1] <= 100:
            keep_every = 40
            keep_doubles_of = 10
        else:
            keep_every = 80 * next_power_of_two(1 + epochs[-1] // 240)
            keep_doubles_of = 20
        for i in count(1):
            n = keep_every * i
            if n > epochs[-1]:
                break
            default_keep_pattern.add(n)
        for i in count():
            n = keep_doubles_of * (2**i)
            if n > epochs[-1]:
                break
            default_keep_pattern.add(n)
        keep_epochs.update(opts.get("keep", default_keep_pattern))
        keep_epochs.update(epochs[-keep_last_n:])
        score_keys = set()  # e.g. "dev_error", "dev_score", etc.
        # Collect all possible score keys. Note that we could have different ones for different epochs.
        for data in lr_control.epoch_data.values():
            score_keys.update(data.error.keys())
        assert score_keys
        score_keys = sorted(score_keys)
        score_values = {key: [] for key in score_keys}
        for epoch in epochs:
            epoch_scores = lr_control.epoch_data[epoch].error
            for key in epoch_scores.keys():
                score_values[key].append(epoch_scores[key])
        for key in list(score_keys):
            scores = score_values[key]
            if min(scores) == max(scores):
                print(
                    "Ignoring score key %r because all epochs have the same value %r." % (key, scores[0]), file=log.v3
                )
                score_keys.remove(key)
                score_values.pop(key)
        # Actually, terminology is a bit confusing. We call it "score" here (and elsewhere), but it's a loss,
        # so the maximum value is the worst possible value.
        worst_score_values = {key: max(scores) for (key, scores) in score_values.items()}
        for key in score_keys:
            scores = sorted(
                [(lr_control.epoch_data[epoch].error.get(key, worst_score_values[key]), epoch) for epoch in epochs]
            )
            scores = scores[:keep_best_n]
            keep_epochs.update([v[1] for v in scores])
        keep_epochs.intersection_update(epochs)
        if len(keep_epochs) == len(epochs):
            print("%i epochs stored so far and keeping all." % len(epochs), file=log.v2)
            return
        remove_epochs = sorted(set(epochs).difference(keep_epochs))
        assert remove_epochs
        if len(epochs) > 6:
            epoch_summary = "[%s, ..., %s]" % (", ".join(map(str, epochs[:3])), ", ".join(map(str, epochs[-3:])))
        else:
            epoch_summary = str(epochs)
        print(
            "We have stored models for epochs %s and keep epochs %s." % (epoch_summary, sorted(keep_epochs)),
            file=log.v3,
        )
        print("We will delete the models of epochs %s." % (remove_epochs,), file=log.v3)
        opts.assert_all_read()
        if self.config.bool("dry_run", False):
            print("Dry-run, will not delete models.", file=log.v2)
            return
        if ask_for_confirmation:
            confirm("Delete those models?", exit_on_false=True)
        count_bytes = 0
        for epoch in remove_epochs:
            count_bytes += self.delete_model(existing_models[epoch])
        print("Deleted %s." % human_bytes_size(count_bytes), file=log.v2)

    def _is_dataset_evaluated(self, name: str, *, epoch: Optional[int] = None) -> bool:
        """
        Check via self.learning_rate_control.

        :param name: e.g. "dev"
        :return: whether there is an entry for the score in the learning rate file
        """
        assert self.learning_rate_control.filename  # otherwise we would not have stored it
        if epoch is None:
            epoch = self.epoch
        error_dict = self.learning_rate_control.get_epoch_error_dict(epoch)
        if not error_dict:
            return False
        return any([k.startswith("%s_score" % name) or k.startswith("%s_loss" % name) for k in error_dict.keys()])
