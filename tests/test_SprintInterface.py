from __future__ import annotations

import sys
import os

import _setup_test_env  # noqa
import returnn.sprint.interface as SprintAPI
from returnn.tf.engine import Engine
from tempfile import mkdtemp
from returnn.config import Config
import shutil
import numpy


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


def test_forward():
    tmpdir = mkdtemp("returnn-test-sprint")
    olddir = os.getcwd()
    os.chdir(tmpdir)

    from returnn.datasets.generating import DummyDataset

    seq_len = 5
    n_data_dim = 2
    n_classes_dim = 3
    train_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=4, seq_len=seq_len)
    train_data.init_seq_order(epoch=1)
    cv_data = DummyDataset(input_dim=n_data_dim, output_dim=n_classes_dim, num_seqs=2, seq_len=seq_len)
    cv_data.init_seq_order(epoch=1)

    config = "\n".join(
        [
            "#!rnn.py",
            "use_tensorflow = True",
            "model = '%s/model'" % tmpdir,
            "num_outputs = %i" % n_classes_dim,
            "num_inputs = %i" % n_data_dim,
            'network = {"output": {"class": "softmax", "loss": "ce", "from": "data:data"}}',
            "num_epochs = 2",
        ]
    )

    open("config_write", "w").write(config)
    open("config", "w").write(config + "\nload= '%s/model'" % tmpdir)

    config = Config()
    config.load_file("config_write")
    engine = Engine(config=config)
    engine.init_train_from_config(config=config, train_data=train_data, dev_data=cv_data, eval_data=None)
    engine.epoch = 1
    engine.save_model(engine.get_epoch_model_filename())
    Engine._epoch_model = None

    # Reset engine
    from returnn.util.basic import BackendEngine

    BackendEngine.selected_engine = None

    inputDim = 2
    outputDim = 3
    SprintAPI.init(
        inputDim=inputDim,
        outputDim=outputDim,
        config="action:forward,configfile:config,epoch:1",
        targetMode="forward-only",
    )
    assert isinstance(SprintAPI.engine, Engine)
    print("used data keys via net:", SprintAPI.engine.network.get_used_data_keys())

    features = numpy.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
    seq_len = features.shape[0]
    posteriors = SprintAPI._forward("segment1", features.T).T
    assert posteriors.shape == (seq_len, outputDim)

    SprintAPI.exit()

    os.chdir(olddir)
    shutil.rmtree(tmpdir)
