"""
tests if the graph values of two different training runs on are equal
can be used to test determinism of demo training on GPU and CPU.
in some cases the test will give a false negative (on GPU)

state 16.04.19 following demos are confirmed not deterministic on GPU:
with: CUDA 10.0, tf: 1.13.0 nvcc:

- demos/demo-tf-vanilla-lstm.12ax.config
- demos/demo-tf-native-lstm-lowmem.12ax.config

- demos/demo-tf-native-lstm.12ax.config

this test can be used on all lstm demos:

to run this test you might need to apply workaround: https://github.com/rwth-i6/returnn/issues/87
Example:

  python3 tests/test_TF_determinismus.py "<democonfig-location>"



"""

import sys
import os

import _setup_test_env  # noqa
from returnn.tf.engine import *
from returnn.config import Config

from numpy.testing import assert_array_equal, assert_equal

config = Config()
if len(sys.argv) == 1:
    config.load_file("demos/demo-tf-vanilla-lstm.12ax.config")
else:
    print(str(sys.argv[1]))
    config.load_file(sys.argv[1])


def load_data():
    from returnn.__main__ import load_data

    dev_data, _ = load_data(
        config, 0, "dev", chunking=config.value("chunking", ""), seq_ordering="sorted", shuffle_frames_of_nseqs=0
    )
    eval_data, _ = load_data(
        config, 0, "eval", chunking=config.value("chunking", ""), seq_ordering="sorted", shuffle_frames_of_nseqs=0
    )
    train_data, _ = load_data(config, 0, "train")
    return dev_data, eval_data, train_data


def test_determinism_of_vanillalstm():
    def create_engine():
        dev_data, eval_data, train_data = load_data()

        engine = Engine()
        engine.init_train_from_config(config, train_data, dev_data, eval_data)
        engine.init_train_epoch()
        engine.train_batches = engine.train_data.generate_batches(
            recurrent_net=engine.network.recurrent,
            batch_size=engine.batch_size,
            max_seqs=engine.max_seqs,
            max_seq_length=engine.max_seq_length,
            seq_drop=engine.seq_drop,
            shuffle_batches=engine.shuffle_batches,
            used_data_keys=engine.network.used_data_keys,
        )
        engine.updater.set_learning_rate(engine.learning_rate, session=engine.tf_session)
        engine.updater.init_optimizer_vars(session=engine.tf_session)
        return engine

    def train_engine_fetch_vars(engine):
        data_provider = engine._get_data_provider(
            dataset=engine.train_data, batches=engine.train_batches, feed_dict=True
        )
        feed_dict, _ = data_provider.get_feed_dict(single_threaded=True)
        trainer = Runner(engine=engine, dataset=engine.train_data, batches=engine.train_batches, train=True)
        feed_dict, _ = data_provider.get_feed_dict(single_threaded=True)
        trainer.run(report_prefix="One Run")

        return [e.eval(engine.tf_session) for e in engine.network.get_params_list()]

    e1 = create_engine()
    r1 = train_engine_fetch_vars(e1)

    e2 = create_engine()
    r2 = train_engine_fetch_vars(e2)

    assert_array_equal(r1, r2)


test_determinism_of_vanillalstm()
