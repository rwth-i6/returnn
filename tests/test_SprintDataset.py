import _setup_test_env  # noqa
from nose.tools import assert_equal, assert_true
from returnn.engine.batch import Batch
from returnn.config import Config
import returnn.util.basic as util
from returnn.datasets.sprint import ExternSprintDataset
import os
import sys
import unittest
from returnn.util import better_exchook

dummyconfig_dict = {
    "num_inputs": 2,
    "num_outputs": 3,
    "hidden_size": (1,),
    "hidden_type": "forward",
    "activation": "relu",
    "bidirectional": False,
}


os.chdir((os.path.dirname(__file__) or ".") + "/..")
assert os.path.exists("rnn.py")
sprintExecPath = "tests/DummySprintExec.py"


def generate_batch(seq_idx, dataset):
    batch = Batch()
    batch.add_frames(seq_idx=seq_idx, seq_start_frame=0, length=dataset.get_seq_length(seq_idx))
    return batch


def test_read_all():
    config = Config()
    config.update(dummyconfig_dict)
    print("Create ExternSprintDataset")
    python_exec = util.which("python")
    if python_exec is None:
        raise unittest.SkipTest("python not found")
    num_seqs = 4
    dataset = ExternSprintDataset(
        [python_exec, sprintExecPath],
        "--*.feature-dimension=2 --*.trainer-output-dimension=3 "
        "--*.crnn-dataset=DummyDataset(2,3,num_seqs=%i,seq_len=10)" % num_seqs,
    )
    dataset.init_seq_order(epoch=1)
    seq_idx = 0
    while dataset.is_less_than_num_seqs(seq_idx):
        dataset.load_seqs(seq_idx, seq_idx + 1)
        for key in dataset.get_data_keys():
            value = dataset.get_data(seq_idx, key)
            print("seq idx %i, data %r: %r" % (seq_idx, key, value))
        seq_idx += 1
    assert seq_idx == num_seqs


def test_assign_dev_data():
    config = Config()
    config.update(dummyconfig_dict)
    print("Create ExternSprintDataset")
    dataset = ExternSprintDataset(
        [sys.executable, sprintExecPath],
        "--*.feature-dimension=2 --*.trainer-output-dimension=3 --*.crnn-dataset=DummyDataset(2,3,num_seqs=4,seq_len=10)",
    )
    dataset.init_seq_order(epoch=1)
    assert_true(dataset.is_less_than_num_seqs(0))
    recurrent = False
    batch_generator = dataset.generate_batches(recurrent_net=recurrent, batch_size=5)
    batches = batch_generator.peek_next_n(2)
    assert_equal(len(batches), 2)


def test_window():
    input_dim = 2
    output_dim = 3
    num_seqs = 2
    seq_len = 5
    window = 3
    dataset_kwargs = dict(
        sprintTrainerExecPath=[sys.executable, sprintExecPath],
        sprintConfigStr=" ".join(
            [
                "--*.feature-dimension=%i" % input_dim,
                "--*.trainer-output-dimension=%i" % output_dim,
                "--*.crnn-dataset=DummyDataset(input_dim=%i,output_dim=%i,num_seqs=%i,seq_len=%i)"
                % (input_dim, output_dim, num_seqs, seq_len),
            ]
        ),
    )
    dataset1 = ExternSprintDataset(**dataset_kwargs)
    dataset2 = ExternSprintDataset(window=window, **dataset_kwargs)
    try:
        dataset1.init_seq_order(epoch=1)
        dataset2.init_seq_order(epoch=1)
        dataset1.load_seqs(0, 1)
        dataset2.load_seqs(0, 1)
        assert_equal(dataset1.get_data_dim("data"), input_dim)
        assert_equal(dataset2.get_data_dim("data"), input_dim * window)
        data1 = dataset1.get_data(0, "data")
        data2 = dataset2.get_data(0, "data")
        assert_equal(data1.shape, (seq_len, input_dim))
        assert_equal(data2.shape, (seq_len, window * input_dim))
        data2a = data2.reshape(seq_len, window, input_dim)
        print("data1:")
        print(data1)
        print("data2:")
        print(data2)
        print("data1[0]:")
        print(data1[0])
        print("data2[0]:")
        print(data2[0])
        print("data2a[0,0]:")
        print(data2a[0, 0])
        assert_equal(list(data2a[0, 0]), [0] * input_dim)  # zero-padded left
        assert_equal(list(data2a[0, 1]), list(data1[0]))
        assert_equal(list(data2a[0, 2]), list(data1[1]))
        assert_equal(list(data2a[1, 0]), list(data1[0]))
        assert_equal(list(data2a[1, 1]), list(data1[1]))
        assert_equal(list(data2a[1, 2]), list(data1[2]))
        assert_equal(list(data2a[-1, 2]), [0] * input_dim)  # zero-padded right
    finally:
        dataset1._exit_handler()
        dataset2._exit_handler()


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
