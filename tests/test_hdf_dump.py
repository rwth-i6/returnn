import hdf_dump
import nose.tools
from os import path
from os import remove
from Log import log


def setup_module():
    """
    this functions runs before the whole test module.
    here all theano stuff need to be loaded from virtualenv
    :return:
    """
    log.initialize()


def test_hdf_dataset_init():
    dataset_name = "nose_dataset.hdf"
    if path.exists(dataset_name):
        remove(dataset_name)
    hdf_dump.hdf_dataset_init(dataset_name)
    assert path.exists(dataset_name)

# TODO: Sprint subprocess fail to solve
def test_hdf_dump_from_dataset():
    dataset_name = "nose_dataset.hdf"
    #hdf_dump.init(configFilename='/u/kulikov/setups/quaero-en-50h/6x2048-theano-new/dump-dataset.crnn.config', commandLineOptions=[])
    pass


def teardown_module():
    dataset_name = "nose_dataset.hdf"
    if path.exists(dataset_name):
        remove(dataset_name)
    assert not path.exists(dataset_name)
