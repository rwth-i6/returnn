import hdf_dump
import nose.tools
from os import path
from os import remove


def setup_module():
    """
    this functions runs before the whole test module.
    here all theano stuff need to be loaded from virtualenv
    :return:
    """
    pass


# TODO: Log.V3 is a problem for nose, figure out why
def test_hdf_dataset_init():
    dataset_name = "nose_dataset.hdf"
    hdf_dump.hdf_dataset_init(dataset_name)
    assert path.exists(dataset_name)


def teardown_module():
    dataset_name = "nose_dataset.hdf"
    remove(dataset_name)
    assert not path.exists(dataset_name)