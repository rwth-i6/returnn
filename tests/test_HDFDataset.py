from HDFDataset import HDFDataset
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import raises
import os


class TestHDFDataset(object):
  @classmethod
  def setup_class(cls):
    """
    :return:
     This method is run once before starting testing
    """

  @classmethod
  def teardown_class(cls):
    """
    This method is run once after completing all tests
    :return:
    """

  def setup(self):
    """
    This method is run before each test is going to be started
    :return:
    """

  def teardown(self):
    """
    This method is run after finishing of each test
    :return:
    """

  def test_init(self):
    """
    This method tests initialization of the HDFDataset class
    """
    toy_dataset = HDFDataset()
    assert_equal(toy_dataset.file_start, [0], "self.file_start init problem, should be [0]")
    assert_equal(toy_dataset.files, [], "self.files init problem, should be []")
    assert_equal(toy_dataset.file_seq_start, [], "self.file_seq_start init problem, should be []")
    return toy_dataset

  def test_addfile(self):
    """
    This method tests self.addfile function
    """
    toy_dataset = self.test_init()
    # TODO: auto-generate file, then use here
    #toy_dataset.add_file("/u/kulikov/develop/crnn/tests/toy_set.hdf")
