
from NetworkBaseLayer import Container
from nose.tools import assert_equal, assert_is_none

class TestContainer(object):

  def test_guess_source_layer_name(self):
    assert_equal(Container.guess_source_layer_name("hidden_1"), "hidden_0")
    assert_is_none(Container.guess_source_layer_name("hidden_0"))

