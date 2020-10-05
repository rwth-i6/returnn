
import sys
import os

import _setup_test_env  # noqa
from returnn.theano.layers.base import Container
from nose.tools import assert_equal, assert_is_none


class TestContainer(object):

  def test_guess_source_layer_name(self):
    assert_equal(Container.guess_source_layer_name("hidden_1"), "hidden_0")
    assert_equal(Container.guess_source_layer_name("hidden_11"), "hidden_10")
    assert_equal(Container.guess_source_layer_name("hidden_10"), "hidden_9")
    assert_is_none(Container.guess_source_layer_name("hidden_0"))
    assert_equal(Container.guess_source_layer_name("hidden_2_fw"), "hidden_1_fw")
    assert_equal(Container.guess_source_layer_name("hidden_20_fw"), "hidden_19_fw")
    assert_is_none(Container.guess_source_layer_name("hidden_0_bw"))
    assert_equal(Container.guess_source_layer_name("h_1_2"), "h_1_1")
    assert_equal(Container.guess_source_layer_name("h_10_20"), "h_10_19")
    assert_equal(Container.guess_source_layer_name("h_1_2_30_40_fw"), "h_1_2_30_39_fw")
