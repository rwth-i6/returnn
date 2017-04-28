
import sys
sys.path += ["."]  # Python 3 hack

from nose.tools import assert_equal, assert_is, assert_in, assert_not_in, assert_true, assert_false
from NetworkLayer import LayerClasses
from NetworkHiddenLayer import ForwardLayer
from NetworkRecurrentLayer import RecurrentUnitLayer



def test_LayerClasses_list():
  # Don't need to be the complete list.
  # This is mostly for the layer_class cleanup, to check whether we still have all.

  assert_in("forward", LayerClasses)  # used in crnn.config format
  assert_in("hidden", LayerClasses)  # used in JSON format
  assert_in("recurrent", LayerClasses)
  assert_in('lstm', LayerClasses)
  assert_in('gru', LayerClasses)
  assert_in('sru', LayerClasses)
  assert_in('sra', LayerClasses)
  assert_in('chunking', LayerClasses)
  assert_in('dual', LayerClasses)
  assert_in("state_to_act", LayerClasses)
  assert_in("base", LayerClasses)
  assert_in('rec', LayerClasses)

  assert_is(LayerClasses["forward"], ForwardLayer)
  assert_is(LayerClasses["hidden"], ForwardLayer)  # alias
  assert_is(LayerClasses["rec"], RecurrentUnitLayer)
