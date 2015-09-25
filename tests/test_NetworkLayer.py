
from nose.tools import assert_equal, assert_is, assert_in, assert_not_in, assert_true, assert_false
from NetworkLayer import LayerClasses
from NetworkHiddenLayer import ForwardLayer, StateToAct, BaseInterpolationLayer, ChunkingLayer, DualStateLayer
from NetworkRecurrentLayer import RecurrentLayer, RecurrentUnitLayer
from NetworkLstmLayer import LstmLayer, OptimizedLstmLayer, FastLstmLayer, SimpleLstmLayer, GRULayer, SRULayer, SRALayer



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
  assert_in('lstm_opt', LayerClasses)
  assert_in('lstm_fast', LayerClasses)
  assert_in('lstm_simple', LayerClasses)

  assert_is(LayerClasses["forward"], ForwardLayer)
  assert_is(LayerClasses["recurrent"], RecurrentLayer)
  assert_is(LayerClasses["rec"], RecurrentUnitLayer)
  assert_is(LayerClasses["gru"], GRULayer)

