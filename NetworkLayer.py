
from NetworkHiddenLayer import ForwardLayer, StateToAct
from NetworkRecurrentLayer import RecurrentLayer, RecurrentUnitLayer
from NetworkLstmLayer import LstmLayer, OptimizedLstmLayer, FastLstmLayer, SimpleLstmLayer, GRULayer, SRULayer, SRALayer


LayerClasses = {
  'forward': ForwardLayer,  # used in crnn.config format
  'hidden': ForwardLayer,  # used in JSON format
  'recurrent': RecurrentLayer,
  'lstm': LstmLayer,
  'gru': GRULayer,
  'sru': SRULayer,
  'sra': SRALayer,
  "state_to_act" : StateToAct,
  'rec' : RecurrentUnitLayer,
  'lstm_opt': OptimizedLstmLayer,
  'lstm_fast': FastLstmLayer,
  'lstm_simple': SimpleLstmLayer
}


def get_layer_class(name):
  """
  :type name: str
  :rtype: type(NetworkHiddenLayer.HiddenLayer)
  """
  if name in LayerClasses:
    return LayerClasses[name]
  assert False, "invalid layer type: %s" % name
