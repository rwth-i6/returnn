
from NetworkHiddenLayer import ForwardLayer
from NetworkRecurrentLayer import RecurrentLayer, RecurrentUnitLayer
from NetworkLstmLayer import LstmLayer, OptimizedLstmLayer, FastLstmLayer, GRULayer, SRULayer, SRALayer


LayerClasses = {
  'forward': ForwardLayer,  # used in crnn.config format
  'hidden': ForwardLayer,  # used in JSON format
  'recurrent': RecurrentLayer,
  'lstm': LstmLayer,
  'gru': GRULayer,
  'sru': SRULayer,
  'sra': SRALayer,
  'rec' : RecurrentUnitLayer,
  'lstm_opt': OptimizedLstmLayer,
  'lstm_fast': FastLstmLayer
}


def get_layer_class(name):
  """
  :type name: str
  :rtype: type(NetworkHiddenLayer.HiddenLayer)
  """
  if name in LayerClasses:
    return LayerClasses[name]
  assert False, "invalid layer type: " + name
