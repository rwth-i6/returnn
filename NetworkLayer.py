
from NetworkHiddenLayer import ForwardLayer
from NetworkRecurrentLayer import RecurrentLayer
from NetworkLstmLayer import LstmLayer, OptimizedLstmLayer, GRULayer, SRULayer, SRALayer


LayerClasses = {
  'forward': ForwardLayer,  # used in crnn.config format
  'hidden': ForwardLayer,  # used in JSON format
  'recurrent': RecurrentLayer,
  'lstm': LstmLayer,
  'gru': GRULayer,
  'sru': SRULayer,
  'sra': SRALayer,
  'lstm_opt': OptimizedLstmLayer,
}


def get_layer_class(name):
  """
  :type name: str
  :rtype: type(NetworkHiddenLayer.HiddenLayer)
  """
  if name in LayerClasses:
    return LayerClasses[name]
  assert False, "invalid layer type: " + name
