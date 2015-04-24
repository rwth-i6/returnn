
from NetworkHiddenLayer import ForwardLayer
from NetworkLstmLayer import LstmLayer, OptimizedLstmLayer, NormalizedLstmLayer, MaxLstmLayer, GateLstmLayer, \
  LstmPeepholeLayer
from NetworkRecurrentLayer import RecurrentLayer


LayerClasses = {
  'forward': ForwardLayer,  # used in crnn.config format
  'hidden': ForwardLayer,  # used in JSON format
  'recurrent': RecurrentLayer,
  'lstm': LstmLayer,
  'lstm_opt': OptimizedLstmLayer,
  'lstm_norm': NormalizedLstmLayer,
  'gatelstm': GateLstmLayer,
  'peep_lstm': LstmPeepholeLayer,
  'maxlstm': MaxLstmLayer
}


def get_layer_class(name):
  """
  :type name: str
  :rtype: type(NetworkHiddenLayer.HiddenLayer)
  """
  if name in LayerClasses:
    return LayerClasses[name]
  assert False, "invalid layer type: " + name
