
LayerClasses = {}

def _initLayerClasses():
  global LayerClasses
  from inspect import isclass
  import NetworkHiddenLayer
  import NetworkRecurrentLayer
  import NetworkLstmLayer
  mods = [NetworkHiddenLayer, NetworkRecurrentLayer, NetworkLstmLayer]
  for mod in mods:
    for _, clazz in vars(mod).items():
      if not isclass(clazz): continue
      layer_class = getattr(clazz, "layer_class", None)
      if not layer_class: continue
      LayerClasses[layer_class] = clazz
  from NetworkHiddenLayer import ForwardLayer
  LayerClasses["forward"] = ForwardLayer  # used in crnn.config format

_initLayerClasses()

def get_layer_class(name):
  """
  :type name: str
  :rtype: type(NetworkHiddenLayer.HiddenLayer)
  """
  if name in LayerClasses:
    return LayerClasses[name]
  if name.startswith("config."):
    from Config import get_global_config
    config = get_global_config()
    cls = config.typed_value(name[len("config."):])
    import inspect
    assert inspect.isclass(cls), "get_layer_class: %s not found" % name
    return cls
  assert False, "get_layer_class: invalid layer type: %s" % name
