
LayerClasses = {}

def _initLayerClasses():
  global LayerClasses
  from inspect import isclass
  import NetworkHiddenLayer
  import NetworkRecurrentLayer
  import NetworkLstmLayer
  import NetworkTwoDLayer
  import NetworkBaseLayer
  import NetworkCNNLayer
  from NetworkOutputLayer import FramewiseOutputLayer
  mods = [NetworkHiddenLayer, NetworkRecurrentLayer, NetworkLstmLayer, NetworkTwoDLayer, NetworkBaseLayer, NetworkCNNLayer]
  for mod in mods:
    for _, clazz in vars(mod).items():
      if not isclass(clazz): continue
      layer_class = getattr(clazz, "layer_class", None)
      if not layer_class: continue
      LayerClasses[layer_class] = clazz
  from NetworkHiddenLayer import ForwardLayer
  LayerClasses["forward"] = ForwardLayer  # used in crnn.config format
  LayerClasses["softmax"] = FramewiseOutputLayer

_initLayerClasses()

def get_layer_class(name, raise_exception=True):
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
    if not inspect.isclass(cls):
      if raise_exception:
        raise Exception("get_layer_class: %s not found" % name)
      else:
        return None
    if cls.layer_class is None:
      # Will make Layer.save() (to HDF) work correctly.
      cls.layer_class = name
    return cls
  if raise_exception:
    raise Exception("get_layer_class: invalid layer type: %s" % name)
  return None
