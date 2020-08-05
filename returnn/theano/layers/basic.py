
LayerClasses = {}


def _init_layer_classes():
  global LayerClasses
  from inspect import isclass
  from . import hidden
  from . import rec
  from . import lstm
  from . import twod
  from . import base
  from . import cnn
  from .output import FramewiseOutputLayer
  mods = [hidden, rec, lstm, twod, base, cnn]
  for mod in mods:
    for _, clazz in vars(mod).items():
      if not isclass(clazz): continue
      layer_class = getattr(clazz, "layer_class", None)
      if not layer_class: continue
      LayerClasses[layer_class] = clazz
  from .hidden import ForwardLayer
  LayerClasses["forward"] = ForwardLayer  # used in returnn.config format
  LayerClasses["softmax"] = FramewiseOutputLayer


_init_layer_classes()


def get_layer_class(name, raise_exception=True):
  """
  :type name: str
  :rtype: type(NetworkHiddenLayer.HiddenLayer)
  """
  if name in LayerClasses:
    return LayerClasses[name]
  if name.startswith("config."):
    from returnn.config import get_global_config
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
