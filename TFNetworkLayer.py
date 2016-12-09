
import tensorflow as tf


class Data(object):
  def __init__(self, name,
               shape=None, dtype=None,
               placeholder=None,
               sparse=None,
               dim=None,
               size_placeholder=None,
               auto_create_placeholders=True):
    """
    :param str name:
    :param tuple[int|None] shape: including time-dim, which can be None. excluding batch-dim
    :param str dtype: e.g. "float32" or "int64"
    :param tf.Tensor|None placeholder: with added batch-dim
    :param bool sparse: whether to treat the value as an index
    :param None|int dim: shape[-1] if not sparse, otherwise like num_classes
    :param dict[int,tf.Tensor] tf.Tensor size_placeholder: for every None in shape, this will describe the size
    """
    self.name = name
    if sparse is None:
      if dtype and dtype.startswith("int"):
        sparse = True
      elif name == "classes":
        sparse = True
      elif shape is not None and len(shape) == 1:
        sparse = True
      if sparse is None:
        sparse = False
    self.sparse = sparse
    if shape is None:
      assert dim, "no shape specified, need dim"
      if sparse:
        shape = (None,)  # assume common (time,)
      else:
        shape = (None, dim)  # assume common (time,dim)
    self.shape = shape
    if dtype is None:
      if sparse:
        dtype = "int32"
      else:
        dtype = "float32"
    if dim is None:
      assert not sparse, "need dim"
      dim = shape[-1]
    self.feature_dim = dim
    self.dtype = dtype
    if placeholder is None and auto_create_placeholders:
      with tf.name_scope("extern_data/placeholders/%s/" % name):
        placeholder = tf.placeholder(name=name, dtype=dtype, shape=(None,) + shape)
    self.placeholder = placeholder
    if size_placeholder is None and auto_create_placeholders:
      size_placeholder = {}  # type: dict[int,tf.Tensor]
      with tf.name_scope("extern_data/placeholders/%s/" % name):
        for i, dim in enumerate(shape):
          if dim is None:
            size_placeholder[i] = tf.placeholder(
              name="%s_dim%i_size" % (name, i), dtype="int64", shape=(None,))
    self.size_placeholder = size_placeholder

  def get_description(self, with_name=True, with_placeholder=False):
    keys = ["shape", "dtype", "sparse"]
    if with_name:
      keys.insert(0, "name")
    if with_placeholder:
      keys.append("placeholder")
    return "Data(%s)" % ", ".join(["%s=%r" % (key, getattr(self, key)) for key in keys])


class LayerBase(object):
  layer_class = None

  def __init__(self, name, network, n_out=None, out_type=None, sources=()):
    """
    :param str name:
    :param TFNetwork.TFNetwork network:
    :param None|int n_out: output dim
    :param dict[str] out_type:
    :param list[LayerBase] sources:
    """
    self.name = name
    self.network = network
    if out_type is None:
      assert n_out
      out_type = {"dim": n_out}
    out_type = out_type.copy()
    out_type.setdefault("name", "%s_output" % self.name)
    if n_out is not None:
      out_type.setdefault("dim", n_out)
      assert out_type["dim"] == n_out
    self.output = Data(auto_create_placeholders=False, **out_type)
    self.sources = sources
    self.params = {}  # type: dict[str,tf.Variable]


class SourceLayer(LayerBase):
  layer_class = "source"

  def __init__(self, data_key=None, **kwargs):
    super(SourceLayer, self).__init__(**kwargs)
    if data_key is None:
      data_key = self.network.extern_data.default_input
    assert not self.sources, "source layer does not expect sources"
    self.output = self.network.extern_data.get_data(data_key)


class SoftmaxLayer(LayerBase):
  def __init__(self, **kwargs):
    super(SoftmaxLayer, self).__init__(**kwargs)


_layer_class_dict = {}  # type: dict[str,type(LayerBase)]

def _init_layer_class_dict():
  for v in globals().values():
    layer_class = getattr(v, "layer_class", None)
    if layer_class:
      assert layer_class not in _layer_class_dict
      _layer_class_dict[layer_class] = v


def get_layer_class(name):
  """
  :param str name: matches layer_class
  :rtype: () -> LayerBase
  """
  if not _layer_class_dict:
    _init_layer_class_dict()
  if name not in _layer_class_dict:
    raise Exception("unknown layer class %r" % name)
  return _layer_class_dict[name]

