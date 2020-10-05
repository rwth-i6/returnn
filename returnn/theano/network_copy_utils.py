
from __future__ import print_function
from returnn.log import log


class LayerDoNotMatchForCopy(Exception):
  pass


def intelli_copy_layer(old_layer, new_layer):
  """
  :type old_layer: NetworkBaseLayer.Layer
  :type new_layer: NetworkBaseLayer.Layer

  Copies from old_layer to new_layer.

  We support slightly different param names. That can happen because the param names
  could encode the source/target layer number, e.g. named "hidden_N".
  Thus we need to translate the parameter names for the new network.

  For the translation, we expect that a sorted list of the old output source layer names
  matches the related list of new output source layer names.
  """

  old_output_param_names = sorted(old_layer.params.keys())
  new_output_param_names = sorted(new_layer.params.keys())
  if len(old_output_param_names) != len(new_output_param_names):
    raise LayerDoNotMatchForCopy("num parameters do not match. old layer: %s, new layer: %s" %
                                 (old_output_param_names, new_output_param_names))
  new_output_param_name_map = {old_param_name: new_param_name
                               for old_param_name, new_param_name in zip(old_output_param_names,
                                                                         new_output_param_names)}
  print("Copy map: %s" % sorted(new_output_param_name_map.items()), file=log.v5)
  old_output_params = old_layer.get_params_dict()
  new_output_params = {new_output_param_name_map[old_param_name]: param
                       for old_param_name, param in old_output_params.items()}
  for p, v in new_output_params.items():
    self_param_shape = new_layer.params[p].get_value(borrow=True, return_internal_type=True).shape
    if self_param_shape != v.shape:
      raise LayerDoNotMatchForCopy(
        "In %s, param %s shape does not match. Expected (new layer) %s, got (old layer) %s." %
        (new_layer, p, self_param_shape, v.shape))
    new_layer.params[p].set_value(v, borrow=True)
