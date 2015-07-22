
from Log import log


def intelli_copy_layer(old_layer, new_layer):
  """
  :type old_layer: NetworkBaseLayer.Layer
  :type old_layer: NetworkBaseLayer.Layer

  Copies from old_layer to new_layer.
  """

  # We support slightly different param names. That can happen because the param names
  # could encode the source/target layer number, e.g. named "hidden_N".
  # Thus we need to translate the parameter names for the new network.

  # For the translation, we expect that a sorted list of the old output source layer names
  # matches the related list of new output source layer names.
  assert len(old_layer.params.keys()) == len(new_layer.params.keys())
  old_output_param_names = sorted(old_layer.params.keys())
  new_output_param_names = sorted(new_layer.params.keys())
  assert len(old_output_param_names) == len(new_output_param_names)
  new_output_param_name_map = {old_param_name: new_param_name
                               for old_param_name, new_param_name in zip(old_output_param_names,
                                                                         new_output_param_names)}
  print >> log.v5, "Copy map: %s" % sorted(new_output_param_name_map.items())
  old_output_params = old_layer.get_params_dict()
  new_output_params = {new_output_param_name_map[old_param_name]: param
                       for old_param_name, param in old_output_params.items()}
  new_layer.set_params_by_dict(new_output_params)
