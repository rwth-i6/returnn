
from Network import LayerNetwork
from NetworkBaseLayer import Layer
from NetworkDescription import LayerNetworkDescription
from NetworkCopyUtils import intelli_copy_layer


class Pretrain:
  # Note: If we want to add other pretraining schemes, make this a base class.

  def __init__(self, original_network_description, copy_output_layer=None, greedy=None):
    """
    :type original_network_description: NetworkDescription.LayerNetworkDescription
    :param bool copy_output_layer: whether to copy the output layer params from last epoch or reinit
    :param bool greedy: if True, only train output+last layer, otherwise train all
    """
    self.original_network_description = original_network_description
    if copy_output_layer is None:
      copy_output_layer = True
    self.copy_output_layer = copy_output_layer
    if greedy is None:
      greedy = False
    self.greedy = greedy

  def __str__(self):
    return "Default layerwise construction+pretraining, starting with input+hidden+output. " + \
           "Epochs: %i" % self.get_train_num_epochs()

  def get_train_num_epochs(self):
    # Start with 1 hidden layers up to N hidden layers -> N epochs.
    # The first hidden layer is the input layer.
    return len(self.original_network_description.hidden_info)

  def get_network_description_for_epoch(self, epoch):
    """
    :type epoch: int
    :rtype: LayerNetworkDescription
    """
    description = self.original_network_description.copy()
    # We start with epoch 1. Start with 1 layer.
    description.hidden_info = description.hidden_info[:epoch]
    return description

  def get_network_for_epoch(self, epoch, mask="unity"):
    """
    :type epoch: int
    :rtype: Network.LayerNetwork
    """
    description = self.get_network_description_for_epoch(epoch)
    Layer.rng_seed = epoch
    return LayerNetwork.from_description(description, mask)

  def copy_params_from_old_network(self, new_network, old_network):
    """
    :type new_network: LayerNetwork
    :type old_network: LayerNetwork
    :returns the remaining hidden layer names which exist only in the new network.
    :rtype: set[str]
    """
    # network.hidden are the input + all hidden layers.
    for layer_name, layer in old_network.hidden.items():
      new_network.hidden[layer_name].set_params_by_dict(layer.get_params_dict())

    # network.output is the remaining output layer.
    if self.copy_output_layer:
      intelli_copy_layer(old_network.output, new_network.output)

  def get_train_param_args_for_epoch(self, epoch):
    """
    :type epoch: int
    :returns the kwargs for LayerNetwork.set_train_params, i.e. which params to train.
    :rtype: dict[str]
    """
    if not self.greedy:
      return {}  # This implies all available args.
    if epoch == 1:
      return {}  # This implies all available args.
    prev_network = self.get_network_for_epoch(epoch - 1)
    cur_network = self.get_network_for_epoch(epoch)
    prev_network_layer_names = prev_network.hidden.keys()
    cur_network_layer_names_set = set(cur_network.hidden.keys())
    assert cur_network_layer_names_set.issuperset(prev_network_layer_names)
    new_hidden_layer_names = cur_network_layer_names_set.difference(prev_network_layer_names)
    return {"hidden_layer_selection": new_hidden_layer_names, "with_output": True}


def pretrainFromConfig(config):
  """
  :type config: Config.Config
  :rtype: Pretrain | None
  """
  pretrainType = config.value("pretrain", "")
  if pretrainType == "default":
    assert config.network_topology_json is None, "Cannot handle JSON network topology in pretrain."
    original_network_description = LayerNetworkDescription.from_config(config)
    copy_output_layer = config.bool("pretrain_copy_output_layer", None)
    greedy = config.bool("pretrain_greedy", None)
    return Pretrain(original_network_description=original_network_description,
                    copy_output_layer=copy_output_layer, greedy=greedy)
  elif pretrainType == "":
    return None
  else:
    raise Exception, "unknown pretrain type: %s" % pretrainType
