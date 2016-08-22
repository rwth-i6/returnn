
from Network import LayerNetwork
from NetworkBaseLayer import Layer
from NetworkCopyUtils import intelli_copy_layer, LayerDoNotMatchForCopy
from Log import log


class Pretrain:
  """
  Start with 1 hidden layers up to N hidden layers -> N pretrain steps -> N epochs (with repetitions == 1).
  The first hidden layer is the input layer.
  This works for generic network constructions. See _construct_epoch().
  """
  # Note: If we want to add other pretraining schemes, make this a base class.

  def __init__(self, original_network_json, network_init_args, copy_output_layer=None, greedy=None, repetitions=None):
    """
    :type original_network_json: dict[str]
    :param dict[str] network_init_args: additional args we use for LayerNetwork.from_json().
      must have n_in, n_out.
    :param bool|str copy_output_layer: whether to copy the output layer params from last epoch or reinit
    :param bool greedy: if True, only train output+last layer, otherwise train all
    :param None | int | list[int] repetitions: how often to repeat certain pretrain steps. default is one epoch
    """
    if copy_output_layer is None:
      copy_output_layer = "ifpossible"
    if copy_output_layer:
      assert copy_output_layer is True or copy_output_layer == "ifpossible"
    self.copy_output_layer = copy_output_layer
    if greedy is None:
      greedy = False
    self.greedy = greedy
    self.network_init_args = network_init_args
    assert "n_in" in network_init_args
    assert "n_out" in network_init_args
    self._step_net_jsons = [original_network_json]
    self._construct_epochs()
    if not repetitions:
      repetitions = 1
    if not isinstance(repetitions, list):
      repetitions = [repetitions]
    assert isinstance(repetitions, list)
    assert 0 < len(repetitions) <= len(self._step_net_jsons)
    if len(repetitions) < len(self._step_net_jsons):
      repetitions = repetitions + [repetitions[-1]] * (len(self._step_net_jsons) - len(repetitions))
    assert len(repetitions) == len(self._step_net_jsons)
    self.repetitions = repetitions
    self._epoch_to_step_idx = []
    for i, n in enumerate(repetitions):
      self._epoch_to_step_idx += [i] * n

  def _get_network_json_for_epoch(self, epoch):
    """
    :param int epoch: starting at 1
    :rtype: dict[str]
    """
    if epoch > len(self._epoch_to_step_idx):
      epoch = len(self._epoch_to_step_idx)  # take the last, which is the original
    step = self._epoch_to_step_idx[epoch - 1]
    return self._step_net_jsons[step]

  def _construct_epoch(self):
    """
    We start from the most simple network which we have constructed so far,
    and try to construct an even simpler network.
    """
    from copy import deepcopy
    new_json = deepcopy(self._step_net_jsons[0])
    assert "output" in new_json
    # From the sources of the output layer, collect all their sources.
    # Then remove the direct output sources and replace them with the indirect sources.
    new_sources = set()
    for source in new_json["output"]["from"]:
      # Except for data sources. Just keep them.
      if source == "data":
        new_sources.add("data")
      else:
        assert source in new_json, "error %r, n: %i, last: %s" % (source, len(self._step_net_jsons), self._step_net_jsons[0])
        new_sources.update(new_json[source].get("from", ["data"]))
        del new_json[source]
    # Check if anything changed.
    # This is e.g. not the case if the only source was data.
    if list(sorted(new_sources)) == list(sorted(new_json["output"]["from"])):
      return False
    # If we have data input, it likely means that the input dimension
    # for the output layer would change. Just avoid that for now.
    if "data" in new_sources:
      return False
    new_json["output"]["from"] = list(sorted(new_sources))
    self._step_net_jsons = [new_json] + self._step_net_jsons
    return True

  def _construct_epochs(self):
    while self._construct_epoch():
      pass

  # -------------- Public interface

  def __str__(self):
    return ("Default layerwise construction+pretraining, starting with input+hidden+output. " +
            "Number of models topologies (including final): %i, " +
            "Number of train epochs: %i, Repetitions: %r") % (
            len(self._step_net_jsons), self.get_train_num_epochs(), self.repetitions)

  def get_train_num_epochs(self):
    return len(self._epoch_to_step_idx)

  def get_network_for_epoch(self, epoch, mask=None):
    """
    :type epoch: int
    :rtype: Network.LayerNetwork
    """
    json_content = self._get_network_json_for_epoch(epoch)
    Layer.rng_seed = epoch
    return LayerNetwork.from_json(json_content, mask=mask, **self.network_init_args)

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
      for layer_name in new_network.output.keys():
        assert layer_name in old_network.output
        try:
          intelli_copy_layer(old_network.output[layer_name], new_network.output[layer_name])
        except LayerDoNotMatchForCopy:
          if self.copy_output_layer == "ifpossible":
            print >>log.v4, "Pretrain: Can not copy output layer %s, will leave it randomly initialized" % layer_name
          else:
            raise
    else:
      print >>log.v4, "Pretrain: Will not copy output layer"

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
    network_init_args = LayerNetwork.init_args_from_config(config)
    original_network_json = LayerNetwork.json_from_config(config)
    copy_output_layer = config.bool_or_other("pretrain_copy_output_layer", "ifpossible")
    greedy = config.bool("pretrain_greedy", None)
    repetitions = config.int_list("pretrain_repetitions", None)
    return Pretrain(original_network_json=original_network_json,
                    network_init_args=network_init_args,
                    copy_output_layer=copy_output_layer,
                    greedy=greedy, repetitions=repetitions)
  elif pretrainType == "":
    return None
  else:
    raise Exception("unknown pretrain type: %s" % pretrainType)
