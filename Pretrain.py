
from Network import LayerNetwork
from NetworkBaseLayer import Layer
from NetworkCopyUtils import intelli_copy_layer, LayerDoNotMatchForCopy
from Log import log
from Util import unicode


class WrapEpochValue:
  """
  Use this wrapper if you want to define some value in your network
  which depends on the pretrain epoch.
  This is going to be part in your network description dict.
  """
  def __init__(self, func):
    self.func = func

  def get_value(self, epoch):
    return self.func(epoch=epoch)


def find_pretrain_wrap_values(net_json):
  assert isinstance(net_json, dict)
  def _check_dict(d):
    for k, v in sorted(d.items()):
      if isinstance(v, dict):
        if _check_dict(v):
          return True
      if isinstance(v, WrapEpochValue):
        return True
    return False
  return _check_dict(net_json)


class Pretrain:
  """
  Start with 1 hidden layers up to N hidden layers -> N pretrain steps -> N epochs (with repetitions == 1).
  The first hidden layer is the input layer.
  This works for generic network constructions. See _construct_epoch().
  """
  # Note: If we want to add other pretraining schemes, make this a base class.

  def __init__(self, original_network_json, network_init_args, copy_output_layer=None, greedy=None, repetitions=None, construction_algo=None):
    """
    :type original_network_json: dict[str]
    :param dict[str] network_init_args: additional args we use for LayerNetwork.from_json().
      must have n_in, n_out.
    :param bool|str copy_output_layer: whether to copy the output layer params from last epoch or reinit
    :param bool greedy: if True, only train output+last layer, otherwise train all
    :param None | int | list[int] | dict repetitions: how often to repeat certain pretrain steps. default is one epoch.
      It can also be a dict, with keys like 'default' and 'final'. See code below.
    :param str construction_algo: e.g. "from_output"
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
    self._original_network_json = original_network_json
    if not construction_algo:
      construction_algo = "from_output"
    self._construction_algo = construction_algo
    getattr(self, "_construct_epochs_%s" % construction_algo)()
    self._remove_non_trainable_added_only()
    if not repetitions:
      repetitions = 1
    if isinstance(repetitions, dict):
      rep_dict = repetitions
      default_rep = rep_dict.pop('default', 1)
      repetitions = [default_rep] * len(self._step_net_jsons)
      for k, v in sorted(rep_dict.items()):
        if k == "final":
          k = len(self._step_net_jsons) - 1
        repetitions[k] = v
    else:
      if not isinstance(repetitions, list):
        assert isinstance(repetitions, (int, long))
        repetitions = [repetitions]
      assert isinstance(repetitions, list)
      assert 0 < len(repetitions) <= len(self._step_net_jsons)
      if len(repetitions) < len(self._step_net_jsons):
        repetitions = repetitions + [repetitions[-1]] * (len(self._step_net_jsons) - len(repetitions))
    self.repetitions = repetitions
    self._make_repetitions()
    self._resolve_wrapped_values()

  def _remove_non_trainable_added_only(self):
    """
    If from one epoch to the next, only non-trainable layers were added, remove this pretrain epoch.
    Output layers are ignored.
    Also handles first epoch.
    """
    assert self._step_net_jsons
    old_net_jsons = self._step_net_jsons
    self._step_net_jsons = []
    # -1 will be the empty net. Until one before final, which we will always add.
    for i in range(-1, len(old_net_jsons) - 2):
      if i == -1:
        net1, net2 = {}, old_net_jsons[0]
      else:
        net1, net2 = old_net_jsons[i:i+2]
      assert isinstance(net1, dict)
      assert isinstance(net2, dict)
      for l in sorted(net1.keys()):
        assert l in net2
      have_new = False
      have_new_trainable = False
      for l in sorted(net2.keys()):
        if self._is_layer_output(net2, l): continue  # ignore output layers
        if l in net1: continue  # already had before
        have_new = True
        if net2[l].get("trainable", True):
          have_new_trainable = True
          break
      #assert have_new, "i: %i,\nold: %r,\nnew: %r" % (i, sorted(net1.keys()), sorted(net2.keys()))
      if have_new_trainable:
        self._step_net_jsons.append(net2)
    # Always add final net.
    self._step_net_jsons.append(old_net_jsons[-1])

  def _make_repetitions(self):
    assert len(self.repetitions) == len(self._step_net_jsons)
    from copy import deepcopy
    old_net_jsons = self._step_net_jsons
    self._step_net_jsons = []
    for n_rep, net_json in zip(self.repetitions, old_net_jsons):
      for i in range(n_rep):
        self._step_net_jsons.append(deepcopy(net_json))

  def _resolve_wrapped_values(self):
    def _check_dict(d, epoch):
      for k, v in sorted(d.items()):
        assert isinstance(k, (str, unicode))
        if isinstance(v, dict):
          _check_dict(v, epoch=epoch)
        if isinstance(v, WrapEpochValue):
          d[k] = v.get_value(epoch=epoch)
    for i, net_json in enumerate(self._step_net_jsons):
      epoch = i + 1
      _check_dict(net_json, epoch=epoch)

  def _find_layer_descendants(self, json, sources):
    l = []
    for other_layer_name, other_layer in sorted(json.items()):
      if other_layer_name in l:
        continue
      other_sources = other_layer.get("from", ["data"])
      for src in sources:
        if src in other_sources:
          l.append(other_layer_name)
          break
    return l

  def _is_layer_output(self, json, layer_name):
    if layer_name == "output":
      return True
    if json[layer_name]["class"] == "softmax":
      return True
    if "target" in json[layer_name]:
      return True
    return False

  def _find_layer_outputs(self, json, sources):
    outs = []
    visited = set()
    while sources:
      visited.update(sources)
      for src in sources:
        if src in outs:
          continue
        if self._is_layer_output(json, src):
          outs.append(src)
      sources = self._find_layer_descendants(self._original_network_json, sources)
      for v in visited:
        if v in sources:
          sources.remove(v)
    return outs

  def _find_existing_inputs(self, json, layer_name):
    l = []
    sources = self._original_network_json[layer_name].get("from", ["data"])
    for src in sources:
      if src in json or src == "data":
        if src not in l:
          l.append(src)
      else:
        for csrc in self._find_existing_inputs(json, src):
          if csrc not in l:
            l.append(csrc)
    return l

  def _construct_next_epoch_from_input(self, num_steps):
    """
    First find all layers which have data as input.
    Then expand from those layers.
    """
    from copy import deepcopy
    new_net = {}
    sources = ["data"]
    # Keep track of other layers which need to be added to make it complete.
    needed = set()
    def update_needed(l):
      needed.update(set(new_net[l].get("from", ["data"])).difference(list(new_net.keys()) + ["data"]))
    # First search for non-trainable layers (like input windows or so).
    # You must specify "trainable": False in the layer at the moment.
    while True:
      descendants = self._find_layer_descendants(self._original_network_json, sources)
      added_something = False
      for l in descendants:
        if l in new_net:
          continue
        if self._original_network_json[l].get("trainable", True):
          continue
        if l in needed:
          needed.remove(l)
        added_something = True
        sources.append(l)
        new_net[l] = deepcopy(self._original_network_json[l])
        update_needed(l)
      if not added_something:
        break
    # First do a search of depth `num_steps` through the net.
    for i in range(num_steps):
      descendants = self._find_layer_descendants(self._original_network_json, sources)
      sources = []
      for l in descendants:
        if l in new_net:
          continue
        if l in needed:
          needed.remove(l)
        sources.append(l)
        new_net[l] = deepcopy(self._original_network_json[l])
        update_needed(l)
      if not sources:  # This means we reached the end.
        return False
    # Add all output layers.
    for l in sorted(self._original_network_json.keys()):
      if l in new_net:
        continue
      if not self._is_layer_output(self._original_network_json, l):
        continue
      if l in needed:
        needed.remove(l)
      new_net[l] = deepcopy(self._original_network_json[l])
      update_needed(l)
    if not needed:  # Nothing needed anymore, i.e. no missing layers, i.e. we arrived at the final network topology.
      return False
    # Now fill in all missing ones.
    for l in sorted(new_net.keys()):
      sources = new_net[l].get("from", ["data"])
      sources2 = self._find_existing_inputs(new_net, l)
      if sources != sources2:
        if "data" in sources2:
          sources2.remove("data")
        new_net[l]["from"] = sources2
    self._step_net_jsons.append(new_net)
    return True

  def _construct_epochs_from_input(self):
    self._step_net_jsons = []
    num_steps = 1
    while self._construct_next_epoch_from_input(num_steps):
      num_steps += 1
    # Just add the original net at the end.
    self._step_net_jsons.append(self._original_network_json)

  def _construct_new_epoch_from_output(self):
    """
    We start from the most simple network which we have constructed so far,
    and try to construct an even simpler network.
    """
    from copy import deepcopy
    new_json = deepcopy(self._step_net_jsons[0])
    while True:
      assert "output" in new_json
      # From the sources of the output layer, collect all their sources.
      # Then remove the direct output sources and replace them with the indirect sources.
      new_sources = set()
      deleted_sources = set()
      for source in new_json["output"]["from"]:
        # Except for data sources. Just keep them.
        if source == "data":
          new_sources.add("data")
        else:
          assert source in new_json, "error %r, n: %i, last: %s" % (source, len(self._step_net_jsons), self._step_net_jsons[0])
          new_sources.update(new_json[source].get("from", ["data"]))
          del new_json[source]
          deleted_sources.add(source)
      # Check if anything changed.
      # This is e.g. not the case if the only source was data.
      if list(sorted(new_sources)) == list(sorted(new_json["output"]["from"])):
        return False
      new_json["output"]["from"] = list(sorted(new_sources))
      # If we have data input, it likely means that the input dimension
      # for the output layer would change. Just avoid that for now.
      if "data" in new_sources:
        # Try again.
        continue
      # If all deleted sources were non-trainable, skip this.
      if all(not self._original_network_json[del_source].get("trainable", True) for del_source in deleted_sources):
        # Try again.
        continue
      self._step_net_jsons = [new_json] + self._step_net_jsons
      return True

  def _construct_epochs_from_output(self):
    self._step_net_jsons = [self._original_network_json]
    while self._construct_new_epoch_from_output():
      pass

  def _construct_epochs_no_network_modifications(self):
    self._step_net_jsons = [self._original_network_json]

  # -------------- Public interface

  def __str__(self):
    return ("Default layerwise construction+pretraining, starting with input+hidden+output. " +
            "Number of pretrain epochs: %i (repetitions: %r)") % (
            self.get_train_num_epochs(), self.repetitions)

  def get_train_num_epochs(self):
    return len(self._step_net_jsons)

  def get_final_network_json(self):
    return self._step_net_jsons[-1]

  def get_network_json_for_epoch(self, epoch):
    """
    :param int epoch: starting at 1
    :rtype: dict[str]
    """
    assert epoch >= 1
    if epoch > len(self._step_net_jsons):
      epoch = len(self._step_net_jsons)  # take the last, which is the original
    return self._step_net_jsons[epoch - 1]

  def get_network_for_epoch(self, epoch, mask=None):
    """
    :type epoch: int
    :rtype: Network.LayerNetwork
    """
    json_content = self.get_network_json_for_epoch(epoch)
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
    if config.is_typed("pretrain_repetitions"):
      repetitions = config.typed_value("pretrain_repetitions")
    else:
      repetitions = config.int_list("pretrain_repetitions", None)
    construction_algo = config.value("pretrain_construction_algo", None)
    return Pretrain(original_network_json=original_network_json,
                    network_init_args=network_init_args,
                    copy_output_layer=copy_output_layer,
                    greedy=greedy, repetitions=repetitions,
                    construction_algo=construction_algo)
  elif pretrainType == "":
    return None
  else:
    raise Exception("unknown pretrain type: %s" % pretrainType)


def demo():
  import better_exchook
  better_exchook.install()
  import rnn
  import sys
  if len(sys.argv) <= 1:
    print("usage: python %s [config] [other options]" % __file__)
    print("example usage: python %s ++pretrain default ++pretrain_construction_algo from_input" % __file__)
  rnn.initConfig(commandLineOptions=sys.argv[1:])
  rnn.config._hack_value_reading_debug()
  if not rnn.config.value("pretrain", ""):
    print("config option 'pretrain' not set, will set it for this demo to 'default'")
    rnn.config.set("pretrain", "default")
  pretrain = pretrainFromConfig(rnn.config)
  print("pretrain: %s" % pretrain)
  num_pretrain_epochs = pretrain.get_train_num_epochs()
  from pprint import pprint
  for epoch in range(1, 1 + num_pretrain_epochs):
    print("epoch %i (of %i) network json:" % (epoch, num_pretrain_epochs))
    net_json = pretrain.get_network_json_for_epoch(epoch)
    pprint(net_json)
  print("done.")


if __name__ == "__main__":
  import sys
  sys.modules["Pretrain"] = sys.modules["__main__"]
  demo()
