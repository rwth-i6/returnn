"""
This module contains pretrain related code.
This is independent from the backend (TF or Theano, etc).
"""

from __future__ import annotations

from returnn.util.basic import unicode, long


class WrapEpochValue:
    """
    Use this wrapper if you want to define some value in your network
    which depends on the pretrain epoch.
    This is going to be part in your network description dict.
    """

    def __init__(self, func):
        """
        :param ((epoch: int) -> object) func: function which should accept one kwd-arg 'epoch'
        """
        self.func = func

    def get_value(self, epoch):
        """
        :param int epoch:
        :return: anything, whatever self.func returns
        :rtype: object
        """
        return self.func(epoch=epoch)


def find_pretrain_wrap_values(net_json):
    """
    See also :func:`Pretrain._resolve_wrapped_values`.
    Recursively goes through dicts, tuples and lists.
    This is a simple check to see if this is needed,
    i.e. if there are any :class:`WrapEpochValue` used.

    :param dict[str] net_json: network dict
    :return: whether there is some :class:`WrapEpochValue` in it
    :rtype: bool
    """
    assert isinstance(net_json, dict)

    def _check(d):
        if isinstance(d, WrapEpochValue):
            return True
        if isinstance(d, dict):
            for k, v in sorted(d.items()):
                if _check(v):
                    return True
        if isinstance(d, (tuple, list)):
            for v in d:
                if _check(v):
                    return True
        return False

    return _check(net_json)


class Pretrain:
    """
    Start with 1 hidden layers up to N hidden layers -> N pretrain steps -> N epochs (with repetitions == 1).
    The first hidden layer is the input layer.
    This works for generic network constructions. See _construct_epoch().
    """

    # Note: If we want to add other pretraining schemes, make this a base class.

    def __init__(
        self,
        original_network_json,
        network_init_args=None,
        copy_param_mode=None,
        copy_output_layer=None,
        greedy=None,
        repetitions=None,
        construction_algo="from_output",
        output_layers=("output",),
        input_layers=("data",),
    ):
        """
        :type original_network_json: dict[str]
        :param dict[str]|None network_init_args: additional args we use for LayerNetwork.from_json().
            must have n_in, n_out. (for Theano only, thus optional now)
        :param str copy_param_mode:
        :param bool|str copy_output_layer: whether to copy the output layer params from last epoch or reinit
        :param bool greedy: if True, only train output+last layer, otherwise train all
        :param None | int | list[int] | dict repetitions: how often to repeat certain pretrain steps.
            default is one epoch.
            It can also be a dict, with keys like 'default' and 'final'. See code below.
        :param str|callable construction_algo: e.g. "from_output"
        :param list[str]|tuple[str] output_layers: used for construction
        :param list[str]|tuple[str] input_layers: used for construction
        """
        assert copy_param_mode in [None, "ifpossible", "subset", "reset"]
        if copy_output_layer is None:
            copy_output_layer = copy_param_mode
        if copy_output_layer is None:
            copy_output_layer = "ifpossible"
        if copy_output_layer == "reset":
            copy_output_layer = False
        if copy_output_layer:
            assert copy_output_layer is True or copy_output_layer in ["ifpossible", "subset"]
        self.copy_param_mode = copy_param_mode
        self.copy_output_layer = copy_output_layer
        if greedy is None:
            greedy = False
        self.greedy = greedy
        self.network_init_args = network_init_args
        self._original_network_json = original_network_json
        self._construction_algo = construction_algo
        self._input_layers = input_layers
        self._output_layers = output_layers
        if construction_algo == "from_input":
            self._construct_epochs_from_input()
        elif construction_algo == "from_output":
            self._construct_epochs_from_output()
        elif callable(construction_algo):
            self._construct_epochs_custom(construction_algo)
        elif construction_algo == "no_network_modifications":
            self._construct_epochs_no_network_modifications()
        else:
            raise Exception("invalid construction_algo %r" % construction_algo)
        if not callable(construction_algo):  # if callable, trust the user
            self._remove_non_trainable_added_only()
        if not repetitions:
            repetitions = 1
        if isinstance(repetitions, dict):
            rep_dict = repetitions.copy()
            default_rep = rep_dict.pop("default", 1)
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
        assert len(repetitions) == len(self._step_net_jsons)
        for i, net_dict in enumerate(self._step_net_jsons):
            if "#repetition" in net_dict:
                repetitions[i] = net_dict.pop("#repetition")
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
                net1, net2 = old_net_jsons[i : i + 2]
            assert isinstance(net1, dict)
            assert isinstance(net2, dict)
            for layer_name in sorted(net1.keys()):
                assert layer_name in net2
            have_new_trainable = False
            for layer_name in sorted(net2.keys()):
                if self._is_layer_output(net2, layer_name):
                    continue  # ignore output layers
                if layer_name in net1:
                    continue  # already had before
                if net2[layer_name].get("trainable", True):
                    have_new_trainable = True
                    break
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
            for _ in range(n_rep):
                self._step_net_jsons.append(deepcopy(net_json))

    def _resolve_wrapped_values(self):
        """
        Resolves any :class:`WrapEpochValue` in the net dicts.
        Recursively goes through dicts, tuples and lists.
        See also :func:`find_pretrain_wrap_values`.
        """

        # noinspection PyShadowingNames
        def _check_dict(d, epoch, depth=0):
            for k, v in sorted(d.items()):
                if depth <= 1:  # 0 - layers, 1 - layer opts
                    assert isinstance(k, (str, unicode))
                d[k] = _check(v, epoch=epoch, depth=depth + 1)

        # noinspection PyShadowingNames
        def _check(v, epoch, depth):
            """
            :param WrapEpochValue|tuple|list|dict|T v:
            :param int epoch:
            :param int depth:
            :rtype: T
            """
            if isinstance(v, WrapEpochValue):
                return v.get_value(epoch=epoch)
            if isinstance(v, (tuple, list)):
                if not any([isinstance(x, WrapEpochValue) for x in v]):
                    return v
                return type(v)([_check(x, epoch=epoch, depth=depth + 1) for x in v])
            if isinstance(v, dict):
                _check_dict(v, epoch=epoch, depth=depth)
                return v
            return v

        for i, net_json in enumerate(self._step_net_jsons):
            epoch = i + 1
            _check_dict(net_json, epoch=epoch)

    # noinspection PyMethodMayBeStatic
    def _find_layer_descendants(self, json, sources):
        ls = []
        for other_layer_name, other_layer in sorted(json.items()):
            if other_layer_name in ls:
                continue
            other_sources = other_layer.get("from", ["data"])
            for src in sources:
                if src in other_sources:
                    ls.append(other_layer_name)
                    break
        return ls

    def _is_layer_output(self, json, layer_name):
        if layer_name in self._output_layers:
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

    def _find_existing_inputs(self, json, layer_name, _collected=None, _visited=None):
        if _collected is None:
            _collected = []
        if _visited is None:
            _visited = {layer_name: None}
        sources = self._original_network_json[layer_name].get("from", ["data"])
        for src in sources:
            if src in json or src == "data":
                if src not in _collected:
                    _collected.append(src)
            else:
                if src not in _visited:
                    _visited[src] = layer_name
                    self._find_existing_inputs(json=json, layer_name=src, _collected=_collected, _visited=_visited)
        return _collected

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

        # noinspection PyShadowingNames
        def update_needed(layer_name):
            """
            :param str layer_name:
            """
            needed.update(set(new_net[layer_name].get("from", ["data"])).difference(list(new_net.keys()) + ["data"]))

        # First search for non-trainable layers (like input windows or so).
        # You must specify "trainable": False in the layer at the moment.
        while True:
            descendants = self._find_layer_descendants(self._original_network_json, sources)
            added_something = False
            for layer_name in descendants:
                if layer_name in new_net:
                    continue
                if self._original_network_json[layer_name].get("trainable", True):
                    continue
                if layer_name in needed:
                    needed.remove(layer_name)
                added_something = True
                sources.append(layer_name)
                new_net[layer_name] = deepcopy(self._original_network_json[layer_name])
                update_needed(layer_name)
            if not added_something:
                break
        # First do a search of depth `num_steps` through the net.
        for _ in range(num_steps):
            descendants = self._find_layer_descendants(self._original_network_json, sources)
            sources = []
            for layer_name in descendants:
                if layer_name in new_net:
                    continue
                if layer_name in needed:
                    needed.remove(layer_name)
                sources.append(layer_name)
                new_net[layer_name] = deepcopy(self._original_network_json[layer_name])
                update_needed(layer_name)
            if not sources:  # This means we reached the end.
                return False
        # Add all output layers.
        for layer_name in sorted(self._original_network_json.keys()):
            if layer_name in new_net:
                continue
            if not self._is_layer_output(self._original_network_json, layer_name):
                continue
            if layer_name in needed:
                needed.remove(layer_name)
            new_net[layer_name] = deepcopy(self._original_network_json[layer_name])
            update_needed(layer_name)
        if not needed:  # Nothing needed anymore, i.e. no missing layers, i.e. we arrived at the final network topology.
            return False
        # Now fill in all missing ones.
        for layer_name in sorted(new_net.keys()):
            sources = new_net[layer_name].get("from", ["data"])
            sources2 = self._find_existing_inputs(new_net, layer_name)
            if sources != sources2:
                if "data" in sources2:
                    sources2.remove("data")
                new_net[layer_name]["from"] = sources2
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
            for out_layer_name in self._output_layers:
                assert out_layer_name in new_json
            # From the sources of the output layer, collect all their sources.
            # Then remove the direct output sources and replace them with the indirect sources.
            new_sources = set()
            deleted_sources = set()
            for out_layer_name in self._output_layers:
                for source in new_json[out_layer_name]["from"]:
                    # Except for data sources. Just keep them.
                    if source in self._input_layers:
                        new_sources.add(source)
                    else:
                        assert source in new_json, "error %r, n: %i, last: %s" % (
                            source,
                            len(self._step_net_jsons),
                            self._step_net_jsons[0],
                        )
                        new_sources.update(new_json[source].get("from", ["data"]))
                        del new_json[source]
                        deleted_sources.add(source)
            # Check if anything changed.
            # This is e.g. not the case if the only source was data.
            if list(sorted(new_sources)) == list(
                sorted(set(sum([new_json[name]["from"] for name in self._output_layers], [])))
            ):
                return False
            for out_layer_name in self._output_layers:
                new_json[out_layer_name]["from"] = list(sorted(new_sources))
            # If we have data input, it likely means that the input dimension
            # for the output layer would change. Just avoid that for now.
            if new_sources.intersection(set(self._input_layers)):
                # Try again.
                continue
            # If all deleted sources were non-trainable, skip this.
            if all(
                not self._original_network_json[del_source].get("trainable", True) for del_source in deleted_sources
            ):
                # Try again.
                continue
            self._step_net_jsons = [new_json] + self._step_net_jsons
            return True

    def _construct_epochs_from_output(self):
        self._step_net_jsons = [self._original_network_json]
        while self._construct_new_epoch_from_output():
            pass

    def _construct_epochs_custom(self, func):
        """
        :param ((idx: int, net_dict: dict[str,dict[str]]) -> dict[str,dict[str]]|None) func:
          ``func`` can work inplace on net_dict and should then return it.
          If ``None`` is returned, it will stop with the construction.
          The original network will always be added at the end.
        """
        from copy import deepcopy

        self._step_net_jsons = []
        idx = 0
        while True:
            d = func(idx=idx, net_dict=deepcopy(self._original_network_json))
            if not d:
                break
            self._step_net_jsons.append(d)
            idx += 1
        self._step_net_jsons.append(self._original_network_json)

    def _construct_epochs_no_network_modifications(self):
        self._step_net_jsons = [self._original_network_json]

    # -------------- Public interface

    def __str__(self):
        parts = [
            "Pretrain construction algo %r," % self._construction_algo,
            "number of pretrain epochs: %i" % self.get_train_num_epochs(),
        ]
        rep_set = set(self.repetitions)
        if rep_set != {1}:
            if len(rep_set) == 1:
                parts.append("(repetitions: %i)" % self.repetitions[0])
            else:
                parts.append("(repetitions: %r)" % self.repetitions)
        return " ".join(parts)

    def get_train_num_epochs(self):
        """
        :rtype: int
        """
        return len(self._step_net_jsons)

    def get_final_network_json(self):
        """
        :rtype: dict[str,dict[str]]
        """
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
        raise NotImplementedError("This feature was removed with dropped Theano support")


def pretrain_from_config(config):
    """
    :type config: returnn.config.Config
    :rtype: Pretrain | None
    """
    from returnn.config import network_json_from_config

    pretrain_type = config.bool_or_other("pretrain", None)
    if pretrain_type == "default" or (isinstance(pretrain_type, dict) and pretrain_type) or pretrain_type is True:
        network_init_args = None
        original_network_json = network_json_from_config(config)
        opts = config.get_of_type("pretrain", dict, {})
        if config.has("pretrain_copy_output_layer"):
            opts.setdefault("copy_output_layer", config.bool_or_other("pretrain_copy_output_layer", "ifpossible"))
        if config.has("pretrain_greedy"):
            opts.setdefault("greedy", config.bool("pretrain_greedy", None))
        if config.has("pretrain_repetitions"):
            if config.is_typed("pretrain_repetitions"):
                opts.setdefault("repetitions", config.typed_value("pretrain_repetitions"))
            else:
                opts.setdefault("repetitions", config.int_list("pretrain_repetitions", None))
        if config.has("pretrain_construction_algo"):
            opts.setdefault("construction_algo", config.value("pretrain_construction_algo", None))
        return Pretrain(original_network_json=original_network_json, network_init_args=network_init_args, **opts)
    elif not pretrain_type:
        return None
    else:
        raise Exception("unknown pretrain type: %s" % pretrain_type)


def demo():
    """
    Will print out the different network topologies of the specified pretraining scheme.
    """
    import returnn.util.better_exchook

    returnn.util.better_exchook.install()
    import returnn.__main__ as rnn
    import argparse
    from returnn.util.basic import obj_diff_str

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config")
    arg_parser.add_argument("--diff", action="store_true", help="show diff only")
    arg_parser.add_argument("other_returnn_args", nargs=argparse.REMAINDER, help="config updates or so")
    args = arg_parser.parse_args()
    rnn.init_config(
        config_filename=args.config, command_line_options=args.other_returnn_args, extra_updates={"log": []}
    )
    # noinspection PyProtectedMember
    rnn.config._hack_value_reading_debug()
    rnn.init_log()
    if not rnn.config.value("pretrain", ""):
        print("config option 'pretrain' not set, will set it for this demo to 'default'")
        rnn.config.set("pretrain", "default")
    pretrain = pretrain_from_config(rnn.config)
    print("pretrain: %s" % pretrain)
    num_pretrain_epochs = pretrain.get_train_num_epochs()
    last_net_json = None
    from pprint import pprint

    for epoch in range(1, 1 + num_pretrain_epochs):
        print("epoch %i (of %i) network json:" % (epoch, num_pretrain_epochs))
        net_json = pretrain.get_network_json_for_epoch(epoch)
        if args.diff:
            if last_net_json is not None:
                print(obj_diff_str(last_net_json, net_json))
            else:
                print("(initial)")
        else:
            pprint(net_json)
        last_net_json = net_json
    print("done.")


if __name__ == "__main__":
    import sys
    import os

    __package__ = "returnn"  # pylint: disable=redefined-builtin
    my_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(my_dir))
    if "" in sys.path:
        sys.path.remove("")
    if my_dir in sys.path:
        sys.path.remove(my_dir)
    import returnn

    sys.modules["returnn.pretrain"] = sys.modules["__main__"]
    try:
        demo()
    except BrokenPipeError:
        print("BrokenPipeError")
        sys.exit(1)
