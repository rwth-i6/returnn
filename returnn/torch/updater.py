"""
This module covers the optimizer (SGD, Adam, etc) logic,
and model param update logic in general.
"""

from __future__ import print_function

import torch
import typing

from returnn.log import log

_OptimizerClassesDictInitialized = False
_OptimizerClassesDict = {}


def _init_optimizer_classes_dict():
  """
  Initializes a global dictionary with all optimizers available in PyTorch.
  """
  global _OptimizerClassesDictInitialized
  if _OptimizerClassesDictInitialized:
    return
  _OptimizerClassesDictInitialized = True
  for name, cls in list(vars(torch.optim).items()):
    assert isinstance(name, str)
    # Check if cls is a valid subclass of torch.optim.Optimizer
    if not isinstance(cls, type) or not issubclass(cls, torch.optim.Optimizer):
      continue
    assert name not in _OptimizerClassesDict
    _OptimizerClassesDict[name.lower()] = cls


def get_optimizer_class(class_name):
  """
  :param str|function|type[torch.optim.Optimizer] class_name: Optimizer data, e.g. "adam", torch.optim.Adam...
  :return: Optimizer class
  :rtype: type[torch.optim.Optimizer]
  """
  _init_optimizer_classes_dict()
  if isinstance(class_name, type):
    assert issubclass(class_name, torch.optim.Optimizer)
  elif callable(class_name):
    class_name = class_name()
  else:
    assert isinstance(class_name, str)
    assert class_name.lower() in _OptimizerClassesDict, (
      "%s not found in the available torch optimizers list: %s." %
      (class_name.lower(), ", ".join("'%s'" % key for key in _OptimizerClassesDict))
    )
    class_name = _OptimizerClassesDict[class_name.lower()]

  return class_name


def _possibly_add_opt_param(opt_name, opt_value, opt_kwargs, cls_vars_list):
  """
  Adds an attribute to the dictionary of attributes to be passed to the optimizer,
  provided that the optimizer accepts this attribute.

  :param str opt_name: Name of the attribute to be added.
  :param typing.Any opt_value: Value of the attribute to be added.
  :param dict[str] opt_kwargs: Attributes that the optimizer accepts.
  :param list[str] cls_vars_list: List of accepted attributes by the optimizer.
  """
  if opt_name in cls_vars_list:
    opt_kwargs.setdefault(opt_name, opt_value)
  else:
    assert not opt_value, (
      "%s not accepted by the chosen optimizer. Accepted values: %s" %
      (opt_name, ", ".join("%s" % optim_name for optim_name in cls_vars_list))
    )


def _get_class_init_kwargs(optim_class):
  """
  Obtains the keyword arguments of the class provided as parameter that the user can add to their optimizer.

  :param type[torch.optim.Optimizer] optim_class: Optimizer class.
  :return: Keyword arguments of the provided class.
  :rtype: List[str]
  """
  from returnn.util.basic import collect_class_init_kwargs
  optim_class_init_kwargs = collect_class_init_kwargs(optim_class)
  # We already provide params by default, remove it so that the user doesn't add it to the optimizer dict.
  optim_class_init_kwargs.remove("params")

  return optim_class_init_kwargs


class Updater(object):
  """
  Wraps a torch.optim.Optimizer, and extends it by some further functionality.
  """

  def __init__(self, config, network, initial_learning_rate=1.):
    """
    :param returnn.config.Config config: config defining the training conditions.
    :param torch.nn.Module network: PyTorch Module defining the network.
    :param float initial_learning_rate:
    """
    self.config = config
    self.learning_rate = initial_learning_rate
    self.network = network
    self.optimizer = None  # type: typing.Optional[torch.optim.Optimizer]

  def set_learning_rate(self, value):
    """
    Updates the learning rate of the optimizer at each (sub)epoch.

    :param float value: New learning rate.
    """
    for param_group in self.optimizer.param_groups:
      param_group["lr"] = value

  def get_current_step_learning_rate(self):
    """
    Obtains an updated learning rate for the current training step inside a (sub)epoch.
    """
    pass

  def create_optimizer(self):
    """
    Creates an optimizer and stores it in self.optimizer.
    """
    optimizer_opts = self.config.typed_value("optimizer", None)
    if not optimizer_opts:
      self.optimizer = self._create_default_optimizer()
    else:
      self.optimizer = self._create_optimizer(optimizer_opts)

  def load_optimizer(self, filename):
    """
    Loads a torch.optim.Optimizer from disk and stores it in self.optimizer.

    :param str filename: File from which to load the optimizer state.
    """
    print("Load optimizer %s" % filename, file=log.v4)
    optimizer_state = torch.load(filename)
    self.optimizer.load_state_dict(optimizer_state)

  def save_optimizer(self, filename):
    """
    Saves the state of self.optimizer to a file.

    :param str filename: File in which to save the optimizer state.
    """
    print("Save optimizer under %s" % filename, file=log.v4)
    torch.save(self.optimizer.state_dict(), filename)

  def get_optimizer(self):
    """
    :return: Wrapped optimizer object.
    :rtype: torch.optim.Optimizer
    """
    return self.optimizer

  def _create_optimizer(self, optimizer_opts):
    """
    Returns a valid optimizer considering the dictionary given by the user in the config.

    :param dict[str]|str optimizer_opts: Optimizer configuration specified by the user.
        If it's a dict, it must contain "class" with the optimizer name or callable.
        If it's a str, it must be the optimizer name.
    :return: A valid optimizer.
    :rtype: torch.optim.Optimizer
    """
    lr = self.learning_rate
    epsilon = self.config.float("optimizer_epsilon", 1e-16)
    momentum = self.config.float("momentum", 0.0)

    # If the parameter is already a valid optimizer, return it without further processing
    if isinstance(optimizer_opts, torch.optim.Optimizer):
      return optimizer_opts

    if isinstance(optimizer_opts, str) or callable(optimizer_opts):
      optimizer_opts = {"class": optimizer_opts}
    assert isinstance(optimizer_opts, dict), (
      "optimizer_opts must be dict, str, callable or torch.optim.Optimizer instance."
    )
    assert "class" in optimizer_opts, "A custom optimizer requires a class (SGD, Adam...)"

    # Resolve the optimizer class
    optim_class_name = optimizer_opts.pop("class")
    optim_class = get_optimizer_class(optim_class_name)

    # Resolve the optimizer arguments
    opt_kwargs = optimizer_opts.copy()
    optim_class_init_kwargs = _get_class_init_kwargs(optim_class)
    # epsilon is named eps in torch.
    # If the user specified it as epsilon, parse it as eps for the optimizer
    if "eps" in optim_class_init_kwargs:
      if "epsilon" in opt_kwargs:
        opt_kwargs["eps"] = opt_kwargs.pop("epsilon")
      else:
        opt_kwargs.setdefault("eps", epsilon)
    else:
      assert "eps" not in opt_kwargs, (
        "epsilon not accepted by the chosen optimizer. Accepted values: %s" %
        ", ".join("%s" % optim_name for optim_name in optim_class_init_kwargs)
      )
    _possibly_add_opt_param("momentum", momentum, opt_kwargs, optim_class_init_kwargs)
    assert "learning_rate" not in optimizer_opts, "learning_rate should be set outside of the optimizer dict."
    lr = lr * optimizer_opts.get("learning_rate_multiplier", 1.0)
    opt_kwargs["lr"] = lr

    param_groups = self._get_optimizer_param_groups(optim_class, opt_kwargs)
    optimizer = optim_class(param_groups, **opt_kwargs)
    print("Optimizer: %s" % optimizer)
    assert isinstance(optimizer, torch.optim.Optimizer)

    return optimizer

  def _create_default_optimizer(self):
    """
    :return: SGD optimizer.
    :rtype: torch.optim.SGD
    """
    print("Create SGD optimizer (default).", file=log.v2)
    optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate)

    return optimizer

  def _get_optimizer_param_groups(self, optim_class, optimizer_opts):
    """
    The weight_decay parameter from AdamW affects the weights of layers such as LayerNorm and Embedding.
    This function creates a blacklist of network modules and splits the optimizer groups in two:
    those who will receive weight decay, and those who won't receive it.
    The weight_decay parameter of the rest of the optimizers is L2 regularization.

    For further reading, see https://github.com/karpathy/minGPT/pull/24#issuecomment-679316025 and
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994.

    This code is based on https://github.com/karpathy/minGPT (MIT license):
    https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136.

    :param type[torch.optim.Optimizer] optim_class: Optimizer class.
    :param dict[str] optimizer_opts: Optimizer configuration specified by the user.
    :return: List of configurations for the different sets of parameters.
    :rtype: List[Dict[str]]
    """
    network_params = self.network.parameters()

    # Split in parameter groups only if decouple_constraints is set and the optimizer accepts weight_decay.
    cls_init_kwargs = _get_class_init_kwargs(optim_class)
    if "weight_decay" not in cls_init_kwargs or self.config.bool("decouple_constraints", False):
      assert "weight_decay" not in optimizer_opts, (
        "weight_decay not accepted by the chosen optimizer. Accepted values: %s" %
        ", ".join("%s" % optim_name for optim_name in cls_init_kwargs)
      )
      return [{"params": network_params}]

    weight_decay = optimizer_opts.get("weight_decay", 0.0)
    if not weight_decay:
      return [{"params": network_params}]

    # Distinguish between parameters with and without weight_decay/L2 regularization.
    # Parameters without weight decay: biases + LayerNorm/Embedding layers.
    wd_params = set()
    no_wd_params = set()
    blacklist_wd_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in self.network.named_modules():
      for pn, p in m.named_parameters():
        fpn = '%s.%s' % (mn, pn) if mn else pn  # Full param name
        if pn.endswith("bias"):
          no_wd_params.add(fpn)
        elif pn.endswith("weight") and isinstance(m, blacklist_wd_modules):
          no_wd_params.add(fpn)
        else:
          wd_params.add(fpn)

    param_dict = {pn: p for pn, p in self.network.named_parameters()}
    optim_groups = [
      {"params": [param_dict[pn] for pn in sorted(list(wd_params))], "weight_decay": weight_decay},
      {"params": [param_dict[pn] for pn in sorted(list(no_wd_params))], "weight_decay": 0.0},
    ]

    return optim_groups
