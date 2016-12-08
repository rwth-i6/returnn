
from Log import log


class TFNetwork(object):
  def __init__(self, rnd_seed=42):
    self.rnd_seed = rnd_seed

  def construct_from_dict(self, net_json):
    pass

  def load_params_from_file(self, filename):
    pass

  def print_network_info(self, name="Network"):
    print >> log.v2, "%s layer topology:" % name
    print >> log.v2, "  input #:", self.n_in
    for layer_name, layer in sorted(self.hidden.items()):
      print >> log.v2, "  hidden %s %r #: %i" % (layer.layer_class, layer_name, layer.attrs["n_out"])
    if not self.hidden:
      print >> log.v2, "  (no hidden layers)"
    for layer_name, layer in sorted(self.output.items()):
      print >> log.v2, "  output %s %r #: %i" % (layer.layer_class, layer_name, layer.attrs["n_out"])
    if not self.output:
      print >> log.v2, "  (no output layers)"
    print >> log.v2, "net params #:", self.num_params()
    print >> log.v2, "net trainable params:", self.train_params_vars
