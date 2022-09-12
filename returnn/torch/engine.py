"""
Main engine for PyTorch
"""

from returnn.config import Config
from returnn.engine.base import EngineBase


class Engine(EngineBase):
  """
  PyTorch engine
  """

  def __init__(self, config):
    """
    :param Config config:
    """
    super(Engine, self).__init__()
    self.config = config
