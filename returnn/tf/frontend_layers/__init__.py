"""
TensorFlow frontend.
"""

from ._backend import ReturnnLayersBackend

# Sorted by name.
from .cond import *
from .config_entry_points import *
from .debug_eager_mode import *
from .dims import *
from .layer import *
from .loop import *
from .make_layer import *
from .masked_computation import *
from .prev_tensor_ref import *


# https://returnn.readthedocs.io/en/latest/configuration_reference/behavior_version.html
# Need Dim.is_equal to be more restrictive (v16).
min_returnn_behavior_version = 16
