"""
This package provides the frontend API.
The frontend API supports multiple backends.
https://github.com/rwth-i6/returnn/issues/1120

The convention for the user is to do::

    import returnn.frontend as rf
"""

# Not necessarily all here, but the most common ones.
# (Take PyTorch `torch.nn` as a reference.)

# Some most come first here when others directly use it,
# e.g. `rf.Module` as a baseclass.
from .module import *
from .state import *

# Now the rest, in alphabetical order.
from .array_ import *
from .attention import *
from .cond import *
from .const import *
from .container import *
from .control_flow_ctx import *
from .conv import *
from .device import *
from .dims import *
from .dropout import *
from .dtype import *
from .gradient import *
from .linear import *
from .loop import *
from .loss import *
from .math_ import *
from .matmul import *
from .normalization import *
from .parameter import *
from .rand import *
from .rec import *
from .reduce import *
from .run_ctx import *
from .signal import *
from .types import *

# Modules not in the main namespace but in sub namespaces.
from . import init

# And some functions from the internal backend API.
from ._backend import select_backend_torch, select_backend_returnn_layers_tf
