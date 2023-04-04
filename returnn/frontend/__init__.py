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

from .array_ import *
from .const import *
from .dims import *
from .dtype import *
from .linear import *
from .loss import *
from .math_ import *
from .matmul import *
from .parameter import *
from .rand import *
from .reduce import *
from .run_ctx import *
from .state import *
from .types import *

from . import init

from ._backend import select_backend_torch, select_backend_returnn_layers_tf
