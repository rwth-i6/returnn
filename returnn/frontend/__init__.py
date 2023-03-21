"""
This package provides the frontend API.
The frontend API supports multiple backends.
https://github.com/rwth-i6/returnn/issues/1120

The convention for the user is to do::

    from returnn import frontend as rf
"""

# Not necessarily all here, but the most common ones.
# (Take PyTorch `torch.nn` as a reference.)
from .array_ import *
from .dims import *
from .dtype import *
from .math_ import *
from .matmul import *
from .module import *
from .outputs import *
from .parameter import *
from .rand import *
from .reduce import *
from .run_ctx import *
from .state import *
from .types import *

from . import init
