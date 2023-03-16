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
from .dot import *
from .outputs import *
from .reduce import *
from .types import *
