
"""
This is here so that the RETURNN repo root directory can be imported as a submodule.
This was used for the old flat code file structure, for usage like::

    import returnn.TFUtil

We want to support the same code.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

if globals().get("__package__", None) is None:
  __package__ = __name__  # https://www.python.org/dev/peps/pep-0366/

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from returnn.__old_mod_loader__ import setup as _old_module_loader_setup  # nopep8

# noinspection PyUnboundLocalVariable
_old_module_loader_setup(package_name=__package__, modules=globals())
