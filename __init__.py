"""
This is here so that the RETURNN repo root directory can be imported as a submodule.
This was used for the old flat code file structure, for usage like::

    import returnn.TFUtil

We want to support the same code.
"""


from __future__ import annotations
import os
import sys

if globals().get("__package__", None) is None:
    __package__ = __name__  # https://www.python.org/dev/peps/pep-0366/ # pylint: disable=redefined-builtin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "returnn":
    # This is a bit tricky. `import returnn` in here will not work, as this is the reference to ourselves!
    # Thus delete this ref, to enforce a reload of the right module.
    # For the user, this will actually replace the `returnn` module by the right one.
    # But this should be compatible.
    sys.modules.pop("returnn", None)
    # It is important that we have some `import returnn` below.

from returnn.__old_mod_loader__ import (
    setup as _old_module_loader_setup,
)  # nopep8 # pylint: disable=wrong-import-position

# noinspection PyUnboundLocalVariable
_old_module_loader_setup(package_name=__package__, modules=globals())
