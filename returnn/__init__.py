
"""
The main RETURNN package __init__.
We provide ``__version__`` and ``__long_version__``. See below.

You are supposed to explicitly import the specific sub-module/sub-package.
Just `import returnn` is not enough.

We also provide some helper code to keep older configs compatible,
which used our old-style module names, like ``import TFUtil`` or ``import returnn.TFUtil``.
"""

import os as _os

from .__setup__ import get_version_str as _get_version_str
__long_version__ = _get_version_str(fallback="1.0.0+unknown", long=True)  # `SemVer <https://semver.org/>`__ compatible
__version__ = __long_version__[:__long_version__.index("+")]  # distutils.version.StrictVersion compatible
__git_version__ = __long_version__  # just an alias, to keep similar to other projects

from .__old_mod_loader__ import setup as _old_mod_loader_setup
_old_mod_loader_setup(modules=globals())

__root_dir__ = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # can be used as __path__
