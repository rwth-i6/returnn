
from .__setup__ import get_version_str as _get_version_str
__version__ = _get_version_str(fallback="1.0.0")

from .__old_mod_loader__ import setup as _old_mod_loader_setup
_old_mod_loader_setup()
