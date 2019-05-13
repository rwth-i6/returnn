
"""
This is here so that whole RETURNN can be imported as a submodule.
"""

# Currently all of RETURNN is written with absolute imports.
# This probably should be changed, see also: https://github.com/rwth-i6/returnn/issues/162
# If we change that to relative imports, for some of the scripts we might need some solution like this:
# https://stackoverflow.com/questions/54576879/

# Anyway, as a very ugly workaround, to make it possible to import RETURNN as a package, we have this hack:
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
del sys, os
