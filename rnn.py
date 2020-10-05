#!/usr/bin/env python3

"""
RETURNN as a tool. Main entry point. Just calls :func:`returnn.__main__.main`.
"""

from returnn.__main__ import main


if __name__ == '__main__':
  main()

else:
  # This is likely an old script, directly importing this file.
  # You should not do that. Import returnn.__main__ instead.
  # Anyway, here some hacky workaround to keep this working.
  import sys
  sys.modules[__name__] = sys.modules["returnn.__main__"]
