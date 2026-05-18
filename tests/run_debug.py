"""
Debug runner for hanging tests.

Installs a faulthandler that dumps all thread tracebacks (including C extensions)
to stderr after ``TIMEOUT`` seconds have elapsed.

Usage::

    python tests/run_debug.py [pytest-args …]

"""

from __future__ import annotations
import faulthandler
import sys

_TIMEOUT_SECS = 300

print(f"[run_debug] faulthandler armed: will dump all tracebacks after {_TIMEOUT_SECS}s", file=sys.stderr, flush=True)
faulthandler.dump_traceback_later(_TIMEOUT_SECS, file=sys.stderr, repeat=True)

import pytest  # noqa: E402

# Pass -v (show each test name) and -s (don't capture stdout/stderr)
extra_args = ["-v", "-s"]
user_args = sys.argv[1:]
sys.exit(pytest.main(extra_args + user_args))
