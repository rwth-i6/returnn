"""
Sets OPENBLAS_NUM_THREADS and similar env vars explicitly to 1.
The idea is that this might make some tests more deterministic.

This is optionial, and separate from :mod:`_setup_test_env`.
"""


def _setup():
    import sys
    import os

    # Do this here such that we always see this log in Travis.
    orig_stdout = sys.stdout
    try:
        sys.stdout = sys.__stdout__  # Nosetests has overwritten sys.stdout

        # Do this very early, before we import numpy/TF, such that it can have an effect.
        for env_var in ["OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS"]:
            print("Env %s = %s" % (env_var, os.environ.get(env_var, None)))
            # Overwrite with 1. This should make the test probably more deterministic. Not sure...
            os.environ[env_var] = "1"

    finally:
        sys.stdout = orig_stdout


_setup()
