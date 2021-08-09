.. _test_suite:

==========
Test suite
==========

RETURNN comes with a huge number of test cases (unit tests and more complex tests)
which are automatically run on every Git push and for every GitHub pull request
by GitHub Actions defined `here <https://github.com/rwth-i6/returnn/blob/master/.github/workflows/main.yml>`__.

The test cases are all in the `tests directory <https://github.com/rwth-i6/returnn/tree/master/tests>`__.

We use nosetests but the tests can also be run manually like::

  python3 tests/test_TFEngine.py

Or::

  python3 tests/test_TFEngine.py test_engine_train
