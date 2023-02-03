"""
This package contains modules which all provide different :class:`Dataset` implementations.
A :class:`Dataset` is totally independent from the backend engine used (TF or Theano),
and also mostly independent from other parts of RETURNN, except some utils and logging.
"""

# Make available the most basic API directly here.
from .basic import Dataset, init_dataset, init_dataset_via_str
