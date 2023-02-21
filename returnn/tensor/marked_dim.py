"""
Marked (optional) or implicit (virtual) dims
"""

from __future__ import annotations

from . import dim as _d
from . import _dim_extra


class MarkedDim:
    """
    Base class for marked dims, e.g. optional dims, or implicit (virtual) dims.
    """

    def __init__(self, tag: _d.Dim):
        """
        :param tag:
        """
        self.tag = tag

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.tag)

    def _eq_tuple(self):
        return self.__class__, self.tag

    def __hash__(self):
        return hash(self._eq_tuple())

    def __eq__(self, other):
        if isinstance(other, MarkedDim):
            return self._eq_tuple() == other._eq_tuple()
        return False

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        """
        See :func:`Dim.__lt__`.
        """
        if not isinstance(other, (_d.Dim, MarkedDim)):
            raise TypeError("cannot compare %r with %r" % (self, other))
        if self == other:
            return False
        return _dim_extra.dim_cmp_value(self) < _dim_extra.dim_cmp_value(other)

    def __gt__(self, other):
        return other < self

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return not self > other


class ImplicitDim(MarkedDim):
    """
    Represents an implicit dim (dim tag) in :class:`Data`.
    https://github.com/rwth-i6/returnn/issues/706
    """


class ImplicitSparseDim(ImplicitDim):
    """
    Represents an implicit dim via Data.sparse_dim.
    """


class ImplicitDynSizeDim(ImplicitDim):
    """
    Represents an implicit dim via dynamic dim sizes.
    https://github.com/rwth-i6/returnn/issues/706
    (For example via :class:`CumConcatLayer`.)
    """


class OptionalDim(MarkedDim):
    """
    Represents a dim which might exist or not.
    """
