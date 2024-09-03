"""
Provides :class:`Data`, :class:`Dim`, :class:`SearchBeam`.

See :ref:`data` for some higher-level description.
"""

from __future__ import annotations

from typing import Optional, Union, Dict, List, Tuple
import typing
import os
import tensorflow as tf

from returnn.util.basic import NotSpecified

# Import also batch_dim, single_step_dim, to support old code.
# noinspection PyUnresolvedReferences
from returnn.tensor import Tensor, Dim, batch_dim, single_step_dim, ControlFlowContext

# Import also more to support old code.
# noinspection PyUnresolvedReferences
from returnn.tensor.marked_dim import MarkedDim, ImplicitSparseDim, ImplicitDynSizeDim, OptionalDim

# Import to support old code.
# noinspection PyUnresolvedReferences
from returnn.tensor.dim import VerifyOutShapeException


# Alias for old code compatibility.
Data = Tensor

# Earlier the class was called DimensionTag. Provide this alias for older code.
DimensionTag = Dim

# Alias for older code.
_MarkedDim = MarkedDim


# Provide some simple wrappers. https://github.com/rwth-i6/returnn/issues/782
# Use CamelCase function names (invalidates PEP8) to make it look like a class instance.


# noinspection PyPep8Naming
def FeatureDim(description, dimension, **kwargs):
    """
    DEPRECATED. Use :class:`Dim` instead, and setting the `kind` is not needed anymore.

    :param str description:
    :param int|None dimension:
    :rtype: Dim
    """
    return Dim(kind=Dim.Types.Feature, description=description, dimension=dimension, **kwargs)


# noinspection PyPep8Naming
def SpatialDim(description, dimension=None, **kwargs):
    """
    DEPRECATED. Use :class:`Dim` instead, and setting the `kind` is not needed anymore.

    :param str description:
    :param int|None dimension:
    :rtype: Dim
    """
    return Dim(kind=Dim.Types.Spatial, description=description, dimension=dimension, **kwargs)


class BatchInfo:
    """
    A batched tensor is a tensor with batch dimension,
    i.e. consisting of multiple samples/sequences
    which are supposed to be totally independent of each other.

    The batch dim can consists out of one or more flattened "virtual" dims,
    which :class:`BatchInfo` keeps track of.
    This class provides some additional information
    about the batch dimension.
    Only one instance per different type of batch-dim is expected
    (i.e. `batch_info1 is batch_info2` <==> same batch info).

    When we pass data from the dataset to the network
    (in all cases (training, inference ...) via :class:`Runner` in the TF engine),
    we get a batch dimension due to the minibatch construction.
    This is a global batch size
    (usually dynamic, because every minibatch/step can have a different amount of samples,
    although we can also support static sizes, which is needed e.g. for TPUs)
    represented by :class:`BatchInfo.GlobalBatchDim`.

    When we do beam search (see :class:`SearchBeam`),
    we have multiple hypotheses per batch item,
    and thus a different batch dimension.

    We can also pack arrays (multiple sequences)
    (also referred to as "flattened tensors", "non-padded tensors", "ragged tensors").
    See e.g. :class:`FlattenBatchLayer` or :func:`flatten_with_seq_len_mask`.
    Also see :class:`tf.RaggedTensor` which also represents
    packed tensors (but only batch-major, although this is just a reinterpretation).
    We do not directly use :class:`tf.RaggedTensor` in :class:`Data`
    to have robust and reliable code (which expects :class:`tf.Tensor`).
    However, we maybe can make use of some of the functions in :mod:`tf.ragged`.
    """

    class VirtualDimBase(object):
        """
        Represents one virtual dim, flattened into the batch dim.
        """

        def short_repr(self):
            """
            :rtype: str
            """
            raise NotImplementedError

        def __repr__(self):
            return "%s{%s}" % (self.__class__.__name__, self.short_repr())

    class FixedDim(VirtualDimBase):
        """
        Represents a dim with fixed size.
        """

        def __init__(self, size: Union[tf.Tensor, int], dim_tag: Optional[Dim] = None):
            """
            :param size:
            :param dim_tag:
            """
            self.size = size
            self.dim_tag = dim_tag

        def short_repr(self):
            """
            :rtype: str
            """
            if isinstance(self.size, int):
                return "F(%i)" % self.size
            return "F(?)"

    class GlobalBatchDim(FixedDim):
        """
        Represents the global batch dim by the network (minibatch construction from the dataset).
        """

        def __init__(self, size: Union[tf.Tensor, int], dim_tag: Optional[Dim] = None):
            if not dim_tag:
                dim_tag = batch_dim  # use global batch dim tag
            super().__init__(size=size, dim_tag=dim_tag)

        def short_repr(self):
            """
            :rtype: str
            """
            if isinstance(self.size, int) and self.size >= 0:
                return "B(%i)" % self.size
            return "B"

    class BeamDim(FixedDim):
        """
        Represents a search beam.
        """

        def __init__(self, beam):
            """
            :param SearchBeam beam:
            """
            super(BatchInfo.BeamDim, self).__init__(size=beam.beam_size)
            self.beam = beam

        def short_repr(self):
            """
            :rtype: str
            """
            return "Beam{%r}(%s)" % (self.beam.name, self.size)

    class PaddedDim(FixedDim):
        """
        Represents a dim with variable size, which is flattened with padding (not packed) into the batch.
        """

        def __init__(self, dim_tag):
            """
            :param Dim dim_tag:
            """
            super(BatchInfo.PaddedDim, self).__init__(size=dim_tag.get_dim_value())
            self.dim_tag = dim_tag

        def short_repr(self):
            """
            :rtype: str
            """
            return "Padded{%r}" % self.dim_tag.description

    class PackedDim(VirtualDimBase):
        """
        Represents a dim with variable sizes, which is packed (un-padded) into the batch.
        Variable w.r.t. other dims (must be per batch entry).
        """

        def __init__(self, dim_tag, key_axes):
            """
            :param Dim dim_tag:
            :param list[BatchInfo.VirtualDimBase] key_axes:
              most common case would be [GlobalBatchDim(...)],
              but [GlobalBatchDim(...),BeamDim(...)] is also common.
            """
            self.dim_tag = dim_tag
            self.key_axes = key_axes

        @property
        def sizes(self):
            """
            :return: shape [B_flat]
            :rtype: tf.Tensor
            """
            assert self.dim_tag.dyn_size is not None
            return self.dim_tag.dyn_size

        def short_repr(self):
            """
            :rtype: str
            """
            return "Packed{%r}" % (self.dim_tag.description,)

    def __init__(self, base, new_dim, new_dim_index=None):
        """
        :param BatchInfo|None base:
          If this is extended or based on another batch.
          Except of the batch dim of the dataset minibatch,
          we would always have a base.
        :param BatchInfo.VirtualDimBase|None new_dim:
        :param int|None new_dim_index:

        In most cases, this constructor would probably not be used directly by the user.
        """
        self.base = base
        virtual_dims = list(base.virtual_dims) if base else []
        if new_dim:
            if new_dim_index is None:
                assert not virtual_dims
                new_dim_index = 0
            if new_dim_index < 0:
                assert new_dim_index == -1
                virtual_dims.append(new_dim)
            else:
                virtual_dims.insert(new_dim_index, new_dim)
        self.virtual_dims = virtual_dims  # type: typing.List[BatchInfo.VirtualDimBase]
        self._dim = None  # type: typing.Optional[typing.Union[tf.Tensor,int]]
        self.batch_dim_tag: Optional[Dim] = None
        if not base and isinstance(new_dim, BatchInfo.GlobalBatchDim):
            self.batch_dim_tag = new_dim.dim_tag
        else:
            self.batch_dim_tag = Dim(
                kind=Dim.Types.Batch, description="batch:%s" % self.short_repr(), batch=self, dimension=self.static_dim
            )
        # These self._global_... attributes are meant
        # to be accessed only via the global (root) object (via get_global_base).
        # They store global information.
        # We don't use class attributes because this should not be global per process but only per network.
        self._global_beam_dims_by_beam_name = {}  # type: Dict[str,BatchInfo.BeamDim]
        self._global_padded_dims_by_dim_tag = {}  # type: Dict[Dim,BatchInfo.PaddedDim]
        self._packed_dims_by_dim_tag = {}  # type: Dict[Dim,BatchInfo.PackedDim]
        self.descendants = []  # type: List[BatchInfo]
        self._descendants_by_beam_name = {}  # type: Dict[str,BatchInfo]
        self._global_descendants_by_virtual_dims = {}  # type: Dict[Tuple[BatchInfo.VirtualDimBase,...],BatchInfo]
        if base:
            base.descendants.append(self)
            if isinstance(new_dim, BatchInfo.BeamDim):
                beam = new_dim.beam
                assert beam.name not in base._descendants_by_beam_name
                base._descendants_by_beam_name[beam.name] = self
        global_base = self.get_global_base()
        assert tuple(self.virtual_dims) not in global_base._global_descendants_by_virtual_dims
        global_base._global_descendants_by_virtual_dims[tuple(self.virtual_dims)] = self

    # noinspection PyShadowingNames
    @classmethod
    def make_global_batch_info(cls, batch_dim):
        """
        :param tf.Tensor|int batch_dim:
        :return: global batch info w.r.t. the network / graph
        :rtype: BatchInfo
        """
        # This is not stored in a class attrib because this is only w.r.t. the network, not global in the process.
        return BatchInfo(base=None, new_dim=BatchInfo.GlobalBatchDim(size=batch_dim))

    _global_broadcast_batch = None  # type: typing.Optional[BatchInfo]

    @classmethod
    def make_global_broadcast_batch_info(cls):
        """
        :return: BatchInfo with no virtual dims, s.t. the dimension is 1 (== prod([])) (broadcastable)
        :rtype: BatchInfo
        """
        if cls._global_broadcast_batch:
            return cls._global_broadcast_batch
        cls._global_broadcast_batch = BatchInfo(base=None, new_dim=None)
        return cls._global_broadcast_batch

    @classmethod
    def get_common_batch_info(cls, batches):
        """
        :param list[BatchInfo|None] batches:
        :rtype: BatchInfo|None
        """
        # Fast paths.
        if not batches:
            return None
        if len(batches) == 1:
            return batches[0]
        # Make unique, and filter non-none.
        batches_ = []
        for batch in batches:
            if batch and batch not in batches_:
                batches_.append(batch)
        batches = batches_
        if not batches_:
            return None
        if len(batches) == 1:
            return batches[0]
        base = batches[0].get_global_base()

        # Collect all dims.
        all_virtual_dims = []
        for batch in batches:
            for dim in batch.virtual_dims:
                if dim not in all_virtual_dims:
                    # We want to get a reasonable order.
                    same_type_last_idx = None
                    for i, dim_ in enumerate(all_virtual_dims):
                        if type(dim_) == type(dim):  # noqa
                            same_type_last_idx = i
                    if same_type_last_idx is not None:
                        all_virtual_dims.insert(same_type_last_idx + 1, dim)
                    else:
                        all_virtual_dims.append(dim)

        # Check if some batch already has them all.
        for batch in batches:
            if set(batch.virtual_dims) == set(all_virtual_dims):  # allow different order here
                return batch

        # Ok, need to extend.
        global_batch_dims = [dim for dim in all_virtual_dims if isinstance(dim, BatchInfo.GlobalBatchDim)]
        assert len(global_batch_dims) == 1, f"got global_batch_dims={global_batch_dims!r}"
        global_batch_dim = global_batch_dims[0]
        assert base.virtual_dims == [global_batch_dim]
        beams = [dim for dim in all_virtual_dims if isinstance(dim, BatchInfo.BeamDim)]
        if beams:
            base = base.copy_extend_with_beam(SearchBeam.get_combined_beam(*(b.beam for b in beams)))
        dim_idx = 0
        for dim in all_virtual_dims:
            if dim in global_batch_dims:
                dim_idx += 1 + len(beams)
                continue
            if dim in beams:
                continue
            base = base._copy_extend_dim(new_dim=dim, new_dim_idx=dim_idx)
            dim_idx += 1
        return base

    def __repr__(self):
        return "BatchInfo{%s}" % ", ".join([dim.short_repr() for dim in self.virtual_dims])

    def short_repr(self):
        """
        :rtype: str
        """
        # "x" is the Theano-style shortcut for a broadcast dim.
        return "&".join([dim.short_repr() for dim in self.virtual_dims] or ["Bx"])

    def __getstate__(self):
        raise Exception("Pickling of BatchInfo is not supported. (%s)" % self)

    @property
    def dim(self):
        """
        :rtype: tf.Tensor|int
        """
        if self._dim is not None:
            return self._dim
        if not self.virtual_dims:
            return 1
        if len(self.virtual_dims) == 1:
            dim = self.virtual_dims[0]
            assert isinstance(dim, BatchInfo.FixedDim)
            return dim.size
        from returnn.tf.util.basic import same_control_flow_ctx, optional_mul

        if all(isinstance(dim, BatchInfo.FixedDim) for dim in self.virtual_dims):
            dims = self.virtual_dims  # type: typing.List[BatchInfo.FixedDim]
            sizes = [dim.size for dim in dims]
            with same_control_flow_ctx(sizes):
                value = optional_mul(*sizes)  # type: typing.Union[tf.Tensor,int]
            self._dim = value
            return value
        if all(isinstance(dim, (BatchInfo.PackedDim, BatchInfo.GlobalBatchDim)) for dim in self.virtual_dims):
            dims = [dim for dim in self.virtual_dims if isinstance(dim, BatchInfo.PackedDim)]
            if len(dims) > 1:
                raise NotImplementedError("%s: currently only support one packed dim but have %r" % (self, dims))
            (dim,) = dims
            assert isinstance(dim, BatchInfo.PackedDim)
            with same_control_flow_ctx(dim.dim_tag.dyn_size_ext.placeholder):
                value = tf.reduce_sum(dim.dim_tag.dyn_size_ext.placeholder)
            self._dim = value
            return value
        raise NotImplementedError("%r.dim()" % self)

    @dim.setter
    def dim(self, value):
        """
        Can only set the global batch dim.

        :param tf.Tensor|int value:
        """
        assert len(self.virtual_dims) == 1
        dim = self.virtual_dims[0]
        assert isinstance(dim, BatchInfo.GlobalBatchDim)
        dim.size = value
        if dim.dim_tag:
            dim.dim_tag.capacity = dim.dim_tag.size = value if (isinstance(value, int) and value > 0) else None
        self._dim = value

    @property
    def static_dim(self):
        """
        :rtype: int|None
        """
        # This should be safe. Do not call self.dim.
        if self._dim is not None:
            return self._dim if isinstance(self._dim, int) else None
        if not self.virtual_dims:
            return 1
        if len(self.virtual_dims) == 1:
            dim = self.virtual_dims[0]
            assert isinstance(dim, BatchInfo.FixedDim)
            return dim.size if isinstance(dim.size, int) else None
        from functools import reduce
        from operator import mul

        if all(isinstance(dim, BatchInfo.FixedDim) for dim in self.virtual_dims):
            dims = self.virtual_dims  # type: typing.List[BatchInfo.FixedDim]
            sizes = [dim.size for dim in dims]
            if all(isinstance(s, int) for s in sizes):
                return reduce(mul, sizes, 1)
            return None
        return None

    @property
    def beam(self):
        """
        :rtype: SearchBeam|None
        """
        beams = [dim for dim in self.virtual_dims if isinstance(dim, BatchInfo.BeamDim)]
        if beams:
            # Just return first. In case you need more custom logic, directly check the dims.
            return beams[0].beam
        return None

    def get_base_chain(self):
        """
        :rtype: list[BatchInfo]
        """
        bases = []
        base = self.base
        while base:
            bases.append(base)
            base = base.base
        return bases

    def get_global_base(self):
        """
        :rtype: BatchInfo
        """
        if not self.base:
            return self
        return self.get_base_chain()[-1]

    def get_global_batch_dim(self):
        """
        :rtype: BatchInfo.GlobalBatchDim
        """
        global_beam_dims = [dim for dim in self.virtual_dims if isinstance(dim, BatchInfo.GlobalBatchDim)]
        assert len(global_beam_dims) == 1
        return global_beam_dims[0]

    def is_global_batch(self):
        """
        :rtype: bool
        """
        global_beam_dims = [dim for dim in self.virtual_dims if isinstance(dim, BatchInfo.GlobalBatchDim)]
        return len(global_beam_dims) == 1 and len(self.virtual_dims) == 1

    def is_broadcast(self):
        """
        :rtype: bool
        """
        return len(self.virtual_dims) == 0

    def _make_beam_dim(self, beam):
        """
        :param SearchBeam beam:
        :rtype: BatchInfo.BeamDim
        """
        assert self.virtual_dims
        root = self.get_global_base()
        if beam.name in root._global_beam_dims_by_beam_name:
            return root._global_beam_dims_by_beam_name[beam.name]
        new_dim = BatchInfo.BeamDim(beam=beam)
        root._global_beam_dims_by_beam_name[beam.name] = new_dim
        return new_dim

    def _make_packed_dim(self, dim_tag):
        """
        :param Dim dim_tag:
        :rtype: BatchInfo.PackedDim
        """
        assert self.virtual_dims
        assert dim_tag.dyn_size is not None
        dim_tag_base = dim_tag.get_same_base()
        if dim_tag_base in self._packed_dims_by_dim_tag:
            return self._packed_dims_by_dim_tag[dim_tag_base]
        new_dim = BatchInfo.PackedDim(dim_tag=dim_tag, key_axes=self.virtual_dims)
        self._packed_dims_by_dim_tag[dim_tag_base] = new_dim
        return new_dim

    def _make_padded_dim(self, dim_tag):
        """
        :param Dim dim_tag:
        :rtype: BatchInfo.PaddedDim
        """
        assert self.virtual_dims
        root = self.get_global_base()
        assert dim_tag.dyn_size is not None
        dim_tag_base = dim_tag.get_for_batch_ctx(self, dim_tag.control_flow_ctx)
        if dim_tag_base in root._global_padded_dims_by_dim_tag:
            return root._global_padded_dims_by_dim_tag[dim_tag_base]
        new_dim = BatchInfo.PaddedDim(dim_tag=dim_tag_base)
        root._global_padded_dims_by_dim_tag[dim_tag_base] = new_dim
        return new_dim

    def _next_spatial_major_index(self):
        idx = None
        for i, dim in enumerate(self.virtual_dims):
            if isinstance(dim, BatchInfo.GlobalBatchDim):
                break
            if isinstance(dim, BatchInfo.BeamDim):
                break
            assert isinstance(dim, BatchInfo.FixedDim)
            idx = i + 1
        if idx is not None:
            return idx
        return 0

    def copy_extend_with_beam(self, beam):
        """
        :param SearchBeam beam:
        :rtype: BatchInfo
        """
        assert self.virtual_dims
        if self.beam == beam:
            return self
        if beam.name in self._descendants_by_beam_name:
            return self._descendants_by_beam_name[beam.name]
        return BatchInfo(
            base=self,
            new_dim=self._make_beam_dim(beam),
            new_dim_index=self.virtual_dims.index(self.get_global_batch_dim()) + 1,
        )

    def copy_remove_beam(self):
        """
        :rtype: BatchInfo
        """
        if not self.beam:
            return self
        assert self.virtual_dims
        root = self.get_global_base()
        dims_wo_beam = [dim for dim in self.virtual_dims if not isinstance(dim, BatchInfo.BeamDim)]
        return root._global_descendants_by_virtual_dims[tuple(dims_wo_beam)]  # must exist

    def copy_remove_dim(self, remove_dim):
        """
        :param VirtualDimBase remove_dim:
        :rtype: BatchInfo
        """
        assert self.virtual_dims
        root = self.get_global_base()
        dims_wo_dim = [dim for dim in self.virtual_dims if dim != remove_dim]
        return root._global_descendants_by_virtual_dims[tuple(dims_wo_dim)]  # must exist

    def copy_set_beam(self, beam):
        """
        :param SearchBeam|None beam:
        :rtype: BatchInfo
        """
        batch = self.copy_remove_beam()
        if beam:
            batch = batch.copy_extend_with_beam(beam)
        return batch

    def copy_extend_with_packed_dim_tag(self, dim_tag, batch_major):
        """
        :param Dim dim_tag:
        :param bool batch_major: if True, add new dim in front. otherwise, add new dim at the end
        :rtype: BatchInfo
        """
        new_dim = self._make_packed_dim(dim_tag)
        new_dim_idx = -1 if batch_major else self._next_spatial_major_index()
        return self._copy_extend_dim(new_dim=new_dim, new_dim_idx=new_dim_idx)

    def copy_extend_with_padded_dim_tag(self, dim_tag, batch_major=None, new_dim_idx=None):
        """
        :param Dim dim_tag:
        :param bool|None batch_major: if True, add new dim in front. otherwise, add new dim at the end
        :param int|None new_dim_idx:
        :rtype: BatchInfo
        """
        new_dim = self._make_padded_dim(dim_tag)
        if new_dim_idx is None:
            assert batch_major is not None
            new_dim_idx = -1 if batch_major else self._next_spatial_major_index()
        else:
            assert batch_major is None
        return self._copy_extend_dim(new_dim=new_dim, new_dim_idx=new_dim_idx)

    def copy_extend_with_padded_or_fixed_dim_tag(self, dim_tag, batch_major=None, new_dim_idx=None):
        """
        :param Dim dim_tag:
        :param bool|None batch_major: if True, add new dim in front. otherwise, add new dim at the end
        :param int|None new_dim_idx:
        :rtype: BatchInfo
        """
        if dim_tag.dyn_size is not None:
            new_dim = self._make_padded_dim(dim_tag)
        else:
            new_dim = BatchInfo.FixedDim(size=dim_tag.get_dim_value(), dim_tag=dim_tag)
        if new_dim_idx is None:
            assert batch_major is not None
            new_dim_idx = -1 if batch_major else self._next_spatial_major_index()
        else:
            assert batch_major is None
        return self._copy_extend_dim(new_dim=new_dim, new_dim_idx=new_dim_idx)

    def _copy_extend_dim(self, new_dim, new_dim_idx):
        """
        :param BatchInfo.VirtualDimBase new_dim:
        :param int new_dim_idx:
        :rtype: BatchInfo
        """
        assert self.virtual_dims
        root = self.get_global_base()
        virtual_dims = list(self.virtual_dims)
        if new_dim_idx < 0:
            assert new_dim_idx == -1
            virtual_dims.append(new_dim)
        else:
            virtual_dims.insert(new_dim_idx, new_dim)
        if tuple(virtual_dims) in root._global_descendants_by_virtual_dims:
            return root._global_descendants_by_virtual_dims[tuple(virtual_dims)]
        return BatchInfo(base=self, new_dim=new_dim, new_dim_index=new_dim_idx)


class SearchBeam:
    """
    Represents info about the beam from some beam search (e.g. via :func:`beam_search`),
    e.g. such as the beam size, but also the dependencies.
    This is somewhat parallel to :class:`SearchChoices`, but simpler,
    and independent from the layers/network (:class:`returnn.tf.layers.base.LayerBase`).
    """

    def __init__(self, beam_size, dependency=NotSpecified, name=None, _next_frame=None):
        """
        :param int beam_size:
        :param SearchBeam|NotSpecified|None dependency:
        :param str|None name:
        :param SearchBeam|None _next_frame:
        """
        if isinstance(dependency, SearchBeam):
            assert name and dependency.name and name != dependency.name
        if name and os.path.basename(name).startswith("prev:"):
            assert _next_frame
        self.beam_size = beam_size
        self.dependency = dependency
        self.name = name
        self._next_frame = _next_frame

    def copy_as_prev_frame(self):
        """
        :rtype: SearchBeam
        """
        if self._next_frame:  # already prev frame -> return self. see logic in RecLayer maybe_transform
            return self
        assert self.name
        name = "%s/prev:%s" % (os.path.dirname(self.name), os.path.basename(self.name))
        return SearchBeam(beam_size=self.beam_size, name=name, _next_frame=self)

    def __repr__(self):
        keys = ["name", "beam_size"]
        if self.dependency is not NotSpecified:
            keys.append("dependency")
        return "%s(%s)" % (self.__class__.__name__, ", ".join(["%s=%r" % (key, getattr(self, key)) for key in keys]))

    def __eq__(self, other):
        """
        :param SearchBeam|object|None other:
        :rtype: bool
        """
        if self is other:
            return True
        if self is None or other is None:
            return False
        if not isinstance(self, SearchBeam) or not isinstance(other, SearchBeam):
            return False
        if self.name is None or other.name is None:
            return False  # cannot identify
        return self.name == other.name

    def __ne__(self, other):
        """
        :param SearchBeam|object|None other:
        :rtype: bool
        """
        return not (self == other)

    def __hash__(self):
        return hash(self.name)

    def _get_dependency_list(self):
        """
        :return: list as far as it is defined
        :rtype: list[SearchBeam]
        """
        ls = [self]
        while isinstance(ls[-1].dependency, SearchBeam):
            ls.append(ls[-1].dependency)
        return ls

    @classmethod
    def get_combined_beam(cls, beam1, beam2=None, *beams):
        """
        Combines beams.
        This will throw an exception if they cannot be combined.
        Note that in beam search (see :class:`SearchChoices`),
        the logic to combine beams from different search choices
        happens in a generic way for all layers automatically
        via :func:`TFNetwork._create_layer_layer_desc`,
        so normally we already have the same beam.
        Unless we are at template construction.

        :param SearchBeam|None beam1:
        :param SearchBeam|None beam2:
        :param SearchBeam|None beams:
        :rtype: SearchBeam|None
        """
        if beams:
            beam12 = cls.get_combined_beam(beam1, beam2)
            return cls.get_combined_beam(beam12, beams[0], *beams[1:])
        if beam2 is None:
            return beam1
        if beam1 is None:
            return beam2
        if beam1 == beam2:
            if beam2.dependency is NotSpecified:
                return beam1
            if beam1.dependency is NotSpecified:
                return beam2
            return beam1
        assert beam1.name and beam2.name
        if beam2._next_frame and not beam1._next_frame:
            return beam1
        if beam1._next_frame and not beam2._next_frame:
            return beam2
        b1 = beam1
        b2 = beam2
        used_next_frame = False
        if b1._next_frame and b2._next_frame:
            b1 = b1._next_frame
            b2 = b2._next_frame
            used_next_frame = True
        l1 = b1._get_dependency_list()
        l2 = b2._get_dependency_list()
        if b2 in l1:
            return beam1
        if b1 in l2:
            return beam2
        if used_next_frame:
            # Example: beam1: prev:out, beam2: prev:t, t->prev:out (l2).
            if beam1 in l2:  # -> beam1 dep on beam2
                return beam1
            if beam2 in l1:
                return beam2
        raise Exception(
            "\n".join(
                [
                    "Cannot combine beams:",
                    "  1: %s (deps: %s, next %s, next deps %s)"
                    % (
                        beam1,
                        beam1._get_dependency_list(),
                        beam1._next_frame,
                        beam1._next_frame._get_dependency_list() if beam1._next_frame else None,
                    ),
                    "  2: %s (deps: %s, next %s, next deps %s)"
                    % (
                        beam2,
                        beam2._get_dependency_list(),
                        beam2._next_frame,
                        beam2._next_frame._get_dependency_list() if beam2._next_frame else None,
                    ),
                ]
            )
        )
