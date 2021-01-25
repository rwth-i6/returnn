
"""
Provides :class:`Data`, :class:`DimensionTag`, :class:`SearchBeam`.

See :ref:`data` for some higher-level description.
"""

from __future__ import print_function, division

import os
import typing
import tensorflow as tf

from returnn.util.basic import NotSpecified
import returnn.tf.compat as tf_compat


class DimensionTag(object):
  """
  This identifies one axis/dimension, like a time-dimension, etc.
  This can be used by :class:`Data`. See :func:`Data.get_dim_tag`.
  It is not to specify the specific axis in a specific Data/tensor,
  but to specify the content and dimension.
  I.e. if we have the same DimensionTag for two Data instances,
  the dimensions should match. I.e.:

      data1.get_dim_tag(i) == data2.get_dim_tag(j)
        =>  tf.shape(data1.placeholder)[i] == tf.shape(data2.placeholder)[j]
  """

  class Types:
    """
    Defines possible values for ``kind``.
    """
    Unspecified = None
    Batch = "batch"
    Spatial = "spatial"  # also time
    Time = "spatial"  # we don't treat this as different
    Feature = "feature"

  def __init__(self, kind=Types.Unspecified, description=None, dimension=None, dyn_size=None,
               src_data=None, src_axis=None):
    """
    :param str|None kind:
    :param str|None description: the description should be unique
    :param int|None dimension:
    :param tf.Tensor|None dyn_size: e.g. seq_len, (batch,)
    :param Data|None src_data:
    :param int|None src_axis:
    """
    self.id = id(self)  # This is just used for __repr__ to distinguish different instances.
    self.kind = kind
    self.description = description
    self.dimension = dimension
    self.dyn_size = dyn_size
    self.same_as = None  # type: typing.Optional[DimensionTag]
    if src_data:
      assert isinstance(src_data, Data) and isinstance(src_axis, int)
    self.src_data = src_data
    self.src_axis = src_axis
    if dyn_size is not None:
      other = DimensionTag.get_tag_from_size_tensor(dyn_size)
      if other:
        self.declare_same_as(other)
      else:
        self.set_tag_on_size_tensor(dyn_size)

  def __repr__(self):
    attribs = ["kind"]
    for attr in ["description", "dimension"]:
      if getattr(self, attr) is not None:
        attribs.append(attr)
    attribs.append("id")
    if self.same_as:
      attribs.append("same_base_id")
    return "DimensionTag(%s)" % ", ".join(["%s=%r" % (attr, getattr(self, attr)) for attr in attribs])

  def set_tag_on_size_tensor(self, x):
    """
    :param tf.Tensor x:
    """
    # It's unusual if self.dimension is not None, but let's accept that.
    if hasattr(x, "_is_size_of_dim_tag"):
      # noinspection PyProtectedMember
      assert x._is_size_of_dim_tag in (None, self)
    if getattr(x, "_is_size_of_dim_tag", None) is None:
      setattr(x, "_is_size_of_dim_tag", self)
    if self.dyn_size is None:
      self.dyn_size = x

  @classmethod
  def get_tag_from_size_tensor(cls, x):
    """
    :param tf.Tensor x: size tensor. has been set before via :func:`set_tag_on_size_tensor`
    :rtype: DimensionTag|None
    """
    return getattr(x, "_is_size_of_dim_tag", None)

  def can_compare(self):
    """
    :return: whether we can clearly identify this axis. for axes with dynamic size, we require the dyn_size.
    :rtype: bool
    """
    if self.same_as:
      return self.same_as.can_compare()
    if self.kind in [self.Types.Batch, self.Types.Feature]:
      return True
    assert self.kind == self.Types.Spatial
    if self.dimension is not None:
      return True
    if self.dyn_size is None:
      return False
    assert self.get_tag_from_size_tensor(self.dyn_size).get_same_base() is self
    return True

  def is_equal(self, other, ignore_feature_dim=False, allow_same_feature_dim=False, allow_same_spatial_dim=None,
               treat_feature_as_spatial=False, broadcast_matches=False, unknown_spatial_matches=False):
    """
    Compares self to other for equality.
    Note that the default behavior is very restrictive.
    Use functions such as :func:`get_all_dimension_tags` or :func:`get_existing_tag_from_collection`
    to explicitly specify the behavior for the comparison.

    :param DimensionTag other:
    :param bool ignore_feature_dim:
    :param bool allow_same_feature_dim:
    :param bool|None allow_same_spatial_dim:
    :param bool treat_feature_as_spatial:
    :param bool broadcast_matches:
    :param bool unknown_spatial_matches:
    :rtype: bool
    """
    if allow_same_spatial_dim is None:
      allow_same_spatial_dim = allow_same_feature_dim
    self_base = self.get_same_base()
    other_base = other.get_same_base()
    if self_base is other_base:
      return True
    self_kind = self.kind
    other_kind = other.kind
    if self_kind == other_kind == self.Types.Feature and ignore_feature_dim:
      return True
    if treat_feature_as_spatial:
      if self_kind == self.Types.Feature:
        self_kind = self.Types.Spatial
      if other_kind == self.Types.Feature:
        other_kind = self.Types.Spatial
    if self.dimension != other.dimension:
      if broadcast_matches and (self.dimension == 1 or other.dimension == 1):
        pass  # pass on
      else:
        return False
    if self_kind != other_kind:
      return False
    if self_kind == other_kind == self.Types.Batch:
      # Note: This might be incorrect in some cases,
      # e.g. for beam search when we have the beam hidden in the batch dim,
      # or when we used MergeDimsLayer on the batch axis, or so.
      # We might need to extend the logic here later.
      return True
    if self_kind == other_kind == self.Types.Feature:
      if allow_same_feature_dim:
        return True
    if self_kind == other_kind == self.Types.Spatial:
      if allow_same_spatial_dim:
        if self.dimension is not None:
          return True
        if broadcast_matches and (self.dimension == 1 or other.dimension == 1):
          return True
      if unknown_spatial_matches and ((self.dyn_size is None) != (other.dyn_size is None)):
        return True
    if self.description == other.description:
      return True
    return False

  def __eq__(self, other):
    """
    :param DimensionTag other:
    :rtype: bool
    """
    if not isinstance(other, DimensionTag):
      return False
    return self.is_equal(other)

  def __ne__(self, other):
    """
    :param DimensionTag other:
    :rtype: bool
    """
    return not (self == other)

  def __hash__(self):
    base = self.get_same_base()
    return hash((base.kind, base.description))

  def get_same_base(self):
    """
    :rtype: DimensionTag
    """
    if self.same_as:
      return self.same_as.get_same_base()
    return self

  @property
  def same_base_id(self):
    """
    :rtype: int
    """
    return self.get_same_base().id

  def declare_same_as(self, other):
    """
    :param DimensionTag other:
    """
    from .basic import same_control_flow_ctx, tile_transposed
    assert not self.same_as or self.same_as is other.get_same_base()
    self.same_as = other.get_same_base()
    # If we have a defined source, and this is a dynamic spatial axis, and it was undefined before,
    # maybe we can overtake the size_placeholder now.
    if self.same_as.dyn_size is not None and self.src_data:
      assert isinstance(self.src_axis, int)
      # Maybe it changed in the meanwhile, so check.
      if self.src_data.get_dim_tag(self.src_axis).description == self.description:
        if self.src_data.size_placeholder is None:
          self.src_data.size_placeholder = {}
        self.src_data.size_placeholder[
          self.src_data.get_batch_axis_excluding_batch(self.src_axis)] = self.same_as.dyn_size
        # if the tag is used in a recurrent layer during search, the placeholder has to be expanded by the beam size
        if self.src_data.beam and (not self.same_as.src_data or not self.same_as.src_data.beam):
          for i, v in sorted(self.src_data.size_placeholder.items()):
            with same_control_flow_ctx(v):
              self.src_data.size_placeholder[i] = tile_transposed(v, axis=0, multiples=self.src_data.beam.beam_size)
            self.set_tag_on_size_tensor(self.src_data.size_placeholder[i])
    # If others dyn_size is None but we have a dyn_size, maybe update others dyn_size.
    if self.dyn_size is not None and self.same_as.dyn_size is not self.dyn_size:
      # Could be unset if it comes from the config, or from prev graph creation.
      # This is important such that self.can_compare() is sane.
      if self.same_as.dyn_size is None or self.same_as.dyn_size.graph is not self.dyn_size.graph:
        self.same_as.dyn_size = self.dyn_size

  @classmethod
  def get_existing_tag_from_collection(cls, other, tags, is_equal_opts=None):
    """
    :param DimensionTag other:
    :param list[DimensionTag]|tuple[DimensionTag]|set[DimensionTag] tags:
    :param dict[str]|None is_equal_opts: passed to DimensionTag.is_equal
    :rtype: DimensionTag|None
    """
    if is_equal_opts is None:
      is_equal_opts = {}
    for _tag in tags:
      if _tag.is_equal(other, **is_equal_opts):
        return _tag
    return None

  @classmethod
  def get_all_dimension_tags(cls, data_list, is_equal_opts=None, unique_separate_axes=True):
    """
    :param list[Data] data_list:
    :param dict[str]|None is_equal_opts: passed to DimensionTag.is_equal
    :param bool unique_separate_axes: e.g. data_list=[Data with shape (B,5,5,10)] results in 4 dim tags, not 3.
    :return: list of dimension tags, dict for data -> list of dimension tags (for each axis)
    :rtype: (list[DimensionTag], dict[Data, list[DimensionTag]])
    """
    tags = []
    data_axes_dict = {}
    for data in data_list:
      data_axes_dict[data] = []
      tags_for_data = []
      for axis in range(data.batch_ndim):
        tag = data.get_dim_tag(axis)
        existing_tag = cls.get_existing_tag_from_collection(tag, tags=tags, is_equal_opts=is_equal_opts)
        if not existing_tag:
          if unique_separate_axes:
            # Don't append it to `tags` directly now, such that e.g. for data with shape (B,5,5,10),
            # we end up with two separate dim tags for the two spatial dims.
            tags_for_data.append(tag)
          else:
            tags.append(tag)
        data_axes_dict[data].append(existing_tag or tag)
      tags.extend(tags_for_data)
    return tags, data_axes_dict

  @classmethod
  def get_uniq_collection(cls, tags, is_equal_opts=None):
    """
    :param list[DimensionTag]|tuple[DimensionTag]|set[DimensionTag] tags:
    :param dict[str]|None is_equal_opts: passed to DimensionTag.is_equal
    :rtype: list[DimensionTag]
    """
    res = []
    for tag in tags:
      ex = cls.get_existing_tag_from_collection(tag, res, is_equal_opts=is_equal_opts)
      if not ex:
        res.append(tag)
    return res


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
    return "%s(%s)" % (
      self.__class__.__name__, ", ".join(["%s=%r" % (key, getattr(self, key)) for key in keys]))

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
      "\n".join([
        "Cannot combine beams:",
        "  1: %s (deps: %s, next %s, next deps %s)" % (
          beam1, beam1._get_dependency_list(),
          beam1._next_frame, beam1._next_frame._get_dependency_list() if beam1._next_frame else None),
        "  2: %s (deps: %s, next %s, next deps %s)" % (
          beam2, beam2._get_dependency_list(), beam2._next_frame,
          beam2._next_frame._get_dependency_list() if beam2._next_frame else None)]))


class Data(object):
  """
  This class is to describe a tensor,
  i.e. its shape and properties like
  whether we should consider it sparse data (i.e. it represents indices).
  This is used in TFNetwork to describe the dataset external data
  as well as in every layer's output.

  See :ref:`data`.
  """

  size_dtype = "int32"

  def __init__(self, name,
               shape=None, dtype=None,
               placeholder=None,
               sparse=None,
               dim=NotSpecified,
               size_placeholder=None,
               batch_dim_axis=0,
               time_dim_axis=NotSpecified,
               feature_dim_axis=NotSpecified,
               available_for_inference=True,
               auto_create_placeholders=False,
               vocab=None,
               same_dim_tags_as=None,
               beam=None):
    """
    :param str name:
    :param tuple[int|None]|list[int|None] shape: including time-dim (can be None). excluding batch-dim.
      e.g. (time,feat)=(None,128)
    :param str dtype: e.g. "float32" or "int64"
    :param tf.Tensor|None placeholder: with added batch-dim
    :param bool sparse: whether to treat the value as an index. do not confuse with tf.SparseTensor
    :param None|int dim: feature dimension, shape[-1] if not sparse, otherwise like num_classes
    :param int|None batch_dim_axis: where we add the batch-dim.
      e.g. shape=(time,...), 0 -> (batch,time,...), 1 -> (time,batch,...).
      This is normally always set, and a lot of code expects this. However, you can set it to None
      if this Data does not have a batch-dim.
    :param int|None time_dim_axis: where we have the time dim axis, after we added the batch-dim.
      this is often 1. however, can be None if there is no time-dim.
    :param int|None|NotSpecified feature_dim_axis: feature dim axis. by default it's the last one
    :param dict[int,tf.Tensor]|None size_placeholder: for every None in shape, this will describe the size.
      The size is always a tensor of shape (batch,), i.e. the size can be different for each sequence in a batch.
    :param bool available_for_inference: e.g. the extern data "classes" is usually not available for inference
    :param str|dict[str]|GeneratingDataset.Vocabulary|None vocab:
    :param dict[int|str,DimensionTag]|None same_dim_tags_as: will mark our dimension tags to be the same
    :param SearchBeam|None beam: the batch-dim could be extended by a beam-size,
      such that it represents the merged dims [batch, beam_size].
    """
    assert isinstance(name, str)
    assert dtype is None or isinstance(dtype, str)
    self.name = name
    if sparse is None:
      sparse = False
    self.sparse = sparse
    if dtype is None:
      if sparse:
        dtype = "int32"
      else:
        dtype = "float32"
    self.dtype = dtype  # type: str
    assert batch_dim_axis is None or isinstance(batch_dim_axis, int)
    self.batch_dim_axis = batch_dim_axis  # type: typing.Optional[int]  # None -> no batch dim axis
    if shape is None:
      if time_dim_axis is NotSpecified:  # need to determine this now
        if self.batch_dim_axis is None:
          time_dim_axis = None
        else:
          # By default if not specified, we have a time dim.
          taken_axes = {self.batch_dim_axis}
          if isinstance(feature_dim_axis, int):
            taken_axes.add(feature_dim_axis)
          time_dim_axis = [i for i in range(max(taken_axes) + 2) if i not in taken_axes][0]
      if time_dim_axis is not None:
        assert time_dim_axis != self.batch_dim_axis
        shape = (None,) * (self.get_batch_axis_excluding_batch(time_dim_axis) + 1)
      else:  # no time-dim-axis
        shape = ()
      if not sparse and feature_dim_axis is not None:
        assert dim is not NotSpecified, "no shape specified, not sparse, feature_dim_axis existing -> need dim"
        if feature_dim_axis is NotSpecified or feature_dim_axis == -1:
          shape = shape + (dim,)
        else:
          assert 0 <= feature_dim_axis != self.batch_dim_axis
          feature_dim_axis_wo_batch = self.get_batch_axis_excluding_batch(feature_dim_axis)
          if feature_dim_axis_wo_batch < len(shape):
            shape = shape[:-feature_dim_axis_wo_batch] + (dim,) + shape[feature_dim_axis_wo_batch + 1:]
          else:
            shape = shape + (None,) * (feature_dim_axis_wo_batch - len(shape)) + (dim,)
            assert len(shape) == feature_dim_axis_wo_batch + 1
    self.shape = tuple(shape)  # type: typing.Tuple[typing.Optional[int], ...]  # excl. batch-dim. see self.batch_shape
    if feature_dim_axis is not NotSpecified:
      if isinstance(feature_dim_axis, int):
        assert not self.sparse, "cannot have feature_dim_axis when sparse"
        if feature_dim_axis < 0:
          feature_dim_axis += self.batch_ndim
        assert 0 <= feature_dim_axis < self.batch_ndim
    self._feature_dim_axis = feature_dim_axis
    if time_dim_axis is NotSpecified:
      if self.batch_dim_axis is None:
        time_dim_axis = None
      else:
        # Do not select the batch dim axis, or any axis with None dim.
        # Note that we currently allow to select the same as the feature dim axis,
        # in case the feature dim is None.
        taken_axes = {self.batch_dim_axis}
        for axis, _dim in enumerate(self.batch_shape):
          if _dim is not None:
            taken_axes.add(axis)
        available_axes = [i for i in range(self.batch_ndim) if i not in taken_axes]
        if available_axes:
          time_dim_axis = available_axes[0]
        else:
          time_dim_axis = None
    if time_dim_axis is not None:
      assert 0 <= time_dim_axis < self.batch_ndim
    self.time_dim_axis = time_dim_axis  # type: typing.Optional[int]  # counted with batch-dim
    if dim is NotSpecified:
      assert not sparse, "need dim (num classes) if sparse"
      if self.feature_dim_axis is None:
        dim = None
      else:
        dim = self.batch_shape[self.feature_dim_axis]
    self.dim = dim  # type: typing.Optional[int]
    if placeholder is None and auto_create_placeholders:
      with tf.name_scope("extern_data/placeholders/%s/" % name):
        placeholder = tf_compat.v1.placeholder(**self.get_placeholder_kwargs(with_batch=True))
    self.placeholder = placeholder  # type: tf.Tensor  # this will hold the data value itself
    # The size_placeholder is for each variable length dimension in shape, i.e. excluding the batch-dim.
    if size_placeholder is not None:
      size_placeholder = size_placeholder.copy()
    if size_placeholder is None and auto_create_placeholders:
      size_placeholder = {}  # type: typing.Dict[int,tf.Tensor]
      with tf.name_scope("extern_data/placeholders/%s/" % name):
        for axis in self.get_axes_with_size():
          size_placeholder[axis] = tf_compat.v1.placeholder(**self.get_size_placeholder_kwargs(axis))
          tag = DimensionTag(
            description="%s:var:extern_data:%s" % (
              "time" if self.get_batch_axis(axis) == self.time_dim_axis else "spatial%i" % axis, self.name),
            kind=DimensionTag.Types.Spatial)
          tag.set_tag_on_size_tensor(size_placeholder[axis])
    if not size_placeholder and (self.ndim_dense <= 1 or all([d is not None for d in shape])):
      size_placeholder = {}
    self.size_placeholder = size_placeholder  # type: typing.Dict[int,tf.Tensor]  # axis w.o. batch -> size (batch,)
    self.available_for_inference = available_for_inference
    self.beam = beam
    if vocab is not None:
      from returnn.datasets.generating import Vocabulary
      if isinstance(vocab, str):
        vocab = Vocabulary(vocab)
      elif isinstance(vocab, dict):
        vocab = Vocabulary.create_vocab(**vocab)
      assert isinstance(vocab, Vocabulary)
      assert self.sparse, "%s should represent indices of %s" % (self, vocab)
      assert self.dim == vocab.num_labels, "%s dims do not match with vocab %s" % (self, vocab)
    self.vocab = vocab  # type: typing.Optional[Vocabulary]
    if same_dim_tags_as:
      # Note that this currently does not work as intended at template construction time...
      for _axis, _dim_tag in sorted(same_dim_tags_as.items()):
        _axis = self.get_axis_from_description(_axis)
        self.get_dim_tag(_axis).declare_same_as(_dim_tag)
    self.sanity_check()

  @classmethod
  def from_tensor(cls, x):
    """
    :param tf.Tensor x:
    :rtype: Data
    """
    assert x.get_shape().ndims == 0, "currently only scalars supported"
    return Data(name=str(x.op.name), shape=(), batch_dim_axis=None, dtype=x.dtype.name, placeholder=x)

  @classmethod
  def template_from_constant(cls, x, name, dtype=None, with_batch_dim=False):
    """
    :param int|float|bool|numpy.ndarray x:
    :param str name:
    :param str|None dtype:
    :param bool with_batch_dim:
    :rtype: Data
    """
    import numpy
    if dtype is None:
      if isinstance(x, int):
        dtype = "int32"
      elif isinstance(x, float):
        dtype = "float32"
      elif isinstance(x, bool):
        dtype = "bool"
      elif isinstance(x, numpy.ndarray):
        dtype = str(x.dtype)
      else:
        raise TypeError("cannot handle value %r of type %r" % (x, type(x)))
    shape = x.shape if isinstance(x, numpy.ndarray) else ()
    return Data(
      name=name,
      shape=shape, batch_dim_axis=0 if with_batch_dim else None, time_dim_axis=None,
      dtype=dtype)

  def sanity_check(self, ignore_placeholder=False):
    """
    Performs some sanity checks on self, and raises exceptions if something is not sane.

    :param bool ignore_placeholder:
    """
    for axis_name, axis in self.get_special_axes_dict(include_batch_dim_axis=True).items():
      assert axis is None or 0 <= axis < self.batch_ndim, "%s: axis %s (%i) invalid" % (self, axis_name, axis)
    if self.batch_dim_axis is not None:
      for axis_name, axis in self.get_special_axes_dict(include_batch_dim_axis=False).items():
        assert axis != self.batch_dim_axis, "%s: axis %s (%i) must be different from batch_dim_axis (%i)" % (
          self, axis_name, axis, self.batch_dim_axis)
    if self.sparse:
      assert self.feature_dim_axis is None, "%s: If sparse, there cannot be a feature dim axis." % self
    else:
      if self.feature_dim_axis is None:  # e.g. scalars, or [B]
        assert self.dim is None, "%s: not sparse but no feature-dim-axis, so dim should be None" % self
    if self.feature_dim_axis is not None:
      assert self.dim == self.batch_shape[self.feature_dim_axis], (
        "%s: inconsistent dim. feature axis or unspecified: %r." % (self, self.feature_dim_axis_or_unspecified))
    if not ignore_placeholder and self.placeholder is not None:
      # Note: We could just call self.placeholder.set_shape.
      # However, we are more explicit. We assume that the placeholder has already a known shape, and error otherwise.
      assert self.placeholder.shape.ndims == self.batch_ndim
      for i in range(self.batch_ndim):
        if self.batch_shape[i] is None:
          continue  # we allow anything in the placeholder
        if self.placeholder.shape[i].value != self.batch_shape[i]:
          print("Mismatching shape: Tensor %r vs Data %r" % (self.placeholder, self))
          from .basic import print_graph_output
          print_graph_output(self.placeholder, max_depth=3)
        assert self.placeholder.shape[i].value == self.batch_shape[i]
      self.placeholder.set_shape(self.batch_shape)
      assert self.placeholder.dtype.base_dtype.name == self.dtype

  def get_placeholder_kwargs(self, with_batch=True):
    """
    :param bool with_batch:
    :return: kwargs for tf.compat.v1.placeholder
    :rtype: dict[str]
    """
    return dict(name=self.name, dtype=self.dtype, shape=self.batch_shape if with_batch else self.shape)

  def get_axes_with_size(self):
    """
    :return: list of axes which can vary in size for each entry of the batch-dim, e.g. the time-dim-axis.
      The axis index is counted without the batch-dim.
    :rtype: list[int]
    """
    return [i for (i, dim) in enumerate(self.shape) if dim is None]

  def get_size_placeholder_kwargs(self, axis, with_batch=True):
    """
    :param int axis:
    :param bool with_batch:
    :return: kwargs for tf.compat.v1.placeholder
    :rtype: dict[str]
    """
    # For each batch a separate size.
    return dict(name="%s_dim%i_size" % (self.name, axis), dtype=self.size_dtype,
                shape=(None,) if with_batch else ())

  def get_kwargs(self, with_size_placeholder=False):
    """
    :param bool with_size_placeholder:
    :return: relevant attrib items for copying
    :rtype: dict[str]
    """
    keys = ["name", "shape", "dtype", "sparse", "dim", "batch_dim_axis", "time_dim_axis"]
    if self._feature_dim_axis is not NotSpecified:
      keys += ["feature_dim_axis"]
    if not self.available_for_inference:
      keys += ["available_for_inference"]
    if self.beam is not None:
      keys += ["beam"]
    if self.vocab:
      keys += ["vocab"]
    if with_size_placeholder and self.size_placeholder is not None:
      keys += ["size_placeholder"]
    return {key: getattr(self, key) for key in keys}

  def get_description(self, with_name=True, with_placeholder=False):
    """
    :param bool with_name:
    :param bool with_placeholder:
    :return: description of self. also used for __repr__
    :rtype: str
    """
    keys = ["shape"]
    if self.sparse:
      keys.append("dtype")
      keys.append("sparse")
      keys.append("dim")
    else:
      if self.dtype != "float32":
        keys.append("dtype")
    if self.batch_dim_axis != 0:
      keys.append("batch_dim_axis")
    if (
          self.time_dim_axis is None or
          self.time_dim_axis >= 2 or
          self.batch_dim_axis is None or
          self.batch_dim_axis >= 2):
      keys.append("time_dim_axis")
    if self._feature_dim_axis is not NotSpecified:
      keys.append("feature_dim_axis")
    if with_name:
      keys.insert(0, "name")
    if with_placeholder:
      keys.append("placeholder")
    if not self.available_for_inference:
      keys.append("available_for_inference")
    if self.beam is not None:
      keys.append("beam")
    args = ["%s=%r" % (key, getattr(self, key)) for key in keys]
    args += ["batch_shape_meta=[%s]" % ",".join(self.get_batch_axes_short_description())]
    return "Data(%s)" % ", ".join(args)

  def get_batch_axes_short_description(self):
    """
    :rtype: list[str]
    """
    res = []
    for axis, dim_tag in enumerate(self.get_batch_shape_dim_tags()):
      descriptions = []
      if axis == self.batch_dim_axis:
        descriptions.append("B")
      if axis == self.time_dim_axis:
        descriptions.append("T")
      if axis == self.feature_dim_axis:
        descriptions.append("F")
      if self.batch_shape[axis] is None:
        if axis == self.batch_dim_axis:
          pass  # expected
        elif self.size_placeholder and self.get_batch_axis_excluding_batch(axis) in self.size_placeholder:
          descriptions.append(repr(dim_tag.description))
        else:
          descriptions.append("?")
      else:
        descriptions.append(str(self.batch_shape[axis]))
        if dim_tag.kind == DimensionTag.Types.Spatial and dim_tag.dyn_size is not None:
          descriptions.append(repr(dim_tag.description))
      res.append("|".join(descriptions))
    return res

  def get_compare_key(self):
    """
    :return: some key which can be used for compare functions, i.e. such that
      cmp(get_compare_key(self), get_compare_key(other)) == cmp(self, other),
      i.e. we define some order by that.
      Note that this order is not totally fixed, and might change.
    :rtype: object
    """
    return (
      self.name, self.dtype,
      self.shape,
      self.batch_dim_axis, self.feature_dim_axis, self.time_dim_axis,
      sorted(self.size_placeholder.keys()),
      [self.get_size_dim_tag(i) for i in range(len(self.size_placeholder))],
      self.beam)

  def __repr__(self):
    return self.get_description()

  def __hash__(self):
    return id(self)

  def copy(self, name=None):
    """
    :param str name: if given, will overwrite this name
    :return: copy of myself, using self.get_kwargs(), and with placeholder and size_placeholder
    :rtype: Data
    """
    data = Data(**self.get_kwargs())
    data.placeholder = self.placeholder
    if self.size_placeholder is not None:
      data.size_placeholder = self.size_placeholder.copy()
    if name:
      data.name = name
    return data

  def copy_as_batch_major(self):
    """
    :return: copy of myself with batch_dim_axis == 0
    :rtype: Data
    """
    return self.copy_with_batch_dim_axis(0)

  def copy_as_time_major(self):
    """
    :return: copy of myself with time_dim_axis == 0
    :rtype: Data
    """
    assert self.time_dim_axis is not None
    return self.copy_with_time_dim_axis(0)

  def copy_with_batch_dim_axis(self, batch_dim_axis):
    """
    :param int batch_dim_axis:
    :return: copy of myself with specific batch_dim_axis
    :rtype: Data
    """
    assert self.batch_dim_axis is not None
    return self.copy_move_axis(self.batch_dim_axis, batch_dim_axis)

  def copy_with_time_dim_axis(self, time_dim_axis):
    """
    :param int time_dim_axis:
    :return: copy of myself with specific time_dim_axis
    :rtype: Data
    """
    assert self.time_dim_axis is not None
    return self.copy_move_axis(self.time_dim_axis, time_dim_axis)

  def copy_move_axis(self, old_axis, new_axis):
    """
    :param int old_axis: counted with batch-dim
    :param int new_axis: counted with batch-dim
    :return: copy of myself with moved axis (see :func:`move_axis`)
    :rtype: Data
    """
    from .basic import move_axis
    if old_axis < 0:
      old_axis += self.batch_ndim
      assert old_axis >= 0
    assert 0 <= old_axis < self.batch_ndim
    if new_axis < 0:
      new_axis += self.batch_ndim
      assert new_axis >= 0
    assert 0 <= new_axis < self.batch_ndim
    if old_axis == new_axis:
      return self.copy()

    def translate_axis(axis):
      """
      :param int|None axis:
      :return: axis after move_axis
      :rtype: int|None
      """
      if axis is None:
        return None
      if old_axis == new_axis:
        return axis
      if axis < min(old_axis, new_axis) or axis > max(old_axis, new_axis):
        return axis
      if axis == old_axis:
        return new_axis
      if old_axis < new_axis:
        assert old_axis < axis <= new_axis
        return axis - 1
      assert new_axis <= axis < old_axis
      return axis + 1

    data = self.copy()
    if data.placeholder is not None:
      data.placeholder = move_axis(data.placeholder, old_axis, new_axis)
    data.batch_dim_axis = translate_axis(self.batch_dim_axis)
    new_feature_dim_axis = translate_axis(self.feature_dim_axis)
    if new_feature_dim_axis != data.feature_dim_axis:
      # Only assign in this case. Otherwise, e.g. if it is NotSpecified, leave it like that.
      data.feature_dim_axis = new_feature_dim_axis
    data.time_dim_axis = translate_axis(self.time_dim_axis)
    if data.size_placeholder:
      data.size_placeholder = {
        data.get_batch_axis_excluding_batch(translate_axis(self.get_batch_axis(i))): size
        for (i, size) in data.size_placeholder.items()}
      assert None not in data.size_placeholder
    new_shape = [None] * data.ndim
    for i, dim in enumerate(self.shape):
      new_shape[data.get_batch_axis_excluding_batch(translate_axis(self.get_batch_axis(i)))] = dim
    data.shape = tuple(new_shape)
    data.sanity_check()
    return data

  def copy_as_bt_or_tb_major(self):
    """
    :rtype: Data
    :return: copy of myself in batch-time-major or time-batch-major
    """
    assert self.have_batch_axis() and self.have_time_axis()
    if self.batch_dim_axis == 0:
      return self.copy_with_time_dim_axis(1)
    if self.time_dim_axis == 0:
      return self.copy_with_batch_dim_axis(1)
    if self.batch_dim_axis > self.time_dim_axis:
      return self.copy_as_time_major().copy_as_bt_or_tb_major()
    return self.copy_as_batch_major().copy_as_bt_or_tb_major()

  def copy_with_feature_dim_axis(self, feature_dim_axis):
    """
    :param int feature_dim_axis: can also be negative
    :return: copy of myself with specific feature dim axis
    :rtype: Data
    """
    assert self.feature_dim_axis is not None
    return self.copy_move_axis(self.feature_dim_axis, feature_dim_axis)

  def copy_as_batch_feature_major(self):
    """
    :return: copy of self with batch_dim_axis == 0 and feature_dim_axis == 1
    :rtype: Data
    """
    assert self.batch_dim_axis is not None
    assert self.feature_dim_axis is not None
    data = self.copy_as_batch_major()
    data = data.copy_with_feature_dim_axis(1)
    return data

  def copy_as_time_batch_major(self):
    """
    :return: copy of self with batch_dim_axis == 1 and time_dim_axis == 0
    :rtype: Data
    """
    assert self.have_batch_axis() and self.have_time_axis()
    data = self.copy_as_bt_or_tb_major()
    if data.time_dim_axis == 1:
      data = data.copy_move_axis(0, 1)
    return data

  def copy_as_batch_spatial_major(self):
    """
    :return: copy with batch_dim_axis == 0, then all dynamic axes, then any other spatial axes, last feature axis
    :rtype: Data
    """
    data = self.copy_as_batch_major()
    if data.feature_dim_axis is not None:
      data = data.copy_with_feature_last()
    if data.size_placeholder:
      for i, (j, size) in enumerate(sorted(data.size_placeholder.items())):
        data = data.copy_move_axis(data.get_batch_axis(j), i + 1)
    if data.feature_dim_axis is not None:
      assert data.feature_dim_axis == data.batch_ndim - 1
      # Maybe reset feature_dim_axis to unspecified.
      if data.feature_dim_axis_or_unspecified is not NotSpecified:
        if data._default_feature_dim_axis() == data.feature_dim_axis:
          data.feature_dim_axis = NotSpecified
    return data

  def copy_with_feature_last(self):
    """
    :return: copy of self with feature_dim_axis being the very last axis
    :rtype: Data
    """
    assert self.feature_dim_axis is not None
    return self.copy_with_feature_dim_axis(-1)

  def copy_add_batch_dim(self, batch_dim_axis):
    """
    :param int batch_dim_axis:
    :return: copy of myself with added batch-dim
    :rtype: Data
    """
    assert self.batch_dim_axis is None
    if batch_dim_axis < 0:
      assert batch_dim_axis + self.batch_ndim + 1 >= 0
      batch_dim_axis += self.batch_ndim + 1
    assert 0 <= batch_dim_axis <= self.batch_ndim
    data = self.copy()
    if data.placeholder is not None:
      data.placeholder = tf.expand_dims(data.placeholder, batch_dim_axis, name="%s_add_batch_dim" % self.name)
    data.batch_dim_axis = batch_dim_axis
    other_special_axes = self.get_special_axes_dict(counted_with_batch_dim=True, only_available=True)
    for k, a in other_special_axes.items():
      setattr(data, k, a if (a < batch_dim_axis) else (a + 1))
    data.sanity_check()
    return data

  def copy_add_spatial_dim(self, spatial_dim_axis=None, dim=1, auto_time_dim_axis=True):
    """
    :param int|None spatial_dim_axis: counted with batch-dim. if there is no time-dim, this will be it.
    :param int|None dim:
    :param bool auto_time_dim_axis:
    :return: copy of myself with added spatial-dim
    :rtype: Data
    """
    from .basic import get_valid_scope_name_from_str
    data = self.copy()
    if spatial_dim_axis is None:
      if self.get_spatial_batch_axes():
        spatial_dim_axis = self.get_spatial_batch_axes()[-1] + 1  # after the existing spatial dim
      elif self.feature_dim_axis is not None:
        spatial_dim_axis = self.feature_dim_axis  # add it before the feature dim
      else:
        spatial_dim_axis = self.batch_ndim  # add it at the end
    else:
      if spatial_dim_axis < 0:
        assert spatial_dim_axis + self.batch_ndim + 1 >= 0
        spatial_dim_axis += self.batch_ndim + 1
      assert 0 <= spatial_dim_axis <= self.batch_ndim
    if data.placeholder is not None:
      assert dim == 1  # not implemented otherwise
      data.placeholder = tf.expand_dims(
        data.placeholder, spatial_dim_axis, name="%s_add_spatial_dim" % get_valid_scope_name_from_str(self.name))
    if self.batch_dim_axis is None:
      axis_wo_batch = spatial_dim_axis
    else:
      axis_wo_batch = spatial_dim_axis if (spatial_dim_axis <= self.batch_dim_axis) else (spatial_dim_axis - 1)
    if data.size_placeholder:
      data.size_placeholder = {
        i if (i < axis_wo_batch) else (i + 1): size
        for (i, size) in data.size_placeholder.items()}
    data.shape = data.shape[:axis_wo_batch] + (dim,) + data.shape[axis_wo_batch:]
    if auto_time_dim_axis and data.time_dim_axis is None:
      data.time_dim_axis = spatial_dim_axis
    other_special_axes = self.get_special_axes_dict(
      counted_with_batch_dim=True, only_available=True, include_batch_dim_axis=True)
    for k, a in other_special_axes.items():
      setattr(data, k, a if (a < spatial_dim_axis) else (a + 1))
    if data.feature_dim_axis is not None:
      # feature dim axis might have changed if unspecified, so just update dim
      data.dim = data.batch_shape[data.feature_dim_axis]
    data.sanity_check()
    return data

  def copy_add_feature_dim(self, axis=None):
    """
    :param int|None axis:
    :return: self with a new feature dim axis with dim 1.
      If there is an existing feature dim, the new feature dim will be added right after.
      If we are sparse, we don't add a feature dim, but it becomes a spatial dim instead.
    :rtype: Data
    """
    if self.sparse:
      # By definition, we don't have a feature dim. We allow this though. We just make it a spatial axis.
      return self.copy_add_spatial_dim(spatial_dim_axis=axis)
    v = self.copy()
    assert not v.sparse
    if axis is None:
      if v.feature_dim_axis is not None:
        new_feature_dim_axis = v.feature_dim_axis + 1
      else:
        new_feature_dim_axis = v.batch_ndim
    else:
      if axis < 0:
        assert axis + v.batch_ndim + 1 >= 0
        axis += v.batch_ndim + 1
      assert 0 <= axis <= v.batch_ndim
      new_feature_dim_axis = axis
    other_special_axes = self.get_special_axes_dict(
      counted_with_batch_dim=True, only_available=True, include_batch_dim_axis=True)
    other_special_axes.pop("feature_dim_axis", None)
    new_feature_dim_axis_wo_batch = self.get_batch_axis_excluding_batch(new_feature_dim_axis)
    v.shape = v.shape[:new_feature_dim_axis_wo_batch] + (1,) + v.shape[new_feature_dim_axis_wo_batch:]
    v.dim = 1
    for k, a in other_special_axes.items():
      setattr(v, k, a if (a < new_feature_dim_axis) else (a + 1))
    if v.feature_dim_axis_or_unspecified is not NotSpecified:
      v.feature_dim_axis = NotSpecified
    if v.feature_dim_axis != new_feature_dim_axis:
      v.feature_dim_axis = new_feature_dim_axis
    if v.placeholder is not None:
      v.placeholder = tf.expand_dims(v.placeholder, new_feature_dim_axis, name="copy_add_feature_dim")
    v.sanity_check()
    return v

  def get_default_new_axis_for_dim_tag(self, dim_tag):
    """
    :param DimensionTag dim_tag:
    :rtype: int
    """
    if dim_tag.kind == DimensionTag.Types.Batch:
      return 0
    # Note: if dim_tag is feature, but we are sparse, we just treat is as spatial, handled below.
    if dim_tag.kind == DimensionTag.Types.Feature and not self.sparse:
      if self.feature_dim_axis is not None:
        return self.feature_dim_axis + 1  # after existing feature-dim
      else:
        return self.batch_ndim  # at the end
    assert dim_tag.kind == DimensionTag.Types.Spatial or (dim_tag.kind == DimensionTag.Types.Feature and self.sparse)
    if dim_tag.dimension is None and self.get_dynamic_axes():
      return self.get_dynamic_axes()[-1] + 1  # after existing dynamic axis
    if self.get_spatial_batch_axes():
      return self.get_spatial_batch_axes()[-1] + 1  # after the existing spatial dim
    elif self.feature_dim_axis is not None:
      return self.feature_dim_axis  # add it before the feature dim
    else:
      return self.batch_ndim  # add it at the end

  def copy_add_dim_by_tag(self, dim_tag, unbroadcast=False, axis=None):
    """
    :param DimensionTag dim_tag:
    :param bool unbroadcast:
    :param int|None axis:
    :rtype: Data
    """
    if axis is None:
      axis = self.get_default_new_axis_for_dim_tag(dim_tag=dim_tag)
    if dim_tag.kind == DimensionTag.Types.Batch:
      res = self.copy_add_batch_dim(batch_dim_axis=axis)
      if unbroadcast:
        assert res.placeholder is None  # not implemented yet...
      return res
    # Note: if dim_tag is feature, but we are sparse, we just treat is as spatial, handled below.
    if dim_tag.kind == DimensionTag.Types.Feature and not self.sparse:
      res = self.copy_add_feature_dim(axis=axis)
      if unbroadcast:
        assert res.placeholder is None  # not implemented yet...
        res.dim = dim_tag.dimension
        shape = list(res.shape)
        shape[res.get_batch_axis_excluding_batch(res.feature_dim_axis)] = dim_tag.dimension
        res.shape = tuple(shape)
        res.sanity_check()
      return res
    assert dim_tag.kind == DimensionTag.Types.Spatial or (dim_tag.kind == DimensionTag.Types.Feature and self.sparse)
    res = self.copy_add_spatial_dim(spatial_dim_axis=axis, dim=1)
    assert res.batch_shape[axis] == 1
    if unbroadcast:
      assert res.placeholder is None  # not implemented yet...
      shape = list(res.shape)
      shape[res.get_batch_axis_excluding_batch(axis)] = dim_tag.dimension
      res.shape = tuple(shape)
      if res.feature_dim_axis is not None:
        # feature dim axis might have changed if unspecified, so just update dim
        res.dim = res.batch_shape[res.feature_dim_axis]
      res.sanity_check()
      if dim_tag.dimension is None and dim_tag.dyn_size is not None:
        if res.size_placeholder is None:
          res.size_placeholder = {}
        res.size_placeholder[res.get_batch_axis_excluding_batch(axis)] = dim_tag.dyn_size
    return res

  def copy_split_feature_dim(self, new_feature_dim):
    """
    :param int new_feature_dim: will be the new dim
    :rtype: Data
    """
    from .basic import get_shape
    assert not self.sparse
    assert self.feature_dim_axis is not None
    assert self.dim is not None
    assert self.dim % new_feature_dim == 0, "must be a multiple of the input feature dim"
    old_feature_dim = self.dim // new_feature_dim
    new_feature_dim_axis = self.feature_dim_axis + 1
    v = self.copy()
    other_special_axes = self.get_special_axes_dict(
      counted_with_batch_dim=True, only_available=True, include_batch_dim_axis=True)
    other_special_axes.pop("feature_dim_axis", None)
    old_feature_dim_axis_wo_batch = self.get_batch_axis_excluding_batch(self.feature_dim_axis)
    v.shape = (v.shape[:old_feature_dim_axis_wo_batch] +
               (old_feature_dim, new_feature_dim) +
               v.shape[old_feature_dim_axis_wo_batch + 1:])
    v.dim = new_feature_dim
    for k, a in other_special_axes.items():
      setattr(v, k, a if (a < new_feature_dim_axis) else (a + 1))
    v.feature_dim_axis = new_feature_dim_axis
    if v.placeholder is not None:
      v.placeholder.set_shape(self.batch_shape)
      old_shape = get_shape(v.placeholder)
      new_shape = (old_shape[:self.feature_dim_axis] +
                   [old_feature_dim, new_feature_dim] +
                   old_shape[new_feature_dim_axis + 1:])
      v.placeholder = tf.reshape(v.placeholder, new_shape, name="copy_split_feature_dim")
    v.sanity_check()
    return v

  def copy_compatible_to(self, data, unbroadcast=False, except_feature=False,
                         data_dyn_shape=None, check_sparse=True, check_dtype=True):
    """
    :param Data data: other data which the returned tensor should be compatible to
      It would add any missing axes with a dim 1 axis for automatic broadcasting.
      It currently does not check whether existing dims match.
    :param bool unbroadcast: if True, all broadcast axes (axes with dim 1) will be tiled such that they match
    :param bool except_feature: if unbroadcast, do not unbroadcast the feature dim
    :param tf.Tensor|list[tf.Tensor|int]|tuple[tf.Tensor|int]|None data_dyn_shape:
      For unbroadcast, if we do not want to rely on tf.shape(data.placeholder).
    :param bool check_sparse:
    :param bool check_dtype:
    :returns: Data, might add broadcast dimensions
    :rtype: Data
    """
    assert not check_sparse or self.sparse == data.sparse
    assert not check_dtype or self.dtype == data.dtype
    v = self.copy()
    v.sparse = data.sparse  # we will later reset it. this is to better count the axes (feature and spatial)
    if not v.sparse:
      # We might need to reset the dim, as it would be invalid otherwise. Reset later.
      if v.feature_dim_axis is not None:
        v.dim = v.batch_shape[v.feature_dim_axis]
      else:
        v.dim = None
    if data.batch_dim_axis is not None and v.batch_dim_axis is None:
      v = v.copy_add_batch_dim(0)  # later we might move the axis
    if v.batch_dim_axis is not None and data.batch_dim_axis is None:
      raise ValueError("copy_compatible_to: self %r has batch-dim, but target data %r has not" % (self, data))
    if data.batch_ndim < v.batch_ndim:
      raise ValueError("copy_compatible_to: self %r already has more dims than target data %r" % (self, data))
    start = v
    _, dim_tags = DimensionTag.get_all_dimension_tags([start, data], dict(allow_same_feature_dim=True))
    assert len(dim_tags[start]) == start.batch_ndim
    assert len(dim_tags[data]) == data.batch_ndim
    # This sets it explicitly. We will later make it NotSpecified if needed.
    # This avoids unexpected behavior after copy_add_spatial_dim and simplifies the logic.
    v.feature_dim_axis = v.feature_dim_axis
    # Add dims, in case we miss any.
    for axis in range(data.batch_ndim):
      if axis == data.batch_dim_axis:
        continue
      axis_wo_batch = data.get_batch_axis_excluding_batch(axis)
      v_axis = v.get_batch_axis(axis_wo_batch)
      existing_axis = None
      if dim_tags[data][axis] in dim_tags[start]:
        existing_axis = dim_tags[start].index(dim_tags[data][axis])
      if existing_axis is None:
        # Try a bit harder to find an existing.
        if axis == data.feature_dim_axis and v.feature_dim_axis is not None:
          if v.batch_shape[v.feature_dim_axis] == data.batch_shape[axis]:
            if v.batch_shape[v.feature_dim_axis] is not None:
              existing_axis = v.feature_dim_axis  # There might be cases that the dim_tags did not match.
          if v.batch_shape[v.feature_dim_axis] == 1:
            existing_axis = v.feature_dim_axis  # Interpret the existing as broadcast dim.
        if axis == data.time_dim_axis and v.time_dim_axis is not None:
          if v.batch_shape[v.time_dim_axis] == data.batch_shape[axis]:
            if v.batch_shape[v.time_dim_axis] is not None:
              existing_axis = v.time_dim_axis  # There might be cases that the dim_tags did not match.
          if v.batch_shape[v.time_dim_axis] == 1:
            existing_axis = v.time_dim_axis  # Interpret the existing as broadcast dim.
      if existing_axis is not None:
        # We go from left to right, so we should have moved it already.
        # However, it could be that we confused some other axis earlier.
        if existing_axis > v_axis:
          v = v.copy_move_axis(old_axis=existing_axis, new_axis=v_axis)
          dim_tags[start].insert(v_axis, dim_tags[start].pop(existing_axis))  # keep consistent
        continue
      if data.batch_ndim > v.batch_ndim:
        if axis == data.feature_dim_axis:
          v = v.copy_add_feature_dim(v_axis)
        else:
          v = v.copy_add_spatial_dim(v_axis, auto_time_dim_axis=False)  # time-dim would be set later
        dim_tags[start].insert(v_axis, v.get_dim_tag(v_axis))  # keep consistent
        if axis == data.time_dim_axis and v.time_dim_axis != v_axis:
          v.time_dim_axis = v_axis
        if axis == data.feature_dim_axis and v.feature_dim_axis != v_axis:
          v.feature_dim_axis = v_axis
    # Now we assume that we have all missing axes added,
    # but they might still be in a wrong order.
    assert v.batch_ndim == data.batch_ndim
    # Now maybe move batch/feature axis.
    # We might do multiple iterations here, depending on which axis comes first.
    # This is a bit ugly, but the code is simpler.
    num_iterations = 0
    while True:
      num_iterations += 1
      assert num_iterations <= 4
      if v.batch_dim_axis != data.batch_dim_axis:
        assert data.batch_dim_axis is not None and v.batch_dim_axis is not None
        v = v.copy_with_batch_dim_axis(data.batch_dim_axis)
        assert v.batch_dim_axis == data.batch_dim_axis
        continue
      if v.feature_dim_axis != data.feature_dim_axis:
        assert data.feature_dim_axis is not None and v.feature_dim_axis is not None
        v = v.copy_with_feature_dim_axis(data.feature_dim_axis)
        assert v.feature_dim_axis == data.feature_dim_axis
        if data.feature_dim_axis_or_unspecified is NotSpecified:
          v.feature_dim_axis = NotSpecified
          assert v.feature_dim_axis == data.feature_dim_axis
        continue
      # Now we have both equal.
      break
    if data.feature_dim_axis_or_unspecified is NotSpecified and v.feature_dim_axis_or_unspecified is not NotSpecified:
      if v._default_feature_dim_axis() == v.feature_dim_axis:
        v.feature_dim_axis = NotSpecified
    if self.sparse:
      v.feature_dim_axis = NotSpecified
      v.sparse = True  # reset
      v.dim = self.dim  # reset
    if unbroadcast and any([d1 != 1 and d2 == 1 for (d1, d2) in zip(data.batch_shape, v.batch_shape)]):
      v.size_placeholder.update(data.size_placeholder or {})
      if v.placeholder is not None:
        with tf.name_scope("copy_compatible_to_unbroadcast"):
          tiles = [1] * v.batch_ndim
          for axis in range(v.batch_ndim):
            if v.batch_shape[axis] != 1:
              continue
            if except_feature and axis == v.feature_dim_axis:
              continue
            if data.batch_shape[axis] is not None:
              tiles[axis] = data.batch_shape[axis]
            elif data_dyn_shape is not None:
              tiles[axis] = data_dyn_shape[axis]
            else:
              assert data.placeholder, "need data.placeholder for unbroadcast (target data: %r)" % v
              tiles[axis] = tf.shape(data.placeholder)[axis]
          if set(tiles) != {1}:
            v.placeholder = tf.tile(v.placeholder, tiles)
      new_shape = list(v.batch_shape)
      for axis in range(v.batch_ndim):
        if except_feature and axis == data.feature_dim_axis:
          continue
        if data.batch_shape[axis] != 1 and new_shape[axis] == 1:
          new_shape[axis] = data.batch_shape[axis]
      if v.feature_dim_axis is not None:
        v.dim = new_shape[v.feature_dim_axis]
      if v.batch_dim_axis is not None:
        del new_shape[v.batch_dim_axis]
      v.shape = tuple(new_shape)
      if v.placeholder is not None and not except_feature:
        v.placeholder.set_shape(v.batch_shape)
    v.sanity_check()
    return v

  def copy_time_flattened(self):
    """
    :return: copy of myself where the time-axis is flattened away into the batch-dim-axis.
      See :func:`get_placeholder_time_flattened` and :func:`flatten_with_seq_len_mask for more details.
    :rtype: Data
    """
    assert self.batch_dim_axis is not None
    assert self.time_dim_axis is not None
    data = self.copy()
    if data.placeholder is not None:
      data.placeholder = data.get_placeholder_time_flattened()
    data.shape = tuple([
      data.batch_shape[i] for i in range(data.batch_ndim)
      if i not in (data.batch_dim_axis, data.time_dim_axis)])
    if data.size_placeholder is not None:
      if data.time_dim_axis_excluding_batch in data.size_placeholder:
        del data.size_placeholder[data.time_dim_axis_excluding_batch]
    data.time_dim_axis = None
    data.batch_dim_axis = 0
    data.sanity_check()
    return data

  def copy_extend_with_beam(self, beam):
    """
    :param SearchBeam|None beam:
    :return: copy of myself where the batch-dim is extended/multiplied by beam_size, using tile_transposed
    :rtype: Data
    """
    from .basic import get_valid_scope_name_from_str, same_control_flow_ctx, tile_transposed
    data = self.copy()
    if data.beam and data.beam == beam:
      return data
    assert data.beam is None, "incompatible beam (%r vs %r)" % (data.beam, beam)
    if beam is None:
      return data
    with tf.name_scope("%s_data_extend_with_beam" % get_valid_scope_name_from_str(self.name)):
      if data.placeholder is not None:
        with same_control_flow_ctx(data.placeholder):
          data.placeholder = tile_transposed(data.placeholder, axis=data.batch_dim_axis, multiples=beam.beam_size)
      if data.size_placeholder is not None:
        for i, v in sorted(data.size_placeholder.items()):
          tag = DimensionTag.get_tag_from_size_tensor(v)
          with same_control_flow_ctx(v):
            data.size_placeholder[i] = tile_transposed(v, axis=0, multiples=beam.beam_size)
          if tag is not None:
            tag.set_tag_on_size_tensor(data.size_placeholder[i])
      data.beam = beam
      return data

  def copy_squeeze_axes(self, axes):
    """
    :param list[int] axes: counted with batch dim
    :return: copy of myself, with squeezed axes
    :rtype: Data
    """
    from .basic import get_valid_scope_name_from_str
    assert isinstance(axes, (list, tuple))
    assert all([self.batch_shape[axis] == 1 for axis in axes])
    if not axes:
      return self.copy()
    data = self.copy()
    if data.placeholder is not None:
      data.placeholder = tf.squeeze(
        data.placeholder, axes,
        name="%s_squeeze_axes" % get_valid_scope_name_from_str(data.name))
    assert data.batch_dim_axis not in axes
    data.shape = tuple([data.shape[i] for i in range(data.ndim) if data.get_batch_axis(i) not in axes])
    if self.time_dim_axis is not None:
      if self.time_dim_axis in axes:
        data.time_dim_axis = None
      else:
        data.time_dim_axis = self.time_dim_axis - len([axis for axis in axes if axis < self.time_dim_axis])
    if not self.sparse:
      if self.feature_dim_axis is not None and self.feature_dim_axis_or_unspecified is not NotSpecified:
        if self.feature_dim_axis in axes:
          data.feature_dim_axis = None
        else:
          data.feature_dim_axis = self.feature_dim_axis - len([axis for axis in axes if axis < self.feature_dim_axis])
      # Always reset dim. We might have a different feature axis now (if it was and is unspecified, i.e. automatic).
      if data.feature_dim_axis is not None:
        data.dim = data.batch_shape[data.feature_dim_axis]
      else:
        data.dim = None
    if self.size_placeholder:
      data.size_placeholder = {
        i - len([axis for axis in axes if self.get_batch_axis_excluding_batch(axis) < i]): size
        for (i, size) in self.size_placeholder.items()}
    data.sanity_check()
    return data

  def copy_template(self, name=None, dtype=None):
    """
    :param str|None name:
    :param str|None dtype:
    :return: copy of myself, using self.get_kwargs(), without placeholder
    :rtype: Data
    """
    kwargs = self.get_kwargs(with_size_placeholder=True)
    if name:
      kwargs["name"] = name
    if dtype:
      kwargs["dtype"] = dtype
    return Data(**kwargs)

  def copy_template_excluding_axis(self, exclude_axis, name=None):
    """
    :param int exclude_axis: axis to be removed.
    :param str|None name: if set, this will be the new name.
    :return: copy of myself excluding exclude_axis axis, without placeholder.
    :rtype: Data
    """
    kwargs = self.get_kwargs()
    if exclude_axis < 0:
      exclude_axis += self.batch_ndim
      assert exclude_axis >= 0
    assert 0 <= exclude_axis < self.batch_ndim
    axis_to_exclude_wo_b = self.get_batch_axis_excluding_batch(exclude_axis)  # None if exclude_axis == batch_dim_axis
    if exclude_axis == self.feature_dim_axis:
      del kwargs["dim"]

    other_special_axes = self.get_special_axes_dict(
      counted_with_batch_dim=True, only_available=True, include_batch_dim_axis=True)
    for axis_name, axis in other_special_axes.items():
      assert axis_name in kwargs
      if axis == exclude_axis:
        del kwargs[axis_name]
      else:
        kwargs[axis_name] = axis if (axis < exclude_axis) else (axis - 1)
    if exclude_axis == self.batch_dim_axis:
      kwargs["batch_dim_axis"] = None

    new_shape = list(self.shape)
    if axis_to_exclude_wo_b is not None:
      del new_shape[axis_to_exclude_wo_b]
    kwargs["shape"] = new_shape

    if self.size_placeholder is not None:
      size_placeholder = {}
      for i, size in self.size_placeholder.items():
        if i == axis_to_exclude_wo_b:
          continue
        if axis_to_exclude_wo_b is not None and i > axis_to_exclude_wo_b:
          i -= 1
        size_placeholder[i] = size
      kwargs["size_placeholder"] = size_placeholder
    if name:
      kwargs["name"] = name
    return Data(**kwargs)

  def copy_template_excluding_spatial_dim(self, spatial_axis_num, name=None):
    """
    :param int spatial_axis_num: index in self.get_spatial_batch_axes()
    :param str|None name: if set, this will be the new name
    :return: copy of myself excluding the time-dimension without placeholder
    :rtype: Data
    """
    spatial_axes = self.get_spatial_batch_axes()
    if spatial_axis_num < 0:
      spatial_axis_num += len(spatial_axes)
      assert spatial_axis_num >= 0
    assert 0 <= spatial_axis_num < len(spatial_axes)
    axis_to_exclude = spatial_axes[spatial_axis_num]
    axis_to_exclude_wo_b = self.get_batch_axis_excluding_batch(axis_to_exclude)
    size_placeholder = None
    if self.size_placeholder is not None:
      size_placeholder = {}
      for i, size in self.size_placeholder.items():
        if i == axis_to_exclude_wo_b:
          continue
        if i > axis_to_exclude_wo_b:
          i -= 1
        size_placeholder[i] = size
    new_shape = list(self.shape)
    del new_shape[axis_to_exclude_wo_b]
    kwargs = self.get_kwargs()
    other_special_axes = self.get_special_axes_dict(
      counted_with_batch_dim=True, only_available=True, include_batch_dim_axis=True)
    for special_axis_name, special_axis in other_special_axes.items():
      if special_axis == axis_to_exclude:
        kwargs.pop(special_axis_name, None)
        continue
      kwargs[special_axis_name] = special_axis if (special_axis < axis_to_exclude) else (special_axis - 1)
    kwargs["shape"] = new_shape
    kwargs["size_placeholder"] = size_placeholder
    if name:
      kwargs["name"] = name
    return Data(**kwargs)

  def copy_template_excluding_time_dim(self, name=None):
    """
    :param str|None name: if set, this will be the new name
    :return: copy of myself excluding the time-dimension without placeholder
    :rtype: Data
    """
    assert self.batch_dim_axis is not None
    assert self.time_dim_axis is not None
    new_shape = list(self.shape)
    del new_shape[self.time_dim_axis_excluding_batch]
    kwargs = self.get_kwargs()
    if self.size_placeholder is not None:
      size = {
        i if i < self.time_dim_axis_excluding_batch else i - 1: s
        for (i, s) in self.size_placeholder.items()
        if i != self.time_dim_axis_excluding_batch}
      kwargs["size_placeholder"] = size
    other_special_axes = self.get_special_axes_dict(
      counted_with_batch_dim=True, only_available=True, include_batch_dim_axis=True)
    other_special_axes.pop("time_dim_axis", None)
    for axis_name, axis in other_special_axes.items():
      kwargs[axis_name] = axis if (axis < self.time_dim_axis) else (axis - 1)
    del kwargs["time_dim_axis"]  # maybe automatically select another one
    kwargs["shape"] = new_shape
    if name:
      kwargs["name"] = name
    return Data(**kwargs)

  def copy_template_adding_time_dim(self, name=None, time_dim_axis=0):
    """
    Adds a time-dim-axis.
    If a time-dim-axis already exists, it will anyway create this new one.

    :param str|None name: if set, this will be the new name
    :param int time_dim_axis: the new time-dim-axis index
    :return: copy of myself adding the time-dimension without placeholder
    :rtype: Data
    """
    kwargs = self.get_kwargs()
    new_shape = list(self.shape)
    new_shape.insert(time_dim_axis, None)
    other_special_axes = self.get_special_axes_dict(
      counted_with_batch_dim=True, only_available=True, include_batch_dim_axis=True)
    other_special_axes.pop("time_dim_axis", None)
    for axis_name, axis in other_special_axes.items():
      kwargs[axis_name] = axis if (axis < time_dim_axis) else (axis + 1)
    kwargs["time_dim_axis"] = time_dim_axis
    kwargs["shape"] = new_shape
    if name:
      kwargs["name"] = name
    return Data(**kwargs)

  def copy_template_replace_dim(self, axis, new_dim, new_size=None):
    """
    :param int axis:
    :param int|None new_dim:
    :param tf.Tensor|None new_size:
    :rtype: Data
    """
    out = self.copy_template()
    if axis < 0:
      assert axis + out.batch_ndim >= 0
      axis += out.batch_ndim
    assert 0 <= axis < out.batch_ndim
    if axis == out.batch_dim_axis:
      assert new_dim is None
      return out  # nothing to do
    axis_wo_b = out.get_batch_axis_excluding_batch(axis)
    new_shape = list(out.shape)
    new_shape[axis_wo_b] = new_dim
    out.shape = tuple(new_shape)
    if axis == out.feature_dim_axis:
      out.dim = new_dim
    if out.size_placeholder and axis_wo_b in out.size_placeholder:
      del out.size_placeholder[axis_wo_b]
    if new_size is not None:
      if out.size_placeholder is None:
        out.size_placeholder = {}
      out.size_placeholder[axis_wo_b] = new_size
    out.sanity_check()
    return out

  def _get_variable_dim_pattern(self):
    """
    :return: tuple with bools specifying which dims of the shape (excluding batch-dim) are of variable length.
     e.g. (time,feature), shape=(None,128), this returns (True, False)
    :rtype: tuple[bool]
    """
    return tuple([dim is None for dim in self.shape])

  def _get_var_len_axes(self):
    return sorted([i for (i, d) in enumerate(self._get_variable_dim_pattern()) if d])

  def matches_var_dim_pattern(self, other):
    """
    :param Data other:
    :return: whether the variable-dims pattern matches,
      i.e. same variable dims (get_variable_dim_pattern), same time dim, excluding batch-dim.
      i.e. the size_placeholder should be compatible.
    :rtype: bool
    """
    if self.time_dim_axis_excluding_batch != other.time_dim_axis_excluding_batch:
      return False
    return self._get_var_len_axes() == other._get_var_len_axes()

  @property
  def batch_shape(self):
    """
    :return: shape with added batch-dim. e.g. (batch,time,feat) = (None,None,128)
    :rtype: tuple[int|None]
    """
    return self.get_batch_shape(batch_dim=None)

  def get_batch_shape(self, batch_dim):
    """
    :param int|tf.Tensor|None batch_dim:
    :return: shape with added batch-dim. e.g. (batch,time,feat) = (None,None,128)
    :rtype: tuple[int|None]
    """
    if self.batch_dim_axis is not None:
      return self.shape[:self.batch_dim_axis] + (batch_dim,) + self.shape[self.batch_dim_axis:]
    return self.shape

  def get_dynamic_batch_shape(self):
    """
    :rtype: list[int|tf.Tensor]
    """
    return [self.get_dim(axis) for axis in range(self.batch_ndim)]

  @property
  def shape_dense(self):
    """
    :return: shape with feature dim axis
    :rtype: tuple[int|None]
    """
    if self.sparse:
      return self.shape + (self.dim,)  # by default, assume at the end
    return self.shape

  @property
  def shape_sparse(self):
    """
    :return: shape without feature dim axis
    :rtype: tuple[int|None]
    """
    if self.sparse:
      return self.shape
    return self.shape[:self.feature_dim_axis] + self.shape[self.feature_dim_axis + 1:]

  @property
  def batch_shape_dense(self):
    """
    :rtype: tuple[int|None]
    """
    if self.sparse:
      return self.batch_shape + (self.dim,)
    return self.batch_shape

  @property
  def ndim(self):
    """
    :rtype: int
    :return: ndim counted without batch-dim
    """
    return len(self.shape)

  @property
  def ndim_dense(self):
    """
    :rtype: int
    :return: ndim counted without batch-dim, added by 1 if we are sparse
    """
    if self.sparse:
      return self.ndim + 1
    return self.ndim

  @property
  def batch_ndim(self):
    """
    :rtype: int
    :return: ndim counted with batch-dim
    """
    if self.batch_dim_axis is not None:
      return self.ndim + 1
    return self.ndim

  @property
  def batch_ndim_dense(self):
    """
    :rtype: int
    :return: ndim counted with batch-dim, added by 1 if we are sparse
    """
    if self.sparse:
      return self.batch_ndim + 1
    return self.batch_ndim

  @property
  def is_time_major(self):
    """
    :return: whether this is in time-major format, i.e. (time,batch,...)
    :rtype: bool
    """
    return self.time_dim_axis == 0

  @property
  def is_batch_major(self):
    """
    :return: whether this is in batch-major format, i.e. (batch,...)
    :rtype: bool
    """
    return self.batch_dim_axis == 0

  @property
  def is_batch_feature_major(self):
    """
    :return: whether this is in batch-feature-major format, i.e. (batch,feature,...) (NC...)
    :rtype: bool
    """
    return self.batch_dim_axis == 0 and self.feature_dim_axis == 1

  def _default_feature_dim_axis(self):
    """
    :return: feature dim axis, counted with batch-dim
    :rtype: int|None
    """
    if self.sparse:
      return None
    if not self.shape:
      return None
    axes = [i for i in range(self.batch_ndim) if i not in [self.batch_dim_axis, self.time_dim_axis]]
    if not axes:
      # Allow same as time-dim-axis...
      axes = [i for i in range(self.batch_ndim) if i != self.batch_dim_axis]
    assert axes
    static_axes = [i for i in axes if self.batch_shape[i] is not None]
    # Prefer last static, if available.
    if static_axes:
      return static_axes[-1]
    return axes[-1]

  @property
  def feature_dim_axis(self):
    """
    :return: feature dim axis, counted with batch-dim
    :rtype: int|None
    """
    if self._feature_dim_axis is not NotSpecified:
      return self._feature_dim_axis
    return self._default_feature_dim_axis()

  @feature_dim_axis.setter
  def feature_dim_axis(self, value):
    """
    :param int|None|NotSpecified value:
    """
    assert value is NotSpecified or value is None or isinstance(value, int)
    if isinstance(value, int):
      assert 0 <= value < self.batch_ndim
    self._feature_dim_axis = value

  @property
  def feature_dim_axis_or_unspecified(self):
    """
    :return: feature dim axis, counted with batch-dim. could also be unspecified
    :rtype: int|None|NotSpecified
    """
    return self._feature_dim_axis

  @property
  def time_dim_axis_excluding_batch(self):
    """
    :rtype: int|None
    """
    if self.time_dim_axis is None:
      return None
    return self.get_batch_axis_excluding_batch(self.time_dim_axis)

  def time_dimension(self):
    """
    :return: shape(placeholder)[time_dim_axis], int scalar
    :rtype: tf.Tensor
    """
    from .basic import reuse_name_scope_of_tensor
    assert self.time_dim_axis is not None
    if self.batch_shape[self.time_dim_axis] is not None:
      return self.batch_shape[self.time_dim_axis]
    with reuse_name_scope_of_tensor(self.placeholder):
      with tf.name_scope("time_dim"):
        return tf.shape(self.placeholder)[self.time_dim_axis]

  def get_dim(self, axis):
    """
    :param int axis: counted with batch-dim
    :return: shape[axis]
    :rtype: tf.Tensor|int
    """
    if self.batch_shape[axis] is not None:
      return self.batch_shape[axis]
    return tf.shape(self.placeholder)[axis]

  def get_placeholder_as_time_major(self):
    """
    :rtype: tf.Tensor
    """
    assert self.placeholder is not None
    return self.copy_as_time_major().placeholder

  def get_placeholder_as_batch_major(self):
    """
    :rtype: tf.Tensor
    """
    assert self.placeholder is not None
    return self.copy_as_batch_major().placeholder

  def get_placeholder_with_specific_batch_dim_axis(self, batch_dim_axis):
    """
    :param int batch_dim_axis:
    :rtype: tf.Tensor
    """
    from .basic import swapaxes
    assert self.placeholder is not None
    if self.batch_dim_axis == batch_dim_axis:
      return self.placeholder
    return swapaxes(self.placeholder, batch_dim_axis, self.batch_dim_axis)

  def get_placeholder_time_flattened(self):
    """
    :return: via :func:`flatten_with_seq_len_mask`
    :rtype: tf.Tensor
    """
    from .basic import flatten_with_seq_len_mask
    assert self.placeholder is not None
    assert self.have_time_axis()
    # flatten_with_seq_len_mask only works if either time_dim_axis or batch_dim_axis is 0:
    assert 0 in [self.time_dim_axis, self.batch_dim_axis]
    seq_lens = self.size_placeholder[self.time_dim_axis_excluding_batch]
    return flatten_with_seq_len_mask(self.placeholder, seq_lens, batch_dim_axis=self.batch_dim_axis,
                                     time_dim_axis=self.time_dim_axis)

  def get_placeholder_flattened(self, keepdims=False):
    """
    :param bool keepdims: if set, it will add broadcast dimensions after the flattening behind the first axis
    :rtype: tf.Tensor
    :return: placeholder where all dynamic axes are flattened into a single axis.
      e.g. for the usual case (batch, time, dim), it becomes (batch'|time', dim),
      or (batch, time, height, dim) will also become (batch'|time', dim).
      with keep_dims, (batch, time, height, dim) will become (batch'|time', 1, 1, dim).
    """
    assert self.placeholder is not None
    x = self.placeholder
    orig_dyn_axes = self.get_spatial_batch_axes() + [self.batch_dim_axis]
    dyn_axes = list(orig_dyn_axes)
    if dyn_axes == [self.batch_dim_axis]:
      return x
    assert 0 in dyn_axes, "would need some transpose, not supported at the moment"
    assert len(dyn_axes) > 1
    orig_num_dyn_axes = len(dyn_axes)
    ndim = len(self.batch_shape)
    if self.have_time_axis():
      x = self.get_placeholder_time_flattened()
      removed_axis = max(self.time_dim_axis, self.batch_dim_axis)
      dyn_axes.remove(removed_axis)
      dyn_axes = [(i if (i < removed_axis) else (i - 1))
                  for i in dyn_axes]
      ndim -= 1
    if len(dyn_axes) > 1:
      shape = tf.shape(x)
      x = tf.reshape(
        x,
        [tf.reduce_prod([shape[i] for i in dyn_axes])] +
        [shape[i] for i in range(ndim) if i not in dyn_axes])
      dyn_axes = [0]
    assert dyn_axes == [0]
    if keepdims and orig_num_dyn_axes >= 2:
      for i in orig_dyn_axes:
        if i not in dyn_axes:
          x = tf.expand_dims(x, axis=i)
      x.set_shape([None] * self.batch_ndim)
    return x

  def get_axes(self, exclude_time=False, exclude_batch=False, exclude_feature=False):
    """
    :param bool exclude_time: will filter out the time-axis
    :param bool exclude_batch: will filter out the batch-axis
    :param bool exclude_feature: will filter out the feature-axis
    :return: list of axes, like `range(len(self.shape))`, calculated with batch dim.
    :rtype: list[int]
    """
    axes = list(range(len(self.batch_shape)))
    if exclude_time and self.time_dim_axis is not None:
      axes.pop(axes.index(self.time_dim_axis))
    if exclude_batch and self.batch_dim_axis is not None:
      axes.pop(axes.index(self.batch_dim_axis))
    if exclude_feature and self.feature_dim_axis is not None:
      axes.pop(axes.index(self.feature_dim_axis))
    return axes

  def get_axes_from_description(self, axes, allow_int=True):
    """
    :param int|list[int]|str|list[str]|None axes: one axis or multiple axis, or none.
      This is counted with batch-dim, which by default is axis 0 (see enforce_batch_dim_axis).
      It also accepts the special tokens "B"|"batch", "spatial", "spatial_except_time", or "F"|"feature",
      and more (see the code).
    :param bool allow_int: whether to allow an int directly. in almost all cases, it is better to use a symbolic name
      to specify an axis, as different layers could reorder them, and maybe also change their behavior in the future.
    :return: list of axes, counted with batch-dim
    :rtype: list[int]
    """
    if axes is None or axes == "":
      return []
    if not allow_int:
      assert not isinstance(axes, int)
    assert isinstance(axes, (str, int, list, tuple))
    if isinstance(axes, (list, tuple)):
      assert all([a is None or isinstance(a, (str, int)) for a in axes])
      if not allow_int:
        assert all([not isinstance(a, int) for a in axes])
    if isinstance(axes, str):
      import re
      axes = axes.lower()
      if axes in ["b", "batch"]:
        assert self.batch_dim_axis is not None
        axes = self.batch_dim_axis
      elif axes == "spatial":
        axes = self.get_spatial_batch_axes()
      elif re.match("(s|spatial):-?\\d+$", axes):
        s = int(axes.split(":")[1])
        spatial_axes = self.get_spatial_batch_axes()
        if s < 0:
          s += len(spatial_axes)
        assert s < len(spatial_axes), "%s get_axes_from_description: %r invalid" % (self, axes)
        axes = spatial_axes[s]
      elif axes in ["dyn", "dynamic"]:
        axes = self.get_dynamic_axes()
      elif re.match("(d|dyn|dynamic):-?\\d+$", axes):
        s = int(axes.split(":")[1])
        dyn_axes = self.get_dynamic_axes()
        if s < 0:
          s += len(dyn_axes)
        assert 0 <= s < len(dyn_axes), "%s get_axes_from_description: %r invalid" % (self, axes)
        axes = dyn_axes[s]
      elif axes == "spatial_except_time":
        axes = self.get_spatial_batch_axes()
        assert self.time_dim_axis is not None
        axes.remove(self.time_dim_axis)
      elif axes in ["t", "time"]:
        assert self.time_dim_axis is not None
        axes = self.time_dim_axis
      elif axes == "t?":
        axes = [self.time_dim_axis] if self.time_dim_axis is not None else []
      elif axes == "except_time":  # also except batch
        axes = list(range(self.batch_ndim))
        axes.remove(self.batch_dim_axis)
        if self.time_dim_axis is not None:
          axes.remove(self.time_dim_axis)
      elif axes == "except_batch":
        axes = list(range(self.batch_ndim))
        axes.remove(self.batch_dim_axis)
      elif re.match("(except_batch):-?\\d+$", axes):
        s = int(axes.split(":")[1])
        non_batch_axes = list(range(self.batch_ndim))
        if self.batch_dim_axis is not None:
          non_batch_axes.remove(self.batch_dim_axis)
        if s < 0:
          s += len(non_batch_axes)
        assert 0 <= s < len(non_batch_axes), "%s get_axes_from_description: %r invalid" % (self, axes)
        axes = non_batch_axes[s]
      elif axes == "*":
        axes = list(range(self.batch_ndim))
      elif axes == "static":
        axes = self.get_static_axes()
      elif re.match("(static):-?\\d+$", axes):
        s = int(axes.split(":")[1])
        static_axes = self.get_static_axes()
        if s < 0:
          s += len(static_axes)
        assert 0 <= s < len(static_axes), "%s get_axes_from_description: %r invalid" % (self, axes)
        axes = static_axes[s]
      elif axes in ["f", "feature", "non_spatial"]:
        axes = self.get_feature_batch_axes()
      elif all([a in "btf" for a in axes]):
        return self.get_axes_from_description(list(axes))
      elif axes.startswith("stag:"):  # spatial tag
        axes = self.get_axis_by_tag_name(axes[len("stag:"):], spatial_only=True)
      else:
        raise Exception("invalid axis mode %r" % axes)
    if isinstance(axes, int):
      axes = [axes]
    assert isinstance(axes, (tuple, list)), "invalid axis %r" % axes
    flat_axes = []
    for i in axes:
      if isinstance(i, int):
        flat_axes += [i]
      else:
        assert isinstance(i, (str, tuple, list))
        flat_axes += self.get_axes_from_description(i)
    flat_axes = [i % self.batch_ndim for i in flat_axes]
    res = []
    for i in flat_axes:
      if i not in res:
        res.append(i)
    return res

  def get_axis_from_description(self, axis, allow_int=True):
    """
    :param int|str axis:
    :param bool allow_int:
    :return: axis, counted with batch-dim
    :rtype: int
    """
    axes = self.get_axes_from_description(axis, allow_int=allow_int)
    assert len(axes) == 1, "%r: %r is not a unique axis but %r" % (self, axis, axes)
    return axes[0]

  def get_axis_by_tag_name(self, name, spatial_only=False):
    """
    :param str name: the tag name, or part of it (must be unique, and must exist)
    :param bool spatial_only:
    :rtype: int
    """
    dim_tags = self.get_batch_shape_dim_tags()
    matching_dim_tags = [(axis, tag) for axis, tag in enumerate(dim_tags) if name.lower() in tag.description.lower()]
    if spatial_only:
      matching_dim_tags = [(axis, tag) for axis, tag in matching_dim_tags if tag.kind == DimensionTag.Types.Spatial]
    assert len(matching_dim_tags) == 1, "%r: tag name %r is not unique in dim tags %r" % (self, name, dim_tags)
    return matching_dim_tags[0][0]

  def get_batch_axis_excluding_batch(self, axis):
    """
    :param int axis: counted with batch-dim
    :return: axis counted without batch-dim
    :rtype: int|None
    """
    if axis < 0:
      assert axis + self.batch_ndim >= 0
      axis += self.batch_ndim
      # Do this check only in this case;
      # we call this function early in construction where batch_ndim might be invalid.
      assert 0 <= axis < self.batch_ndim
    if self.batch_dim_axis is None:
      return axis
    if axis == self.batch_dim_axis:
      return None
    if axis < self.batch_dim_axis:
      return axis
    return axis - 1

  def get_batch_axis(self, axis):
    """
    :param int axis: counted without batch-dim
    :return: axis counted with batch-dim
    :rtype: int
    """
    if self.batch_dim_axis is None:
      return axis
    if axis >= self.batch_dim_axis:
      return axis + 1
    return axis

  def have_batch_axis(self):
    """
    :rtype: bool
    """
    return self.batch_dim_axis is not None

  def have_time_axis(self):
    """
    :rtype: bool
    """
    return self.time_dim_axis is not None

  def have_feature_axis(self):
    """
    :rtype: bool
    """
    return self.feature_dim_axis is not None

  def is_time_axis_dynamic(self):
    """
    :return: whether there are different seq-lens for the time, or all the same (static)
    :rtype: bool
    """
    assert self.time_dim_axis is not None
    if self.placeholder is None and self.size_placeholder is None:
      # Run at template construction time.
      return self.batch_shape[self.time_dim_axis_excluding_batch] is None
    if self.time_dim_axis_excluding_batch in self.size_placeholder:
      return True
    assert isinstance(self.shape[self.time_dim_axis_excluding_batch], int), (
      "%s: dynamic time axis dim (None) (axis %i) but size_placeholder %r misses information" % (
        self, self.time_dim_axis, self.size_placeholder))
    return False

  def is_axis_dynamic(self, axis):
    """
    :param int axis: counted with batch-dim axis
    :return: dynamic, i.e. we have it in size_placeholder.
      Note that this does not perfectly match with :func:`get_dynamic_axes`, but more with :func:`is_time_axis_dynamic`,
      although probably in most (all?) cases it should match.
      If True, you can get the size via :func:`get_dynamic_size`.
    :rtype: bool
    """
    if axis == self.batch_dim_axis:
      return False
    if self.placeholder is None and self.size_placeholder is None:
      # Run at template construction time.
      return self.batch_shape[axis] is None
    axis_wo_batch = self.get_batch_axis_excluding_batch(axis)
    if axis_wo_batch in self.size_placeholder:
      return True  # not quite the same as get_dynamic_axes
    assert isinstance(self.batch_shape[axis], int), (
      "%s: the requested axis has neither a size_placeholder entry nor a fixed size" % self)
    return False

  def get_dynamic_size(self, axis):
    """
    :param int axis: counted with batch-dim axis. :func:`is_axis_dynamic` should be True
    :return: shape (B,)
    :rtype: tf.Tensor
    """
    axis_wo_batch = self.get_batch_axis_excluding_batch(axis)
    return self.size_placeholder[axis_wo_batch]

  def get_dynamic_axes(self):
    """
    :return: list of axes, counted with batch-dim axis (but we exclude the batch dim axis itself)
    :rtype: list[int]
    """
    return [axis for axis, dim in enumerate(self.batch_shape)
            if axis != self.batch_dim_axis and dim is None]

  def get_static_axes(self):
    """
    :return: list of axes, counted with batch-dim axis (but we exclude the batch dim axis itself)
    :rtype: list[int]
    """
    return [axis for axis, dim in enumerate(self.batch_shape)
            if axis != self.batch_dim_axis and dim is not None]

  def mark_same_time(self, other):
    """
    If the dimension tag of others time axis matches any of our axes, we set our time axis to the selected one.

    :param Data other:
    :return: whether we have found the same
    :rtype: bool
    """
    assert other.have_time_axis()
    tag_other = other.get_dim_tag(other.time_dim_axis)
    for axis, dim_tag in enumerate(self.get_batch_shape_dim_tags()):
      if dim_tag == tag_other:
        self.time_dim_axis = axis
        return True
    return False

  def is_same_time_dim(self, other):
    """
    Checks whether we have a matching/compatible time dim.

    :param Data other:
    :rtype: bool
    """
    assert self.have_time_axis()
    if not other.have_time_axis():
      return False
    tag_self = self.get_dim_tag(self.time_dim_axis)
    tag_other = other.get_dim_tag(other.time_dim_axis)
    return tag_self == tag_other

  def get_sequence_lengths(self):
    """
    :return: seq lens tensor of shape (batch,) of dtype int32. also see :func:`get_dynamic_size`
    :rtype: tf.Tensor
    """
    from .basic import same_control_flow_ctx, expand_dims_unbroadcast
    assert self.time_dim_axis is not None
    if self.is_time_axis_dynamic():
      return self.size_placeholder[self.time_dim_axis_excluding_batch]
    assert self.shape[self.time_dim_axis_excluding_batch] is not None
    with same_control_flow_ctx(self.placeholder), tf.name_scope("fixed_seq_len"):
      return expand_dims_unbroadcast(
        self.shape[self.time_dim_axis_excluding_batch], axis=0, dim=self.get_batch_dim())

  def get_sequence_mask(self):
    """
    :return: seq mask of shape (batch,time) if we are batch-major, else (time,batch) if we are time-major
    :rtype: tf.Tensor
    """
    from .basic import sequence_mask_time_major, sequence_mask
    assert self.time_dim_axis is not None
    assert self.batch_dim_axis is not None
    if self.is_time_major:
      assert self.batch_dim_axis == 1
      return sequence_mask_time_major(self.get_sequence_lengths())
    else:
      assert self.batch_dim_axis == 0
      assert self.time_dim_axis == 1
      return sequence_mask(self.get_sequence_lengths())

  def get_sequence_mask_broadcast(self, axis=None):
    """
    :param int|None axis:
    :return: seq mask of shape ((batch,time) or (time,batch)) + (1,)s for remaining dims
      if BT or TB major, and axis is T or None.
      In general compatible to placeholder, i.e. same ndim, with broadcast dims.
      We assert here that the axis is dynamic (:func:`is_axis_dynamic`), i.e. we have the size.
    :rtype: tf.Tensor
    """
    from .basic import sequence_mask_time_major, sequence_mask
    if axis is None:
      assert self.time_dim_axis is not None
      axis = self.time_dim_axis
    assert axis != self.batch_dim_axis
    size = self.get_dynamic_size(axis)
    if axis >= self.batch_dim_axis:
      seq_mask = sequence_mask(size)  # (B,T)
    else:  # axis < batch_dim_axis
      seq_mask = sequence_mask_time_major(size)  # (T,B)
    shape = [1] * self.batch_ndim  # type: typing.List[typing.Union[int,tf.Tensor]]
    with tf.name_scope("get_sequence_mask_broadcast"):
      placeholder_shape = tf.shape(self.placeholder)
      shape[self.batch_dim_axis] = placeholder_shape[self.batch_dim_axis]
      shape[axis] = placeholder_shape[axis]
      seq_mask = tf.reshape(seq_mask, shape, name="seq_mask_reshape")
      assert seq_mask.get_shape().ndims == self.batch_ndim
    return seq_mask

  def get_batch_dim(self):
    """
    :rtype: tf.Tensor
    """
    assert self.placeholder is not None
    assert self.batch_dim_axis is not None
    return tf.shape(self.placeholder)[self.batch_dim_axis]

  def get_spatial_batch_axes(self):
    """
    :rtype: list[int]
    :return: list of axes which are not batch axes and not feature or which are time axis or dynamic.
      counted with batch-dim.
    """
    return [
      axis
      for axis in range(self.batch_ndim)
      if axis != self.batch_dim_axis
      and (axis != self.feature_dim_axis or
           axis == self.time_dim_axis or
           self.batch_shape[axis] is None)]

  def get_spatial_axes(self):
    """
    :rtype: list[int]
    :return: list of axes which are not feature and batch axes, counted without batch-dim.
    """
    return [self.get_batch_axis_excluding_batch(axis) for axis in self.get_spatial_batch_axes()]

  def get_feature_batch_axes(self):
    """
    :rtype: list[int]
    :return: list of axes which are feature axes, counted with batch-dim. currently there is only one or zero such axis.
    """
    if self.feature_dim_axis is not None:
      return [self.feature_dim_axis]
    return []

  def get_feature_axes(self):
    """
    :rtype: list[int]
    :return: list of axes which are feature axes, counted without batch-dim.
    """
    return [self.get_batch_axis_excluding_batch(axis) for axis in self.get_feature_batch_axes()]

  SpecialAxesNames = ("batch_dim_axis", "time_dim_axis", "feature_dim_axis")

  def get_special_axes_dict(self, counted_with_batch_dim=True, include_batch_dim_axis=False, only_available=False):
    """
    :param bool counted_with_batch_dim:
    :param bool include_batch_dim_axis:
    :param bool only_available:
    :return: dict axis-name -> axis
    :rtype: dict[str,int]
    """
    axes = list(self.SpecialAxesNames)
    if include_batch_dim_axis:
      assert counted_with_batch_dim
    else:
      axes.remove("batch_dim_axis")
    d = {k: getattr(self, k) for k in axes}
    if not counted_with_batch_dim:
      d = {k: self.get_batch_axis_excluding_batch(v) if (v is not None) else None
           for (k, v) in d.items()}
    if only_available:
      d = {k: v for (k, v) in d.items() if v is not None}
      if self._feature_dim_axis is NotSpecified:  # special rule
        d.pop("feature_dim_axis", None)
    return d

  def get_bc_spatial_batch_shape(self):
    """
    :return: shape which will broadcast along all spatial dimensions and time/batch dim
    :rtype: tuple[int|None]
    """
    dyn_axes = self.get_spatial_batch_axes()
    if self.batch_dim_axis is not None:
      dyn_axes += [self.batch_dim_axis]
    return tuple([1 if (axis in dyn_axes) else dim
                  for axis, dim in enumerate(self.batch_shape)])

  def get_bc_shape(self, opts=None):
    """
    :param dict[str|list|tuple,int|str|None]|None opts:
      ``key`` specifies the axes.
      ``value`` 1 ('x') is broadcasting, -1 (None) is not broadcasting
      Axes should not be defined multiple times.
      The default behavior if an axis is not specified is like :func:`get_bc_spatial_batch_shape`,
      i.e. it will broadcast in batch and spatial dims only.
    :return: shape where 1 means broadcasting, None or >1 means not broadcasting. can be used for :func:`TFUtil.dropout`
    :rtype: tuple[int|None]
    """
    if opts is None:
      opts = {}
    axes_map = {}  # int -> int|None
    for key, value in opts.items():
      assert value in (-1, 1, 'x', None), "%r get_bc_shape: invalid value in opts %r" % (self, opts)
      if value == 'x':
        value = 1
      if value == -1:
        value = None
      key_axes = self.get_axes_from_description(key)
      for key_axis in key_axes:
        assert key_axis not in axes_map, (
          "%r get_bc_shape: axis %i is defined multiple times in opts %r" % (self, key_axis, opts))
        assert 0 <= key_axis < self.batch_ndim, "%r get_bc_shape: invalid axis %i in opts %r" % (self, key_axis, opts)
        axes_map[key_axis] = self.batch_shape[key_axis] if value is None else value
    # Fill in remaining axes by defaults, just as in get_bc_spatial_batch_shape.
    remaining_axes = sorted(set(range(self.batch_ndim)).difference(axes_map.keys()))
    if remaining_axes:
      dyn_axes_list = self.get_spatial_batch_axes()
      if self.batch_dim_axis is not None:
        dyn_axes_list += [self.batch_dim_axis]
      for axis in remaining_axes:
        axes_map[axis] = 1 if axis in dyn_axes_list else self.batch_shape[axis]
    assert sorted(axes_map.keys()) == list(range(self.batch_ndim))
    return tuple([axes_map[i] for i in range(self.batch_ndim)])

  def get_scope_name(self):
    """
    :return: via self.placeholder or any self.size_placeholder, or None
    :rtype: str|None
    """
    if self.placeholder is not None:
      return os.path.dirname(self.placeholder.name)
    if self.size_placeholder:
      for i, v in sorted(self.size_placeholder.items()):
        if v is not None:
          return os.path.dirname(v.name)
    return None

  def get_full_name(self):
    """
    :return: if we have a defined scope (via :func:`self.get_scope_name`), then scope_name + "/" + self.name,
      otherwise just self.name
    :rtype: str
    """
    scope_name = self.get_scope_name()
    if scope_name:
      return "%s/%s" % (scope_name, self.name)
    return self.name

  def get_dim_tag(self, axis):
    """
    :param int axis: counted with batch-dim
    :rtype: DimensionTag
    """
    name = self.get_full_name()
    if axis == self.batch_dim_axis:
      return DimensionTag(
        kind=DimensionTag.Types.Batch, description="batch:%s" % name,
        src_data=self, src_axis=axis)
    axis_wo_b = self.get_batch_axis_excluding_batch(axis)
    dyn_size = self.size_placeholder.get(axis_wo_b) if self.size_placeholder else None
    # Note: Prefer interpretation as spatial axis if there is a dynamic size or this is marked as time axis.
    if axis == self.feature_dim_axis and dyn_size is None and axis != self.time_dim_axis:
      return DimensionTag(
        kind=DimensionTag.Types.Feature, dimension=self.dim, description="feature:%s" % name,
        src_data=self, src_axis=axis)
    if dyn_size is not None:
      tag = DimensionTag.get_tag_from_size_tensor(dyn_size)
      if tag:
        return tag
    spatial_axes = self.get_spatial_batch_axes()
    assert axis in spatial_axes
    description = "time" if axis == self.time_dim_axis else "spatial%i" % spatial_axes.index(axis)
    if dyn_size is not None:
      # Note: This case is uncommon/unexpected (we should have a dim-tag on the dyn_size above), so be verbose,
      # and fix such cases if possible (i.e. for all newly created dynamic size tensors, set the dim-tag).
      description += ":var:%r" % dyn_size.name
    elif self.batch_shape[axis] is None:
      description += ":var-unk"
    else:
      description += ":static%i" % self.batch_shape[axis]
    description += ":%s" % name
    tag = DimensionTag(
      kind=DimensionTag.Types.Spatial, description=description,
      dimension=self.batch_shape[axis], dyn_size=dyn_size,
      src_data=self, src_axis=axis)
    return tag

  def get_time_dim_tag(self):
    """
    :rtype: DimensionTag
    """
    assert self.time_dim_axis is not None
    return self.get_dim_tag(self.time_dim_axis)

  def get_size_dim_tag(self, number):
    """
    :param int number: index in sorted(size_placeholder.keys())
    :rtype: DimensionTag
    """
    axis_wo_batch = sorted(self.size_placeholder.keys())[number]
    return self.get_dim_tag(self.get_batch_axis(axis_wo_batch))

  def get_batch_shape_dim_tags(self):
    """
    :return: list of dimension tags, for each axis (counted with batch dim, i.e. len is batch_ndim)
    :rtype: tuple[DimensionTag]
    """
    return tuple([self.get_dim_tag(i) for i in range(self.batch_ndim)])

  @classmethod
  def get_common_data(cls, sources, warnings_out=None, out_shape=None):
    """
    :param list[Data] sources:
    :param io.TextIOBase|io.StringIO|typing.TextIO|None warnings_out:
    :param list[int|tf.Tensor]|None out_shape: will insert the shape dynamically
    :return: some generic data where the sources should be compatible to (with copy_compatible_to),
      i.e. it contains the union of all axes from all sources (least common multiple).
    :rtype: Data|None
    """
    assert not out_shape
    if not sources:
      return None
    assert sources
    if len(sources) == 1:
      if out_shape is not None:
        out_shape.extend(sources[0].get_dynamic_batch_shape())
      return sources[0]
    max_ndim = max([s.batch_ndim for s in sources])
    # Try with the (first) largest.
    common = [s for s in sources if s.batch_ndim == max_ndim][0]
    common = common.copy()
    if out_shape is not None:
      out_shape.extend(common.get_dynamic_batch_shape())
    if any([s.beam for s in sources]):
      # Note: we don't use copy_extend_with_beam because we don't want to create any ops in the TF graph at this point.
      common.beam = SearchBeam.get_combined_beam(*[s.beam for s in sources])
    is_equal_opts = dict(ignore_feature_dim=True, allow_same_spatial_dim=True, broadcast_matches=True)
    all_dim_tags, tags_dict = DimensionTag.get_all_dimension_tags(sources, is_equal_opts=is_equal_opts)
    # Note: We cannot compare len(all_dims_tags) to len(shape) as e.g. shape (B,1,1,D) would have only 3 dim tags.
    largest_dim_tags, tags_dict_ = DimensionTag.get_all_dimension_tags([common], is_equal_opts=is_equal_opts)
    tags_dict.update(tags_dict_)
    if len(largest_dim_tags) == len(all_dim_tags):
      return common
    # Some dim-tags are maybe not comparable (e.g. undefined time-dim-tag).
    # We fix this in some cases, i.e. by selecting unique time-dim.
    defined_var_spatial_tags = [
      tag for tag in all_dim_tags
      if tag.kind == DimensionTag.Types.Spatial and tag.get_same_base().dyn_size is not None]
    if len(defined_var_spatial_tags) == 1:
      for data in sources + [common]:
        non_comparable_dim_tags = [dim_tag for dim_tag in tags_dict[data] if not dim_tag.can_compare()]
        non_comparable_dim_tags = DimensionTag.get_uniq_collection(non_comparable_dim_tags, is_equal_opts=is_equal_opts)
        if len(non_comparable_dim_tags) == 1 and non_comparable_dim_tags[0].kind == DimensionTag.Types.Spatial:
          non_comparable_dim_tags[0].declare_same_as(defined_var_spatial_tags[0])
    non_comparable_dim_tags = [dim_tag for dim_tag in largest_dim_tags if not dim_tag.can_compare()]
    if non_comparable_dim_tags:
      if warnings_out:
        from pprint import pformat
        print(
          "get_common_data(\n%s),\ndim tags\n%s,\nlargest source\n(%s)\nhas incomplete dim tag info:\n%s" % (
            pformat(sources), pformat(all_dim_tags), pformat(common), pformat(non_comparable_dim_tags)),
          file=warnings_out)
      # The further code would be unreliable, so better have this simple fallback.
      return common
    # Ok, there is some other axis (or multiple), or we cannot identify/compare them because of incomplete information.
    # Try something more complex: Make all axes unique.
    # Note that this should also work at template construction time,
    # where we do not have access to the size_placeholder,
    # and thus the dimension tags are not reliable (in the current implementation).
    tags_dict_ext = {
      id(tag): [(data, tags_dict[data].index(tag)) for data in sources if tag in tags_dict[data]]
      for tag in all_dim_tags}
    for dim_tag in all_dim_tags:
      if not dim_tag.can_compare():
        if warnings_out:
          from pprint import pformat
          print(
            "get_common_data(\n%s),\ndim tags\n%s,\ncommon\n(%s),\ncannot compare\n%s" % (
              pformat(sources), pformat(all_dim_tags), pformat(common), pformat(dim_tag)),
            file=warnings_out)
        continue
      if not DimensionTag.get_existing_tag_from_collection(dim_tag, largest_dim_tags, is_equal_opts=is_equal_opts):
        largest_dim_tags.append(dim_tag)
        axis = common.get_default_new_axis_for_dim_tag(dim_tag)
        common = common.copy_template().copy_add_dim_by_tag(dim_tag, unbroadcast=True, axis=axis)
        if out_shape is not None:
          tag_data, tag_data_axis = tags_dict_ext[id(dim_tag)][0]
          assert isinstance(tag_data, Data)
          out_shape.insert(axis, tag_data.get_dim(tag_data_axis))
    # Simple fallback: Use first with biggest batch_ndim.
    # Was even simpler before: Use first.
    return common

  def find_matching_dims(self, dim_tag, is_equal_opts):
    """
    Finds the dimensions of this Data that match another DimensionTag

    :param DimensionTag dim_tag:
    :param dict[str,bool]|None is_equal_opts: passed to DimensionTag.is_equal
    :rtype: list[int] a list of matching axes, counted with batch dim. Sorted in ascending order
    """
    return [axis for axis in range(self.batch_ndim) if self.get_dim_tag(axis).is_equal(dim_tag, **is_equal_opts)]

  def find_matching_dim_map(self, other, other_axes):
    """
    Looks up all other_axes of another Data in this Data. Does not allow duplicates.

    :param Data other:
    :param list[int] other_axes: a list of axes of ``other``, counted with batch dim
    :return: a dict mapping other axes to own axes, all counted with batch dim
    :rtype: dict[int,int]
    """
    def map_other_axis_to_self(other_axis, taken_self_axes):
      """
      :param int other_axis: counted with batch dim
      :param set[int] taken_self_axes: axes that should not be used again
      :return: the axis of ``self`` that matches ``other_axis``, counted with batch dim
      :rtype: int
      """
      is_equal_opts = dict(
        allow_same_feature_dim=True, allow_same_spatial_dim=True, treat_feature_as_spatial=True)
      other_axis_dim_tag = other.get_dim_tag(other_axis)
      matching = [
        self_axis for self_axis in self.find_matching_dims(other_axis_dim_tag, is_equal_opts)
        if self_axis not in taken_self_axes]
      if not matching:
        # The DimensionTags do not match. Then we also allow one single dyn_size to be unknown
        is_equal_opts["unknown_spatial_matches"] = True
        matching = [
          self_axis for self_axis in self.find_matching_dims(other_axis_dim_tag, is_equal_opts)
          if self_axis not in taken_self_axes]
        assert len(matching) == 1, 'cannot match the axes %s from %s to %s' % (other_axes, other, self)
      assert matching, 'cannot match the axes %s from %s to %s' % (other_axes, other, self)
      # If there are multiple matches (e.g. because two axes have the same feature dim), leave their order intact.
      # We do this by always choosing the first unused match which is the smallest axes
      return matching[0]

    other_to_self_mapping = {}
    for axis in other_axes:
      other_to_self_mapping[axis] = map_other_axis_to_self(axis, set(other_to_self_mapping.values()))
    assert len(other_to_self_mapping) == len(other_axes), 'other_axes may not contain duplicates'
    return other_to_self_mapping
