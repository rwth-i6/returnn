
"""
Provides :class:`Data`, :class:`DimensionTag`, :class:`SearchBeam`.

See :ref:`data` for some higher-level description.
"""

from __future__ import print_function, division

import os
import typing
import tensorflow as tf
import traceback

from returnn.util.basic import NotSpecified, Entity
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
    Batch = Entity("batch")
    Spatial = Entity("spatial")  # also time
    Time = Spatial  # we don't treat this as different
    Feature = Entity("feature")

  def __init__(self, kind=Types.Unspecified, description=None,
               dimension=None, dyn_size=None, dyn_size_ext=None,
               undefined=False,
               derived_from_tag=None,
               batch=None, control_flow_ctx=None,
               src_data=None, src_axis=None):
    """
    :param Entity|None kind:
    :param str|None description: the description should be unique
    :param int|None dimension:
    :param tf.Tensor|None dyn_size: e.g. seq_len, (batch,)
    :param Data|None dyn_size_ext: seq_len or extended
    :param bool undefined: When this is specified as `None` by the user via `shape`.
    :param DimensionTag|None derived_from_tag:
      Whether this new tag is reduced, down/up sampled, padded etc from this given other tag.
      In situations where dim tags are being matched (Data.get_common_data),
      the behavior is to consider them as equal,
      and assume that the chain of operations (e.g. padding + valid conv) results in the same dim.
    :param BatchInfo|None batch: for batch-dim, or dynamic dims per batch
    :param ControlFlowContext|None control_flow_ctx:
    :param Data|None src_data:
    :param int|None src_axis:
    """
    assert not kind or isinstance(kind, Entity)
    self.kind = kind
    self.description = description
    self.dimension = dimension
    self.same_as = None  # type: typing.Optional[DimensionTag]
    self._same_as_tb = None  # type: typing.Optional[traceback.StackSummary]  # for debugging
    self.derived_from_tag = derived_from_tag
    if src_data:
      assert isinstance(src_data, Data) and isinstance(src_axis, int)
    if not batch and dyn_size_ext:
      batch = dyn_size_ext.batch
    self.batch = batch
    self.control_flow_ctx = control_flow_ctx
    self.src_data = src_data
    self.src_axis = src_axis
    if dyn_size_ext and not dyn_size_ext.batch and batch:
      dyn_size_ext.batch = batch
    if dyn_size_ext:
      assert batch == dyn_size_ext.batch
    self.dyn_size_ext = dyn_size_ext  # type: typing.Optional[Data]
    self._dyn_size_same = set()  # type: typing.Set[tf.Tensor]
    self._undefined = undefined
    # We can have different tag variants per batch info (e.g. with beam), or per control flow ctx.
    # They each have same_as = self. The same_base should have the base (global) batch info.
    self._same_for_batch_ctx = {}  # type: typing.Dict[typing.Tuple[BatchInfo,typing.Optional[ControlFlowContext]],DimensionTag]  # nopep8
    if dyn_size is not None:
      assert not dyn_size_ext
      self.dyn_size = dyn_size

  def __repr__(self):
    return "DimensionTag{%s}" % self.short_repr()

  def short_repr(self):
    """
    :return: some short repr
    :rtype: str
    """
    if self.is_batch_dim():
      return "B"
    desc = "%s%r" % ("F" if self.is_feature_dim() else "", self.get_same_base().description)
    if self.dimension is not None:
      desc += "(%i)" % self.dimension
    else:
      if self.dyn_size_ext:
        desc += "[%s]" % ",".join(self.dyn_size_ext.get_batch_axes_short_description(special_axes=False))
      else:
        desc += "[?]"
      if self.control_flow_ctx:
        desc += "{ctx=%s}" % self.control_flow_ctx.repr_inner()
    return desc

  def copy(self, kind=None):
    """
    :param Entity|None kind: if set, overwrites self.kind
    :return: copy, maybe as new kind. setting same_as to self
    :rtype: DimensionTag
    """
    tag = DimensionTag(
      kind=kind or self.kind, description=self.description,
      dimension=self.dimension, dyn_size_ext=self.dyn_size_ext,
      batch=self.batch,
      src_data=self.src_data, src_axis=self.src_axis)
    tag.same_as = self  # not declare_same_as, none of the extra checks needed
    tag._same_as_tb = traceback.extract_stack()
    return tag

  def _can_use_in_ctx(self, ctx):
    """
    :param ControlFlowContext|None ctx:
    :rtype: bool
    """
    if self.control_flow_ctx == ctx:
      return True
    if not ControlFlowContext.is_parent_or_same(self.control_flow_ctx, ctx):
      return False
    assert ctx
    # E.g. ctx == loop(time_dim), when self.control_flow_ctx == None,
    # we can use self in ctx, iff time_dim not in self.dyn_size_ext.dim_tags.
    # We can only do this check if we know about dyn_size_ext.
    if not self.dyn_size_ext:
      return False
    parent_dims = ControlFlowContext.collect_parent_dims(ctx)
    for dim in self.dyn_size_ext.dim_tags:
      if dim in parent_dims:
        return False
    return True

  def _validate_in_current_graph(self):
    """
    :rtype: bool
    """
    tensor = None
    if self.batch:
      batch_base = self.batch.get_global_base()
      if batch_base.is_global_batch():
        tensor = batch_base.get_global_batch_dim().size
    if not isinstance(tensor, tf.Tensor):
      if self.dyn_size_ext and self.dyn_size_ext.placeholder is not None:
        tensor = self.dyn_size_ext.placeholder
    if isinstance(tensor, tf.Tensor):
      g = tf_compat.v1.get_default_graph()
      if tensor.graph is not g:  # maybe from an earlier run which reuses the dim tag
        # Reset and cleanup.
        self.dyn_size_ext = None
        same_base = self.get_same_base()
        same_base._same_for_batch_ctx.pop((self.batch, self.control_flow_ctx), None)
        self.batch = None  # it is invalid in the new graph
        self.control_flow_ctx = None  # also invalid
        return False
    return True

  def _maybe_update(self):
    if self.is_batch_dim():
      return
    if isinstance(self.dimension, int):
      return
    if self.dyn_size_ext:
      return
    if not self.batch:
      return
    # Check if we can find more in
    same = self.get_for_batch_ctx(self.batch, self.control_flow_ctx, allow_none=True)
    if self is same or not same or not same.dyn_size_ext:
      return
    self.dyn_size_ext = same.dyn_size_ext

  def get_for_batch_ctx(self, batch, ctx, allow_none=False):
    """
    :param BatchInfo batch:
    :param ControlFlowContext|None ctx:
    :param bool allow_none:
    :rtype: DimensionTag|None
    """
    if self.batch == batch and self.control_flow_ctx == ctx and self.dyn_size_ext:
      self._validate_in_current_graph()
      return self
    if self.is_batch_dim():
      # We ignore the ctx for the batch dim currently.
      if self.batch == batch:
        return self
      return DimensionTag(kind=DimensionTag.Types.Batch, description="batch:%s" % batch.short_repr(), batch=batch)
    if self.dimension is not None:
      # If static dim, no effect.
      assert not self.batch
      return self
    if batch.is_broadcast():
      return self  # just leave as-is. should not matter.
    same_base = self.get_same_base()
    same_base._validate_in_current_graph()
    # Might be uninitialized in some cases. Assume batch is global.
    if not same_base.batch:
      batch_base = batch.get_global_base()
      if same_base.dyn_size_ext:
        assert batch == batch_base
        same_base.batch = batch
        assert not same_base.dyn_size_ext.batch
        same_base.dyn_size_ext.batch = batch
      else:
        same_base.batch = batch_base
    if same_base.dyn_size_ext:
      assert same_base.batch == same_base.dyn_size_ext.batch
      assert same_base.control_flow_ctx == same_base.dyn_size_ext.control_flow_ctx
    for ctx_ in ControlFlowContext.abs_ctx_stack_with_root(ctx):
      tag = same_base._same_for_batch_ctx.get((batch, ctx_), None)
      if tag and tag._can_use_in_ctx(ctx) and tag._validate_in_current_graph():
        return tag
    if same_base.batch == batch and same_base._can_use_in_ctx(ctx) and same_base.dyn_size_ext:
      return same_base
    # Ok, nothing matching found.
    dyn_size_ext = None
    # Maybe we have sth with the base batch without beam or padded batch which we can extend.
    if batch != batch.get_global_base():
      batch_base = batch.get_global_base()
      base_can_use_in_ctx = None
      if same_base.batch == batch_base and same_base._can_use_in_ctx(ctx) and same_base.dyn_size_ext:
        base_can_use_in_ctx = same_base
      else:
        for ctx_ in ControlFlowContext.abs_ctx_stack_with_root(ctx):
          tag = same_base._same_for_batch_ctx.get((batch_base, ctx_), None)
          if tag and tag._can_use_in_ctx(ctx) and tag._validate_in_current_graph() and tag.dyn_size_ext:
            base_can_use_in_ctx = tag
            break
      if base_can_use_in_ctx and base_can_use_in_ctx.dyn_size_ext:
        # The same_base has some dyn size without any beam nor control flow context.
        # We can expand it to the current beam, or extend by padded batch.
        dyn_size_ext = base_can_use_in_ctx.dyn_size_ext.copy_extend_batch(batch)
        if batch.beam:
          dyn_size_ext = base_can_use_in_ctx.dyn_size_ext.copy_extend_with_beam(batch.beam)
        assert dyn_size_ext.batch == batch
        beam_expanded_base_data = getattr(dyn_size_ext.placeholder, "_RETURNN_beam_expanded_base_data", None)
        if batch.beam:
          assert beam_expanded_base_data
        # Note: The beam expansion used tiling, which can be cached.
        # This means that we could end up with the same size tensor (placeholder) for multiple different beams,
        # when there are different beams with same beam size!
        # This breaks the current logic in get_tag_from_size_tensor.
        # As a workaround, we make an explicit new tensor here.
        from .basic import get_valid_scope_name_from_str, same_control_flow_ctx
        with same_control_flow_ctx(dyn_size_ext.placeholder):
          dyn_size_ext.placeholder = tf.identity(
            dyn_size_ext.placeholder,
            name=get_valid_scope_name_from_str("%s_get_for_batch_ctx_%s" % (dyn_size_ext.name, batch.short_repr())))
        if batch.beam:
          dyn_size_ext.placeholder._RETURNN_dyn_size_beam = batch.beam
          dyn_size_ext.placeholder._RETURNN_beam_expanded_base_data = beam_expanded_base_data
    if not dyn_size_ext and allow_none:
      return None
    dim_tag = DimensionTag(
      kind=self.kind, description=self.description, dimension=self.dimension,
      batch=batch, control_flow_ctx=dyn_size_ext.control_flow_ctx if dyn_size_ext else ctx,
      dyn_size_ext=dyn_size_ext)
    dim_tag.same_as = same_base
    dim_tag._same_as_tb = traceback.extract_stack()
    if dyn_size_ext:
      dim_tag.set_tag_on_size_tensor(dyn_size_ext.placeholder, batch=batch)
    same_base._same_for_batch_ctx[(dim_tag.batch, dim_tag.control_flow_ctx)] = dim_tag
    return dim_tag

  def set_dyn_size_ext_for_batch_ctx(self, batch, ctx, dyn_size_ext):
    """
    :param BatchInfo batch:
    :param ControlFlowContext|None ctx:
    :param Data dyn_size_ext:
    """
    same = self.get_for_batch_ctx(batch, ctx)
    same.dyn_size_ext = dyn_size_ext
    self._maybe_update()

  def get_dyn_size_ext_for_batch_ctx(self, batch, ctx):
    """
    :param BatchInfo|None batch:
    :param ControlFlowContext|None ctx:
    :rtype: Data|None
    """
    if not batch and self.batch:
      # Assume global batch.
      batch = self.batch.get_global_base()
    if not batch:
      # This is usually not valid. However, this case can happen early at initialization.
      assert batch == self.batch and ctx == self.control_flow_ctx
      return self.dyn_size_ext
    same = self.get_for_batch_ctx(batch, ctx, allow_none=True)
    if not same:
      return None
    return same.dyn_size_ext

  @property
  def dyn_size(self):
    """
    :return: dyn size / seq len (usually of shape [B]), or None
      If the dyn size can potentially be of a different shape, directly access dyn_size_ext.
    :rtype: tf.Tensor|None
    """
    if self.dyn_size_ext:
      return self.dyn_size_ext.placeholder
    return None

  @dyn_size.setter
  def dyn_size(self, dyn_size):
    """
    :param tf.Tensor dyn_size:
    """
    assert isinstance(dyn_size, tf.Tensor) and dyn_size.shape.ndims == 1
    if self.dyn_size_ext:
      # Do not allow resetting it to sth different.
      assert self.dyn_size_ext.placeholder is dyn_size
      return
    beam = getattr(dyn_size, "_RETURNN_dyn_size_beam", None)
    self.dyn_size_ext = Data(
      name=("%s:dyn_size" % self.description) if self.description else dyn_size.op.name,
      dtype=Data.size_dtype, placeholder=dyn_size, shape=(), batch_dim_axis=0,
      batch=self.batch, beam=beam, control_flow_ctx=self.control_flow_ctx)
    other = DimensionTag.get_tag_from_size_tensor(dyn_size)
    if other:
      self.declare_same_as(other)
    else:
      self.set_tag_on_size_tensor(dyn_size)

  def is_batch_dim(self):
    """
    :return: whether this dim tag is of kind batch
    :rtype: bool
    """
    return self.kind == DimensionTag.Types.Batch

  def is_feature_dim(self):
    """
    :return: whether this dim tag is of kind feature
    :rtype: bool
    """
    return self.kind == DimensionTag.Types.Feature

  def is_spatial_dim(self):
    """
    :return: whether this dim tag is of kind spatial
    :rtype: bool
    """
    return self.kind == DimensionTag.Types.Spatial

  def is_same_size_tensor(self, x):
    """
    :param tf.Tensor x:
    :return: whether this dim tag for this specific batch (incl beam) is the same as the given size
    :rtype: bool
    """
    if x is self.dyn_size:
      return True
    if x in self._dyn_size_same:
      return True
    return False

  def set_tag_on_size_tensor(self, x, batch=None, same_as_before=False):
    """
    This function is used
    to couple a tf.Tensor instance representing the dyn size
    with the dim tag.

    This is usually a newly created dim tag,
    which is yet unset.

    It is also used to couple an existing dim tag with other dyn sizes
    which just differ by an expansion of the batch (e.g. search beam).

    See also :func:`get_tag_from_size_tensor`.

    :param tf.Tensor x:
    :param BatchInfo|None batch:
    :param bool same_as_before: implies it was set before, and the new size is the same.
      e.g. it could be some identity with added checks, or other change.
    :return: self or new dim tag
    :rtype: DimensionTag
    """
    # It's unusual if self.dimension is not None, but let's accept that.
    if hasattr(x, "_is_size_of_dim_tag"):
      # noinspection PyProtectedMember
      assert x._is_size_of_dim_tag in (None, self)
    # If we already have another dyn size set or different batch, create a new DimensionTag instance.
    if self.batch and batch and self.batch != batch:
      assert not same_as_before  # it cannot be the same when it is another batch...
      new_dim_tag = self.get_for_batch_ctx(batch=batch, ctx=self.control_flow_ctx)
      new_dim_tag.set_tag_on_size_tensor(x, batch=batch)
      return new_dim_tag
    if self.dyn_size is not None and self.dyn_size is not x:
      if x in self._dyn_size_same:
        pass  # ok, pass on
      elif same_as_before:
        self._dyn_size_same.add(x)
        # And now pass on.
      else:
        assert self.batch and batch
        # It's not clear what to do. We could create a new dim tag, but the sizes might be different.
        # Usually we should not get here.
        # So for now, just error.
        from .basic import format_graph_output
        raise Exception("\n".join([
          "%r (%r) already has size %r, and another incompatible size %r (batch %r) is being assigned." % (
            self, self.description, self.dyn_size, x, batch),
          "\nNew size computation graph:",
          format_graph_output(x, max_depth=3),
          "\nThis is maybe the result of an incorrect declare_same_as. Traceback of declare_same_as:",
          "".join(self._same_as_tb.format()) if self._same_as_tb else ("same_as = %s" % self.same_as)]))
    if batch and getattr(x, "_RETURNN_dyn_size_beam", None):
      assert batch.beam == getattr(x, "_RETURNN_dyn_size_beam")
    if self.batch and batch:
      assert self.batch == batch
    elif batch and not self.batch:
      self.batch = batch  # overtake
    if getattr(x, "_is_size_of_dim_tag", None) is None:
      setattr(x, "_is_size_of_dim_tag", self)
    if self.dyn_size is None:
      self.dyn_size = x
    return self

  @classmethod
  def get_tag_from_size_tensor(cls, x):
    """
    :param tf.Tensor x: size tensor. has been set before via :func:`set_tag_on_size_tensor`
    :rtype: DimensionTag|None
    """
    return getattr(x, "_is_size_of_dim_tag", None)

  def is_equal(self, other, ignore_feature_dim=False, allow_same_feature_dim=False, allow_same_spatial_dim=None,
               treat_feature_as_spatial=False, broadcast_matches=False, unknown_spatial_matches=False,
               undefined_matches=False, derived_matches=False):
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
    :param bool undefined_matches:
    :param bool derived_matches:
    :rtype: bool
    """
    if self is other:  # first some fast path check
      return True
    if allow_same_spatial_dim is None:
      allow_same_spatial_dim = allow_same_feature_dim
    self_base = self.get_same_derived_base() if derived_matches else self.get_same_base()
    other_base = other.get_same_derived_base() if derived_matches else other.get_same_base()
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
      if unknown_spatial_matches and ((self.dyn_size is None) or (other.dyn_size is None)):
        return True
      if undefined_matches and (self.undefined or other.undefined):
        return True
    # In principle, we would want to check for identity (self is other).
    # We currently use the description because the identity would not be the same
    # in case of template construction where a dim tag is once created for a template layer,
    # and then later again for the real layer.
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
    # This must match the behavior in __eq__, which is is_equal with default options.
    # I.e. different hash implies not equal (but same hash not necessarily equal).
    if self.is_batch_dim():
      return hash(())
    base = self.get_same_base()
    return hash((base.kind, base.dimension, base.description))

  def get_same_base(self):
    """
    :rtype: DimensionTag
    """
    base = self
    while base.same_as:
      base = base.same_as
    return base

  def get_same_derived_base(self):
    """
    :rtype: DimensionTag
    """
    base = self
    while base.same_as or base.derived_from_tag:
      if base.same_as:
        base = base.same_as
        continue
      base = base.derived_from_tag
      assert base
    return base

  @property
  def undefined(self):
    """
    :rtype: bool
    """
    base = self
    while base.same_as or base.derived_from_tag:
      if base._undefined:
        return True
      if base.same_as:
        base = base.same_as
        continue
      base = base.derived_from_tag
      assert base
    return base._undefined

  def declare_same_as(self, other):
    """
    :param DimensionTag other:
    """
    self._maybe_update()
    self._validate_in_current_graph()
    if self is other:
      return
    other_same_base = other.get_same_base()
    if self is other_same_base or self.same_as is other_same_base:
      return
    if self.same_as:
      self_same_as = self.get_same_base()
      assert not self_same_as.same_as
      if self_same_as is other_same_base:
        return
      other_same_base._merge_same_for_batch_ctx_dict(self_same_as)
      self_same_as.same_as = other_same_base
      self_same_as._same_as_tb = traceback.extract_stack()
      if self_same_as.dyn_size_ext is None or not self_same_as._validate_in_current_graph():
        self_same_as.dyn_size_ext = other_same_base.get_dyn_size_ext_for_batch_ctx(
          self_same_as.batch, self_same_as.control_flow_ctx)
      elif other_same_base.dyn_size_ext is None or not other_same_base._validate_in_current_graph():
        other_same_base.dyn_size_ext = self_same_as.get_dyn_size_ext_for_batch_ctx(
          other_same_base.batch, other_same_base.control_flow_ctx)
      if (self.dyn_size_ext is None or not self._validate_in_current_graph()) and self_same_as.dyn_size_ext:
        self.dyn_size_ext = self_same_as.get_dyn_size_ext_for_batch_ctx(self.batch, self.control_flow_ctx)
    other_same_base._merge_same_for_batch_ctx_dict(self)
    self.same_as = other_same_base
    self._same_as_tb = traceback.extract_stack()
    self._maybe_update()
    if self.dyn_size is not None and other_same_base.dyn_size is not None:
      if self.dyn_size is not other_same_base.dyn_size:
        if self.batch == other_same_base.batch and self.control_flow_ctx == other_same_base.control_flow_ctx:
          # Note: Instead of making this a warning, we could also enforce this at some point.
          #   The user should be able to fix `extern_data` in the config such that this is correct in the first place.
          #   Also, in addition to this warning, we might want to add some runtime check on the eq of the dyn sizes.
          print(
            "Warning: assuming dim tags are same with different size placeholders: %r vs %r" % (
              self.dyn_size, other_same_base.dyn_size))
    # If we have a defined source, and this is a dynamic spatial axis, and it was undefined before,
    # maybe we can overtake the size_placeholder now.
    if self.same_as.dyn_size is not None and self.src_data:
      assert isinstance(self.src_axis, int)
      # Maybe it changed in the meanwhile, so check.
      tag = self.src_data.get_dim_tag(self.src_axis)
      if tag.description == self.description and (not tag.dyn_size_ext or not tag._validate_in_current_graph()):
        tag.dyn_size_ext = self.get_dyn_size_ext_for_batch_ctx(tag.batch, tag.control_flow_ctx)
    # If others dyn_size is None but we have a dyn_size, maybe update others dyn_size.
    if self.dyn_size is not None and self.same_as.dyn_size is not self.dyn_size:
      # Could be unset if it comes from the config, or from prev graph creation.
      # This is important such that self.can_compare() is sane.
      if self.same_as.dyn_size is None or not self.same_as._validate_in_current_graph():
        self.same_as.dyn_size_ext = self.get_dyn_size_ext_for_batch_ctx(
          self.same_as.batch, self.same_as.control_flow_ctx)
    if (not self.dyn_size_ext or not self._validate_in_current_graph()) and other.dyn_size_ext:
      self.dyn_size_ext = other.get_dyn_size_ext_for_batch_ctx(self.batch, self.control_flow_ctx)

  def _merge_same_for_batch_ctx_dict(self, other):
    """
    :param DimensionTag other:
    """
    self._validate_in_current_graph()
    for _, dim in list(self._same_for_batch_ctx.items()):
      assert isinstance(dim, DimensionTag)
      dim._validate_in_current_graph()
    for key, dim in other._same_for_batch_ctx.items():
      if not dim._validate_in_current_graph():
        continue
      self_dim = self._same_for_batch_ctx.get(key, None)
      if self_dim and (self_dim.dyn_size_ext or not dim.dyn_size_ext):
        continue  # keep ours
      if not dim.dyn_size_ext:
        continue  # undefined, do not overtake
      self._same_for_batch_ctx[key] = dim
    other._same_for_batch_ctx.clear()  # we only want to have it once

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
    # We do potential multiple rounds, such that we prefer "more equal" (using less is_equal_opts).
    rounds = [{}]
    if is_equal_opts:
      if "broadcast_matches" in is_equal_opts:
        rounds.append({k: v for (k, v) in is_equal_opts.items() if k != "broadcast_matches"})
      rounds.append(is_equal_opts)
    for _is_equal_opts in rounds:
      for _tag in tags:
        if _tag.is_equal(other, **_is_equal_opts):
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
        if existing_tag:
          if existing_tag.undefined and not tag.undefined and tag.dimension == existing_tag.dimension:
            # Replace the existing by the new tag.
            tags[tags.index(existing_tag)] = tag
            existing_tag = tag
        else:  # no existing tag
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

  def get_dim_value(self):
    """
    Infers the dim this axis should have if unbroadcasted.
    If `self.src_data` has a placeholder, will use the shape from there.
    Otherwise, uses `self.dimension` (if static) or `self.dyn_size` (if dynamic).
    :rtype: int|tf.Tensor
    """
    if self.dimension is not None:
      return self.dimension
    if self.is_batch_dim():
      if self.src_data:
        return self.src_data.get_batch_dim()
      from returnn.tf.layers.base import LayerBase
      return LayerBase.get_recent_layer().get_batch_dim()
    if self.src_data is not None and self.src_axis is not None and self.src_data.placeholder is not None:
      from returnn.tf.util.basic import get_shape_dim
      return get_shape_dim(self.src_data.placeholder, self.src_axis)
    if self.dyn_size is not None:
      return tf.math.reduce_max(self.dyn_size)
    raise Exception('%s: need placeholder, self.dimension or self.dyn_size for dim value' % self)


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
    def __init__(self, size):
      """
      :param tf.Tensor|int size:
      """
      self.size = size

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
    def short_repr(self):
      """
      :rtype: str
      """
      if isinstance(self.size, int):
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
      :param DimensionTag dim_tag:
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
      :param DimensionTag dim_tag:
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
    # These self._global_... attributes are meant
    # to be accessed only via the global (root) object (via get_global_base).
    # They store global information.
    # We don't use class attributes because this should not be global per process but only per network.
    self._global_beam_dims_by_beam_name = {}  # type: typing.Dict[str,BatchInfo.BeamDim]
    self._global_padded_dims_by_dim_tag = {}  # type: typing.Dict[DimensionTag,BatchInfo.PaddedDim]
    self._packed_dims_by_dim_tag = {}  # type: typing.Dict[DimensionTag,BatchInfo.PackedDim]
    self.descendants = []  # type: typing.List[BatchInfo]
    self._descendants_by_beam_name = {}  # type: typing.Dict[str,BatchInfo]
    self._global_descendants_by_virtual_dims = {}  # type: typing.Dict[typing.Tuple[BatchInfo.VirtualDimBase,...],BatchInfo]  # noqa
    if base:
      base.descendants.append(self)
      if isinstance(new_dim, BatchInfo.BeamDim):
        beam = new_dim.beam
        assert beam.name not in base._descendants_by_beam_name
        base._descendants_by_beam_name[beam.name] = self
    global_base = self.get_global_base()
    assert tuple(self.virtual_dims) not in global_base._global_descendants_by_virtual_dims
    global_base._global_descendants_by_virtual_dims[tuple(self.virtual_dims)] = self

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
            if type(dim_) == type(dim):
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
    assert len(global_batch_dims) == 1
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
    raise NotImplementedError("%r.dim()" % self)

  @dim.setter
  def dim(self, value):
    """
    :param tf.Tensor|int value:
    """
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
    :param DimensionTag dim_tag:
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
    :param DimensionTag dim_tag:
    :rtype: BatchInfo.PaddedDim
    """
    assert self.virtual_dims
    root = self.get_global_base()
    assert dim_tag.dyn_size is not None
    dim_tag_base = dim_tag.get_same_base()
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
      new_dim_index=self.virtual_dims.index(self.get_global_batch_dim()) + 1)

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
    :param DimensionTag dim_tag:
    :param bool batch_major: if True, add new dim in front. otherwise, add new dim at the end
    :rtype: BatchInfo
    """
    new_dim = self._make_packed_dim(dim_tag)
    new_dim_idx = -1 if batch_major else self._next_spatial_major_index()
    return self._copy_extend_dim(new_dim=new_dim, new_dim_idx=new_dim_idx)

  def copy_extend_with_padded_dim_tag(self, dim_tag, batch_major):
    """
    :param DimensionTag dim_tag:
    :param bool batch_major: if True, add new dim in front. otherwise, add new dim at the end
    :rtype: BatchInfo
    """
    new_dim = self._make_padded_dim(dim_tag)
    new_dim_idx = -1 if batch_major else self._next_spatial_major_index()
    return self._copy_extend_dim(new_dim=new_dim, new_dim_idx=new_dim_idx)

  def copy_extend_with_padded_or_fixed_dim_tag(self, dim_tag, batch_major):
    """
    :param DimensionTag dim_tag:
    :param bool batch_major: if True, add new dim in front. otherwise, add new dim at the end
    :rtype: BatchInfo
    """
    if dim_tag.dyn_size is not None:
      new_dim = self._make_padded_dim(dim_tag)
    else:
      new_dim = BatchInfo.FixedDim(size=dim_tag.get_dim_value())
    new_dim_idx = -1 if batch_major else self._next_spatial_major_index()
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
               batch_dim_axis=NotSpecified,
               time_dim_axis=NotSpecified,
               feature_dim_axis=NotSpecified,
               available_for_inference=True,
               auto_create_placeholders=False,
               vocab=None,
               dim_tags=None,
               same_dim_tags_as=None,
               batch=None,
               beam=None,
               control_flow_ctx=None):
    """
    :param str name:
    :param tuple[int|None]|list[int|None] shape: including time-dim (can be None). excluding batch-dim.
      e.g. (time,feat)=(None,128)
    :param str dtype: e.g. "float32" or "int64"
    :param tf.Tensor|None placeholder: with added batch-dim
    :param bool sparse: whether to treat the value as an index. do not confuse with tf.SparseTensor
    :param None|int dim: feature dimension, shape[-1] if not sparse, otherwise like num_classes
    :param int|None|NotSpecified batch_dim_axis: where we add the batch-dim.
      e.g. shape=(time,...), 0 -> (batch,time,...), 1 -> (time,batch,...).
      Default is 0.
      This is normally always set, and a lot of code expects this. However, you can set it to None
      if this Data does not have a batch-dim.
    :param int|None|NotSpecified time_dim_axis: where we have the time dim axis, after we added the batch-dim.
      this is often 1. however, can be None if there is no time-dim.
    :param int|None|NotSpecified feature_dim_axis: feature dim axis. by default it's the last one
    :param dict[int,tf.Tensor]|None size_placeholder: for every None in shape, this will describe the size.
      The size is always a tensor of shape (batch,), i.e. the size can be different for each sequence in a batch.
    :param bool available_for_inference: e.g. the extern data "classes" is usually not available for inference
    :param bool auto_create_placeholders: This will create a tf.placeholder.
    :param str|dict[str]|GeneratingDataset.Vocabulary|None vocab:
    :param tuple[DimensionTag]|list[DimensionTag]|dict[int,DimensionTag]|None dim_tags:
      If tuple/list, this specifies the whole (batch) shape.
      If dict, explicitly specified dimension tags per axis (axis counted with batch-dim)
    :param dict[int|str,DimensionTag]|None same_dim_tags_as: will mark our dimension tags to be the same
    :param BatchInfo|None batch:
    :param SearchBeam|None beam: the batch-dim could be extended by a beam-size,
      such that it represents the merged dims [batch, beam_size].
    :param ControlFlowContext|None control_flow_ctx:
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
    if beam and batch:
      assert batch.beam == beam
    self._batch = batch
    self._beam = beam
    self.control_flow_ctx = control_flow_ctx
    if isinstance(dim_tags, (tuple, list)):
      # We do a couple of sanity checks, and maybe set special axes attribs.
      shape_ = tuple(tag.dimension for tag in dim_tags if not tag.is_batch_dim())
      if shape is not None:
        assert tuple(shape) == shape_
      del shape
      batch_dim_axis_ = _batch_dim_axis_from_dim_tags_tuple(dim_tags)
      if batch_dim_axis is not NotSpecified:
        assert batch_dim_axis == batch_dim_axis_
      del batch_dim_axis
      if time_dim_axis is NotSpecified:
        time_dim_axis = _default_time_dim_axis_dim_tags(dim_tags)
      dim_tags = tuple(dim_tags)
      if auto_create_placeholders:
        _auto_create_size_placeholders_on_dim_tags(name=name, dim_tags=dim_tags)
      del shape_
      del batch_dim_axis_
    else:
      if batch_dim_axis is NotSpecified:
        batch_dim_axis = 0
      if shape is None:
        if time_dim_axis is NotSpecified:
          time_dim_axis = _default_time_dim_axis_no_shape(
            batch_dim_axis=batch_dim_axis, feature_dim_axis=feature_dim_axis)
        shape, time_dim_axis = _infer_default_shape_and_time(
          batch_dim_axis=batch_dim_axis, feature_dim_axis=feature_dim_axis, time_dim_axis=time_dim_axis,
          sparse=sparse, dim=dim)
      else:
        if time_dim_axis is NotSpecified:
          time_dim_axis = _default_time_dim_axis(batch_dim_axis=batch_dim_axis, shape=shape)
      dim_tags = _infer_dim_tags_tuple_from_shape(
        shape, batch_dim_axis=batch_dim_axis, time_dim_axis=time_dim_axis, feature_dim_axis=feature_dim_axis,
        size_placeholder=size_placeholder, name=name,
        auto_create_placeholders=auto_create_placeholders,
        dim_tags=dim_tags, sparse=sparse)
      del batch_dim_axis
      del shape
    self._dim_tags = dim_tags  # type: typing.Tuple[DimensionTag]
    if feature_dim_axis is not NotSpecified:
      if isinstance(feature_dim_axis, int):
        assert not self.sparse, "cannot have feature_dim_axis when sparse"
        if feature_dim_axis < 0:
          feature_dim_axis += self.batch_ndim
        assert 0 <= feature_dim_axis < self.batch_ndim
    self._feature_dim_axis = feature_dim_axis
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
    self._placeholder = placeholder  # type: tf.Tensor  # this will hold the data value itself
    self.available_for_inference = available_for_inference
    if vocab is not None:
      from returnn.datasets.util.vocabulary import Vocabulary
      if isinstance(vocab, str):
        vocab = Vocabulary(vocab)
      elif isinstance(vocab, dict):
        vocab = Vocabulary.create_vocab(**vocab)
      assert isinstance(vocab, Vocabulary)
      assert self.sparse, "%s should represent indices of %s" % (self, vocab)
      assert self.dim == vocab.num_labels, "%s dims do not match with vocab %s" % (self, vocab)
    self.vocab = vocab  # type: typing.Optional[Vocabulary]
    # The size_placeholder is for each variable length dimension in shape, i.e. excluding the batch-dim.
    if size_placeholder:
      self.size_placeholder = size_placeholder  # type: typing.Dict[int,tf.Tensor]  # axis w.o. batch -> size (batch,)
    if same_dim_tags_as:
      for _axis, _dim_tag in sorted(same_dim_tags_as.items()):
        _axis = self.get_axis_from_description(_axis)
        assert isinstance(_dim_tag, DimensionTag)
        base_tag = self._dim_tags[_axis]
        if base_tag != _dim_tag:
          base_tag.declare_same_as(_dim_tag)
          if _dim_tag.dyn_size is not None:
            self.set_dynamic_size(_axis, _dim_tag.dyn_size)
    self._adapt_batch_consistent_dim_tags()
    self.sanity_check(assume_complete=False)

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

  def sanity_check(self, ignore_placeholder=False, assume_complete=True):
    """
    Performs some sanity checks on self, and raises exceptions if something is not sane.

    :param bool ignore_placeholder:
    :param bool assume_complete:
    """
    for axis_name, axis in self.get_special_axes_dict().items():
      assert axis is None or 0 <= axis < self.batch_ndim, "%s: axis %s (%i) invalid" % (self, axis_name, axis)
    if self.batch_dim_axis is not None:
      for axis_name, axis in self.get_special_axes_dict().items():
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
    for axis, tag in enumerate(self.dim_tags):
      assert self.batch_shape[axis] == tag.dimension
      if tag.is_batch_dim():
        assert axis == self.batch_dim_axis, "%s: invalid %s" % (self, tag)
        continue  # further checks will assume not batch
      assert axis != self.batch_dim_axis, "%s: invalid %s" % (self, tag)
      # Note: tag.kind (feature or spatial) is independent from self.feature_dim_axis.
      if tag.batch and self.batch:
        assert tag.batch == self.batch or self.batch.is_broadcast()
      if tag.dyn_size_ext:
        assert tag.dyn_size_ext.dtype in {"int32", "int64"}
        assert tag.batch == tag.dyn_size_ext.batch
        tag.dyn_size_ext.sanity_check()
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
      # Currently only if placeholder is set.
      # We can later always do the check even without placeholder.
      if assume_complete:
        for tag in self.dim_tags:
          if tag.dimension is None:
            if tag.is_batch_dim():
              continue
            assert tag.dyn_size is not None

  def get_runtime_sanity_check_op(self):
    """
    :return: op which does a couple of runtime sanity checks on the placeholder
    :rtype: tf.Operation
    """
    assert self.placeholder is not None
    checks = []
    with tf.name_scope("runtime_sanity_check"):
      shape = tf.shape(self.placeholder)
      batch_dim = shape[self.batch_dim_axis] if self.have_batch_axis() else 1
      rank = tf.rank(self.placeholder)
      data = ["Data.get_runtime_sanity_check_op:", str(self), "shape", shape]
      for i, tag in enumerate(self.dim_tags):
        if tag.dyn_size is not None:
          data += [
            "dyn_size[%i] (%s)" % (i, tag), tag.dyn_size, ".shape", tf.shape(tag.dyn_size)]
      checks += [tf.Assert(tf.equal(rank, self.batch_ndim), data + ["-> invalid rank"])]
      if self.have_batch_axis():
        batch_dim_via_info = self.get_batch_dim()
        checks += [
          tf.Assert(tf.equal(batch_dim, batch_dim_via_info), data + ["-> invalid batch dim info", batch_dim_via_info])]
      for i in range(self.batch_ndim):
        if self.batch_shape[i] is not None:
          checks += [tf.Assert(tf.equal(shape[i], self.batch_shape[i]), data + ["-> invalid shape[%i]" % i])]
        dyn_size_ext = self.dim_tags[i].dyn_size_ext
        if dyn_size_ext and dyn_size_ext.placeholder is not None:
          dyn_size = dyn_size_ext.placeholder
          if dyn_size_ext.have_batch_axis() and self.have_batch_axis():
            checks += [tf.Assert(
              tf.equal(tf.shape(dyn_size)[dyn_size_ext.batch_dim_axis], batch_dim),
              data + ["-> invalid axis %i tag dyn size batch dim" % i])]
          checks += [tf.Assert(
            # Note: in almost all cases, we have equality here.
            # However, not strictly in all cases, e.g. DecideLayer, maybe some others...
            # But that should not be more than 1 less.
            tf.logical_or(
              tf.logical_and(
                tf.less_equal(tf.reduce_max(dyn_size), shape[i]),
                tf.greater_equal(tf.reduce_max(dyn_size), shape[i] - 1)),
              # In other rare cases, this might be a broadcast dim
              # (e.g. as initial values of att weights for a rec loop).
              tf.equal(1, shape[i])),
            data + ["-> invalid shape[%i] or max(dyn_size[%i])" % (i, i)])]
          checks += [dyn_size_ext.get_runtime_sanity_check_op()]
    return tf.group(*checks)

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

  def get_kwargs(self, include_special_axes=True):
    """
    :param bool include_special_axes: whether to include time and feature special axis marker
    :return: relevant attrib items for copying
    :rtype: dict[str]
    """
    keys = ["name", "dim_tags", "dtype"]
    if include_special_axes:
      keys += ["time_dim_axis"]
      if self._feature_dim_axis is not NotSpecified:
        keys += ["feature_dim_axis"]
    if self.sparse:
      # Sparse is False by default. And the dim is inferred from the feature dim, or otherwise does not make sense.
      keys += ["sparse", "dim"]
    if self.vocab:
      keys += ["vocab"]
    if self.batch is not None:
      keys += ["batch"]
    if self.beam is not None:
      keys += ["beam"]
    if self.control_flow_ctx:
      keys += ["control_flow_ctx"]
    if not self.available_for_inference:
      keys += ["available_for_inference"]
    return {key: getattr(self, key) for key in keys}

  def get_description(self, with_name=True, with_placeholder=False, catch_exceptions=False):
    """
    :param bool with_name:
    :param bool with_placeholder:
    :param bool catch_exceptions:
    :return: description of self. also used for __repr__
    :rtype: str
    """
    # Avoid redundant information (most information is covered in batch_shape_meta).
    # Also try to avoid confusion (e.g. `shape` vs `batch_shape`).
    keys = []
    if self.sparse:
      keys.append("dtype")
      keys.append("sparse")
      keys.append("dim")
    else:
      if self.dtype != "float32":
        keys.append("dtype")
    if with_placeholder:
      keys.append("placeholder")
    if not self.available_for_inference:
      keys.append("available_for_inference")
    if self.beam is not None:
      # With batch, it should be contained already in batch_shape_meta (if everything is correct).
      # We anyway add it, in case sth is incorrect or incomplete.
      if not self.batch or self.batch.beam != self.beam:
        keys.append("beam")
    args = []
    if with_name:
      name = getattr(self, "name", None)
      args += [repr(name) if name else "<undefined>"]
    try:
      batch_shape_meta = "[%s]" % ",".join(self.get_batch_axes_short_description())
    except Exception as exc:
      if catch_exceptions:
        batch_shape_meta = "<!%s: %s>" % (type(exc).__name__, exc)
      else:
        raise
    args += [batch_shape_meta]
    for key in keys:
      try:
        value_repr = repr(getattr(self, key))
      except Exception as exc:
        if catch_exceptions:
          value_repr = "<!%s: %s>" % (type(exc).__name__, exc)
        else:
          raise
      args += ["%s=%s" % (key, value_repr)]
    if self.control_flow_ctx:
      try:
        value_repr = self.control_flow_ctx.repr_inner()
      except Exception as exc:
        if catch_exceptions:
          value_repr = "<!%s: %s>" % (type(exc).__name__, exc)
        else:
          raise
      args += ["ctx=" + value_repr]
    return "Data{%s}" % ", ".join(args)

  def get_batch_axes_short_description(self, special_axes=True):
    """
    :param bool special_axes: special markers for old-style time_dim_axis and feature_dim_axis
    :rtype: list[str]
    """
    res = []
    for axis, dim_tag in enumerate(self.dim_tags):
      descriptions = []
      if axis == self.batch_dim_axis:
        if self.batch:
          descriptions.append(self.batch.short_repr())
        else:
          descriptions.append("B?")
      if special_axes:
        if axis == self.time_dim_axis:
          descriptions.append("T")
        if axis == self.feature_dim_axis:
          descriptions.append("F")
      if self.batch_shape[axis] is None:
        if axis == self.batch_dim_axis:
          pass  # expected
        else:
          descriptions.append(dim_tag.short_repr())
      elif axis != self.batch_dim_axis or not self.batch:
        descriptions.append(dim_tag.short_repr())
      res.append("|".join(descriptions) or "?")
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
      self.dim_tags,
      self.batch, self.beam)

  def __repr__(self):
    return self.get_description(catch_exceptions=True)

  def __hash__(self):
    return id(self)

  def _adapt_batch_consistent_dim_tags(self):
    if not self.batch:  # uninitialized
      return
    self._dim_tags = tuple(
      tag.get_for_batch_ctx(batch=self.batch, ctx=self.control_flow_ctx) for tag in self._dim_tags)

  def copy(self, name=None):
    """
    :param str name: if given, will overwrite this name
    :return: copy of myself, using self.get_kwargs(), and with placeholder and size_placeholder
    :rtype: Data
    """
    data = Data(**self.get_kwargs())
    data.placeholder = self.placeholder
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

  def copy_transpose(self, perm):
    """
    :param list[int] perm: permutation of the axes, counted with batch-dim.
      Maps the new axes to the old axes
    :return: copy of myself with permuted axes
    :rtype: Data
    """
    assert len(perm) == self.batch_ndim
    assert set(perm) == set(range(self.batch_ndim))
    if all(perm[axis] == axis for axis in range(self.batch_ndim)):
      return self.copy()
    inv_perm_ = {j: i for (i, j) in enumerate(perm)}
    inv_perm = [j for (i, j) in sorted(inv_perm_.items())]

    def translate_axis(axis):
      """
      :param int|None axis: counted with batch-dim
      :return: translated axis (if not None)
      :rtype: int|None
      """
      if axis is None:
        return None
      return inv_perm[axis]

    data_opts = self.get_kwargs(include_special_axes=False)
    if self.placeholder is not None:
      from returnn.tf.util.basic import get_valid_scope_name_from_str
      data_opts["placeholder"] = tf.transpose(
        self.placeholder, perm, name="%s_transpose" % get_valid_scope_name_from_str(self.name))
    if self.feature_dim_axis_or_unspecified is not NotSpecified:
      data_opts["feature_dim_axis"] = translate_axis(self.feature_dim_axis)
    data_opts["time_dim_axis"] = translate_axis(self.time_dim_axis)
    data_opts["dim_tags"] = tuple(self.dim_tags[perm[i]] for i in range(self.batch_ndim))
    data = Data(**data_opts)
    data.sanity_check()
    return data

  def copy_move_axis(self, old_axis, new_axis):
    """
    :param int old_axis: counted with batch-dim
    :param int new_axis: counted with batch-dim
    :return: copy of myself with moved axis (see :func:`move_axis`)
    :rtype: Data
    """
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

    perm = list(range(self.batch_ndim))
    old = perm.pop(old_axis)
    perm.insert(new_axis, old)
    return self.copy_transpose(perm)

  def copy_swap_axes(self, axis1, axis2):
    """
    Like :func:`Data.copy_move_axis`, but keeps all other axes unchanged.
    :param int axis1: counted with batch-dim
    :param int axis2: counted with batch-dim
    :return: copy of myself with moved axis (see :func:`swapaxes`)
    :rtype: Data
    """
    if axis1 < 0:
      axis1 += self.batch_ndim
    assert 0 <= axis1 < self.batch_ndim
    if axis2 < 0:
      axis2 += self.batch_ndim
    assert 0 <= axis2 < self.batch_ndim
    if axis1 == axis2:
      return self.copy()

    perm = list(range(self.batch_ndim))
    perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
    return self.copy_transpose(perm)

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

  def copy_add_batch_dim(self, batch_dim_axis, batch=None, dim_tag=None):
    """
    :param int batch_dim_axis:
    :param BatchInfo|None batch:
    :param DimensionTag|None dim_tag:
    :return: copy of myself with added batch-dim
    :rtype: Data
    """
    assert self.batch_dim_axis is None
    if not batch:
      from returnn.tf.layers.base import LayerBase
      batch = LayerBase.get_recent_layer().get_batch_info()
    if batch_dim_axis < 0:
      assert batch_dim_axis + self.batch_ndim + 1 >= 0
      batch_dim_axis += self.batch_ndim + 1
    assert 0 <= batch_dim_axis <= self.batch_ndim
    data_opts = self.get_kwargs(include_special_axes=False)
    placeholder = self.placeholder
    if placeholder is not None:
      from .basic import get_valid_scope_name_from_str
      placeholder = tf.expand_dims(
        self.placeholder, batch_dim_axis, name=get_valid_scope_name_from_str("%s_add_batch_dim" % self.name))
      if not isinstance(batch.dim, int) or batch.dim != 1:
        tiles = [1] * batch_dim_axis + [batch.dim] + [1] * (self.batch_ndim - batch_dim_axis)
        placeholder = tf.tile(placeholder, tiles)
    dim_tags = list(self.dim_tags)
    if dim_tag:
      assert dim_tag.is_batch_dim()
      assert dim_tag.dimension == batch.static_dim
      assert dim_tag.batch == batch
    else:
      dim_tag = DimensionTag(
        kind=DimensionTag.Types.Batch, description="batch", dimension=batch.static_dim, batch=batch)
    dim_tags.insert(batch_dim_axis, dim_tag)
    data_opts["dim_tags"] = dim_tags
    data_opts["batch"] = batch
    data_opts["beam"] = batch.beam
    other_special_axes = self.get_special_axes_dict(counted_with_batch_dim=True, only_available=True)
    for k, a in other_special_axes.items():
      data_opts[k] = a if (a < batch_dim_axis) else (a + 1)
    return Data(placeholder=placeholder, **data_opts)

  def copy_add_spatial_dim(self, spatial_dim_axis=None, dim=1, auto_time_dim_axis=True):
    """
    :param int|None spatial_dim_axis: counted with batch-dim. if there is no time-dim, this will be it.
    :param int|None dim:
    :param bool auto_time_dim_axis:
    :return: copy of myself with added spatial-dim
    :rtype: Data
    """
    if dim is None:
      assert not self.placeholder
    dim_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description="added_spatial", dimension=dim)
    if spatial_dim_axis is None:
      spatial_dim_axis = self.get_default_new_axis_for_dim_tag(dim_tag)
    v = self.copy_add_dim_by_tag(dim_tag, unbroadcast=True, axis=spatial_dim_axis)
    if auto_time_dim_axis and self.time_dim_axis is None:
      v.time_dim_axis = spatial_dim_axis
    return v

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
    dim_tag = DimensionTag(kind=DimensionTag.Types.Feature, description="feature1", dimension=1)
    if axis is None:
      axis = self.get_default_new_axis_for_dim_tag(dim_tag)
    v = self.copy_add_dim_by_tag(dim_tag, axis=axis)
    if v.feature_dim_axis_or_unspecified is not NotSpecified:
      v.feature_dim_axis = NotSpecified
    if v.feature_dim_axis != axis:
      v.feature_dim_axis = axis
    return v

  def get_default_new_axis_for_dim_tag(self, dim_tag):
    """
    :param DimensionTag dim_tag:
    :rtype: int
    """
    if dim_tag.is_batch_dim():
      return 0
    # Note: if dim_tag is feature, but we are sparse, we just treat is as spatial, handled below.
    if dim_tag.is_feature_dim() and not self.sparse:
      if self.feature_dim_axis is not None:
        return self.feature_dim_axis + 1  # after existing feature-dim
      else:
        return self.batch_ndim  # at the end
    assert dim_tag.is_spatial_dim() or (dim_tag.is_feature_dim() and self.sparse)
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
    :param bool unbroadcast: If True unbroadcast the newly added axis.
      Will infer the unbroadcast shape via :func:`DimensionTag.get_dim_value`
    :param int|None axis:
    :rtype: Data
    """
    from .basic import get_valid_scope_name_from_str
    if axis is None:
      axis = self.get_default_new_axis_for_dim_tag(dim_tag=dim_tag)
    if axis < 0:
      axis += self.batch_ndim + 1
    assert 0 <= axis <= self.batch_ndim

    if dim_tag.is_batch_dim():
      if unbroadcast:
        batch_info = dim_tag.src_data.batch if dim_tag.src_data else None
        return self.copy_add_batch_dim(batch_dim_axis=axis, batch=batch_info, dim_tag=dim_tag)
      else:
        batch_info = BatchInfo.make_global_broadcast_batch_info()
        return self.copy_add_batch_dim(
          batch_dim_axis=axis, batch=batch_info,
          dim_tag=dim_tag if (dim_tag.dimension == 1 and dim_tag.batch == batch_info) else None)

    data_opts = self.get_kwargs()
    # Note: if dim_tag is feature, but we are sparse, we just make it spatial
    if self.sparse and dim_tag.is_feature_dim():
      dim_tag = dim_tag.copy(kind=DimensionTag.Types.Spatial)
    if not unbroadcast and dim_tag.dimension != 1:
      dim_tag = DimensionTag(
        kind=dim_tag.kind, description="%s_dummy_dim1" % (dim_tag.description or "unnamed"), dimension=1)
    data_opts["dim_tags"] = self.dim_tags[:axis] + (dim_tag,) + self.dim_tags[axis:]
    other_special_axes = self.get_special_axes_dict(counted_with_batch_dim=True, only_available=True)
    for k, a in other_special_axes.items():
      data_opts[k] = a if (a < axis) else (a + 1)
    if dim_tag.is_feature_dim() and self.feature_dim_axis is None:
      data_opts.pop("feature_dim_axis", None)  # fall back to default
    if dim_tag.is_spatial_dim() and self.time_dim_axis is None:
      data_opts.pop("time_dim_axis", None)  # fall back to default
    if self.placeholder is not None:
      with tf.name_scope("%s_copy_add_dim_by_tag" % get_valid_scope_name_from_str(self.name)):
        placeholder = tf.expand_dims(self.placeholder, axis)
        if dim_tag.dimension is None or dim_tag.dimension > 1:
          tiles = [1] * axis + [dim_tag.get_dim_value()] + [1] * (self.batch_ndim - axis)
          placeholder = tf.tile(placeholder, tiles)
      data_opts["placeholder"] = placeholder
    return Data(**data_opts)

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
    feature_dim_rem = self.dim // new_feature_dim
    new_feature_dim_axis = self.feature_dim_axis + 1
    data_opts = self.get_kwargs(include_special_axes=False)
    dim_tag_split_rem = DimensionTag(
      kind=DimensionTag.Types.Spatial, description="feature_split_rem_%i" % feature_dim_rem,
      dimension=feature_dim_rem)
    dim_tag_new = DimensionTag(
      kind=self.dim_tags[self.feature_dim_axis].kind,
      description="feature_split_new_%i" % new_feature_dim,
      dimension=new_feature_dim)
    dim_tags = (
      self.dim_tags[:self.feature_dim_axis] +
      (dim_tag_split_rem, dim_tag_new) +
      self.dim_tags[self.feature_dim_axis + 1:])
    data_opts["dim_tags"] = dim_tags
    other_special_axes = self.get_special_axes_dict(counted_with_batch_dim=True, only_available=True)
    other_special_axes.pop("feature_dim_axis", None)
    for k, a in other_special_axes.items():
      data_opts[k] = a if (a < new_feature_dim_axis) else (a + 1)
    if self.placeholder is not None:
      self.placeholder.set_shape(self.batch_shape)
      old_shape = get_shape(self.placeholder)
      new_shape = (
        old_shape[:self.feature_dim_axis] +
        [feature_dim_rem, new_feature_dim] +
        old_shape[self.feature_dim_axis + 1:])
      data_opts["placeholder"] = tf.reshape(self.placeholder, new_shape, name="copy_split_feature_dim")
    return Data(**data_opts)

  def copy_extend_batch(self, batch):
    """
    Similar as copy_compatible_to with unbroadcast=True,
    we would possibly extend/expand our batch dim.
    See :class:`BatchInfo`.
    This assumes that we already have a batch dim
    (otherwise see :func:`copy_add_batch_dim`).

    This excludes any beam expansion, which is handled explicitly elsewhere
    (e.g. see :func:`copy_extend_with_beam`).

    :param BatchInfo batch:
    :rtype: Data
    """
    assert self.have_batch_axis()
    assert self.batch, "%s: batch unset" % self
    data = self.copy()
    batch = batch.copy_set_beam(data.beam)
    if data.batch.beam != data.beam:  # Check for some buggy code.
      data.batch = data.batch.copy_set_beam(data.beam)
    if data.batch == batch:
      return data
    data.batch = batch
    self._adapt_batch_consistent_dim_tags()
    if self.placeholder is not None:
      # This can only work if the batch is expanded.
      assert set(self.batch.virtual_dims).issubset(batch.virtual_dims)
      from .basic import get_shape
      from returnn.util.basic import ensure_list_of_type
      with tf.name_scope("copy_extend_batch"):
        axis = self.batch_dim_axis
        x = self.placeholder
        shape = get_shape(x)
        # Only fixed dims supported/implemented (no packed dims).
        old_dims = ensure_list_of_type(self.batch.virtual_dims, BatchInfo.FixedDim)
        new_dims = ensure_list_of_type(batch.virtual_dims, BatchInfo.FixedDim)
        batch_broadcast_shape = []  # type: typing.List[typing.Union[tf.Tensor,int]]  # fill below
        ndim_batch_split = self.batch_ndim - 1 + len(new_dims)
        tiles = [1] * ndim_batch_split  # type: typing.List[typing.Union[tf.Tensor,int]]  # overwrite below
        old_idx = 0
        for new_idx, new_dim in enumerate(new_dims):
          old_dim = old_dims[old_idx] if old_idx < len(old_dims) else None
          if old_dim == new_dim:
            batch_broadcast_shape.append(old_dim.size)
            old_idx += 1
          else:
            batch_broadcast_shape.append(1)
            tiles[axis + new_idx] = new_dim.size
        assert old_idx == len(old_dims)
        shape_batch_split = shape[:axis] + batch_broadcast_shape + shape[axis + 1:]
        x = tf.reshape(x, shape_batch_split)
        x = tf.tile(x, tiles)
        shape = shape[:axis] + [batch.dim] + shape[axis + 1:]
        x = tf.reshape(x, shape)
        data.placeholder = x
    return data

  def copy_compatible_to(self, data, unbroadcast=False, except_feature=False, check_sparse=True, check_dtype=True):
    """
    :param Data data: other data which the returned tensor should be compatible to
      It would add any missing axes with a dim 1 axis for automatic broadcasting.
      It currently does not check whether existing dims match.
    :param bool unbroadcast: if True, all broadcast axes (axes with dim 1) will be tiled such that they match
    :param bool except_feature: if unbroadcast, do not unbroadcast the feature dim
    :param bool check_sparse:
    :param bool check_dtype:
    :returns: Data, might add broadcast dimensions
    :rtype: Data
    """
    assert not check_sparse or self.sparse == data.sparse
    assert not check_dtype or self.dtype == data.dtype
    v = self.copy()
    if v.batch and data.batch and v.batch != data.batch:
      v = v.copy_extend_batch(data.batch)
    v.sparse = data.sparse  # we will later reset it. this is to better count the axes (feature and spatial)
    if not v.sparse:
      # We might need to reset the dim, as it would be invalid otherwise. Reset later.
      if v.feature_dim_axis is not None:
        v.dim = v.batch_shape[v.feature_dim_axis]
      else:
        v.dim = None
    if v.batch_dim_axis is not None and data.batch_dim_axis is None:
      raise ValueError("copy_compatible_to: self %r has batch-dim, but target data %r has not" % (self, data))
    if data.batch_ndim < v.batch_ndim:
      raise ValueError("copy_compatible_to: self %r already has more dims than target data %r" % (self, data))

    is_equal_opts = dict(
      allow_same_feature_dim=True, allow_same_spatial_dim=True, treat_feature_as_spatial=True, ignore_feature_dim=True)
    mapped_axes = data.find_matching_dim_map(v, list(range(v.batch_ndim)), is_equal_opts)  # maps v -> data
    assert len(mapped_axes) == v.batch_ndim

    for target_axis in range(data.batch_ndim):
      new_v_axis = min(target_axis, v.batch_ndim)
      if target_axis not in mapped_axes.values():
        # Dim in data, but not in v
        unbroadcast_axis = unbroadcast and not (except_feature and data.feature_dim_axis == target_axis)
        v = v.copy_add_dim_by_tag(data.get_dim_tag(target_axis), axis=new_v_axis, unbroadcast=unbroadcast_axis)
        # Keep mapped_axes consistent
        mapped_axes = {v_ax + (1 if v_ax >= new_v_axis else 0): trg_ax for v_ax, trg_ax in mapped_axes.items()}
        mapped_axes[new_v_axis] = target_axis
      else:
        # Dim exists in both data and in v. Maybe order is wrong.
        matching_v_axes = [v_ax for v_ax, trg_ax in mapped_axes.items() if trg_ax == target_axis]
        assert len(matching_v_axes) == 1
        matching_v_axis = matching_v_axes[0]
        if target_axis != matching_v_axis:
          # Order was wrong
          v = v.copy_swap_axes(matching_v_axis, new_v_axis)
          # Keep mapped_axes consistent
          mapped_axes[matching_v_axis], mapped_axes[new_v_axis] = mapped_axes[new_v_axis], mapped_axes[matching_v_axis]

    assert v.batch_ndim == data.batch_ndim
    assert all(mapped_axes[ax] == ax for ax in range(v.batch_ndim))

    # Ensure time_dim_axis and feature_dim_axis is same as in data
    assert v.batch_dim_axis == data.batch_dim_axis  # there is only at most one batch_dim_axis
    v.time_dim_axis = data.time_dim_axis
    v.feature_dim_axis = data.feature_dim_axis_or_unspecified
    if v.feature_dim_axis is not None:
      v.dim = v.batch_shape[v.feature_dim_axis]
    else:
      v.dim = None

    # Reset sparse
    if self.sparse:
      v.feature_dim_axis = NotSpecified
      v.sparse = True  # reset
      v.dim = self.dim  # reset

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
    data_opts = self.get_kwargs(include_special_axes=False)
    if self.placeholder is not None:
      data_opts["placeholder"] = self.get_placeholder_time_flattened()
    dim_tag = self.dim_tags[self.time_dim_axis]
    dim_tag = DimensionTag(
      kind=DimensionTag.Types.Spatial, description="%s_flattened" % (dim_tag.description or "unnamed"))
    data_opts["dim_tags"] = (
      (dim_tag,) +
      tuple(tag for (i, tag) in enumerate(self.dim_tags) if i not in (self.batch_dim_axis, self.time_dim_axis)))
    data_opts["time_dim_axis"] = None
    data_opts.pop("feature_dim_axis", None)
    return Data(**data_opts)

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
    data.beam = beam
    assert data.batch
    data.batch = data.batch.copy_set_beam(beam)
    with tf.name_scope("%s_data_extend_with_beam" % get_valid_scope_name_from_str(self.name)):
      if data.placeholder is not None:
        with same_control_flow_ctx(data.placeholder):
          data.placeholder = tile_transposed(data.placeholder, axis=data.batch_dim_axis, multiples=beam.beam_size)
          setattr(data.placeholder, "_RETURNN_beam_expanded_base_data", self)
    data._adapt_batch_consistent_dim_tags()
    return data

  def copy_squeeze_axes(self, axes):
    """
    :param list[int] axes: counted with batch dim
    :return: copy of myself, with squeezed axes
    :rtype: Data
    """
    from .basic import get_valid_scope_name_from_str
    assert isinstance(axes, (list, tuple))
    assert all(self.batch_shape[axis] == 1 for axis in axes)
    assert all(0 <= axis < self.batch_ndim for axis in axes)
    if not axes:
      return self.copy()
    data_opts = self.get_kwargs(include_special_axes=False)
    if self.placeholder is not None:
      data_opts["placeholder"] = tf.squeeze(
        self.placeholder, axes,
        name="%s_squeeze_axes" % get_valid_scope_name_from_str(self.name))
    data_opts["dim_tags"] = [tag for (i, tag) in enumerate(self.dim_tags) if i not in axes]
    if self.time_dim_axis is not None:
      if self.time_dim_axis in axes:
        data_opts.pop("time_dim_axis", None)
      else:
        data_opts["time_dim_axis"] = self.time_dim_axis - len([axis for axis in axes if axis < self.time_dim_axis])
    if not self.sparse:
      if self.feature_dim_axis is not None and self.feature_dim_axis_or_unspecified is not NotSpecified:
        if self.feature_dim_axis in axes:
          data_opts.pop("feature_dim_axis", None)
        else:
          data_opts["feature_dim_axis"] = (
            self.feature_dim_axis - len([axis for axis in axes if axis < self.feature_dim_axis]))
    return Data(**data_opts)

  def copy_template(self, name=None, dtype=None):
    """
    :param str|None name:
    :param str|None dtype:
    :return: copy of myself, using self.get_kwargs(), without placeholder
    :rtype: Data
    """
    kwargs = self.get_kwargs()
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
    kwargs = self.get_kwargs(include_special_axes=False)
    if exclude_axis < 0:
      exclude_axis += self.batch_ndim
      assert exclude_axis >= 0
    assert 0 <= exclude_axis < self.batch_ndim
    if exclude_axis == self.feature_dim_axis:
      kwargs.pop("dim", None)
    other_special_axes = self.get_special_axes_dict(counted_with_batch_dim=True, only_available=True)
    for axis_name, axis in other_special_axes.items():
      if axis == exclude_axis:
        continue
      kwargs[axis_name] = axis if (axis < exclude_axis) else (axis - 1)
    new_dim_tags = list(self.dim_tags)
    del new_dim_tags[exclude_axis]
    kwargs["dim_tags"] = new_dim_tags
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
    return self.copy_template_excluding_axis(exclude_axis=axis_to_exclude, name=name)

  def copy_template_excluding_time_dim(self, name=None):
    """
    :param str|None name: if set, this will be the new name
    :return: copy of myself excluding the time-dimension without placeholder
    :rtype: Data
    """
    assert self.batch_dim_axis is not None
    assert self.time_dim_axis is not None
    return self.copy_template_excluding_axis(exclude_axis=self.time_dim_axis, name=name)

  def copy_template_adding_time_dim(self, name=None, time_dim_axis=0):
    """
    Adds a time-dim-axis.
    If a time-dim-axis already exists, it will anyway create this new one.

    :param str|None name: if set, this will be the new name
    :param int time_dim_axis: the new time-dim-axis index
    :return: copy of myself adding the time-dimension without placeholder
    :rtype: Data
    """
    if time_dim_axis < 0:
      time_dim_axis += self.batch_ndim + 1
      assert time_dim_axis >= 0
    assert 0 <= time_dim_axis <= self.batch_ndim
    kwargs = self.get_kwargs(include_special_axes=False)
    dim_tag = DimensionTag(kind=DimensionTag.Types.Time, description="unknown_time", dimension=None)
    dim_tags = self.dim_tags[:time_dim_axis] + (dim_tag,) + self.dim_tags[time_dim_axis:]
    kwargs["dim_tags"] = dim_tags
    other_special_axes = self.get_special_axes_dict(counted_with_batch_dim=True, only_available=True)
    other_special_axes.pop("time_dim_axis", None)
    for axis_name, axis in other_special_axes.items():
      kwargs[axis_name] = axis if (axis < time_dim_axis) else (axis + 1)
    kwargs["time_dim_axis"] = time_dim_axis
    if name:
      kwargs["name"] = name
    return Data(**kwargs)

  def copy_template_replace_dim_tag(self, axis, new_dim_tag, name=None):
    """
    :param int axis:
    :param DimensionTag new_dim_tag:
    :param str|None name: new name
    :rtype: Data
    """
    if axis < 0:
      assert axis + self.batch_ndim >= 0
      axis += self.batch_ndim
    assert 0 <= axis < self.batch_ndim
    opts = self.get_kwargs()
    dim_tags = self.dim_tags[:axis] + (new_dim_tag,) + self.dim_tags[axis + 1:]
    opts["dim_tags"] = dim_tags
    if self.feature_dim_axis_or_unspecified is not NotSpecified:
      if self.feature_dim_axis == axis and self.dim_tags[axis].is_feature_dim() and not new_dim_tag.is_feature_dim():
        opts["feature_dim_axis"] = None
    if name:
      opts["name"] = name
    return Data(**opts)

  def copy_template_replace_dim(self, axis, new_dim, new_size=None):
    """
    :param int axis:
    :param int|None new_dim:
    :param tf.Tensor|None new_size:
    :rtype: Data
    """
    dim_tag = self.dim_tags[axis]
    if dim_tag.is_batch_dim():
      assert new_dim is None
      return self.copy_template()  # nothing to do
    dim_tag = DimensionTag(
      kind=dim_tag.kind, description="%s_replaced" % (dim_tag.description or "unnamed"),
      dimension=new_dim, dyn_size=new_size)
    return self.copy_template_replace_dim_tag(axis=axis, new_dim_tag=dim_tag)

  def copy_template_new_dim_tags(self, new_dim_tags, name=None, keep_special_axes=False):
    """
    :param list[DimensionTag]|tuple[DimensionTag] new_dim_tags:
    :param str|None name:
    :param bool keep_special_axes:
    :rtype: Data
    """
    if keep_special_axes:
      assert len(new_dim_tags) == self.batch_ndim
    opts = self.get_kwargs(include_special_axes=keep_special_axes)
    opts["dim_tags"] = new_dim_tags
    if name:
      opts["name"] = name
    return Data(**opts)

  def copy_template_set_ctx(self, ctx):
    """
    :param ControlFlowContext ctx:
    :return: new Data instance
    :rtype: Data
    """
    kwargs = self.get_kwargs()
    kwargs["control_flow_ctx"] = ctx
    return Data(**kwargs)

  def _get_variable_dim_pattern(self):
    """
    :return: tuple with bools specifying which dims of the shape (excluding batch-dim) are of variable length.
     e.g. (time,feature), shape=(None,128), this returns (True, False)
    :rtype: tuple[bool]
    """
    return tuple([dim is None for dim in self.shape])

  def _get_var_len_axes(self):
    return [i for (i, d) in enumerate(self._get_variable_dim_pattern()) if d]

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
  def dim_tags(self):
    """
    :rtype: tuple[DimensionTag]
    """
    return self._dim_tags

  @property
  def shape(self):
    """
    :return: shape without batch-dim. e.g. (time,feat) = (None,128)
    :rtype: tuple[int|None]
    """
    return tuple(tag.dimension for tag in self._dim_tags if not tag.is_batch_dim())

  @shape.setter
  def shape(self, shape):
    """
    :param tuple[int|None] shape:
    """
    if tuple(shape) == self.shape:
      return
    raise Exception("%s: setting the shape is not allowed (new shape %s)" % (self, shape))

  @property
  def batch_shape(self):
    """
    :return: shape with added batch-dim. e.g. (batch,time,feat) = (None,None,128)
    :rtype: tuple[int|None]
    """
    return tuple(tag.dimension for tag in self.dim_tags)

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

  def have_varying_shape_in_ctx(self):
    """
    :return: whether the (dynamic) shape can change in this control flow context.
      E.g. when self.control_flow_context is a loop, and we have one dynamic dim
      where dyn_size_ext has the same control_flow_context
      (such that dyn_size_ext has e.g. shape [B,T] outside the loop).
      This can be relevant for accumulating values of self.placeholder
      e.g. via tf.TensorArray.
    :rtype: bool
    """
    return any(tag.control_flow_ctx for tag in self.dim_tags)

  @property
  def size_placeholder(self):
    """
    For compatibility, return a proxy object which behaves like the original dict.

    :rtype: _SizePlaceholderProxy
    """
    return _SizePlaceholderProxy(self)

  @size_placeholder.setter
  def size_placeholder(self, sizes):
    """
    :param dict[int,tf.Tensor]|None sizes:
    """
    if sizes is None:
      return
    for axis_wo_b, size in sizes.items():
      self.set_dynamic_size(axis=self.get_batch_axis(axis_wo_b), sizes=size)

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
  def batch_shape_dense(self):
    """
    :rtype: tuple[int|None]
    """
    if self.sparse:
      return self.batch_shape + (self.dim,)
    return self.batch_shape

  @property
  def dim_tags_sparse(self):
    """
    :return: dim tags without feature dim axis
    :rtype: tuple[DimensionTag]
    """
    if self.sparse or not self.have_feature_axis():
      return self.dim_tags
    return self.dim_tags[:self.feature_dim_axis] + self.dim_tags[self.feature_dim_axis + 1:]

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
    return len(self._dim_tags)

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

  @property
  def batch_dim_axis(self):
    """
    :return: batch dim axis, counted with batch-dim
    :rtype: int|None
    """
    return _batch_dim_axis_from_dim_tags_tuple(self._dim_tags)

  @batch_dim_axis.setter
  def batch_dim_axis(self, axis):
    """
    :param int|None axis:
    """
    if axis == self.batch_dim_axis:
      return
    raise Exception("%s: cannot set batch_dim_axis = %s" % (self, axis))

  def _default_feature_dim_axis(self):
    """
    :return: feature dim axis, counted with batch-dim
    :rtype: int|None
    """
    return _default_feature_dim_axis(
      batch_dim_axis=self.batch_dim_axis, time_dim_axis=self.time_dim_axis,
      batch_shape=self.batch_shape, sparse=self.sparse)

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

  @property
  def placeholder(self):
    """
    :rtype: tf.Tensor|None
    """
    return self._placeholder

  @placeholder.setter
  def placeholder(self, value):
    """
    :param tf.Tensor|None value:
    """
    self._placeholder = value
    self.sanity_check(assume_complete=False)

  @property
  def batch(self):
    """
    :rtype: BatchInfo|None
    """
    return self._batch

  @batch.setter
  def batch(self, batch):
    """
    :param BatchInfo|None batch:
    """
    if batch:
      assert batch.beam == self.beam
    self._batch = batch
    self._adapt_batch_consistent_dim_tags()

  @property
  def beam(self):
    """
    :rtype: SearchBeam|None
    """
    if self._beam:
      return self._beam
    if self._batch:
      return self._batch.beam
    return None

  @beam.setter
  def beam(self, beam):
    """
    :param SearchBeam|None beam:
    """
    # No check for batch.beam, as the batch is usually set only later.
    self._beam = beam
    if self._batch:
      self._batch = self._batch.copy_set_beam(beam=beam)

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

  def get_placeholder_with_runtime_sanity_checks(self):
    """
    :return: identity(self.placeholder) with added checks
    :rtype: tf.Tensor
    """
    with tf.control_dependencies([self.get_runtime_sanity_check_op()]):
      return tf.identity(self.placeholder, name="identity_with_runtime_sanity_checks")

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
    :param int|list[int]|str|list[str|DimensionTag]|DimensionTag|None axes: one axis or multiple axis, or none.
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
    if isinstance(axes, DimensionTag):
      return [i for (i, tag) in enumerate(self.dim_tags) if tag == axes]
    if not allow_int:
      assert not isinstance(axes, int)
    assert isinstance(axes, (str, int, list, tuple))
    if isinstance(axes, (list, tuple)):
      assert all([a is None or isinstance(a, (str, int, DimensionTag)) for a in axes])
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
      elif axes.startswith("stag-single:"):  # spatial tag which possibly matches multiple spatial axes
        # in this case, a name of form "stag-single:<idx>:<name> is expected.
        # idx is relative to the matching stags, i.e., it is the index among the list of spatial dims matching the name
        _, idx_s, name = axes.split(":", 2)  # stag-single:<idx>:<name>
        idx = int(idx_s)
        axes = self.get_axes_by_tag_name(name, spatial_only=True)[idx]
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
        assert isinstance(i, (str, tuple, list, DimensionTag))
        flat_axes += self.get_axes_from_description(i)
    flat_axes = [i % self.batch_ndim for i in flat_axes]
    res = []
    for i in flat_axes:
      if i not in res:
        res.append(i)
    return res

  def get_axis_from_description(self, axis, allow_int=True):
    """
    :param int|str|DimensionTag axis:
    :param bool allow_int:
    :return: axis, counted with batch-dim
    :rtype: int
    """
    axes = self.get_axes_from_description(axis, allow_int=allow_int)
    assert axes, "%s: %r axis not found" % (self, axis)
    assert len(axes) == 1, "%r: %r is not a unique axis but %r" % (self, axis, axes)
    return axes[0]

  def has_axis(self, axis):
    """
    :param str|DimensionTag axis:
    :return: whether the axis exists
    :rtype: bool
    """
    axes = self.get_axes_from_description(axis, allow_int=False)
    return len(axes) > 0

  def get_axes_by_tag_name(self, name, spatial_only=False):
    """
    :param str name: the tag name, or part of it (must be unique, and must exist)
    :param bool spatial_only:
    :rtype: list[int]
    """
    dim_tags = self.get_batch_shape_dim_tags()
    matching_dim_tags = [
      (axis, tag) for axis, tag in enumerate(dim_tags)
      if name.lower() in tag.description.lower()
      or name.lower() in tag.get_same_base().description.lower()]
    if spatial_only:
      spatial_axes = self.get_spatial_batch_axes()
      matching_dim_tags = [
        (axis, tag) for axis, tag in matching_dim_tags if axis in spatial_axes or tag.is_spatial_dim()]
    return [ax for ax, _ in matching_dim_tags]

  def get_axis_by_tag_name(self, name, spatial_only=False):
    """
    :param str name: the tag name, or part of it (must be unique, and must exist)
    :param bool spatial_only:
    :rtype: int
    """
    matching_dim_tags = self.get_axes_by_tag_name(name, spatial_only)
    assert len(matching_dim_tags) == 1, "%r: tag name %r is not unique in dim tags %r" % (
      self, name, self.get_batch_shape_dim_tags())
    return matching_dim_tags[0]

  def get_batch_axis_excluding_batch(self, axis):
    """
    :param int axis: counted with batch-dim
    :return: axis counted without batch-dim
    :rtype: int|None
    """
    return _get_axis_wo_b(axis, batch_dim_axis=self.batch_dim_axis, batch_ndim=self.batch_ndim)

  def get_batch_axis(self, axis):
    """
    :param int axis: counted without batch-dim
    :return: axis counted with batch-dim
    :rtype: int
    """
    return _get_axis_wb(axis, batch_dim_axis=self.batch_dim_axis)

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
    if self.placeholder is None:
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
    if self.placeholder is None:
      # Run at template construction time.
      return self.batch_shape[axis] is None
    axis_wo_batch = self.get_batch_axis_excluding_batch(axis)
    if axis_wo_batch in self.size_placeholder:
      return True  # not quite the same as get_dynamic_axes
    assert isinstance(self.batch_shape[axis], int), (
      "%s: the requested axis has neither a size_placeholder entry nor a fixed size" % self)
    return False

  def has_dynamic_size(self, axis):
    """
    :param int axis: counted with batch-dim axis. :func:`is_axis_dynamic` should be True
    :rtype: bool
    """
    return self.dim_tags[axis].dyn_size is not None

  def get_dynamic_size(self, axis):
    """
    :param int axis: counted with batch-dim axis. :func:`is_axis_dynamic` should be True
    :return: shape (B,)
    :rtype: tf.Tensor
    """
    tag = self.dim_tags[axis]
    assert tag.dyn_size is not None, "%s: axis %i has no dyn size" % (self, axis)
    return tag.dyn_size

  def set_dynamic_size(self, axis, sizes):
    """
    :param int axis: counted with batch-dim
    :param tf.Tensor sizes: shape [B]
    """
    # Note: The following code is somewhat ugly patchwork
    # to fix some other currently incomplete or buggy behavior of some layers
    # which introduce sizes without correctly setting the dim tag.
    # The beam information is also missing currently.
    # We make the ugly assumption that when it is unset,
    # the first usage should hopefully define the correct beam.
    if getattr(sizes, "_RETURNN_dyn_size_beam", NotSpecified) is NotSpecified:
      sizes._RETURNN_dyn_size_beam = self.beam
    if self.beam and getattr(sizes, "_RETURNN_dyn_size_beam", None) != self.beam:
      tag = DimensionTag.get_tag_from_size_tensor(sizes)
      assert tag and self.batch
      tag = tag.get_for_batch_ctx(batch=self.batch, ctx=self.control_flow_ctx)
      assert tag.dyn_size is not None
      sizes = tag.dyn_size

    sizes_tag = DimensionTag.get_tag_from_size_tensor(sizes)
    if sizes_tag:
      assert sizes_tag.is_same_size_tensor(sizes)
    tag = self.dim_tags[axis]
    assert tag.dimension is None  # dynamic axis
    if tag.is_same_size_tensor(sizes):
      return  # nothing to do
    if tag.dyn_size is None:
      if sizes_tag:  # special rule for older code: overtake previous existing
        assert sizes_tag.is_same_size_tensor(sizes)
        self._dim_tags = self.dim_tags[:axis] + (sizes_tag,) + self.dim_tags[axis + 1:]
        # Also assume the existing dim tag should be expected as equal.
        # Likely there is anyway no reference so this does not matter.
        tag.declare_same_as(sizes_tag)
      else:
        # Assign now. This should also set the dim tag on sizes.
        new_tag = tag.set_tag_on_size_tensor(sizes, batch=self.batch)
        if new_tag is not tag:
          self._dim_tags = self.dim_tags[:axis] + (new_tag,) + self.dim_tags[axis + 1:]
    else:
      # Reset to some new size.
      # Use new dim tag, or previous existing attached to size.
      assert sizes_tag, "%s: assign dyn sizes %s without defined dim tag" % (self, sizes)
      self._dim_tags = self.dim_tags[:axis] + (sizes_tag,) + self.dim_tags[axis + 1:]

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

  def mark_same_time(self, tags):
    """
    If the given dimension tag matches any of our axes, we set our time axis to the selected one.

    :param set[DimensionTag] tags:
    :return: whether we have found the same
    :rtype: bool
    """
    for axis, dim_tag in enumerate(self.dim_tags):
      if dim_tag in tags:
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
    if axis < 0:
      assert axis + self.batch_ndim > 0
      axis += self.batch_ndim
    assert 0 <= axis < self.batch_ndim
    assert axis != self.batch_dim_axis
    tag = self.dim_tags[axis]
    assert tag.dyn_size_ext
    with tf.name_scope("get_sequence_mask_broadcast"):
      if tag.dyn_size_ext.have_batch_axis() and tag.dyn_size_ext.batch_ndim == 1:  # just [B]
        # This is the common case where the size is of shape [B].
        # We make use of sequence_mask or sequence_mask_time_major in that case,
        # which is optimized by caching.
        size = tag.dyn_size
        if axis >= self.batch_dim_axis:
          seq_mask = sequence_mask(size)  # (B,T)
        else:  # axis < batch_dim_axis
          seq_mask = sequence_mask_time_major(size)  # (T,B)
        shape = [1] * self.batch_ndim  # type: typing.List[typing.Union[int,tf.Tensor]]
        placeholder_shape = tf.shape(self.placeholder)
        shape[self.batch_dim_axis] = placeholder_shape[self.batch_dim_axis]
        shape[axis] = placeholder_shape[axis]
        seq_mask = tf.reshape(seq_mask, shape, name="seq_mask_reshape")
        assert seq_mask.get_shape().ndims == self.batch_ndim
      else:  # size is something unusual
        max_idx = tf.reduce_max(tag.dyn_size)
        # We use the assumption that self.placeholder.shape[axis] == max_idx.
        idx_range = tf.range(max_idx)
        idx_range = tf.reshape(idx_range, [1] * (axis - 1) + [max_idx] + [1] * (self.batch_ndim - axis - 1))
        assert tag.dyn_size_ext
        assert set(tag.dyn_size_ext.dim_tags).issubset(self.dim_tags)
        size_ext = tag.dyn_size_ext.copy_compatible_to(self, check_sparse=False, check_dtype=False)
        seq_mask = tf.less(idx_range, size_ext.placeholder)
        assert seq_mask.get_shape().ndims == self.batch_ndim
    return seq_mask

  def copy_masked(self, mask_value):
    """
    :param float|int mask_value:
    :rtype: Data
    """
    assert self.placeholder is not None
    from .basic import mask_dyn_seq_len_nd
    dyn_axes = [axis for axis, dim in enumerate(self.dim_tags) if not dim.is_batch_dim() and dim.dimension is None]
    res = self.copy()
    res.placeholder = mask_dyn_seq_len_nd(self, pad_value=mask_value, axes=dyn_axes)
    return res

  def get_batch_dim(self):
    """
    :rtype: tf.Tensor|int
    """
    assert self.batch_dim_axis is not None
    if self.batch:
      if self.beam:
        assert self.batch.beam == self.beam
      return self.batch.dim
    # Note: We need this fallback code for now
    # until we consistently have set self.batch correctly in all cases.
    from returnn.tf.layers.base import LayerBase
    batch = LayerBase.get_recent_layer().get_batch_info()
    batch = batch.copy_set_beam(self.beam)
    return batch.dim

  def get_batch_dim_tag(self):
    """
    :rtype: DimensionTag
    """
    assert self.have_batch_axis()
    return self.dim_tags[self.batch_dim_axis]

  def get_static_batch_dim(self):
    """
    :rtype: int|None
    """
    # Do not fallback to get_batch_dim or get_recent_layer or so. This should be safe.
    if not self.batch:
      return None
    return self.batch.static_dim

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

  # Exclude "batch_dim_axis" now because that is always inferred from dim tags.
  SpecialAxesNames = ("time_dim_axis", "feature_dim_axis")

  def get_special_axes_dict(self, counted_with_batch_dim=True, only_available=False):
    """
    :param bool counted_with_batch_dim:
    :param bool only_available:
    :return: dict axis-name -> axis
    :rtype: dict[str,int]
    """
    axes = list(self.SpecialAxesNames)
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
    return self._dim_tags[axis]

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
    return self.dim_tags

  @classmethod
  def get_common_data(cls, sources, ignore_feature_dim=False):
    """
    :param list[Data] sources:
    :param bool ignore_feature_dim: when set, the feature dim does not have to match in the sources
    :return: some generic data where the sources should be compatible to (with copy_compatible_to),
      i.e. it contains the union of all axes from all sources (least common multiple).
      This is always a template, and a new copy.
    :rtype: Data|None
    """
    if not sources:
      return None
    assert sources
    if len(sources) == 1:
      return sources[0].copy_template()
    max_ndim = max([s.batch_ndim for s in sources])
    common_batch = BatchInfo.get_common_batch_info([src.batch for src in sources if src.batch])
    # Try with the (first) largest.
    common = [s for s in sources if s.batch_ndim == max_ndim][0]
    common = common.copy_template()
    common.beam = None  # this will be reset
    if common_batch:
      common.batch = common_batch.copy_set_beam(None)  # the beam will be reset
    if any([s.beam for s in sources]):
      # Note: we don't use copy_extend_with_beam because we don't want to create any ops in the TF graph at this point.
      common.beam = SearchBeam.get_combined_beam(*[s.beam for s in sources])
    is_equal_opts = dict(
      ignore_feature_dim=ignore_feature_dim, treat_feature_as_spatial=True,
      allow_same_spatial_dim=True, broadcast_matches=True,
      undefined_matches=True, derived_matches=True)
    all_dim_tags, tags_dict = DimensionTag.get_all_dimension_tags(sources, is_equal_opts=is_equal_opts)
    # Check for potential undefined tags, and replace those with defined tags if possible.
    for axis, dim_tag in enumerate(common.dim_tags):
      if dim_tag.undefined:
        other = DimensionTag.get_existing_tag_from_collection(dim_tag, all_dim_tags, is_equal_opts=is_equal_opts)
        if other and not other.undefined:
          # We found another dim tag which matches and which is defined.
          # Replace it.
          common = common.copy_template_replace_dim_tag(axis=axis, new_dim_tag=other)
    # Check for missing tags, and add those.
    for dim_tag in all_dim_tags:
      if not DimensionTag.get_existing_tag_from_collection(dim_tag, common.dim_tags, is_equal_opts=is_equal_opts):
        axis = common.get_default_new_axis_for_dim_tag(dim_tag)
        common = common.copy_add_dim_by_tag(dim_tag, unbroadcast=True, axis=axis)
    return common

  def find_matching_dims(self, dim_tag, is_equal_opts):
    """
    Finds the dimensions of this Data that match another DimensionTag

    :param DimensionTag dim_tag:
    :param dict[str,bool]|None is_equal_opts: passed to DimensionTag.is_equal
    :rtype: list[int] a list of matching axes, counted with batch dim. Sorted in ascending order
    """
    return [axis for axis in range(self.batch_ndim) if self.get_dim_tag(axis).is_equal(dim_tag, **is_equal_opts)]

  def find_matching_dim_map(self, other, other_axes, is_equal_opts=None):
    """
    Looks up all other_axes of another Data in this Data. Does not allow duplicates.

    :param Data other:
    :param list[int] other_axes: a list of axes of ``other``, counted with batch dim
    :return: a dict mapping other axes to own axes, all counted with batch dim
    :param dict[str,bool]|None is_equal_opts: passed to DimensionTag.is_equal
    :rtype: dict[int,int]
    """
    if is_equal_opts is None:
      is_equal_opts = dict(
        allow_same_feature_dim=True, allow_same_spatial_dim=True, treat_feature_as_spatial=True)

    def map_other_axis_to_self(other_axis, taken_self_axes):
      """
      :param int other_axis: counted with batch dim
      :param set[int] taken_self_axes: axes that should not be used again
      :return: the axis of ``self`` that matches ``other_axis``, counted with batch dim
      :rtype: int
      """
      other_axis_dim_tag = other.get_dim_tag(other_axis)
      matching = [
        self_axis for self_axis in self.find_matching_dims(other_axis_dim_tag, is_equal_opts)
        if self_axis not in taken_self_axes]
      if not matching:
        # Try harder by allowing broadcasting to match
        is_equal_opts["broadcast_matches"] = True
        matching = [
          self_axis for self_axis in self.find_matching_dims(other_axis_dim_tag, is_equal_opts)
          if self_axis not in taken_self_axes]
      if not matching:
        # If still not, then also allow one single dyn_size to be unknown
        is_equal_opts["unknown_spatial_matches"] = True
        matching = [
          self_axis for self_axis in self.find_matching_dims(other_axis_dim_tag, is_equal_opts)
          if self_axis not in taken_self_axes]
        assert len(matching) == 1, 'cannot match the axes %s from %s to %s. Failing to match axis %s' % (
          other_axes, other, self, other_axis)
      assert matching, 'cannot match the axes %s from %s to %s. Failing at axis %s' % (
        other_axes, other, self, other_axis)
      # If there are multiple matches (e.g. because two axes have the same feature dim), leave their order intact.
      # We do this by always choosing the first unused match which is the smallest axes
      return matching[0]

    other_to_self_mapping = {}
    for axis in other_axes:
      other_to_self_mapping[axis] = map_other_axis_to_self(axis, set(other_to_self_mapping.values()))
    assert len(other_to_self_mapping) == len(other_axes), 'other_axes may not contain duplicates'
    return other_to_self_mapping


class _SizePlaceholderProxy:
  """
  This is a proxy object to emulate the original Data.size_placeholder behavior,
  which was a dict[int,tf.Tensor], axis_wo_batch -> sizes.
  """

  def __init__(self, data):
    """
    :param Data data:
    """
    self.data = data

  def _assert_sane_axis_wo_batch(self, idx):
    assert isinstance(idx, int) and 0 <= idx < self.data.ndim

  def __contains__(self, item):
    if not isinstance(item, int):
      return False
    if not 0 <= item < self.data.ndim:
      return False
    return self.data.has_dynamic_size(axis=self.data.get_batch_axis(item))

  def __getitem__(self, item):
    self._assert_sane_axis_wo_batch(item)
    return self.data.get_dynamic_size(axis=self.data.get_batch_axis(item))

  def __setitem__(self, key, value):
    self._assert_sane_axis_wo_batch(key)
    self.data.set_dynamic_size(axis=self.data.get_batch_axis(key), sizes=value)

  def __delitem__(self, key):
    self._assert_sane_axis_wo_batch(key)
    raise Exception("%s: cannot delete items from size_placeholder" % self.data)

  def __iter__(self):
    return iter(self.keys())

  def __len__(self):
    return len(self.keys())

  def __bool__(self):
    return bool(self.keys())

  __nonzero__ = __bool__  # Python 3 wants __bool__, Python 2.7 wants __nonzero__

  def __repr__(self):
    return repr(self.as_dict())

  def get(self, axis_wo_b, default=None):
    """
    :param int axis_wo_b:
    :param tf.Tensor|None default:
    :rtype: tf.Tensor|None
    """
    if axis_wo_b in self:
      return self[axis_wo_b]
    return default

  def pop(self, axis_wo_b, *default):
    """
    :param int axis_wo_b:
    """
    if default and axis_wo_b not in self:
      default, = default
      return default
    res = self[axis_wo_b]
    del self[axis_wo_b]
    return res

  def clear(self):
    """
    Remove all.
    """
    raise Exception("%s: cannot clear size_placeholder" % self.data)

  def keys(self):
    """
    :rtype: list[int]
    """
    return [i for i in range(self.data.ndim) if i in self]

  def values(self):
    """
    :rtype: list[tf.Tensor]
    """
    return [self[i] for i in self.keys()]

  def items(self):
    """
    :rtype: list[(int,tf.Tensor)]
    """
    return [(i, self[i]) for i in self.keys()]

  def copy(self):
    """
    :return: a copy-like object
    :rtype: dict[int,tf.Tensor]
    """
    return self.as_dict()

  def as_dict(self):
    """
    :rtype: dict[int,tf.Tensor]
    """
    return dict(self.items())


def _batch_dim_axis_from_dim_tags_tuple(dim_tags):
  """
  :param tuple[DimensionTag] dim_tags:
  :return: batch_dim_axis. int or None if not existing
  :rtype: int|None
  """
  for axis, dim_tag in enumerate(dim_tags):
    if dim_tag.is_batch_dim():
      return axis
  return None


def _batch_shape_from_shape(shape, batch_dim_axis):
  """
  :param tuple[int|None]|list[int|None] shape: without batch-dim
  :param int|None batch_dim_axis:
  :return: shape with batch dim if existing
  :rtype: tuple[int|None]
  """
  shape = tuple(shape)
  if batch_dim_axis is not None:
    assert 0 <= batch_dim_axis <= len(shape)
    return shape[:batch_dim_axis] + (None,) + shape[batch_dim_axis:]
  else:
    return shape


def _create_size_placeholder(name, axis_wo_b, tag):
  """
  :param str name:
  :param int axis_wo_b:
  :param DimensionTag tag:
  """
  from .basic import reuse_name_scope
  with reuse_name_scope("extern_data/placeholders/%s" % name, absolute=True):
    dyn_size = tf_compat.v1.placeholder(
      name="%s_dim%i_size" % (name, axis_wo_b), dtype=Data.size_dtype, shape=(None,))
    tag.set_tag_on_size_tensor(dyn_size)


def _infer_dim_tags_tuple_from_shape(
  shape,
  batch_dim_axis, time_dim_axis, feature_dim_axis,
  sparse,
  size_placeholder,
  dim_tags,
  name,
  auto_create_placeholders
):
  """
  :param tuple[int|None]|list[int|None] shape: this is without batch-dim-axis
  :param int|None batch_dim_axis:
  :param int|None time_dim_axis:
  :param int|None|NotSpecified feature_dim_axis:
  :param bool sparse:
  :param dict[int,tf.Tensor]|None size_placeholder: key is axis without batch-dim
  :param dict[int,DimensionTag]|None dim_tags: some existing explicitly specified dim tags. key is axis with batch-dim
  :param bool auto_create_placeholders:
  :param str name:
  :return: dim tags tuple
  :rtype: tuple[DimensionTag]
  """
  assert isinstance(shape, (tuple, list))
  shape = tuple(shape)
  batch_shape = _batch_shape_from_shape(shape, batch_dim_axis=batch_dim_axis)
  if feature_dim_axis is NotSpecified:
    feature_dim_axis = _default_feature_dim_axis(
      batch_dim_axis=batch_dim_axis, time_dim_axis=time_dim_axis, batch_shape=batch_shape, sparse=sparse)
  elif feature_dim_axis is not None:
    if feature_dim_axis < 0:
      feature_dim_axis += len(batch_shape)
    assert 0 <= feature_dim_axis < len(batch_shape)
  dim_tags = dim_tags.copy() if dim_tags else {}
  if batch_dim_axis is not None and batch_dim_axis not in dim_tags:
    dim_tags[batch_dim_axis] = DimensionTag(kind=DimensionTag.Types.Batch, description="batch:%s" % name)
  # Note: Consistent to Data.get_dim_tag,
  # prefer interpretation as spatial axis if there is a dynamic size or this is marked as time axis.
  if size_placeholder:
    for axis_wo_b, size in size_placeholder.items():
      axis = _get_axis_wb(axis_wo_b, batch_dim_axis=batch_dim_axis)
      if axis in dim_tags:
        continue
      tag = DimensionTag.get_tag_from_size_tensor(size)
      if tag:
        dim_tags[axis] = tag
  # See Data.get_spatial_batch_axes
  spatial_axes = [
    axis
    for axis in range(len(batch_shape))
    if axis != batch_dim_axis
    and (axis != feature_dim_axis or
         axis == time_dim_axis or
         batch_shape[axis] is None)]
  for axis in range(len(batch_shape)):
    tag = dim_tags.get(axis)
    axis_wo_b = _get_axis_wo_b(axis, batch_dim_axis=batch_dim_axis)
    dyn_size = size_placeholder.get(axis_wo_b) if (size_placeholder and axis_wo_b is not None) else None
    dim = batch_shape[axis]
    if auto_create_placeholders and dim is None and dyn_size is None and axis != batch_dim_axis:
      if not tag:
        if axis == time_dim_axis:
          tag_name = "time"
        else:
          tag_name = "spatial%i" % axis
        tag = DimensionTag(
          description="%s:var:extern_data:%s" % (tag_name, name),
          # Spatial dim tag, even if axis == feature_dim_axis. This is to keep the old behavior.
          # This is such that DimensionTag.is_equal behaves as before, e.g. in Data.get_common_data.
          kind=DimensionTag.Types.Spatial)
        dim_tags[axis] = tag
      _create_size_placeholder(name=name, axis_wo_b=axis_wo_b, tag=tag)
      dyn_size = tag.dyn_size
    if tag:
      # Just some sanity checks.
      assert isinstance(tag, DimensionTag)
      assert tag.dimension == dim
      assert tag.is_same_size_tensor(dyn_size)
      continue
    if axis == feature_dim_axis and dyn_size is None and axis != time_dim_axis:
      tag = DimensionTag(
        kind=DimensionTag.Types.Feature, dimension=dim, description="feature:%s" % name,
        undefined=dim is None)
    else:
      assert axis in spatial_axes
      description = "time" if axis == time_dim_axis else "spatial%i" % spatial_axes.index(axis)
      if dyn_size is not None:
        # Note: This case is uncommon/unexpected (we should have a dim-tag on the dyn_size above), so be verbose,
        # and fix such cases if possible (i.e. for all newly created dynamic size tensors, set the dim-tag).
        description += ":var:%r" % dyn_size.name
      elif dim is None:
        description += ":var-unk"
      else:
        description += ":static%i" % dim
      description += ":%s" % name
      tag = DimensionTag(
        kind=DimensionTag.Types.Spatial, description=description, dimension=dim, dyn_size=dyn_size,
        undefined=dim is None and dyn_size is None)
    dim_tags[axis] = tag
  assert sorted(dim_tags.keys()) == list(range(len(batch_shape)))
  return tuple(dim_tags[axis] for axis in range(len(batch_shape)))


def _auto_create_size_placeholders_on_dim_tags(name, dim_tags):
  """
  :param str name:
  :param tuple[DimensionTag] dim_tags:
  """
  batch_dim_axis = _batch_dim_axis_from_dim_tags_tuple(dim_tags)
  for axis, tag in enumerate(dim_tags):
    if tag.is_batch_dim():
      continue
    if tag.dimension is not None:
      continue
    if tag.dyn_size is not None:
      continue
    axis_wo_b = _get_axis_wo_b(axis, batch_dim_axis=batch_dim_axis)
    _create_size_placeholder(name=name, axis_wo_b=axis_wo_b, tag=tag)


def _get_axis_wo_b(axis_wb, batch_dim_axis, batch_ndim=None):
  """
  :param int axis_wb: counted with batch-dim
  :param int|None batch_dim_axis:
  :param int|None batch_ndim: only used for axis_wb < 0. might be unknown (None)
  :return: axis counted without batch-dim
  :rtype: int|None
  """
  if axis_wb < 0:
    assert batch_ndim is not None
    assert axis_wb + batch_ndim >= 0
    axis_wb += batch_ndim
    # Do this check only in this case;
    # we call this function early in construction where batch_ndim might be invalid.
    assert 0 <= axis_wb < batch_ndim
  if batch_dim_axis is None:
    return axis_wb
  if axis_wb == batch_dim_axis:
    return None
  if axis_wb < batch_dim_axis:
    return axis_wb
  return axis_wb - 1


def _get_axis_wb(axis_wo_b, batch_dim_axis):
  """
  :param int axis_wo_b: counted without batch-dim
  :param int|None batch_dim_axis:
  :return: axis counted with batch-dim
  :rtype: int
  """
  if batch_dim_axis is None:
    return axis_wo_b
  if axis_wo_b >= batch_dim_axis:
    return axis_wo_b + 1
  return axis_wo_b


def _infer_default_shape_and_time(batch_dim_axis, time_dim_axis, feature_dim_axis, sparse, dim):
  """
  This is the logic to infer some sensible/default shape when it is not specified.
  As this is somewhat adhoc, this is not recommended to be used anymore.

  :param int|None batch_dim_axis:
  :param int|None time_dim_axis:
  :param int|None|NotSpecified feature_dim_axis:
  :param bool sparse:
  :param int|None dim:
  :return: shape (without batch dim), time_dim_axis
  :rtype: (tuple[int|None],int|None)
  """
  if time_dim_axis is not None:
    assert time_dim_axis != batch_dim_axis
    shape = (None,) * (_get_axis_wo_b(time_dim_axis, batch_dim_axis=batch_dim_axis) + 1)
  else:  # no time-dim-axis
    shape = ()
  if not sparse and feature_dim_axis is not None:
    assert dim is not NotSpecified, "no shape specified, not sparse, feature_dim_axis existing -> need dim"
    if feature_dim_axis is NotSpecified or feature_dim_axis == -1:
      shape = shape + (dim,)
    else:
      assert 0 <= feature_dim_axis != batch_dim_axis
      feature_dim_axis_wo_batch = _get_axis_wo_b(feature_dim_axis, batch_dim_axis=batch_dim_axis)
      if feature_dim_axis_wo_batch < len(shape):
        shape = shape[:-feature_dim_axis_wo_batch] + (dim,) + shape[feature_dim_axis_wo_batch + 1:]
      else:
        shape = shape + (None,) * (feature_dim_axis_wo_batch - len(shape)) + (dim,)
        assert len(shape) == feature_dim_axis_wo_batch + 1
  return shape, time_dim_axis


def _default_time_dim_axis(batch_dim_axis, shape):
  """
  :param int|None batch_dim_axis:
  :param tuple[int|None]|list[int|None] shape: without batch-dim
  :return: time dim axis, counted with batch-dim
  :rtype: int|None
  """
  if batch_dim_axis is None:
    time_dim_axis = None
  else:
    # Do not select the batch dim axis, or any axis with None dim.
    # Note that we currently allow to select the same as the feature dim axis,
    # in case the feature dim is None.
    taken_axes = {batch_dim_axis}
    batch_shape = _batch_shape_from_shape(shape, batch_dim_axis=batch_dim_axis)
    for axis, _dim in enumerate(batch_shape):
      if _dim is not None:
        taken_axes.add(axis)
    available_axes = [i for i in range(len(batch_shape)) if i not in taken_axes]
    if available_axes:
      time_dim_axis = available_axes[0]
    else:
      time_dim_axis = None
  return time_dim_axis


def _default_time_dim_axis_no_shape(batch_dim_axis, feature_dim_axis):
  """
  :param int|None batch_dim_axis:
  :param int|None|NotSpecified feature_dim_axis:
  :return: time dim axis, counted with batch-dim
  :rtype: int|None
  """
  if batch_dim_axis is None:
    time_dim_axis = None
  else:
    # By default if not specified, we have a time dim.
    taken_axes = {batch_dim_axis}
    if isinstance(feature_dim_axis, int):
      taken_axes.add(feature_dim_axis)
    time_dim_axis = [i for i in range(max(taken_axes) + 2) if i not in taken_axes][0]
  return time_dim_axis


def _default_time_dim_axis_dim_tags(dim_tags):
  """
  :param list[DimensionTag]|tuple[DimensionTag] dim_tags:
  :return: time dim axis, counted with batch-dim
  :rtype: int|None
  """
  # Consistent to _default_time_dim_axis.
  # Any spatial dynamic axis.
  # Or otherwise any dynamic axis (including maybe feature).
  # Not using any static axes.
  dim_tags_dyn_spatial = [i for i, tag in enumerate(dim_tags) if tag.is_spatial_dim() and tag.dimension is None]
  if dim_tags_dyn_spatial:
    return dim_tags_dyn_spatial[0]
  dim_tags_dyn = [i for i, tag in enumerate(dim_tags) if not tag.is_batch_dim() and tag.dimension is None]
  if dim_tags_dyn:
    return dim_tags_dyn[0]
  return None


def _default_feature_dim_axis(batch_dim_axis, time_dim_axis, batch_shape, sparse):
  """
  :param int|None batch_dim_axis:
  :param int|None time_dim_axis:
  :param tuple[int|None] batch_shape:
  :param bool sparse:
  :return: feature dim axis, counted with batch-dim
  :rtype: int|None
  """
  if sparse:
    return None
  batch_ndim = len(batch_shape)
  ndim = batch_ndim if batch_dim_axis is None else (batch_ndim - 1)
  if ndim == 0:
    return None
  axes = [i for i in range(batch_ndim) if i not in [batch_dim_axis, time_dim_axis]]
  if not axes:
    # Allow same as time-dim-axis...
    axes = [i for i in range(batch_ndim) if i != batch_dim_axis]
  assert axes
  static_axes = [i for i in axes if batch_shape[i] is not None]
  # Prefer last static, if available.
  if static_axes:
    return static_axes[-1]
  return axes[-1]


class ControlFlowContext:
  """
  This is a simple wrapper around the TF ControlFlowContext, i.e. tf.while_loop or tf.cond.

  We have this wrapper to refer to a context which might not exist yet (e.g. at template construction time).
  Also, we might want to store additional information, such the spatial dim tag of the loop.
  """

  class Types:
    """
    Possible types of context.
    """
    Loop = "loop"
    Cond = "cond"

  def __init__(self, kind, outer_ctx=None):
    """
    :param str kind: from ControlFlowContext.Types
    :param ControlFlowContext outer_ctx:
    """
    self.kind = kind
    self._outer_ctx = outer_ctx
    from tensorflow.python.ops.control_flow_ops import ControlFlowContext as TFControlFlowCtx
    self._tf_control_flow_ctx = None  # type: typing.Optional[TFControlFlowCtx]
    self._loop_spatial_dim = None  # type: typing.Optional[DimensionTag]

  def __repr__(self):
    return "ControlFlowContext{%s}" % self.repr_inner()

  def repr_inner(self):
    """
    :rtype: str
    """
    return "/".join(ctx._repr_single() for ctx in self._abs_ctx_stack())

  def _repr_single(self):
    """
    :rtype: str
    """
    s = self.kind
    if self.is_loop() and self.loop_spatial_dim:
      s += "(%s)" % self.loop_spatial_dim.short_repr()
    return s

  def _abs_ctx_stack(self):
    """
    :rtype: list[ControlFlowContext]
    :return: chain of ctx, last is self
    """
    chain = []
    ctx = self
    while ctx:
      chain.append(ctx)
      ctx = ctx.outer_ctx
    chain.reverse()
    return chain

  @classmethod
  def abs_ctx_stack(cls, ctx):
    """
    :param ControlFlowContext|None ctx:
    :rtype: list[ControlFlowContext]
    """
    if ctx:
      return ctx._abs_ctx_stack()
    return []

  @classmethod
  def abs_ctx_stack_with_root(cls, ctx):
    """
    :param ControlFlowContext|None ctx:
    :rtype: list[ControlFlowContext|None]
    :return: chain of ctx, last is self, first is None
    """
    ls = [None]  # type: typing.List[typing.Optional[ControlFlowContext]]
    if ctx:
      ls += ctx._abs_ctx_stack()
    return ls

  @classmethod
  def is_parent_or_same(cls, parent, child):
    """
    :param ControlFlowContext|None parent:
    :param ControlFlowContext|None child:
    :rtype: bool
    """
    if parent == child:
      return True
    if not parent:
      return True  # parent is root
    if not child:
      return False  # child is root but parent is not
    while child:
      if child == parent:
        return True
      child = child.outer_ctx
    return False

  @classmethod
  def collect_parent_dims(cls, ctx):
    """
    :param ControlFlowContext|None ctx:
    :rtype: list[DimensionTag]
    """
    dims = []
    for ctx_ in ControlFlowContext.abs_ctx_stack(ctx):
      if ctx_.is_loop() and ctx_.loop_spatial_dim:
        dims.append(ctx_.loop_spatial_dim)
    return dims

  def is_loop(self):
    """
    :rtype: bool
    """
    return self.kind == self.Types.Loop

  def is_cond(self):
    """
    :rtype: bool
    """
    return self.kind == self.Types.Cond

  @property
  def outer_ctx(self):
    """
    :rtype: ControlFlowContext|None
    """
    return self._outer_ctx

  @property
  def tf_control_flow_ctx(self):
    """
    :rtype: tensorflow.python.ops.control_flow_ops.ControlFlowContext|None
    """
    return self._tf_control_flow_ctx

  @tf_control_flow_ctx.setter
  def tf_control_flow_ctx(self, ctx):
    """
    :param tensorflow.python.ops.control_flow_ops.ControlFlowContext ctx:
    """
    if self.is_loop():
      assert ctx.IsWhileContext()
    if self.is_cond():
      assert ctx.IsCondContext()
    self._tf_control_flow_ctx = ctx

  @property
  def loop_spatial_dim(self):
    """
    :rtype: DimensionTag|None
    """
    assert self.is_loop()
    return self._loop_spatial_dim

  @loop_spatial_dim.setter
  def loop_spatial_dim(self, dim):
    """
    :param DimensionTag dim:
    """
    assert self.is_loop()
    self._loop_spatial_dim = dim
