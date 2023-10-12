"""
make layer
"""

from __future__ import annotations
from typing import Optional, Union, Dict, Type
from types import FunctionType
import numpy
import tensorflow as tf
from tensorflow.python.util import nest
from returnn.tensor import Tensor, Dim, batch_dim
from returnn.tf.util.data import BatchInfo
from .. import frontend_layers as rfl
from . import dims as _dims


__all__ = ["make_layer", "_get_sub_layer", "register_extern_data"]


def make_layer(
    layer_dict: rfl.LayerDictRaw,
    *,
    name: Optional[Union[str, rfl.Layer]] = None,
    out: Optional[Tensor] = None,
    predefined_out_data: Optional[Tensor] = None,
    name_ctx_ignore_top_stack_frames: int = 0,
) -> Tensor[rfl.Layer]:
    """
    Creates the layer. This also registers the layer instance in the top name ctx.
    When no name is given, this assumes that the top name ctx corresponds to this module.

    If a layer has params, and you want the param sharing logic,
    you should instead derive a new class from :class:`Module`.
    Usually, you do not need either of these,
    as all standard layers should already be wrapped,
    and it should be possible to define any possible logic
    using that.
    (If this is not the case, please report an issue.)

    :param layer_dict: can contain :class:`Tensor` instances
    :param name:
      if str: (suggested) layer name. if given, will create a new :class:`NameCtx`
      if NameCtx, will use this.
    :param out:
    :param predefined_out_data: normally we can derive the out data automatically.
      If this should be skipped, you can pass this explicitly.
    :param name_ctx_ignore_top_stack_frames: for :func:`Layer.current_ctx`.
      If your calling function creates exactly one single layer, you might want to ignore its stack frame
      and set ignore_top_stack_frames=1 and also set a name for the layer.
      If you are potentially creating multiple layers in your calling function,
      leave the default ignore_top_stack_frames=0.
      Some postprocessing step might anyway simplify obsolete subnetworks,
      see :mod:`naming`.
    """
    if isinstance(name, str) or not name:
        parent_ctx = rfl.Layer.current_ctx(ignore_top_stack_frames=name_ctx_ignore_top_stack_frames + 1)
        if not name:
            name = layer_dict["class"]
        layer = rfl.Layer(suggested_name=name, parent=parent_ctx)
        created_name_ctx = True
    elif isinstance(name, rfl.Layer):
        layer = name
        created_name_ctx = False
    else:
        raise TypeError(f"name must be str or Layer, not {type(name)}; or you should pass a module")
    assert not layer.tensor and not layer.layer_dict  # not yet assigned
    layer_dict = layer_dict.copy()

    # Just some sanity checks.
    for value in nest.flatten(layer_dict):
        if isinstance(value, (Tensor, Dim, bool, int, float, str, numpy.ndarray, type(None), FunctionType, rfl.Net)):
            pass
        else:
            raise TypeError(f"{layer}: unexpected type {type(value)} in layer_dict: {layer_dict}")

    try:

        if out is not None:
            assert isinstance(out, Tensor)
        elif predefined_out_data is not None:
            assert isinstance(predefined_out_data, Tensor)
            out = predefined_out_data.copy_template()
        else:
            out = _tensor_from_layer_dict(layer_dict, layer=layer)

        # Do not assign name_ctx.tensor yet because we potentially could raise exceptions later.
        assert layer.tensor is None
        assert layer.layer_dict is None
        assert not layer.usages

        assert layer_dict is not None

        out.control_flow_ctx = rfl.Layer.inner_control_flow()
        if out.have_batch_axis() and not out.batch:
            # You could say this is a bug of RETURNN. Or at least RETURNN is just incomplete here.
            # RETURNN usually would fix that later when the layer is actually created,
            # but we don't do that here.
            # We can still try to look at dependencies and use those batch info.
            batches = []
            for dep in layer.get_tensor_dependencies(_extra_layer_dict=layer_dict):
                if dep.tensor is not None and dep.tensor.batch and dep.tensor.batch not in batches:
                    batches.append(dep.tensor.batch)
            if batches:
                out.batch = BatchInfo.get_common_batch_info(batches)
            elif layer.root.global_batch:
                out.batch = layer.root.global_batch

        for value in nest.flatten(layer_dict):
            if isinstance(value, Tensor) and value.raw_tensor is not None:
                value: Tensor[rfl.Layer]
                assert isinstance(value.raw_tensor, rfl.Layer)
                value.raw_tensor.usages.append(layer)

        layer.layer_dict = layer_dict
        layer.tensor = out
        out.raw_tensor = layer

    except Exception:
        # Just forward the exception.
        # However, if we already created a new name_ctx for it, we can clean this up now.
        if created_name_ctx:
            assert layer.parent
            layer.parent.children.pop(layer.name)
        raise

    for tag in out.dim_tags:
        # noinspection PyProtectedMember
        _dims._register_dim_deps_when_novel(tag, [out])
    # Debug out. Similar as RETURNN template log. Maybe put this behind a flag? Anyway, useful for now.
    print(out)
    return out


def _get_sub_layer(layer: Tensor[rfl.Layer], name: str, *, data: Tensor) -> Tensor:
    """
    Like the "{layer}/{name}" syntax in RETURNN.
    Normally this should only be needed for internal usage.
    """
    out = layer.raw_tensor.get_child_tensor(name, data=data)
    if rfl.is_debug_eager_mode_enabled():
        assert layer.raw_tensor.debug_layer
        import returnn.tf.layers.base

        assert isinstance(layer.raw_tensor.debug_layer, returnn.tf.layers.base.LayerBase)
        sub_layer = layer.raw_tensor.debug_layer.get_sub_layer(name)
        assert sub_layer and sub_layer.output.dim_tags == out.data.dim_tags
        out.raw_tensor.debug_layer = sub_layer
        out.data = sub_layer.output
    return out


def _tensor_from_layer_dict(layer_dict: rfl.LayerDictRaw, *, layer: rfl.Layer) -> Tensor[rfl.Layer]:
    """
    Use RETURNN layer_class.get_out_data_from_opts to get the :class:`Data`.
    For this function, we need to set up some dummy network and dummy source layers.
    """
    from returnn.tf.network import TFNetwork, ExternData
    from returnn.tf.layers.base import InternalLayer, LayerBase
    from returnn.config import get_global_config

    config = get_global_config()
    loop = rfl.Layer.inner_loop()  # Note: for control_flow_ctx, we should also check Cond
    net = TFNetwork(
        config=config,
        extern_data=ExternData(),
        name="dummy_net",
        train_flag=True,  # should not have an effect usually for templates, except maybe in debug-eager-mode
        inside_rec_time_dim=loop.loop_spatial_dim if loop else None,
        control_flow_ctx=rfl.Layer.inner_control_flow(),
    )
    net.extern_data.set_batch_info(_init_global_batch(), init_batch_info=False)

    ref_to_layer_name = {}  # type: Dict[rfl.Layer, str]

    def _get_unique_name(name) -> str:
        name = name.replace("/", "_")
        name = LayerBase.cls_get_tf_scope_name(name)
        reserved_names = set(net.layers.keys()) | {"data"}
        if name not in reserved_names:
            return name
        i = 0
        while True:
            name_ = f"{name}_{i}"
            if name_ not in reserved_names:
                return name_
            i += 1

    def _get_layer_name(ref: Tensor) -> str:
        if ref.raw_tensor in ref_to_layer_name:
            return ref_to_layer_name[ref.raw_tensor]
        name = _get_unique_name(ref.raw_tensor.name)
        ref_to_layer_name[ref.raw_tensor] = name
        assert name not in net.layers
        data = ref.copy_template()
        net.layers[name] = InternalLayer(name=name, network=net, output=data)
        return name

    def _map_layer_dict_elem(value):
        if isinstance(value, Tensor):
            return _get_layer_name(value)
        return value

    layer_dict = nest.map_structure(_map_layer_dict_elem, layer_dict)
    out_name = _get_unique_name(layer.name)
    net_dict = {
        out_name: layer_dict,
        # Simple workaround in case the layer wants to access its previous layer.
        # https://github.com/rwth-i6/returnn_common/issues/243
        f"prev:{out_name}": {"class": "constant", "shape": ()},
    }

    if rfl.is_debug_eager_mode_enabled():
        _add_layer = None  # implies to really construct the layer
    else:
        # Creates only a template layer.
        def _add_layer(name: str, layer_class: Type[LayerBase], **layer_desc) -> LayerBase:
            # noinspection PyProtectedMember
            layer_desc = net._create_layer_layer_desc(name=name, layer_desc=layer_desc, template=True)
            try:
                out_data = layer_class.get_out_data_from_opts(**layer_desc)
                out_data = layer_class.fixup_out_data(out_data, **layer_desc)
            except Exception as exc:
                msgs = ["The RETURNN call\n", f"  {layer_class.__name__}.get_out_data_from_opts(\n"]
                for key, v in layer_desc.items():
                    msgs.append(f"    {key}={v!r},\n")
                msgs += [
                    "  )\n",
                    "raised the exception:\n",
                    f"  {type(exc).__name__} {exc!s}\n",
                    "(See above for the RETURNN exception traceback.)",
                ]
                # Use `with_traceback`, such that the user directly sees the full traceback,
                # and also that debuggers stop right where it matters.
                # Still use `from exc` to keep the original exception,
                # which might additionally look nicer in the output.
                raise ReturnnConstructTemplateException("".join(msgs)).with_traceback(exc.__traceback__) from exc
            layer_ = InternalLayer(name=name, network=net, output=out_data)
            net.layers[name] = layer_
            return layer_

    # Use construct_layer to automatically handle more complex logic such as subnetworks.
    net_layer = net.construct_layer(net_dict=net_dict, name=out_name, add_layer=_add_layer)

    if rfl.is_debug_eager_mode_enabled():
        layer.debug_layer = net_layer

    return net_layer.output.copy_template()


class ReturnnConstructTemplateException(Exception):
    """
    In :func:`_data_from_layer_dict`, when we call layer_class.get_out_data_from_opts,
    we potentially can get errors, often due to user mistakes.
    We wrap those errors in this exception for better reporting.
    """


def _init_global_batch() -> BatchInfo:
    root_name_ctx = rfl.Layer.top().root
    if root_name_ctx.global_batch:
        return root_name_ctx.global_batch
    if rfl.is_debug_eager_mode_enabled():
        root_name_ctx.global_batch = BatchInfo.make_global_batch_info(
            tf.constant(3, name="global_batch")
        )  # https://xkcd.com/221/, but prime
    else:
        # We need some global batch info, and this needs a tensor (e.g. placeholder),
        # but we don't have any tensor yet, nor do we want to create any tensors at this point.
        # So we pass the dummy value -1.
        # Such dummy global batch info with -1 will be handled specially in RETURNN init_batch_info,
        # and it will be replaced with the real global batch.
        root_name_ctx.global_batch = BatchInfo.make_global_batch_info(-1)
    return root_name_ctx.global_batch


def _get_raw_layer_by_name(name: str, *, scope: rfl.Layer, data: Tensor):
    """
    Special layer can be "data:..." or whatever.
    """
    scope.get_child_with_tensor(name, data=data)


def register_extern_data(data: Tensor[rfl.Layer]):
    """
    Register extern data from root ctx.
    As a side effect, it registers the given data as extern data,
    and this will be included when creating the RETURNN config,
    via :func:`NameCtx.get_returnn_config`.
    """
    assert isinstance(data, Tensor)  # the usage was different before. make sure we get this correct
    if data.raw_tensor is not None:
        assert isinstance(data.raw_tensor, rfl.Layer)
    # Assigning data.batch might get variants of the dim tags,
    # but keep some refs to the original dim tags instances here,
    # and also assign them below.
    # This is a workaround the complexity of dim tags internals
    # (https://github.com/rwth-i6/returnn/issues/975)
    # which is actually not used by the RETURNN frontend,
    # so we ignore this mostly, and thus use this workaround.
    # It's only needed because we assign data.batch,
    # which is needed for some of the layers to operate properly.
    orig_dim_tags = data.dim_tags
    if data.raw_tensor is None:
        data.batch = _init_global_batch()
        root_scope = rfl.Layer.top().root  # must exist
        _get_raw_layer_by_name(f"data:{data.name}", scope=root_scope, data=data)
    for i, (tag, orig_tag) in enumerate(zip(data.dim_tags, orig_dim_tags)):
        if not tag.is_batch_dim() and tag.is_dynamic() and not tag.dyn_size_ext:
            # Undefined dynamic dim tag. Set default data template.
            orig_tag.dyn_size_ext = tag.dyn_size_ext = Tensor(
                name=f"{tag.name or (data.name + f'[{i}]')}_default_dyn_size_ext",
                dims=[batch_dim],
                dtype=data.size_dtype,
                batch=data.batch,
            )
        if tag.is_batch_dim() and not tag.dyn_size_ext and tag.dimension is None:
            # Undefined batch dim tag. Set default data template.
            batch_dim.dyn_size_ext = orig_tag.dyn_size_ext = tag.dyn_size_ext = Tensor(
                name=f"batch_dim_default_dyn_size_ext",
                dims=[],
                dtype=data.size_dtype,
                batch=data.batch,
            )
        # noinspection PyProtectedMember
        _dims._register_dim_deps_when_novel(tag, [data])
    if rfl.is_debug_eager_mode_enabled() and not data.raw_tensor.debug_layer:
        from returnn.tf.layers.basic import InternalLayer

        data_tf = data.copy_template()
        data_tf.raw_tensor = _make_random_tf_tensor_for_returnn_data(data)
        data.raw_tensor.debug_layer = rfl.make_layer(
            {"class": InternalLayer, "output": data_tf, "debug_type_name": f"data:{data.name}"}
        )


def _make_random_tf_tensor_for_returnn_data(data: Tensor) -> tf.Tensor:
    shape = []
    for dim in data.dim_tags:
        if dim.is_batch_dim():
            assert data.batch
            shape.append(data.batch.dim)
        elif dim.dimension is not None:
            shape.append(dim.dimension)
        else:
            dim.complete_dyn_size()
            if dim.dyn_size_ext is None:
                assert data.batch
                dim.dyn_size_ext = Tensor(
                    name=f"{data.name}_dummy_dyn_size_ext",
                    dim_tags=[batch_dim],
                    dtype=data.size_dtype,
                    batch=data.batch,
                )
            if dim.dyn_size_ext.placeholder is None:
                dim.dyn_size_ext.placeholder = _make_random_tf_tensor_for_returnn_data(dim.dyn_size_ext)
            shape.append(tf.reduce_max(dim.dyn_size_ext.placeholder))
    dtype = tf.as_dtype(data.dtype)
    if dtype.is_integer:
        if data.sparse:
            return tf.random.uniform(shape=shape, dtype=dtype, minval=0, maxval=data.dim)
        else:
            import binascii

            c = abs(binascii.crc32(data.name.encode("utf8"))) % 21 + 3
            shape = tf.convert_to_tensor(shape)
            c_tf = tf.constant(c, name="dummy_random_const", dtype=dtype)
            rnd = tf.broadcast_to(c_tf, shape)
            rnd_diff = tf.random.uniform(shape=shape, minval=0, maxval=2**31 - 1, dtype=dtype)
            rnd_diff = rnd_diff % tf.reshape(tf.minimum(tf.range(0, tf.size(rnd), dtype=dtype) + 1, c_tf - 2), shape)
            rnd = tf.clip_by_value(rnd - rnd_diff, 1, c_tf)
            return rnd
    assert dtype.is_floating  # not implemented otherwise
    return tf.random.normal(shape=shape, dtype=dtype)
