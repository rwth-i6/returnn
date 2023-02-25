from __future__ import annotations

import tensorflow as tf
import returnn.tf.compat as tf_compat
from returnn.tf.layers.basic import LayerBase, _ConcatInputLayer, get_concat_sources_data_template


# ---------- Utilities ----------


def batch_sizes_after_windowing(sizes, window):
    """
    :param tf.Tensor sizes: (batch_sizes)
    :param int window: size of the applied window
    :return: sizes for each batch after applying a window on each batch
    :rtype: tf.Tensor
    """

    def fold_times(acc, x):
        r1 = tf.tile([window], [tf.maximum(x - window + 1, 0)])
        r2 = tf.range(tf.minimum(x, window - 1), 0, -1)
        return tf.concat([acc, r1, r2], 0)

    return tf.foldl(
        fold_times,
        sizes,
        tf_compat.v1.placeholder_with_default(tf.zeros([0], dtype="int32"), [None]),
        name="fold_sizes",
    )


def batch_indices_after_windowing(sizes, window):
    """
    here we compute the start and end times for each of the new batches when applying a window
    :param tf.Tensor sizes: (batch_sizes)
    :param int window: size of the applied window
    :return: tensor of shape (?, 3), contains batch index, start-frame and end-frame for each batch after applying a window
    :rtype: tf.Tensor
    """

    def fold_batches(acc, x):
        b = x[0]
        l = x[1]
        batch = tf.tile([b], [l])
        start = tf.range(l)
        end = tf.minimum(tf.range(window, l + window), l)
        return tf.concat([acc, tf.transpose(tf.stack([batch, start, end]))], axis=0)

    return tf.foldl(
        fold_batches,
        tf.stack([tf.range(tf.shape(sizes)[0]), sizes], axis=1),
        tf_compat.v1.placeholder_with_default(tf.zeros([0, 3], dtype="int32"), [None, 3]),
        name="fold_batches",
    )


# ---------- Layers ----------


class SegmentInputLayer(_ConcatInputLayer):
    """
    This layer takes the input data, applies a window and outputs each window as a new batch, this is more
    efficient than a window as a new dimension if sequences have varying lengths
    """

    layer_class = "segment_input"

    def __init__(self, window=15, **kwargs):
        super(SegmentInputLayer, self).__init__(**kwargs)
        sizes = self.input_data.size_placeholder[0]
        new_sizes = batch_sizes_after_windowing(sizes, window)

        def fold_data(acc, x):
            batch_idx = x[0]
            num_frames = x[1]
            res = tf.expand_dims(tf.range(num_frames), -1)  # start times
            res = tf.tile(res, [1, window])  # fill add time dimension
            res += tf.range(window)  # add offsets
            res = tf.where(res >= num_frames, tf.zeros_like(res), res)  # filter frames that go past the end
            if self.input_data.is_batch_major:
                res = tf.stack([tf.ones_like(res) * batch_idx, res], axis=2)  # add batch_index in first dim
            else:
                res = tf.stack([res, tf.ones_like(res) * batch_idx], axis=2)  # add batch_index in second dim
            return tf.concat([acc, res], 0)

        initial = tf_compat.v1.placeholder_with_default((tf.zeros([0, window, 2], dtype=tf.int32)), [None, window, 2])
        indices = tf.foldl(
            fold_data, tf.stack([tf.range(tf.shape(sizes)[0]), sizes], axis=1), initial, name="fold_data"
        )

        if self.input_data.is_time_major:
            indices = tf.transpose(indices, [1, 0, 2])

        self.output.placeholder = tf.gather_nd(self.input_data.placeholder, indices)
        self.output.size_placeholder[0] = new_sizes

    @classmethod
    def get_out_data_from_opts(cls, name, sources, window, **kwargs):
        out = get_concat_sources_data_template(sources, name="%s_output" % name)
        out.size_placeholder = {}
        out.size_placeholder[0] = None
        return out


class ClassesToSegmentsLayer(_ConcatInputLayer):
    """
    This layer takes a sequence of classes (=> sparse input) and applies a window (same as SegmentInput) to it.
    For each position t in the window it computes the relative frequencies of the classes up to and including
    that position t.
    """

    layer_class = "classes_to_segments"

    def __init__(self, num_classes, window=15, **kwargs):
        super(ClassesToSegmentsLayer, self).__init__(**kwargs)
        assert self.input_data.sparse

        sizes = self.input_data.size_placeholder[0]
        new_sizes = batch_sizes_after_windowing(sizes, window)
        batches = batch_indices_after_windowing(sizes, window)

        onehot = tf.one_hot(self.input_data.get_placeholder_as_batch_major(), num_classes)

        def compute(x):
            batch = x[0]
            start = x[1]
            end = x[2]

            padded_onehot = tf.concat(
                [onehot[batch][start:end], tf.zeros([tf.maximum(window - (end - start), 0), num_classes])], axis=0
            )
            classes = tf.cumsum(padded_onehot)
            normalization = tf.cast(tf.expand_dims(tf.range(1, window + 1), -1), classes.dtype)
            return classes / normalization

        self.output.placeholder = tf.map_fn(compute, batches, dtype="float32")
        self.output.size_placeholder[0] = new_sizes

    @classmethod
    def get_out_data_from_opts(cls, name, sources, num_classes, window, **kwargs):
        out = get_concat_sources_data_template(sources, name="%s_output" % name).copy_as_batch_major()
        out.size_placeholder = {}
        out.size_placeholder[0] = None
        out.sparse = False
        out.shape += (num_classes,)
        out.dtype = "float32"
        out.dim = num_classes
        return out


class ClassesToLengthDistributionLayer(_ConcatInputLayer):
    layer_class = "classes_to_length_distribution"

    def __init__(self, window=15, scale=1.0, **kwargs):
        super(ClassesToLengthDistributionLayer, self).__init__(**kwargs)
        assert self.input_data.sparse

        sizes = self.input_data.size_placeholder[0]
        batches = batch_indices_after_windowing(sizes, window)
        new_sizes = tf.fill((tf.shape(batches)[0],), 1)
        classes = self.input_data.get_placeholder_as_batch_major()

        def compute(bse):
            batch = bse[0]
            start = bse[1]
            end = bse[2]

            batch_cls = classes[batch][start:end]
            cls_not_eq = tf.not_equal(batch_cls[:-1], batch_cls[1:])
            cls_changed = tf.concat([cls_not_eq, [True]], axis=0)
            idx = tf.where(cls_changed)
            count = tf.squeeze(tf.concat([[idx[0] + 1], idx[1:] - idx[:-1]], axis=0), axis=1)
            freq = tf.cast(count, dtype="float32")

            res = tf.scatter_nd(idx, tf.cast(count, dtype="float32") / tf.cast(end - start, dtype="float32"), (window,))

            return res

        self.output.placeholder = tf.expand_dims(tf.map_fn(compute, batches, back_prop=False, dtype="float32"), axis=-2)
        self.output.size_placeholder[0] = new_sizes

    @classmethod
    def get_out_data_from_opts(cls, name, sources, window, **kwargs):
        out = get_concat_sources_data_template(sources, name="%s_output" % name).copy_as_batch_major()
        out.size_placeholder = {}
        out.size_placeholder[0] = None
        out.shape = (1, window)
        out.dim = window
        out.sparse = False
        out.dtype = "float32"
        return out


class ClassesToLengthDistributionGlobalLayer(_ConcatInputLayer):
    layer_class = "classes_to_length_distribution_global"

    def __init__(
        self, window=15, weight_falloff=1.0, target_smoothing=None, min_length=1, broadcast_axis="time", **kwargs
    ):
        super(ClassesToLengthDistributionGlobalLayer, self).__init__(**kwargs)
        assert self.input_data.sparse
        assert broadcast_axis in ["time", "feature"]

        sizes = self.input_data.size_placeholder[0]
        batches = batch_indices_after_windowing(sizes, window)
        new_sizes = tf.fill((tf.shape(batches)[0],), 1)
        classes = self.input_data.get_placeholder_as_batch_major()
        cls_not_eq = tf.not_equal(classes[:, :-1], classes[:, 1:])
        cls_changed = tf.concat([cls_not_eq, tf.fill((tf.shape(classes)[0], 1), True)], axis=1)

        lengths = tf.range(0.0, float(window), 1.0)
        end_distribution = tf.pow((1.0 - weight_falloff), lengths) * weight_falloff

        # add small weight at the last frame in case there is no ending label, then we want the last frame to be a label end
        # because the windows have different lengths the last frame might not be at index (window -1), but earlier, thus we
        # also add some zeros at the end
        no_label_backup = tf.convert_to_tensor([0.0] * (window - 1) + [1e-4] + [0.0] * (window - 1))

        # As the label ends are not well-defined in most cases, we allow the model a little bit of wiggle-room in the
        # decision where to put the label ends. This is achieved by smoothing the targets for one segment with a kernel
        # that is specified by target_smoothing.
        smoothing = None
        if target_smoothing is not None:
            import numpy as np

            assert len(target_smoothing) % 2 == 1

            smoothing = tf.TensorArray(
                dtype="float32", size=window, clear_after_read=False, infer_shape=False, name="smoothing_matrices"
            )

            s = sum(target_smoothing)
            target_smoothing = [v / s for v in target_smoothing]
            center = len(target_smoothing) // 2

            mat = np.zeros((window, window), dtype="float32")
            for i, v in enumerate(target_smoothing):
                mat += np.diag([v] * (window - abs(i - center)), i - center)

            for i in range(window):
                submat = np.diag([1.0] * window).astype("float32")
                submat[: i + 1, : i + 1] = mat[: i + 1, : i + 1]
                submat /= submat.sum(axis=1).reshape((-1, 1))
                smoothing = smoothing.write(i, submat)

        min_length_filter = tf.concat(
            [tf.zeros((min_length - 1,), dtype="float32"), tf.ones((window - min_length + 1), dtype="float32")], axis=0
        )

        def compute(bse):
            batch = bse[0]
            start = bse[1]
            end = bse[2]
            size = end - start

            cls_chg = cls_changed[batch][start:end]
            idx = tf.where(cls_chg)
            res = tf.scatter_nd(idx, end_distribution[: tf.shape(idx)[0]], (window,))
            if min_length > 1:
                res *= min_length_filter
            res += no_label_backup[window - size : 2 * window - size]
            res = res / tf.reduce_sum(res)
            if smoothing is not None:
                res = tf.tensordot(res, smoothing.read(size - 1), [[0], [0]])

            return res

        targets = tf.map_fn(compute, batches, back_prop=False, dtype="float32")
        if broadcast_axis == "time":
            self.output.placeholder = tf.expand_dims(targets, axis=-2)
        elif broadcast_axis == "feature":
            self.output.placeholder = tf.expand_dims(targets, axis=-1)
        self.output.size_placeholder[0] = new_sizes

    @classmethod
    def get_out_data_from_opts(cls, name, sources, window, broadcast_axis="time", **kwargs):
        out = get_concat_sources_data_template(sources, name="%s_output" % name).copy_as_batch_major()
        out.size_placeholder = {}
        out.size_placeholder[0] = None
        out.shape = (1, window) if broadcast_axis == "time" else (window, 1)
        out.dim = window if broadcast_axis == "time" else 1
        out.sparse = False
        out.dtype = "float32"
        return out


class SegmentAlignmentLayer(_ConcatInputLayer):
    layer_class = "segment_alignment"

    def __init__(self, num_classes, window=15, **kwargs):
        super(SegmentAlignmentLayer, self).__init__(**kwargs)
        assert self.input_data.sparse

        sizes = tf_compat.v1.div(self.input_data.size_placeholder[0], 2)
        batches = batch_indices_after_windowing(sizes, window)
        new_sizes = tf.fill((tf.shape(batches)[0],), window)

        input = self.input_data.get_placeholder_as_batch_major()
        input = tf.reshape(input, (tf.shape(input)[0], -1, 2))  # reinterpret last dimension as (dim, 2)
        end_distribution = tf.convert_to_tensor([1.0] + [0.0] * (window - 1))
        onehot = tf.one_hot(input[:, :, 0], num_classes)
        onehot = tf.pad(onehot, tf.constant([[0, 0], [0, window - 1], [0, 0]]), "CONSTANT")

        # add small weight at the last frame in case there is no ending label, then we want the last frame to be a label end
        # because the windows have different lengths the last frame might not be at index (window -1), but earlier, thus we
        # also add some zeros at the end
        no_label_backup = tf.constant([0.0] * (window - 1) + [1e-4] + [0.0] * (window - 1))

        def compute(bse):
            batch = bse[0]
            start = bse[1]
            end = bse[2]
            size = end - start

            seg_ended = input[batch, start:end, 1]
            idx = tf.where(tf.not_equal(seg_ended, 0))
            length_dist = tf.scatter_nd(idx, end_distribution[: tf.shape(idx)[0]], (window,))
            length_dist += no_label_backup[window - size : 2 * window - size]
            length_dist = length_dist / tf.reduce_sum(length_dist)
            length_dist = tf.expand_dims(length_dist, -1)

            result = onehot[batch, start : start + window, :] * length_dist
            return result

        targets = tf.map_fn(compute, batches, back_prop=False, dtype="float32")
        self.output.placeholder = targets
        self.output.size_placeholder[0] = new_sizes

    @classmethod
    def get_out_data_from_opts(cls, name, sources, num_classes, window, **kwargs):
        out = get_concat_sources_data_template(sources, name="%s_output" % name).copy_as_batch_major()
        out.size_placeholder = {}
        out.size_placeholder[0] = None
        out.shape = (window, num_classes)
        out.dim = num_classes
        out.sparse = False
        out.dtype = "float32"
        return out


class UnsegmentInputLayer(_ConcatInputLayer):
    """
    Takes the output of SegmentInput (sequences windowed over time and folded into batch-dim)
    and restores the original batch dimension. The feature dimension contains window * original_features
    many entries. The entries at time t all correspond to windows ending at time t. The window
    that started in the same frame comes first, then the window that started in the frame before and so on.
    This is also the format used for the segmental decoder in RASR.
    """

    layer_class = "unsegment_input"

    def __init__(self, **kwargs):
        super(UnsegmentInputLayer, self).__init__(**kwargs)
        sizes = self.input_data.size_placeholder[0]
        end_times = tf.squeeze(
            tf.where(tf.equal(sizes, 1)) + 1, axis=1
        )  # a batch of size one indicates that a sequence ended there
        start_times = tf.concat([[0], end_times[:-1]], axis=0)
        new_sizes = end_times - start_times
        max_size = tf.reduce_max(new_sizes)

        # first we shift the data in the time dimension (to get all windows that end at the same time into one batch)
        data = self.input_data.get_placeholder_as_time_major()

        def map_data(x):
            time = x[0]
            batches = x[1]
            size = tf.shape(batches)[0]
            out = tf.concat([batches[size - time :, :], batches[: size - time, :]], axis=0)
            return out

        data = tf.map_fn(map_data, [tf.range(tf.shape(data)[0]), data], dtype=self.input_data.dtype)

        # now we take the start and end times that we extracted above and feed them into extract_batch, which will get
        # the data for one of the original batches
        def extract_batch(x):
            start_batch = x[0]
            end_batch = x[1]
            # the next three lines are a convoluted way of writing data[:,start:end,:], but this notation did not work
            start = [tf.constant(0, dtype="int64"), start_batch, tf.constant(0, dtype="int64")]
            end = [tf.constant(-1, dtype="int64"), end_batch - start_batch, tf.constant(-1, dtype="int64")]
            d = tf.slice(data, start, end)
            d = tf.transpose(d, perm=[1, 0, 2])
            d = tf.reshape(d, [tf.shape(d)[0], -1])
            s = tf.convert_to_tensor([tf.cast(max_size - (end_batch - start_batch), dtype="int32"), tf.shape(d)[1]])
            d = tf.concat([d, tf.zeros(s, dtype="float32")], axis=0)
            return d

        data = tf.map_fn(extract_batch, [start_times, end_times], dtype=data.dtype)

        self.output.size_placeholder = {}
        self.output.size_placeholder[0] = new_sizes
        self.output.batch_dim_axis = 0
        self.output.time_dim_axis = 1
        self.output.placeholder = data

    @classmethod
    def get_out_data_from_opts(cls, name, sources, **kwargs):
        out = get_concat_sources_data_template(sources, name="%s_output" % name).copy_as_batch_major()
        out.size_placeholder[0] = None
        return out


class FillUnusedMemoryLayer(_ConcatInputLayer):
    """
    Fills all unused entries in the time/batch/feature tensor with a constant
    """

    layer_class = "fill_unused"

    def __init__(self, fill_value=0.0, **kwargs):
        super(FillUnusedMemoryLayer, self).__init__(**kwargs)

        mask = self.input_data.get_sequence_mask()
        mask = tf.expand_dims(mask, dim=-1)
        mask = tf.tile(mask, [1, 1, tf.shape(self.input_data.placeholder)[2]])

        x = self.input_data.placeholder
        x = tf.where(mask, x, tf.fill(tf.shape(x), float(fill_value)))
        self.output.placeholder = x
        self.output.size_placeholder = self.input_data.size_placeholder.copy()

    @classmethod
    def get_out_data_from_opts(cls, name, sources=(), **kwargs):
        return get_concat_sources_data_template(sources, name="%s_output" % name)


class SwapTimeFeatureLayer(_ConcatInputLayer):
    layer_class = "swap_time_feature"

    def __init__(self, **kwargs):
        super(SwapTimeFeatureLayer, self).__init__(**kwargs)
        assert self.input_data.batch_ndim == 3
        assert not self.input_data.sparse
        perm = [self.input_data.batch_dim_axis, self.input_data.feature_dim_axis, self.input_data.time_dim_axis]
        self.output.placeholder = tf.transpose(self.output.placeholder, perm=perm)
        shape = tf.shape(self.output.placeholder)
        self.output.size_placeholder[0] = tf.fill((shape[0],), value=shape[1])
        self.output.dim = self.output.placeholder.get_shape()[-1].value
        self.output.shape = (self.output.placeholder.get_shape()[1].value, self.output.dim)

    @classmethod
    def get_out_data_from_opts(cls, name, sources=(), **kwargs):
        out = get_concat_sources_data_template(sources, name="%s_output" % name)
        out.batch_dim_axis = 0
        out.time_dim_axis = 1
        out.dim = None
        out.size_placeholder = {}
        return out


class FlattenTimeLayer(_ConcatInputLayer):
    layer_class = "flatten_time"

    def __init__(self, **kwargs):
        super(FlattenTimeLayer, self).__init__(**kwargs)
        out = self.input_data.get_placeholder_as_batch_major()
        out = tf.reshape(out, [tf.shape(out)[0], 1, -1])  # (B, 1, T*D)
        self.output.placeholder = out
        self.output.size_placeholder[0] = tf.ones_like(self.input_data.get_sequence_lengths())

    @classmethod
    def get_out_data_from_opts(cls, name, sources, **kwargs):
        out = get_concat_sources_data_template(sources, name="%s_output" % name).copy_as_batch_major()
        out.dim = None
        out.shape = (1, None)
        out.size_placeholder = {}
        return out


class ApplyLengthDistributionLayer(LayerBase):
    layer_class = "apply_length_distribution"

    def __init__(self, length_model_scale=1.0, **kwargs):
        super(ApplyLengthDistributionLayer, self).__init__(**kwargs)
        self.output = self.sources[0].output.copy()
        len_dist_layer = self.sources[1]
        perm = []
        if self.output.is_batch_major:
            perm.append(len_dist_layer.output.batch_dim_axis)
            perm.append(len_dist_layer.output.feature_dim_axis)
        else:
            perm.append(len_dist_layer.output.feature_dim_axis)
            perm.append(len_dist_layer.output.batch_dim_axis)
        perm.append(len_dist_layer.output.time_dim_axis)
        len_mod = tf.transpose(len_dist_layer.output.placeholder, perm=perm)
        if length_model_scale != 1.0:
            len_mod = tf.pow(len_mod, length_model_scale)
        self.output.placeholder *= len_mod

    @classmethod
    def get_out_data_from_opts(cls, name, sources, **kwargs):
        return sources[0].output.copy()


class NormalizeLengthScoresLayer(LayerBase):
    layer_class = "normalize_length_scores"

    def __init__(self, **kwargs):
        super(NormalizeLengthScoresLayer, self).__init__(**kwargs)
        time_axis = 0 if self.sources[0].output.is_time_major else 1
        batch_axis = 1 if self.sources[0].output.is_time_major else 0

        self.output = self.sources[0].output.copy()
        p = self.sources[0].output.placeholder
        win_size = tf.cast(tf.shape(p)[time_axis], dtype=tf.float32)
        s = tf_compat.v1.log(p) * tf.expand_dims(
            tf.expand_dims(tf.range(win_size, dtype=tf.float32) + 1.0, -1), batch_axis
        )
        self.output.placeholder = tf.exp(s)

    @classmethod
    def get_out_data_from_opts(cls, name, sources, **kwargs):
        return sources[0].output.copy()
