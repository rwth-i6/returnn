
from __future__ import print_function

import tensorflow as tf
from TFNetworkLayer import LayerBase, _ConcatInputLayer


class AlternatingRealToComplexLayer(_ConcatInputLayer):
  """
  This layer converts a real valued input tensor into a complex valued output
  tensor.
  For this even and odd features are considered the real and imaginary part of
  one complex number, respectively
  """

  layer_class = "alternating_real_to_complex"

  def __init__(self, **kwargs):
    """
    """
    super(AlternatingRealToComplexLayer, self).__init__(**kwargs)

    # set order of axes
    self.output.batch_dim_axis = 0
    self.output.time_dim_axis = 1
    self.output.feature_dim_axis = 2
    # - ensure correct ordering of input placeholder
    input_placeholder = self.input_data.placeholder
    if ((self.input_data.batch_dim_axis != self.output.batch_dim_axis) or (self.input_data.time_dim_axis != self.output.time_dim_axis) or (self.input_data.feature_dim_axis != self.output.feature_dim_axis)):
      input_placeholder = tf.transpose(input_placeholder, [self.input_data.batch_dim_axis, self.input_data.time_dim_axis, self.input_data.feature_dim_axis])

    real_value = tf.strided_slice(input_placeholder, [0, 0, 0], tf.shape(input_placeholder), [1, 1, 2])
    imag_value = tf.strided_slice(input_placeholder, [0, 0, 1], tf.shape(input_placeholder), [1, 1, 2])
    self.output.placeholder = tf.complex(real_value, imag_value)


class BatchMedianPoolingLayer(_ConcatInputLayer):
  """
  This layer is used to pool together batches by taking their medium value.
  Thus the batch size is divided by pool_size. The stride is hard coded to be
  equal to the pool size
  """

  layer_class = "batch_median_pooling"

  def __init__(self, pool_size=1, **kwargs):
    """
    :param pool_size int: size of the pool to take median of (is also used as stride size)
    """
    super(BatchMedianPoolingLayer, self).__init__(**kwargs)

    # set order of axes
    self.output.batch_dim_axis = 0
    self.output.time_dim_axis = 1
    self.output.feature_dim_axis = 2
    # - ensure correct ordering of input placeholder
    input_placeholder = self.input_data.placeholder
    if ((self.input_data.batch_dim_axis != self.output.batch_dim_axis) or (self.input_data.time_dim_axis != self.output.time_dim_axis) or (self.input_data.feature_dim_axis != self.output.feature_dim_axis)):
      input_placeholder = tf.transpose(input_placeholder, [self.input_data.batch_dim_axis, self.input_data.time_dim_axis, self.input_data.feature_dim_axis])

    # loop over batch pools and extract median
    pool_start_idx = tf.constant(0)
    output = tf.Variable(tf.zeros([1, 1, 1], dtype=tf.float32), dtype=tf.float32, trainable=False)

    def iteratePools(pool_start_idx, output):
        return tf.less(pool_start_idx, tf.shape(input_placeholder)[self.output.batch_dim_axis])

    def poolMedian(pool_start_idx, output):
      pool = input_placeholder[pool_start_idx:(pool_start_idx + pool_size), :, :]
      median = tf.transpose(tf.reshape(tf.nn.top_k(tf.transpose(pool, [self.output.time_dim_axis, self.output.feature_dim_axis, self.output.batch_dim_axis]), tf.cast(tf.ceil(tf.cast(tf.shape(pool)[self.output.batch_dim_axis], dtype=tf.float32) / 2), dtype=tf.int32)).values[:, :, -1], (tf.shape(pool)[1], tf.shape(pool)[2], 1)), [2, 0, 1])
      output = tf.cond(tf.greater(pool_start_idx, 0), lambda: tf.concat([output, median], axis=self.output.batch_dim_axis), lambda: median)
      return tf.add(pool_start_idx, pool_size), output

    r = tf.while_loop(iteratePools, poolMedian, [pool_start_idx, output], shape_invariants=[pool_start_idx.get_shape(), tf.TensorShape([None, None, None])])
    self.output.placeholder = r[-1]
    self.output.size_placeholder = {self.output.time_dim_axis_excluding_batch: tf.strided_slice(self.input_data.size_placeholder[self.input_data.time_dim_axis_excluding_batch], [0], tf.shape(self.input_data.size_placeholder[self.input_data.time_dim_axis_excluding_batch]), [pool_size])}


class MelFilterbankLayer(_ConcatInputLayer):
  """
  This layer applies the log Mel filterbank to the input
  """

  layer_class = "mel_filterbank"

  def __init__(self, sampling_rate=16000, fft_size=1024, nr_of_filters=80, **kwargs):
    """
    :param sampling_rate int: sampling rate of the signal which the input originates from
    :param fft_size int: fft_size with which the time signal was transformed into the intput
    :param nr_of_filters int: number of output filter bins
    """
    if ('n_out' in kwargs and (kwargs['n_out'] != nr_of_filters)):
        raise Exception('argument n_out of layer MelFilterbankLayer can not be different from nr_of_filters')
    kwargs['n_out'] = nr_of_filters
    super(MelFilterbankLayer, self).__init__(**kwargs)

    from tfSi6Proc.basics.transformation.fourier import tfMelFilterBank

    # set order of axes
    self.output.batch_dim_axis = 0
    self.output.time_dim_axis = 1
    self.output.feature_dim_axis = 2
    # - ensure correct ordering of input placeholder
    input_placeholder = self.input_data.placeholder
    if ((self.input_data.batch_dim_axis != self.output.batch_dim_axis) or (self.input_data.time_dim_axis != self.output.time_dim_axis) or (self.input_data.feature_dim_axis != self.output.feature_dim_axis)):
      input_placeholder = tf.transpose(input_placeholder, [self.input_data.batch_dim_axis, self.input_data.time_dim_axis, self.input_data.feature_dim_axis])

    mel_fbank_mat = tfMelFilterBank(0, sampling_rate / 2, sampling_rate, fft_size, nr_of_filters)
    self.output.placeholder = tf.einsum('btf,bfc->btc', input_placeholder, tf.tile(tf.expand_dims(mel_fbank_mat, axis=0), [tf.shape(input_placeholder)[0], 1, 1]))


class MaskBasedGevBeamformingLayer(LayerBase):
  """
  This layer applies GEV beamforming to a multichannel signal. The different
  channels are assumed to be concatenated to the
  input feature vector. The first source to the layer must contain the complex
  spectrograms of the single channels and the
  second source must contain the noise and speech masks
  """

  layer_class = "mask_based_gevbeamforming"

  def __init__(self, nr_of_channels=1, postfilter_id=0, **kwargs):
    """
    :param nr_of_channels int: number of input channels to beamforming (needed to split the feature vector)
    :param postfilter_id int: Id which is specifying which post filter to apply in gev beamforming.
                              For more information see tfSi6Proc.audioProcessing.enhancement.beamforming.TfMaskBasedGevBeamformer
    """
    super(MaskBasedGevBeamformingLayer, self).__init__(**kwargs)
    assert len(self.sources) == 2

    # set order of axes
    self.output.batch_dim_axis = 0
    self.output.time_dim_axis = 1
    self.output.feature_dim_axis = 2

    from tfSi6Proc.audioProcessing.enhancement.beamforming import TfMaskBasedGevBeamformer

    complexSpectrogram = tf.transpose(self.sources[0].output.placeholder, [self.sources[0].output.batch_dim_axis, self.sources[0].output.time_dim_axis, self.sources[0].output.feature_dim_axis])
    complexSpectrogram = tf.transpose(tf.reshape(complexSpectrogram, (tf.shape(complexSpectrogram)[0], tf.shape(complexSpectrogram)[1], nr_of_channels, tf.shape(complexSpectrogram)[2] / nr_of_channels)), [0, 1, 3, 2])
    masks = tf.transpose(self.sources[1].output.placeholder, [self.sources[1].output.batch_dim_axis, self.sources[1].output.time_dim_axis, self.sources[1].output.feature_dim_axis])
    masks = tf.transpose(tf.reshape(masks, (tf.shape(masks)[0], tf.shape(masks)[1], nr_of_channels, tf.shape(masks)[2] / nr_of_channels)), [0, 1, 3, 2])
    noiseMasks = masks[:, :, :(tf.shape(masks)[2] / 2), :]
    speechMasks = masks[:, :, (tf.shape(masks)[2] / 2):, :]

    with tf.name_scope("beamformingBatchLoop"):
      batchIdx = tf.constant(0)
      output = tf.Variable(tf.zeros([1, 1, 1], dtype=tf.complex64), dtype=tf.complex64, trainable=False)

      def iterateBatch(batchIdx, output):
          return tf.less(batchIdx, tf.shape(complexSpectrogram)[0])

      def beamform(batchIdx, output):
        gevBf = TfMaskBasedGevBeamformer(tfFreqDomInput=complexSpectrogram[batchIdx, :, :, :], tfNoiseMask=noiseMasks[batchIdx, :, :, :], tfSpeechMask=speechMasks[batchIdx, :, :, :], postFilterId=postfilter_id)
        bfOut = gevBf.getFrequencyDomainOutputSignal()
        bfOut = tf.reshape(bfOut, (1, tf.shape(bfOut)[0], tf.shape(bfOut)[1]))
        output = tf.cond(tf.greater(batchIdx, 0), lambda: tf.concat([output, bfOut], axis=self.output.batch_dim_axis), lambda: bfOut)
        return tf.add(batchIdx, 1), output

      r = tf.while_loop(iterateBatch, beamform, [batchIdx, output], shape_invariants=[batchIdx.get_shape(), tf.TensorShape([None, None, None])])
      self.output.placeholder = r[-1]


class SplitConcatMultiChannel(_ConcatInputLayer):
  """
  This layer assumes the feature vector to be a concatenation of features of
  multiple channels (of the same size). It splits the feature dimension into
  equisized number of channel features and stacks them in the batch dimension.
  Thus the batch size is multiplied with the number of channels and the feature
  size is divided by the number of channels.
  The channels of one singal will have consecutive batch indices, meaning the
  signal of the original batch index n is split
  and can now be found in batch indices (n * nr_of_channels) to
  ((n+1) * nr_of_channels - 1)
  """

  layer_class = "split_concatenated_multichannel"

  def __init__(self, nr_of_channels=1, **kwargs):
    """
    :param nr_of_channels int: the number of concatenated channels in the feature dimension
    """
    super(SplitConcatMultiChannel, self).__init__(**kwargs)

    # set order of axes
    self.output.batch_dim_axis = 0
    self.output.time_dim_axis = 1
    self.output.feature_dim_axis = 2
    # - ensure correct ordering of input placeholder
    input_placeholder = self.input_data.placeholder
    if ((self.input_data.batch_dim_axis != self.output.batch_dim_axis) or (self.input_data.time_dim_axis != self.output.time_dim_axis) or (self.input_data.feature_dim_axis != self.output.feature_dim_axis)):
      input_placeholder = tf.transpose(input_placeholder, [self.input_data.batch_dim_axis, self.input_data.time_dim_axis, self.input_data.feature_dim_axis])

    output = tf.reshape(input_placeholder, [tf.shape(input_placeholder)[0], tf.shape(input_placeholder)[1], nr_of_channels, tf.shape(input_placeholder)[2] / nr_of_channels])
    self.output.placeholder = tf.transpose(tf.reshape(tf.transpose(output, [1, 3, 0, 2]), (tf.shape(output)[1], tf.shape(output)[3], tf.shape(output)[0] * tf.shape(output)[2])), [2, 0, 1])
    # work around to obtain result like numpy.repeat(size_placeholder, nr_of_channels)
    self.output.size_placeholder = {self.output.time_dim_axis_excluding_batch: tf.reshape(tf.tile(tf.reshape(self.input_data.size_placeholder[self.input_data.time_dim_axis_excluding_batch], [-1, 1]), [1, nr_of_channels]), [-1])}


class TileFeaturesLayer(_ConcatInputLayer):
  """
  This function is tiling features with giving number of repetitions
  """

  layer_class = "tile_features"

  def __init__(self, repetitions=1, **kwargs):
    """
    :param repetitions int: number of tiling repetitions in feature domain
    """
    super(TileFeaturesLayer, self).__init__(**kwargs)

    # set order of axes
    self.output.batch_dim_axis = 0
    self.output.time_dim_axis = 1
    self.output.feature_dim_axis = 2
    # - ensure correct ordering of input placeholder
    input_placeholder = self.input_data.placeholder
    if ((self.input_data.batch_dim_axis != self.output.batch_dim_axis) or (self.input_data.time_dim_axis != self.output.time_dim_axis) or (self.input_data.feature_dim_axis != self.output.feature_dim_axis)):
      input_placeholder = tf.transpose(input_placeholder, [self.input_data.batch_dim_axis, self.input_data.time_dim_axis, self.input_data.feature_dim_axis])

    self.output.placeholder = tf.tile(input_placeholder, [1, 1, repetitions])
