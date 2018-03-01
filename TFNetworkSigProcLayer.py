
from __future__ import print_function

import tensorflow as tf
from TFNetworkLayer import LayerBase, _ConcatInputLayer, get_concat_sources_data_template
from TFUtil import Data


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

    input_placeholder = self.input_data.get_placeholder_as_batch_major()

    real_value = tf.strided_slice(input_placeholder, [0, 0, 0], tf.shape(input_placeholder), [1, 1, 2])
    imag_value = tf.strided_slice(input_placeholder, [0, 0, 1], tf.shape(input_placeholder), [1, 1, 2])
    self.output.placeholder = tf.complex(real_value, imag_value)
    self.output.size_placeholder = {0: self.input_data.size_placeholder[self.input_data.time_dim_axis_excluding_batch]}

  @classmethod
  def get_out_data_from_opts(cls, name, sources, n_out=None, **kwargs):
    return super(AlternatingRealToComplexLayer, cls).get_out_data_from_opts(name=name, sources=sources, out_type={"dim": n_out, "dtype": "complex64", "batch_dim_axis": 0, "time_dim_axis": 1}, **kwargs)


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

    input_placeholder = self.input_data.get_placeholder_as_batch_major()

    # get median over pooled batches
    # - reshape input for usage with tf.nn.top_k
    reshaped_input = tf.reshape(tf.transpose(input_placeholder, [1, 2, 0]), shape=(tf.shape(input_placeholder)[1], tf.shape(input_placeholder)[2], tf.shape(input_placeholder)[0] / pool_size, pool_size))
    # - get median of each pool
    median = tf.nn.top_k(reshaped_input, k=tf.cast(tf.ceil(tf.constant(pool_size, dtype=tf.float32) / 2), dtype=tf.int32)).values[:, :, :, -1]
    median_batch_major = tf.transpose(median, [2, 0, 1])
    self.output.placeholder = median_batch_major
    self.output.size_placeholder = {self.output.time_dim_axis_excluding_batch: tf.strided_slice(self.input_data.size_placeholder[self.input_data.time_dim_axis_excluding_batch], [0], tf.shape(self.input_data.size_placeholder[self.input_data.time_dim_axis_excluding_batch]), [pool_size])}

  @classmethod
  def get_out_data_from_opts(cls, name, sources, pool_size, n_out=None, **kwargs):
    input_data = get_concat_sources_data_template(sources)
    assert not input_data.sparse
    return Data(
      name="%s_output" % name,
      shape=[input_data.get_placeholder_as_batch_major().shape[1].value, input_data.get_placeholder_as_batch_major().shape[2].value],
      dtype=input_data.dtype,
      size_placeholder={0: tf.strided_slice(input_data.size_placeholder[input_data.time_dim_axis_excluding_batch], [0], tf.shape(input_data.size_placeholder[input_data.time_dim_axis_excluding_batch]), [pool_size])},
      sparse=False,
      batch_dim_axis=0,
      time_dim_axis=1)


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

    input_placeholder = self.input_data.get_placeholder_as_batch_major()

    mel_fbank_mat = tfMelFilterBank(0, sampling_rate / 2.0, sampling_rate, fft_size, nr_of_filters)
    self.output.placeholder = tf.einsum('btf,bfc->btc', input_placeholder, tf.tile(tf.expand_dims(mel_fbank_mat, axis=0), [tf.shape(input_placeholder)[0], 1, 1]))
    self.output.size_placeholder = {0: self.input_data.size_placeholder[self.input_data.time_dim_axis_excluding_batch]}

  @classmethod
  def get_out_data_from_opts(cls, name, sources, n_out=None, **kwargs):
    return super(MelFilterbankLayer, cls).get_out_data_from_opts(name=name, sources=sources, out_type={"dim": n_out, "batch_dim_axis": 0, "time_dim_axis": 1}, **kwargs)


class MaskBasedGevBeamformingLayer(LayerBase):
  """
  This layer applies GEV beamforming to a multichannel signal. The different
  channels are assumed to be concatenated to the
  input feature vector. The first source to the layer must contain the complex
  spectrograms of the single channels and the
  second source must contain the noise and speech masks
  """

  layer_class = "mask_based_gevbeamforming"

  def __init__(self, nr_of_channels=1, postfilter_id=0, qralgorithm_steps=None, **kwargs):
    """
    :param int nr_of_channels: number of input channels to beamforming (needed to split the feature vector)
    :param int postfilter_id: Id which is specifying which post filter to apply in gev beamforming.
                              For more information see
                              tfSi6Proc.audioProcessing.enhancement.beamforming.TfMaskBasedGevBeamformer
    """
    super(MaskBasedGevBeamformingLayer, self).__init__(**kwargs)
    assert len(self.sources) == 2

    from tfSi6Proc.audioProcessing.enhancement.beamforming import TfMaskBasedGevBeamformer

    complexSpectrogram = self.sources[0].output.get_placeholder_as_batch_major()
    complexSpectrogram = tf.transpose(tf.reshape(complexSpectrogram, (tf.shape(complexSpectrogram)[0], tf.shape(complexSpectrogram)[1], nr_of_channels, tf.shape(complexSpectrogram)[2] // nr_of_channels)), [0, 1, 3, 2])
    masks = tf.transpose(self.sources[1].output.placeholder, [self.sources[1].output.batch_dim_axis, self.sources[1].output.time_dim_axis, self.sources[1].output.feature_dim_axis])
    masks = tf.transpose(tf.reshape(masks, (tf.shape(masks)[0], tf.shape(masks)[1], nr_of_channels, tf.shape(masks)[2] / nr_of_channels)), [0, 1, 3, 2])
    noiseMasks = masks[:, :, :(tf.shape(masks)[2] // 2), :]
    speechMasks = masks[:, :, (tf.shape(masks)[2] // 2):, :]

    gevBf = TfMaskBasedGevBeamformer(flag_inputHasBatch=1, tfFreqDomInput=complexSpectrogram, tfNoiseMask=noiseMasks, tfSpeechMask=speechMasks, postFilterId=postfilter_id, qrAlgorithmSteps=qralgorithm_steps)
    bfOut = gevBf.getFrequencyDomainOutputSignal()
    self.output.placeholder = bfOut

  @classmethod
  def get_out_data_from_opts(cls, out_type={}, n_out=None, **kwargs):
    out_type.setdefault("dim", n_out)
    out_type["batch_dim_axis"] = 0
    out_type["time_dim_axis"] = 1
    return super(MaskBasedGevBeamformingLayer, cls).get_out_data_from_opts(out_type=out_type, **kwargs)


class MaskBasedMvdrBeamformingWithDiagLoadingLayer(LayerBase):
  """
  This layer applies GEV beamforming to a multichannel signal. The different
  channels are assumed to be concatenated to the
  input feature vector. The first source to the layer must contain the complex
  spectrograms of the single channels and the
  second source must contain the noise and speech masks
  """

  layer_class = "mask_based_mvdrbeamforming"

  def __init__(self, nr_of_channels=1, diag_loading_coeff=0, qralgorithm_steps=None, **kwargs):
    """
    :param int nr_of_channels: number of input channels to beamforming (needed to split the feature vector)
    :param int diag_loading_coeff: weighting coefficient for diagonal loading.
    """
    super(MaskBasedMvdrBeamformingWithDiagLoadingLayer, self).__init__(**kwargs)
    assert len(self.sources) == 2

    from tfSi6Proc.audioProcessing.enhancement.beamforming import TfMaskBasedMvdrBeamformer

    complexSpectrogramWithConcatChannels = self.sources[0].output.get_placeholder_as_batch_major()
    complexSpectrogram = tf.transpose(tf.reshape(complexSpectrogramWithConcatChannels, (tf.shape(complexSpectrogramWithConcatChannels)[0], tf.shape(complexSpectrogramWithConcatChannels)[1], nr_of_channels, tf.shape(complexSpectrogramWithConcatChannels)[2] // nr_of_channels)), [0, 1, 3, 2])
#    noiseMasks = tf.transpose(self.sources[1].output.placeholder, [self.sources[1].output.batch_dim_axis, self.sources[1].output.time_dim_axis, self.sources[1].output.feature_dim_axis])
    noiseMasks = self.sources[1].output.get_placeholder_as_batch_major()
    noiseMasks = tf.transpose(tf.reshape(noiseMasks, (tf.shape(noiseMasks)[0], tf.shape(noiseMasks)[1], nr_of_channels, tf.shape(noiseMasks)[2] // nr_of_channels)), [0, 1, 3, 2])

    mvdrBf = TfMaskBasedMvdrBeamformer(flag_inputHasBatch=1, tfFreqDomInput=complexSpectrogram, tfNoiseMask=noiseMasks, tfDiagLoadingCoeff=tf.constant(diag_loading_coeff, dtype=tf.float32), qrAlgorithmSteps=qralgorithm_steps)
    bfOut = mvdrBf.getFrequencyDomainOutputSignal()
    self.output.placeholder = bfOut

  @classmethod
  def get_out_data_from_opts(cls, out_type={}, n_out=None, **kwargs):
    out_type.setdefault("dim", n_out)
    out_type["batch_dim_axis"] = 0
    out_type["time_dim_axis"] = 1
    return super(MaskBasedMvdrBeamformingWithDiagLoadingLayer, cls).get_out_data_from_opts(out_type=out_type, **kwargs)


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
    :param int nr_of_channels: the number of concatenated channels in the feature dimension
    """
    super(SplitConcatMultiChannel, self).__init__(**kwargs)

    input_placeholder = self.input_data.get_placeholder_as_batch_major()

    output = tf.reshape(input_placeholder, [tf.shape(input_placeholder)[0], tf.shape(input_placeholder)[1], nr_of_channels, tf.shape(input_placeholder)[2] / nr_of_channels])
    self.output.placeholder = tf.transpose(tf.reshape(tf.transpose(output, [1, 3, 0, 2]), (tf.shape(output)[1], tf.shape(output)[3], tf.shape(output)[0] * tf.shape(output)[2])), [2, 0, 1])
    # work around to obtain result like numpy.repeat(size_placeholder, nr_of_channels)
    self.output.size_placeholder = {self.output.time_dim_axis_excluding_batch: tf.reshape(tf.tile(tf.reshape(self.input_data.size_placeholder[self.input_data.time_dim_axis_excluding_batch], [-1, 1]), [1, nr_of_channels]), [-1])}

  @classmethod
  def get_out_data_from_opts(cls, name, sources, nr_of_channels, n_out=None, **kwargs):
    input_data = get_concat_sources_data_template(sources)
    assert not input_data.sparse
    return Data(
      name="%s_output" % name,
      shape=[input_data.get_placeholder_as_batch_major().shape[1].value, input_data.get_placeholder_as_batch_major().shape[2].value // nr_of_channels],
      dtype=input_data.dtype,
      size_placeholder={0: tf.reshape(tf.tile(tf.reshape(input_data.size_placeholder[input_data.time_dim_axis_excluding_batch], [-1, 1]), [1, nr_of_channels]), [-1])},
      sparse=False,
      batch_dim_axis=0,
      time_dim_axis=1)


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

    input_placeholder = self.input_data.get_placeholder_as_batch_major()

    self.output.placeholder = tf.tile(input_placeholder, [1, 1, repetitions])

  @classmethod
  def get_out_data_from_opts(cls, name, sources, repetitions, n_out=None, **kwargs):
    input_data = get_concat_sources_data_template(sources)
    assert not input_data.sparse
    return Data(
      name="%s_output" % name,
      shape=[input_data.get_placeholder_as_batch_major().shape[1].value, input_data.get_placeholder_as_batch_major().shape[2].value * repetitions],
      dtype=input_data.dtype,
      sparse=False,
      size_placeholder={0: input_data.size_placeholder[input_data.time_dim_axis_excluding_batch]},
      batch_dim_axis=0,
      time_dim_axis=1)
