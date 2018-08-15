
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


class ComplexLinearProjectionLayer(_ConcatInputLayer):
  layer_class = "complex_linear_projection"

  def __init__(self, nr_of_filters, clp_weights_init="glorot_uniform", **kwargs):
    if ('n_out' in kwargs and (kwargs['n_out'] != nr_of_filters)):
        raise Exception('argument n_out of layer MelFilterbankLayer can not be different from nr_of_filters')
    kwargs['n_out'] = nr_of_filters
    self._nr_of_filters = nr_of_filters
    super(ComplexLinearProjectionLayer, self).__init__(**kwargs)
    self._clp_kernel = self._build_kernel(clp_weights_init)
    self.output.placeholder = self._build_clp_multiplication(self._clp_kernel)

  def _build_kernel(self, clp_weights_init):
    from TFUtil import get_initializer
    input_placeholder = self.input_data.get_placeholder_as_batch_major()
    kernel_width = input_placeholder.shape[2].value // 2
    kernel_height = self._nr_of_filters
    with self.var_creation_scope():
      clp_weights_initializer = get_initializer(
        clp_weights_init, seed=self.network.random.randint(2 ** 31), eval_local_ns={"layer": self})
      clp_kernel = self.add_param(tf.get_variable(
        name="clp_kernel", shape=(2, kernel_width, kernel_height), dtype=tf.float32, initializer=clp_weights_initializer))
    return clp_kernel

  def _build_clp_multiplication(self, clp_kernel):
    from TFUtil import safe_log
    input_placeholder = self.input_data.get_placeholder_as_batch_major()
    tf.assert_equal(tf.shape(clp_kernel)[1], tf.shape(input_placeholder)[2] // 2)
    tf.assert_equal(tf.shape(clp_kernel)[2], self._nr_of_filters)
    input_real = tf.strided_slice(input_placeholder, [0, 0, 0], tf.shape(input_placeholder), [1, 1, 2])
    input_imag = tf.strided_slice(input_placeholder, [0, 0, 1], tf.shape(input_placeholder), [1, 1, 2])
    kernel_real = self._clp_kernel[0, :, :]
    kernel_imag = self._clp_kernel[1, :, :]
    output_real = tf.einsum('btf,fp->btp', input_real, kernel_real) - tf.einsum('btf,fp->btp', input_imag, kernel_imag)
    output_imag = tf.einsum('btf,fp->btp', input_imag, kernel_real) + tf.einsum('btf,fp->btp', input_real, kernel_imag)
    output_uncompressed = tf.sqrt(tf.pow(output_real, 2) + tf.pow(output_imag, 2))
    output_compressed = safe_log(output_uncompressed)
    return output_compressed

  @classmethod
  def get_out_data_from_opts(cls, nr_of_filters, **kwargs):
    if 'n_out' not in kwargs:
      kwargs['n_out'] = nr_of_filters
    return super(ComplexLinearProjectionLayer, cls).get_out_data_from_opts(**kwargs)


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
    def tfMelFilterBank(fMin, fMax, samplingRate, fftSize, nrOfFilters):
      """
      Returns the filter matrix which yields the mel filter bank features, when applied to the spectrum as
      tf.matmul(freqDom, filterMatrix), where freqDom has dimension (time, frequency) and filterMatrix is the matrix returned
      by this function
      The filter matrix is computed according to equation 6.141 in
      [Huang & Acero+, 2001] "Spoken Language Processing - A Guide to Theroy, Algorithm, and System Development"

      :type fMin: float | int
      :param fMin: minimum frequency
      :type fMax: float | int
      :param fMax: maximum frequency
      :type samplingRate: float
      :param samplingRate: sampling rate of audio signal
      :type fftSize: int
      :param fftSize: dimension of discrete fourier transformation
      :type nrOfFilters: int
      :param nrOfFilters: number of mel frequency filter banks to be created

      :rtype: tf.tensor, shape=(filterValue, nrOfFilters)
      :return: matrix yielding the mel frequency cepstral coefficients
      """
      import numpy as np

      def melScale(freq):
        """
        returns the respective value on the mel scale

        :type freq: float
        :param freq: frequency value to transform onto mel scale
        :rtype: float
        """
        return 1125.0 * np.log(1 + float(freq) / 700)

      def invMelScale(melVal):
        """
        returns the respective value in the frequency domain

        :type melVal: float
        :param melVal: value in mel domain
        :rtype: float
        """
        return 700.0 * (np.exp(float(melVal) / 1125) - 1)

      def filterCenter(filterId, fMin, fMax, samplingRate, fftSize, nrOfFilters):
        """
        :type filterId: int
        :param filterId: filter to compute the center frequency for
        :type fMin: float | int
        :param fMin: minimum frequency
        :type fMax: float | int
        :param fMax: maximum frequency
        :type samplingRate: float
        :param samplingRate: sampling rate of audio signal
        :type fftSize: int
        :param fftSize: dimension of discrete fourier transformation
        :type nrOfFilters: int
        :param nrOfFilters: number of mel frequency filter banks to be created

        :rtype: float
        :return: center frequency of filter
        """
        return (float(fftSize) / samplingRate) * invMelScale(melScale(fMin) + filterId * ((melScale(fMax) - melScale(fMin)) / (nrOfFilters + 1)))

      filtCent = np.zeros(shape=(nrOfFilters + 2,), dtype=np.float32)
      for i1 in range(nrOfFilters + 2):
        filtCent[i1] = filterCenter(i1, fMin, fMax, samplingRate, fftSize, nrOfFilters)
      fMat = np.zeros(shape=(int(np.floor(fftSize / 2) + 1), nrOfFilters))
      for i1 in range(fMat.shape[0]):
        for i2 in range(1, nrOfFilters + 1):
          if (i1 > filtCent[i2 - 1]) and (i1 < filtCent[i2 + 1]):
            if i1 < filtCent[i2]:
              num = i1 - filtCent[i2 - 1]
              denom = filtCent[i2] - filtCent[i2 - 1]
            else:
              num = filtCent[i2 + 1] - i1
              denom = filtCent[i2 + 1] - filtCent[i2]
            elVal = num / denom
          else:
            elVal = 0
          fMat[i1, i2 - 1] = elVal
      return tf.constant(fMat, dtype=tf.float32)

    if ('n_out' in kwargs and (kwargs['n_out'] != nr_of_filters)):
        raise Exception('argument n_out of layer MelFilterbankLayer can not be different from nr_of_filters')
    kwargs['n_out'] = nr_of_filters
    super(MelFilterbankLayer, self).__init__(**kwargs)

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


class MultiChannelStftLayer(_ConcatInputLayer):
  """
  The layer applys a STFT to every channel separately and concatenates the frequency domain vectors for every frame
  """
  layer_class = "multichannel_stft_layer"
  recurrent = True

  def __init__(self, frame_shift, frame_size, fft_size, window="hanning", use_rfft=True, nr_of_channels=1, pad_last_frame=False, **kwargs):
    """
    :param int frame_shift: frame shift for stft in samples
    :param int frame_size: frame size for stft in samples 
    :param int fft_size: fft size in samples 
    :param str window: id of the windowing function used. Possible options are:
      - hanning
    :param bool use_rfft: if set to true a real input signal is expected and only
      the significant half of the FFT bins are returned
    :param int nr_of_channels: number of input channels 
    :param bool pad_last_frame: padding of last frame with zeros or discarding of
      last frame
    """
    n_out = self._get_n_out_by_fft_config(fft_size, use_rfft, nr_of_channels)
    if ('n_out' in kwargs and (kwargs['n_out'] != n_out)):
        raise Exception('argument n_out of layer MultiChannelStftLayer does not match the fft configuration')
    kwargs['n_out'] = n_out
    super(MultiChannelStftLayer, self).__init__(**kwargs)
    tf.assert_equal(nr_of_channels, self._get_nr_of_channels_from_input_placeholder())
    self._nr_of_channels = nr_of_channels
    self._frame_shift = frame_shift
    self._frame_size = frame_size
    self._fft_size = fft_size
    self._window = window
    self._use_rfft = use_rfft
    self._pad_last_frame = pad_last_frame
    self.output.placeholder = self._apply_stft_to_input()

  def _get_nr_of_channels_from_input_placeholder(self):
    input_placeholder = self.input_data.get_placeholder_as_batch_major()
    return input_placeholder.shape[2]

  def _apply_stft_to_input(self):
    input_placeholder = self.input_data.get_placeholder_as_batch_major()
    if self._use_rfft:
      channel_wise_stft = tf.contrib.signal.stft(
        signals=tf.transpose(input_placeholder, [0, 2, 1]),
        frame_length=self._frame_size,
        frame_step=self._frame_shift,
        fft_length=self._fft_size,
        window_fn=self._get_window,
        pad_end=self._pad_last_frame
      )
      channel_wise_stft = tf.transpose(channel_wise_stft, [0, 2, 1, 3])
      batch_dim = tf.shape(channel_wise_stft)[0]
      time_dim = tf.shape(channel_wise_stft)[1]
      concat_feature_dim = channel_wise_stft.shape[2] * channel_wise_stft.shape[3]
      channel_concatenated_stft = tf.reshape(channel_wise_stft, (batch_dim, time_dim, concat_feature_dim))
      output_placeholder = channel_concatenated_stft
    return output_placeholder

  def _get_window(self, window_length, dtype):
    if self._window == "hanning":
        window = tf.contrib.signal.hann_window(window_length, dtype=dtype)
    if self._window == "None" or self._window == "ones":
      window = tf.ones((window_length,), dtype=dtype)
    return window

  @classmethod
  def _get_n_out_by_fft_config(cls, fft_size, use_rfft, nr_of_channels):
    n_out = fft_size
    if use_rfft:
        n_out = fft_size // 2 + 1
    n_out *= nr_of_channels
    return n_out

  @classmethod
  def get_out_data_from_opts(cls, fft_size, use_rfft=True, nr_of_channels=1, **kwargs):
    n_out = cls._get_n_out_by_fft_config(fft_size, use_rfft, nr_of_channels)
    if 'n_out' not in kwargs:
      kwargs['n_out'] = n_out
    return super(MultiChannelStftLayer, cls).get_out_data_from_opts(**kwargs)


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
