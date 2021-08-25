"""
Defines multiple signal processing-related layers.
"""

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import returnn.tf.compat as tf_compat
from returnn.tf.layers.basic import LayerBase, _ConcatInputLayer, get_concat_sources_data_template
from returnn.tf.util.basic import Data


class AlternatingRealToComplexLayer(_ConcatInputLayer):
  """
  This layer converts a real valued input tensor into a complex valued output tensor.
  For this even and odd features are considered the real and imaginary part of one complex number, respectively
  """

  layer_class = "alternating_real_to_complex"

  def __init__(self, **kwargs):
    super(AlternatingRealToComplexLayer, self).__init__(**kwargs)

    input_placeholder = self.input_data.get_placeholder_as_batch_major()

    real_value = tf.strided_slice(input_placeholder, [0, 0, 0], tf.shape(input_placeholder), [1, 1, 2])
    imag_value = tf.strided_slice(input_placeholder, [0, 0, 1], tf.shape(input_placeholder), [1, 1, 2])
    self.output.placeholder = tf.complex(real_value, imag_value)
    self.output.size_placeholder = {0: self.input_data.size_placeholder[self.input_data.time_dim_axis_excluding_batch]}

  @classmethod
  def get_out_data_from_opts(cls, name, sources, n_out=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int|None|returnn.util.basic.NotSpecified n_out:
    :rtype: Data
    """
    return super(AlternatingRealToComplexLayer, cls).get_out_data_from_opts(
      name=name, sources=sources,
      out_type={"dim": n_out, "dtype": "complex64", "batch_dim_axis": 0, "time_dim_axis": 1}, **kwargs)


class BatchMedianPoolingLayer(_ConcatInputLayer):
  """
  This layer is used to pool together batches by taking their medium value.
  Thus the batch size is divided by pool_size.
  The stride is hard coded to be equal to the pool size.
  """

  layer_class = "batch_median_pooling"

  def __init__(self, pool_size=1, **kwargs):
    """
    :param int pool_size: size of the pool to take median of (is also used as stride size)
    """
    super(BatchMedianPoolingLayer, self).__init__(**kwargs)

    input_placeholder = self.input_data.get_placeholder_as_batch_major()

    # get median over pooled batches
    # - reshape input for usage with tf.nn.top_k
    reshaped_input = tf.reshape(
      tf.transpose(input_placeholder, [1, 2, 0]),
      shape=(
        tf.shape(input_placeholder)[1],
        tf.shape(input_placeholder)[2],
        tf.shape(input_placeholder)[0] // pool_size,
        pool_size))
    # - get median of each pool
    median = tf.nn.top_k(
      reshaped_input,
      k=tf.cast(tf_compat.v1.ceil(tf.constant(pool_size, dtype=tf.float32) / 2.), dtype=tf.int32)).values[:, :, :, -1]
    median_batch_major = tf.transpose(median, [2, 0, 1])
    self.output.placeholder = median_batch_major
    self.output.size_placeholder = {
      self.output.time_dim_axis_excluding_batch: tf.strided_slice(
        self.input_data.size_placeholder[self.input_data.time_dim_axis_excluding_batch],
        [0], tf.shape(self.input_data.size_placeholder[self.input_data.time_dim_axis_excluding_batch]), [pool_size])}

  @classmethod
  def get_out_data_from_opts(cls, name, sources, pool_size, n_out=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int pool_size:
    :param int|None|returnn.util.basic.NotSpecified n_out:
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources)
    assert not input_data.sparse
    return Data(
      name="%s_output" % name,
      shape=[
        input_data.get_placeholder_as_batch_major().shape[1].value,
        input_data.get_placeholder_as_batch_major().shape[2].value],
      dtype=input_data.dtype,
      size_placeholder={0: tf.strided_slice(
        input_data.size_placeholder[input_data.time_dim_axis_excluding_batch],
        [0], tf.shape(input_data.size_placeholder[input_data.time_dim_axis_excluding_batch]), [pool_size])},
      sparse=False,
      batch_dim_axis=0,
      time_dim_axis=1)


class ComplexLinearProjectionLayer(_ConcatInputLayer):
  """
  Complex linear projection layer
  For the original idea, see:
  Variani, Ehsan, et al. "Complex linear projection (CLP): A discriminative approach to joint feature extraction and
  acoustic modeling." (2016).
  """
  layer_class = "complex_linear_projection"

  def __init__(self, nr_of_filters, clp_weights_init="glorot_uniform", **kwargs):
    """
    :param int nr_of_filters:
    :param str|dict[str]|float|numpy.ndarray clp_weights_init:
    """
    if 'n_out' in kwargs and kwargs['n_out'] != nr_of_filters:
      raise Exception('argument n_out of layer MelFilterbankLayer can not be different from nr_of_filters')
    kwargs['n_out'] = nr_of_filters
    self._nr_of_filters = nr_of_filters
    super(ComplexLinearProjectionLayer, self).__init__(**kwargs)
    self._clp_kernel = self._build_kernel(clp_weights_init)
    self.output.placeholder = self._build_clp_multiplication(self._clp_kernel)

  def _build_kernel(self, clp_weights_init):
    from returnn.tf.util.basic import get_initializer
    input_placeholder = self.input_data.get_placeholder_as_batch_major()
    kernel_width = input_placeholder.shape[2].value // 2
    kernel_height = self._nr_of_filters
    with self.var_creation_scope():
      clp_weights_initializer = get_initializer(
        clp_weights_init, seed=self.network.random.randint(2 ** 31), eval_local_ns={"layer": self})
      clp_kernel = self.add_param(tf_compat.v1.get_variable(
        name="clp_kernel", shape=(2, kernel_width, kernel_height), dtype=tf.float32,
        initializer=clp_weights_initializer))
    return clp_kernel

  def _build_clp_multiplication(self, clp_kernel):
    from returnn.tf.util.basic import safe_log
    input_placeholder = self.input_data.get_placeholder_as_batch_major()
    tf_compat.v1.assert_equal(tf.shape(clp_kernel)[1], tf.shape(input_placeholder)[2] // 2)
    tf_compat.v1.assert_equal(tf.shape(clp_kernel)[2], self._nr_of_filters)
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
    """
    :param int nr_of_filters:
    :rtype: Data
    """
    if 'n_out' not in kwargs:
      kwargs['n_out'] = nr_of_filters
    return super(ComplexLinearProjectionLayer, cls).get_out_data_from_opts(**kwargs)


class ComplexToAlternatingRealLayer(_ConcatInputLayer):
  """
  This layer converts a complex valued input tensor into a real valued output tensor.
  For this the even and odd parts of the output are considered the real and imaginary part of one complex number,
  respectively.
  """

  layer_class = "complex_to_alternating_real"

  def __init__(self, **kwargs):
    def _interleave_vectors(vec1, vec2):
      vec1 = tf.expand_dims(vec1, 3)
      vec2 = tf.expand_dims(vec2, 3)
      interleaved = tf.concat([vec1, vec2], 3)
      interleaved = tf.reshape(interleaved, (tf.shape(vec1)[0], tf.shape(vec1)[1], tf.shape(vec1)[2] * 2))
      return interleaved

    super(ComplexToAlternatingRealLayer, self).__init__(**kwargs)

    input_placeholder = self.input_data.get_placeholder_as_batch_major()

    real_value = tf_compat.v1.real(input_placeholder)
    imag_value = tf_compat.v1.imag(input_placeholder)
    self.output.placeholder = _interleave_vectors(real_value, imag_value)
    self.output.size_placeholder = {0: self.input_data.size_placeholder[self.input_data.time_dim_axis_excluding_batch]}


class MaskBasedGevBeamformingLayer(LayerBase):
  """
  This layer applies GEV beamforming to a multichannel signal. The different channels are assumed to be concatenated to
  the input feature vector. The first source to the layer must contain the complex spectrograms of the single channels
  and the second source must contain the noise and speech masks
  """

  layer_class = "mask_based_gevbeamforming"

  def __init__(self, nr_of_channels=1, postfilter_id=0, qralgorithm_steps=None, output_nan_filter=False, **kwargs):
    """
    :param int nr_of_channels: number of input channels to beamforming (needed to split the feature vector)
    :param int postfilter_id: Id which is specifying which post filter to apply in gev beamforming.
                              For more information see
                              tfSi6Proc.audioProcessing.enhancement.beamforming.TfMaskBasedGevBeamformer
    :param int|None: nr of steps of the qr algorithm to compute eigen vector for beamforming
    :param bool output_nan_filter: if set to true nan values in the beamforming output are replaced by zero
    """
    super(MaskBasedGevBeamformingLayer, self).__init__(**kwargs)
    assert len(self.sources) == 2
    # noinspection PyUnresolvedReferences
    from tfSi6Proc.audioProcessing.enhancement.beamforming import TfMaskBasedGevBeamformer

    complex_spectrogram = self.sources[0].output.get_placeholder_as_batch_major()
    complex_spectrogram = tf.transpose(tf.reshape(complex_spectrogram, (
      tf.shape(complex_spectrogram)[0], tf.shape(complex_spectrogram)[1], nr_of_channels,
      tf.shape(complex_spectrogram)[2] // nr_of_channels)), [0, 1, 3, 2])
    masks = tf.transpose(
      self.sources[1].output.placeholder,
      [self.sources[1].output.batch_dim_axis, self.sources[1].output.time_dim_axis,
       self.sources[1].output.feature_dim_axis])
    masks = tf.transpose(
      tf.reshape(masks, (tf.shape(masks)[0], tf.shape(masks)[1], nr_of_channels, tf.shape(masks)[2] // nr_of_channels)),
      [0, 1, 3, 2])
    noise_masks = masks[:, :, :(tf.shape(masks)[2] // 2), :]
    speech_masks = masks[:, :, (tf.shape(masks)[2] // 2):, :]

    gev_bf = TfMaskBasedGevBeamformer(
      flag_inputHasBatch=1, tfFreqDomInput=complex_spectrogram, tfNoiseMask=noise_masks, tfSpeechMask=speech_masks,
      postFilterId=postfilter_id, qrAlgorithmSteps=qralgorithm_steps, outputNanFilter=output_nan_filter)
    bf_out = gev_bf.getFrequencyDomainOutputSignal()
    self.output.placeholder = bf_out

  @classmethod
  def get_out_data_from_opts(cls, out_type=None, n_out=None, **kwargs):
    """
    :param dict[str]|None out_type:
    :param int|None|returnn.util.basic.NotSpecified n_out:
    :rtype: Data
    """
    if out_type is None:
      out_type = {}
    out_type.setdefault("dim", n_out)
    out_type["batch_dim_axis"] = 0
    out_type["time_dim_axis"] = 1
    return super(MaskBasedGevBeamformingLayer, cls).get_out_data_from_opts(out_type=out_type, **kwargs)


class MaskBasedMvdrBeamformingWithDiagLoadingLayer(LayerBase):
  """
  This layer applies GEV beamforming to a multichannel signal. The different channels are assumed to be concatenated to
  the input feature vector. The first source to the layer must contain the complex spectrograms of the single channels
  and the second source must contain the noise and speech masks.
  """

  layer_class = "mask_based_mvdrbeamforming"

  def __init__(self, nr_of_channels=1, diag_loading_coeff=0, qralgorithm_steps=None, output_nan_filter=False, **kwargs):
    """
    :param int nr_of_channels: number of input channels to beamforming (needed to split the feature vector)
    :param int diag_loading_coeff: weighting coefficient for diagonal loading.
    :param int|None qralgorithm_steps: nr of steps of the qr algorithm to compute eigen vector for beamforming
    :param bool output_nan_filter: if set to true nan values in the beamforming output are replaced by zero
    """
    super(MaskBasedMvdrBeamformingWithDiagLoadingLayer, self).__init__(**kwargs)
    assert len(self.sources) == 2

    # noinspection PyUnresolvedReferences
    from tfSi6Proc.audioProcessing.enhancement.beamforming import TfMaskBasedMvdrBeamformer

    complex_spectrogram_with_concat_channels = self.sources[0].output.get_placeholder_as_batch_major()
    complex_spectrogram = tf.transpose(tf.reshape(complex_spectrogram_with_concat_channels, (
      tf.shape(complex_spectrogram_with_concat_channels)[0], tf.shape(complex_spectrogram_with_concat_channels)[1],
      nr_of_channels, tf.shape(complex_spectrogram_with_concat_channels)[2] // nr_of_channels)), [0, 1, 3, 2])
    noise_masks = self.sources[1].output.get_placeholder_as_batch_major()
    noise_masks = tf.transpose(tf.reshape(
      noise_masks,
      (tf.shape(noise_masks)[0], tf.shape(noise_masks)[1], nr_of_channels, tf.shape(noise_masks)[2] // nr_of_channels)),
      [0, 1, 3, 2])

    mvdr_bf = TfMaskBasedMvdrBeamformer(
      flag_inputHasBatch=1, tfFreqDomInput=complex_spectrogram, tfNoiseMask=noise_masks,
      tfDiagLoadingCoeff=tf.constant(diag_loading_coeff, dtype=tf.float32), qrAlgorithmSteps=qralgorithm_steps,
      outputNanFilter=output_nan_filter)
    bf_out = mvdr_bf.getFrequencyDomainOutputSignal()
    self.output.placeholder = bf_out

  @classmethod
  def get_out_data_from_opts(cls, out_type=None, n_out=None, **kwargs):
    """
    :param dict[str]|None out_type:
    :param int|None|returnn.util.basic.NotSpecified n_out:
    :rtype: Data
    """
    if out_type is None:
      out_type = {}
    out_type.setdefault("dim", n_out)
    out_type["batch_dim_axis"] = 0
    out_type["time_dim_axis"] = 1
    return super(MaskBasedMvdrBeamformingWithDiagLoadingLayer, cls).get_out_data_from_opts(out_type=out_type, **kwargs)


class MelFilterbankLayer(_ConcatInputLayer):
  """
  This layer applies the Mel filterbank to the input.
  """

  layer_class = "mel_filterbank"

  def __init__(self, sampling_rate=16000, fft_size=1024, nr_of_filters=80, **kwargs):
    """
    :param int sampling_rate: sampling rate of the signal which the input originates from
    :param int fft_size: fft_size with which the time signal was transformed into the intput
    :param int nr_of_filters: number of output filter bins
    """

    # noinspection PyShadowingNames
    def tf_mel_filter_bank(f_min, f_max, sampling_rate, fft_size, nr_of_filters):
      """
      Returns the filter matrix which yields the mel filter bank features, when applied to the spectrum as
      tf.matmul(freqDom, filterMatrix), where freqDom has dimension (time, frequency)
      and filterMatrix is the matrix returned
      by this function
      The filter matrix is computed according to equation 6.141 in
      [Huang & Acero+, 2001] "Spoken Language Processing - A Guide to Theroy, Algorithm, and System Development"

      :param float|int f_min: minimum frequency
      :param float|int f_max: maximum frequency
      :param float sampling_rate: sampling rate of audio signal
      :param int fft_size: dimension of discrete fourier transformation
      :param int nr_of_filters: number of mel frequency filter banks to be created
      :rtype: tf.tensor, shape=(filterValue, nrOfFilters)
      :return: matrix yielding the mel frequency cepstral coefficients
      """
      import numpy as np

      def mel_scale(freq):
        """
        returns the respective value on the mel scale

        :param float freq: frequency value to transform onto mel scale
        :rtype: float
        """
        return 1125.0 * np.log(1 + float(freq) / 700)

      def inv_mel_scale(mel_val):
        """
        returns the respective value in the frequency domain

        :param float mel_val: value in mel domain
        :rtype: float
        """
        return 700.0 * (np.exp(float(mel_val) / 1125) - 1)

      # noinspection PyShadowingNames
      def filter_center(filter_id, f_min, f_max, sampling_rate, fft_size, nr_of_filters):
        """
        :param int filter_id: filter to compute the center frequency for
        :param float|int f_min: minimum frequency
        :param float|int f_max: maximum frequency
        :param float sampling_rate: sampling rate of audio signal
        :param int fft_size: dimension of discrete fourier transformation
        :param int nr_of_filters: number of mel frequency filter banks to be created
        :rtype: float
        :return: center frequency of filter
        """
        return (float(fft_size) / sampling_rate) * inv_mel_scale(
          mel_scale(f_min) + filter_id * ((mel_scale(f_max) - mel_scale(f_min)) / (nr_of_filters + 1)))

      filt_cent = np.zeros(shape=(nr_of_filters + 2,), dtype=np.float32)
      for i1 in range(nr_of_filters + 2):
        filt_cent[i1] = filter_center(i1, f_min, f_max, sampling_rate, fft_size, nr_of_filters)
      f_mat = np.zeros(shape=(int(np.floor(fft_size / 2) + 1), nr_of_filters))
      for i1 in range(f_mat.shape[0]):
        for i2 in range(1, nr_of_filters + 1):
          if (i1 > filt_cent[i2 - 1]) and (i1 < filt_cent[i2 + 1]):
            if i1 < filt_cent[i2]:
              num = i1 - filt_cent[i2 - 1]
              denom = filt_cent[i2] - filt_cent[i2 - 1]
            else:
              num = filt_cent[i2 + 1] - i1
              denom = filt_cent[i2 + 1] - filt_cent[i2]
            el_val = num / denom
          else:
            el_val = 0
          f_mat[i1, i2 - 1] = el_val
      return tf.constant(f_mat, dtype=tf.float32)

    if 'n_out' in kwargs and (kwargs['n_out'] != nr_of_filters):
      raise Exception('argument n_out of layer MelFilterbankLayer can not be different from nr_of_filters')
    kwargs['n_out'] = nr_of_filters
    super(MelFilterbankLayer, self).__init__(**kwargs)

    input_placeholder = self.input_data.get_placeholder_as_batch_major()

    mel_fbank_mat = tf_mel_filter_bank(0, sampling_rate / 2.0, sampling_rate, fft_size, nr_of_filters)
    self.output.placeholder = tf.einsum(
      'btf,bfc->btc', input_placeholder,
      tf.tile(tf.expand_dims(mel_fbank_mat, axis=0), [tf.shape(input_placeholder)[0], 1, 1]))
    self.output.size_placeholder = {0: self.input_data.size_placeholder[self.input_data.time_dim_axis_excluding_batch]}

  @classmethod
  def get_out_data_from_opts(cls, name, sources, n_out=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int|None|returnn.util.basic.NotSpecified n_out:
    :rtype: Data
    """
    return super(MelFilterbankLayer, cls).get_out_data_from_opts(
      name=name, sources=sources, out_type={"dim": n_out, "batch_dim_axis": 0, "time_dim_axis": 1}, **kwargs)


class MultiChannelMultiResolutionStftLayer(_ConcatInputLayer):
  """
  The layer applys a STFT to every channel separately and concatenates the frequency domain vectors for every frame.
  The STFT is applied with multiple different frame- and FFT-sizes and the resulting multi-channel STFTs are
  concatenated.
  Resulting in a tensor with the content [res_0-ch_0, ..., res_0-ch_N, res_1-ch_0, ... res_M-ch_N]
  The subsampling from T input samples to T' output frames is computed as follows:
  T' = (T - frame_size) / frame_shift + 1
  frame_shift is the same for all resolutions and T' is computed according to a reference frame_size which is taken to
  be frame_sizes[0].
  For all other frame sizes the input is zero-padded or the output is cut to obtain the same T' as for the reference
  frame_size.
  """
  layer_class = "multichannel_multiresolution_stft_layer"
  recurrent = True

  def __init__(self, frame_shift, frame_sizes, fft_sizes, window="hanning", use_rfft=True,
               nr_of_channels=1, pad_last_frame=False, **kwargs):
    """
    :param int frame_shift: frame shift for stft in samples
    :param list[int] frame_sizes: frame size for stft in samples
    :param list[int] fft_sizes: fft size in samples
    :param str window: id of the windowing function used. Possible options are:
      - hanning
    :param bool use_rfft: if set to true a real input signal is expected and only the significant half of the FFT bins
    are returned
    :param int nr_of_channels: number of input channels
    :param bool pad_last_frame: padding of last frame with zeros or discarding of last frame
    """

    def _compute_size_placeholder():
      size_placeholder_dict = {}
      nr_of_full_frames = (self.input_data.size_placeholder[0] - self._reference_frame_size) // self._frame_shift + 1
      nf_of_paded_frames = 0
      if (
        self._pad_last_frame and
        ((self.input_data.size_placeholder[0] - self._reference_frame_size) -
         (nr_of_full_frames - 1) * self._frame_shift > 0)):
        nf_of_paded_frames = 1
      new_size = nr_of_full_frames + nf_of_paded_frames
      from ..util.data import DimensionTag
      DimensionTag(
        kind=DimensionTag.Types.Spatial, description="MultiChannelMultiResolutionStft",
        dyn_size=new_size, batch=self.output.batch,
        src_data=self.output, src_axis=self.output.get_batch_axis(0))
      size_placeholder_dict[0] = new_size
      return size_placeholder_dict

    import numpy as np
    n_out = np.sum([self._get_n_out_by_fft_config(fft_size, use_rfft, nr_of_channels) for fft_size in fft_sizes])
    if 'n_out' in kwargs and (kwargs['n_out'] != n_out):
      raise Exception('argument n_out of layer MultiChannelStftLayer does not match the fft configuration')
    kwargs['n_out'] = n_out
    super(MultiChannelMultiResolutionStftLayer, self).__init__(**kwargs)
    tf_compat.v1.assert_equal(nr_of_channels, self._get_nr_of_channels_from_input_placeholder())
    self._nr_of_channels = nr_of_channels
    self._frame_shift = frame_shift
    self._frame_sizes = frame_sizes
    self._reference_frame_size = frame_sizes[0]
    self._fft_sizes = fft_sizes
    self._window = window
    self._use_rfft = use_rfft
    self._pad_last_frame = pad_last_frame
    self.output.placeholder = self._apply_stft_to_input()
    self.output.size_placeholder = _compute_size_placeholder()

  def _get_nr_of_channels_from_input_placeholder(self):
    input_placeholder = self.input_data.get_placeholder_as_batch_major()
    return input_placeholder.shape[2]

  def _apply_stft_to_input(self):
    from returnn.tf.util.basic import get_shape

    # noinspection PyShadowingNames
    def _crop_stft_output_to_reference_frame_size_length(channel_concatenated_stft, crop_size):
      return tf.slice(
        channel_concatenated_stft, [0, 0, 0],
        [get_shape(channel_concatenated_stft)[0], crop_size, get_shape(channel_concatenated_stft)[2]])

    input_placeholder = self.input_data.get_placeholder_as_batch_major()
    channel_wise_stft_res_list = list()
    for fft_size, frame_size in zip(self._fft_sizes, self._frame_sizes):
      def _get_window(window_length, dtype):
        if self._window == "hanning":
          try:
            # noinspection PyPackageRequirements
            from tensorflow.signal import hann_window
          except ImportError:
            # noinspection PyPackageRequirements,PyUnresolvedReferences
            from tensorflow.contrib.signal import hann_window
          window = hann_window(window_length, dtype=dtype)
        elif self._window == "blackman":
          # noinspection PyPackageRequirements
          import scipy.signal
          window = tf.constant(scipy.signal.windows.blackman(frame_size), dtype=tf.float32)
        elif self._window == "None" or self._window == "ones":
          window = tf.ones((window_length,), dtype=dtype)
        else:
          assert False, "Window was not parsed correctly: {}".format(self._window)
        return window

      # noinspection PyShadowingNames
      def _pad_time_signal(input_placeholder, frame_size):
        if frame_size > self._reference_frame_size:
          return tf.concat(
            [input_signal, tf.ones(
              [get_shape(input_signal)[0], frame_size - self._reference_frame_size,
               get_shape(input_signal)[2]]) * 1e-7], axis=1)
        else:
          return input_placeholder

      input_signal = _pad_time_signal(input_placeholder, frame_size)
      if self._use_rfft:
        try:
          # noinspection PyPackageRequirements
          from tensorflow.signal import stft
        except ImportError:
          # noinspection PyPackageRequirements,PyUnresolvedReferences
          from tensorflow.contrib.signal import stft
        channel_wise_stft = stft(
          signals=tf.transpose(input_signal, [0, 2, 1]),
          frame_length=frame_size,
          frame_step=self._frame_shift,
          fft_length=fft_size,
          window_fn=_get_window,
          pad_end=self._pad_last_frame)
        channel_wise_stft = tf.transpose(channel_wise_stft, [0, 2, 1, 3])
        batch_dim = tf.shape(channel_wise_stft)[0]
        time_dim = tf.shape(channel_wise_stft)[1]
        concat_feature_dim = channel_wise_stft.shape[2] * channel_wise_stft.shape[3]
        channel_concatenated_stft = tf.reshape(channel_wise_stft, (batch_dim, time_dim, concat_feature_dim))
        if channel_wise_stft_res_list:
          channel_concatenated_stft = (
            _crop_stft_output_to_reference_frame_size_length(
              channel_concatenated_stft,
              get_shape(channel_wise_stft_res_list[0])[1]))
        channel_wise_stft_res_list.append(channel_concatenated_stft)
    output_placeholder = tf.concat(channel_wise_stft_res_list, axis=2)
    return output_placeholder

  @classmethod
  def _get_n_out_by_fft_config(cls, fft_size, use_rfft, nr_of_channels):
    n_out = fft_size
    if use_rfft:
      n_out = fft_size // 2 + 1
    n_out *= nr_of_channels
    return n_out

  @classmethod
  def get_out_data_from_opts(cls, fft_sizes, use_rfft=True, nr_of_channels=1, **kwargs):
    """
    :param list[int] fft_sizes:
    :param bool use_rfft:
    :param int nr_of_channels:
    :rtype: Data
    """
    import numpy as np
    n_out = np.sum([cls._get_n_out_by_fft_config(fft_size, use_rfft, nr_of_channels) for fft_size in fft_sizes])
    if 'n_out' not in kwargs:
      kwargs['n_out'] = n_out
    return (super(MultiChannelMultiResolutionStftLayer, cls)
            .get_out_data_from_opts(**kwargs)
            .copy_template(dtype="complex64"))


class MultiChannelStftLayer(MultiChannelMultiResolutionStftLayer):
  """
  The layer applys a STFT to every channel separately and concatenates the frequency domain vectors for every frame.
  """
  recurrent = True
  layer_class = "multichannel_stft_layer"

  def __init__(self, frame_shift, frame_size, fft_size, window="hanning", use_rfft=True, nr_of_channels=1,
               pad_last_frame=False, **kwargs):
    kwargs['frame_shift'] = frame_shift
    kwargs['window'] = window
    kwargs['use_rfft'] = use_rfft
    kwargs['nr_of_channels'] = nr_of_channels
    kwargs['pad_last_frame'] = pad_last_frame
    super(MultiChannelStftLayer, self).__init__(frame_sizes=[frame_size], fft_sizes=[fft_size], **kwargs)

  @classmethod
  def get_out_data_from_opts(cls, fft_size, use_rfft=True, nr_of_channels=1, **kwargs):
    """
    :param int fft_size:
    :param bool use_rfft:
    :param int nr_of_channels:
    :rtype: Data
    """
    return (super(MultiChannelStftLayer, cls)
            .get_out_data_from_opts(fft_sizes=[fft_size], use_rfft=use_rfft, nr_of_channels=nr_of_channels, **kwargs)
            .copy_template(dtype="complex64"))


class NoiseEstimationByFirstTFramesLayer(_ConcatInputLayer):
  """
  Estimates noise from the first t time frames.
  """
  layer_class = "first_t_frames_noise_estimator"
  recurrent = True

  def __init__(self, nr_of_frames, **kwargs):
    """
    :param int nr_of_frames: first nr_of_frames frames are used for averaging
                             all frames are used if nr_of_frames is -1
    """
    super(NoiseEstimationByFirstTFramesLayer, self).__init__(**kwargs)
    self._nr_of_frames = nr_of_frames
    noise_vector = self._get_noise_vector()
    self.output.placeholder = tf.tile(
      noise_vector, (1, tf.shape(self.input_data.get_placeholder_as_batch_major())[1], 1))

  def _get_noise_vector(self):
    input_placeholder = self.input_data.get_placeholder_as_batch_major()
    if self._nr_of_frames != -1:
      noise_vector = tf.reduce_mean(input_placeholder[:, :self._nr_of_frames, :], axis=1, keepdims=True)
    else:
      noise_vector = tf.reduce_mean(input_placeholder, axis=1, keepdims=True)
    return noise_vector


class ParametricWienerFilterLayer(LayerBase):
  """
  Parametric Wiener Filter
  For related paper, see:
  Menne, Tobias, Ralf Schlueter, and Hermann Ney.
  "Investigation into joint optimization of single channel speech enhancement and acoustic modeling for robust ASR."
  IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
  IEEE, 2019.
  """
  layer_class = "parametric_wiener_filter"

  def __init__(self, l_overwrite=None, p_overwrite=None, q_overwrite=None, filter_input=None, parameters=None,
               noise_estimation=None, average_parameters=False, **kwargs):
    """
    :param float|None l_overwrite: if given overwrites the l value of the parametric wiener filter with given constant
    :param float|None p_overwrite: if given overwrites the p value of the parametric wiener filter with given constant
    :param float|None q_overwrite: if given overwrites the q value of the parametric wiener filter with given constant
    :param LayerBase|None filter_input: name of layer containing input for wiener filter
    :param LayerBase|None parameters: name of layer containing parameters for wiener filter
    :param LayerBase|None noise_estimation: name of layer containing noise estimate for wiener filter
    :param bool average_parameters: if set to true the parameters l, p and q are averaged over the time axis
    """
    # noinspection PyUnresolvedReferences
    from tfSi6Proc.audioProcessing.enhancement.singleChannel import TfParametricWienerFilter
    super(ParametricWienerFilterLayer, self).__init__(**kwargs)

    class _NoiseEstimator(object):
      """
      Noise estimator. For details, see tfSi6Proc.audioProcessing.enhancement.singleChannel
      """
      def __init__(self, noise_power_spectrum_tensor):
        self._noise_power_spectrum_tensor = noise_power_spectrum_tensor

      @classmethod
      def from_layer(cls, layer):
        """
        :param LayerBase layer:
        :rtype: _NoiseEstimator
        """
        return cls(layer.output.get_placeholder_as_batch_major())

      def get_noise_power_spectrum(self):
        """
        :rtype: tf.Tensor
        """
        return self._noise_power_spectrum_tensor

    # noinspection PyShadowingNames
    def _get_parameters_from_constructor_inputs(parameters, l_overwrite, p_overwrite, q_overwrite, average_parameters):
      parameter_vector = None
      if parameters is not None:
        parameter_vector = parameters.output.get_placeholder_as_batch_major()
        tf_compat.v1.assert_equal(parameter_vector.shape[-1], 3)
      if (l_overwrite is None) or (p_overwrite is None) or (q_overwrite is None):
        assert parameter_vector is not None
        if average_parameters:
          parameter_vector = tf.tile(
            tf.reduce_mean(parameter_vector, axis=1, keepdims=True), [1, tf.shape(parameter_vector)[1], 1])
      if l_overwrite is not None:
        l_ = tf.constant(l_overwrite, dtype=tf.float32)
      else:
        l_ = tf.expand_dims(parameter_vector[:, :, 0], axis=-1)
      if p_overwrite is not None:
        p_ = tf.constant(p_overwrite, dtype=tf.float32)
      else:
        p_ = tf.expand_dims(parameter_vector[:, :, 1], axis=-1)
      if q_overwrite is not None:
        q_ = tf.constant(q_overwrite, dtype=tf.float32)
      else:
        q_ = tf.expand_dims(parameter_vector[:, :, 2], axis=-1)
      return l_, p_, q_

    filter_input_placeholder = filter_input.output.get_placeholder_as_batch_major()
    if filter_input_placeholder.dtype != tf.complex64:
      filter_input_placeholder = tf.cast(filter_input_placeholder, dtype=tf.complex64)
    ne = _NoiseEstimator.from_layer(noise_estimation)
    l, p, q = _get_parameters_from_constructor_inputs(parameters, l_overwrite, p_overwrite, q_overwrite,
                                                      average_parameters)
    wiener = TfParametricWienerFilter(ne, [], l, p, q, inputTensorFreqDomain=filter_input_placeholder)
    self.output.placeholder = wiener.getFrequencyDomainOutputSignal()

  @classmethod
  def get_out_data_from_opts(cls, **kwargs):
    """
    :rtype: Data
    """
    kwargs_with_sources = kwargs
    if ("sources" in kwargs_with_sources) and (len(kwargs_with_sources["sources"]) == 0):
      kwargs_with_sources["sources"] = [kwargs["filter_input"]]
    return cls._base_get_out_data_from_opts(**kwargs_with_sources)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer:
    """
    if "from" in d and len(d["from"]) > 0:
      # This if block is kept for backwards compatibility only and should not be used
      assert ("filter_input" not in d) and ("parameters" not in d) and ("noise_estimation" not in d)
      if len(d["from"]) == 2:
        d["filter_input"] = d["from"][0]
        d["parameters"] = None
        d["noise_estimation"] = d["from"][1]
      if len(d["from"]) == 3:
        d["filter_input"] = d["from"][0]
        d["parameters"] = d["from"][1]
        d["noise_estimation"] = d["from"][2]
    d.setdefault("from", [])
    super(ParametricWienerFilterLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["filter_input"] = get_layer(d["filter_input"])
    if d["parameters"] is not None:
      d["parameters"] = get_layer(d["parameters"])
    d["noise_estimation"] = get_layer(d["noise_estimation"])


class SignalMaskingLayer(LayerBase):
  """
  Mask a given signal using a given mask.
  """
  layer_class = "signal_masking"

  def __init__(self, signal, mask, **kwargs):
    """
    :param LayerBase signal: name of layer the signal to be masked
    :param LayerBase mask: name of layer containing the mask
    """

    # noinspection PyShadowingNames
    def _cast_signal_and_mask_if_iecessary(signal, mask):
      if signal.dtype != mask.dtype:
        if signal.dtype == tf.complex64 and mask.dtype == tf.float32:
          return signal, tf.cast(mask, dtype=tf.complex64)
        else:
          raise NotImplementedError('difference in dtype between mask and signal is not supported yet.')
      return signal, mask

    super(SignalMaskingLayer, self).__init__(**kwargs)
    self._signal = signal.output.get_placeholder_as_batch_major()
    self._mask = mask.output.get_placeholder_as_batch_major()
    self._signal, self._mask = _cast_signal_and_mask_if_iecessary(self._signal, self._mask)
    self.output.placeholder = tf.multiply(self._signal, self._mask)
    self.output.size_placeholder = signal.output.size_placeholder

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer:
    """
    d.setdefault("from", [])
    super(SignalMaskingLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["signal"] = get_layer(d["signal"])
    d["mask"] = get_layer(d["mask"])


class SplitConcatMultiChannel(_ConcatInputLayer):
  """
  This layer assumes the feature vector to be a concatenation of features of multiple channels (of the same size).
  It splits the feature dimension into equisized number of channel features and stacks them in the batch dimension.
  Thus the batch size is multiplied with the number of channels and the feature size is divided by the number of
  channels.
  The channels of one singal will have consecutive batch indices, meaning the signal of the original batch index n is
  split and can now be found in batch indices (n * nr_of_channels) to ((n+1) * nr_of_channels - 1)
  """

  layer_class = "split_concatenated_multichannel"

  def __init__(self, nr_of_channels=1, **kwargs):
    """
    :param int nr_of_channels: the number of concatenated channels in the feature dimension
    """
    super(SplitConcatMultiChannel, self).__init__(**kwargs)

    input_placeholder = self.input_data.get_placeholder_as_batch_major()

    # For shape comments: C = nr_of_channels
    output = tf.reshape(input_placeholder, [
      tf.shape(input_placeholder)[0], tf.shape(input_placeholder)[1], nr_of_channels,
      tf.shape(input_placeholder)[2] // nr_of_channels])  # shape B x T x C x (F / C)
    new_shape = (tf.shape(output)[1], tf.shape(output)[3], tf.shape(output)[0] * tf.shape(output)[2])
    output = tf.transpose(output, [1, 3, 0, 2])  # shape T x (F / C) x B x C
    output = tf.reshape(output, new_shape)  # shape T x (F / C) x (B * C)
    self.output.placeholder = tf.transpose(output, [2, 0, 1])  # shape (B * C) x T x (F / C)
    # work around to obtain result like numpy.repeat(size_placeholder, nr_of_channels)
    size_placeholder_np_repeat = tf.reshape(
      self.input_data.size_placeholder[self.input_data.time_dim_axis_excluding_batch], [-1, 1])  # shape T x 1
    size_placeholder_np_repeat = tf.tile(size_placeholder_np_repeat, [1, nr_of_channels])  # shape T x C
    size_placeholder_np_repeat = tf.reshape(size_placeholder_np_repeat, [-1])  # shape (T * C)
    self.output.size_placeholder = {self.output.time_dim_axis_excluding_batch: size_placeholder_np_repeat}

  @classmethod
  def get_out_data_from_opts(cls, name, sources, nr_of_channels, n_out=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int nr_of_channels:
    :param int|None|returnn.util.basic.NotSpecified n_out:
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources).copy_as_batch_major()
    assert not input_data.sparse
    return Data(
      name="%s_output" % name,
      shape=[input_data.batch_shape[1], input_data.batch_shape[2] // nr_of_channels],
      dtype=input_data.dtype,
      sparse=False,
      batch_dim_axis=0,
      time_dim_axis=1)


class TileFeaturesLayer(_ConcatInputLayer):
  """
  This function is tiling features with given number of repetitions.
  """

  layer_class = "tile_features"

  def __init__(self, repetitions=1, **kwargs):
    """
    :param int repetitions: number of tiling repetitions in feature domain
    """
    super(TileFeaturesLayer, self).__init__(**kwargs)

    input_placeholder = self.input_data.get_placeholder_as_batch_major()

    self.output.placeholder = tf.tile(input_placeholder, [1, 1, repetitions])

  @classmethod
  def get_out_data_from_opts(cls, name, sources, repetitions, n_out=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int repetitions:
    :param int|None|returnn.util.basic.NotSpecified n_out:
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources)
    assert not input_data.sparse
    return Data(
      name="%s_output" % name,
      shape=[input_data.shape[1], input_data.shape[2] * repetitions],
      dtype=input_data.dtype,
      sparse=False,
      size_placeholder={0: input_data.size_placeholder[input_data.time_dim_axis_excluding_batch]},
      batch_dim_axis=0,
      time_dim_axis=1)
