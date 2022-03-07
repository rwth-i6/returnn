"""
ExtractAudioFeatures class and related helpers
"""

import numpy

from returnn.util.basic import CollectionReadCheckCovered


class ExtractAudioFeatures:
  """
  Currently uses librosa to extract MFCC/log-mel features.
  (Alternatives: python_speech_features, talkbox.features.mfcc, librosa)
  """

  def __init__(self,
               window_len=0.025, step_len=0.010,
               num_feature_filters=None, with_delta=False,
               norm_mean=None, norm_std_dev=None,
               features="mfcc", feature_options=None, random_permute=None, random_state=None, raw_ogg_opts=None,
               pre_process=None, post_process=None,
               sample_rate=None, num_channels=None,
               peak_normalization=True, preemphasis=None, join_frames=None):
    """
    :param float window_len: in seconds
    :param float step_len: in seconds
    :param int num_feature_filters:
    :param bool|int with_delta:
    :param numpy.ndarray|str|int|float|None norm_mean: if str, will interpret as filename, or "per_seq"
    :param numpy.ndarray|str|int|float|None norm_std_dev: if str, will interpret as filename, or "per_seq"
    :param str|function features: "mfcc", "log_mel_filterbank", "log_log_mel_filterbank", "raw", "raw_ogg"
    :param dict[str]|None feature_options: provide additional parameters for the feature function
    :param CollectionReadCheckCovered|dict[str]|bool|None random_permute:
    :param numpy.random.RandomState|None random_state:
    :param dict[str]|None raw_ogg_opts:
    :param function|None pre_process:
    :param function|None post_process:
    :param int|None sample_rate:
    :param int|None num_channels: number of channels in audio
    :param bool peak_normalization: set to False to disable the peak normalization for audio files
    :param float|None preemphasis: set a preemphasis filter coefficient
    :param int|None join_frames: concatenate multiple frames together to a superframe
    :return: float32 data of shape
    (audio_len // int(step_len * sample_rate), num_channels (optional), (with_delta + 1) * num_feature_filters)
    :rtype: numpy.ndarray
    """
    self.window_len = window_len
    self.step_len = step_len
    if num_feature_filters is None:
      if features == "raw":
        num_feature_filters = 1
      elif features == "raw_ogg":
        raise Exception("you should explicitly specify num_feature_filters (dimension) for raw_ogg")
      else:
        num_feature_filters = 40  # was the old default
    self.num_feature_filters = num_feature_filters
    self.preemphasis = preemphasis
    if isinstance(with_delta, bool):
      with_delta = 1 if with_delta else 0
    assert isinstance(with_delta, int) and with_delta >= 0
    self.with_delta = with_delta
    # join frames needs to be set before norm loading
    self.join_frames = join_frames
    if norm_mean is not None:
      if not isinstance(norm_mean, (int, float)):
        norm_mean = self._load_feature_vec(norm_mean)
    if norm_std_dev is not None:
      if not isinstance(norm_std_dev, (int, float)):
        norm_std_dev = self._load_feature_vec(norm_std_dev)
    self.norm_mean = norm_mean
    self.norm_std_dev = norm_std_dev
    if random_permute and not isinstance(random_permute, CollectionReadCheckCovered):
      random_permute = CollectionReadCheckCovered.from_bool_or_dict(random_permute)
    self.random_permute_opts = random_permute
    self.random_state = random_state
    self.features = features
    self.feature_options = feature_options
    self.pre_process = pre_process
    self.post_process = post_process
    self.sample_rate = sample_rate
    if num_channels is not None:
      assert self.features == "raw", "Currently, multiple channels are only supported for raw waveforms"
      self.num_dim = 3
    else:
      self.num_dim = 2
    self.num_channels = num_channels
    self.raw_ogg_opts = raw_ogg_opts
    self.peak_normalization = peak_normalization

  def _load_feature_vec(self, value):
    """
    :param str|None value:
    :return: shape (self.num_inputs,), float32
    :rtype: numpy.ndarray|str|None
    """
    if value is None:
      return None
    if isinstance(value, str):
      if value == "per_seq":
        return value
      value = numpy.loadtxt(value)
    assert isinstance(value, numpy.ndarray)
    assert value.shape == (self.get_feature_dimension(),)
    return value.astype("float32")

  def get_audio_features_from_raw_bytes(self, raw_bytes, seq_name=None):
    """
    :param io.BytesIO raw_bytes:
    :param str|None seq_name:
    :return: shape (time,feature_dim)
    :rtype: numpy.ndarray
    """
    if self.features == "raw_ogg":
      assert self.with_delta == 0 and self.norm_mean is None and self.norm_std_dev is None
      # We expect that raw_bytes comes from a Ogg file.
      try:
        from returnn.extern.ParseOggVorbis.returnn_import import ParseOggVorbisLib
      except ImportError:
        print("Maybe you did not clone the submodule extern/ParseOggVorbis?")
        raise
      return ParseOggVorbisLib.get_instance().get_features_from_raw_bytes(
        raw_bytes=raw_bytes.getvalue(), output_dim=self.num_feature_filters, **(self.raw_ogg_opts or {}))

    # Don't use librosa.load which internally uses audioread which would use Gstreamer as a backend,
    # which has multiple issues:
    # https://github.com/beetbox/audioread/issues/62
    # https://github.com/beetbox/audioread/issues/63
    # Instead, use PySoundFile, which is also faster. See here for discussions:
    # https://github.com/beetbox/audioread/issues/64
    # https://github.com/librosa/librosa/issues/681
    import soundfile  # noqa  # pip install pysoundfile
    # integer audio formats are automatically transformed in the range [-1,1]
    audio, sample_rate = soundfile.read(raw_bytes)
    return self.get_audio_features(audio=audio, sample_rate=sample_rate, seq_name=seq_name)

  def get_audio_features(self, audio, sample_rate, seq_name=None):
    """
    :param numpy.ndarray audio: raw audio samples, shape (audio_len,)
    :param int sample_rate: e.g. 22050
    :param str|None seq_name:
    :return: array (time,dim), dim == self.get_feature_dimension()
    :rtype: numpy.ndarray
    """
    if self.sample_rate is not None:
      assert sample_rate == self.sample_rate, "currently no conversion implemented..."

    if self.preemphasis:
      from scipy import signal  # noqa
      audio = signal.lfilter([1, -self.preemphasis], [1], audio)

    if self.peak_normalization:
      peak = numpy.max(numpy.abs(audio))
      if peak != 0.0:
        audio /= peak

    if self.random_permute_opts and self.random_permute_opts.truth_value:
      audio = _get_random_permuted_audio(
        audio=audio,
        sample_rate=sample_rate,
        opts=self.random_permute_opts,
        random_state=self.random_state)

    if self.pre_process:
      audio = self.pre_process(audio=audio, sample_rate=sample_rate, random_state=self.random_state)
      assert isinstance(audio, numpy.ndarray) and len(audio.shape) == 1

    if self.features == "raw":
      assert self.num_feature_filters == 1
      if audio.ndim == 1:
        audio = numpy.expand_dims(audio, axis=1)  # add dummy feature axis
      if self.num_channels is not None:
        if audio.ndim == 2:
          audio = numpy.expand_dims(audio, axis=2)  # add dummy feature axis
        assert audio.shape[1] == self.num_channels
        assert audio.ndim == 3  # time, channel, feature
      feature_data = audio.astype("float32")

    else:
      kwargs = {
        "sample_rate": sample_rate,
        "window_len": self.window_len,
        "step_len": self.step_len,
        "num_feature_filters": self.num_feature_filters,
        "audio": audio}

      if self.feature_options is not None:
        assert isinstance(self.feature_options, dict)
        kwargs.update(self.feature_options)

      if callable(self.features):
        feature_data = self.features(random_state=self.random_state, **kwargs)
      elif self.features == "mfcc":
        feature_data = _get_audio_features_mfcc(**kwargs)
      elif self.features == "log_mel_filterbank":
        feature_data = _get_audio_log_mel_filterbank(**kwargs)
      elif self.features == "log_log_mel_filterbank":
        feature_data = _get_audio_log_log_mel_filterbank(**kwargs)
      elif self.features == "db_mel_filterbank":
        feature_data = _get_audio_db_mel_filterbank(**kwargs)
      elif self.features == "linear_spectrogram":
        feature_data = _get_audio_linear_spectrogram(**kwargs)
      else:
        raise Exception("non-supported feature type %r" % (self.features,))

    assert feature_data.ndim == self.num_dim, "got feature data shape %r" % (feature_data.shape,)
    assert feature_data.shape[-1] == self.num_feature_filters

    if self.with_delta:
      import librosa  # noqa
      deltas = [librosa.feature.delta(feature_data, order=i, axis=0).astype("float32")
                for i in range(1, self.with_delta + 1)]
      feature_data = numpy.concatenate([feature_data] + deltas, axis=-1)
      assert feature_data.shape[1] == (self.with_delta + 1) * self.num_feature_filters

    if self.norm_mean is not None:
      if isinstance(self.norm_mean, str) and self.norm_mean == "per_seq":
        feature_data -= numpy.mean(feature_data, axis=0, keepdims=True)
      elif isinstance(self.norm_mean, (int, float)):
        feature_data -= self.norm_mean
      else:
        if self.num_dim == 2:
          feature_data -= self.norm_mean[numpy.newaxis, :]
        elif self.num_dim == 3:
          feature_data -= self.norm_mean[numpy.newaxis, numpy.newaxis, :]
        else:
          assert False, "Unexpected number of dimensions: {}".format(self.num_dim)

    if self.norm_std_dev is not None:
      if isinstance(self.norm_std_dev, str) and self.norm_std_dev == "per_seq":
        feature_data /= numpy.maximum(numpy.std(feature_data, axis=0, keepdims=True), 1e-2)
      elif isinstance(self.norm_std_dev, (int, float)):
        feature_data /= self.norm_std_dev
      else:
        if self.num_dim == 2:
          feature_data /= self.norm_std_dev[numpy.newaxis, :]
        elif self.num_dim == 3:
          feature_data /= self.norm_std_dev[numpy.newaxis, numpy.newaxis, :]
        else:
          assert False, "Unexpected number of dimensions: {}".format(self.num_dim)

    if self.join_frames is not None:
      pad_len = self.join_frames - (feature_data.shape[0] % self.join_frames)
      pad_len = pad_len % self.join_frames
      new_len = feature_data.shape[0] + pad_len
      if self.num_channels is None:
        new_shape = (new_len // self.join_frames, feature_data.shape[-1] * self.join_frames)
        pad_width = ((0, pad_len), (0, 0))
      else:
        new_shape = (new_len // self.join_frames, self.num_channels, feature_data.shape[-1] * self.join_frames)
        pad_width = ((0, pad_len), (0, 0), (0, 0))
      feature_data = numpy.pad(feature_data, pad_width=pad_width, mode="edge")
      feature_data = numpy.reshape(feature_data, newshape=new_shape, order='C')

    assert feature_data.shape[-1] == self.get_feature_dimension()
    if self.post_process:
      feature_data = self.post_process(feature_data, seq_name=seq_name)
      assert isinstance(feature_data, numpy.ndarray) and feature_data.ndim == self.num_dim
      assert feature_data.shape[-1] == self.get_feature_dimension()
    return feature_data

  def get_feature_dimension(self):
    """
    :rtype: int
    """
    return (self.with_delta + 1) * self.num_feature_filters * (self.join_frames or 1)


def _get_audio_linear_spectrogram(audio, sample_rate, window_len=0.025, step_len=0.010, num_feature_filters=512):
  """
  Computes linear spectrogram features from an audio signal.
  Drops the DC component.

  :param numpy.ndarray audio: raw audio samples, shape (audio_len,)
  :param int sample_rate: e.g. 22050
  :param float window_len: in seconds
  :param float step_len: in seconds
  :return: (audio_len // int(step_len * sample_rate), num_feature_filters), float32
  :rtype: numpy.ndarray
  """
  import librosa  # noqa

  min_n_fft = int(window_len * sample_rate)
  assert num_feature_filters*2 >= min_n_fft
  assert num_feature_filters % 2 == 0

  librosa_version = librosa.__version__.split(".")
  if int(librosa_version[0]) >= 1 or (int(librosa_version[0]) == 0 and int(librosa_version[1]) >= 9):
    stft_func = librosa.stft
  else:
    stft_func = librosa.core.stft
  spectrogram = numpy.abs(stft_func(
    audio, hop_length=int(step_len * sample_rate),
    win_length=int(window_len * sample_rate), n_fft=num_feature_filters*2))

  # remove the DC part
  spectrogram = spectrogram[1:]

  assert spectrogram.shape[0] == num_feature_filters
  spectrogram = spectrogram.transpose().astype("float32")  # (time, dim)
  return spectrogram


def _get_audio_features_mfcc(audio, sample_rate, window_len=0.025, step_len=0.010, num_feature_filters=40,
                             n_mels=128, fmin=0, fmax=None):
  """
  :param numpy.ndarray audio: raw audio samples, shape (audio_len,)
  :param int sample_rate: e.g. 22050
  :param float window_len: in seconds
  :param float step_len: in seconds
  :param int num_feature_filters: number of dct outputs to use
  :param int n_mels: number of mel filters
  :param int fmin: minimum frequency for mel filters
  :param int|None fmax: maximum frequency for mel filters (None -> use sample_rate/2)
  :return: (audio_len // int(step_len * sample_rate), num_feature_filters), float32
  :rtype: numpy.ndarray
  """
  import librosa  # noqa
  features = librosa.feature.mfcc(
    y=audio, sr=sample_rate,
    n_mfcc=num_feature_filters,
    n_mels=n_mels, fmin=fmin, fmax=fmax,
    hop_length=int(step_len * sample_rate), n_fft=int(window_len * sample_rate))
  librosa_version = librosa.__version__.split(".")
  if int(librosa_version[0]) >= 1 or (int(librosa_version[0]) == 0 and int(librosa_version[1]) >= 7):
    rms_func = librosa.feature.rms
  else:
    rms_func = librosa.feature.rmse  # noqa
  energy = rms_func(
    y=audio,
    hop_length=int(step_len * sample_rate), frame_length=int(window_len * sample_rate))
  features[0] = energy  # replace first MFCC with energy, per convention
  assert features.shape[0] == num_feature_filters  # (dim, time)
  features = features.transpose().astype("float32")  # (time, dim)
  return features


def _get_audio_log_mel_filterbank(audio, sample_rate, window_len=0.025, step_len=0.010, num_feature_filters=80):
  """
  Computes log Mel-filterbank features from an audio signal.
  References:

    https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/speech_recognition.py

  :param numpy.ndarray audio: raw audio samples, shape (audio_len,)
  :param int sample_rate: e.g. 22050
  :param float window_len: in seconds
  :param float step_len: in seconds
  :param int num_feature_filters:
  :return: (audio_len // int(step_len * sample_rate), num_feature_filters), float32
  :rtype: numpy.ndarray
  """
  import librosa  # noqa
  mel_filterbank = librosa.feature.melspectrogram(
    y=audio, sr=sample_rate,
    n_mels=num_feature_filters,
    hop_length=int(step_len * sample_rate), n_fft=int(window_len * sample_rate))
  log_noise_floor = 1e-3  # prevent numeric overflow in log
  log_mel_filterbank = numpy.log(numpy.maximum(log_noise_floor, mel_filterbank))
  assert log_mel_filterbank.shape[0] == num_feature_filters
  log_mel_filterbank = log_mel_filterbank.transpose().astype("float32")  # (time, dim)
  return log_mel_filterbank


def _get_audio_db_mel_filterbank(audio, sample_rate,
                                 window_len=0.025, step_len=0.010, num_feature_filters=80,
                                 fmin=0, fmax=None, min_amp=1e-10, center=True):
  """
  Computes log Mel-filterbank features in dezibel values from an audio signal.
  Provides adjustable minimum frequency and minimual amplitude clipping

  :param numpy.ndarray audio: raw audio samples, shape (audio_len,)
  :param int sample_rate: e.g. 22050
  :param float window_len: in seconds
  :param float step_len: in seconds
  :param int num_feature_filters: number of mel-filterbanks
  :param int fmin: minimum frequency covered by mel filters
  :param int|None fmax: maximum frequency covered by mel filters
  :param int min_amp: silence clipping for small amplitudes
  :param bool center: pads the signal with reflection so that the window center starts at 0.
  :return: (audio_len // int(step_len * sample_rate), num_feature_filters), float32
  :rtype: numpy.ndarray
  """
  # noinspection PyPackageRequirements
  assert fmin >= 0
  assert min_amp > 0

  import librosa  # noqa
  mel_filterbank = librosa.feature.melspectrogram(
    y=audio, sr=sample_rate,
    n_mels=num_feature_filters,
    hop_length=int(step_len * sample_rate),
    n_fft=int(window_len * sample_rate),
    fmin=fmin, fmax=fmax, center=center
   )

  log_mel_filterbank = 20 * numpy.log10(numpy.maximum(min_amp, mel_filterbank))
  assert log_mel_filterbank.shape[0] == num_feature_filters
  log_mel_filterbank = log_mel_filterbank.transpose().astype("float32")  # (time, dim)
  return log_mel_filterbank


def _get_audio_log_log_mel_filterbank(audio, sample_rate, window_len=0.025, step_len=0.010, num_feature_filters=80):
  """
  Computes log-log Mel-filterbank features from an audio signal.
  References:

    https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/speech_recognition.py

  :param numpy.ndarray audio: raw audio samples, shape (audio_len,)
  :param int sample_rate: e.g. 22050
  :param float window_len: in seconds
  :param float step_len: in seconds
  :param int num_feature_filters:
  :return: (audio_len // int(step_len * sample_rate), num_feature_filters), float32
  :rtype: numpy.ndarray
  """
  import librosa  # noqa
  librosa_version = librosa.__version__.split(".")
  if int(librosa_version[0]) >= 1 or (int(librosa_version[0]) == 0 and int(librosa_version[1]) >= 9):
    db_func = librosa.amplitude_to_db
  else:
    db_func = librosa.core.amplitude_to_db
  mel_filterbank = librosa.feature.melspectrogram(
    y=audio, sr=sample_rate,
    n_mels=num_feature_filters,
    hop_length=int(step_len * sample_rate), n_fft=int(window_len * sample_rate))
  log_noise_floor = 1e-3  # prevent numeric overflow in log
  log_mel_filterbank = numpy.log(numpy.maximum(log_noise_floor, mel_filterbank))
  log_log_mel_filterbank = db_func(log_mel_filterbank)
  assert log_log_mel_filterbank.shape[0] == num_feature_filters
  log_log_mel_filterbank = log_log_mel_filterbank.transpose().astype("float32")  # (time, dim)
  return log_log_mel_filterbank


def _get_random_permuted_audio(audio, sample_rate, opts, random_state):
  """
  :param numpy.ndarray audio: raw time signal
  :param int sample_rate:
  :param CollectionReadCheckCovered opts:
  :param numpy.random.RandomState random_state:
  :return: audio randomly permuted
  :rtype: numpy.ndarray
  """
  import librosa  # noqa
  import scipy.ndimage  # noqa
  import warnings
  audio = audio * random_state.uniform(opts.get("rnd_scale_lower", 0.8), opts.get("rnd_scale_upper", 1.0))
  if opts.get("rnd_zoom_switch", 1.) > 0.:
    opts.get("rnd_zoom_lower"), opts.get("rnd_zoom_upper"), opts.get("rnd_zoom_order")  # Mark as read.
  if random_state.uniform(0.0, 1.0) < opts.get("rnd_zoom_switch", 0.2):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      # Alternative: scipy.interpolate.interp2d
      factor = random_state.uniform(opts.get("rnd_zoom_lower", 0.9), opts.get("rnd_zoom_upper", 1.1))
      audio = scipy.ndimage.zoom(audio, factor, order=opts.get("rnd_zoom_order", 3))
  if opts.get("rnd_stretch_switch", 1.) > 0.:
    opts.get("rnd_stretch_lower"), opts.get("rnd_stretch_upper")  # Mark as read.
  if random_state.uniform(0.0, 1.0) < opts.get("rnd_stretch_switch", 0.2):
    rate = random_state.uniform(opts.get("rnd_stretch_lower", 0.9), opts.get("rnd_stretch_upper", 1.2))
    audio = librosa.effects.time_stretch(audio, rate=rate)
  if opts.get("rnd_pitch_switch", 1.) > 0.:
    opts.get("rnd_pitch_lower"), opts.get("rnd_pitch_upper", 1.)  # Mark as read.
  if random_state.uniform(0.0, 1.0) < opts.get("rnd_pitch_switch", 0.2):
    n_steps = random_state.uniform(opts.get("rnd_pitch_lower", -1.), opts.get("rnd_pitch_upper", 1.))
    audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
  opts.assert_all_read()
  return audio
