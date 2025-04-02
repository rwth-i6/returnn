"""
Mel filterbank.
"""


from __future__ import annotations
from typing import Optional, Union, Tuple
import functools
import math
import numpy
from returnn.util import math as util_math
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


__all__ = ["mel_filterbank", "log_mel_filterbank_from_raw"]


def mel_filterbank(
    x: Tensor,
    *,
    in_dim: Dim,
    out_dim: Dim,
    sampling_rate: Union[int, float],
    fft_length: Optional[int] = None,
    f_min: Optional[Union[int, float]] = None,
    f_max: Optional[Union[int, float]] = None,
):
    """
    Applies the Mel filterbank to the input.

    :param x:
    :param in_dim: expected to be fft_length // 2 + 1. E.g. via :func:`stft`.
    :param out_dim: nr of mel filters.
    :param sampling_rate:
    :param fft_length: fft_size, n_fft. Should match fft_length from :func:`stft`.
        If not given, infer this from in_dim, as (in_dim - 1) * 2.
    :param f_min:
    :param f_max:
    :return:
    """
    if not fft_length:
        fft_length = (in_dim.dimension - 1) * 2
    f_min = f_min or 0
    f_max = f_max or sampling_rate / 2.0
    # noinspection PyProtectedMember
    backend = x._raw_backend
    cache_key = (f_min, f_max, sampling_rate, fft_length, out_dim.dimension, backend, x.dtype, x.device)
    if cache_key in _mel_filter_bank_matrix_cache:
        filter_bank_matrix_: Tensor = _mel_filter_bank_matrix_cache[cache_key]
        # The in_dim might be temporarily created, so replace it by the new one.
        # Even allow for temporary out_dim.
        filter_bank_matrix = Tensor("mel-filter-bank", dims=(in_dim, out_dim), dtype=x.dtype)
        filter_bank_matrix.raw_tensor = filter_bank_matrix_.raw_tensor
    else:
        filter_bank_matrix_np = _mel_filter_bank_matrix_np(
            f_min=f_min, f_max=f_max, sampling_rate=sampling_rate, fft_size=fft_length, nr_of_filters=out_dim.dimension
        )
        filter_bank_matrix = rf.convert_to_tensor(filter_bank_matrix_np, dims=(in_dim, out_dim), _backend=backend)
        filter_bank_matrix = rf.cast(filter_bank_matrix, dtype=x.dtype)
        filter_bank_matrix = rf.copy_to_device(filter_bank_matrix, x.device)
        if backend.executing_eagerly():
            if len(_mel_filter_bank_matrix_cache) > 100:
                # Very simple cache management. No LRU logic or anything like that.
                _mel_filter_bank_matrix_cache.clear()
            _mel_filter_bank_matrix_cache[cache_key] = filter_bank_matrix
    out = rf.matmul(x, filter_bank_matrix, reduce=in_dim)
    out.feature_dim = out_dim
    return out


# Used for eager frameworks. See mel_filterbank.
_mel_filter_bank_matrix_cache = {}


@functools.lru_cache()
def _mel_filter_bank_matrix_np(
    *,
    f_min: Union[int, float],
    f_max: Union[int, float],
    sampling_rate: Union[int, float],
    fft_size: int,
    nr_of_filters: int,
) -> numpy.ndarray:
    """
    Returns the filter matrix which yields the mel filter bank features, when applied to the spectrum as
    matmul(freqDom, filterMatrix), where freqDom has dimension (time, frequency)
    and filterMatrix is the matrix returned
    by this function.
    The filter matrix is computed according to equation 6.141 in
    [Huang & Acero+, 2001] "Spoken Language Processing - A Guide to Theroy, Algorithm, and System Development"

    :param float|int f_min: minimum frequency
    :param float|int f_max: maximum frequency
    :param float sampling_rate: sampling rate of audio signal
    :param int fft_size: dimension of discrete fourier transformation
    :param int nr_of_filters: number of mel frequency filter banks to be created
    :return: shape=(fft_size // 2 + 1, nr_of_filters), matrix yielding the mel frequency cepstral coefficients
    """

    def mel_scale(freq):
        """
        returns the respective value on the mel scale

        :param float freq: frequency value to transform onto mel scale
        :rtype: float
        """
        return 1125.0 * numpy.log(1 + float(freq) / 700)

    def inv_mel_scale(mel_val):
        """
        returns the respective value in the frequency domain

        :param float mel_val: value in mel domain
        :rtype: float
        """
        return 700.0 * (numpy.exp(float(mel_val) / 1125) - 1)

    # noinspection PyShadowingNames
    def filter_center(filter_id, f_min, f_max, sampling_rate, fft_size, nr_of_filters):
        """
        :param int filter_id: filter to compute the center frequency for
        :param float|int f_min: minimum frequency
        :param float|int f_max: maximum frequency
        :param float|int sampling_rate: sampling rate of audio signal
        :param int fft_size: dimension of discrete fourier transformation
        :param int nr_of_filters: number of mel frequency filter banks to be created
        :rtype: float
        :return: center frequency of filter
        """
        return (float(fft_size) / sampling_rate) * inv_mel_scale(
            mel_scale(f_min) + filter_id * ((mel_scale(f_max) - mel_scale(f_min)) / (nr_of_filters + 1))
        )

    filt_cent = numpy.zeros(shape=(nr_of_filters + 2,), dtype=numpy.float32)
    for i1 in range(nr_of_filters + 2):
        filt_cent[i1] = filter_center(i1, f_min, f_max, sampling_rate, fft_size, nr_of_filters)
    f_mat = numpy.zeros(shape=(fft_size // 2 + 1, nr_of_filters))
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

    return f_mat


def log_mel_filterbank_from_raw(
    raw_audio: Tensor,
    *,
    in_spatial_dim: Dim,
    out_dim: Dim,
    sampling_rate: int = 16_000,
    window_len: float = 0.025,
    step_len: float = 0.010,
    n_fft: Optional[int] = None,
    log_base: Union[int, float] = 10,
) -> Tuple[Tensor, Dim]:
    """
    log mel filterbank features

    :param raw_audio: (..., in_spatial_dim, ...). if it has a feature_dim with dimension 1, it is squeezed away.
    :param in_spatial_dim:
    :param out_dim: nr of mel filters.
    :param sampling_rate: samples per second
    :param window_len: in seconds
    :param step_len: in seconds
    :param n_fft: fft_size, n_fft. Should match fft_length from :func:`stft`.
        If not provided, next power-of-two from window_num_frames.
    :param log_base: e.g. 10 or math.e
    """
    if raw_audio.feature_dim and raw_audio.feature_dim.dimension == 1:
        raw_audio = rf.squeeze(raw_audio, axis=raw_audio.feature_dim)
    window_num_frames = int(window_len * sampling_rate)
    step_num_frames = int(step_len * sampling_rate)
    if not n_fft:
        n_fft = util_math.next_power_of_two(window_num_frames)
    spectrogram, out_spatial_dim, in_dim_ = rf.stft(
        raw_audio,
        in_spatial_dim=in_spatial_dim,
        frame_step=step_num_frames,
        frame_length=window_num_frames,
        fft_length=n_fft,
    )
    power_spectrogram = rf.abs(spectrogram) ** 2.0
    # stft might have upcasted this to float32 because some PyTorch versions don't support stft on bfloat16.
    # https://github.com/pytorch/pytorch/issues/117844
    power_spectrogram = rf.cast(power_spectrogram, dtype=raw_audio.dtype)
    mel_fbank = mel_filterbank(power_spectrogram, in_dim=in_dim_, out_dim=out_dim, sampling_rate=sampling_rate)
    log_mel_fbank = rf.safe_log(mel_fbank, eps=1e-10)
    if log_base != math.e:
        log_mel_fbank = log_mel_fbank * (1.0 / math.log(log_base))
    return log_mel_fbank, out_spatial_dim
