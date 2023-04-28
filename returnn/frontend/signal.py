"""
stft etc
"""


from __future__ import annotations
from typing import Optional, Union, Tuple
import numpy
import functools
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


def stft(
    x: Tensor,
    *,
    in_spatial_dim: Dim,
    frame_step: int,
    frame_length: int,
    fft_length: Optional[int] = None,
    window_use_frame_length: bool = True,
    align_window_left: bool = True,
    window_enforce_even: bool = True,
    out_spatial_dim: Optional[Dim] = None,
    out_dim: Optional[Dim] = None,
) -> Tuple[Tensor, Dim, Dim]:
    """
    Calculate the short-time Fourier transform (STFT) of a signal.

    We currently always use the Hann window.

    Note that there are inconsistencies between frameworks (e.g. PyTorch vs TensorFlow).
    We have options where you can explicitly specify the behavior,
    to match either PyTorch or TensorFlow.
    See here for some discussion and demonstration:
    https://github.com/pytorch/pytorch/issues/100177
    https://github.com/librosa/librosa/issues/1701
    https://github.com/albertz/playground/blob/master/tf_pt_stft.py

    :param x: (..., in_spatial_dim, ...). any other dims are treated as batch dims.
    :param in_spatial_dim:
    :param frame_length: win_length in PT/librosa, frame_size in StftLayer, frame_length in TF
    :param frame_step: hop_length in PT/librosa, frame_shift in StftLayer, frame_step in TF
    :param fft_length: n_fft in PT/librosa, fft_size in StftLayer, fft_length in TF.
        If not given, will use frame_length.
    :param window_use_frame_length: Whether to use window size = frame_length.
        If False, uses window size = fft_length.
        This has only an effect if frame_length != fft_length.
        This has an effect on the output seq length.
        If True: out_seq_len = ⌈(T - frame_length + 1) / frame_step⌉ (T = in_seq_len).
        If False: out_seq_len = ⌈(T - fft_length + 1) / frame_step⌉.
        Note that in TF/SciPy, the behavior matches to window_use_frame_length=True,
        but in PT/librosa, the behavior matches to window_use_frame_length=False.
    :param align_window_left: whether to align the window to the left inside the potential wider fft_length window.
        This only has an effect if frame_length != fft_length.
        If True, the window will be aligned to the left and the right side will be padded with zeros.
        Then effectively the remaining phases are not used.
        If False, the window will be aligned to the center and the left and right side will be padded with zeros.
        Then effectively the beginning and end phases are not used.
        In TF/SciPy, the behavior matches to align_window_left=True,
        but in PT/librosa, the behavior matches to align_window_left=False.
    :param window_enforce_even: enforce that the window size is even, potentially lowering the window size by 1.
        This only has an effect if the window size is uneven.
        In TF, the behavior matches to window_enforce_even=True,
        but in most other frameworks, the behavior matches to window_enforce_even=False.
    :param out_spatial_dim:
    :param out_dim:
    :return: (stft, out_spatial_dim, out_dim)
    """
    fft_length = fft_length or frame_length
    if out_dim is None:
        out_dim = Dim(fft_length // 2 + 1, name="stft-freq")
    if out_spatial_dim is None:
        from .conv import make_conv_out_spatial_dims

        (out_spatial_dim,) = make_conv_out_spatial_dims(
            [in_spatial_dim], filter_size=frame_length, strides=frame_step, padding="valid"
        )
    # noinspection PyProtectedMember
    return (
        x._raw_backend.stft(
            x,
            in_spatial_dim=in_spatial_dim,
            frame_step=frame_step,
            frame_length=frame_length,
            fft_length=fft_length,
            window_use_frame_length=window_use_frame_length,
            align_window_left=align_window_left,
            window_enforce_even=window_enforce_even,
            out_spatial_dim=out_spatial_dim,
            out_dim=out_dim,
        ),
        out_spatial_dim,
        out_dim,
    )


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
        filter_bank_matrix = _mel_filter_bank_matrix_cache[cache_key]
    else:
        filter_bank_matrix_np = _mel_filter_bank_matrix_np(
            f_min=f_min, f_max=f_max, sampling_rate=sampling_rate, fft_size=fft_length, nr_of_filters=out_dim.dimension
        )
        filter_bank_matrix_np = filter_bank_matrix_np.astype(x.dtype)
        filter_bank_matrix = rf.convert_to_tensor(filter_bank_matrix_np, dims=(in_dim, out_dim), _backend=backend)
        filter_bank_matrix = rf.copy_to_device(filter_bank_matrix, x.device)
        if backend.executing_eagerly():
            if len(_mel_filter_bank_matrix_cache) > 100:
                # Very simple cache management. No LRU logic or anything like that.
                _mel_filter_bank_matrix_cache.clear()
            _mel_filter_bank_matrix_cache[cache_key] = filter_bank_matrix
    return rf.matmul(x, filter_bank_matrix, reduce=in_dim)


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
