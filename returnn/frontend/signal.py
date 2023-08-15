"""
stft etc
"""


from __future__ import annotations
from typing import Optional, Tuple
from returnn.tensor import Tensor, Dim


__all__ = ["stft"]


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
