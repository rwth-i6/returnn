"""
Collection of generic utilities related to signal processing
"""

import numpy


def greenwood_function(x, scaling_constant=165.4, constant_of_integration=0.88, slope=2.1):
    """
    Greenwood function, convert fractional length to frequency, see
    Schlueter, Ralf, et al. "Gammatone features and feature combination for large vocabulary speech recognition."
    ICASSP 2007 and also
    https://en.wikipedia.org/wiki/Greenwood_function
    The default values are taken from RASR, see https://github.com/rwth-i6/rasr/blob/master/src/Signal/GammaTone.cc
    They correspond to the recommended values for human data according to
    Greenwood, Donald D. "A cochlear frequency‐position function for several species—29 years later."
    The Journal of the Acoustical Society of America, 1990

    :param float x: fractional length
    :param float scaling_constant: A in [1]
    :param float constant_of_integration: k in [1]
    :param float slope: a in [1]
    :return: frequency corresponding to given fractional length
    :rtype: float
    """
    assert 0 <= x <= 1, "fractional length has to be between 0 and 1"
    return scaling_constant * (numpy.power(10.0, slope * x) - constant_of_integration)


def inv_greenwood_function(freq, scaling_constant=165.4, constant_of_integration=0.88, slope=2.1):
    """
    Inverse greenwood function, convert frequency to fractional length, see
    Schlueter, Ralf, et al. "Gammatone features and feature combination for large vocabulary speech recognition."
    ICASSP 2007 and also
    https://en.wikipedia.org/wiki/Greenwood_function
    The default values are taken from RASR, see https://github.com/rwth-i6/rasr/blob/master/src/Signal/GammaTone.cc
    They correspond to the recommended values for human data according to
    Greenwood, Donald D. "A cochlear frequency‐position function for several species—29 years later."
    The Journal of the Acoustical Society of America, 1990

    :param float freq: frequency
    :param float scaling_constant: A in [1]
    :param float constant_of_integration: k in [1]
    :param float slope: a in [1]
    :return: fractional length corresponding to given frequency
    :rtype: float
    """
    return numpy.log10(freq / scaling_constant + constant_of_integration) / slope


class GammatoneFilterbank(object):
    """
    Class representing a gammatone filterbank.
    Based on
    [1] Schlueter, Ralf, et al. "Gammatone features and feature combination for large vocabulary speech recognition."
    ICASSP 2007
    """

    def __init__(self, num_channels, length, sample_rate=16000, freq_max=7500.0, freq_min=100.0, normalization=True):
        """
        :param int num_channels: number of filters
        :param int|float length: length of FIR filters in seconds
        :param int sample_rate: sample rate of audio signal in Hz
        :param float freq_max: maximum frequency of filterbank
        :param float freq_min: minimum frequency of filterbank
        :param bool normalization: normalize filterbanks to maximum frequency response of 0 dB
        """
        self.num_channels = num_channels
        self.length = length
        self.sample_rate = sample_rate
        self.freq_max = freq_max
        self.freq_min = freq_min
        self.normalization = normalization

    def get_gammatone_filterbank(self):
        """
        Returns an array with the parameters of the gammatone filterbank

        :return: gammatone filterbank of shape (self.length * self.sample_rate, self.num_channels)
        :rtype: numpy.array
        """
        center_freqs = self.center_frequencies(self.num_channels, self.freq_max, self.freq_min)
        fbank = []
        for freq in center_freqs:
            fbank.append(self.gammatone_impulse_response(freq, self.length, self.sample_rate))
        fbank = numpy.vstack(fbank)
        if self.normalization:
            fbank = self.normalize_filters(fbank)
        fbank = numpy.transpose(fbank)
        return fbank

    @staticmethod
    def center_frequencies(num_channels, freq_max, freq_min):
        """
        Determine center frequencies for gammatone filterbank

        :param int num_channels: number of filters
        :param float freq_max: maximum frequency of filterbank
        :param float freq_min: minimum frequency of filterbank
        :return: center frequencies
        :rtype: numpy.array
        """
        x_min = inv_greenwood_function(freq_min)
        x_max = inv_greenwood_function(freq_max)
        x_linear = numpy.linspace(x_min, x_max, num_channels)
        freq_center = numpy.vectorize(greenwood_function)(x_linear)
        return freq_center

    @staticmethod
    def bandwidth_by_center_frequency(freq, lin_approx_coeff=24.7, quality_factor=9.264491981582191):
        """
        Get bandwidth (named B in [1]) by center frequency using a linear approximation of the equivalent rectangular
        bandwidth (ERB) from
        Glasberg, Brian R., and Brian CJ Moore. "Derivation of auditory filter shapes from notched-noise data."
        Hearing research, 1990
        The default values are taken from there and are also used in RASR, see
        https://github.com/rwth-i6/rasr/blob/master/src/Signal/GammaTone.cc

        :param float freq: center frequency
        :param float lin_approx_coeff: coefficient for the linear approximation of the ERB
        :param float quality_factor: audiological (ERB) based filter quality factor
        :return: bandwidth
        :rtype: float
        """
        return lin_approx_coeff * (1 / (lin_approx_coeff * quality_factor) * freq + 1.0)

    def gammatone_impulse_response(
        self, f_center, length, sample_rate, output_gain=1.0, filter_order=4, phase_shift=0.0
    ):
        """
        Compute gammatone impulse response based on [1]

        :param float f_center: center frequency
        :param int|float length: length of finite impulse response in seconds
        :param int sample_rate: sample rate of audio signal in Hz
        :param float output_gain: output gain, named k in [1]
        :param int filter_order: order of filter, named n in [1]
        :param float phase_shift: phase shift, named phi in [1]
        :return: gammatone impulse response
        :rtype: numpy.array
        """
        bandwidth = self.bandwidth_by_center_frequency(f_center)  # bandwidth (duration of impulse response)
        num_samples = int(numpy.floor(sample_rate * length))
        t = numpy.linspace(1.0 / sample_rate, length, num_samples)
        return (
            output_gain
            * numpy.power(t, filter_order - 1)
            * numpy.exp(-2 * numpy.pi * bandwidth * t)
            * numpy.cos(2 * numpy.pi * f_center * t + phase_shift)
        )

    @staticmethod
    def normalize_filters(filters):
        """
        Normalize filterbank such that the maximum frequency response is 0 dB

        :param numpy.array filters: filterbank with shape number_channels x filter_length
        :return: normalized filterbank
        :rtype: numpy.array
        """
        from scipy import signal

        for filt in range(filters.shape[0]):
            _, f_resp = signal.freqz(filters[filt, :])
            filters[filt, :] = filters[filt, :] / numpy.max(numpy.abs(f_resp))
        return filters
