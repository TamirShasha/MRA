import numpy as np

from src.utils import shift_signal


def _add_gaussian_noise(signal, sigma):
    """
    :param signal: Clean signal
    :param sigma: Noise STD
    :return: Noisy signal
    """
    noise = np.random.normal(0, sigma, len(signal))
    return signal + noise


def create_mra_data(signal, num_of_samples, sigma, shift_dist=None):
    """
    :param signal: Clean signal inside a window of length L
    :param num_of_samples: The number of signal shifts to generate
    :param sigma: Noise STD
    :param shift_dist: shift distribution
    :return: noisy signals shifted according to given shift distribution
    """
    signals = []
    for i in range(num_of_samples):
        shifted_signal = shift_signal(signal, dist=shift_dist)
        noisy_signal = _add_gaussian_noise(shifted_signal, sigma)
        signals.append(noisy_signal)

    return np.array(signals)


def classic_signal():
    L = 20
    signal = np.zeros(L)
    signal[6:10] = 0.35
    signal[11:14] = -0.35
    return signal
