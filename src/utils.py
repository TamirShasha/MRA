import numpy as np


def shift_signal(signal, shift=None, dist=None):
    """
    Circular Shifts given signal upon given dist
    """
    n = len(signal)
    if shift is None:
        if dist is None:
            shift = np.random.randint(n)
        else:
            shift = np.random.choice(np.arange(n), p=dist)
    shifted_signal = np.roll(signal, shift)
    return shifted_signal


def relative_error(estimated_signal, true_signal):
    """
    Calculate the relative error between estimated signal and true signal up to circular shift
    :return: relative error
    """
    n = len(true_signal)
    corr = [np.linalg.norm(true_signal - np.roll(estimated_signal, i)) for i in range(n)]
    shift = np.argmin(corr)
    error = np.min(corr) / np.linalg.norm(true_signal)
    return error, shift


def generate_shift_dist(s, L):
    """
    Generates experiment distribution of length L
    :param s: regulation param
    :param L: length
    """
    shift_dist = np.array([np.exp(-np.square(t / s)) for t in np.arange(1, L + 1)])
    shift_dist /= np.sum(shift_dist)
    return shift_dist
