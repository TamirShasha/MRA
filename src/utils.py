import numpy as np


def relative_error(estimated_signal, true_signal):
    """
    Calculate the relative error between estimated signal and true signal up to circular shift
    :return: relative error
    """
    n = len(true_signal)
    corr = [np.linalg.norm(true_signal - np.roll(estimated_signal, i)) for i in range(n)]
    error = np.min(corr) / np.linalg.norm(true_signal)
    return error
