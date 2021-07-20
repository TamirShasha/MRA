import matplotlib.pyplot as plt
import numpy as np

from src.data_generator import classic_signal, create_mra_data


def generate_shift_dist(s):
    def shift_dist(t):
        return np.exp(-np.square(t / s))

    return shift_dist


def em_experiment():
    signal = classic_signal()
    shift_dists = [generate_shift_dist(s) for s in np.linspace(3, 9, 10)]
    num_of_samples = 2000
    sigma = 1

    times = 20
    mra_data = create_mra_data(signal, 1000, 0.25)
