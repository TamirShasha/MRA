import numpy as np
import matplotlib.pyplot as plt

from src.data_generator import classic_signal, create_mra_data
from src.utils import generate_shift_dist, relative_error, shift_signal
from src.em_algorithm import EmAlgorithmFFT, EmAlgorithm


# np.random.seed(500)


def sample_complexity_experiment():
    signal = classic_signal()
    N = 100000
    L = signal.shape[0]
    shift_dist = generate_shift_dist(3, L)

    noise_std = 1
    mra_data = create_mra_data(signal, N, noise_std, shift_dist)
    # plt.plot(mra_data[0])
    # plt.show()
    # exit()

    # mra_data = np.array([[1, 2, 3, 3], [4, 5, 5, 6], [7, 7, 8, 9]]).T
    em_algo = EmAlgorithmFFT(mra_data.T, noise_std)
    # em_algo = EmAlgorithm(mra_data, noise_std)
    em_results = em_algo.run()

    signal_est = em_results[-1, 0]
    err, shift = relative_error(signal_est, signal)

    shifted_estimated_signal = shift_signal(signal_est, shift)

    print(f'Error for noise std {noise_std} is {err}')

    plt.plot(shifted_estimated_signal.real)
    plt.plot(signal)
    plt.show()

    est_shift_dist = em_results[-1, 1]
    err2, shift2 = relative_error(est_shift_dist, shift_dist)
    shifted_estimated_dist = shift_signal(est_shift_dist, shift2).real

    plt.plot(shifted_estimated_dist)
    plt.plot(shift_dist)
    plt.show()


sample_complexity_experiment()
