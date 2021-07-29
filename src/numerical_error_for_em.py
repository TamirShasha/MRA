import numpy as np
import matplotlib.pyplot as plt

from src.data_generator import classic_signal, create_mra_data
from src.utils import generate_shift_dist, relative_error
from src.em_algorithm import EmAlgorithm, EmAlgorithmFFT


def em_numerical_error():
    signal = classic_signal()
    N = 100000
    L = signal.shape[0]
    # shift_dist = generate_shift_dist(3, L)
    # shift_dist = np.random.random(L)
    # shift_dist /= np.sum(shift_dist)
    shift_dist = np.full(L, 3 / L)

    times = 30
    noise_stds = np.exp(np.linspace(-3, 0, 20))
    exp_errs = np.zeros_like(noise_stds, dtype=float)
    for i, noise_std in enumerate(noise_stds):
        for t in range(times):
            print(f'At experiment #{t} for noise_std {noise_std}')
            mra_data = create_mra_data(signal, N, noise_std, shift_dist)

            em_algo = EmAlgorithmFFT(mra_data, noise_std)
            em_results = em_algo.run()

            signal_est = em_results[-1, 0]
            err, shift = relative_error(signal_est, signal)
            exp_errs[i] += err

        exp_errs[i] /= times
        print(f'Error for noise std {noise_std} is {exp_errs[i]}')

    print(exp_errs)
    plt.plot(np.log(noise_stds), np.log(exp_errs), marker='o')
    plt.show()


em_numerical_error()
