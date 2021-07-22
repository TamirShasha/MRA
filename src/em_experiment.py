import matplotlib.pyplot as plt
import numpy as np

from src.data_generator import classic_signal, create_mra_data
from src.utils import shift_signal, relative_error, generate_shift_dist
from src.em_algorithm import EmAlgorithm
from src.em_un_algorithm import EmUnAlgorithm


def em_experiment(em_alg: bool = True, debug_plot=False):
    """
    :param em_alg: which EM algorithm to run, True for EM with distribution and False without distribution
    :param debug_plot:
    :return:
    """
    signal = classic_signal()
    N = 2000
    L = signal.shape[0]
    noise_std = 1
    ss = [3, 4, 5, 6, 7, 8, 9]
    shift_dists = [generate_shift_dist(s, L) for s in ss]

    times = 2
    exp_errs = np.zeros_like(ss, dtype=float)
    for i, shift_dist in enumerate(shift_dists):
        for t in range(times):
            print(f'At experiment #{t} for shift_dist #{i}')
            mra_data = create_mra_data(signal, N, noise_std, shift_dist)

            em_algo = EmAlgorithm(mra_data, noise_std) if em_alg else EmUnAlgorithm(mra_data, noise_std)
            em_results = em_algo.run(1)

            signal_est = em_results[-1, 0] if em_alg else em_results[0]
            err, shift = relative_error(signal_est, signal)
            exp_errs[i] += err

            if debug_plot:
                errs = []
                for signal_est, dist_est in em_results:
                    err, shift = relative_error(signal_est, signal)
                    errs.append(err)

                shifted_estimated_signal = shift_signal(signal_est, shift)

                plt.plot(np.arange(L), signal)
                plt.plot(np.arange(L), shifted_estimated_signal)
                plt.show()

                plt.plot(np.arange(L), shift_dist)
                plt.plot(np.arange(L), np.roll(em_results[-1, 1], shift))
                plt.show()

        exp_errs[i] /= times

    plt.plot(ss, exp_errs)
    plt.show()


em_experiment(em_alg=False)
em_experiment(em_alg=True)
