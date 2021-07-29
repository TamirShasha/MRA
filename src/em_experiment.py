import matplotlib.pyplot as plt
import numpy as np

from src.data_generator import classic_signal, create_mra_data
from src.utils import shift_signal, relative_error, generate_shift_dist
from src.em_algorithm import EmAlgorithm, EmAlgorithmFFT
from src.em_un_algorithm import EmUnAlgorithm, EmUnAlgorithmFFT

# np.random.seed(500)


def em_experiment(em_alg: bool = True, use_fft: bool = False, debug_plot=False):
    """
    :param em_alg: which EM algorithm to run, True for EM with distribution and False without distribution
    :param debug_plot:
    :return:
    """
    signal = classic_signal()
    N = 2000
    L = signal.shape[0]
    noise_std = 1
    ss = np.arange(3, 13)
    # ss = [3, 4]
    shift_dists = [generate_shift_dist(s, L) for s in ss]

    times = 30
    modified_em_exp_errs = np.zeros_like(ss, dtype=float)
    uniform_em_exp_errs = np.zeros_like(ss, dtype=float)
    for i, shift_dist in enumerate(shift_dists):
        for t in range(times):
            print(f'At experiment #{t} for shift_dist #{i}')
            mra_data = create_mra_data(signal, N, noise_std, shift_dist).T
            if use_fft:
                em_algo = EmAlgorithmFFT(mra_data, noise_std) if em_alg else EmUnAlgorithmFFT(mra_data, noise_std)
            else:
                em_algo = EmAlgorithm(mra_data, noise_std) if em_alg else EmUnAlgorithm(mra_data, noise_std)

            em_results = em_algo.run(50)
            signal_est = em_results[-1, 0] if em_alg else em_results[-1]
            err, shift = relative_error(signal_est, signal)
            modified_em_exp_errs[i] += err

            unmodified_em_algo = EmUnAlgorithmFFT(mra_data, noise_std)
            em_results = unmodified_em_algo.run(50)
            signal_est = em_results[-1]
            err, shift = relative_error(signal_est, signal)
            uniform_em_exp_errs[i] += err

        uniform_em_exp_errs[i] /= times
        modified_em_exp_errs[i] /= times

    plt.plot(ss, uniform_em_exp_errs, label='Uniform EM')
    plt.plot(ss, modified_em_exp_errs, label='Modified EM')
    plt.xlabel('Uniformity')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.show()


em_experiment(em_alg=True, use_fft=True, debug_plot=False)
# em_experiment(em_alg=True)
