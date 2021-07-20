import matplotlib.pyplot as plt
import numpy as np
import time

from src.data_generator import classic_signal, create_mra_data
from src.utils import shift_signal, relative_error


def generate_shift_dist(s, L):
    shift_dist = np.array([np.exp(-np.square(t / s)) for t in np.arange(1, L + 1)])
    shift_dist /= np.sum(shift_dist)
    return shift_dist


class EmAlgorithm:

    def __init__(self,
                 data: np.ndarray,
                 noise_std):
        self.data = data
        self.N = data.shape[0]
        self.L = data.shape[1]
        self.noise_std = noise_std

    def calc_weights(self, curr_signal_est, curr_shift_dist_est):
        weights = np.zeros(shape=(self.N, self.L))

        for j in range(self.N):
            for l in range(self.L):
                term1 = np.square(np.linalg.norm(shift_signal(curr_signal_est, l) - self.data[j]))
                term2 = curr_shift_dist_est[l]
                term = np.exp(-1 / (2 * self.noise_std ** 2) * term1) * term2
                weights[j, l] = term
            weights[j] /= np.sum(weights[j])

        return weights

    def calc_next_estimations(self, curr_signal_est, curr_shift_dist_est):

        # t0 = time.time()
        weights = self.calc_weights(curr_signal_est, curr_shift_dist_est)

        # t1 = time.time()
        # print(t1 - t0)

        next_signal_estimation = np.zeros_like(curr_signal_est)

        for j in range(self.N):
            for l in range(self.L):
                next_signal_estimation += weights[j, l] * shift_signal(self.data[j], - l)
        next_signal_estimation /= self.N

        # t2 = time.time()
        # print(t2 - t1)

        summed_weights = np.sum(weights, axis=0)
        next_shift_dist_est = np.zeros_like(curr_shift_dist_est)
        for l in range(self.L):
            next_shift_dist_est[l] = summed_weights[l]
        next_shift_dist_est /= np.sum(summed_weights)

        # t3 = time.time()
        # print(t3 - t2)

        return next_signal_estimation, next_shift_dist_est

    def run(self, iterations=20):

        curr_signal_est = np.arange(self.L, dtype=float) / self.L
        curr_shift_dist_est = np.full(self.L, fill_value=1 / self.L)

        results = []
        for t in range(iterations):
            print(f'At iteration {t}')
            curr_signal_est, curr_shift_dist_est = self.calc_next_estimations(curr_signal_est, curr_shift_dist_est)
            results.append((curr_signal_est, curr_shift_dist_est))

        return np.array(results)


def em_experiment(debug_plot=False):
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

            em_algo = EmAlgorithm(mra_data, noise_std)
            em_results = em_algo.run(1)

            signal_est = em_results[-1, 0]
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


em_experiment()
