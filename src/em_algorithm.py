import numpy as np
import time

from src.utils import shift_signal


class EmAlgorithm:

    def __init__(self,
                 data: np.ndarray,
                 noise_std):
        self.data = data
        self.N = data.shape[0]
        self.L = data.shape[1]
        self.noise_std = noise_std

        self.shifted_data = np.array([np.roll(self.data, l, axis=1) for l in range(self.L)])

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
                next_signal_estimation += weights[j, l] * self.shifted_data[-l, j]
        next_signal_estimation /= self.N

        summed_weights = np.sum(weights, axis=0)
        next_shift_dist_est = np.zeros_like(curr_shift_dist_est)
        for l in range(self.L):
            next_shift_dist_est[l] = summed_weights[l]
        next_shift_dist_est /= np.sum(summed_weights)

        return next_signal_estimation, next_shift_dist_est

    def run(self, iterations=50):

        curr_signal_est = np.arange(self.L, dtype=float) / self.L
        curr_shift_dist_est = np.full(self.L, fill_value=1 / self.L)

        results = []
        for t in range(iterations):
            print(f'At iteration {t}')
            curr_signal_est, curr_shift_dist_est = self.calc_next_estimations(curr_signal_est, curr_shift_dist_est)
            results.append((curr_signal_est, curr_shift_dist_est))

        return np.array(results)


class EmAlgorithmFFT(EmAlgorithm):
    def em_iteration(self, fftx, fftX, rho, sqnormX, sigma):
        C = np.fft.ifft(np.conjugate(fftx) * fftX).real
        T = (2 * C - sqnormX) / (2 * sigma ** 2)
        T = (T.T - T.max(axis=1)).T
        W = np.exp(T)
        W = W * rho
        W = (W.T / np.sum(W, axis=1)).T
        fftW = np.fft.fft(W)
        return np.mean(np.conjugate(fftW) * fftX, axis=0), np.mean(W, axis=0)

    def run(self, iterations=20):
        curr_signal_est = np.arange(self.L, dtype=float) / self.L
        curr_shift_dist_est = np.full(self.L, fill_value=1 / self.L)

        fftx = np.fft.fft(curr_signal_est)
        fftX = np.fft.fft(self.data)
        sqnormX = (np.abs(self.data) ** 2).max(axis=0)

        results = []
        for t in range(iterations):
            # print(f'At iteration {t}')
            fftx, curr_shift_dist_est = self.em_iteration(fftx, fftX, curr_shift_dist_est, sqnormX, self.noise_std)
            results.append((np.fft.ifft(fftx).real, curr_shift_dist_est))

        return np.array(results)
