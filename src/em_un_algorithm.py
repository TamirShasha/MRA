import numpy as np
import time

from src.utils import shift_signal, relative_error
from src.em_algorithm import EmAlgorithm


class EmUnAlgorithm(EmAlgorithm):

    def calc_weights(self, curr_signal_est, curr_shift_dist_est):
        weights = np.zeros(shape=(self.N, self.L))

        for j in range(self.N):
            for l in range(self.L):
                term1 = np.square(np.linalg.norm(shift_signal(curr_signal_est, l) - self.data[j]))
                term = np.exp(-1 / (2 * self.noise_std ** 2) * term1)
                weights[j, l] = term
            weights[j] /= np.sum(weights[j])

        return weights

    def calc_next_estimations(self, curr_signal_est, curr_shift_dist_est):

        # t0 = time.time()
        weights = self.calc_weights(curr_signal_est, None)

        # t1 = time.time()
        # print(t1 - t0)

        next_signal_estimation = np.zeros_like(curr_signal_est)

        for j in range(self.N):
            for l in range(self.L):
                next_signal_estimation += weights[j, l] * self.shifted_data[-l, j]
        next_signal_estimation /= self.N

        return next_signal_estimation

    def run(self, iterations=20):

        curr_signal_est = np.arange(self.L, dtype=float) / self.L

        results = []
        for t in range(iterations):
            print(f'At iteration {t}')
            curr_signal_est = self.calc_next_estimations(curr_signal_est, None)
            results.append(curr_signal_est)

        return np.array(results)


class EmUnAlgorithmFFT(EmAlgorithm):
    def em_iteration(self, fftx, fftX, sqnormX, sigma):
        C = np.fft.ifft((fftX.T * np.conjugate(fftx))).T
        T = (2 * C - sqnormX.T) / (2 * sigma ** 2)
        T -= T.max(axis=0)
        W = np.exp(T)
        W /= np.sum(W, axis=0)
        fftW = np.fft.fft(W.T)
        return np.mean(np.conjugate(fftW).T * fftX, axis=1)

    def run(self, iterations=20, tol=1e-4):
        curr_signal_est = (np.arange(self.L, dtype=float) / self.L).T

        fftx = np.fft.fft(curr_signal_est)
        fftX = np.fft.fft(self.data.T).T
        sqnormX = np.square(self.data).sum(axis=0)

        results = []
        for t in range(iterations):
            fftx = self.em_iteration(fftx, fftX, sqnormX, self.noise_std)
            next_signal_est = np.fft.ifft(fftx).real
            results.append(next_signal_est)
            if relative_error(curr_signal_est, next_signal_est)[0] < tol:
                print(f'Reached the tolerance at iteration #{t}')
                break

            curr_signal_est = next_signal_est

        return np.array(results)
