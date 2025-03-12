from layers.layer import Layer
from typing import Tuple
from numpy.random import rand
from scipy.signal import convolve2d, correlate2d
import numpy as np
from utils.optimizer import Optimizer, Optimizers, Adam, GradientDescent


class Convolution(Layer):
    def __init__(self, input_shape: Tuple[int, int, int], filters: int,
                 kernel_size: Tuple[int, int],
                 optimizer: Optimizers = Optimizers.ADAM,
                 alpha: float = 0.01):
        self.input_shape = input_shape
        self.height, self.width, self.depth = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_height, self.kernel_width = kernel_size
        self.output_height = self.height - self.kernel_height + 1
        self.output_width = self.width - self.kernel_width + 1
        self.kernels = np.asarray(rand(self.filters,
                                       self.depth,
                                       self.kernel_height,
                                       self.kernel_width)) - 0.5
        self.biases = np.asarray(rand(self.filters,
                                      self.height - self.kernel_height + 1,
                                      self.width - self.kernel_width + 1)) - 0.5
        self.input: np.ndarray | None = None
        self.output: np.ndarray | None = None
        opt = None
        if optimizer == Optimizers.ADAM:
            opt = Adam()
        elif optimizer == Optimizers.GRAD:
            opt = GradientDescent(alpha)
        if opt is None:
            raise ValueError('invalid optimizer')
        self.optimizer: Optimizer = opt

    def prop(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        input_t = input.T
        out = np.zeros((input_t.shape[0], self.filters, self.output_height,
                        self.output_width))
        for b in range(input_t.shape[0]):
            for i in range(self.filters):
                for j in range(self.depth):
                    out[b, i] += correlate2d(input_t[b, j], self.kernels[i, j],
                                             mode='valid')
        return (out + self.biases).T

    def back_prop(self, grad: np.ndarray) -> np.ndarray:
        if self.input is None:
            raise ValueError('self.input is None')

        dk = np.zeros(self.kernels.shape)
        dx = np.zeros(self.input.T.shape)

        input_t = self.input.T
        grad = grad.T
        for b in range(grad.shape[0]):
            for i in range(self.filters):
                for j in range(self.depth):
                    dk[i, j] += correlate2d(input_t[b, j], grad[b, i],
                                            mode='valid')
                    dx[b, j] += convolve2d(grad[b, i], self.kernels[i, j],
                                           mode='full')

        db = grad.sum(axis=0) / grad.shape[0]
        dk = dk / grad.shape[0]
        u_dk, u_db = self.optimizer.update(dk, db)

        self.kernels -= u_dk
        self.biases -= u_db

        return dx.T

    def save(self, path: str, i: int) -> dict:
        np.save(f'{path}/convolution_{i}_k', self.kernels)
        np.save(f'{path}/convolution_{i}_b', self.biases)
        return {
            'type': 'Convolution',
            'input_shape': self.input_shape,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'k': f'convolution_{i}_k.npy',
            'b': f'convolution_{i}_b.npy'
        }

    def open(self, path: str, info: dict) -> None:
        k_file = info['k']
        self.kernels = np.load(f'{path}/{k_file}')
        b_file = info['b']
        self.biases = np.load(f'{path}/{b_file}')
