from layers.layer import Layer
from numpy.random import rand
import numpy as np


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        weights = rand(output_size, input_size) - 0.5
        self.w = np.asarray(weights)
        self.b = np.asarray(rand(output_size, 1)) - 0.5
        self.input: np.ndarray | None = None

    def prop(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        try:
            val = np.dot(self.w, input) + self.b
            return val
        except ValueError:
            self.b = self.b[:, 0].reshape(self.b.shape[0], 1)
            return np.dot(self.w, input) + self.b

    def back_prop(self, grad: np.ndarray, alpha: float) -> np.ndarray:
        if self.input is None:
            raise ValueError('No input data')

        dw = np.dot(grad, self.input.T)
        db = grad.sum(axis=1) / grad.shape[1]

        self.w = np.subtract(self.w, alpha * dw)
        self.b = np.subtract(self.b, alpha * db)

        dx = np.dot(self.w.T, grad)
        return dx

    def save(self, path: str, i: int) -> dict:
        np.save(f'{path}/dense_{i}_w', self.w)
        np.save(f'{path}/dense_{i}_b', self.b)
        return {
            'type': 'Dense',
            'input_size': self.input_size,
            'output_size': self.output_size,
            'w': f'dense_{i}_w.npy',
            'b': f'dense_{i}_b.npy'
        }

    def open(self, path: str, info: dict) -> None:
        w_file = info['w']
        self.w = np.load(f'{path}/{w_file}')
        b_file = info['b']
        self.b = np.load(f'{path}/{b_file}')
