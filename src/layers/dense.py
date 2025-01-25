from layers.layer import Layer
from numpy.random import rand
import jax.numpy as np


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        weights = rand(output_size, input_size) - 0.5
        print(weights)
        self.w = np.asarray(weights)
        print(self.w)
        self.b = np.asarray(rand(output_size, 1)) - 0.5
        print(self.b.shape)
        self.input: np.ndarray | None = None

    def prop(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        try:
            return np.dot(self.w, input) + self.b
        except TypeError:
            self.b = self.b[:, 0].reshape(self.b.shape[0], 1)
            return np.dot(self.w, input) + self.b

    def back_prop(self, grad: np.ndarray, alpha: float) -> np.ndarray:
        if self.input is None:
            raise ValueError('No input data')

        dw = np.dot(grad, self.input.T)
        db = grad

        self.w = np.subtract(self.w, alpha * dw)
        self.b = np.subtract(self.b, alpha * db)

        return np.dot(self.w.T, grad)
