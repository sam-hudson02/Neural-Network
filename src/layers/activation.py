from layers.layer import Layer
from utils.activation import ActivationFunction
import numpy as np


class Activation(Layer):
    def __init__(self, activation: ActivationFunction):
        self.a_func = activation

    def prop(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return self.a_func.eval(input)

    def back_prop(self, grad: np.ndarray) -> np.ndarray:
        # the ReLU derivative is a boolean array, and bool * float
        # keeps the float's dtype, so nothing is promoted here
        return grad * self.a_func.derivative(self.input)

    def save(self, path: str, i: int) -> dict:
        return {
            'type': 'Activation',
            'activation': self.a_func.name
        }


class Softmax(Layer):
    """
    Softmax over axis 0 (one column per sample).
    """

    def __init__(self, input_size: tuple[int, int, int] | None = None):
        self.input_size = input_size
        self.output: np.ndarray | None = None

    def prop(self, input: np.ndarray) -> np.ndarray:
        exps = np.exp(input - np.max(input, axis=0, keepdims=True))
        self.output = exps / np.sum(exps, axis=0, keepdims=True)
        return self.output

    def back_prop(self, grad: np.ndarray) -> np.ndarray:
        if self.output is None:
            raise ValueError('self.output is None')
        # J = diag(s) - s s^T per sample, so J^T g collapses to
        # s * (g - sum(s * g)) without ever forming the Jacobian
        s = self.output
        return s * (grad - np.sum(s * grad, axis=0, keepdims=True))

    def save(self, path: str, i: int) -> dict:
        return {'type': 'Softmax'}
