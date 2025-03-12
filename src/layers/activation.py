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
        return grad * self.a_func.derivative(self.input)

    def save(self, path: str, i: int) -> dict:
        return {
            'type': 'Activation',
            'activation': self.a_func.name
        }


class Softmax(Layer):
    def __init__(self, input_size: tuple[int, int, int]):
        self.input_size = input_size

    def prop(self, input: np.ndarray) -> np.ndarray:
        exps = np.exp(input)
        softmax = exps / np.sum(exps, axis=0)
        if np.isnan(softmax).any():
            raise ValueError('Softmax returned nan values')
        self.output = softmax
        return softmax

    def back_prop(self, grad: np.ndarray, alpha: float) -> np.ndarray:
        print(self.output.shape)
        n = np.shape(self.output)[1]
        print(n)
        matrix = np.subtract(np.eye(n), self.output.T) * self.output
        return np.dot(matrix, grad)
