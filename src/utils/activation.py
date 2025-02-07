import numpy as np


class ActivationFunction:
    def __init__(self):
        self.name = self.__class__.__name__.lower()
        pass

    def eval(self, val: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, val: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(ActivationFunction):
    def __init__(self):
        self.name = 'relu'

    def eval(self, val: np.ndarray) -> np.ndarray:
        return np.maximum(0, val)

    def derivative(self, val: np.ndarray) -> np.ndarray:
        return val > 0


class Sigmoid(ActivationFunction):
    def __init__(self):
        self.name = 'sigmoid'

    def eval(self, val: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-val))

    def derivative(self, val: np.ndarray) -> np.ndarray:
        return self.eval(val) * (1 - self.eval(val))


class Tanh(ActivationFunction):
    def __init__(self):
        self.name = 'tanh'

    def eval(self, val: np.ndarray) -> np.ndarray:
        return np.tanh(val)

    def derivative(self, val: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(val) ** 2


class Softmax(ActivationFunction):
    def eval(self, val: np.ndarray) -> np.ndarray:
        exps = np.exp(val)
        softmax = exps / np.sum(exps, axis=0)
        if np.isnan(softmax).any():
            raise ValueError('Softmax returned nan values')
        return softmax

    def derivative(self, val: np.ndarray) -> np.ndarray:
        n = val.shape[0]
        matrix = np.subtract(np.eye(n), val.T)
        return np.dot(matrix, val)
