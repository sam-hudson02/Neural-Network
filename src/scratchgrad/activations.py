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
    def __init__(self):
        self.name = 'softmax'

    def eval(self, val: np.ndarray) -> np.ndarray:
        exps = np.exp(val - np.max(val, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def derivative(self, val: np.ndarray) -> np.ndarray:
        # softmax couples every output to every input, so it has no
        # elementwise derivative for Activation to multiply through
        raise NotImplementedError(
            'softmax has no elementwise derivative; use '
            'layers.activation.Softmax, or Network(softmax=True) with '
            'cce_softmax_prime')
