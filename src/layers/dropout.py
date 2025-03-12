from layers.layer import Layer
import numpy as np


class Dropout(Layer):
    def __init__(self, rate):
        self.rate = rate

    def prop(self, input: np.ndarray) -> np.ndarray:
        # random distribution between 0 and 1
        rand = np.random.rand(*input.shape)
        # mask is a boolean array of values that are greater than the rate
        self.mask = (rand > self.rate).astype(int)
        return input * self.mask

    def back_prop(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask

    def save(self, path: str, i: int) -> dict:
        return {
            'type': 'Dropout',
            'rate': self.rate
        }

    def open(self, path: str, info: dict) -> None:
        self.rate = info['rate']
