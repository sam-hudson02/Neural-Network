from time import time
import numpy as np
from layers.layer import Layer
from utils.utils import eta_fancy, stable_softmax, pad
from typing import Callable
import sys


class Network:
    def __init__(self, layers: list[Layer], softmax: bool,
                 loss: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 loss_prime: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 verbose: bool = False):
        self.layers: list[Layer] = layers
        self.layers_reverse: list[Layer] = layers[::-1]
        self.softmax: bool = softmax
        self.loss: Callable[[np.ndarray, np.ndarray], np.ndarray] = loss
        self.loss_prime: Callable[[np.ndarray,
                                   np.ndarray], np.ndarray] = loss_prime
        self.verbose: bool = verbose
        self.last_update = time()

    def prop(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.prop(x)
        if self.softmax:
            x = stable_softmax(x)
        return x

    def back_prop(self, x: np.ndarray, y: np.ndarray,
                  alpha: float) -> np.ndarray:
        a = self.prop(x)
        dy = self.loss_prime(y, a)
        for layer in self.layers_reverse:
            dy = layer.back_prop(dy, alpha)
        return a

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 500,
              batch_size: int = 4000,
              alpha: Callable[[int], float] = lambda _: 0.01) -> None:
        n = x.shape[0]
        batches = n // batch_size
        a = 0
        start_time = time()
        for i in range(epochs):
            for j in range(batches):
                batch_start = time()
                x_act = x[j * batch_size:(j + 1) * batch_size].T
                y_act = y[j * batch_size:(j + 1) * batch_size].T
                a = self.back_prop(x_act, y_act, alpha(i))
                batch_end = time()
                if self.verbose:
                    ct = time()
                    if ct - self.last_update > 1:
                        eta = (batch_end - batch_start) * \
                            (batches * (epochs - i) - j)
                        eta = eta_fancy(eta)
                        elaped_time = round(time() - start_time, 2)
                        sys.stdout.write(f'\rEpoch: {i}, Accuracy: {self.accuracy(
                            a, y_act)}, Elapsed time: {elaped_time}, ETA: {eta}')

    def accuracy(self, a_2: np.ndarray, y: np.ndarray) -> str:
        predictions = np.argmax(a_2, axis=0)
        correct = np.sum(predictions == np.argmax(y, axis=0))
        return pad(round(float(correct / y.shape[1]), 4), 5)

    def save(self, path: str) -> None:
        pass
