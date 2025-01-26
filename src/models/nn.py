import jax.numpy as np
from layers.layer import Layer
from utils.utils import stable_softmax
from typing import Callable


class Network:
    def __init__(self, layers: list[Layer], softmax: bool,
                 loss: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 loss_prime: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.layers: list[Layer] = layers
        self.layers_reverse: list[Layer] = layers[::-1]
        self.softmax: bool = softmax
        self.loss: Callable[[np.ndarray, np.ndarray], np.ndarray] = loss
        self.loss_prime: Callable[[np.ndarray,
                                   np.ndarray], np.ndarray] = loss_prime

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
              alpha: float = 0.1, batch_size: int = 4000) -> None:
        n = x.shape[0]
        batches = n // batch_size
        a = 0
        print(f'x: {x.shape}')
        for i in range(epochs):
            for j in range(batches):
                x_act = x[j * batch_size:(j + 1) * batch_size].T
                y_act = y[j * batch_size:(j + 1) * batch_size].T
                a = self.back_prop(x_act, y_act, alpha)
                if i % 10 == 0 and j == 0:
                    print(
                        f'Epoch: {i} \nAccuracy: {self.accuracy(a, y_act)}')

    def accuracy(self, a_2: np.ndarray, y: np.ndarray) -> float:
        predictions = np.argmax(a_2, axis=0)
        correct = np.sum(predictions == np.argmax(y, axis=0))
        print(f'correct: {correct}')
        print(f'size: {y.shape[1]}')
        return float(correct / y.shape[1])

    def save(self, path: str) -> None:
        pass
