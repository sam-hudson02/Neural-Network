import numpy as np
from utils.utils import Activation, activation_func, softmax, \
    activation_derivative
from typing import Tuple


class Classifier:
    def __init__(self, x: np.ndarray, y: np.ndarray,
                 activation: Activation = Activation.RELU,
                 hidden: int = 128):
        """
        Basic classifier neural network to train and predict MNIST data.
        :param x: np.ndarray: The input data, where each column is a sample.
        :param y: np.ndarray: The output data, where each column is a one-hot
                              encoded label.
        :param activation: Activation(optional): The activation function to be
                           used, default is RELU.
        :param hidden: int(optional): Width of the hidden layer. It used to be
                       pinned to the number of classes, which caps the model
                       at one hidden unit per class.
        """
        self.x: np.ndarray = x
        self.y: np.ndarray = y
        self.n: int = x.shape[1]
        self.activation: Activation = activation
        self.classes: int = y.shape[0]
        print(f'classes: {self.classes}')
        self.size: int = x.shape[0]
        self.hidden: int = hidden
        print(f'size: {self.size}')
        self.w_1, self.w_2, self.b_1, self.b_2 = self.init_weights()

    def propagate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
        """
        Forward propagation of the neural network. This method takes some input
        (x) and propagates it through the network to get the output. where a_2
        is the final output of the network.
        :param x: np.ndarray: The input data, where each column is a sample.
        """
        z_1 = np.dot(self.w_1, x) + self.b_1
        a_1 = activation_func(z_1, self.activation)
        z_2 = np.dot(self.w_2, a_1) + self.b_2
        a_2 = softmax(z_2)
        return z_1, a_1, z_2, a_2

    def init_weights(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray]:
        # He initialisation, matching layers.dense.Dense
        w_1 = np.random.normal(0.0, np.sqrt(2.0 / self.size),
                               (self.hidden, self.size))
        w_2 = np.random.normal(0.0, np.sqrt(2.0 / self.hidden),
                               (self.classes, self.hidden))
        b_1 = np.zeros((self.hidden, 1))
        b_2 = np.zeros((self.classes, 1))
        return w_1, w_2, b_1, b_2

    def back_prop(self, x: np.ndarray, y: np.ndarray,
                  alpha: float) -> np.ndarray:
        n = x.shape[1]
        z_1, a_1, _, a_2 = self.propagate(x)
        dz_2 = np.subtract(a_2, y)
        dw_2 = np.dot(dz_2, a_1.T) / n
        db_2 = np.sum(dz_2, axis=1, keepdims=True) / n
        dz_1 = np.dot(self.w_2.T, dz_2) * \
            activation_derivative(z_1, self.activation)
        dw_1 = np.dot(dz_1, x.T) / n
        db_1 = np.sum(dz_1, axis=1, keepdims=True) / n
        self.w_1 = np.subtract(self.w_1, alpha * dw_1)
        self.b_1 = np.subtract(self.b_1, alpha * db_1)
        self.w_2 = np.subtract(self.w_2, alpha * dw_2)
        self.b_2 = np.subtract(self.b_2, alpha * db_2)
        if np.isnan(self.w_1).any() or np.isnan(self.b_1).any() or \
                np.isnan(self.w_2).any() or np.isnan(self.b_2).any():
            raise ValueError('NaN values detected')
        return a_2

    def train(self, epochs: int = 200, batch_size: int = 1000,
              alpha=0.10) -> None:
        batches = self.n // batch_size
        for i in range(epochs):
            for j in range(batches):
                x = self.x[:, j * batch_size:(j + 1) * batch_size]
                y = self.y[:, j * batch_size:(j + 1) * batch_size]
                a_2 = self.back_prop(x, y, alpha)
                if i % 10 == 0:
                    print(
                        f'Epoch: {i} \nAccuracy: {self.accuracy(a_2, y)}')

    def accuracy(self, a_2: np.ndarray, y: np.ndarray) -> float:
        predictions = np.argmax(a_2, axis=0)
        correct = np.sum(predictions == np.argmax(y, axis=0))
        return float(correct / y.shape[1])

    def test(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        predictions = self.propagate(x)
        a_2 = predictions[-1]
        return self.accuracy(a_2, y), a_2
