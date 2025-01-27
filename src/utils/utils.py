from enum import Enum
import numpy as np


class Activation(Enum):
    RELU = 1
    SIGMOID = 2
    TANH = 3


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.power(np.subtract(y_true, y_pred), 2))


def mse_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2 * np.subtract(y_pred, y_true) / y_true.size

def pad(num: float, length: int) -> str:
    return f'{num:.{length}f}'

def activation_func(x: np.ndarray, func: Activation) -> np.ndarray:
    if func == Activation.RELU:
        return np.maximum(0, x)
    elif func == Activation.SIGMOID:
        return 1 / (1 + np.exp(-x))
    elif func == Activation.TANH:
        return np.tanh(x)
    else:
        return x


def activation_derivative(x: np.ndarray, func: Activation) -> np.ndarray | int:
    if func == Activation.RELU:
        return x > 0
    elif func == Activation.SIGMOID:
        return activation_func(x, Activation.SIGMOID) * \
            (1 - activation_func(x, Activation.SIGMOID))
    elif func == Activation.TANH:
        return 1 / np.cosh(x) ** 2
    else:
        return x


def one_hot(y: np.ndarray) -> np.ndarray:
    return np.eye(np.max(y) + 1)[y].T


def softmax(x: np.ndarray) -> np.ndarray:
    exps = np.exp(x)
    softmax = exps / np.sum(exps, axis=0)
    # check for any nan values
    if np.isnan(softmax).any():
        raise ValueError('Softmax returned nan values')
    return softmax


def stable_softmax(x: np.ndarray) -> np.ndarray:
    exps = np.exp(x - np.max(x))
    y = exps / np.sum(exps, axis=0)
    # check for any nan values
    if np.isnan(y).any():
        # find a column with nan values
        for i in range(y.shape[1]):
            if np.isnan(y[:, i]).any():
                print(x[:, i])
                tmp = np.exp(x[:, i]-np.max(x[:, i]))
                print(tmp)
                print(np.sum(tmp))
                print(tmp / np.sum(tmp))
                y[:, i] = tmp / np.sum(tmp)
        if np.isnan(y).any():
            raise ValueError('Got nan value')
    return y
