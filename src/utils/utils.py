from enum import Enum
import numpy as np


class Activation(Enum):
    RELU = 1
    SIGMOID = 2
    TANH = 3


def eta_fancy(sl: float) -> str:
    output = ''
    if sl > (60 * 60):
        hours = sl // (60 * 60)
        output += f'{hours} hours, '
        sl = sl % (60 * 60)
    if sl > 60:
        minutes = sl // 60
        output += f'{minutes} minutes, '
        sl = sl % 60
    output += f'{int(sl)} seconds'
    return output


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.power(np.subtract(y_true, y_pred), 2)))


def mse_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2 * np.subtract(y_pred, y_true) / y_true.size


def cce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-15
    x = y_true * np.log(y_pred + eps)
    return float(np.sum(x)) / -y_true.shape[1]


def cce_softmax_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_pred - y_true


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
                tmp = np.exp(x[:, i]-np.max(x[:, i]))
                y[:, i] = tmp / np.sum(tmp)
        if np.isnan(y).any():
            raise ValueError('Got nan value')
    return y
