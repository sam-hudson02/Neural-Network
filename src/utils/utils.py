from enum import Enum
import numpy as np

from utils.dtype import DTYPE


class Activation(Enum):
    RELU = 1
    SIGMOID = 2
    TANH = 3


def eta_fancy(sl: float) -> str:
    output = ''
    if sl > (60 * 60):
        hours = int(sl // (60 * 60))
        output += f'{hours} hours, '
        sl = sl % (60 * 60)
    if sl > 60:
        minutes = int(sl // 60)
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
    # summed in double precision whatever the network computes in: this
    # adds up one term per sample, and a float32 accumulator loses the
    # small ones once the running total is a few thousand times their size
    return float(np.sum(x, dtype=np.float64)) / -y_true.shape[1]


def cce_softmax_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_pred - y_true


def bce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-15
    x = y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
    return float(np.sum(x, dtype=np.float64)) / -y_true.shape[1]


def bce_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denominator = np.maximum(y_pred * (1 - y_pred) * y_true.shape[1], 1e-15)
    return (y_pred - y_true) / denominator


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
        return 1 - np.tanh(x) ** 2
    else:
        return x


def one_hot(y: np.ndarray, classes: int | None = None) -> np.ndarray:
    """
    One-hot encode integer labels as a (classes, samples) array.
    :param y: np.ndarray: 1-D array of integer labels.
    :param classes: int(optional): Size of the label space. Defaults to
                    max(y) + 1, which is only correct when every class is
                    present in y -- always pass it explicitly for a subset.
    """
    y = np.asarray(y).astype(int)
    if classes is None:
        classes = int(np.max(y)) + 1
    if y.size and (y.min() < 0 or y.max() >= classes):
        raise ValueError(f'labels outside [0, {classes}) for one_hot')
    return np.eye(classes, dtype=DTYPE)[y].T


def softmax(x: np.ndarray) -> np.ndarray:
    return stable_softmax(x)


def stable_softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax over axis 0, shifted by the per-column max so that no column can
    overflow or underflow to an all-zero denominator.
    """
    exps = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exps / np.sum(exps, axis=0, keepdims=True)
