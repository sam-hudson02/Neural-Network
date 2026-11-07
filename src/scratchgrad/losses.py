import numpy as np


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
