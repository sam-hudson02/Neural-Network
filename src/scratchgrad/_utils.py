import numpy as np


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


def pad(num: float, length: int) -> str:
    return f'{num:.{length}f}'


def softmax(x: np.ndarray) -> np.ndarray:
    return stable_softmax(x)


def stable_softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax over axis 0, shifted by the per-column max so that no column can
    overflow or underflow to an all-zero denominator.
    """
    exps = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exps / np.sum(exps, axis=0, keepdims=True)
