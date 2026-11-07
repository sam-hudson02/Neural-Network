import numpy as np

from scratchgrad._dtype import DTYPE


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
