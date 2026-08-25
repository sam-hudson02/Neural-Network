from layers.layer import Layer
import numpy as np
from utils.dtype import DTYPE


class Dropout(Layer):
    """
    Inverted dropout.

    While learning, each unit is zeroed with probability rate and the
    survivors are divided by (1 - rate), which keeps the expected activation
    the same as it would be without dropout. That scaling is what lets the
    layer do nothing at all at inference time: the network the following
    layers were trained against already has the right magnitude, so no
    correction is needed when every unit is kept.
    """

    def __init__(self, rate: float):
        """
        :param rate: float: Probability of zeroing a unit, in [0, 1).
        """
        if not 0.0 <= rate < 1.0:
            raise ValueError(f'dropout rate must be in [0, 1), got {rate}')
        self.rate = rate
        self.mask: np.ndarray | None = None

    def prop(self, input: np.ndarray) -> np.ndarray:
        if not self.training or self.rate == 0.0:
            # inference: keep everything, and leave the scale alone
            self.mask = None
            return input
        keep = 1.0 - self.rate
        # built in the working dtype: a float64 mask would promote the
        # activations and every matmul after this layer
        self.mask = ((np.random.rand(*input.shape) < keep) / keep).astype(DTYPE)
        return input * self.mask

    def back_prop(self, grad: np.ndarray) -> np.ndarray:
        # the gradient has to travel through exactly the units that carried
        # the activation forward, with the same scaling
        if self.mask is None:
            return grad
        return grad * self.mask

    def save(self, path: str, i: int) -> dict:
        return {
            'type': 'Dropout',
            'rate': self.rate
        }

    def open(self, path: str, info: dict) -> None:
        self.rate = info['rate']
