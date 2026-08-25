import numpy as np
from typing import Tuple


class Layer:
    # layers that behave differently while learning (dropout, etc) read this
    # network sets it for the duration of training forward pass and clears
    # it again, so validation and prediction always run in inference mode
    training: bool = False

    def __init__(self, input_size: Tuple[int, int, int]):
        self.input_size = input_size

    def prop(self, input: np.ndarray) -> np.ndarray:
        raise (ValueError('method prop not implemented'))

    def back_prop(self, grad: np.ndarray) -> np.ndarray:
        raise (ValueError('method back_prop not implemented'))

    def save(self, path: str, i: int) -> dict:
        pass

    def open(self, path: str, info: dict) -> None:
        # most layers hold no state, so loading one is a no-op; the layers
        # that do carry weights (Dense, Convolution) override this
        pass

    def __str__(self):
        # return class name
        return f'{self.__class__.__name__}'
