import numpy as np
from typing import Tuple


class Layer:
    def __init__(self, input_size: Tuple[int, int, int]):
        self.input_size = input_size

    def prop(self, input: np.ndarray) -> np.ndarray:
        raise (ValueError('method prop not implemented'))

    def back_prop(self, grad: np.ndarray) -> np.ndarray:
        raise (ValueError('method back_prop not implemented'))

    def save(self, path: str, i: int) -> dict:
        pass

    def open(self, path: str, info: dict) -> None:
        print('Warning: method open not implemented in',
              self.__class__.__name__)
        pass

    def __str__(self):
        # return class name
        return f'{self.__class__.__name__}'
