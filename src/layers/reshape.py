from layers.layer import Layer
import numpy as np
from typing import Tuple


class Reshape(Layer):
    def __init__(self, input_size: Tuple):
        self.input_size: Tuple = input_size
        self.output_size: int = input_size[0] * input_size[1] * input_size[2]

    def prop(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        images = input.shape[3]
        return np.reshape(input, (self.output_size, images))

    def back_prop(self, grad: np.ndarray, alpha: float) -> np.ndarray:
        images = self.input.shape[3]
        return np.reshape(grad, (self.input_size[0], self.input_size[1],
                                 self.input_size[2], images))
