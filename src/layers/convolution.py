from layers.layer import Layer
from typing import Tuple
from jax.scipy.signal import correlate2d
from numpy.random import rand
import jax.numpy as np
import time
from scipy.signal import fftconvolve


class Convolution(Layer):
    def __init__(self, input_shape: Tuple[int, int, int], filters: int,
                 kernel_size: Tuple[int, int]):
        self.height, self.width, self.depth = input_shape
        self.filters = filters
        self.kernel_height, self.kernel_width = kernel_size
        self.output_height = self.height - self.kernel_height + 1
        self.output_width = self.width - self.kernel_width + 1
        self.kernels = np.asarray(rand(self.kernel_height,
                                       self.kernel_width,
                                       self.depth,
                                       self.filters)) - 0.5
        self.biases = np.asarray(rand(self.height - self.kernel_height + 1,
                                      self.width - self.kernel_width + 1,
                                      self.filters)) - 0.5
        self.input: np.ndarray | None = None
        self.output: np.ndarray | None = None

    def prop(self, input: np.ndarray) -> np.ndarray:
        start_time = time.time()
        print(input.shape)
        images = input.shape[3]
        self.input = input
        self.output = np.zeros((self.filters * self.depth, self.height - 2,
                                self.width - 2, images))
        outputs = []
        count = 0
        for i in range(images):
            image = input[:, :, :, i]
            image_outputs = []
            for j in range(self.filters):
                image_out = np.zeros((self.output_height, self.output_width))
                bias = self.biases[:, :, j]
                for k in range(self.depth):
                    count += 1
                    channel = image[:, :, k]
                    filter = self.kernels[:, :, k, j]
                    correlation = fftconvolve(channel, filter, "valid")
                    image_out += correlation + bias
                image_outputs.append(image_out)
            image_outputs = np.asarray(image_outputs)
            outputs.append(image_outputs)
        self.output = np.asarray(outputs).T
        end_time = time.time()
        print(f'Convolution time: {end_time - start_time}')
        print(f'Count: {count}')
        return self.output

    def back_prop(self, grad: np.ndarray, alpha: float) -> np.ndarray:
        start_time = time.time()
        if self.input is None:
            raise ValueError('self.input is None')

        n = self.input.shape[3]

        dk = np.zeros((self.kernel_height, self.kernel_width,
                      self.depth, self.filters))
        # dx = np.zeros((self.depth, self.height, self.width))

        for i in range(n):
            image = self.input[:, :, :, i]
            filter_errors = []
            for j in range(self.filters):
                depth_errors = []
                for k in range(self.depth):
                    channel = image[:, :, k]
                    grad_image = grad[:, :, j, i]
                    error = fftconvolve(channel, grad_image, "valid")
                    depth_errors.append(error)
                filter_errors.append(np.asarray(depth_errors).T)
            filter_errors = np.asarray(filter_errors).T
            dk += filter_errors

        db = np.sum(grad, axis=3) / n

        self.kernels = np.subtract(self.kernels, alpha * dk)
        self.biases = np.subtract(self.biases, alpha * db)
        end_time = time.time()
        print(f'Convolution backprop time: {end_time - start_time}')

        return grad
