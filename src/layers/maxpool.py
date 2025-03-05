from layers.layer import Layer
import torch.nn.functional as F
import torch
import numpy as np


class MaxPool(Layer):
    def __init__(self, pool_size: int, stride: int):
        self.pool_size = pool_size
        self.stride = stride
        self.input: np.ndarray | None = None
        self.x: torch.Tensor | None = None
        self.pool: torch.Tensor | None = None

    def prop(self, input: np.ndarray) -> np.ndarray:
        self.input = input.T
        self.x = torch.tensor(self.input, requires_grad=True)
        self.pool = F.max_pool2d(self.x, self.pool_size, self.stride)
        out = self.pool.detach().numpy().T
        return out

    def back_prop(self, grad: np.ndarray, alpha: float) -> np.ndarray:
        if self.x is None:
            raise ValueError('self.x is None')
        if self.pool is None:
            raise ValueError('self.pool is None')
        self.pool.backward(torch.tensor(grad.T, dtype=torch.float32))
        if self.x.grad is None:
            raise ValueError('self.x.grad is None')
        out = self.x.grad.detach().numpy().T
        return out

    def save(self, path: str, i: int) -> dict:
        return {
            'type': 'MaxPool',
            'pool_size': self.pool_size,
            'stride': self.stride
        }
