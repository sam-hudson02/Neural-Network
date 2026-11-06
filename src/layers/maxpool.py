from layers.layer import Layer
import torch.nn.functional as F
import torch
import numpy as np
from utils.dtype import as_dtype, torch_dtype


class MaxPool(Layer):
    """
    Max pooling over volumes laid out as (height, width, depth, batch), which
    is the layout Convolution and Reshape use. torch does the pooling and the
    gradient routing; it only ever sees the (batch, depth, h, w) view it wants.
    """

    _TO_INNER = (3, 2, 0, 1)
    _TO_OUTER = (2, 3, 1, 0)

    def __init__(self, pool_size: int, stride: int):
        self.pool_size = pool_size
        self.stride = stride
        self.input: np.ndarray | None = None
        self.x: torch.Tensor | None = None
        self.pool: torch.Tensor | None = None

    def prop(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        inner = np.ascontiguousarray(as_dtype(input).transpose(self._TO_INNER))
        self.x = torch.tensor(inner, dtype=torch_dtype(), requires_grad=True)
        self.pool = F.max_pool2d(self.x, self.pool_size, self.stride)
        out = self.pool.detach().numpy()
        return out.transpose(self._TO_OUTER)

    def back_prop(self, grad: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise ValueError('self.x is None')
        if self.pool is None:
            raise ValueError('self.pool is None')
        inner = np.ascontiguousarray(as_dtype(grad).transpose(self._TO_INNER))
        # x is rebuilt every forward pass, so its .grad starts empty and
        # matching the dtype of x keeps torch from rejecting the seed
        self.pool.backward(torch.tensor(inner, dtype=torch_dtype()))
        if self.x.grad is None:
            raise ValueError('self.x.grad is None')
        out = self.x.grad.detach().numpy()
        return out.transpose(self._TO_OUTER)

    def save(self, path: str, i: int) -> dict:
        return {
            'type': 'MaxPool',
            'pool_size': self.pool_size,
            'stride': self.stride
        }
