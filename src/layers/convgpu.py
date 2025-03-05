from layers.layer import Layer
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch


class ConvGPU(Layer):
    def __init__(self, input_shape: tuple[int, int, int], out_c: int,
                 kernel_size: tuple[int, int],
                 stride: int = 1,
                 alpha: float = 0.01):
        self.out_channels = out_c
        self.in_channels = input_shape[2]
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias: torch.Tensor = torch.randn(out_c, requires_grad=True)
        self.weights: torch.Tensor = torch.randn(
            out_c, self.in_channels, kernel_size[0], kernel_size[1],
            requires_grad=True)
        self.conv: torch.Tensor | None = None
        self.optimizer = optim.Adam([self.weights, self.bias])

    def prop(self, input: np.ndarray) -> np.ndarray:
        self.input = input.T.astype(np.float32)
        self.x = torch.tensor(self.input, requires_grad=True)
        self.conv = F.conv2d(self.x, self.weights, self.bias, self.stride)
        out = self.conv.detach().numpy().T
        return out

    def back_prop(self, grad: np.ndarray, alpha: float) -> np.ndarray:
        if self.x is None:
            raise ValueError('self.x is None')
        if self.conv is None:
            raise ValueError('self.conv is None')
        self.optimizer.zero_grad()

        self.conv.backward(torch.tensor(grad.T, dtype=torch.float32))
        if self.x.grad is None:
            raise ValueError('self.x.grad is None')

        self.optimizer.step()
        out = self.x.grad.detach().numpy().T
        return out
