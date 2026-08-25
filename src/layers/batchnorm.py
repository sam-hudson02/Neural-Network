from layers.layer import Layer
import numpy as np
from utils.dtype import DTYPE, as_dtype
from utils.optimizer import Optimizer, Optimizers, Adam, GradientDescent


class BatchNorm(Layer):
    """
    Batch normalisation for either a convolution volume or a dense vector.
    """

    def __init__(self, channels: int,
                 optimizer: Optimizers = Optimizers.ADAM,
                 alpha: float = 0.01, momentum: float = 0.9,
                 epsilon: float = 1e-5):
        """
        :param channels: number of filters for a convolution volume, or of
                         units for a dense vector.
        :param momentum: weight kept from the existing running statistics at
                         each update. 0.9 averages over roughly the last ten
                         batches.
        :param epsilon: guard against dividing by the standard deviation of a
                        channel that is constant across the batch. 1e-5 rather
                        than the 1e-8 used elsewhere because in float32 a
                        smaller value is lost against a variance of order one.
        """
        if channels < 1:
            raise ValueError(f'channels must be positive, got {channels}')
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f'momentum must be in [0, 1), got {momentum}')
        self.channels = channels
        self.momentum = momentum
        self.epsilon = epsilon

        # start as the identity, so inserting the layer does not disturb a
        # network that was working without it
        self.gamma = np.ones(channels, dtype=DTYPE)
        self.beta = np.zeros(channels, dtype=DTYPE)
        # the inference statistics, matching what an untrained layer's inputs
        # would be after He initialisation
        self.running_mean = np.zeros(channels, dtype=DTYPE)
        self.running_var = np.ones(channels, dtype=DTYPE)

        self.x_hat: np.ndarray | None = None
        self.std: np.ndarray | None = None

        opt = None
        if optimizer == Optimizers.ADAM:
            opt = Adam(alpha)
        elif optimizer == Optimizers.GRAD:
            opt = GradientDescent(alpha)
        if opt is None:
            raise ValueError('invalid optimizer')
        self.optimizer: Optimizer = opt

    def _axes(self, ndim: int) -> tuple:
        """The axes to pool statistics over, leaving the channel axis alone."""
        if ndim == 4:
            # (height, width, depth, batch)
            return 0, 1, 3
        if ndim == 2:
            # (features, batch)
            return 1,
        raise ValueError(f'BatchNorm expects a 2-D or 4-D volume, got {ndim} '
                         f'dimensions')

    def _shape(self, ndim: int) -> tuple:
        """Per-channel parameters broadcast against an input of this rank."""
        return (1, 1, self.channels, 1) if ndim == 4 else (self.channels, 1)

    def prop(self, input: np.ndarray) -> np.ndarray:
        input = as_dtype(input)
        axes = self._axes(input.ndim)
        shape = self._shape(input.ndim)
        channel_axis = 2 if input.ndim == 4 else 0
        if input.shape[channel_axis] != self.channels:
            raise ValueError(f'BatchNorm({self.channels}) got an input with '
                             f'{input.shape[channel_axis]} channels')

        if self.training:
            mean = input.mean(axis=axes, keepdims=True)
            var = input.var(axis=axes, keepdims=True)
            # the running statistics only ever feed inference, so they are
            # updated here and never take part in the gradient
            keep = self.momentum
            self.running_mean = (keep * self.running_mean
                                 + (1 - keep) * mean.reshape(-1))
            self.running_var = (keep * self.running_var
                                + (1 - keep) * var.reshape(-1))
        else:
            mean = self.running_mean.reshape(shape)
            var = self.running_var.reshape(shape)

        self.std = np.sqrt(var + self.epsilon)
        self.x_hat = (input - mean) / self.std
        return self.gamma.reshape(shape) * self.x_hat + self.beta.reshape(shape)

    def back_prop(self, grad: np.ndarray) -> np.ndarray:
        if self.x_hat is None or self.std is None:
            raise ValueError('back_prop before prop')
        grad = as_dtype(grad)
        axes = self._axes(grad.ndim)
        shape = self._shape(grad.ndim)
        batch = grad.shape[-1]

        d_gamma = np.sum(grad * self.x_hat, axis=axes).reshape(-1) / batch
        d_beta = np.sum(grad, axis=axes).reshape(-1) / batch

        # the mean and variance are functions of every sample in the batch
        d_x_hat = grad * self.gamma.reshape(shape)
        dx = (d_x_hat
              - d_x_hat.mean(axis=axes, keepdims=True)
              - self.x_hat * (d_x_hat * self.x_hat).mean(axis=axes,
                                                         keepdims=True))
        dx /= self.std

        u_gamma, u_beta = self.optimizer.update(d_gamma, d_beta)
        self.gamma -= u_gamma
        self.beta -= u_beta

        return dx

    def save(self, path: str, i: int) -> dict:
        np.save(f'{path}/batchnorm_{i}_g', self.gamma)
        np.save(f'{path}/batchnorm_{i}_b', self.beta)
        np.save(f'{path}/batchnorm_{i}_rm', self.running_mean)
        np.save(f'{path}/batchnorm_{i}_rv', self.running_var)
        return {
            'type': 'BatchNorm',
            'channels': self.channels,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'g': f'batchnorm_{i}_g.npy',
            'b': f'batchnorm_{i}_b.npy',
            'rm': f'batchnorm_{i}_rm.npy',
            'rv': f'batchnorm_{i}_rv.npy'
        }

    def open(self, path: str, info: dict) -> None:
        self.gamma = as_dtype(np.load(f'{path}/{info["g"]}'))
        self.beta = as_dtype(np.load(f'{path}/{info["b"]}'))
        # without the running statistics the layer would normalise a
        # prediction against zero mean and unit variance, which is not what it
        # was trained against, so these are as much a parameter as gamma is
        self.running_mean = as_dtype(np.load(f'{path}/{info["rm"]}'))
        self.running_var = as_dtype(np.load(f'{path}/{info["rv"]}'))
        self.epsilon = info.get('epsilon', self.epsilon)
        self.momentum = info.get('momentum', self.momentum)
