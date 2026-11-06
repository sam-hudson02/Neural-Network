from layers.layer import Layer
import numpy as np
from utils.dtype import DTYPE, as_dtype
from utils.optimizer import Optimizer, Optimizers, Adam, GradientDescent


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int,
                 optimizer: Optimizers = Optimizers.ADAM, alpha: float = 0.01):
        self.input_size = input_size
        self.output_size = output_size
        # He initialisation: variance 2 / fan_in keeps activations from
        # growing or vanishing as they pass through a stack of ReLU layers.
        # A flat rand() - 0.5 has variance ~0.083 whatever the fan-in, which
        # is far too wide for a 784-input layer and too narrow for a small one
        self.w = as_dtype(np.random.normal(0.0, np.sqrt(2.0 / input_size),
                                           (output_size, input_size)))
        # biases start at zero; random ones only add noise the network has to
        # learn its way out of
        self.b = np.zeros((output_size, 1), dtype=DTYPE)
        self.input: np.ndarray | None = None
        opt = None
        if optimizer == Optimizers.ADAM:
            opt = Adam(alpha)
        elif optimizer == Optimizers.GRAD:
            opt = GradientDescent(alpha)
        if opt is None:
            raise ValueError('invalid optimizer')
        self.optimizer: Optimizer = opt

    def prop(self, input: np.ndarray) -> np.ndarray:
        self.input = as_dtype(input)
        return np.dot(self.w, self.input) + self.b

    def back_prop(self, grad: np.ndarray) -> np.ndarray:
        if self.input is None:
            raise ValueError('No input data')

        dw = np.dot(grad, self.input.T) / grad.shape[1]
        db = grad.sum(axis=1) / grad.shape[1]
        db = db.reshape(db.shape[0], 1)

        # the gradient w.r.t. the input must use the weights this forward
        # pass actually used, so take it before applying the update
        dx = np.dot(self.w.T, grad)

        u_dw, u_db = self.optimizer.update(dw, db)
        self.w = np.subtract(self.w, u_dw)
        self.b = np.subtract(self.b, u_db)

        return dx

    def save(self, path: str, i: int) -> dict:
        np.save(f'{path}/dense_{i}_w', self.w)
        np.save(f'{path}/dense_{i}_b', self.b)
        return {
            'type': 'Dense',
            'input_size': self.input_size,
            'output_size': self.output_size,
            'w': f'dense_{i}_w.npy',
            'b': f'dense_{i}_b.npy'
        }

    def open(self, path: str, info: dict) -> None:
        w_file = info['w']
        self.w = as_dtype(np.load(f'{path}/{w_file}'))
        b_file = info['b']
        self.b = as_dtype(np.load(f'{path}/{b_file}'))
