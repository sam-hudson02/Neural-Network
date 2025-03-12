from layers.layer import Layer
from numpy.random import rand
import numpy as np
from utils.optimizer import Optimizer, Optimizers, Adam, GradientDescent


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int,
                 optimizer: Optimizers = Optimizers.ADAM, alpha: float = 0.01):
        self.input_size = input_size
        self.output_size = output_size
        weights = rand(output_size, input_size) - 0.5
        self.w = np.asarray(weights)
        self.b = np.asarray(rand(output_size, 1)) - 0.5
        self.input: np.ndarray | None = None
        opt = None
        if optimizer == Optimizers.ADAM:
            opt = Adam()
        elif optimizer == Optimizers.GRAD:
            opt = GradientDescent(alpha)
        if opt is None:
            raise ValueError('invalid optimizer')
        self.optimizer: Optimizer = opt

    def prop(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        try:
            val = np.dot(self.w, input) + self.b
            return val
        except ValueError:
            self.b = self.b[:, 0].reshape(self.b.shape[0], 1)
            return np.dot(self.w, input) + self.b

    def back_prop(self, grad: np.ndarray) -> np.ndarray:
        if self.input is None:
            raise ValueError('No input data')

        dw = np.dot(grad, self.input.T) / grad.shape[1]
        db = grad.sum(axis=1) / grad.shape[1]
        db = db.reshape(db.shape[0], 1)

        u_dw, u_db = self.optimizer.update(dw, db)

        self.w = np.subtract(self.w, u_dw)
        self.b = np.subtract(self.b, u_db)

        dx = np.dot(self.w.T, grad)
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
        self.w = np.load(f'{path}/{w_file}')
        b_file = info['b']
        self.b = np.load(f'{path}/{b_file}')
