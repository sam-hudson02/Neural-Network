import numpy as np
from typing import Tuple
from enum import Enum


class Optimizers(Enum):
    GRAD = 1
    ADAM = 2


class Optimizer:
    def __init__(self) -> None:
        pass

    def update(self, dw: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray,
                                                              np.ndarray]:
        raise NotImplementedError('update method not implemented')

    def set_learning_rate(self, alpha: float) -> None:
        """Retune the step size mid-run, for a learning rate schedule."""
        raise NotImplementedError('set_learning_rate not implemented')


class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float):
        self.alpha = learning_rate

    def set_learning_rate(self, alpha: float) -> None:
        self.alpha = alpha

    def update(self, dw: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray,
                                                              np.ndarray]:
        dw = self.alpha * dw
        db = self.alpha * db
        return dw, db


class Adam(Optimizer):
    def __init__(self, eta: float = 0.01, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 t: int = 1) -> None:
        self.v_dw, self.m_dw = 0, 0
        self.v_db, self.m_db = 0, 0
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = t

    def set_learning_rate(self, alpha: float) -> None:
        # only eta changes: the moment estimates are what Adam has learned
        # about each parameter's gradient, and resetting them alongside the
        # rate would throw that away at every step of the schedule
        self.eta = alpha

    def update(self, dw: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray,
                                                              np.ndarray]:
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        self.m_db = self.beta1*self.m_db + (1-self.beta1)*db

        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db**2)

        # bias correction
        m_dw_corr = self.m_dw/(1-self.beta1**self.t)
        m_db_corr = self.m_db/(1-self.beta1**self.t)
        v_dw_corr = self.v_dw/(1-self.beta2**self.t)
        v_db_corr = self.v_db/(1-self.beta2**self.t)

        # update weights and biases
        dw = self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        db = self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
        self.t += 1
        return dw, db
