from time import time
import numpy as np
from layers.layer import Layer
from utils.utils import stable_softmax, pad
from typing import Callable
import os
import json
from alive_progress import alive_bar


class Network:
    def __init__(self, layers: list[Layer], softmax: bool,
                 loss: Callable[[np.ndarray, np.ndarray], float],
                 loss_prime: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 verbose: bool = False, name: str = 'network'):
        self.name: str = name
        self.layers: list[Layer] = layers
        self.layers_reverse: list[Layer] = layers[::-1]
        self.softmax: bool = softmax
        self.loss: Callable[[np.ndarray, np.ndarray], float] = loss
        self.loss_prime: Callable[[np.ndarray,
                                   np.ndarray], np.ndarray] = loss_prime
        self.verbose: bool = verbose
        self.last_update = time()
        self.loss_history: list[list[float]] = []
        self.accuracy_history: list[list[float]] = []
        self.validation_loss_history: list[float] = []
        self.validation_accuracy_history: list[float] = []
        self.validation_strict_accuracy_history: list[float] = []

    def prop(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.prop(x)
        if self.softmax:
            x = stable_softmax(x)
        return x

    def back_prop(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        a = self.prop(x)
        dy = self.loss_prime(y, a)
        for layer in self.layers_reverse:
            dy = layer.back_prop(dy)
        return a

    def train(self, x: np.ndarray, y: np.ndarray,
              validation: tuple[np.ndarray, np.ndarray] | None = None,
              epochs: int = 500,
              multi: bool = False,
              batch_size: int = 4000) -> list[list[float]]:
        n = x.shape[0]
        batches = n // batch_size
        a = 0
        for i in range(epochs):
            with alive_bar(batches, length=13, spinner=None,
                           receipt_text=True) as bar:
                bar.title = f'Epoch {i+1}/{epochs}'
                loss_epoch = []
                acc_epoch = []
                acc_tot = 0
                loss_tot = 0
                sacc_tot = 0
                for j in range(batches):
                    x_act = x[j * batch_size:(j + 1) * batch_size].T
                    y_act = y[j * batch_size:(j + 1) * batch_size].T
                    a = self.back_prop(x_act, y_act)
                    loss = self.loss(y_act, a)
                    if multi:
                        acc = self.multi_accuracy(a, y_act)
                        sacc = self.strict_accuracy(a, y_act)
                    else:
                        acc = self.accuracy(a, y_act)
                        sacc = 0
                    acc_tot += acc
                    loss_tot += loss
                    sacc_tot += sacc
                    loss_str = self.str_loss(loss_tot/(j+1))
                    acc_str = self.str_acc(acc_tot/(j+1))
                    sacc_str = self.str_acc(sacc_tot/(j+1))
                    loss_epoch.append((self.loss(y_act, a)))
                    acc_epoch.append(self.accuracy(a, y_act))
                    if self.verbose:
                        if multi:
                            bar.text(f'{loss_str} {acc_str} {sacc_str}')
                        else:
                            bar.text(f'{loss_str} {acc_str}')
                        bar()

                self.loss_history.append(loss_epoch)
                self.accuracy_history.append(acc_epoch)

                if validation is not None:
                    v_loss, v_acc, v_sacc = self.validate(multi, validation)
                    self.validation_loss_history.append(v_loss)
                    self.validation_accuracy_history.append(v_acc)
                    self.validation_strict_accuracy_history.append(v_sacc)
                    loss_str = self.str_loss(v_loss)
                    acc_str = self.str_acc(v_acc)
                    sacc_str = self.str_acc(v_sacc)
                    if multi:
                        text = f'{loss_str} {acc_str} {sacc_str}'
                    else:
                        text = f'{loss_str} {acc_str}'
                    bar.text(text)
            x, y = self.shuffle_data(x, y)
        return self.loss_history

    def average_loss(self) -> list[float]:
        return [float(np.mean(x)) for x in self.loss_history]

    def average_accuracy(self) -> list[float]:
        return [float(np.mean(x)) for x in self.accuracy_history]

    def validate(self, multi: bool,
                 val: tuple[np.ndarray, np.ndarray]) -> tuple[float, float, float]:
        x, y = val
        a = self.prop(x.T)
        loss = self.loss(y.T, a)
        if multi:
            acc = self.multi_accuracy(a, y.T)
            strict_acc = self.strict_accuracy(a, y.T)
        else:
            acc = self.accuracy(a, y.T)
            strict_acc = 0
        return loss, acc, strict_acc

    def shuffle_data(self, x: np.ndarray,
                     y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.random.permutation(x.shape[0])
        return x[p], y[p]

    def save(self, path: str) -> None:
        arch = {}
        history = {'loss': self.loss_history,
                   'accuracy': self.accuracy_history,
                   'validation_loss': self.validation_loss_history,
                   'validation_accuracy': self.validation_accuracy_history,
                   'validation_strict_accuracy': self.validation_strict_accuracy_history}
        if not os.path.exists(path):
            os.makedirs(path)
        for i, layer in enumerate(self.layers):
            arch[i] = layer.save(path, i)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f'{path}/arch.json', 'w') as f:
            json.dump(arch, f)
        with open(f'{path}/history.json', 'w') as f:
            json.dump(history, f)

    def open(self, path: str) -> None:
        with open(f'{path}/history.json', 'r') as f:
            history = json.load(f)
            self.loss_history = history['loss']
            self.accuracy_history = history['accuracy']
            self.validation_loss_history = history['validation_loss']
            self.validation_accuracy_history = history['validation_accuracy']
        with open(f'{path}/arch.json', 'r') as f:
            arch = json.load(f)
        for i, info in arch.items():
            self.layers[int(i)].open(path, info)

    def accuracy(self, a: np.ndarray, y: np.ndarray) -> float:
        predictions = np.argmax(a, axis=0)
        correct = np.sum(predictions == np.argmax(y, axis=0))
        return correct / y.shape[1]

    def multi_accuracy(self, a: np.ndarray, y: np.ndarray) -> float:
        # round to 1 or 0
        predictions = a.T.round()
        correct = np.sum(predictions == y.T)
        return correct / (y.shape[1] * y.shape[0])

    def strict_accuracy(self, a: np.ndarray, y: np.ndarray) -> float:
        predictions = a.T.round().astype(int)
        return np.mean(np.all(predictions == y.T, axis=1))

    def str_acc(self, acc: float) -> str:
        return f'acc: {pad(round(acc, 4), 5)}'

    def str_loss(self, loss: float) -> str:
        return f'loss: {pad(round(loss, 4), 5)}'
