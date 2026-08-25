import math
import numpy as np
from layers.layer import Layer
from utils.dtype import as_dtype
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
        self.loss_history: list[list[float]] = []
        self.accuracy_history: list[list[float]] = []
        self.validation_loss_history: list[float] = []
        self.validation_accuracy_history: list[float] = []
        self.validation_strict_accuracy_history: list[float] = []

    @staticmethod
    def batch_last(a: np.ndarray) -> np.ndarray:
        """
        Move the sample axis from the front to the back, which is the layout
        every layer expects. For 2-D (samples, features) this is a plain
        transpose; for a 4-D (samples, height, width, depth) volume a plain
        .T would also reverse height against width, which silently transposes
        every image and only survives because MNIST happens to be square.
        """
        return np.moveaxis(a, 0, -1)

    def prop(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Run the input forward through every layer.
        :param training: bool(optional): Whether this is the forward half of a
                         learning step. Defaults to False so that validating
                         or predicting never applies dropout.
        """
        # one cast at the door rather than one per layer: numpy promotes
        # float32 @ float64 to float64, so a double precision input would
        # quietly drag every matmul in the stack back up with it
        x = as_dtype(x)
        for layer in self.layers:
            layer.training = training
            x = layer.prop(x)
        if self.softmax:
            x = stable_softmax(x)
        return x

    def back_prop(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        a = self.prop(x, training=True)
        # leave the stack in inference mode so that a layer used directly,
        # outside a prop call, does not silently keep dropping units
        for layer in self.layers:
            layer.training = False
        dy = self.loss_prime(y, a)
        for layer in self.layers_reverse:
            dy = layer.back_prop(dy)
        return a

    def train(self, x: np.ndarray, y: np.ndarray,
              validation: tuple[np.ndarray, np.ndarray] | None = None,
              epochs: int = 500,
              multi: bool = False,
              batch_size: int = 4000) -> list[list[float]]:
        # cast the whole set once here instead of every batch, every epoch
        x, y = as_dtype(x), as_dtype(y)
        n = x.shape[0]
        if batch_size < 1:
            raise ValueError('batch_size must be at least 1')
        if n == 0:
            raise ValueError('no training samples')
        # keep the trailing partial batch instead of dropping those samples
        batches = math.ceil(n / batch_size)
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
                    sl = slice(j * batch_size, (j + 1) * batch_size)
                    x_act = self.batch_last(x[sl])
                    y_act = self.batch_last(y[sl])
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
                    loss_epoch.append(loss)
                    acc_epoch.append(acc)
                    if self.verbose:
                        loss_str = self.str_loss(loss_tot/(j+1))
                        acc_str = self.str_acc(acc_tot/(j+1))
                        if multi:
                            sacc_str = self.str_acc(sacc_tot/(j+1))
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

    def set_learning_rate(self, alpha: float) -> None:
        """
        Set the learning rate on every layer that learns, for use by a
        schedule between epochs. Activation, MaxPool, Reshape and Dropout
        hold no optimizer and are skipped.
        """
        for layer in self.layers:
            optimizer = getattr(layer, 'optimizer', None)
            if optimizer is not None:
                optimizer.set_learning_rate(alpha)

    def average_loss(self) -> list[float]:
        return [float(np.mean(x)) for x in self.loss_history]

    def average_accuracy(self) -> list[float]:
        return [float(np.mean(x)) for x in self.accuracy_history]

    def validate(self, multi: bool,
                 val: tuple[np.ndarray, np.ndarray]) -> tuple[float, float, float]:
        x, y = val
        x, y = self.batch_last(x), self.batch_last(y)
        a = self.prop(x)
        loss = self.loss(y, a)
        if multi:
            acc = self.multi_accuracy(a, y)
            strict_acc = self.strict_accuracy(a, y)
        else:
            acc = self.accuracy(a, y)
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
        with open(f'{path}/arch.json', 'w') as f:
            json.dump(arch, f)
        with open(f'{path}/history.json', 'w') as f:
            json.dump(history, f)

    def open(self, path: str) -> None:
        # history.json postdates the earliest checkpoints, so treat it as
        # optional rather than refusing to load the weights without it
        history_path = f'{path}/history.json'
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            self.loss_history = history.get('loss', [])
            self.accuracy_history = history.get('accuracy', [])
            self.validation_loss_history = history.get('validation_loss', [])
            self.validation_accuracy_history = history.get(
                'validation_accuracy', [])
            self.validation_strict_accuracy_history = history.get(
                'validation_strict_accuracy', [])
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
