"""
Check every layer's analytic gradient against central finite differences.

Run in double precision, or the check is measuring float32 rounding rather
than the gradients:

    PHYS379_DTYPE=float64 uv run python src/check_gradients.py

A central difference is accurate to O(h^2), so with h = 1e-6 the numeric
gradient is good to about 1e-12 relative -- but only if the forward pass it
differences is itself precise to well below that. In float32, where the
forward pass carries about 1e-7 relative error, the difference of two nearby
losses is almost entirely noise and the check reports failures for gradients
that are perfectly correct. This is the one job the float64 setting exists
for.

Every reported figure is a relative error. Anything below about 1e-7 is a
pass; the float64 run comes out around 1e-9 to 1e-10.
"""
import sys

import numpy as np

from layers.activation import Activation as ActivationLayer
from layers.batchnorm import BatchNorm
from layers.convolution import Convolution
from layers.dense import Dense
from utils.activation import ReLU, Sigmoid, Tanh
from utils.dtype import NAME
from utils.optimizer import Optimizer, Optimizers


class Capture(Optimizer):
    """
    Stand-in optimizer that records the gradients and applies no update.

    The layers here update their weights inside back_prop, so a check has to
    intercept the gradient on its way to the optimizer; letting the update
    happen would also move the weights out from under the finite differences.
    """

    def __init__(self):
        self.dw: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def update(self, dw: np.ndarray,
               db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.dw, self.db = dw.copy(), db.copy()
        return np.zeros_like(dw), np.zeros_like(db)

    def set_learning_rate(self, alpha: float) -> None:
        pass


def numeric(loss, a: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """The gradient of loss with respect to a, by central differences."""
    grad = np.zeros_like(a)
    it = np.nditer(a, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        original = a[i]
        a[i] = original + h
        high = loss()
        a[i] = original - h
        low = loss()
        a[i] = original
        grad[i] = (high - low) / (2 * h)
        it.iternext()
    return grad


def relative(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-12))


def report(label: str, *pairs) -> bool:
    """Print one line per layer, and say whether it passed."""
    worst = max(relative(got, want) for _, got, want in pairs)
    detail = '  '.join(f'{name} {relative(got, want):.1e}'
                       for name, got, want in pairs)
    status = 'ok  ' if worst < 1e-7 else 'FAIL'
    print(f'  {status} {label:34} {detail}')
    return worst < 1e-7


def check_convolution(padding, shape=(6, 7, 2), filters=3, kernel=(3, 3),
                      batch=4) -> bool:
    np.random.seed(1)
    layer = Convolution(shape, filters, kernel, Optimizers.GRAD, 0.0,
                        padding=padding)
    captured = Capture()
    layer.optimizer = captured
    x = np.random.randn(*shape, batch)
    seed = np.random.randn(layer.output_height, layer.output_width, filters,
                           batch)

    def loss():
        return float(np.sum(layer.prop(x) * seed))

    layer.prop(x)
    dx = layer.back_prop(seed)
    # the layer averages its parameter gradients over the batch, so the
    # numeric ones have to be divided to match
    label = (f'Convolution padding={padding!r} '
             f'-> {layer.output_height}x{layer.output_width}')
    return report(label,
                  ('dx', dx, numeric(loss, x)),
                  ('dk', captured.dw, numeric(loss, layer.kernels) / batch),
                  ('db', captured.db, numeric(loss, layer.biases) / batch))


def check_batchnorm(shape, channels) -> bool:
    np.random.seed(2)
    layer = BatchNorm(channels, Optimizers.GRAD, 0.0)
    captured = Capture()
    layer.optimizer = captured
    # a non-trivial scale and shift, or the check passes on the identity
    layer.gamma = np.random.randn(channels) * 0.5 + 1
    layer.beta = np.random.randn(channels) * 0.3
    layer.training = True
    x = np.random.randn(*shape) * 2 + 0.5
    seed = np.random.randn(*shape)
    batch = shape[-1]

    def loss():
        return float(np.sum(layer.prop(x) * seed))

    layer.prop(x)
    dx = layer.back_prop(seed)
    return report(f'BatchNorm {shape} c={channels}',
                  ('dx', dx, numeric(loss, x)),
                  ('dgamma', captured.dw, numeric(loss, layer.gamma) / batch),
                  ('dbeta', captured.db, numeric(loss, layer.beta) / batch))


def check_dense(input_size=5, output_size=4, batch=6) -> bool:
    np.random.seed(3)
    layer = Dense(input_size, output_size, Optimizers.GRAD, 0.0)
    captured = Capture()
    layer.optimizer = captured
    x = np.random.randn(input_size, batch)
    seed = np.random.randn(output_size, batch)

    def loss():
        return float(np.sum(layer.prop(x) * seed))

    layer.prop(x)
    dx = layer.back_prop(seed)
    return report(f'Dense({input_size}, {output_size})',
                  ('dx', dx, numeric(loss, x)),
                  ('dw', captured.dw, numeric(loss, layer.w) / batch),
                  ('db', captured.db, numeric(loss, layer.b) / batch))


def check_activation(function) -> bool:
    np.random.seed(4)
    layer = ActivationLayer(function)
    # kept away from zero, where the ReLU kink has no derivative to check
    x = np.random.randn(4, 5) + np.sign(np.random.randn(4, 5)) * 0.5
    seed = np.random.randn(4, 5)

    def loss():
        return float(np.sum(layer.prop(x) * seed))

    layer.prop(x)
    dx = layer.back_prop(seed)
    return report(f'Activation({function.name})',
                  ('dx', dx, numeric(loss, x)))


def check_batchnorm_inference() -> bool:
    """
    Inference must normalise with the running statistics, so that what a
    sample is predicted to be does not depend on which other samples happen
    to share its batch.
    """
    np.random.seed(5)
    layer = BatchNorm(3, Optimizers.GRAD, 0.0, momentum=0.0)
    x = np.random.randn(4, 5, 3, 8) * 2 + 1
    layer.training = True
    batch_output = layer.prop(x)
    layer.training = False
    # momentum 0 means the running statistics are exactly that batch's, so
    # inference should reproduce the training output
    same = np.allclose(layer.prop(x), batch_output, atol=1e-5)
    # and one sample alone must come out as it did with company
    alone = np.allclose(layer.prop(x[:, :, :, :1]), batch_output[:, :, :, :1],
                        atol=1e-5)
    status = 'ok  ' if same and alone else 'FAIL'
    print(f'  {status} {"BatchNorm inference statistics":34} '
          f'batch {same}  single sample {alone}')
    return same and alone


def main() -> int:
    print(f'gradient checks in {NAME}')
    if NAME != 'float64':
        print('  warning: run with PHYS379_DTYPE=float64, or these will fail '
              'on rounding alone')
    passed = [
        check_dense(),
        check_activation(ReLU()),
        check_activation(Sigmoid()),
        check_activation(Tanh()),
        check_convolution('valid'),
        check_convolution('same'),
        check_convolution(1),
        check_convolution((2, 0)),
        check_convolution('same', kernel=(5, 5)),
        check_batchnorm((5, 4, 3, 8), 3),
        check_batchnorm((7, 6), 7),
        check_batchnorm_inference(),
    ]
    print(f'\n{sum(passed)}/{len(passed)} passed')
    return 0 if all(passed) else 1


if __name__ == '__main__':
    sys.exit(main())
