from layers.layer import Layer
from typing import Tuple
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
from utils.dtype import DTYPE, as_dtype
from utils.optimizer import Optimizer, Optimizers, Adam, GradientDescent


class Convolution(Layer):
    """
    Convolution (strictly, cross-correlation) over a stack of images.

    Layers exchange volumes as (height, width, depth, batch). Internally the
    batch comes first, (batch, depth, height, width), and each forward pass is
    a single matrix multiply: the input patches are unrolled into a matrix
    (im2col) and multiplied by the flattened kernels. That is the same
    arithmetic as correlating each channel with each filter one pair at a
    time, but it hands the whole batch to BLAS instead of making
    batch * filters * depth separate scipy calls.
    """

    _TO_INNER = (3, 2, 0, 1)
    _TO_OUTER = (2, 3, 1, 0)

    def __init__(self, input_shape: Tuple[int, int, int], filters: int,
                 kernel_size: Tuple[int, int],
                 optimizer: Optimizers = Optimizers.ADAM,
                 alpha: float = 0.01,
                 padding: str | int | Tuple[int, int] = 'valid'):
        """
        :param padding: 'valid' for no padding, which is the default and what
                        every model in this repository was trained with;
                        'same' to ring the input with enough zeros that the
                        output keeps the input's height and width; or an
                        explicit number of rows and columns to add on each
                        side.
        """
        self.input_shape = input_shape
        self.height, self.width, self.depth = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_height, self.kernel_width = kernel_size
        self.padding = padding
        self.pad_height, self.pad_width = self._resolve_padding(padding)
        self.output_height = (self.height + 2 * self.pad_height
                              - self.kernel_height + 1)
        self.output_width = (self.width + 2 * self.pad_width
                             - self.kernel_width + 1)
        if self.output_height < 1 or self.output_width < 1:
            raise ValueError(
                f'kernel {kernel_size} with padding {padding} leaves an '
                f'output of {self.output_height}x{self.output_width} for an '
                f'input of {self.height}x{self.width}')

        # He initialisation: a ReLU zeroes half its inputs, so the surviving
        # weights need a variance of 2 / fan_in to keep the activation scale
        # steady as it passes through the stack
        fan_in = self.depth * self.kernel_height * self.kernel_width
        self.kernels = as_dtype(np.random.normal(
            0.0, np.sqrt(2.0 / fan_in),
            (self.filters, self.depth, self.kernel_height, self.kernel_width)))
        # one bias per filter, shared across every output position. A bias per
        # position would let the layer learn "this pixel is usually bright",
        # which destroys the translation invariance that makes a convolution
        # worth using and costs filters * out_h * out_w parameters instead of
        # filters
        self.biases = np.zeros((self.filters, 1, 1), dtype=DTYPE)

        self.input: np.ndarray | None = None
        self.output: np.ndarray | None = None
        opt = None
        if optimizer == Optimizers.ADAM:
            opt = Adam(alpha)
        elif optimizer == Optimizers.GRAD:
            opt = GradientDescent(alpha)
        if opt is None:
            raise ValueError('invalid optimizer')
        self.optimizer: Optimizer = opt

    def _resolve_padding(self, padding: str | int | Tuple[int, int]
                         ) -> Tuple[int, int]:
        """Turn the padding argument into rows and columns per side."""
        if isinstance(padding, str):
            if padding == 'valid':
                return 0, 0
            if padding == 'same':
                if self.kernel_height % 2 == 0 or self.kernel_width % 2 == 0:
                    # an even kernel needs one more row on one side than the
                    # other, and asymmetric padding is not worth the
                    # complication for kernels nothing here uses
                    raise ValueError("'same' padding needs an odd kernel, "
                                     f'got {self.kernel_size}')
                return (self.kernel_height - 1) // 2, (self.kernel_width - 1) // 2
            raise ValueError(f"padding must be 'valid', 'same', or a number "
                             f'of pixels, got {padding!r}')
        if isinstance(padding, int):
            pair = (padding, padding)
        else:
            pair = tuple(padding)
        if len(pair) != 2 or any(p < 0 for p in pair):
            raise ValueError(f'padding must be two non-negative numbers, '
                             f'got {padding!r}')
        return pair

    def _pad(self, inner: np.ndarray) -> np.ndarray:
        """Ring a (batch, depth, h, w) volume with zeros."""
        if not (self.pad_height or self.pad_width):
            return inner
        return np.pad(inner, ((0, 0), (0, 0),
                              (self.pad_height, self.pad_height),
                              (self.pad_width, self.pad_width)))

    def _columns(self, inner: np.ndarray) -> np.ndarray:
        """
        (batch, depth, h, w) -> (batch, depth * kh * kw, out_h * out_w), one
        column per output pixel holding the patch that produced it. The input
        must already be padded.
        """
        patches = sliding_window_view(
            inner, (self.kernel_height, self.kernel_width), axis=(2, 3))
        patches = patches.transpose(0, 1, 4, 5, 2, 3)
        return patches.reshape(inner.shape[0], -1,
                               self.output_height * self.output_width)

    def _inner(self, outer: np.ndarray) -> np.ndarray:
        """(h, w, depth, batch) -> a contiguous (batch, depth, h, w) copy."""
        return np.ascontiguousarray(as_dtype(outer).transpose(self._TO_INNER))

    def prop(self, input: np.ndarray) -> np.ndarray:
        if input.shape[:3] != tuple(self.input_shape):
            raise ValueError(f'expected input of shape {self.input_shape} + '
                             f'(batch,), got {input.shape}')
        self.input = input
        cols = self._columns(self._pad(self._inner(input)))
        kernels = self.kernels.reshape(self.filters, -1)
        out = (kernels @ cols).reshape(-1, self.filters, self.output_height,
                                       self.output_width)
        out += self.biases
        self.output = out.transpose(self._TO_OUTER)
        if self.output is None:
            raise ValueError('self.output is None')
        return self.output

    def back_prop(self, grad: np.ndarray) -> np.ndarray:
        if self.input is None:
            raise ValueError('self.input is None')

        padded = self._pad(self._inner(self.input))
        grad_t = self._inner(grad)
        batch = grad_t.shape[0]
        # rebuilt rather than cached from prop: holding the unrolled patches
        # across a forward-only pass (validation) would pin a lot of memory
        cols = self._columns(padded)
        flat_grad = grad_t.reshape(batch, self.filters, -1)
        kernels = self.kernels.reshape(self.filters, -1)

        dk = np.einsum('bfp,bcp->fc', flat_grad, cols,
                       optimize=True).reshape(self.kernels.shape) / batch
        # each bias is shared across every output position, so its gradient
        # collects from all of them as well as from the batch
        db = grad_t.sum(axis=(0, 2, 3)).reshape(self.filters, 1, 1) / batch

        # col2im: scatter each patch's share of the gradient back to the
        # pixels it came from, one kernel offset at a time
        dcols = (kernels.T @ flat_grad).reshape(
            batch, self.depth, self.kernel_height, self.kernel_width,
            self.output_height, self.output_width)
        dx = np.zeros(padded.shape, dtype=DTYPE)
        for i in range(self.kernel_height):
            for j in range(self.kernel_width):
                dx[:, :, i:i + self.output_height,
                   j:j + self.output_width] += dcols[:, :, i, j]
        if self.pad_height or self.pad_width:
            # the zeros that were added have no upstream layer to send a
            # gradient to, so drop the border back off
            dx = dx[:, :, self.pad_height:self.pad_height + self.height,
                    self.pad_width:self.pad_width + self.width]

        u_dk, u_db = self.optimizer.update(dk, db)
        self.kernels -= u_dk
        self.biases -= u_db

        return dx.transpose(self._TO_OUTER)

    def save(self, path: str, i: int) -> dict:
        np.save(f'{path}/convolution_{i}_k', self.kernels)
        np.save(f'{path}/convolution_{i}_b', self.biases)
        return {
            'type': 'Convolution',
            'input_shape': self.input_shape,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'k': f'convolution_{i}_k.npy',
            'b': f'convolution_{i}_b.npy'
        }

    def open(self, path: str, info: dict) -> None:
        k_file = info['k']
        self.kernels = as_dtype(np.load(f'{path}/{k_file}'))
        b_file = info['b']
        biases = as_dtype(np.load(f'{path}/{b_file}'))
        if biases.shape != self.biases.shape:
            # checkpoints written before the bias became per-filter hold one
            # value per output position; average them so the layer stays
            # consistent instead of silently keeping the old shape
            print(f'note: {b_file} has per-position biases {biases.shape}, '
                  f'averaging to {self.biases.shape}')
            biases = biases.mean(axis=(1, 2)).reshape(self.filters, 1, 1)
        self.biases = biases
        saved = info.get('padding', 'valid')
        if saved != self.padding:
            print(f'note: {k_file} was trained with padding {saved!r}, this '
                  f'layer is built with {self.padding!r}')
