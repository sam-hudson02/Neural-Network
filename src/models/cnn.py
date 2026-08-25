"""
The convolution stack both training scripts use, built to a description.

Kept in one place because train_mnist_conv.py and train_math_conv.py want the
same architecture with different widths, and a copy in each is a copy to keep
in step every time the stack changes.
"""
from layers.activation import Activation as ActivationLayer
from layers.batchnorm import BatchNorm
from layers.convolution import Convolution
from layers.dense import Dense
from layers.dropout import Dropout
from layers.layer import Layer
from layers.maxpool import MaxPool
from layers.reshape import Reshape
from utils.activation import ReLU
from utils.optimizer import Optimizers


def build_cnn(classes: int, alpha: float, filters: tuple[int, int] = (6, 12),
              dropout: float = 0.0, padding: str = 'valid',
              batch_norm: bool = False, hidden: int = 128,
              image_size: int = 28, depth: int = 1,
              optimizer: Optimizers = Optimizers.ADAM) -> list[Layer]:
    """
    Three 3x3 convolutions with a max pool after the first two, then a fully
    connected head. With valid padding on a 28x28 input that runs
    28 -> 26 -> 13 -> 11 -> 5 -> 3; with 'same' padding the convolutions leave
    the size alone and only the pooling shrinks it, 28 -> 14 -> 7.

    :param filters: widths of the first convolution and of the two after it.
    :param dropout: rate for the fully connected head, 0 to omit the layer.
                    The head holds the great majority of the weights, so that
                    is where dropout has something to regularise; the
                    convolutions share their few hundred weights across every
                    image position and overfit far less.
    :param padding: 'valid' or 'same'. Valid throws away the border at every
                    convolution, which over three layers costs most of the
                    outermost ring of pixels and leaves a 3x3 map to classify
                    from. 'same' keeps it, giving the head a 7x7 map, about
                    five times as many features, and a correspondingly larger
                    first dense layer.
    :param batch_norm: insert a BatchNorm after each convolution, before its
                       activation. This is where it belongs: the point is to
                       fix the distribution the non-linearity sees, and a ReLU
                       fed inputs that have drifted negative passes nothing on.
    :param hidden: units in the hidden dense layer.
    """
    filters_1, filters_2 = filters
    widths = (filters_1, filters_2, filters_2)
    size = image_size
    layers: list[Layer] = []
    for i, width in enumerate(widths):
        conv = Convolution((size, size, depth), width, (3, 3), optimizer,
                           alpha, padding=padding)
        layers.append(conv)
        size, depth = conv.output_height, width
        if batch_norm:
            layers.append(BatchNorm(width, optimizer, alpha))
        # the last convolution feeds the reshape directly, as it did in
        # final_mnist.py and final_math.py; kept that way so a run with the
        # defaults is comparable with the results already in the report
        if i < len(widths) - 1:
            layers.append(ActivationLayer(ReLU()))
            layers.append(MaxPool(2, 2))
            size = (size - 2) // 2 + 1
    layers.append(Reshape((size, size, depth)))
    layers.append(Dense(size * size * depth, hidden, optimizer, alpha))
    layers.append(ActivationLayer(ReLU()))
    if dropout:
        layers.append(Dropout(dropout))
    layers.append(Dense(hidden, classes, optimizer, alpha))
    return layers


def describe(layers: list[Layer]) -> str:
    """The learned parameter count, for the run banner."""
    import numpy as np
    total = 0
    for layer in layers:
        for name in ('w', 'b', 'kernels', 'biases', 'gamma', 'beta'):
            value = getattr(layer, name, None)
            if isinstance(value, np.ndarray):
                total += value.size
    return f'{len(layers)} layers, {total:,} learned parameters'
