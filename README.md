# Neural Network

A small neural network library written from scratch on top of numpy. It
started as the PHYS379 machine learning project at university and is being
turned into a standalone library; the experiment scripts and datasets from
that project have been removed, and the write-up is kept in `paper/`.

What it has:

- **Layers**: dense, convolution (valid, same or explicit padding), batch
  normalisation, max pooling, dropout, reshape, activation and softmax
- **Activations**: ReLU, sigmoid, tanh, softmax
- **Losses**: mean squared error, categorical cross-entropy (with a fused
  softmax gradient), binary cross-entropy
- **Optimisers**: SGD and Adam, with constant, cosine and step learning rate
  schedules
- **Training**: minibatching, shuffling, validation each epoch, loss and
  accuracy history, and saving/loading weights to a directory
- **A gradient checker** that tests every layer's backward pass against
  finite differences

The forward and backward passes are plain numpy. torch is used only for max
pooling and the optional `ConvGPU` layer.

## status

This is a work in progress. The code still lives as top-level modules under
`src/` (`layers`, `models`, `utils`, `plots`) rather than as an importable
package, so for now `src` has to be on the path:

```
PYTHONPATH=src uv run python your_script.py
```

The package name in `pyproject.toml` is still `phys379`, and the `scrape` and
`nlp` optional dependency groups belong to experiments that are no longer
here.

## setup

The project uses [uv](https://docs.astral.sh/uv/):

```
uv sync
```

That installs the exact versions in `uv.lock`. torch comes from the PyTorch
CPU index (configured in `pyproject.toml`) so it does not pull in the CUDA
stack.

Without uv, `requirements.txt` is generated from the lock file:

```
pip install -r requirements.txt
```

Regenerate it after changing dependencies with:

```
uv export --all-extras --no-hashes --no-emit-project \
    --format requirements-txt -o requirements.txt
```

then re-add the `--extra-index-url` line at the top, which uv does not emit.

## usage

A network is a list of layers plus a loss and its derivative. Inputs and
labels go in samples-first, `(samples, features)` or
`(samples, height, width, channels)`:

```python
import numpy as np
from layers.activation import Activation
from layers.dense import Dense
from models.nn import Network
from utils.activation import ReLU
from utils.optimizer import Optimizers
from utils.utils import cce, cce_softmax_prime, one_hot

x = np.random.rand(1000, 4)
y = one_hot((x.sum(axis=1) > 2).astype(int), 2).T  # one_hot is classes-first

net = Network(
    layers=[
        Dense(4, 16, Optimizers.ADAM, alpha=0.01),
        Activation(ReLU()),
        Dense(16, 2, Optimizers.ADAM, alpha=0.01),
    ],
    softmax=True,
    loss=cce,
    loss_prime=cce_softmax_prime,
)
net.train(x, y, validation=(x[:200], y[:200]), epochs=5, batch_size=64)

net.save('models/example')   # weights as .npy, plus arch.json and history.json
net.open('models/example')   # load into a network with the same layers
```

Internally every layer works batch-last, so to call `prop` yourself, move the
sample axis to the end first: `net.prop(Network.batch_last(x))`, which returns
`(classes, samples)`.

### convolutional networks

`models/cnn.py` has `build_cnn`, which builds three 3x3 convolutions with a
max pool after the first two and a fully connected head:

```python
from models.cnn import build_cnn

layers = build_cnn(classes=10, alpha=1e-3, padding='same', batch_norm=True,
                   dropout=0.3)
```

| option | default | what it does |
|---|---|---|
| `padding` | `valid` | `valid` discards the border at every convolution, so a 28x28 image reaches the head as a 3x3 map. `same` keeps the size through the convolutions so only pooling shrinks it, leaving a 7x7 map and about five times as many features. Costs roughly 3.5x the time per step. |
| `batch_norm` | off | Normalises each convolution's output before its activation and learns a scale and shift. Starts from a lower loss and tolerates a larger learning rate; adds about 35% to the time per step. |
| `dropout` | 0 | Dropout rate in the fully connected head, where most of the weights are. |

### learning rate schedules

`utils/schedule.py` gives the rate for each epoch. Apply it between epochs
with `Network.set_learning_rate`:

```python
from utils.schedule import learning_rate

for epoch in range(epochs):
    net.set_learning_rate(learning_rate(1e-3, epoch, epochs, 'cosine'))
    net.train(x, y, validation=val, epochs=1)
```

### precision

Everything computes in float32 by default. Set `PHYS379_DTYPE=float64` to
switch to double precision, which is mainly useful for gradient checks.

## gradient check

```
PHYS379_DTYPE=float64 uv run python src/check_gradients.py
```

Compares each layer's analytic gradients with central finite differences and
prints the relative error; anything below about 1e-7 passes.

## notes and paper

- `notes/layers/` -- the maths behind the forward and backward pass of each
  layer
- `notes/classifier/` -- notes on the original MNIST classifier
- `paper/` -- the PHYS379 project report
