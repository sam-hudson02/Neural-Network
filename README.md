# 379 Machine Learning Project

Neural networks written from scratch on top of numpy: dense, convolution,
batch normalisation, max pooling, dropout and reshape layers, with SGD and
Adam optimisers, learning rate schedules and optional padding. All code is in
the /src folder.

## setup

The project uses [uv](https://docs.astral.sh/uv/). To create the environment
and install everything:

```
uv sync --all-extras
```

That installs the exact versions in `uv.lock`, so everyone gets the same
environment. Then run anything with `uv run`:

```
uv run python src/train_mnist_conv.py
```

### dependency groups

The core dependencies (numpy, pandas, matplotlib, pillow, alive-progress,
torch) are all that the MNIST and maths-symbol experiments need. Two optional
groups cover the rest:

| group    | packages                                | needed by                  |
|----------|-----------------------------------------|----------------------------|
| `scrape` | requests, beautifulsoup4                | `utils/reddit_scrape.py`   |
| `nlp`    | gensim, nltk, sentence-transformers     | `utils/bow.py`, `final_books.py` |

`uv sync` on its own installs just the core; `uv sync --extra nlp` adds one
group; `uv sync --all-extras` adds both. torch is pulled from the PyTorch CPU
index (configured in `pyproject.toml`) so that installing it does not drag in
the whole CUDA stack.

To add a dependency later, use `uv add <package>`, or
`uv add --optional nlp <package>` to put it in a group. Both update
`pyproject.toml` and `uv.lock` and install it.

### without uv

`requirements.txt` is generated from the lock file for anyone not using uv:

```
pip install -r requirements.txt
```

Regenerate it after changing dependencies with:

```
uv export --all-extras --no-hashes --no-emit-project \
    --format requirements-txt -o requirements.txt
```

then re-add the `--extra-index-url` line at the top, which uv does not emit.

## running

Run everything from the repository root, the data and model paths are
relative to it.

Training a CNN, with per-epoch learning rate, validation accuracy, epoch time
and an estimate of the time remaining. Both scripts take `--dropout`, `--lr`,
`--epochs`, `--batch-size`, `--filters`, `--padding`, `--batch-norm`,
`--lr-schedule`, `--save` and `--plots`; pass `--help` for the full list.

```
uv run python src/train_mnist_conv.py --epochs 20
uv run python src/train_math_conv.py --epochs 10
uv run python src/train_mnist_conv.py --epochs 20 --padding same --batch-norm
```

Both build the same stack, from `models/cnn.py`: three 3x3 convolutions with a
max pool after the first two, then a fully connected head.

| option | default | what it does |
|---|---|---|
| `--padding` | `valid` | `valid` discards the border at every convolution, so a 28x28 image reaches the classifier as a 3x3 map. `same` pads instead, keeping the size through the convolutions so only the pooling shrinks it, which leaves a 7x7 map and about five times as many features for the head. Costs roughly 3.5x the time per step. |
| `--batch-norm` | off | Normalises each convolution's output across the batch before its activation, and learns a scale and shift. Starts from a lower loss and tolerates a larger learning rate; adds about 35% to the time per step. |
| `--lr-schedule` | `cosine` | Anneals the rate from `--lr` down to `--lr-min` over the run. `step` multiplies by `--lr-gamma` every `--lr-step` epochs; `constant` reproduces earlier runs. |

Note that `cosine` is the default, so a run is no longer directly comparable
with one from before this was added -- pass `--lr-schedule constant` for that.

The maths script additionally balances the dataset, which is otherwise skewed
1619:1 between its largest and smallest class. `--min-per-class` drops the
rarest classes and `--max-per-class` caps the commonest, defaulting to a
57-class selection of 167,970 images. It splits 80/10/10 with stratification
and only touches the test set when given `--test`. Decoded images are cached,
so only the first run pays the few minutes of jpeg decoding.

The full experiments for each dataset. Each trains its networks (or reloads
them from /models if they are already there) and writes plots to /plots:

```
uv run python src/final_mnist.py
uv run python src/final_math.py
uv run python src/final_books.py
```

`src/main.py` is a scratchpad of one-off experiments rather than an entry
point; several of its functions need datasets that are not in the repo.
