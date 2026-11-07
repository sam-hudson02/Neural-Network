"""
The floating point type the whole library computes in.

float32 is the default. Every serious deep learning library trains in single
precision, for two reasons: BLAS runs sgemm at roughly twice the throughput of
dgemm on the same hardware because it moves half the bytes and fits twice as
much in cache, and the extra precision buys nothing when the gradients are
stochastic estimates in the first place. Weights hold about seven significant
figures in float32, which is far more than the noise floor of a minibatch
gradient.

Set PHYS379_DTYPE=float64 to switch back. That is worth doing for gradient
checks, where central differences need the precision of the baseline to be far
below the step size, and not much else.
"""
import os

import numpy as np

_NAME = os.environ.get('PHYS379_DTYPE', 'float32').lower()
if _NAME not in ('float32', 'float64'):
    raise ValueError(f'PHYS379_DTYPE must be float32 or float64, got {_NAME}')

DTYPE: type = np.float32 if _NAME == 'float32' else np.float64
NAME: str = _NAME


def as_dtype(a: np.ndarray) -> np.ndarray:
    """
    Return a in the working dtype, without copying if it is already there.

    Mixing precisions is silently expensive rather than wrong: numpy promotes
    float32 @ float64 to float64, so a single stray array anywhere in the
    stack drags every downstream matmul back to double precision and undoes
    the speed-up without changing any result enough to notice.
    """
    return np.asarray(a, dtype=DTYPE)


def torch_dtype():
    """The matching torch dtype, imported lazily so torch stays optional."""
    import torch
    return torch.float32 if DTYPE is np.float32 else torch.float64
