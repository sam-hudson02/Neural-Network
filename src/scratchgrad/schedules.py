"""
Learning rate schedules.

Adam adapts the size of each step to that parameter's own gradient history,
but it does not know when training is nearly over. The global rate still has
to come down: a rate large enough to cross the loss surface quickly at the
start is too large to settle into a minimum at the end, and the usual symptom
is a validation curve that stops improving and then bounces around for the
rest of the run.
"""
import math

SCHEDULES = ('constant', 'cosine', 'step')


def learning_rate(base: float, epoch: int, epochs: int,
                  schedule: str = 'cosine', minimum: float = 0.0,
                  step: int = 5, gamma: float = 0.5) -> float:
    """
    The learning rate to train a given epoch at.

    :param base: the rate the run starts from.
    :param epoch: which epoch is about to be trained, counting from 1.
    :param epochs: total epochs in the run, which the schedules stretch
                   themselves across.
    :param schedule: 'constant' leaves the rate alone. 'cosine' follows a half
                     cosine from base down to minimum, spending a long time
                     near the top, falling fastest in the middle and flattening
                     out at the end, which is what lets it fine-tune without
                     an abrupt change of regime. 'step' multiplies by gamma
                     every step epochs, which is cruder but easier to describe
                     in a report.
    :param minimum: the floor for 'cosine'.
    :param step: epochs between drops, for 'step'.
    :param gamma: the factor applied at each drop, for 'step'.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f'schedule must be one of {SCHEDULES}, '
                         f'got {schedule!r}')
    if epoch < 1 or epochs < 1:
        raise ValueError('epoch and epochs must be at least 1')
    if schedule == 'constant':
        return base
    if schedule == 'step':
        return base * gamma ** ((epoch - 1) // step)
    # divide by epochs rather than epochs - 1 so that the last epoch still
    # trains at a small positive rate instead of exactly the floor
    progress = (epoch - 1) / epochs
    return minimum + (base - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def describe(base: float, epochs: int, schedule: str, minimum: float,
             step: int, gamma: float) -> str:
    """A one-line summary of where the rate starts and ends."""
    if schedule == 'constant':
        return f'constant lr {base:g}'
    last = learning_rate(base, epochs, epochs, schedule, minimum, step, gamma)
    detail = f'every {step} epochs x{gamma:g}' if schedule == 'step' \
        else f'floor {minimum:g}'
    return f'{schedule} lr {base:g} -> {last:.2g} ({detail})'
