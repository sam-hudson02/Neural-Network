
from typing import Callable

def const_alpha(alpha: float) -> Callable[[int], float]:
    def optimized(epoch: int):
        return alpha
    return optimized 

def step_alpha(start_alpha: float, delta: float, step: int) -> Callable[[int], float]:
    def optimized(epoch: int):
        return start_alpha - delta * (epoch // step)
    return optimized

def exp_alpha(start_alpha: float, decay: float) -> Callable[[int], float]:
    def optimized(epoch: int):
        return start_alpha * (decay ** epoch)
    return optimized
