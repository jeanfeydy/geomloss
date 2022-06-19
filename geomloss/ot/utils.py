from typing import Tensor

def stable_log(a: Tensor) -> Tensor:
    """Returns the log of the input, with values clamped to -100k to avoid numerical bugs."""
    a_log = a.log()
    a_log[a <= 0] = -100000
    return a_log