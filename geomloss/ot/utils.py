from typing import RealTensor


def stable_log(a: RealTensor) -> RealTensor:
    """Returns the log of the input, with values clamped to -100k to avoid numerical bugs."""
    a_log = a.log()
    a_log[a <= 0] = -100000
    return a_log


def dot_products(a: RealTensor, f: RealTensor) -> RealTensor:
    """Performs a batchwise computation of dot products."""
    assert a.shape == f.shape
    B = a.shape[0]
    return (a.reshape(B, -1) * f.reshape(B, -1)).sum(1)
