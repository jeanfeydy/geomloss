import numpy as np
from scipy import special

from ..typing import RealTensor


def device(a: RealTensor):
    return "cpu"


def dtype(a: RealTensor):
    return a.dtype


def abs(a: RealTensor) -> RealTensor:
    return np.abs(a)


def exp(a: RealTensor) -> RealTensor:
    return np.exp(a)


def stable_log(a: RealTensor) -> RealTensor:
    """Returns the log of the input, with values clamped to -100k to avoid numerical bugs."""
    a_log = np.log(a)
    a_log[a <= 0] = -100000
    return a_log


def dot_products(a: RealTensor, f: RealTensor) -> RealTensor:
    """Performs a batchwise computation of dot products."""
    assert a.shape == f.shape
    B = a.shape[0]
    return np.sum(a.reshape(B, -1) * f.reshape(B, -1), axis=1)


def norm(a: RealTensor) -> RealTensor:
    return np.sqrt(np.sum(a**2))


def einsum(formula, *args):
    return np.einsum(formula, *args)


def any(x, axis=None, keepdims=False):
    return np.any(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    return np.sum(x, axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    return np.mean(x, axis=axis, keepdims=keepdims)


def amin(x, axis=None):
    return np.amin(x, axis)


def amax(x, axis=None):
    return np.amax(x, axis)


def logsumexp(x, axis=None, keepdims=False):
    return special.logsumexp(x, axis=axis, keepdims=keepdims)


def transpose(x, axes):
    return np.transpose(x, axes=axes)


def stack(*args):
    return np.stack(args)


def ascontiguousarray(x):
    return np.ascontiguousarray(x)


def ones_like(x):
    return np.ones_like(x)


# Numpy does not support autograd, so the functions below are trivial:
def detach(x):
    return x


def is_grad_enabled(typical_array):
    return False


def set_grad_enabled(typical_array, b):
    return None


def unbalanced_weight(f, *, eps, rho):
    return (rho + eps / 2) * f
