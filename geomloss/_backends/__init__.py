from .common import pick, get_library
from . import numpy as bk_numpy

try:
    from . import torch as bk_torch
except:
    from . import numpy as bk_torch

from . import keops as bk_keops
from .keops import keops_available, numpy_LazyTensor, torch_LazyTensor

# Low-level attributes:
device = pick(numpy=bk_numpy.device, torch=bk_torch.device)
dtype = pick(numpy=bk_numpy.dtype, torch=bk_torch.dtype)

# Simple mathematical functions:
abs = pick(numpy=bk_numpy.abs, torch=bk_torch.abs)
exp = pick(numpy=bk_numpy.exp, torch=bk_torch.exp)
stable_log = pick(numpy=bk_numpy.stable_log, torch=bk_torch.stable_log)
dot_products = pick(numpy=bk_numpy.dot_products, torch=bk_torch.dot_products)
norm = pick(numpy=bk_numpy.norm, torch=bk_torch.norm)

# Einstein summation: the first arg is a string:
einsum = pick(numpy=bk_numpy.einsum, torch=bk_torch.einsum, main_arg=1)

# Array manipulations and reductions:
any = pick(numpy=bk_numpy.any, torch=bk_torch.any)
sum = pick(numpy=bk_numpy.sum, torch=bk_torch.sum, keops=bk_keops.sum)
mean = pick(numpy=bk_numpy.mean, torch=bk_torch.mean)
amin = pick(numpy=bk_numpy.amin, torch=bk_torch.amin, keops=bk_keops.amin)
amax = pick(numpy=bk_numpy.amax, torch=bk_torch.amax)
logsumexp = pick(
    numpy=bk_numpy.logsumexp, torch=bk_torch.logsumexp, keops=bk_keops.logsumexp
)

allclose = pick(numpy=bk_numpy.allclose, torch=bk_torch.allclose)

transpose = pick(numpy=bk_numpy.transpose, torch=bk_torch.transpose)
stack = pick(numpy=bk_numpy.stack, torch=bk_torch.stack)
ascontiguousarray = pick(
    numpy=bk_numpy.ascontiguousarray, torch=bk_torch.ascontiguousarray
)

# Array creation:
ones_like = pick(numpy=bk_numpy.ones_like, torch=bk_torch.ones_like)


# Conversion between NumPy arrays, PyTorch tensors...:
def cast(x, *, shape, dtype, device, library):
    # `library` denotes the target library.
    source = get_library(x)

    assert source in ["numpy", "torch"]
    assert library in ["numpy", "torch"]

    if library == "numpy":
        if source == "torch":
            x = bk_torch.to_numpy(x)
        return bk_numpy.to(x, shape=shape, dtype=dtype, device=device)

    elif library == "torch":
        if source == "numpy":
            x = bk_torch.from_numpy(x)
        return bk_torch.to(x, shape=shape, dtype=dtype, device=device)


def LazyTensor(x):
    assert keops_available, "Please report this issue."
    library = get_library(x)
    if library == "numpy":
        return numpy_LazyTensor(x)
    elif library == "torch":
        return torch_LazyTensor(x)
    else:
        raise ValueError(
            "Expected a NumPy array or a PyTorch tensor, "
            f"but found {x} "
            f"of type {type(x)}."
        )


# Autograd magic:
detach = pick(numpy=bk_numpy.detach, torch=bk_torch.detach)
is_grad_enabled = pick(numpy=bk_numpy.is_grad_enabled, torch=bk_torch.is_grad_enabled)
set_grad_enabled = pick(
    numpy=bk_numpy.set_grad_enabled, torch=bk_torch.set_grad_enabled
)

scale = pick(numpy=bk_numpy.scale, torch=bk_torch.scale)
view = pick(numpy=bk_numpy.view, torch=bk_torch.view)
