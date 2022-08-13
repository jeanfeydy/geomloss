from .common import pick, library
from . import numpy as bk_numpy

try:
    from . import torch as bk_torch
except:
    from . import numpy as bk_torch


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
sum = pick(numpy=bk_numpy.sum, torch=bk_torch.sum)
mean = pick(numpy=bk_numpy.mean, torch=bk_torch.mean)
amin = pick(numpy=bk_numpy.amin, torch=bk_torch.amin)
amax = pick(numpy=bk_numpy.amax, torch=bk_torch.amax)
logsumexp = pick(numpy=bk_numpy.logsumexp, torch=bk_torch.logsumexp)

transpose = pick(numpy=bk_numpy.transpose, torch=bk_torch.transpose)
stack = pick(numpy=bk_numpy.stack, torch=bk_torch.stack)
ascontiguousarray = pick(
    numpy=bk_numpy.ascontiguousarray, torch=bk_torch.ascontiguousarray
)

# Array creation:
ones_like = pick(numpy=bk_numpy.ones_like, torch=bk_torch.ones_like)

# Autograd magic:
detach = pick(numpy=bk_numpy.detach, torch=bk_torch.detach)
is_grad_enabled = pick(numpy=bk_numpy.is_grad_enabled, torch=bk_torch.is_grad_enabled)
set_grad_enabled = pick(
    numpy=bk_numpy.set_grad_enabled, torch=bk_torch.set_grad_enabled
)

unbalanced_weight = pick(
    numpy=bk_numpy.unbalanced_weight, torch=bk_torch.unbalanced_weight
)
