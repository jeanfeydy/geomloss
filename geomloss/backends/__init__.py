from .common import pick
from . import torch as bk_torch
from . import numpy as bk_numpy


# Simple mathematical functions:
exp = pick(numpy=bk_numpy.exp, torch=bk_torch.exp)
stable_log = pick(numpy=bk_numpy.stable_log, torch=bk_torch.stable_log)
dot_products = pick(numpy=bk_numpy.dot_products, torch=bk_torch.dot_products)
norm = pick(numpy=bk_numpy.norm, torch=bk_torch.norm)

# Einstein summation: the first arg is a string:
einsum = pick(numpy=bk_numpy.einsum, torch=bk_torch.einsum, main_arg=1)

# Array manipulation:
sum = pick(numpy=bk_numpy.sum, torch=bk_torch.sum)
amin = pick(numpy=bk_numpy.amin, torch=bk_torch.amin)
amax = pick(numpy=bk_numpy.amax, torch=bk_torch.amax)
stack = pick(numpy=bk_numpy.stack, torch=bk_torch.stack)

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
