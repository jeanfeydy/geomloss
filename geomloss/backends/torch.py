import torch
from ..typing import RealTensor


def device(a: RealTensor):
    return a.device


def dtype(a: RealTensor):
    return a.dtype


def abs(a: RealTensor) -> RealTensor:
    return a.abs()


def exp(a: RealTensor) -> RealTensor:
    return a.exp()


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


def norm(a: RealTensor) -> RealTensor:
    return (a**2).sum().sqrt()


def einsum(formula, *args):
    return torch.einsum(formula, *args)


def any(x, axis=None, keepdims=False):
    if axis is None:
        assert keepdims == False
        return x.any()
    else:
        return x.any(dim=axis, keepdim=keepdims)


def sum(x, axis=None, keepdims=False):
    return x.sum(dim=axis, keepdim=keepdims)


def mean(x, axis=None, keepdims=False):
    return x.mean(dim=axis, keepdim=keepdims)


def amin(x, axis=None):
    return torch.amin(x, axis)


def amax(x, axis=None):
    return torch.amax(x, axis)


def logsumexp(x, axis=None, keepdims=False):
    return x.logsumexp(dim=axis, keepdim=keepdims)


def transpose(x, axes):
    return x.permute(axes)


def stack(*args):
    return torch.stack(args)


def ascontiguousarray(x):
    return x.contiguous()


def ones_like(x):
    return torch.ones_like(x)


def from_numpy(x):
    return torch.from_numpy(x)


def to_numpy(x):
    return x.detach().cpu().numpy()


def to(x, *, shape, dtype, device):
    return x.view(shape).to(dtype=dtype, device=device)


def detach(x):
    return x.detach()


def is_grad_enabled(typical_array):
    return torch.is_grad_enabled()


def set_grad_enabled(typical_array, b):
    return torch.set_grad_enabled(b)


class UnbalancedWeight(torch.nn.Module):
    """Applies the correct scaling to the dual variables in the Sinkhorn divergence formula.

    Remarkably, the exponentiated potentials should be scaled
    by "rho + eps/2" in the forward pass and "rho + eps" in the backward.
    For an explanation of this surprising "inconsistency"
    between the forward and backward formulas,
    please refer to Proposition 12 (Dual formulas for the Sinkhorn costs)
    in "Sinkhorn divergences for unbalanced optimal transport",
    Sejourne et al., https://arxiv.org/abs/1910.12958.
    """

    def __init__(self, *, eps: float, rho: float):
        super(UnbalancedWeight, self).__init__()
        self.eps, self.rho = eps, rho

    def forward(self, x: RealTensor):
        return (self.rho + self.eps / 2) * x

    def backward(self, g: RealTensor):
        return (self.rho + self.eps) * g


def unbalanced_weight(f, *, eps, rho):
    return UnbalancedWeight(eps=eps, rho=rho)(f)
