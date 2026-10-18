import pytest
import torch

from geomloss import SamplesLoss
from geomloss._legacy import utils


@pytest.mark.parametrize("loss", ["energy", "laplacian", "gaussian"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [None, 2])
def test_kernel_online_without_keops(monkeypatch, loss, dtype, batch_size):
    monkeypatch.setattr(utils, "keops_available", False)
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    y = torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=dtype)
    if batch_size is not None:
        x = x.repeat(batch_size, 1, 1)
        y = y.repeat(batch_size, 1, 1)
    x.requires_grad_()
    y.requires_grad_()

    reference = SamplesLoss(loss, backend="tensorized")(x, y)
    result = SamplesLoss(loss, backend="online")(x, y)
    reference_grad = torch.autograd.grad(reference.sum(), (x, y))
    result_grad = torch.autograd.grad(result.sum(), (x, y))

    for actual, expected in zip(result_grad, reference_grad):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(result, reference)
