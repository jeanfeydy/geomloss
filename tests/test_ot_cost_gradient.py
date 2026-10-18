"""Gradient of ot.solve(...).value with respect to the cost."""

import pytest
import torch

from geomloss import _backends as bk
from geomloss import ot


def test_solve_cost_gradient_is_the_plan():
    """One point on each side forces the plan to 1, so d(value)/dc is 1."""
    cost = torch.tensor([[1.7]], dtype=torch.float64, requires_grad=True)
    solution = ot.solve(
        cost,
        a=torch.tensor([1.0], dtype=torch.float64),
        b=torch.tensor([1.0], dtype=torch.float64),
        reg=0.5,
        max_iter=100,
    )

    (gradient,) = torch.autograd.grad(solution.value, cost)

    assert solution.value.item() == 1.7
    torch.testing.assert_close(gradient, torch.ones_like(cost), atol=0, rtol=0)


def test_converged_cost_gradient_matches_plan_and_potentials():
    """Envelope theorem: d(value)/dC is the plan, and d(value)/da is f."""
    cost = torch.tensor(
        [[0.0, 1.0, 4.0], [2.0, 1.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    weights_a = torch.tensor([2.0, 2.0], dtype=torch.float64, requires_grad=True)
    weights_b = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
    solution = ot.solve(
        cost,
        a=weights_a,
        b=weights_b,
        reg=5.0,
        max_iter=100000,
    )

    grad_cost, grad_a, grad_b = torch.autograd.grad(
        solution.value, (cost, weights_a, weights_b)
    )

    torch.testing.assert_close(grad_cost, solution.plan, atol=1e-12, rtol=0)
    torch.testing.assert_close(grad_a, solution.potential_a, atol=0, rtol=0)
    torch.testing.assert_close(grad_b, solution.potential_b, atol=0, rtol=0)


@pytest.mark.skipif(
    bk.keops_available,
    reason="KeOps LazyTensor costs stay in the graph on this path.",
)
def test_solve_sample_coordinate_gradient_counts_the_plan_once():
    """d(value)/dX_a matches the derivative of <plan, C> with the plan held fixed."""
    points_a = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.2, 0.7], [-0.4, 0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    points_b = torch.tensor(
        [[0.2, 0.1], [0.8, -0.2], [1.5, 0.3]],
        dtype=torch.float64,
    )
    solution = ot.solve_sample(
        points_a,
        points_b,
        a=torch.ones(4, dtype=torch.float64) / 4,
        b=torch.ones(3, dtype=torch.float64) / 3,
        reg=2.0,
        max_iter=5000,
    )

    separation = points_a.detach()[:, None, :] - points_b[None, :, :]
    expected = (solution.plan[:, :, None] * (2 * separation)).sum(dim=1)
    (gradient,) = torch.autograd.grad(solution.value, points_a)

    torch.testing.assert_close(gradient, expected, atol=1e-4, rtol=0)
