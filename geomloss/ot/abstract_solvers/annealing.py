r"""This file controls the schedule for annealing schemes 
(also known as "epsilon-scaling") in Sinkhorn solvers.

The main reference for this file is Chapter 3.3 in Jean Feydy's PhD thesis:
Geometric data analysis, beyond convolutions (2020), 
https://www.jeanfeydy.com/geometric_data_analysis.pdf
"""

from typing import List, Dict, Optional
from numpy.typing import ArrayLike
from collections.abc import Callable
from collections import NamedTuple

Tensor = ArrayLike


class AnnealingParameters(NamedTuple):
    diameter: float
    blur: float
    blur_list: List[float]
    eps: float
    eps_list: List[float]
    rho: float


# ==============================================================================
#                            eps-scaling heuristic
# ==============================================================================


def max_diameter(x: Tensor, y: Tensor) -> float:
    """Returns a rough estimation of the diameter of a pair of point clouds.

    This quantity is used as a maximum "starting scale" in the epsilon-scaling
    annealing heuristic.

    Args:
        x ((N, D) Tensor): First point cloud.
        y ((M, D) Tensor): Second point cloud.

    Returns:
        float: Upper bound on the largest distance between points `x[i]` and `y[j]`.
    """
    mins = torch.stack((x.min(dim=0)[0], y.min(dim=0)[0])).min(dim=0)[0]
    maxs = torch.stack((x.max(dim=0)[0], y.max(dim=0)[0])).max(dim=0)[0]
    diameter = (maxs - mins).norm().item()
    return diameter


def epsilon_schedule(*, p, diameter, blur, n_iter=None, scaling=None):
    r"""Creates a list of values for the temperature "epsilon" across Sinkhorn iterations.

    We use an aggressive strategy with an exponential cooling
    schedule: starting from a value of :math:`\text{diameter}^p`,
    the temperature epsilon is divided
    by :math:`\text{scaling}^p` at every iteration until reaching
    a minimum value of :math:`\text{blur}^p`.

    The number of iterations can be specified in two different ways,
    using either an integer number (n_iter) or a ratio between successive scales(scaling).

    Args:
        p (integer or float): The exponent of the Euclidean distance
            :math:`\|x_i-y_j\|` that defines the cost function
            :math:`\text{C}(x_i,y_j) =\tfrac{1}{p} \|x_i-y_j\|^p`.

        diameter (float, positive): Upper bound on the largest distance between
            points :math:`x_i` and :math:`y_j`.

        blur (float, positive): Target value for the entropic regularization
            (":math:`\varepsilon = \text{blur}^p`").

        n_iter (int): Number of iterations.

        scaling (float, in (0,1)): Ratio between two successive
            values of the blur scale.

    Returns:
        list of float: list of values for the temperature epsilon.
    """
    eps_list = (
        [diameter**p]
        + [
            np.exp(p * e)
            for e in np.arange(np.log(diameter), np.log(blur), np.log(scaling))
        ]
        + [blur**p]
    )
    return eps_list


def annealing_parameters(
    *,
    x: Tensor,
    y: Tensor,
    p: int,
    blur: float,
    reach: float,
    diameter: Optional[float] = None,
    n_iter: Optional[int] = None,
    scaling: Optional[float] = None,
) -> ScalingParameters:
    r"""Turns high-level arguments into numerical values for the Sinkhorn loop."""

    if diameter is None:
        D = x.shape[-1]
        diameter = max_diameter(x.view(-1, D), y.view(-1, D))

    eps = blur**p
    rho = None if reach is None else reach**p
    eps_list = epsilon_schedule(p, diameter, blur, scaling)
    return diameter, eps, eps_list, rho
