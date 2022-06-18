from typing import List, Dict, Optional, Any
from numpy.typing import ArrayLike
from collections.abc import Callable
from collections import NamedTuple


Tensor = ArrayLike
CostMatrix = Any
CostFunction = Any


# =================================================================
#             Functions used in the Sinkhorn loop
# =================================================================

# The softmin function is at the heart of any (stable) implementation
# of the Sinkhorn algorithm. It takes as input:
# - a temperature eps(ilon),
# - a cost matrix C_xy[i,j] = C(x[i],y[j]),
# - a weighted dual potential G_y[j] = log(b(y[j])) + g_ab(y[j]) / eps.
#
# It returns a new dual potential supported on the points x[i]:
#   f_x[i] = - eps * log(sum_j(exp( G_y[j]  -  C_xy[i,j] / eps )))
#
# In the Sinkhorn loop, we typically use calls like:
#   ft_ba = softmin(eps, C_xy, b_log + g_ab / eps)

SoftMin = Callable[[float, CostMatrix, Tensor], Tensor]


# The extrapolate function is used in the multiscale Sinkhorn scheme.
# We use it to extrapolate the values of a potential f_x[i] = f(x[i])
# from a coarse point cloud x[i] to a finer point cloud x_fine[i].
#
# Depending on the context, this function may either rely on straightforward
# bi/tri-linear interpolation or on the analytical formula for the
# Sinkhorn updates between point clouds.
#
# It takes as input:
# - a coarse dual potential f_x[i] = f(x[i])
# - a coarse dual potential g_y[j] = g(y[j]), which is supported by the other measure,
# - a temperature eps(ilon),
# - a damping factor (which is equal to 1 for balanced OT),
# - a coarse cost matrix C_xy[i,j] = C(x[i], y[j]),
# - a coarse vector of log-weights b_log[j] = log(b(y[j])), supported by the other measure,
# - a fine cost matrix C_xy_fine[i,j] = C(x_fine[i], y_fine[j]).
#
# It returns a new dual potential supported on the points x_fine[i].
#
# In the multiscale Sinkhorn loop, we typically use calls like:
#   f_ba = extrapolate(f_ba, g_ab, eps, damping, C_xy, b_log, C_xy_fine),

Extrapolator = Callable[
    [
        Tensor,
        Tensor,
        float,
        float,
        CostMatrix,
        Tensor,
        CostMatrix,
    ],
    Tensor,
]


# The kernel truncation function is used in the multiscale Sinkhorn scheme.
# Following ideas that were introduced by Bernhard Schmitzer in ~2016,
# we use it to discard un-necessary blocks from a cost matrix at a fine scale,
# using information at a coarse scale.
#
# It takes as input:
# - a coarse cost matrix C_xy[i,j] = C(x[i], y[j]),
# - a coarse cost matrix C_yx[i,j] = C(y[i], x[j]),
#   typically the transpose of C_xy,
# - a fine cost matrix C_xy_fine[i,j] = C(x_fine[i], y_fine[j]),
# - a fine cost matrix C_yx_fine[i,j] = C(y_fine[i], x_fine[j]),
#   typically the transpose of C_xy_fine,
# - a coarse dual potential f_x[i] = f(x[i]),
# - a coarse dual potential g_y[j] = g(y[j]),
# - a temperature eps(ilon),
# - a truncation ratio,
# - a cost function,
#
# It returns new values for C_xy_fine and C_yx_fine, with irrelevant
# blocks of values being pruned out.
#
# In the multiscale Sinkhorn loop, we typically use calls like:
#
# C_xy_fine, C_yx_fine = kernel_truncation(
#                     C_xy,
#                     C_yx,
#                     C_xy_fine,
#                     C_yx_fine,
#                     f_ba,
#                     g_ab,
#                     eps,
#                     truncate=truncate,
#                     cost=cost,
#                     )

KernelTruncation = Callable[
    [
        CostMatrix,
        CostMatrix,
        CostMatrix,
        CostMatrix,
        Tensor,
        Tensor,
        float,
        Optional[float],
        Optional[CostFunction],
    ],
    tuple[CostMatrix, CostMatrix],
]


# The annealing parameters contains the lists of blur scales and temperatures
# epsilon that we use in successive iterations of the Sinkhorn loop.
# It also contains a rough estimation of the diameter of our configuration.
class AnnealingParameters(NamedTuple):
    diameter: float
    blur_list: List[float]
    eps_list: List[float]
