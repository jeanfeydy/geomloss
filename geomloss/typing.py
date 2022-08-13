from typing import Union, Tuple, List, Optional, Any, NamedTuple, Callable
from numpy.typing import ArrayLike


RealTensor = ArrayLike
CostMatrix = Union[RealTensor, Any]
CostFunction = Any

# A CostMatrices object encodes the full information about cost
# values between the supports of our two distributions,
# the source points x[i] and the target points y[j].
# At the very least, we require the cost matrix between the points
# x[i] and y[j] - typically, C.yx = transpose(C.xy).
# Moreover, many objects (kernel distances, debiased Sinkhorn divergences...)
# also require the cost matrices between points x[i] <-> x[j]
# and y[i] <-> y[j].
class CostMatrices(NamedTuple):
    xx: Optional[CostMatrix]  # C(x[i], x[j])
    yy: Optional[CostMatrix]  # C(y[i], y[j])
    xy: CostMatrix  # C(x[i], y[j])
    yx: CostMatrix  # C(y[i], x[j])


# The Sinkhorn potentials contains the four output of the Sinkhorn loop
#
# Please note that the symmetric potentials are only used
# for debiased entropic OT: biased OT uses None instead.
class SinkhornPotentials(NamedTuple):
    f_aa: Optional[RealTensor]  # Symmetric potential f_aa(x_i)
    g_bb: Optional[RealTensor]  # Symmetric potential g_bb(y_j)
    g_ab: RealTensor  # Dual potential g_ab(y_j)
    f_ba: RealTensor  # Dual potential f_ba(x_i)


# The descent parameters contains the lists of blur scales, temperatures
# epsilon and strength of the marginal constraints rho
# that we use in successive iterations of the Sinkhorn loop.
# It also contains a rough estimation of the diameter of our configuration,
# and a list of integer indices that describe at which "scale" of a multiscale
# representation we should perform each iteration.
class DescentParameters(NamedTuple):
    diameter: float
    scale_list: List[int]  # = [0, ..., 0] for single-scale mode
    blur_list: List[float]
    eps_list: List[float]
    rho_list: List[float]


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

SoftMin = Callable[
    [
        float,  # eps
        CostMatrix,  # C_xy
        RealTensor,  # G_y
    ],
    RealTensor,  # f_x
]


# The extrapolate function is used in the multiscale Sinkhorn scheme.
# We use it to extrapolate the values of a potential f_x[i] = f(x[i])
# from a coarse point cloud x[i] to a finer point cloud x_fine[i].
#
# Depending on the context, this function may either rely on straightforward
# bi/tri-linear interpolation or on the analytical formula for the
# Sinkhorn updates between point clouds.
#
# It takes as input:
# - a coarse dual potential f_x[i] = f(x[i]) that we want to refine,
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
        RealTensor,  # f_x
        RealTensor,  # g_y
        float,  # eps
        float,  # damping
        CostMatrix,  # C_xy
        RealTensor,  # b_log
        CostMatrix,  # C_xy_fine
    ],
    RealTensor,  # f_x_fine
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
        CostMatrix,  # C_xy
        CostMatrix,  # C_yx
        CostMatrix,  # C_xy_fine
        CostMatrix,  # C_yx_fine
        RealTensor,  # f_ba
        RealTensor,  # g_ab
        float,  # eps
        Optional[float],  # truncate
        Optional[CostFunction],  # cost
    ],
    Tuple[CostMatrix, CostMatrix],  # C_xy_fine, C_yx_fine
]
