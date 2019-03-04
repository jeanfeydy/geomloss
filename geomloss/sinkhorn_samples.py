"""Implements the (unbiased) Sinkhorn divergence between sampled measures.

.. math::
    \\text{S}_{\\varepsilon,\\rho}(\\alpha,\\beta) 
        ~&=~ \\text{OT}_{\\varepsilon,\\rho}(\\alpha, \\beta)
         ~-~\\tfrac{1}{2} \\text{OT}_{\\varepsilon,\\rho}(\\alpha, \\alpha)
         ~-~\\tfrac{1}{2} \\text{OT}_{\\varepsilon,\\rho}(\\beta, \\beta)
         ~+~ \\tfrac{\\varepsilon}{2} \| \\langle \\alpha, 1\\rangle - \\langle \\beta, 1\\rangle \|^2

where:

.. math::
    \\text{OT}_{\\varepsilon,\\rho}(\\alpha, \\beta)
    ~&=~ \\min_{\pi\geqslant 0} \\langle\, \pi\,,\, \\text{C} \,\\rangle
        ~+~\\varepsilon \, \\text{KL}(\pi,\\alpha\otimes\\beta) \\\\
        ~&+~\\rho \, \\text{KL}(\pi\,\mathbf{1},\\alpha)
        ~+~\\rho \, \\text{KL}(\pi^\intercal \,\mathbf{1},\\beta ) \\
    &=~ \\max_{b,a} -\\rho \\langle\, \\alpha \,,\, e^{-b/\\rho} - 1\,\\rangle
        -\\rho \\langle\, \\beta \,,\, e^{-a/\\rho} - 1\,\\rangle \\\\
        &-~
        \\epsilon \\langle\, \\alpha\otimes\\beta \,,\, e^{(b\oplus a - \\text{C})/\\epsilon} - 1\,\\rangle,

with a Kullback-Leibler divergence defined through:

.. math::
    \\text{KL}(\\alpha, \\beta)~=~
    \\langle \, \\alpha  \,,\, \\log \\tfrac{\\text{d}\\alpha}{\\text{d}\\beta} \,\\rangle
    ~-~ \\langle \, \\alpha  \,,\, 1 \,\\rangle
    ~+~ \\langle \, \\beta   \,,\, 1 \,\\rangle ~\geqslant~ 0.
"""

import numpy as np
import torch
from functools import partial

try:  # Import the keops library, www.kernel-operations.io
    from pykeops.torch import generic_logsumexp
    from pykeops.torch.cluster import grid_cluster, cluster_ranges_centroids, sort_clusters, from_matrix
    keops_available = True
except:
    keops_available = False
    
from .utils import scal, squared_distances, distances


# ==============================================================================
#                            ε-scaling heuristic
# ==============================================================================

def max_diameter(x, y):
    mins = torch.stack((x.min(dim=0)[0], y.min(dim=0)[0])).min(dim=0)[0]
    maxs = torch.stack((x.max(dim=0)[0], y.max(dim=0)[0])).max(dim=0)[0]
    diameter = (maxs-mins).norm().item()
    return diameter


def epsilon_schedule(p, diameter, blur, scaling):
    ε_s = [ diameter**p ] \
        + [ np.exp(e) for e in np.arange(p*np.log(diameter), p*np.log(blur), p*np.log(scaling)) ] \
        + [ blur**p ]
    return ε_s


def scaling_parameters( x, y, p, blur, reach, diameter, scaling):

    if diameter is None:
        D = x.shape[-1]
        diameter = max_diameter(x.view(-1,D), y.view(-1,D))

    ε   = blur**p
    ε_s = epsilon_schedule( p, diameter, blur, scaling )
    ρ   = None if reach is None else reach**p
    return ε, ε_s, ρ  


# ==============================================================================
#                              Sinkhorn loop
# ==============================================================================

def dampening(ε, ρ):
    return 1 if ρ is None else 1 / ( 1 + ε / ρ )


def log_weights(α):
    α_log = α.log()
    α_log[α <= 0] = -100000
    return α_log


def sinkhorn_cost(ε, ρ, α, β, a_i, b_j, a_j, b_i, batch=False):
    if ρ is None:
        return scal( α, b_i - a_i, batch=batch ) + scal( β, a_j - b_j, batch=batch )


def sinkhorn_loop( softmin, α, β, C_xx, C_yy, C_xy, C_yx, ε_s, ρ ):

    with torch.no_grad():
        ε = ε_s[0]
        λ = dampening(ε, ρ)

        α_log, β_log = log_weights(α), log_weights(β)
        a_i = λ * softmin(ε, C_xx, α_log )  # (B,N)
        b_j = λ * softmin(ε, C_yy, β_log )  # (B,M)
        a_j = λ * softmin(ε, C_yx, α_log )  # (B,M)
        b_i = λ * softmin(ε, C_xy, β_log )  # (B,N)

        for ε in ε_s:
            λ = dampening(ε, ρ)

            # "Coordinate ascent" on the dual problems:
            at_i = λ * softmin(ε, C_xx, α_log + a_i/ε )  # (B,N)
            bt_j = λ * softmin(ε, C_yy, β_log + b_j/ε )  # (B,M)
            at_j = λ * softmin(ε, C_yx, α_log + b_i/ε )  # (B,M)
            bt_i = λ * softmin(ε, C_xy, β_log + a_j/ε )  # (B,N)

            # Symmetrized updates:
            a_i, b_j = .5 * ( a_i + at_i ), .5 * ( b_j + bt_j )  # OT(α,α), OT(β,β)
            a_j, b_i = .5 * ( a_j + at_j ), .5 * ( b_i + bt_i )  # OT(α,β)

    # Last extrapolation, to get the correct gradients:
    a_i = λ * softmin(ε, C_xx, (α_log + a_i/ε).detach() )  # (B,N)
    b_j = λ * softmin(ε, C_yy, (β_log + b_j/ε).detach() )  # (B,M)

    # The cross-updates should be done in parallel!
    a_j, b_i = λ * softmin(ε, C_yx, (α_log + b_i/ε).detach() ), \
               λ * softmin(ε, C_xy, (β_log + a_j/ε).detach() )

    return a_i, b_j, a_j, b_i


# ==============================================================================
#                          backend == "tensorized"
# ==============================================================================

cost_routines = {
    1 : (lambda x,y : distances(x,y)),
    2 : (lambda x,y : squared_distances(x,y) / 2),
}

def softmin_tensorized(ε, C, f):
    B = C.shape[0]
    return - ε * ( f.view(B,1,-1) - C/ε ).logsumexp(2)

def sinkhorn_tensorized(α, x, β, y, p=2, blur=.05, reach=None, diameter=None, scaling=.5, cost=None, **kwargs):
    
    B, N, D = x.shape
    _, M, _ = y.shape

    if cost is None:
        cost = cost_routines[p]
        
    C_xx, C_yy = cost( x, x.detach()), cost( y, y.detach())  # (B,N,N), (B,M,M)
    C_xy, C_yx = cost( x, y.detach()), cost( y, x.detach())  # (B,N,M), (B,M,N)


    ε, ε_s, ρ = scaling_parameters( x, y, p, blur, reach, diameter, scaling )

    a_i, b_j, a_j, b_i = sinkhorn_loop( softmin_tensorized, α, β, 
                                        C_xx, C_yy, C_xy, C_yx, ε_s, ρ )

    return sinkhorn_cost(ε, ρ, α, β, a_i, b_j, a_j, b_i, batch=True)


# ==============================================================================
#                          backend == "online"
# ==============================================================================

cost_formulas = {
    1 : "Norm2(X-Y)",
    2 : "(SqDist(X,Y) / IntCst(2))",
}

def softmin_online(ε, C_ij, f_j, log_conv=None):
    x_i, y_j = C_ij
    return - ε * log_conv( x_i, y_j, f_j.view(-1,1), torch.Tensor([1/ε]).type_as(x_i) ).view(-1)


def sinkhorn_online(α, x, β, y, p=2, blur=.05, reach=None, diameter=None, scaling=.5, cost=None, **kwargs):
    
    N, D = x.shape
    M, _ = y.shape

    if cost is None:
        cost = cost_formulas[p]

    log_conv = generic_logsumexp("( B - (P * " + cost + " ) )",
                                 "A = Vx(1)",
                                 "X = Vx({})".format(D),
                                 "Y = Vy({})".format(D),
                                 "B = Vy(1)",
                                 "P = Pm(1)")

    softmin = partial(softmin_online, log_conv=log_conv)

    # The "cost matrices" are implicitely encoded in the point clouds,
    # and re-computed on-the-fly:
    C_xx, C_yy = (x, x.detach()), (y, y.detach())
    C_xy, C_yx = (x, y.detach()), (y, x.detach())

    ε, ε_s, ρ = scaling_parameters( x, y, p, blur, reach, diameter, scaling )

    a_i, b_j, a_j, b_i = sinkhorn_loop( softmin, α, β, 
                                        C_xx, C_yy, C_xy, C_yx, ε_s, ρ )

    return sinkhorn_cost(ε, ρ, α, β, a_i, b_j, a_j, b_i)