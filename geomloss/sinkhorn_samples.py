"""Implements the (unbiased) Sinkhorn divergence between sampled measures."""

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

from .sinkhorn_divergence import epsilon_schedule, scaling_parameters
from .sinkhorn_divergence import dampening, log_weights, sinkhorn_cost, sinkhorn_loop



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

    a_x, b_y, a_y, b_x = sinkhorn_loop( softmin_tensorized, 
                                        log_weights(α), log_weights(β), 
                                        C_xx, C_yy, C_xy, C_yx, ε_s, ρ )

    return sinkhorn_cost(ε, ρ, α, β, a_x, b_y, a_y, b_x, batch=True)


# ==============================================================================
#                          backend == "online"
# ==============================================================================

cost_formulas = {
    1 : "Norm2(X-Y)",
    2 : "(SqDist(X,Y) / IntCst(2))",
}

def softmin_online(ε, C_xy, f_y, log_conv=None):
    x, y = C_xy
    # KeOps is pretty picky on the input shapes...
    return - ε * log_conv( x, y, f_y.view(-1,1), torch.Tensor([1/ε]).type_as(x) ).view(-1)


def softmin_keops(cost, D):
    log_conv = generic_logsumexp("( B - (P * " + cost + " ) )",
                                 "A = Vx(1)",
                                 "X = Vx({})".format(D),
                                 "Y = Vy({})".format(D),
                                 "B = Vy(1)",
                                 "P = Pm(1)")
    return partial(softmin_online, log_conv=log_conv)


def sinkhorn_online(α, x, β, y, p=2, blur=.05, reach=None, diameter=None, scaling=.5, cost=None, **kwargs):
    
    N, D = x.shape
    M, _ = y.shape

    if cost is None:
        cost = cost_formulas[p]

    softmin = softmin_keops(cost, D)

    # The "cost matrices" are implicitely encoded in the point clouds,
    # and re-computed on-the-fly:
    C_xx, C_yy = (x, x.detach()), (y, y.detach())
    C_xy, C_yx = (x, y.detach()), (y, x.detach())

    ε, ε_s, ρ = scaling_parameters( x, y, p, blur, reach, diameter, scaling )

    a_x, b_y, a_y, b_x = sinkhorn_loop( softmin,
                                        log_weights(α), log_weights(β), 
                                        C_xx, C_yy, C_xy, C_yx, ε_s, ρ )

    return sinkhorn_cost(ε, ρ, α, β, a_x, b_y, a_y, b_x)



# ==============================================================================
#                          backend == "multiscale"
# ==============================================================================

