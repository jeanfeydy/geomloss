"""Implements the (unbiased) Sinkhorn divergence between sampled measures."""

import numpy as np
import torch
from functools import partial

try:  # Import the keops library, www.kernel-operations.io
    from pykeops.torch import generic_logsumexp
    from pykeops.torch.cluster import grid_cluster, cluster_ranges_centroids
    from pykeops.torch.cluster import sort_clusters, from_matrix, swap_axes
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


    diameter, ε, ε_s, ρ = scaling_parameters( x, y, p, blur, reach, diameter, scaling )

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


def keops_lse(cost, D):
    log_conv = generic_logsumexp("( B - (P * " + cost + " ) )",
                                 "A = Vx(1)",
                                 "X = Vx({})".format(D),
                                 "Y = Vy({})".format(D),
                                 "B = Vy(1)",
                                 "P = Pm(1)")
    return log_conv


def sinkhorn_online(α, x, β, y, p=2, blur=.05, reach=None, diameter=None, scaling=.5, cost=None, **kwargs):
    
    N, D = x.shape
    M, _ = y.shape

    if cost is None: cost = cost_formulas[p]

    softmin = partial(softmin_online, log_conv=keops_lse(cost, D)) 

    # The "cost matrices" are implicitely encoded in the point clouds,
    # and re-computed on-the-fly:
    C_xx, C_yy = (x, x.detach()), (y, y.detach())
    C_xy, C_yx = (x, y.detach()), (y, x.detach())

    diameter, ε, ε_s, ρ = scaling_parameters( x, y, p, blur, reach, diameter, scaling )

    a_x, b_y, a_y, b_x = sinkhorn_loop( softmin,
                                        log_weights(α), log_weights(β), 
                                        C_xx, C_yy, C_xy, C_yx, ε_s, ρ )

    return sinkhorn_cost(ε, ρ, α, β, a_x, b_y, a_y, b_x)



# ==============================================================================
#                          backend == "multiscale"
# ==============================================================================


def softmin_multiscale(ε, C_xy, f_y, log_conv=None):
    x, y, ranges_x, ranges_y, ranges_xy = C_xy
    # KeOps is pretty picky on the input shapes...
    return - ε * log_conv( x, y, f_y.view(-1,1), torch.Tensor([1/ε]).type_as(x), ranges=ranges_xy ).view(-1)


def clusterize(α, x, scale=None) :
    """
    Performs a simple 'voxelgrid' clustering on the input measure,
    putting points into cubic bins of size 'scale' = σ_c.
    The weights are summed, and the centroid position is that of the bin's center of mass.
    Most importantly, the "fine" lists of weights and points are *sorted*
    so that clusters are *contiguous in memory*: this allows us to perform
    kernel truncation efficiently on the GPU.

    If 
        [α_c, α], [x_c, x], [x_ranges] = clusterize(α, x, σ_c),
    then
        α_c[k], x_c[k] correspond to
        α[x_ranges[k,0]:x_ranges[k,1]], x[x_ranges[k,0]:x_ranges[k,1],:]
    """
    if scale is None : # No clustering, single-scale Sinkhorn on the way...
        return [α], [x], []

    else : # As of today, only two-scale Sinkhorn is implemented:
        # Compute simple (voxel-like) class labels:
        x_lab = grid_cluster(x, scale)  
        # Compute centroids and weights:
        ranges_x, x_c, α_c = cluster_ranges_centroids(x, x_lab, weights=α)
        # Make clusters contiguous in memory:
        (α, x), x_labels = sort_clusters( (α,x), x_lab)

        return [α_c, α], [x_c, x], [ranges_x]

def kernel_truncation( C_xy, C_yx, C_xy_, C_yx_, 
                       b_x, a_y, ε, truncate=None, cost=None):
    """Prunes out useless parts of the (block-sparse) cost matrices for finer scales.

    This is where our approximation takes place.
    To be mathematically rigorous, we should make several coarse-to-fine passes,
    making sure that we're not forgetting anyone. A good reference here is
    Bernhard Schmitzer's work: "Stabilized Sparse Scaling Algorithms for 
    Entropy Regularized Transport Problems, (2016)".
    """
    if truncate is None:
        return C_xy_, C_yx_
    else:
        x,  yd,   ranges_x,  ranges_y, _ = C_xy
        y,  xd,          _,         _, _ = C_yx
        x_, yd_, ranges_x_, ranges_y_, _ = C_xy_
        y_, xd_,         _,         _, _ = C_yx_

        with torch.no_grad():
            C      = cost(x, y)
            keep   = b_x.view(-1,1) + a_y.view(1,-1) > C - truncate*ε
            ranges_xy_ = from_matrix(ranges_x, ranges_y, keep)
            print("Keep fraction:", (keep.sum().float() / (C.shape[0]*C.shape[1])).item() )
    

        return (x_, yd_, ranges_x_, ranges_y_, ranges_xy_), \
               (y_, xd_, ranges_y_, ranges_x_, swap_axes(ranges_xy_))


def extrapolate_samples( b_x, a_y, ε, λ, C_xy, β_log, C_xy_, softmin=None ):
    yd = C_xy[1]   # Source points (coarse)
    x_ = C_xy_[0]  # Target points (fine)

    C = (x_, yd, None, None, None)  # "Rectangular" cost matrix, don't bother with ranges
    return λ * softmin(ε, C, (β_log + a_y/ε).detach() )


def sinkhorn_multiscale(α, x, β, y, p=2, blur=.05, reach=None, diameter=None, 
                        scaling=.5, truncate=5, cost=None, cluster_scale=None, **kwargs):
    
    N, D = x.shape
    M, _ = y.shape

    if cost is None: cost = cost_formulas[p], cost_routines[p]
    cost_formula, cost_routine = cost[0], cost[1]

    softmin = partial(softmin_multiscale, log_conv=keops_lse(cost_formula, D)) 
    extrapolate = partial(extrapolate_samples, softmin=softmin)

    diameter, ε, ε_s, ρ = scaling_parameters( x, y, p, blur, reach, diameter, scaling )
    
    # Clusterize and sort our point clouds:
    if cluster_scale is None:
        cluster_scale = diameter / (np.sqrt(D) * 2000**(1/D))
    [α_c, α], [x_c, x], [ranges_x] = clusterize(α, x, scale=cluster_scale)
    [β_c, β], [y_c, y], [ranges_y] = clusterize(β, y, scale=cluster_scale)

    jumps = [ len(ε_s)-1 ]
    for i, ε in enumerate(ε_s[2:]):
        if cluster_scale**p > ε:
            jumps = [i+1]
            break
    

    print(x_c.shape, y_c.shape)
    print([ x**(1/2) for x in ε_s])
    print(jumps)

    # The input measures are stored at two levels: coarse and fine
    α_logs = [ log_weights(α_c), log_weights(α) ]
    β_logs = [ log_weights(β_c), log_weights(β) ]

    # We do the same [ coarse, fine ] decomposition for "cost matrices",
    # which are implicitely encoded as point clouds
    # + integer summation ranges, and re-computed on-the-fly:
    C_xxs = [ (x_c, x_c.detach(), ranges_x, ranges_x, None), 
              (  x,   x.detach(),     None,     None, None) ] 
    C_yys = [ (y_c, y_c.detach(), ranges_y, ranges_y, None), 
              (  y,   y.detach(),     None,     None, None) ] 
    C_xys = [ (x_c, y_c.detach(), ranges_x, ranges_y, None), 
              (  x,   y.detach(),     None,     None, None) ] 
    C_yxs = [ (y_c, x_c.detach(), ranges_y, ranges_x, None), 
              (  y,   x.detach(),     None,     None, None) ] 

    a_x, b_y, a_y, b_x = sinkhorn_loop( softmin,
                                        α_logs, β_logs, 
                                        C_xxs, C_yys, C_xys, C_yxs, ε_s, ρ,
                                        jumps=jumps,
                                        cost=cost_routine,
                                        kernel_truncation=kernel_truncation,
                                        truncate=truncate,
                                        extrapolate=extrapolate, )

    return sinkhorn_cost(ε, ρ, α, β, a_x, b_y, a_y, b_x)
