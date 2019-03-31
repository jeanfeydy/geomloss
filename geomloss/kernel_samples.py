"""Implements kernel ("gaussian", "laplacian", "energy") norms between sampled measures.

.. math::
    \\text{Loss}(\\alpha,\\beta) 
        ~&=~ \\text{Loss}\\big( \sum_{i=1}^N \\alpha_i \,\delta_{x_i} \,,\, \sum_{j=1}^M \\beta_j \,\delta_{y_j} \\big) 
        ~=~ \\tfrac{1}{2} \|\\alpha-\\beta\|_k^2 \\\\
        &=~ \\tfrac{1}{2} \langle \\alpha-\\beta \,,\, k\star (\\alpha - \\beta) \\rangle \\\\
        &=~ \\tfrac{1}{2} \sum_{i=1}^N \sum_{j=1}^N  \\alpha_i \\alpha_j \cdot k(x_i,x_j) 
          + \\tfrac{1}{2} \sum_{i=1}^M \sum_{j=1}^M  \\beta_i \\beta_j \cdot k(y_i,y_j) \\\\
        &-~\sum_{i=1}^N \sum_{j=1}^M  \\alpha_i \\beta_j \cdot k(x_i,y_j)

where:

.. math::
    k(x,y)~=~\\begin{cases}
        \exp( -\|x-y\|^2/2\sigma^2) & \\text{if loss = ``gaussian''} \\\\
        \exp( -\|x-y\|/\sigma) & \\text{if loss = ``laplacian''} \\\\
        -\|x-y\| & \\text{if loss = ``energy''} \\\\
    \\end{cases}
"""

import numpy as np
import torch

try:  # Import the keops library, www.kernel-operations.io
    from pykeops.torch import generic_sum
    from pykeops.torch.cluster import grid_cluster, cluster_ranges_centroids, sort_clusters, from_matrix, swap_axes
    keops_available = True
except:
    keops_available = False

from .utils import scal, squared_distances, distances

class DoubleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return 2*grad_output

def double_grad(x):
    return DoubleGrad.apply(x)


# ==============================================================================
#                          backend == "tensorized"
# ==============================================================================

def gaussian_kernel(x, y, blur=.05):
    C2 = squared_distances(x / blur, y / blur)
    return (- .5 * C2 ).exp()

def laplacian_kernel(x, y, blur=.05):
    C = distances(x / blur, y / blur)
    return (- C ).exp()

def energy_kernel(x, y, blur=None):
    return - distances(x, y)

kernel_routines = {
    "gaussian" : gaussian_kernel,
    "laplacian": laplacian_kernel,
    "energy"   : energy_kernel,
}

def kernel_tensorized(α, x, β, y, blur=.05, kernel=None, name=None, potentials=False,**kwargs):
    
    B, N, D = x.shape
    _, M, _ = y.shape

    if kernel is None:
        kernel = kernel_routines[name]

    K_xx = kernel( double_grad(x), x.detach(), blur=blur)  # (B,N,N) tensor
    K_yy = kernel( double_grad(y), y.detach(), blur=blur)  # (B,M,M) tensor
    K_xy = kernel( x, y, blur=blur)                        # (B,N,M) tensor

    a_x = torch.matmul( K_xx, α.detach().unsqueeze(-1) ).squeeze(-1)  # (B,N,N) @ (B,N) = (B,N) 
    b_y = torch.matmul( K_yy, β.detach().unsqueeze(-1) ).squeeze(-1)  # (B,M,M) @ (B,M) = (B,M)
    b_x = torch.matmul( K_xy, β.unsqueeze(-1)          ).squeeze(-1)  # (B,N,M) @ (B,M) = (B,N) 

    
    if potentials:
        a_y = torch.matmul( K_xy.transpose(1,2), α.unsqueeze(-1)).squeeze(-1)  # (B,M,N) @ (B,N) = (B,M)
        return a_x - b_x, b_y - a_y

    else:  # Return the Kernel norm. N.B.: we assume that 'kernel' is symmetric:
        return .5 * (double_grad(α) * a_x).sum(1) \
             + .5 * (double_grad(β) * b_y).sum(1) \
             -  (α * b_x).sum(1)



# ==============================================================================
#                           backend == "online"
# ==============================================================================

kernel_formulas = {
    "gaussian" : ("Exp(-SqDist(X,Y) / IntCst(2))", True ),
    "laplacian": ("Exp(-Norm2(X-Y))",   True ),
    "energy"   : ("(-Norm2(X-Y))",      False),
}


def kernel_keops(kernel, α, x, β, y, potentials=False, ranges_xx = None, ranges_yy = None, ranges_xy = None):

    D = x.shape[1]
    kernel_conv = generic_sum( "(" + kernel + " * B)",   # Formula
                               "A = Vi(1)",              # Output:    a_i
                               "X = Vi({})".format(D),   # 1st input: x_i
                               "Y = Vj({})".format(D),   # 2nd input: y_j
                               "B = Vj(1)" )             # 3rd input: b_j
    
    a_x = kernel_conv(double_grad(x), x.detach(), α.detach().view(-1,1), ranges=ranges_xx)
    b_y = kernel_conv(double_grad(y), y.detach(), β.detach().view(-1,1), ranges=ranges_yy)
    b_x = kernel_conv(x, y, β.view(-1,1), ranges=ranges_xy)

    if potentials:
        a_y = kernel_conv(y, x, α.view(-1,1), ranges=swap_axes(ranges_xy))
        return a_x - b_x, b_y - a_y

    else:  # Return the Kernel norm. N.B.: we assume that 'kernel' is symmetric:
        return .5 * scal( double_grad(α), a_x ) \
             + .5 * scal( double_grad(β), b_y )  -  scal( α, b_x )
              


def kernel_preprocess(kernel, name, x, y, blur):
    if not keops_available:
        raise ImportError("The 'pykeops' library could not be loaded: " \
                        + "'online' and 'multiscale' backends are not available.")
    
    if kernel is None: kernel, rescale = kernel_formulas[name]
    else:              rescale = True
    
    # Center the point clouds just in case, to prevent numeric overflows:
    center = (x.mean(0, keepdim=True) + y.mean(0,  keepdim=True)) / 2
    x, y = x - center, y - center
    # Rescaling on x and y is cheaper than doing it for all pairs of points 
    if rescale : x, y = x / blur, y / blur
    
    return kernel, x, y


def kernel_online(α, x, β, y, blur=.05, kernel=None, name=None, potentials=False, **kwargs):

    kernel, x, y = kernel_preprocess(kernel, name, x, y, blur)
    return kernel_keops(kernel, α, x, β, y, potentials=potentials)


# ==============================================================================
#                          backend == "multiscale"
# ==============================================================================

def max_diameter(x, y):
    mins = torch.stack((x.min(dim=0)[0], y.min(dim=0)[0])).min(dim=0)[0]
    maxs = torch.stack((x.max(dim=0)[0], y.max(dim=0)[0])).max(dim=0)[0]
    diameter = (maxs-mins).norm().item()
    return diameter


def kernel_multiscale(α, x, β, y, blur=.05, kernel=None, name=None, 
                      truncate=5, diameter=None, cluster_scale=None, 
                      potentials=False, verbose=False, **kwargs):

    if truncate is None or name == "energy":
        return kernel_online( α, x, β, y, blur=blur, kernel=kernel, 
                              truncate=truncate, name=name, potentials=potentials, **kwargs )

    # Renormalize our point cloud so that blur = 1:
    kernel, x, y = kernel_preprocess(kernel, name, x, y, blur)

    # Don't forget to normalize the diameter too!
    if cluster_scale is None: 
        D = x.shape[-1]
        if diameter is None:
            diameter = max_diameter(x.view(-1,D), y.view(-1,D))
        else:
            diameter = diameter / blur
        cluster_scale = diameter / (np.sqrt(D) * 2000**(1/D))

    # Put our points in cubic clusters:
    cell_diameter = cluster_scale * np.sqrt( x.shape[1] )
    x_lab = grid_cluster(x, cluster_scale) 
    y_lab = grid_cluster(y, cluster_scale) 

    # Compute the ranges and centroids of each cluster:
    ranges_x, x_c, α_c = cluster_ranges_centroids(x, x_lab, weights=α)
    ranges_y, y_c, β_c = cluster_ranges_centroids(y, y_lab, weights=β)


    if verbose: 
        print("{}x{} clusters, computed at scale = {:2.3f}".format(
              len(x_c), len(y_c), cluster_scale))

    # Sort the clusters, making them contiguous in memory:
    (α, x), x_lab = sort_clusters( (α, x), x_lab)
    (β, y), y_lab = sort_clusters( (β, y), y_lab)

    with torch.no_grad():  # Compute our block-sparse reduction ranges:
        # Compute pairwise distances between clusters:
        C_xx = squared_distances( x_c, x_c)
        C_yy = squared_distances( y_c, y_c)
        C_xy = squared_distances( x_c, y_c)

        # Compute the boolean masks:
        keep_xx = ( C_xx <= (truncate + cell_diameter)**2 )
        keep_yy = ( C_yy <= (truncate + cell_diameter)**2 )
        keep_xy = ( C_xy <= (truncate + cell_diameter)**2 )

        # Compute the KeOps reduction ranges:
        ranges_xx = from_matrix(ranges_x, ranges_x, keep_xx)
        ranges_yy = from_matrix(ranges_y, ranges_y, keep_yy)
        ranges_xy = from_matrix(ranges_x, ranges_y, keep_xy)
    
    return kernel_keops(kernel, α, x, β, y, potentials=potentials,
                ranges_xx=ranges_xx, ranges_yy=ranges_yy, ranges_xy=ranges_xy)

