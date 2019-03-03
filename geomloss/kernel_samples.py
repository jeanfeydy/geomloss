"""Implements kernel ("gaussian", "laplacian", "energy") norms between sampled measures.

.. math::
    \\text{Loss}(\\al,\\be) 
        ~&=~ \\text{Loss}\\big( \sum_{i=1}^N \\al_i \,\delta_{x_i} \,,\, \sum_{j=1}^M \\be_j \,\delta_{y_j} \\big) 
        ~=~ \\tfrac{1}{2} \|\\al-\\be\|_k^2 \\\\
        &=~ \\tfrac{1}{2} \langle \\al-\\be \,,\, k\star (\\al - \\be) \\rangle \\\\
        &=~ \\tfrac{1}{2} \sum_{i=1}^N \sum_{j=1}^N  \\al_i \\al_j \cdot k(x_i,x_j) 
          + \\tfrac{1}{2} \sum_{i=1}^M \sum_{j=1}^M  \\be_i \\be_j \cdot k(y_i,y_j) \\\\
        &-~\sum_{i=1}^N \sum_{j=1}^M  \\al_i \\be_j \cdot k(x_i,y_j)

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
    from pykeops.torch.cluster import grid_cluster, cluster_ranges_centroids, sort_clusters, from_matrix
    keops_available = True
except:
    keops_available = False

def scal(α, f) :
    return torch.dot( α.view(-1), f.view(-1) )


class DoubleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return 2*grad_output

def double_grad(x):
    return DoubleGrad.apply(x)

class Sqrt0(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        result = input.sqrt()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        grad_input = grad_output / (2*result)
        grad_input[result == 0] = 0
        return grad_input

def sqrt_0(x):
    return Sqrt0.apply(x)


# ==============================================================================
#                          backend == "tensorized"
# ==============================================================================

def squared_distances(x, y):
    D_xx = (x*x).sum(-1).unsqueeze(2)  # (B,N,1)
    D_xy = torch.matmul( x, y.permute(0,2,1) )  # (B,N,D) @ (B,D,M) = (B,N,M)
    D_yy = (y*y).sum(-1).unsqueeze(1)  # (B,1,M)
    return D_xx - 2*D_xy + D_yy


def distances(x, y):
    return sqrt_0( squared_distances(x,y) )


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

def kernel_tensorized(α, x, β, y, blur=.05, kernel=None, name=None, **kwargs):
    
    B, N, D = x.shape
    _, M, _ = y.shape

    if kernel is None:
        kernel = kernel_routines[name]

    K_xx = kernel( double_grad(x), x.detach(), blur=blur)  # (B,N,N) tensor
    K_yy = kernel( double_grad(y), y.detach(), blur=blur)  # (B,M,M) tensor
    K_xy = kernel( x, y, blur=blur)                              # (B,N,M) tensor

    a_i = torch.matmul( K_xx, α.detach().unsqueeze(-1) ).squeeze(-1)  # (B,N,N) @ (B,N) = (B,N) 
    b_j = torch.matmul( K_yy, β.detach().unsqueeze(-1) ).squeeze(-1)  # (B,M,M) @ (B,M) = (B,M)
    b_i = torch.matmul( K_xy, β.unsqueeze(-1)          ).squeeze(-1)  # (B,N,M) @ (B,M) = (B,N) 

    # N.B.: we assume that 'kernel' is symmetric:
    return .5 * (double_grad(α) * a_i).sum(1) \
         + .5 * (double_grad(β) * b_j).sum(1) \
         -  (α * b_i).sum(1)



# ==============================================================================
#                           backend == "online"
# ==============================================================================

kernel_formulas = {
    "gaussian" : ("Exp(-SqDist(X,Y) / IntCst(2))", True ),
    "laplacian": ("Exp(-Dist(X,Y))",   True ),
    "energy"   : ("(-Dist(X,Y))",      False),
}


def kernel_keops(kernel, α, x, β, y, ranges_xx = None, ranges_yy = None, ranges_xy = None):

    D = x.shape[1]
    kernel_conv = generic_sum( "(" kernel +  " * B)",    # Formula
                               "A = Vx(1)",              # Output:    a_i
                               "X = Vx({})".format(D),   # 1st input: x_i
                               "Y = Vy({})".format(D),   # 2nd input: y_j
                               "B = Vy(1)" )             # 3rd input: b_j
    
    a_i = generic_sum(double_grad(x), x.detach(), α.detach().view(-1,1), ranges=ranges_xx)
    b_j = generic_sum(double_grad(y), y.detach(), β.detach().view(-1,1), ranges=ranges_yy)
    b_i = generic_sum(x, y, β.view(-1,1), ranges=ranges_xy)

    # N.B.: we assume that 'kernel' is symmetric:
    return .5 * scal( double_grad(α), a_i ) \
         + .5 * scal( double_grad(β), b_j )  -  scal( α, b_i )
              


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


def kernel_online(α, x, β, y, blur=.05, kernel=None, name=None, **kwargs):

    kernel, x, y = kernel_preprocess(kernel, name, x, y, blur)
    return kernel_keops(kernel, α, x, β, y)


# ==============================================================================
#                          backend == "multiscale"
# ==============================================================================

def kernel_multiscale(α, x, β, y, blur=.05, kernel=None, truncate=5, name=None, **kwargs):

    if truncate is None or name == "energy":
        return kernel_online( α, x, β, y, blur=.05, kernel=None, 
                              truncate=None, name=None, **kwargs )

    # Renormalize our point cloud so that blur = 1:
    kernel, x, y = kernel_preprocess(kernel, name, x, y, blur)

    # Put our points in cubic clusters of diameter 1/2:
    diameter = .5
    edge = diameter / np.sqrt( x.shape[1] )
    x_lab = grid_cluster(x, edge) 
    y_lab = grid_cluster(y, edge) 

    # Compute the ranges and centroids of each cluster:
    ranges_x, x_c, α_c = cluster_ranges_centroids(x, x_lab, weights=α)
    ranges_y, y_c, β_c = cluster_ranges_centroids(y, y_lab, weights=β)

    # Sort the clusters, making them contiguous in memory:
    (α, x), x_lab = sort_clusters( (α, x), x_lab)
    (β, y), y_lab = sort_clusters( (β, y), y_lab)

    with torch.no_grad():  # Compute our block-sparse reduction ranges:
        # Compute pairwise distances between clusters:
        C_xx = squared_distances( x_c, x_c)
        C_yy = squared_distances( y_c, y_c)
        C_xy = squared_distances( x_c, y_c)

        # Compute the boolean masks:
        keep_xx = ( C_xx <= (truncate + 2 * diameter)**2 )
        keep_yy = ( C_yy <= (truncate + 2 * diameter)**2 )
        keep_xy = ( C_xy <= (truncate + 2 * diameter)**2 )

        # Compute the KeOps reduction ranges:
        ranges_xx = from_matrix(ranges_x, ranges_x, keep_xx)
        ranges_yy = from_matrix(ranges_y, ranges_y, keep_yy)
        ranges_xy = from_matrix(ranges_x, ranges_y, keep_xy)
    
    return kernel_keops(kernel, α, x, β, y, 
                ranges_xx=ranges_xx, ranges_yy=ranges_yy, ranges_xy=ranges_xy)

