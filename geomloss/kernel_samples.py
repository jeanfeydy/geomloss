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
    from pykeops.torch.cluster import (
        grid_cluster,
        cluster_ranges_centroids,
        sort_clusters,
        from_matrix,
        swap_axes,
    )
    from pykeops.torch import LazyTensor

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
        return 2 * grad_output


def double_grad(x):
    return DoubleGrad.apply(x)


# ==============================================================================
#                               All backends
# ==============================================================================


def gaussian_kernel(x, y, blur=0.05, use_keops=False, ranges=None):
    C2 = squared_distances(x / blur, y / blur, use_keops=use_keops)
    K = (-C2 / 2).exp()

    if use_keops and ranges is not None:
        K.ranges = ranges
    return K


def laplacian_kernel(x, y, blur=0.05, use_keops=False, ranges=None):
    C = distances(x / blur, y / blur, use_keops=use_keops)
    K = (-C).exp()

    if use_keops and ranges is not None:
        K.ranges = ranges
    return K


def energy_kernel(x, y, blur=None, use_keops=False, ranges=None):
    # N.B.: We never truncate the energy distance kernel
    return -distances(x, y, use_keops=use_keops)


kernel_routines = {
    "gaussian": gaussian_kernel,
    "laplacian": laplacian_kernel,
    "energy": energy_kernel,
}


def kernel_loss(
    α,
    x,
    β,
    y,
    blur=0.05,
    kernel=None,
    name=None,
    potentials=False,
    use_keops=False,
    ranges_xx=None,
    ranges_yy=None,
    ranges_xy=None,
    **kwargs
):
    if kernel is None:
        kernel = kernel_routines[name]

    # Center the point clouds just in case, to prevent numeric overflows:
    # N.B.: This may break user-provided kernels and comes at a non-negligible
    #       cost for small problems, so let's disable this by default.
    # center = (x.mean(-2, keepdim=True) + y.mean(-2, keepdim=True)) / 2
    # x, y = x - center, y - center

    # (B,N,N) tensor
    K_xx = kernel(
        double_grad(x), x.detach(), blur=blur, use_keops=use_keops, ranges=ranges_xx
    )
    # (B,M,M) tensor
    K_yy = kernel(
        double_grad(y), y.detach(), blur=blur, use_keops=use_keops, ranges=ranges_yy
    )
    # (B,N,M) tensor
    K_xy = kernel(x, y, blur=blur, use_keops=use_keops, ranges=ranges_xy)

    # (B,N,N) @ (B,N) = (B,N)
    a_x = (K_xx @ α.detach().unsqueeze(-1)).squeeze(-1)
    # (B,M,M) @ (B,M) = (B,M)
    b_y = (K_yy @ β.detach().unsqueeze(-1)).squeeze(-1)
    # (B,N,M) @ (B,M) = (B,N)
    b_x = (K_xy @ β.unsqueeze(-1)).squeeze(-1)

    if potentials:
        # (B,M,N) @ (B,N) = (B,M)
        Kt = K_xy.t() if use_keops else K_xy.transpose(1, 2)
        a_y = (Kt @ α.unsqueeze(-1)).squeeze(-1)
        return a_x - b_x, b_y - a_y

    else:  # Return the Kernel norm. N.B.: we assume that 'kernel' is symmetric:
        batch = x.dim() > 2
        return (
            0.5 * scal(double_grad(α), a_x, batch=batch)
            + 0.5 * scal(double_grad(β), b_y, batch=batch)
            - scal(α, b_x, batch=batch)
        )


# ==============================================================================
#                          backend == "tensorized"
# ==============================================================================

from functools import partial

kernel_tensorized = partial(kernel_loss, use_keops=False)


# ==============================================================================
#                           backend == "online"
# ==============================================================================

kernel_online = partial(kernel_loss, use_keops=True)


# ==============================================================================
#                          backend == "multiscale"
# ==============================================================================


def max_diameter(x, y):
    mins = torch.stack((x.min(dim=0)[0], y.min(dim=0)[0])).min(dim=0)[0]
    maxs = torch.stack((x.max(dim=0)[0], y.max(dim=0)[0])).max(dim=0)[0]
    diameter = (maxs - mins).norm().item()
    return diameter


def kernel_multiscale(
    α,
    x,
    β,
    y,
    blur=0.05,
    kernel=None,
    name=None,
    truncate=5,
    diameter=None,
    cluster_scale=None,
    potentials=False,
    verbose=False,
    **kwargs
):

    if truncate is None or name == "energy":
        return kernel_online(
            α.unsqueeze(0),
            x.unsqueeze(0),
            β.unsqueeze(0),
            y.unsqueeze(0),
            blur=blur,
            kernel=kernel,
            truncate=truncate,
            name=name,
            potentials=potentials,
            **kwargs
        )

    # Renormalize our point cloud so that blur = 1:
    # Center the point clouds just in case, to prevent numeric overflows:
    center = (x.mean(-2, keepdim=True) + y.mean(-2, keepdim=True)) / 2
    x, y = x - center, y - center
    x_ = x / blur
    y_ = y / blur

    # Don't forget to normalize the diameter too!
    if cluster_scale is None:
        D = x.shape[-1]
        if diameter is None:
            diameter = max_diameter(x_.view(-1, D), y_.view(-1, D))
        else:
            diameter = diameter / blur
        cluster_scale = diameter / (np.sqrt(D) * 2000 ** (1 / D))

    # Put our points in cubic clusters:
    cell_diameter = cluster_scale * np.sqrt(x_.shape[-1])
    x_lab = grid_cluster(x_, cluster_scale)
    y_lab = grid_cluster(y_, cluster_scale)

    # Compute the ranges and centroids of each cluster:
    ranges_x, x_c, α_c = cluster_ranges_centroids(x_, x_lab, weights=α)
    ranges_y, y_c, β_c = cluster_ranges_centroids(y_, y_lab, weights=β)

    if verbose:
        print(
            "{}x{} clusters, computed at scale = {:2.3f}".format(
                len(x_c), len(y_c), cluster_scale
            )
        )

    # Sort the clusters, making them contiguous in memory:
    (α, x), x_lab = sort_clusters((α, x), x_lab)
    (β, y), y_lab = sort_clusters((β, y), y_lab)

    with torch.no_grad():  # Compute our block-sparse reduction ranges:
        # Compute pairwise distances between clusters:
        C_xx = squared_distances(x_c, x_c)
        C_yy = squared_distances(y_c, y_c)
        C_xy = squared_distances(x_c, y_c)

        # Compute the boolean masks:
        keep_xx = C_xx <= (truncate + cell_diameter) ** 2
        keep_yy = C_yy <= (truncate + cell_diameter) ** 2
        keep_xy = C_xy <= (truncate + cell_diameter) ** 2

        # Compute the KeOps reduction ranges:
        ranges_xx = from_matrix(ranges_x, ranges_x, keep_xx)
        ranges_yy = from_matrix(ranges_y, ranges_y, keep_yy)
        ranges_xy = from_matrix(ranges_x, ranges_y, keep_xy)

    return kernel_loss(
        α,
        x,
        β,
        y,
        blur=blur,
        kernel=kernel,
        name=name,
        potentials=potentials,
        use_keops=True,
        ranges_xx=ranges_xx,
        ranges_yy=ranges_yy,
        ranges_xy=ranges_xy,
    )
