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

import torch
try:  # Import the keops library, www.kernel-operations.io
    from pykeops.torch import generic_sum
    keops_available = True
except:
    keops_available = False

def scal(α, f) :
    return torch.dot( α.view(-1), f.view(-1) )


# ==============================================================================
#                          backend == "tensorized"
# ==============================================================================

def squared_distances(x,y):

    D_xx = (x*x).sum(-1).unsqueeze(2)  # (B,N,1)
    D_xy = (x.unsqueeze())  # (B,N,1,D) @ (B,1,M,D) = (B,N,M)


def gaussian_kernel(x, y, blur=.05):
    C = squared_distances(x, y)
    return (- C / (2 * blur**2)).exp()


def laplacian_kernel(x, y, blur=.05):
    C = distances(x, y)
    return (- C / blur).exp()


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

    K_xx = kernel( x, x, blur=blur)  # (B,N,N) tensor
    K_xy = kernel( x, y, blur=blur)  # (B,N,M) tensor
    K_yy = kernel( y, y, blur=blur)  # (B,M,M) tensor

    a_i = torch.matmul( K_xx, α.unsqueeze(-1) ).squeeze(-1)  # (B,N,N) @ (B,N) = (B,N) 
    b_i = torch.matmul( K_xy, β.unsqueeze(-1) ).squeeze(-1)  # (B,N,M) @ (B,M) = (B,N) 
    b_j = torch.matmul( K_yy, β.unsqueeze(-1) ).squeeze(-1)  # (B,M,M) @ (B,M) = (B,M)

    # N.B.: we assume that 'kernel' is symmetric:
    return (α * (.5 * a_i - b_i)).sum(1) + .5*(β * b_j).sum(1) 



# ==============================================================================
#                           backend == "online"
# ==============================================================================

def kernel_online(α, x, β, y, blur=.05, kernel=None, name=None, **kwargs):

    if not keops_available:
        raise ImportError("The 'pykeops' library could not be loaded: 'online' backend is not available.")


# ==============================================================================
#                          backend == "multiscale"
# ==============================================================================

def kernel_mumltiscale(α, x, β, y, blur=.05, kernel=None, truncate=None, name=None, **kwargs):

    if not keops_available:
        raise ImportError("The 'pykeops' library could not be loaded: 'multiscale' backend is not available.")

