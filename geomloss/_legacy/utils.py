import numpy as np
import torch
from torch.nn.functional import conv1d, avg_pool2d, avg_pool3d, interpolate


try:  # Import the keops library, www.kernel-operations.io
    from pykeops.torch import LazyTensor

    keops_available = True
except:
    keops_available = False


def scal(a, f, batch=False):
    if batch:
        B = a.shape[0]
        return (a.reshape(B, -1) * f.reshape(B, -1)).sum(1)
    else:
        return torch.dot(a.reshape(-1), f.reshape(-1))


#######################################
# On point clouds
#######################################


def squared_distances(x, y, use_keops=False):
    if use_keops and keops_available:
        if x.dim() == 2:
            x_i = LazyTensor(x[:, None, :])  # (N,1,D)
            y_j = LazyTensor(y[None, :, :])  # (1,M,D)
        elif x.dim() == 3:  # Batch computation
            x_i = LazyTensor(x[:, :, None, :])  # (B,N,1,D)
            y_j = LazyTensor(y[:, None, :, :])  # (B,1,M,D)
        else:
            print("x.shape : ", x.shape)
            raise ValueError("Incorrect number of dimensions")

        return ((x_i - y_j) ** 2).sum(-1)

    else:
        if x.dim() == 2:
            D_xx = (x * x).sum(-1).unsqueeze(1)  # (N,1)
            D_xy = torch.matmul(x, y.permute(1, 0))  # (N,D) @ (D,M) = (N,M)
            D_yy = (y * y).sum(-1).unsqueeze(0)  # (1,M)
        elif x.dim() == 3:  # Batch computation
            D_xx = (x * x).sum(-1).unsqueeze(2)  # (B,N,1)
            D_xy = torch.matmul(x, y.permute(0, 2, 1))  # (B,N,D) @ (B,D,M) = (B,N,M)
            D_yy = (y * y).sum(-1).unsqueeze(1)  # (B,1,M)
        else:
            print("x.shape : ", x.shape)
            raise ValueError("Incorrect number of dimensions")

        return D_xx - 2 * D_xy + D_yy


def distances(x, y, use_keops=False):
    if use_keops:
        return squared_distances(x, y, use_keops=use_keops).sqrt()

    else:
        return torch.sqrt(torch.clamp_min(squared_distances(x, y), 1e-8))


#######################################
# On grids
#######################################


BATCH, CHANNEL, HEIGHT, WIDTH, DEPTH = 0, 1, 2, 3, 4


def dimension(I):
    """Returns 2 if we are working with 2D images and 3 for volumes."""
    return I.dim() - 2


subsample = {
    2: (lambda x: 4 * avg_pool2d(x, 2)),
    3: (lambda x: 8 * avg_pool3d(x, 2)),
}

upsample_mode = {
    2: "bilinear",
    3: "trilinear",
}


def pyramid(I):
    D = dimension(I)
    I_s = [I]

    for i in range(int(np.log2(I.shape[HEIGHT]))):
        I = subsample[D](I)
        I_s.append(I)

    I_s.reverse()
    return I_s


def upsample(I):
    D = dimension(I)
    return interpolate(I, scale_factor=2, mode=upsample_mode[D], align_corners=False)


def log_dens(α):
    α_log = α.log()
    α_log[α <= 0] = -10000.0
    return α_log


########################
# "Hard" C-transform:
#


def C_transform(G, tau=1, p=2):
    """
    Computes the forward C-transform of an array G of shape:
     - (Batch, Nx)         in 1D
     - (Batch, Nx, Ny)     in 2D
     - (Batch, Nx, Ny, Nz) in 3D

    i.e.
    F(x_i) <- max_j [G(x_j) - C(x_i, x_j)]

    with:
    C(x,y) = |x-y|^p / (p * tau)

    In this first demo, we assume that:
      - We are working with square images: Nx = Ny = Nz = N.
      - p = 1 or 2  (Manhattan or Euclidean distance).
      - Pixels have unit length in all dimensions.
    """
    D = G.ndim - 1  # D = 1, 2 or 3
    B, N = G.shape[0], G.shape[1]

    x = torch.arange(N).type_as(G)  # [0, ..., N-1], on the same device as G.
    if p == 1:
        x = x / tau
    if p == 2:
        x = x / np.sqrt(2 * tau)
    else:
        raise NotImplementedError()

    if not keops_available:
        raise ImportError("This routine depends on the pykeops library.")

    def lines(g):
        g = g.contiguous()  # Make sure that g is not "transposed" implicitely,
        # but stored as a contiguous array of numbers.

        g_j = LazyTensor(g.view(-1, 1, N, 1))
        x_i = LazyTensor(x.view(1, N, 1, 1))
        x_j = LazyTensor(x.view(1, 1, N, 1))

        if p == 1:
            Cg_ij = g_j - (x_i - x_j).abs()  # (B * N, N, N, 1)
        elif p == 2:
            Cg_ij = g_j - (x_i - x_j) ** 2  # (B * N, N, N, 1)

        f_i = Cg_ij.max(dim=2)  # (B * N, N, 1)

        if D == 1:
            return f_i.view(B, N)
        elif D == 2:
            return f_i.view(B, N, N)
        elif D == 3:
            return f_i.view(B, N, N, N)

    if D == 1:
        G = lines(G)

    if D == 2:
        G = lines(G)  # Act on lines
        G = lines(G.permute([0, 2, 1])).permute([0, 2, 1])  # Act on columns

    elif D == 3:
        G = lines(G)  # Act on dim 4
        G = lines(G.permute([0, 1, 3, 2])).permute([0, 1, 3, 2])  # Act on dim 3
        G = lines(G.permute([0, 3, 2, 1])).permute([0, 3, 2, 1])  # Act on dim 2

    return G


########################
# "Soft" C-transform:
#


def softmin_grid(eps, C_xy, h_y):
    r"""Soft-C-transform, implemented using seperable KeOps operations.

    This routine implements the (soft-)C-transform
    between dual vectors, which is the core computation for
    Auction- and Sinkhorn-like optimal transport solvers.

    If `eps` is a float number, `C_xy` is a tuple of axes dimensions
    and `h_y` encodes a dual potential :math:`h_j` that is supported by the 1D/2D/3D grid
    points :math:`y_j`'s, then `softmin_tensorized(eps, C_xy, h_y)` returns a dual potential
    `f` for ":math:`f_i`", supported by the :math:`x_i`'s, that is equal to:

    .. math::
        f_i \gets - \varepsilon \log \sum_{j=1}^{\text{M}} \exp
        \big[ h_j - C(x_i, y_j) / \varepsilon \big]~.

    For more detail, see e.g. Section 3.3 and Eq. (3.186) in Jean Feydy's PhD thesis.

    Args:
        eps (float, positive): Temperature :math:`\varepsilon` for the Gibbs kernel
            :math:`K_{i,j} = \exp(-C(x_i, y_j) / \varepsilon)`.

        C_xy (): Encodes the implicit cost matrix :math:`C(x_i,y_j)`.

        h_y ((B, Nx), (B, Nx, Ny) or (B, Nx, Ny, Nz) Tensor):
            Grid of logarithmic "dual" values, with a batch dimension.
            Most often, this image will be computed as `h_y = b_log + g_j / eps`,
            where `b_log` is an array of log-weights :math:`\log(\beta_j)`
            for the :math:`y_j`'s and :math:`g_j` is a dual variable
            in the Sinkhorn algorithm, so that:

            .. math::
                f_i \gets - \varepsilon \log \sum_{j=1}^{\text{M}} \beta_j
                \exp \tfrac{1}{\varepsilon} \big[ g_j - C(x_i, y_j) \big]~.

    Returns:
        (B, Nx), (B, Nx, Ny) or (B, Nx, Ny, Nz) Tensor: Dual potential `f` of values
            :math:`f_i`, supported by the points :math:`x_i`.
    """
    D = dimension(h_y)
    B, K, N = h_y.shape[BATCH], h_y.shape[CHANNEL], h_y.shape[WIDTH]

    if not keops_available:
        raise ImportError("This routine depends on the pykeops library.")

    x = torch.arange(N).type_as(h_y) / N
    p = C_xy
    if p == 1:
        x = x / eps
    elif p == 2:
        x = x / np.sqrt(2 * eps)
    else:
        raise NotImplementedError()

    def softmin(a_log):
        a_log = a_log.contiguous()
        # print(a_log.shape)
        a_log_j = LazyTensor(a_log.view(-1, 1, N, 1))
        x_i = LazyTensor(x.view(1, N, 1, 1))
        x_j = LazyTensor(x.view(1, 1, N, 1))

        if p == 1:
            kA_log_ij = a_log_j - (x_i - x_j).abs()  # (B * N, N, N, 1)
        elif p == 2:
            kA_log_ij = a_log_j - (x_i - x_j) ** 2  # (B * N, N, N, 1)

            # kA_log_ij =  (x_i - x_j)**2 - g_j

        # print(kA_log_ij)
        kA_log = kA_log_ij.logsumexp(dim=2)  # (B * N, N, 1)

        if D == 2:
            return kA_log.view(B, K, N, N)
        elif D == 3:
            return kA_log.view(B, K, N, N, N)

    if D == 2:
        h_y = softmin(h_y)  # Act on lines
        h_y = softmin(h_y.permute([0, 1, 3, 2])).permute([0, 1, 3, 2])  # Act on columns

    elif D == 3:
        h_y = softmin(h_y)  # Act on dim 4
        h_y = softmin(h_y.permute([0, 1, 2, 4, 3])).permute(
            [0, 1, 2, 4, 3]
        )  # Act on dim 3
        h_y = softmin(h_y.permute([0, 1, 4, 3, 2])).permute(
            [0, 1, 4, 3, 2]
        )  # Act on dim 2

    return -eps * h_y
