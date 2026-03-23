import torch
from .utils import log_dens, pyramid, upsample, softmin_grid
from .sinkhorn_divergence import epsilon_schedule, scaling_parameters
from .sinkhorn_divergence import sinkhorn_cost, sinkhorn_loop


def extrapolate(f_ba, g_ab, eps, damping, C_xy, b_log, C_xy_fine):
    return upsample(f_ba)


def kernel_truncation(
    C_xy,
    C_yx,
    C_xy_fine,
    C_yx_fine,
    f_ba,
    g_ab,
    eps,
    truncate=None,
    cost=None,
    verbose=False,
):
    return C_xy_fine, C_yx_fine


def sinkhorn_divergence(
    a,
    b,
    p=2,
    blur=None,
    reach=None,
    axes=None,
    scaling=0.5,
    cost=None,
    debias=True,
    potentials=False,
    verbose=False,
    **kwargs,
):
    r"""Sinkhorn divergence between measures supported on 1D/2D/3D grids.

    Args:
        a ((B, Nx), (B, Nx, Ny) or (B, Nx, Ny, Nz) Tensor): Weights :math:`\alpha_i`
            for the first measure, with a batch dimension.

        b ((B, Nx), (B, Nx, Ny) or (B, Nx, Ny, Nz) Tensor): Weights :math:`\beta_j`
            for the second measure, with a batch dimension.

        p (int, optional): Exponent of the ground cost function
            :math:`C(x_i,y_j)`, which is equal to
            :math:`\tfrac{1}{p}\|x_i-y_j\|^p` if it is not provided
            explicitly through the `cost` optional argument.
            Defaults to 2.

        blur (float or None, optional): Target value for the blurring scale
            of the "point spread function" or Gibbs kernel
            :math:`K_{i,j} = \exp(-C(x_i,y_j)/\varepsilon) = \exp(-\|x_i-y_j\|^p / p \text{blur}^p).
            In the Sinkhorn algorithm, the temperature :math:`\varepsilon`
            is computed as :math:`\text{blur}^p`.
            Defaults to None: we pick the smallest pixel size across
            the Nx, Ny and Nz dimensions (if applicable).

        axes (tuple of pairs of floats or None (= [0, 1)^(1/2/3)), optional):
            Dimensions of the image domain, specified through a 1/2/3-uple
            of [vmin, vmax] bounds.
            For instance, if the batched 2D images correspond to sampled
            measures on [-10, 10) x [-3, 5), you may use "axes = ([-10, 10], [-3, 5])".
            The (implicit) pixel coordinates are computed using a "torch.linspace(...)"
            across each dimension: along any given axis, the spacing between two pixels
            is equal to "(vmax - vmin) / npixels".

            Defaults to None: we assume that the signal / image / volume
            is sampled on the unit interval [0, 1) / square [0, 1)^2 / cube [0, 1)^3.

        scaling (float in (0, 1), optional): Ratio between two successive
            values of the blur radius in the epsilon-scaling annealing descent.
            Defaults to 0.5.

        cost (function or None, optional): ...
            Defaults to None: we use a Euclidean cost
            :math:`C(x_i,y_j) = \tfrac{1}{p}\|x_i-y_j\|^p`.

        debias (bool, optional): Should we used the "de-biased" Sinkhorn divergence
            :math:`\text{S}_{\varepsilon, \rho}(\al,\be)` instead
            of the "raw" entropic OT cost
            :math:`\text{OT}_{\varepsilon, \rho}(\al,\be)`?
            This slows down the OT solver but guarantees that our approximation
            of the Wasserstein distance will be positive and definite
            - up to convergence of the Sinkhorn loop.
            For a detailed discussion of the influence of this parameter,
            see e.g. Fig. 3.21 in Jean Feydy's PhD thesis.
            Defaults to True.

        potentials (bool, optional): Should we return the optimal dual potentials
            instead of the cost value?
            Defaults to False.

    Returns:
        (B,) Tensor or pair of (B, Nx, ...), (B, Nx, ...) Tensors: If `potentials` is True,
            we return a pair of (B, Nx, ...), (B, Nx, ...) Tensors that encode the optimal
            dual vectors, respectively supported by :math:`x_i` and :math:`y_j`.
            Otherwise, we return a (B,) Tensor of values for the Sinkhorn divergence.
    """

    if blur is None:
        blur = 1 / a.shape[-1]

    # Pre-compute a multiscale decomposition (=Binary/Quad/OcTree)
    # of the input measures, stored as logarithms
    a_s, b_s = pyramid(a)[1:], pyramid(b)[1:]
    a_logs = list(map(log_dens, a_s))
    b_logs = list(map(log_dens, b_s))

    # By default, our cost function :math:`C(x_i,y_j)` is a halved,
    # squared Euclidean distance (p=2) or a simple Euclidean distance (p=1):
    depth = len(a_logs)
    if cost is None:
        C_s = [p] * depth  # Dummy "cost matrices"
    else:
        raise NotImplementedError()

    # Diameter of the configuration:
    diameter = 1
    # Target temperature epsilon:
    eps = blur**p
    # Strength of the marginal constraints:
    rho = None if reach is None else reach**p

    # Schedule for the multiscale descent, with ε-scaling:
    """
    sigma = diameter
    for n in range(depth):
        for _ in range(scaling_N):  # Number of steps per scale
            eps_list.append(sigma ** p)

            # Decrease the kernel radius, making sure that
            # the radius sigma is divided by two at every scale until we reach
            # the target value, "blur":
            scale = max(sigma * (2 ** (-1 / scaling_N)), blur)

    jumps = [scaling_N * (i + 1) - 1 for i in range(depth - 1)]
    """
    if scaling < 0.5:
        raise ValueError(
            f"Scaling value of {scaling} is too small: please use a number in [0.5, 1)."
        )

    diameter, eps, eps_list, rho = scaling_parameters(
        None, None, p, blur, reach, diameter, scaling
    )

    # List of pixel widths:
    pyramid_scales = [diameter / a.shape[-1] for a in a_s]
    if verbose:
        print("Pyramid scales:", pyramid_scales)

    current_scale = pyramid_scales.pop(0)
    jumps = []
    for i, eps in enumerate(eps_list[1:]):
        if current_scale**p > eps:
            jumps.append(i + 1)
            current_scale = pyramid_scales.pop(0)

    if verbose:
        print("Temperatures: ", eps_list)
        print("Jumps: ", jumps)

    assert (
        len(jumps) == len(a_s) - 1
    ), "There's a bug in the multicale pre-processing..."

    # Use an optimal transport solver to retrieve the dual potentials:
    f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
        softmin_grid,
        a_logs,
        b_logs,
        C_s,
        C_s,
        C_s,
        C_s,
        eps_list,
        rho,
        jumps=jumps,
        kernel_truncation=kernel_truncation,
        extrapolate=extrapolate,
        debias=debias,
    )

    # Optimal transport cost:
    return sinkhorn_cost(
        eps,
        rho,
        a,
        b,
        f_aa,
        g_bb,
        g_ab,
        f_ba,
        batch=True,
        debias=debias,
        potentials=potentials,
    )
