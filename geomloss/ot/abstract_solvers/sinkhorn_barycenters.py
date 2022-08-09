from ... import backends as bk
from ...typing import (
    List,
    RealTensor,
    Tuple,
    Optional,
    SoftMin,
    DescentParameters,
    CostMatrices,
    Extrapolator,
)


def barycenter_iteration(
    *,
    softmin: SoftMin,
    f_k: RealTensor,
    g_k: RealTensor,
    log_d: RealTensor,
    eps: float,
    C: CostMatrices,
    log_b_k: RealTensor,
    w_k: RealTensor,
) -> Tuple[RealTensor, RealTensor, RealTensor, RealTensor]:
    """Implements a Sinkhorn iteration for the Wasserstein barycenter problem.

    Args:
        softmin (function): a softmin (~ soft distance) operator,
            as in the standard Sinkhorn loop.

        f_k ((B,K,...) Tensor): dual potentials, supported by the barycenters
            on points x[i].

        g_k ((B,K,...) Tensor): dual potentials, supported by the input measures
            on points y[j].

        log_d ((B,1,...) Tensor): de-biasing weight, supported by the barycenters
            on points x[i].

        eps (float > 0): the positive regularization parameter (= temperature)
            for the Sinkhorn algorithm.

        C (CostMatrices): a NamedTuple of objects that encode the (B,K)
            cost matrices C.xx[i,j] = C(x[i], x[j]), C.xy[i,j] = C(x[i], y[j])
            and C.yx[i,j] = C(y[i], x[j]) in a way that is compatible with softmin.

        log_b_k ((B,K,...) Tensor): logarithms of the measure weights ("masses")
            for the input measures.

        w_k ((B,K) Tensor): desired weights for the barycentric interpolation.
    """

    # Sinkhorn "pseudo-step" - from the measures to the barycenter:
    ft_k = softmin(eps, C.xy, log_b_k + g_k / eps)  # (B,K,...)
    # Update the barycenter:
    # (B,1,...) = (B,1,...) - (B,K,...) @ (B,K,1,..,1)
    # With e.g. 2 trailing dimensions, the einsum below is equivalent to:
    # log_bar = log_d - (ft_k * w_k[:,:,None,None]).sum(1, keepdim=True) / eps
    log_bar = log_d - bk.einsum("bk..., bk -> b...", ft_k, w_k)[:, None, ...] / eps

    # Symmetric Sinkhorn updates:
    # From the measures to the barycenter:
    ft_k = softmin(eps, C.xy, log_b_k + g_k / eps)  # (B,K,n,n)
    # From the barycenter to the measures:
    gt_k = softmin(eps, C.yx, log_bar + f_k / eps)  # (B,K,n,n)
    f_k = (f_k + ft_k) / 2
    g_k = (g_k + gt_k) / 2

    # Sinkhorn "pseudo-step" - from the measures to the barycenter:
    ft_k = softmin(eps, C.xy, log_b_k + g_k / eps)
    # Update the barycenter - log_bar is (B,1,...):
    log_bar = log_d - bk.einsum("bk..., bk -> b...", ft_k, w_k)[:, None, ...] / eps

    # Update the de-biasing measure:
    # (B,1,...) = (B,1,...) + (B,1,...) + (B,1,...)
    log_d = 0.5 * (log_d + log_bar + softmin(eps, C.xx, log_d) / eps)

    return f_k, g_k, log_d, log_bar


def sinkhorn_barycenter_loop(
    *,
    softmin: SoftMin,
    log_b_k_list: List[RealTensor],
    w_k: RealTensor,
    C_list: List[CostMatrices],
    descent: DescentParameters,
    extrapolate: Optional[Extrapolator] = None,
    backward_iterations: int = 5,
):
    r"""Implements the (possibly multiscale) symmetric Sinkhorn loop for barycenters.

    We solve:
        A* = arg min_{probability distribution A} Sum_{k=1}^K [ w_k * S_e(A, B_k) ]

    Where:
    - S_e denotes the de-biased Sinkhorn divergence at temperature e(psilon),
    - w_k >= 0 are barycentric weights that sum up to 1,
    - B_k are K probability distributions.

    This function runs batch-wise: we compute B Wasserstein barycenters over
    B*K probability measures in parallel.

    The solver that is implemented below assumes that the support of the barycenter
    A* is known in advance and only optimizes the distribution of mass on this support.
    It is generally well-suited to the computation of barycenters in low-dimensional spaces
    (especially on 2D/3D grids), where the support of A* is easy to sample as the
    convex hull of the supports of the B_k.

    However, it is far from being optimal in high-dimensional settings, where estimating
    the location of the support of A* without having to use an exponential number
    of samples is the "hard" part of the Wasserstein barycenter computation.

    For information on why the Wasserstein barycenter problem is hard in high dimensions,
    see e.g. the two papers by Jason M. Altschuler and Enric Boix-Adsera:
    - Wasserstein barycenters are NP-hard to compute (2021),
    - Wasserstein barycenters can be bomputed in polynomial time in fixed dimension (2021).


    This file implements ideas that have been presented in two separate works:

    - Debiased Sinkhorn barycenters (2020) by Hicham Janati, Marco Cuturi
    and Alexandre Gramfort, for the introduction of the de-biasing density
    and the general structure of the algorithm.

    - Chapter 3.3 in Jean Feydy's PhD thesis:
    Geometric data analysis, beyond convolutions (2020),
    https://www.jeanfeydy.com/geometric_data_analysis.pdf,
    for the definition of the Sinkhorn divergence, the annealing heuristic,
    the multiscale scheme, de-biasing and stable log-sum-exp computations.

    Args:
        softmin (function): This routine must implement the (soft-)C-transform
            between dual vectors, which is the core computation for
            Auction- and Sinkhorn-like optimal transport solvers.
            If `eps` is a float number, `C_xy` encodes a cost matrix :math:`C(x_i,y_j)`
            and `g` encodes a dual potential :math:`g_j` that is supported by the points
            :math:`y_j`'s, then `softmin(eps, C_xy, g)` must return a dual potential
            `f` for ":math:`f_i`", supported by the :math:`x_i`'s, that is equal to:

            .. math::
                f_i \gets - \varepsilon \log \sum_{j=1}^{\text{M}} \exp
                \big[ g_j - C(x_i, y_j) / \varepsilon \big]~.

            For more detail, see e.g. Section 3.3 and Eq. (3.186) in Jean Feydy's PhD thesis.

        log_b_k_list (list of S real-valued Tensors of shape (B,K,...)):
            List of log-weights :math:`\log(\beta^k_i)` for the target measure,
            at S different scales (from coarse to fine).

        w_k ((B,K) Tensor): desired weights for the barycentric interpolation.

        C_list (list of S CostMatrices): List of NamedTuples with attributes
            that represent the cost matrix at S different scales.
            These will be passed to the `softmin` function as second arguments.
            We expect the attributes:
            - `xx` for the cost matrix :math:`C_{xx}[i,j] = C(x[i], x[j])`.
            - `yy` for the cost matrix :math:`C_{yy}[i,j] = C(y[i], y[j])`.
            - `xy` for the cost matrix :math:`C_{xy}[i,j] = C(x[i], y[j])`.
            - `yx` for the cost matrix :math:`C_{yx}[i,j] = C(y[i], x[j])`.
            Where x[i] denotes the positions of the samples for the barycenter(s),
            and y[j] denotes the positions of the samples for the input measures.

        descent (DescentParameters): A NamedTuple with attributes that describe
            the evolution of our main parameters along the iterations of the
            Sinkhorn loop.
            We expect the attributes:
            - eps_list (list of n_iter float > 0): List of successive values for
              the Sinkhorn regularization parameter, the temperature :math:`\varepsilon`.
              The number of iterations in the loop is equal to the length of this list
              + the number of "backward_iterations".

            - scale_list (list of n_iter int): List of scale indices at which we
              perform our iterations.
              Each scale index should satisfy `0 <= scale < S`.
              For single-scale mode, use `scale_list = [0] * n_iter`.

        extrapolate (function, optional): Coarse-to-fine extrapolation for the
            dual potentials. If
            `self` is a dual potential that is supported by the :math:`x_i`'s,
            `other` is a dual potential that is supported by the :math:`y_j`'s,
            `log_weights` denotes the log-weights :math:`\log(\beta_j)`
            that are supported by the :math:`y_j`'s at the coarse resolution,
            `C` encodes the cost matrix :math:`C(x_i, y_j)` at the current
            ("coarse") resolution,
            `C_fine` encodes the cost matrix :math:`C(x_i, y_j)` at the next
            ("fine") resolution,
            `eps` is the current value of the temperature :math:`\varepsilon`,
            `dampen` is a pointwise dampening function for unbalanced OT,
            then
            `extrapolate(self, other, log_weights, C, C_fine, eps, dampen)`
            will be used to compute the new values of the dual potential
            `self` on the point cloud :math:`x_i` at a finer resolution.
            This function may either use a simple bi/tri-linear interpolation
            method on the first potential (`self`), or rely on the analytic
            expression of the dual potential that is induced by the
            :math:`y_j`'s and :math:`\beta_j`'s.
            Defaults to None: this function is not needed in single-scale mode.

        backward_iterations (int, optional): Number of extra iterations,
            performed at the end of the multiscale Sinkhorn loop,
            with autograd enabled. If you are interested in the gradient of the
            Wasserstein barycenter with respect to the input measures,
            a larger number ensures a better approximation
            of this gradient in the backward step. Defaults to 5.

    Returns:
        (B,1,...) real-valued Tensor: Solutions for the B Wasserstein barycenter problem
            that have been solved in parallel.
            These weights correspond to the sample positions at the finest scale.
    """

    with bk.set_grad_enabled(backward_iterations == 0 and bk.is_grad_enabled()):
        # We (usually) start at the coarsest resolution available:
        scale = descent.scale_list[0]  # Scale index
        log_b_k = log_b_k_list[scale]  # (B,K,...) log-weights for the measures
        C = C_list[scale]  # implicit (B,K,N,M), (B,K,N,N)... cost matrices

        # First value of the temperature (typically, eps = diameter**p):
        eps = descent.eps_list[0]  # eps = temperature

        # Initialize the dual variables:
        # (B,K,...) tensor, supported by the barycenter points x:
        f_k = softmin(eps, C.xy, log_b_k)
        # TODO: the line below is not great...
        g_k = softmin(eps, C.yx, log_b_k)  # (B,K,...), supported by the input points y

        # Logarithm of the debiasing term:
        log_d = bk.sum(bk.ones_like(log_b_k), axis=1, keepdims=True)  # (B,1,...)
        # Normalize each of these:
        log_d = log_d - bk.logsumexp(log_d, axis=log_d.shape[2:], keepdims=True)

        # Multiscale descent, with eps-scaling ----------------------------------

        # See Fig. 3.25-26 in Jean Feydy's PhD thesis for intuitions.
        for i, eps in enumerate(descent.eps_list):
            f_k, g_k, log_d, log_bar = barycenter_iteration(
                softmin=softmin,
                f_k=f_k,
                g_k=g_k,
                log_d=log_d,
                eps=eps,
                C=C,
                log_b_k=log_b_k,
                w_k=w_k,
            )

            # In single-scale mode, scale_list = [0, ..., 0]: we never run the code below.
            if i + 1 < len(descent.scale_list) and scale != descent.scale_list[i + 1]:

                # "Full" cost matrix at the next scale:
                next_scale = descent.scale_list[i + 1]
                C_fine = C_list[next_scale]

                # N.B.: this code does not currently support unbalanced OT
                dampen = None

                # The extrapolation formulas are coherent with the
                # softmin(...) updates of the barycenter iteration.
                f_k = extrapolate(
                    self=f_k,
                    other=g_k,
                    log_weights=log_b_k,
                    C=C.xy,
                    C_fine=C_fine.xy,
                    eps=eps,
                    dampen=dampen,
                )

                g_k = extrapolate(
                    self=g_k,
                    other=f_k,
                    log_weights=log_bar,
                    C=C.yx,
                    C_fine=C_fine.yx,
                    eps=eps,
                    dampen=dampen,
                )

                # N.B.: This has not been tested at all outside of grids.
                log_d = extrapolate(
                    self=log_d,
                    other=0 * log_d,
                    log_weights=log_d,
                    C=C.xx,
                    C_fine=C_fine.xx,
                    eps=eps,
                    dampen=dampen,
                )

                # Update the representations and cost matrices at a finer scale:
                scale = next_scale
                C = C_fine  # = C_list[scale]
                eps = descent.eps_list[scale]  # eps = temperature
                log_b_k = log_b_k_list[scale]  # (B,K,...) log-weights

    # N.B.: PyTorch autograd may be enabled here.
    for _ in range(backward_iterations):
        f_k, g_k, log_d, log_bar = barycenter_iteration(
            softmin=softmin,
            f_k=f_k,
            g_k=g_k,
            log_d=log_d,
            eps=eps,
            C=C,
            log_b_k=log_b_k,
            w_k=w_k,
        )

    return bk.exp(log_bar)
