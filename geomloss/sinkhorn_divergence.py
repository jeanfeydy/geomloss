r"""Implements the "raw" and "de-biased" Sinkhorn divergences between abstract measures.

.. math::
    \text{S}_{\varepsilon,\rho}(\alpha,\beta) 
        ~&=~ \text{OT}_{\varepsilon,\rho}(\alpha, \beta)
         ~-~\tfrac{1}{2} \text{OT}_{\varepsilon,\rho}(\alpha, \alpha)
         ~-~\tfrac{1}{2} \text{OT}_{\varepsilon,\rho}(\beta, \beta)
         ~+~ \tfrac{\varepsilon}{2} \| \langle \alpha, 1\rangle - \langle \beta, 1\rangle \|^2

where:

.. math::
    \text{OT}_{\varepsilon,\rho}(\alpha, \beta)
    ~&=~ \min_{\pi\geqslant 0} \langle\, \pi\,,\, \text{C} \,\rangle
        ~+~\varepsilon \, \text{KL}(\pi,\alpha\otimes\beta) \\
        ~&+~\rho \, \text{KL}(\pi\,\mathbf{1},\alpha)
        ~+~\rho \, \text{KL}(\pi^\intercal \,\mathbf{1},\beta ) \\
    &=~ \max_{f,g} -\rho \langle\, \alpha \,,\, e^{-f/\rho} - 1\,\rangle
        -\rho \langle\, \beta \,,\, e^{-g/\rho} - 1\,\rangle \\
        &-~
        \epsilon \langle\, \alpha\otimes\beta \,,\, e^{(f\oplus g - \text{C})/\epsilon} - 1\,\rangle,

with a Kullback-Leibler divergence defined through:

.. math::
    \text{KL}(\alpha, \beta)~=~
    \langle \, \alpha  \,,\, \log \tfrac{\text{d}\alpha}{\text{d}\beta} \,\rangle
    ~-~ \langle \, \alpha  \,,\, 1 \,\rangle
    ~+~ \langle \, \beta   \,,\, 1 \,\rangle ~\geqslant~ 0.
"""

import numpy as np
import torch
from functools import partial

try:  # Import the keops library, www.kernel-operations.io
    from pykeops.torch import generic_logsumexp
    from pykeops.torch.cluster import (
        grid_cluster,
        cluster_ranges_centroids,
        sort_clusters,
        from_matrix,
    )

    keops_available = True
except:
    keops_available = False

from .utils import scal


# ==============================================================================
#                             Utility functions
# ==============================================================================


def dampening(eps, rho):
    """Dampening factor for entropy+unbalanced OT with KL penalization of the marginals."""
    return 1 if rho is None else 1 / (1 + eps / rho)


def log_weights(a):
    """Returns the log of the input, with values clamped to -100k to avoid numerical bugs."""
    a_log = a.log()
    a_log[a <= 0] = -100000
    return a_log


class UnbalancedWeight(torch.nn.Module):
    """Applies the correct scaling to the dual variables in the Sinkhorn divergence formula.

    Remarkably, the exponentiated potentials should be scaled
    by "rho + eps/2" in the forward pass and "rho + eps" in the backward.
    For an explanation of this surprising "inconsistency"
    between the forward and backward formulas,
    please refer to Proposition 12 (Dual formulas for the Sinkhorn costs)
    in "Sinkhorn divergences for unbalanced optimal transport",
    Sejourne et al., https://arxiv.org/abs/1910.12958.
    """

    def __init__(self, eps, rho):
        super(UnbalancedWeight, self).__init__()
        self.eps, self.rho = eps, rho

    def forward(self, x):
        return (self.rho + self.eps / 2) * x

    def backward(self, g):
        return (self.rho + self.eps) * g


# ==============================================================================
#                            eps-scaling heuristic
# ==============================================================================


def max_diameter(x, y):
    """Returns a rough estimation of the diameter of a pair of point clouds.

    This quantity is used as a maximum "starting scale" in the epsilon-scaling
    annealing heuristic.

    Args:
        x ((N, D) Tensor): First point cloud.
        y ((M, D) Tensor): Second point cloud.

    Returns:
        float: Upper bound on the largest distance between points `x[i]` and `y[j]`.
    """
    mins = torch.stack((x.min(dim=0)[0], y.min(dim=0)[0])).min(dim=0)[0]
    maxs = torch.stack((x.max(dim=0)[0], y.max(dim=0)[0])).max(dim=0)[0]
    diameter = (maxs - mins).norm().item()
    return diameter


def epsilon_schedule(p, diameter, blur, scaling):
    r"""Creates a list of values for the temperature "epsilon" across Sinkhorn iterations.

    We use an aggressive strategy with an exponential cooling
    schedule: starting from a value of :math:`\text{diameter}^p`,
    the temperature epsilon is divided
    by :math:`\text{scaling}^p` at every iteration until reaching
    a minimum value of :math:`\text{blur}^p`.

    Args:
        p (integer or float): The exponent of the Euclidean distance
            :math:`\|x_i-y_j\|` that defines the cost function
            :math:`\text{C}(x_i,y_j) =\tfrac{1}{p} \|x_i-y_j\|^p`.

        diameter (float, positive): Upper bound on the largest distance between
            points :math:`x_i` and :math:`y_j`.

        blur (float, positive): Target value for the entropic regularization
            (":math:`\varepsilon = \text{blur}^p`").

        scaling (float, in (0,1)): Ratio between two successive
            values of the blur scale.

    Returns:
        list of float: list of values for the temperature epsilon.
    """
    eps_list = (
        [diameter**p]
        + [
            np.exp(e)
            for e in np.arange(
                p * np.log(diameter), p * np.log(blur), p * np.log(scaling)
            )
        ]
        + [blur**p]
    )
    return eps_list


def scaling_parameters(x, y, p, blur, reach, diameter, scaling):
    r"""Turns high-level arguments into numerical values for the Sinkhorn loop."""
    if diameter is None:
        D = x.shape[-1]
        diameter = max_diameter(x.view(-1, D), y.view(-1, D))

    eps = blur**p
    rho = None if reach is None else reach**p
    eps_list = epsilon_schedule(p, diameter, blur, scaling)
    return diameter, eps, eps_list, rho


# ==============================================================================
#                              Sinkhorn divergence
# ==============================================================================


def sinkhorn_cost(
    eps, rho, a, b, f_aa, g_bb, g_ab, f_ba, batch=False, debias=True, potentials=False
):
    r"""Returns the required information (cost, etc.) from a set of dual potentials.

    Args:
        eps (float): Target (i.e. final) temperature.
        rho (float or None (:math:`+\infty`)): Strength of the marginal constraints.

        a ((..., N) Tensor, nonnegative): Weights for the "source" measure on the points :math:`x_i`.
        b ((..., M) Tensor, nonnegative): Weights for the "target" measure on the points :math:`y_j`.
        f_aa ((..., N) Tensor)): Dual potential for the "a <-> a" problem.
        g_bb ((..., M) Tensor)): Dual potential for the "b <-> b" problem.
        g_ab ((..., M) Tensor)): Dual potential supported by :math:`y_j` for the "a <-> b" problem.
        f_ba ((..., N) Tensor)): Dual potential supported by :math:`x_i`  for the "a <-> a" problem.
        batch (bool, optional): Are we working in batch mode? Defaults to False.
        debias (bool, optional): Are we working with the "debiased" or the "raw" Sinkhorn divergence?
            Defaults to True.
        potentials (bool, optional): Shall we return the dual vectors instead of the cost value?
            Defaults to False.

    Returns:
        Tensor or pair of Tensors: if `potentials` is True, we return a pair
            of (..., N), (..., M) Tensors that encode the optimal dual vectors,
            respectively supported by :math:`x_i` and :math:`y_j`.
            Otherwise, we return a (,) or (B,) Tensor of values for the Sinkhorn divergence.
    """

    if potentials:  # Just return the dual potentials
        if debias:  # See Eq. (3.209) in Jean Feydy's PhD thesis.
            # N.B.: This formula does not make much sense in the unbalanced mode
            #       (i.e. if reach is not None).
            return f_ba - f_aa, g_ab - g_bb
        else:  # See Eq. (3.207) in Jean Feydy's PhD thesis.
            return f_ba, g_ab

    else:  # Actually compute the Sinkhorn divergence
        if (
            debias
        ):  # UNBIASED Sinkhorn divergence, S_eps(a,b) = OT_eps(a,b) - .5*OT_eps(a,a) - .5*OT_eps(b,b)
            if rho is None:  # Balanced case:
                # See Eq. (3.209) in Jean Feydy's PhD thesis.
                return scal(a, f_ba - f_aa, batch=batch) + scal(
                    b, g_ab - g_bb, batch=batch
                )
            else:
                # Unbalanced case:
                # See Proposition 12 (Dual formulas for the Sinkhorn costs)
                # in "Sinkhorn divergences for unbalanced optimal transport",
                # Sejourne et al., https://arxiv.org/abs/1910.12958.
                return scal(
                    a,
                    UnbalancedWeight(eps, rho)(
                        (-f_aa / rho).exp() - (-f_ba / rho).exp()
                    ),
                    batch=batch,
                ) + scal(
                    b,
                    UnbalancedWeight(eps, rho)(
                        (-g_bb / rho).exp() - (-g_ab / rho).exp()
                    ),
                    batch=batch,
                )

        else:  # Classic, BIASED entropized Optimal Transport OT_eps(a,b)
            if rho is None:  # Balanced case:
                # See Eq. (3.207) in Jean Feydy's PhD thesis.
                return scal(a, f_ba, batch=batch) + scal(b, g_ab, batch=batch)
            else:
                # Unbalanced case:
                # See Proposition 12 (Dual formulas for the Sinkhorn costs)
                # in "Sinkhorn divergences for unbalanced optimal transport",
                # Sejourne et al., https://arxiv.org/abs/1910.12958.
                # N.B.: Even if this quantity is never used in practice,
                #       we may want to re-check this computation...
                return scal(
                    a, UnbalancedWeight(eps, rho)(1 - (-f_ba / rho).exp()), batch=batch
                ) + scal(
                    b, UnbalancedWeight(eps, rho)(1 - (-g_ab / rho).exp()), batch=batch
                )


# ==============================================================================
#                              Sinkhorn loop
# ==============================================================================


def sinkhorn_loop(
    softmin,
    a_logs,
    b_logs,
    C_xxs,
    C_yys,
    C_xys,
    C_yxs,
    eps_list,
    rho,
    jumps=[],
    kernel_truncation=None,
    truncate=5,
    cost=None,
    extrapolate=None,
    debias=True,
    last_extrapolation=True,
):
    r"""Implements the (possibly multiscale) symmetric Sinkhorn loop,
    with the epsilon-scaling (annealing) heuristic.

    This is the main "core" routine of GeomLoss. It is written to
    solve optimal transport problems efficiently in all the settings
    that are supported by the library: (generalized) point clouds,
    images and volumes.

    This algorithm is described in Section 3.3.3 of Jean Feydy's PhD thesis,
    "Geometric data analysis, beyond convolutions" (Universite Paris-Saclay, 2020)
    (https://www.jeanfeydy.com/geometric_data_analysis.pdf).
    Algorithm 3.5 corresponds to the case where `kernel_truncation` is None,
    while Algorithm 3.6 describes the full multiscale algorithm.

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

        a_logs (list of Tensors): List of log-weights :math:`\log(\alpha_i)`
            for the first input measure at different resolutions.

        b_logs (list of Tensors): List of log-weights :math:`\log(\beta_i)`
            for the second input measure at different resolutions.

        C_xxs (list): List of objects that encode the cost matrices
            :math:`C(x_i, x_j)` between the samples of the first input
            measure at different scales.
            These will be passed to the `softmin` function as second arguments.

        C_yys (list): List of objects that encode the cost matrices
            :math:`C(y_i, y_j)` between the samples of the second input
            measure at different scales.
            These will be passed to the `softmin` function as second arguments.

        C_xys (list): List of objects that encode the cost matrices
            :math:`C(x_i, y_j)` between the samples of the first and second input
            measures at different scales.
            These will be passed to the `softmin` function as second arguments.

        C_yxs (list): List of objects that encode the cost matrices
            :math:`C(y_i, x_j)` between the samples of the second and first input
            measures at different scales.
            These will be passed to the `softmin` function as second arguments.

        eps_list (list of float): List of successive values for the temperature
            :math:`\varepsilon`. The number of iterations in the loop
            is equal to the length of this list.

        rho (float or None): Strength of the marginal constraints for unbalanced OT.
            None stands for :math:`\rho = +\infty`, i.e. balanced OT.

        jumps (list, optional): List of iteration numbers where we "jump"
            from a coarse resolution to a finer one by looking
            one step further in the lists `a_logs`, `b_logs`, `C_xxs`, etc.
            Count starts at iteration 0.
            Defaults to [] - single-scale mode without jumps.

        kernel_truncation (function, optional): Implements the kernel truncation trick.
            Defaults to None.

        truncate (int, optional): Optional argument for `kernel_truncation`.
            Defaults to 5.

        cost (string or function, optional): Optional argument for `kernel_truncation`.
            Defaults to None.

        extrapolate (function, optional): Function.
            If
            `f_ba` is a dual potential that is supported by the :math:`x_i`'s,
            `g_ab` is a dual potential that is supported by the :math:`y_j`'s,
            `eps` is the current value of the temperature :math:`\varepsilon`,
            `damping` is the current value of the damping coefficient for unbalanced OT,
            `C_xy` encodes the cost matrix :math:`C(x_i, y_j)` at the current
            ("coarse") resolution,
            `b_log` denotes the log-weights :math:`\log(\beta_j)`
            that are supported by the :math:`y_j`'s at the coarse resolution,
            and
            `C_xy_fine` encodes the cost matrix :math:`C(x_i, y_j)` at the next
            ("fine") resolution,
            then
            `extrapolate(f_ba, g_ab, eps, damping, C_xy, b_log, C_xy_fine)`
            will be used to compute the new values of the dual potential
            `f_ba` on the point cloud :math:`x_i` at a finer resolution.
            Defaults to None - it is not needed in single-scale mode.

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

        last_extrapolation (bool, optional): Should we perform a last, "full"
            Sinkhorn iteration before returning the dual potentials?
            This allows us to retrieve correct gradients without having
            to backpropagate trough the full Sinkhorn loop.
            Defaults to True.

    Returns:
        4-uple of Tensors: The four optimal dual potentials
            `(f_aa, g_bb, g_ab, f_ba)` that are respectively
            supported by the first, second, second and first input measures
            and associated to the "a <-> a", "b <-> b",
            "a <-> b" and "a <-> b" optimal transport problems.
    """

    # Number of iterations, specified by our epsilon-schedule
    Nits = len(eps_list)

    # The multiscale algorithm may loop over several representations
    # of the input measures.
    # In this routine, the convention is that "myvars" denotes
    # the list of "myvar" across different scales.
    if type(a_logs) is not list:
        # The "single-scale" use case is simply encoded
        # using lists of length 1.

        # Logarithms of the weights:
        a_logs, b_logs = [a_logs], [b_logs]

        # Cost "matrices" C(x_i, y_j) and C(y_i, x_j):
        C_xys, C_yxs = [C_xys], [C_yxs]  # Used for the "a <-> b" problem.

        # Cost "matrices" C(x_i, x_j) and C(y_i, y_j):
        if debias:  # Only used for the "a <-> a" and "b <-> b" problems.
            C_xxs, C_yys = [C_xxs], [C_yys]

    # N.B.: We don't let users backprop through the Sinkhorn iterations
    #       and branch instead on an explicit formula "at convergence"
    #       using some "advanced" PyTorch syntax at the end of the loop.
    #       This acceleration "trick" relies on the "envelope theorem":
    #       it works very well if users are only interested in the gradient
    #       of the Sinkhorn loss, but may not produce correct results
    #       if one attempts to compute order-2 derivatives,
    #       or differentiate "non-standard" quantities that
    #       are defined using the optimal dual potentials.
    #
    #       We may wish to alter this behaviour in the future.
    #       For reference on the question, see Eq. (3.226-227) in
    #       Jean Feydy's PhD thesis and e.g.
    #       "Super-efficiency of automatic differentiation for
    #       functions defined as a minimum", Ablin, Peyré, Moreau (2020)
    #       https://arxiv.org/pdf/2002.03722.pdf.
    torch.autograd.set_grad_enabled(False)

    # Line 1 (in Algorithm 3.6 from Jean Feydy's PhD thesis) ---------------------------

    # We start at the coarsest resolution available:
    k = 0  # Scale index
    eps = eps_list[k]  # First value of the temperature (typically, = diameter**p)

    # Damping factor: equal to 1 for balanced OT,
    # < 1 for unbalanced OT with KL penalty on the marginal constraints.
    # For reference, see Table 1 in "Sinkhorn divergences for unbalanced
    # optimal transport", Sejourne et al., https://arxiv.org/abs/1910.12958.
    damping = dampening(eps, rho)

    # Load the measures and cost matrices at the current scale:
    a_log, b_log = a_logs[k], b_logs[k]
    C_xy, C_yx = C_xys[k], C_yxs[k]  # C(x_i, y_j), C(y_i, x_j)
    if debias:  # Info for the "a <-> a" and "b <-> b" problems
        C_xx, C_yy = C_xxs[k], C_yys[k]  # C(x_i, x_j), C(y_j, y_j)

    # Line 2 ---------------------------------------------------------------------------
    # Start with a decent initialization for the dual vectors:
    # N.B.: eps is really large here, so the log-sum-exp behaves as a sum
    #       and the softmin is basically
    #       a convolution with the cost function (i.e. the limit for eps=+infty).
    #       The algorithm was originally written with this convolution
    #       - but in this implementation, we use "softmin" for the sake of simplicity.
    g_ab = damping * softmin(eps, C_yx, a_log)  # a -> b
    f_ba = damping * softmin(eps, C_xy, b_log)  # b -> a
    if debias:
        f_aa = damping * softmin(eps, C_xx, a_log)  # a -> a
        g_bb = damping * softmin(eps, C_yy, b_log)  # a -> a

    # Lines 4-5: eps-scaling descent ---------------------------------------------------
    for i, eps in enumerate(eps_list):  # See Fig. 3.25-26 in Jean Feydy's PhD thesis.
        # Line 6: update the damping coefficient ---------------------------------------
        damping = dampening(eps, rho)  # eps and damping change across iterations

        # Line 7: "coordinate ascent" on the dual problems -----------------------------
        # N.B.: As discussed in Section 3.3.3 of Jean Feydy's PhD thesis,
        #       we perform "symmetric" instead of "alternate" updates
        #       of the dual potentials "f" and "g".
        #       To this end, we first create buffers "ft", "gt"
        #       (for "f-tilde", "g-tilde") using the standard
        #       Sinkhorn formulas, and update both dual vectors
        #       simultaneously.
        ft_ba = damping * softmin(eps, C_xy, b_log + g_ab / eps)  # b -> a
        gt_ab = damping * softmin(eps, C_yx, a_log + f_ba / eps)  # a -> b

        # See Fig. 3.21 in Jean Feydy's PhD thesis to see the importance
        # of debiasing when the target "blur" or "eps**(1/p)" value is larger
        # than the average distance between samples x_i, y_j and their neighbours.
        if debias:
            ft_aa = damping * softmin(eps, C_xx, a_log + f_aa / eps)  # a -> a
            gt_bb = damping * softmin(eps, C_yy, b_log + g_bb / eps)  # b -> b

        # Symmetrized updates - see Fig. 3.24.b in Jean Feydy's PhD thesis:
        f_ba, g_ab = 0.5 * (f_ba + ft_ba), 0.5 * (g_ab + gt_ab)  # OT(a,b) wrt. a, b
        if debias:
            f_aa, g_bb = 0.5 * (f_aa + ft_aa), 0.5 * (g_bb + gt_bb)  # OT(a,a), OT(b,b)

        # Line 8: jump from a coarse to a finer scale ----------------------------------
        # In multi-scale mode, we work we increasingly detailed representations
        # of the input measures: this type of strategy is known as "multi-scale"
        # in computer graphics, "multi-grid" in numerical analysis,
        # "coarse-to-fine" in signal processing or "divide and conquer"
        # in standard complexity theory (e.g. for the quick-sort algorithm).
        #
        # In the Sinkhorn loop with epsilon-scaling annealing, our
        # representations of the input measures are fine enough to ensure
        # that the typical distance between any two samples x_i, y_j is always smaller
        # than the current value of "blur = eps**(1/p)".
        # As illustrated in Fig. 3.26 of Jean Feydy's PhD thesis, this allows us
        # to reach a satisfying level of precision while speeding up the computation
        # of the Sinkhorn iterations in the first few steps.
        #
        # In practice, different multi-scale representations of the input measures
        # are generated by the "parent" code of this solver and stored in the
        # lists a_logs, b_logs, C_xxs, etc.
        #
        # The switch between different scales is specified by the list of "jump" indices,
        # that is generated in conjunction with the list of temperatures "eps_list".
        #
        # N.B.: In single-scale mode, jumps = []: the code below is never executed
        #       and we retrieve "Algorithm 3.5" from Jean Feydy's PhD thesis.
        if i in jumps:
            if i == len(eps_list) - 1:  # Last iteration: just extrapolate!
                C_xy_fine, C_yx_fine = C_xys[k + 1], C_yxs[k + 1]
                if debias:
                    C_xx_fine, C_yy_fine = C_xxs[k + 1], C_yys[k + 1]

                last_extrapolation = False  # No need to re-extrapolate after the loop
                torch.autograd.set_grad_enabled(True)

            else:  # It's worth investing some time on kernel truncation...
                # The lines below implement the Kernel truncation trick,
                # described in Eq. (3.222-3.224) in Jean Feydy's PhD thesis and in
                # "Stabilized sparse scaling algorithms for entropy regularized transport
                #  problems", Schmitzer (2016-2019), (https://arxiv.org/pdf/1610.06519.pdf).
                #
                # A more principled and "controlled" variant is also described in
                # "Capacity constrained entropic optimal transport, Sinkhorn saturated
                #  domain out-summation and vanishing temperature", Benamou and Martinet
                #  (2020), (https://hal.archives-ouvertes.fr/hal-02563022/).
                #
                # On point clouds, this code relies on KeOps' block-sparse routines.
                # On grids, it is a "dummy" call: we do not perform any "truncation"
                # and rely instead on the separability of the Gaussian convolution kernel.

                # Line 9: a <-> b ------------------------------------------------------
                C_xy_fine, C_yx_fine = kernel_truncation(
                    C_xy,
                    C_yx,
                    C_xys[k + 1],
                    C_yxs[k + 1],
                    f_ba,
                    g_ab,
                    eps,
                    truncate=truncate,
                    cost=cost,
                )

                if debias:
                    # Line 10: a <-> a  ------------------------------------------------
                    C_xx_fine, _ = kernel_truncation(
                        C_xx,
                        C_xx,
                        C_xxs[k + 1],
                        C_xxs[k + 1],
                        f_aa,
                        f_aa,
                        eps,
                        truncate=truncate,
                        cost=cost,
                    )
                    # Line 11: b <-> b -------------------------------------------------
                    C_yy_fine, _ = kernel_truncation(
                        C_yy,
                        C_yy,
                        C_yys[k + 1],
                        C_yys[k + 1],
                        g_bb,
                        g_bb,
                        eps,
                        truncate=truncate,
                        cost=cost,
                    )

            # Line 12: extrapolation step ----------------------------------------------
            # We extra/inter-polate the values of the dual potentials from
            # the "coarse" to the "fine" resolution.
            #
            # On point clouds, we use the expressions of the dual potentials
            # detailed e.g. in Eqs. (3.194-3.195) of Jean Feydy's PhD thesis.
            # On images and volumes, we simply rely on (bi/tri-)linear interpolation.
            #
            # N.B.: the cross-updates below *must* be done in parallel!
            f_ba, g_ab = (
                extrapolate(f_ba, g_ab, eps, damping, C_xy, b_log, C_xy_fine),
                extrapolate(g_ab, f_ba, eps, damping, C_yx, a_log, C_yx_fine),
            )

            # Extrapolation for the symmetric problems:
            if debias:
                f_aa = extrapolate(f_aa, f_aa, eps, damping, C_xx, a_log, C_xx_fine)
                g_bb = extrapolate(g_bb, g_bb, eps, damping, C_yy, b_log, C_yy_fine)

            # Line 13: update the measure weights and cost "matrices" ------------------
            k = k + 1
            a_log, b_log = a_logs[k], b_logs[k]
            C_xy, C_yx = C_xy_fine, C_yx_fine
            if debias:
                C_xx, C_yy = C_xx_fine, C_yy_fine

    # As a very last step, we perform a final "Sinkhorn" iteration.
    # As detailed above (around "torch.autograd.set_grad_enabled(False)"),
    # this allows us to retrieve correct expressions for the gradient
    # without having to backprop through the whole Sinkhorn loop.
    torch.autograd.set_grad_enabled(True)

    if last_extrapolation:
        # The cross-updates should be done in parallel!
        f_ba, g_ab = (
            damping * softmin(eps, C_xy, (b_log + g_ab / eps).detach()),
            damping * softmin(eps, C_yx, (a_log + f_ba / eps).detach()),
        )

        if debias:
            f_aa = damping * softmin(eps, C_xx, (a_log + f_aa / eps).detach())
            g_bb = damping * softmin(eps, C_yy, (b_log + g_bb / eps).detach())

    if debias:
        return f_aa, g_bb, g_ab, f_ba
    else:
        return None, None, g_ab, f_ba
