r"""Implements the "biased" and "de-biased" Sinkhorn divergences between positive measures.
    
    S_e,r(A, B)  (debiased)

        =        OT_e,r(A, B)
         - 1/2 * OT_e,r(A, B)
         - 1/2 * OT_e,r(A, B)
         + e/2 * | mass(A) - mass(B) |^2

where:

    OT_e,r(A, B)  (biased)

        (Primal problem:)
        = min_{ transport plan P >= 0 } 
              Sum(P * C) 
        + e * KL(P, A @ B.T)
        + r * KL(P @ 1, A)
        + r * KL(P.T @ 1, B)

        (Dual problem:)
        = max_{dual vectors F, G} 
        - r * Sum(A * (exp(-F/r) - 1))
        - r * Sum(B * (exp(-G/r) - 1))
        - e * Sum((A @ B.T) * (exp((F + G.T - C)/e) - 1))

where the Kullback-Leibler divergence between two non-negative measures is defined through:

    KL(A, B) = Sum(A * log(A/B)) - Sum(A) + Sum(B) >= 0.

The main reference for this file is Chapter 3.3 in Jean Feydy's PhD thesis:
Geometric data analysis, beyond convolutions (2020), 
https://www.jeanfeydy.com/geometric_data_analysis.pdf
"""

from ..typing import (
    Tensor,
    Optional,
    AnnealingParameters,
    SoftMin,
    CostMatrix,
    CostFunction,
    Extrapolator,
    KernelTruncation,
)
from .unbalanced_ot import dampening


def log_weights(a: Tensor) -> Tensor:
    """Returns the log of the input, with values clamped to -100k to avoid numerical bugs."""
    a_log = a.log()
    a_log[a <= 0] = -100000
    return a_log


def sinkhorn_loop(
    *,
    softmin: SoftMin,
    a_logs: List[Tensor],
    b_logs: List[Tensor],
    C_xxs: List[CostMatrix],
    C_yys: List[CostMatrix],
    C_xys: List[CostMatrix],
    C_yxs: List[CostMatrix],
    eps_list: List[float],
    rho: Optional[float] = None,
    jumps: List[int] = [],
    kernel_truncation: Optional[KernelTruncation] = None,
    truncate: Optional[float] = 5.0,
    cost: Optional[CostFunction] = None,
    extrapolate: Optional[Extrapolator] = None,
    debias: bool = True,
    last_extrapolation: bool = True,
) -> tuple[Optional[Tensor], Optional[Tensor], Tensor, Tensor]:
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

        truncate (float, optional): Optional argument for `kernel_truncation`.
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
    # In this routine, the convention is that "myvars" (with an 's')
    # denotes the list of "myvar" across different scales.
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
    eps = eps_list[k]  # First value of the temperature (typically, eps = diameter**p)

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
