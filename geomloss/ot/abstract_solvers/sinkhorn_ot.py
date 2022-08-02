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

import torch
from ..typing import (
    List,
    RealTensor,
    Optional,
    SinkhornPotentials,
    SoftMin,
    CostMatrices,
    Extrapolator,
    KernelTruncation,
    DescentParameters,
)
from .unbalanced_ot import dampening


def sinkhorn_loop(
    *,
    softmin: SoftMin,
    log_a_list: List[RealTensor],
    log_b_list: List[RealTensor],
    C_list: List[CostMatrices],
    descent: DescentParameters,
    kernel_truncation: Optional[KernelTruncation] = None,
    extrapolate: Optional[Extrapolator] = None,
    debias: bool = True,
    last_extrapolation: bool = True,
) -> SinkhornPotentials:
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

    In the description below, we assume that S >= 1 is the number of scales.

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

        log_a_list (list of S real-valued Tensors): List of log-weights
            :math:`\log(\alpha_i)` for the source measure, at different resolutions.

        log_b_list (list of S real-valued Tensors): List of log-weights
            :math:`\log(\beta_i)` for the target measure, at different resolutions.

        C_list (list of S CostMatrices): List of NamedTuples with attributes
            that represent the cost matrix at different scales.
            These will be passed to the `softmin` function as second arguments.
            We expect the attributes:
            - `xx` for the cost matrix :math:`C_{xx}[i,j] = C(x[i], x[j])`.
            - `yy` for the cost matrix :math:`C_{yy}[i,j] = C(y[i], y[j])`.
            - `xy` for the cost matrix :math:`C_{xy}[i,j] = C(x[i], y[j])`.
            - `yx` for the cost matrix :math:`C_{yx}[i,j] = C(y[i], x[j])`.

        descent (DescentParameters): A NamedTuple with attributes that describe
            the evolution of our main parameters along the iterations of the
            Sinkhorn loop.
            We expect the attributes:
            - eps_list (list of n_iter float > 0): List of successive values for
              the Sinkhorn regularization parameter, the temperature :math:`\varepsilon`.
              The number of iterations in the loop is equal to the length of this list.

            - rho_list (list of n_iter (float > 0 or None)): List of successive values for
              the strength of the marginal constraints in unbalanced OT.
              None values stand for :math:`\rho = +\infty`, i.e. balanced OT.

            - jumps (list of S-1 int): Sorted list of iteration numbers where we "jump"
              from a coarse resolution to a finer one by looking one step further
              in the lists `log_a_list`, `log_b_list` and `C_list`.
              Each integer jump index `jump` should satisfy `0 <= jump < n_iter`.
              For single-scale mode, use `jumps = []`.

        kernel_truncation (function, optional): Implements the kernel truncation trick.
            Defaults to None: this function is not needed in single-scale mode.

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

        debias (bool, optional): Should we used the "de-biased" Sinkhorn divergence
            :math:`\text{S}_{\varepsilon, \rho}(\al,\be)` instead
            of the "raw" entropic OT cost
            :math:`\text{OT}_{\varepsilon, \rho}(\al,\be)`?
            This slows down the OT solver by a factor 2, but guarantees that
            our approximation of the Wasserstein distance will be positive and definite
            - up to convergence of the Sinkhorn loop.
            For a detailed discussion of the influence of this parameter,
            see e.g. Fig. 3.21 in Jean Feydy's PhD thesis.
            Defaults to True.

        last_extrapolation (bool, optional): Should we perform a last,
            "full" Sinkhorn iteration before returning the dual potentials?
            This allows us to retrieve correct gradients without having
            to backpropagate trough the full Sinkhorn loop.
            Defaults to True.

    Returns:
        Named 4-uple of Tensors: The four optimal dual potentials
            `(f_aa, g_bb, g_ab, f_ba)` that are respectively
            supported by the first, second, second and first input measures
            and associated to the "a <-> a", "b <-> b",
            "a <-> b" and "a <-> b" optimal transport problems.
    """

    # The multiscale algorithm may loop over several representations
    # of the input measures.
    # In this routine, the convention is that "myvar_s" (with a '_s' suffix)
    # denotes the list of "myvar" across different scales.

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
    prev_autograd = torch.is_grad_enabled()
    torch.autograd.set_grad_enabled(False)

    # Line 1 (in Algorithm 3.6 from Jean Feydy's PhD thesis) ---------------------------

    # We start at the coarsest resolution available:
    k = 0  # Scale index

    # First value of the temperature (typically, eps = diameter**p)
    # and of the strength of the marginal constraints (typically, rho = reach**p).
    eps = descent.eps_list[0]
    rho = descent.rho_list[0]

    # Damping factor: contractant function on the dual potentials.
    # Equal to the identity for balanced OT, and to a scaling by a constant < 1 
    # for unbalanced OT with KL penalty on the marginal constraints.
    # For reference, see Table 1 in "Sinkhorn divergences for unbalanced
    # optimal transport", Sejourne et al., https://arxiv.org/abs/1910.12958.
    dampen = dampening(eps=eps, rho=rho)

    # Load the masses (more precisely, the logarithms of the point weights/densities)
    # and cost matrices (C(x[i], y[j]), ...) at the current scale:
    log_a, log_b, C = log_a_list[k], log_b_list[k], C_list[k]

    # Line 2 ---------------------------------------------------------------------------
    # Start with a decent initialization for the dual vectors:
    # N.B.: eps is really large here, so the log-sum-exp behaves as a sum
    #       and the softmin is basically
    #       a convolution with the cost function (i.e. the limit for eps=+infty).
    #       The algorithm was originally written with this convolution
    #       - but in this implementation, we use "softmin" for the sake of simplicity.
    g_ab = dampen(softmin(eps, C.yx, log_a))  # a -> b
    f_ba = dampen(softmin(eps, C.xy, log_b))  # b -> a
    if debias:
        f_aa = dampen(softmin(eps, C.xx, log_a))  # a -> a
        g_bb = dampen(softmin(eps, C.yy, log_b))  # a -> a

    # Lines 4-5: eps-scaling descent ---------------------------------------------------
    # See Fig. 3.25-26 in Jean Feydy's PhD thesis for intuitions.
    for i, (eps, rho) in enumerate(zip(descent.eps_list, descent.rho_list)):

        # Line 6: update the damping coefficient ---------------------------------------
        dampen = dampening(eps=eps, rho=rho)  # eps and damping change across iterations

        # Line 7: "coordinate ascent" on the dual problems -----------------------------
        # N.B.: As discussed in Section 3.3.3 of Jean Feydy's PhD thesis,
        #       we perform "symmetric" instead of "alternate" updates
        #       of the dual potentials "f" and "g".
        #       To this end, we first create buffers "ft", "gt"
        #       (for "f-tilde", "g-tilde") using the standard
        #       Sinkhorn formulas, and update both dual vectors
        #       simultaneously.
        ft_ba = dampen(softmin(eps, C.xy, log_b + g_ab / eps))  # b -> a
        gt_ab = dampen(softmin(eps, C.yx, log_a + f_ba / eps))  # a -> b

        # See Fig. 3.21 in Jean Feydy's PhD thesis to see the importance
        # of debiasing when the target "blur" or "eps**(1/p)" value is larger
        # than the average distance between samples x_i, y_j and their neighbours.
        if debias:
            ft_aa = dampen(softmin(eps, C.xx, log_a + f_aa / eps))  # a -> a
            gt_bb = dampen(softmin(eps, C.yy, log_b + g_bb / eps))  # b -> b

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
        if i in descent.jumps:

            if i == len(descent.eps_list) - 1:  # Last iteration: just extrapolate!
                C_fine = C_list[k + 1]
                last_extrapolation = False  # No need to re-extrapolate after the loop
                torch.autograd.set_grad_enabled(prev_autograd)

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
                C_fine_xy, C_fine_yx = kernel_truncation(
                    C=C.xy,
                    CT=C.yx,
                    C_fine=C_list[k + 1].xy,
                    CT_fine=C_list[k + 1].yx,
                    f=f_ba,
                    g=g_ab,
                    eps=eps,
                )

                if debias:
                    # Line 10: a <-> a  ------------------------------------------------
                    C_fine_xx, _ = kernel_truncation(
                        C=C.xx,
                        C_fine=C_list[k + 1].xx,
                        f=f_aa,
                        eps=eps,
                    )
                    # Line 11: b <-> b -------------------------------------------------
                    C_fine_yy, _ = kernel_truncation(
                        C=C.yy,
                        C_fine=C_list[k + 1].yy,
                        f=g_bb,
                        eps=eps,
                    )
                else:
                    C_fine_xx, C_fine_yy = None, None

                # Update our cost object with the truncated matrices:
                C_fine = CostMatrices(
                    xx=C_fine_xx,
                    yy=C_fine_yy,
                    xy=C_fine_xy,
                    yx=C_fine_yx,
                )

            # Line 12: extrapolation step ----------------------------------------------
            # We extra/inter-polate the values of the dual potentials from
            # the "coarse" to the "fine" resolution.
            #
            # On point clouds, we use the expressions of the dual potentials
            # detailed e.g. in Eqs. (3.194-3.195) of Jean Feydy's PhD thesis.
            # On images and volumes, we simply rely on (bi/tri-)linear interpolation.
            #
            # N.B.: The cross-updates below *must* be done in parallel!
            #       Do *not* split this coupled update.
            f_ba, g_ab = (
                extrapolate(
                    self=f_ba,
                    other=g_ab,
                    log_weights=log_b,
                    C=C.xy,
                    C_fine=C_fine.xy,
                    eps=eps,
                    dampen=dampen,
                ),
                extrapolate(
                    self=g_ab,
                    other=f_ba,
                    lob_weights=log_a,
                    C=C.yx,
                    C_fine=C_fine.yx,
                    eps=eps,
                    dampen=dampen,
                ),
            )

            # Extrapolation for the symmetric problems:
            if debias:
                f_aa = extrapolate(
                    self=f_aa,
                    other=f_aa,
                    log_weights=log_a,
                    C=C.xx,
                    C_fine=C_fine.xx,
                    eps=eps,
                    dampen=dampen,
                )
                g_bb = extrapolate(
                    self=g_bb,
                    other=g_bb,
                    log_weights=log_b,
                    C=C.yy,
                    C_fine=C_fine.yy,
                    eps=eps,
                    dampen=dampen,
                )

            # Line 13: update the measure weights and cost "matrices" ------------------
            k = k + 1
            log_a, log_b = log_a_list[k], log_b_list[k]
            C = C_fine

    # As a very last step, we perform a final "Sinkhorn" iteration.
    # As detailed above (around "torch.autograd.set_grad_enabled(False)"),
    # this allows us to retrieve correct expressions for the gradient
    # without having to backprop through the whole Sinkhorn loop.
    torch.autograd.set_grad_enabled(prev_autograd)

    if last_extrapolation:
        # The cross-updates *must* be done in parallel!
        # Do *not* split this coupled update.
        f_ba, g_ab = (
            dampen(softmin(eps, C.xy, (log_b + g_ab / eps).detach())),
            dampen(softmin(eps, C.yx, (log_a + f_ba / eps).detach())),
        )

        if debias:
            f_aa = dampen(softmin(eps, C.xx, (log_a + f_aa / eps).detach()))
            g_bb = dampen(softmin(eps, C.yy, (log_b + g_bb / eps).detach()))

    # If there is no de-biasing, we should define empty "self-attention"
    # potentials.
    if not debias:
        f_aa, g_bb = None, None

    return SinkhornPotentials(
        f_aa=f_aa,
        g_bb=g_bb,
        g_ab=g_ab,
        f_ba=f_ba,
    )
