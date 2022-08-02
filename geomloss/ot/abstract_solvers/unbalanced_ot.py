r"""Provides support for several variations of unbalanced optimal transport.

The main reference for this file is:
Sinkhorn Divergences for Unbalanced Optimal Transport (2019)
Thibault Séjourné, Jean Feydy, François-Xavier Vialard, Alain Trouvé, Gabriel Peyré.
https://arxiv.org/pdf/1910.12958.pdf
"""

from ..utils import dot_products
from ..typing import Optional, RealTensor, SinkhornPotentials


def dampening(*, eps: float, rho: Optional[float]):
    """Dampening function for entropy+unbalanced OT with KL penalization of the marginals."""
    if rho is None:
        return lambda f: f
    else:
        return lambda f: f / (1 + eps / rho)


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

    def __init__(self, *, eps: float, rho: float):
        super(UnbalancedWeight, self).__init__()
        self.eps, self.rho = eps, rho

    def forward(self, x: Tensor):
        return (self.rho + self.eps / 2) * x

    def backward(self, g: Tensor):
        return (self.rho + self.eps) * g


def sinkhorn_cost(
    *,
    a: Tensor,
    b: Tensor,
    potentials: SinkhornPotentials,
    eps: float,
    rho: Optional[float],
    debias: bool = True,
):
    r"""Returns the values of the Sinkhorn divergence from a set of dual potentials.

    For reference, please look at Eqs. (3.207-3.209) in Jean Feydy's PhD thesis
    (balanced case) and Proposition 12 in "Sinkhorn divergences for unbalanced
    optimal transport" by Sejourne et al.

    This function is batched: it computes B values in parallel.
    If you only want to compute a single value, please prefix a "dummy"
    dimension (=1) in front of your measure weights and dual potentials.

    Args:
        a ((B,...) real-valued Tensor >= 0): Weights for the "source" measure on the points :math:`x_i`.
        b ((B,...) real-valued Tensor >= 0): Weights for the "target" measure on the points :math:`y_j`.
        potentials (SinkhornPotentials): A NamedTuple that contains the solutions
           of a (dual) regularized optimal transport problem.
           We expect the attributes:
            - f_aa ((B,...) real-valued Tensor or None): Dual potential for the "a <-> a" problem.
            - g_bb ((B,...) real-valued Tensor or None): Dual potential for the "b <-> b" problem.
            - g_ab ((B,...) real-valued Tensor)): Dual potential supported by :math:`y_j`
              for the "a <-> b" problem.
            - f_ba ((B,...) real-valued Tensor): Dual potential supported by :math:`x_i`
              for the "a <-> a" problem.

        eps (float): Target (i.e. final) temperature.
        rho (float or None (:math:`+\infty`)): Strength of the marginal constraints.
        debias (bool, optional): Do we compute the "debiased" Sinkhorn divergence
            instead of the "raw" entropic transport cost?
            Defaults to True.

    Returns:
        (B,) real-valued Tensor:
    """

    # For the sake of convenience, unwrap the dual potentials returned by the
    # Sinkhorn loop:
    f_aa = potentials.f_aa  # may be None with balanced OT
    g_bb = potentials.g_bb  # may be None with balanced OT
    g_ab = potentials.g_ab
    f_ba = potentials.f_ba

    if rho is None:
        # Balanced case: the cost is linear wrt. the potentials.
        if not debias:
            # Classic, BIASED entropic Optimal Transport OT_eps(a,b)
            # See Eq. (3.207) in Jean Feydy's PhD thesis.
            F_a = f_ba
            G_b = g_ab
        else:
            # DEBIASED Sinkhorn divergence, S_eps(a,b) = OT_eps(a,b) - .5*OT_eps(a,a) - .5*OT_eps(b,b)
            # See Eq. (3.209) in Jean Feydy's PhD thesis.
            F_a = f_ba - f_aa
            G_b = g_ab - g_bb

    else:
        # Unbalanced case: we must dampen the dual potentials with a function
        # ~ rho * (1 - exp(-f/rho))  (but not exactly, hence the UnbalancedWeight...)
        # that is ~ f when |f| << rho, and that kills the high values of f.

        # First: compute a dampened version of the update above.
        if not debias:
            # Classic, BIASED entropic Optimal Transport OT_eps,rho(a,b),
            # as in the ~2016 papers by Lenaic Chizat, Bernhard Schmitzer...
            #
            # See Proposition 12 (Dual formulas for the Sinkhorn costs)
            # in "Sinkhorn divergences for unbalanced optimal transport",
            # Sejourne et al., https://arxiv.org/abs/1910.12958.

            # N.B.: Even if this quantity is never used in practice,
            #       we may want to re-check this computation...
            F_a = 1 - (-f_ba / rho).exp()
            G_b = 1 - (-g_ab / rho).exp()

        else:
            # DEBIASED Sinkhorn divergence, S_eps,rho(a,b)
            # = OT_eps,rho(a,b) - .5*OT_eps,rho(a,a) - .5*OT_eps,rho(b,b)
            #   + e/2 * | mass(a) - mass(b) |^2
            # studied extensively in the PhD thesis of Thibault Sejourne.
            #
            # See Proposition 12 (Dual formulas for the Sinkhorn costs)
            # in "Sinkhorn divergences for unbalanced optimal transport",
            # Sejourne et al., https://arxiv.org/abs/1910.12958.
            F_a = (-f_aa / rho).exp() - (-f_ba / rho).exp()
            G_b = (-g_bb / rho).exp() - (-g_ab / rho).exp()

        # Second: weight it by the correct factor,
        # in a way that is coherent for the backward pass.
        # TODO: make this compatible with order 2 derivatives.
        F_a = UnbalancedWeight(eps=eps, rho=rho)(F_a)
        G_b = UnbalancedWeight(eps=eps, rho=rho)(G_b)

    a_costs = dot_products(a, F_a)  # (B,)
    b_costs = dot_products(b, G_b)  # (B,)
    return a_costs + b_costs


"""
if potentials:  # Just return the dual potentials
    if debias:  # See Eq. (3.209) in Jean Feydy's PhD thesis.
        # N.B.: This formula does not make much sense in the unbalanced mode
        #       (i.e. if reach is not None).
        return f_ba - f_aa, g_ab - g_bb
    else:  # See Eq. (3.207) in Jean Feydy's PhD thesis.
        return f_ba, g_ab
"""
