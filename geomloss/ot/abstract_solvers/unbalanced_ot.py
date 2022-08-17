r"""Provides support for several variations of unbalanced optimal transport.

The main reference for this file is:
Sinkhorn Divergences for Unbalanced Optimal Transport (2019)
Thibault Séjourné, Jean Feydy, François-Xavier Vialard, Alain Trouvé, Gabriel Peyré.
https://arxiv.org/pdf/1910.12958.pdf
"""

from ... import backends as bk
from ...typing import Optional, RealTensor, SinkhornPotentials


def dampening(*, eps: float, rho: Optional[float]):
    """Dampening function for entropy+unbalanced OT with KL penalization of the marginals."""
    if rho is None:
        return lambda f: f
    else:
        return lambda f: f / (1 + eps / rho)


def sinkhorn_cost(
    *,
    a: RealTensor,
    b: RealTensor,
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

    # Make sure that every Tensor has the expected shape:
    assert f_ba.shape == a.shape
    assert g_ab.shape == b.shape

    if f_aa is not None:
        assert f_aa.shape == a.shape
    if g_bb is not None:
        assert g_bb.shape == b.shape

    # Make sure that eps and rho are None or > 0:
    assert eps > 0
    assert rho is None or rho > 0

    # Actual formulas:
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
            F_a = -bk.exp(-f_ba / rho)
            G_b = -bk.exp(-g_ab / rho)

            # TODO: make this compatible with order 2 derivatives.
            # Total masses:
            m_a = bk.sum(a, axis=tuple(range(1, len(a.shape))), keepdims=True)
            m_b = bk.sum(b, axis=tuple(range(1, len(b.shape))), keepdims=True)

            # Constant terms that disappear in the debiased divergence:
            Cst_a = bk.scale(
                bk.ones_like(F_a),
                forward=rho + (eps / 2) * m_b,
                backward=rho + eps * m_b,
            )
            Cst_b = bk.scale(
                bk.ones_like(G_b),
                forward=rho + (eps / 2) * m_a,
                backward=rho + eps * m_a,
            )

            F_a = Cst_a + bk.scale(
                F_a,
                forward=rho + eps / 2,
                backward=rho + eps,
            )
            G_b = Cst_b + bk.scale(
                G_b,
                forward=rho + eps / 2,
                backward=rho + eps,
            )

        else:
            # DEBIASED Sinkhorn divergence, S_eps,rho(a,b)
            # = OT_eps,rho(a,b) - .5*OT_eps,rho(a,a) - .5*OT_eps,rho(b,b)
            #   + e/2 * | mass(a) - mass(b) |^2
            # studied extensively in the PhD thesis of Thibault Sejourne.
            #
            # See Proposition 12 (Dual formulas for the Sinkhorn costs)
            # in "Sinkhorn divergences for unbalanced optimal transport",
            # Sejourne et al., https://arxiv.org/abs/1910.12958.
            F_a = bk.exp(-f_aa / rho) - bk.exp(-f_ba / rho)
            G_b = bk.exp(-g_bb / rho) - bk.exp(-g_ab / rho)

            # Second: weight it by the correct factor,
            # in a way that is coherent for the backward pass.
            # TODO: make this compatible with order 2 derivatives.
            F_a = bk.scale(F_a, forward=rho + eps / 2, backward=rho + eps)
            G_b = bk.scale(G_b, forward=rho + eps / 2, backward=rho + eps)

    a_costs = bk.dot_products(a, F_a)  # (B,)
    b_costs = bk.dot_products(b, G_b)  # (B,)
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
