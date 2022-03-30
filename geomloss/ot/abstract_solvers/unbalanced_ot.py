r"""Provides support for several variations of unbalanced optimal transport.

The main reference for this file is:
Sinkhorn Divergences for Unbalanced Optimal Transport (2019)
Thibault Séjourné, Jean Feydy, François-Xavier Vialard, Alain Trouvé, Gabriel Peyré.
https://arxiv.org/pdf/1910.12958.pdf
"""



def dampening(eps, rho):
    """Dampening factor for entropy+unbalanced OT with KL penalization of the marginals."""
    return 1 if rho is None else 1 / (1 + eps / rho)




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
