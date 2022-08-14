# Our generic backend, to use instead of NumPy/PyTorch/...
from ... import backends as bk

# Typing annotations:
from ...typing import RealTensor, CostMatrices

# Abstract class for our results:
from ..ot_result import OTResult

# Abstract solvers and annealing strategy:
from ..abstract_solvers import (
    sinkhorn_loop,
    # sinkhorn_barycenter_loop,
    annealing_parameters,
)

# Utility functions:
from ...arguments import (
    ArrayProperties,
    check_library,
    check_dtype,
    check_device,
    check_marginals,
)


# ========================================================================================
#                          High-level public interface
# ========================================================================================

# ----------------------------------------------------------------------------------------
#                                 Standard OT solver
# ----------------------------------------------------------------------------------------


def solve(
    C,  # (N, M) or (B, N, M)  (B is the batch dimension)
    *,
    a=None,  # (N,) or (B, N)
    b=None,  # (M,) or (B, M)
    # Regularization:
    reg=0,  # -> None by default
    reg_type="relative entropy",  # "entropy", "quadratic", etc.
    # Unbalanced OT:
    unbalanced=None,  # None = +infty -> balanced by default;
    # We also support scalar numbers, pairs of scalar numbers,
    # ((N,), (M,)) and ((B, N), (B, M)) point-dependent penalties.
    unbalanced_type="relative entropy",  # ="KL", "TV", etc.
    # Partial OT?
    debias=False,  # Use debiasing? This also requires C_aa, C_bb
    # Optim parameters, following SciPy convention:
    method="auto",  # We can match keywords in this
    # string to activate some options such as
    # "symmetric", "annealing", etc.
    # Tolerance values.
    maxiter=None,
    ftol=None,  # Changes of the objective?
    xtol=None,  # Changes of the variables?
    # We may just use "tol" and let the solver decide...
    # As far as I can tell, most users are interested in a
    # simple, one-dimensional parameter that lets them trade
    # time for accuracy. For instance, we could allow
    # users to specify an "rtol" ratio in (0, 1)
    # that would drive a sensible heuristic for the parameter choices.
    # The main thing to guarantee is a monotonic improvement
    # of the solution quality as rtol goes to 0.
    rtol=1e-2,  # ~1% relative accuracy
):
    # Basic checks on the parameters =====================================================
    if reg < 0:
        raise ValueError(f"Parameter 'reg' should be >= 0. Received {reg}.")
    elif reg == 0:
        raise NotImplementedError("Currently, we require that reg > 0.")

    if reg_type != "relative entropy":
        raise NotImplementedError("Currently, we only support a Sinkhorn solver.")

    if unbalanced is not None and unbalanced <= 0:
        raise ValueError(
            "Parameter 'unbalanced' should be None (= +infty) "
            f"or > 0. Received {unbalanced}."
        )

    if unbalanced_type != "relative entropy":
        raise NotImplementedError(
            "Currently, we only support unbalanced OT with "
            "a 'relative entropy' penalty on the marginal constraints."
        )

    if debias:
        raise NotImplementedError(
            "Currently, we do not support debiasing " "for the matrix-mode OT solver."
        )

    if method != "auto":
        raise NotImplementedError("Currently, we only support a single method.")

    if ftol is not None or xtol is not None:
        raise NotImplementedError(
            "Currently, we do not support rigorous stopping criteria."
        )

    # Check the parameters ===============================================================

    # Cost matrix ------------------------------------------------------------------------
    # Check the shape:
    if len(C.shape) == 2:
        C = C[None, :, :]  # Add an extra, dummy batch dimension
        B = 0  # No batch size
    elif len(C.shape) == 3:
        B = C.shape[0]
    else:
        raise ValueError(
            "The 'cost' matrix should be an array with 2 or 3 dimensions. "
            f"Instead, ot.solve received an array of shape {C.shape}."
        )

    # At this point, we know that C is a (max(1, B), N, M) array.
    N, M = C.shape[1], C.shape[2]

    # First marginal a -------------------------------------------------------------------
    if a is None:
        a = bk.ones_like(C[:, :, 0])  # (max(1, B), N) array
        a = a / N  # By default, the marginal sums up to 1

    else:
        if B == 0:  # No batch mode
            if len(a.shape) != 1:
                raise ValueError(
                    "Since 'cost' was given as a 2-dimensional array, "
                    "the first marginal 'a' should be a vector with 1 dimension. "
                    f"Instead, ot.solve received an array of shape {a.shape}."
                )

            if a.shape[0] != N:
                raise ValueError(
                    f"The dimension of 'cost' ({N},{M}) "
                    f"is not compatible with that of the first marginal 'a' {a.shape}. "
                    f"We expect a vector of shape ({N},)."
                )

            a = a[None, :]  # Add a dummy batch dimension

        else:  # Batch mode
            if len(a.shape) != 2:
                raise ValueError(
                    "Since 'cost' was given as a 3-dimensional array, "
                    "we work in batch mode an expect that "
                    "the first marginal 'a' is an array with 2 dimensions. "
                    f"Instead, ot.solve received an array of shape {a.shape}."
                )

            if a.shape[0] != B or a.shape[1] != N:
                raise ValueError(
                    f"The dimension of 'cost' ({B},{N},{M}) "
                    f"is not compatible with that of the first marginal 'a' {a.shape}. "
                    f"We expect an array of shape ({B},{N})."
                )

    # Check that all values are non-negative:
    if bk.any(a < 0):
        raise ValueError(
            "The first marginal 'a' contains negative values. "
            "ot.solve requires that a >= 0."
        )

    # Add this point, we know that a is a (max(1, B), N) array with >= 0 values.

    # Second marginal b ------------------------------------------------------------------
    if b is None:
        b = bk.ones_like(C[:, 0, :])  # (max(1, B), M) array
        b = b / M  # By default, the marginal sums up to 1

    else:
        if B == 0:  # No batch mode
            if len(b.shape) != 1:
                raise ValueError(
                    "Since 'cost' was given as a 2-dimensional array, "
                    "the second marginal 'b' should be a vector with 1 dimension. "
                    f"Instead, ot.solve received an array of shape {b.shape}."
                )

            if b.shape[0] != M:
                raise ValueError(
                    f"The dimension of 'cost' ({N},{M}) "
                    f"is not compatible with that of the second marginal 'b' {b.shape}. "
                    f"We expect a vector of shape ({M},)."
                )

            b = b[None, :]  # Add a dummy batch dimension

        else:  # Batch mode
            if len(b.shape) != 2:
                raise ValueError(
                    "Since 'cost' was given as a 3-dimensional array, "
                    "we work in batch mode an expect that "
                    "the second marginal 'b' is an array with 2 dimensions. "
                    f"Instead, ot.solve received an array of shape {b.shape}."
                )

            if b.shape[0] != B or b.shape[1] != M:
                raise ValueError(
                    f"The dimension of 'cost' ({B},{N},{M}) "
                    f"is not compatible with that of the second marginal 'b' {b.shape}. "
                    f"We expect an array of shape ({B},{M})."
                )

    # Check that all values are non-negative:
    if bk.any(b < 0):
        raise ValueError(
            "The second marginal 'b' contains negative values. "
            "ot.solve requires that b >= 0."
        )

    # Add this point, we know that b is a (max(1, B), M) array with >= 0 values.

    # Check that the marginals have the same total mass ----------------------------------
    if unbalanced is None:  # if we work in balanced mode
        sums_a = bk.sum(a, axis=1)  # (B,)
        sums_b = bk.sum(b, axis=1)  # (B,)
        check_marginals(sums_a, sums_b, rtol=1e-3)

    # Low-level compatibility ------------------------------------------------------------

    # Check that all the arrays come from the same library (numpy, torch...):
    library = check_library(a, b, C)
    # Check that every array has the same numerical precision:
    dtype = check_dtype(a, b, C)
    # Check that every array is on the same device:
    device = check_device(a, b, C)

    array_properties = ArrayProperties(
        B=B,
        N=N,
        M=M,
        dtype=dtype,
        device=device,
        library=library,
    )

    # Actual computations ================================================================
    descent = annealing_parameters(
        diameter=bk.amax(C),
        p=1,
        blur=reg,
        reach=unbalanced,
        n_iter=maxiter,
    )

    # N.B.: With a fixed cost matrix, there is no debiasing.
    potentials = sinkhorn_loop(
        softmin=softmin_dense,
        log_a_list=[bk.stable_log(a)],
        log_b_list=[bk.stable_log(b)],
        C_list=[
            CostMatrices(
                xy=C,
                yx=bk.ascontiguousarray(bk.transpose(C, (0, 2, 1))),
            )
        ],
        descent=descent,
        debias=False,
        last_extrapolation=True,
    )
    # solve exact OT by default
    # solve generic regularization ('enropic','l2','entropic+group lasso')
    # (reg_type can be a function)
    # default a and b are uniform (do they sum up to 1?)
    return OTResultMatrix(
        a=a,
        b=b,
        C=C,
        potentials=potentials,
        array_properties=array_properties,
        reg=reg,
        reg_type=reg_type,
        unbalanced=unbalanced,
        unbalanced_type=unbalanced_type,
        debias=debias,
    )


class OTResultMatrix(OTResult):
    def __init__(
        self,
        *,
        a,
        b,
        C,
        potentials,
        array_properties,
        reg,
        reg_type,
        unbalanced,
        unbalanced_type,
        debias,
    ):
        super().__init__(
            a=a,
            b=b,
            C=C,
            potentials=potentials,
            array_properties=array_properties,
            reg=reg,
            reg_type=reg_type,
            unbalanced=unbalanced,
            unbalanced_type=unbalanced_type,
            debias=debias,
        )

        # Fill the dictionary of "expected shapes", that will be used to format the
        # result as expected by the user:
        ap = self._array_properties

        if ap.B == 0:
            batchdim = ()  # No batch dimension
        else:
            batchdim = (ap.B,)  # One batch dimension

        self._shapes = {
            "a": batchdim + (ap.N,),
            "b": batchdim + (ap.M,),
            "C": batchdim + (ap.N, ap.M),
            "B": batchdim,
        }

    @property
    def _exp_fg_C(self):
        """Computes the pseudo transport plan exp((f[i] + g[j] - C[i,j]) / eps)."""
        # Load the relevant quantities:
        f = self._potentials.f_ba  # (B, N)
        g = self._potentials.g_ab  # (B, M)
        C = self._C  # (B, N, M)
        eps = self._reg  # float, > 0

        # Make sure that everyone has the expected shape:
        ap = self._array_properties
        # If ap.B == 0 (no batch mode), we still use B = 1:
        B, N, M = max(1, ap.B), ap.N, ap.M

        assert f.shape == (B, N)
        assert g.shape == (B, M)
        assert C.shape == (B, N, M)
        assert eps > 0

        # Compute the main term in the expression of the optimal plan:
        return bk.exp((f[:, :, None] + g[:, None, :] - C) / eps)  # (B,N,M)

    @property
    def plan(self):
        # Load the relevant quantities:
        a = self._a  # (B, N)
        b = self._b  # (B, M)
        plan = self._exp_fg_C  # (B, N, M)

        # Make sure that everyone has the expected shape:
        ap = self._array_properties
        # If ap.B == 0 (no batch mode), we still use B = 1:
        B, N, M = max(1, ap.B), ap.N, ap.M

        assert a.shape == (B, N)
        assert b.shape == (B, M)
        assert plan.shape == (B, N, M)

        # Actual computation:
        if self._reg_type == "relative entropy":
            # Multiply by the reference product measure:
            plan = a[:, :, None] * b[:, None, :] * plan  # (B,N,1) * (B,1,M) * (B,N,M)
        else:
            raise NotImplementedError(
                "Currently, we only support the computation "
                "of transport plans when `reg_type = 'relative entropy'`."
            )

        return self.cast(plan, "C")  # Cast as a (N,M) or (B,N,M) Tensor

    @property
    def marginal_a(self):
        """First marginal of the transport plan, with the same shape as the source weights `a`."""
        # Compute a[i] * sum_j ( b[j] * exp( (f[i] + g[j] - C[i,j]) / eps) )
        marginal = self._a * bk.sum(
            self._b[:, None, :] * self._exp_fg_C, axis=2
        )  # (B, N)
        return self.cast(marginal, "a")

    @property
    def marginal_b(self):
        """Second marginal of the transport plan, with the same shape as the target weights `b`."""
        # Compute b[j] * sum_i ( a[i] * exp( (f[i] + g[j] - C[i,j]) / eps) )
        marginal = self._b * bk.sum(
            self._a[:, :, None] * self._exp_fg_C, axis=1
        )  # (B, M)
        return self.cast(marginal, "b")


# ----------------------------------------------------------------------------------------
#                              Wasserstein barycenters
# ----------------------------------------------------------------------------------------


# Convention:
# - B is the batch dimension
# - K is the number of measures per barycenter
# - N is the number of samples "for the data"
# - M is the number of samples "for the barycenter"
def barycenter(
    cost,  # (N, M) or (K, N, M) or (B, K, N, M)
    a=None,  # (N,) or (K, N) or (B, K, N)
    weights=None,  # (K,) or (B, K)
    # + all the standard parameters for ot.solve
):
    # masses will be a (M,) or (B, M) array of weights
    return OTResult(potentials=potentials, masses=masses)


# ========================================================================================
#                                 Low-level routines
# ========================================================================================


def softmin_dense(eps: float, C_xy: RealTensor, G_y: RealTensor) -> RealTensor:
    """Softmin function implemented on dense arrays, without using KeOps.

    The softmin function is at the heart of any (stable) implementation
    of the Sinkhorn algorithm. It takes as input:
    - a temperature eps(ilon),
    - a cost matrix C_xy[i,j] = C(x[i],y[j]),
    - a weighted dual potential G_y[j] = log(b(y[j])) + g_ab(y[j]) / eps.

    It returns a new dual potential supported on the points x[i]:
    f_x[i] = - eps * log(sum_j(exp( G_y[j]  -  C_xy[i,j] / eps )))

    In the Sinkhorn loop, we typically use calls like:
        ft_ba = softmin(eps, C_xy, b_log + g_ab / eps)

    Args:
        eps (float > 0): Positive temperature eps(ilon), the main regularization parameter
            of the Sinkhorn algorithm.
        C_xy ((B,N,M) real-valued Tensor): Batch of B cost matrices of shape (N,M).
        G_y ((B,M) real-valued Tensor): Batch of B vectors of shape (M,).

    Returns:
        (B,N) real-valued Tensor:
    """
    assert eps > 0, "We only support positive temperatures (eps > 0)."
    assert len(C_xy.shape) == 3, "C_xy should be a (B,N,M) Tensor."
    assert len(G_y.shape) == 2, "G_y should be a (B,M) Tensor."
    assert C_xy.shape[0] == G_y.shape[0], "Batch dimensions 'B' are incompatible."
    assert C_xy.shape[2] == G_y.shape[1], "Numbers of columns 'M' are incompatible."

    scores_xy = G_y[:, None, :] - C_xy / eps  # (B,N,M)
    return -eps * bk.logsumexp(scores_xy, axis=2)
