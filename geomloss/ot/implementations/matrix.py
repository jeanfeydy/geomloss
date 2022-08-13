# Our generic backend, to use instead of NumPy/PyTorch/...
from ... import backends as bk

# Typing annotations:
from ...typing import RealTensor, CostMatrix

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


def solve(
    cost,  # (N, M) or (B, N, M)  (B is the batch dimension)
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
        raise ValueError("Parameter 'reg' should be >= 0. " f"Received {reg}.")
    elif reg == 0:
        raise NotImplementedError("Currently, we require that reg > 0.")

    if reg_type != "relative entropy":
        raise NotImplementedError("Currently, we only support a Sinkhorn solver.")

    if unbalanced_type != "relative entropy":
        raise NotImplementedError(
            "Currently, we only support unbalanced OT with "
            "a 'relative entropy' penalty on the marginal constraints."
        )

    if method != "auto":
        raise NotImplementedError("Currently, we only support a single method.")

    if ftol is not None or xtol is not None:
        raise NotImplementedError(
            "Currently, we do not support rigorous stopping criteria."
        )

    # Check the parameters ===============================================================

    # Cost matrix ------------------------------------------------------------------------
    C = cost

    # Check the shapes:
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
                    f"The dimension of 'cost' {cost.shape} "
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
                    f"The dimension of 'cost' {cost.shape} "
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
                    f"The dimension of 'cost' {cost.shape} "
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
                    f"The dimension of 'cost' {cost.shape} "
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

    arrays = ArrayProperties(
        B=B,
        N=N,
        M=M,
        dtype=dtype,
        device=device,
        library=library,
    )

    # Actual computations ================================================================
    descent = annealing_parameters(
        diameter=bk.max(C),
        p=1,
        blur=reg,
        reach=unbalanced,
        n_iter=maxiter,
    )

    potentials = sinkhorn_loop(
        softmin=softmin_dense,
        log_a_list=[bk.stable_log(a)],
        log_b_list=[bk.stable_log(b)],
        C_list=[C],
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
        arrays=arrays,
        potentials=potentials,
    )


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
