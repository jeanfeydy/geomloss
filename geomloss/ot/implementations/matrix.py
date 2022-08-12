# Our generic backend, to use instead of NumPy/PyTorch/...
from ... import backends as bk

# Typing annotations:
from ...typing import RealTensor, CostMatrix

# Abstract class for our results:
from ..ot_result import OTResult

# Abstract solvers and annealing strategy:
from ..abstract_solvers import (
    sinkhorn_loop,
    sinkhorn_barycenter_loop,
    annealing_parameters,
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
    xtol=None,  # Changes of the variables? We may just use "tol" and let the solver decide...
    # As far as I can tell, most users are interested in a
    # simple, one-dimensional parameter that lets them trade
    # time for accuracy. For instance, we could allow
    # users to specify an "rtol" ratio in (0, 1)
    # that would drive a sensible heuristic for the parameter choices.
    # The main thing to guarantee is a monotonic improvement
    # of the solution quality as rtol goes to 0.
    rtol=1e-2,  # ~1% relative accuracy
):
    # Basic checks on the parameters -----------------------------------------------------
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
        
    # Check the parameters ---------------------------------------------------------------

    # Actual computations ----------------------------------------------------------------
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
    return OTResultMatrix(potentials)


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
