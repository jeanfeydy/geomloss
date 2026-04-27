# Our generic backend, to use instead of NumPy/PyTorch/...
from ... import backends as bk

# Typing annotations:
from ...typing import RealTensor, CostMatrices, CostMatrix

# Abstract class for our results:
from ..ot_result import OTResult

# Abstract solvers and annealing strategy:
from ..abstract_solvers import (
    sinkhorn_loop,
    # sinkhorn_barycenter_loop,
    max_diameter,
    annealing_parameters,
)

# Utility functions:
from ...arguments import (
    ArrayProperties,
    check_library_dtype_device,
    check_regularization,
    check_marginal,
    check_marginal_masses,
)

# ========================================================================================
#                              Cost functions and softmin
# ========================================================================================


def squared_distances(x, y):
    N, D = x.shape
    M, D_ = y.shape
    assert D == D_, "x and y should have the same number of coordinates per sample."

    if bk.keops_available:
        x_i = bk.LazyTensor(bk.view(x, (N, 1, D)))  # (N,1,D)
        y_j = bk.LazyTensor(bk.view(y, (1, M, D)))  # (1,M,D)

        return ((x_i - y_j) ** 2).sum(-1)

    else:
        D_xx = bk.view(bk.sum(x * x, axis=-1), (N, 1))  # (N,1)
        D_xy = x @ bk.transpose(y, (1, 0))  # (N,D) @ (D,M) = (N,M)
        D_yy = bk.view(bk.sum(y * y, axis=-1), (1, M))  # (1,M)

        return D_xx - 2 * D_xy + D_yy


def distances(x, y):
    if bk.use_keops:
        return squared_distances(x, y).sqrt()
    else:
        return bk.sqrt(bk.clamp_min(squared_distances(x, y), 1e-8))


def cost_matrix(x, y, *, cost="sqeuclidean"):
    N, D = x.shape
    M, D_ = y.shape
    assert D == D_, "x and y should have the same number of coordinates per sample."

    if cost == "sqeuclidean":
        C_ij = squared_distances(x, y)

    else:
        raise NotImplementedError()

    assert C_ij.shape == (N, M), "Cost matrix should have shape (N,M)."
    return C_ij


def softmin_sample(
    eps: float,
    log_weights: RealTensor,
    costs: CostMatrix,
    potentials: RealTensor,
) -> RealTensor:
    """Softmin function implemented with KeOps.

    The softmin function is at the heart of any (stable) implementation
    of the Sinkhorn algorithm. It takes as input:
    - a temperature eps(ilon),
    - log_weights lb[j] = log(b(y[j])) of shape (M,),
    - a cost matrix C_xy[i,j] = C(x[i],y[j]) of shape (N, M),
    - a weighted dual potential G_y[j] = g_ab(y[j]) of shape (M,).

    It returns a new dual potential supported on the points x[i]:
    f_x[i] = - eps * log(sum_j(exp[ lb[j] + (G_y[j]  -  C_xy[i,j]) / eps ] ))

    In the Sinkhorn loop, we typically use calls like:
        ft_ba = softmin(eps, b_log, C_xy, g_ab)

    Args:
        eps (float >= 0): Temperature eps(ilon), the main regularization parameter
            of the Sinkhorn algorithm.
        log_weights ((M,) real-valued Tensor): Batch of B vectors of shape (M,) containing
            the logarithm of the weights of the measure b.
        costs ((N,M) LazyTensor): Cost matrix of shape (N,M).
        potentials ((M,) real-valued Tensor): Vector of shape (M,).

    Returns:
        (N,) real-valued Tensor:
    """
    log_b_y = log_weights
    C_xy = costs
    g_y = potentials

    eps = float(eps)  # To avoid stealthy dtype issues with NumPy/PyTorch scalars

    assert eps >= 0, "We only support non-negative temperatures (eps >= 0)."
    assert len(C_xy.shape) == 2, "C_xy should be a (N,M) Tensor."
    N, M = C_xy.shape

    assert g_y.shape == (M,), "g_y should be a (M,) Tensor."
    assert log_b_y.shape == (M,), "log_b_y should be a (M,) Tensor."

    C_is_lazy = hasattr(C_xy, "_shape")

    if eps == float("inf"):
        # TODO: handle the case where b is not a probability measure
        # Currently, we're "missing" the -eps * log(b_y.sum()) term.
        b_y = bk.exp(log_b_y)  # (M,)
        sum_b = bk.sum(b_y, axis=0, keepdims=True)  # (1,)

        # Compute f_i of shape (N,):
        if C_is_lazy:
            g_y_j = bk.LazyTensor(bk.view(g_y, (1, M, 1)))
            b_y_j = bk.LazyTensor(bk.view(b_y, (1, M, 1)))
            # f_i = ((C_xy - g_y_j) * b_y_j).sum(axis=1)  # (N,)
        else:
            g_y_j = bk.view(g_y, (1, M))
            b_y_j = bk.view(b_y, (1, M))

        f_i = bk.sum((C_xy - g_y_j) * b_y_j, axis=1)
        f_i = bk.view(f_i, (N,))  # May be important with KeOps
        return f_i / sum_b

    elif eps == 0:
        # TODO: handle the case where some of the b_y are zero
        if C_is_lazy:
            g_y_j = bk.LazyTensor(bk.view(g_y, (1, M, 1)))
            # f_i = (C_xy - g_y_j).min(axis=1)  # (N,) TODO: gradient?
        else:
            g_y_j = bk.view(g_y, (1, M))

        f_i = bk.amin(C_xy - g_y_j, axis=1)  # (N,)
        f_i = bk.view(f_i, (N,))  # May be important with KeOps
        return f_i

    else:
        if C_is_lazy:
            s_j = bk.LazyTensor(bk.view(log_b_y + g_y / eps, (1, M, 1)))
            # scores_xy = s_j - C_xy / eps  # (N, M)
            # s_i = scores_xy.logsumexp(axis=1)
        else:
            s_j = bk.view(log_b_y + g_y / eps, (1, M))

        scores_xy = s_j - C_xy / eps  # (N, M)
        s_i = bk.logsumexp(scores_xy, axis=1)
        s_i = bk.view(s_i, (N,))  # May be important with KeOps
        return -eps * s_i


# ========================================================================================
#                                  Actual solvers
# ========================================================================================


# OT on empirical distributions
def solve_sample(
    X_a,  # (N, D)
    X_b,  # (M, D)
    a=None,  # (N,)
    b=None,  # (M,)
    cost="sqeuclidean",
    # We will also support simple functions such as "lambda C(x_i,y_j) = ((x_i - y_j) ** 2).sum(-1) / 2".
    debias=False,
    # Regularization:
    reg=None,  # -> None by default
    reg_type="KL",
    # Unbalanced OT:
    unbalanced=None,  # None = +infty -> balanced by default;
    # We will also support scalar numbers, pairs of scalar numbers,
    # ((N,), (M,)) and ((B, N), (B, M)) point-dependent penalties.
    unbalanced_type="KL",
    # Optim parameters, following SciPy convention:
    method="auto",  # We can match keywords in this
    # string to activate some options such as
    # "symmetric", "annealing", etc.
    # Tolerance values.
    max_iter=None,
    tol=None,
    # Redundant parameters, that make sense for geometric problems:
    blur=None,  # Specifies "epsilon" = p * blur^p
    reach=None,  # Specifies "rho" = p * reach^p
    # + same other params as above
):
    if cost == "sqeuclidean":
        p = 2
    else:
        p = 1

    if blur is not None:
        if reg is not None:
            raise ValueError(
                "Parameters 'reg' and 'blur' are redundant. Please specify only one of them."
            )
        reg = p * (blur**p)  # Multiply by p because there is no 1/p in the cost

    if reach is not None:
        if unbalanced is not None:
            raise ValueError(
                "Parameters 'unbalanced' and 'reach' are redundant. Please specify only one of them."
            )
        unbalanced = p * (reach**p)  # Multiply by p because there is no 1/p in the cost

    # Basic checks on the solver parameters ==============================================
    check_regularization(
        reg=reg,
        reg_type=reg_type,
        unbalanced=unbalanced,
        unbalanced_type=unbalanced_type,
        method=method,
        tol=tol,
    )

    # Check the input data ===============================================================
    # Samples ----------------------------------------------------------------------------
    if len(X_a.shape) != 2:
        raise ValueError(f"Expected X_a to be a (N, D) array. Received {X_a.shape}.")
    if len(X_b.shape) != 2:
        raise ValueError(f"Expected X_b to be a (M, D) array. Received {X_b.shape}.")

    N, D = X_a.shape
    M, D_ = X_b.shape
    if D != D_:
        raise ValueError(
            f"Expected X_a and X_b to have the same number of coordinates per sample. "
            f"Received D={D} for X_a and D={D_} for X_b."
        )

    # Marginals --------------------------------------------------------------------------
    a = check_marginal(a, ones_like=X_a[:, 0], marginal_size=N, name="a")
    b = check_marginal(b, ones_like=X_b[:, 0], marginal_size=M, name="b")

    if unbalanced is None:  # if we work in balanced mode
        sums_a = bk.sum(a, axis=0, keepdims=True)  # (1,)
        sums_b = bk.sum(b, axis=0, keepdims=True)  # (1,)
        check_marginal_masses(sums_a, sums_b)

    # Low-level compatibility ------------------------------------------------------------
    library, dtype, device = check_library_dtype_device(X_a, X_b, a, b)

    array_properties = ArrayProperties(
        B=0,  # No batch dimension
        N=N,
        M=M,
        dtype=dtype,
        device=device,
        library=library,
    )

    # Actual computations ================================================================
    descent = annealing_parameters(
        maxmin_cost=max_diameter(X_a, X_b) ** p,
        eps=reg,
        rho=unbalanced,
        n_iter=max_iter,
    )

    # TODO: implement the two-scales method
    C_xy = cost_matrix(X_a, X_b, cost=cost)
    C_yx = cost_matrix(X_b, X_a, cost=cost)

    if debias:
        C_xx = cost_matrix(X_a, X_a, cost=cost)
        C_yy = cost_matrix(X_b, X_b, cost=cost)

    else:
        C_xx = None
        C_yy = None

    C_list = [CostMatrices(xy=C_xy, yx=C_yx, xx=C_xx, yy=C_yy)]

    potentials = sinkhorn_loop(
        softmin=softmin_sample,
        log_a_list=[bk.stable_log(a)],
        log_b_list=[bk.stable_log(b)],
        C_list=C_list,
        descent=descent,
        debias=debias,
        last_extrapolation=True,
    )

    return OTResultSample(
        X_a=X_a,
        X_b=X_b,
        a=a,
        b=b,
        C=C_list[-1],  # The finest scale
        cost=cost,
        reg=reg,
        reg_type=reg_type,
        unbalanced=unbalanced,
        unbalanced_type=unbalanced_type,
        debias=debias,
        potentials=potentials,
        array_properties=array_properties,
    )


# To support heterogeneous batches (which are very common in shape analysis),
# we will at some point let users specify "batch vectors" following PyTorch_Geometric's convention:
# https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#mini-batches
# batch_a=None,  # (N,) vector of increasing integer values in [0,B-1]
# batch_b=None,  # (M,) vector of increasing integer values in [0,B-1]


def solve_sample_batch(
    X_a,  # (B, N, D)
    X_b,  # (B, M, D)
    a=None,  # (B, N)
    b=None,  # (B, M)
    cost="sqeuclidean",
    debias=False,
    # Regularization:
    reg=0,  # -> None by default
    reg_type="KL",
    # Unbalanced OT:
    unbalanced=None,  # None = +infty -> balanced by default;
    # We will also support scalar numbers, pairs of scalar numbers,
    # ((N,), (M,)) and ((B, N), (B, M)) point-dependent penalties.
    unbalanced_type="KL",
    # Optim parameters, following SciPy convention:
    method="auto",  # We can match keywords in this
    # string to activate some options such as
    # "symmetric", "annealing", etc.
    # Tolerance values.
    max_iter=None,
    tol=None,
    p=None,
    blur=None,
    reach=None,
):
    args, output_shapes = cast_input(
        xa=(xa, "B,N,D"),
        xb=(xb, "B,M,D"),
        a=(a, "B,N"),
        b=(b, "B,M"),
        a_batch=(a_batch, "N"),
        b_batch=(b_batch, "M"),
    )

    return SinkhornSamplesOTResult(potentials)


class OTResultSample(OTResult):
    def __init__(
        self,
        *,
        X_a,
        X_b,
        a,
        b,
        C,
        cost,
        reg,
        reg_type,
        unbalanced,
        unbalanced_type,
        debias,
        potentials,
        array_properties,
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

        self._X_a = X_a
        self._X_b = X_b
        self._cost = cost

        # Fill the dictionary of "expected shapes", that will be used to format the
        # result as expected by the user:
        ap = self._array_properties

        if ap.B == 0:
            # Under the hood, we always work with batch dimensions, even if the user did not provide one.
            self._shapes = {
                "a": (ap.N,),
                "b": (ap.M,),
                "C": (ap.N, ap.M),
                "B": (),
            }
        else:
            raise NotImplementedError()


# Convention:
# - D is the number of coordinates per sample (= point)
def barycenter_sample(
    xa,  # (N, D) or (K, N, D) or (B, K, N, D)
    a=None,  # (N,) or (K, N) or (B, K, N)
    weights=None,  # (K,) or (B, K)
    # + all the standard parameters for ot.solve_samples
):
    # masses will be a (M,) or (B, M) array of weights
    # samples will be a (M, D) or (B, M, D) array of coordinates
    return OTResult(potentials=potentials, masses=masses, samples=samples)
