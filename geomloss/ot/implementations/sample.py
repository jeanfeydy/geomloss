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


def squared_distances(x, y, use_keops=False):

    if use_keops and keops_available:
        if x.dim() == 2:
            x_i = LazyTensor(x[:, None, :])  # (N,1,D)
            y_j = LazyTensor(y[None, :, :])  # (1,M,D)
        elif x.dim() == 3:  # Batch computation
            x_i = LazyTensor(x[:, :, None, :])  # (B,N,1,D)
            y_j = LazyTensor(y[:, None, :, :])  # (B,1,M,D)
        else:
            print("x.shape : ", x.shape)
            raise ValueError("Incorrect number of dimensions")

        return ((x_i - y_j) ** 2).sum(-1)

    else:
        if x.dim() == 2:
            D_xx = (x * x).sum(-1).unsqueeze(1)  # (N,1)
            D_xy = torch.matmul(x, y.permute(1, 0))  # (N,D) @ (D,M) = (N,M)
            D_yy = (y * y).sum(-1).unsqueeze(0)  # (1,M)
        elif x.dim() == 3:  # Batch computation
            D_xx = (x * x).sum(-1).unsqueeze(2)  # (B,N,1)
            D_xy = torch.matmul(x, y.permute(0, 2, 1))  # (B,N,D) @ (B,D,M) = (B,N,M)
            D_yy = (y * y).sum(-1).unsqueeze(1)  # (B,1,M)
        else:
            print("x.shape : ", x.shape)
            raise ValueError("Incorrect number of dimensions")

        return D_xx - 2 * D_xy + D_yy


def distances(x, y, use_keops=False):
    if use_keops:
        return squared_distances(x, y, use_keops=use_keops).sqrt()

    else:
        return torch.sqrt(torch.clamp_min(squared_distances(x, y), 1e-8))


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
    if debias:
        # TODO
        pass

    potentials = sinkhorn_loop(
        softmin=softmin_sample,
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

    return SinkhornSamplesOTResult(potentials)


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


class SinkhornSamplesOTResult(OTResult):
    pass


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
