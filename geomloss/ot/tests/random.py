import numpy as np
from scipy.optimize import linear_sum_assignment
from .common import ExpectedOTResult, cast


def random_matrix(
    *,
    N,
    batchsize,
    **kwargs,
):
    """Generates a random (N,N) cost matrix and computes the expected solution with scipy.

    This example is used by tests/test_ot_solve_matrix.py.
    """

    # Generate some random data ----------------------------------------------------------
    # We use a simple configuration with one source point and one target point:
    B, M = max(1, batchsize), N  # M = N, since we are dealing with square matrices

    # The marginals sum up to N:
    a = np.ones((B, N))
    b = np.ones((B, M))
    C = 2 * (np.random.rand(B, N, M) - 0.5)  # Random values in [-1,1)

    # Use the SciPy implementation of the O(N^3) Jonker-Volgenant algorithm:
    value = np.zeros((B,))
    plan = np.zeros((B, N, M))
    for k in range(B):
        row_ind, col_ind = linear_sum_assignment(C[k, :, :])
        value[k] = C[k, row_ind, col_ind].sum()
        plan[k, row_ind, col_ind] = 1

    # Convert to match the expected signatures and return the result ---------------------
    if batchsize == 0:  # No batch mode:
        # (B,) -> (), (B,N) -> (N,), (B,M) -> (M,), (B,N,M) -> (N,M)
        a, b, C, value, plan = a[0], b[0], C[0], value[0], plan[0]

    # N.B.: Sinkhorn multiscale really isn't very good for this type of unstructured 
    #       problem, so we have to use a lot of iterations to ensure correctness.
    return {
        "cost": cast(C, **kwargs),
        "a": cast(a, **kwargs),
        "b": cast(b, **kwargs),
        "maxiter": 1000,
        "reg": 1e-4,
        "atol": 1e-2,
        "result": cast(
            ExpectedOTResult(
                value=value,
                # value_linear=value,
                plan=plan,
                marginal_a=a,
                marginal_b=b,
            ),
            **kwargs,
        ),
    }
