import numpy as np
from .common import ExpectedOTResult, cast


def permutations_matrix(
    *,
    N,
    batchsize,
    **kwargs,
):
    """Generates a random (N,N) cost matrix whose associated transport plan is a permutation matrix.

    This example is used by tests/test_ot_solve_matrix.py.
    """

    # Generate some random data ----------------------------------------------------------
    B, M = max(1, batchsize), N  # M = N, since we are dealing with square matrices

    # The marginals sum up to N:
    a = np.ones((B, N))
    b = np.ones((B, M))
    C = np.random.rand(B, N, M) + 2  # Random values in [2,3)

    # Ensure that we will retrieve a given permutation matrix by putting
    # small values in C[i, sigma[i]]:
    value = np.zeros((B,))
    plan = np.zeros((B, N, M))
    for k in range(B):
        # Draw a random permutation sigma of the row indices:
        row_ind = np.arange(N)
        col_ind = np.random.permutation(N)
        # Put N small coefficients in [0,1) in C[i, sigma[i]]:
        C[k,row_ind,col_ind] = np.random.rand(N)
        # This ensures that the "correct" result corresponds to sigma:
        value[k] = C[k, row_ind, col_ind].sum()
        plan[k, row_ind, col_ind] = 1

    # Convert to match the expected signatures and return the result ---------------------
    if batchsize == 0:  # No batch mode:
        # (B,) -> (), (B,N) -> (N,), (B,M) -> (M,), (B,N,M) -> (N,M)
        a, b, C, value, plan = a[0], b[0], C[0], value[0], plan[0]

    # N.B.: Sinkhorn multiscale really isn't very good for this type of unstructured
    #       problem, so we have to use a lot of iterations to ensure correctness.
    return cast(
        {
            "a": a,
            "b": b,
            "C": C,
            "maxiter": 100,
            "reg": 1e-4,
            "atol": 1e-2,
            "result": ExpectedOTResult(
                value=value,
                # value_linear=value,
                plan=plan,
                marginal_a=a,
                marginal_b=b,
            ),
        },
        **kwargs,
    )
