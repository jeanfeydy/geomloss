import numpy as np
from scipy.optimize import linear_sum_assignment
from .common import OTExperimentConfig, ExpectedOTResult, cast
from .common import st_N, st_batchsize, st_library_dtype_device
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as st_arrays


@st.composite
def st_random_matrix(draw):
    """Generates a random (N,N) cost matrix and computes the expected solution with scipy.

    This example is used by tests/test_ot_solve_matrix.py.
    """

    N = draw(st.integers(min_value=1, max_value=4))  # small integer
    batchsize = draw(st_batchsize)  # integer, 0 means no batch mode

    # Generate some random data ----------------------------------------------------------
    B, M = max(1, batchsize), N  # M = N, since we are dealing with square matrices

    # The marginals sum up to N:
    a = np.ones((B, N))
    b = np.ones((B, M))

    C = draw(
        st_arrays(
            dtype=np.float64,
            shape=(B, N, M),
            elements=st.floats(min_value=-1, max_value=1),
        )
    )

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
    return cast(
        OTExperimentConfig(
            a=a,
            b=b,
            C=C,
            maxiter=1000,
            reg=1e-4,
            atol=1e-2,
            result=ExpectedOTResult(
                value=value,
                # value_linear=value,
                # plan=plan,
                marginal_a=a,
                marginal_b=b,
            ),
        ),
        **draw(st_library_dtype_device),
    )
