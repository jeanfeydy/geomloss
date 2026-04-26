import numpy as np
from .common import OTExperimentConfig, ExpectedOTResult, cast
from .common import st_N, st_batchsize, st_library_dtype_device
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as st_arrays


@st.composite
def st_permutations_matrix(draw):
    """Generates a random (N,N) cost matrix whose associated transport plan is a permutation matrix.

    This example is used by tests/test_ot_solve_matrix.py.
    """

    N = draw(st_N)  # small integer
    batchsize = draw(st_batchsize)  # integer, 0 means no batch mode

    # Generate some random data ----------------------------------------------------------
    B, M = max(1, batchsize), N  # M = N, since we are dealing with square matrices

    # The marginals sum up to N:
    a = np.ones((B, N))
    b = np.ones((B, M))

    threshold = draw(st.floats(min_value=0.0, max_value=10.0))  # large positive float
    gap = draw(st.floats(min_value=5, max_value=10.0))  # large positive float

    # Generate a random cost matrix with values larger than threshold
    # TODO: support inf and NaN
    C = draw(
        st_arrays(
            dtype=np.float64,
            shape=(B, N, M),
            elements=st.floats(min_value=threshold, max_value=20),
        )
    )
    # except for N small values in each batch, which we will put in C[i, sigma[i]] to ensure that the optimal plan is a permutation matrix:
    small_values = draw(
        st_arrays(
            dtype=np.float64,
            shape=(B, N),
            elements=st.floats(min_value=-20, max_value=threshold - gap),
        )
    )

    # Ensure that we will retrieve a given permutation matrix by putting
    # small values in C[i, sigma[i]]:
    value = np.zeros((B,))
    plan = np.zeros((B, N, M))
    for batch in range(B):
        # Draw a random permutation sigma of the row indices:
        row_ind = np.arange(N)
        col_ind = draw(st.permutations(row_ind))
        # Put N small coefficients in C[i, sigma[i]]:
        C[batch, row_ind, col_ind] = small_values[batch]
        # This ensures that the "correct" result corresponds to sigma:
        value[batch] = C[batch, row_ind, col_ind].sum()
        plan[batch, row_ind, col_ind] = 1

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
            maxiter=100,
            reg=1e-1,
            atol=5e-2,
            rtol=5e-2,
            result=ExpectedOTResult(
                value=value,
                # value_linear=value,
                plan=plan,
                marginal_a=a,
                marginal_b=b,
            ),
        ),
        **draw(st_library_dtype_device),
    )
