import numpy as np
from .common import ExpectedOTResult, cast


def diracs_matrix(
    *,
    batchsize,
    library,
    dtype,
    device,
):
    """Generates a minimal example, used by tests/test_ot_solve_matrix.py."""

    # Generate some random data ----------------------------------------------------------
    # We use a simple configuration with one source point and one target point:
    B, N, M = max(1, batchsize), 1, 1

    a = np.ones(B, N)
    b = np.ones(B, M)
    C = 200 * (np.random.rand(B, N, M) - .5)  # Random values in [-100,100)
    value = C.reshape(B)
    plan = np.ones((B, N, M))
    potential_a = C.reshape(B, N) / 2
    potential_b = C.reshape(B, M) / 2

    # Convert to match the expected signatures and return the result ---------------------
    if batchsize == 0:  # No batch mode:
        # (B,) -> (), (B,N) -> (N,), (B,M) -> (M,), (B,N,M) -> (N,M)
        a, b, C, value, plan = a[0], b[0], C[0], value[0], plan[0]
        potential_a, potential_b = potential_a[0], potential_b[0]

    return {
        "cost": cast(C, library=library, dtype=dtype, device=device),
        "maxiter": 100,
        "reg": 1e-4,
        "atol": 1e-2,
        "result": cast(
            ExpectedOTResult(
                value=value,
                # value_linear=value,
                plan=plan,
                potential_a=potential_a,
                potential_b=potential_b,
                marginal_a=a,
                marginal_b=b,
            ),
            library=library,
            dtype=dtype,
            device=device,
        ),
    }
