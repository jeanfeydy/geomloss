import numpy as np
from .common import OTExperimentConfig, ExpectedOTResult, cast
from .common import st_D, st_batchsize, st_library_dtype_device
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as st_arrays

# import torch
# torch.set_printoptions(precision=8)


@st.composite
def st_diracs_matrix(draw):
    """Generates a minimal (1,1) cost matrix and computes the expected solutions in closed form.

    Since this is a trivial example, we should have perfect results
    with any regularization strength and number of iterations.
    However, due to float32 cancellation issues, a value of epsilon < 1e-2
    may lead to small inaccuracies in the transport plan.

    This example is used by tests/test_ot_solve_matrix.py.
    """

    batchsize = draw(st_batchsize)  # integer, 0 means no batch mode

    # Generate some random data ----------------------------------------------------------
    # We use a simple configuration with one source point and one target point:
    B, N, M = max(1, batchsize), 1, 1

    a = np.ones((B, N))
    b = np.ones((B, M))
    # TODO: support inf and NaN
    C = draw(
        st_arrays(
            dtype=np.float64,
            shape=(B, N, M),
            elements=st.floats(min_value=-100, max_value=100),
        )
    )

    value = C.reshape(B)
    plan = np.ones((B, N, M))
    potential_a = C.reshape(B, N) / 2
    potential_b = C.reshape(B, M) / 2

    # Convert to match the expected signatures and return the result ---------------------
    if batchsize == 0:  # No batch mode:
        # (B,) -> (), (B,N) -> (N,), (B,M) -> (M,), (B,N,M) -> (N,M)
        a, b, C, value, plan = a[0], b[0], C[0], value[0], plan[0]
        potential_a, potential_b = potential_a[0], potential_b[0]

    return cast(
        OTExperimentConfig(
            # We also want to test the case where a and b are None (i.e. uniform weights)
            a=draw(st.just(a) | st.none()),
            b=draw(st.just(b) | st.none()),
            C=C,
            max_iter=draw(st.integers(min_value=1, max_value=50)),
            reg=draw(st.floats(min_value=1e-2, max_value=10.0)),
            atol=1e-2,
            result=ExpectedOTResult(
                value=value,
                # value_linear=value,
                plan=plan,
                potential_a=potential_a,
                potential_b=potential_b,
                marginal_a=a,
                marginal_b=b,
            ),
        ),
        **draw(st_library_dtype_device),
    )


@st.composite
def st_diracs_sample(draw):
    """Generates a trivial example with one point on each side, and computes the expected solutions in closed form.

    This example is used by tests/test_ot_solve_sample.py.
    """

    batchsize = 0  # draw(st_batchsize)  # integer, 0 means no batch mode
    D = draw(st_D)  # small integer, dimension of the ambient space
    cost = draw(st.sampled_from(["sqeuclidean"]))

    # Generate some random data ----------------------------------------------------------
    # We use a simple configuration with one source point and one target point:
    B, N, M = max(1, batchsize), 1, 1

    X_a = draw(
        st_arrays(
            dtype=np.float64,
            shape=(B, N, D),
            elements=st.floats(min_value=-10, max_value=10),
        )
    )
    X_b = draw(
        st_arrays(
            dtype=np.float64,
            shape=(B, M, D),
            elements=st.floats(min_value=-10, max_value=10),
        )
    )
    a = np.ones((B, N))
    b = np.ones((B, M))

    if cost == "sqeuclidean":
        C = np.sum((X_a - X_b) ** 2, axis=-1)
        value = C.reshape(B)
        plan = np.ones((B, N, M))
        potential_a = C.reshape(B, N) / 2
        potential_b = C.reshape(B, M) / 2

    # Convert to match the expected signatures and return the result ---------------------
    if batchsize == 0:  # No batch mode:
        X_a, X_b, a, b, C, value, plan = (
            X_a[0],
            X_b[0],
            a[0],
            b[0],
            C[0],
            value[0],
            plan[0],
        )
        potential_a, potential_b = potential_a[0], potential_b[0]

    return cast(
        OTExperimentConfig(
            X_a=X_a,
            X_b=X_b,
            a=draw(st.just(a) | st.none()),
            b=draw(st.just(b) | st.none()),
            cost=cost,
            max_iter=draw(st.integers(min_value=1, max_value=50)),
            reg=draw(st.floats(min_value=1e-2, max_value=10.0)),
            atol=1e-2,
            result=ExpectedOTResult(
                value=value,
                # value_linear=value,
                plan=plan,
                potential_a=potential_a,
                potential_b=potential_b,
                marginal_a=a,
                marginal_b=b,
            ),
        ),
        **draw(st_library_dtype_device),
    )
