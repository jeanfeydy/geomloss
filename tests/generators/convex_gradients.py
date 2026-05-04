import numpy as np
from .common import OTExperimentConfig, ExpectedOTResult, cast
from .common import st_N, st_D, st_batchsize, st_library_dtype_device
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as st_arrays


def random_points(*, draw, B, N, D):
    """Generates a batch of B clouds of N points in dimension D.

    We apply a synthetic deformation which is the gradient of a random sum of Euclidean
    distances to N key points.
    """

    # We use random weights that sum up to 1:
    weights = draw(
        st_arrays(
            dtype=np.float64,
            shape=(B, N),
            elements=st.floats(min_value=0.01, max_value=1.0),
        )
    )
    weights = weights / np.sum(weights, axis=1, keepdims=True)  # rows sum up to 1

    # TODO: Fix this with a better error criterion on the plan and marginals...
    if False:
        x_i = np.random.rand(B, N, D)  # (B,N,D) source point cloud
    else:
        # Uniform spacing of the source points along a segment
        x_i = 0.5 * np.ones((B, N, D))
        for k in range(B):
            x_i[k, :, 0] = np.arange(N) / N

    # As a random convex function, we use a sum of Euclidean norms
    # f(x) = sum_j=1^N v_j * |x - z_j|
    # with gradient
    # g(x) = sum_j=1^N v_j * normalize(x - z_j)
    v_j = draw(
        st_arrays(
            dtype=np.float64,
            shape=(B, N),
            elements=st.floats(min_value=0.1, max_value=2.0),
        )
    )
    z_j = draw(
        st_arrays(
            dtype=np.float64,
            shape=(B, N, D),
            elements=st.floats(min_value=0.0, max_value=1.0),
        )
    )

    # Compute normalize(x_i - z_j):
    diff_ij = x_i.reshape(B, N, 1, D) - z_j.reshape(B, 1, N, D)  # (B,N,N,D)
    norms_ij = np.sqrt(np.sum(diff_ij**2, axis=-1, keepdims=True))  # (B,N,N,1)
    norms_ij[norms_ij == 0] = 1  # Make 100% sure that we won't see any NaN
    diff_ij = diff_ij / norms_ij  # (B,N,N,D)

    # Compute the weighted sum:
    delta_i = np.sum(v_j.reshape(B, 1, N, 1) * diff_ij.reshape(B, N, N, D), axis=2)

    # Add it to the source points x_i to generate the target points y_i:
    y_i = x_i + delta_i  # (B,N,D)

    # Compute the expected OT value, for a ground cost C(x,y) = 0.5 * |x - y|^2
    sqdists = np.sum((x_i - y_i) ** 2, axis=2)  # (B,N)
    value = 0.5 * np.sum(weights * sqdists, axis=1)  # (B,)

    return {
        "x": x_i,
        "y": y_i,
        "weights": weights,
        "value": value,
    }


@st.composite
def st_convex_gradients_matrix(draw):
    """Generates a random cloud of N points in dimension D and applies a random gradient of a convex function.

    This example is used by tests/test_ot_solve_matrix.py.
    """

    N, D = draw(st_N), draw(st_D)  # small integers
    batchsize = draw(st_batchsize)  # integer, 0 means no batch mode

    # Generate some random data ----------------------------------------------------------
    B, M = max(1, batchsize), N  # M = N, since we just move points around.

    # Generate a random configuration, encoded using (B,N,D) arrays:
    points = random_points(draw=draw, B=B, N=N, D=D)

    # Turn this data into "matrix" format:
    a = points["weights"]  # (B,N)
    b = points["weights"]  # (B,M)

    x_i = points["x"].reshape(B, N, 1, D)
    y_j = points["y"].reshape(B, 1, M, D)
    C = np.sum(0.5 * (x_i - y_j) ** 2, axis=3)  # (B,N,M)

    value = points["value"]  # (B,)

    # The transport plan is a diagonal matrix with entries that correspond to the weights:
    plan = np.zeros((B, N, M))
    for k in range(B):
        plan[k, np.arange(N), np.arange(N)] = points["weights"][k, :]

    # Convert to match the expected signatures and return the result ---------------------
    if batchsize == 0:  # No batch mode:
        # (B,) -> (), (B,N) -> (N,), (B,M) -> (M,), (B,N,M) -> (N,M)
        a, b, C, value, plan = a[0], b[0], C[0], value[0], plan[0]

    return cast(
        OTExperimentConfig(
            a=a,
            b=b,
            C=C,
            max_iter=1000,
            reg=1e-3,
            atol=1e-2,
            rtol=1e-2,
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
