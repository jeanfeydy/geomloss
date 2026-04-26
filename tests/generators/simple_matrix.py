import numpy as np
from .common import OTExperimentConfig, ExpectedOTResult, cast
from .common import st_N, st_M, st_batchsize, st_library_dtype_device
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as st_arrays
from geomloss import backends as bk


@st.composite
def st_simple_matrix(draw):
    """Generates a minimal input configuration for ot.solve(...).

    This example is used by tests/test_ot_solve_matrix.py.
    """

    N = draw(st_N)  # small integer
    M = draw(st_M)  # small integer
    batchsize = draw(st_batchsize)  # integer, 0 means no batch mode
    probability = draw(st.booleans())
    unbalanced = draw(st.one_of(st.none(), st.floats(min_value=1e-2, max_value=10.0)))

    # Generate some random data ----------------------------------------------------------
    B = max(1, batchsize)

    # TODO: support inf and NaN
    C = draw(
        st_arrays(
            dtype=np.float64,
            shape=(B, N, M),
            elements=st.floats(min_value=-10, max_value=10),
        )
    )  # (B,N,M)

    CT = np.transpose(C, (0, 2, 1))  # (B,M,N)

    a = draw(
        st_arrays(
            dtype=np.float64,
            shape=(B, N),
            elements=st.floats(min_value=0.1, max_value=10),
        )
    )  # (B,N)
    b = draw(
        st_arrays(
            dtype=np.float64,
            shape=(B, M),
            elements=st.floats(min_value=0.1, max_value=10),
        )
    )  # (B,M)

    # If we use balanced OT, the measures must have the same mass:
    if probability:
        a = a / bk.sum(a, axis=1, keepdims=True)
        b = b / bk.sum(b, axis=1, keepdims=True)

    elif unbalanced is None:
        total_mass = draw(
            st_arrays(
                dtype=np.float64,
                shape=(B, 1),
                elements=st.floats(min_value=0.1, max_value=10),
            )
        )
        a = total_mass * (a / bk.sum(a, axis=1, keepdims=True))
        b = total_mass * (b / bk.sum(b, axis=1, keepdims=True))

    if batchsize == 0:  # No batch mode
        C, CT, a, b = C[0], CT[0], a[0], b[0]

    return cast(
        OTExperimentConfig(
            a=a,
            b=b,
            C=C,
            CT=CT,
            max_iter=draw(st.integers(min_value=1, max_value=100)),
            reg=draw(st.floats(min_value=1e-2, max_value=10.0)),
            atol=1e-3,
            rtol=1e-3,
            unbalanced=unbalanced,
        ),
        **draw(st_library_dtype_device),
    )
