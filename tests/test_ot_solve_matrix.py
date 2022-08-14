from hypothesis import given
from hypothesis import strategies as st

from geomloss import ot
from .check_ot_result import check_ot_result


@given(
    batchsize=st.integers(min_value=0, max_value=5),
    library=st.sampled_from(["numpy", "torch"]),
    dtype=st.sampled_from(["float32", "float64"]),
    device=st.sampled_from(["cpu", "cuda"]),
    method=st.sampled_from(["auto"]),
)
def test_correct_values_diracs(batchsize, library, dtype, device, method):
    """Checks correctness on trivial 1-by-1 cost matrices."""

    # Load our test cases - batchsize=0 means no batch mode:
    ex = ot.tests.diracs_matrix(
        batchsize=batchsize,
        library=library,
        dtype=dtype,
        device=device,
    )

    # Compute a solution with high precision settings:
    us = ot.solve(ex["cost"], reg=ex["reg"], maxiter=ex["maxiter"], method=method)

    # Check that all the attributes have the expected values:
    check_ot_result(us, ex["result"], atol=ex["atol"])
