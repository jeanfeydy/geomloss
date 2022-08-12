from hypothesis import given
from hypothesis import strategies as st

from geomloss import ot
from .check_ot_result import check_ot_result


@given(
    dim=st.integers(min_value=1),
    batchsize=st.integers(min_value=0),
    array=st.sampled_from(["numpy", "torch"]),
    dtype=st.sampled_from(["float32", "float64"]),
    device=st.sampled_from(["cpu", "gpu"]),
    backend=st.sampled_from(["auto"]),
)
def test_correct_values_diracs(dim, batchsize, array, dtype, device, backend):
    """Checks correctness on trivial 1-by-1 cost matrices."""

    # Load our test cases - batchsize=0 means no batch mode:
    ex = ot.tests.diracs_matrix(
        dim=dim,
        batchsize=batchsize,
        array=array,
        dtype=dtype,
        device=device,
    )

    # Compute a solution with high precision settings:
    us = ot.solve(ex["cost"], reg=ex["cost"], maxiter=ex["maxiter"], backend=backend)

    # Check that all the attributes have the expected values:
    check_ot_result(us, ex["result"], atol=ex["atol"])
