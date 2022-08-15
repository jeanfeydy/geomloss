from hypothesis import given
from hypothesis import strategies as st

from geomloss import ot
from .check_ot_result import check_ot_result


# Supported configurations:
all_configs = {
    "method": st.sampled_from(["auto"]),
    "batchsize": st.integers(min_value=0, max_value=5),  # 0 means no batch mode
    "library": st.sampled_from(["numpy", "torch"]),
    "dtype": st.sampled_from(["float32", "float64"]),
    "device": st.sampled_from(["cpu", "cuda"]),
}

# ========================================================================================
#                        Corectness checks for ot.solve(...)
# ========================================================================================


def check_solve_correct_values(ex, *, method):
    """Runs the ot.solve() matrix solver and checks the result."""

    # Compute a solution with high precision settings:
    us = ot.solve(
        ex["cost"],
        a=ex["a"],
        b=ex["b"],
        reg=ex["reg"],
        maxiter=ex["maxiter"],
        method=method,
    )
    # Check that all the attributes have the expected values:
    check_ot_result(us, ex["result"], atol=ex["atol"])


@given(**all_configs)
def test_correct_values_diracs(method, **kwargs):
    """Checks correctness on trivial 1-by-1 cost matrices."""

    # Load our test case:
    ex = ot.tests.diracs_matrix(**kwargs)
    # Run it and check correctness:
    check_solve_correct_values(ex, method=method)


@given(N=st.integers(min_value=1, max_value=10), **all_configs)
def test_correct_values_random(N, method, **kwargs):
    """Checks correctness on random (N,N) cost matrices (ground truth = scipy)."""

    # Load our test case:
    ex = ot.tests.random_matrix(N=N, **kwargs)
    # Run it and check correctness:
    check_solve_correct_values(ex, method=method)
