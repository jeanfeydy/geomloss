from datetime import timedelta
from hypothesis import given, settings
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
        ex["C"],
        a=ex["a"],
        b=ex["b"],
        reg=ex["reg"],
        unbalanced=ex.get("unbalanced", None),
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


# In the test below, we use ~100**D samples per distribution.
# To keep run times reasonable, it's best to stick to D=1.
@given(
    N=st.integers(min_value=50, max_value=60),  # Spice things up...
    M=st.integers(min_value=51, max_value=60),  # with different values for N and M.
    D=st.integers(min_value=1, max_value=1),  # We stick to 1D examples
    debias=st.sampled_from([False]),
    # We generate Gaussian distributions on [0,1]:
    # blur=st.one_of(st.sampled_from([0]), st.floats(min_value=0.1, max_value=1.0)),
    blur=st.floats(min_value=0.1, max_value=1.0),
    reach=st.one_of(st.none(), st.floats(min_value=1e-2, max_value=100.0)),
    # reach=st.one_of(st.none()),
    **all_configs,
)
@settings(deadline=timedelta(milliseconds=500))
def test_correct_values_gaussians(N, M, D, debias, blur, reach, method, **kwargs):
    """Checks correctness on Gaussian distributions, sampled on a regular grid.

    This test relies on the formulas found in:
    "Entropic optimal transport between unbalanced Gaussian measures has a closed form"
    by Janati, Muzellec, Peyré and Cuturi, NeurIPS 2020.
    """

    # Load our test case:
    ex = ot.tests.gaussians_matrix(
        N=N,
        M=M,
        D=D,
        debias=debias,
        blur=blur,
        reach=reach,
        cov_type="diagonal",
        **kwargs,
    )
    # Run it and check correctness:
    check_solve_correct_values(ex, method=method)


@given(
    N=st.integers(min_value=1, max_value=10),
    **all_configs,
)
def test_correct_values_permutations(N, method, **kwargs):
    """Checks correctness on (N,N) cost matrix whose associated transport plan is a permutation matrix."""

    # Load our test case:
    ex = ot.tests.permutations_matrix(N=N, **kwargs)
    # Run it and check correctness:
    check_solve_correct_values(ex, method=method)


@given(
    N=st.integers(min_value=1, max_value=10),
    **all_configs,
)
def test_correct_values_random(N, method, **kwargs):
    """Checks correctness on random (N,N) cost matrices (ground truth = scipy)."""

    # Load our test case:
    ex = ot.tests.random_matrix(N=N, **kwargs)
    # Run it and check correctness:
    check_solve_correct_values(ex, method=method)


@given(
    N=st.integers(min_value=1, max_value=10),
    D=st.integers(min_value=1, max_value=10),
    **all_configs,
)
def test_correct_values_convex_gradients(N, D, method, **kwargs):
    """Checks correctness on clouds of N points in dimension D on which we applied a synthetic deformation.

    This test relies on the fact that OT with a squared Euclidean cost retrieves
    the unique gradient of a convex function that maps the source onto the target.
    """

    # Load our test case:
    ex = ot.tests.convex_gradients_matrix(N=N, D=D, **kwargs)
    # Run it and check correctness:
    check_solve_correct_values(ex, method=method)
