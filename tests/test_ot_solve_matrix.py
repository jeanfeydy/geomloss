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
        ex["C"],
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


# In the test below, we use 100**D samples per distribution.
# To keep run times reasonable, it's best to stick to D=1.
@given(
    D=st.integers(min_value=1, max_value=1),
    debias=st.sampled_from([False, True]),
    reg=st.floats(min_value=1e-4, max_value=1.0),
    unbalanced=st.one_of(st.none(), st.floats(min_value=1e-4, max_value=100.0)),
    **all_configs,
)
def test_correct_values_gaussians(D, debias, reg, unbalanced, method, **kwargs):
    """Checks correctness on Gaussian distributions, sampled on a regular grid.

    This test relies on the formulas found in:
    "Entropic optimal transport between unbalanced Gaussian measures has a closed form"
    by Janati, Muzellec, Peyré and Cuturi, NeurIPS 2020.
    """

    # Load our test case:
    ex = ot.tests.gaussians_matrix(
        N=100, D=D, debias=debias, reg=reg, unbalanced=unbalanced, **kwargs
    )
    # Run it and check correctness:
    check_solve_correct_values(ex, method=method)
