import numpy as np

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geomloss import ot
from geomloss import backends as bk
from geomloss.ot.tests.common import cast
from .check_ot_result import check_ot_result, check_ot_result_symmetric

# Generic parameters:
generic_parameters = {
    "N": st.integers(min_value=1, max_value=10),
    "M": st.integers(min_value=1, max_value=10),
    "maxiter": st.integers(min_value=1, max_value=100),
    "debias": st.sampled_from([False]),
    "reg": st.floats(min_value=1e-2, max_value=10.0),
    "reg_type": st.sampled_from(["relative entropy"]),
    "unbalanced": st.one_of(st.none(), st.floats(min_value=1e-2, max_value=10.0)),
    "unbalanced_type": st.sampled_from(["relative entropy"]),
}

# Supported configurations:
all_configs = {
    "method": st.sampled_from(["auto"]),
    "batchsize": st.integers(min_value=0, max_value=2),  # 0 means no batch mode
    "library": st.sampled_from(["numpy", "torch"]),
    "dtype": st.sampled_from(["float32", "float64"]),
    "device": st.sampled_from(["cpu", "cuda"]),
}


# ========================================================================================
#           Check that the main properties of OT are respected by ot.solve(...)
# ========================================================================================


def basic_example(*, N, M, batchsize, unbalanced, library, dtype, device):
    """Generates a minimal input configuration for ot.solve(...)."""

    B = max(1, batchsize)

    C = np.random.randn(B, N, M)  # (B,N,M)
    CT = np.transpose(C, (0, 2, 1))  # (B,M,N)
    a = np.random.rand(B, N)  # (B,N)
    b = np.random.rand(B, M)  # (B,N)

    # If we use balanced OT, the measures must have the same mass:
    if unbalanced is None:
        total_mass = np.random.rand(B, 1)
        a = total_mass * (a / bk.sum(a, axis=1, keepdims=True))
        b = total_mass * (b / bk.sum(b, axis=1, keepdims=True))

    # Cast the arrays as expected:
    C, CT, a, b = [
        cast(x, library=library, dtype=dtype, device=device) for x in [C, CT, a, b]
    ]
    if batchsize == 0:  # No batch mode
        C, CT, a, b = C[0], CT[0], a[0], b[0]

    return {
        "C": C,
        "CT": CT,
        "a": a,
        "b": b,
    }


# Running solvers with very low reg and maxiter may produce +-inf and then NaN in the
# final computation of the OT cost: we disable this known warning by hand.
@given(
    **generic_parameters,
    **all_configs,
)
@pytest.mark.filterwarnings("ignore:overflow encountered in exp")
def test_symmetry(
    N,
    M,
    batchsize,
    library,
    dtype,
    device,
    **params,
):
    """Checks that OT(a,b) = OT(b,a)."""
    ex = basic_example(
        N=N,
        M=M,
        batchsize=batchsize,
        unbalanced=params["unbalanced"],
        library=library,
        dtype=dtype,
        device=device,
    )

    # Compute a direct solution:
    a_to_b = ot.solve(ex["C"], a=ex["a"], b=ex["b"], **params)
    # Compute a reverse solution:
    b_to_a = ot.solve(ex["CT"], a=ex["b"], b=ex["a"], **params)

    # Check that all the attributes coincide as expected:
    dims = (1, 0) if batchsize == 0 else (0, 2, 1)
    transpose = lambda plan: bk.transpose(plan, dims)
    check_ot_result_symmetric(a_to_b, b_to_a, transpose=transpose)


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
@settings(deadline=None)
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
@settings(deadline=None)
def test_correct_values_convex_gradients(N, D, method, **kwargs):
    """Checks correctness on clouds of N points in dimension D on which we applied a synthetic deformation.

    This test relies on the fact that OT with a squared Euclidean cost retrieves
    the unique gradient of a convex function that maps the source onto the target.
    """

    # Load our test case:
    ex = ot.tests.convex_gradients_matrix(N=N, D=D, **kwargs)
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
    blur=st.one_of(st.sampled_from([0]), st.floats(min_value=0.1, max_value=1.0)),
    # blur=st.floats(min_value=0.1, max_value=1.0),
    # N.B.: If rho is too large, the cost is dominated by the marginal constraints
    #       and we cannot satisfy |error| < atol = 1e-2.
    reach=st.one_of(st.none(), st.floats(min_value=1e-2, max_value=10.0)),
    # reach=st.one_of(st.none()),
    **all_configs,
)
@settings(deadline=None)
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
