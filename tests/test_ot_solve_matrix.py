import numpy as np

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geomloss import ot
from geomloss import backends as bk
from .generators.common import cast, OTExperimentConfig
from . import generators
from .check_ot_result import (
    check_ot_result,
    check_ot_result_symmetric,
    check_ot_result_cost_linearity,
)

# Generic parameters:
generic_parameters = {
    "N": st.integers(min_value=1, max_value=10),
    "M": st.integers(min_value=1, max_value=10),
    "maxiter": st.integers(min_value=1, max_value=50),
    "debias": st.sampled_from([False]),
    "reg": st.floats(min_value=1e-2, max_value=10.0),
    "reg_type": st.sampled_from(["relative entropy"]),
}

unbalanced_parameters = {
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

st_method = st.sampled_from(["auto"])


# ========================================================================================
#           Check that the main properties of OT are respected by ot.solve(...)
# ========================================================================================


# Running solvers with very low reg and maxiter may produce +-inf and then NaN in the
# final computation of the OT cost: we disable this known warning by hand.
@given(
    ex=generators.st_simple_matrix(),
    method=st_method,
)
@pytest.mark.filterwarnings("ignore:overflow encountered in exp")
@pytest.mark.filterwarnings("ignore:overflow encountered in cast")
def test_symmetry(ex, method):
    """Checks that OT(a,b) = OT(b,a)."""

    # Compute a direct solution:
    a_to_b = ot.solve(
        ex.C,
        a=ex.a,
        b=ex.b,
        reg=ex.reg,
        unbalanced=ex.unbalanced,
        maxiter=ex.maxiter,
        method=method,
    )
    # Compute a reverse solution:
    b_to_a = ot.solve(
        ex.CT,
        a=ex.b,
        b=ex.a,
        reg=ex.reg,
        unbalanced=ex.unbalanced,
        maxiter=ex.maxiter,
        method=method,
    )

    # Check that all the attributes coincide as expected:
    dims = (1, 0) if len(ex.C.shape) == 2 else (0, 2, 1)
    transpose = lambda plan: bk.transpose(plan, dims)
    check_ot_result_symmetric(
        a_to_b, b_to_a, transpose=transpose, atol=ex.atol, rtol=ex.rtol
    )


@given(
    ex=generators.st_simple_matrix(),
    scaling=st.floats(min_value=0.01, max_value=100.0),
    offset=st.floats(min_value=-100.0, max_value=100.0),
    method=st_method,
)
@pytest.mark.filterwarnings("ignore:overflow encountered in exp")
@pytest.mark.filterwarnings("ignore:overflow encountered in cast")
def test_cost_linearity(ex, scaling, offset, method):
    """Checks that OT_{scaling * C + offset}(a,b) = scaling * OT(a,b) + offset if scaling > 0."""

    # TODO: Augment this test with an offset, only for the balanced case
    # This will require a "translation-invariant" initialization of the dual potentials.
    use_offset = 0  # 1 if unbalanced is None else 0
    offset = use_offset * offset

    # Compute a direct solution:
    normal = ot.solve(
        ex.C,
        a=ex.a,
        b=ex.b,
        reg=ex.reg,
        unbalanced=ex.unbalanced,
        maxiter=100,
        method=method,
    )

    # Compute a scaled solution:
    s_unbalanced = None if ex.unbalanced is None else scaling * ex.unbalanced

    scaled = ot.solve(
        scaling * ex.C + offset,
        a=ex.a,
        b=ex.b,
        reg=scaling * ex.reg,
        unbalanced=s_unbalanced,
        maxiter=100,
        method=method,
    )

    # Check that all the attributes coincide as expected:
    check_ot_result_cost_linearity(
        normal, scaled, scaling=scaling, offset=offset, atol=1e-2, rtol=1e-2
    )


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


def check_solver(
    ex: OTExperimentConfig,
    *,
    method: str,
):
    """Runs the ot.solve() matrix solver and checks the result."""

    # Compute a solution with high precision settings:
    ours = ot.solve(
        ex.C,
        a=ex.a,
        b=ex.b,
        reg=ex.reg,
        unbalanced=ex.unbalanced,
        maxiter=ex.maxiter,
        method=method,
    )
    # Check that all the attributes have the expected values:
    check_ot_result(ours, ex.result, atol=ex.atol)


@given(
    experiment=generators.st_diracs_matrix(),
    method=st_method,
)
def test_correct_values_diracs(experiment, method):
    """Checks correctness on trivial 1-by-1 cost matrices."""
    check_solver(experiment, method=method)


@given(
    experiment=generators.st_permutations_matrix(),
    method=st_method,
)
def test_correct_values_permutations(experiment, method):
    """Checks correctness on (N,N) cost matrix whose associated transport plan is a permutation matrix."""
    check_solver(experiment, method=method)


@given(
    experiment=generators.st_random_matrix(),
    method=st_method,
)
@settings(deadline=None)
def test_correct_values_random(experiment, method):
    """Checks correctness on random (N,N) cost matrices (ground truth = scipy)."""
    check_solver(experiment, method=method)


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
    ex = generators.convex_gradients_matrix(N=N, D=D, **kwargs)
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
    ex = generators.gaussians_matrix(
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
