import numpy as np

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from geomloss import ot
from geomloss import backends as bk
from .generators.common import st_method, cast, OTExperimentConfig
from . import generators
from .check_ot_result import (
    check_ot_result,
    check_ot_result_symmetric,
    check_ot_result_cost_linearity,
)

# ========================================================================================
#           Check that the main properties of OT are respected by ot.solve(...)
# ========================================================================================


# Running solvers with very low reg and maxiter may produce +-inf and then NaN in the
# final computation of the OT cost: we disable this known warning by hand.
@given(
    ex=generators.st_simple_matrix(),
    method=st_method,
)
@pytest.mark.filterwarnings("ignore:overflow encountered in")
def test_symmetry(ex, method):
    """Checks that OT(a,b) = OT(b,a)."""

    solver = ot.solve if len(ex.C.shape) == 2 else ot.solve_batch

    # Compute a direct solution:
    a_to_b = solver(
        ex.C,
        a=ex.a,
        b=ex.b,
        reg=ex.reg,
        unbalanced=ex.unbalanced,
        maxiter=ex.maxiter,
        method=method,
    )
    # Compute a reverse solution:
    b_to_a = solver(
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
@pytest.mark.filterwarnings("ignore:overflow encountered in")
def test_cost_linearity(ex, scaling, offset, method):
    """Checks that OT_{scaling * C + offset}(a,b) = scaling * OT(a,b) + offset if scaling > 0."""

    # TODO: Augment this test with an offset, only for the balanced case
    # This will require a "translation-invariant" initialization of the dual potentials.
    use_offset = 0  # 1 if unbalanced is None else 0
    offset = use_offset * offset

    solver = ot.solve if len(ex.C.shape) == 2 else ot.solve_batch

    # Compute a direct solution:
    normal = solver(
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

    scaled = solver(
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
        normal, scaled, scaling=scaling, offset=offset, atol=1e-2, rtol=5e-2
    )


# ========================================================================================
#                        Corectness checks for ot.solve(...)
# ========================================================================================


def check_solver(
    ex: OTExperimentConfig,
    *,
    method: str,
):
    """Runs the ot.solve() or ot.solve_batch() matrix solver and checks the result."""
    
    solver = ot.solve if len(ex.C.shape) == 2 else ot.solve_batch

    ours = solver(
        ex.C,
        a=ex.a,
        b=ex.b,
        reg=ex.reg,
        unbalanced=ex.unbalanced,
        maxiter=ex.maxiter,
        method=method,
    )
    # Check that all the attributes have the expected values:
    check_ot_result(ours, ex.result, atol=ex.atol, rtol=ex.rtol)


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
    experiment=generators.st_convex_gradients_matrix(),
    method=st_method,
)
@settings(deadline=None)
def test_correct_values_convex_gradients(experiment, method):
    """Checks correctness on clouds of N points in dimension D on which we applied a synthetic deformation.

    This test relies on the fact that OT with a squared Euclidean cost retrieves
    the unique gradient of a convex function that maps the source onto the target.
    """
    check_solver(experiment, method=method)


# In the test below, we use ~100**D samples per distribution.
# To keep run times reasonable, it's best to stick to D=1.
@given(
    experiment=generators.st_gaussians_matrix(),
    method=st_method,
)
@settings(deadline=None)
def test_correct_values_gaussians(experiment, method):
    """Checks correctness on Gaussian distributions, sampled on a regular grid.

    This test relies on the formulas found in:
    "Entropic optimal transport between unbalanced Gaussian measures has a closed form"
    by Janati, Muzellec, Peyré and Cuturi, NeurIPS 2020.
    """
    check_solver(experiment, method=method)
