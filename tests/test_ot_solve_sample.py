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
#                     Corectness checks for ot.solve_sample(...)
# ========================================================================================


def check_solver(
    ex: OTExperimentConfig,
    *,
    method: str,
):
    """Runs ot.solve_sample() or ot.solve_sample_batch() and checks the result."""

    solver = ot.solve_sample if len(ex.X_a.shape) == 2 else ot.solve_sample_batch

    ours = solver(
        ex.X_a,
        ex.X_b,
        a=ex.a,
        b=ex.b,
        cost=ex.cost,
        reg=ex.reg,
        unbalanced=ex.unbalanced,
        max_iter=ex.max_iter,
        method=method,
    )
    # Check that all the attributes have the expected values:
    check_ot_result(ours, ex.result, atol=ex.atol, rtol=ex.rtol)


@given(
    experiment=generators.st_diracs_sample(),
    method=st_method,
)
@settings(deadline=None)
def test_correct_values_diracs(experiment, method):
    """Checks correctness on trivial examples with one point on each side."""
    check_solver(experiment, method=method)
