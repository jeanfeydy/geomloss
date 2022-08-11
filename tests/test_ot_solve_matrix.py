from hypothesis import given
from hypothesis import strategies as st

from geomloss import ot
from .check_ot_result import check_ot_result


@given(dim=st.integers(min_value=1), batchsize=st.integers(min_value=0))
def test_correct_values_diracs(dim, batchsize):
    """Checks correctness on trivial 1-by-1 cost matrices."""

    # Load our test cases - batchsize=0 means no batch mode:
    test_case = ot.tests.diracs_matrix(dim=dim, batchsize=batchsize)
    
    # Compute a solution with high precision settings:
    us = ot.solve(test_case["cost"], reg=1e-4, maxiter=100)
    
    # Check that all the attributes have the expected values:
    check_ot_result(us, test_case["result"])