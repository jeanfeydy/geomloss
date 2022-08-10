import pytest_check as check


def check_approx_equal(a, b):
    """Checks that two numerical arrays are nearly the same."""

    # First of all, our arrays should have the same shape and dtype:
    check.equal(a.dtype, b.dtype)
    check.equal(a.shape, b.shape)




def check_ot_result(us, gt):
    
    # Check that the value is correct:
    check_approx_equal(us.value, gt.value)
    
    # Check that the transport plans are correct:
    check_approx_equal(us.plan, gt.plan)

    # Check the two dual potentials are correct.
    # Note that these are only defined up to an additive constant,
    # so the current comparison is problematic:
    check_approx_equal(us.potential_a, gt.potential_a)
    check_approx_equal(us.potential_b, gt.potential_b)



