import pytest_check as check
from geomloss import backends as bk


def check_approx_equal(a, b, atol=1e-3, name=""):
    """Checks that two numerical arrays are nearly the same.

    If b is None, we skip the checks.
    """

    if b is not None:
        # First of all, our arrays should have the same shape and dtype:
        check.equal(a.dtype, b.dtype, f"The dtype of `{name}` is not correct.")
        check.equal(a.shape, b.shape, f"The shape of `{name}` is not correct.")

        # Also check values, including +-inf and NaN:
        check.is_true(
            bk.allclose(a, b, atol=atol, equal_nan=True),
            f"The values of `{name}` are not correct.",
        )


def check_ot_result(us, gt, atol=1e-3):

    # Check that the value is correct:
    check_approx_equal(us.value, gt.value, atol=atol, name="value")

    # Check that the "linear" value is correct:
    if gt.value_linear is not None:
        check_approx_equal(
            us.value_linear, gt.value_linear, atol=atol, name="value_linear"
        )

    # Check that the transport plans are correct:
    check_approx_equal(us.plan, gt.plan, atol=atol, name="plan")

    # Check that the two dual potentials are correct.
    if gt.potential_a is not None:  # (Only if we expect to return the dual potentials)
        # Since the pair of dual potentials is only defined up to an additive constant
        # for the Monge-Kantorovitch (= unregularized OT) problem,
        # we check that both vectors are equal when we remove the mean,
        # and that the sums of their means coincide.
        us_a, gt_a = us.potential_a, gt.potential_a
        us_b, gt_b = us.potential_b, gt.potential_b

        # To compute the means of the dual potentials, we rely on the shape of gt.value
        # to infer if we are working in batch mode.
        if len(gt.value.shape) == 0:
            # Reduce on all dimensions:
            red_dims = tuple(i for i in range(0, len(gt_a.shape)))
        elif len(gt.value.shape) == 1:
            # Reduce on all but the first dimension:
            red_dims = tuple(i for i in range(1, len(gt_a.shape)))
        else:
            assert False, "The field gt.value should be a scalar or a vector."

        mean_us_a = bk.mean(us_a, axis=red_dims)
        mean_gt_a = bk.mean(gt_a, axis=red_dims)
        mean_us_b = bk.mean(us_b, axis=red_dims)
        mean_gt_b = bk.mean(gt_b, axis=red_dims)

        check_approx_equal(
            mean_us_a + mean_us_b,
            mean_gt_a + mean_gt_b,
            atol=atol,
            name="sum(dual_potentials)",
        )
        check_approx_equal(
            us_a - mean_us_a, gt_a - mean_gt_a, atol=atol, name="potential_a"
        )
        check_approx_equal(
            us_b - mean_us_b, gt_b - mean_gt_b, atol=atol, name="potential_b"
        )

    # Check that the two marginals are correct:
    check_approx_equal(us.marginal_a, gt.marginal_a, atol=atol, name="marginal_a")
    check_approx_equal(us.marginal_b, gt.marginal_b, atol=atol, name="marginal_b")

    # Check that the barycentric mappings are correct:
    if gt.a_to_b is not None:
        check_approx_equal(us.a_to_b, gt.a_to_b, atol=atol, name="a_to_b")
        check_approx_equal(us.b_to_a, gt.b_to_a, atol=atol, name="b_to_a")


def check_ot_result_symmetric(a_to_b, b_to_a, *, transpose, atol=1e-4):
    """Checks that OT(a,b) = OT(b,a)."""
    # Values:
    check_approx_equal(a_to_b.value, b_to_a.value, atol=atol, name="value")

    if a_to_b.value_linear is not None:
        check_approx_equal(
            a_to_b.value_linear, b_to_a.value_linear, atol=atol, name="value_linear"
        )

    # Transport plan:
    check_approx_equal(a_to_b.plan, transpose(b_to_a.plan), atol=atol, name="plan")

    # Dual potentials:
    if a_to_b.potential_a is not None:
        check_approx_equal(
            a_to_b.potential_a, b_to_a.potential_b, atol=atol, name="potential_a"
        )
        check_approx_equal(
            a_to_b.potential_b, b_to_a.potential_a, atol=atol, name="potential_b"
        )

    # Two marginals:
    check_approx_equal(
        a_to_b.marginal_a, b_to_a.marginal_b, atol=atol, name="marginal_a"
    )
    check_approx_equal(
        a_to_b.marginal_b, b_to_a.marginal_a, atol=atol, name="marginal_b"
    )

    # Check that the barycentric mappings are correct:
    if a_to_b.a_to_b is not None:
        check_approx_equal(a_to_b.a_to_b, b_to_a.b_to_a, atol=atol, name="a_to_b")
        check_approx_equal(a_to_b.b_to_a, b_to_a.a_to_b, atol=atol, name="b_to_a")


def check_ot_result_cost_linearity(normal, scaled, *, scaling, offset, atol=1e-2):
    """Checks that OT_{scaling * C}(a,b) = scaling * OT(a,b) if scaling > 0."""

    # Values:
    check_approx_equal(
        scaling * normal.value + offset, scaled.value, atol=atol, name="value"
    )

    if normal.value_linear is not None:
        check_approx_equal(
            scaling * normal.value_linear + offset,
            scaled.value_linear,
            atol=atol,
            name="value_linear",
        )

    # Transport plan:
    check_approx_equal(normal.plan, scaled.plan, atol=atol, name="plan")

    # Check that the two dual potentials are correct.
    if False: # normal.potential_a is not None:  # (Only if we expect to return the dual potentials)
        # Since the pair of dual potentials is only defined up to an additive constant
        # for the Monge-Kantorovitch (= unregularized OT) problem,
        # we check that both vectors are equal when we remove the mean,
        # and that the sums of their means coincide - up to scaling and offset.
        normal_a, scaled_a = normal.potential_a, scaled.potential_a
        normal_b, scaled_b = normal.potential_b, scaled.potential_b

        # To compute the means of the dual potentials, we rely on the shape of normal.value
        # to infer if we are working in batch mode.
        if len(normal.value.shape) == 0:
            # Reduce on all dimensions:
            red_dims = tuple(i for i in range(0, len(normal_a.shape)))
        elif len(normal.value.shape) == 1:
            # Reduce on all but the first dimension:
            red_dims = tuple(i for i in range(1, len(normal_a.shape)))
        else:
            assert False, "The field normal.value should be a scalar or a vector."

        mean_normal_a = bk.mean(normal_a, axis=red_dims)
        mean_scaled_a = bk.mean(scaled_a, axis=red_dims)
        mean_normal_b = bk.mean(normal_b, axis=red_dims)
        mean_scaled_b = bk.mean(scaled_b, axis=red_dims)

        check_approx_equal(
            scaling * (mean_normal_a + mean_normal_b) + offset,
            mean_scaled_a + mean_scaled_b,
            atol=atol,
            name="sum(dual_potentials)",
        )
        check_approx_equal(
            scaling * (normal_a - mean_normal_a), scaled_a - mean_scaled_a, atol=atol, name="potential_a"
        )
        check_approx_equal(
            scaling * (normal_b - mean_normal_b), scaled_b - mean_scaled_b, atol=atol, name="potential_b"
        )

    # Two marginals:
    check_approx_equal(
        normal.marginal_a, scaled.marginal_a, atol=atol, name="marginal_a"
    )
    check_approx_equal(
        normal.marginal_b, scaled.marginal_b, atol=atol, name="marginal_b"
    )

    # Check that the barycentric mappings are correct:
    if normal.a_to_b is not None:
        check_approx_equal(normal.a_to_b, scaled.a_to_b, atol=atol, name="a_to_b")
        check_approx_equal(normal.b_to_a, scaled.b_to_a, atol=atol, name="b_to_a")
