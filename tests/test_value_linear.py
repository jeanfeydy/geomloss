import numpy as np
import pytest

import pytest_check as check

from geomloss import ot
from geomloss import _backends as bk

# ========================================================================================
#                      Focused checks for OTResult.value_linear
# ========================================================================================
#
# The tests below complement the property-based tests of test_ot_solve_matrix.py and
# test_ot_solve_sample.py, which check value_linear against closed-form or SciPy
# ground truths. Here, we focus on the aspects that are specific to this quantity:
# its exact mathematical definition, the fact that it is computed lazily, its
# interaction with the caching mechanism, and its behaviour in the unbalanced
# and debiased settings.


def sqdists(X_a, X_b):
    """Returns the matrix of squared Euclidean distances, computed independently of GeomLoss."""
    return np.sum((X_a[:, None, :] - X_b[None, :, :]) ** 2, axis=-1)


def test_value_linear_matrix():
    """Checks that value_linear is the linear cost of the transport plan, on a hand-written example.

    The transport plan of this problem is known explicitly, so that the expected
    value of the linear cost can be written down by hand.
    """

    C = [[0.0, 1.0, 4.0], [2.0, 1.0, 0.0]]
    solution = ot.solve(C=C, a=[2, 2], b=[1, 1, 2], reg=0.001, max_iter=100)

    # With such a low regularization, the optimal plan is the (2,3) matrix
    # [[1, 1, 0], [0, 0, 2]], whose linear cost is:
    #     1 * C[0,0] + 1 * C[0,1] + 2 * C[1,2] = 0 + 1 + 0 = 1.
    check_plan = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    check.is_true(
        bk.allclose(solution.plan, check_plan, atol=1e-3),
        "The transport plan of the example is not the expected one.",
    )
    check.is_true(
        bk.allclose(solution.value_linear, 1.0, atol=1e-3),
        f"Expected value_linear = 1.0, but found {solution.value_linear}.",
    )

    # A direct, explicit computation of sum_ij plan_ij * C_ij must coincide:
    expected = np.sum(np.asarray(solution.plan) * np.asarray(C))
    check.is_true(
        bk.allclose(solution.value_linear, expected, atol=1e-10),
        "value_linear does not match the explicit sum of plan * C.",
    )


def test_value_linear_matrix_batch():
    """Checks that value_linear is returned with the expected layout, in batch mode.

    N.B.: We do not compare the batched solver with a sequence of single-problem
    solves. ot.solve_batch() derives its annealing parameters from the cost matrices
    of the whole batch, whereas ot.solve() only sees one problem: the two results are
    legitimately different, and this is not related to value_linear.
    """

    C = np.array(
        [
            [[0.0, 1.0, 4.0], [2.0, 1.0, 0.0]],
            [[0.0, 2.0, 3.0], [1.0, 0.0, 1.0]],
        ]
    )
    a = np.array([[2.0, 2.0], [1.0, 3.0]])
    b = np.array([[1.0, 1.0, 2.0], [2.0, 1.0, 1.0]])

    batched = ot.solve_batch(C, a=a, b=b, reg=0.001, max_iter=1000)

    check.equal(batched.value_linear.shape, (2,))
    check.equal(batched.value_linear.dtype, batched.value.dtype)

    for k in range(2):
        # The linear value must be the linear cost of the plan that is returned:
        check.is_true(
            bk.allclose(
                batched.value_linear[k], np.sum(batched.plan[k] * C[k]), atol=1e-8
            ),
            f"value_linear does not match the explicit linear cost of the plan (k={k}).",
        )

    # A single-problem solve must return a scalar, not a (1,) array:
    single = ot.solve(C[0], a=a[0], b=b[0], reg=0.001, max_iter=1000)
    check.equal(single.value_linear.shape, ())
    check.is_true(
        bk.allclose(single.value_linear, np.sum(single.plan * C[0]), atol=1e-8)
    )


def test_value_linear_sample():
    """Checks value_linear for the sample-based solver, in the balanced and unbalanced settings."""

    rng = np.random.RandomState(0)
    X_a = rng.randn(7, 2)
    X_b = rng.randn(5, 2)
    a = rng.rand(7) + 0.1
    b = rng.rand(5) + 0.1
    a, b = a / a.sum(), b / b.sum()
    C_xy = sqdists(X_a, X_b)

    balanced = ot.solve_sample(X_a, X_b, a=a, b=b, reg=0.1, max_iter=1000)
    check.is_true(
        bk.allclose(balanced.value_linear, np.sum(balanced.plan * C_xy), atol=1e-8),
        "value_linear does not match sum(plan * C) for the balanced sample solver.",
    )

    # In the unbalanced setting, the total mass of the plan is not constrained:
    # value_linear must still be the linear cost of the plan that is returned.
    unbalanced = ot.solve_sample(
        X_a, X_b, a=a, b=b, reg=0.1, unbalanced=0.5, max_iter=1000
    )
    check.is_true(
        bk.allclose(unbalanced.value_linear, np.sum(unbalanced.plan * C_xy), atol=1e-8),
        "value_linear does not match sum(plan * C) for the unbalanced sample solver.",
    )
    # Make sure that we really are in the unbalanced regime:
    check.is_true(
        not np.isclose(np.sum(unbalanced.plan), np.sum(a), atol=1e-3),
        "The unbalanced test case is degenerate: the plan transports all the mass.",
    )


def test_value_linear_sample_debiased():
    """Checks value_linear for debiased solvers, on a symmetric configuration with a known answer.

    When the source and the target measures are identical, the three optimal transport
    problems involved in the debiased Sinkhorn divergence (a <-> b, a <-> a and b <-> b)
    coincide, so that the linear part of the divergence must vanish exactly.
    """

    rng = np.random.RandomState(1)
    X = rng.randn(8, 2)
    a = rng.rand(8) + 0.1
    a = a / a.sum()

    solution = ot.solve_sample(X, X, a=a, b=a, reg=0.1, debias=True, max_iter=1000)

    # The three transport plans are the same, and so are the three cost matrices:
    C_xy = sqdists(X, X)
    check.is_true(
        bk.allclose(solution.value_linear, 0.0, atol=1e-10),
        "The linear part of the Sinkhorn divergence between identical measures "
        f"should vanish, but found {solution.value_linear}.",
    )

    # Sanity check on the non-debiased quantity, for the very same problem:
    plain = ot.solve_sample(X, X, a=a, b=a, reg=0.1, max_iter=1000)
    check.is_true(bk.allclose(plain.value_linear, np.sum(plain.plan * C_xy), atol=1e-8))

    # N.B.: The general formula <pi_ab, C_xy> - 0.5 <pi_aa, C_xx> - 0.5 <pi_bb, C_yy>
    # is checked in test_value_linear_sample_debiased_nontrivial() below, on a problem
    # where the three transport plans are distinct and the expected value is not zero.


def test_value_linear_sample_debiased_nontrivial():
    """Checks the debiased value_linear on distinct measures with distinct weights.

    Unlike the symmetric test above, the three optimal transport problems involved here
    are genuinely different, and the expected value is not zero. It is built from three
    *independent* solves, performed with the matrix solver on explicitly constructed cost
    matrices: one for the cross problem (a <-> b) and one for each self-interaction
    problem (a <-> a and b <-> b). The definition

        <pi_ab, C_xy> - 0.5 <pi_aa, C_xx> - 0.5 <pi_bb, C_yy>

    is then written out explicitly with plain sums of `plan * C`. No object's
    `value_linear` is read while building the expectation, and the formula is not
    factored out into a shared helper, so that dropping a term, flipping a sign or
    forgetting a factor 1/2 all make this test fail: each of those mistakes changes the
    expected value by at least 7e-2, while the implementation agrees with it to ~1e-5.
    """

    X_a = np.array([[0.0, 0.0], [1.0, 0.2], [0.1, 1.4], [1.6, 1.1]])
    X_b = np.array([[0.3, 0.9], [1.4, 0.4], [0.8, 1.7]])
    a = np.array([0.4, 0.3, 0.2, 0.1])  # sums up to 1
    b = np.array([0.5, 0.3, 0.2])  # sums up to 1, and differs from a
    reg = 0.5
    max_iter = 5000

    C_xy = sqdists(X_a, X_b)
    C_xx = sqdists(X_a, X_a)
    C_yy = sqdists(X_b, X_b)

    # The three transport plans, each obtained from its own balanced OT problem:
    pi_ab = ot.solve(C_xy, a=a, b=b, reg=reg, max_iter=max_iter).plan
    pi_aa = ot.solve(C_xx, a=a, b=a, reg=reg, max_iter=max_iter).plan
    pi_bb = ot.solve(C_yy, a=b, b=b, reg=reg, max_iter=max_iter).plan

    cross = np.sum(pi_ab * C_xy)
    self_a = 0.5 * np.sum(pi_aa * C_xx)
    self_b = 0.5 * np.sum(pi_bb * C_yy)
    expected = cross - self_a - self_b

    # The two self-interaction terms must be large enough for this test to be able to
    # detect a missing term or a wrong sign:
    check.is_true(
        self_a > 1e-2, f"The self-interaction term for a is too small: {self_a}."
    )
    check.is_true(
        self_b > 1e-2, f"The self-interaction term for b is too small: {self_b}."
    )
    check.is_true(
        abs(expected) > 1e-2, f"The expected value is too close to zero: {expected}."
    )

    solution = ot.solve_sample(
        X_a, X_b, a=a, b=b, reg=reg, debias=True, max_iter=max_iter
    )

    # The cross plan of the debiased solver must be the one that we solved for above,
    # so that the expectation really is the one that the implementation should produce:
    check.is_true(
        bk.allclose(solution.plan, pi_ab, atol=1e-3),
        "The debiased solver did not return the expected cross transport plan.",
    )
    check.is_true(
        bk.allclose(solution.value_linear, expected, atol=1e-3),
        f"Expected value_linear = {expected}, but found {solution.value_linear}.",
    )


def test_value_linear_is_computed_lazily():
    """Checks that value_linear is only computed when it is requested."""

    solution = ot.solve(
        C=[[0.0, 1.0, 4.0], [2.0, 1.0, 0.0]],
        a=[2, 2],
        b=[1, 1, 2],
        reg=0.001,
        max_iter=100,
    )

    # Nothing should have been computed yet:
    check.is_false(
        hasattr(solution, "_cached_value_linear"),
        "value_linear should not be computed before it is requested.",
    )
    check.is_false(
        hasattr(solution, "_cached_value"),
        "value should not be computed before it is requested.",
    )

    value_linear = solution.value_linear
    check.is_true(hasattr(solution, "_cached_value_linear"))
    # Computing value_linear must not trigger the computation of value:
    check.is_false(
        hasattr(solution, "_cached_value"),
        "Computing value_linear should not compute value as a side effect.",
    )

    # The result must be memoized:
    check.is_true(solution.value_linear is value_linear)


def test_value_linear_cache_clear():
    """Checks that value_linear is recomputed, with the same result, after a cache reset."""

    solution = ot.solve(
        C=[[0.0, 1.0, 4.0], [2.0, 1.0, 0.0]],
        a=[2, 2],
        b=[1, 1, 2],
        reg=0.001,
        max_iter=100,
    )
    value_linear = solution.value_linear

    solution.cache_clear()
    check.is_false(hasattr(solution, "_cached_value_linear"))

    check.is_true(
        bk.allclose(solution.value_linear, value_linear, atol=0.0, rtol=0.0),
        "value_linear is not reproducible after a cache reset.",
    )


def test_value_linear_dtype_and_shape():
    """Checks that value_linear has the same shape and dtype as value."""

    for dtype in [np.float32, np.float64]:
        solution = ot.solve(
            C=np.array([[0.0, 1.0, 4.0], [2.0, 1.0, 0.0]], dtype=dtype),
            a=np.array([2, 2], dtype=dtype),
            b=np.array([1, 1, 2], dtype=dtype),
            reg=0.001,
            max_iter=100,
        )
        check.equal(solution.value_linear.shape, solution.value.shape)
        check.equal(solution.value_linear.dtype, solution.value.dtype)

    # Torch backend:
    torch = pytest.importorskip("torch")

    solution = ot.solve(
        C=torch.tensor([[0.0, 1.0, 4.0], [2.0, 1.0, 0.0]], dtype=torch.float32),
        a=torch.tensor([2.0, 2.0]),
        b=torch.tensor([1.0, 1.0, 2.0]),
        reg=0.001,
        max_iter=100,
    )
    check.equal(solution.value_linear.shape, solution.value.shape)
    check.equal(solution.value_linear.dtype, solution.value.dtype)
    # N.B.: bk.allclose maps to torch.allclose() with torch inputs, which does not
    # accept a Python scalar as its second argument: we compare floats instead.
    check.is_true(
        abs(float(solution.value_linear) - 1.0) < 1e-3,
        f"Expected value_linear = 1.0, but found {solution.value_linear}.",
    )


def test_value_linear_not_implemented():
    """Checks that the abstract OTResult class does not silently return a wrong value."""

    result = ot.OTResult(
        a=None,
        b=None,
        potentials=None,
        array_properties=None,
        batchsize=0,
        reg=1.0,
        reg_type="KL",
        unbalanced=None,
        unbalanced_type="KL",
        debias=False,
    )
    with pytest.raises(NotImplementedError):
        result.value_linear


@pytest.mark.skipif(
    not bk.keops_available,
    reason="KeOps is not available: the symbolic (LazyTensor) code path cannot be exercised.",
)
def test_value_linear_with_keops():
    """Checks that the symbolic KeOps code path is consistent with an explicit computation.

    N.B.: With KeOps, accessing value_linear must not materialize the dense cost
    matrix, nor the dense transport plan.
    """

    rng = np.random.RandomState(2)
    X_a = rng.randn(9, 3)
    X_b = rng.randn(6, 3)
    a = rng.rand(9) + 0.1
    b = rng.rand(6) + 0.1
    a, b = a / a.sum(), b / b.sum()

    solution = ot.solve_sample(X_a, X_b, a=a, b=b, reg=0.1, max_iter=1000)

    # Make sure that we did go through the KeOps code path:
    check.is_true(solution._C_lazy is not None, "KeOps was not used for this problem.")

    value_linear = solution.value_linear

    # The result must have the expected layout, and be finite:
    check.equal(np.asarray(value_linear).shape, ())
    check.is_true(np.all(np.isfinite(np.asarray(value_linear))))

    # Accessing value_linear must not have materialized the dense cost matrix:
    check.is_true(
        solution._C_dense is None,
        "Accessing value_linear materialized the dense cost matrix.",
    )

    # The value must coincide with an explicit, independent computation:
    # N.B.: The two computations are performed in different orders and with
    # different accumulators, so we allow for a relative tolerance.
    C_xy = sqdists(X_a, X_b)
    expected = np.sum(np.asarray(solution.plan, dtype=np.float64) * C_xy)
    check.is_true(
        np.allclose(
            np.asarray(value_linear, dtype=np.float64), expected, rtol=1e-4, atol=1e-8
        ),
        "The symbolic value_linear does not match the explicit sum of plan * C: "
        f"found {value_linear}, expected {expected}.",
    )
