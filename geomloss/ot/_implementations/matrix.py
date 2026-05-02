# Our generic backend, to use instead of NumPy/PyTorch/...
from ... import _backends as bk

# Typing annotations:
from ..._typing import RealTensor, CostMatrices

# Input converter:
from ..._input_validation import convert_inputs

# Abstract class for our results:
from .._ot_result import OTResult, LinearOperator

# Abstract solvers and annealing strategy:
from .._abstract_solvers import (
    sinkhorn_loop,
    # sinkhorn_barycenter_loop,
    annealing_parameters,
)

# Utility functions:
from ..._arguments import (
    ArrayProperties,
    check_library_dtype_device,
    check_regularization,
    check_marginal,
    check_marginal_masses,
)

# ========================================================================================
#                          High-level public interface
# ========================================================================================

# ----------------------------------------------------------------------------------------
#                                 Standard OT solver
# ----------------------------------------------------------------------------------------


@convert_inputs("C", "a", "b")
def solve(
    C,  # (N, M)
    *,
    a=None,  # (N,)
    b=None,  # (M,)
    # Regularization:
    reg=0,  # -> None by default
    reg_type="KL",
    # Unbalanced OT:
    unbalanced=None,  # None = +infty -> balanced by default;
    # We will also support scalar numbers, pairs of scalar numbers,
    # ((N,), (M,)) and ((B, N), (B, M)) point-dependent penalties.
    unbalanced_type="KL",
    # Optim parameters, following SciPy convention:
    method="auto",  # We can match keywords in this
    # string to activate some options such as
    # "symmetric", "annealing", etc.
    # Tolerance values.
    max_iter=None,
    tol=None,
):
    r"""

    Examples
    --------

    .. testcode::

        from geomloss import ot

        # Solve a balanced, 2x3 OT problem with entropic regularization:
        solution = ot.solve(
            C=[
                [0., 1., 4.],
                [2., 1., 0.]
            ],
            a=[2, 2],
            b=[1, 1, 2],
            reg=0.001,
            max_iter=100,
        )
        print(solution.plan)

    .. testoutput::

        [[1. 1. 0.]
         [0. 0. 2.]]

    .. testcode::

        print(f"{solution.value:.3f}")

    .. testoutput::

        0.997


    """
    if len(C.shape) != 2:
        raise ValueError(
            "The 'cost' matrix should be an array with 2 dimensions. "
            f"Instead, ot.solve received an array of shape {C.shape}."
        )

    N, M = C.shape

    a = check_marginal(a, ones_like=C[:, 0], marginal_size=N, name="a")
    b = check_marginal(b, ones_like=C[0, :], marginal_size=M, name="b")

    # We simply call the batch version of the solver, which will add a dummy batch dimension if needed.
    result = solve_batch(
        C[None, :, :],
        a=a[None, :],
        b=b[None, :],
        reg=reg,
        reg_type=reg_type,
        unbalanced=unbalanced,
        unbalanced_type=unbalanced_type,
        method=method,
        max_iter=max_iter,
        tol=tol,
    )
    # Since we know that there is no batch dimension, we can remove it from the result:
    result._squeeze_batchdim()
    return result


@convert_inputs("C", "a", "b")
def solve_batch(
    C,  # (B, N, M)  (B is the batch dimension)
    *,
    a=None,  # (B, N)
    b=None,  # (B, M)
    # Regularization:
    reg=0,  # -> None by default
    reg_type="KL",
    # Unbalanced OT:
    unbalanced=None,  # None = +infty -> balanced by default;
    # We will also support scalar numbers, pairs of scalar numbers,
    # ((N,), (M,)) and ((B, N), (B, M)) point-dependent penalties.
    unbalanced_type="KL",
    # Optim parameters, following SciPy convention:
    method="auto",  # We can match keywords in this
    # string to activate some options such as
    # "symmetric", "annealing", etc.
    # Tolerance values.
    max_iter=None,
    tol=None,
):
    # Basic checks on the solver parameters
    check_regularization(
        reg=reg,
        reg_type=reg_type,
        unbalanced=unbalanced,
        unbalanced_type=unbalanced_type,
        method=method,
        tol=tol,
        max_iter=max_iter,
    )

    # Check the input data ===============================================================

    # Cost matrix ------------------------------------------------------------------------
    # Check the shape:
    if len(C.shape) != 3:
        raise ValueError(
            "The 'cost' matrix should be an array with 3 dimensions (batch, N, M). "
            f"Instead, ot.solve received an array of shape {C.shape}."
        )

    # At this point, we know that C is a (B, N, M) array.
    B, N, M = C.shape

    # First marginal a -------------------------------------------------------------------

    a = check_marginal(a, ones_like=C[:, :, 0], marginal_size=N, name="a")
    b = check_marginal(b, ones_like=C[:, 0, :], marginal_size=M, name="b")

    # Add this point, we know that:
    # - a is a (B, N) array with >= 0 values.
    # - b is a (B, M) array with >= 0 values.

    # Check that the marginals have the same total mass ----------------------------------
    if unbalanced is None:  # if we work in balanced mode
        sums_a = bk.sum(a, axis=1)  # (B,)
        sums_b = bk.sum(b, axis=1)  # (B,)
        check_marginal_masses(sums_a, sums_b)

    # Low-level compatibility ------------------------------------------------------------
    library, dtype, device = check_library_dtype_device(a, b, C)

    array_properties = ArrayProperties(
        B=B,
        N=N,
        M=M,
        dtype=dtype,
        device=device,
        library=library,
    )

    # Actual computations ================================================================
    descent = annealing_parameters(
        maxmin_cost=bk.amax(C) - bk.amin(C),
        eps=reg,
        rho=unbalanced,
        n_iter=max_iter,
    )

    # N.B.: With a fixed cost matrix, there is no debiasing.
    potentials = sinkhorn_loop(
        softmin=softmin_dense,
        log_a_list=[bk.stable_log(a)],
        log_b_list=[bk.stable_log(b)],
        C_list=[
            CostMatrices(
                xy=C,
                yx=bk.ascontiguousarray(bk.transpose(C, (0, 2, 1))),
            )
        ],
        descent=descent,
        debias=False,
        last_extrapolation=True,
    )
    # solve exact OT by default
    # solve generic regularization ('enropic','l2','entropic+group lasso')
    # (reg_type can be a function)
    # default a and b are uniform (do they sum up to 1?)
    return OTResultMatrix(
        a=a,
        b=b,
        C=C,
        potentials=potentials,
        array_properties=array_properties,
        reg=reg,
        reg_type=reg_type,
        unbalanced=unbalanced,
        unbalanced_type=unbalanced_type,
    )


class OTResultMatrix(OTResult):
    def __init__(
        self,
        *,
        a,
        b,
        C,
        potentials,
        array_properties,
        reg,
        reg_type,
        unbalanced,
        unbalanced_type,
    ):
        super().__init__(
            a=a,
            b=b,
            C=C,
            potentials=potentials,
            array_properties=array_properties,
            batchsize=array_properties.B,
            reg=reg,
            reg_type=reg_type,
            unbalanced=unbalanced,
            unbalanced_type=unbalanced_type,
            debias=False,
        )

        # Fill the dictionary of "expected shapes", that will be used to format the
        # result as expected by the user:
        ap = self._array_properties

        # Under the hood, we always work with batch dimensions, even if the user did not provide one.
        self._shapes = {
            "a": (ap.B, ap.N),
            "b": (ap.B, ap.M),
            "C": (ap.B, ap.N, ap.M),
            "B": (ap.B,),
        }

    def _squeeze_batchdim(self):
        """Removes the batch dimension, assuming that it is a dummy one."""
        ap = self._array_properties
        assert ap.B == 1
        assert self._batchsize == 1
        self._batchsize = 0

        self._shapes = {
            "a": (ap.N,),
            "b": (ap.M,),
            "C": (ap.N, ap.M),
            "B": (),
        }

    def _density(self):
        # Load the relevant quantities:
        f = self._potentials.f_ba  # (B, N)
        g = self._potentials.g_ab  # (B, M)
        C = self._C  # (B, N, M)
        eps = self._reg  # float, > 0

        # Make sure that everyone has the expected shape:
        ap = self._array_properties
        B, N, M = ap.B, ap.N, ap.M

        assert f.shape == (B, N)
        assert g.shape == (B, M)
        assert C.shape == (B, N, M)
        assert eps > 0

        # Compute the main term in the expression of the optimal plan:
        D_ij = bk.exp((f[:, :, None] + g[:, None, :] - C) / eps)  # (B,N,M)
        return self.cast(D_ij, "C")  # Cast as a (N,M) or (B,N,M) Tensor

    def _density_operator(self):
        return LinearOperator.from_dense(
            self.density,
            input_shape=self._shapes["b"],
            output_shape=self._shapes["a"],
        )

    def _plan(self):
        # Load the relevant quantities:
        a = self._a  # (B, N)
        b = self._b  # (B, M)
        dens = self.density  # (N, M) or (B, N, M)

        # Make sure that everyone has the expected shape:
        ap = self._array_properties
        B, N, M = ap.B, ap.N, ap.M

        if self._batchsize == 0:
            assert dens.shape == (N, M)
            assert B == 1
            # Add a dummy batch dimension
            dens = bk.view(dens, (B, N, M))

        assert a.shape == (B, N)
        assert b.shape == (B, M)
        assert dens.shape == (B, N, M)

        # Actual computation:
        if self._reg_type == "KL":
            # Multiply by the reference product measure:
            plan = a[:, :, None] * b[:, None, :] * dens  # (B,N,1) * (B,1,M) * (B,N,M)
        else:
            raise NotImplementedError(
                "Currently, we only support the computation "
                "of transport plans when `reg_type = 'KL'`."
            )

        return self.cast(plan, "C")  # Cast as a (N,M) or (B,N,M) Tensor


# ----------------------------------------------------------------------------------------
#                              Wasserstein barycenters
# ----------------------------------------------------------------------------------------


# Convention:
# - B is the batch dimension
# - K is the number of measures per barycenter
# - N is the number of samples "for the data"
# - M is the number of samples "for the barycenter"
def barycenter(
    cost,  # (N, M) or (K, N, M) or (B, K, N, M)
    a=None,  # (N,) or (K, N) or (B, K, N)
    weights=None,  # (K,) or (B, K)
    # + all the standard parameters for ot.solve
):
    # masses will be a (M,) or (B, M) array of weights
    return OTResult(potentials=potentials, masses=masses)


# ========================================================================================
#                                 Low-level routines
# ========================================================================================


def softmin_dense(
    eps: float,
    log_weights: RealTensor,
    costs: RealTensor,
    potentials: RealTensor,
) -> RealTensor:
    """Softmin function implemented on dense arrays, without using KeOps.

    The softmin function is at the heart of any (stable) implementation
    of the Sinkhorn algorithm. It takes as input:
    - a temperature eps(ilon),
    - log_weights lb[j] = log(b(y[j])) of shape (B,M),
    - a cost matrix C_xy[i,j] = C(x[i],y[j]) of shape (B, N, M),
    - a weighted dual potential G_y[j] = g_ab(y[j]) of shape (B, M).

    It returns a new dual potential supported on the points x[i]:
    f_x[i] = - eps * log(sum_j(exp[ lb[j] + (G_y[j]  -  C_xy[i,j]) / eps ] ))

    In the Sinkhorn loop, we typically use calls like:
        ft_ba = softmin(eps, b_log, C_xy, g_ab)

    Args:
        eps (float >= 0): Temperature eps(ilon), the main regularization parameter
            of the Sinkhorn algorithm.
        log_weights ((B,M) real-valued Tensor): Batch of B vectors of shape (M,) containing
            the logarithm of the weights of the measure b.
        costs ((B,N,M) real-valued Tensor): Batch of B cost matrices of shape (N,M).
        potentials ((B,M) real-valued Tensor): Batch of B vectors of shape (M,).

    Returns:
        (B,N) real-valued Tensor:
    """
    log_b_y = log_weights
    C_xy = costs
    g_y = potentials

    assert eps >= 0, "We only support non-negative temperatures (eps >= 0)."
    assert len(C_xy.shape) == 3, "C_xy should be a (B,N,M) Tensor."
    B, N, M = C_xy.shape

    assert g_y.shape == (B, M), "g_y should be a (B,M) Tensor."
    assert log_b_y.shape == (B, M), "log_b_y should be a (B,M) Tensor."

    if eps == float("inf"):
        # TODO: handle the case where b is not a probability measure
        # Currently, we're "missing" the -eps * log(b_y.sum()) term.
        b_y = bk.exp(log_b_y)  # (B,M)
        sum_b = b_y.sum(axis=1, keepdims=True)  # (B,1)
        f_i = ((C_xy - bk.view(g_y, (B, 1, M))) * bk.view(b_y, (B, 1, M))).sum(
            axis=2
        )  # (B,N)
        return f_i / sum_b

    elif eps == 0:
        # TODO: handle the case where some of the b_y are zero
        f_i = bk.amin(C_xy - bk.view(g_y, (B, 1, M)), axis=2)  # (B,N)
        return f_i

    else:
        scores_xy = bk.view(log_b_y + g_y / eps, (B, 1, M)) - C_xy / eps  # (B,N,M)
        return -eps * bk.logsumexp(scores_xy, axis=2)
