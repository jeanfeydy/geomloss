# Our generic backend, to use instead of NumPy/PyTorch/...
from ... import _backends as bk

# Typing annotations:
from ..._typing import RealTensor, CostMatrices, UnbalancedType, SupportedMethods

# Input converter:
from ..._input_validation import convert_inputs

# Abstract class for our results:
from .._ot_result import OTResult, LinearOperator

# Support for caching results without messing up the documentation:
from ..._cache import add_cached_methods_to_sphinx

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


# ========================================================================================
#                          High-level public interface
# ========================================================================================


# ----------------------------------------------------------------------------------------
#                                Return type
# ----------------------------------------------------------------------------------------


@add_cached_methods_to_sphinx
class OTResultMatrix(OTResult):
    """Stores the result of an optimal transport problem computed using an explicit cost matrix.

    Users may access the optimal transport plan, the dual potentials and other related
    quantities as attributes of this class. Under the hood, these attributes are computed
    lazily and cached for efficiency. In the documentation below,
    $B$ denotes the batch size while $N$ and $M$ denote the number of samples in the
    source and target measures, respectively.
    """

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

    _cached_properties = (
        "potential_a",
        "potential_b",
        "density",
        "lazy_density",
        "density_operator",
        "plan",
        "lazy_plan",
        "plan_operator",
        "value",
        "marginal_a",
        "marginal_b",
        "citation",
    )

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
        r"""Density $P_{ij}$ of the transport plan with respect to the reference product measure $\alpha \otimes \beta$.

        Using the notations of the documentation of :func:`~geomloss.ot.solve`, we have:

        .. math::

            P_{ij} = \exp\left( \frac{f_i + g_j - C_{ij}}{\varepsilon} \right)~,

        where $f$ and $g$ are the dual potentials of the optimal transport problem, $C$ is the cost matrix and $\varepsilon$ is the regularization strength.

        Returns
        -------
        (N,M) or (B,N,M) array-like >= 0
            If this solution was computed with :func:`~geomloss.ot.solve`, the density is returned as a :mod:`(N,M)` array.
            If this solution was computed with :func:`~geomloss.ot.solve_batch`, the density is returned as a :mod:`(B,N,M)` array, where B is the batch size.
        """
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
        r"""Linear operator associated to :attr:`density`.

        Returns
        -------
        LinearOperator
            If this solution was computed with :func:`~geomloss.ot.solve`, the returned operator can be applied to
            arrays of shape :mod:`(M,...)` to produce arrays of shape :mod:`(N,...)`.

            If this solution was computed with :func:`~geomloss.ot.solve_batch`, the returned operator can be applied to
            arrays of shape :mod:`(B,M,...)` to produce arrays of shape :mod:`(B,N,...)`.

        """
        return LinearOperator.from_dense(
            self.density,
            input_shape=self._shapes["b"],
            output_shape=self._shapes["a"],
        )

    def _plan(self):
        r"""Optimal transport plan $\pi_{ij}$.

        Using the notations of the documentation of :func:`~geomloss.ot.solve`, we have:

        .. math::

            \pi_{ij} = \alpha_i \beta_j \cdot \exp\left( \frac{f_i + g_j - C_{ij}}{\varepsilon} \right)~,

        where $f$ and $g$ are the dual potentials of the optimal transport problem, $C$ is the cost matrix and $\varepsilon$ is the regularization strength.

        Returns
        -------
        (N,M) or (B,N,M) array-like >= 0
            If this solution was computed with :func:`~geomloss.ot.solve`, the plan is returned as a :mod:`(N,M)` array.
            If this solution was computed with :func:`~geomloss.ot.solve_batch`, the plan is returned as a :mod:`(B,N,M)` array, where B is the batch size.
        """

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

    def _plan_operator(self):
        r"""Linear operator associated to :attr:`plan`.

        Example
        -------

        .. testcode::

            import numpy as np
            from geomloss import ot

            # Solve a balanced, 2x3 OT problem with entropic regularization:
            solution = ot.solve(
                C=[[0., 1., 4.],
                   [2., 1., 0.]],
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

            print(solution.plan_operator @ np.array([-2., 1., 1.]))

        .. testoutput::

            [-1.  2.]


        Returns
        -------
        LinearOperator
            If this solution was computed with :func:`~geomloss.ot.solve`, the returned operator can be applied to
            arrays of shape :mod:`(M,...)` to produce arrays of shape :mod:`(N,...)`.

            If this solution was computed with :func:`~geomloss.ot.solve_batch`, the returned operator can be applied to
            arrays of shape :mod:`(B,M,...)` to produce arrays of shape :mod:`(B,N,...)`.

        """
        return super()._plan_operator()


# ----------------------------------------------------------------------------------------
#                                 Standard OT solver
# ----------------------------------------------------------------------------------------


@convert_inputs("C", "a", "b")
def solve(
    C,  # (N, M)
    *,
    reg,  # float > 0
    a=None,  # (N,)
    b=None,  # (M,)
    # Unbalanced OT:
    unbalanced=None,  # None = +infty -> balanced by default;
    # We will also support scalar numbers, pairs of scalar numbers,
    # ((N,), (M,)) and ((B, N), (B, M)) point-dependent penalties.
    unbalanced_type: UnbalancedType = "KL",
    # Optim parameters, following SciPy convention:
    method: SupportedMethods = "auto",  # We can match keywords in this
    # string to activate some options such as
    # "symmetric", "annealing", etc.
    # Tolerance values.
    max_iter=None,
    tol=None,
) -> OTResultMatrix:
    r"""Solves an optimal transport problem with an explicit cost matrix.

    .. warning::
        The interface of this solver is still in development and may change in future releases.
        We welcome any feedback and suggestions for improvement.


    We focus on **entropy-regularized OT** and support both the **balanced** setting
    (with strict transport constraints) and the **unbalanced** setting
    (with softer constraints on the marginals of the transport plan).

    Given a source measure $\alpha$ sampled with $N$ points, 
    a target measure $\beta$ sampled with $M$ points
    and a $N$-by-$M$ cost matrix $C$, the **primal formulation** of the problem 
    $\text{OT}_{\varepsilon, \rho}(\alpha, \beta)$
    reads:

    .. math::

        \min_{\pi \in \mathbb{R}_+^{N \times M}} \sum_{i=1}^N \sum_{j=1}^M C_{ij} \pi_{ij}
        + \varepsilon \text{KL}(\pi, \alpha \otimes \beta)
        + \text{D}_\rho(\pi 1_M, \alpha)
        + \text{D}_\rho(\pi^\top 1_N, \beta)~,

    where:

    - $\pi$ is the **transport plan**, a non-negative matrix of shape $(N,M)$.
    - $\varepsilon > 0$ is the **regularization strength** or **temperature**.
    - $\text{KL}(\pi, \alpha \otimes \beta)$ is the **relative entropy** of $\pi$ with respect to the reference product measure $\alpha \otimes \beta$:
    
    .. math::

        \text{KL}(\pi, \alpha \otimes \beta) = \sum_{i=1}^N \sum_{j=1}^M \pi_{ij} \log\left( \frac{\pi_{ij}}{\alpha_i \beta_j} \right) - \pi_{ij} + \alpha_i \beta_j~.

    - $\text{D}_\rho$ is a penalty on the **marginal constraints**, weighted by a positive parameter $\rho$.
      
      - If $\rho = +\infty$, we are in the **balanced** setting and
        $\text{D}_{\rho}$ encodes a hard constraint on the marginals of $\pi$:
        $\pi 1_M = \alpha$ and $\pi^\top 1_N = \beta$.
      
      - If $\rho < +\infty$, we are in the **unbalanced** setting and $\text{D}_\rho$ encodes a softer penalty on the marginal constraints,
        which allows for partial transport of the source measure $\alpha$ onto the target $\beta$.
        Currently, we only support the Kullback-Leibler divergence as unbalanced OT penalty, in which case we have:

    .. math::

        \text{D}_\rho(\pi 1_M, \alpha) = \rho \cdot \text{KL}(\pi 1_M, \alpha)~,~ \\
        \text{D}_\rho(\pi^\top 1_N, \beta) = \rho \cdot \text{KL}(\pi^\top 1_N, \beta)~.

    Parameters
    ----------
    C:  (N,M) array-like
        **Cost** matrix: $C_{ij}$ denotes the cost of transporting a unit of mass from
        point $x_i$ in the source measure $\alpha$ to point $y_j$ in the target measure $\beta$.
    reg: float > 0
        Regularization strength or **temperature** $\varepsilon$.
        It should be understood as a fuzziness parameter that is applied to the cost matrix:
        $\varepsilon = \max_{i,j} C_{ij} - \min_{i,j} C_{ij}$ corresponds to a very fuzzy problem, and leads to an optimal transport plan that is close to the product measure $\alpha \otimes \beta$. On the other hand, $\varepsilon \rightarrow 0$ corresponds to the original cost matrix, without any fuzziness, and leads to a sparse optimal transport plan. In practice, we recommend using a value of $\varepsilon$ that is small compared to the scale of the cost matrix, but not too small to avoid numerical issues. A good default value is $\varepsilon = 0.001 \cdot (\max_{i,j} C_{ij} - \min_{i,j} C_{ij})$.
    a:  (N,) array-like >= 0, default: None
        **First marginal**: $\alpha_i$ denotes the mass associated to point $x_i$ in the source measure $\alpha$.
        If not provided, we assume that $a$ is the uniform distribution on $N$ points,
        i.e. $\alpha_i = 1/N$ for all $i$.
    b:  (M,) array-like >= 0, default: None
        **Second marginal**: $\beta_j$ denotes the mass associated to point $y_j$ in the target measure $\beta$.
        If not provided, we assume that $b$ is the uniform distribution on $M$ points,
        i.e. $\beta_j = 1/M$ for all $j$.
    unbalanced: float > 0 or None, default: None
        Scaling factor $\rho$ for the **unbalanced OT penalty** on the marginal constraints.
        If unbalanced is :mod:`None`, we solve the balanced OT problem
        and look for a full transport of the source marginal $\alpha$ onto the target $\beta$.
        If unbalanced is a positive :mod:`float` $\rho > 0$, we solve an unbalanced OT problem where the marginal constraints are relaxed and penalized with a soft penalty parameterized by $\rho$.
    unbalanced_type:
        Type of unbalanced OT penalty to use when unbalanced is not :mod:`None`.
        Currently, we only support the Kullback-Leibler divergence ``"KL"``, also known as the relative entropy.
    method:
        Optimization method to use. Currently, we only support the symmetrized Sinkhorn algorithm
        with annealing, which is activated by the keyword ``"auto"``.
    max_iter: int > 0 or None, default: None
        Maximum **number of iterations** for the optimization algorithm.
        If :mod:`None`, the algorithm runs until convergence.
    tol: float > 0 or None, default: None
        **Tolerance** for convergence.
        If :mod:`None`, we use a default tolerance based on the
        regularization strength $\varepsilon$ and the scale of the cost matrix $C$.

    Returns
    -------
    OTResultMatrix
        An object containing the optimal transport plan, the dual potentials
        and other related quantities as attributes.

    Examples
    --------

    .. testcode::

        from geomloss import ot

        # Solve a balanced, 2x3 OT problem with entropic regularization:
        solution = ot.solve(
            C=[[0., 1., 4.],
               [2., 1., 0.]],
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
    reg,  # float > 0
    a=None,  # (B, N)
    b=None,  # (B, M)
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
    r"""Batched version of :func:`~geomloss.ot.solve`, which can solve multiple OT problems in parallel.

    .. warning::
        The interface of this solver is still in development and may change in future releases.
        We welcome any feedback and suggestions for improvement.

    Parameters
    ----------

    C:  (B,N,M) array-like
        Batch of $B$ **cost matrices**: $C_{ij}^{(b)}$ denotes the cost
        of transporting a unit of mass from point $x_i$ in the source measure $\alpha^{(b)}$ to point $y_j$ in the target measure $\beta^{(b)}$ for the $b$-th OT problem in the batch.
    a:  (B,N) array-like >= 0, default: None
        Batch of $B$ **first marginals**: $\alpha_i^{(b)}$ denotes the mass associated to point $x_i$ in the source measure $\alpha^{(b)}$ for the $b$-th OT problem in the batch.
        If not provided, we assume that $a$ is the uniform distribution on $N$ points for each problem in the batch,
        i.e. $\alpha_i^{(b)} = 1/N$ for all $i$ and $b$.
    b:  (B,M) array-like >= 0, default: None
        Batch of $B$ **second marginals**: $\beta_j^{(b)}$ denotes the mass associated to point $y_j$ in the target measure $\beta^{(b)}$ for the $b$-th OT problem in the batch.
        If not provided, we assume that $b$ is the uniform distribution on $M$ points for each problem in the batch,
        i.e. $\beta_j^{(b)} = 1/M$ for all $j$ and $b$.


    **Other parameters** are identical to those of :func:`~geomloss.ot.solve`.
    They are applied uniformly to all problems in the batch.

    Returns
    -------
    OTResultMatrix
        An object containing the optimal transport plans, the dual potentials
        and other related quantities for each problem in the batch as attributes.

    Examples
    --------

    .. testcode::

        from geomloss import ot
        # Solve a batch of 2 balanced, 2x3 OT problems with entropic regularization:
        solution = ot.solve_batch(
            C=[[[0., 1., 4.],
                [2., 1., 0.]],
               [[0., 2., 3.],
                [1., 0., 1.]]],
            a=[[2, 2],
               [1, 3]],
            b=[[1, 1, 2],
               [2, 1, 1]],
            reg=0.001,
            max_iter=1000,
        )
        print(solution.plan)

    .. testoutput::

        [[[1.    1.    0.   ]
          [0.    0.    2.   ]]

         [[1.005 0.    0.   ]
          [0.988 1.003 1.003]]]

    .. testcode::

        print(f"{solution.value}")

    .. testoutput::

        [0.997 1.995]

    """

    # Basic checks on the solver parameters
    check_regularization(
        reg=reg,
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
        reg_type="KL",
        unbalanced=unbalanced,
        unbalanced_type=unbalanced_type,
    )


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
