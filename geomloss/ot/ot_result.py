from .. import backends as bk
from ..cache import add_cached_methods_to_sphinx, cache_methods_and_properties
from .abstract_solvers.unbalanced_ot import sinkhorn_cost
import math


class LinearOperator:
    """Linear operator that can be applied to vectors, without being explicitly instantiated as a matrix."""

    def __init__(self, *, matmat, rmatmat, input_shape, output_shape):
        self._matmat = matmat
        self._rmatmat = rmatmat
        self._input_shape = input_shape
        self._output_shape = output_shape

    def __matmul__(self, x):
        if (
            len(x.shape) < len(self._input_shape)
            or x.shape[: len(self._input_shape)] != self._input_shape
        ):
            raise ValueError(
                f"Expects an input of shape {self._input_shape} with, maybe, additional trailing dimensions, but found an array of shape {x.shape}."
            )

        trailing_shape = x.shape[len(self._input_shape) :]

        # For the sake of simplicity, make sure that x has shape (input_shape, V) for some V,
        # with a single trailing dimension.
        x_reshaped = bk.view(x, self._input_shape + (-1,))
        out = self._matmat(x_reshaped)  # (output_shape, V)
        return bk.view(out, self._output_shape + trailing_shape)

    @property
    def shape(self):
        """For compatibility with e.g. SciPy's LinearOperator class."""
        return (math.prod(self._output_shape), math.prod(self._input_shape))

    def transpose(self):
        return LinearOperator(
            matmat=self._rmatmat,
            rmatmat=self._matmat,
            input_shape=self._output_shape,
            output_shape=self._input_shape,
        )

    @property
    def T(self):
        return self.transpose()

    @classmethod
    def from_dense(cls, dense_matrix, *, input_shape, output_shape):
        """Returns a LinearOperator that behaves like the given dense matrix."""
        if len(dense_matrix.shape) == 2:
            N, M = dense_matrix.shape
            assert input_shape == (M,)
            assert output_shape == (N,)

            def matmat(s):
                M_, V_ = s.shape
                assert M_ == M
                return dense_matrix @ s  # (N,M) @ (M,V) -> (N,V)

            def rmatmat(s):
                N_, V_ = s.shape
                assert N_ == N
                return bk.transpose(dense_matrix, (1, 0)) @ s  # (M,N) @ (N,V) -> (M,V)

        elif len(dense_matrix.shape) == 3:
            B, N, M = dense_matrix.shape
            assert input_shape == (B, M)
            assert output_shape == (B, N)

            def matmat(s):
                B_, M_, V_ = s.shape
                assert B_ == B and M_ == M
                return dense_matrix @ s  # (B,N,M) @ (B,M,V) -> (B,N,V)

            def rmatmat(s):
                B_, N_, V_ = s.shape
                assert B_ == B and N_ == N
                # (B,M,N) @ (B,N,V) -> (B,M,V)
                return bk.transpose(dense_matrix, (0, 2, 1)) @ s

        else:
            raise ValueError(
                f"Expected a dense matrix of shape (N, M) or (B, N, M), but found an array of shape {dense_matrix.shape}."
            )

        return cls(
            matmat=matmat,
            rmatmat=rmatmat,
            input_shape=input_shape,
            output_shape=output_shape,
        )

    @classmethod
    def from_lazy_tensor(cls, lazy_tensor, *, input_shape, output_shape):
        """Returns a LinearOperator that behaves like the given KeOps LazyTensor."""
        if len(lazy_tensor.shape) == 2:
            N, M = lazy_tensor.shape
            assert input_shape == (M,)
            assert output_shape == (N,)

            def matmat(s):
                M_, V_ = s.shape
                assert M_ == M
                return lazy_tensor @ s  # (N,M) @ (M,V) -> (N,V)

            def rmatmat(s):
                N_, V_ = s.shape
                assert N_ == N
                return lazy_tensor.T @ s  # (M,N) @ (N,V) -> (M,V)

        else:
            raise ValueError(
                f"Expected a LazyTensor of shape (N, M), but found an array of shape {lazy_tensor.shape}."
            )

        return cls(
            matmat=matmat,
            rmatmat=rmatmat,
            input_shape=input_shape,
            output_shape=output_shape,
        )

    def rescale(self, *, input_scaling, output_scaling):
        """Returns a new LinearOperator that behaves like the original one, but with rescaled inputs and outputs."""
        b = input_scaling
        a = output_scaling

        assert a.shape == self._output_shape
        assert b.shape == self._input_shape

        def matmat(s):
            # __matmul__ reshapes the input to have shape (input_shape, V) for some V, with a single trailing dimension.
            V = s.shape[-1]
            assert s.shape == b.shape + (V,)
            a_broadcasted = bk.view(a, a.shape + (1,))
            b_broadcasted = bk.view(b, b.shape + (1,))
            return a_broadcasted * (self @ (b_broadcasted * s))

        def rmatmat(s):
            # __matmul__ reshapes the input to have shape (input_shape, V) for some V.
            V = s.shape[-1]
            assert s.shape == b.shape + (V,)
            a_broadcasted = bk.view(a, a.shape + (1,))
            b_broadcasted = bk.view(b, b.shape + (1,))
            return b_broadcasted * (self.T @ (a_broadcasted * s))

        return LinearOperator(
            matmat=matmat,
            rmatmat=rmatmat,
            input_shape=self._input_shape,
            output_shape=self._output_shape,
        )


@add_cached_methods_to_sphinx
class OTResult:
    """Abstract class for optimal transport results.

    An OT solver returns an object that inherits from OTResult
    (e.g. SinkhornOTResult) and implements the relevant
    methods (e.g. "plan" and "lazy_plan" but not "sparse_plan", etc.).
    log is a dictionary containing potential information about the solver
    """

    def __init__(
        self,
        *,
        a,
        b,
        potentials,
        array_properties,
        batchsize,
        reg,
        reg_type,
        unbalanced,
        unbalanced_type,
        debias,
        C=None,
        value=None,
        value_linear=None,
        plan=None,
        log=None,
        backend=None,
        sparse_plan=None,
        lazy_plan=None,
    ):

        self._a = a
        self._b = b
        self._C = C
        self._potentials = potentials
        self._array_properties = array_properties
        self._batchsize = batchsize

        self._reg = reg
        self._reg_type = reg_type
        self._unbalanced = unbalanced
        self._unbalanced_type = unbalanced_type
        self._debias = debias

        self._value = value
        self._value_linear = value_linear
        self._plan = plan
        self._log = log
        self._sparse_plan = sparse_plan
        self._lazy_plan = lazy_plan
        self._backend = backend

        # I assume that other solvers may return directly
        # some primal objects?
        # In the code below, let's define the main quantities
        # that may be of interest to users.
        # An OT solver returns an object that inherits from OTResult
        # (e.g. SinkhornOTResult) and implements the relevant
        # methods (e.g. "plan" and "lazy_plan" but not "sparse_plan", etc.).
        # log is a dictionary containing potential information about the solver

        # ----------------------------------------------------------------------------
        # Start of the Python magic (hack?) to load the features-computing methods,
        # memoize the properties (with cache clearing when users update the points,
        # edges or triangles), and add the methods to the class with a docstring that
        # is fully compatible with Sphinx autodoc.
        # ----------------------------------------------------------------------------

        # Cached methods: for reference on the Python syntax,
        # see "don't lru_cache methods! (intermediate) anthony explains #382",
        # https://www.youtube.com/watch?v=sVjtp6tGo0g
        cache_methods_and_properties(
            cls=self.__class__,  # OTResult,
            instance=self,
            cache_size=1,
        )

    _cached_methods = ()
    _cached_properties = (
        "potential_a",
        "potential_b",
        "potential_aa",
        "potential_bb",
        "density",
        "sparse_density",
        "lazy_density",
        "density_operator",
        "plan",
        "sparse_plan",
        "lazy_plan",
        "plan_operator",
        "value",
        "value_linear",
        "marginal_a",
        "marginal_b",
        "a_to_b",
        "b_to_a",
        "citation",
    )

    from ..cache import cache_clear

    def cast(self, x, shape):
        return bk.cast(
            x,
            shape=self._shapes[shape],
            dtype=self._array_properties.dtype,
            device=self._array_properties.device,
            library=self._array_properties.library,
        )

    # Dual potentials ====================================================================
    def _potential_a(self):
        """First dual potential, associated to the source measure `a`.

        This real-valued Tensor has the same shape and numerical dtype as the
        Tensor of source weights `a` that was provided as input to the OT solver.
        It is also hosted on the same device (RAM, GPU memory...), using the same
        tensor computing library (NumPy, PyTorch...).
        """
        return self.cast(self._potentials.f_ba, "a")

    def _potential_b(self):
        """Second dual potential, associated to the target measure `b`.

        This real-valued Tensor has the same shape and numerical dtype as the
        Tensor of target weights `b` that was provided as input to the OT solver.
        It is also hosted on the same device (RAM, GPU memory...), using the same
        tensor computing library (NumPy, PyTorch...).
        """
        return self.cast(self._potentials.g_ab, "b")

    def _potential_aa(self):
        """Dual potential associated to the self-interaction of the source measure `a`.

        This potential is only defined when using a debiased Sinkhorn solver.
        This real-valued Tensor has the same shape and numerical dtype as the
        Tensor of source weights `a` that was provided as input to the OT solver.
        It is also hosted on the same device (RAM, GPU memory...), using the same
        tensor computing library (NumPy, PyTorch...).
        """
        if self._potentials.f_aa is None:
            raise ValueError(
                "The self-interaction potential `f_aa` is not defined. "
                "To fix this issue, run your OT solver with `debias = True`."
            )

        return self.cast(self._potentials.f_aa, "a")

    def _potential_bb(self):
        """Dual potential associated to the self-interaction of the target measure `b`.

        This potential is only defined when using a debiased Sinkhorn solver.
        This real-valued Tensor has the same shape and numerical dtype as the
        Tensor of target weights `b` that was provided as input to the OT solver.
        It is also hosted on the same device (RAM, GPU memory...), using the same
        tensor computing library (NumPy, PyTorch...).
        """
        if self._potentials.g_bb is None:
            raise ValueError(
                "The self-interaction potential `g_bb` is not defined. "
                "To fix this issue, run your OT solver with `debias = True`."
            )

        return self.cast(self._potentials.g_bb, "b")

    # Transport plan =====================================================================
    def _density(self):
        """Density of the transport plan with respect to the reference measure, encoded as a dense array."""
        return None

    def _sparse_density(self):
        """Density of the transport plan with respect to the reference measure, encoded as a sparse array."""
        return None

    def _lazy_density(self):
        """Density of the transport plan with respect to the reference measure, encoded as a symbolic KeOps LazyTensor."""
        return None

    def _density_operator(self):
        """Density of the transport plan with respect to the reference measure, encoded as a linear operator."""
        return None

    def _plan(self):
        """Transport plan, encoded as a dense array."""
        # N.B.: We may catch out-of-memory errors and suggest
        # the use of lazy_plan or sparse_plan when appropriate.
        return None

    def _sparse_plan(self):
        """Transport plan, encoded as a sparse array."""
        return None

    def _lazy_plan(self):
        """Transport plan, encoded as a symbolic KeOps LazyTensor."""
        return None

    def _plan_operator(self):
        """Transport plan, encoded as a linear operator."""

        a = self.cast(self._a, "a")
        b = self.cast(self._b, "b")
        return self.density_operator.rescale(input_scaling=b, output_scaling=a)

    # Loss values ========================================================================
    def _value(self):
        """Full transport cost, including possible regularization terms."""
        if self._reg_type != "KL":
            raise NotImplementedError(
                "Currently, we only support 'KL' "
                "as regularization for the OT problem."
            )

        if self._unbalanced_type != "KL":
            raise NotImplementedError(
                "Currently, we only support 'KL' "
                "as regularization for the marginal constraints."
            )

        # sinkhorn_cost assumes that the potentials in self._potentials have the same
        # shapes as self._a, self._b:
        values = sinkhorn_cost(
            a=self._a,
            b=self._b,
            potentials=self._potentials,
            eps=self._reg,
            rho=self._unbalanced,
            debias=self._debias,
            batchsize=self._batchsize,
        )
        return self.cast(values, "B")

    def _value_linear(self):
        """The "bare bones" transport cost, i.e. the product between the transport plan and the cost."""
        return None

    # Marginal constraints ===============================================================
    def _marginal_a(self):
        """First marginal of the transport plan, with the same shape as the source weights `a`."""
        a = self.cast(self._a, "a")
        b = self.cast(self._b, "b")

        density = self.density_operator @ b
        assert density.shape == a.shape
        marginal = a * density
        return self.cast(marginal, "a")

    def _marginal_b(self):
        """Second marginal of the transport plan, with the same shape as the target weights `b`."""
        a = self.cast(self._a, "a")
        b = self.cast(self._b, "b")

        density = self.density_operator.T @ a
        assert density.shape == b.shape
        marginal = b * density
        return self.cast(marginal, "b")

    # Barycentric mappings ===============================================================
    # Return the displacement vectors as an array
    # that has the same shape as "xa"/"xb" (for samples)
    # or "a"/"b" * D (for images)?
    def _a_to_b(self):
        """Displacement vectors from the first to the second measure."""
        return None

    def _b_to_a(self):
        """Displacement vectors from the second to the first measure."""
        return None

    # Wasserstein barycenters ============================================================
    # @property
    # def masses(self):
    #     """Masses for the Wasserstein barycenter."""
    #     raise NotImplementedError()

    # @property
    # def samples(self):
    #     """Sample locations for the Wasserstein barycenter."""
    #     raise NotImplementedError()

    # Miscellaneous ======================================================================
    def _citation(self):
        r"""Appropriate citation(s) for this result, in plain text and BibTex formats."""

        # The string below refers to the GeomLoss library:
        # successor methods may concatenate the relevant references
        # to the original definitions, solvers and underlying numerical backends.
        return r"""GeomLoss library: 
        
            "Interpolating between optimal transport and MMD using Sinkhorn divergences." 
            In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 2681-2690. PMLR, 2019.
            Feydy, Jean, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari, Alain Trouvé, and Gabriel Peyré.

            @inproceedings{feydy2019interpolating,
                title={Interpolating between Optimal Transport and MMD using Sinkhorn Divergences},
                author={Feydy, Jean and S{\'e}journ{\'e}, Thibault and Vialard, Fran{\c{c}}ois-Xavier and Amari, Shun-ichi and Trouve, Alain and Peyr{\'e}, Gabriel},
                booktitle={The 22nd International Conference on Artificial Intelligence and Statistics},
                pages={2681--2690},
                year={2019}
            }
        """
