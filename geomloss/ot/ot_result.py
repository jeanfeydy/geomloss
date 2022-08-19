from .. import backends as bk
from .abstract_solvers.unbalanced_ot import sinkhorn_cost


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

    def cast(self, x, shape):
        return bk.cast(
            x,
            shape=self._shapes[shape],
            dtype=self._array_properties.dtype,
            device=self._array_properties.device,
            library=self._array_properties.library,
        )

    # Dual potentials ====================================================================
    @property
    def potential_a(self):
        """First dual potential, associated to the source measure `a`.

        This real-valued Tensor has the same shape and numerical dtype as the
        Tensor of source weights `a` that was provided as input to the OT solver.
        It is also hosted on the same device (RAM, GPU memory...), using the same
        tensor computing library (NumPy, PyTorch...).
        """
        return self.cast(self._potentials.f_ba, "a")

    @property
    def potential_b(self):
        """Second dual potential, associated to the target measure `b`.

        This real-valued Tensor has the same shape and numerical dtype as the
        Tensor of target weights `b` that was provided as input to the OT solver.
        It is also hosted on the same device (RAM, GPU memory...), using the same
        tensor computing library (NumPy, PyTorch...).
        """
        return self.cast(self._potentials.g_ab, "b")

    @property
    def potential_aa(self):
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

    @property
    def potential_bb(self):
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
    @property
    def plan(self):
        """Transport plan, encoded as a dense array."""
        # N.B.: We may catch out-of-memory errors and suggest
        # the use of lazy_plan or sparse_plan when appropriate.
        return None

    @property
    def sparse_plan(self):
        """Transport plan, encoded as a sparse array."""
        return None

    @property
    def lazy_plan(self):
        """Transport plan, encoded as a symbolic KeOps LazyTensor."""
        return None

    # Loss values ========================================================================
    @property
    def value(self):
        """Full transport cost, including possible regularization terms."""
        if self._reg_type != "relative entropy":
            raise NotImplementedError(
                "Currently, we only support 'relative entropy' "
                "as regularization for the OT problem."
            )

        if self._unbalanced_type != "relative entropy":
            raise NotImplementedError(
                "Currently, we only support 'relative entropy' "
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
        )
        return self.cast(values, "B")

    @property
    def value_linear(self):
        """The "bare bones" transport cost, i.e. the product between the transport plan and the cost."""
        return None

    # Marginal constraints ===============================================================
    @property
    def marginal_a(self):
        """First marginal of the transport plan, with the same shape as the source weights `a`."""
        return None

    @property
    def marginal_b(self):
        """Second marginal of the transport plan, with the same shape as the target weights `b`."""
        return None

    # Barycentric mappings ===============================================================
    # Return the displacement vectors as an array
    # that has the same shape as "xa"/"xb" (for samples)
    # or "a"/"b" * D (for images)?
    @property
    def a_to_b(self):
        """Displacement vectors from the first to the second measure."""
        return None

    @property
    def b_to_a(self):
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
    @property
    def citation(self):
        r"""Appropriate citation(s) for this result, in plain text and BibTex formats."""

        # The string below refers to the GeomLoss library:
        # successor methods may concatenate the relevant references
        # to the original definitions, solvers and underlying numerical backends.
        return """GeomLoss library: 
        
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
