class OTResult:
    """Abstract class for optimal transport results.

    An OT solver returns an object that inherits from OTResult
    (e.g. SinkhornOTResult) and implements the relevant
    methods (e.g. "plan" and "lazy_plan" but not "sparse_plan", etc.).
    log is a dictionary containing potential information about the solver
    """

    def __init__(
        self,
        potentials=None,
        value=None,
        value_linear=None,
        plan=None,
        log=None,
        backend=None,
        sparse_plan=None,
        lazy_plan=None,
    ):

        self._potentials = potentials
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

    # Dual potentials --------------------------------------------
    @property
    def potentials(self):
        """Dual potentials, i.e. Lagrange multipliers for the marginal constraints.

        This pair of arrays has the same shape, numerical type
        and properties as the input weights "a" and "b".
        """
        if self._potentials is not None:
            return self._potentials
        else:
            raise NotImplementedError()

    @property
    def potential_a(self):
        """First dual potential, associated to the "source" measure "a"."""
        if self._potentials is not None:
            return self._potentials[0]
        else:
            raise NotImplementedError()

    @property
    def potential_b(self):
        """Second dual potential, associated to the "target" measure "b"."""
        if self._potentials is not None:
            return self._potentials[1]
        else:
            raise NotImplementedError()

    # Transport plan -------------------------------------------
    @property
    def plan(self):
        """Transport plan, encoded as a dense array."""
        # N.B.: We may catch out-of-memory errors and suggest
        # the use of lazy_plan or sparse_plan when appropriate.

        if self._plan is not None:
            return self._plan
        else:
            raise NotImplementedError()

    @property
    def sparse_plan(self):
        """Transport plan, encoded as a sparse array."""
        raise NotImplementedError()

    @property
    def lazy_plan(self):
        """Transport plan, encoded as a symbolic KeOps LazyTensor."""
        raise NotImplementedError()

    # Loss values --------------------------------
    @property
    def value(self):
        """Full transport cost, including possible regularization terms."""
        if self._value is not None:
            return self._value
        else:
            raise NotImplementedError()

    @property
    def value_linear(self):
        """The "minimal" transport cost, i.e. the product between the transport plan and the cost."""
        if self._value_linear is not None:
            return self._value_linear
        else:
            raise NotImplementedError()

    # Marginal constraints -------------------------
    @property
    def marginals(self):
        """Marginals of the transport plan: should be very close to "a" and "b" for balanced OT."""
        raise NotImplementedError()

    @property
    def marginal_a(self):
        """First marginal of the transport plan, with the same shape as "a"."""
        raise NotImplementedError()

    @property
    def marginal_b(self):
        """Second marginal of the transport plan, with the same shape as "b"."""
        raise NotImplementedError()

    # Barycentric mappings -------------------------
    # Return the displacement vectors as an array
    # that has the same shape as "xa"/"xb" (for samples)
    # or "a"/"b" * D (for images)?
    @property
    def a_to_b(self):
        """Displacement vectors from the first to the second measure."""
        raise NotImplementedError()

    @property
    def b_to_a(self):
        """Displacement vectors from the second to the first measure."""
        raise NotImplementedError()

    # Wasserstein barycenters ----------------------
    # @property
    # def masses(self):
    #     """Masses for the Wasserstein barycenter."""
    #     raise NotImplementedError()

    # @property
    # def samples(self):
    #     """Sample locations for the Wasserstein barycenter."""
    #     raise NotImplementedError()

    # Miscellaneous --------------------------------
    @property
    def citation(self):
        """Appropriate citation(s) for this result, in plain text and BibTex formats."""

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
