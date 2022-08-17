import numpy as np
from .common import ExpectedOTResult, cast

from numpy import log, eye, trace, allclose, block, concatenate, tile
from scipy.linalg import inv, det, sqrtm


def sqdist(x, y):
    return np.sum((x - y) ** 2)


# ========================================================================================
#                                Mathematical formulas
# ========================================================================================

# The code below implements the formulas found in
# "Entropic optimal transport between unbalanced Gaussian measures has a closed form"
# by Janati, Muzellec, Peyré and Cuturi, NeurIPS 2020.


def gaussian(*, mean, cov):
    """Creates a Gaussian density function on regular grids.

    Args:
        mean ((D,) array): Mean vector.
        cov ((D,D) array): Covariance matrix.

    Returns:
        function: A function that takes as input a (N,D) array of point positions
            and returns a (N,) array of pointwise densities, normalized to sum up to 1.
    """
    D = mean.shape[0]
    assert mean.shape == (D,)
    assert cov.shape == (D, D)
    sens = inv(cov)

    def density(x):
        N = x.shape[0]
        assert x.shape == (N, D)
        dev = x - mean  # (N, D)
        sqnorms = np.sum((dev @ sens) * dev, axis=1)  # (N,)
        weights = np.exp(-0.5 * sqnorms)
        return weights / np.sum(weights)  # (N,)

    return density


# ----------------------------------------------------------------------------------------
#                                      Section 2
# ----------------------------------------------------------------------------------------


def Wasserstein_Bures_distance(*, a, A, b, B):
    """Implements Eq. (3).

    Args:
        a ((D,) array): Mean of the source Gaussian.
        A ((D,D) array): Covariance of the source Gaussian.
        b ((D,) array): Mean of the target Gaussian.
        B ((D,D) array): Covariance of the target Gaussian.

    Returns:
        float: The squared Wasserstein distance between N(a,A) and N(b,B).
    """
    return sqdist(a, b) + Bures_distance(A=A, B=B)


def Bures_distance(*, A, B):
    """Implements Eq. (4).

    Args:
        A ((D,D) array): Covariance of the source Gaussian.
        B ((D,D) array): Covariance of the target Gaussian.

    Returns:
        float: The squared Bures distance between A and B.
    """
    A_12 = sqrtm(A)
    return trace(A) + trace(B) - 2 * trace(sqrtm(A_12 @ B @ A_12))


def Monge_map_gaussians(*, a, A, b, B):
    """Implements Eq. (5).

    Args:
        a ((D,) array): Mean of the source Gaussian.
        A ((D,D) array): Covariance of the source Gaussian.
        b ((D,) array): Mean of the target Gaussian.
        B ((D,D) array): Covariance of the target Gaussian.

    Returns:
        function: The Monge map that turns N(a,A) into N(b,B).
    """
    # First expression:
    A_12 = sqrtm(A)
    A_m12 = inv(A_12)
    T_AB = A_m12 @ sqrtm(A_12 @ B @ A_12) @ A_m12

    # Second expression:
    B_12 = sqrtm(B)
    T_AB_bis = B_12 @ inv(sqrtm(B_12 @ A @ B_12)) @ B_12

    # Check that the two expressions coincide:
    assert allclose(T_AB, T_AB_bis)

    def T_star(x):
        return T_AB @ (x - a) + b

    return T_star


# ----------------------------------------------------------------------------------------
#                                      Section 3
# ----------------------------------------------------------------------------------------


def OT_sigma(*, a, A, b, B, sigma):
    """Implements Eq. (13).

    Args:
        a ((D,) array): Mean of the source Gaussian.
        A ((D,D) array): Covariance of the source Gaussian.
        b ((D,) array): Mean of the target Gaussian.
        B ((D,D) array): Covariance of the target Gaussian.
        sigma (float > 0): Entropic blur.

    Returns:
        float: The entropy-regularized squared Wasserstein distance
            between N(a,A) and N(b,B).
    """
    return sqdist(a, b) + Bures_sigma_distance(A=A, B=B, sigma=sigma)


def Bures_sigma_distance(*, A, B, sigma):
    """Implements Eq. (14).

    Args:
        A ((D,D) array): Covariance of the source Gaussian.
        B ((D,D) array): Covariance of the target Gaussian.
        sigma (float > 0): Entropic blur.

    Returns:
        float: The entropy-regularized squared Bures distance between A and B.
    """
    d = len(A)
    s2 = sigma**2
    D_s = D_sigma(A=A, B=B, sigma=sigma)

    return (
        trace(A)
        + trace(B)
        - trace(D_s)
        + d * s2 * (1 - log(2 * s2))
        + s2 * log(det(D_s + s2 * eye(d)))
    )


def D_sigma(*, A, B, sigma):
    """Implements the formula at the start of Theorem 1.

    Args:
        A ((D,D) array): Covariance of the source Gaussian.
        B ((D,D) array): Covariance of the target Gaussian.
        sigma (float > 0): Entropic blur.

    Returns:
        (D,D) array: The covariance factor in Theorem 1.
    """
    d = len(A)
    A_12 = sqrtm(A)
    return sqrtm(4 * A_12 @ B @ A_12 + sigma**4 * eye(d))


def C_sigma(*, A, B, sigma):
    """Implements the formula between Eq. (14) and Eq. (15).

    Args:
        A ((D,D) array): Covariance of the source Gaussian.
        B ((D,D) array): Covariance of the target Gaussian.
        sigma (float > 0): Entropic blur.

    Returns:
        (D,D) array: The off_diagonal factor in the covariance
            of the entropy-regularized transport plan between N(a,A) and N(b,B).
    """
    d = len(A)
    A_12 = sqrtm(A)
    A_m12 = inv(A_12)
    return 0.5 * A_12 @ D_sigma(A=A, B=B, sigma=sigma) @ A_m12 - 0.5 * sigma**2 * eye(
        d
    )


def pi_sigma(*, a, A, b, B, sigma):
    """Implements Eq. (15).

    Args:
        a ((D,) array): Mean of the source Gaussian.
        A ((D,D) array): Covariance of the source Gaussian.
        b ((D,) array): Mean of the target Gaussian.
        B ((D,D) array): Covariance of the target Gaussian.
        sigma (float > 0): Entropic blur.

    Returns:
        function: The entropy-regularized transport plan pi^*,
            encoded as a function that takes as input:
            - a (N,D) array of source coordinates `x`,
            - a (M,D) array of target coordinates `y`,
            and returns a (N,M) array of probabilities that sums up to 1.
    """
    C_s = C_sigma(A=A, B=B, sigma=sigma)
    mean = concatenate((a, b), axis=0)
    cov = block([[A, C_s], [C_s.T, B]])

    def pi_star(*, x, y):
        N, M, D = x.shape[0], y.shape[0], y.shape[1]
        x_i = tile(x.reshape(N, 1, D), (1, M, 1)).reshape(N * M, D)
        y_j = tile(y.reshape(1, M, D), (N, 1, 1)).reshape(N * M, D)
        xy_ij = concatenate((x_i, y_j), axis=1)  # (N*M, D+D)
        return gaussian(mean=mean, cov=cov)(xy_ij).reshape(N, M)

    return pi_star


def Q(X):
    """Creates a quadratic function, associated to the (D,D) matrix X.

    Args:
        X ((D,D) array): Quadratic form.

    Returns:
        function: Function that takes as input a (N,D) array of point positions x
            and returns a (N,) array of values.
    """
    D = X.shape[0]
    assert X.shape == (D, D)

    def q(x):
        N = x.shape[0]
        assert x.shape == (N, D)
        return -0.5 * np.sum((x @ X) * x, axis=1)  # (N,)

    return q


def OT_sigma_potentials(*, a, A, b, B, sigma):

    if np.sum(np.abs(a)) != 0:
        raise NotImplementedError()
    if np.sum(np.abs(b)) != 0:
        raise NotImplementedError()

    d = A.shape[0]
    s2 = sigma**2
    C_s = C_sigma(A=A, B=B, sigma=sigma)

    # Eq. (23), multiplied by sigma ** 2:
    inv_C_s2 = inv(C_s + s2 * eye(d))  # (D, D)
    s2_U = B @ inv_C_s2 - eye(d)
    s2_V = inv_C_s2 @ A - eye(d)

    # Eq. (19):
    f = lambda x: 2 * Q(s2_U)(x)
    g = lambda y: 2 * Q(s2_V)(y)


def Sinkhorn_barycenters(*, w, a, A, sigma):
    r"""Implements Eq. (31).

    Returns the parameters of the Gaussian distribution, solution of Eq. (30):
    beta = argmin_{beta} \sum_{k=1}^K w[k] * S_sigma(N(a[k], A[k]), beta) .

    where S_sigma is the debiased Sinkhorn divergence with blur sigma.

    Args:
        w ((K,) array): Non-negative weights that sum up to 1.
        a ((K,D) array): Means of the input measures.
        A ((K,D,D) array): Covariance matrices of the input measures.

    Returns:
        (D,) array, (D, D) array: Mean and covariance of the barycenter.
    """

    K, D = a.shape
    assert w.shape == (K,)
    assert A.shape == (K, D, D)
    assert np.sum(w) == 1.0

    if sigma != 0:
        raise NotImplementedError()

    mean = (w.reshape(K, 1) * a).sum(axis=0)

    pass


# ----------------------------------------------------------------------------------------
#                                      Section 4
# ----------------------------------------------------------------------------------------


def UOT_cost_masses(*, m_a, m_b, m_pi, sigma, gamma):
    """Implements Eq. (37)."""
    return (
        gamma * (m_a + m_b)
        + 2 * sigma**2 * m_a * m_b
        - 2 * (sigma**2 + gamma) * m_pi
    )


def UOT_tau(*, sigma, gamma):
    """Implements the formula above Eq. (39)."""
    return gamma / (2 * sigma**2 + gamma)


def UOT_lambda(*, sigma, gamma):
    """Implements the formula above Eq. (39)."""
    return sigma**2 + gamma / 2


def UOT_mu(*, a, A, b, B, sigma, gamma):
    """Implements Eq. (39)."""
    inv_X = inv(UOT_X(A=A, B=B, sigma=sigma, gamma=gamma))
    mu_a = a + A @ inv_X @ (b - a)
    mu_b = b + B @ inv_X @ (a - b)
    return np.concatenate((mu_a, mu_b), axis=0)


def UOT_H(*, A, B, sigma, gamma):
    """Implements Eq. (40)."""

    l = UOT_lambda(sigma=sigma, gamma=gamma)
    inv_X = inv(UOT_X(A=A, B=B, sigma=sigma, gamma=gamma))
    C = UOT_C(A=A, B=B, sigma=sigma, gamma=gamma)
    Id = eye(A.shape[0])

    return block(
        [
            [
                (Id + C / l) @ (A - A @ inv_X @ A),
                C + (Id + C / l) @ A @ inv_X @ B,
            ],
            [
                C.T + (Id + C.T / l) @ B @ inv_X @ A,
                (Id + C.T / l) @ (B - B @ inv_X @ B),
            ],
        ]
    )


def UOT_m_pi(*, m_a, a, A, m_b, b, B, sigma, gamma):
    """Implements Eq. (41)."""

    d = A.shape[0]
    tau = UOT_tau(sigma=sigma, gamma=gamma)
    A_t = A_tilde(A=A, sigma=sigma, gamma=gamma)
    B_t = B_tilde(B=B, sigma=sigma, gamma=gamma)
    inv_X = inv(UOT_X(A=A, B=B, sigma=sigma, gamma=gamma))
    C = UOT_C(A=A, B=B, sigma=sigma, gamma=gamma)

    term_1 = sigma ** ((d * sigma**2) / (gamma + sigma**2))

    term_2 = m_a * m_b * det(C) * np.sqrt((det(A_t @ B_t) ** tau) / det(A @ B))
    term_2 = term_2 ** (1 / (tau + 1))

    term_3 = 1 / np.sqrt(det(C - (2 / gamma) * A_t @ B_t))

    term_4 = np.sum((a - b).reshape(-1) * (inv_X @ (a - b)).reshape(-1))
    term_4 = np.exp(-term_4 / (2 * (tau + 1)))

    return term_1 * term_2 * term_3 * term_4


def UOT_X(*, A, B, sigma, gamma):
    """Implements the formula below Eq. (41)."""
    d = A.shape[0]
    return A + B + UOT_lambda(sigma=sigma, gamma=gamma) * eye(d)


def A_tilde(*, A, sigma, gamma):
    """Implements the formula below Eq. (41)."""
    Id = eye(A.shape[0])
    l = UOT_lambda(sigma=sigma, gamma=gamma)
    return 0.5 * gamma * (Id - l * inv(A + l * Id))


def B_tilde(*, B, sigma, gamma):
    """Implements the formula below Eq. (41)."""
    d = B.shape[0]
    l = UOT_lambda(sigma=sigma, gamma=gamma)
    return 0.5 * gamma * (eye(d) - l * inv(B + l * eye(d)))


def UOT_C(*, A, B, sigma, gamma):
    """Implements the formula below Eq. (41)."""
    tau = UOT_tau(sigma=sigma, gamma=gamma)
    A_t = A_tilde(A=A, sigma=sigma, gamma=gamma)
    B_t = B_tilde(B=B, sigma=sigma, gamma=gamma)
    Id = eye(A.shape[0])

    return sqrtm(A_t @ B_t / tau + sigma**4 * Id / 4) - sigma**2 * Id / 2


def pi_sigma_gamma(*, m_a, a, A, m_b, b, B, sigma, gamma):
    """Implements Theorem 3.i.

    Args:
        m_a (float): Total mass of the source Gaussian.
        a ((D,) array): Mean of the source Gaussian.
        A ((D,D) array): Covariance of the source Gaussian.
        m_b (float): Total mass of the target Gaussian.
        b ((D,) array): Mean of the target Gaussian.
        B ((D,D) array): Covariance of the target Gaussian.
        sigma (float > 0): Entropic blur.

    Returns:
        function: The unbalanced, entropy-regularized transport plan pi^*,
            encoded as a function that takes as input:
            - a (N,D) array of source coordinates `x`,
            - a (M,D) array of target coordinates `y`,
            and returns a (N,M) transport plan.
    """
    m_pi = UOT_m_pi(m_a=m_a, a=a, A=A, m_b=m_b, b=b, B=B, sigma=sigma, gamma=gamma)
    mean = UOT_mu(a=a, A=A, b=b, B=B, sigma=sigma, gamma=gamma)
    cov = UOT_H(A=A, B=B, sigma=sigma, gamma=gamma)

    def pi_star(*, x, y):
        N, M, D = x.shape[0], y.shape[0], y.shape[1]
        x_i = tile(x.reshape(N, 1, D), (1, M, 1)).reshape(N * M, D)
        y_j = tile(y.reshape(1, M, D), (N, 1, 1)).reshape(N * M, D)
        xy_ij = concatenate((x_i, y_j), axis=1)  # (N*M, D+D)
        return m_pi * gaussian(mean=mean, cov=cov)(xy_ij).reshape(N, M)

    return pi_star


def OT_sigma_gamma(*, m_a, a, A, m_b, b, B, sigma, gamma):
    """Implements Theorem 3.ii.

    Args:
        m_a (float): Total mass of the source Gaussian.
        a ((D,) array): Mean of the source Gaussian.
        A ((D,D) array): Covariance of the source Gaussian.
        m_b (float): Total mass of the target Gaussian.
        b ((D,) array): Mean of the target Gaussian.
        B ((D,D) array): Covariance of the target Gaussian.
        sigma (float > 0): Entropic blur.

    Returns:
        float: The unbalanced, entropy-regularized squared Wasserstein distance
            between m_a * N(a,A) and m_b * N(b,B).
    """
    m_pi = UOT_m_pi(m_a=m_a, a=a, A=A, m_b=m_b, b=b, B=B, sigma=sigma, gamma=gamma)
    return UOT_cost_masses(m_a=m_a, m_b=m_b, m_pi=m_pi, sigma=sigma, gamma=gamma)


#


# ========================================================================================
#                             Test cases for the OT solvers
# ========================================================================================


def gaussians_matrix(
    *,
    N,
    M,
    D,
    debias,
    blur,
    reach,
    cov_type,
    batchsize,
    **kwargs,
):
    """Generates two Gaussian distributions sampled on a regular grid.

    This example is used by tests/test_ot_solve_matrix.py.
    """

    # Generate some random data ----------------------------------------------------------
    B = max(1, batchsize)

    # Geometry:
    # We sample the two distributions on regular grids on [-1,2]^D:
    x_i = np.linspace(-1, 2, N)
    x_i = np.stack(np.meshgrid(*((x_i,) * D), indexing="ij"), axis=-1).reshape(
        N**D, D
    )

    y_j = np.linspace(-1, 2, M)
    y_j = np.stack(np.meshgrid(*((y_j,) * D), indexing="ij"), axis=-1).reshape(
        M**D, D
    )

    # Matrix of squared distances (not halved), following the convention of Janati et al.:
    C = np.sum((x_i.reshape(N**D, 1, D) - y_j.reshape(1, M**D, D)) ** 2, axis=-1)
    C = np.tile(C, (B, 1, 1))
    assert C.shape == (B, N**D, M**D)

    # Gaussian distributions:
    # Means for the sources and the targets are in [0,1]:
    if True:
        means = np.random.rand(2, B, D)
    else:
        means = np.array([[[0.0]], [[1.0]]])
    min_std = 3 * 3 / min(N, M)  # Typical distance between samples in 3/N
    # Make sure that the support of our Gaussians at +-5*sigma is in [-1,1]:
    max_std = 0.2
    assert max_std > min_std

    # Total mass for the unbalanced case, in [0, 2):
    if True:
        total_mass = 2 * np.random.rand(2, B)  # (2, B)
    else:
        total_mass = 0.5 * np.ones((2, B))

    # TODO: Add non-diagonal test cases
    if cov_type == "diagonal":
        stds = np.random.rand(2, B, D)  # Standard deviations = cov^1/2
        stds = min_std + (max_std - min_std) * stds
        covs = np.zeros((2, B, D, D))
        covs[:, :, np.arange(D), np.arange(D)] = stds**2
    else:
        raise NotImplementedError()

    # Compute the densities for the source and target distributions:
    source_weights = np.zeros((B, N))
    target_weights = np.zeros((B, M))
    for k in range(B):
        source_weights[k, :] = gaussian(mean=means[0, k, :], cov=covs[0, k, :, :])(x_i)
        target_weights[k, :] = gaussian(mean=means[1, k, :], cov=covs[1, k, :, :])(y_j)

        if reach is not None:
            source_weights[k, :] *= total_mass[0, k]
            target_weights[k, :] *= total_mass[1, k]

    # Apply our formulas -----------------------------------------------------------------
    value = np.zeros((B,))  # (B,)
    plan = np.zeros_like(C)  # (B,N**D,M**D)
    marginal_a = np.copy(source_weights)  # (B,N**D)
    marginal_b = np.copy(target_weights)  # (B,M**D)

    def source_target(k):
        means_cov = {
            "a": means[0, k, :],
            "A": covs[0, k, :, :],
            "b": means[1, k, :],
            "B": covs[1, k, :, :],
        }

        if reach is None:
            # Balanced case
            return means_cov
        else:
            # Unbalanced case
            return {"m_a": total_mass[0, k], "m_b": total_mass[1, k], **means_cov}

    if blur == 0 and reach is None:
        # Balanced case, i.e. the Bures metric
        for k in range(B):
            value[k] = Wasserstein_Bures_distance(**source_target(k))

        # We approximate this solution with a small blur value at .01.
        eps = 1e-4
        rho = None

        # The true transport plan is singular, so we'd rather not check its correctness
        # against a sampled array:
        plan = None

    elif reach is None:

        eps = 2 * blur**2
        rho = None
        # Entropy-regularized and balanced case
        for k in range(B):
            value[k] = OT_sigma(sigma=blur, **source_target(k))
            plan_k = pi_sigma(sigma=blur, **source_target(k))
            plan[k, :, :] = plan_k(x=x_i, y=y_j)  # (N**D, M**D)

    else:
        # Entropy-regularized and unbalanced case
        if blur == 0:
            # The true transport plan is singular, so we use a smoother value instead:
            blur = 0.1

        eps = 2 * blur**2
        rho = reach**2

        for k in range(B):
            value[k] = OT_sigma_gamma(sigma=blur, gamma=reach**2, **source_target(k))
            plan_k = pi_sigma_gamma(sigma=blur, gamma=reach**2, **source_target(k))
            plan[k, :, :] = plan_k(x=x_i, y=y_j)  # (N**D, M**D)


        marginal_a = np.sum(plan, axis=2)
        marginal_b = np.sum(plan, axis=1)

    # Convert to match the expected signatures and return the result ---------------------
    if batchsize == 0:  # No batch mode:
        # (B,) -> (), (B,N) -> (N,), (B,M) -> (M,), (B,N,M) -> (N,M)
        source_weights, target_weights = source_weights[0], target_weights[0]
        marginal_a, marginal_b = marginal_a[0], marginal_b[0]
        C, value = C[0], value[0]
        if plan is not None:
            plan = plan[0]

    return cast(
        {
            "a": source_weights,
            "b": target_weights,
            "C": C,
            "x": x_i,
            "y": y_j,
            "means": means,
            "covs": covs,
            "total_mass": total_mass,
            "maxiter": 1000,
            "reg": eps,
            "unbalanced": rho,
            "atol": 0.01,
            "result": ExpectedOTResult(
                value=value,
                # value_linear=value,
                plan=plan,
                marginal_a=marginal_a,
                marginal_b=marginal_b,
            ),
        },
        **kwargs,
    )
