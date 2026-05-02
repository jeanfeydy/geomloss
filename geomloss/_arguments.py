from . import _backends as bk
from typing import NamedTuple, Any


class ArrayProperties(NamedTuple):
    B: int  # Batch dimension, 0 if not batch mode
    N: int  # Number of source samples
    M: int  # Number of target samples
    dtype: Any  # Numerical dtype: may be torch.dtype, a string, etc.
    device: Any  # Physical device: may be a string ("cpu", "gpu:0"...), torch.device...
    library: str  # Underlying framework: one of "numpy", "torch".


def check_regularization(
    *,
    reg,
    reg_type,
    unbalanced,
    unbalanced_type,
    method,
    tol,
    max_iter,
):
    if reg < 0:
        raise ValueError(f"Parameter 'reg' should be >= 0. Received {reg}.")
    elif reg == 0:
        raise NotImplementedError("Currently, we require that reg > 0.")

    if reg_type != "KL":
        raise NotImplementedError("Currently, we only support a Sinkhorn solver.")

    if unbalanced is not None and unbalanced <= 0:
        raise ValueError(
            "Parameter 'unbalanced' should be None (= +infty) "
            f"or > 0. Received {unbalanced}."
        )

    if unbalanced_type != "KL":
        raise NotImplementedError(
            "Currently, we only support unbalanced OT with "
            "a 'KL' penalty on the marginal constraints."
        )

    if method != "auto":
        raise NotImplementedError("Currently, we only support a single method.")

    if max_iter is None:
        raise ValueError("The 'max_iter' parameter should be a positive integer.")

    if tol is not None:
        raise NotImplementedError(
            "Currently, we do not support rigorous stopping criteria."
        )


def check_library(*args):
    """Checks that all input arrays come from the same library (numpy, torch...)."""

    libraries = list(set([bk.get_library(a) for a in args]))
    if len(libraries) > 1:
        raise ValueError(
            "The input arrays do not come from the same tensor library: "
            f"received a collection of {libraries}, which is ambiguous. "
            "To fix this error, please cast all arrays using a single library."
        )
    else:
        return libraries[0]


def check_dtype(*args):
    """Checks that all input arrays have the same numerical dtype."""

    dtypes = list(set([bk.dtype(a) for a in args]))
    if len(dtypes) > 1:
        raise ValueError(
            "The input arrays do not have the same numerical dtype: "
            f"received a collection of {dtypes}, which is ambiguous. "
            "To fix this error, please cast all arrays to the same numerical dtype."
        )
    else:
        return dtypes[0]


def check_device(*args):
    """Checks that all input arrays are stored on the same device."""

    devices = list(set([bk.device(a) for a in args]))
    if len(devices) > 1:
        raise ValueError(
            "The input arrays are not stored on the same device: "
            f"received a collection of {devices}, which is ambiguous."
            "To fix this error, please move all arrays to the same RAM or GPU device."
        )
    else:
        return devices[0]


def check_library_dtype_device(*args):
    # Check that all the arrays come from the same library (numpy, torch...):
    library = check_library(*args)
    # Check that every array has the same numerical precision:
    dtype = check_dtype(*args)
    # Check that every array is on the same device:
    device = check_device(*args)

    return library, dtype, device


def check_marginal(m, *, ones_like, marginal_size, name):
    if m is None:
        m = bk.ones_like(ones_like)
        m = m / marginal_size  # By default, the marginal sums up to 1

    if m.shape != ones_like.shape:
        raise ValueError(
            f"The marginal '{name}' should be of shape {ones_like.shape}. "
            f"Instead, received an array of shape {m.shape}."
        )

    # Check that all values are non-negative:
    if bk.any(m < 0):
        raise ValueError(
            f"The marginal '{name}' contains negative values. "
            f"We require that {name} >= 0."
        )

    return m


def check_marginal_masses(sums_a, sums_b, rtol=1e-3):
    """Raises an error if two vectors of total sums do not coincide.

    Args:
        sums_a ((B,) real-valued Tensor): Vector of B non-negative values.
        sums_b ((B,) real-valued Tensor): Vector of B non-negative values.
        rtol (float, optional): Relative tolerance. Defaults to 1e-3.

    Raises:
        ValueError: If sums_a and sums_b are too different from each other,
            we let the user know that we cannot use a balanced OT solver.
    """
    rel_diffs = bk.abs(sums_a - sums_b) / (sums_a + sums_b)

    if bk.any(rel_diffs > rtol):
        if len(sums_a) == 1:
            s = "do not sum up to the same value."
        else:
            s = "have rows that do not sum up to the same values."

        raise ValueError(
            "The two arrays of marginal weights 'a' and 'b' "
            f"{s}"
            "As a consequence, the balanced OT problem is not feasible. "
            "To fix this error, you may either normalize the two marginals ",
            "to make sure that their weights sum up to compatible values "
            "(= 1 for probability distributions), or use UNbalanced optimal "
            "transport with the 'unbalanced' keyword argument.",
        )
