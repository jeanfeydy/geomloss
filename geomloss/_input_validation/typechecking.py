"""Runtime checker for function's arguments."""

# from beartype import beartype
# from jaxtyping import jaxtyped


def typecheck(func):
    """Runtime checker for function's arguments.

    This is a combination of the beartype and jaxtyping decorators. Jaxtyped
    allows to use jaxtyping typing hints for arrays/tensors while beartype is a
    runtime type checker. This decorator allows to use both.

    Parameters
    ----------
    func : callable
        the function to decorate

    Returns
    -------
    callable
        the decorated function
    """
    return jaxtyped(typechecker=beartype)(func)
