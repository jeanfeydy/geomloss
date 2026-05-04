import numpy as np

try:
    import torch as pt

    tensor = pt.Tensor
    torch_available = True
except:
    torch_available = False

from .keops import keops_available


def get_library(x):
    if isinstance(x, np.ndarray) or isinstance(x, np.ScalarType):
        return "numpy"
    elif torch_available and isinstance(x, tensor):
        return "torch"
    elif keops_available and hasattr(x, "_shape"):
        return "keops"
    else:
        raise ValueError(
            "Expected a NumPy array, a PyTorch tensor or a KeOps LazyTensor, "
            f"but found {x} "
            f"of type {type(x)}."
        )


def pick(*, numpy, torch, keops=None, main_arg=0):
    def out_fn(*args, **kwargs):
        arg = args[main_arg]
        library = get_library(arg)
        if library == "numpy":
            return numpy(*args, **kwargs)
        elif library == "torch":
            return torch(*args, **kwargs)
        elif keops is not None and library == "keops":
            return keops(*args, **kwargs)
        else:
            raise ValueError(
                "Expected a NumPy array, a PyTorch tensor or a KeOps LazyTensor, "
                f"but found {arg} "
                f"of type {type(arg)}."
            )

    return out_fn
