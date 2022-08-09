import numpy as np

try:
    import torch as pt

    tensor = pt.Tensor
    torch_available = True
except:
    torch_available = False


def pick(*, numpy, torch, main_arg=0):
    def out_fn(*args, **kwargs):
        arg = args[main_arg]
        if isinstance(arg, np.array):
            return numpy(*args, **kwargs)
        elif torch_available and isinstance(arg, tensor):
            return torch(*args, **kwargs)
        else:
            raise ValueError(
                "Expected a NumPy array or a PyTorch tensor, "
                f"but found {arg} "
                f"of type {type(arg)}."
            )

    return out_fn
