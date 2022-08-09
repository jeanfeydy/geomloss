import numpy as np

def pick(*, numpy, torch, main_arg = 0):
    def out_fn(*args, **kwargs):
        arg = args[main_arg]
        if isinstance(arg, np.array):
            return numpy(*args, **kwargs)
        else:
            return torch(*args, **kwargs)
    return out_fn