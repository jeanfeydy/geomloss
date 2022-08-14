import numpy as np
from typing import NamedTuple, Any

try:
    import torch
    torch_from_numpy = torch.from_numpy
    torch_available = True
    
except:
    torch_from_numpy = None
    torch_available = False


class ExpectedOTResult(NamedTuple):
    """Stores the expected results of an OT solver following the OTResult API."""

    value: Any
    value_linear: Any
    plan: Any
    potential_a: Any
    potential_b: Any
    potential_aa: Any
    potential_bb: Any
    sparse_plan: Any
    lazy_plan: Any
    marginal_a: Any
    marginal_b: Any
    a_to_b: Any
    b_to_a: Any


def cast(x, *, library, dtype, device):
    """Casts a NumPy array to the expected Tensor type."""

    if isinstance(x, np.array):
        x = x.astype(dtype)
        if library == "torch":
            if not torch_available:
                raise ImportError(
                    "Could not load PyTorch, so could not create a test case "
                    "with torch Tensors."
                )
            x = torch_from_numpy(x).to(device=device)
        return x

    elif x is None:
        return None

    elif isinstance(x, ExpectedOTResult):
        return ExpectedOTResult(
            **{
                key: cast(val, library=library, dtype=dtype, device=device)
                for (key, val) in x.items()
            }
        )

    else:
        raise ValueError("Expected a NumPy array, None or an ExpectedOTResult object.")
