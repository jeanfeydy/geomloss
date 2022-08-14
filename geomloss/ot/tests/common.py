import numpy as np
from typing import NamedTuple, Any

try:
    import torch
    torch_from_numpy = torch.from_numpy
    torch_from_numpy_scalar = torch.tensor
    torch_available = True
    cuda_available = torch.cuda.is_available()
    
except:
    torch_from_numpy = None
    torch_from_numpy_scalar = None
    torch_available = False
    cuda_available = False


class ExpectedOTResult(NamedTuple):
    """Stores the expected results of an OT solver following the OTResult API."""

    value: Any = None
    value_linear: Any = None
    plan: Any = None
    potential_a: Any = None
    potential_b: Any = None
    potential_aa: Any = None
    potential_bb: Any = None
    sparse_plan: Any = None
    lazy_plan: Any = None
    marginal_a: Any = None
    marginal_b: Any = None
    a_to_b: Any = None
    b_to_a: Any = None


def cast(x, *, library, dtype, device):
    """Casts a NumPy array to the expected Tensor type."""

    if library == "torch" and not torch_available:
        raise ImportError(
            "Could not load PyTorch, so could not create a test case "
            "with torch Tensors."
        )
    
    if not cuda_available:
        device = "cpu"

    if isinstance(x, np.ndarray):
        x = x.astype(dtype)
        if library == "torch":
            x = torch_from_numpy(x).to(device=device)
        return x
    
    elif isinstance(x, np.ScalarType):  # Typically, ot_result.value in no batch mode
        x = x.astype(dtype)
        if library == "torch":
            x = torch_from_numpy_scalar(x).to(device=device)
        return x

    elif x is None:
        return None

    elif isinstance(x, ExpectedOTResult):
        return ExpectedOTResult(
            **{
                key: cast(val, library=library, dtype=dtype, device=device)
                for (key, val) in x._asdict().items()
            }
        )

    else:
        raise ValueError("Expected a NumPy array, None or an ExpectedOTResult object.")
