import numpy as np
from typing import NamedTuple, Any
from hypothesis import strategies as st
from dataclasses import dataclass

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


st_batchsize = st.integers(min_value=0, max_value=2)  # 0 means no batch mode
st_N = st.integers(min_value=1, max_value=10)
st_M = st.integers(min_value=1, max_value=10)

st_library = st.sampled_from(["numpy", "torch"])
st_dtype = st.sampled_from(["float32", "float64"])
st_device = st.sampled_from(["cpu", "cuda"])

st_library_dtype_device = st.fixed_dictionaries(
    {
        "library": st_library,
        "dtype": st_dtype,
        "device": st_device,
    }
)


@dataclass
class ExpectedOTResult:
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


@dataclass
class OTExperimentConfig:
    a: Any
    b: Any
    C: Any
    maxiter: int
    reg: float
    atol: float
    rtol: float = 0.0
    result: ExpectedOTResult = None
    unbalanced: Any = None
    CT: Any = None


def cast(x, *, library, dtype, device):
    """Casts a NumPy array to the expected Tensor type.

    Containers (dict and ExpectedOTResult) are handled recursively.
    """

    # We may need to apply cast recursively:
    def transform_mapping(mapping):
        return {
            k: cast(v, library=library, dtype=dtype, device=device)
            for k, v in mapping.items()
        }

    if library == "torch" and not torch_available:
        raise ImportError(
            "Could not load PyTorch, so could not create a test case "
            "with torch Tensors."
        )

    if not cuda_available:
        device = "cpu"

    if type(x) in [int, float, str]:
        return x

    elif isinstance(x, np.ndarray):
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

    elif isinstance(x, dict):
        return transform_mapping(x)

    elif isinstance(x, (OTExperimentConfig, ExpectedOTResult)):
        return type(x)(**transform_mapping(x.__dict__))

    else:
        raise ValueError(
            "Expected a NumPy array, int, float, None or an ExpectedOTResult object."
        )
