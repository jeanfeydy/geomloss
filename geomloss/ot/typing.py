from typing import List, Dict, Optional
from numpy.typing import ArrayLike
from collections.abc import Callable
from collections import NamedTuple

Tensor = ArrayLike


class AnnealingParameters(NamedTuple):
    diameter: float
    blur_list: List[float]
    eps_list: List[float]
