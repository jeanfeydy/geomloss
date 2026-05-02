from ._implementations.matrix import solve, solve_batch, barycenter, OTResultMatrix
from ._implementations.sample import (
    solve_sample,
    solve_sample_batch,
    barycenter_sample,
    OTResultSample,
)
from ._implementations.grid import solve_grid, barycenter_grid
from ._ot_result import OTResult, LinearOperator
