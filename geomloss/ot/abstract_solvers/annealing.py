r"""This file controls the schedule for annealing schemes 
(also known as "epsilon-scaling") in Sinkhorn solvers.

The main reference for this file is Chapter 3.3 in Jean Feydy's PhD thesis:
Geometric data analysis, beyond convolutions (2020), 
https://www.jeanfeydy.com/geometric_data_analysis.pdf
"""

from ..typing import Tensor, Optional, AnnealingParameters

# ==============================================================================
#                         epsilon-scaling heuristic
# ==============================================================================


def max_diameter(x: Tensor, y: Tensor) -> float:
    """Returns a rough estimation of the diameter of a pair of point clouds.

    This quantity is used as a maximum "starting scale" in the epsilon-scaling
    annealing heuristic.

    Args:
        x ((N, D) Tensor): First point cloud.
        y ((M, D) Tensor): Second point cloud.

    Returns:
        float: Upper bound on the largest distance between points `x[i]` and `y[j]`.
    """
    mins = torch.stack((x.min(dim=0)[0], y.min(dim=0)[0])).min(dim=0)[0]  # (D,)
    maxs = torch.stack((x.max(dim=0)[0], y.max(dim=0)[0])).max(dim=0)[0]  # (D,)
    diameter = (maxs - mins).norm().item()
    return diameter


def annealing_parameters(
    *,
    x: Tensor,
    y: Tensor,
    p: int,
    blur: float,
    reach: Optional[float] = None,
    diameter: Optional[float] = None,
    n_iter: Optional[int] = None,
    scaling: Optional[float] = None,
) -> AnnealingParameters:
    r"""Turns high-level arguments into numerical values for the Sinkhorn loop.

    We use an aggressive strategy with an exponential cooling
    schedule: starting from a value of :math:`\text{diameter}^p`,
    the temperature epsilon is divided
    by :math:`\text{scaling}^p` at every iteration until reaching
    a minimum value of :math:`\text{blur}^p`.

    The number of iterations can be specified in two different ways,
    using either an integer number (n_iter) or a ratio between successive scales(scaling).

    Args:
        x (Tensor): Sample positions for the source measure.

        y (Tensor): Sample positions for the target measure.

        p (integer or float): The exponent of the Euclidean distance
            :math:`\|x_i-y_j\|` that defines the cost function
            :math:`\text{C}(x_i,y_j) =\tfrac{1}{p} \|x_i-y_j\|^p`.
            The relation between the blur scales (that are homogeneous to a distance)
            and the temperatures eps (that are homogeneous to the cost function)
            across iterations is that eps = blur**p.

        blur (float > 0): Target value for the blur scale and the
            temperature (= entropic regularization parameter)
            ":math:`\varepsilon = \text{blur}^p`".

        reach (float > 0 or None): Strength of the marginal constraints.
            None stands for +infinity, i.e. balanced optimal transport.

        diameter (float > 0 or None): Upper bound on the largest distance between
            points :math:`x_i` and :math:`y_j`.

        n_iter (int >= 1 or None): Number of iterations.

        scaling (float in (0,1) or None): Ratio between two successive
            values of the blur scale.

    Returns:
        list of float: list of values for the temperature epsilon.
    """

    if n_iter is not None and n_iter <= 0:
        raise ValueError(
            "The number of iterations should be >= 1. " f"Received n_iter={n_iter}."
        )

    if scaling is not None and (scaling <= 0 or scaling > 1):
        raise ValueError(
            "The scaling factor should be in (0,1]. " f"Received scaling={scaling}."
        )

    if n_iter is None and scaling is None:
        raise ValueError(
            "Please specify a number of iterations using either "
            "the n_iter or scaling parameters."
        )

    # Compute the typical scale of our configuration:
    if diameter is None:
        # Flatten the batch (if present)
        D = x.shape[-1]
        diameter = max_diameter(x.view(-1, D), y.view(-1, D))

    # Make sure that the diameter is >= blur:
    diameter = max(diameter, blur)

    # Compute the appropriate number of iterations, if it has not been provided already:
    if n_iter is None:
        if scaling == 1:
            raise ValueError(
                "If n_iter is not specified, the scaling coefficient "
                "should be < 1. Keeping a constant value for the temperature epsilon "
                "(with scaling = 1) does not allow us to stop convergence and may lead "
                "to an infinite loop."
            )
        else:
            # Ensure that we have enough iterations to go from diameter to blur
            # with geometric steps of size scaling:
            n_iter = (np.log(blur) - np.log(diameter)) / np.log(scaling)
            n_iter = int(np.floor(n_iter)) + 2

            # With the formula above, assuming that e.g.
            # diameter = 1, blur = 0.01 and scaling = 0.1,
            # we find n_iter = 2 + 2 = 4, that will eventually produce
            # blur_list = [1, 0.1, 0.01, 0.01]

    # At this point, we know that n_iter >= 1 and scaling is None or 0 < scaling <= 1.
    if scaling == 1:
        # The user has specified a number of iterations and a "constant" scaling:
        # this is the regular Sinkhorn algorithm, without annealing.
        blur_list = [blur] * n_iter

    elif scaling is None:
        # The user has specified a number of iterations but no scaling:
        # we follow a geometric progession from diameter to the target
        # blur value.
        if n_iter == 1:
            blur_list = [blur]
        else:
            blur_list = np.geomspace(diameter, blur, n_iter)

    else:
        # The user has specified a number of iterations *and* a scaling in (0,1):
        # we follow a geometric progression of factor scaling,
        # with n_iter terms and a "floor" minimum value at blur.
        blur_list = np.arange(n_iter)
        blur_list = np.log(diameter) + blur_list * np.log(scaling)
        blur_list = np.maximum(blur_list, np.log(blur))
        blur_list = np.exp(blur_list)

    # Turn our scales into temperature values:
    eps_list = [b**p for b in blur_list]

    return AnnealingParameters(
        diameter=diameter,
        eps_list=eps_list,
        blur_list=blur_list,
    )
