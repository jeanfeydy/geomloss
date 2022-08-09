r"""This file controls the schedule for annealing schemes 
(also known as "epsilon-scaling") in Sinkhorn solvers.

The main reference for this file is Chapter 3.3 in Jean Feydy's PhD thesis:
Geometric data analysis, beyond convolutions (2020), 
https://www.jeanfeydy.com/geometric_data_analysis.pdf
"""

import numpy as np
from ... import backends as bk
from ...typing import RealTensor, Optional, List, DescentParameters

# ==============================================================================
#                         epsilon-scaling heuristic
# ==============================================================================


def max_diameter(x: RealTensor, y: RealTensor) -> float:
    """Returns a rough estimation of the diameter of a pair of point clouds.

    This quantity can be used as a maximum "starting scale" in the epsilon-scaling
    annealing heuristic.

    Args:
        x ((N, D) real-valued Tensor): First point cloud.
        y ((M, D) real-valued Tensor): Second point cloud.

    Returns:
        float: Upper bound on the largest distance between points `x[i]` and `y[j]`.
    """
    mins = bk.amin(bk.stack((bk.amin(x, axis=0), bk.amin(y, axis=0))), axis=0)  # (D,)
    maxs = bk.amax(bk.stack((bk.amax(x, axis=0), bk.amax(y, axis=0))), axis=0)  # (D,)
    diameter = float(bk.norm(maxs - mins))
    return diameter


"""
# Compute the typical scale of our configuration:
if diameter is None:
    # Flatten the batch (if present)
    D = x.shape[-1]
    diameter = max_diameter(x.view(-1, D), y.view(-1, D))
"""


def annealing_parameters(
    *,
    diameter: float,
    p: int,
    blur: float,
    reach: Optional[float] = None,
    n_iter: Optional[int] = None,
    scaling: Optional[float] = None,
    resolutions: Optional[List[float]] = None,
) -> DescentParameters:
    r"""Turns high-level arguments into numerical values for the Sinkhorn loop.

    We use an aggressive strategy with an exponential cooling
    schedule: starting from a value of :math:`\text{diameter}^p`,
    the temperature epsilon is divided
    by :math:`\text{scaling}^p` at every iteration until reaching
    a minimum value of :math:`\text{blur}^p`.

    The number of iterations can be specified in two different ways, using either
    an integer number (n_iter) or a ratio between successive scales (scaling).

    Args:
        diameter (float > 0 or None): Upper bound on the largest distance between
            sample locations :math:`x_i` and :math:`y_j`.

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

        n_iter (int >= 1 or None): Number of iterations.

        scaling (float in (0,1) or None): Ratio between two successive
            values of the blur scale.

        resolutions (list of S float or None): List of successive scales at which
            we represent the input distributions. These typically correspond
            to sampling scales, i.e. to average distances between two nearest samples.
            These scales should be decreasing (we always work in a coarse-to-fine
            fashion). Note that this parameter is only relevant for multi-scale
            implementations. If scales is None or is a list of length 1, we assume
            that we work in single-scale mode and stick to a single representation
            of the input measures throughout the Sinkhorn iterations.

    Returns:
        descent (DescentParameters): A NamedTuple with attributes that describe
            the evolution of the main parameters along the iterations of the
            Sinkhorn loop.
            We return the attributes:
            - diameter (float): The value of the diameter that we used as an estimate
              in the descent. Typically, it is equal to max(diameter, blur).

            - blur_list (list of n_iter float > 0): List of successive values for
              the blur length of the Sinkhorn kernel, at which we process the samples.
              The number of iterations in the loop is equal to the length of this list.

            - eps_list (list of n_iter float > 0): List of successive values for
              the Sinkhorn regularization parameter, the temperature :math:`\varepsilon`.
              At every iteration, the temperature is equal to blur**p.
              The number of iterations in the loop is equal to the length of this list.

            - rho_list (list of n_iter (float > 0 or None)): List of successive values for
              the strength of the marginal constraints in unbalanced OT.
              None values stand for :math:`\rho = +\infty`, i.e. balanced OT.

            - scale_list (list of n_iter int): List of scale indices at which we
              perform our iterations.
              Each scale index should satisfy `0 <= scale < S`.
              For single-scale mode, we return `scale_list = [0] * n_iter`.
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

    # We use a constant value for the unbalanced parameter rho:
    if reach is None:
        rho = None
    else:
        rho = reach**p
    rho_list = [rho] * len(blur_list)

    # We perform an iteration that is associated to a precision "blur"
    # at the coarsest available resolution such that blur >= resolution.
    # This means that jumps from a coarse to a finer scale will happen when
    # blur[next_iteration] < current scale.
    #
    # More precisely, let's assume that:
    # resolutions = [1., .5, .1]
    # blur_list = [.7, .6, .5, .3, .2, .1, .05]
    # then:
    # scale_list = [1, 1, 1, 2, 2, 2, 2]
    # This will induce a "jump" from scale 1 to scale 2 between the
    # 3rd (blur=.5) and the 4th (blur=.3) iterations.
    #
    # Note that in this example:
    # - We don't use the first scale (resolution = 1.0), because our first iteration
    #   is already performed at blur = 0.7.
    # - We perform the 3rd iteration (blur = 0.5) at the 2nd scale (resolution = 0.5).
    #   Our convention is that we only jump if the blur becomes strictly smaller
    #   than the current resolution of our representation.
    # - We perform the last iteration (blur = .05) at the last scale (resolution = 0.1)
    #   because we don't have access to any finer representation of the distributions.

    if resolutions is None or len(resolutions) < 2:
        # Single-scale mode
        scale_list = [0] * len(blur_list)
    else:
        scale_list = []
        scale = 0  # Index for the current scale
        for blur in blur_list:
            while scale + 1 < len(resolutions) and blur < resolutions[scale]:
                scale = scale + 1
            scale_list.append(scale)

        # By convention, we always return a result at the finest scale available:
        scale_list[-1] = len(resolutions) - 1

    return DescentParameters(
        diameter=diameter,
        scale_list=scale_list,
        eps_list=eps_list,
        blur_list=blur_list,
        rho_list=rho_list,
    )
