"""
Utility routines for benchmarks on OT solvers
===================================================

"""

import time
import torch
import numpy as np

use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
numpy = lambda x: x.detach().cpu().numpy()

################################################################################
# 3D dataset
# -------------------------
#
# Reading **.ply** files:

from plyfile import PlyData, PlyElement


def load_ply_file(fname, offset=[-0.011, 0.109, -0.008], scale=0.04):
    """Loads a .ply mesh to return a collection of weighted Dirac atoms: one per triangle face."""

    # Load the data, and read the connectivity information:
    plydata = PlyData.read(fname)
    triangles = np.vstack(plydata["face"].data["vertex_indices"])

    # Normalize the point cloud, as specified by the user:
    points = np.vstack([[x, y, z] for (x, y, z) in plydata["vertex"]])
    points -= offset
    points /= 2 * scale

    # Our mesh is given as a collection of ABC triangles:
    A, B, C = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]

    # Locations and weights of our Dirac atoms:
    X = (A + B + C) / 3  # centers of the faces
    S = np.sqrt(np.sum(np.cross(B - A, C - A) ** 2, 1)) / 2  # areas of the faces

    print(
        "File loaded, and encoded as the weighted sum of {:,} atoms in 3D.".format(
            len(X)
        )
    )

    # We return a (normalized) vector of weights + a "list" of points
    return tensor(S / np.sum(S)), tensor(X)


################################################################################
# Synthetic sphere - a typical source measure:


def create_sphere(n_samples=1000):
    """Creates a uniform sample on the unit sphere."""
    n_samples = int(n_samples)

    indices = np.arange(0, n_samples, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_samples)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    points = np.vstack((x, y, z)).T
    weights = np.ones(n_samples) / n_samples

    return tensor(weights), tensor(points)


############################################################
# Simple (slow) display routine:


def display_cloud(ax, measure, color):
    w_i, x_i = numpy(measure[0]), numpy(measure[1])

    ax.view_init(elev=110, azim=-90)
    # ax.set_aspect('equal')

    weights = w_i / w_i.sum()
    ax.scatter(x_i[:, 0], x_i[:, 1], x_i[:, 2], s=25 * 500 * weights, c=color)

    ax.axes.set_xlim3d(left=-1.4, right=1.4)
    ax.axes.set_ylim3d(bottom=-1.4, top=1.4)
    ax.axes.set_zlim3d(bottom=-1.4, top=1.4)


#############################################################
# Measuring the error made on the marginal constraints
# ---------------------------------------------------------
#
# Computing the marginals of the implicit transport plan:
#
# .. math::
#   \pi ~&=~ \exp \tfrac{1}{\varepsilon}( f\oplus g - \text{C})~\cdot~ \alpha\otimes\beta,\\
#   \text{i.e.}~~\pi_{x_i \leftrightarrow y_j}~&=~ \exp \tfrac{1}{\varepsilon}( F_i + G_j - \text{C}(x_i,y_j))~\cdot~ \alpha_i \beta_j.
#
#


from pykeops.torch import LazyTensor


def plan_marginals(blur, a_i, x_i, b_j, y_j, F_i, G_j):
    """Returns the marginals of the transport plan encoded in the dual vectors F_i and G_j."""

    x_i = LazyTensor(x_i[:, None, :])
    y_j = LazyTensor(y_j[None, :, :])
    F_i = LazyTensor(F_i[:, None, None])
    G_j = LazyTensor(G_j[None, :, None])

    # Cost matrix:
    C_ij = ((x_i - y_j) ** 2).sum(-1) / 2

    # Scaled kernel matrix:
    K_ij = ((F_i + G_j - C_ij) / blur**2).exp()

    A_i = a_i * (K_ij @ b_j)  # First marginal
    B_j = b_j * (K_ij.t() @ a_i)  # Second marginal

    return A_i, B_j


########################################################
# Compare the marginals using the relevant kernel norm
#
# .. math::
#   \|\alpha - \beta\|^2_{k_\varepsilon} ~=~
#   \langle \alpha - \beta , k_\varepsilon \star (\alpha -\beta) \rangle,
#
# with :math:`k_\varepsilon(x,y) = \exp(-\text{C}(x,y)/\varepsilon)`.
#


def blurred_relative_error(blur, x_i, a_i, A_i):
    """Computes the relative error |A_i-a_i| / |a_i| with respect to the kernel norm k_eps."""

    x_j = LazyTensor(x_i[None, :, :])
    x_i = LazyTensor(x_i[:, None, :])

    C_ij = ((x_i - x_j) ** 2).sum(-1) / 2
    K_ij = (-C_ij / blur**2).exp()

    squared_error = (A_i - a_i).dot(K_ij @ (A_i - a_i))
    squared_norm = a_i.dot(K_ij @ a_i)

    return (squared_error / squared_norm).sqrt()


##############################################################################
# Simple error routine:


def marginal_error(blur, a_i, x_i, b_j, y_j, F_i, G_j, mode="blurred"):
    """Measures how well the transport plan encoded in the dual vectors F_i and G_j satisfies the marginal constraints."""

    A_i, B_j = plan_marginals(blur, a_i, x_i, b_j, y_j, F_i, G_j)

    if mode == "TV":
        # Return the (average) total variation error on the marginal constraints:
        return ((A_i - a_i).abs().sum() + (B_j - b_j).abs().sum()) / 2

    elif mode == "blurred":
        # Use the kernel norm k_eps to measure the discrepancy
        norm_x = blurred_relative_error(blur, x_i, a_i, A_i)
        norm_y = blurred_relative_error(blur, y_j, b_j, B_j)
        return (norm_x + norm_y) / 2

    else:
        raise NotImplementedError()


#############################################################
# Computing the entropic Wasserstein distance
# ---------------------------------------------------------
#
# Computing the transport cost, assuming that the dual vectors satisfy
# the equations at optimality:
#
# .. math::
#   \text{OT}_\varepsilon(\alpha,\beta)~=~ \langle \alpha, f^\star\rangle + \langle \beta, g^\star \rangle.
#


def transport_cost(a_i, b_j, F_i, G_j):
    """Returns the entropic transport cost associated to the dual variables F_i and G_j."""
    return a_i.dot(F_i) + b_j.dot(G_j)


##############################################################################
# Compute the "entropic Wasserstein distance"
#
# .. math::
#   \text{D}_\varepsilon(\alpha,\beta)~=~ \sqrt{2 \cdot \text{OT}_\varepsilon(\alpha,\beta)},
#
# which is **homogeneous to a distance on the ambient space** and is
# associated to the (biased) Sinkhorn cost :math:`\text{OT}_\varepsilon`
# with cost :math:`\text{C}(x,y) = \tfrac{1}{2}\|x-y\|^2`.


def wasserstein_distance(a_i, b_j, F_i, G_j):
    """Returns the entropic Wasserstein "distance" associated to the dual variables F_i and G_j."""
    return (2 * transport_cost(a_i, b_j, F_i, G_j)).sqrt()


##############################################################################
# Compute all these quantities simultaneously, with a proper clock:


def benchmark_solver(OT_solver, blur, source, target):
    """Returns a (timing, relative error on the marginals, wasserstein distance) triplet for OT_solver(source, target)."""
    a_i, x_i = source
    b_j, y_j = target

    a_i, x_i = a_i.contiguous(), x_i.contiguous()
    b_j, y_j = b_j.contiguous(), y_j.contiguous()

    if x_i.is_cuda:
        torch.cuda.synchronize()
    start = time.time()
    F_i, G_j = OT_solver(a_i, x_i, b_j, y_j)
    if x_i.is_cuda:
        torch.cuda.synchronize()
    end = time.time()

    F_i, G_j = F_i.view(-1), G_j.view(-1)

    return (
        end - start,
        marginal_error(blur, a_i, x_i, b_j, y_j, F_i, G_j).item(),
        wasserstein_distance(a_i, b_j, F_i, G_j).item(),
    )


#############################################################
# Benchmarking a collection of OT solvers
# ---------------------------------------------------------
#


def benchmark_solvers(
    name,
    OT_solvers,
    source,
    target,
    ground_truth,
    blur=0.01,
    display=False,
    maxtime=None,
):
    timings, errors, costs = [], [], []
    break_loop = False
    print(
        'Benchmarking the "{}" family of OT solvers - ground truth = {:.6f}:'.format(
            name, ground_truth
        )
    )
    for i, OT_solver in enumerate(OT_solvers):
        try:
            timing, error, cost = benchmark_solver(OT_solver, blur, source, target)

            timings.append(timing)
            errors.append(error)
            costs.append(cost)
            print(
                "{}-th solver : t = {:.4f}, error on the constraints = {:.3f}, cost = {:.6f}".format(
                    i + 1, timing, error, cost
                )
            )

        except RuntimeError:
            print("** Memory overflow ! **")
            break_loop = True
            timings.append(np.nan)
            errors.append(np.nan)
            costs.append(np.nan)

        if break_loop or (maxtime is not None and timing > maxtime):
            not_performed = len(OT_solvers) - (i + 1)
            timings += [np.nan] * not_performed
            errors += [np.nan] * not_performed
            costs += [np.nan] * not_performed
            break
    print("")

    timings, errors, costs = np.array(timings), np.array(errors), np.array(costs)

    if display:  # Fancy display
        fig = plt.figure(figsize=(12, 8))

        ax_1 = fig.subplots()
        ax_1.set_title(
            'Benchmarking "{}"\non a {:,}-by-{:,} entropic OT problem, with a blur radius of {:.3f}'.format(
                name, len(source[0]), len(target[0]), blur
            )
        )
        ax_1.set_xlabel("time (s)")

        ax_1.plot(timings, errors, color="b")
        ax_1.set_ylabel("Relative error on the marginal constraints", color="b")
        ax_1.tick_params("y", colors="b")
        ax_1.set_yscale("log")
        ax_1.set_ylim(bottom=1e-5)

        ax_2 = ax_1.twinx()

        ax_2.plot(timings, abs(costs - ground_truth) / ground_truth, color="r")
        ax_2.set_ylabel("Relative error on the cost value", color="r")
        ax_2.tick_params("y", colors="r")
        ax_2.set_yscale("log")
        ax_2.set_ylim(bottom=1e-5)

    return timings, errors, costs
