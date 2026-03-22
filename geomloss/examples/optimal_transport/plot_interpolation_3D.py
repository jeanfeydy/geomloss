"""Creating a fancy interpolation video between 3D meshes.
==============================================================

N.B.: I am currently very busy writing my PhD thesis. Comments will come soon!
"""


################################################################################
# Setup
# ----------------------
#
# Standard imports.
#


import numpy as np
import torch
import os

use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
numpy = lambda x: x.detach().cpu().numpy()

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from geomloss import SamplesLoss
from pykeops.torch import LazyTensor


################################################################################
# Utility: turn a triangle mesh into a weighted point cloud.


def to_measure(points, triangles):
    """Turns a triangle into a weighted point cloud."""

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
# Utility: load ".ply" mesh file.
#

from plyfile import PlyData, PlyElement


def load_ply_file(fname):
    """Loads a .ply mesh to return a collection of weighted Dirac atoms: one per triangle face."""

    # Load the data, and read the connectivity information:
    plydata = PlyData.read(fname)
    triangles = np.vstack(plydata["face"].data["vertex_indices"])

    # Normalize the point cloud, as specified by the user:
    points = np.vstack([[v[0], v[1], v[2]] for v in plydata["vertex"]])

    return to_measure(points, triangles)


################################################################################
# Utility: load ".nii" volume file.
#

import SimpleITK as sitk
from skimage.measure import marching_cubes


def load_nii_file(fname, threshold=0.5):
    """Uses the marching cube algorithm to turn a .nii binary mask into a surface weighted point cloud."""

    mask = sitk.GetArrayFromImage(sitk.ReadImage(fname))
    # mask = skimage.transform.downscale_local_mean(mask, (4,4,4))
    verts, faces, normals, values = marching_cubes(mask, threshold)

    return to_measure(verts, faces)


################################################################################
# Synthetic sphere - a typical source measure:
#


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
#


def display_cloud(ax, measure, color):
    w_i, x_i = numpy(measure[0]), numpy(measure[1])

    ax.view_init(elev=110, azim=-90)
    # ax.set_aspect('equal')

    weights = w_i / w_i.sum()
    ax.scatter(x_i[:, 0], x_i[:, 1], x_i[:, 2], s=25 * 500 * weights, c=color)

    ax.axes.set_xlim3d(left=-1.4, right=1.4)
    ax.axes.set_ylim3d(bottom=-1.4, top=1.4)
    ax.axes.set_zlim3d(bottom=-1.4, top=1.4)


############################################################
# Save the output as a VTK folder, to be rendered with Paraview:

folder = "output/wasserstein_3D/"
os.makedirs(os.path.dirname("output/wasserstein_3D/"), exist_ok=True)

import pyvista as pv


def save_vtk(fname, points, colors):
    """N.B.: Paraview is a good VTK viewer, which supports ray-tracing."""

    # Use PyVista to save the point cloud as a VTK file:
    points = pv.PolyData(points)
    points["colors"] = colors
    points.save(folder + fname)


#################################################################
# Data
# ----------------
#
# Shall we work on subsampled data or at full resolution?

fast_demo = False if use_cuda else True

if use_cuda:
    Npoints = 1e4 if fast_demo else 2e5
else:
    Npoints = 1e3

##############################################################
# Create a reference template:

template = create_sphere(Npoints)

#################################################
# Use color labels to track the particles:
#

K = 12
colors = (K * template[1][:, 0]).cos()
colors = colors.view(-1).detach().cpu().numpy()


#################################################
# Fetch the data:
#


os.makedirs(os.path.dirname("data/"), exist_ok=True)
if not os.path.exists("data/wasserstein_3D_models/Stanford_dragon_200k.ply"):
    print("Fetching the data... ", end="", flush=True)
    import urllib.request

    urllib.request.urlretrieve(
        "http://www.kernel-operations.io/data/wasserstein_3D_models.zip",
        "data/wasserstein_3D_models.zip",
    )

    import shutil

    shutil.unpack_archive("data/wasserstein_3D_models.zip", "data")
    print("Done.")


#############################################################
# Load the data on the GPU:


print("Loading the data:")
# N.B.: Since Plyfile is far from being optimized, this may take some time!
targets = [
    load_ply_file("data/wasserstein_3D_models/Stanford_dragon_200k.ply"),
    load_ply_file("data/wasserstein_3D_models/vertebrae_400k_biol260_sketchfab_CC.ply"),
    load_nii_file("data/wasserstein_3D_models/brain.nii.gz"),
]

#################################################################
# Normalize and subsample everyone, if required:


def normalize(measure, n=None):
    """Reduce a point cloud to at most n points and normalize the weights and point cloud."""
    weights, locations = measure
    N = len(weights)

    if n is not None and n < N:
        n = int(n)
        indices = torch.randperm(N)
        indices = indices[:n]
        weights, locations = weights[indices], locations[indices]

    weights = weights / weights.sum()
    weights, locations = weights.contiguous(), locations.contiguous()

    # Center, normalize the point cloud
    mean = (weights.view(-1, 1) * locations).sum(dim=0)
    locations -= mean
    std = (weights.view(-1) * (locations**2).sum(dim=1).view(-1)).sum().sqrt()
    locations /= std

    return weights, locations


targets = [normalize(t, n=Npoints) for t in targets]

########################################################################
# Fine tuning:

template = template[0], template[1] / 2 + tensor(
    [0.5, 0.0, 0.0]
)  # Smaller sphere, towards the back of the dragon
targets[1] = targets[1][0], targets[1][1] @ tensor(
    [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
)  # Turn the vertebra
targets[2] = targets[2][0], -targets[2][1]  # Flip the brain

#########################################################################
# Optimal Transport matchings
# --------------------------------
#
# Define our solver:


import time

Loss = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=0.5, truncate=1)


def OT_registration(source, target, name):
    a, x = source  # weights, locations
    b, y = target  # weights, locations

    x.requires_grad = True
    z = x.clone()  # Moving point cloud

    if use_cuda:
        torch.cuda.synchronize()
    start = time.time()

    nits = 4 if fast_demo else 10

    for it in range(nits):
        wasserstein_zy = Loss(a, z, b, y)
        [grad_z] = torch.autograd.grad(wasserstein_zy, [z])
        z -= grad_z / a[:, None]  # Apply the regularized Brenier map

        # save_vtk(f"matching_{name}_it_{it}.vtk", numpy(z), colors)

    end = time.time()
    print("Registered {} in {:.3f}s.".format(name, end - start))

    return z


#################################################################
# Register the source onto the targets:
#

matchings = [
    OT_registration(template, target, f"shape{i+1}")
    for (i, target) in enumerate(targets)
]

#################################################################
# Display our matchings:

for i, (matching, target) in enumerate(zip(matchings, targets)):
    fig = plt.figure(figsize=(6, 6))
    plt.set_cmap("hsv")

    ax = fig.add_subplot(1, 1, 1, projection="3d")

    display_cloud(ax, (template[0], matching), colors)
    display_cloud(ax, target, "blue")
    ax.set_title(
        "Registered (N={:,}) and target {} (M={:,}) point clouds".format(
            len(matching), i + 1, len(target[0])
        )
    )
    plt.tight_layout()


#################################################################
# Movie
# -------------
#
# Save them as a collection of VTK files:

FPS = 32 if fast_demo else 32

source = template[1]
pairs = [
    (source, source),
    (source, matchings[0]),
    (matchings[0], matchings[0]),
    (matchings[0], matchings[1]),
    (matchings[1], matchings[1]),
    (matchings[1], matchings[2]),
    (matchings[2], matchings[2]),
    (matchings[2], source),
]

frame = 0

print("Save as a VTK movie...", end="", flush=True)
for A, B in pairs:
    A, B = numpy(A), numpy(B)
    for t in np.linspace(0, 1, FPS):
        save_vtk(f"frame_{frame}.vtk", (1 - t) * A + t * B, colors)
        frame += 1

print("Done.")
plt.show()
