"""
Transferring labels from a segmented atlas
=============================================

We use a new multiscale algorithm for solving regularized Optimal Transport 
problems on the GPU, with a linear memory footprint. 

We use the resulting smooth assignments to perform label transfer for atlas-based 
segmentation of fiber tractograms. The parameters -- \emph{blur} and \emph{reach} -- 
of our method are meaningful, defining the minimum and maximum distance at which 
two fibers are compared with each other. They can be set according to anatomical knowledge.
"""


##############################################
# Setup
# ---------------------
#
# Standard imports:

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from geomloss import SamplesLoss

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor if use_cuda else torch.LongTensor

###############################################
# Loading and saving data routines
#

from tract_io import read_vtk, streamlines_resample, save_vtk, save_vtk_labels
from tract_io import save_tract, save_tract_numpy
from tract_io import save_tract_with_labels, save_tracts_labels_separate

##############################################
# Dataset
# ---------------------
#
# Fetch data from the KeOps website:

import os


def fetch_file(name):
    if not os.path.exists(f"data/{name}.npy"):
        import urllib.request

        print("Fetching the atlas... ", end="", flush=True)
        urllib.request.urlretrieve(
            f"https://www.kernel-operations.io/data/{name}.npy", f"data/{name}.npy"
        )
        print("Done.")


fetch_file("tracto_atlas")
fetch_file("atlas_labels")
fetch_file("tracto1")


##############################################
# Fibers do not have a canonical orientation. Since our ground distance is a simple
# L2-distance on the sampled fibers, we augment the dataset with the mirror flip
# of all fibers and perform the OT on this augmented dataset.


def torch_load(X, dtype=dtype):
    return torch.from_numpy(X).type(dtype).contiguous()


def add_flips(X):
    """Adds flips and loads on the GPU the input fiber track."""
    #    X = X[:,None,:,:]
    X_flip = torch.flip(X, (1,))
    X = torch.stack((X, X_flip), dim=1)  # (Nfibers, 2, NPOINTS, 3)
    return X


###############################################################################
# Source atlas
# ~~~~~~~~~~~~~~~~~~~
#
# Load atlas (segmented, each fiber has a label):

Y_j = torch_load(np.load("data/tracto_atlas.npy"))
labels_j = torch_load(np.load("data/atlas_labels.npy"), dtype=dtypeint)

###############################################################################
#

M, NPOINTS = Y_j.shape[0], Y_j.shape[1]  # Number of fibers, points per fiber

###############################################################################
#

Y_j = Y_j.view(M, NPOINTS, 3) / np.sqrt(NPOINTS)

###############################################################################
#

Y_j = add_flips(Y_j)  # Shape (M, 2, NPOINTS, 3)
##############################################
# Target subject
# ~~~~~~~~~~~~~~~~~~~~
#
# Load a new subject (unlabelled)
#

X_i = torch_load(np.load("data/tracto1.npy"))
N, NPOINTS_i = X_i.shape[0], X_i.shape[1]  # Number of fibers, points per fiber

if NPOINTS != NPOINTS_i:
    raise ValueError(
        "The atlas and the subject are not sampled with the same number of points: "
        + f"{NPOINTS} and {NPOINTS_i}, respectively."
    )

X_i = X_i.view(N, NPOINTS, 3) / np.sqrt(NPOINTS)
X_i = add_flips(X_i)  # Shape (N, 2, NPOINTS, 3)


##############################################
# Feature engineering
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Add some weight on both ends of our fibers:
#

gamma = 2.0
X_i[:, :, 0, :] *= gamma
X_i[:, :, -1, :] *= gamma
Y_j[:, :, 0, :] *= gamma
Y_j[:, :, -1, :] *= gamma

###############################################################################
# Optimizing performances
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Contiguous memory accesses are critical for performances on the GPU.
#

from pykeops.torch.cluster import sort_clusters, cluster_ranges

ranges_j = cluster_ranges(labels_j)  # Ranges for all clusters
Y_j, labels_j = sort_clusters(
    Y_j, labels_j
)  # Make sure that all clusters are contiguous in memory

C = len(ranges_j)  # Number of classes

if C != labels_j.max() + 1:
    raise ValueError("???")

for j, (start_j, end_j) in enumerate(ranges_j):
    if start_j >= end_j:
        raise ValueError(f"The {j}-th cluster of the atlas seems to be empty.")

###############################################################################
# Each fiber is sampled with 20 points in R^3.
# Thus, one tractogram is a matrix of size n x 60 where n is the number of fibers
# The atlas is labelled, wich means that each fiber belong to a cluster.
# This is summarized by the vector labels_j of size n x 1. labels_j[i] is the label of the fiber i.
# Subsample the data by a factor 4 if you want to reduce the computational time:

subsample = 20 if True else 1


##############################################
#

to_keep = []
for start_j, end_j in ranges_j:
    to_keep += list(range(start_j, end_j, subsample))

Y_j, labels_j = Y_j[to_keep].contiguous(), labels_j[to_keep].contiguous()
ranges_j = cluster_ranges(labels_j)  # Keep the ranges up to date!

##############################################
#

X_i = X_i[::subsample].contiguous()


##############################################
#

N, M = len(X_i), len(Y_j)

print("Data loaded.")


##############################################
# Pre-computing cluster prototypes
# --------------------------------------
#

from pykeops.torch import LazyTensor


def nn_search(x_i, y_j, ranges=None):
    x_i = LazyTensor(x_i[:, None, :])  # Broadcasted "line" variable
    y_j = LazyTensor(y_j[None, :, :])  # Broadcasted "column" variable

    D_ij = ((x_i - y_j) ** 2).sum(-1)  # Symbolic matrix of squared distances
    D_ij.ranges = ranges  # Apply our block-sparsity pattern

    return D_ij.argmin(dim=1).view(-1)


################################################################################
# K-Means loop:
#


def KMeans(x_i, c_j, Nits=10, ranges=None):
    D = x_i.shape[1]
    for i in range(10):
        # Points -> Nearest cluster
        labs_i = nn_search(x_i, c_j, ranges=ranges)
        # Class cardinals:
        Ncl = torch.bincount(labs_i.view(-1)).type(dtype)
        # Compute the cluster centroids with torch.bincount:
        for d in range(D):  # Unfortunately, vector weights are not supported...
            c_j[:, d] = torch.bincount(labs_i.view(-1), weights=x_i[:, d]) / Ncl

    return c_j, labs_i


##############################################
# On the subject
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# For new subject (unlabelled), we perform a simple Kmean
# on R^60 to obtain a cluster of the data.
#

K = 1000

# Pick K fibers at random:
perm = torch.randperm(N)
random_labels = perm[:K]
C_i = X_i[random_labels]  # (K, 2, NPOINTS, 3)

# Reshape our data as "N-by-60" tensors:
C_i_flat = C_i.view(K * 2, NPOINTS * 3)  # Flattened list of centroids
X_i_flat = X_i.view(N * 2, NPOINTS * 3)  # Flattened list of fibers

# Retrieve our new centroids:
C_i_flat, labs_i = KMeans(X_i_flat, C_i_flat)
C_i = C_i_flat.view(K, 2, NPOINTS, 3)
# Standard deviation of our clusters:
std_i = ((X_i_flat - C_i_flat[labs_i.view(-1), :]) ** 2).sum(dim=1).mean().sqrt()


############################################################################################
#
# On the atlas
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# To use the multiscale version of the regularized OT,
# we need to have a cluster of our input data (atlas and new subject).
# For the atlas, the cluster is given by the segmentation. We use a Kmeans to
# separate the fibers and the flips within a cluser, in order to have clusters whose fibers have similar
# orientation
#
ranges_yi = 2 * ranges_j

ranges_cj = 2 * torch.arange(C).type_as(ranges_j)
ranges_cj = torch.stack((ranges_cj, ranges_cj + 2)).t().contiguous()

slices_i = 1 + torch.arange(C).type_as(ranges_j)
ranges_yi_cj = (ranges_yi, slices_i, ranges_cj, ranges_cj, slices_i, ranges_yi)


################################################################################
# Pick one unoriented (i.e. two oriented) fibers per class:

first_labels = ranges_j[:, 0]  # One label per class

C_j = Y_j[first_labels.type(dtypeint), :, :, :]  # (C, 2, NPOINTS, 3)
C_j_flat = C_j.view(C * 2, NPOINTS * 3)  # Flattened list of centroids

############################################################################################
#


Y_j_flat = Y_j.view(M * 2, NPOINTS * 3)
C_j_flat, labs_j = KMeans(Y_j_flat, C_j_flat, ranges=ranges_yi_cj)
C_j = C_j_flat.view(C, 2, NPOINTS, 3)

std_j = ((Y_j_flat - C_j_flat[labs_j.view(-1), :]) ** 2).sum(dim=1).mean().sqrt()


########################################################
# Compute the OT plan with the multiscale algorithm
# ------------------------------------------------------
#
# To use the **multiscale** Sinkhorn algorithm,
# we should simply provide:
#
# - explicit **labels** and **weights** for both input measures,
# - a typical **cluster_scale** which specifies the iteration at which
#   the Sinkhorn loop jumps from a **coarse** to a **fine** representation
#   of the data.
#
blur = 3.0
OT_solver = SamplesLoss(
    "sinkhorn",
    p=2,
    blur=blur,
    reach=20,
    scaling=0.9,
    cluster_scale=max(std_i, std_j),
    debias=False,
    potentials=True,
    verbose=True,
)

############################################################################################
# To specify explicit cluster labels, SamplesLoss also requires
# explicit weights. Let's go with the default option - a uniform distribution:

a_i = torch.ones(2 * N).type(dtype) / (2 * N)
b_j = torch.ones(2 * M).type(dtype) / (2 * M)

start = time.time()

# Compute the dual vectors F_i and G_j:
# 6 args -> labels_i, weights_i, locations_i, labels_j, weights_j, locations_j
F_i, G_j = OT_solver(
    labs_i, a_i, X_i.view(N * 2, NPOINTS * 3), labs_j, b_j, Y_j.view(M * 2, NPOINTS * 3)
)

if use_cuda:
    torch.cuda.synchronize()
end = time.time()

print("OT computed in  in {:.3f}s.".format(end - start))


##############################################
# Use the OT to perform the transfer of labels
# ----------------------------------------------
#
# The transport plan pi_{i,j} gives the probability for
# a fiber i of the subject to be assigned to the (labeled) fiber j of the atlas.
# We assign a label l to the fiber i as the label with maximum probability for all the soft assignement of i.

# Return to the original data (unflipped)
X_i = X_i[:, 0, :, :].contiguous()  # (N, NPOINTS, 3)
F_i = F_i[::2].contiguous()  # (N,)


################################################
# Compute the transport plan:
#

XX_i = LazyTensor(X_i.view(N, 1, NPOINTS * 3))
YY_j = LazyTensor(Y_j.view(1, M * 2, NPOINTS * 3))
FF_i = LazyTensor(F_i.view(N, 1, 1))
GG_j = LazyTensor(G_j.view(1, M * 2, 1))

# Cost matrix:
CC_ij = ((XX_i - YY_j) ** 2).sum(-1) / 2  # (N, M * 2, 1) LazyTensor

# Scaled kernel matrix:
KK_ij = ((FF_i + GG_j - CC_ij) / blur**2).exp()  # (N, M * 2, 1) LazyTensor


################################################
# Transfer the labels, bypassing the one-hot vector encoding
# for the sake of efficiency:


def slicing_ranges(start, end):
    """KeOps does not yet support sliced indexing of LazyTensors, so we have to resort to some black magic..."""
    ranges_i = (
        torch.Tensor([[0, N]]).type(dtypeint).int()
    )  # Int32, on the correct device
    slices_i = torch.Tensor([1]).type(dtypeint).int()
    redranges_j = torch.Tensor([[start, end]]).type(dtypeint).int()
    return (ranges_i, slices_i, redranges_j, redranges_j, slices_i, ranges_i)


weights_i = torch.zeros(C + 1, N).type(torch.FloatTensor)  # C classes + outliers

for c in range(C):
    start, end = 2 * ranges_j[c]
    KK_ij.ranges = slicing_ranges(
        start, end
    )  # equivalent to "PP_ij[:, start:end]", which is not supported yet...
    weights_i[c] = (KK_ij.sum(dim=1).view(N) / (2 * M)).cpu()

weights_i[C] = 0.2  # If no label has a bigger weight than .01, this fiber is an outlier

labels_i = weights_i.argmax(dim=0)  # (N,) vector


################################################
# Save our new cluster information as a signal:

# Come back to the original data
X_i[:, 0, :] /= gamma
X_i[:, -1, :] /= gamma

save_tracts_labels_separate(
    "output/labels_subject", X_i, labels_i, 0, labels_i.max() + 1
)  # save the data
