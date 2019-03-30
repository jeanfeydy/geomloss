"""
Optimal Transport in high dimension
====================================

Let's use a custom clustering scheme to generalize the
multiscale Sinkhorn algorithm to high-dimensional settings.
"""



##############################################
# Setup
# ---------------------

import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from geomloss import SamplesLoss

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def display_4d_samples(ax1, ax2, x, color) :
    x_ = x.detach().cpu().numpy()
    if not type(color) in [str, list] : color = color.detach().cpu().numpy()
    ax1.scatter( x_[:,0], x_[:,1], 25*500 / len(x_), color, edgecolors='none', cmap="tab10" )
    ax2.scatter( x_[:,2], x_[:,3], 25*500 / len(x_), color, edgecolors='none', cmap="tab10" )


###############################################
# Dataset
# ~~~~~~~~~~~~~~~~~~
#
# Our source and target samples are drawn from measures whose densities
# are stored in simple PNG files. They allow us to define a pair of discrete 
# probability measures:
#
# .. math::
#   \alpha ~=~ \frac{1}{N}\sum_{i=1}^N \delta_{x_i}, ~~~
#   \beta  ~=~ \frac{1}{M}\sum_{j=1}^M \delta_{y_j}.

N, M = (100, 100) if not use_cuda else (10000, 10000)

# Generate some kind of 4d-helix:
t = torch.linspace(0, 2*np.pi, N).type(dtype)
X_i = torch.stack((
    t * (2*t).cos() / 7,
    t * (2*t).sin() / 7,
    t / 7,
    t**2 / 50
    )).t().contiguous()
X_i = X_i + .05 * torch.randn(N, 4).type(dtype)

# The y_j's are sampled non-uniformly on the unit sphere of R^4:
Y_j = torch.randn(M, 4).type(dtype)
Y_j[:,0] += 2
Y_j = Y_j / (1e-4 + Y_j.norm(dim=1, keepdim=True))

#######################################
# Display our 4d-samples using two 2d-views:

plt.figure(figsize=(12,6))

ax1 = plt.subplot(1,2,1) ; plt.title("Dimensions 0, 1")
ax2 = plt.subplot(1,2,2) ; plt.title("Dimensions 2, 3")

display_4d_samples(ax1, ax2, X_i, [(.95,.55,.55)] )
display_4d_samples(ax1, ax2, Y_j, [(.55,.55,.95)] )


###############################################
# K-means clustering
# -------------------------------
# 

from pykeops.torch import generic_argmin

def KMeans(x, K=10, Niter=10, verbose=True):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # Define our KeOps kernel:
    nn_search = generic_argmin(
        'SqDist(x,y)',           # A simple squared L2 distance
        'ind = Vi(1)',           # Output one index per line
        'x = Vi({})'.format(D),  # 1st arg: one point per line
        'y = Vj({})'.format(D))  # 2nd arg: one point per column
    
    # K-means loop:
    # - x  is the point cloud, 
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()

    # Simplistic random initialization:
    perm = torch.randperm(N)
    idx = perm[:K]
    c = x[idx, :].clone()  

    for i in range(Niter):
        cl  = nn_search(x,c).view(-1)  # Points -> Nearest cluster
        Ncl = torch.bincount(cl).type(dtype)  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    end = time.time()
    if verbose: print("KMeans performed in {:.3f}s.".format(end-start))

    return cl, c

lab_i, c_i = KMeans(X_i, K=100)
lab_j, c_j = KMeans(Y_j, K=400)


###############################################
# Estimate the average cluster size:

std_i = (( X_i - c_i[lab_i, :] )**2).sum(1).mean().sqrt()
std_j = (( Y_j - c_j[lab_j, :] )**2).sum(1).mean().sqrt()

print("Our clusters have standard deviations of {:.5f} and {:.5f}.".format(std_i, std_j))

###############################################
# Display our clusters:

plt.figure(figsize=(12,12))

ax1 = plt.subplot(2,2,1) ; plt.title("Dimensions 0, 1")
ax2 = plt.subplot(2,2,2) ; plt.title("Dimensions 2, 3")
ax3 = plt.subplot(2,2,3) ; plt.title("Dimensions 0, 1")
ax4 = plt.subplot(2,2,4) ; plt.title("Dimensions 2, 3")

display_4d_samples(ax1, ax2, X_i, lab_i )
display_4d_samples(ax3, ax4, Y_j, lab_j )


###############################################
# Multiscale Sinkhorn algorithm
# -------------------------------
# 

from geomloss import SamplesLoss

Loss =  SamplesLoss("sinkhorn", p=2, blur=.01, scaling=.9) 

Wass_xy = Loss( X_i, Y_j )

print(Wass_xy.item())


###############################################
# Multiscale Sinkhorn algorithm
# -------------------------------
# 

Loss =  SamplesLoss("sinkhorn", p=2, blur=.01, scaling=.9,
                    cluster_scale = max(std_i, std_j), verbose=True) 

a_i = torch.ones(N).type(dtype) / N
b_j = torch.ones(M).type(dtype) / M

Wass_xy = Loss( lab_i, a_i, X_i, lab_j, b_j, Y_j )
print(Wass_xy.item())


plt.show()