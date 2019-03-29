"""
Transfer of labels with Optimal Transport
============================================

Let's use a regularized Optimal Transport plan
to transfer labels from one point cloud to another.
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

###############################################
# Display routines
# ~~~~~~~~~~~~~~~~~

import numpy as np
import torch
from random import choices
import imageio
from matplotlib import pyplot as plt


def load_image(fname) :
    img = imageio.imread(fname)[::-1,:,:3]  # RGB
    return img / 255.         # Normalized to [0,1]

def draw_samples(fname, n, dtype=torch.FloatTensor, labels=False) :
    A = load_image(fname)
    xg, yg = np.meshgrid( np.arange(A.shape[0]), np.arange(A.shape[1]) )
    
    A_gray = (1 - A).sum(2)
    grid = list( zip(xg.ravel(), yg.ravel()) )
    dens = A_gray.ravel() / A_gray.sum()
    dots = np.array( choices(grid, dens, k=n ) )

    if labels: labs = A[ dots[:,1], dots[:,0] ].reshape((n,3))

    dots = (dots.astype(float) + .5) / np.array([A.shape[0], A.shape[1]])
    dots += (.5/A.shape[0]) * np.random.standard_normal(dots.shape)

    if labels:
        return torch.from_numpy(dots).type(dtype), torch.from_numpy(labs).type(dtype)
    else:
        return torch.from_numpy(dots).type(dtype)

def display_samples(ax, x, color="black") :
    x_ = x.detach().cpu().numpy()
    ax.scatter( x_[:,0], x_[:,1], 25*500 / len(x_), color, edgecolors='none' )



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

N, M = (500, 500) if not use_cuda else (10000, 10000)
 
X_i      = draw_samples("data/threeblobs_a.png", N, dtype)
Y_j, l_j = draw_samples("data/threeblobs_b.png", M, dtype, labels=True)


###############################################
# Fancy display:

plt.figure(figsize=(8,8)) ; ax = plt.gca()
ax.scatter( [10], [10] ) # shameless hack to prevent a slight change of axis...

display_samples(ax, Y_j, l_j)
display_samples(ax, X_i)

ax.set_title("Source and Target (Labeled) point clouds")

ax.axis([0,1,0,1]) ; ax.set_aspect('equal', adjustable='box')
ax.set_xticks([], []); ax.set_yticks([], []) ; plt.tight_layout()


###############################################
# Regularized Optimal Transport
# -------------------------------
# 

blur = .05
OT_solver = SamplesLoss("sinkhorn", p=2, blur=blur, scaling=.9, debias=False, potentials=True)
a_y, b_x = OT_solver(X_i, Y_j)


###############################################
# Transfer of labels:

from pykeops.torch import generic_sum

transfer = generic_sum(
    "Exp( (B + A - IntInv(2)*SqDist(X,Y)) / E ) * L",
    "Label = Vi(3)",
    "E = Pm(1)",
    "X = Vi(2)",
    "Y = Vj(2)",
    "A = Vj(1)",
    "B = Vi(1)",
    "L = Vj(3)"
)

labels_i = transfer(torch.Tensor( [blur**2] ).type(dtype), X_i, Y_j, 
                    a_y.view(-1,1), b_x.view(-1,1), l_j ) / M


print( (labels_i.sum(1) - 1).abs().mean().item() )

###############################################
# Fancy display:

plt.figure(figsize=(8,8)) ; ax = plt.gca()
ax.scatter( [10], [10] ) # shameless hack to prevent a slight change of axis...

display_samples(ax, Y_j, l_j)
display_samples(ax, X_i, labels_i.clamp(0,1))

ax.set_title("Source and Target (Labeled) point clouds")

ax.axis([0,1,0,1]) ; ax.set_aspect('equal', adjustable='box')
ax.set_xticks([], []); ax.set_yticks([], []) ; plt.tight_layout()


plt.show()