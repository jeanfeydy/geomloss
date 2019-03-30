"""
Transfer of labels with Optimal Transport
============================================

Let's use a regularized Optimal Transport plan
to transfer labels from one point cloud to another.
"""



##############################################
# Setup
# ---------------------
# Standard imports:

import numpy as np
import matplotlib.pyplot as plt

import time

import torch
from geomloss import SamplesLoss

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

###############################################
# Display routines:

import numpy as np
import torch
import imageio
from matplotlib import pyplot as plt


def load_image(fname) :
    img = imageio.imread(fname)[::-1,:,:3]  # RGB, without Alpha channel
    return img / 255.                       # Normalized to [0,1]


def display_samples(ax, x, color="black") :
    x_ = x.detach().cpu().numpy()
    if type(color) is not str : color = color.detach().cpu().numpy()
    ax.scatter( x_[:,0], x_[:,1], 25*500 / len(x_), color, edgecolors='none' )


###############################################
# Draw labeled samples from an RGB image:

from random import choices

def draw_samples(fname, n, dtype=torch.FloatTensor, labels=False) :
    A = load_image(fname)
    xg, yg = np.meshgrid( np.arange(A.shape[0]), np.arange(A.shape[1]) )
    
    # Draw random coordinates according to the input density:
    A_gray = (1 - A).sum(2)
    grid = list( zip(xg.ravel(), yg.ravel()) )
    dens = A_gray.ravel() / A_gray.sum()
    dots = np.array( choices(grid, dens, k=n ) )

    # Pick the correct labels:
    if labels: labs = A[ dots[:,1], dots[:,0] ].reshape((n,3))

    # Normalize the coordinates to fit in the unit square, and add some noise
    dots = (dots.astype(float) + .5) / np.array([A.shape[0], A.shape[1]])
    dots += (.5/A.shape[0]) * np.random.standard_normal(dots.shape)

    if labels: return torch.from_numpy(dots).type(dtype), torch.from_numpy(labs).type(dtype)
    else:      return torch.from_numpy(dots).type(dtype)


###############################################
# Dataset
# -------------------------
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
# In this tutorial, we endow the :math:`y_j`'s with **color labels**,
# encoded as one-hot vectors :math:`\ell_j` which are equal to:
#
# - :math:`(1,0,0)` for **red** points,
# - :math:`(0,1,0)` for **green** points,
# - :math:`(0,0,1)` for **blue** points.
#
# In the next few paragraphs, we'll see how to use **regularized Optimal Transport plans**
# to transfer these labels from the :math:`y_j`'s onto the :math:`x_i`'s.
# But first, let's display our **source** (labeled) and **target** point clouds:


plt.figure(figsize=(8,8)) ; ax = plt.gca()
ax.scatter( [10], [10] ) # shameless hack to prevent a slight change of axis...

# Fancy display:
display_samples(ax, Y_j, l_j)  
display_samples(ax, X_i)
ax.set_title("Source (Labeled) and Target  point clouds")

ax.axis([0,1,0,1]) ; ax.set_aspect('equal', adjustable='box')
ax.set_xticks([], []); ax.set_yticks([], []) ; plt.tight_layout()


###############################################
# Regularized Optimal Transport
# -------------------------------
# 

blur = .01
OT_solver = SamplesLoss("sinkhorn", p=2, blur=blur, scaling=.9, debias=False, potentials=True)
F_i, G_j = OT_solver(X_i, Y_j)


###############################################
# Transfer of labels:

from pykeops.torch import generic_sum

transfer = generic_sum(
    "Exp( (F_i + G_j - IntInv(2)*SqDist(X_i,Y_j)) / E ) * L_j",
    "Label = Vi(3)",
    "E = Pm(1)",
    "X_i = Vi(2)",
    "Y_j = Vj(2)",
    "F_i = Vi(1)",
    "G_j = Vj(1)",
    "L_j = Vj(3)"
)

labels_i = transfer(torch.Tensor( [blur**2] ).type(dtype), X_i, Y_j, 
                    F_i.view(-1,1), G_j.view(-1,1), l_j ) / M


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



###############################################
# Unbalanced Optimal Transport
# -------------------------------
# 

blur = .01
OT_solver = SamplesLoss("sinkhorn", p=2, blur=blur, reach=.2, scaling=.9, debias=False, potentials=True)
F_i, G_j = OT_solver(X_i, Y_j)

labels_i = transfer(torch.Tensor( [blur**2] ).type(dtype), X_i, Y_j, 
                    F_i.view(-1,1), G_j.view(-1,1), l_j ) / M


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