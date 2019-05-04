"""
2) Kernel truncation, log-linear runtimes
=====================================================

In the previous notebook, we've seen that **simulated annealing**
could be used to define efficient coarse-to-fine solvers
of the entropic :math:`\\text{OT}_\\varepsilon` problem.
Adapting ideas from `(Schmitzer, 2016) <https://arxiv.org/abs/1610.06519>`_,
we now explain how the :mod:`SamplesLoss("sinkhorn", backend="multiscale") <geomloss.SamplesLoss>`
layer combines this strategy with a **multiscale encoding of the input measures** to
compute Sinkhorn divergences in :math:`O(n \log(n))` times, on the GPU.
"""

##################################################
# Multiscale Optimal Transport
# -----------------------------
#
# Starting with the seminal work of `Quentin Mérigot <http://quentin.mrgt.fr/>`_
# `(Mérigot, 2011) <https://hal.archives-ouvertes.fr/hal-00604684>`_
# 
# .. warning::
#   The recent line of Stats-ML papers started by `(Cuturi, 2013) <https://arxiv.org/abs/1306.0895>`_
#   has prioritized the study of the **statistical properties** of entropic OT
#   over computational efficiency.
#   Consequently, in spite of their impact on
#   `fluid mechanics <https://arxiv.org/abs/1505.03306>`_,
#   `computer graphics <https://arxiv.org/abs/1409.1279>`_ and all fields
#   where a `manifold assumption <https://arxiv.org/abs/1708.02469>`_ 
#   may be done on the input measures,
#   **multiscale methods have been mostly ignored by authors in the Machine Learning community**.
#
#   By providing a fast discrete OT solver that relies on key ideas from both worlds,
#   GeomLoss aims at **bridging the gap** between these two bodies of work.
#   As researchers become aware of both **geometric** and **statistical**
#   points of view on discrete OT, we will hopefully converge towards
#   robust, efficient and well-understood generalizations of the Wasserstein distance.
#


##############################################
# Setup
# ---------------------
#
# Standard imports:

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import os
from torch.autograd import grad

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

###############################################
# Display routines:

from imageio import imread


def load_image(fname) :
    img = np.mean( imread(fname), axis=2 )  # Grayscale
    img = (img[::-1, :])  / 255.
    return 1 - img


def draw_samples(fname, sampling, dtype=torch.FloatTensor) :
    A = load_image(fname)
    A = A[::sampling, ::sampling]
    A[A<=0] = 1e-8

    a_i = A.ravel() / A.sum()

    x, y = np.meshgrid( np.linspace(0,1,A.shape[0]), np.linspace(0,1,A.shape[1]) )
    x += .5 / A.shape[0] ; y += .5 / A.shape[1]

    x_i = np.vstack( (x.ravel(), y.ravel()) ).T

    return torch.from_numpy(a_i).type(dtype), \
           torch.from_numpy(x_i).contiguous().type(dtype)


def display_potential(ax, F, color, nlines=21):
    # Assume that the image is square...
    N = int( np.sqrt(len(F)) )  
    F = F.view(N,N).detach().cpu().numpy()
    F = np.nan_to_num(F)

    # And display it with contour lines:
    levels = np.linspace(-1, 1, nlines)
    ax.contour(F, origin='lower', linewidths = 2., colors = color,
               levels = levels, extent=[0,1,0,1]) 


def display_samples(ax, x, weights, color, v=None) :
    x_ = x.detach().cpu().numpy()
    weights_ = weights.detach().cpu().numpy()

    weights_[weights_ < 1e-5] = 0
    ax.scatter( x_[:,0], x_[:,1], 10 * 500 * weights_, color, edgecolors='none' )

    if v is not None :
        v_ = v.detach().cpu().numpy()
        ax.quiver( x_[:,0], x_[:,1], v_[:,0], v_[:,1], 
                    scale = 1, scale_units="xy", color="#5CBF3A", 
                    zorder= 3, width= 2. / len(x_) )

###############################################
# Dataset
# --------------
#
# Our source and target samples are drawn from measures whose densities
# are stored in simple PNG files. They allow us to define a pair of discrete 
# probability measures:
#
# .. math::
#   \alpha ~=~ \sum_{i=1}^N \alpha_i\,\delta_{x_i}, ~~~
#   \beta  ~=~ \sum_{j=1}^M \beta_j\,\delta_{y_j}.

sampling = 10 if not use_cuda else 2

A_i, X_i = draw_samples("data/ell_a.png", sampling)
B_j, Y_j = draw_samples("data/ell_b.png", sampling)

###############################################
# Scaling heuristic
# -------------------
#
# We now display the behavior of the Sinkhorn loss across
# our iterations.

from pykeops.torch.cluster import grid_cluster, cluster_ranges_centroids
from geomloss import SamplesLoss


scaling, Nits = .5, 9
cluster_scale = .1 if not use_cuda else .05

plt.figure(figsize=( (12, ((Nits-1)//3 + 1) * 4)))

for i in range(Nits):
    blur = scaling**i
    Loss = SamplesLoss("sinkhorn", p=2, blur=blur, diameter=1., cluster_scale = cluster_scale,
                        scaling=scaling, verbose=True, backend="multiscale")

    # Create a copy of the data...
    a_i, x_i = A_i.clone(), X_i.clone()
    b_j, y_j = B_j.clone(), Y_j.clone()

    # And require grad:
    a_i.requires_grad = True
    x_i.requires_grad = True
    b_j.requires_grad = True

    # Compute the loss + gradients:
    Loss_xy = Loss(a_i, x_i, b_j, y_j)
    [F_i, G_j, dx_i] = grad( Loss_xy, [a_i, b_j, x_i] )

    # The generalized "Brenier map" is (minus) the gradient of the Sinkhorn loss
    # with respect to the Wasserstein metric:
    BrenierMap = - dx_i / (a_i.view(-1,1) + 1e-7)

    # Compute the coarse measures for display ----------------------------------

    x_lab = grid_cluster(x_i, cluster_scale)
    _, x_c, a_c = cluster_ranges_centroids(x_i, x_lab, weights=a_i)

    y_lab = grid_cluster(y_j, cluster_scale)
    _, y_c, b_c = cluster_ranges_centroids(y_j, y_lab, weights=b_j)


    # Fancy display: -----------------------------------------------------------

    ax = plt.subplot( ((Nits-1)//3 + 1) , 3, i+1)
    ax.scatter( [10], [10] )  # shameless hack to prevent a slight change of axis...
    
    display_potential(ax, G_j, "#E2C5C5")
    display_potential(ax, F_i, "#C8DFF9")


    if blur > cluster_scale:
        display_samples(ax, y_j, b_j, [(.55,.55,.95, .2)])
        display_samples(ax, x_i, a_i, [(.95,.55,.55, .2)], v = BrenierMap)
        display_samples(ax, y_c, b_c, [(.55,.55,.95)])
        display_samples(ax, x_c, a_c, [(.95,.55,.55)])

    else:
        display_samples(ax, y_j, b_j, [(.55,.55,.95)])
        display_samples(ax, x_i, a_i, [(.95,.55,.55)], v = BrenierMap)


    ax.set_title("iteration {}, blur = {:.3f}".format(i+1, blur))

    ax.set_xticks([0, 1]) ; ax.set_yticks([0, 1])
    ax.axis([0,1,0,1]) ; ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()
