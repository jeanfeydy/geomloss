"""
Gradient flows in 1D
====================

This example showcases the properties of **kernel MMDs**, **Hausdorff**
and **Sinkhorn** divergences on a simple toy problem:
the registration of an interval onto another.
"""



##############################################
# Setup
# ---------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity  # display as density curves

import torch
from geomloss import SamplesLoss

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

###############################################
# Display routine
# ~~~~~~~~~~~~~~~~~
 
t_plot = np.linspace(-0.1, 1.1, 1000)[:,np.newaxis]

def display_samples(ax, x, color):
    """Displays samples on the unit interval using a density curve."""
    kde  = KernelDensity(kernel='gaussian', bandwidth= .005 ).fit(x.data.cpu().numpy())
    dens = np.exp( kde.score_samples(t_plot) )
    dens[0] = 0 ; dens[-1] = 0
    ax.fill(t_plot, dens, color=color)


###############################################
# Dataset
# ~~~~~~~~~~~~~~~~~~
#
# Our source and target samples are drawn from intervals of the real line
# and define discrete probability measures:
#
# .. math::
#   \alpha ~=~ \frac{1}{N}\sum_{i=1}^N \delta_{x_i}, ~~~
#   \beta  ~=~ \frac{1}{M}\sum_{j=1}^M \delta_{y_j}.

N, M = (250, 250) if not use_cuda else (10000, 10000)
 
t_i = torch.linspace(0, 1, N).type(dtype).view(-1,1)
t_j = torch.linspace(0, 1, M).type(dtype).view(-1,1)

X_i, Y_j = 0.2 * t_i,  0.4 * t_j + 0.6

###############################################
# Wasserstein gradient flow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# To study the influence of the :math:`\text{Loss}` function in measure-fitting
# applications, we perform gradient descent on the positions
# :math:`x_i` of the samples that make up :math:`\alpha`
# as we minimize the cost :math:`\text{Loss}(\alpha,\beta)`.
# This procedure can be understood as a discrete (Lagrangian) 
# `Wasserstein gradient flow <https://arxiv.org/abs/1609.03890>`_
# and as a "model-free" machine learning program, where
# we optimize directly on the samples' locations.
 
def gradient_flow(loss, lr=.05) :
    """Flows along the gradient of the cost function, using a simple Euler scheme.
    
    Parameters:
        loss ((x_i,y_j) -> torch float number): 
            Real-valued loss function.
        lr (float, default = .05):
            Learning rate, i.e. time step.
    """
    
    # Parameters for the gradient descent
    Nsteps = int(5/lr)+1 
    display_its = [int(t/lr) for t in [0, .25, .50, 1., 2., 5.]]
    
    # Make sure that we won't modify the reference samples
    x_i, y_j = X_i.clone(), Y_j.clone()

    # We're going to perform gradient descent on Loss(α, β) 
    # wrt. the positions x_i of the diracs masses that make up α:
    x_i.requires_grad = True  
    
    plt.figure(figsize=(12,8)) ; k = 1
    for i in range(Nsteps): # Euler scheme ===============
        # Compute cost and gradient
        L_αβ = loss(x_i, y_j)
        [g]  = torch.autograd.grad(L_αβ, [x_i])

        if i in display_its : # display
            ax = plt.subplot(2,3,k) ; k = k+1

            display_samples(ax, y_j, (.55,.55,.95))
            display_samples(ax, x_i, (.95,.55,.55))
            
            ax.set_title("t = {:1.2f}".format(lr*i))
            plt.axis([-.1,1.1,-.1,5.5])
            plt.xticks([], []); plt.yticks([], [])
            plt.tight_layout()
        
        # in-place modification of the tensor's values
        x_i.data -= lr * len(x_i) * g 



###############################################
# Kernel norms, MMDs
# ------------------------------------
#
# Gaussian MMD
# ~~~~~~~~~~~~~~~
# 
# The smooth Gaussian kernel
# :math:`k(x,y) = \exp(-\|x-y\|^2/2\sigma^2)`
# is blind to details which are smaller than the blurring scale :math:`\sigma`:
# its gradient stops being informative when :math:`\alpha`
# and :math:`\beta` become equal "up to the high frequencies".

gradient_flow( SamplesLoss("gaussian", blur=.5) )


###############################################
# On the other hand, if the radius :math:`\sigma`
# of the kernel is too small, particles :math:`x_i`
# won't be attracted to the target, and may **spread out**
# to minimize the auto-correlation term 
# :math:`\tfrac{1}{2}\langle \alpha, k\star\alpha\rangle`.

gradient_flow( SamplesLoss("gaussian", blur=.1) )


###############################################
# Laplacian MMD
# ~~~~~~~~~~~~~~~~
#
# The pointy exponential kernel
# :math:`k(x,y) = \exp(-\|x-y\|/\sigma)`
# tends to provide a better fit, but tends to zero at infinity
# and is still very prone to **screening artifacts**.

gradient_flow( SamplesLoss("laplacian", blur=.1) )


###############################################
# Energy Distance MMD
# ~~~~~~~~~~~~~~~~~~~~~~
# 
# The scale-equivariant kernel
# :math:`k(x,y)=-\|x-y\|` provides a robust baseline:
# the Energy Distance.


# sphinx_gallery_thumbnail_number = 4
gradient_flow( SamplesLoss("energy") )