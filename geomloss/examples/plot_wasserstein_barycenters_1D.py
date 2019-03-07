"""
Wasserstein barycenters in 1D
==================================

Let's compute Wasserstein barycenters
using the Lagrangian and Eulerian formulations
of the Sinkhorn divergence.
"""



##############################################
# Setup
# ---------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity  # display as density curves
import time

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
    kde  = KernelDensity(kernel='gaussian', bandwidth= .002 ).fit(x.data.cpu().numpy())
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
#   \beta  ~=~ \frac{1}{M}\sum_{j=1}^M \delta_{y_j}, ~~~
#   \gamma  ~=~ \frac{1}{M}\sum_{k=1}^M \delta_{z_k}.

N, M = (500, 500) if not use_cuda else (10000, 10000)
 
t_i = torch.linspace(0, 1, N).type(dtype).view(-1,1)
t_j = torch.linspace(0, 1, M).type(dtype).view(-1,1)
t_k = torch.linspace(0, 1, M).type(dtype).view(-1,1)

X_i, Y_j, Z_k = t_i,  .1 * t_j, .1 * t_k + .9

###############################################
# Wasserstein barycenter
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
 
def lagrangian_barycenter(loss, w=.5, lr=.1) :
    """Flows along the gradient of the cost function, using a simple Euler scheme.
    
    Parameters:
        loss ((x_i,y_j) -> torch float number): 
            Real-valued loss function.
        lr (float, default = .05):
            Learning rate, i.e. time step.
    """
    
    # Parameters for the gradient descent
    Nsteps = int(5/lr)+1 
    display_its = [int(t/lr) for t in [0, 1., 5.]]
    
    # Make sure that we won't modify the reference samples
    x_i, y_j, z_k = X_i.clone(), Y_j.clone(), Z_k.clone()

    # We're going to perform gradient descent on w*Loss(x_i,y_j) + (1-w)*Loss(x_i,z_k))
    # wrt. the positions x_i of the diracs masses that make up our source measure:
    x_i.requires_grad = True  
    
    t_0 = time.time()
    plt.figure(figsize=(12,4)) ; k = 1
    for i in range(Nsteps): # Euler scheme ===============
        # Compute cost and gradient
        L_α  = w * loss(x_i, y_j) + (1 - w) * loss(x_i, z_k)
        [g]  = torch.autograd.grad(L_α, [x_i])

        if i in display_its : # display
            ax = plt.subplot(1,3,k) ; k = k+1

            display_samples(ax, y_j, (.55,.55,.95))
            display_samples(ax, z_k, (.95,.55,.55))
            display_samples(ax, x_i, (.55,.95,.55))
            
            ax.set_title("t = {:1.2f}".format(lr*i))
            plt.axis([-.1,1.1,-.1,10.5])
            plt.xticks([], []); plt.yticks([], [])
            plt.tight_layout()
        
        # in-place modification of the tensor's values
        x_i.data -= lr * len(x_i) * g 
    plt.title("t = {:1.2f}, elapsed time: {:.2f}s/it".format(lr*i, (time.time() - t_0)/Nsteps ))



###############################################
# Wasserstein-2 distance
# ~~~~~~~~~~~~~~~~~~~~~~~~
#

lagrangian_barycenter( SamplesLoss("sinkhorn", p=2, blur=.1, scaling=.9), w=.5 )

###############################################
# Blabla

lagrangian_barycenter( SamplesLoss("sinkhorn", p=2, blur=.01, scaling=.9), w=.5 )

###############################################
# Blabla

lagrangian_barycenter( SamplesLoss("sinkhorn", p=2, blur=.001, scaling=.9), w=.5 )


###############################################
# Blabla

lagrangian_barycenter( SamplesLoss("sinkhorn", p=2, blur=.001, scaling=.5), w=.5 )

plt.show()