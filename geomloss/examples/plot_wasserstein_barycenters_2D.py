"""
Wasserstein barycenters in 2D
==================================

Let's compute Wasserstein barycenters between 2D densities,
using a simple Lagrangian scheme to minimize a weighted Sinkhorn divergence.
"""

##############################################
# Setup
# ---------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

import torch
from geomloss import SamplesLoss

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

###############################################
# Dataset
# ~~~~~~~~~~~~~~~~~~
#


N, M = (50, 50) if not use_cuda else (500, 500)
 
t_i = torch.linspace(0, 1, M).type(dtype).view(-1,1)
t_j = torch.linspace(0, 1, M).type(dtype).view(-1,1)

X_i, Y_j  = .1 * t_i, .2 * t_j + .8  # Intervals [0., 0.1] and [.8, 1.].


###############################################
#

t_k = torch.linspace(0, 1, N).type(dtype).view(-1,1)
Z_k = t_k


###############################################
# Display routine
# ~~~~~~~~~~~~~~~~~
#
# We display our samples using (smoothed) density
# curves, computed with a straightforward Gaussian convolution:
 
t_plot = np.linspace(-0.1, 1.1, 1000)[:,np.newaxis]

def display_samples(ax, x, color, blur=.01):
    """Displays samples on the unit square using a density estimator."""
    kde  = KernelDensity(kernel='gaussian', bandwidth= blur ).fit( x.data.cpu().numpy() )
    dens = np.exp( kde.score_samples(t_plot) )
    dens[0] = 0 ; dens[-1] = 0
    ax.fill(t_plot, dens, color=color)



###############################################
#

from geomloss.examples.model_fitting import fit_model  # Wrapper around scipy.optimize
from torch.nn import Module, Parameter  # PyTorch syntax for optimization problems

class LagrangianBarycenter(Module):
    
    def __init__(self, loss, *targets):
        super(LagrangianBarycenter, self).__init__()
        self.loss = loss        # Sinkhorn divergence to optimize
        self.targets = targets

        self.g_k = (torch.ones(len(Z_k)) / len(Z_k)).type_as(Z_k)  # Default weights
        self.z_k = Parameter( Z_k.clone() )


    def optimize(self, display=False, tol=1e-10):
        """Uses a custom wrapper around the scipy.optimize module."""
        fit_model(self, method = "L-BFGS", lr = 1., display = display, tol=tol, gtol=tol)


    def forward(self) :
        """Returns the cost to minimize."""
        cost = 0
        for (w, weights, samples) in self.targets:
            cost = cost + w * self.loss(self.g_k, self.z_k, weights, samples)
        return cost


    def plot(self, nit=0, cost=0, ax=None, title=None):
        """Displays the descent using a custom 'waffle' layout."""
        if ax is None:
            if nit == 0 or nit % 16 == 4: 
                plt.pause(.01)
                plt.figure(figsize=(16,4))

            if nit <= 4 or nit % 4 == 0:
                if nit < 4: index = nit + 1
                else:       index = (nit//4 - 1) % 4 + 1
                ax = plt.subplot(1,4, index)
                
        if ax is not None:
            display_samples(ax, self.x_i, (.95,.55,.55))
            display_samples(ax, self.y_j, (.55,.55,.95))
            display_samples(ax, self.z_k, (.55,.95,.55), weights = self.weights(), blur=.005)

            if title is None:
                ax.set_title("nit = {}, cost = {:3.4f}".format(nit, cost))
            else:
                ax.set_title(title)

            ax.axis([-.1,1.1,-.1,20.5])
            ax.set_xticks([], []); ax.set_xticks([], [])
            plt.tight_layout()


    

# LagrangianBarycenter( SamplesLoss("sinkhorn", blur=.01, scaling=.5) ).optimize(display=True, tol=1e-5)

 
plt.show()
