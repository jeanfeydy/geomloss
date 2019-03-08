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
from geomloss.examples.model_fitting import fit_model
from torch.nn import Module, Parameter

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

###############################################
# Dataset
# ~~~~~~~~~~~~~~~~~~
#
# Starting from the uniform measure on :math:`[0,1]`,
# we are going to interpolate between samples 
# drawn from intervals of the real line.
# First, we should thus define our *leftmost*, *rightmost*
# and *variable* measures: 
#
# .. math::
#   \alpha ~=~ \frac{1}{M}\sum_{i=1}^M \delta_{x_i}, ~~~
#   \beta  ~=~ \frac{1}{M}\sum_{j=1}^M \delta_{y_j}, ~~~
#   \gamma  ~=~ \frac{1}{N}\sum_{k=1}^N \delta_{z_k}.

N, M = (100, 100) if not use_cuda else (1000, 1000)
 
t_i = torch.linspace(0, 1, M).type(dtype).view(-1,1)
t_j = torch.linspace(0, 1, M).type(dtype).view(-1,1)
t_k = torch.linspace(0, 1, N).type(dtype).view(-1,1)

X_i, Y_j, Z_k = .1 * t_i, .2 * t_j + .8, t_k



###############################################
# Display routine
# ~~~~~~~~~~~~~~~~~
 
t_plot = np.linspace(-0.1, 1.1, 1000)[:,np.newaxis]

def display_samples(ax, x, color, weights=None):
    """Displays samples on the unit interval using a density curve."""
    kde  = KernelDensity(kernel='gaussian', bandwidth= .002 ).fit(
            x.data.cpu().numpy(), 
            sample_weight = None if weights is None else weights.data.cpu().numpy())
    dens = np.exp( kde.score_samples(t_plot) )
    dens[0] = 0 ; dens[-1] = 0
    ax.fill(t_plot, dens, color=color)


class Barycenter(Module):
    """Abstract model that will be optimized."""
    
    def __init__(self, loss, w=.5):
        super(Barycenter, self).__init__()
        self.loss = loss
        self.w = w
        self.x_i, self.y_j, self.z_k = X_i.clone(), Y_j.clone(), Z_k.clone()

    def optimize(self):
        fit_model(self, method = "L-BFGS", lr = 1., verbose = True)

    def weights(self):
        return (torch.ones(len(self.z_k)) / len(self.z_k)).type_as(self.z_k)
    
    def plot(self, nit=0, cost=0):
        if nit == 0 or nit % 16 == 4: 
            plt.pause(.01)
            plt.figure(figsize=(16,4))

        if nit <= 4 or nit % 4 == 0:
            if nit < 4: index = nit + 1
            else:       index = (nit//4 - 1) % 4 + 1
            ax = plt.subplot(1,4, index)
            display_samples(ax, self.x_i, (.95,.55,.55))
            display_samples(ax, self.y_j, (.55,.55,.95))
            display_samples(ax, self.z_k, (.55,.95,.55), weights = self.weights())
            
            ax.set_title("nit = {}, cost = {:3.4f}".format(nit, cost))
            
            plt.axis([-.1,1.1,-.1,20.5])
            plt.xticks([], []); plt.yticks([], [])
            plt.tight_layout()


###############################################
# Lagrangian gradient flow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

class LagrangianBarycenter(Barycenter) :
    def __init__(self, loss, w=.5) :
        super(LagrangianBarycenter, self).__init__(loss, w)
        self.z_k = Parameter( Z_k.clone() )
      
    def forward(self) :
        """Returns the cost to minimize."""
        return self.w  * self.loss( self.z_k, self.x_i) \
        + (1 - self.w) * self.loss( self.z_k, self.y_j)
    

###############################################
# Blabla
#

LagrangianBarycenter( SamplesLoss("sinkhorn", blur=.1, scaling=.9) ).optimize()

###############################################
# Blabla

LagrangianBarycenter( SamplesLoss("sinkhorn", blur=.01, scaling=.9) ).optimize()

###############################################
# Blabla

LagrangianBarycenter( SamplesLoss("sinkhorn", blur=.001, scaling=.9) ).optimize()

 
###############################################
# Eulerian gradient flow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

from torch.nn.functional import conv1d

def gaussian_conv(x, blur=2):
    D = torch.arange( - 5*blur, 5*blur+1 ).type_as(x)
    k_ε = (-D**2 / (2*blur**2)).exp()
    return conv1d( x.view(1,1,-1), k_ε.view(1,1,-1), padding=k_ε.shape[-1]//2 ).view(-1)


class EulerianBarycenter(Barycenter) :
    def __init__(self, loss, w=.5) :
        super(EulerianBarycenter, self).__init__(loss, w)

        # We're going to work with weights:
        self.a_i = (torch.ones(len(self.x_i)) / len(self.x_i)).type_as(self.x_i)
        self.b_j = (torch.ones(len(self.y_j)) / len(self.y_j)).type_as(self.y_j)

        # Our parameter to optimize: the logarithms of our weights
        self.l_k = Parameter( torch.zeros(len(self.z_k)).type_as(self.z_k) )
 
    def weights(self):
        soft_lk = gaussian_conv( self.l_k, blur= int(.02*len(self.l_k)  ))
        return torch.nn.functional.softmax(soft_lk, dim=0)
      
    def forward(self) :
        """Returns the cost to minimize."""
        c_k  = self.weights()
        return self.w  * self.loss(c_k, self.z_k, self.a_i, self.x_i) \
        + (1 - self.w) * self.loss(c_k, self.z_k, self.b_j, self.y_j)


###############################################
# Blabla

EulerianBarycenter( SamplesLoss("sinkhorn", blur=.001, scaling=.9) ).optimize()

plt.show()