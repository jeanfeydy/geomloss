"""
Wasserstein barycenters in 2D
==================================

Let's compute Wasserstein barycenters between 2D densities,
using a simple descent scheme to minimize a weighted Sinkhorn divergence.
"""

##############################################
# Setup
# ---------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.neighbors import KernelDensity
from torch.nn.functional import avg_pool2d

import torch
from geomloss import SamplesLoss

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

###############################################
# Dataset
# ~~~~~~~~~~~~~~~~~~
#

def grid(W):
    y, x = torch.meshgrid( [ torch.arange(0.,W).type(dtype) / W ] * 2 )
    return torch.stack( (x,y), dim=2 ).view(-1,2)


def load_image(fname) :
    img = misc.imread(fname, flatten = True) # Grayscale
    img = (img[:, :])  / 255.
    return 1 - img


def as_measure(fname, size):
    weights = torch.from_numpy( load_image(fname) ).type(dtype)
    sampling = weights.shape[0] // size
    weights = avg_pool2d( weights.unsqueeze(0).unsqueeze(0), sampling).squeeze(0).squeeze(0)
    weights = weights / weights.sum()

    samples = grid( size )
    return weights.view(-1), samples
    
import matplotlib
matplotlib.rc('image', cmap='gray')

N, M = (8, 8) if not use_cuda else (128, 64)

A = as_measure("data/A.png", M)
B = as_measure("data/B.png", M)
C = as_measure("data/C.png", M)
D = as_measure("data/D.png", M)


###############################################
#

Z_k = grid(N)


###############################################
# Display routine
# ~~~~~~~~~~~~~~~~~
#
# We display our samples using (smoothed) density
# curves, computed with a straightforward Gaussian convolution:
 
grid_plot = grid(M).view(-1,2).cpu().numpy()

def display_samples(ax, x, weights=None):
    """Displays samples on the unit square using a density estimator."""
    x = x.clamp(0, 1 - .1/M)
    bins = (x[:,0] * M).floor() + M * (x[:,1] * M).floor()
    count = bins.int().bincount(weights=weights, minlength=M*M)
    ax.imshow( count.detach().float().view(M,M).cpu().numpy(), vmin=0 )



###############################################
#

from geomloss.examples.optimal_transport.model_fitting import fit_model  # Wrapper around scipy.optimize
from torch.nn import Module, Parameter  # PyTorch syntax for optimization problems

class Barycenter(Module):
    
    def __init__(self, loss, targets):
        super(Barycenter, self).__init__()
        self.loss = loss        # Sinkhorn divergence to optimize
        self.targets = targets

        # Our parameter to optimize: sample locations and log-weights
        self.z_k = Parameter( Z_k.clone() )
        self.l_k = Parameter( torch.zeros(len(self.z_k)).type_as(self.z_k) )
 

    def weights(self):
        """Turns the l_k's into the weights of a positive probabilty measure."""
        return torch.nn.functional.softmax(self.l_k, dim=0)


    def fit(self, display=False, tol=1e-10):
        """Uses a custom wrapper around the scipy.optimize module."""
        fit_model(self, method = "L-BFGS", lr = 1., display = display, tol=tol, gtol=tol)


    def forward(self) :
        """Returns the cost to minimize."""
        cost = 0
        for (w, (weights, samples) ) in self.targets:
            cost = cost + w * self.loss(self.weights(), self.z_k, weights, samples)
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
            display_samples(ax, self.z_k, weights = self.weights())

            if title is None:
                ax.set_title("nit = {}, cost = {:3.4f}".format(nit, cost))
            else:
                ax.set_title(title)

            ax.set_xticks([], []); ax.set_yticks([], [])
            if nit != 2 and nit % 16 != 12: plt.tight_layout()



Barycenter( SamplesLoss("sinkhorn", blur=.01, scaling=.5),
            [ (0., A), (1., B) ] 
            ).fit(display=True, tol=1e-5)

 

plt.figure(figsize=(9.7,14))

ax = plt.subplot(7,7,1)  ; ax.imshow( A[0].reshape(M,M) ) ; ax.set_xticks([], []); ax.set_yticks([], [])
ax = plt.subplot(7,7,7)  ; ax.imshow( B[0].reshape(M,M) ) ; ax.set_xticks([], []); ax.set_yticks([], [])
ax = plt.subplot(7,7,43) ; ax.imshow( C[0].reshape(M,M) ) ; ax.set_xticks([], []); ax.set_yticks([], [])
ax = plt.subplot(7,7,49) ; ax.imshow( D[0].reshape(M,M) ) ; ax.set_xticks([], []); ax.set_yticks([], [])

for i in range(5):
    for j in range(5):
        x, y = j/4, i/4

        bary = Barycenter( SamplesLoss("sinkhorn", blur=.01, scaling=.5),
                        [ ( (1-x)*(1-y), A ),  ( x*(1-y), B ), 
                          ( (1-x)*  y,   C ),  ( x*  y,   D ),  ] )
        bary.fit(tol=1e-5)

        ax = plt.subplot(7,7, 7*(i+1) + j+2 )
        bary.plot(ax=ax, title="")

plt.tight_layout()
plt.show()
