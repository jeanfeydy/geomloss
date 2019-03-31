"""
Influence of the blur parameter, scaling strategy
=====================================================

Blabla
"""



##############################################
# Setup
# ---------------------

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch.autograd import grad

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

###############################################
# Display routines
# ~~~~~~~~~~~~~~~~~

from scipy import misc


def load_image(fname) :
    img = misc.imread(fname, flatten = True) # Grayscale
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
    ax.contour(F, origin='lower', linewidths = 1., colors = color,
               levels = levels, extent=[0,1,0,1]) 



def display_samples(ax, x, weights, color, v=None) :
    x_ = x.detach().cpu().numpy()
    weights_ = weights.detach().cpu().numpy()

    weights_[weights_ < 1e-5] = 0
    ax.scatter( x_[:,0], x_[:,1], 5 * 500 * weights_, color, edgecolors='none' )

    if v is not None :
        v_ = v.detach().cpu().numpy()
        ax.quiver( x_[:,0], x_[:,1], v_[:,0], v_[:,1], 
                    scale = 1, scale_units="xy", color="#5CBF3A", 
                    zorder= 3, width= 2. / len(x_) )

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

sampling = 10 if not use_cuda else 2

A_i, X_i = draw_samples("ell_a.png", sampling)
B_j, Y_j = draw_samples("ell_b.png", sampling)

from geomloss import SamplesLoss


def display_scaling(scaling = .5, Nits = 9) :

    plt.figure(figsize=( (12, ((Nits-1)//3 + 1) * 4)))

    for i in range(Nits):
        blur = scaling**i
        Loss = SamplesLoss("sinkhorn", p=2, blur=blur, diameter=1., scaling=scaling)

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


        # Fancy display: -----------------------------------------------------------
        ax = plt.subplot( ((Nits-1)//3 + 1) , 3, i+1)
        ax.scatter( [10], [10] )  # shameless hack to prevent a slight change of axis...

        display_potential(ax, G_j, "#C8DFF9")
        display_potential(ax, F_i, "#E2C5C5")

        display_samples(ax, y_j, b_j, [(.55,.55,.95)])
        display_samples(ax, x_i, a_i, [(.95,.55,.55)], v = BrenierMap)

        ax.set_title("iteration {}, blur = {:.3f}".format(i+1, blur))

        ax.set_xticks([0, 1]) ; ax.set_yticks([0, 1])
        ax.axis([0,1,0,1]) ; ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()


display_scaling(scaling = .5, Nits = 9)
display_scaling(scaling = .7, Nits = 18)
plt.show()

