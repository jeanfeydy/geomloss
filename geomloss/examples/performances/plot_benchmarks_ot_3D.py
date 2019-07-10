r"""
Wasserstein distances between 3D meshes
==========================================================

Let's compare the performances of several OT solvers
on subsampled versions of the `Stanford dragon <http://graphics.stanford.edu/data/3Dscanrep/>`_,
a standard test surface made up of more than **870,000 triangles**.
In this benchmark, we measure timings on a simple registration task:
the **optimal transport of a sphere onto the (subsampled) dragon**, using
a quadratic ground cost 
:math:`\text{C}(x,y) = \tfrac{1}{2}\|x-y\|^2`
in the ambient space :math:`\mathbb{R}^3`.

"""


##############################################
# Setup
# ---------------------
#
# First, let's fetch our model from the Stanford repository:

import os

if not os.path.exists('data/dragon_recon/dragon_vrip_res4.ply'):
    import urllib.request
    urllib.request.urlretrieve(
        'http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz', 
        'data/dragon.tar.gz')

    import shutil
    shutil.unpack_archive('data/dragon.tar.gz', 'data')


##############################################
# To read the raw ``.ply`` ascii files, we rely on the
# `plyfile <https://github.com/dranjan/python-plyfile>`_ package:

import time
import importlib

import numpy as np
import torch

use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
numpy = lambda x : x.detach().cpu().numpy()

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plyfile import PlyData, PlyElement

def load_ply_file(fname, offset = [-0.011,  0.109, -0.008], scale = .04) :
    """Loads a .ply mesh to return a collection of weighted Dirac atoms: one per triangle face."""

    # Load the data, and read the connectivity information:
    plydata = PlyData.read(fname)
    triangles = np.vstack( plydata['face'].data['vertex_indices'] )

    # Normalize the point cloud, as specified by the user:
    points = np.vstack( [ [x,y,z] for (x,y,z) in  plydata['vertex'] ] )
    points -= offset
    points /= 2 * scale

    # Our mesh is given as a collection of ABC triangles:
    A, B, C = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]

    # Locations and weights of our Dirac atoms:
    X = (A + B + C) / 3  # centers of the faces
    S = np.sqrt(np.sum(np.cross(B - A, C - A) ** 2, 1)) / 2  # areas of the faces

    print("File loaded, and encoded as the weighted sum of {:,} atoms in 3D.".format(len(X)))

    # We return a (normalized) vector of weights + a "list" of points
    return tensor(S / np.sum(S)), tensor(X)


############################################################
# To keep things simple, we use as **targets** the subsamplings provided
# in the reference Stanford archive. Feel free to re-run
# this script with your own models!

# N.B.: Since Plyfile is far from being optimized, this may take some time!
targets = [ load_ply_file( fname ) for fname in 
            ['data/dragon_recon/dragon_vrip_res4.ply',
             'data/dragon_recon/dragon_vrip_res3.ply',
             'data/dragon_recon/dragon_vrip_res2.ply',
             #'data/dragon_recon/dragon_vrip.ply',  
          ] ]

###########################################################
# Our **source measures**: unit spheres, sampled with the same number of points
# as the target meshes.
#

def create_sphere(n_samples = 1000):
    """Creates a uniform sample on the unit sphere."""

    indices = np.arange(0, n_samples, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_samples)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
    points  = np.vstack( (x, y, z)).T
    weights = np.ones(n_samples) / n_samples

    return tensor(weights), tensor(points)

sources = [ create_sphere( len(X) ) for (_, X) in targets ]

############################################################
# As expected, our source and target point clouds are roughly aligned with each other.
# Now, let's move on to the interesting part!

def display_cloud(ax, measure, color) :

    w_i, x_i = numpy( measure[0] ), numpy( measure[1] )

    ax.view_init(elev=110, azim=-90)
    ax.set_aspect('equal')

    weights = w_i / w_i.sum()
    ax.scatter( x_i[:,0], x_i[:,1], x_i[:,2], 
                s = 25*500 * weights, c = color, edgecolors='none' )

    ax.axes.set_xlim3d(left=-1.4, right=1.4) 
    ax.axes.set_ylim3d(bottom=-1.4, top=1.4) 
    ax.axes.set_zlim3d(bottom=-1.4, top=1.4) 


fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(1, 1, 1, projection='3d')
display_cloud(ax, sources[0], 'red')
display_cloud(ax, targets[0], 'blue')
ax.set_title("Source and target point clouds - low resolution")
plt.tight_layout()




#############################################################
# With 


from pykeops import LazyTensor
from geomloss import SamplesLoss

blur = .01
OT_solver = SamplesLoss("sinkhorn", p=2, blur=blur,
                        scaling=.9, debias=False, potentials=True, backend="online")

print("Coucou")

a_i, x_i = sources[0]
b_j, y_j = targets[0]
F_i, G_j = OT_solver(a_i, x_i, b_j, y_j)

print("Voila")


def marginal_constraints(blur, a_i, x_i, b_j, y_j, F_i, G_j) :

    x_i = LazyTensor( x_i[:,None,:] )
    y_j = LazyTensor( y_j[None,:,:] )
    F_i = LazyTensor( F_i[:,None,None] )
    G_j = LazyTensor( G_j[None,:,None] )

    C_ij = ((x_i - y_j) ** 2).sum(-1) / 2
    K_ij = (( F_i + G_j - C_ij ) / blur**2 ).exp()

    A_i = K_ij@b_j
    B_j = K_ij.t()@a_i

    return (A_i - a_i).abs().sum(), (B_j - b_j).abs().sum()


print( marginal_constraints(blur, a_i, x_i, b_j, y_j, F_i, G_j) )

if False:
    # Creates a pyplot figure:
    plt.figure()
    linestyles = ["o-", "s-", "^-"]
    for i, backend in enumerate(backends):
        plt.plot( benches[:,0], benches[:,i+1], linestyles[i], 
                    linewidth=2, label='backend="{}"'.format(backend) )

    plt.title('Runtime for SamplesLoss("{}") in dimension {}'.format(Loss.loss, D))
    plt.xlabel('Number of samples per measure')
    plt.ylabel('Seconds')
    plt.yscale('log') ; plt.xscale('log')
    plt.legend(loc='upper left')
    plt.grid(True, which="major", linestyle="-")
    plt.grid(True, which="minor", linestyle="dotted")
    plt.axis([ NS[0], NS[-1], 1e-3, MAXTIME ])
    plt.tight_layout()

    # Save as a .csv to put a nice Tikz figure in the papers:
    header = "Npoints " + " ".join(backends)
    np.savetxt("output/benchmark_"+Loss.loss+"_3D.csv", benches, 
                fmt='%-9.5f', header=header, comments='')


plt.show()