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


from pykeops.torch import LazyTensor

def plan_marginals(blur, a_i, x_i, b_j, y_j, F_i, G_j) :

    x_i = LazyTensor( x_i[:,None,:] )
    y_j = LazyTensor( y_j[None,:,:] )
    F_i = LazyTensor( F_i[:,None,None] )
    G_j = LazyTensor( G_j[None,:,None] )

    # Cost matrix:
    C_ij = ((x_i - y_j) ** 2).sum(-1) / 2

    # Scaled kernel matrix:
    K_ij = (( F_i + G_j - C_ij ) / blur**2 ).exp()

    A_i = a_i * (K_ij@b_j)      # First marginal
    B_j = b_j * (K_ij.t()@a_i)  # Second marginal

    return A_i, B_j

def blurred_relative_error(blur, x_i, a_i, A_i):
    x_j = LazyTensor( x_i[None,:,:] )
    x_i = LazyTensor( x_i[:,None,:] )

    C_ij = ((x_i - x_j) ** 2).sum(-1) / 2
    K_ij = ( - C_ij / blur**2 ).exp()

    squared_error = (A_i - a_i).dot( K_ij@(A_i - a_i) )
    squared_norm  = a_i.dot( K_ij@a_i )

    return ( squared_error / squared_norm ).sqrt()

def marginal_error(blur, a_i, x_i, b_j, y_j, F_i, G_j, mode="blurred"):

    A_i, B_j = plan_marginals(blur, a_i, x_i, b_j, y_j, F_i, G_j)

    if mode == "TV":
        # Return the (average) total variation error on the marginal constraints:
        return ( (A_i - a_i).abs().sum() + (B_j - b_j).abs().sum() ) / 2
    
    elif mode == "blurred":
        norm_x = blurred_relative_error(blur, x_i, a_i, A_i)
        norm_y = blurred_relative_error(blur, y_j, b_j, B_j)
        return ( norm_x + norm_y ) / 2

    else:
        raise NotImplementedError()



def transport_cost(a_i, b_j, F_i, G_j):
    return a_i.dot(F_i) + b_j.dot(G_j)

def wasserstein_distance(a_i, b_j, F_i, G_j):
    return (2 * transport_cost(a_i, b_j, F_i, G_j)).sqrt()


def benchmark_solver(OT_solver, blur, source, target):
    a_i, x_i = source
    b_j, y_j = target

    start = time.time()
    F_i, G_j = OT_solver(a_i, x_i, b_j, y_j)
    if use_cuda: torch.cuda.synchronize()
    end = time.time()

    return end - start, \
           marginal_error(blur, a_i, x_i, b_j, y_j, F_i, G_j).item(), \
           wasserstein_distance(a_i, b_j, F_i, G_j).item()
        


def benchmark_solvers(name, OT_solvers, source, target, ground_truth, 
                      blur = .01, display=False, maxtime=None):

    timings, errors, costs = [], [], []

    print('Benchmarking the "{}" family of OT solvers:'.format(name))
    for i, OT_solver in enumerate(OT_solvers):

        timing, error, cost = benchmark_solver(OT_solver, blur, source, target)

        timings.append(timing) ; errors.append(error) ; costs.append(cost)
        print("{}-th solver : t = {:.4f}, error on the constraints = {:.3f}, cost = {:.6f}".format(
                i+1, timing, error, cost))

        if maxtime is not None and timing > maxtime:
            not_performed = len(OT_solvers) - (i + 1)
            timings += [np.nan] * not_performed
            errors  += [np.nan] * not_performed
            costs   += [np.nan] * not_performed
            break

    timings, errors, costs = np.array(timings), np.array(errors), np.array(costs)


    if display: # Fancy display
        fig = plt.figure(figsize=(12,8))

        ax_1 = fig.subplots()
        ax_1.set_title("Benchmarking \"{}\"\non a {:,}-by-{:,} entropic OT problem, with a blur radius of {:.3f}".format(
            name, len(source[0]), len(target[0]), blur
        ))
        ax_1.set_xlabel("time (s)")

        ax_1.plot(timings, errors, color="b")
        ax_1.set_ylabel("Relative error on the marginal constraints", color="b")
        ax_1.tick_params("y", colors="b") ; ax_1.set_ylim(bottom=0)

        ax_2 = ax_1.twinx()

        ax_2.plot(timings, abs(costs - ground_truth) / ground_truth, color="r")
        ax_2.set_ylabel("Relative error on the cost value", color="r")
        ax_2.tick_params("y", colors="r") ; ax_2.set_ylim(bottom=0)

    return timings, errors, costs



def sinkhorn_loop(a_i, x_i, b_j, y_j, blur = .01, nits = 100, backend = "keops"):

    loga_i, logb_j = a_i.log(), b_j.log()
    loga_i, logb_j = loga_i[:,None,None], logb_j[None,:,None]


    if backend == "keops":
        x_i, y_j = LazyTensor( x_i[:,None,:] ), LazyTensor( y_j[None,:,:] )
        C_ij = ((x_i - y_j) ** 2).sum(-1) / 2
        
    elif backend == "pytorch":
      
        # C_ij = ((x_i[:,None,:] - y_j[None,:,:]) ** 2).sum(-1) / 2
        D_xx = (x_i ** 2).sum(-1)[:,None]  # (N,1)
        D_xy = x_i@y_j.t()   # (N,D)@(D,M) = (N,M)
        D_yy = (y_j ** 2).sum(-1)[None,:]  # (1,M)
        C_ij = (D_xx + D_yy) / 2 - D_xy

        C_ij = C_ij[:,:,None]

    eps = blur**2
    F_i, G_j = torch.zeros_like(loga_i), torch.zeros_like(logb_j)

    for _ in range(nits):
        F_i = - ( (- C_ij / eps + (G_j + logb_j) ) ).logsumexp(dim=1)[:,None,:]
        G_j = - ( (- C_ij / eps + (F_i + loga_i) ) ).logsumexp(dim=0)[None,:,:]

    return eps * F_i, eps * G_j

from functools import partial
sinkhorn_solver = lambda blur, nits, backend: partial(sinkhorn_loop, blur=blur, nits=nits, backend=backend)



def full_benchmark(source, target, blur, maxtime=None):

    # Compute a suitable "ground truth"
    OT_solver = SamplesLoss("sinkhorn", p=2, blur=blur,
                            scaling=.999, debias=False, potentials=True)
    _, _, ground_truth = benchmark_solver(OT_solver, blur, sources[0], targets[0])

    results = {}

    # Compute statistics for the three backends of GeomLoss: -------------------

    for backend in ["multiscale", "online", "tensorized"]:
        OT_solvers = [ SamplesLoss("sinkhorn", p=2, blur=blur, scaling=scaling,
                                backend=backend, debias=False, potentials=True)
                    for scaling in [.6, .7, .8, .9, .95, .99] ]

        results[backend] = benchmark_solvers("GeomLoss "+backend, OT_solvers, 
                                              source, target, ground_truth, 
                                              blur = blur, display=True, maxtime=maxtime)


    # Compute statistics for a naive Sinkhorn loop -----------------------------

    for backend in ["pytorch", "keops"]:
        OT_solvers = [ sinkhorn_solver(blur, nits = nits, backend = backend)
                       for nits in [5, 10, 20, 50, 100, 200, 500, 1000] ]

        results[backend] = benchmark_solvers("Sinkhorn loop - " + backend, OT_solvers, 
                                                source, target, ground_truth, 
                                                blur = blur, display=True, maxtime=maxtime)

    
from geomloss import SamplesLoss

blur = .01
results = full_benchmark(sources[0], targets[0], blur)



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