"""
Benchmark SamplesLoss in 3D
=====================================

Let's compare the performances of our losses and backends
as the number of samples grows from 100 to 1,000,000.
"""


##############################################
# Setup
# ---------------------

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if False:
    import urllib.request
    urllib.request.urlretrieve(
        'http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz', 
        'data/dragon.tar.gz')

    import shutil
    shutil.unpack_archive('data/dragon.tar.gz', 'data')

from plyfile import PlyData, PlyElement


def load_ply_file(fname, centroid = [-0.011 ,  0.109, -0.008], scale = .04) :

    plydata = PlyData.read(fname)
    triangles = np.vstack( plydata['face'].data['vertex_indices'] )

    points = np.vstack( [ [x,y,z] for (x,y,z) in  plydata['vertex'] ] )
    points -= centroid
    points /= 2 * scale

    A, B, C = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]

    X = (A + B + C) / 3
    S = np.sqrt(np.sum(np.cross(B - A, C - A) ** 2, 1)) / 2

    return S / np.sum(S), X

import math, random

def create_sphere(n_samples = 1000):

    indices = np.arange(0, n_samples, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_samples)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
    points  = np.vstack( (x, y, z)).T
    weights = np.ones(n_samples) / n_samples

    return weights, points


targets = [ load_ply_file( fname ) for fname in 
            ['data/dragon_recon/dragon_vrip_res4.ply',
             'data/dragon_recon/dragon_vrip_res3.ply',] ]

sources = [ create_sphere( len(X) ) for (_, X) in targets ]


def display_cloud(ax, w_i, x_i, color) :

    ax.view_init(elev=110, azim=-90)
    ax.set_aspect('equal')

    weights = w_i / w_i.sum()
    ax.scatter( x_i[:,0], x_i[:,1], x_i[:,2], 
                s = 25*500 * weights, c = color, edgecolors='none' )

    ax.axes.set_xlim3d(left=-1.4, right=1.4) 
    ax.axes.set_ylim3d(bottom=-1.4, top=1.4) 
    ax.axes.set_zlim3d(bottom=-1.4, top=1.4) 


fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1, 2, 1, projection='3d') ; display_cloud(ax, sources[0][0], sources[0][1], 'red') ; ax.set_title("Source point cloud")
ax = fig.add_subplot(1, 2, 2, projection='3d') ; display_cloud(ax, targets[0][0], targets[0][1], 'blue') ; ax.set_title("Target point cloud")
plt.tight_layout()

plt.show()


if False:
    import time

    import importlib
    import torch

    use_cuda = torch.cuda.is_available()

    from geomloss import SamplesLoss



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

