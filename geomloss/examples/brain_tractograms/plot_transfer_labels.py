"""
Fast and Scalable Optimal Transport for Brain Tractograms: Application to Labels Transfer
==================================

We use a new multiscale algorithm for solving regularized Optimal Transport 
problems on the GPU, with a linear memory footprint. 

We use the resulting smooth assignments to perform label transfer for atlas-based 
segmentation of fiber tractograms. The parameters -- \emph{blur} and \emph{reach} -- 
of our method are meaningful, defining the minimum and maximum distance at which 
two fibers are compared with each other. They can be set according to anatomical knowledge.
"""
##################################################
#
# Multiscale Optimal Transport
# -----------------------------
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
from geomloss import SamplesLoss
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor  if use_cuda else torch.LongTensor

###############################################
#Loading and saving data routines
#----------------------------------------
#

NPOINTS = 20  # Number of points per fiber. 

from tract_io import read_vtk, streamlines_resample, save_vtk, save_vtk_labels
from tract_io import save_tract, save_tract_numpy
from tract_io import save_tract_with_labels, save_tracts_labels_separate

##############################################
# Dataset
# ---------------------
#
# Fetch data from the KeOps website:

import os

def fetch_file(name):
    if not os.path.exists(f'data/{name}.npy'):
        import urllib.request
        print("Fetching the atlas... ", end="", flush=True)
        urllib.request.urlretrieve(
            f'https://www.kernel-operations.io/data/{name}.npy', 
            f'data/{name}.npy')
        print("Done.")

fetch_file("tracto_atlas")
fetch_file("atlas_labels")
fetch_file("tracto1")

###############################################################################
# Each fiber is sampled with 20 points in R^3. 
# Thus, one tractogram is a matrix of size n x 60 where n is the number of fibers
# The atlas is labelled, wich means that each fiber belong to a cluster. 
# This is summarized by the vector lab_j of size n x 1. lab_j[i] is the label of the fiber i. 
# Subsample the data by a factor 4 if you want to reduce the computational time:

subsample = 4 if True else 1    

###############################################################################
# 
# Load atlas (segmented, each fiber has a label):

Y_j   = np.load("data/tracto_atlas.npy")[::subsample, :]  / np.sqrt(NPOINTS)
lab_j = np.load("data/atlas_labels.npy")[::subsample]

##############################################
# Fibers do not have a canonical orientation. Since our ground distance is a simple
# L2-distance on the sampled fibers, we augment the dataset with the mirror flip 
# of all fibers and perform the OT on this augmented dataset.
    
Y_j_flip = Y_j.reshape( (-1, NPOINTS, 3) )[:,::-1,:].copy().reshape( Y_j.shape )

##############################################
# 

Y_j   = np.concatenate( (Y_j,   Y_j_flip), axis = 0)
lab_j = np.concatenate( (lab_j, lab_j),    axis = 0)

##############################################
# 

Y_j   = torch.from_numpy( Y_j   ).type( dtype ).view( len(Y_j), -1 ).contiguous()
lab_j = torch.from_numpy( lab_j ).type( dtypeint ).contiguous()
nf_j  = len( Y_j ) // 2

##############################################
# Load a new subject (unlabelled)
#

# load the unlabeled fibers
X_i = ( np.load( "data/tracto1.npy" ) / np.sqrt(NPOINTS) )[::subsample,:]
# add the flip:
X_i_flip = X_i.reshape( (-1, NPOINTS, 3) )[:,::-1,:].copy().reshape( X_i.shape ) 
X_i = np.concatenate( (X_i, X_i_flip), axis = 0) 
X_i = torch.from_numpy( X_i ).type( dtype ).view( len(X_i), -1).contiguous()

##############################################
# Add some weight on both ends of our fibers:
#

gamma = 3.
X_i[:, 0], X_i[:, -1] = gamma * X_i[:, 0] , gamma * X_i[:, -1] 
Y_j[:, 0], Y_j[:, -1] = gamma * Y_j[:, 0] , gamma * Y_j[:, -1] 

##############################################
# 

N, M = len( X_i ), len( Y_j )
print("Data loaded.")

n_labels = lab_j.max() + 1



##############################################
# Pre processing for the multi-scale      
# --------------------------------------
#
# To use the multiscale version of the regularized OT, 
# we need to have a cluster of our input data (atlas and new subject).
# For the atlas, the cluster is given by the segmentation. We use a Kmeans to 
# separate the fibers and the flips within a cluser, in order to have clusters whose fibers have similar
# orientation
#

from pykeops.torch import generic_argmin

############################################################################################
# Kmeans adapted on our labeled atlas (estimate centroids + labels given the initial segmented atlas)
# number of clusters : twice the number of labels (since we augmented the data with the flips)   
# We perform this in order to have clusters with the same orientation. 
#

def KMeans_atlas(x, lab, nf, Niter = 10, verbose = True):
    N, D = x.shape  # Number of samples, dimension of the ambient space
    nn_search = generic_argmin(  # Argmin reduction for generic formulas:
        'SqDist(x,y)',           # A simple squared L2 distance
        'ind = Vi(1)',           # Output one index per "line" (reduction over "j")
        'x = Vi({})'.format(D),  # 1st arg: one point per "line"
        'y = Vj({})'.format(D))  # 2nd arg: one point per "column"


    #centroids:
    c_j = torch.zeros(lab.max()+1,x.shape[1]).type_as(x)
    c_j_flip = torch.zeros(lab.max()+1,x.shape[1]).type_as(x)
    
    cl_j = torch.zeros(len(x)).type(dtypeint)
    nf_visited = 0
    for l in range(lab.max() + 1):  # loop on the labels
        i_l = (lab == l).nonzero()
        nf_l = len(i_l)
        #careful initialization of the centroids : one for the original fibers, one for the flips taken in the dataset:
        c_l, c_l_flip  = x[i_l[0],:], x[nf + i_l[0],:]
        fibers, fibers_flip = x[i_l.view(-1),:], x[nf + i_l.view(-1),:] #fibers and mirror flips in the current label
        y = torch.cat((fibers,fibers_flip),0) 
        c = torch.cat((c_l,c_l_flip),0) 
        for i in range(Niter): #Kmeans estimation with two classes : one original, one flip. 
            cl  = nn_search(y,c).view(-1)  # Points -> Nearest cluster
            Ncl = torch.bincount(cl).type(dtype)  # Class weights
            for d in range(D):  # Compute the cluster centroids with torch.bincount:
                c[:, d] = torch.bincount(cl, weights=y[:, d]) / Ncl
        c_j[l,:], c_j_flip[l,:] = c[0,:], c[1,:]   #update the centroids (original and flip)
        cl_j[ torch.cat((i_l.view(-1), i_l.view(-1) + nf), 0)] = l*(1-cl) + cl*(l+lab.max()+1) #update the classes (original and flips)
        nf_visited = nf_visited + 2*nf_l
        print(l)
        print('fibers labeled = ', nf_visited)
    return cl_j, torch.cat((c_j,c_j_flip),0)


##############################################
# For new subject (unlabelled), we perform a simple Kmean
# on R^60 to obtain a cluster of the data.
#


#KMeans on the data with flips
def KMeans_withflip(x, K, Niter= 10, verbose = True):
    N, D = x.shape  # Number of samples, dimension of the ambient space
    
    # Define our KeOps CUDA kernel:Y_j,lab_j = import_tract_with_labels('k_means_atlas_left.vtk')
    nn_search = generic_argmin(  # Argmin reduction for generic formulas:
        'SqDist(x,y)',           # A simple squared L2 distanceY_j, lab_j
        'ind = Vi(1)',           # Output one index per "line" (reduction over "j")
        'x = Vi({})'.format(D),  # 1st arg: one point per "line"
        'y = Vj({})'.format(D))  # 2nd arg: one point per "column"

    # K-means loop:
    # - x  is the point cloud, 
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()

    # Simplistic random initialization for the cluster centroids:
    perm = torch.randperm(N // 2)
    idx = perm[:K]
    #initialize centroids carefully with the flips (same random initialization for the original data and the flips)
    c = torch.cat((x[idx, :].clone(), x[N // 2 + idx, :].clone()),0)

    for i in range(Niter):
        cl  = nn_search(x,c).view(-1)  # Points -> Nearest cluster
        Ncl = torch.bincount(cl).type(dtype)  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl
    if use_cuda: torch.cuda.synchronize()
    end = time.time()
    if verbose: print("KMeans performed in {:.3f}s.".format(end-start))

    return cl, c

# Perform the computation:
cl_j, c_j = KMeans_atlas(Y_j,lab_j[:len(lab_j)//2],nf_j)
lab_i, c_i = KMeans_withflip(X_i, K = 1000)
print("Computed.")

#standard deviation of the clusters size
std_i = (( X_i - c_i[lab_i, :] )**2).sum(1).mean().sqrt()
std_j = (( Y_j - c_j[cl_j , :] )**2).sum(1).mean().sqrt()

print("K means Done.")



##############################################
# Compute the OT plan with the multiscale algorithm    
# ------------------------------------------------------
#
# To use the **multiscale** Sinkhorn algorithm,
# we should simply provide:
#
# - explicit **labels** and **weights** for both input measures,
# - a typical **cluster_scale** which specifies the iteration at which
#   the Sinkhorn loop jumps from a **coarse** to a **fine** representation
#   of the data.
#
blur = 3.
Loss =  SamplesLoss("sinkhorn", p=2, blur= blur, reach = 20,  scaling=.9, cluster_scale = max(std_i,std_j), debias = False, potentials = True, verbose=True) 

############################################################################################
# To specify explicit cluster labels, SamplesLoss also requires
# explicit weights. Let's go with the default option - a uniform distribution:

a_i = torch.ones(N).type(dtype) / N
b_j = torch.ones(M).type(dtype) / M

start = time.time()
# 6 args -> labels_i, weights_i, locations_i, labels_j, weights_j, locations_j
F_i, G_j = Loss( lab_i, a_i, X_i , cl_j, b_j, Y_j ) #Compute the dual vectors F_i and G_j
if use_cuda: torch.cuda.synchronize()
end = time.time()

print('OT computed in  in {:.3f}s.'.format(end-start))


##############################################
# Use the OT to perform thes label transfer
# ---------------------
# The transport plan pi_{i,j} gives the probability for 
# a fiber i of the subject to be assigned to the (labeled) fiber j of the atlas.
# We assign a label l to the fiber i as the label with maximum probability for all the soft assignement of i. 

X_i = X_i[:len(X_i)//2,:]#return to the original data (unflipped)
N_batch = len(X_i) // 10
F_i = F_i[:len(F_i)//2]
new_lab = torch.zeros(0).cuda().type(dtypeint) #label assignement 
value = torch.zeros(0).cuda()

from pykeops.torch import generic_sum

# Define our KeOps CUDA kernel:
print('lab_j max = ', lab_j.max())
#Compute soft-segmentation score
transfer = generic_sum(
    "Exp( (F_i + G_j - IntInv(2)*SqDist(X_i,Y_j)) / E )",  # See the formula above
    "Lab = Vi(1)",  # Output:  one vector of size 3 per line
    "E   = Pm(1)",  # 1st arg: a scalar parameter, the temperature
    "X_i = Vi({})".format(NPOINTS*3),  # 2nd arg: one 2d-point per line
    "Y_j = Vj({})".format(NPOINTS*3),  # 3rd arg: one 2d-point per column
    "F_i = Vi(1)",  # 4th arg: one scalar value per line
    "G_j = Vj(1)") # 5th arg: one scalar value per column

for i in range(10):
    Lab_i_batch = torch.zeros( N_batch, lab_j.max()+1).type(dtype)
    start = i * N_batch
    end = (i + 1 ) * N_batch
    print(start, end)
    new_labels_i = torch.zeros( len(X_i) // 10, 1, n_labels ).cuda()
    for k in range(n_labels):
        # And apply it on the data (KeOps is pretty picky on the input shapes...):
        G_j_lab_k = G_j[lab_j == k]
        Y_j_lab_k = Y_j[lab_j == k,:]
        new_labels_i[:,:,k] = transfer(torch.Tensor( [blur**2] ).type(dtype), X_i[start:end,:], Y_j_lab_k, 
                                    F_i[start:end].view(-1,1), G_j_lab_k.view(-1,1)) / M
    
    value_batch, new_lab_batch = new_labels_i.squeeze().max(1)
    new_lab = torch.cat((new_lab, new_lab_batch),0)
    value = torch.cat((value, value_batch),0)

X_i[ : , 0 ], X_i[ : , -1 ] = X_i[ : , 0 ] / gamma ,  X_i[ : , -1 ] / gamma 

new_lab[(value < 10**(-2)) ] = new_lab.max() + 1 #we add a new labels of outliers : fibers that were not assign during the OT. 
save_tracts_labels_separate('Output/segmented_subject/labels_subject', X_i, new_lab, 0, new_lab.max() + 1) #save the data


