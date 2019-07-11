"""
Create an atlas using Wasserstein barycenters
==================================================
"""

#############################################
# Setup
# ---------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.neighbors import KernelDensity
from torch.nn.functional import avg_pool2d
import torch
from geomloss import SamplesLoss
import time
use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
from scipy.interpolate import RegularGridInterpolator

import nibabel as nib
import matplotlib.pyplot as plt


###############################################
# Dataset
# ~~~~~~~~~~~~~~~~~~
#
# In this tutorial, we work with square images
# understood as densities on the unit square.

affine_transform = nib.load('Data_MRI/manual_ifof1.nii').affine

def load_data_nii(fname):
    img = nib.load(fname)
    affine_mat=img.affine
    hdr=img.header
    data = img.get_fdata()
    data_norm=data/np.max(data)
    data_norm = torch.from_numpy(data_norm).type(dtype)
    return data_norm

def grid(nx,ny,nz):
    x,y,z = torch.meshgrid( torch.arange(0.,nx).type(dtype) ,  torch.arange(0.,ny).type(dtype), torch.arange(0.,nz).type(dtype) )
    return torch.stack( (x,y,z), dim=3 ).view(-1,3).detach().cpu().numpy()

dataset = []
for i in range(5):
    fname = 'Data_MRI/manual_ifof'+str(i+1)+'.nii'
    image_norm = load_data_nii(fname)
    print(image_norm.shape)
    dataset.append(image_norm)

###############################################


def img_to_points_cloud(data_norm):#normalized images (between 0 and 1)
    nx,ny,nz = data_norm.shape
    ind = data_norm.nonzero()
    indx = ind[:,0]
    indy = ind[:,1]
    indz = ind[:,2] 
    data_norm = data_norm/data_norm.sum()
    a_i = data_norm[indx,indy,indz]
    
    return ind.type(dtype),a_i

def measure_to_image(x,nx,ny,nz, weights = None):
    bins = (x[:,2]).floor() + nz*(x[:,1]).floor() + nz*ny*(x[:,0]).floor()
    count = bins.int().bincount(weights = weights, minlength = nx*ny*nz)
    return count.view(nx,ny,nz)
     

Loss = SamplesLoss( "sinkhorn", blur=1, scaling=.9, debias = False)
models = []
nx,ny,nz = image_norm.shape

def initialize_barycenter(dataset):
    mean = torch.zeros(nx,ny,nz).type(dtype)
    for k in range(len(dataset)):
        img = dataset[k]
        mean = mean + img
    mean = mean/len(dataset)
    x_i,a_i = img_to_points_cloud(mean)
    bar_pos, bar_weight = torch.tensor([]).type(dtype), torch.tensor([]).type(dtype)
    for d in range(3):
        x_i_d1, x_i_d2 = x_i.clone(), x_i.clone()
        x_i_d1[:,d], a_i_d1 = x_i_d1[:,d] +0.25, a_i/6
        x_i_d2[:,d], a_i_d2 = x_i_d2[:,d] -0.25, a_i/6
        bar_pos, bar_weight = torch.cat((bar_pos,x_i_d1,x_i_d2),0), torch.cat((bar_weight,a_i_d1,a_i_d2),0)
    return bar_pos, bar_weight

x_i,a_i = initialize_barycenter(dataset)
x_i.requires_grad = True
   
start = time.time()
for j in range(len(dataset)):
    img_j = dataset[j]
    y_j, b_j = img_to_points_cloud(img_j)
    L_ab = Loss( a_i, x_i, b_j, y_j)
    [g_i] = torch.autograd.grad(L_ab, [x_i])
    models.append( x_i - g_i / a_i.view(-1,1) )

a, b, c, d, e = models

###############################################
# If the weights :math:`w_k` sum up to 1, this update is a barycentric
# combination of the **target points** :math:`x_i + v_i^A`, :math:`~\dots\,`, :math:`x_i + v_i^D`,
# images of the source sample :math:`x_i`
# under the action of the :doc:`generalized Monge maps <plot_interpolation>` that transport
# our uniform sample onto the four target measures.
# 
# Using the resulting sample as an **ersatz for the true Wasserstein barycenter**
# is thus an approximation that holds in dimension 1, and is reasonable
# for most applications. As evidenced below, it allows us to interpolate
# between arbitrary densities at a low numerical cost:


barycenter = (a+b+c+d+e)/5
if use_cuda: torch.cuda.synchronize()
end = time.time()
print('barycenter computed in {:.3f}s.'.format(end-start))
img_barycenter = measure_to_image(barycenter, nx,ny,nz,a_i)
plt.figure()
plt.imshow(img_barycenter.detach().cpu().numpy()[20,:,:])
plt.show()
linear_transform_inv = np.linalg.inv(affine_transform[:3,:3])
translation_inv = -affine_transform[:3,3]
affine_inv = np.r_[np.c_[linear_transform_inv,translation_inv],np.array([[0,0,0,1]])]
barycenter_nib = nib.Nifti1Image(521*(img_barycenter/img_barycenter.max()).detach().cpu().numpy(), affine_transform)
nib.save(barycenter_nib,'barycenter_image.nii')
