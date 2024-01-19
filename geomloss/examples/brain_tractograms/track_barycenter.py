"""
Create an atlas using Wasserstein barycenters
==================================================

In this tutorial, we compute the barycenter of a dataset of probability tracks. 
The barycenter is computed as the Fréchet mean for the Sinkhorn divergence, using a Lagrangian optimization scheme. 
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
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
from scipy.interpolate import RegularGridInterpolator


import gzip
import shutil
import pdb


import nibabel as nib
import matplotlib.pyplot as plt


###############################################
# Dataset
# ~~~~~~~~~~~~~~~~~~
#
# In this tutorial, we work with probability tracks, that can be understood as normalized 3D images. We will compute the Wasserstein barycenter of this dataset.


import os


def fetch_file(name):
    if not os.path.exists(f"data/{name}.nii.gz"):
        import urllib.request

        print("Fetching the atlas... ", end="", flush=True)
        urllib.request.urlretrieve(
            f"https://www.kernel-operations.io/data/{name}.nii.gz",
            f"data/{name}.nii.gz",
        )
        with gzip.open(f"data/{name}.nii.gz", "rb") as f_in:
            with open(f"data/{name}.nii", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("Done.")


for i in range(5):
    fetch_file(f"manual_ifof{i+1}")


affine_transform = nib.load("data/manual_ifof1.nii").affine


# load data in the nii format to a 3D, normalized array.
def load_data_nii(fname):
    img = nib.load(fname)
    affine_mat = img.affine
    hdr = img.header
    data = img.get_fdata()
    data_norm = data / np.max(data)
    data_norm = torch.from_numpy(data_norm).type(dtype)
    return data_norm


def grid(nx, ny, nz):
    x, y, z = torch.meshgrid(
        torch.arange(0.0, nx).type(dtype),
        torch.arange(0.0, ny).type(dtype),
        torch.arange(0.0, nz).type(dtype),
        indexing="ij",
    )
    return torch.stack((x, y, z), dim=3).view(-1, 3).detach().cpu().numpy()


# load the data set (here, we have 5 subjects)
dataset = []
for i in range(5):
    fname = "data/manual_ifof" + str(i + 1) + ".nii"
    image_norm = load_data_nii(fname)
    print(image_norm.shape)
    dataset.append(image_norm)

###############################################
# In this tutorial, we work with 3D images, understood as densities on the cube.


def img_to_points_cloud(data_norm):  # normalized images (between 0 and 1)
    nx, ny, nz = data_norm.shape
    ind = data_norm.nonzero()
    indx = ind[:, 0]
    indy = ind[:, 1]
    indz = ind[:, 2]
    data_norm = data_norm / data_norm.sum()
    a_i = data_norm[indx, indy, indz]

    return ind.type(dtype), a_i


def measure_to_image(x, nx, ny, nz, weights=None):
    bins = (x[:, 2]).floor() + nz * (x[:, 1]).floor() + nz * ny * (x[:, 0]).floor()
    count = bins.int().bincount(weights=weights, minlength=nx * ny * nz)
    return count.view(nx, ny, nz)


###############################################################################
# To perform our computations, we turn these 3D arrays into weighted point cloud, regularly spaced in the grid.


a, b = img_to_points_cloud(dataset[0]), img_to_points_cloud(dataset[1])
c, d, e = (
    img_to_points_cloud(dataset[2]),
    img_to_points_cloud(dataset[3]),
    img_to_points_cloud(dataset[4]),
)

###############################################################################
# We initialize the barycenter as an upsampled, arithmetic mean of the data set.


nx, ny, nz = image_norm.shape


def initialize_barycenter(dataset):
    mean = torch.zeros(nx, ny, nz).type(dtype)
    for k in range(len(dataset)):
        img = dataset[k]
        mean = mean + img
    mean = mean / len(dataset)
    x_i, a_i = img_to_points_cloud(mean)
    bar_pos, bar_weight = torch.tensor([]).type(dtype), torch.tensor([]).type(dtype)
    for d in range(3):
        x_i_d1, x_i_d2 = x_i.clone(), x_i.clone()
        x_i_d1[:, d], a_i_d1 = x_i_d1[:, d] + 0.25, a_i / 6
        x_i_d2[:, d], a_i_d2 = x_i_d2[:, d] - 0.25, a_i / 6
        bar_pos, bar_weight = torch.cat((bar_pos, x_i_d1, x_i_d2), 0), torch.cat(
            (bar_weight, a_i_d1, a_i_d2), 0
        )
    return bar_pos, bar_weight


x_i, a_i = initialize_barycenter(dataset)

###############################################################################
# The barycenter will be the minimizer of the sum of Sinkhorn distances to the dataset.
# It is computed through a Lagrangian gradient descent on the particles' positions.

Loss = SamplesLoss("sinkhorn", blur=1, scaling=0.9, debias=False)
models = []
x_i.requires_grad = True


start = time.time()
for j in range(len(dataset)):
    img_j = dataset[j]
    y_j, b_j = img_to_points_cloud(img_j)
    L_ab = Loss(a_i, x_i, b_j, y_j)
    [g_i] = torch.autograd.grad(L_ab, [x_i])
    models.append(x_i - g_i / a_i.view(-1, 1))

a, b, c, d, e = models
barycenter = (a + b + c + d + e) / 5
if use_cuda:
    torch.cuda.synchronize()
end = time.time()
print("barycenter computed in {:.3f}s.".format(end - start))

##############################################################################
# We can plot slices of the computed barycenters
img_barycenter = measure_to_image(barycenter, nx, ny, nz, a_i)
plt.figure()
plt.imshow(img_barycenter.detach().cpu().numpy()[20, :, :])
plt.show()

#############################################################################
# Or save the 3D image in .nii format, once put in the same coordinates system as the data images.
linear_transform_inv = np.linalg.inv(affine_transform[:3, :3])
translation_inv = -affine_transform[:3, 3]
affine_inv = np.r_[
    np.c_[linear_transform_inv, translation_inv], np.array([[0, 0, 0, 1]])
]
barycenter_nib = nib.Nifti1Image(
    521 * (img_barycenter / img_barycenter.max()).detach().cpu().numpy(),
    affine_transform,
)
nib.save(barycenter_nib, "barycenter_image.nii")
