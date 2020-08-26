"""
Multiple targets in batch mode
====================================

Let's consider multiple ways to use the
loss function when working with multiple targets.
"""

####################################
# Setup
# -------------------

import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from geomloss import SamplesLoss
from random import choices
from imageio import imread
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def load_image(fname) :
    img = imread(fname, as_gray = True) # Grayscale
    img = (img[::-1, :])  / 255.
    return 1 - img

def draw_samples(fname, n, dtype=torch.FloatTensor) :
    A = load_image(fname)
    xg, yg = np.meshgrid( np.linspace(0,1,A.shape[0]), np.linspace(0,1,A.shape[1]) )

    grid = list( zip(xg.ravel(), yg.ravel()) )
    dens = A.ravel() / A.sum()
    dots = np.array( choices(grid, dens, k=n ) )
    dots += (.5/A.shape[0]) * np.random.standard_normal(dots.shape)
    return torch.from_numpy(dots).type(dtype)


##############################
# Padding warning
# ------------------------
#
# Let's consider the case in :math:`\mathbb{R}^2` where you have one source 
# :math:`x` of size :math:`N` and two targets to consider :math:`y` and :math:`z` of size
# :math:`M` and :math:`P`. This means we have the three probability measures:
#
# .. math::
#       \alpha ~=~ \frac{1}{N}\sum_{i=1}^N \delta_{x_i}, ~~~ \beta ~=~ \frac{1}{M}\sum_{j=1}^M \delta_{y_j}, ~~~ \gamma ~=~ \frac{1}{P}\sum_{j=1}^P \delta_{z_j}.
#

N, M, P = (100, 200, 100) if not use_cuda else (5000, 1000, 2000)

Y_j = draw_samples("../comparisons/data/density_b.png", M, dtype)
X_i = draw_samples("../comparisons/data/density_a.png", N, dtype)
Z_j = draw_samples("../comparisons/data/density_c.png", P, dtype)

x_i, y_j, z_j = X_i.clone(), Y_j.clone(), Z_j.clone() # make sure we don't change it

####################
# We use the Wasserstein-:math:`2` distance with the ``tensorized``` backend.

loss = SamplesLoss("sinkhorn", p=2, blur=.05, backend="tensorized")

######################
# When :math:`M=P`, both targets can be put in the same tensor without
# thinking twice about it. When :math:`M\neq P,\ M<P`, one must consider that
# padding :math:`y_j` with zeros will induce a ill-apportionment of the weights 
# as the padding will now be considered with a weight of :math:`\frac{(P-M)}{P}`.

x_ = x_i.unsqueeze(0).repeat(2,1,1) # 2 tiles of x_i

padding = max(y_j.shape[0], z_j.shape[0])

t_ = torch.stack([ # 2 targets as 1 tensor
    F.pad(y_j, (0, 0, 0, padding - y_j.shape[0])),
    F.pad(z_j, (0, 0, 0, padding - z_j.shape[0]))
    ])

print("----------- 1 source to 1 target ----------")
x_to_y = loss(x_i, y_j)
x_to_z = loss(x_i, z_j)

print("loss (alpha,x) -> (beta,y):", x_to_y.item())
print("loss (alpha,x) -> (gamma,z):", x_to_z.item())

print("----------- 1 source to 2 targets (batches) ---------")
x_to_yz = loss(x_, t_)
print("Is loss close for the 1st target?", torch.isclose(x_to_y, x_to_yz[0], atol=1e-3).item())
print("Is loss close for the 2nd target?", torch.isclose(x_to_z, x_to_yz[1], atol=1e-3).item())

###################################
# Solution 1: give padded weights
# ---------------------------------
# To avoid taking into account the padded zeros, one can use the weights
# and assign the nought-value to the padded elements. Then :math:`\beta` becomes:
#
# .. math::
#        \beta ~=~ \sum_{j=1}^M \delta_{y_j} + \sum_{k=M+1}^P 0.
#

def get_weights(sample): # uniform distribution
    if sample.dim() == 2:  # 
        N = sample.shape[0]
        return torch.ones(N).type_as(sample) / N
    elif sample.dim() == 3:
        B, N, _ = sample.shape
        return torch.ones(B,N).type_as(sample) / N

alpha = get_weights(x_i)
beta = get_weights(y_j)
gamma = get_weights(z_j)

alpha_ = get_weights(x_) # 2 tiles of alpha
weights_ = torch.stack([ # padded measures
    F.pad(beta, (0, padding - y_j.shape[0])),
    F.pad(gamma, (0, padding - z_j.shape[0]))    
])

print("----------- Giving α_, x_, β_, y_ (the stacked versions) ---------")
stacked_alpha_t_ = loss(alpha_, x_, weights_, t_)
print("Is loss close for the 1st target?", torch.isclose(x_to_y, stacked_alpha_t_[0], atol=1e-3).item())
print("Is loss close for the 2nd target?", torch.isclose(x_to_z, stacked_alpha_t_[1], atol=1e-3).item())

###########################
# Solution 2: using lists
# --------------------------
# When using batches, we often store our data in a list
# to iterate over later. ``SamplesLoss`` is compatible with
# lists with or without precising the weights (in the later case,
# the uniform distribution is applied).
#
# Giving weights and targets as lists.

print("----------- Giving x_, α_ and y, β as lists ---------")
all_list = loss(alpha_, x_i, [beta, gamma], [y_j, z_j])
print("Is loss close for the 1st target?", torch.isclose(x_to_y, all_list[0], atol=1e-3).item())
print("Is loss close for the 2nd target?", torch.isclose(x_to_z, all_list[1], atol=1e-3).item())

#######################
# .. note::
#        Here, giving the list of weights induces that the target weights at the end must have the
#        same dimension than the source weights in input. To avoid any shapes issues, one can give in input
#        either ``x_i`` or ``x_`` in this situation. 

#######################
# Giving targets as a list with uniform weights.

print("----------- Giving x, y (y as a list, nothing stacked or padded) ---------")
no_weight = loss(x_i, [y_j, z_j])
print("Is loss close for the 1st target?", torch.isclose(x_to_y, no_weight[0], atol=1e-3).item())
print("Is loss close for the 2nd target?", torch.isclose(x_to_z, no_weight[1], atol=1e-3).item())

######################
# Small benchmark 
# -------------------
# Besides being more user-friendly, this method isn't costlier than doing
# a naive ``for`` loop over the batches.
#

import time

def bench_list(source, target):
    loss = SamplesLoss("sinkhorn", p=2, blur=.01, backend="tensorized")
    start = time.perf_counter()
    for _ in range(10):
        loss(source, target)
    end = time.perf_counter()
    return (end - start) / 10

def bench_old_batch(source, target):
    loss = SamplesLoss("sinkhorn", p=2, blur=.01, backend="tensorized")
    start = time.perf_counter()
    for _ in range(10):
        for j in range(len(target)):
            loss(source, target[j])
    end = time.perf_counter()
    return (end - start) / 10


def bench_weights_list(source, target):
    # the target is 2 batchs of dim 2 and the source is of dim 2 too
    source_ = source.unsqueeze(0).repeat(2,1,1)
    alpha = get_weights(source_)
    beta = [get_weights(target[i]) for i in range(len(target))]

    loss = SamplesLoss("sinkhorn", p=2, blur=.01, backend="tensorized")
    start = time.perf_counter()
    for _ in range(10):
        loss(alpha, source_, beta, target)
    end = time.perf_counter()
    return (end - start) / 10

def bench_weights_loop(source, target):
    loss = SamplesLoss("sinkhorn", p=2, blur=.01, backend="tensorized")
    alpha = get_weights(source)
    beta = [get_weights(target[i]) for i in range(len(target))]
    start = time.perf_counter()
    for _ in range(10):
        for j in range(len(target)):
            loss(alpha, source, beta[j], target[j])
    end = time.perf_counter()
    return (end - start) / 10


def bench(source, target, func):
    if func == "list":
        return bench_list(source, target)
    elif func == "loop":
        return bench_old_batch(source, target)
    elif func == "weight_list":
        return bench_weights_list(source, target)
    elif func == "weight_loop":
        return bench_weights_loop(source, target)

def make_bench(X_i, Y_j, Z_j):
    print("Begin bench for increasing number of batch --------------------")
    source = X_i.clone()
    time_batch, time_loop = [], []
    n = []
    target = [Y_j, Z_j] * 8
    for i in range(1, len(target) + 1):
        try:
            time_batch.append(bench(source, target[0:i], func="list"))
            time_loop.append(bench(source, target[0:i], func="loop"))
            print("Finished {} targets in {:.3f}s for the 'loop' version and {:.3f}s for the batched.".format(i, time_loop[i-1], time_batch[i-1]))

        except:
            time_batch.append(np.nan)
            time_loop.append(np.nan)
        n.append(i)

    return n, time_loop, time_batch


def mobile_bench(diffs, N):
    print("Begin bench for different sizes of 2 batch -----------------------")
    time_loop, time_batch = [], []
    for idx, val in enumerate(diffs):
        X_i = draw_samples("../comparisons/data/density_a.png", N, dtype)
        Y_j = draw_samples("../comparisons/data/density_b.png", N - val, dtype)
        Z_j = draw_samples("../comparisons/data/density_c.png", N + val, dtype)
        target = [Y_j, Z_j]
        source = X_i.clone()
        try:
            time_batch.append(bench(source, target, func="list"))
            time_loop.append(bench(source, target, func="loop"))
            print("Finished diff {} in {:.3f}s for the 'loop' version and {:.3f}s for the batched.".format(val, time_loop[idx], time_batch[idx]))

        except:
            time_batch.append(np.nan)
            time_loop.append(np.nan)

    return diffs, time_loop, time_batch


def weight_batch(diffs, N):
    print("Begin bench for different sizes with given weights of 2 batch -----------------------")
    time_loop, time_batch = [], []
    for idx, val in enumerate(diffs):
        X_i = draw_samples("../comparisons/data/density_a.png", N, dtype)
        Y_j = draw_samples("../comparisons/data/density_b.png", N - val, dtype)
        Z_j = draw_samples("../comparisons/data/density_c.png", N + val, dtype)
        target = [Y_j, Z_j]
        source = X_i.clone()
        try:
            time_batch.append(bench(source, target, func="weight_list"))
            time_loop.append(bench(source, target, func="weight_loop"))
            print("Finished diff {} in {:.3f}s for the 'loop' version and {:.3f}s for the batched.".format(val, time_loop[idx], time_batch[idx]))

        except:
            time_batch.append(np.nan)
            time_loop.append(np.nan)

    return diffs, time_loop, time_batch


def plot_bench():

    N, M, P = (100, 200, 100) if not use_cuda else (3000, 2100, 2000)
    diffs = [10, 100, 200, 500, 750, 1000, 1500, 2000, 2500, 3000] if use_cuda else [10, 100, 200, 300, 500]


    X_i = draw_samples("../comparisons/data/density_a.png", N, dtype)
    Y_j = draw_samples("../comparisons/data/density_b.png", M, dtype)
    Z_j = draw_samples("../comparisons/data/density_c.png", P, dtype)

    n, time_loop, time_batch = make_bench(X_i, Y_j, Z_j)

    N = 5000 if use_cuda else 700
    diffs, time_loop_diffs, time_batch_diffs = mobile_bench(diffs, N)
    _, weight_loop, weight_list = weight_batch(diffs, N)

    plt.style.use("ggplot")
    plt.figure()
    plt.title("Targets of size {} and {} for a source of size {}".format(M, P, N))
    plt.plot(n, time_batch, label="batched version")
    plt.plot(n, time_loop, label="loop over the targets")
    plt.xlabel("Number of targets to compute the cost")
    plt.ylabel("time (s)")
    plt.ylim([5e-2,2])
    plt.yscale('log')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title(r"Targets of size {}$-diff$ and {}$+ diff$ for a source of size {}".format(N, N, N))
    plt.plot(diffs, time_batch_diffs, label="batched version")
    plt.plot(diffs, time_loop_diffs, label="loop over the targets")
    plt.plot(diffs, weight_list, label="batched weights given")
    plt.plot(diffs, weight_loop, label="loop weights given")
    plt.xlabel(r"Parameter $diff$ in the number of points used")
    plt.ylabel("time (s)")
    plt.ylim([1e-1, 1])
    plt.yscale('log'); plt.xscale("log")
    plt.legend()
    plt.show()


plot_bench()

###########################################
# So, as long as there isn't a large difference :math:`(>3000)` between the number of points
# in the two batches, there is no cost in time to use the more practical way.
