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
import time
from matplotlib import pyplot as plt

import importlib
import torch

use_cuda = torch.cuda.is_available()

from geomloss import SamplesLoss

MAXTIME = 10 if use_cuda else 1  # Max number of seconds before we break the loop
REDTIME = (
    2 if use_cuda else 0.2
)  # Decrease the number of runs if computations take longer than 2s...
D = 3  # Let's do this in 3D

# Number of samples that we'll loop upon
NS = [
    100,
    200,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
    200000,
    500000,
    1000000,
]


##############################################
# Synthetic dataset. Feel free to use
# a Stanford Bunny, or whatever!


def generate_samples(N, device):
    """Create point clouds sampled non-uniformly on a sphere of diameter 1."""

    x = torch.randn(N, D, device=device)
    x[:, 0] += 1
    x = x / (2 * x.norm(dim=1, keepdim=True))

    y = torch.randn(N, D, device=device)
    y[:, 1] += 2
    y = y / (2 * y.norm(dim=1, keepdim=True))

    x.requires_grad = True

    # Draw random weights:
    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)

    # And normalize them:
    a = a.abs()
    b = b.abs()
    a = a / a.sum()
    b = b / b.sum()

    return a, x, b, y


##############################################
# Benchmarking loops.


def benchmark(Loss, dev, N, loops=10):
    """Times a loss computation+gradient on an N-by-N problem."""

    # NB: torch does not accept reloading anymore.
    # importlib.reload(torch)  # In case we had a memory overflow just before...
    device = torch.device(dev)
    a, x, b, y = generate_samples(N, device)

    # We simply benchmark a Loss + gradien wrt. x
    code = "L = Loss( a, x, b, y ) ; L.backward()"
    Loss.verbose = True
    exec(code, locals())  # Warmup run, to compile and load everything
    Loss.verbose = False

    t_0 = time.perf_counter()  # Actual benchmark --------------------
    if use_cuda:
        torch.cuda.synchronize()
    for i in range(loops):
        exec(code, locals())
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_0  # ---------------------------

    print(
        "{:3} NxN loss, with N ={:7}: {:3}x{:3.6f}s".format(
            loops, N, loops, elapsed / loops
        )
    )
    return elapsed / loops


def bench_config(Loss, dev):
    """Times a loss computation+gradient for an increasing number of samples."""

    print("Backend : {}, Device : {} -------------".format(Loss.backend, dev))

    times = []

    def run_bench():
        try:
            Nloops = [100, 10, 1]
            nloops = Nloops.pop(0)
            for n in NS:
                elapsed = benchmark(Loss, dev, n, loops=nloops)

                times.append(elapsed)
                if (nloops * elapsed > MAXTIME) or (
                    nloops * elapsed > REDTIME and len(Nloops) > 0
                ):
                    nloops = Nloops.pop(0)

        except IndexError:
            print("**\nToo slow !")

    try:
        run_bench()

    except RuntimeError as err:
        if str(err)[:4] == "CUDA":
            print("**\nMemory overflow !")

        else:
            # CUDA memory overflows semi-break the internal
            # torch state and may cause some strange bugs.
            # In this case, best option is simply to re-launch
            # the benchmark.
            run_bench()

    return times + (len(NS) - len(times)) * [np.nan]


def full_bench(loss, *args, **kwargs):
    """Benchmarks the varied backends of a geometric loss function."""

    print("Benchmarking : ===============================")

    lines = [NS]
    backends = ["tensorized", "online", "multiscale"]
    for backend in backends:
        Loss = SamplesLoss(*args, **kwargs, backend=backend)
        lines.append(bench_config(Loss, "cuda" if use_cuda else "cpu"))

    benches = np.array(lines).T

    # Creates a pyplot figure:
    plt.figure()
    linestyles = ["o-", "s-", "^-"]
    for i, backend in enumerate(backends):
        plt.plot(
            benches[:, 0],
            benches[:, i + 1],
            linestyles[i],
            linewidth=2,
            label='backend="{}"'.format(backend),
        )

    plt.title('Runtime for SamplesLoss("{}") in dimension {}'.format(Loss.loss, D))
    plt.xlabel("Number of samples per measure")
    plt.ylabel("Seconds")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.grid(True, which="major", linestyle="-")
    plt.grid(True, which="minor", linestyle="dotted")
    plt.axis([NS[0], NS[-1], 1e-3, MAXTIME])
    plt.tight_layout()

    # Save as a .csv to put a nice Tikz figure in the papers:
    header = "Npoints " + " ".join(backends)
    np.savetxt(
        "output/benchmark_" + Loss.loss + "_3D.csv",
        benches,
        fmt="%-9.5f",
        header=header,
        comments="",
    )


##############################################
# Gaussian MMD, with a small blur
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

full_bench(SamplesLoss, "gaussian", blur=0.1, truncate=3)


##############################################
# Energy Distance MMD
# ~~~~~~~~~~~~~~~~~~~~~~
#

full_bench(SamplesLoss, "energy")


##############################################
# Sinkhorn divergence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# With a medium blurring scale, at one twentieth of the
# configuration's diameter:

full_bench(SamplesLoss, "sinkhorn", p=2, blur=0.05, diameter=1)


##############################################
# With a small blurring scale, at one hundredth of the
# configuration's diameter:

full_bench(SamplesLoss, "sinkhorn", p=2, blur=0.01, diameter=1)

plt.show()
