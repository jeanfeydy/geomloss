"""
Profiling the GeomLoss routines
===================================

This example explains how to **profile** the geometric losses
to select the backend and truncation/scaling values that
are best suited to your data.
"""



##############################################
# Setup
# ---------------------

import torch
from geomloss import SamplesLoss
from time import time

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

##############################################
# Sample points on the unit sphere:
# 

N, M = (250, 250) if not use_cuda else (5000, 5000) 
x, y = torch.randn(N,3).type(dtype), torch.randn(M,3).type(dtype)
x, y = x / x.norm(dim=1,keepdim=True), y / y.norm(dim=1,keepdim=True)
x.requires_grad = True

##########################################################
# Use the PyTorch profiler to output Chrome trace files:

for backend in ["tensorized", "online", "multiscale"]:
    with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
        loss = SamplesLoss("gaussian", blur=.1, backend=backend, truncate=3)
        t_0 = time()
        L_xy = loss(x, y)
        L_xy.backward()
        t_1 = time()
        print("{:.2f}s, cost = {:.6f}".format( t_1-t_0, L_xy.item()) )

    prof.export_chrome_trace("output/profile_"+backend+".json")


######################################################################
# Now, all you have to do is to open the "Easter egg" address
# ``chrome://tracing`` in Google Chrome/Chromium, 
# and load the ``profile_*`` files one after
# another. Enjoy :-)

print("Done.")