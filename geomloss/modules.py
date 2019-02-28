import torch
from torch.nn import Module
from functools import partial

routines = {
    "sinkhorn" : {
        "tensorized" : sinkhorn_tensorized,
        "online"     : sinkhorn_online,
        "multiscale" : sinkhorn_multiscale,
        },
    "hausdorff" : {
        "tensorized" : hausdorff_tensorized,
        "online"     : hausdorff_online,
        "multiscale" : hausdorff_multiscale,
        },
    "energy" : {
        "tensorized" : partial( kernel_tensorized, name="energy" ),
        "online"     : partial( kernel_online,     name="energy" ),
        "multiscale" : partial( kernel_multiscale, name="energy" ),
    },
    "gaussian" : {
        "tensorized" : partial( kernel_tensorized, name="gaussian" ),
        "online"     : partial( kernel_online,     name="gaussian" ),
        "multiscale" : partial( kernel_multiscale, name="gaussian" ),
    },
    "laplacian" : {
        "tensorized" : partial( kernel_tensorized, name="laplacian" ),
        "online"     : partial( kernel_online,     name="laplacian" ),
        "multiscale" : partial( kernel_multiscale, name="laplacian" ),
    },
}


class SamplesLoss(Module):
    """Creates a criterion that computes distances between sampled measures on a vector space.

    Blabla measures.

    Warning:
        If **loss** is ``"sinkhorn"`` and 

    Parameters:
        loss (string, default = ``"sinkhorn"``): The loss function to compute.
            The supported values are:

              - ``"sinkhorn"``: (Un-biased) Sinkhorn divergence, which interpolates
                between Wasserstein (blur=0) and kernel (blur=:math:`+\infty`).
              - ``"hausdorff"``: Weighted Hausdorff distance, which interpolates
                between
              - ``"energy"``: Energy Distance MMD, computed using the kernel
                :math:`k(x,y) = -\|x-y\|_2`.
              - ``"gaussian"``: Gaussian MMD, computed using the kernel
                :math:`k(x,y) = \exp \\big( -\|x-y\|_2^2 \,/\, 2\sigma^2)`
                of standard deviation :math:`\sigma` = ``blur``.
              - ``"laplacian"``: Laplacian MMD, computed using the kernel
                :math:`k(x,y) = \exp \\big( -\|x-y\|_2 \,/\, \sigma)`
                of standard deviation :math:`\sigma` = ``blur``.
        
        p (int, default=2): If **loss** is ``"sinkhorn"`` or ``"hausdorff"``,
            specifies the ground cost function between points.
            The supported values are:
              - 1: :math:`C(x,y) ~=~ \|x-y\|_2`.
              - 2: :math:`C(x,y) ~=~ \\tfrac{1}{2}\|x-y\|_2^2`.
        
        blur (float, default=.05): The finest level of detail that
            should be handled by the loss function - in
            order to prevent overfitting on the samples' locations.
            It is typically set to a fraction of the diameter
            of the input configuration.
            
            - If **loss** is ``"gaussian"`` or ``"laplacian"``,
              it is the standard deviation :math:`\sigma` of the convolution kernel.
            - If **loss** is ``"sinkhorn"`` or ``"haudorff"``,
              it is the typical scale :math:`\sigma` associated
              to the temperature :math:`\epsilon = \sigma^p`.
              The default value of .05 is sensible for input
              measures that lie in the unit square/cube.

            Note that the *Energy Distance* is scale-equivariant, and won't 
            be affected by this parameter.

        reach (float, default=None=:math:`+\infty`): If 

        scaling (float, default=.5): If **loss** is ``"sinkhorn"``,
            specifies the ratio between successive values
            of :math:`\sigma=\epsilon^{1/p}` in the
            :math:`\epsilon`-scaling descent.

        truncate (float, default=None=:math:`+\infty`):

        cost (function, default=None):


        backend (string, default = ``"auto"``): The implementation that
            will be used in the background; this choice has a major impact
            on performance. The supported values are:

              - ``"auto"``: Choose automatically, depending on the inputs' shapes.
              - ``"tensorized"``: Relies on a full cost/kernel matrix, computed
                once and for all and stored on the device memory. 
                This method is fast, but has a quadratic
                memory footprint and does not scale beyond ~5,000 samples per measure.
              - ``"online"``: Computes cost/kernel values on-the-fly, leveraging
                online map-reduce CUDA routines provided by 
                the `pykeops <www.kernel-operations.io>`_ library.
              - ``"multiscale"``: Fast implementation that scales to millions
                of samples in dimension 1-2-3, relying on the block-sparse
                reductions provided by the `pykeops <www.kernel-operations.io>`_ library.

    """
    def __init__(self, loss="sinkhorn", p=2, blur=.05, reach=None, 
                       scaling=.5, truncate=None, cost=None, kernel=None, backend="auto"):

        super(SamplesLoss, self).__init__()
        self.loss = loss
        self.backend = backend
        self.p = p
        self.blur = blur
        self.reach = reach
        self.truncate = truncate
        self.scaling = scaling
        self.cost = cost
        self.kernel = kernel

    def forward(self, α, x, β, y):
        M, N = len(α), len(β)
        D = α.shape[1]

        backend = self.backend
        if backend == "auto":
            if M*N <= 5000**2 : backend = "tensorized"
            elif D <= 3:        backend = "multiscale"
            else :              backend = "online"

        return routines[self.loss][backend]( α, x, β, y, 
                    p = self.p, blur = self.blur, reach = self.reach, scaling = self.scaling, 
                    truncate = self.truncate, cost = self.cost, kernel = self.kernel )