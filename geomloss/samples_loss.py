import torch
from torch.nn import Module
from functools import partial
import warnings

from .kernel_samples import kernel_tensorized, kernel_online, kernel_multiscale

from .sinkhorn_samples import sinkhorn_tensorized
from .kernel_samples import kernel_online     as sinkhorn_online
from .kernel_samples import kernel_multiscale as sinkhorn_multiscale

from .kernel_samples import kernel_tensorized as hausdorff_tensorized
from .kernel_samples import kernel_online     as hausdorff_online
from .kernel_samples import kernel_multiscale as hausdorff_multiscale


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
                between Wasserstein (blur=0) and kernel (blur= :math:`+\infty` ).
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

              - **p** = 1: :math:`~~C(x,y) ~=~ \|x-y\|_2`.
              - **p** = 2: :math:`~~C(x,y) ~=~ \\tfrac{1}{2}\|x-y\|_2^2`.
        
        blur (float, default=.05): The finest level of detail that
            should be handled by the loss function - in
            order to prevent overfitting on the samples' locations.
            
            - If **loss** is ``"gaussian"`` or ``"laplacian"``,
              it is the standard deviation :math:`\sigma` of the convolution kernel.
            - If **loss** is ``"sinkhorn"`` or ``"haudorff"``,
              it is the typical scale :math:`\sigma` associated
              to the temperature :math:`\\varepsilon = \sigma^p`.
              The default value of .05 is sensible for input
              measures that lie in the unit square/cube.

            Note that the *Energy Distance* is scale-equivariant, and won't 
            be affected by this parameter.

        reach (float, default=None= :math:`+\infty` ): If 

        scaling (float, default=.5): If **loss** is ``"sinkhorn"``,
            specifies the ratio between successive values
            of :math:`\sigma=\\varepsilon^{1/p}` in the
            :math:`\\varepsilon`-scaling descent.

        truncate (float, default=None=:math:`+\infty`):

        cost (function, default=None):


        backend (string, default = ``"auto"``): The implementation that
            will be used in the background; this choice has a major impact
            on performance. The supported values are:

              - ``"auto"``: Choose automatically, using a simple
                heuristic based on the inputs' shapes.
              - ``"tensorized"``: Relies on a full cost/kernel matrix, computed
                once and for all and stored on the device memory. 
                This method is fast, but has a quadratic
                memory footprint and does not scale beyond ~5,000 samples per measure.
              - ``"online"``: Computes cost/kernel values on-the-fly, leveraging
                online map-reduce CUDA routines provided by 
                the `pykeops <https://www.kernel-operations.io>`_ library.
              - ``"multiscale"``: Fast implementation that scales to millions
                of samples in dimension 1-2-3, relying on the block-sparse
                reductions provided by the `pykeops <https://www.kernel-operations.io>`_ library.

    """
    def __init__(self, loss="sinkhorn", p=2, blur=.05, reach=None, diameter=None, scaling=.5, truncate=5, cost=None, kernel=None, backend="auto"):

        super(SamplesLoss, self).__init__()
        self.loss = loss
        self.backend = backend
        self.p = p
        self.blur = blur
        self.reach = reach
        self.truncate = truncate
        self.diameter = diameter
        self.scaling = scaling
        self.cost = cost
        self.kernel = kernel


    def forward(self, *args):
        """Computes the loss between sampled measures."""
        
        α, x, β, y = self.process_args(*args)
        B, N, M, D = self.check_shapes(α, x, β, y)

        backend = self.backend  # Choose the backend -----------------------------------------
        if backend == "auto":
            if M*N <= 5000**2 : backend = "tensorized"  # Fast backend, with a quadratic memory footprint
            elif D <= 3:        backend = "multiscale"  # Super scalable algorithm in low dimension
            else :              backend = "online"      # Worst case scenario

        # Check compatibility between the batchsize and the backend --------------------------
        if backend in ["online", "multiscale"]:  # KeOps routines work on single measures
            if B == 1:
                α, x, β, y = α.squeeze(0), x.squeeze(0), β.squeeze(0), y.squeeze(0)
            elif B > 1:
                warnings.warn("'online' and 'multiscale' backends do not support batchsize > 1. " \
                             +"Using 'tensorized' instead: beware of memory overflows!")
                backend = "tensorized"

        if B == 0 and backend == "tensorized":  # tensorized routines work on batched tensors
            α, x, β, y = α.unsqueeze(0), x.unsqueeze(0), β.unsqueeze(0), y.unsqueeze(0)


        # Run --------------------------------------------------------------------------------
        values = routines[self.loss][backend]( α, x, β, y, 
                    p = self.p, blur = self.blur, reach = self.reach, 
                    diameter=self.diameter, scaling = self.scaling, truncate = self.truncate, 
                    cost = self.cost, kernel = self.kernel )


        # Make sure that the output has the correct shape ------------------------------------
        if backend in ["online", "multiscale"]:  # KeOps backends return a single scalar value
            if B == 0: return values           # The user expects a scalar value
            else:      return values.view(-1)  # The user expects a "batch list" of distances

        else:  # "tensorized" backend returns a "batch vector" of values
            if B == 0: return values[0]  # The user expects a scalar value
            else:      return values     # The user expects a "batch vector" of distances
                

    def process_args(self, *args):
        if len(args) == 4:
            return args
        elif len(args) == 2:
            x, y = args
            α = self.generate_weights(x)
            β = self.generate_weights(y)
            return α, x, β, y
        else:
            raise ValueError("A SamplesLoss accepts two (x, y) or four (α, x, β, y) arguments.")


    def generate_weights(self, x):
        if x.dim() == 2:  # 
            N = x.shape[0]
            return torch.ones(N).type_as(x) / N
        elif x.dim() == 3:
            B, N, _ = x.shape
            return torch.ones(B,N).type_as(x) / N
        else:
            raise ValueError("Input samples 'x' and 'y' should be encoded as (N,D) or (B,N,D) (batch) tensors.")


    def check_shapes(self, α, x, β, y):

        if α.dim() != β.dim(): raise ValueError("Input weights 'α' and 'β' should have the same number of dimensions.")
        if x.dim() != y.dim(): raise ValueError("Input samples 'x' and 'y' should have the same number of dimensions.")
        if x.shape[-1] != y.shape[-1]: raise ValueError("Input samples 'x' and 'y' should have the same last dimension.")

        if x.dim() == 2:  # No batch --------------------------------------------------------------------
            B = 0  # Batchsize
            N, D = x.shape  # Number of "i" samples, dimension of the feature space
            M, _ = y.shape  # Number of "j" samples, dimension of the feature space
            
            if α.dim() not in [1,2] :
                raise ValueError("Without batches, input weights 'α' and 'β' should be encoded as (N,) or (N,1) tensors.")
            elif α.dim() == 2:
                if α.shape[1] > 1: raise ValueError("Without batches, input weights 'α' should be encoded as (N,) or (N,1) tensors.")
                if β.shape[1] > 1: raise ValueError("Without batches, input weights 'β' should be encoded as (M,) or (M,1) tensors.")
                α, β = α.view(-1), β.view(-1)
            N2, M2 = α.shape[0], β.shape[0]

        elif x.dim() == 3:  # batch computation ---------------------------------------------------------
            B, N, D = x.shape  # Batchsize, number of "i" samples, dimension of the feature space
            B2,M, _ = y.shape  # Batchsize, number of "j" samples, dimension of the feature space
            if B != B2: raise ValueError("Samples 'x' and 'y' should have the same batchsize.")

            if α.dim() not in [2,3] :
                raise ValueError("With batches, input weights 'α' and 'β' should be encoded as (B,N) or (B,N,1) tensors.")
            elif α.dim() == 3:
                if α.shape[2] > 1: raise ValueError("With batches, input weights 'α' should be encoded as (B,N) or (B,N,1) tensors.")
                if β.shape[2] > 1: raise ValueError("With batches, input weights 'β' should be encoded as (B,M) or (B,M,1) tensors.")
                α, β = α.squeeze(-1), β.squeeze(-1)
            
            B2, N2 = α.shape
            B3, M2 = β.shape
            if B != B2: raise ValueError("Samples 'x' and weights 'α' should have the same batchsize.")
            if B != B3: raise ValueError("Samples 'y' and weights 'β' should have the same batchsize.")

        else:
            raise ValueError("Input samples 'x' and 'y' should be encoded as (N,D) or (B,N,D) (batch) tensors.")
        
        if N != N2: raise ValueError("Weights 'α' and samples 'x' should have compatible shapes.")
        if M != M2: raise ValueError("Weights 'β' and samples 'y' should have compatible shapes.")

        return B, N, M, D