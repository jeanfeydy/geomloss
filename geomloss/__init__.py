import sys, os.path

__version__ = "0.3.0"

from .samples_loss import SamplesLoss
from .wasserstein_barycenter_images import ImagesBarycenter
from .sinkhorn_images import sinkhorn_divergence

__all__ = sorted(["SamplesLoss, ImagesBarycenter"])
