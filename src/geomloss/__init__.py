import sys, os.path

__version__ = "0.3.1"

from ._legacy.samples_loss import SamplesLoss
from ._legacy.wasserstein_barycenter_images import ImagesBarycenter
from ._legacy.sinkhorn_images import sinkhorn_divergence

__all__ = sorted(["SamplesLoss, ImagesBarycenter"])
