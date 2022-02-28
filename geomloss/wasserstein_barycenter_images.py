import torch
from .utils import log_dens, pyramid, upsample
from .utils import softmin_grid as softmin


def barycenter_iteration(f_k, g_k, d_log, eps, p, ak_log, w_k):

    # Sinkhorn "pseudo-step" - from the measures to the barycenter:
    ft_k = softmin(eps, p, ak_log + g_k / eps) / eps  # (B,K,n,n)
    # Update the barycenter:
    # (B,1,n,n) = (B,1,n,n) - (B,K,n,n) @ (B,K,1,1)
    bar_log = d_log - (ft_k * w_k[:, :, None, None]).sum(1, keepdim=True)

    # Symmetric Sinkhorn updates:
    # From the measures to the barycenter:
    ft_k = softmin(eps, p, ak_log + g_k / eps)  # (B,K,n,n)
    # From the barycenter to the measures:
    gt_k = softmin(eps, p, bar_log + f_k / eps)  # (B,K,n,n)
    f_k = (f_k + ft_k) / 2
    g_k = (g_k + gt_k) / 2

    # Sinkhorn "pseudo-step" - from the measures to the barycenter:
    ft_k = softmin(eps, p, ak_log + g_k / eps) / eps
    # Update the barycenter:
    # (B,1,n,n) = (B,1,n,n) - (B,K,n,n) @ (B,K,1,1)
    bar_log = d_log - (ft_k * w_k[:, :, None, None]).sum(1, keepdim=True)

    # Update the de-biasing measure:
    # (B,1,n,n) = (B,1,n,n) + (B,1,n,n) + (B,1,n,n)
    d_log = 0.5 * (d_log + bar_log + softmin(eps, p, d_log) / eps)

    return f_k, g_k, d_log, bar_log


def ImagesBarycenter(
    measures, weights, blur=0, p=2, scaling_N=10, backward_iterations=5
):

    a_k = measures  # Densities, (B,K,N,N)
    w_k = weights  # Barycentric weights, (B,K)

    # Default precision settings: blur = pixel size.
    if blur == 0:
        blur = 1 / measures.shape[-1]

    with torch.set_grad_enabled(backward_iterations == 0):

        # Initialize the barycenter as a pointwise linear combination:
        bar = (a_k * w_k[:, :, None, None]).sum(1)  # (B,K,N,N) @ (B,K,1,1) -> (B,N,N)

        # Pre-compute a multiscale decomposition (=QuadTree)
        # of the input measures, stored as logarithms
        ak_s = pyramid(a_k)[1:]  # We remove the 1x1 image, keep the 2x2, 4x4...
        ak_log_s = list(map(log_dens, ak_s))  # The code below relies on log-sum-exps

        # Initialize the blur scale at 1, i.e. the full image length:
        sigma = 1  # sigma = blur scale
        eps = sigma ** p  # eps = temperature

        # Initialize the dual variables
        f_k, g_k = softmin(eps, p, ak_log_s[0]), softmin(eps, p, ak_log_s[0])

        # Logarithm of the debiasing term:
        d_log = torch.ones_like(ak_log_s[0]).sum(dim=1, keepdim=True)  # (B,1,2,2)
        d_log = d_log - d_log.logsumexp(
            [2, 3], keepdim=True
        )  # Normalize each 2x2 image

        # Multiscale descent, with eps-scaling:
        # We iterate over sub-sampled images of shape nxn = 2x2, 4x4, ..., NxN
        for n, ak_log in enumerate(ak_log_s):
            for _ in range(scaling_N):  # Number of steps per scale
                # Update the temperature:
                eps = sigma ** p

                f_k, g_k, d_log, bar_log = barycenter_iteration(
                    f_k, g_k, d_log, eps, p, ak_log, w_k
                )

                # Decrease the kernel radius, making sure that
                # sigma is divided by two at every scale until we reach
                # the target value, "blur":
                sigma = max(sigma * (2 ** (-1 / scaling_N)), blur)

            if n + 1 < len(ak_s):  # Re-fine the maps, if needed
                f_k = upsample(f_k)
                g_k = upsample(g_k)
                d_log = upsample(d_log)

    if (measures.requires_grad or weights.requires_grad) and backward_iterations > 0:
        for _ in range(backward_iterations):
            f_k, g_k, d_log, bar_log = barycenter_iteration(
                f_k, g_k, d_log, eps, p, ak_log, w_k
            )

    return bar_log.exp()
