from ..typing import (
    Tensor,
    Tuple,
    Optional,
    SoftMin,
    CostMatrix,
    CostFunction,
)

from ..utils import stable_log


def barycenter_iteration(
    *,
    softmin: SoftMin,
    f_k: Tensor, 
    g_k: Tensor, 
    d_log: Tensor, 
    eps: float, 
    C_xx: CostMatrix,
    C_xy: CostMatrix,
    C_yx: CostMatrix, 
    ak_log: Tensor, 
    w_k: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements a Sinkhorn iteration for the Wasserstein barycenter problem.
    
    Args:
        softmin (function): a softmin (~ soft distance) operator,
            as in the standard Sinkhorn loop.
        
        f_k ((B,K,...) Tensor): dual potentials, supported by the barycenters
            on points x[i].
        
        g_k ((B,K,...) Tensor): dual potentials, supported by the input measures
            on points y[j].
        
        d_log ((B,1,...) Tensor): de-biasing weight, supported by the barycenters
            on points x[i].
        
        eps (float > 0): the positive regularization parameter (= temperature)
            for the Sinkhorn algorithm.
        
        C_xx (CostMatrix): arbitary object that encodes the cost matrix
            C_xx[i,j] = C(x[i], x[j]) in a way that is compatible with softmin.
        
        C_xy (CostMatrix): arbitary object that encodes the cost matrix
            C_xy[i,j] = C(x[i], y[j]) in a way that is compatible with softmin.
            
        C_yx (CostMatrix): arbitary object that encodes the cost matrix
            C_yx[i,j] = C(y[i], x[j]) in a way that is compatible with softmin.
        
        ak_log ((B,K,...) Tensor): logarithms of the measure weights ("masses")
            for the input measures.
          
        w_k ((B,K) Tensor): desired weights for the barycentric interpolation. 
    """
    
    # Sinkhorn "pseudo-step" - from the measures to the barycenter:
    ft_k = softmin(eps, C_xy, ak_log + g_k / eps) / eps  # (B,K,...)
    # Update the barycenter:
    # (B,1,...) = (B,1,...) - (B,K,...) @ (B,K,1,..,1)
    # With e.g. 2 trailing dimensions, the einsum below is equivalent to:
    # bar_log = d_log - (ft_k * w_k[:,:,None,None]).sum(1, keepdim=True)
    bar_log = d_log - torch.einsum("bk..., bk -> b...", ft_k, w_k)[:,None,...]
    
    # Symmetric Sinkhorn updates:
    # From the measures to the barycenter:
    ft_k = softmin(eps, C_xy, ak_log + g_k / eps)  # (B,K,n,n)
    # From the barycenter to the measures:
    gt_k = softmin(eps, C_yx, bar_log + f_k / eps)  # (B,K,n,n)
    f_k = (f_k + ft_k) / 2
    g_k = (g_k + gt_k) / 2

    # Sinkhorn "pseudo-step" - from the measures to the barycenter:
    ft_k = softmin(eps, C_xy, ak_log + g_k / eps) / eps
    # Update the barycenter - bar_log is (B,1,...):
    bar_log = d_log - torch.einsum("bk..., bk -> b...", ft_k, w_k)[:,None,...]
    
    # Update the de-biasing measure:
    # (B,1,...) = (B,1,...) + (B,1,...) + (B,1,...)
    d_log = 0.5 * (d_log + bar_log + softmin(eps, C_xx, d_log) / eps)

    return f_k, g_k, d_log, bar_log

    
def sinkhorn_barycenter_loop(
    *,
    softmin: SoftMin,
    a_ks: List[Tensor],
    w_k: Tensor,
    C_xxs: List[CostMatrix],
    C_yys: List[CostMatrix],
    C_xys: List[CostMatrix],
    C_yxs: List[CostMatrix],
    eps_list: List[float],
    jumps: List[int] = [],
    extrapolate: Optional[Extrapolator] = None,
    backward_iterations: int = 5,
):

    with torch.set_grad_enabled(backward_iterations == 0 and torch.is_grad_enabled()):
        
        # The code below relies on log-sum-exps, 
        # so we handle the logarithms of the densities:
        log_a_ks = list(map(stable_log, a_ks))
        
        # Initialize the blur scale at 1, i.e. the full image length:
        sigma = 1  # sigma = blur scale
        eps = sigma**p  # eps = temperature

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
                eps = sigma**p

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
    
    

def ImagesBarycenter(
    measures, weights, blur=0, p=2, scaling_N=10, backward_iterations=5
):

    a_k = measures  # Densities, (B,K,...)
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
        eps = sigma**p  # eps = temperature

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
                eps = sigma**p

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
