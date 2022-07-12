from ..typing import (
    Tensor,
    Tuple,
    Optional,
    SoftMin,
    CostMatrix,
    CostMatrices,
    CostFunction,
)

from ..utils import stable_log


def barycenter_iteration(
    *,
    softmin: SoftMin,
    f_k: Tensor, 
    g_k: Tensor, 
    log_d: Tensor, 
    eps: float, 
    C: CostMatrices, 
    log_a_k: Tensor, 
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
        
        log_d ((B,1,...) Tensor): de-biasing weight, supported by the barycenters
            on points x[i].
        
        eps (float > 0): the positive regularization parameter (= temperature)
            for the Sinkhorn algorithm.
        
        C (CostMatrices): a NamedTuple of objects that encode the (B,K) 
            cost matrices C.xx[i,j] = C(x[i], x[j]), C.xy[i,j] = C(x[i], y[j])
            and C.yx[i,j] = C(y[i], x[j]) in a way that is compatible with softmin.
            
        log_a_k ((B,K,...) Tensor): logarithms of the measure weights ("masses")
            for the input measures.
          
        w_k ((B,K) Tensor): desired weights for the barycentric interpolation. 
    """
    
    # Sinkhorn "pseudo-step" - from the measures to the barycenter:
    ft_k = softmin(eps, C.xy, log_a_k + g_k / eps) / eps  # (B,K,...)
    # Update the barycenter:
    # (B,1,...) = (B,1,...) - (B,K,...) @ (B,K,1,..,1)
    # With e.g. 2 trailing dimensions, the einsum below is equivalent to:
    # log_bar = log_d - (ft_k * w_k[:,:,None,None]).sum(1, keepdim=True)
    log_bar = log_d - torch.einsum("bk..., bk -> b...", ft_k, w_k)[:,None,...]
    
    # Symmetric Sinkhorn updates:
    # From the measures to the barycenter:
    ft_k = softmin(eps, C.xy, log_a_k + g_k / eps)  # (B,K,n,n)
    # From the barycenter to the measures:
    gt_k = softmin(eps, C.yx, log_bar + f_k / eps)  # (B,K,n,n)
    f_k = (f_k + ft_k) / 2
    g_k = (g_k + gt_k) / 2

    # Sinkhorn "pseudo-step" - from the measures to the barycenter:
    ft_k = softmin(eps, C.xy, log_a_k + g_k / eps) / eps
    # Update the barycenter - log_bar is (B,1,...):
    log_bar = log_d - torch.einsum("bk..., bk -> b...", ft_k, w_k)[:,None,...]
    
    # Update the de-biasing measure:
    # (B,1,...) = (B,1,...) + (B,1,...) + (B,1,...)
    log_d = 0.5 * (log_d + log_bar + softmin(eps, C.xx, log_d) / eps)

    return f_k, g_k, log_d, log_bar

    
def sinkhorn_barycenter_loop(
    *,
    softmin: SoftMin,
    log_a_k_list: List[Tensor],
    w_k_list: Tensor,
    C_list: List[CostMatrices],
    descent: DescentParameters,
    extrapolate: Optional[Extrapolator] = None,
    backward_iterations: int = 5,
):

    with torch.set_grad_enabled(backward_iterations == 0 and torch.is_grad_enabled()):
        # Setup the input measures at the coarsest level:
        sigma = descent.blur_list[0]  # sigma = blur scale
        eps = descent.eps_list[0]  # eps = temperature
        log_a_k = log_a_k_list[0]  # (B,K,...) log-weights for the input measure
        C = C_list[0]  # implicit (B,K,N,M), (B,K,N,N)... cost matrices
        
        # Initialize the dual variables:
        f_k = softmin(eps, C.xy, log_a_k)  # (B,K,...), supported by the barycenter points x
        # TODO: the line below is not great...
        g_k = softmin(eps, C.yx, log_a_k)  # (B,K,...), supported by the input points y
        
        # Logarithm of the debiasing term:
        log_d = torch.ones_like(log_a_k).sum(dim=1, keepdim=True)  # (B,1,...)
        # Normalize each of these:
        log_d = log_d - log_d.logsumexp(log_d.shape[2:], keepdim=True)

        # Multiscale descent, with eps-scaling ----------------------------------
        scale = 0  # integer counter
        # See Fig. 3.25-26 in Jean Feydy's PhD thesis for intuitions.
        for i, eps in enumerate(descent.eps_list):
            f_k, g_k, log_d, log_bar = barycenter_iteration(
                softmin=softmin,
                f_k=f_k,
                g_k=g_k,
                log_d=log_d,
                eps=eps,
                C=C,
                log_a_k=log_a_k,
                w_k=w_k,
            )
            
            if i in jumps:  # Re-fine the maps, if needed
                
                C_fine = C_list[scale + 1]
                
                # N.B.: this code does not currently support unbalanced OT
                dampen = None
                
                # The extrapolation formulas are coherent with the
                # softmin(...) updates of the barycenter iteration.
                f_k = extrapolate(
                    self=f_k, 
                    other=g_k, 
                    log_weights=log_a_k,
                    C=C.xy, 
                    C_fine=C_fine.xy,
                    eps=eps, 
                    dampen=dampen,
                    )
                    
                g_k = extrapolate(
                    self=g_k,
                    other=f_k,
                    log_weights=log_bar,
                    C=C.yx,
                    C_fine=C_fine.yx,
                    eps=eps,
                    dampen=dampen,
                    )
                    
                # N.B.: This is not tested at all outside of grids.
                log_d = extrapolate(
                    self=log_d,
                    other=0 * log_d,
                    log_weights=log_d,
                    C=C.xx,
                    C_fine=C_fine.xx,
                    eps=eps,
                    dampen=dampen,
                    )

    if (measures.requires_grad or weights.requires_grad) and backward_iterations > 0:
        for _ in range(backward_iterations):
            f_k, g_k, d_log, bar_log = barycenter_iteration(
                f_k, g_k, d_log, eps, p, ak_log, w_k
            )

    return log_bar.exp()
    
    

        # The code below relies on log-sum-exps, 
        # so we handle the logarithms of the densities:
        log_a_ks = list(map(stable_log, a_ks))
        
        
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
