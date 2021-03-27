


def sinkhorn_barycenter(measures, weights, blur=0, p=2, scaling_N = 10):

    a_k = measures  # (B, K, N, N)
    w_k = weights   # (K,)
    if blur == 0:
        blur = 1 / measures.shape[-1]

    bar = (a_k * w_k[None,:,None,None]).sum(1)

    with torch.no_grad() :
        # Pre-compute a multiscale decomposition (=QuadTree)
        # of the input measures, stored as logarithms
        ak_s = pyramid(a_k)[1:]
        ak_log_s = list(map(log_dens, ak_s))

        # Initialize the dual variables
        sigma = 1 ; eps = sigma ** p  # sigma = blurring scale, eps = temperature
        f_k, g_k = logconv(ak_log_s[0], eps, p), logconv(ak_log_s[0], eps, p)
        
        d_log = torch.ones_like(ak_log_s[0]).sum(dim=1, keepdim=True)
        d_log -= d_log.logsumexp([2, 3])
        
        # Multiscale descent, with eps-scaling:
        for n, ak_log in enumerate( ak_log_s ):
            for _ in range(scaling_N) : # Number of steps per scale
                eps = sigma ** p

                # Update the barycenter:
                ft_k = logconv(ak_log  + g_k / eps, eps, p) / eps
                bar_log = d_log - (ft_k * w_k[None,:,None,None]).sum(1)

                # symmetric Sinkhorn updates:
                ft_k = logconv(ak_log  + g_k / eps, eps, p)
                gt_k = logconv(bar_log + f_k / eps, eps, p)
                f_k += ft_k ; f_k *= .5 ; g_k += gt_k ; g_k *= .5

                # Update the barycenter:
                ft_k = logconv(ak_log  + g_k / eps, eps, p) / eps
                bar_log = d_log - (ft_k * w_k[None,:,None,None]).sum(1)

                # Update the de-biasing measure:
                d_log = .5 * (d_log + bar_log + logconv(d_log, eps, p) / eps)

                # Decrease the kernel radius, making sure that
                # σ is divided by two at every scale until we reach
                # the target value, "blur":
                sigma = max(sigma * 2**(-1/scaling_N), blur)

            if n+1 < len(ak_s):  # Re-fine the maps, if needed
                f_k = upsample(f_k) ; g_k = upsample(g_k)
                d_log = upsample(d_log)

    return bar_log.exp()


    
