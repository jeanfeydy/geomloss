

def sinkhorn_divergence(α, β, blur=.05, p=2, scaling_N = 1) :
    
    with torch.no_grad() :
        # Pre-compute a multiscale decomposition (=QuadTree)
        # of the input measures, stored as logarithms
        α_s, β_s = pyramid(α), pyramid(β)
        α_log_s = list(map(log_dens, α_s))
        β_log_s = list(map(log_dens, β_s))

        # Initialize the dual variables
        σ = 1 ; ε = σ**p  # σ = blurring scale, ε = temperature
        a_i, b_j = logconv(α_log_s[0], ε, p), logconv(β_log_s[0], ε, p)
        a_j, b_i = logconv(α_log_s[0], ε, p), logconv(β_log_s[0], ε, p)

        # Multiscale descent, with ε-scaling:
        for n, (α_log, β_log) in enumerate( zip(α_log_s, β_log_s) ):
            for _ in range(scaling_N) : # Number of steps per scale
                ε = σ**p

                # Symmetric updates:
                a_i  = .5 * ( a_i + logconv(α_log + a_i/ε, ε, p) )
                b_j  = .5 * ( b_j + logconv(β_log + b_j/ε, ε, p) )

                # OT_αβ updates:
                at_j = logconv(α_log + b_i/ε, ε, p)
                bt_i = logconv(β_log + a_j/ε, ε, p)
                a_j += at_j ; a_j *= .5 ; b_i += bt_i ; b_i *= .5

                # Decrease the kernel radius, making sure that
                # σ is divided by two at every scale until we reach
                # the target value, "blur":
                σ = max(σ * 2**(-1/scaling_N), blur)

            if n+1 < len(α_s):  # Re-fine the maps, if needed
                a_i = upsample(a_i) ; b_j = upsample(b_j)
                a_j = upsample(a_j) ; b_i = upsample(b_i)

    S_ε = scal( α, b_i - a_i ) + scal( β, a_j - b_j )
    return S_ε
