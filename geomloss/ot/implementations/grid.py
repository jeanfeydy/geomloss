# OT on grids
def solve_grid(
    a=None,  # (B, Nx) or (B, Nx, Ny) or (B, Nx, Ny, Nz)  (D = 1, 2, 3)
    b=None,  # (B, Nx) or (B, Nx, Ny) or (B, Nx, Ny, Nz)
    cost="sqeuclidean",  # We also support functions
    # and D-uples of functions that correspond to the separable
    # cost function on each axis. These functions should take as input
    # (B, N, 1) and (B, 1, N) arrays/LazyTensors to return
    # a (B, N, N) array/LazyTensor. The default corresponds to
    # "lambda (x_i, y_j) : ((x_i - y_j) ** 2).sum(-1) / 2".
    axes=None,  # pair of [vmin, vmax) bounds or
    # D-uple of [vmin, vmax) pairs for each axis.
    # Users may also specify explicitly the coordinates
    # along each dimension using a D-uple
    # of (Nx), (Ny,), (Nz,) arrays
    # or even (B, Nx), (B, Ny), (B, Nz).
    # The default None corresponds to [0, 1)^D,
    # with coordinates that are equal to
    # (.5/Nx, 1.5/Nx, ..., (Nx-.5)/Nx),
    # (.5/Ny, 1.5/Ny, ..., (Ny-.5)/Ny),
    # (.5/Nz, 1.5/Nz, ..., (Nz-.5)/Nz).
    periodic=False,  # We also support D-uples of booleans along each axis.
    # Redundant parameters, that make sense for geometric problems:
    p=None,  # Specifies cost(x,y) = (1/p) * |x-y|^p
    blur=None,  # Specifies "epsilon" = blur^p
    reach=None,  # Specifies "rho" = reach^p
    # + same other params as above
):
    return OTResult(potentials)


def barycenter_grid(
    a=None,  # (B, K, Nx) or (B, K, Nx, Ny) or (B, K, Nx, Ny, Nz)  (D = 1, 2, 3)
    # + all the standard parameters for ot.solve_images
):
    # masses will be a (B, Nx) or (B, Nx, Ny) or (B, Nx, Ny, Nz) array of weights
    return OTResult(potentials=potentials, masses=masses)
