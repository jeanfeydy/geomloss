from .ot_result import OTResult
from .check_input_output import cast_input



# OT on empirical distributions
def solve_samples(
    xa,  # (N, D) or (B, N, D)
    xb,  # (M, D) or (B, M, D)
    a=None,  # (N,) or (B, N)
    b=None,  # (M,) or (B, M)
    # To support heterogeneous batches (which are very common in shape analysis),
    # we let users specify "batch vectors" following PyTorch_Geometric's convention:
    # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#mini-batches
    a_batch=None,  # (N,) vector of increasing integer values in [0,B-1]
    b_batch=None,  # (M,) vector of increasing integer values in [0,B-1]
    cost="sqeuclidean",
    # We also support simple functions such as "lambda C(x_i,y_j) = ((x_i - y_j) ** 2).sum(-1) / 2".
    # Depending on the context, these will be run on NumPy arrays, Torch tensors or KeOps LazyTensors
    # of shape (B, N, 1, D) and (B, M, 1, D) and return a (B, N, M) "array".
    # (B may be equal to 1 but *not* collapsed if no batch dimension was provided)
    debias=False,
    # Redundant parameters, that make sense for geometric problems:
    p=None,  # Specifies cost(x,y) = (1/p) * |x-y|^p
    blur=None,  # Specifies "epsilon" = blur^p
    reach=None,  # Specifies "rho" = reach^p
    # + same other params as above
):
    args, output_shapes = cast_input(
        xa=(xa, "B,N,D"),
        xb=(xb, "B,M,D"),
        a=(a, "B,N"),
        b=(b, "B,M"),
        a_batch=(a_batch, "N"),
        b_batch=(b_batch, "M"),
    )
    
    return SinkhornSamplesOTResult(potentials)





class SinkhornSamplesOTResult(OTResult):
    pass




# Convention:
# - D is the number of coordinates per sample (= point)
def barycenter_samples(
    xa,  # (N, D) or (K, N, D) or (B, K, N, D)
    a=None,  # (N,) or (K, N) or (B, K, N)
    weights=None,  # (K,) or (B, K)
    # + all the standard parameters for ot.solve_samples
):
    # masses will be a (M,) or (B, M) array of weights
    # samples will be a (M, D) or (B, M, D) array of coordinates
    return OTResult(potentials=potentials, masses=masses, samples=samples)
