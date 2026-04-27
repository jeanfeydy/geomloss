try:  # Import the keops library, www.kernel-operations.io
    from pykeops.torch.cluster import grid_cluster, cluster_ranges_centroids
    from pykeops.torch.cluster import sort_clusters, from_matrix, swap_axes
    from pykeops.torch import LazyTensor as torch_LazyTensor
    from pykeops.numpy import LazyTensor as numpy_LazyTensor

    keops_available = True
except ImportError:
    keops_available = False
    torch_LazyTensor = None
    numpy_LazyTensor = None


def sum(K_ij, axis=None, keepdims=False):
    assert axis is not None
    assert not keepdims
    return K_ij.sum(dim=axis)


def logsumexp(K_ij, axis=None, keepdims=False):
    assert axis is not None
    assert not keepdims
    return K_ij.logsumexp(dim=axis)


def amin(K_ij, axis=None, keepdims=False):
    assert axis is not None
    assert not keepdims
    return K_ij.min(dim=axis)
