import torch

try:  # Import the keops library, www.kernel-operations.io
    from pykeops.torch import LazyTensor

    keops_available = True
except:
    keops_available = False


def scal(α, f, batch=False):
    if batch:
        B = α.shape[0]
        return (α.view(B, -1) * f.view(B, -1)).sum(1)
    else:
        return torch.dot(α.view(-1), f.view(-1))


def squared_distances(x, y, use_keops=False):

    if use_keops and keops_available:
        if x.dim() == 2:
            x_i = LazyTensor(x[:,None,:])  # (N,1,D)
            y_j = LazyTensor(y[None,:,:])  # (1,M,D)
        elif x.dim() == 3:  # Batch computation
            x_i = LazyTensor(x[:,:,None,:])  # (B,N,1,D)
            y_j = LazyTensor(y[:,None,:,:])  # (B,1,M,D)
        else:
            print("x.shape : ", x.shape)
            raise ValueError("Incorrect number of dimensions")

        return ((x_i - y_j) ** 2).sum(-1)

    else:
        if x.dim() == 2:
            D_xx = (x * x).sum(-1).unsqueeze(1)  # (N,1)
            D_xy = torch.matmul(x, y.permute(1, 0))  # (N,D) @ (D,M) = (N,M)
            D_yy = (y * y).sum(-1).unsqueeze(0)  # (1,M)
        elif x.dim() == 3:  # Batch computation
            D_xx = (x * x).sum(-1).unsqueeze(2)  # (B,N,1)
            D_xy = torch.matmul(x, y.permute(0, 2, 1))  # (B,N,D) @ (B,D,M) = (B,N,M)
            D_yy = (y * y).sum(-1).unsqueeze(1)  # (B,1,M)
        else:
            print("x.shape : ", x.shape)
            raise ValueError("Incorrect number of dimensions")

        return D_xx - 2 * D_xy + D_yy


def distances(x, y, use_keops=False):
    if use_keops:
        return squared_distances(x, y, use_keops=use_keops).sqrt()

    else:
        return torch.sqrt( torch.clamp_min(squared_distances(x,y), 1e-8) )
