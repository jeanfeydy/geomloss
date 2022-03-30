"""Common utility functions."""


def scal(a, f, batch=False):
    if batch:
        B = a.shape[0]
        return (a.reshape(B, -1) * f.reshape(B, -1)).sum(1)
    else:
        return torch.dot(a.reshape(-1), f.reshape(-1))
