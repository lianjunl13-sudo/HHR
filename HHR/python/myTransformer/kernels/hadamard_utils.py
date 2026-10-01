import torch


def hadamard_transform(x, normalize=True):
    """Fast Walsh-Hadamard Transform along last dimension.
    If normalize=True, divides by sqrt(n) to preserve norm.
    """
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard dimension must be a power of two, got {n}")
    h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], -1, 2, h)
        x1, x2 = x.chunk(2, dim=-2)
        x1, x2 = x1.squeeze(-2), x2.squeeze(-2)
        x = torch.cat([x1 + x2, x1 - x2], dim=-1)
        h *= 2
        x = x.reshape(*x.shape[:-2], -1)
    if normalize:
        x = x / (n ** 0.5)
    return x.contiguous()
