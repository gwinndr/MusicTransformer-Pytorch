import torch

from .constants import TORCH_DEVICE

# Assortment of tensor creations functions that map to the main device

# as_tensor
def as_tensor(data, dtype, device=TORCH_DEVICE):
    """
    ----------
    Author: Damon Gwinn
    ----------
    """

    return torch.as_tensor(data, dtype=dtype, device=device)

# create_tensor
def create_tensor(data, dtype, device=TORCH_DEVICE):
    """
    ----------
    Author: Damon Gwinn
    ----------
    """

    return torch.tensor(data, dtype=dtype, device=device)

# create_full_tensor
def create_full_tensor(shape, value, dtype, device=TORCH_DEVICE):
    """
    ----------
    Author: Damon Gwinn
    ----------
    """

    return torch.full(shape, value, dtype=dtype, device=device)

# create_random_tensor
def create_random_tensor(shape, dtype, device=TORCH_DEVICE):
    """
    ----------
    Author: Damon Gwinn
    ----------
    """

    return torch.rand(shape, dtype=dtype, device=device)

# create_ones_tensor
def create_ones_tensor(shape, dtype, device=TORCH_DEVICE):
    """
    ----------
    Author: Damon Gwinn
    ----------
    """

    return torch.ones(shape, dtype=dtype, device=device)

# create_zeros_tensor
def create_zeros_tensor(shape, dtype, device=TORCH_DEVICE):
    """
    ----------
    Author: Damon Gwinn
    ----------
    """

    return torch.zeros(shape, dtype=dtype, device=device)
