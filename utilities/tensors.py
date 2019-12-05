import torch

from .constants import TORCH_DEVICE

# Assortment of tensor creations functions that map to the main device

# as_tensor
def as_tensor(data, dtype):
    return torch.as_tensor(data, dtype=dtype, device=TORCH_DEVICE)

# create_tensor
def create_tensor(data, dtype):
    return torch.tensor(data, dtype=dtype, device=TORCH_DEVICE)

# create_random_tensor
def create_random_tensor(shape, dtype):
    return torch.rand(shape, dtype=dtype, device=TORCH_DEVICE)

# create_ones_tensor
def create_ones_tensor(shape, dtype):
    return torch.ones(shape, dtype=dtype, device=TORCH_DEVICE)

# create_zeros_tensor
def create_zeros_tensor(shape, dtype):
    return torch.zeros(shape, dtype=dtype, device=TORCH_DEVICE)
