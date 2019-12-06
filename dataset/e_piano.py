import os
import numpy as np
import torch
from torch.utils.data import Dataset

from utilities.constants import TORCH_INT

# EPianoDataset
class EPianoDataset(Dataset):
    def __init__(self, root):
        self.root = root

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        self.data_files = [f for f in fs if os.path.isfile(f)]


    # __len__
    def __len__(self):
        return len(self.data_files)

    # __getitem__
    def __getitem__(self, idx):
        return torch.tensor(pickle.load(self.data_files[idx]), dtype=TORCH_INT)

# create_epiano_datasets
def create_epiano_datasets(dataset_root):
    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
    test_root = os.path.join(dataset_root, "test")

    train_dataset = EPianoDataset(train_root)
    val_dataset = EPianoDataset(val_root)
    test_dataset = EPianoDataset(test_root)

    return train_dataset, val_dataset, test_dataset
