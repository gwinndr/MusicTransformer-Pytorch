import os
import pickle
import torch
from torch.utils.data import Dataset

from utilities.constants import *
from utilities.tensors import create_tensor, create_full_tensor

# EPianoDataset
class EPianoDataset(Dataset):
    def __init__(self, root, max_seq=2048):
        self.root       = root
        self.max_seq    = max_seq

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        self.data_files = [f for f in fs if os.path.isfile(f)]

    # __len__
    def __len__(self):
        return len(self.data_files)

    # __getitem__
    def __getitem__(self, idx):
        # All data on cpu to allow for the Dataloader to multithread
        i_stream    = open(self.data_files[idx], "rb")
        raw_mid     = create_tensor(pickle.load(i_stream), TORCH_LABEL_TYPE, device=TORCH_CPU)
        i_stream.close()

        seq = create_full_tensor((self.max_seq, ), TOKEN_END, TORCH_LABEL_TYPE, device=TORCH_CPU)

        # Must always start with TOKEN_START
        seq[0]      = TOKEN_START
        valid_seq   = self.max_seq - 1

        if(len(raw_mid) < valid_seq):
            seq[1:len(raw_mid)+1] = raw_mid
        else:
            # TODO: Would it be better to randomly select a range?
            seq[1:] = raw_mid[:valid_seq]

        return seq

# create_epiano_datasets
def create_epiano_datasets(dataset_root, max_seq):
    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
    test_root = os.path.join(dataset_root, "test")

    train_dataset = EPianoDataset(train_root)
    val_dataset = EPianoDataset(val_root)
    test_dataset = EPianoDataset(test_root)

    return train_dataset, val_dataset, test_dataset
