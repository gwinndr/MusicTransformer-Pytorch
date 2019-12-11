import os
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utilities.constants import *
from utilities.tensors import create_tensor, create_full_tensor

SEQUENCE_START = -1

# EPianoDataset
class EPianoDataset(Dataset):
    def __init__(self, root, max_seq=2048, random_seq=True):
        self.root       = root
        self.max_seq    = max_seq
        self.random_seq = random_seq

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

        x   = create_full_tensor((self.max_seq, ), TOKEN_END, TORCH_LABEL_TYPE, device=TORCH_CPU)
        tgt = create_full_tensor((self.max_seq, ), TOKEN_END, TORCH_LABEL_TYPE, device=TORCH_CPU)

        # Model expects TOKEN_START at the first position
        # tgt will ideally have TOKEN_END at the end
        x[0]      = TOKEN_START

        ideal_len   = self.max_seq - 1
        raw_len     = len(raw_mid)

        if(raw_len <= ideal_len):
            x[1:raw_len+1] = raw_mid
            tgt[:raw_len]  = raw_mid
        else:
            # Randomly selecting a range
            if(self.random_seq):
                end_range = raw_len - self.max_seq
                x_start = random.randint(SEQUENCE_START, end_range)

            # Always taking from the start to as far as we can
            else:
                x_start = SEQUENCE_START

            x_end = x_start + self.max_seq

            tgt_start = x_start + 1
            tgt_end = x_end + 1

            # Includes TOKEN_START
            if(x_start == SEQUENCE_START):
                x[1:] = raw_mid[:x_end]
            else:
                x = raw_mid[x_start:x_end]

            # Includes TOKEN_END
            if(tgt_end == raw_len + 1):
                tgt[:ideal_len] = raw_mid[tgt_start:tgt_end-1]
            else:
                tgt = raw_mid[tgt_start:tgt_end]

        # print("x:",x)
        # print("tgt:",tgt)

        return x, tgt

# create_epiano_datasets
def create_epiano_datasets(dataset_root, max_seq):
    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
    test_root = os.path.join(dataset_root, "test")

    train_dataset = EPianoDataset(train_root, max_seq)
    val_dataset = EPianoDataset(val_root, max_seq)
    test_dataset = EPianoDataset(test_root, max_seq)

    return train_dataset, val_dataset, test_dataset

def compute_epiano_accuracy(x, y):
    softmax = nn.Softmax(dim=-1)
    y_out = torch.argmax(softmax(y), dim=-1)

    num_right = (y_out == x)
    num_right = torch.sum(num_right, dim=-1).type(TORCH_FLOAT)

    acc = num_right / x.shape[1]
    acc = torch.sum(acc) / x.shape[0]

    return acc
