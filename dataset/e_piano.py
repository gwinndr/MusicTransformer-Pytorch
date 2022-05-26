import os
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utilities.constants import *
from utilities.device import cpu_device

SEQUENCE_START = 0

# EPianoDataset
class EPianoDataset(Dataset):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Pytorch Dataset for the Maestro e-piano dataset (https://magenta.tensorflow.org/datasets/maestro).
    Recommended to use with Dataloader (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

    Uses all files found in the given root directory of pre-processed (preprocess_midi.py)
    Maestro midi files.
    ----------
    """

    def __init__(self, root, max_seq=2048, random_seq=True):
        self.root       = root
        self.max_seq    = max_seq
        self.random_seq = random_seq

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        self.data_files = [f for f in fs if os.path.isfile(f)]

    # __len__
    def __len__(self):
        """
        ----------
        Author: Damon Gwinn
        ----------
        How many data files exist in the given directory
        ----------
        """

        return len(self.data_files)

    # __getitem__
    def __getitem__(self, idx):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Gets the indexed midi batch. Gets random sequence or from start depending on random_seq.

        Returns the input and the target.
        ----------
        """

        # All data on cpu to allow for the Dataloader to multithread
        i_stream    = open(self.data_files[idx], "rb")
        # return pickle.load(i_stream), None
        raw_mid     = torch.tensor(pickle.load(i_stream), dtype=TORCH_LABEL_TYPE, device=cpu_device())
        i_stream.close()

        x, tgt = process_midi(raw_mid, self.max_seq, self.random_seq)

        return x, tgt

# process_midi
def process_midi(raw_mid, max_seq, random_seq):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Takes in pre-processed raw midi and returns the input and target. Can use a random sequence or
    go from the start based on random_seq.
    ----------
    """

    x   = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=cpu_device())
    tgt = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=cpu_device())

    raw_len     = len(raw_mid)
    full_seq    = max_seq + 1 # Performing seq2seq

    if(raw_len == 0):
        return x, tgt

    if(raw_len < full_seq):
        x[:raw_len]         = raw_mid
        tgt[:raw_len-1]     = raw_mid[1:]
        tgt[raw_len-1]      = TOKEN_END
    else:
        # Randomly selecting a range
        if(random_seq):
            end_range = raw_len - full_seq
            start = random.randint(SEQUENCE_START, end_range)

        # Always taking from the start to as far as we can
        else:
            start = SEQUENCE_START

        end = start + full_seq

        data = raw_mid[start:end]

        x = data[:max_seq]
        tgt = data[1:full_seq]


    # print("x:",x)
    # print("tgt:",tgt)

    return x, tgt


# create_epiano_datasets
def create_epiano_datasets(dataset_root, max_seq, random_seq=True):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Creates train, evaluation, and test EPianoDataset objects for a pre-processed (preprocess_midi.py)
    root containing train, val, and test folders.
    ----------
    """

    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
    test_root = os.path.join(dataset_root, "test")

    train_dataset = EPianoDataset(train_root, max_seq, random_seq)
    val_dataset = EPianoDataset(val_root, max_seq, random_seq)
    test_dataset = EPianoDataset(test_root, max_seq, random_seq)

    return train_dataset, val_dataset, test_dataset

# compute_epiano_accuracy
def compute_epiano_accuracy(out, tgt):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Computes the average accuracy for the given input and output batches. Accuracy uses softmax
    of the output.
    ----------
    """

    softmax = nn.Softmax(dim=-1)
    out = torch.argmax(softmax(out), dim=-1)

    out = out.flatten()
    tgt = tgt.flatten()

    mask = (tgt != TOKEN_PAD)

    out = out[mask]
    tgt = tgt[mask]

    # Empty
    if(len(tgt) == 0):
        return 1.0

    num_right = (out == tgt)
    num_right = torch.sum(num_right).type(TORCH_FLOAT)

    acc = num_right / len(tgt)

    return acc
