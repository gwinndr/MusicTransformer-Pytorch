import torch
import torch.nn as nn
import os
import random

from third_party.midi_processor.processor import decode_midi

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy
from torch.utils.data import DataLoader
from torch.optim import Adam

from utilities.constants import *
from utilities.tensors import create_random_tensor

# main
def main():
    args = parse_generate_args()
    print_generate_args(args)

    dataset, _, _ = create_epiano_datasets(args.midi_root, args.max_sequence, random_seq=False)

    model = MusicTransformer(args).to(TORCH_DEVICE)
    model.load_state_dict(torch.load(args.model_weights))

    idx     = random.randrange(len(dataset))
    primer, _  = dataset[idx]
    primer = primer.to(TORCH_DEVICE)

    model.eval()
    with torch.set_grad_enabled(False):
        seq = model.generate(primer[:args.num_prime], args.target_seq_length)

    decode_midi(primer[:args.num_prime].cpu().numpy(), file_path="./primer.mid")
    decode_midi(seq[0].cpu().numpy(), file_path="./full.mid")


if __name__ == "__main__":
    main()
