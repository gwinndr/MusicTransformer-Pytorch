import torch
import torch.nn as nn
import os
import random

from third_party.midi_processor.processor import decode_midi, encode_midi

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

    _, _, dataset = create_epiano_datasets(args.midi_root, args.max_sequence, random_seq=False)

    model = MusicTransformer(args).to(TORCH_DEVICE)
    model.load_state_dict(torch.load(args.model_weights))

    idx     = random.randrange(len(dataset))
    primer, _  = dataset[idx]
    primer = primer.to(TORCH_DEVICE)

    # mid = encode_midi("C:/Users/CO-Cap0010/Desktop/Fire/Base/rand.mid")
    # print(mid)
    # primer = torch.tensor(, dtype=TORCH_LABEL_TYPE, device=TORCH_DEVICE)

    decode_midi(primer[:args.num_prime].cpu().numpy(), file_path="./primer.mid")

    model.eval()
    with torch.set_grad_enabled(False):
        if(args.beam > 0):
            print("BEAM:", args.beam)
            beam_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=args.beam)
            decode_midi(beam_seq[0].cpu().numpy(), file_path="./beam.mid")

        print("RAND DIST")
        rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0)
        decode_midi(rand_seq[0].cpu().numpy(), file_path="./rand.mid")




if __name__ == "__main__":
    main()
