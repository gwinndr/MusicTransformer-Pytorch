import torch
import torch.nn as nn
import os
import random
import pretty_midi
import numpy as np
#os.chdir('/tegridy-tools/tegridy-tools')
import TMIDI
#os.chdir('/content/MusicTransformer-Pytorch')
import pretty_midi as pyd

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy, process_midi
from torch.utils.data import DataLoader
from torch.optim import Adam

from utilities.constants import *
from utilities.device import get_device, use_cuda

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Generates music from a model specified by command line arguments
    ----------
    """

    args = parse_generate_args()
    print_generate_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.output_dir, exist_ok=True)

    # Grabbing dataset if needed
    _, _, dataset = create_epiano_datasets(args.midi_root, args.num_prime, random_seq=False)

    # Can be None, an integer index to dataset, or a file path
    if(args.primer_file is None):
        f = str(random.randrange(len(dataset)))
    else:
        f = args.primer_file

    if(f.isdigit()):
        idx = int(f)
        primer, _  = dataset[idx]
        primer = primer.to(get_device())

        print("Using primer index:", idx, "(", dataset.data_files[idx], ")")

    else:
        raw_mid = encode_midi(f)
        if(len(raw_mid) == 0):
            print("Error: No midi messages in primer file:", f)
            return

        primer, _  = process_midi(raw_mid, args.num_prime, random_seq=False)
        primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())

        print("Using primer file:", f)

    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    model.load_state_dict(torch.load(args.model_weights))

    # Saving primer first
    f_path = os.path.join(args.output_dir, "primer")
    #decode_midi(primer[:args.num_prime].cpu().numpy(), file_path=f_path)
    x = primer[:args.num_prime].cpu().numpy()
    y = x.tolist()
    # Create a PrettyMIDI object
    output = pyd.PrettyMIDI()
    # Create an Instrument instance for a piano instrument
    output_program = pyd.instrument_name_to_program('Acoustic Grand Piano')
    piano = pyd.Instrument(program=output_program)

    # Decode representations into Pretty_MIDI notes
    mpr = TMIDI.Tegridy_RPR_MidiEventProcessor()
    notes = mpr.decode(y)

    # Add notes to the Pretty MIDI object
    piano.notes.extend(notes)

    output.instruments.append(piano)
    # Write out the MIDI data
    output.write(f_path+'.mid')
    print('Saved as', f_path+'.mid')

    # GENERATION
    model.eval()
    with torch.set_grad_enabled(False):
        if(args.beam > 0):
            print("BEAM:", args.beam)
            beam_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=args.beam)

            f_path = os.path.join(args.output_dir, "beam")

            x = beam_seq[0].cpu().numpy()
            y = x.tolist()
            # Create a PrettyMIDI object
            output = pyd.PrettyMIDI()
            # Create an Instrument instance for a piano instrument
            output_program = pyd.instrument_name_to_program('Acoustic Grand Piano')
            piano = pyd.Instrument(program=output_program)

            # Decode representations into Pretty_MIDI notes
            mpr = TMIDI.Tegridy_RPR_MidiEventProcessor()
            notes = mpr.decode(y)

            # Add notes to the Pretty MIDI object
            piano.notes.extend(notes)

            output.instruments.append(piano)
            # Write out the MIDI data
            output.write(f_path+'.mid')
            print('Saved as', f_path+'.mid')

        else:
            print("RAND DIST")
            rand_seq = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0)

            f_path = os.path.join(args.output_dir, "rand")
            #decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)
            #print('Seq =', rand_seq[0].cpu().numpy())
            x = rand_seq[0].cpu().numpy()
            y = x.tolist()
            
            # Create a PrettyMIDI object
            output = pyd.PrettyMIDI()
            # Create an Instrument instance for a piano instrument
            output_program = pyd.instrument_name_to_program('Acoustic Grand Piano')
            piano = pyd.Instrument(program=output_program)

            # Decode representations into Pretty_MIDI notes
            mpr = TMIDI.Tegridy_RPR_MidiEventProcessor()
            notes = mpr.decode(y)

            # Add notes to the Pretty MIDI object
            piano.notes.extend(notes)

            output.instruments.append(piano)
            # Write out the MIDI data
            output.write(f_path+'.mid')
            print('Saved as', f_path+'.mid')

if __name__ == "__main__":
    main()