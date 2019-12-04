import argparse
import os
import pickle

import third_party.midi_processor.processor as midi_processor

MIDI_EXT = ".midi"

# prep_midi
def prep_midi(midi_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for mid in midi_files:
        prepped = midi_processor.encode_midi(mid)

        f_name = mid.split("/")[-1] + ".pickle"
        o_stream = open(os.path.join(output_dir, f_name), "wb")

        pickle.dump(prepped, o_stream)
        o_stream.close()

        count += 1
        if(count % 50 == 0):
            print(count, "/", len(midi_files))

# accumulate_midi_files
def accumulate_midi_files(root):
    midi_files = []
    for elem in os.listdir(root):
        full_path = os.path.join(root, elem)

        # If a folder, recursively search the subfolders
        if(os.path.isdir(full_path)):
            midi_files.extend(accumulate_midi_files(full_path))

        # If a file, check the extension
        else:
            _, ext = os.path.splitext(elem)
            if(ext == MIDI_EXT):
                midi_files.append(full_path)

    return midi_files


# parse_args
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("maestro_root", type=str, help="Root folder for the Maestro dataset")
    parser.add_argument("output_dir", type=str, help="Output folder to put the preprocessed midi into")

    return parser.parse_args()

# main
def main():
    args            = parse_args()
    maestro_root    = args.maestro_root
    output_dir      = args.output_dir

    print("Accumulating midi files in", maestro_root)
    midi_files = accumulate_midi_files(maestro_root)
    print("Accumulated", len(midi_files), "midi files")
    print("")

    print("Preprocessing midi files and saving to", output_dir)
    prep_midi(midi_files, output_dir)
    print("Done!")
    print("")

if __name__ == "__main__":
    main()
