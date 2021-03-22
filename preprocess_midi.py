# Requirements for this updated version/TMIDI version

# git clone https://github.com/asigalov61/tegridy-tools
# Then import TMIDI module from ./tegridy-tools/tegridy-tools dir

# pip install pretty_midi
 

import argparse
import os
import pickle
import json

import TMIDI

import pickle
import sys

from abc import ABC, abstractmethod

import pretty_midi as pyd
from pretty_midi import Note

from pprint import pprint

JSON_FILE = "maestro-v2.0.0.json"

'''
This is the data processing script

Courtesy of X-Labs
https://github.com/music-x-lab/POP909-Dataset

PLEASE NOTE THAT THE LICENSE FOR THIS CODE IS MIT


============
This script will allow you to quickly process the MIDI Files into the Google Magenta's music representation 
    as like [Music Transformer](https://magenta.tensorflow.org/music-transformer) 
            [Performance RNN](https://magenta.tensorflow.org/performance-rnn).

'''

total = 0
def process_midi(path):
    global total
    data = pyd.PrettyMIDI(path)
    main_notes = []
    acc_notes = []
    for ins in data.instruments:
        acc_notes.extend(ins.notes)
    for i in range(len(main_notes)):
        main_notes[i].start = round(main_notes[i].start,2)
        main_notes[i].end = round(main_notes[i].end,2)
    for i in range(len(acc_notes)):
        acc_notes[i].start = round(acc_notes[i].start,2)
        acc_notes[i].end = round(acc_notes[i].end,2)
    main_notes.sort(key = lambda x:x.start)
    acc_notes.sort(key = lambda x:x.start)
    mpr = TMIDI.Tegridy_RPR_MidiEventProcessor()
    repr_seq = mpr.encode(acc_notes)
    total += len(repr_seq)
    print('Converted file:', path)
    print('Total INTs count:', len(repr_seq))
    print('=' * 70)
    return repr_seq

def process_all_midis(midi_root, save_dir):
    save_py = []
    midi_paths = [d for d in os.listdir(midi_root)]
    i = 0
    out_fmt = '{}-{}.data'
    for path in midi_paths:
        pprint(path)
        filename = midi_root + path
        try:
            data = process_midi(filename)
        except KeyboardInterrupt:
            print(' Abort')
            return
        except EOFError:
            print('EOF Error')
            return
        save_py.append(data)
    # pprint(save_py, compact=True)    
    save_py = np.array(save_py, dtype='object')
    print('=' * 70)
    print('Total number of converted MIDIs:', save_py.size)
    print('Total INTs count:', total)
    np.save(save_dir + 'notes_representations.npy', save_py)

# prep_midi
def prep_midi(maestro_root, output_dir):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Pre-processes the maestro dataset, putting processed midi data (train, eval, test) into the
    given output folder
    ----------
    """

    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    maestro_json_file = os.path.join(maestro_root, JSON_FILE)
    if(not os.path.isfile(maestro_json_file)):
        print("ERROR: Could not find file:", maestro_json_file)
        return False

    maestro_json = json.load(open(maestro_json_file, "r"))
    print("Found", len(maestro_json), "pieces")
    print("Preprocessing...")

    total_count = 0
    train_count = 0
    val_count   = 0
    test_count  = 0

    for piece in maestro_json:
        mid         = os.path.join(maestro_root, piece["midi_filename"])
        split_type  = piece["split"]
        f_name      = mid.split("/")[-1] + ".pickle"

        if(split_type == "train"):
            o_file = os.path.join(train_dir, f_name)
            train_count += 1
        elif(split_type == "validation"):
            o_file = os.path.join(val_dir, f_name)
            val_count += 1
        elif(split_type == "test"):
            o_file = os.path.join(test_dir, f_name)
            test_count += 1
        else:
            print("ERROR: Unrecognized split type:", split_type)
            return False

        prepped = process_midi(mid)

        o_stream = open(o_file, "wb")
        pickle.dump(prepped, o_stream)
        o_stream.close()

        total_count += 1
        if(total_count % 50 == 0):
            print(total_count, "/", len(maestro_json))

    print("Num Train:", train_count)
    print("Num Val:", val_count)
    print("Num Test:", test_count)
    return True



# parse_args
def parse_args():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Parses arguments for preprocess_midi using argparse
    ----------
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("maestro_root", type=str, help="Root folder for the Maestro dataset")
    parser.add_argument("-output_dir", type=str, default="./dataset/e_piano", help="Output folder to put the preprocessed midi into")

    return parser.parse_args()

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Preprocesses maestro and saved midi to specified output folder.
    ----------
    """

    args            = parse_args()
    maestro_root    = args.maestro_root
    output_dir      = args.output_dir

    print("Preprocessing midi files and saving to", output_dir)
    prep_midi(maestro_root, output_dir)
    print("Done!")
    print("")

if __name__ == "__main__":
    main()