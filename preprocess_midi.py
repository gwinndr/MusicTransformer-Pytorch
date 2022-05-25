import argparse
import os
import pickle
import json
import random

import third_party.midi_processor.processor as midi_processor

JSON_FILE = "maestro-v2.0.0.json"

# prep_midi
def prep_maestro_midi(maestro_root, output_dir):
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

        prepped = midi_processor.encode_midi(mid)

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

def prep_custom_midi(custom_midi_root, output_dir, valid_p = 0.1, test_p = 0.2):
    """
    ----------
    Author: Corentin Nelias
    ----------
    Pre-processes custom midi files that are not part of the maestro dataset, putting processed midi data (train, eval, test) into the
    given output folder. 
    ----------
    """
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    
    print("Found", len(os.listdir(custom_midi_root)), "pieces")
    print("Preprocessing custom data...")
    total_count = 0
    train_count = 0
    val_count   = 0
    test_count  = 0
    
    for piece in os.listdir(custom_midi_root):
        #deciding whether the data should be part of train, valid or test dataset
        is_train = True if random.random() > valid_p else False
        if not is_train:
            is_valid = True if random.random() > test_p else False
        if is_train:
            split_type  = "train"
        elif is_valid:
            split_type = "validation"
        else:
            split_type = "test"
            
        mid         = os.path.join(custom_midi_root, piece)
        f_name      = piece.split(".")[0] + ".pickle"

        if(split_type == "train"):
            o_file = os.path.join(train_dir, f_name)
            train_count += 1
        elif(split_type == "validation"):
            o_file = os.path.join(val_dir, f_name)
            val_count += 1
        elif(split_type == "test"):
            o_file = os.path.join(test_dir, f_name)
            test_count += 1
        
        prepped = midi_processor.encode_midi(mid)

        o_stream = open(o_file, "wb")
        pickle.dump(prepped, o_stream)
        o_stream.close()

        total_count += 1
        if(total_count % 50 == 0):
            print(total_count, "/", len(os.listdir(custom_midi_root)))

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

    parser.add_argument("root", type=str, help="Root folder for the Maestro dataset or for custom data.")
    parser.add_argument("-output_dir", type=str, default="./dataset/e_piano", help="Output folder to put the preprocessed midi into.")
    parser.add_argument("--custom_dataset", action="store_true", help="Whether or not the specified root folder contains custom data.")

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
    root    = args.root
    output_dir      = args.output_dir

    print("Preprocessing midi files and saving to", output_dir)
    if args.custom_dataset:
        prep_custom_midi(root, output_dir)
    else:
        prep_maestro_midi(root, output_dir)
    print("Done!")
    print("")

if __name__ == "__main__":
    main()
