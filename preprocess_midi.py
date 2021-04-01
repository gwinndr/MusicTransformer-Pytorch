import argparse
import os
import pickle
import json

import third_party.midi_processor.processor as midi_processor

# Possible maestro filenames with the version mapping
MAESTRO_JSON_FILENAMES = ["maestro-v3.0.0.json", "maestro-v2.0.0.json"]
MAESTRO_VERSIONS = [3, 2]

# parse_v3
def parse_v3(maestro_json, maestro_root):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Parses V3 maestro json into lists of train, validation, and test files

    Returns 3 lists: train, val, and test in that order
    ----------
    """

    train = []
    val = []
    test = []

    midi_files = list(maestro_json["midi_filename"].values())
    splits = list(maestro_json["split"].values())

    for i, midi_file in enumerate(midi_files):
        split = splits[i]

        mid = os.path.join(maestro_root, midi_file)

        if(split == "train"):
            train.append(mid)
        elif(split == "validation"):
            val.append(mid)
        elif(split == "test"):
            test.append(mid)
        else:
            print("ERROR: Unrecognized split type:", split)

    return train, val, test

# parse_v2
def parse_v2(maestro_json, maestro_root):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Parses V2 maestro json into lists of train, validation, and test files

    Returns 3 lists: train, val, and test in that order
    ----------
    """

    train = []
    val = []
    test = []

    for piece in maestro_json:
        mid = os.path.join(maestro_root, piece["midi_filename"])
        split_type = piece["split"]

        if(split_type == "train"):
            train.append(mid)
        elif(split_type == "validation"):
            val.append(mid)
        elif(split_type == "test"):
            test.append(mid)
        else:
            print("ERROR: Unrecognized split type:", split_type)

    return train, val, test


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

    maestro_json_files = [os.path.join(maestro_root, json) for json in MAESTRO_JSON_FILENAMES]

    # Set maestro json depending on which version is found
    maestro_json_file = None
    maestro_version = -1
    for i, json_candidate in enumerate(maestro_json_files):
        if(os.path.isfile(json_candidate)):
            maestro_json_file = json_candidate
            maestro_version = MAESTRO_VERSIONS[i]
            print("Found file:", maestro_json_file)
            print("Maestro version:", maestro_version)
            break

    # Error if no maestro json found
    if(maestro_json_file is None):
        print("ERROR: Could not find maestro json file!")
        print("Possible maestro json names (in maestro_root):", MAESTRO_JSON_FILENAMES)
        return False

    maestro_json = json.load(open(maestro_json_file, "r"))

    # Parse json into train, val, and test files based on version
    if(maestro_version == 3):
        train, val, test = parse_v3(maestro_json, maestro_root)
    elif(maestro_version == 2):
        train, val, test = parse_v2(maestro_json, maestro_root)
    else:
        print("BUG: No parser for version:", maestro_version)
        return False

    num_train = len(train)
    num_val = len(val)
    num_test = len(test)

    print("Found", num_train, "train pieces")
    print("Found", num_val, "val pieces")
    print("Found", num_test, "test pieces")

    all_pieces = ( ("train", train), ("validation", val), ("test", test) )
    for piece in all_pieces:
        split_type = piece[0]
        mids = piece[1]
        total_count = 0

        print("Prepping %s..." % split_type)
        for mid in mids:
            o_name = mid.split("/")[-1] + ".pickle"

            if(split_type == "train"):
                o_file = os.path.join(train_dir, o_name)
            elif(split_type == "validation"):
                o_file = os.path.join(val_dir, o_name)
            elif(split_type == "test"):
                o_file = os.path.join(test_dir, o_name)
            else:
                print("ERROR: Unrecognized split type:", split_type)

            prepped = midi_processor.encode_midi(mid)

            o_stream = open(o_file, "wb")
            pickle.dump(prepped, o_stream)
            o_stream.close()

            total_count += 1
            if(total_count % 50 == 0):
                print(total_count, "/", len(mids), split_type)

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
    parser.add_argument("-output_dir", type=str, default="./dataset/e_piano",
        help="Output folder to put the preprocessed midi into (recommend you leave at default)")


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
