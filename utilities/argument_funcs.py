import argparse

from .constants import SEPERATOR

# parse_args
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-input_dir", type=str, default="./prepped_midi", help="Folder of preprocessed and pickled midi files")
    parser.add_argument("-output_dir", type=str, default="./saved_models", help="Folder to save model weights. Saves one every epoch")

    parser.add_argument("-lr", type=float, default=None, help="Constant learn rate. Leave as None for a custom scheduler.")
    parser.add_argument("-batch_size", type=float, default=2, help="Batch size to use")
    parser.add_argument("-epochs", type=int, default=100, help="Number of epochs to use")

    parser.add_argument("-dec_layers", type=int, default=6, help="Number of decoder layers to use")
    parser.add_argument("-num_heads", type=int, default=8, help="Number of heads to use for multi-head attention")
    parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")

    return parser.parse_args()

# print_args
def print_args(args_dict):
    print(SEPERATOR)
    print("input_dir:", args_dict["input_dir"])
    print("output_dir:", args_dict["output_dir"])
    print("")
    print("lr:", args_dict["lr"])
    print("batch_size:", args_dict["batch_size"])
    print("epochs:", args_dict["epochs"])
    print("")
    print("dec_layers:", args_dict["dec_layers"])
    print("num_heads:", args_dict["num_heads"])
    print("d_model:", args_dict["d_model"])
    print(SEPERATOR)
    print("")

# args2dict
def args2dict(args):
    args_dict = dict()

    args_dict["input_dir"]      = args.input_dir
    args_dict["output_dir"]     = args.output_dir
    args_dict["lr"]             = args.lr
    args_dict["batch_size"]     = args.batch_size
    args_dict["epochs"]         = args.epochs
    args_dict["dec_layers"]     = args.dec_layers
    args_dict["num_heads"]      = args.num_heads
    args_dict["d_model"]        = args.d_model

    return args_dict
