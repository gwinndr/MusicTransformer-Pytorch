import argparse

from .constants import SEPERATOR

# parse_args
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-input_dir", type=str, default="./dataset/e_piano", help="Folder of preprocessed and pickled midi files")
    parser.add_argument("-output_dir", type=str, default="./saved_models", help="Folder to save model weights. Saves one every epoch")

    parser.add_argument("-lr", type=float, default=None, help="Constant learn rate. Leave as None for a custom scheduler.")
    parser.add_argument("-batch_size", type=int, default=2, help="Batch size to use")
    parser.add_argument("-epochs", type=int, default=100, help="Number of epochs to use")

    parser.add_argument("-max_sequence", type=int, default=2048, help="Maximum midi sequence to consider")
    parser.add_argument("-n_layers", type=int, default=6, help="Number of decoder layers to use")
    parser.add_argument("-num_heads", type=int, default=8, help="Number of heads to use for multi-head attention")
    parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")

    parser.add_argument("-dim_feedforward", type=int, default=2048, help="Dimension of the feedforward layer")
    # parser.add_argument("-feedforward_activation", type=str, default="relu", help="Activation of the feedforward layer")

    parser.add_argument("-dropout", type=float, default=0.1, help="Dropout rate")

    return parser.parse_args()

# print_args
def print_args(args):
    print(SEPERATOR)
    print("input_dir:", args.input_dir)
    print("output_dir:", args.output_dir)
    print("")
    print("lr:", args.lr)
    print("batch_size:", args.batch_size)
    print("epochs:", args.epochs)
    print("")
    print("max_sequence:", args.max_sequence)
    print("n_layers:", args.n_layers)
    print("num_heads:", args.num_heads)
    print("d_model:", args.d_model)
    print("")
    print("dim_feedforward:", args.dim_feedforward)
    # print("feedforward_activation:", args.feedforward_activation)
    print("dropout:", args.dropout)
    print(SEPERATOR)
    print("")
