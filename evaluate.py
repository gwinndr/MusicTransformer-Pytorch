import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy

from model.music_transformer import MusicTransformer

from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.argument_funcs import parse_eval_args, print_eval_args
from utilities.run_model import eval_model

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Evaluates a model specified by command line arguments
    ----------
    """

    args = parse_eval_args()
    print_eval_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    # Test dataset
    _, _, test_dataset = create_epiano_datasets(args.dataset_dir, args.max_sequence)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    model.load_state_dict(torch.load(args.model_weights))

    # No smoothed loss
    loss = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)

    print("Evaluating:")
    model.eval()

    avg_loss, avg_acc = eval_model(model, test_loader, loss)

    print("Avg loss:", avg_loss)
    print("Avg acc:", avg_acc)
    print(SEPERATOR)
    print("")


if __name__ == "__main__":
    main()
