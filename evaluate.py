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

    # We will evaluate the test_dataset and eval_dataset
    _, eval_dataset, test_dataset = create_epiano_datasets(args.dataset_dir, args.max_sequence)

    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    model.load_state_dict(torch.load(args.model_weights))

    if(args.ce_smoothing is None):
        loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
    else:
        loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, VOCAB_SIZE, ignore_index=TOKEN_PAD)

    print("Evaluating:")
    model.eval()

    avg_eval_loss, avg_eval_acc = eval_model(model, eval_loader, loss_func)
    avg_test_loss, avg_test_acc = eval_model(model, test_loader, loss_func)

    print("Average accuracy and loss on the evaluation set and the test set")
    print("Eval represents a 'lab' environment while Test represents the model 'in the wild'")
    print("")
    print("Avg eval loss:", avg_eval_loss)
    print("Avg eval acc:", avg_eval_acc)
    print("Avg test loss:", avg_test_loss)
    print("Avg test acc:", avg_test_acc)
    print(SEPERATOR)
    print("")


if __name__ == "__main__":
    main()
