import torch
from torch.utils.data import DataLoader

from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy

from model.music_transformer import MusicTransformer
from model.loss import SmoothCrossEntropyLoss

from utilities.constants import *
from utilities.argument_funcs import parse_eval_args, print_eval_args
from utilities.run_model import eval_model

# main
def main():
    args = parse_eval_args()
    print_eval_args(args)

    _, _, test_dataset = create_epiano_datasets(args.dataset_dir, args.max_sequence)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)
    model = MusicTransformer(args).to(TORCH_DEVICE)
    model.load_state_dict(torch.load(args.model_weights))

    # loss    = SmoothCrossEntropyLoss(LABEL_SMOOTHING_E, VOCAB_SIZE, ignore_index=TOKEN_PAD)
    loss    = torch.nn.CrossEntropyLoss()

    print("Evaluating:")
    model.eval()

    avg_loss, avg_acc = eval_model(model, test_loader, loss)

    print("Avg loss:", avg_loss)
    print("Avg acc:", avg_acc)
    print(SEPERATOR)
    print("")


if __name__ == "__main__":
    main()
