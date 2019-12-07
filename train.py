from utilities.argument_funcs import parse_args, print_args
from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets
from torch.utils.data import DataLoader

from utilities.constants import *
from utilities.tensors import create_random_tensor

# main
def main():
    args = parse_args()
    print_args(args)

    train_dataset, val_dataset, test_dataset = create_epiano_datasets(args.input_dir, args.max_sequence)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    for batch in train_loader:
        print(batch)

    # model = MusicTransformer(args).to(TORCH_DEVICE)
    # y = model(x)
    # model.eval()
    # y = model.generate()

if __name__ == "__main__":
    main()
