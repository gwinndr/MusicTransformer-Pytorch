from utilities.argument_funcs import parse_args, print_args
from model.music_transformer import MusicTransformer
# from dataset.e_piano import create_epiano_datasets

from utilities.constants import *
from utilities.tensors import create_random_tensor

# main
def main():
    args = parse_args()
    print_args(args)

    # train_dataset, val_dataset, test_dataset = create_epiano_datasets(args.input_dir)

    x = create_random_tensor((args.batch_size, args.max_sequence), dtype=TORCH_FLOAT)
    x = (x * VOCAB_SIZE).type(TORCH_LABEL_TYPE)
    # print(x.shape)

    model = MusicTransformer(args).to(TORCH_DEVICE)
    y = model(x)
    model.eval()
    y = model.generate()

if __name__ == "__main__":
    main()
