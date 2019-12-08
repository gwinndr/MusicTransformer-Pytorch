import torch
import torch.nn as nn
import os

from utilities.argument_funcs import parse_args, print_args
from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy
from torch.utils.data import DataLoader
from torch.optim import Adam

from utilities.constants import *
from utilities.tensors import create_random_tensor

# main
def main():
    args = parse_args()
    print_args(args)

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset, val_dataset, test_dataset = create_epiano_datasets(args.input_dir, args.max_sequence)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    model = MusicTransformer(args).to(TORCH_DEVICE)

    loss    = nn.CrossEntropyLoss()
    opt     = Adam(model.parameters(), lr=0.0001, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)


    for epoch in range(args.epochs):
        print(SEPERATOR)
        print("NEW EPOCH:", epoch+1)
        print(SEPERATOR)
        print("")

        model.train()
        for batch_num, batch in enumerate(train_loader):
            opt.zero_grad()

            batch = batch.to(TORCH_DEVICE)

            y       = model(batch)
            batch   = batch[:, 1:]

            y   = y.permute(0,2,1) # (batch_size, classes, max_seq)
            out = loss(y, batch)

            out.backward()
            opt.step()

            print(SEPERATOR)
            print("Epoch", epoch+1, " Batch", batch_num+1, "/", len(train_loader))
            print("Loss:", float(out))
            print(SEPERATOR)
            print("")

        print(SEPERATOR)
        print("Evaluating:")
        model.eval()

        with torch.set_grad_enabled(False):
            n_test      = len(test_loader)
            full_loss   = 0.0
            full_acc    = 0.0
            for batch in test_loader:
                batch = batch.to(TORCH_DEVICE)

                y       = model(batch)
                batch   = batch[:, 1:]

                full_acc += float(compute_epiano_accuracy(batch, y))

                y   = y.permute(0,2,1) # (batch_size, classes, max_seq)
                out = loss(y, batch)

                full_loss += float(out)

            print("Avg loss:", full_loss / n_test)
            print("Avg acc:", full_acc / n_test)
            print(SEPERATOR)
            print("")



        path = os.path.join(args.output_dir, "epoch_" + str(epoch+1) + ".pickle")
        torch.save(model.state_dict(), path)


    # model = MusicTransformer(args).to(TORCH_DEVICE)
    # y = model(x)
    # model.eval()
    # y = model.generate()

if __name__ == "__main__":
    main()
