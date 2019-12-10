import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.optim import Adam

from utilities.lr_scheduling import LrStepTracker, get_lr
from utilities.argument_funcs import parse_args, print_args
from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy
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

    # Lr Scheduler vs static lr
    if(args.lr is None):
        lr = LR_DEFAULT_START
        lr_stepper = LrStepTracker(args.d_model, SCHEDULER_WARMUP_STEPS)
    else:
        lr = args.lr

    loss    = nn.CrossEntropyLoss()
    opt     = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)

    if(args.lr is None):
        lr_scheduler = LambdaLR(opt, lr_stepper.step)

    for epoch in range(args.epochs):
        print(SEPERATOR)
        print("NEW EPOCH:", epoch+1)
        print(SEPERATOR)
        print("")

        model.train()
        for batch_num, batch in enumerate(train_loader):
            opt.zero_grad()

            x       = batch[0].to(TORCH_DEVICE)
            tgt     = batch[1].to(TORCH_DEVICE)

            y = model(x)

            y   = y.permute(0,2,1) # (batch_size, classes, max_seq)
            out = loss(y, tgt)

            out.backward()
            opt.step()

            if(args.lr is None):
                lr_scheduler.step()

            print(SEPERATOR)
            print("Epoch", epoch+1, " Batch", batch_num+1, "/", len(train_loader))
            print("LR:", get_lr(opt))
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
                x       = batch[0].to(TORCH_DEVICE)
                tgt     = batch[1].to(TORCH_DEVICE)

                y = model(x)

                full_acc += float(compute_epiano_accuracy(tgt, y))

                y   = y.permute(0,2,1) # (batch_size, classes, max_seq)
                out = loss(y, tgt)

                full_loss += float(out)

            print("Avg loss:", full_loss / n_test)
            print("Avg acc:", full_acc / n_test)
            print(SEPERATOR)
            print("")



        path = os.path.join(args.output_dir, "epoch_" + str(epoch+1) + ".pickle")
        torch.save(model.state_dict(), path)


if __name__ == "__main__":
    main()
