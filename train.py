import os
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy

from model.music_transformer import MusicTransformer
from model.loss import SmoothCrossEntropyLoss

from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.lr_scheduling import LrStepTracker, get_lr
from utilities.argument_funcs import parse_train_args, print_train_args, write_model_params
from utilities.run_model import train_epoch, eval_model

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Trains a model specified by command line arguments
    ----------
    """

    args = parse_train_args()
    print_train_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.output_dir, exist_ok=True)

    # Output prep
    params_file = os.path.join(args.output_dir, "model_params.txt")
    write_model_params(args, params_file)

    weights_folder = os.path.join(args.output_dir, "weights")
    os.makedirs(weights_folder, exist_ok=True)

    results_folder = os.path.join(args.output_dir, "results")
    os.makedirs(results_folder, exist_ok=True)

    # Datasets
    train_dataset, val_dataset, test_dataset = create_epiano_datasets(args.input_dir, args.max_sequence)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    # Continuing from previous training session
    start_epoch = 0
    if(args.continue_weights is not None):
        if(args.continue_epoch is None):
            print("ERROR: Need epoch number to continue from (-continue_epoch) when using continue_weights")
            return
        else:
            model.load_state_dict(torch.load(args.continue_weights))
            start_epoch = args.continue_epoch
    elif(args.continue_epoch is not None):
        print("ERROR: Need continue weights (-continue_weights) when using continue_epoch")
        return

    # Lr Scheduler vs static lr
    if(args.lr is None):
        if(args.continue_epoch is None):
            init_step = 0
        else:
            init_step = args.continue_epoch * len(train_loader)

        lr = LR_DEFAULT_START
        lr_stepper = LrStepTracker(args.d_model, SCHEDULER_WARMUP_STEPS, init_step)
    else:
        lr = args.lr

    # Not smoothing evaluation loss
    eval_loss = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)

    # SmoothCrossEntropyLoss or CrossEntropyLoss for training
    if(args.ce_smoothing is None):
        train_loss = eval_loss
    else:
        train_loss = SmoothCrossEntropyLoss(args.ce_smoothing, VOCAB_SIZE, ignore_index=TOKEN_PAD)

    # Optimizer
    opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)

    if(args.lr is None):
        lr_scheduler = LambdaLR(opt, lr_stepper.step)
    else:
        lr_scheduler = None

    best_acc        = 0.0
    best_acc_epoch  = -1
    best_loss       = float("inf")
    best_loss_epoch = -1

    # TRAIN LOOP
    for epoch in range(start_epoch, args.epochs):
        print(SEPERATOR)
        print("NEW EPOCH:", epoch+1)
        print(SEPERATOR)
        print("")

        train_epoch(epoch+1, model, train_loader, train_loss, opt, lr_scheduler)

        print(SEPERATOR)
        print("Evaluating:")

        cur_loss, cur_acc = eval_model(model, test_loader, eval_loss)

        print("Avg loss:", cur_loss)
        print("Avg acc:", cur_acc)
        print(SEPERATOR)
        print("")

        if(cur_acc > best_acc):
            best_acc    = cur_acc
            best_acc_epoch  = epoch+1
        if(cur_loss < best_loss):
            best_loss       = cur_loss
            best_loss_epoch = epoch+1



        epoch_str = str(epoch+1).zfill(PREPEND_ZEROS_WIDTH)

        if((epoch+1) % args.weight_modulus == 0):
            path = os.path.join(weights_folder, "epoch_" + epoch_str + ".pickle")
            torch.save(model.state_dict(), path)

        path = os.path.join(results_folder, "epoch_" + epoch_str + ".txt")
        o_stream = open(path, "w")
        o_stream.write(str(cur_acc) + "\n")
        o_stream.write(str(cur_loss) + "\n")
        o_stream.close()

    print(SEPERATOR)
    print("Best acc epoch:", best_acc_epoch)
    print("Best acc:", best_acc)
    print("")
    print("Best loss epoch:", best_loss_epoch)
    print("Best loss:", best_loss)


if __name__ == "__main__":
    main()
