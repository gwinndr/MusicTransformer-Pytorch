import torch
import time

from .constants import *
from utilities.device import get_device
from .lr_scheduling import get_lr

from dataset.e_piano import compute_epiano_accuracy


# train_epoch
def train_epoch(cur_epoch, model, dataloader, loss, opt, lr_scheduler=None, print_modulus=1):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Trains a single model epoch
    ----------
    """

    out = -1
    model.train()
    for batch_num, batch in enumerate(dataloader):
        time_before = time.time()

        opt.zero_grad()

        x   = batch[0].to(get_device())
        tgt = batch[1].to(get_device())

        y = model(x)

        y   = y.reshape(y.shape[0] * y.shape[1], -1)
        tgt = tgt.flatten()

        out = loss.forward(y, tgt)

        out.backward()
        opt.step()

        if(lr_scheduler is not None):
            lr_scheduler.step()

        time_after = time.time()
        time_took = time_after - time_before

        if((batch_num+1) % print_modulus == 0):
            print(SEPERATOR)
            print("Epoch", cur_epoch, " Batch", batch_num+1, "/", len(dataloader))
            print("LR:", get_lr(opt))
            print("Train loss:", float(out))
            print("")
            print("Time (s):", time_took)
            print(SEPERATOR)
            print("")

    return

# eval_model
def eval_model(model, dataloader, loss):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Evaluates the model and prints the average loss and accuracy
    ----------
    """

    model.eval()

    avg_acc     = -1
    avg_loss    = -1
    with torch.set_grad_enabled(False):
        n_test      = len(dataloader)
        sum_loss   = 0.0
        sum_acc    = 0.0
        for batch in dataloader:
            x   = batch[0].to(get_device())
            tgt = batch[1].to(get_device())

            y = model(x)

            sum_acc += float(compute_epiano_accuracy(y, tgt))

            y   = y.reshape(y.shape[0] * y.shape[1], -1)
            tgt = tgt.flatten()

            out = loss.forward(y, tgt)

            sum_loss += float(out)

        avg_loss    = sum_loss / n_test
        avg_acc     = sum_acc / n_test

    return avg_loss, avg_acc
