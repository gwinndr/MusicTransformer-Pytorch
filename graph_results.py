import argparse
import os
import csv
import math
import matplotlib.pyplot as plt

RESULTS_FILE = "results.csv"
EPOCH_IDX = 0
LR_IDX = 1
EVAL_LOSS_IDX = 4
EVAL_ACC_IDX = 5

SPLITTER = '?'

# graph_results
def graph_results(input_dirs="./saved_models/results", output_dir=None, model_names=None, epoch_start=0, epoch_end=None):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Graphs model training and evaluation data
    ----------
    """

    input_dirs = input_dirs.split(SPLITTER)

    if(model_names is not None):
        model_names = model_names.split(SPLITTER)
        if(len(model_names) != len(input_dirs)):
            print("Error: len(model_names) != len(input_dirs)")
            return

    #Initialize Loss and Accuracy arrays
    loss_arrs = []
    accuracy_arrs = []
    epoch_counts = []
    lrs = []

    for input_dir in input_dirs:
        loss_arr = []
        accuracy_arr = []
        epoch_count = []
        lr_arr = []

        f = os.path.join(input_dir, RESULTS_FILE)
        with open(f, "r") as i_stream:
            reader = csv.reader(i_stream)
            next(reader)

            lines = [line for line in reader]

        if(epoch_end is None):
            epoch_end = math.inf

        epoch_start = max(epoch_start, 0)
        epoch_start = min(epoch_start, epoch_end)

        for line in lines:
            epoch = line[EPOCH_IDX]
            lr = line[LR_IDX]
            accuracy = line[EVAL_ACC_IDX]
            loss = line[EVAL_LOSS_IDX]

            if(int(epoch) >= epoch_start and int(epoch) < epoch_end):
                accuracy_arr.append(float(accuracy))
                loss_arr.append(float(loss))
                epoch_count.append(int(epoch))
                lr_arr.append(float(lr))

        loss_arrs.append(loss_arr)
        accuracy_arrs.append(accuracy_arr)
        epoch_counts.append(epoch_count)
        lrs.append(lr_arr)

    if(output_dir is not None):
        try:
            os.mkdir(output_dir)
        except OSError:
            print ("Creation of the directory %s failed" % output_dir)
        else:
            print ("Successfully created the directory %s" % output_dir)

    ##### LOSS #####
    for i in range(len(loss_arrs)):
        if(model_names is None):
            name = None
        else:
            name = model_names[i]

        #Create and save plots to output folder
        plt.plot(epoch_counts[i], loss_arrs[i], label=name)
        plt.title("Loss Results")
        plt.ylabel('Loss (Cross Entropy)')
        plt.xlabel('Epochs')
        fig1 = plt.gcf()

    plt.legend(loc="upper left")

    if(output_dir is not None):
        fig1.savefig(os.path.join(output_dir, 'loss_graph.png'))

    plt.show()

    ##### ACCURACY #####
    for i in range(len(accuracy_arrs)):
        if(model_names is None):
            name = None
        else:
            name = model_names[i]

        #Create and save plots to output folder
        plt.plot(epoch_counts[i], accuracy_arrs[i], label=name)
        plt.title("Accuracy Results")
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        fig2 = plt.gcf()

    plt.legend(loc="upper left")

    if(output_dir is not None):
        fig2.savefig(os.path.join(output_dir, 'accuracy_graph.png'))

    plt.show()

    ##### LR #####
    for i in range(len(lrs)):
        if(model_names is None):
            name = None
        else:
            name = model_names[i]

        #Create and save plots to output folder
        plt.plot(epoch_counts[i], lrs[i], label=name)
        plt.title("Learn Rate Results")
        plt.ylabel('Learn Rate')
        plt.xlabel('Epochs')
        fig2 = plt.gcf()

    plt.legend(loc="upper left")

    if(output_dir is not None):
        fig2.savefig(os.path.join(output_dir, 'lr_graph.png'))

    plt.show()

# graph_results_legacy
def graph_results_legacy(input_dirs="./saved_models/results", output_dir=None, model_names=None, epoch_start=0, epoch_end=None):
    """
    ----------
    Author: Ben Myrick
    Modified: Damon Gwinn
    ----------
    Graphs model training and evaluation data using the old results format (legacy)
    ----------
    """

    input_dirs = input_dirs.split(SPLITTER)

    if(model_names is not None):
        model_names = model_names.split(SPLITTER)
        if(len(model_names) != len(input_dirs)):
            print("Error: len(model_names) != len(input_dirs)")
            return

    #Initialize Loss and Accuracy arrays
    loss_arrs = []
    accuracy_arrs = []
    epoch_counts = []

    for input_dir in input_dirs:
        loss_arr = []
        accuracy_arr = []
        epoch_count = []

        fs = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir))]
        fs = [f for f in fs if os.path.isfile(f)]

        if(epoch_end is None):
            epoch_end = len(fs)
        else:
            epoch_end = min(epoch_end, len(fs))

        epoch_start = max(epoch_start, 0)
        epoch_start = min(epoch_start, epoch_end)

        for x in range(epoch_start, epoch_end):
            path = fs[x]

            #Read file and parse accuracy and loss values
            file = open(path, 'r')
            temp_average_accuracy = file.readline()
            temp_average_loss = file.readline()

            #Update accuracy and loss arrays for each epoch
            accuracy_arr.append(float(temp_average_accuracy))
            loss_arr.append(float(temp_average_loss))
            epoch_count.append(x)

            file.close()

        loss_arrs.append(loss_arr)
        accuracy_arrs.append(accuracy_arr)
        epoch_counts.append(epoch_count)

    if(output_dir is not None):
        try:
            os.mkdir(output_dir)
        except OSError:
            print ("Creation of the directory %s failed" % output_dir)
        else:
            print ("Successfully created the directory %s" % output_dir)

    for i in range(len(loss_arrs)):
        if(model_names is None):
            name = input_dirs[i]
        else:
            name = model_names[i]

        #Create and save plots to output folder
        plt.plot(epoch_counts[i], loss_arrs[i], label=name)
        plt.title("Loss Results")
        plt.ylabel('Loss (Cross Entropy)')
        plt.xlabel('Epochs')
        fig1 = plt.gcf()

    plt.legend(loc="upper left")

    if(output_dir is not None):
        fig1.savefig(os.path.join(output_dir, 'loss_graph.png'))

    plt.show()

    for i in range(len(loss_arrs)):
        if(model_names is None):
            name = input_dirs[i]
        else:
            name = model_names[i]

        #Create and save plots to output folder
        plt.plot(epoch_counts[i], accuracy_arrs[i], label=name)
        plt.title("Accuracy Results")
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        fig2 = plt.gcf()

    plt.legend(loc="upper left")

    if(output_dir is not None):
        fig2.savefig(os.path.join(output_dir, 'accuracy_graph.png'))

    plt.show()

# parse_args
def parse_args():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Argparse arguments
    ----------
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-input_dirs", type=str, default="./saved_models/results", help="Input results folder from trained model ('results' folder). Seperate with '?' symbol for comparisons between models")
    parser.add_argument("-output_dir", type=str, default=None, help="Optional output folder to save graph pngs")
    parser.add_argument("-model_names", type=str, default=None, help="Names to display when color coding, seperate with ':'.")
    parser.add_argument("-epoch_start", type=int, default=0, help="Epoch start. Defaults to first file.")
    parser.add_argument("-epoch_end", type=int, default=None, help="Epoch end (non-inclusive). Defaults to None.")
    parser.add_argument("--legacy", action="store_true", help="Use legacy results output format (you likely don't need this)")

    return parser.parse_args()

def main():
    """
    ----------
    Author: Ben Myrick
    Modified: Damon Gwinn
    ----------
    Entry point
    ----------
    """

    args = parse_args()

    if(not args.legacy):
        graph_results(args.input_dirs, args.output_dir, args.model_names, args.epoch_start, args.epoch_end)
    else:
        graph_results_legacy(args.input_dirs, args.output_dir, args.model_names, args.epoch_start, args.epoch_end)

if __name__ == "__main__":
    main()
