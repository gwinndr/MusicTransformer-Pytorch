import argparse
import os
import matplotlib.pyplot as plt

# graph_results
def graph_results(input_folder="./saved_models", output_dir=None, epoch_start=0, epoch_end=None):
    """
    ----------
    Author: Ben Myrick
    Modified: Damon Gwinn
    ----------
    Graphs model training and evaluation data
    ----------
    """

    #Initialize Loss and Accuracy arrays
    loss_arr = []
    accuracy_arr = []
    epochs = []

    fs = [os.path.join(input_folder, f) for f in sorted(os.listdir(input_folder))]
    fs = [f for f in fs if os.path.isfile(f)]

    if(epoch_end is None):
        epoch_end = len(fs)

    print("Gathering results...")
    for x in range(epoch_start, epoch_end):
        path = fs[x]

        #Read file and parse accuracy and loss values
        file = open(path, 'r')
        temp_average_accuracy = file.readline()
        temp_average_loss = file.readline()

        #Update accuracy and loss arrays for each epoch
        accuracy_arr.append(float(temp_average_accuracy))
        loss_arr.append(float(temp_average_loss))
        epochs.append(x)

        file.close()
    print("Done!")
    print()

    if(output_dir is not None):
        try:
            os.mkdir(output_dir)
        except OSError:
            print ("Creation of the directory %s failed" % output_dir)
        else:
            print ("Successfully created the directory %s" % output_dir)

    #Create and save plots to output folder
    plt.plot(epochs,loss_arr)
    plt.title("Loss Results")
    plt.ylabel('Loss (Cross Entropy)')
    plt.xlabel('Epochs')
    fig1 = plt.gcf()
    plt.show()

    if(output_dir is not None):
        fig1.savefig(os.path.join(output_dir, 'loss_graph.png'))

    plt.plot(epochs,accuracy_arr)
    plt.title("Accuracy Results")
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')
    fig2 = plt.gcf()
    plt.show()

    if(output_dir is not None):
        fig2.savefig(os.path.join(output_dir, 'accuracy_graph.png'))

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

    parser.add_argument("-input_dir", type=str, default="./saved_models/results", help="Input results folder from trained model ('results' folder)")
    parser.add_argument("-output_dir", type=str, default=None, help="Optional output folder to save graph pngs")
    parser.add_argument("-epoch_start", type=int, default=0, help="Epoch start. Defaults to first file.")
    parser.add_argument("-epoch_end", type=int, default=None, help="Epoch end (non-inclusive). Defaults to None.")

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

    graph_results(args.input_dir, args.output_dir, args.epoch_start, args.epoch_end)

if __name__ == "__main__":
    main()
