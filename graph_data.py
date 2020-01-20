import argparse
import os, sys
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

def main():
    """
    ----------
    Author: Ben Myrick
    ----------
    Entry point
    ----------
    """

    inputFolder = 'test_data/'
    outputFolder = 'outputGraphs/'
    parseInput(inputFolder, outputFolder)

def parseInput(InputFolder, outputFolder):
    """
    ----------
    Author: Ben Myrick
    ----------
    Graphs model training and evaluation data
    ----------
    """

    parser = argparse.ArgumentParser()

    #Initialize Loss and Accuracy arrays
    loss_arr = []
    accuracy_arr = []
    epochs = []

    number_of_epochs = 3000
    for x in range(1,number_of_epochs):
        ep_num = ""
        if x < 10:
            ep_num = "epoch_000" + str(x)
        elif x < 100 and x > 9:
            ep_num = "epoch_00" + str(x)
        elif x < 1000 and x > 99:
            ep_num = "epoch_0" + str(x)
        elif x < 3001 and x > 999:
            ep_num = "epoch_" + str(x)

        path = InputFolder + ep_num + ".txt"

        #Read file and parse accuracy and loss values
        file = open(path, 'r')
        temp_average_accuracy = file.readline()
        temp_average_loss = file.readline()

        #Update accuracy and loss arrays for each epoch
        accuracy_arr.append(float(temp_average_accuracy))
        loss_arr.append(float(temp_average_loss))

        epochs += [x]

    try:
        os.mkdir(outputFolder)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

    #Create and save plots to output folder
    plt.plot(epochs,loss_arr)
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(outputFolder + 'loss_graph.png')

    plt.plot(epochs,accuracy_arr)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Epochs')
    fig2 = plt.gcf()
    plt.show()
    fig2.savefig(outputFolder + 'accuracy_graph.png')

if __name__ == "__main__":
    main()
