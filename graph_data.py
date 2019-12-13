import argparse
import os, sys
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


def main():
    inputFolder = 'test_data/'
    outputFolder = 'outputGraphs/'

    parseInput(inputFolder, outputFolder)


def parseInput(InputFolder, outputFolder):
    parser = argparse.ArgumentParser()

    #Initialize Loss and Accuracy arrays
    loss_arr = []
    accuracy_arr = []
    epochs = []

    number_of_epochs = 3000
    for x in range(1,number_of_epochs):
        ep_num = ""
        if number_of_epochs < 10:
            ep_num = "epoch_000" + str(x)
        elif number_of_epochs < 100 and number_of_epochs > 9:
            ep_num = "epoch_00" + str(x)
        elif number_of_epochs < 1000 and number_of_epochs > 99:
            ep_num = "epoch_0" + str(x)
        elif number_of_epochs < 3001 and number_of_epochs > 999:
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
