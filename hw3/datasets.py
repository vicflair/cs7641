__author__ = 'vma'

import csv
import numpy as np
import pandas
import random


class Dataset():
    def __init__(self, input, labels):
        self.X = input
        self.Y = labels
        self.N = len(np.unique(labels))
        

class Segmentation():
    def __init__(self):
        self.n_class = 7
        self.train_file = 'data/segmentation.train'
        self.test_file = 'data/segmentation.test'
        self.train = self.load_data(self.train_file)
        self.test = self.load_data(self.test_file)

    def load_data(self, fname):
        with open(fname) as f:
            data = csv.reader(f, delimiter=',')
            data = [row for row in data]
        tx = [map(float, x[1:len(x)]) for x in data[6:]]
        ty = [x[0] for x in data[6:]]
        for i, label in enumerate(ty):
            if label == 'BRICKFACE':
                ty[i] = 0
            elif label == 'SKY':
                ty[i] = 1
            elif label == 'FOLIAGE':
                ty[i] = 2
            elif label == 'CEMENT':
                ty[i] = 3
            elif label == 'WINDOW':
                ty[i] = 4
            elif label == 'PATH':
                ty[i] = 5
            elif label == 'GRASS':
                ty[i] = 6
            else:
                print 'error'
        txy = Dataset(tx, ty)
        return txy


class Forest():
    """ Load Forest Cover type data set. 
    """
    def __init__(self):
        self.n_class = 2
        self.train, self.test = self.load_data()

    def load_data(self):
        with open('data/forest_data_clean.csv') as f:
            data = pandas.read_csv(f, delimiter=',')

        # A little bit of munging: drop Id column and change int to float
        data = data.iloc[:, 1:-1].astype(np.float)

        # Perform a 4:1 split for training and test data
        msk = np.random.rand(len(data)) < 0.8
        trainx = data.iloc[msk, :-1].as_matrix()
        trainy = data.iloc[msk, -1].as_matrix()
        testx = data.iloc[~msk, :-1].as_matrix()
        testy = data.iloc[~msk, -1].as_matrix()

        # Return training and test sets
        trainSet = Dataset(trainx, trainy)
        testSet = Dataset(testx, testy)
        return trainSet, testSet


class Alertness():
    def __init__(self):
        self.n_class = 2
        self.train, self.test = self.load_data()

    def load_data(self):
        """ Load Ford alertness data set. 
        """
        with open('data/fordTrain.csv') as f:
            data = pandas.read_csv(f, sep=',')

        with open('data/fordTest.csv') as f:
            test = pandas.read_csv(f, sep=',')

        with open('data/Solution.csv') as f:
            soln = pandas.read_csv(f, sep=',')

        trainx = data.loc[:, [col for col in data.columns if col != 'IsAlert']]
        trainy = data.loc[:, 'IsAlert']
        testx = test.loc[:, [col for col in test.columns if col != 'IsAlert']]
        testy = soln.Prediction

        # Pre-process data.
        pass

        # Return training and test sets
        trainSet = Dataset(trainx.as_matrix(), trainy.as_matrix())
        testSet = Dataset(testx.as_matrix(), testy.as_matrix())
        return trainSet, testSet
