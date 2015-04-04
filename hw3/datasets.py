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


class Insurance():
    def __init__(self):
        self.n_class = 2
        self.train = self.load_train()
        self.test = self.load_test()

    def load_train(self):
        with open('data/train.txt') as f:
            data = csv.reader(f, delimiter='\t')
            data = [row for row in data]
        train_x = [map(int, x[0:len(x)-1]) for x in data]
        train_y = [int(x[-1]) for x in data]
        return Dataset(train_x, train_y)

    def load_test(self):
        with open('data/testx.txt') as f:
            data = csv.reader(f, delimiter='\t')
            data = [row for row in data]
        test_x = [map(int, x) for x in data]

        with open('data/testy.txt') as f:
            data = csv.reader(f, delimiter='\t')
            data = [row for row in data]
        test_y = [int(x[0]) for x in data]
        return Dataset(test_x, test_y)


class Forest():
    """ Load Forest Cover type data set. 
    """
    def __init__(self):
        self.n_class = 7
        self.train, self.test = self.load_data()

    def load_data(self):
        with open('data/forest_data_clean.csv') as f:
            data = csv.reader(f, delimiter=',')
            # A little bit of munging: drop Id column and change int16 
            data = np.asarray([x[1:] for i, x in enumerate(data) if i > 0], 
                              dtype=np.int16)

        # Perform a 4:1 split for training and test data
        msk = np.random.rand(len(data)) < 0.8
        trainx = data[msk, :-1]
        trainy = data[msk, -1]
        testx = data[~msk, :-1]
        testy = data[~msk, -1]

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
            data = csv.reader(f, delimiter=',')
            train = [x for i, x in enumerate(data) if i > 0] 
            # Extract features and target variable separately
            trainx = [x[3:] for x in train]
            trainy = [x[2] for x in train]

        with open('data/fordTest.csv') as f:
            data = csv.reader(f, delimiter=',')
            testx = [x[3:] for i, x in enumerate(data) if i > 0] 

        with open('data/Solution.csv') as f:
            data = csv.reader(f, delimiter=',')
            testy = [x[2] for i, x in enumerate(data) if i > 0] 

        # Extract features and target variable, convert to numpy array
        trainx = np.asarray(trainx, dtype=np.float32)
        trainy = np.asarray(trainy, dtype=np.int8)
        testx = np.asarray(testx, dtype=np.float32)
        testy = np.asarray(testy, dtype=np.int8)

        # Return training and test sets
        trainSet = Dataset(trainx, trainy)
        testSet = Dataset(testx, testy)
        return trainSet, testSet
