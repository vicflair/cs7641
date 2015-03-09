__author__ = 'vma'

import csv
import random
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.neuralnets import NNclassifier
from sklearn.metrics import accuracy_score


class ANN():
    def __init__(self, num_hidden=5, momentum=0.1, weightdecay=0.01, 
                 verbose=False):
        self.num_hidden = num_hidden
        self.train_file = 'imgseg/segmentation.train'
        self.test_file = 'imgseg/segmentation.test'
        self.train = self.load_data(self.train_file)
        self.test = self.load_data(self.test_file)
        self.network = self.make_network(num_hidden)
        self.domain = [(float('-Inf'), float('Inf'))]*len(self.network.params)
        self.trainer = self.make_trainer(momentum=momentum, 
                                         weightdecay=weightdecay, 
                                         verbose=verbose)


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
        ds = ClassificationDataSet(19, 1, nb_classes=7)
        for x, y in zip(tx, ty):
            ds.addSample(x, y)
        ds._convertToOneOfMany()
        return ds

    def make_network(self, num_hidden=5):
        return buildNetwork(self.train.indim, num_hidden, self.train.outdim)

    def make_trainer(self, momentum=0.1, weightdecay=0.01, verbose=False):
        return BackpropTrainer(self.network, dataset=self.train,
                               momentum=momentum, weightdecay=weightdecay,
                               verbose=verbose)

    def train_network(self):
        self.trainer.trainEpochs(1)

    def fitf(self, weights=[], train=True):
        weights = list(weights)
        if not weights:
            weights = self.network.params
        if train:
            ds = self.train
        else:
            ds = self.test
        self.network._setParameters(weights)
        pred = self.network.activateOnDataset(ds)
        preds = [y.argmax() for y in pred]
        return accuracy_score(preds, ds['class'], normalize=True)

    def neighbors(self, weights):
        weights = list(weights)
        neighbors = []
        for i in range(len(weights)):
            step = random.gauss(0, 1)
            neighbors.append(weights[0:i] + [weights[i]+step] + weights[i+1:])
            neighbors.append(weights[0:i] + [weights[i]-step] + weights[i+1:])
        return neighbors

    def random_solution(self):
        self.network.randomize()
        return list(self.network.params)
