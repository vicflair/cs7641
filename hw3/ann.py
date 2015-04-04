__author__ = 'vma'

import csv
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.neuralnets import NNclassifier
from sklearn.metrics import accuracy_score


class ANN():
    def __init__(self, verbose=False):
        self.train = None
        self.test = None 
        self.network = None
        self.trainer = None 

    def load_data(self, X, Y, N):
        input_size = len(X[0])
        ds = ClassificationDataSet(input_size, 1, nb_classes=N)
        for x, y in zip(X, Y):
            ds.addSample(x, y)
        ds._convertToOneOfMany()
        return ds

    def make_network(self, num_hidden=5):
        self.network = buildNetwork(self.train.indim, num_hidden, self.train.outdim)

    def make_trainer(self, momentum=0.1, weightdecay=0.01, verbose=False):
        self.trainer = BackpropTrainer(self.network, dataset=self.train,
                               momentum=momentum, weightdecay=weightdecay,
                               verbose=verbose)

    def train_network(self):
        self.trainer.trainEpochs(1)

    def fitf(self, train=True):
        if train:
            ds = self.train
        else:
            ds = self.test
        pred = self.network.activateOnDataset(ds)
        preds = [y.argmax() for y in pred]
        return accuracy_score(preds, ds['class'], normalize=True)
