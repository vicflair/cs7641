__author__ = 'vma'

import csv
import numpy as np

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
