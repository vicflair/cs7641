__author__ = 'vma'

import csv
import numpy as np
import neurolab as nl

with open('train.txt') as f:
    data = csv.reader(f, delimiter='\t')
    data = [row for row in data]
trainx = [map(int, row[0:85]) for row in data]
trainy = [[int(row[-1])] for row in data]

input = np.asarray(trainx)
target = np.asarray(trainy)

with open('testx.txt') as f:
    data = csv.reader(f, delimiter='\t')
    testx = [row for row in data]
with open('testy.txt') as f:
    data = csv.reader(f, delimiter='\t')
    testy = [row[0] for row in data]

net = nl.net.newff([[0, 50]]*85, [5, 5, 1])
err = net.train(input, target, epochs=1000, show=15)
#net.sim([[0.2, 0.2]])