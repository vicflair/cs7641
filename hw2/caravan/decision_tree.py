__author__ = 'vma'

import csv
from sklearn import tree

with open('train.txt') as f:
    data = csv.reader(f, delimiter='\t')
    data = [row for row in data]
train_x = [map(int, x[0:len(x)-1]) for x in data]
train_y = [int(x[-1]) for x in data]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x, train_y)

with open('testx.txt') as f:
    data = csv.reader(f, delimiter='\t')
    data = [row for row in data]
test_x = [map(int, x) for x in data]

with open('testy.txt') as f:
    data = csv.reader(f, delimiter='\t')
    data = [row for row in data]
test_y = [int(x[0]) for x in data]

pred = clf.predict(test_x)
results = (pred == test_y)
accuracy = 1. * sum(results)/len(results)

