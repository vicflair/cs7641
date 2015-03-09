__author__ = 'vma'

import csv
import numpy
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

with open('train.txt') as f:
    data = csv.reader(f, delimiter='\t')
    data = [row for row in data]
train = ClassificationDataSet(85, target=1, nb_classes=2)
for i in range(1,len(data)):
    train.addSample(data[i][0:len(data[0])-1], data[i][-1])

with open('testx.txt') as f:
    data = csv.reader(f, delimiter='\t')
    testx = [row for row in data]
with open('testy.txt') as f:
    data = csv.reader(f, delimiter='\t')
    testy = [row[0] for row in data]
testset = ClassificationDataSet(85, target=1, nb_classes=2)
for i in range(1, len(testy)):
    testset.addSample(testx[i], testy[i])

train._convertToOneOfMany()
testset._convertToOneOfMany()

# ####
print "Number of training patterns: ", len(train)
print "Input and output dimensions: ", train.indim, train.outdim
print "First sample (input, target):"
print train['input'][0], train['target'][0]

# ffnn = FeedForwardNetwork()
# inLayer = LinearLayer(85)
# hiddenLayer = SigmoidLayer(2)
# outLayer = LinearLayer(2)
# ffnn.addInputModule(inLayer)
# ffnn.addModule(hiddenLayer)
# ffnn.addOutputModule(outLayer)
# in_to_hidden = FullConnection(inLayer, hiddenLayer)
# hidden_to_out = FullConnection(hiddenLayer, outLayer)
# ffnn.addConnection(in_to_hidden)
# ffnn.addConnection(hidden_to_out)
# ffnn.sortModules()

fnn = buildNetwork(train.indim, 3, train.outdim, bias=True, outclass=SoftmaxLayer)

trainer = BackpropTrainer(fnn, dataset=train, learningrate=0.01, momentum=0.1, verbose=True, weightdecay=0.01)
trainer.trainUntilConvergence(verbose=True, trainingData=train, validationData=testset, maxEpochs=5)

# for i in range(10):
#     trainer.trainEpochs(1)
#
#     trnresult = percentError(trainer.testOnClassData(), train['target'])
#     tstresult = percentError(trainer.testOnClassData(dataset=testset), testset['target'])
#
#     print "epoch: %4d" % trainer.totalepochs, \
#           "  train error: %5.7f%%" % trnresult, \
#           "  test error: %5.7f%%" % tstresult
