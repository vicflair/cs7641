import csv
import numpy as np
import pandas
import pickle
import pylab as plt
import random
import time
from functools import wraps
from matplotlib.font_manager import FontProperties
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer
from pybrain.structure import TanhLayer, FullConnection
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from sklearn import cross_validation
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.learning_curve import learning_curve
from sklearn.metrics import accuracy_score

__author__ = 'vma'


def timeit(func):
    """ Decorator function used to time execution.
    """
    @wraps(func)
    def timed_function(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        end = time.time()
        print '%s execution time: %f secs' % (func.__name__, end - start)
        return output
    return timed_function


def load_data_ford():
    """ Load Ford alertness data set. Return variables are:
    trainx - features of the training set
    trainy - labels of the training set
    testx - features of the test set
    testy - labels of the test set
    """
    with open('../data/fordTrain.csv') as f:
        data = pandas.read_csv(f, sep=',')

    with open('../data/fordTest.csv') as f:
        test = pandas.read_csv(f, sep=',')

    with open('../data/Solution.csv') as f:
        soln = pandas.read_csv(f, sep=',')

    trainx = data.loc[:, [col for col in data.columns if col != 'IsAlert']]
    trainy = data.loc[:, 'IsAlert']
    testx = test.loc[:, [col for col in test.columns if col != 'IsAlert']]
    testy = soln.Prediction

    # Pre-process data.
    trainx, trainy, testx, testy = munge_ford_data(trainx, trainy, testx, testy)
    return trainx, trainy, testx, testy


def load_data_forest():
    """ Load Forest Cover type data set. Return variables are:
    trainx - features of the training set
    trainy - labels of the training set
    testx - features of the test set
    testy - labels of the test set
    """

    with open('../data/forest_data_clean.csv') as f:
        data = pandas.read_csv(f, delimiter=',')
    # Optionally sub-sample, then
    rows = random.sample(data.index, 580000)
    data = data.iloc[rows, 1:]  # also drop Id column
    data_to_scale = data.iloc[:, :-1].astype(np.float)  # change int to float
    scaler = preprocessing.StandardScaler().fit(data_to_scale)
    data.iloc[:, :-1] = scaler.transform(data_to_scale)
    # Perform a 4:1 split for training and test data
    msk = np.random.rand(len(data)) < 0.8
    trainx = data.iloc[msk, :-1]
    trainy = data.iloc[msk, -1]
    testx = data.iloc[~msk, :-1]
    testy = data.iloc[~msk, -1]

    return trainx, trainy, testx, testy


def munge_ford_data(trainx, trainy, testx, testy):
    """ This function is used to pre-processed the Ford data set in order
    to reduce the feature space and improve run time.
    """

    # Search for highly correlated things
    # c = trainx.corr().abs()
    # s = c.unstack()
    # so = s.order(kind="quicksort", na_last=False)[::-1]

    # Drop TrialID and ObsNum because they are irrelevant
    trainx = trainx.drop(['TrialID', 'ObsNum'], 1)
    testx = testx.drop(['TrialID', 'ObsNum'], 1)

    # Drop P4, V6, V10, E9, E2, E9, V8 b/c they are highly correlated (>0.5)
    trainx = trainx.drop(['P4', 'V6', 'V10', 'E9', 'E2'], 1)
    testx = testx.drop(['P4', 'V6', 'V10', 'E9', 'E2'], 1)

    # Drop P8, V7, V9 because their values are all 0
    trainx = trainx.drop(['P8', 'V7', 'V9'], 1)
    testx = testx.drop(['P8', 'V7', 'V9'], 1)

    # Lump together all buckets greater than 4 in E7 and E8
    trainx.ix[trainx.E7 > 4, 'E7'] = 4
    trainx.ix[trainx.E8 > 4, 'E8'] = 4
    testx.ix[testx.E7 > 4, 'E7'] = 4
    testx.ix[testx.E8 > 4, 'E8'] = 4

    # Check columns for NaN , i.e. missing values
    # has_nan = pandas.isnull(trainx).any(0).nonzero()

    # Scale zero mean and unit variance for numerical/continous data
    # Exclude categorical variables E3,E7, E8, V5
    numerical_features = [x for x in trainx.columns if x not in
                          ['E3', 'E7', 'E8', 'V5']]
    scaler = preprocessing.StandardScaler().fit(
        trainx.loc[:, numerical_features])
    trainx.loc[:, numerical_features] = scaler.transform(
        trainx.loc[:, numerical_features])
    testx.loc[:, numerical_features] = scaler.transform(
        testx.loc[:, numerical_features])

    # Optionally sub-sample the train data
    rows = random.sample(trainx.index, 600000)
    trainx = trainx.iloc[rows, ]
    trainy = trainy.iloc[rows, ]

    # Optionally sub-sample test data
    rows = random.sample(testx.index, 120000)
    testx = testx.iloc[rows, ]
    testy = testy.iloc[rows, ]

    return trainx, trainy, testx, testy


@timeit
def test_decision_tree(trainx, trainy, testx, testy, max_depth=6):
    """ Train and test a decision tree. max_depth is the max depth
    of the tree that can be grown."""
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    dt = clf.fit(trainx, trainy)
    print 'max_depth = {}'.format(max_depth)
    print 'train accuracy: {}'.format(dt.score(trainx, trainy))
    print 'test accuracy: {}'.format(dt.score(testx, testy))
    return dt


@timeit
def test_svm(trainx, trainy, testx, testy, kernel='poly', C=1):
    """ Train and test a SVM classifier. kernel is the kernel you
    want to use and can be either 'poly', 'linear', 'rbf', or 'sigmoid'.
    """
    sv = svm.SVC(C=C, kernel=kernel)
    sv.fit(trainx, trainy)
    print 'kernel = {}, C = {}'.format(kernel, C)
    print 'train accuracy: {}'.format(sv.score(trainx, trainy))
    print 'test accuracy: {}'.format(sv.score(testx, testy))
    return sv


@timeit
def test_knn(trainx, trainy, testx, testy, k=14, weights='distance', p=1):
    """ Train and test a K-NN classifier. k is the number of neighbors,
    weights is either 'distance' or 'uniform' and refers to how the
    K neighbors are weighted, and p is the norm of the distance, i.e. p=2
    gives L2 norm.
    """
    knn = neighbors.KNeighborsClassifier(k, weights=weights, p=p)
    knn.fit(trainx, trainy)
    print 'k = {}, weights = {}, p = {}'.format(k, weights, p)
    print 'train accuracy: {}'.format(knn.score(trainx, trainy))
    print 'test accuracy: {}'.format(knn.score(testx, testy))
    return knn


@timeit
def test_boosting(trainx, trainy, testx, testy, max_depth=6, n_estimators=600,
                  learning_rate=1.5, algorithm="SAMME"):
    """ Train and test an AdaBoost classifier using trees as a weak learner.
    max_depth is the maximum size of the trees used as weak learners, n_estimators
    is the number learners to combine, learning_rate is self-explanatory, and
    algorithm can be either "SAMME" or "SAMME.R".
    """
    bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators,
                             learning_rate=learning_rate, algorithm=algorithm)
    bdt.fit(trainx, trainy)
    print 'max_depth = {}, n_estimators = {}, learning_rate = {}, algorithm = {}'.format(
        max_depth, n_estimators, learning_rate, algorithm)
    print 'train accuracy: {}'.format(bdt.score(trainx, trainy))
    print 'test accuracy: {}'.format(bdt.score(testx, testy))
    return bdt


@timeit
def test_ann(trainx, trainy, testx, testy):
    """ Train and test an artificial neural network
    """
    input_size = len(trainx[0])
    output_size = len(np.unique(trainy))
    train = ClassificationDataSet(input_size, 1, nb_classes=output_size)
    for x, y in zip(trainx, trainy):
        train.addSample(x, y)

    test = ClassificationDataSet(input_size, 1, nb_classes=output_size)
    for x, y in zip(testx, testy):
        test.addSample(x, y)

    train._convertToOneOfMany()
    test._convertToOneOfMany()

    print "Number of training patterns: ", len(train)
    print "Input and output dimensions: ", train.indim, train.outdim
    print "First sample (input, target):"
    print train['input'][0], train['target'][0]

    n_hidden = 3
    fnn = buildNetwork(train.indim, n_hidden, train.outdim)
    trainer = BackpropTrainer(
        fnn, dataset=train, momentum=0.1, verbose=True, weightdecay=0.01)

    print "# hidden nodes: {}".format(n_hidden)
    for i in range(25):
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(), train['target'])
        tstresult = percentError(
            trainer.testOnClassData(dataset=test), test['target'])
        print "epoch: %4d" % trainer.totalepochs, "  train error: %5.2f%%" % trnresult, "  test error: %5.2f%%" % tstresult
    pred = fnn.activateOnDataset(test)
    preds = [y.argmax() for y in pred]
    print accuracy_score(preds, testy, normalize=True)


@timeit
def test_ann_pd(trainx, trainy, testx, testy, n_hidden, n_iter=25):
    input_size = len(trainx.iloc[0])
    output_size = len(np.unique(trainy))
    train = ClassificationDataSet(input_size, 1, nb_classes=output_size)
    for i in range(len(trainx.index)):
        train.addSample(trainx.iloc[i].values, trainy.iloc[i])

    test = ClassificationDataSet(input_size, 1, nb_classes=output_size)
    for i in range(len(testx.index)):
        test.addSample(testx.iloc[i].values, testy.iloc[i])

    train._convertToOneOfMany()
    test._convertToOneOfMany()

    print "Number of training patterns: ", len(train)
    print "Input and output dimensions: ", train.indim, train.outdim
    print "First sample (input, target):"
    print train['input'][0], train['target'][0]

    fnn = buildNetwork(train.indim, n_hidden, train.outdim)
    trainer = BackpropTrainer(
        fnn, dataset=train, momentum=0.0, verbose=True, weightdecay=0.0)

    print "# hidden nodes: {}".format(n_hidden)

    for i in range(n_iter):
        trainer.trainEpochs(i)
        trnresult = percentError(trainer.testOnClassData(), train['target'])
        tstresult = percentError(
            trainer.testOnClassData(dataset=test), test['target'])
        print "epoch: %4d" % trainer.totalepochs, "  train error: %5.2f%%" % trnresult, "  test error: %5.2f%%" % tstresult

        pred = fnn.activateOnDataset(test)
        preds = [y.argmax() for y in pred]
        print accuracy_score(preds, testy, normalize=True)
    pred = fnn.activateOnDataset(test)
    preds = [y.argmax() for y in pred]
    print accuracy_score(preds, testy, normalize=True)


def net_data(trainx, trainy, testx, testy):
    input_size = len(trainx.iloc[0])
    output_size = len(np.unique(trainy))
    train = ClassificationDataSet(input_size, 1, nb_classes=output_size)
    for i in range(len(trainx.index)):
        train.addSample(trainx.iloc[i].values, trainy.iloc[i])

    test = ClassificationDataSet(input_size, 1, nb_classes=output_size)
    for i in range(len(testx.index)):
        test.addSample(testx.iloc[i].values, testy.iloc[i])

    train._convertToOneOfMany()
    test._convertToOneOfMany()

    print "Number of training patterns: ", len(train)
    print "Input and output dimensions: ", train.indim, train.outdim
    print "First sample (input, target):"
    print train['input'][0], train['target'][0]
    return train, test

def build_2net(input_size, output_size, n_hidden=[5, 3]):
    """ Build a 2 hidden layer network give the layer sizes. """
    # Create network and modules
    net = FeedForwardNetwork()
    inp = LinearLayer(input_size)
    h1 = SigmoidLayer(n_hidden[0])
    h2 = TanhLayer(n_hidden[1])
    outp = LinearLayer(output_size)
    # Add modules
    net.addOutputModule(outp)
    net.addInputModule(inp)
    net.addModule(h1)
    net.addModule(h2)
    # Create connections
    net.addConnection(FullConnection(inp, h1, inSliceTo=6))
    net.addConnection(FullConnection(inp, h2, inSliceFrom=6))
    net.addConnection(FullConnection(h1, h2))
    net.addConnection(FullConnection(h2, outp))
    # Finish up
    net.sortModules()
    return net


@timeit
def test_ann2(trainx, trainy, testx, testy, n_hidden=[5, 3], n_iter=25):
    """ Test and train a 2-hidden layer neural network, where the first layer is
    composed of sigmoid units and the second layer is composed of tanh units.
    n_hibben is a 2-element list of the sizes of the hidden layers, and n_iter
    is the number of epochs to stop at.
    """
    train, test = net_data(trainx, trainy, testx, testy)
    input_size = len(trainx.iloc[0])
    output_size = len(np.unique(trainy))
    net = build_2net(input_size, output_size, n_hidden=n_hidden)
    # Train the network using back-propagation
    trainer = BackpropTrainer(net, dataset=train, momentum=0.0, verbose=True, weightdecay=0.0)

    for i in range(n_iter):
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(), train['target'])
        tstresult = percentError(
            trainer.testOnClassData(dataset=test), test['target'])

        # Calculate current training and test set accuracy (not error!)
        pred = net.activateOnDataset(test)
        preds = [y.argmax() for y in pred]
        test_acc = accuracy_score(preds, testy, normalize=True)
        pred = net.activateOnDataset(train)
        preds = [y.argmax() for y in pred]
        train_acc = accuracy_score(preds, trainy, normalize=True)
        print 'epoch: {}, train accuracy: {}, test accuracy: {}'.format(trainer.totalepochs,
                                                          train_acc, test_acc)

    return train_acc, test_acc


@timeit
def ann_learning_curve(trainx, trainy, testx, testy, n_hidden=[5, 3],
                       n_iter=10, cv=5, train_sizes=np.linspace(.1, 1.0, 10)):
    """ Returns the learning curve for artificial neural networks, i.e. the
    training and test accuracies (not error!). The input variables are:
    trainx - features of the training data
    trainy - labels of the training data
    testx - features of the test data
    testy - labels of the test data
    n_hidden - 2-element list of hidden layer sizes
    n_iter - # of epochs to stop training at
    cv - the number of trainings to be average for more accurate estimates
    train_sizes - list of training size proportions, from (0.0, 1.0]
                   corresponding to 0% to 100% of the full training set set size

    The return variables are:
    train_sizes - list of the training set (proportional) sizes, i.e. x axis
    average_train_scores - the average training accuracy at each training set size
    average_test_scores - the average test accuracy at each training set size
    """

    cv_train_scores = [[0] * len(train_sizes)]
    cv_test_scores = [[0] * len(train_sizes)]
    for c in range(cv):
        train_scores = []
        test_scores = []
        for ts in train_sizes:
            n_examples = int(round(len(trainx) * ts))
            rows = random.sample(range(len(trainx)), n_examples)
            subx = trainx.iloc[rows, ]
            suby = trainy.iloc[rows, ]
            start = time.time()
            a, b = test_ann2(subx, suby, testx, testy,
                             n_hidden=n_hidden, n_iter=n_iter)
            print 'training time: {} secs'.format(time.time() - start)
            current_train_score = a
            current_test_score = b
            train_scores.append(current_train_score)
            test_scores.append(current_test_score)
        cv_train_scores.append(train_scores)
        cv_test_scores.append(test_scores)
    average_train_scores = [sum(i) / cv for i in zip(*cv_train_scores)]
    average_test_scores = [sum(i) / cv for i in zip(*cv_test_scores)]
    return train_sizes, average_train_scores, average_test_scores


@timeit
def get_learning_curve(estimator, trainx, trainy, testx, testy, cv=1,
                       train_sizes=np.linspace(.1, 1.0, 10)):
    """ Returns the learning curve for scikit lassifiers (no neural nets!), i.e.
    training and test accuracies (not error!).

    The input variables are:
    estimator - the scikit classifier to be used (parameters should alread by set)
    trainx - features of the training data
    trainy - labels of the training data
    testx - features of the test data
    testy - labels of the test data
    cv - the number of trainings to be average for more accurate estimates
    train_sizes - list of training size proportions, from (0.0, 1.0]
                   corresponding to 0% to 100% of the full training set set size

    The return variables are:
    train_sizes - list of the training set (proportional) sizes, i.e. x axis
    average_train_scores - the average training accuracy at each training set size
    average_test_scores - the average test accuracy at each training set size
    """

    cv_train_scores = [[0] * len(train_sizes)]
    cv_test_scores = [[0] * len(train_sizes)]
    for c in range(cv):
        train_scores = []
        test_scores = []
        for ts in train_sizes:
            n_examples = int(round(len(trainx) * ts))
            rows = random.sample(range(len(trainx)), n_examples)
            subx = trainx.iloc[rows, ]
            suby = trainy.iloc[rows, ]
            start = time.time()
            estimator.fit(subx, suby)
            print 'training time: {} secs'.format(time.time() - start)
            current_train_score = estimator.score(subx, suby)
            current_test_score = estimator.score(testx, testy)
            train_scores.append(current_train_score)
            test_scores.append(current_test_score)
        cv_train_scores.append(train_scores)
        cv_test_scores.append(test_scores)
    average_train_scores = [sum(i) / cv for i in zip(*cv_train_scores)]
    average_test_scores = [sum(i) / cv for i in zip(*cv_test_scores)]
    return train_sizes, average_train_scores, average_test_scores


def run_all(dataset='forest', train_sizes=np.linspace(0.1, 1, 10)):
    """ Collect the training size based learning curves for all classifiers.
    Input variables are:
    dataset - either 'forest' or 'ford' depending which dataset you want
    train_sizes - list of the training (proportional) sizes, e.g. [0, 0.5 1]

    Outputs are written to file in the same directory as this script.

    Classifier parameters are set inside this function. Edit them as desired.
    Refer to Scikit-learn documentation for more details.
    """
    ts = train_sizes
    if dataset == 'forest':
        tx, ty, vx, vy = load_data_forest()
        dt = tree.DecisionTreeClassifier(max_depth=22)
        bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=12),
                                 n_estimators=400, learning_rate=1.5, algorithm="SAMME")
        sv = svm.SVC(C=1, kernel="linear")
        knn = neighbors.KNeighborsClassifier(3, weights='distance', p=1)

        a, b, c = get_learning_curve(dt, tx, ty, vx, vy, cv=5, train_sizes=ts)
        results = [a, b, c]
        with open('forest_dt_results.pickle', 'w') as f:
            pickle.dump(results, f)

        a, b, c = get_learning_curve(bdt, tx, ty, vx, vy, cv=5, train_sizes=ts)
        results = [a, b, c]
        with open('forest_bdt_results.pickle', 'w') as f:
            pickle.dump(results, f)

        a, b, c = get_learning_curve(sv, tx, ty, vx, vy, cv=5, train_sizes=ts)
        results = [a, b, c]
        with open('forest_sv_results.pickle', 'w') as f:
            pickle.dump(results, f)

        a, b, c = get_learning_curve(knn, tx, ty, vx, vy, cv=5, train_sizes=ts)
        results = [a, b, c]
        with open('forest_knn_results.pickle', 'w') as f:
            pickle.dump(results, f)

        a, b, c = ann_learning_curve(tx, ty, vx, vy, n_hidden=[30, 15], n_iter=10,
                                     cv=5, train_sizes=ts)
        with open('forest_ann_results.pickle', 'w') as f:
            pickle.dump(results, f)


    elif dataset == 'ford':
        tx, ty, vx, vy = load_data_ford()
        dt = tree.DecisionTreeClassifier(max_depth=3)
        bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1),
                                 n_estimators=400, learning_rate=1.5, algorithm="SAMME")
        sv = svm.SVC(C=1, kernel="linear")
        knn = neighbors.KNeighborsClassifier(25, weights='uniform', p=2)

        a, b, c = get_learning_curve(dt, tx, ty, vx, vy, cv=5, train_sizes=ts)
        results = [a, b, c]
        with open('ford_dt_results.pickle', 'w') as f:
            pickle.dump(results, f)

        a, b, c = get_learning_curve(bdt, tx, ty, vx, vy, cv=5, train_sizes=ts)
        results = [a, b, c]
        with open('ford_bdt_results.pickle', 'w') as f:
            pickle.dump(results, f)

        a, b, c = get_learning_curve(sv, tx, ty, vx, vy, cv=5, train_sizes=ts)
        results = [a, b, c]
        with open('ford_sv_results.pickle', 'w') as f:
            pickle.dump(results, f)

        a, b, c = get_learning_curve(knn, tx, ty, vx, vy, cv=5, train_sizes=ts)
        results = [a, b, c]
        with open('ford_knn_results.pickle', 'w') as f:
            pickle.dump(results, f)

        a, b, c = ann_learning_curve(tx, ty, vx, vy, n_hidden=[15, 7], n_iter=4,
                                     cv=5, train_sizes=ts)
        with open('ford_ann_results.pickle', 'w') as f:
            pickle.dump(results, f)


def err_plot(data, title=''):
    """ Make a plot of the training and test error given the data
    resulting from get_learning_curve() or ann_learning_curve().

    Input variables are:
    data - a list of the three outputs from get_learning_curve()
           or ann_learning_curve()
    title - the title of the plot
    """
    ts, te, ve = data
    te = [1-x for x in te]  # convert to error
    ve = [1-x for x in ve]  # error
    plt.plot(ts, te, 'r-', ts, ve, 'b-')
    plt.ylabel('Error (out of 1)')
    plt.xlabel('Fractional training set size')
    plt.title(title)
    plt.legend(['Training error', 'Test error'], loc='best',
               fancybox=True, shadow=True,
               prop=FontProperties().set_size('small'))
    plt.show()


def plot_learning_curves():
    """ Automatically plots the learning curve data generated by
    get_learning_curve() or ann_learning_curve() functions,
    and also from the run_all() function. If you run the run_all() function,
    this will save data to be read by this function.
    """

    """
    Plot Ford dataset results
    """
    # decision tree
    with open('ford_dt_results.pickle') as f:
        data = pickle.load(f)
    err_plot(data, title='Ford alertness data - Decision tree')
    # boosted decision tree
    with open('ford_bdt_results.pickle') as f:
        data = pickle.load(f)
    err_plot(data, title='Ford alertness data - Boosted decision tree (AdaBoost)')
    # KNN
    with open('ford_knn_results.pickle') as f:
        data = pickle.load(f)
    err_plot(data, title='Ford alertness data - K-NN')
    # SVM
    with open('ford_sv_results.pickle') as f:
        data = pickle.load(f)
    err_plot(data, title='Ford alertness data - SVM')
    # ANN
    with open('ford_ann_results.pickle') as f:
        data = pickle.load(f)
    err_plot(data, title='Ford alertness data - ANN')


    """
    Plot Forest Cover dataset results
    """
    # decision tree
    with open('forest_dt_results.pickle') as f:
        data = pickle.load(f)
    err_plot(data, title='Forest cover data - Decision tree')
    # boosted decision tree
    with open('forest_bdt_results.pickle') as f:
        data = pickle.load(f)
    err_plot(data, title='Forest cover data - Boosted decision tree (AdaBoost)')
    # KNN
    with open('forest_knn_results.pickle') as f:
        data = pickle.load(f)
    err_plot(data, title='Forest cover data - K-NN')
    # SVM
    with open('forest_sv_results.pickle') as f:
        data = pickle.load(f)
    err_plot(data, title='Forest cover data - SVM')
    # ANN
    with open(results+'forest_ann_results.pickle') as f:
        data = pickle.load(f)
    err_plot(data, title='Forest cover data - ANN')


def plot_my_figures():
    """ Automatically generate all the plots used in the report. This uses the
    pickled data from the get_learning_curve() or ann_learning_curve() functions,
    and also from the run_all() function.
    """

    """
    Plot Ford dataset results
    """
    results = '../results/'
    # decision tree
    with open(results+'ford_dt_results.pickle') as f:
        data = pickle.load(f)
    err_plot(data, title='Ford alertness data - Decision tree')
    # boosted decision tree
    with open(results+'ford_bdt_results.pickle') as f:
        data = pickle.load(f)
    err_plot(data, title='Ford alertness data - Boosted decision tree (AdaBoost)')
    # KNN
    with open(results+'ford_knn/ford_knn_results_2_uni_1.pickle') as f:
        knn2 = pickle.load(f)
    with open(results+'ford_knn/ford_knn_results_7_uni_1.pickle') as f:
        knn7 = pickle.load(f)
    with open(results+'ford_knn/ford_knn_results_15_uni_1.pickle') as f:
        knn15 = pickle.load(f)
    with open(results+'ford_knn/ford_knn_results_50_uni_1.pickle') as f:
        knn50 = pickle.load(f)
    knn2[1] = [1-x for x in knn2[1]]  # change accuracy to error
    knn2[2] = [1-x for x in knn2[2]]
    knn7[1] = [1-x for x in knn7[1]]
    knn7[2] = [1-x for x in knn7[2]]
    knn15[1] = [1-x for x in knn15[1]]
    knn15[2] = [1-x for x in knn15[2]]
    knn50[1] = [1-x for x in knn50[1]]
    knn50[2] = [1-x for x in knn50[2]]
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(knn2[0], knn2[1], 'r--', knn2[0], knn2[2], 'r-',
             knn7[0], knn7[1], 'b--', knn7[0], knn7[2], 'b-',
             knn15[0], knn15[1], 'g--', knn15[0], knn15[2], 'g-',
             knn15[0], knn15[1], 'c--', knn50[0], knn50[2], 'c-')
    plt.xlabel('Train set size (proportion of maximum)')
    plt.ylabel('Error (out of 1)')
    plt.title('Ford alertness data - KNNs')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(['2-NN train error',
               '2-NN test error',
               '7-NN train error',
               '7-NN test error',
               '15-NN train error',
               '15-NN test error',
               '50-NN train error',
               '50-NN test error'],
               loc='center left', bbox_to_anchor=(1, 0.5),
               ncol=1, fancybox=True, shadow=True,
               prop=FontProperties().set_size('small'))
    plt.show()
    # SVM
    with open(results+'ford_sv1_results.pickle') as f:
        sv1 = pickle.load(f)
    with open(results+'ford_sv2_results.pickle') as f:
        sv2 = pickle.load(f)
    sv1[1] = [1-x for x in sv1[1]]
    sv1[2] = [1-x for x in sv1[2]]
    sv2[1] = [1-x for x in sv2[1]]
    sv2[2] = [1-x for x in sv2[2]]
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(sv1[0], sv1[1], 'r--', sv1[0], sv1[2], 'r-',
             sv2[0], sv2[1], 'b--', sv2[0], sv2[2], 'b-')
    plt.xlabel('Train set size (proportion of maximum)')
    plt.ylabel('Error (out of 1)')
    plt.title('Ford alertness data - SVMs')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    plt.legend(['Polynomial kernel SVM train error',
               'Polynomial kernel SVM test error',
               'Linear kernel SVM train error',
               'Linear kernel SVM test error'],
               loc='center right',
               fancybox=True, shadow=True, ncol=1,
               prop = FontProperties().set_size('small'))
    plt.show()
    # ANN
    with open(results+'ford_ann_results.pickle') as f:
        data = pickle.load(f)
    err_plot(data, title='Ford alertness data - ANN')


    """
    Plot Forest Cover dataset results
    """
    # decision tree
    with open(results+'forest_dt_results.pickle') as f:
        data = pickle.load(f)
    err_plot(data, title='Forest cover data - Decision tree')
    # boosted decision tree
    with open(results+'forest_bdt_results.pickle') as f:
        data = pickle.load(f)
    err_plot(data, title='Forest cover data - Boosted decision tree (AdaBoost)')
    # KNN
    with open(results+'forest_knn/forest_knn_results_3_dist_2.pickle') as f:
        knn3 = pickle.load(f)
    with open(results+'forest_knn/forest_knn_results_8_dist_2.pickle') as f:
        knn8 = pickle.load(f)
    with open(results+'forest_knn/forest_knn_results_15_dist_2.pickle') as f:
        knn15 = pickle.load(f)
    fig = plt.figure()
    ax = plt.subplot(111)
    knn3[1] = [1-x for x in knn3[1]]
    knn3[2] = [1-x for x in knn3[2]]
    knn8[1] = [1-x for x in knn8[1]]
    knn8[2] = [1-x for x in knn8[2]]
    knn15[1] = [1-x for x in knn15[1]]
    knn15[2] = [1-x for x in knn15[2]]
    plt.plot(knn3[0], knn3[1], 'r--', knn3[0], knn3[2], 'r-',
             knn8[0], knn8[1], 'b--', knn8[0], knn8[2], 'b-',
             knn15[0], knn15[1], 'g--', knn15[0], knn15[2], 'g-')
    plt.xlabel('Train set size (proportion of maximum)')
    plt.ylabel('Error (out of 1)')
    plt.title('Forest cover data - KNNs')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    plt.legend(['3-NN training error',
               '3-NN test error',
               '8-NN training error',
               '8-NN test error',
               '15-NN training error',
               '15-NN test error'],
               loc='center left', bbox_to_anchor=(1, 0.5),
               ncol=1, fancybox=True, shadow=True,
               prop=FontProperties().set_size('small'))
    plt.show()
    # SVM
    with open(results+'forest_sv1_results.pickle') as f:
        sv1 = pickle.load(f)
    with open(results+'forest_sv2_results.pickle') as f:
        sv2 = pickle.load(f)
    sv1[1] = [1-x for x in sv1[1]]
    sv1[2] = [1-x for x in sv1[2]]
    sv2[1] = [1-x for x in sv2[1]]
    sv2[2] = [1-x for x in sv2[2]]
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(sv1[0], sv1[1], 'r--', sv1[0], sv1[2], 'r-',
             sv2[0], sv2[1], 'b--', sv2[0], sv2[2], 'b-')
    plt.xlabel('Train set size (proportion of maximum)')
    plt.ylabel('Error (out of 1)')
    plt.title('Forest cover data - SVMs')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    plt.legend(['Polynomial kernel SVM train error',
               'Polynomial kernel SVM test error',
               'Linear kernel SVM train error',
               'Linear kernel SVM test error'],
               loc='upper right',
               fancybox=True, shadow=True, ncol=1,
               prop = FontProperties().set_size('small'))
    plt.show()
    # ANN
    with open(results+'forest_ann_results_30_15_10.pickle') as f:
        data = pickle.load(f)
    err_plot(data, title='Forest cover data - ANN')


def main():
    # Run all classifiers on both data sets, saving data
    run_all()
    # Plot results
    plot_learning_curves()

    # # Load Ford data
    # tx, ty, vx, vy = load_data_ford()
    # # Test decision tree on Ford Data
    # test_decision_tree(tx, ty, vx, vy, max_depth=3)
    # # Create Adaboost classifier
    # bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1), n_estimators=600,
    #                          learning_rate=1.5, algorithm="SAMME")
    # # Training sizes = 10%, 20%, ..., 100%
    # train_sizes = np.linspace(0.1, 1, 10)
    # # Get learning curve for AdaBoost on Ford data
    # train_sizes, train_scores, test_scores = get_learning_curve(bdt, tx, ty, vx, vy)


if __name__ == '__main__':
    main()
