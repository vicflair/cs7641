import algs
import numpy as np
from ann import ANN
from datasets import Segmentation


def exp1():
    """Run the clustering algorithms on the datasets and describe what you see.
    """
    # Load segmentation datasets
    seg = Segmentation()

    # Find K-means cluster
    km = algs.km(seg.train)
    X = km.predict(seg.train.X)
    
    # Find EM clusters
    em = algs.em(seg.train)
    X = em.predict(seg.train.X)
    

def exp2():
    """Apply the dimensionality reduction algorithms to the two datasets and
    describe what you see."""

    # Load segmentation datasets
    seg = Segmentation()

    # Apply PCA
    pca = algs.pca(seg.train)
    X = pca.transform(seg.train.X)

    # Apply ICA
    ica = algs.ica(seg.train)
    X = ica.transform(seg.train.X)

    # Apply Randomized PCA
    rca = algs.rca(seg.train)
    X = rca.transform(seg.train.X)

    # Apply Linear Discriminant Analysis
    lda = algs.lda(seg.train)
    X = lda.transform(seg.train.X)


def exp3():
    """Reproduce your clustering experiments, but on the data after you've run
    dimensionality reduction on it."""

    dim_red_algs = [algs.pca, algs.ica, algs.rca, algs.lda]
    cluster_algs = [algs.km, algs.em]
    for dra in dim_red_algs:
        for ca in cluster_algs:
            # Load segmentation data set
            seg = Segmentation()
            seg.train.X = dra(seg.train).transform(seg.train.X)
            C = ca(seg.train).predict(seg.train.X)


def exp4():
    """Apply the dimensionality reduction algorithms to one of your datasets
    from assignment #1, then rerun your neural network learner on the newly
    projected data."""

    # Load segmentaton dataset
    seg = Segmentation()

    # Build the neural network without dimensionality reduction
    nn = ANN()
    nn.train = nn.load_data(seg.train.X, seg.train.Y)
    nn.test = nn.load_data(seg.test.X, seg.test.Y)
    nn.make_network()
    nn.make_trainer()

    # Train and run the neural network
    for iter in range(50):
        nn.train_network()
        print 'iter: {}  train: {}  test: {}'.format(iter, nn.fitf(),
                                                     nn.fitf(train=False))

    dim_red_algs = [algs.pca, algs.ica, algs.rca, algs.lda]
    for dra in dim_red_algs:
        # Load a new instance of the segmentation dataset
        seg = Segmentation()

        # Apply dimensionality reduction algorithm to training and test sets.
        est = dra(seg.train)
        seg.train.X = est.transform(seg.train.X)
        seg.test.X = est.transform(seg.test.X)

        # Build neural network
        nn = ANN()
        nn.train = nn.load_data(seg.train.X, seg.train.Y)
        nn.test = nn.load_data(seg.test.X, seg.test.Y)
        nn.make_network()
        nn.make_trainer()

        # Run neural network
        for iter in range(50):
            nn.train_network()
            print 'iter: {}  train: {}  test: {}'.format(iter, nn.fitf(),
                                                     nn.fitf(train=False))

def exp5():
    """Apply the clustering algorithms to the same dataset to which you just
    applied the dimensionality reduction algorithms, treating the clusters as
    if they were new (additional) features. Rerun your neural network leaner
    on the newly projected data."""

    # Load segmentaton dataset
    seg = Segmentation()

    # Build the neural network without dimensionality reduction
    nn = ANN()
    nn.train = nn.load_data(seg.train.X, seg.train.Y)
    nn.test = nn.load_data(seg.test.X, seg.test.Y)
    nn.make_network()
    nn.make_trainer()

    # Train and run the neural network
    for iter in range(10):
        nn.train_network()
        print 'iter: {}  train: {}  test: {}'.format(iter, nn.fitf(),
                                                     nn.fitf(train=False))

    dim_red_algs = [algs.pca, algs.ica, algs.rca, algs.lda]
    cluster_algs = [algs.km, algs.em]
    for dra in dim_red_algs:
        for ca in cluster_algs:
            # Load a new instance of the segmentation dataset
            seg = Segmentation()

            # Apply dimensionality reduction algorithm to training and test sets.
            est = dra(seg.train)
            seg.train.X = est.transform(seg.train.X)
            seg.test.X = est.transform(seg.test.X)

            # Apply clustering to the reduced dimensionality dataset
            est = ca(seg.train)
            clusters = est.predict(seg.train.X)
            seg.train.X = [np.append(x, c) for x, c 
                           in zip(seg.train.X, clusters)]
            clusters = est.predict(seg.test.X)
            seg.test.X = [np.append(x, c) for x, c 
                          in zip(seg.test.X, clusters)]

            # Build neural network
            nn = ANN()
            nn.train = nn.load_data(seg.train.X, seg.train.Y)
            nn.test = nn.load_data(seg.test.X, seg.test.Y)
            nn.make_network()
            nn.make_trainer()

            # Run neural network
            for iter in range(10):
                nn.train_network()
                print 'iter: {}  train: {}  test: {}'.format(iter, nn.fitf(),
                                                         nn.fitf(train=False))
