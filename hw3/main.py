import algs  # My algorithms
import numpy as np
import matplotlib.pyplot as plt
from ann import ANN  # ANN code
from datasets import Segmentation  # Segmentation dataset
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.lda import LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def exp1():
    """Run the clustering algorithms on the datasets and describe what you see.
    """
    # Load segmentation datasets
    seg = Segmentation()

    # Find K-means cluster
    print '-'*20 + ' K-means ' + '-'*20
    scaler = StandardScaler(with_mean=False)
    km = KMeans(n_clusters=7)
    X = scaler.fit_transform(seg.train.X)
    Y = km.fit_predict(X)
    
    # Do the clusters line up with the labels?
    print 'ARI: {}'.format( metrics.adjusted_rand_score(seg.train.Y, Y))
    print 'AMI: {}'.format(metrics.adjusted_mutual_info_score(seg.train.Y, Y))
    
    # How good are the clusters?
    print 'Homogeneity: {}'.format(metrics.homogeneity_score(seg.train.Y, Y))
    print 'Completeness: {}'.format(metrics.completeness_score(seg.train.Y, Y))
    print 'Silhouette: {}'.format(metrics.silhouette_score(X, km.labels_))

    # Find EM clusters
    print '-'*20 + ' EM ' + '-'*20
    em = GMM(n_components=7)
    em.fit(seg.train.X)
    Y = em.predict(seg.train.X)
    
    # Do the clusters line up with the labels?
    print 'ARI: {}'.format( metrics.adjusted_rand_score(seg.train.Y, Y))
    print 'AMI: {}'.format(metrics.adjusted_mutual_info_score(seg.train.Y, Y))
    
    # How good are the clusters?
    print 'Homogeneity: {}'.format(metrics.homogeneity_score(seg.train.Y, Y))
    print 'Completeness: {}'.format(metrics.completeness_score(seg.train.Y, Y))
    print 'Silhouette: {}'.format(metrics.silhouette_score(X, Y))


def exp2():
    """Apply the dimensionality reduction algorithms to the two datasets and
    describe what you see."""

    # Parameters
    N = 5  # Number of components

    # Load segmentation datasets
    seg = Segmentation()

    # Apply PCA
    print '-'*20 + ' PCA ' + '-'*20
    scaler = StandardScaler()
    pca = PCA(n_components=N)
    X = scaler.fit_transform(seg.train.X)
    X = pca.fit_transform(X)
    
    # Describe PCA results
    eigvals = np.linalg.eigvals(pca.get_covariance())
    expl_var = sum(pca.explained_variance_ratio_) 
    R = scaler.inverse_transform(pca.inverse_transform(X))  # Reconstruction
    R_error = sum(map(np.linalg.norm, R-seg.train.X))
    print 'Eigenvalues:'
    print '{}'.format(eigvals)
    print 'Explained variance (%): {}'.format(expl_var)
    print 'Reconstruction error: {}'.format(R_error) 

    # Apply ICA
    print '-'*20 + ' ICA ' + '-'*20
    ica = FastICA(n_components=N)
    X = ica.fit_transform(seg.train.X)
    
    # Describe ICA results
    pass

    # Apply "Randomized Components Analysis"
    print '-'*20 + ' RCA ' + '-'*20
    scaler = StandardScaler()
    grp = GaussianRandomProjection(n_components=N)
    X = scaler.fit_transform(seg.train.X)
    X = grp.fit_transform(X)

    # Describe RCA results
    inv = np.linalg.pinv(grp.components_)
    R = scaler.inverse_transform(np.dot(X, inv.T))  # Reconstruction
    R_error = sum(map(np.linalg.norm, R-seg.train.X))
    print 'Reconstruction error: {}'.format(R_error) 

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
